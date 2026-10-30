"""Evaluation evidence and fairness rates.

Kognita does not run an evaluation harness. It records results and refuses to
activate a tier 2 or higher agent until those results meet the use case
thresholds. A model-only result is not enough: the gate accepts an agent-level
or a system-level result for the same agent version and model version.

Fairness rates are computed from origination evidence across segments the bank
defines, including proxy attributes. A disparity above the bank's threshold is
flagged to the risk owner. Quality drift in 0.5 watches decision quality and
reliability only, so a disparity is not registered as one of those metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from sqlmodel import Session, select

from kognita.canonical import canonical_hash, canonical_json
from kognita.models import EvidenceEvent
from kognita.retention import RetentionStore
from kognita.vocabulary import ActorType, EventType

METRICS: tuple[str, ...] = (
    "accuracy",
    "groundedness",
    "completeness",
    "relevance",
    "hallucination_rate",
    "freshness",
    "policy_compliance",
    "entitlement",
    "action_correctness",
    "escalation",
)
LOWER_IS_BETTER = frozenset({"hallucination_rate"})
LEVELS = frozenset({"model", "agent", "system"})
APPLICABLE_LEVELS = frozenset({"agent", "system"})
_RATE_NAMES = ("recommendation_rate", "contact_rate", "prioritisation_rate")
_UNASSIGNED = "(unassigned)"
_KNOWN_OUTCOMES = frozenset({"recommend", "do not contact", "reversal"})


def _text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(code)
    return value.strip()


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not a number")
    number = float(value)
    if number != number or number < 0 or number > 1:
        raise ValueError("out of range")
    return number


def validate_thresholds(thresholds: object) -> dict[str, float]:
    """Closed metric names and bounds. An empty mapping is valid."""
    if not isinstance(thresholds, Mapping) or isinstance(thresholds, (str, bytes)):
        raise ValueError("EVALUATION_THRESHOLDS_INVALID")
    cleaned: dict[str, float] = {}
    for key, value in thresholds.items():
        if not isinstance(key, str) or key not in METRICS:
            raise ValueError(f"EVALUATION_THRESHOLDS_INVALID: {key}")
        try:
            cleaned[key] = _number(value)
        except ValueError as exc:
            raise ValueError(f"EVALUATION_THRESHOLDS_INVALID: {key}") from exc
    return cleaned


def _complete_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, Mapping) or isinstance(metrics, (str, bytes)):
        raise ValueError("EVALUATION_METRICS_INVALID")
    if set(metrics) != set(METRICS):
        raise ValueError("EVALUATION_METRICS_INVALID")
    cleaned: dict[str, float] = {}
    for key in METRICS:
        try:
            cleaned[key] = _number(metrics[key])
        except ValueError as exc:
            raise ValueError(f"EVALUATION_METRICS_INVALID: {key}") from exc
    return cleaned


@dataclass(frozen=True)
class EvaluationResult:
    """One recorded result for an agent version and a model version."""

    agent_id: str
    version: str
    model_version: str
    level: str
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        agent_id = _text(self.agent_id, "EVALUATION_AGENT_REQUIRED")
        version = _text(self.version, "EVALUATION_VERSION_REQUIRED")
        model_version = _text(self.model_version, "EVALUATION_MODEL_REQUIRED")
        if self.level not in LEVELS:
            raise ValueError("EVALUATION_LEVEL_INVALID")
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "metrics", _complete_metrics(self.metrics))


def _body(result: EvaluationResult) -> dict[str, Any]:
    return {
        "agent_id": result.agent_id,
        "version": result.version,
        "model_version": result.model_version,
        "level": result.level,
        "metrics": dict(result.metrics),
    }


def record_evaluation(session: Session, result: EvaluationResult, *, evidence) -> EvaluationResult:
    """Retain a result and append EVALUATION_RECORDED. Invalid results are not stored."""
    if not isinstance(result, EvaluationResult):
        raise ValueError("EVALUATION_METRICS_INVALID")
    body = _body(result)
    digest = RetentionStore().retain_value(
        session, body, kind="evaluation", correlation_id=f"evaluation:{result.agent_id}",
    )
    evidence.emit(
        session,
        correlation_id=f"evaluation:{result.agent_id}",
        event_type=EventType.EVALUATION_RECORDED,
        actor_type=ActorType.HUMAN,
        actor_id=result.agent_id,
        payload={"action": "RECORDED", **body, "content_hash": digest},
    )
    return result


def _gated(session: Session, agent) -> list[Any]:
    from kognita.usecases import load_register

    names = getattr(agent, "use_cases", None) or []
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("EVALUATION_REQUIRED: use cases are unreadable")
    register = load_register(session)
    found = []
    for name in names:
        entry = register.get(name)
        if entry is not None and entry.proposed_tier >= 2:
            found.append(entry)
    return found


def _thresholds_for(entries: Sequence[Any]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for entry in entries:
        raw = getattr(entry, "evaluation_thresholds", None)
        if not raw:
            raw = (getattr(entry, "record", None) or {}).get("evaluation_thresholds") or {}
        try:
            parsed = validate_thresholds(raw)
        except ValueError as exc:
            raise ValueError(f"EVALUATION_THRESHOLDS_INVALID: {entry.usecase_id}") from exc
        if not parsed:
            raise ValueError(f"EVALUATION_THRESHOLDS_MISSING: {entry.usecase_id}")
        for key, value in parsed.items():
            current = merged.get(key)
            if current is None:
                merged[key] = value
            elif key in LOWER_IS_BETTER:
                merged[key] = min(current, value)
            else:
                merged[key] = max(current, value)
    return merged


def _stored_result(session: Session, event: EvidenceEvent) -> dict[str, Any]:
    payload = event.payload if isinstance(event.payload, dict) else {}
    if canonical_hash(payload) != event.payload_hash:
        raise ValueError("EVALUATION_UNREADABLE")
    raw = RetentionStore().read(session, str(payload.get("content_hash") or ""))
    if not raw:
        raise ValueError("EVALUATION_UNREADABLE")
    try:
        stored = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("EVALUATION_UNREADABLE") from exc
    expected = {key: payload.get(key) for key in ("agent_id", "version", "model_version", "level", "metrics")}
    if not isinstance(stored, dict) or canonical_json(stored) != canonical_json(expected):
        raise ValueError("EVALUATION_UNREADABLE")
    return stored


def _latest_applicable(session: Session, agent) -> dict[str, Any] | None:
    rows = session.exec(
        select(EvidenceEvent)
        .where(EvidenceEvent.event_type == EventType.EVALUATION_RECORDED)
        .order_by(EvidenceEvent.sequence)
    ).all()
    chosen: dict[str, Any] | None = None
    for event in rows:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("action") != "RECORDED":
            continue
        if payload.get("agent_id") != agent.name or payload.get("version") != agent.version:
            continue
        if payload.get("model_version") != agent.model:
            continue
        stored = _stored_result(session, event)
        if stored.get("level") in APPLICABLE_LEVELS:
            chosen = stored
    return chosen


def _misses(metrics: Mapping[str, Any], thresholds: Mapping[str, float]) -> list[str]:
    misses = []
    for key, threshold in thresholds.items():
        if key not in metrics:
            misses.append(f"{key} missing")
            continue
        try:
            value = _number(metrics[key])
        except ValueError:
            misses.append(f"{key} unreadable")
            continue
        if key in LOWER_IS_BETTER:
            if value > threshold:
                misses.append(f"{key} {value} above {threshold}")
        elif value < threshold:
            misses.append(f"{key} {value} below {threshold}")
    return misses


def check_activation(session: Session, agent) -> None:
    """Refuse activation when a tier 2 or higher agent lacks meeting results.

    Tier 1 agents are not gated. A linked use case at tier 2 or above must
    name thresholds, and the latest agent-level or system-level result for
    this agent version and model version must meet every one of them.
    Hallucination rate is lower-is-better. Model-level results do not count.
    """
    entries = _gated(session, agent)
    if not entries:
        return
    if not isinstance(getattr(agent, "model", None), str) or not agent.model.strip():
        raise ValueError("EVALUATION_MODEL_REQUIRED")
    if not isinstance(getattr(agent, "version", None), str) or not agent.version.strip():
        raise ValueError("EVALUATION_VERSION_REQUIRED")
    thresholds = _thresholds_for(entries)
    stored = _latest_applicable(session, agent)
    if stored is None:
        raise ValueError("EVALUATION_REQUIRED: no agent or system result for this agent and model version")
    try:
        metrics = _complete_metrics(stored.get("metrics"))
    except ValueError as exc:
        raise ValueError("EVALUATION_UNREADABLE") from exc
    misses = _misses(metrics, thresholds)
    if misses:
        raise ValueError("EVALUATION_BELOW_THRESHOLD: " + "; ".join(misses))


@dataclass(frozen=True)
class FairnessReport:
    """Recommendation, contact and prioritisation rates, and flagged disparities."""

    rates: tuple[Mapping[str, Any], ...]
    disparities: tuple[Mapping[str, Any], ...]
    threshold: float
    proxy_attributes: tuple[str, ...]
    risk_owners: tuple[str, ...]
    flagged: bool
    monitored_metric: None = None


def _segments(segments: object, proxies: object) -> tuple[dict[str, dict[str, str]], tuple[str, ...]]:
    if not isinstance(segments, Mapping) or isinstance(segments, (str, bytes)) or not segments:
        raise ValueError("FAIRNESS_SEGMENTS_REQUIRED")
    cleaned: dict[str, dict[str, str]] = {}
    for name, mapping in segments.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("FAIRNESS_SEGMENTS_REQUIRED")
        if not isinstance(mapping, Mapping) or isinstance(mapping, (str, bytes)) or not mapping:
            raise ValueError(f"FAIRNESS_SEGMENTS_REQUIRED: {name}")
        clients: dict[str, str] = {}
        for client_id, value in mapping.items():
            if not isinstance(client_id, str) or not client_id.strip():
                raise ValueError(f"FAIRNESS_SEGMENTS_REQUIRED: {name}")
            if not isinstance(value, str) or not value.strip() or value.strip() == _UNASSIGNED:
                raise ValueError(f"FAIRNESS_SEGMENTS_REQUIRED: {name}")
            clients[client_id.strip()] = value.strip()
        cleaned[name.strip()] = clients
    if not isinstance(proxies, (list, tuple)):
        raise ValueError("FAIRNESS_PROXY_UNKNOWN")
    proxy_names = []
    for name in proxies:
        if not isinstance(name, str) or name.strip() not in cleaned:
            raise ValueError("FAIRNESS_PROXY_UNKNOWN")
        proxy_names.append(name.strip())
    return cleaned, tuple(dict.fromkeys(proxy_names))


def _threshold(value: object) -> float:
    try:
        number = _number(value)
    except ValueError as exc:
        raise ValueError("FAIRNESS_THRESHOLD_INVALID") from exc
    return number


def _originations(session: Session, usecase_id: str) -> dict[str, dict[str, Any]]:
    from kognita.models import Interaction

    rows = session.exec(
        select(EvidenceEvent)
        .where(EvidenceEvent.event_type == EventType.ORIGINATION)
        .order_by(EvidenceEvent.sequence)
    ).all()
    state: dict[str, dict[str, Any]] = {}
    for event in rows:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if canonical_hash(payload) != event.payload_hash:
            raise ValueError("FAIRNESS_EVIDENCE_MISMATCH")
        outcome = payload.get("outcome")
        client_id = payload.get("client_id")
        interaction_id = payload.get("interaction_id")
        if outcome not in _KNOWN_OUTCOMES or not isinstance(client_id, str) or not client_id.strip():
            raise ValueError("FAIRNESS_EVIDENCE_MISMATCH")
        if not isinstance(interaction_id, str) or not interaction_id.strip():
            raise ValueError("FAIRNESS_EVIDENCE_MISMATCH")
        interaction = session.get(Interaction, interaction_id)
        if interaction is None or interaction.client_id != client_id:
            raise ValueError("FAIRNESS_EVIDENCE_MISMATCH")
        if usecase_id and interaction.use_case != usecase_id:
            continue
        row = state.setdefault(client_id, {"prioritised": False, "latest": "", "product": ""})
        if outcome == "reversal":
            if row["latest"] == "do not contact":
                row["latest"] = "reversal"
            continue
        row["prioritised"] = True
        row["latest"] = outcome
        row["product"] = str(payload.get("product_id") or "").strip()
    return state


def _flags(row: Mapping[str, Any] | None) -> tuple[bool, bool, bool]:
    if not row or not row.get("prioritised"):
        return False, False, False
    contacted = row.get("latest") == "recommend"
    recommended = contacted and bool(row.get("product"))
    return recommended, contacted, True


def _risk_owners(session: Session, usecase_id: str, client_ids: Sequence[str]) -> tuple[str, ...]:
    from kognita.models import Interaction
    from kognita.usecases import load_register

    register = load_register(session)
    ids: list[str] = [usecase_id] if usecase_id else []
    if not ids:
        rows = session.exec(select(Interaction)).all()
        for row in rows:
            if row.client_id in client_ids and row.use_case:
                ids.append(row.use_case)
    owners = []
    for ident in dict.fromkeys(ids):
        entry = register.get(ident)
        owner = (entry.risk_owner if entry is not None else "").strip()
        if owner:
            owners.append(owner)
    return tuple(dict.fromkeys(owners))


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def _above(hits: int, total: int, overall_hits: int, overall_total: int, threshold: float) -> bool:
    if total <= 0 or overall_total <= 0:
        return False
    difference = abs(hits * overall_total - overall_hits * total)
    return difference > threshold * total * overall_total


def fairness_report(
    session: Session,
    *,
    segments: Mapping[str, Mapping[str, str]],
    disparity_threshold: float,
    proxy_attributes: Sequence[str] = (),
    evidence=None,
    usecase_id: str = "",
    register_monitored: bool = False,
) -> FairnessReport:
    """Rates from origination evidence, flagged when a disparity is above the threshold.

    The population is the bank's segment map. Clients who were originated but
    left off a map are reported as unassigned, so omitting a group does not
    hide it. ``register_monitored`` records a disparity metric only when a
    general monitored-metric register exists. The 0.5 gates do not provide one.
    """
    parsed, proxies = _segments(segments, proxy_attributes)
    threshold = _threshold(disparity_threshold)
    standing = _originations(session, usecase_id.strip())
    rates: list[dict[str, Any]] = []
    disparities: list[dict[str, Any]] = []
    for name, mapping in parsed.items():
        population = dict(mapping)
        for client_id in standing:
            if client_id not in population:
                population[client_id] = _UNASSIGNED
        grouped: dict[str, list[str]] = {}
        for client_id, value in population.items():
            grouped.setdefault(value, []).append(client_id)
        totals = [0, 0, 0]
        per_value: list[tuple[str, list[str], tuple[int, int, int]]] = []
        for value, clients in grouped.items():
            counts = [0, 0, 0]
            for client_id in clients:
                flags = _flags(standing.get(client_id))
                for index, flag in enumerate(flags):
                    if flag:
                        counts[index] += 1
                        totals[index] += 1
            per_value.append((value, clients, (counts[0], counts[1], counts[2])))
        overall_total = len(population)
        proxy = name in proxies
        for value, clients, counts in sorted(per_value, key=lambda item: item[0]):
            row = {
                "segment": name,
                "value": value,
                "proxy": proxy,
                "clients": len(clients),
                "recommendation_rate": _rate(counts[0], len(clients)),
                "contact_rate": _rate(counts[1], len(clients)),
                "prioritisation_rate": _rate(counts[2], len(clients)),
            }
            rates.append(row)
            for index, rate_name in enumerate(_RATE_NAMES):
                if not _above(counts[index], len(clients), totals[index], overall_total, threshold):
                    continue
                disparities.append({
                    "segment": name,
                    "value": value,
                    "proxy": proxy,
                    "rate": rate_name,
                    "segment_rate": row[rate_name],
                    "overall_rate": _rate(totals[index], overall_total),
                    "difference": abs(row[rate_name] - _rate(totals[index], overall_total)),
                })
    owners = _risk_owners(session, usecase_id.strip(), tuple(standing))
    flagged = bool(disparities)
    if flagged:
        if not owners:
            raise ValueError("FAIRNESS_RISK_OWNER_REQUIRED")
        if evidence is None:
            raise ValueError("FAIRNESS_FLAG_REQUIRED")
        event = evidence.emit(
            session,
            correlation_id=f"fairness:{usecase_id.strip() or 'report'}",
            event_type=EventType.REVIEW,
            actor_type=ActorType.SYSTEM,
            actor_id="evaluation",
            payload={
                "action": "REVIEW_FLAG",
                "reason": "FAIRNESS_DISPARITY",
                "risk_owner": owners[0],
                "recipients": list(owners),
                "usecase_id": usecase_id.strip(),
                "threshold": threshold,
                "disparities": disparities,
                "citation": "Evaluation evidence: fairness disparity",
            },
        )
        if event.payload.get("action") != "REVIEW_FLAG" or event.payload.get("reason") != "FAIRNESS_DISPARITY":
            raise ValueError("FAIRNESS_FLAG_REQUIRED")
        if not any(owner in (event.payload.get("recipients") or []) for owner in owners):
            raise ValueError("FAIRNESS_FLAG_REQUIRED")
    monitored = None
    if register_monitored:
        # Decision quality and reliability are the only 0.5 monitored metrics.
        # There is no register for an arbitrary disparity with tiered thresholds.
        monitored = None
    return FairnessReport(
        rates=tuple(rates),
        disparities=tuple(disparities),
        threshold=threshold,
        proxy_attributes=proxies,
        risk_owners=owners if flagged else owners,
        flagged=flagged,
        monitored_metric=monitored,
    )


__all__ = [
    "EvaluationResult",
    "FairnessReport",
    "check_activation",
    "fairness_report",
    "record_evaluation",
]
