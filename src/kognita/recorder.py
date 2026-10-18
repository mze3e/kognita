"""The flight recorder.

The dashboard, the run export, the alerts, and the effectiveness rates are
read from the evidence chain. A run the chain does not name is not listed.
A rate with no denominator is unmeasured.

Cycle time, decision quality, and reliability stay in
:mod:`kognita.outcomes`. This module does not copy them into another store.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from sqlmodel import Session, select

from kognita.evidence import ChainBreak, export_chain, verify_chain
from kognita.exceptions import KognitaError
from kognita.models import EvidenceEvent, as_utc, utcnow
from kognita.usecases import load_register
from kognita.vocabulary import EventType, Outcome

CITE_RUN = "Flight recorder: a run is the evidence that names it"
CITE_EXPORT = "Flight recorder: a run export is the evidence chain"
CITE_BUDGET = "Flight recorder: budget exceeded"
CITE_DENIALS = "Flight recorder: repeated denials from one actor"
CITE_CHAIN = "Flight recorder: chain break detected"
CITE_DRIFT = "Flight recorder: classifier confidence drift"
CITE_EFFECT = "Flight recorder: effectiveness rates are read from evidence"
CITE_UNMEASURED = "Flight recorder: a missing measurement is unmeasured"
CITE_DECISION = "Flight recorder: a decision is the evidence event"
CITE_DENIAL = "Flight recorder: a denial is a decision outcome"
CITE_LATENCY = (
    "Flight recorder: latency is measured beside the decision, not stored on it"
)
CITE_TOKENS = "Flight recorder: token use is the model-call evidence"
CITE_TOOL = "Flight recorder: a tool failure is the tool-call evidence"
CITE_INJECTION = "Flight recorder: an injection attempt is not a recorded event"
CITE_ENTITLEMENT = "Flight recorder: an entitlement block is not a recorded event"

#: Three denials by one actor is the alert. Two is not yet repeated.
REPEATED_DENIALS = 3
#: A series shorter than this is unmeasured. It is not treated as steady.
CONFIDENCE_DRIFT_MIN_SAMPLE = 4
#: Later-half mean at least this far below the earlier-half mean, on the
#: classifier's 0 to 1 scale. Readings from two classifier versions are not
#: one series.
CONFIDENCE_DRIFT_DROP = 0.05
ALERT_DEFINITION_VERSION = 1
EFFECTIVENESS_DEFINITION_VERSION = 1
RECENT_RUNS = 20

EFFECTIVENESS_DEFINITION = (
    "RM acceptance is the share of RM final decisions whose outcome is approve. "
    "Edit is the share whose outcome is modify. "
    "Rejection is the share whose outcome is reject. "
    "Wrong-client is the share of opened interactions that have a "
    "wrong-client targeting outcome. "
    "A rate with no denominator is unmeasured. "
    "An RM final decision whose interaction was not opened in evidence is not "
    "attributed, and the rate stays unmeasured."
)

_OUTCOMES = {item.value for item in Outcome}
_PDP_ACTORS = frozenset({"governance-pdp", "system"})
_RM_OUTCOMES = frozenset({"approve", "modify", "reject", "escalate"})


class RecorderError(KognitaError):
    """A flight-recorder read could not be answered from evidence."""


def recent_runs(
    session: Session,
    *,
    actor: str | None = None,
    tool: str | None = None,
    outcome: str | None = None,
    limit: int = RECENT_RUNS,
    as_of: datetime | None = None,
) -> tuple[dict[str, Any], ...]:
    """Recent runs, newest first, filtered by actor, tool, and outcome.

    A filter matches when that value is on the run. ``limit`` below 1 is
    refused: an empty page would hide runs that are there.
    """
    if limit < 1:
        raise RecorderError(
            f"the flight recorder lists at least one run ({CITE_RUN})"
        )
    events = _events(session, as_of)
    rows = [_summarise(run_id, owned) for run_id, owned in _runs(events).items()]
    wanted_actor = (actor or "").strip()
    wanted_tool = (tool or "").strip()
    wanted_outcome = (outcome or "").strip()
    selected = [
        row
        for row in rows
        if (not wanted_actor or wanted_actor in row["actors"])
        and (not wanted_tool or wanted_tool in row["tools"])
        and (not wanted_outcome or wanted_outcome in row["outcomes"])
    ]
    selected.sort(key=lambda row: row["last_sequence"], reverse=True)
    return tuple(selected[:limit])


def show_run(
    session: Session, run_id: str, *, as_of: datetime | None = None
) -> dict[str, Any]:
    """One run: each decision, its citations, and the evidence that names it."""
    name = run_id.strip()
    events = _events(session, as_of)
    owned = _runs(events).get(name)
    if not owned:
        raise RecorderError(f"run {name} is not in the evidence ({CITE_RUN})")
    summary = _summarise(name, owned)
    decisions = []
    evidence = []
    for event in owned:
        payload = _payload(event)
        evidence.append(_event_view(event, payload))
        if event.event_type != EventType.POLICY_DECISION:
            continue
        checks = _checks(payload)
        decisions.append(
            {
                "sequence": event.sequence,
                "correlation_id": event.correlation_id,
                "recorded_at": _recorded(event),
                "outcome": payload.get("outcome"),
                "actor": _decision_actor(event, payload),
                "tool": _tool_of(payload),
                "event_hash": event.event_hash,
                "citations": [
                    {
                        "check": item.get("check"),
                        "regime": item.get("regime"),
                        "result": item.get("result"),
                        "citation": item.get("citation"),
                        "policy_id": item.get("policy_id"),
                    }
                    for item in checks
                ],
                "citations_recorded": all(
                    isinstance(item.get("citation"), str) and item.get("citation")
                    for item in checks
                )
                and bool(checks),
            }
        )
    summary["decisions"] = decisions
    summary["evidence"] = evidence
    summary["citation"] = CITE_RUN
    return summary


def export_run(session: Session, run_id: str) -> dict[str, Any]:
    """The chain as a self-verifying artifact, with this run marked.

    The chain is not truncated. A slice that starts mid-chain cannot be
    verified. Events that are not part of the run stay in the artifact and
    are not marked of interest. A run the chain does not name is refused.
    """
    name = run_id.strip()
    events = _events(session, None)
    owned = _runs(events).get(name)
    if not owned:
        raise RecorderError(f"run {name} is not in the evidence ({CITE_EXPORT})")
    artifact = export_chain(session)
    sequences = {event.sequence for event in owned}
    for row in artifact["events"]:
        row["of_interest"] = row["sequence"] in sequences
    artifact["interest_count"] = sum(1 for row in artifact["events"] if row["of_interest"])
    artifact["filters"] = {"since": None, "correlation_id": None, "run_id": name}
    artifact["citation"] = CITE_EXPORT
    return artifact


def alerts(session: Session, *, as_of: datetime | None = None) -> dict[str, Any]:
    """Budget, repeated denials, chain break, and classifier confidence drift.

    Each alert says whether it fired and whether it was measured. A series
    that was not measured is unmeasured. It is not reported as clear.
    """
    events = _events(session, as_of)
    try:
        checked = verify_chain(session)
        chain: dict[str, Any] = {
            "kind": "chain break",
            "fired": False,
            "measured": True,
            "checked": checked,
            "detail": "",
            "citation": CITE_CHAIN,
        }
    except ChainBreak as exc:
        chain = {
            "kind": "chain break",
            "fired": True,
            "measured": True,
            "sequence": exc.sequence,
            "reason": exc.reason,
            "detail": exc.reason,
            "citation": CITE_CHAIN,
        }
    injections, blocks = _special(events)
    unmeasured = []
    if not injections:
        unmeasured.append(
            {"signal": "injection", "detail": "unmeasured", "citation": CITE_INJECTION}
        )
    if not blocks:
        unmeasured.append(
            {
                "signal": "entitlement",
                "detail": "unmeasured",
                "citation": CITE_ENTITLEMENT,
            }
        )
    return {
        "definition_version": ALERT_DEFINITION_VERSION,
        "citation": CITE_UNMEASURED,
        "alerts": [
            _budget_alert(events),
            _denial_alert(events),
            chain,
            _drift_alert(events),
        ],
        "injection_attempts": injections,
        "entitlement_blocks": blocks,
        "unmeasured": unmeasured,
    }


def effectiveness(
    session: Session,
    *,
    usecase_id: str | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """RM acceptance, edit, rejection, and wrong-client rates per use case.

    The rates are counts over evidence. The register supplies the use-case
    list and the recorded primary-metric target. It does not supply the rates.
    """
    at = _aware(as_of or utcnow())
    events = _events(session, at)
    register = load_register(session, as_of=at)
    opened = _opened(events)
    names: list[str] = []
    if usecase_id is not None:
        names.append(usecase_id.strip())
    else:
        names.extend(entry.usecase_id for entry in register.entries)
        for use_case in opened.values():
            if use_case and use_case not in names:
                names.append(use_case)
    reviews, unattributed_reviews = _rm_reviews(events, opened)
    wrong, unattributed_wrong = _wrong_client(events, opened)
    rows = [
        _effectiveness_row(
            name,
            register.get(name),
            reviews.get(name),
            wrong.get(name),
            sum(1 for use_case in opened.values() if use_case == name),
        )
        for name in names
    ]
    any_unmeasured = any(
        not rate["measured"] for row in rows for rate in row["rates"].values()
    )
    detail = ""
    if not rows or any_unmeasured or unattributed_reviews or unattributed_wrong:
        detail = "unmeasured"
    return {
        "definition_version": EFFECTIVENESS_DEFINITION_VERSION,
        "definition": EFFECTIVENESS_DEFINITION,
        "citation": CITE_EFFECT,
        "use_cases": rows,
        "unattributed_reviews": unattributed_reviews,
        "unattributed_wrong_client": unattributed_wrong,
        "detail": detail,
    }


def telemetry_signals(snapshot: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Signals one snapshot can support.

    A fact the snapshot does not carry is unmeasured. A zero is emitted only
    when the event recorded the opposite, such as a decision that was not a
    denial.
    """
    kind = str(snapshot.get("kind") or "evidence")
    if kind == "latency":
        seconds = snapshot.get("seconds")
        measured = (
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and float(seconds) >= 0
        )
        return (
            _signal(
                "latency",
                measured=measured,
                value=float(seconds) if measured else None,
                citation=CITE_LATENCY,
                attributes={"correlation_id": str(snapshot.get("correlation_id") or "")},
            ),
        )
    event_type = str(snapshot.get("event_type") or "")
    payload = snapshot.get("payload") if isinstance(snapshot.get("payload"), dict) else {}
    attributes = {
        "correlation_id": str(snapshot.get("correlation_id") or ""),
        "sequence": snapshot.get("sequence"),
    }
    if event_type == EventType.POLICY_DECISION.value:
        outcome = payload.get("outcome")
        measured = isinstance(outcome, str) and outcome in _OUTCOMES
        signals = [
            _signal(
                "decision",
                measured=measured,
                value=1 if measured else None,
                citation=CITE_DECISION,
                attributes={**attributes, "outcome": outcome if measured else ""},
            )
        ]
        if measured and outcome == Outcome.DENY.value:
            signals.append(
                _signal(
                    "denial",
                    measured=True,
                    value=1,
                    citation=CITE_DENIAL,
                    attributes=attributes,
                )
            )
        elif measured:
            signals.append(
                _signal(
                    "denial",
                    measured=True,
                    value=0,
                    citation=CITE_DENIAL,
                    attributes=attributes,
                )
            )
        else:
            signals.append(
                _signal("denial", measured=False, value=None, citation=CITE_DENIAL)
            )
        signals.extend(_special_signals(payload, attributes, absent="unmeasured"))
        return tuple(signals)
    if event_type == EventType.MODEL_CALL.value:
        tokens = payload.get("tokens")
        measured = isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0
        return (
            _signal(
                "tokens",
                measured=measured,
                value=tokens if measured else None,
                citation=CITE_TOKENS,
                attributes=attributes,
            ),
            *_special_signals(payload, attributes, absent="omit"),
        )
    if event_type == EventType.TOOL_CALL.value:
        outcome = payload.get("outcome")
        if not isinstance(outcome, str) or outcome not in _OUTCOMES:
            return (
                _signal(
                    "tool_failure",
                    measured=False,
                    value=None,
                    citation=CITE_TOOL,
                    attributes=attributes,
                ),
                *_special_signals(payload, attributes, absent="omit"),
            )
        failed = outcome != Outcome.ALLOW.value
        return (
            _signal(
                "tool_failure",
                measured=True,
                value=1 if failed else 0,
                citation=CITE_TOOL,
                attributes={**attributes, "tool": _tool_of(payload), "outcome": outcome},
            ),
            *_special_signals(payload, attributes, absent="omit"),
        )
    return tuple(_special_signals(payload, attributes, absent="omit"))


def _special_signals(
    payload: Mapping[str, Any],
    attributes: dict[str, Any],
    *,
    absent: str,
) -> tuple[dict[str, Any], ...]:
    """Injection and entitlement. Absent means unmeasured, or omitted.

    A decision always carries both. Any other event carries one only when
    that event recorded it. A missing record is not a count of zero.
    """
    checks = payload.get("checks")
    rows = checks if isinstance(checks, list) else []
    injected = payload.get("injection") is True or any(
        isinstance(item, dict) and item.get("check") == "INJECTION" for item in rows
    )
    blocked = any(
        isinstance(item, dict)
        and item.get("check") == "ENTITLEMENT"
        and item.get("result") == "fail"
        for item in rows
    )
    found: list[dict[str, Any]] = []
    if injected:
        found.append(
            _signal(
                "injection",
                measured=True,
                value=1,
                citation=CITE_INJECTION,
                attributes=attributes,
            )
        )
    elif absent == "unmeasured":
        found.append(
            _signal(
                "injection",
                measured=False,
                value=None,
                citation=CITE_INJECTION,
                attributes=attributes,
            )
        )
    if blocked:
        found.append(
            _signal(
                "entitlement",
                measured=True,
                value=1,
                citation=CITE_ENTITLEMENT,
                attributes=attributes,
            )
        )
    elif absent == "unmeasured":
        found.append(
            _signal(
                "entitlement",
                measured=False,
                value=None,
                citation=CITE_ENTITLEMENT,
                attributes=attributes,
            )
        )
    return tuple(found)


def _signal(
    name: str,
    *,
    measured: bool,
    value: float | int | None,
    citation: str,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "measured": measured,
        "value": value,
        "detail": "" if measured else "unmeasured",
        "citation": citation,
        "attributes": attributes or {},
    }


def _runs(events: Sequence[EvidenceEvent]) -> dict[str, list[EvidenceEvent]]:
    """Events of each run the chain names, in chain order."""
    correlations: dict[str, set[str]] = {}
    for event in events:
        payload = _payload(event)
        for run_id in _run_ids(event, payload):
            bucket = correlations.setdefault(run_id, set())
            bucket.add(event.correlation_id)
            if payload.get("action") == "ATTACHED":
                attached = str(payload.get("correlation_id") or "").strip()
                if attached:
                    bucket.add(attached)
    grouped: dict[str, list[EvidenceEvent]] = {run_id: [] for run_id in correlations}
    for event in events:
        payload = _payload(event)
        named = _run_ids(event, payload)
        for run_id, ids in correlations.items():
            if event.correlation_id in ids or run_id in named:
                grouped[run_id].append(event)
    return grouped


def _summarise(run_id: str, owned: Sequence[EvidenceEvent]) -> dict[str, Any]:
    actors: set[str] = set()
    tools: set[str] = set()
    outcomes: list[str] = []
    for event in owned:
        payload = _payload(event)
        actors.update(_actors(event, payload))
        tool = _tool_of(payload)
        if tool:
            tools.add(tool)
        if event.event_type == EventType.POLICY_DECISION:
            outcome = payload.get("outcome")
            if isinstance(outcome, str) and outcome:
                outcomes.append(outcome)
    last = owned[-1]
    return {
        "run_id": run_id,
        "actors": sorted(actors),
        "tools": sorted(tools),
        "outcomes": outcomes,
        "latest_outcome": outcomes[-1] if outcomes else "",
        "decisions": len(outcomes),
        "last_sequence": last.sequence,
        "recorded_at": _recorded(last),
        "citation": CITE_RUN,
    }


def _budget_alert(events: Sequence[EvidenceEvent]) -> dict[str, Any]:
    found = []
    for event in events:
        payload = _payload(event)
        citations = [
            str(check.get("citation") or "")
            for check in _checks(payload)
            if check.get("check") == "BUDGET" and check.get("result") == "fail"
        ]
        if payload.get("budget") == "wall_clock_seconds":
            citations.append("wall_clock_seconds")
        if not citations:
            continue
        found.append(
            {
                "sequence": event.sequence,
                "correlation_id": event.correlation_id,
                "run_id": next(iter(_run_ids(event, payload)), ""),
                "citations": citations,
            }
        )
    return {
        "kind": "budget exceeded",
        "fired": bool(found),
        "measured": True,
        "detail": "",
        "events": found,
        "citation": CITE_BUDGET,
    }


def _denial_alert(events: Sequence[EvidenceEvent]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    unattributed = 0
    for event in events:
        if event.event_type != EventType.POLICY_DECISION:
            continue
        payload = _payload(event)
        if payload.get("outcome") != Outcome.DENY.value:
            continue
        actor = _decision_actor(event, payload)
        if not actor:
            unattributed += 1
            continue
        counts[actor] = counts.get(actor, 0) + 1
    actors = [
        {"actor": actor, "denials": count, "measured": True}
        for actor, count in sorted(counts.items())
        if count >= REPEATED_DENIALS
    ]
    fired = bool(actors) or unattributed > 0
    measured = unattributed == 0
    return {
        "kind": "repeated denials",
        "fired": fired,
        "measured": measured,
        "detail": "" if measured else "unmeasured",
        "threshold": REPEATED_DENIALS,
        "actors": actors,
        "unattributed": unattributed,
        "citation": CITE_DENIALS,
    }


def _drift_alert(events: Sequence[EvidenceEvent]) -> dict[str, Any]:
    series: dict[tuple[str, str], list[float]] = {}
    unread = 0
    for event in events:
        for reading in _confidences(event):
            model, version, confidence = reading
            if confidence is None or not model or not version:
                unread += 1
                continue
            series.setdefault((model, version), []).append(confidence)
    rows = []
    fired = False
    short = False
    for (model, version), values in sorted(series.items()):
        sample = len(values)
        if sample < CONFIDENCE_DRIFT_MIN_SAMPLE:
            short = True
            rows.append(
                {
                    "model": model,
                    "version": version,
                    "sample": sample,
                    "measured": False,
                    "fired": False,
                    "detail": "unmeasured",
                    "earlier_mean": None,
                    "later_mean": None,
                }
            )
            continue
        mid = sample // 2
        earlier = _mean(values[:mid])
        later = _mean(values[mid:])
        declined = (
            earlier is not None
            and later is not None
            and later <= earlier - CONFIDENCE_DRIFT_DROP
        )
        fired = fired or declined
        rows.append(
            {
                "model": model,
                "version": version,
                "sample": sample,
                "measured": True,
                "fired": declined,
                "detail": "",
                "earlier_mean": earlier,
                "later_mean": later,
                "drop": CONFIDENCE_DRIFT_DROP,
            }
        )
    measured = unread == 0 and not short and bool(rows)
    if not rows and unread == 0:
        measured = False
    return {
        "kind": "classifier confidence drift",
        "fired": fired,
        "measured": measured,
        "detail": "" if measured else "unmeasured",
        "definition_version": ALERT_DEFINITION_VERSION,
        "minimum_sample": CONFIDENCE_DRIFT_MIN_SAMPLE,
        "drop": CONFIDENCE_DRIFT_DROP,
        "unread": unread,
        "series": rows,
        "citation": CITE_DRIFT,
    }


def _confidences(
    event: EvidenceEvent,
) -> list[tuple[str, str, float | None]]:
    payload = _payload(event)
    found: list[Mapping[str, Any]] = []
    attributes = payload.get("attributes")
    if isinstance(attributes, dict) and isinstance(attributes.get("classifier"), dict):
        found.append(attributes["classifier"])
    if isinstance(payload.get("response_classifier"), dict):
        found.append(payload["response_classifier"])
    readings = []
    for record in found:
        model = str(record.get("model") or "").strip()
        version = str(record.get("version") or "").strip()
        confidence = record.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            readings.append((model, version, None))
        else:
            readings.append((model, version, float(confidence)))
    return readings


def _special(
    events: Sequence[EvidenceEvent],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    injections = []
    blocks = []
    for event in events:
        payload = _payload(event)
        checks = _checks(payload)
        injected = payload.get("injection") is True or any(
            check.get("check") == "INJECTION" for check in checks
        )
        if injected:
            injections.append(
                {
                    "sequence": event.sequence,
                    "correlation_id": event.correlation_id,
                    "citation": CITE_INJECTION,
                }
            )
        if any(
            check.get("check") == "ENTITLEMENT" and check.get("result") == "fail"
            for check in checks
        ):
            blocks.append(
                {
                    "sequence": event.sequence,
                    "correlation_id": event.correlation_id,
                    "citation": CITE_ENTITLEMENT,
                }
            )
    return injections, blocks


def _opened(events: Sequence[EvidenceEvent]) -> dict[str, str]:
    found: dict[str, str] = {}
    for event in events:
        if event.event_type != EventType.INTERACTION:
            continue
        payload = _payload(event)
        if payload.get("action") != "OPENED":
            continue
        ident = str(payload.get("interaction_id") or event.correlation_id)
        found[ident] = str(payload.get("use_case") or "")
    return found


def _rm_reviews(
    events: Sequence[EvidenceEvent], opened: Mapping[str, str]
) -> tuple[dict[str, dict[str, Any]], int]:
    grouped: dict[str, dict[str, Any]] = {}
    unattributed = 0
    for event in events:
        if event.event_type != EventType.FINAL_DECISION:
            continue
        payload = _payload(event)
        if payload.get("kind") != "rm_review":
            continue
        outcome = str(payload.get("outcome") or "")
        interaction_id = str(payload.get("interaction_id") or "")
        if interaction_id not in opened or outcome not in _RM_OUTCOMES:
            unattributed += 1
            continue
        use_case = opened[interaction_id]
        bucket = grouped.setdefault(use_case, {"counts": {}})
        counts = bucket["counts"]
        counts[outcome] = int(counts.get(outcome, 0)) + 1
    return grouped, unattributed


def _wrong_client(
    events: Sequence[EvidenceEvent], opened: Mapping[str, str]
) -> tuple[dict[str, dict[str, int]], int]:
    grouped: dict[str, dict[str, int]] = {}
    seen: dict[str, set[str]] = {}
    unattributed = 0
    for event in events:
        if event.event_type != EventType.OUTCOME:
            continue
        payload = _payload(event)
        if payload.get("action") != "RECORDED":
            continue
        if payload.get("kind") != "wrong-client targeting":
            continue
        interaction_id = str(payload.get("interaction_id") or "")
        if interaction_id not in opened:
            unattributed += 1
            continue
        use_case = opened[interaction_id]
        bucket = grouped.setdefault(use_case, {"events": 0})
        bucket["events"] += 1
        seen.setdefault(use_case, set()).add(interaction_id)
    for use_case, interactions in seen.items():
        grouped[use_case]["interactions"] = len(interactions)
    return grouped, unattributed


def _effectiveness_row(
    usecase_id: str,
    entry: Any,
    reviews: Mapping[str, Any] | None,
    wrong: Mapping[str, int] | None,
    interactions: int,
) -> dict[str, Any]:
    counts = dict((reviews or {}).get("counts") or {})
    sample = sum(int(value) for value in counts.values())
    rates = {
        "acceptance": _rate("acceptance", counts.get("approve", 0), sample, 0),
        "edit": _rate("edit", counts.get("modify", 0), sample, 0),
        "rejection": _rate("rejection", counts.get("reject", 0), sample, 0),
    }
    wrong_events = int((wrong or {}).get("events") or 0)
    wrong_interactions = int((wrong or {}).get("interactions") or 0)
    if interactions == 0:
        wrong_rate = _unmeasured_rate("wrong-client")
    else:
        wrong_rate = {
            "name": "wrong-client",
            "value": wrong_interactions / interactions,
            "measured": True,
            "detail": "",
            "sample": interactions,
            "count": wrong_interactions,
            "events": wrong_events,
            "citation": CITE_EFFECT,
        }
    rates["wrong_client"] = wrong_rate
    primary = ""
    baseline = None
    target = None
    registered = entry is not None
    if entry is not None:
        primary = entry.primary_metric
        baseline = entry.baseline
        target = entry.target
    return {
        "usecase_id": usecase_id,
        "registered": registered,
        "primary_metric": primary,
        "baseline": baseline,
        "target": target,
        "definition_version": EFFECTIVENESS_DEFINITION_VERSION,
        "sample_reviews": sample,
        "sample_interactions": interactions,
        "rates": rates,
        "citation": CITE_EFFECT,
    }


def _rate(name: str, count: int, sample: int, unattributed: int) -> dict[str, Any]:
    if sample == 0 or unattributed:
        body = _unmeasured_rate(name)
        body["count"] = count
        body["sample"] = sample
        return body
    return {
        "name": name,
        "value": count / sample,
        "measured": True,
        "detail": "",
        "sample": sample,
        "count": count,
        "citation": CITE_EFFECT,
    }


def _unmeasured_rate(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "value": None,
        "measured": False,
        "detail": "unmeasured",
        "sample": 0,
        "count": 0,
        "citation": CITE_UNMEASURED,
    }


def _event_view(event: EvidenceEvent, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sequence": event.sequence,
        "event_type": EventType(event.event_type).value,
        "correlation_id": event.correlation_id,
        "actor_type": str(event.actor_type.value if hasattr(event.actor_type, "value") else event.actor_type),
        "actor_id": event.actor_id,
        "event_hash": event.event_hash,
        "recorded_at": _recorded(event),
        "payload": dict(payload),
    }


def _run_ids(event: EvidenceEvent, payload: Mapping[str, Any]) -> set[str]:
    found = set()
    if isinstance(event.run_id, str) and event.run_id:
        found.add(event.run_id)
    named = payload.get("run_id")
    if isinstance(named, str) and named:
        found.add(named)
    return found


def _actors(event: EvidenceEvent, payload: Mapping[str, Any]) -> set[str]:
    found = set()
    actor = _decision_actor(event, payload)
    if actor:
        found.add(actor)
    envelope = _envelope(payload)
    for key in ("principal", "agent_name"):
        value = str(envelope.get(key) or "").strip()
        if value:
            found.add(value)
    for key in ("rm", "approver"):
        value = str(payload.get(key) or "").strip()
        if value:
            found.add(value)
    return found


def _decision_actor(event: EvidenceEvent, payload: Mapping[str, Any]) -> str:
    envelope = _envelope(payload)
    agent = str(envelope.get("agent_name") or "").strip()
    if agent:
        return agent
    principal = str(envelope.get("principal") or "").strip()
    if principal:
        return principal
    actor = str(event.actor_id or "").strip()
    if actor and actor not in _PDP_ACTORS:
        return actor
    return ""


def _tool_of(payload: Mapping[str, Any]) -> str:
    envelope = _envelope(payload)
    return str(envelope.get("tool") or payload.get("tool") or "").strip()


def _envelope(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    envelope = payload.get("envelope")
    if isinstance(envelope, dict):
        return envelope
    return {}


def _checks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return []
    return [item for item in checks if isinstance(item, dict)]


def _events(session: Session, at: datetime | None) -> list[EvidenceEvent]:
    rows = list(
        session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    )
    if at is None:
        return rows
    limit = _aware(at)
    kept = []
    for event in rows:
        recorded = as_utc(event.recorded_at)
        if recorded is None or recorded <= limit:
            kept.append(event)
    return kept


def _recorded(event: EvidenceEvent) -> str | None:
    recorded = as_utc(event.recorded_at)
    if recorded is None:
        return None
    return recorded.isoformat()


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _payload(event: EvidenceEvent) -> dict[str, Any]:
    payload = event.payload
    if isinstance(payload, dict):
        return payload
    return {}


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        from datetime import timezone

        return value.replace(tzinfo=timezone.utc)
    return value


__all__ = [
    "ALERT_DEFINITION_VERSION",
    "CITE_BUDGET",
    "CITE_CHAIN",
    "CITE_DECISION",
    "CITE_DENIAL",
    "CITE_DENIALS",
    "CITE_DRIFT",
    "CITE_LATENCY",
    "CITE_TOKENS",
    "CITE_TOOL",
    "CITE_EFFECT",
    "CITE_ENTITLEMENT",
    "CITE_EXPORT",
    "CITE_INJECTION",
    "CITE_RUN",
    "CITE_UNMEASURED",
    "CONFIDENCE_DRIFT_DROP",
    "CONFIDENCE_DRIFT_MIN_SAMPLE",
    "EFFECTIVENESS_DEFINITION",
    "EFFECTIVENESS_DEFINITION_VERSION",
    "RECENT_RUNS",
    "REPEATED_DENIALS",
    "RecorderError",
    "alerts",
    "effectiveness",
    "export_run",
    "recent_runs",
    "show_run",
    "telemetry_signals",
]
