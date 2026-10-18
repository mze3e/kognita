"""Outcome events, the near-miss register, and outcome metrics.

Outcome events are client complaints, corrections, wrong-client targeting,
unsuitable suggestions, communication errors, policy breaches and operational
incidents. Each one is linked to the interaction. A complaint is refused
unless it names the output, the model version and the approver already on
that interaction.

The near-miss register aggregates blocked actions and RM rejections from
decisions, and records verifier blocks, unsafe drafts and unusual agent
behaviour. Verifier blocks have no producer in this package; they appear
when they are recorded.

Cycle time, decision quality and reliability are computed from the
interaction record. Definitions are versioned. A trend is not averaged
across two definition versions. A use case that names a primary metric and
has no measurement is a breach.

A recorded exception or a recorded human recovery counts against reliability.
A run that never finished is not treated as clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from sqlmodel import Session, select

from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.interaction import linked_correlation_ids, require_interaction
from kognita.models import (
    EvidenceEvent,
    Interaction,
    InteractionLink,
    RmReview,
    RunRecord,
    as_utc,
    utcnow,
)
from kognita.usecases import PRIMARY_METRICS, RegisteredUseCase, load_register
from kognita.vocabulary import ActorType, EventType

#: Bump this when a formula changes. Snapshots from another version are not
#: averaged into the same trend.
METRIC_DEFINITION_VERSION = 1

OUTCOME_KINDS: tuple[str, ...] = (
    "client complaint",
    "correction",
    "wrong-client targeting",
    "unsuitable suggestion",
    "communication error",
    "policy breach",
    "operational incident",
)

#: Kinds taken from decisions. Recording them again would count the same fact twice.
DERIVED_NEAR_MISS_KINDS: tuple[str, ...] = ("blocked action", "rm rejection")

RECORDED_NEAR_MISS_KINDS: tuple[str, ...] = (
    "verifier block",
    "unsafe draft",
    "unusual agent behaviour",
)

NEAR_MISS_KINDS: tuple[str, ...] = DERIVED_NEAR_MISS_KINDS + RECORDED_NEAR_MISS_KINDS

NOT_RECORDED = "not recorded"

CITE_COMPLAINT = (
    "Outcomes: a complaint links to the output, the model version and the approver"
)
CITE_OUTCOME = "Outcomes: an outcome event links to the originating interaction"
CITE_NEAR_MISS = (
    "Near misses: blocked actions, verifier blocks, RM rejections, unsafe drafts "
    "and unusual agent behaviour"
)
CITE_CYCLE = "Outcome metrics: cycle time is trigger to final decision"
CITE_QUALITY = "Outcome metrics: decision quality"
CITE_RELIABILITY = "Outcome metrics: reliability"
CITE_UNMEASURED = "Outcome metrics: a missing measurement is a breach"
CITE_VERSION = "Outcome metrics: definitions are versioned"
CITE_EXCEPTION = "Outcome metrics: reliability excludes an exception"
CITE_RECOVERY = "Outcome metrics: reliability excludes human recovery"

_CLEAN_FINALS = frozenset({"allow", "approve", "modify"})
_EDITED_PROVENANCE = "AI then human-edited"

_DEFINITIONS: dict[str, dict[str, str]] = {
    "cycle time": {
        "unit": "seconds",
        "direction": "lower",
        "statement": (
            "Mean seconds from the interaction open to the latest final decision, "
            "over interactions that have a final decision in the window. "
            "RM effort is the mean seconds from the recommendation being shown "
            "to that decision, reported beside the mean. "
            "An interaction with no final decision is not in the mean. "
            "A final decision with no usable timestamps leaves the metric unmeasured. "
            "No final decision means the metric is not measured."
        ),
        "citation": CITE_CYCLE,
    },
    "decision quality": {
        "unit": "share",
        "direction": "higher",
        "statement": (
            "Share of decided interactions with no edit, no rejection, no "
            "escalation, and no outcome event. "
            "Edit rate, rejection rate, escalation rate, and outcome events per "
            "interaction are reported beside the share. "
            "No decided interaction means the metric is not measured."
        ),
        "citation": CITE_QUALITY,
    },
    "reliability": {
        "unit": "share",
        "direction": "higher",
        "statement": (
            "Share of runs attached to an interaction of the use case that "
            "finished with an allow and without a denial, a budget breach, a "
            "recorded exception, or a recorded human recovery. "
            "A run still suspended, or with no allow, is not clean. "
            "No attached run means the metric is not measured."
        ),
        "citation": CITE_RELIABILITY,
    },
}


class OutcomeError(KognitaError):
    """An outcome, a near miss, or a metric snapshot could not be recorded."""


@dataclass(frozen=True)
class MetricValue:
    """One metric for one use case, under one definition version."""

    metric: str
    definition_version: int
    definition: str
    unit: str
    direction: str
    citation: str
    value: float | None
    measured: bool
    sample: int
    components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "definition_version": self.definition_version,
            "definition": self.definition,
            "unit": self.unit,
            "direction": self.direction,
            "citation": self.citation,
            "value": self.value,
            "measured": self.measured,
            "sample": self.sample,
            "components": self.components,
        }


@dataclass(frozen=True)
class UseCaseMetrics:
    """The three metrics, and the primary compared with the registered target."""

    usecase_id: str
    registered: bool
    status: str
    primary_metric: str
    baseline: float | None
    target: float | None
    definition_version: int
    compared: bool
    measured: bool
    value: float | None
    breached: bool
    detail: str
    citation: str
    accountable_owner: str
    metrics: dict[str, MetricValue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "usecase_id": self.usecase_id,
            "registered": self.registered,
            "status": self.status,
            "primary_metric": self.primary_metric,
            "baseline": self.baseline,
            "target": self.target,
            "definition_version": self.definition_version,
            "compared": self.compared,
            "measured": self.measured,
            "value": self.value,
            "breached": self.breached,
            "detail": self.detail,
            "citation": self.citation,
            "accountable_owner": self.accountable_owner,
            "metrics": {name: item.to_dict() for name, item in self.metrics.items()},
        }


@dataclass(frozen=True)
class NearMissCount:
    """How many near misses share a use case, an agent and a model version."""

    usecase_id: str
    agent: str
    model_version: str
    kind: str
    count: int
    agent_recorded: bool
    model_version_recorded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "usecase_id": self.usecase_id,
            "agent": self.agent,
            "model_version": self.model_version,
            "kind": self.kind,
            "count": self.count,
            "agent_recorded": self.agent_recorded,
            "model_version_recorded": self.model_version_recorded,
        }


def record_outcome(
    session: Session,
    interaction_id: str,
    *,
    kind: str,
    summary: str,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    output_hash: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record an outcome event linked to the interaction.

    The output, the model version and the approver are taken from evidence
    already on the interaction. A complaint that cannot name all three is
    refused and nothing is appended. The same bar applies to every outcome
    kind: an event that does not trace to the output is not recorded.
    """
    at = now or utcnow()
    label = kind.strip()
    if label not in OUTCOME_KINDS:
        raise OutcomeError(
            f"outcome kind {kind!r} is not a recorded outcome ({CITE_OUTCOME})"
        )
    text = summary.strip()
    basis = citation.strip()
    who = actor_id.strip()
    if not text or not basis or not who:
        raise OutcomeError(
            "an outcome event names what happened, who recorded it, and the "
            f"rule it cites ({CITE_OUTCOME})"
        )
    interaction = require_interaction(session, interaction_id)
    link = _output_link(session, interaction, output_hash=output_hash.strip(), at=at)
    if link is None:
        rule = CITE_COMPLAINT if label == "client complaint" else CITE_OUTCOME
        raise OutcomeError(
            "the outcome does not link to one output, its model version and "
            f"the approver ({rule})"
        )
    rule = CITE_COMPLAINT if label == "client complaint" else CITE_OUTCOME
    payload = {
        "action": "RECORDED",
        "kind": label,
        "summary": text,
        "interaction_id": interaction.id,
        "usecase_id": interaction.use_case,
        "output_hash": link["output_hash"],
        "model": link["model"],
        "model_version": link["model_version"],
        "provider": link["provider"],
        "prompt_template_version": link["prompt_template_version"],
        "approver": link["approver"],
        "citation": basis,
        "rule": rule,
    }
    event = evidence.emit(
        session,
        correlation_id=interaction.id,
        event_type=EventType.OUTCOME,
        actor_type=ActorType.HUMAN,
        actor_id=who,
        payload=payload,
    )
    _enqueue(
        session,
        interaction.id,
        evidence=evidence,
        reasons=(f"outcome event: {label}",),
        risk="high",
        citation=rule,
        actor_id=who,
        now=at,
    )
    body = dict(payload)
    body["sequence"] = event.sequence
    return body


def record_near_miss(
    session: Session,
    *,
    kind: str,
    summary: str,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    interaction_id: str = "",
    usecase_id: str = "",
    agent: str = "",
    model_version: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record a verifier block, an unsafe draft, or unusual agent behaviour.

    Blocked actions and RM rejections are read from decisions. Recording
    those kinds here is refused so the register does not count them twice.
    """
    at = now or utcnow()
    label = kind.strip()
    if label in DERIVED_NEAR_MISS_KINDS:
        raise OutcomeError(
            f"{label} is taken from decisions and is not recorded again "
            f"({CITE_NEAR_MISS})"
        )
    if label not in RECORDED_NEAR_MISS_KINDS:
        raise OutcomeError(
            f"near miss kind {kind!r} is not a recorded near miss ({CITE_NEAR_MISS})"
        )
    text = summary.strip()
    basis = citation.strip()
    who = actor_id.strip()
    if not text or not basis or not who:
        raise OutcomeError(
            "a near miss names what happened, who recorded it, and the rule "
            f"it cites ({CITE_NEAR_MISS})"
        )
    interaction = interaction_id.strip()
    usecase = usecase_id.strip()
    if interaction:
        row = require_interaction(session, interaction)
        usecase = usecase or row.use_case
    elif not usecase:
        raise OutcomeError(
            "a near miss names the interaction or the use case "
            f"({CITE_NEAR_MISS})"
        )
    agent_name = agent.strip()
    version = model_version.strip()
    payload = {
        "action": "RECORDED",
        "kind": label,
        "summary": text,
        "interaction_id": interaction,
        "usecase_id": usecase,
        "agent": agent_name,
        "model_version": version,
        "agent_recorded": bool(agent_name),
        "model_version_recorded": bool(version),
        "citation": basis,
        "rule": CITE_NEAR_MISS,
    }
    event = evidence.emit(
        session,
        correlation_id=interaction or f"usecase:{usecase}",
        event_type=EventType.NEAR_MISS,
        actor_type=ActorType.HUMAN,
        actor_id=who,
        payload=payload,
    )
    if interaction:
        _enqueue(
            session,
            interaction,
            evidence=evidence,
            reasons=(f"near miss: {label}",),
            risk="high",
            citation=CITE_NEAR_MISS,
            actor_id=who,
            now=at,
        )
    body = dict(payload)
    body["sequence"] = event.sequence
    return body


def record_exception(
    session: Session,
    interaction_id: str,
    *,
    run_id: str,
    summary: str,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record an exception on a run. Reliability does not count that run as clean."""
    return _record_run_fault(
        session,
        interaction_id,
        run_id=run_id,
        summary=summary,
        evidence=evidence,
        actor_id=actor_id,
        citation=citation,
        action="EXCEPTION",
        rule=CITE_EXCEPTION,
        now=now,
    )


def record_human_recovery(
    session: Session,
    interaction_id: str,
    *,
    run_id: str,
    summary: str,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record that a person recovered a run. That run is not a clean completion."""
    return _record_run_fault(
        session,
        interaction_id,
        run_id=run_id,
        summary=summary,
        evidence=evidence,
        actor_id=actor_id,
        citation=citation,
        action="HUMAN_RECOVERY",
        rule=CITE_RECOVERY,
        now=now,
    )


def near_miss_register(
    session: Session, *, as_of: datetime | None = None
) -> tuple[NearMissCount, ...]:
    """Near misses grouped by use case, agent and model version."""
    at = _aware(as_of or utcnow())
    events = _events_until(session, at)
    interactions = _interactions_until(session, at)
    links = list(session.exec(select(InteractionLink)).all())
    counts: dict[tuple[str, str, str, str, bool, bool], int] = {}

    def add(
        usecase: str,
        agent: str,
        version: str,
        kind: str,
        *,
        agent_recorded: bool,
        version_recorded: bool,
    ) -> None:
        agent_label = agent if agent_recorded and agent else NOT_RECORDED
        version_label = version if version_recorded and version else NOT_RECORDED
        key = (
            usecase,
            agent_label,
            version_label,
            kind,
            agent_recorded and bool(agent),
            version_recorded and bool(version),
        )
        counts[key] = counts.get(key, 0) + 1

    for event in events:
        payload = _payload(event)
        if event.event_type == EventType.NEAR_MISS and payload.get("action") == "RECORDED":
            add(
                str(payload.get("usecase_id") or ""),
                str(payload.get("agent") or ""),
                str(payload.get("model_version") or ""),
                str(payload.get("kind") or ""),
                agent_recorded=bool(payload.get("agent_recorded")),
                version_recorded=bool(payload.get("model_version_recorded")),
            )
            continue
        if event.event_type == EventType.FINAL_DECISION and payload.get("outcome") == "reject":
            interaction = _interaction_by_id(interactions, str(payload.get("interaction_id") or ""))
            if interaction is None:
                continue
            agent, agent_ok, version, version_ok = _agent_and_version(
                events, interaction, links
            )
            add(
                interaction.use_case,
                agent,
                version,
                "rm rejection",
                agent_recorded=agent_ok,
                version_recorded=version_ok,
            )
            continue
        if _is_blocked(event):
            attributed = _blocked_attribution(event, events, interactions, links)
            if attributed is None:
                continue
            usecase, agent, agent_ok, version, version_ok = attributed
            add(
                usecase,
                agent,
                version,
                "blocked action",
                agent_recorded=agent_ok,
                version_recorded=version_ok,
            )
    rows = [
        NearMissCount(
            usecase_id=key[0],
            agent=key[1],
            model_version=key[2],
            kind=key[3],
            count=count,
            agent_recorded=key[4],
            model_version_recorded=key[5],
        )
        for key, count in counts.items()
    ]
    return tuple(
        sorted(rows, key=lambda item: (item.usecase_id, item.kind, item.agent, item.model_version))
    )


def outcome_metrics(
    session: Session,
    *,
    as_of: datetime | None = None,
    usecase_id: str | None = None,
) -> tuple[UseCaseMetrics, ...]:
    """Cycle time, decision quality and reliability for each use case.

    The primary metric is compared with the baseline and target on the
    register. A named primary with no measurement is a breach. Baseline is
    reported; the comparison that breaches is the target.
    """
    at = _aware(as_of or utcnow())
    register = load_register(session, as_of=at)
    events = _events_until(session, at)
    interactions = _interactions_until(session, at)
    links = list(session.exec(select(InteractionLink)).all())
    reviews = list(session.exec(select(RmReview)).all())
    runs = {row.id: row for row in session.exec(select(RunRecord)).all()}
    by_id = {item.usecase_id: item for item in register.entries}
    wanted: list[str] = []
    if usecase_id is not None:
        wanted.append(usecase_id)
    else:
        wanted.extend(entry.usecase_id for entry in register.entries)
        for item in interactions:
            if item.use_case not in by_id and item.use_case not in wanted:
                wanted.append(item.use_case)
    reports: list[UseCaseMetrics] = []
    for name in wanted:
        entry = by_id.get(name)
        owned = [item for item in interactions if item.use_case == name]
        metrics = {
            metric: _one_metric(
                metric,
                owned,
                events,
                links,
                reviews,
                runs,
                at,
            )
            for metric in PRIMARY_METRICS
        }
        reports.append(_pack(name, entry, metrics))
    return tuple(reports)


def interaction_measurement(
    session: Session, entry: RegisteredUseCase, at: datetime
) -> tuple[bool, float | None, bool]:
    """The computed primary metric when the use case has an interaction.

    ``has_interaction`` is false when nothing has been opened for the use
    case at ``at``. The appetite measure then keeps the recorded primary-metric
    event. When an interaction exists, the computed value is the measurement,
    and an unmeasured primary is not a number.
    """
    at = _aware(at)
    opened = [
        row
        for row in _interactions_until(session, at)
        if row.use_case == entry.usecase_id
    ]
    if not opened:
        return False, None, False
    if entry.primary_metric not in PRIMARY_METRICS:
        return True, None, False
    report = next(
        (
            item
            for item in outcome_metrics(session, as_of=at, usecase_id=entry.usecase_id)
            if item.usecase_id == entry.usecase_id
        ),
        None,
    )
    if report is None:
        return True, None, False
    return True, report.value, report.measured


def publish_outcome_metrics(
    session: Session,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    at: datetime | None = None,
) -> tuple[dict[str, Any], ...]:
    """Append one snapshot per use case that names a primary metric.

    An earlier snapshot is left as it was. A later definition version is a
    new event, not an edit.
    """
    when = _aware(at or utcnow())
    who = actor_id.strip()
    if not who:
        raise OutcomeError(
            f"publishing outcome metrics names who published them ({CITE_VERSION})"
        )
    written: list[dict[str, Any]] = []
    for report in outcome_metrics(session, as_of=when):
        if not report.registered or not report.primary_metric:
            continue
        metric = report.metrics[report.primary_metric]
        body = {
            "action": "METRIC",
            "usecase_id": report.usecase_id,
            "primary_metric": report.primary_metric,
            "definition_version": METRIC_DEFINITION_VERSION,
            "baseline": report.baseline,
            "target": report.target,
            "value": report.value,
            "measured": report.measured,
            "breached": report.breached,
            "unit": metric.unit,
            "definition": metric.definition,
            "components": metric.components,
            "citation": report.citation,
        }
        evidence.emit(
            session,
            correlation_id=f"usecase:{report.usecase_id}",
            event_type=EventType.OUTCOME,
            actor_type=ActorType.HUMAN,
            actor_id=who,
            payload=body,
        )
        written.append(body)
    return tuple(written)


def metric_trend(
    session: Session, usecase_id: str, *, as_of: datetime | None = None
) -> dict[str, Any]:
    """Published snapshots grouped by definition version.

    The groups are not averaged together. :func:`latest_metric` refuses a
    use case whose snapshots span more than one version.
    """
    at = _aware(as_of or utcnow())
    groups: dict[int, list[dict[str, Any]]] = {}
    for event in _events_until(session, at):
        payload = _payload(event)
        if event.event_type != EventType.OUTCOME or payload.get("action") != "METRIC":
            continue
        if str(payload.get("usecase_id") or "") != usecase_id:
            continue
        version = payload.get("definition_version")
        if isinstance(version, bool) or not isinstance(version, int):
            continue
        groups.setdefault(version, []).append(
            {
                "sequence": event.sequence,
                "definition_version": version,
                "value": payload.get("value"),
                "measured": payload.get("measured"),
                "breached": payload.get("breached"),
                "primary_metric": payload.get("primary_metric"),
            }
        )
    versions = [
        {"definition_version": version, "snapshots": groups[version]}
        for version in sorted(groups)
    ]
    return {
        "usecase_id": usecase_id,
        "versions": versions,
        "citation": CITE_VERSION,
    }


def latest_metric(
    session: Session, usecase_id: str, *, as_of: datetime | None = None
) -> dict[str, Any]:
    """The latest published snapshot, when every snapshot shares one version."""
    trend = metric_trend(session, usecase_id, as_of=as_of)
    versions = trend["versions"]
    if not versions:
        raise OutcomeError(
            f"'{usecase_id}' has no published outcome metric ({CITE_UNMEASURED})"
        )
    if len(versions) != 1:
        raise OutcomeError(
            f"'{usecase_id}' outcome metrics are not averaged across definition "
            f"versions ({CITE_VERSION})"
        )
    snapshots = versions[0]["snapshots"]
    latest = snapshots[-1]
    latest = dict(latest)
    latest["citation"] = CITE_VERSION
    return latest


def _record_run_fault(
    session: Session,
    interaction_id: str,
    *,
    run_id: str,
    summary: str,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    action: str,
    rule: str,
    now: datetime | None,
) -> dict[str, Any]:
    at = _aware(now or utcnow())
    interaction = require_interaction(session, interaction_id)
    run = run_id.strip()
    text = summary.strip()
    basis = citation.strip()
    who = actor_id.strip()
    if not run or not text or not basis or not who:
        raise OutcomeError(
            f"a run fault names the run, what happened, who recorded it, and "
            f"the rule it cites ({rule})"
        )
    if session.get(RunRecord, run) is None:
        raise OutcomeError(f"run {run} is not recorded ({rule})")
    attached = session.exec(
        select(InteractionLink).where(InteractionLink.interaction_id == interaction.id)
    ).all()
    if not any(link.run_id == run for link in attached):
        raise OutcomeError(
            f"run {run} is not attached to interaction {interaction.id} ({rule})"
        )
    payload = {
        "action": action,
        "interaction_id": interaction.id,
        "usecase_id": interaction.use_case,
        "run_id": run,
        "summary": text,
        "at": at.isoformat(),
        "citation": basis,
        "rule": rule,
    }
    event = evidence.emit(
        session,
        correlation_id=interaction.id,
        event_type=EventType.INTERACTION,
        actor_type=ActorType.HUMAN,
        actor_id=who,
        payload=payload,
    )
    body = dict(payload)
    body["sequence"] = event.sequence
    return body


def _enqueue(
    session: Session,
    interaction_id: str,
    *,
    evidence: EvidenceWriter,
    reasons: Sequence[str],
    risk: str,
    citation: str,
    actor_id: str,
    now: datetime,
) -> None:
    from kognita.review import enqueue_for_review

    enqueue_for_review(
        session,
        interaction_id,
        evidence=evidence,
        reasons=reasons,
        risk=risk,
        citation=citation,
        actor_id=actor_id,
        now=now,
    )


def _output_link(
    session: Session,
    interaction: Interaction,
    *,
    output_hash: str,
    at: datetime,
) -> dict[str, str] | None:
    events = _events_until(session, at)
    correlations = linked_correlation_ids(session, interaction.id)
    calls = [
        event
        for event in events
        if event.event_type == EventType.MODEL_CALL
        and event.correlation_id in correlations
        and _response_hash(_payload(event))
    ]
    if output_hash:
        calls = [
            event
            for event in calls
            if _response_hash(_payload(event)) == output_hash
        ]
    hashes = {_response_hash(_payload(event)) for event in calls}
    if len(hashes) != 1:
        return None
    chosen = calls[-1]
    payload = _payload(chosen)
    version = str(payload.get("model_version") or "").strip()
    if not version:
        return None
    approver = _approver(events, interaction.id)
    if not approver:
        return None
    return {
        "output_hash": _response_hash(payload),
        "model": str(payload.get("model") or ""),
        "model_version": version,
        "provider": str(payload.get("provider") or ""),
        "prompt_template_version": str(payload.get("prompt_template_version") or ""),
        "approver": approver,
    }


def _approver(events: Sequence[EvidenceEvent], interaction_id: str) -> str:
    found: list[tuple[datetime, str]] = []
    for event in events:
        if event.event_type != EventType.FINAL_DECISION:
            continue
        payload = _payload(event)
        if str(payload.get("interaction_id") or "") != interaction_id:
            continue
        name = str(payload.get("approver") or "").strip()
        when = _parse_time(payload.get("decided_at")) or as_utc(event.recorded_at)
        if name and when is not None:
            found.append((when, name))
    if not found:
        return ""
    found.sort(key=lambda item: item[0])
    return found[-1][1]


def _pack(
    usecase_id: str,
    entry: RegisteredUseCase | None,
    metrics: Mapping[str, MetricValue],
) -> UseCaseMetrics:
    if entry is None:
        return UseCaseMetrics(
            usecase_id=usecase_id,
            registered=False,
            status="",
            primary_metric="",
            baseline=None,
            target=None,
            definition_version=METRIC_DEFINITION_VERSION,
            compared=False,
            measured=False,
            value=None,
            breached=True,
            detail="not registered",
            citation=CITE_UNMEASURED,
            accountable_owner="",
            metrics=dict(metrics),
        )
    primary = entry.primary_metric if entry.primary_metric in PRIMARY_METRICS else ""
    baseline = entry.baseline
    target = entry.target
    named = bool(primary) and baseline is not None and target is not None
    if not named:
        return UseCaseMetrics(
            usecase_id=usecase_id,
            registered=True,
            status=entry.status,
            primary_metric=primary,
            baseline=baseline,
            target=target,
            definition_version=METRIC_DEFINITION_VERSION,
            compared=False,
            measured=False,
            value=None,
            breached=False,
            detail="no primary metric",
            citation=CITE_UNMEASURED,
            accountable_owner=entry.owner,
            metrics=dict(metrics),
        )
    chosen = metrics[primary]
    misses = False
    if chosen.measured and chosen.value is not None and target is not None:
        misses = _misses(primary, chosen.value, float(target))
    breached = (not chosen.measured) or misses
    detail = ""
    if not chosen.measured:
        detail = "unmeasured"
    elif misses:
        detail = "beyond target"
    return UseCaseMetrics(
        usecase_id=usecase_id,
        registered=True,
        status=entry.status,
        primary_metric=primary,
        baseline=baseline,
        target=target,
        definition_version=METRIC_DEFINITION_VERSION,
        compared=True,
        measured=chosen.measured,
        value=chosen.value,
        breached=breached,
        detail=detail,
        citation=chosen.citation if chosen.measured else CITE_UNMEASURED,
        accountable_owner=entry.owner,
        metrics=dict(metrics),
    )


def _one_metric(
    metric: str,
    interactions: Sequence[Interaction],
    events: Sequence[EvidenceEvent],
    links: Sequence[InteractionLink],
    reviews: Sequence[RmReview],
    runs: Mapping[str, RunRecord],
    at: datetime,
) -> MetricValue:
    spec = _DEFINITIONS[metric]
    if metric == "cycle time":
        value, measured, sample, components = _cycle_time(
            interactions, events, reviews, at
        )
    elif metric == "decision quality":
        value, measured, sample, components = _decision_quality(
            interactions, events, at
        )
    else:
        value, measured, sample, components = _reliability(
            interactions, events, links, runs
        )
    return MetricValue(
        metric=metric,
        definition_version=METRIC_DEFINITION_VERSION,
        definition=spec["statement"],
        unit=spec["unit"],
        direction=spec["direction"],
        citation=spec["citation"],
        value=value,
        measured=measured,
        sample=sample,
        components=components,
    )


def _cycle_time(
    interactions: Sequence[Interaction],
    events: Sequence[EvidenceEvent],
    reviews: Sequence[RmReview],
    at: datetime,
) -> tuple[float | None, bool, int, dict[str, Any]]:
    durations: list[float] = []
    invalid = False
    decided = 0
    for interaction in interactions:
        final_at = _final_at(events, interaction.id, at)
        if final_at is None:
            continue
        decided += 1
        opened = as_utc(interaction.opened_at)
        seconds = _seconds(opened, final_at)
        if seconds is None:
            invalid = True
            continue
        durations.append(seconds)
    effort, effort_measured, effort_sample = _rm_effort(interactions, events, reviews, at)
    components = {
        "rm_effort_seconds": effort,
        "rm_effort_measured": effort_measured,
        "rm_effort_interactions": effort_sample,
        "citation": CITE_CYCLE,
    }
    if decided == 0 or invalid:
        return None, False, decided, components
    return _mean(durations), True, decided, components


def _rm_effort(
    interactions: Sequence[Interaction],
    events: Sequence[EvidenceEvent],
    reviews: Sequence[RmReview],
    at: datetime,
) -> tuple[float | None, bool, int]:
    durations: list[float] = []
    invalid = False
    sample = 0
    for interaction in interactions:
        rows = [
            row
            for row in reviews
            if row.interaction_id == interaction.id
            and (requested := as_utc(row.requested_at)) is not None
            and requested <= at
        ]
        if not rows:
            continue
        sample += 1
        final_at = _final_at(events, interaction.id, at)
        row = next((item for item in rows if item.status == "final"), rows[-1])
        end = as_utc(row.decided_at) or final_at
        start = as_utc(row.requested_at)
        seconds = _seconds(start, end)
        if seconds is None:
            invalid = True
            continue
        durations.append(seconds)
    if sample == 0:
        return None, False, 0
    if invalid:
        return None, False, sample
    return _mean(durations), True, sample


def _decision_quality(
    interactions: Sequence[Interaction],
    events: Sequence[EvidenceEvent],
    at: datetime,
) -> tuple[float | None, bool, int, dict[str, Any]]:
    decided_ids = [
        interaction.id
        for interaction in interactions
        if _final_at(events, interaction.id, at) is not None
    ]
    sample = len(decided_ids)
    if sample == 0:
        return None, False, 0, _quality_components(0, 0, 0, 0, 0)
    edited = rejected = escalated = bad = 0
    outcome_events = 0
    for interaction_id in decided_ids:
        finals = _finals(events, interaction_id)
        outcomes = {str(payload.get("outcome") or "") for payload in finals}
        edited_here = bool(outcomes & {"modify"}) or any(
            str(payload.get("provenance") or "") == _EDITED_PROVENANCE for payload in finals
        )
        edited_here = edited_here or _diff_edited(events, interaction_id)
        rejected_here = "reject" in outcomes
        escalated_here = "escalate" in outcomes
        linked = _outcome_events(events, interaction_id)
        outcome_events += linked
        if edited_here:
            edited += 1
        if rejected_here:
            rejected += 1
        if escalated_here:
            escalated += 1
        if edited_here or rejected_here or escalated_here or linked:
            bad += 1
    good = sample - bad
    return (
        good / sample,
        True,
        sample,
        _quality_components(sample, edited, rejected, escalated, outcome_events),
    )


def _quality_components(
    sample: int, edited: int, rejected: int, escalated: int, outcome_events: int
) -> dict[str, Any]:
    def rate(count: int) -> float | None:
        if sample == 0:
            return None
        return count / sample

    return {
        "edit_rate": rate(edited),
        "rejection_rate": rate(rejected),
        "escalation_rate": rate(escalated),
        "outcome_events_per_interaction": rate(outcome_events),
        "outcome_events": outcome_events,
        "citation": CITE_QUALITY,
    }


def _reliability(
    interactions: Sequence[Interaction],
    events: Sequence[EvidenceEvent],
    links: Sequence[InteractionLink],
    runs: Mapping[str, RunRecord],
) -> tuple[float | None, bool, int, dict[str, Any]]:
    interaction_ids = {item.id for item in interactions}
    run_ids = sorted(
        {
            link.run_id
            for link in links
            if link.run_id and link.interaction_id in interaction_ids
        }
    )
    if not run_ids:
        return None, False, 0, _reliability_components(0, 0, 0, 0, 0, 0, 0)
    clean = denials = budgets = exceptions = recoveries = suspended = incomplete = 0
    for run_id in run_ids:
        correlations = {
            link.correlation_id
            for link in links
            if link.run_id == run_id and link.correlation_id
        }
        row = runs.get(run_id)
        faults = _run_faults(events, run_id, correlations)
        held = row is None or bool(row.continuation_hash) or bool(row.approvals_pending)
        finished = _run_finished(events, correlations)
        if faults["denial"]:
            denials += 1
        if faults["budget"]:
            budgets += 1
        if faults["exception"]:
            exceptions += 1
        if faults["recovery"]:
            recoveries += 1
        if held:
            suspended += 1
        unclean = held or any(faults.values()) or not finished
        if not finished and not held and not any(faults.values()):
            incomplete += 1
        if not unclean:
            clean += 1
    sample = len(run_ids)
    return (
        clean / sample,
        True,
        sample,
        _reliability_components(
            sample, clean, denials, budgets, exceptions, recoveries, suspended + incomplete
        ),
    )


def _reliability_components(
    sample: int,
    clean: int,
    denials: int,
    budgets: int,
    exceptions: int,
    recoveries: int,
    unfinished: int,
) -> dict[str, Any]:
    return {
        "runs": sample,
        "clean": clean,
        "denials": denials,
        "budget_breaches": budgets,
        "exceptions": exceptions,
        "human_recoveries": recoveries,
        "unfinished": unfinished,
        "citation": CITE_RELIABILITY,
    }


def _run_faults(
    events: Sequence[EvidenceEvent], run_id: str, correlations: set[str]
) -> dict[str, bool]:
    denial = budget = exception = recovery = False
    for event in events:
        payload = _payload(event)
        on_run = event.run_id == run_id or event.correlation_id in correlations
        if not on_run and str(payload.get("run_id") or "") != run_id:
            continue
        if event.event_type == EventType.POLICY_DECISION:
            if str(payload.get("outcome") or "") == "DENY":
                denial = True
            if _budget_failed(payload):
                budget = True
        action = str(payload.get("action") or "")
        if action == "EXCEPTION" and str(payload.get("run_id") or "") == run_id:
            exception = True
        if action == "HUMAN_RECOVERY" and str(payload.get("run_id") or "") == run_id:
            recovery = True
    return {
        "denial": denial,
        "budget": budget,
        "exception": exception,
        "recovery": recovery,
    }


def _run_finished(events: Sequence[EvidenceEvent], correlations: set[str]) -> bool:
    if not correlations:
        return False
    for event in events:
        if event.correlation_id not in correlations:
            continue
        payload = _payload(event)
        if event.event_type == EventType.POLICY_DECISION and payload.get("outcome") == "ALLOW":
            return True
        if (
            event.event_type == EventType.FINAL_DECISION
            and str(payload.get("outcome") or "") in _CLEAN_FINALS
        ):
            return True
    return False


def _budget_failed(payload: Mapping[str, Any]) -> bool:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return False
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("check") == "BUDGET" and check.get("result") == "fail":
            return True
    return False


def _final_at(
    events: Sequence[EvidenceEvent], interaction_id: str, at: datetime
) -> datetime | None:
    found: list[datetime] = []
    for payload in _finals(events, interaction_id):
        when = _parse_time(payload.get("decided_at"))
        if when is not None and when <= at:
            found.append(when)
    if not found:
        return None
    return max(found)


def _finals(events: Sequence[EvidenceEvent], interaction_id: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for event in events:
        if event.event_type != EventType.FINAL_DECISION:
            continue
        payload = _payload(event)
        if str(payload.get("interaction_id") or "") == interaction_id:
            found.append(payload)
    return found


def _diff_edited(events: Sequence[EvidenceEvent], interaction_id: str) -> bool:
    for event in events:
        if event.event_type != EventType.INTERACTION:
            continue
        payload = _payload(event)
        if payload.get("action") != "RM_REVIEW":
            continue
        if str(payload.get("interaction_id") or "") != interaction_id:
            continue
        try:
            ops = int(payload.get("diff_ops") or 0)
        except (TypeError, ValueError):
            ops = 0
        if ops > 0:
            return True
    return False


def _outcome_events(events: Sequence[EvidenceEvent], interaction_id: str) -> int:
    count = 0
    for event in events:
        if event.event_type != EventType.OUTCOME:
            continue
        payload = _payload(event)
        if payload.get("action") != "RECORDED":
            continue
        if str(payload.get("interaction_id") or "") == interaction_id:
            count += 1
    return count


def _is_blocked(event: EvidenceEvent) -> bool:
    payload = _payload(event)
    if event.event_type == EventType.POLICY_DECISION and payload.get("outcome") == "DENY":
        return True
    return event.event_type == EventType.EGRESS and payload.get("decision") == "DENY"


def _blocked_attribution(
    event: EvidenceEvent,
    events: Sequence[EvidenceEvent],
    interactions: Sequence[Interaction],
    links: Sequence[InteractionLink],
) -> tuple[str, str, bool, str, bool] | None:
    payload = _payload(event)
    interaction = _interaction_for_correlation(event.correlation_id, interactions, links)
    envelope = payload.get("envelope")
    if not isinstance(envelope, dict):
        envelope = {}
    usecase = (
        interaction.use_case
        if interaction is not None and interaction.use_case
        else str(envelope.get("purpose") or "")
    )
    if not usecase:
        usecase = NOT_RECORDED
    agent = str(envelope.get("agent_name") or "").strip()
    version, version_ok = _version_on_correlation(events, event.correlation_id)
    return usecase, agent, bool(agent), version, version_ok


def _agent_and_version(
    events: Sequence[EvidenceEvent],
    interaction: Interaction,
    links: Sequence[InteractionLink],
) -> tuple[str, bool, str, bool]:
    correlations = {interaction.id}
    for link in links:
        if link.interaction_id == interaction.id and link.correlation_id:
            correlations.add(link.correlation_id)
    agent = ""
    for event in events:
        if event.correlation_id not in correlations:
            continue
        if event.event_type != EventType.POLICY_DECISION:
            continue
        envelope = _payload(event).get("envelope")
        if isinstance(envelope, dict) and envelope.get("agent_name"):
            agent = str(envelope.get("agent_name") or "").strip()
            break
    versions: set[str] = set()
    for event in events:
        if event.event_type != EventType.MODEL_CALL:
            continue
        if event.correlation_id not in correlations:
            continue
        version = str(_payload(event).get("model_version") or "").strip()
        if version:
            versions.add(version)
    if len(versions) == 1:
        return agent, bool(agent), next(iter(versions)), True
    return agent, bool(agent), "", False


def _version_on_correlation(
    events: Sequence[EvidenceEvent], correlation_id: str
) -> tuple[str, bool]:
    versions = {
        str(_payload(event).get("model_version") or "").strip()
        for event in events
        if event.event_type == EventType.MODEL_CALL and event.correlation_id == correlation_id
    }
    versions.discard("")
    if len(versions) == 1:
        return next(iter(versions)), True
    return "", False


def _interaction_for_correlation(
    correlation_id: str,
    interactions: Sequence[Interaction],
    links: Sequence[InteractionLink],
) -> Interaction | None:
    by_id = {item.id: item for item in interactions}
    if correlation_id in by_id:
        return by_id[correlation_id]
    for link in links:
        if link.correlation_id == correlation_id and link.interaction_id in by_id:
            return by_id[link.interaction_id]
    return None


def _interaction_by_id(
    interactions: Sequence[Interaction], interaction_id: str
) -> Interaction | None:
    for item in interactions:
        if item.id == interaction_id:
            return item
    return None


def _response_hash(payload: Mapping[str, Any]) -> str:
    return str(payload.get("response_hash") or "").strip()


def _misses(metric: str, value: float, target: float) -> bool:
    if metric == "cycle time":
        return value > target
    return value < target


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds()
    if delta < 0:
        return None
    return delta


def _events_until(session: Session, at: datetime) -> list[EvidenceEvent]:
    rows = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    kept: list[EvidenceEvent] = []
    for row in rows:
        recorded = as_utc(row.recorded_at)
        if recorded is not None and recorded <= at:
            kept.append(row)
    return kept


def _interactions_until(session: Session, at: datetime) -> list[Interaction]:
    rows = session.exec(select(Interaction)).all()
    return [
        row
        for row in rows
        if (opened := as_utc(row.opened_at)) is not None and opened <= at
    ]


def _payload(event: EvidenceEvent) -> dict[str, Any]:
    raw = event.payload
    return raw if isinstance(raw, dict) else {}


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _aware(value)
    try:
        return _aware(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


__all__ = [
    "CITE_COMPLAINT",
    "CITE_CYCLE",
    "CITE_EXCEPTION",
    "CITE_NEAR_MISS",
    "CITE_QUALITY",
    "CITE_RECOVERY",
    "CITE_RELIABILITY",
    "CITE_UNMEASURED",
    "CITE_VERSION",
    "DERIVED_NEAR_MISS_KINDS",
    "METRIC_DEFINITION_VERSION",
    "NEAR_MISS_KINDS",
    "NOT_RECORDED",
    "OUTCOME_KINDS",
    "RECORDED_NEAR_MISS_KINDS",
    "MetricValue",
    "NearMissCount",
    "OutcomeError",
    "UseCaseMetrics",
    "interaction_measurement",
    "latest_metric",
    "metric_trend",
    "near_miss_register",
    "outcome_metrics",
    "publish_outcome_metrics",
    "record_exception",
    "record_human_recovery",
    "record_near_miss",
    "record_outcome",
]
