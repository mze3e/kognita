"""Circuit breaker, affected clients, and kill-switch drills.

Flags are review findings, outcome events, recorded near misses, and denials
that are not themselves a breaker, a revocation, or a kill switch. When the
count for a use case, model version, prompt version, or policy version is
greater than or equal to a registered threshold inside its window, Kognita
inserts a prohibiting policy, escalates to the accountable owner, and records
the trip as an incident of impact 1. Nothing trips until a threshold is
registered. There is no built-in default.

The prohibiting policy is a ``CIRCUIT_BREAKER`` rule. It is applied on the
next decision even when a pack's ``engages`` would skip it. Resetting it is a
governed close of that row, by the accountable owner, with a citation.

``affected_clients`` lists clients who received model output or released tool
output in a window, when that output is linked to an interaction. A denial
is not receipt.

A high-materiality use case records kill-switch drills. The drill engages the
agent's existing kill switch, records who activated it, how long until the
next decision was denied, and how the fallback was reached, then releases the
switch unless a revocation is holding it. The next due date is
:data:`DRILL_DAYS` (90) after the drill. A drill past its due date denies the
use case the way a missed review date does. A high-materiality pilot or live
use case with no drill scheduled is already overdue, when the decision is
given the register.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from sqlalchemy import or_
from sqlmodel import Session, select

from kognita.envelope import Check, Envelope, RuleContext
from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.models import (
    BreakerThreshold,
    DrillDue,
    EvidenceEvent,
    Interaction,
    InteractionLink,
    Policy,
    as_utc,
    in_force,
    utcnow,
)
from kognita.registry import set_kill_switch
from kognita.revocation import OpenRevocation, load_revocations
from kognita.rules import circuit_breaker
from kognita.usecases import is_material, load_register, record_incident
from kognita.vocabulary import (
    ActorType,
    CheckResult,
    EventType,
    Outcome,
)

#: Days from a completed drill to the next due date. The roadmap does not set
#: an interval. 90 is the same period as a tier 3 review, and it stays this
#: constant.
DRILL_DAYS = 90

BREAKER_DIMENSIONS: tuple[str, ...] = ("usecase", "model", "prompt", "policy")

_CONTAINMENT = frozenset(
    {"CIRCUIT_BREAKER", "REVOCATION", "KILL_SWITCH", "KILL_SWITCH_DRILL"}
)

CITE = "Circuit breaker"
CITE_DRILL = "Kill-switch drill"
CITE_AFFECTED = "Affected clients"


class BreakerError(KognitaError):
    """A breaker, a drill, or an affected-client query could not be completed."""


@dataclass(frozen=True)
class OpenDrill:
    """A drill due date in force at one instant."""

    usecase_id: str
    next_due: datetime


def set_breaker_threshold(
    session: Session,
    dimension: str,
    value: str,
    *,
    threshold: int,
    window_seconds: int,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    owner: str = "",
    now: datetime | None = None,
) -> BreakerThreshold:
    """Record how many flags trip a breaker. An open threshold for the same value is closed.

    A count greater than or equal to ``threshold`` trips it. Nothing trips
    until this row exists: there is no built-in default threshold.
    """
    at = _aware(now or utcnow())
    kind = _breaker_dimension(dimension)
    named = value.strip()
    if not named:
        raise BreakerError(f"{CITE}: a threshold names the {kind}")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
        raise BreakerError(f"{CITE}: the threshold is a positive count")
    if (
        isinstance(window_seconds, bool)
        or not isinstance(window_seconds, int)
        or window_seconds < 1
    ):
        raise BreakerError(f"{CITE}: the window is a positive number of seconds")
    actor = _human(session, actor_id, "threshold")
    basis = _basis(citation, "threshold")
    from kognita.revocation import accountable_owners

    owners = accountable_owners(session, kind, named, explicit=owner, at=at)
    if len(owners) != 1:
        raise BreakerError(
            f"{CITE}: name the accountable owner for {kind} {named}"
        )
    accountable = owners[0]
    _close_thresholds(session, kind, named, at)
    row = BreakerThreshold(
        dimension=kind,
        value=named,
        threshold=threshold,
        window_seconds=window_seconds,
        citation=basis,
        actor_id=actor,
        accountable_owner=accountable,
        effective_from=at,
    )
    session.add(row)
    session.flush()
    _emit(
        session,
        evidence,
        actor=actor,
        correlation_id=f"breaker:{kind}:{named}",
        body={
            "action": "SET_BREAKER_THRESHOLD",
            "dimension": kind,
            "value": named,
            "threshold": threshold,
            "window_seconds": window_seconds,
            "citation": basis,
            "accountable_owner": accountable,
            "at": at.isoformat(),
        },
    )
    return row


def evaluate_breakers(
    session: Session,
    *,
    evidence: EvidenceWriter,
    now: datetime | None = None,
) -> tuple[dict[str, Any], ...]:
    """Trip each threshold whose flags in the window are at or above it.

    A count greater than or equal to the registered threshold trips it. With
    no threshold registered, nothing trips. A threshold that already has an
    open prohibiting policy is left as it is. Called when a flag is recorded,
    so the trip is the next decision's problem and not a scan on the gateway
    path. The trip is an incident of impact 1.
    """
    at = _aware(now or utcnow())
    thresholds = in_force(session, BreakerThreshold, at)
    if not thresholds:
        return ()
    # The widest open window bounds which flags are read, on recorded_at.
    # A decision linked to a flag inside that window is loaded by id even
    # when the decision itself is older. The read does not walk the chain.
    window = max(int(row.window_seconds) for row in thresholds)
    flags = _flags(session, at, window_seconds=window)
    tripped: list[dict[str, Any]] = []
    for row in thresholds:
        if _open_breaker(session, row.dimension, row.value, at) is not None:
            continue
        matched = [
            flag
            for flag in flags
            if row.value in flag["keys"].get(row.dimension, ())
            and _in_window(flag["at"], at, row.window_seconds)
        ]
        # At or above the registered threshold. No threshold means this loop
        # never runs, so nothing trips by default.
        if len(matched) < row.threshold:
            continue
        tripped.append(_trip(session, row, matched, evidence, at))
    return tuple(tripped)


def reset_breaker(
    session: Session,
    dimension: str,
    value: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    now: datetime | None = None,
) -> Policy:
    """Close a tripped breaker. The actor must be the owner named on it."""
    at = _aware(now or utcnow())
    kind = _breaker_dimension(dimension)
    named = value.strip()
    actor = _human(session, actor_id, "reset")
    basis = _basis(citation, "reset")
    policy = _open_breaker(session, kind, named, at)
    if policy is None:
        raise BreakerError(f"{CITE}: {kind} {named} is not tripped")
    rule = policy.rule if isinstance(policy.rule, dict) else {}
    owner = str(rule.get("accountable_owner") or "").strip()
    if not owner or actor != owner:
        raise BreakerError(
            f"{CITE}: {actor} cannot reset {kind} {named} · accountable owner: {owner or 'none'}"
        )
    policy.effective_to = at
    session.add(policy)
    session.flush()
    _emit(
        session,
        evidence,
        actor=actor,
        correlation_id=f"breaker:{kind}:{named}",
        body={
            "action": "BREAKER_RESET",
            "dimension": kind,
            "value": named,
            "policy_id": policy.id,
            "citation": basis,
            "accountable_owner": owner,
            "at": at.isoformat(),
        },
    )
    return policy


def circuit_breaker_checks(
    envelope: Envelope,
    policies: Sequence[Policy],
    attributes: dict[str, Any],
    at: datetime,
) -> list[Check]:
    """Breaker policies in force, whether or not a pack engages them."""
    context = RuleContext(envelope=envelope, attributes=attributes, as_of=at)
    checks: list[Check] = []
    for policy in policies:
        if policy.rule_type != "CIRCUIT_BREAKER" or not policy.is_effective(at):
            continue
        checks.extend(circuit_breaker(policy, context))
    return checks


def load_drill_dues(session: Session, at: datetime) -> tuple[OpenDrill, ...]:
    when = _aware(at)
    return tuple(
        OpenDrill(usecase_id=row.usecase_id, next_due=_aware(row.next_due))
        for row in in_force(session, DrillDue, when)
    )


def drill_checks(
    envelope: Envelope,
    dues: Sequence[OpenDrill],
    *,
    register: Any = None,
    at: datetime,
) -> list[Check]:
    """Deny a high-materiality pilot or live use case that is past its drill.

    A recorded due date at or before ``at`` is overdue. With a register, a
    high-materiality pilot or live use case that has no drill scheduled is
    already overdue. A use case that is not high-materiality, or not pilot or
    live, is not denied for a missing schedule. With no register, only a
    recorded overdue date denies: the decision cannot see materiality.
    """
    when = _aware(at)
    checks: list[Check] = []
    scheduled = False
    for due in dues:
        if due.usecase_id != envelope.purpose:
            continue
        scheduled = True
        if due.next_due > when:
            continue
        owner = ""
        if register is not None:
            entry = register.get(envelope.purpose)
            if (
                entry is None
                or not is_material(entry)
                or entry.status not in ("pilot", "live")
            ):
                continue
            owner = entry.owner.strip()
        checks.append(_drill_denial(envelope.purpose, owner, scheduled=True))
    if checks or scheduled or register is None or not envelope.purpose.strip():
        return checks
    entry = register.get(envelope.purpose)
    if entry is None or not is_material(entry) or entry.status not in ("pilot", "live"):
        return checks
    checks.append(
        _drill_denial(envelope.purpose, entry.owner.strip(), scheduled=False)
    )
    return checks


def _drill_denial(purpose: str, owner: str, *, scheduled: bool) -> Check:
    if scheduled:
        citation = f"{CITE_DRILL} — '{purpose}' is past its drill date"
    else:
        citation = (
            f"{CITE_DRILL} — '{purpose}' has no drill scheduled "
            f"and is past its drill date"
        )
    if owner:
        citation = f"{citation} · accountable owner: {owner}"
    return Check(
        check="KILL_SWITCH_DRILL",
        regime="INTERNAL",
        result=CheckResult.FAIL,
        citation=citation,
    )


def schedule_kill_switch_drill(
    session: Session,
    usecase_id: str,
    *,
    due_at: datetime,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    now: datetime | None = None,
) -> DrillDue:
    """Set the next drill date. A high-materiality use case past it is denied."""
    at = _aware(now or utcnow())
    due = _aware(due_at)
    entry = _material(session, usecase_id, at)
    actor = _human(session, actor_id, "drill")
    basis = _basis(citation, "drill")
    _close_dues(session, entry.usecase_id, at)
    row = DrillDue(usecase_id=entry.usecase_id, next_due=due, effective_from=at)
    session.add(row)
    session.flush()
    _emit(
        session,
        evidence,
        actor=actor,
        correlation_id=f"usecase:{entry.usecase_id}",
        body={
            "action": "DRILL_DUE",
            "usecase_id": entry.usecase_id,
            "next_due": due.isoformat(),
            "citation": basis,
            "accountable_owner": entry.owner,
            "at": at.isoformat(),
        },
    )
    return row


def run_kill_switch_drill(
    session: Session,
    usecase_id: str,
    agent_name: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    fallback: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Engage the kill switch, record the denial, and release the switch.

    The switch stays engaged when the next decision is not denied by it, and
    when an open revocation is holding that agent. The drill interval then
    starts at ``now``.
    """
    at = _aware(now or utcnow())
    entry = _material(session, usecase_id, at)
    name = agent_name.strip()
    if not name:
        raise BreakerError(f"{CITE_DRILL}: a drill names the agent")
    if entry.agents and name not in set(entry.agents):
        raise BreakerError(
            f"{CITE_DRILL}: {name} is not an agent of {entry.usecase_id}"
        )
    actor = _human(session, actor_id, "drill")
    basis = _basis(citation, "drill")
    reached = fallback.strip()
    if not reached:
        raise BreakerError(f"{CITE_DRILL}: a drill records how the fallback was reached")
    from kognita.registry import get as get_agent

    if get_agent(session, name) is None:
        raise BreakerError(f"{CITE_DRILL}: agent {name} is not registered")

    started = time.perf_counter()
    set_kill_switch(
        session,
        name,
        True,
        evidence=evidence,
        actor_id=actor,
        correlation_id=f"usecase:{entry.usecase_id}",
        reason=basis,
        now=at,
    )
    evaluation = _probe(session, entry.usecase_id, name, actor, evidence, at)
    elapsed = time.perf_counter() - started
    denied = next(
        (
            check
            for check in evaluation.checks
            if check.check == "KILL_SWITCH" and check.result == CheckResult.FAIL
        ),
        None,
    )
    if denied is None:
        raise BreakerError(
            f"{CITE_DRILL}: the next decision was not denied by the kill switch "
            f"· accountable owner: {entry.owner}"
        )
    if not _revocation_holds(session, name, at):
        set_kill_switch(
            session,
            name,
            False,
            evidence=evidence,
            actor_id=actor,
            correlation_id=f"usecase:{entry.usecase_id}",
            reason=basis,
            now=at,
        )
    next_due = at + timedelta(days=DRILL_DAYS)
    _close_dues(session, entry.usecase_id, at)
    session.add(
        DrillDue(usecase_id=entry.usecase_id, next_due=next_due, effective_from=at)
    )
    session.flush()
    payload = {
        "action": "KILL_SWITCH_DRILL",
        "usecase_id": entry.usecase_id,
        "agent": name,
        "activated_by": actor,
        "citation": basis,
        "activated_at": at.isoformat(),
        "denied_at": at.isoformat(),
        "latency_seconds": elapsed,
        "fallback": reached,
        "denied_citation": denied.citation,
        "next_due": next_due.isoformat(),
        "accountable_owner": entry.owner,
    }
    _emit(
        session,
        evidence,
        actor=actor,
        correlation_id=f"usecase:{entry.usecase_id}",
        body=payload,
    )
    return payload


def affected_clients(
    session: Session,
    *,
    model: str = "",
    policy_id: str = "",
    usecase: str = "",
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Clients who received model or released tool output in ``[start, end)``.

    Receipt is a ``MODEL_CALL`` or a released ``TOOL_CALL``. A denial is not
    receipt, and neither is a model response the gateway withheld. Output linked to an interaction is listed under ``clients``.
    Output with no interaction, such as a gateway or MCP call, is not given a
    client id; it is listed under ``unlinked`` as "client unknown" so the
    query does not read as complete when it is not.
    """
    opened = _aware(start)
    closed = _aware(end)
    if closed <= opened:
        raise BreakerError(f"{CITE_AFFECTED}: --to is after --from")
    selectors = [name for name, value in (("model", model), ("policy", policy_id), ("usecase", usecase)) if value.strip()]
    if len(selectors) != 1:
        raise BreakerError(
            f"{CITE_AFFECTED}: name one of --model, --policy, or --usecase"
        )
    selector = selectors[0]
    target = {"model": model, "policy": policy_id, "usecase": usecase}[selector].strip()
    events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    interactions = {row.id: row for row in session.exec(select(Interaction)).all()}
    links = list(session.exec(select(InteractionLink)).all())
    by_correlation: dict[str, list[EvidenceEvent]] = {}
    for event in events:
        by_correlation.setdefault(event.correlation_id, []).append(event)
    rows: list[dict[str, Any]] = []
    unlinked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for event in events:
        if event.event_type not in (EventType.MODEL_CALL, EventType.TOOL_CALL):
            continue
        recorded = as_utc(event.recorded_at)
        if recorded is None or recorded < opened or recorded >= closed:
            continue
        payload = _payload(event)
        if not _released(event.event_type, payload):
            continue
        if event.event_type == EventType.MODEL_CALL and _withheld_later(
            by_correlation.get(event.correlation_id, [])
        ):
            continue
        interaction = _interaction_for(event.correlation_id, interactions, links)
        if interaction is None or not interaction.client_id:
            same = by_correlation.get(event.correlation_id, [])
            if _unlinked_matches(selector, target, payload, same):
                unlinked.append(
                    {
                        "client_id": None,
                        "client": "client unknown",
                        "correlation_id": event.correlation_id,
                        "received_at": recorded.isoformat(),
                        "model_version": str(payload.get("model_version") or ""),
                        "usecase_id": _decided_purpose(same),
                    }
                )
            continue
        if not _output_matches(
            selector,
            target,
            payload,
            interaction,
            by_correlation.get(event.correlation_id, []),
        ):
            continue
        key = (interaction.client_id, interaction.id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "client_id": interaction.client_id,
                "interaction_id": interaction.id,
                "correlation_id": event.correlation_id,
                "received_at": recorded.isoformat(),
                "model_version": str(payload.get("model_version") or ""),
                "usecase_id": interaction.use_case,
            }
        )
    label = {"model": "model", "policy": "policy", "usecase": "use case"}[selector]
    return {
        "selector": selector,
        "value": target,
        "from": opened.isoformat(),
        "to": closed.isoformat(),
        "clients": rows,
        "unlinked": unlinked,
        "citation": (
            f"{CITE_AFFECTED}: output received under {label} {target} "
            f"from {opened.isoformat()} until {closed.isoformat()}"
        ),
    }


def _trip(
    session: Session,
    row: BreakerThreshold,
    matched: Sequence[dict[str, Any]],
    evidence: EvidenceWriter,
    at: datetime,
) -> dict[str, Any]:
    usecases = sorted({str(flag.get("usecase") or "") for flag in matched if flag.get("usecase")})
    rule: dict[str, Any] = {
        "dimension": row.dimension,
        "value": row.value,
        "on_violation": "fail",
        "accountable_owner": row.accountable_owner,
    }
    if row.dimension == "policy":
        rule["usecases"] = usecases
    citation = (
        f"{CITE} — {row.dimension} {row.value} crossed {len(matched)} flags "
        f"in {row.window_seconds}s · {row.citation} · "
        f"accountable owner: {row.accountable_owner}"
    )
    policy = Policy(
        regime="CIRCUIT_BREAKER",
        rule_type="CIRCUIT_BREAKER",
        rule=rule,
        citation=citation,
        effective_from=at,
    )
    session.add(policy)
    session.flush()
    register = load_register(session, as_of=at)
    incident_usecase = ""
    if row.dimension == "usecase" and register.get(row.value) is not None:
        incident_usecase = row.value
    else:
        for usecase in usecases:
            entry = register.get(usecase)
            if entry is not None and entry.owner == row.accountable_owner:
                incident_usecase = usecase
                break
        if not incident_usecase:
            for usecase in usecases:
                if register.get(usecase) is not None:
                    incident_usecase = usecase
                    break
    summary = (
        f"{CITE}: {row.dimension} {row.value} tripped at {len(matched)} flags "
        f"and escalated to {row.accountable_owner}"
    )
    record_incident(
        session,
        incident_usecase,
        impact=1,
        summary=summary,
        evidence=evidence,
        actor_id=row.accountable_owner or "circuit-breaker",
        at=at,
    )
    body = {
        "action": "CIRCUIT_BREAKER",
        "dimension": row.dimension,
        "value": row.value,
        "policy_id": policy.id,
        "flag_count": len(matched),
        "threshold": row.threshold,
        "window_seconds": row.window_seconds,
        "citation": citation,
        "accountable_owner": row.accountable_owner,
        "escalated_to": row.accountable_owner,
        "usecases": usecases,
        "at": at.isoformat(),
    }
    _emit(
        session,
        evidence,
        actor=row.accountable_owner or "circuit-breaker",
        correlation_id=f"breaker:{row.dimension}:{row.value}",
        body=body,
    )
    return body


def _flags(
    session: Session, at: datetime, *, window_seconds: int
) -> list[dict[str, Any]]:
    """Flags inside the window, with the decisions those flags cite.

    The window is when the problem was flagged or confirmed. A decision that
    the flag cites can be older than that window. Those decisions are loaded
    by correlation id, and the interactions and links that join them are
    loaded by id. Nothing else on the chain is read.
    """
    when = _aware(at)
    start = when - timedelta(seconds=window_seconds)
    windowed = session.exec(
        select(EvidenceEvent)
        .where(
            EvidenceEvent.recorded_at >= start,
            EvidenceEvent.recorded_at <= when,
            EvidenceEvent.event_type.in_(
                (
                    EventType.REVIEW,
                    EventType.OUTCOME,
                    EventType.NEAR_MISS,
                    EventType.POLICY_DECISION,
                    EventType.EGRESS,
                )
            ),
        )
        .order_by(EvidenceEvent.sequence)
    ).all()
    events = [
        event
        for event in windowed
        if _is_flag(event, _payload(event))
        and (recorded := as_utc(event.recorded_at)) is not None
        and start <= recorded <= when
    ]
    interactions, links, decisions = _flag_context(session, events, at)
    by_correlation, by_interaction = _policy_ids(decisions, interactions, links)
    found: list[dict[str, Any]] = []
    for event in events:
        payload = _payload(event)
        if not _is_flag(event, payload):
            continue
        recorded = as_utc(event.recorded_at)
        if recorded is None:
            continue
        usecase = _usecase(event, payload, interactions, links)
        keys: dict[str, set[str]] = {name: set() for name in BREAKER_DIMENSIONS}
        if usecase:
            keys["usecase"].add(usecase)
        keys["model"].update(_carried(payload, "model_version", "model_version_recorded"))
        keys["model"].update(_carried(payload, "model", "model_recorded"))
        keys["model"].update(_carried(payload, "model_card", "model_card_recorded"))
        providers = _carried(payload, "provider", "provider_recorded")
        names = _carried(payload, "model", "model_recorded")
        versions = _carried(payload, "model_version", "model_version_recorded")
        for provider in providers:
            for name in names:
                for version in versions:
                    keys["model"].add(f"{provider}/{name}@{version}")
        keys["prompt"].update(
            _carried(
                payload,
                "prompt_template_version",
                "prompt_template_version_recorded",
            )
        )
        keys["policy"].update(
            _flag_policies(
                event, payload, by_correlation, by_interaction, interactions, links
            )
        )
        found.append({"at": recorded, "usecase": usecase, "keys": keys})
    return found


def _flag_context(
    session: Session,
    events: Sequence[EvidenceEvent],
    at: datetime,
) -> tuple[dict[str, Interaction], list[InteractionLink], list[EvidenceEvent]]:
    """Interactions, links and decisions a windowed flag can cite.

    The flag names a correlation id and, when it has one, an interaction id.
    Links for those ids name the decision's correlation. The decisions are
    read by that id, with no time bound.
    """
    correlation_ids: set[str] = set()
    interaction_ids: set[str] = set()
    for event in events:
        if event.correlation_id:
            correlation_ids.add(event.correlation_id)
            interaction_ids.add(event.correlation_id)
        named = _payload(event).get("interaction_id")
        if isinstance(named, str) and named:
            interaction_ids.add(named)
    links = _links_for(session, correlation_ids, interaction_ids)
    for link in links:
        if link.correlation_id:
            correlation_ids.add(link.correlation_id)
        if link.interaction_id:
            interaction_ids.add(link.interaction_id)
    interactions = _interactions_for(session, interaction_ids, at)
    decisions = _decisions_for(session, correlation_ids)
    return interactions, links, decisions


def _links_for(
    session: Session, correlation_ids: set[str], interaction_ids: set[str]
) -> list[InteractionLink]:
    clauses = []
    if correlation_ids:
        clauses.append(InteractionLink.correlation_id.in_(tuple(correlation_ids)))
    if interaction_ids:
        clauses.append(InteractionLink.interaction_id.in_(tuple(interaction_ids)))
    if not clauses:
        return []
    return list(session.exec(select(InteractionLink).where(or_(*clauses))).all())


def _interactions_for(
    session: Session, interaction_ids: set[str], at: datetime
) -> dict[str, Interaction]:
    if not interaction_ids:
        return {}
    rows = session.exec(
        select(Interaction).where(Interaction.id.in_(tuple(interaction_ids)))
    ).all()
    return {
        row.id: row
        for row in rows
        if (opened := as_utc(row.opened_at)) is not None and opened <= at
    }


def _decisions_for(
    session: Session, correlation_ids: set[str]
) -> list[EvidenceEvent]:
    if not correlation_ids:
        return []
    return list(
        session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.event_type == EventType.POLICY_DECISION,
                EvidenceEvent.correlation_id.in_(tuple(correlation_ids)),
            )
        ).all()
    )


def _is_flag(event: EvidenceEvent, payload: dict[str, Any]) -> bool:
    if event.event_type == EventType.REVIEW and payload.get("action") == "REVIEW_FLAG":
        return True
    if event.event_type == EventType.OUTCOME and payload.get("action") == "RECORDED":
        return True
    if event.event_type == EventType.NEAR_MISS and payload.get("action") == "RECORDED":
        return True
    if event.event_type == EventType.POLICY_DECISION and payload.get("outcome") == Outcome.DENY.value:
        return not _containment_only(payload)
    if event.event_type == EventType.EGRESS and payload.get("decision") == "DENY":
        return True
    return False


def _containment_only(payload: dict[str, Any]) -> bool:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return False
    fails = [
        check
        for check in checks
        if isinstance(check, dict) and check.get("result") == CheckResult.FAIL.value
    ]
    if not fails:
        return False
    return all(str(check.get("check") or "") in _CONTAINMENT for check in fails)


def _policy_ids(
    events: Sequence[EvidenceEvent],
    interactions: dict[str, Interaction],
    links: Sequence[InteractionLink],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Policy ids cited on decisions, by correlation and by interaction."""
    by_correlation: dict[str, set[str]] = {}
    by_interaction: dict[str, set[str]] = {}
    for event in events:
        if event.event_type != EventType.POLICY_DECISION:
            continue
        ids = _check_policy_ids(_payload(event))
        if not ids:
            continue
        by_correlation.setdefault(event.correlation_id, set()).update(ids)
        for interaction_id in _interaction_ids(event.correlation_id, interactions, links):
            by_interaction.setdefault(interaction_id, set()).update(ids)
    return by_correlation, by_interaction


def _flag_policies(
    event: EvidenceEvent,
    payload: dict[str, Any],
    by_correlation: dict[str, set[str]],
    by_interaction: dict[str, set[str]],
    interactions: dict[str, Interaction],
    links: Sequence[InteractionLink],
) -> set[str]:
    found = set(_check_policy_ids(payload))
    found.update(by_correlation.get(event.correlation_id, ()))
    interaction = str(payload.get("interaction_id") or "")
    if interaction:
        found.update(by_interaction.get(interaction, ()))
    for interaction_id in _interaction_ids(event.correlation_id, interactions, links):
        found.update(by_interaction.get(interaction_id, ()))
    return found


def _check_policy_ids(payload: dict[str, Any]) -> set[str]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return set()
    found: set[str] = set()
    for check in checks:
        if not isinstance(check, dict) or check.get("policy_id") is None:
            continue
        if str(check.get("check") or "") == "CIRCUIT_BREAKER":
            continue
        found.add(str(check["policy_id"]))
    return found


def _usecase(
    event: EvidenceEvent,
    payload: dict[str, Any],
    interactions: dict[str, Interaction],
    links: Sequence[InteractionLink],
) -> str:
    named = str(payload.get("usecase_id") or "").strip()
    if named:
        return named
    envelope = payload.get("envelope")
    if isinstance(envelope, dict) and str(envelope.get("purpose") or "").strip():
        return str(envelope.get("purpose")).strip()
    interaction = _interaction_for(event.correlation_id, interactions, links)
    if interaction is not None and interaction.use_case:
        return interaction.use_case
    interaction_id = str(payload.get("interaction_id") or "")
    row = interactions.get(interaction_id)
    if row is not None:
        return row.use_case
    return ""


def _carried(payload: dict[str, Any], key: str, recorded_key: str) -> set[str]:
    """Versions a flag names, including those on the recorded envelope.

    A recorded flag that says the version was not recorded contributes nothing.
    A denial has no such marker, so the envelope and the attributes count.
    """
    if payload.get(recorded_key) is False:
        return set()
    found: set[str] = set()
    direct = payload.get(key)
    if isinstance(direct, str) and direct.strip():
        found.add(direct.strip())
    attributes = payload.get("attributes")
    if isinstance(attributes, dict):
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            found.add(value.strip())
    envelope = payload.get("envelope")
    if isinstance(envelope, dict):
        arguments = envelope.get("arguments")
        if isinstance(arguments, dict):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                found.add(value.strip())
        if key == "model" and envelope.get("subject_type") == "model":
            subject = envelope.get("subject_id")
            if isinstance(subject, str) and subject.strip():
                found.add(subject.strip())
    return found


def _output_matches(
    selector: str,
    target: str,
    payload: dict[str, Any],
    interaction: Interaction,
    same_correlation: Sequence[EvidenceEvent],
) -> bool:
    if selector == "usecase":
        return interaction.use_case == target
    if selector == "model":
        keys = {
            str(payload.get("model_version") or "").strip(),
            str(payload.get("model") or "").strip(),
            str(payload.get("model_card") or "").strip(),
        }
        provider = str(payload.get("provider") or "").strip()
        model = str(payload.get("model") or "").strip()
        version = str(payload.get("model_version") or "").strip()
        if provider and model and version:
            keys.add(f"{provider}/{model}@{version}")
        keys.discard("")
        return target in keys
    return any(
        target in _check_policy_ids(_payload(event))
        for event in same_correlation
        if event.event_type == EventType.POLICY_DECISION
    )


def _decided_purpose(same_correlation: Sequence[EvidenceEvent]) -> str:
    """The purpose on the decision that released this output, if recorded."""
    for event in same_correlation:
        if event.event_type != EventType.POLICY_DECISION:
            continue
        envelope = _payload(event).get("envelope")
        if isinstance(envelope, dict) and isinstance(envelope.get("purpose"), str):
            return envelope["purpose"]
    return ""


def _unlinked_matches(
    selector: str,
    target: str,
    payload: dict[str, Any],
    same_correlation: Sequence[EvidenceEvent],
) -> bool:
    """Whether output with no interaction falls under the selector."""
    if selector == "usecase":
        return _decided_purpose(same_correlation) == target
    stand_in = Interaction(id="", client_id="", use_case="")
    return _output_matches(selector, target, payload, stand_in, same_correlation)


def _released(event_type: EventType, payload: dict[str, Any]) -> bool:
    """True when the client was given the output.

    A model call is receipt when it was sent or carries a response hash.
    A tool call is receipt when it carries a response hash and was not a
    denial. ``sent`` false or ``withheld`` true is not receipt for either.
    """
    if payload.get("sent") is False or payload.get("withheld") is True:
        return False
    if event_type == EventType.TOOL_CALL:
        outcome = str(payload.get("outcome") or "").strip().upper()
        if outcome == Outcome.DENY.value:
            return False
        return bool(payload.get("response_hash"))
    if payload.get("response_hash"):
        return True
    return payload.get("sent") is True


def _withheld_later(same_correlation: Sequence[EvidenceEvent]) -> bool:
    """True when the gateway withheld the model response after recording it.

    A model call is recorded with its response before the served model is
    checked. When that check withholds the body, a ``POLICY_DECISION`` with
    ``withheld`` true follows on the same correlation id, and the client was
    given the denial, not the output.
    """
    return any(
        event.event_type == EventType.POLICY_DECISION
        and _payload(event).get("withheld") is True
        for event in same_correlation
    )


def _interaction_for(
    correlation_id: str,
    interactions: dict[str, Interaction],
    links: Sequence[InteractionLink],
) -> Interaction | None:
    if correlation_id in interactions:
        return interactions[correlation_id]
    for link in links:
        if link.correlation_id == correlation_id and link.interaction_id in interactions:
            return interactions[link.interaction_id]
    return None


def _interaction_ids(
    correlation_id: str,
    interactions: dict[str, Interaction],
    links: Sequence[InteractionLink],
) -> set[str]:
    found: set[str] = set()
    if correlation_id in interactions:
        found.add(correlation_id)
    for link in links:
        if link.correlation_id == correlation_id and link.interaction_id:
            found.add(link.interaction_id)
    return found


def _probe(
    session: Session,
    usecase_id: str,
    agent_name: str,
    actor: str,
    evidence: EvidenceWriter,
    at: datetime,
) -> Any:
    from kognita.governance import decide, load_snapshot, record

    envelope = Envelope(
        principal=actor,
        purpose=usecase_id,
        tool="kill_switch_drill",
        agent_name=agent_name,
    )
    register = load_register(session, as_of=at)
    evaluation = decide(
        envelope,
        load_snapshot(session, as_of=at),
        usecases=register,
        as_of=at,
    )
    return record(session, evaluation, evidence=evidence, now=at)


def _revocation_holds(session: Session, agent_name: str, at: datetime) -> bool:
    open_rows: tuple[OpenRevocation, ...] = load_revocations(session, at)
    return any(row.dimension == "agent" and row.value == agent_name for row in open_rows)


def _open_breaker(
    session: Session, dimension: str, value: str, at: datetime
) -> Policy | None:
    rows = [
        policy
        for policy in in_force(session, Policy, at)
        if policy.rule_type == "CIRCUIT_BREAKER"
    ]
    for policy in rows:
        if not policy.is_effective(at) or not isinstance(policy.rule, dict):
            continue
        if policy.rule.get("dimension") == dimension and policy.rule.get("value") == value:
            return policy
    return None


def _close_thresholds(session: Session, dimension: str, value: str, at: datetime) -> None:
    rows = session.exec(
        select(BreakerThreshold).where(
            BreakerThreshold.dimension == dimension,
            BreakerThreshold.value == value,
        )
    ).all()
    for row in rows:
        if row.is_effective(at):
            row.effective_to = at
            session.add(row)
    session.flush()


def _close_dues(session: Session, usecase_id: str, at: datetime) -> None:
    rows = session.exec(select(DrillDue).where(DrillDue.usecase_id == usecase_id)).all()
    for row in rows:
        if row.is_effective(at):
            row.effective_to = at
            session.add(row)
    session.flush()


def _material(session: Session, usecase_id: str, at: datetime) -> Any:
    entry = load_register(session, as_of=at).get(usecase_id.strip())
    if entry is None:
        raise BreakerError(f"{CITE_DRILL}: {usecase_id} is not a registered use case")
    if not is_material(entry):
        raise BreakerError(
            f"{CITE_DRILL}: {entry.usecase_id} is not high-materiality "
            f"· accountable owner: {entry.owner}"
        )
    return entry


def _in_window(recorded: datetime, at: datetime, window_seconds: int) -> bool:
    return recorded >= at - timedelta(seconds=window_seconds) and recorded <= at


def _breaker_dimension(value: str) -> str:
    kind = value.strip()
    if kind not in BREAKER_DIMENSIONS:
        raise BreakerError(f"{CITE}: {value!r} is not a breaker dimension")
    return kind


def _human(session: Session, actor_id: str, action: str) -> str:
    from kognita.registry import get as get_agent

    actor = actor_id.strip()
    if not actor:
        raise BreakerError(f"{CITE}: a {action} names the person")
    if get_agent(session, actor) is not None:
        raise BreakerError(
            f"{CITE}: {actor} is a registered agent and cannot {action}"
        )
    return actor


def _basis(citation: str, action: str) -> str:
    basis = citation.strip()
    if not basis:
        raise BreakerError(f"{CITE}: a {action} cites its rule")
    return basis


def _emit(
    session: Session,
    evidence: EvidenceWriter,
    *,
    actor: str,
    correlation_id: str,
    body: dict[str, Any],
) -> None:
    evidence.emit(
        session,
        correlation_id=correlation_id,
        event_type=EventType.POLICY_CHANGE,
        actor_type=ActorType.HUMAN,
        actor_id=actor,
        payload=body,
    )


def _payload(event: EvidenceEvent) -> dict[str, Any]:
    payload = event.payload
    return payload if isinstance(payload, dict) else {}


def _aware(value: datetime) -> datetime:
    when = as_utc(value)
    if when is None:
        return utcnow()
    return when


__all__ = [
    "BREAKER_DIMENSIONS",
    "DRILL_DAYS",
    "BreakerError",
    "OpenDrill",
    "affected_clients",
    "circuit_breaker_checks",
    "drill_checks",
    "evaluate_breakers",
    "load_drill_dues",
    "reset_breaker",
    "run_kill_switch_drill",
    "schedule_kill_switch_drill",
    "set_breaker_threshold",
]
