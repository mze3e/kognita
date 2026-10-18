"""Live-run intervention.

A human can observe a live run, pause it, override its latest decision with
their own, restrict its permissions, revoke the agent through the kill switch,
and recover it. Each intervention is an ``INTERVENTION`` event on the
interaction the run is attached to. None of them rewrites a stored decision.

The next governed call on a paused, recovered, or revoked run is a denial.
A deny override is a denial of the next call. An allow override is recorded
and does not outrank a later policy denial. A restrict adds tools to the
denylist and may lower the classification ceiling; it does not raise either.

Recover executes registered compensations in reverse order and records every
outcome. Missing callbacks and failed rollbacks remain explicit in evidence.
Revoke engages the per-agent kill switch and signals active proxy calls.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlmodel import Session, select

from kognita.envelope import Check
from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.models import (
    CompensatingAction,
    EvidenceEvent,
    InteractionLink,
    RunRecord,
    as_utc,
    utcnow,
)
from kognita.origination import authorised_humans
from kognita.registry import set_kill_switch
from kognita.tools import Run
from kognita.vocabulary import (
    ActorType,
    CheckResult,
    Classification,
    EventType,
    Outcome,
    classification_rank,
)

_LIVE = frozenset({"live", "paused"})
_STOPPED = frozenset({"recovered", "revoked"})
_OVERRIDE = frozenset({"allow", "deny", "escalate"})


class InterventionError(KognitaError):
    """An intervention could not be recorded as requested."""


def observe_run(
    session: Session,
    run_id: str,
    *,
    evidence: EvidenceWriter,
    human: str,
    citation: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record what the run is doing. The run is not changed."""
    at = now or utcnow()
    run, interaction_id = _ready(session, run_id, human, citation, at)
    payload = _base(run, interaction_id, human, "observe", citation, at)
    payload["snapshot"] = _snapshot(run)
    _emit(session, evidence, human, payload)
    return payload


def pause_run(
    session: Session,
    run_id: str,
    *,
    evidence: EvidenceWriter,
    human: str,
    citation: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pause a live run. The next governed call is a denial."""
    at = now or utcnow()
    run, interaction_id = _ready(session, run_id, human, citation, at)
    if run.control_state != "live":
        raise InterventionError(
            f"run {run_id} is {run.control_state} and cannot be paused "
            "(Intervention: pause)"
        )
    from kognita._actions import _stop
    _stop(run=run_id)
    run.control_state = "paused"
    session.add(run)
    session.flush()
    payload = _base(run, interaction_id, human, "pause", citation, at)
    _emit(session, evidence, human, payload)
    return payload


def override_run(
    session: Session,
    run_id: str,
    *,
    evidence: EvidenceWriter,
    human: str,
    citation: str,
    outcome: str,
    decision_sequence: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record the human's outcome against the run's latest policy decision.

    The stored decision is not rewritten. ``deny`` stops the next call.
    ``escalate`` makes the next call an escalation. ``allow`` is recorded
    and leaves policy in force.
    """
    at = now or utcnow()
    if outcome not in _OVERRIDE:
        raise InterventionError(
            f"override outcome {outcome!r} is not allow, deny, or escalate "
            "(Intervention: override)"
        )
    run, interaction_id = _ready(session, run_id, human, citation, at)
    if run.control_state in _STOPPED:
        raise InterventionError(
            f"run {run_id} is {run.control_state} and cannot be overridden "
            "(Intervention: override)"
        )
    decision = _decision(session, interaction_id, decision_sequence)
    if decision is None:
        raise InterventionError(
            f"run {run_id} has no policy decision to override "
            "(Intervention: override)"
        )
    original = _payload(decision).get("outcome")
    run.override_outcome = outcome.upper()
    session.add(run)
    session.flush()
    payload = _base(run, interaction_id, human, "override", citation, at)
    payload["overrides_sequence"] = decision.sequence
    payload["original_outcome"] = original
    payload["outcome"] = outcome
    _emit(session, evidence, human, payload)
    return payload


def restrict_run(
    session: Session,
    run_id: str,
    *,
    evidence: EvidenceWriter,
    human: str,
    citation: str,
    tools: Sequence[str] = (),
    ceiling: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Narrow a live run. Tools are added to the denylist. A ceiling only falls."""
    at = now or utcnow()
    run, interaction_id = _ready(session, run_id, human, citation, at)
    if run.control_state in _STOPPED:
        raise InterventionError(
            f"run {run_id} is {run.control_state} and cannot be restricted "
            "(Intervention: restrict)"
        )
    added = _tool_names(tools)
    new_ceiling = _narrowed_ceiling(run.classification_ceiling, ceiling)
    if not added and new_ceiling is None:
        raise InterventionError(
            "a restriction names a tool to deny or a lower classification ceiling "
            "(Intervention: restrict)"
        )
    before_tools = list(run.tool_denylist or [])
    before_ceiling = run.classification_ceiling
    if added:
        run.tool_denylist = list(dict.fromkeys([*before_tools, *added]))
    if new_ceiling is not None:
        run.classification_ceiling = new_ceiling
    session.add(run)
    session.flush()
    payload = _base(run, interaction_id, human, "restrict", citation, at)
    payload["tools_before"] = before_tools
    payload["tools_after"] = list(run.tool_denylist or [])
    payload["ceiling_before"] = before_ceiling
    payload["ceiling_after"] = run.classification_ceiling
    _emit(session, evidence, human, payload)
    return payload


def recover_run(
    session: Session,
    run_id: str,
    *,
    evidence: EvidenceWriter,
    human: str,
    citation: str,
    now: datetime | None = None,
    _registry=None,
) -> dict[str, Any]:
    """Stop the run and execute its compensating actions, latest first."""
    at = now or utcnow()
    run, interaction_id = _ready(session, run_id, human, citation, at)
    if run.control_state == "recovered":
        raise InterventionError(
            f"run {run_id} is already recovered (Intervention: recover)"
        )
    if run.control_state == "revoked":
        raise InterventionError(
            f"run {run_id} is revoked and is not recovered (Intervention: revoke)"
        )
    steps = _steps(session, run.id)
    run.control_state = "recovered"
    session.add(run)
    session.flush()
    payload = _base(run, interaction_id, human, "recover", citation, at)
    from kognita._actions import _stop, _recover
    _stop(run=run_id)
    _recover(session, steps, evidence, interaction_id, _registry)
    payload["steps"] = steps
    if not steps:
        payload["note"] = (
            "no compensating action was registered; the run is stopped"
        )
    _emit(session, evidence, human, payload)
    return payload


def revoke_run(
    session: Session,
    run_id: str,
    *,
    evidence: EvidenceWriter,
    human: str,
    citation: str,
    agent_name: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Stop the run and engage that agent's kill switch."""
    at = now or utcnow()
    name = agent_name.strip()
    if not name:
        raise InterventionError(
            "revoke names the agent (Intervention: revoke)"
        )
    run, interaction_id = _ready(session, run_id, human, citation, at)
    if run.control_state == "revoked":
        raise InterventionError(
            f"run {run_id} is already revoked (Intervention: revoke)"
        )
    try:
        set_kill_switch(
            session,
            name,
            True,
            evidence=evidence,
            actor_id=human.strip(),
            correlation_id=interaction_id,
            reason=citation.strip(),
            now=at,
        )
    except LookupError as exc:
        raise InterventionError(
            f"{name} is not a registered agent (Intervention: revoke)"
        ) from exc
    from kognita.credentials import revoke
    from kognita.models import _Authority, _RunScope
    from kognita.authority import _emit as _authority_emit
    revoke(session, name, None, evidence=evidence, actor_id=human.strip(), _automatic=True)
    scope = session.get(_RunScope, run_id, populate_existing=True)
    bound_id = scope.body.get('delegation_id') if scope else None
    # The primary binding must be ended even when the agent has a large history.
    bound = session.get(_Authority, bound_id, populate_existing=True) if bound_id else None
    authorities = session.exec(select(_Authority).where(
        _Authority.kind == "delegation", _Authority.ended == False,
        _Authority.grantee == name, _Authority.body['_run'].as_string() == run_id)
        .limit(1001).execution_options(populate_existing=True)).all()
    if bound is not None and bound not in authorities:
        authorities.append(bound)
    for authority in authorities:
        if not authority.ended:
            authority.ended = True
            session.add(authority)
            _authority_emit(session, evidence, human.strip(), "AUTHORITY_REVOKED", authority)
    run.control_state = "revoked"
    session.add(run)
    session.flush()
    payload = _base(run, interaction_id, human, "revoke", citation, at)
    payload["agent"] = name
    _emit(session, evidence, human, payload)
    return payload


def register_compensating_action(
    session: Session,
    run_id: str,
    *,
    evidence: EvidenceWriter,
    human: str,
    citation: str,
    name: str,
    now: datetime | None = None,
) -> CompensatingAction:
    """Append one compensating action for a later recover."""
    at = now or utcnow()
    action = name.strip()
    if not action:
        raise InterventionError(
            "a compensating action has a name (Intervention: recover)"
        )
    run, interaction_id = _ready(session, run_id, human, citation, at)
    if run.control_state not in _LIVE:
        raise InterventionError(
            f"run {run_id} is {run.control_state}; a compensating action is "
            "registered on a live run (Intervention: recover)"
        )
    row = CompensatingAction(
        run_id=run.id,
        name=action,
        citation=citation.strip(),
        recorded_at=at,
    )
    session.add(row)
    session.flush()
    payload = _base(run, interaction_id, human, "register_compensating_action", citation, at)
    payload["name"] = action
    payload["compensating_action_id"] = row.id
    _emit(session, evidence, human, payload)
    return row


def control_checks(
    session: Session,
    run: Run,
    *,
    tool: str,
    classification: Classification,
) -> list[Check]:
    """Checks the next call must carry because of an intervention.

    An empty list means no intervention is in force. The stored decision is
    not read in order to change it.
    """
    state = getattr(run, "control_state", "live") or "live"
    checks: list[Check] = []
    if state == "paused":
        checks.append(_control("pause", CheckResult.FAIL, "the run is paused"))
    elif state == "recovered":
        checks.append(_control("recover", CheckResult.FAIL, "the run was recovered"))
    elif state == "revoked":
        checks.append(_control("revoke", CheckResult.FAIL, "the run was revoked"))
    override = str(getattr(run, "override_outcome", "") or "")
    if override == Outcome.DENY.value:
        checks.append(_control("override", CheckResult.FAIL, "the run was overridden to deny"))
    elif override == Outcome.ESCALATE.value:
        checks.append(
            _control("override", CheckResult.ESCALATE, "the run was overridden to escalate")
        )
    denied = set(getattr(run, "tool_denylist", []) or [])
    if tool and tool in denied:
        checks.append(
            _control("restrict", CheckResult.FAIL, f"{tool} was restricted on this run")
        )
    ceiling = getattr(run, "classification_ceiling", None)
    if (
        ceiling is not None
        and not _at_or_below(classification, ceiling)
        and _was_restricted(session, run.id)
    ):
        checks.append(
            _control(
                "restrict",
                CheckResult.FAIL,
                f"the run ceiling is {ceiling}",
            )
        )
    return checks


def _ready(
    session: Session,
    run_id: str,
    human: str,
    citation: str,
    at: datetime,
) -> tuple[RunRecord, str]:
    party = human.strip()
    if not party:
        raise InterventionError(
            "an intervention names the human (Intervention: human)"
        )
    if not citation.strip():
        raise InterventionError(
            "an intervention cites its rule (Intervention: human)"
        )
    run = session.get(RunRecord, run_id)
    if run is None:
        raise InterventionError(
            f"run {run_id} is not recorded (Intervention: live run)"
        )
    interaction_id = _interaction_id(session, run.id)
    if not interaction_id:
        raise InterventionError(
            f"run {run_id} is not attached to an interaction "
            "(Intervention: interaction)"
        )
    from kognita.interaction import require_interaction

    interaction = require_interaction(session, interaction_id)
    if _is_agent(session, party):
        raise InterventionError(
            f"{party} is a registered agent and cannot intervene "
            "(Intervention: human)"
        )
    allowed = authorised_humans(session, interaction.use_case, at)
    if party not in allowed:
        raise InterventionError(
            f"{party} is not an authorised human for this interaction "
            "(Intervention: human)"
        )
    return run, interaction_id


def _decision(
    session: Session, interaction_id: str, sequence: int | None
) -> EvidenceEvent | None:
    from kognita.interaction import linked_correlation_ids

    ids = linked_correlation_ids(session, interaction_id)
    rows = session.exec(
        select(EvidenceEvent)
        .where(EvidenceEvent.event_type == EventType.POLICY_DECISION)
        .order_by(EvidenceEvent.sequence)
    ).all()
    chosen: EvidenceEvent | None = None
    for event in rows:
        payload = _payload(event)
        on_span = (
            event.correlation_id in ids or payload.get("interaction_id") == interaction_id
        )
        if not on_span:
            continue
        if sequence is not None and event.sequence != sequence:
            continue
        chosen = event
    return chosen


def _interaction_id(session: Session, run_id: str) -> str:
    rows = session.exec(
        select(InteractionLink).where(InteractionLink.run_id == run_id)
    ).all()
    for link in rows:
        if link.interaction_id:
            return link.interaction_id
    return ""


def _steps(session: Session, run_id: str) -> list[dict[str, Any]]:
    rows = session.exec(
        select(CompensatingAction)
        .where(CompensatingAction.run_id == run_id)
        .order_by(CompensatingAction.id)
    ).all()
    ordered = list(reversed(rows))
    return [
        {"name": row.name, "citation": row.citation, "id": row.id} for row in ordered
    ]


def _tool_names(tools: Sequence[str]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, str) or not tool.strip():
            raise InterventionError(
                "a restricted tool has a name (Intervention: restrict)"
            )
        name = tool.strip()
        if name not in names:
            names.append(name)
    return names


def _narrowed_ceiling(current: str | None, ceiling: str | None) -> str | None:
    """The new ceiling when it is lower, else None when the caller left it."""
    if ceiling is None:
        return None
    try:
        requested = Classification(ceiling).value
    except ValueError as exc:
        raise InterventionError(
            f"classification ceiling {ceiling!r} is not a classification "
            "(Intervention: restrict)"
        ) from exc
    if current is not None and classification_rank(requested) > classification_rank(current):
        raise InterventionError(
            "a restriction does not raise the classification ceiling "
            "(Intervention: restrict)"
        )
    if current == requested:
        return None
    return requested


def _was_restricted(session: Session, run_id: str) -> bool:
    # The gateway also uses this check: fetch one match through the event-type
    # index rather than materialising the entire intervention history.
    return session.exec(
        select(EvidenceEvent.id).where(
            EvidenceEvent.event_type == EventType.INTERVENTION,
            EvidenceEvent.payload["run_id"].as_string() == run_id,
            EvidenceEvent.payload["action"].as_string() == "restrict",
        ).limit(1)
    ).first() is not None


def _snapshot(run: RunRecord) -> dict[str, Any]:
    started = as_utc(run.started_at)
    return {
        "control_state": run.control_state,
        "calls_used": run.calls_used,
        "tokens_used": run.tokens_used,
        "cost_usd_used": run.cost_usd_used,
        "tool_denylist": list(run.tool_denylist or []),
        "classification_ceiling": run.classification_ceiling,
        "override_outcome": run.override_outcome,
        "approvals_pending": list(run.approvals_pending or []),
        "started_at": started.isoformat() if started else None,
    }


def _base(
    run: RunRecord,
    interaction_id: str,
    human: str,
    action: str,
    citation: str,
    at: datetime,
) -> dict[str, Any]:
    basis = citation.strip()
    if not basis:
        raise InterventionError(
            f"an intervention cites its rule (Intervention: {action})"
        )
    return {
        "action": action,
        "run_id": run.id,
        "interaction_id": interaction_id,
        "human": human.strip(),
        "citation": basis,
        "control_state": run.control_state,
        "recorded_at": at.isoformat(),
        "checks": [
            {
                "check": action.upper(),
                "regime": "INTERVENTION",
                "result": CheckResult.PASS.value,
                "citation": f"Intervention: {action} — {basis}",
            }
        ],
    }


def _emit(
    session: Session, evidence: EvidenceWriter, human: str, payload: dict[str, Any]
) -> EvidenceEvent:
    return evidence.emit(
        session,
        correlation_id=str(payload.get("interaction_id") or payload["run_id"]),
        event_type=EventType.INTERVENTION,
        actor_type=ActorType.HUMAN,
        actor_id=human.strip(),
        payload=payload,
    )


def _control(action: str, result: CheckResult, detail: str) -> Check:
    return Check(
        check=action.upper(),
        regime="INTERVENTION",
        result=result,
        citation=f"Intervention: {action} — {detail}",
    )


def _payload(event: EvidenceEvent) -> dict[str, Any]:
    payload = event.payload
    if isinstance(payload, dict):
        return payload
    return {}


def _is_agent(session: Session, name: str) -> bool:
    from kognita.models import Agent

    return session.exec(select(Agent).where(Agent.name == name)).first() is not None


def _at_or_below(value: Classification | str, ceiling: Classification | str) -> bool:
    return classification_rank(value) <= classification_rank(ceiling)


__all__ = [
    "InterventionError",
    "control_checks",
    "observe_run",
    "override_run",
    "pause_run",
    "recover_run",
    "register_compensating_action",
    "restrict_run",
    "revoke_run",
]
