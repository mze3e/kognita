"""The Policy Decision Point.

Two functions, deliberately separated:

:func:`decide` is **pure**. Given a policy snapshot it returns an
:class:`~kognita.core.envelope.Evaluation` and touches nothing — no database, no
clock, no randomness beyond an injected request id. That is what makes a decision
replayable: the same envelope against the same snapshot always yields the same
outcome and the same envelope hash, so "what would this have decided last March?"
is a question with an answer.

:func:`record` persists an evaluation and its evidence, and opens an approval
when one is required.

Splitting them is not tidiness. A decision function that writes cannot be tested
without a database, cannot be replayed without side effects, and cannot be reused
to answer a hypothetical.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Sequence

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from kognita.core.envelope import Check, Envelope, Evaluation, RuleContext, envelope_hash
from kognita.core.evidence import EvidenceWriter
from kognita.core.models import (
    Agent,
    Approval,
    GovernanceDecision,
    Policy,
    as_utc,
    utcnow,
)
from kognita.core.rules import Evaluator, build_registry
from kognita.core.vocabulary import (
    ActorType,
    Classification,
    CheckResult,
    EventType,
    Outcome,
)

#: How long a freshly opened approval stays valid.
DEFAULT_APPROVAL_TTL = timedelta(hours=48)


@dataclass(frozen=True)
class PolicySnapshot:
    """The policies in force at one instant, plus the agent registry.

    Built once by :func:`load_snapshot` and handed to :func:`decide`, so the
    decision function itself never queries anything.
    """

    policies: tuple[Policy, ...] = ()
    agents: dict[str, Agent] = field(default_factory=dict)

    def effective(self, at: datetime) -> tuple[Policy, ...]:
        return tuple(p for p in self.policies if p.is_effective(at))


def load_snapshot(session: Session, *, as_of: datetime | None = None) -> PolicySnapshot:
    """Read the policy set and agent registry, filtered to those in force."""
    at = as_of or utcnow()
    policies = tuple(
        p for p in session.exec(select(Policy)).all() if p.is_effective(at)
    )
    agents = {a.name: a for a in session.exec(select(Agent)).all()}
    return PolicySnapshot(policies=policies, agents=agents)


def resolve_outcome(checks: Sequence[Check]) -> Outcome:
    """Fail closed: the most severe check result present decides.

    Precedence is DENY > ESCALATE > HUMAN_APPROVAL > ALLOW, so a single failure
    among any number of passes still denies. An empty check list allows — a
    request nothing objects to is permitted, which is why the registry checks
    below always contribute at least a purpose check.
    """
    results = {c.result for c in checks}
    if CheckResult.FAIL in results:
        return Outcome.DENY
    if CheckResult.ESCALATE in results:
        return Outcome.ESCALATE
    if CheckResult.REQUIRES_HUMAN in results:
        return Outcome.HUMAN_APPROVAL
    return Outcome.ALLOW


def registry_checks(envelope: Envelope, snapshot: PolicySnapshot) -> list[Check]:
    """Pre-policy checks the core always applies: agent identity and kill switch.

    The registry is an allowlist. An agent nobody registered is denied, because
    the alternative — an unknown agent acting and being logged afterwards — is
    exactly the failure mode a registry exists to prevent.
    """
    if not envelope.agent_name:
        return []

    agent = snapshot.agents.get(envelope.agent_name)
    if agent is None:
        return [
            Check(
                check="AGENT_REGISTRY",
                regime="INTERNAL",
                result=CheckResult.FAIL,
                citation=f"Agent inventory — '{envelope.agent_name}' is not registered",
            )
        ]
    if agent.kill_switch:
        return [
            Check(
                check="KILL_SWITCH",
                regime="INTERNAL",
                result=CheckResult.FAIL,
                citation=f"Kill switch engaged — accountable owner: {agent.owner_exec}",
            )
        ]
    return [
        Check(
            check="AGENT_REGISTRY",
            regime="INTERNAL",
            result=CheckResult.PASS,
            citation=(
                f"{agent.name} v{agent.version} · accountable: {agent.owner_exec} "
                f"· tier {agent.materiality_tier}"
            ),
        )
    ]


def purpose_check(envelope: Envelope, purposes: Sequence[str]) -> Check:
    """The purpose must come from the closed vocabulary the deployment declared."""
    ok = not purposes or envelope.purpose in set(purposes)
    return Check(
        check="PURPOSE",
        regime="INTERNAL",
        result=CheckResult.PASS if ok else CheckResult.FAIL,
        citation="Lawful-basis registry — closed purpose vocabulary",
    )


def decide(
    envelope: Envelope,
    snapshot: PolicySnapshot,
    *,
    attributes: dict[str, Any] | None = None,
    subjects: dict[str, Any] | None = None,
    rules: dict[str, Evaluator] | None = None,
    purposes: Sequence[str] = (),
    engages: Callable[[Policy, RuleContext], bool] | None = None,
    as_of: datetime | None = None,
    request_id: str | None = None,
) -> Evaluation:
    """Evaluate an envelope against a policy snapshot. Pure — writes nothing.

    ``engages`` lets a pack say whether a policy is in scope for this request at
    all. Evaluating a rule from a regime the request never touches produces
    noise at best and a wrong denial at worst, so an un-engaged policy is skipped
    rather than passed.
    """
    at = as_of or utcnow()
    attrs = dict(attributes or {})
    subject_rows = dict(subjects or {})
    registry = rules if rules is not None else build_registry()

    checks: list[Check] = []
    checks.extend(registry_checks(envelope, snapshot))
    checks.append(purpose_check(envelope, purposes))

    context = RuleContext(
        envelope=envelope, attributes=attrs, subjects=subject_rows, as_of=at
    )

    for policy in snapshot.effective(at):
        if engages is not None and not engages(policy, context):
            continue
        evaluator = registry.get(policy.rule_type)
        if evaluator is None:
            # A policy nobody can evaluate must not be silently ignored: it is a
            # rule the deployment believes is in force.
            checks.append(
                Check(
                    check=f"UNEVALUABLE_POLICY: {policy.rule_type}",
                    regime=policy.regime,
                    result=CheckResult.ESCALATE,
                    citation=policy.citation
                    or f"No evaluator registered for rule type '{policy.rule_type}'",
                    policy_id=policy.id,
                )
            )
            continue
        checks.extend(evaluator(policy, context))

    frozen = tuple(checks)
    return Evaluation(
        request_id=request_id or str(uuid.uuid4()),
        outcome=resolve_outcome(frozen),
        checks=frozen,
        attributes=attrs,
        envelope=envelope,
        as_of=at,
        envelope_hash=envelope_hash(envelope, attrs, frozen),
    )


def record(
    session: Session,
    evaluation: Evaluation,
    *,
    evidence: EvidenceWriter,
    classification: Classification = Classification.C1,
    approval_ttl: timedelta = DEFAULT_APPROVAL_TTL,
    now: datetime | None = None,
) -> Evaluation:
    """Persist a decision, evidence it, and open an approval if one is required.

    Returns the evaluation with ``decision_id`` filled in. Everything happens in
    the caller's session, so a rolled-back transaction leaves no record claiming
    a decision was made.
    """
    at = now or utcnow()
    envelope = evaluation.envelope

    decision = GovernanceDecision(
        request_id=evaluation.request_id,
        principal=envelope.principal,
        agent_name=envelope.agent_name,
        purpose=envelope.purpose,
        tool=envelope.tool,
        subject_type=envelope.subject_type,
        subject_id=envelope.subject_id,
        attributes=dict(evaluation.attributes),
        outcome=evaluation.outcome,
        checks=[c.to_dict() for c in evaluation.checks],
        envelope_hash=evaluation.envelope_hash,
        decided_at=at,
        as_of=evaluation.as_of,
    )
    session.add(decision)
    session.flush()

    evidence.emit(
        session,
        correlation_id=evaluation.request_id,
        event_type=EventType.POLICY_DECISION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        classification=classification,
        payload={
            "envelope": envelope.to_dict(),
            "attributes": evaluation.attributes,
            "outcome": evaluation.outcome.value,
            "checks": [c.to_dict() for c in evaluation.checks],
            "envelope_hash": evaluation.envelope_hash,
            "as_of": evaluation.as_of.isoformat(),
        },
    )

    if evaluation.outcome == Outcome.HUMAN_APPROVAL:
        subjects = envelope.all_subjects()
        scope = f"{envelope.tool} · " + (
            ", ".join(f"{k}={v}" for k, v in sorted(subjects.items())) or "no subject"
        )
        approval = Approval(
            decision_id=decision.id or 0,
            envelope_hash=evaluation.envelope_hash,
            scope=scope,
            created_at=at,
            expires_at=at + approval_ttl,
        )
        session.add(approval)
        session.flush()
        evidence.emit(
            session,
            correlation_id=evaluation.request_id,
            event_type=EventType.APPROVAL,
            actor_type=ActorType.SYSTEM,
            actor_id="governance-pdp",
            classification=classification,
            payload={
                "action": "OPENED",
                "approval_id": approval.id,
                "envelope_hash": evaluation.envelope_hash,
                "scope": scope,
                "expires_at": approval.expires_at.isoformat(),
            },
        )

    return Evaluation(
        request_id=evaluation.request_id,
        outcome=evaluation.outcome,
        checks=evaluation.checks,
        attributes=evaluation.attributes,
        envelope=evaluation.envelope,
        as_of=evaluation.as_of,
        envelope_hash=evaluation.envelope_hash,
        decision_id=decision.id,
    )


__all__ = [
    "PolicySnapshot",
    "load_snapshot",
    "decide",
    "record",
    "resolve_outcome",
    "registry_checks",
    "purpose_check",
    "DEFAULT_APPROVAL_TTL",
]
