"""The Policy Decision Point.

Two functions, deliberately separated:

:func:`decide` is **pure**. Given a policy snapshot it returns an
:class:`~kognita.envelope.Evaluation` and touches nothing — no database, no
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

import json
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Callable, Sequence

from sqlmodel import Session, select

from kognita.canonical import canonical_hash, canonical_json
from kognita.classify import PatternClassifier, classifier_record
from kognita.envelope import Check, Envelope, Evaluation, RuleContext, envelope_hash
from kognita.evidence import EvidenceWriter
from kognita.exceptions import PolicyEditError
from kognita.approvals import open_approval
from kognita.models import (
    Agent,
    Approval,
    GovernanceDecision,
    Policy,
    as_utc,
    policy_content_hash,
    utcnow,
)
from kognita.rules import Evaluator, build_registry
from kognita.vocabulary import (
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

    frozen = tuple(_pin_policy_hashes(checks, snapshot.policies))
    return Evaluation(
        request_id=request_id or str(uuid.uuid4()),
        outcome=resolve_outcome(frozen),
        checks=frozen,
        attributes=attrs,
        envelope=envelope,
        as_of=at,
        envelope_hash=envelope_hash(envelope, attrs, frozen),
    )


def _pin_policy_hashes(checks: list[Check], policies: tuple[Policy, ...]) -> list[Check]:
    """Record the hash of each policy row the check evaluated.

    The hash is taken from the row, not from the evaluator. A check that does
    not name a policy is left alone.
    """
    by_id = {policy.id: policy for policy in policies if policy.id is not None}
    pinned: list[Check] = []
    for check in checks:
        policy = by_id.get(check.policy_id) if check.policy_id is not None else None
        if policy is None:
            pinned.append(check)
            continue
        pinned.append(replace(check, policy_hash=policy_content_hash(policy)))
    return pinned


def _copy_rule(rule: dict[str, Any]) -> dict[str, Any]:
    """A rule payload that does not alias the row it was copied from."""
    return json.loads(canonical_json(rule))


def supersede_policy(
    session: Session,
    policy: Policy,
    *,
    at: datetime,
    rule: dict[str, Any] | None = None,
    citation: str | None = None,
    evidence: EvidenceWriter | None = None,
    actor_id: str = "system",
    correlation_id: str | None = None,
) -> Policy:
    """Close ``policy`` at ``at`` and insert the successor.

    The closed row keeps the rule that already ran. The successor is a new
    row starting at ``at``. An in-place edit of the closed row is refused.
    """
    start = as_utc(policy.effective_from)
    end = as_utc(policy.effective_to)
    if start is None or start > at or not policy.is_effective(at):
        raise PolicyEditError(
            "policy is not effective at this instant; a policy that has not "
            "yet come into force can be edited in place"
        )
    if end is not None and end <= at:
        raise PolicyEditError("policy is already closed")
    policy.effective_to = at
    successor = Policy(
        regime=policy.regime,
        rule_type=policy.rule_type,
        applies_to=policy.applies_to,
        rule=_copy_rule(policy.rule if rule is None else rule),
        citation=policy.citation if citation is None else citation,
        effective_from=at,
    )
    session.add(policy)
    session.add(successor)
    session.flush()
    if evidence is not None:
        evidence.emit(
            session,
            correlation_id=correlation_id or f"policy:{policy.id}",
            event_type=EventType.POLICY_CHANGE,
            actor_type=ActorType.HUMAN,
            actor_id=actor_id,
            payload={
                "action": "SUPERSEDE",
                "policy_id": policy.id,
                "successor_id": successor.id,
                "policy_hash": policy_content_hash(policy),
                "successor_hash": policy_content_hash(successor),
                "at": at.isoformat(),
            },
        )
    return successor


def classifier_derived_envelope(
    text: str,
    envelope: Envelope,
    *,
    attributes: dict[str, Any] | None = None,
    classifier: Any | None = None,
) -> tuple[Envelope, dict[str, Any]]:
    """Fill gaps in a typed envelope from a text classifier. Does not decide.

    The classifier builds attributes. :func:`decide` is what turns them into an
    outcome, and it reads the recorded label rather than calling the classifier
    again. Replay passes those attributes back to :func:`decide` and does not
    come through here.

    Rules that hold:

    - A typed attribute wins. ``classification`` is written only when the
      caller did not supply one.
    - Identity is not read from the text. ``principal``, ``roles``,
      ``is_admin``, ``agent_name``, ``scopes`` and ``actor_location`` stay as
      the caller authenticated them.
    - Purpose and subject are not read from the text either. A purpose token
      would satisfy the purpose vocabulary and turn a closed denial into an
      allow. ``subject_type`` and ``subject_id`` choose which record is in
      scope, and a name in the prose does not get to choose it.
    - The classifier record (model, version, label, calibrated confidence,
      input hash) is stored on the attributes when this function classified.
      The text itself is not stored.
    """
    attrs = dict(attributes or {})
    if attrs.get("classification"):
        return envelope, attrs

    chosen = classifier if classifier is not None else PatternClassifier()
    recorded = classifier_record(chosen, text)
    recorded["filled"] = ["classification"]
    attrs["classification"] = recorded["label"]
    attrs["classifier"] = recorded
    return envelope, attrs


def _evidence_envelope(envelope: Envelope) -> dict[str, Any]:
    """The envelope for a POLICY_DECISION, without the governed argument text.

    ``Envelope.to_dict`` still carries the arguments, and ``envelope_hash``
    still binds them. The evidence payload keeps the hash and size from
    ``hashes_only`` instead of the argument text, the same way MODEL_CALL
    and EGRESS keep hashes and sizes instead of content.
    """
    body = envelope.to_dict()
    arguments = body["arguments"]
    body["arguments"] = {
        "sha256": canonical_hash(arguments),
        "bytes": len(canonical_json(arguments)),
    }
    return body


def record(
    session: Session,
    evaluation: Evaluation,
    *,
    evidence: EvidenceWriter,
    classification: Classification = Classification.C1,
    approval_ttl: timedelta = DEFAULT_APPROVAL_TTL,
    now: datetime | None = None,
    budget: dict[str, Any] | None = None,
    create_approval: bool = True,
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

    # Consumption is evidence of the call, not an input to the decision. Keeping
    # it out of the envelope hash means a budget counter cannot retarget an
    # approval onto a different request. The hash still binds the arguments;
    # the chain keeps that hash, not the text.
    payload: dict[str, Any] = {
        "envelope": _evidence_envelope(envelope),
        "attributes": evaluation.attributes,
        "outcome": evaluation.outcome.value,
        "checks": [c.to_dict() for c in evaluation.checks],
        "envelope_hash": evaluation.envelope_hash,
        "as_of": evaluation.as_of.isoformat(),
    }
    if budget is not None:
        payload["budget"] = budget

    evidence.emit(
        session,
        correlation_id=evaluation.request_id,
        event_type=EventType.POLICY_DECISION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        classification=classification,
        payload=payload,
    )

    if evaluation.outcome == Outcome.HUMAN_APPROVAL and create_approval:
        subjects = envelope.all_subjects()
        scope = f"{envelope.tool} · " + (
            ", ".join(f"{k}={v}" for k, v in sorted(subjects.items())) or "no subject"
        )

        # Check if this is a two-signature approval
        confirmation_required = False
        for check in evaluation.checks:
            if check.policy_id is not None:
                policy = session.exec(
                    select(Policy).where(Policy.id == check.policy_id)
                ).first()
                if policy and policy.rule_type == "TWO_SIGNATURE_APPROVAL":
                    confirmation_required = True
                    break

        approval = open_approval(
            session,
            decision_id=decision.id or 0,
            envelope_hash=evaluation.envelope_hash,
            scope=scope,
            evidence=evidence,
            correlation_id=evaluation.request_id,
            ttl=approval_ttl,
            requester_id=envelope.principal,
            confirmation_required=confirmation_required,
            now=at,
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
    "classifier_derived_envelope",
    "supersede_policy",
    "DEFAULT_APPROVAL_TTL",
]
