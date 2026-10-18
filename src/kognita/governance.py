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
from kognita.breaker import (
    OpenDrill,
    circuit_breaker_checks,
    drill_checks,
    load_drill_dues,
)
from kognita.models import (
    Agent,
    Approval,
    GovernanceDecision,
    Policy,
    as_utc,
    in_force,
    policy_content_hash,
    utcnow,
)
from kognita.revocation import OpenRevocation, load_revocations, revocation_checks
from kognita.rules import Evaluator, build_registry
from kognita.usecases import UseCaseRegister, usecase_checks
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
    revocations: tuple[OpenRevocation, ...] = ()
    drill_dues: tuple[OpenDrill, ...] = ()
    _authority: tuple = ()
    _usage: dict = field(default_factory=dict)

    def effective(self, at: datetime) -> tuple[Policy, ...]:
        return tuple(p for p in self.policies if p.is_effective(at))


def load_snapshot(session: Session, *, as_of: datetime | None = None,
                  _evidence=None, _agent_name=None) -> PolicySnapshot:
    """Read the policy set and agent registry in force at ``as_of``.

    Closed policy rows stay in the table and are not loaded. Revocations and
    drill dues are the same kind of read: the open window, not the history.
    """
    at = as_of or utcnow()
    policies = tuple(in_force(session, Policy, at))
    query = select(Agent)
    if _agent_name is not None:
        query = query.where(Agent.name == _agent_name)
    agents = {a.name: a for a in session.exec(query).all()}
    from kognita.autonomy import _step_down
    for agent in agents.values():
        if _evidence is not None and agent.autonomy_level:
            _step_down(session, agent.name, evidence=_evidence, now=at)
    from kognita.authority import _load
    from kognita._limits import _load_usage
    return PolicySnapshot(
        _usage=_load_usage(session, agents, at),
        _authority=_load(session, _agent_name, at, _evidence),
        policies=policies,
        agents=agents,
        revocations=load_revocations(session, at),
        drill_dues=load_drill_dues(session, at),
    )


def resolve_outcome(checks: Sequence[Check]) -> Outcome:
    """Fail closed: the most severe check result present decides.

    Precedence is DENY > ESCALATE > HUMAN_APPROVAL > ALLOW, so a single failure
    among any number of passes still denies. An empty check list allows — a
    request nothing objects to is permitted, which is why the registry checks
    below always contribute at least a purpose check, or a use-case check when
    the caller supplies a register.
    """
    results = {c.result for c in checks}
    if CheckResult.FAIL in results:
        return Outcome.DENY
    if CheckResult.ESCALATE in results:
        return Outcome.ESCALATE
    if CheckResult.REQUIRES_HUMAN in results:
        return Outcome.HUMAN_APPROVAL
    return Outcome.ALLOW


def registry_checks(envelope: Envelope, snapshot: PolicySnapshot, *, at=None, usecases=None) -> list[Check]:
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
    from kognita.autonomy import _checks
    return _checks(envelope, agent, at or utcnow(), usecases) + [
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


def _accountable_owner(envelope: Envelope, snapshot: PolicySnapshot) -> str:
    """Who is accountable on the purpose-list path.

    An agent call cites that agent's ``owner_exec``. A call with no agent cites
    the human principal. An unregistered agent, an agent with no owner, or a
    call with neither an agent owner nor a principal cites nobody, and the
    purpose check then fails closed.
    """
    if envelope.agent_name:
        agent = snapshot.agents.get(envelope.agent_name)
        if agent is None:
            return ""
        return agent.owner_exec.strip()
    return envelope.principal.strip()


def purpose_check(
    envelope: Envelope, purposes: Sequence[str], *, owner: str | None = None
) -> Check:
    """The purpose must come from the closed vocabulary the deployment declared.

    A missing or empty list declares nothing. The check fails closed, the same
    way a purpose outside a declared list fails. It does not pass every purpose.

    Every decision on this path also cites an accountable owner. ``owner`` is
    the agent's accountable owner when :func:`decide` has one. Omitting it
    cites the human principal, which is what a direct call did before an owner
    was a separate argument. An empty owner fails the check even when the
    purpose is listed.

    When :func:`decide` is given a use-case register, this check is not run.
    The register check replaces it: the purpose must be a registered use case,
    and the citation names that use case's accountable owner.
    """
    cited = envelope.principal.strip() if owner is None else owner.strip()
    listed = bool(purposes) and envelope.purpose in set(purposes)
    if not cited:
        return Check(
            check="PURPOSE",
            regime="INTERNAL",
            result=CheckResult.FAIL,
            citation=(
                "Lawful-basis registry — closed purpose vocabulary · "
                "no accountable owner can be cited"
            ),
        )
    citation = (
        "Lawful-basis registry — closed purpose vocabulary · "
        f"accountable owner: {cited}"
    )
    return Check(
        check="PURPOSE",
        regime="INTERNAL",
        result=CheckResult.PASS if listed else CheckResult.FAIL,
        citation=citation,
    )


def decide(
    envelope: Envelope,
    snapshot: PolicySnapshot,
    *,
    attributes: dict[str, Any] | None = None,
    subjects: dict[str, Any] | None = None,
    rules: dict[str, Evaluator] | None = None,
    purposes: Sequence[str] = (),
    usecases: UseCaseRegister | None = None,
    engages: Callable[[Policy, RuleContext], bool] | None = None,
    as_of: datetime | None = None,
    request_id: str | None = None,
) -> Evaluation:
    """Evaluate an envelope against a policy snapshot. Pure — writes nothing.

    ``purposes`` is the list the deployment declared. Missing or empty fails
    closed. A purpose on that list passes this check; any other purpose fails it.
    The check cites an accountable owner: the agent's owner when the envelope
    names an agent, otherwise the principal. No owner fails closed.

    ``usecases`` is a :class:`~kognita.usecases.UseCaseRegister`. When it is
    given, it replaces the purpose list: the envelope's purpose must be a
    registered use case, and a passing check cites the accountable owner.

    ``engages`` lets a pack say whether a policy is in scope for this request at
    all. Evaluating a rule from a regime the request never touches produces
    noise at best and a wrong denial at worst, so an un-engaged policy is skipped
    rather than passed. A revocation, a kill-switch drill that is overdue or
    missing on a high-materiality pilot or live use case, and a tripped
    circuit breaker are applied even when ``engages`` would skip them.
    """
    at = as_of or utcnow()
    request_id = request_id or str(uuid.uuid4())
    attrs = dict(attributes or {})
    agent = snapshot.agents.get(envelope.agent_name)
    if agent is not None:
        attrs["deployment_hash"] = agent.deployment_hash
        if agent.blast_radius is not None:
            attrs["_s4_usage"] = [[t.isoformat(), cs, v, req] for t, cs, v, req in snapshot._usage.get(agent.name, ())]
    from kognita.authority import _grant_ids
    if envelope.agent_name:
        attrs["_authority_grants"] = _grant_ids(envelope, snapshot, at)
    if envelope.agent_name:
        attrs["_s4_clients"] = [envelope.all_subjects()["client"]] if "client" in envelope.all_subjects() else []
    value = envelope.arguments.get("value", 0)
    from math import isfinite
    attrs["_s4_value"] = value if type(value) in (int, float) and isfinite(value) and value >= 0 else 0
    subject_rows = dict(subjects or {})
    registry = rules if rules is not None else build_registry()

    checks: list[Check] = []
    from kognita.authority import _checks as _authority_checks
    from kognita._limits import _checks as _limit_checks
    checks.extend(_limit_checks(envelope, snapshot, at))
    checks.extend(_authority_checks(envelope, snapshot, at))
    checks.extend(registry_checks(envelope, snapshot, at=at, usecases=usecases))
    if usecases is not None:
        checks.extend(usecase_checks(envelope, usecases, at))
    else:
        checks.append(
            purpose_check(
                envelope, purposes, owner=_accountable_owner(envelope, snapshot)
            )
        )
    checks.extend(
        revocation_checks(
            envelope,
            snapshot.revocations,
            attributes=attrs,
            register=usecases,
        )
    )
    checks.extend(
        drill_checks(envelope, snapshot.drill_dues, register=usecases, at=at)
    )
    checks.extend(circuit_breaker_checks(envelope, snapshot.policies, attrs, at))

    context = RuleContext(
        envelope=envelope, attributes=attrs, subjects=subject_rows, as_of=at
    )

    for policy in snapshot.effective(at):
        if policy.rule_type == "CIRCUIT_BREAKER":
            continue
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


def _standing_checks(
    envelope: Envelope,
    snapshot: PolicySnapshot,
    *,
    attributes: dict[str, Any] | None = None,
    usecases: UseCaseRegister | None = None,
    at: datetime,
    _request_id: str | None = None,
) -> list[Check]:
    """The gates that can close after a decision, as they stand at ``at``.

    The agent registry and kill switch, the use-case register when it
    governs, open revocations, overdue drills and tripped circuit breakers.
    A held call re-reads these before it is released, because an approval
    granted later does not outrank a kill switch or a revocation that came
    first. Only failures and escalations are returned.
    """
    attrs = dict(attributes or {})
    checks: list[Check] = []
    agent = snapshot.agents.get(envelope.agent_name)
    if agent is not None and attrs.get("deployment_hash") != agent.deployment_hash:
        checks.append(Check(check="S1-DEPLOYMENT-HOLD", regime="INTERNAL",
                            result=CheckResult.FAIL, citation="S1-DEPLOYMENT-HOLD"))
    from kognita.authority import _checks as _authority_checks
    from kognita._limits import _checks as _limit_checks
    checks.extend(_limit_checks(envelope, snapshot, at, _exclude=_request_id))
    checks.extend(_authority_checks(envelope, snapshot, at))
    checks.extend(registry_checks(envelope, snapshot, at=at, usecases=usecases))
    if usecases is not None:
        checks.extend(usecase_checks(envelope, usecases, at))
    checks.extend(
        revocation_checks(envelope, snapshot.revocations, attributes=attrs, register=usecases)
    )
    checks.extend(drill_checks(envelope, snapshot.drill_dues, register=usecases, at=at))
    checks.extend(circuit_breaker_checks(envelope, snapshot.policies, attrs, at))
    return [
        check
        for check in _pin_policy_hashes(checks, snapshot.policies)
        if check.result in (CheckResult.FAIL, CheckResult.ESCALATE)
    ]


def _approval_policy_recheck(
    envelope: Envelope,
    snapshot: PolicySnapshot,
    *,
    attributes: dict[str, Any] | None,
    usecases: UseCaseRegister | None,
    at: datetime,
    original: Evaluation,
    known_policy_ids: set[int],
    subjects: dict[str, Any] | None = None,
    rules: dict[str, Evaluator] | None = None,
    engages: Callable[[Policy, RuleContext], bool] | None = None,
    pack_unavailable: str | None = None,
) -> list[Check]:
    """Policies that were not in force at the hold and are stricter than this approval.

    The approval satisfies a ``HUMAN_APPROVAL`` check whose policy was already
    in force. A policy that was not in that set, and that now fails, escalates,
    or requires a human approval this grant does not satisfy, withholds the
    release. Policies already known at the checkpoint are not re-litigated
    here. Standing checks cover the kill switch, the register, revocations,
    drills and breakers.

    The new policies are evaluated with the pack that held the call: its
    ``rules``, its ``engages`` scope, and its ``subjects``, as the original
    decision was. When that pack cannot be restored, ``pack_unavailable``
    says why, and every policy that came into force during the hold
    escalates with that reason in its citation. It is not evaluated without
    the pack's scope, and it is not skipped.
    """
    if pack_unavailable is not None:
        unseen = [
            policy
            for policy in snapshot.effective(at)
            if policy.id is not None and policy.id not in known_policy_ids
        ]
        return _pin_policy_hashes(
            [
                Check(
                    check="PACK_CONTEXT",
                    regime=policy.regime,
                    result=CheckResult.ESCALATE,
                    citation=(
                        f"Policy {policy.id} ({policy.citation}) came into force while "
                        f"the call was held, and the pack that held it could not be "
                        f"restored at resume: {pack_unavailable}. It cannot be "
                        f"evaluated in that pack's scope, so the release is withheld."
                    ),
                    policy_id=policy.id,
                )
                for policy in unseen
            ],
            snapshot.policies,
        )
    fresh = decide(
        envelope,
        snapshot,
        attributes=attributes,
        subjects=subjects,
        rules=rules,
        usecases=usecases,
        engages=engages,
        as_of=at,
        request_id=original.request_id,
    )
    satisfied = {
        check.policy_id
        for check in original.checks
        if check.result is CheckResult.REQUIRES_HUMAN and check.policy_id is not None
    }
    withheld: list[Check] = []
    for check in fresh.checks:
        if check.policy_id is None or check.policy_id in known_policy_ids:
            continue
        if check.result in (CheckResult.FAIL, CheckResult.ESCALATE):
            withheld.append(check)
            continue
        if check.result is CheckResult.REQUIRES_HUMAN and check.policy_id not in satisfied:
            withheld.append(check)
    return withheld


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
    run_id: str | None = None,
) -> Evaluation:
    """Persist a decision, evidence it, and open an approval if one is required.

    Returns the evaluation with ``decision_id`` filled in. Everything happens in
    the caller's session, so a rolled-back transaction leaves no record claiming
    a decision was made.
    """
    at = now or utcnow()
    from kognita._limits import _recheck
    evaluation = _recheck(session, evaluation, at)
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
    from kognita._limits import _reserve
    _reserve(session, evaluation, at)
    from kognita.authority import _record_use
    _record_use(session, evaluation, evidence)

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
    named_run = (run_id or "").strip()
    if named_run:
        # The run row must already exist: this id is a foreign key, and the
        # flight recorder reads the run from this event rather than from the
        # run table.
        payload["run_id"] = named_run

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
