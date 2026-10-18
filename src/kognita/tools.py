"""The governed capability runner.

One ordering, and it is the invariant the whole library rests on::

    envelope → decide → record → (deny: nothing runs) → execute
             → TOOL_CALL evidence → EGRESS evidence

Nothing may reach a tool body before the decision point has allowed it, because
a tool that runs and *then* has its output filtered has already read the data.
Fail-closed means a denied request returns no data at all, not redacted data.

``HUMAN_APPROVAL`` is not permission to release. The outcome means the work may
be prepared, but the tool body does not run and nothing is returned until
:func:`continue_run` finds the approval actually granted. A denial does not
execute. The continuation for that pause is a content-hash handle into a local
store; the run record keeps the handle, not the payload.

A single-signature hold is granted by ``continue_run`` itself when the
resolution is true. A two-signature hold is not. ``mark`` records the first
signature. ``continue_run`` then confirms the second, and only when the
approval is already marked. A true resolution on an approval that is still
pending does not grant it and does not resume the run. The second
signature is ``confirmer`` when that argument is given, and ``approver_name``
otherwise. A blank confirming party, or the same identity as the marker or
the requester, fails closed and leaves the continuation in place.

A tool call that a contact-frequency policy marks as contacting the client
is refused while do-not-contact is in force for that client. The refusal is
a denial on ``run_governed``. ``continue_run`` raises and does not resume.

``EGRESS`` records that data crossed the boundary, by reference: its size and the
request it belongs to, never its content.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

from sqlmodel import Session, select

from kognita.approvals import (
    ApprovalError,
    confirm,
    find_granted,
    grant,
    reject,
)
from kognita.interaction import attach, require_interaction
from kognita.origination import OriginationError, contact_refusal
from kognita.canonical import canonical_hash, canonical_json
from kognita.envelope import Check, Envelope, Evaluation, envelope_hash
from kognita.evidence import EvidenceWriter
from kognita.governance import (
    PolicySnapshot,
    _approval_policy_recheck,
    _standing_checks,
    classifier_derived_envelope,
    decide,
    load_snapshot,
    record,
    resolve_outcome,
)
from kognita.usecases import _governing_register
from kognita.models import Approval, Continuation, RunRecord, as_utc, utcnow
from kognita.retention import RetentionStore
from kognita.vocabulary import (
    ActorType,
    ApprovalStatus,
    CheckResult,
    Classification,
    EventType,
    Outcome,
    at_or_below,
)

ToolFn = Callable[[Envelope, Evaluation, Session], Any]


class ToolNotRegistered(LookupError):
    """No tool with that name is registered."""


@dataclass
class ToolSpec:
    """A registered capability and the classification its output carries."""

    name: str
    fn: ToolFn
    classification: Classification = Classification.C2
    description: str = ""


class ToolRegistry:
    """The set of capabilities a deployment exposes.

    Registration is explicit: a function that is not registered cannot be called
    through the runner, and the runner is the only path that produces evidence.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        fn: ToolFn,
        *,
        classification: Classification = Classification.C2,
        description: str = "",
    ) -> ToolSpec:
        spec = ToolSpec(
            name=name, fn=fn, classification=classification, description=description
        )
        self._tools[name] = spec
        return spec

    def tool(
        self,
        name: str,
        *,
        classification: Classification = Classification.C2,
        description: str = "",
    ) -> Callable[[ToolFn], ToolFn]:
        """Decorator form of :meth:`register`."""

        def decorate(fn: ToolFn) -> ToolFn:
            self.register(
                name, fn, classification=classification, description=description
            )
            return fn

        return decorate

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotRegistered(f"no tool registered as {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


@dataclass
class ToolRun:
    """The outcome of a governed call."""

    evaluation: Evaluation
    data: Any = None
    approval_required: bool = False

    @property
    def outcome(self) -> Outcome:
        return self.evaluation.outcome

    @property
    def denied(self) -> bool:
        return self.data is None and not self.evaluation.allowed


@dataclass
class Run:
    """Budgets for one piece of agent work, across every step that shares it.

    ``max_calls``, ``max_tokens``, ``max_cost_usd`` and ``wall_clock_seconds``
    are hard stops. ``classification_ceiling`` is the most sensitive data a
    call in this run may touch. ``approvals_pending`` lists open
    ``HUMAN_APPROVAL`` decisions waiting on a person.

    Counters (``calls_used``, ``tokens_used``, ``cost_usd_used``) are what the
    next call is checked against. Token spend is recorded whenever a call site
    supplies it, so a later gateway can feed the cap without this slice
    talking to a model provider.

    ``continuation_hash`` is a handle into the local continuation store. The
    payload it names is not kept on the run.
    """

    max_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    wall_clock_seconds: float | None = None
    classification_ceiling: Classification | None = None
    approvals_pending: list[int] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    calls_used: int = 0
    tokens_used: int = 0
    cost_usd_used: float = 0.0
    started_at: datetime | None = None
    continuation_hash: str | None = None
    #: Set by a live-run intervention. ``_refresh_run`` loads them from the store
    #: so a later step sees a pause, a restriction, or an override.
    control_state: str = "live"
    tool_denylist: list[str] = field(default_factory=list)
    override_outcome: str = ""


def _budget_payload(run: Run) -> dict[str, Any]:
    ceiling = run.classification_ceiling
    return {
        "calls_used": run.calls_used,
        "tokens_used": run.tokens_used,
        "cost_usd_used": run.cost_usd_used,
        "max_calls": run.max_calls,
        "max_tokens": run.max_tokens,
        "max_cost_usd": run.max_cost_usd,
        "wall_clock_seconds": run.wall_clock_seconds,
        "classification_ceiling": None
        if ceiling is None
        else Classification(ceiling).value,
    }


def _budget_breaches(
    run: Run,
    *,
    classification: Classification,
    cost_usd: float | None,
    tokens: int | None,
    now: datetime,
) -> list[str]:
    """Names of the caps this call would exceed. Empty means the call is admitted.

    Token and cost caps are enforced only when the call site already knows the
    figure. A missing figure is not a free pass to invent one, and it is not
    treated as zero.
    """
    breached: list[str] = []
    if run.max_calls is not None and run.calls_used >= run.max_calls:
        breached.append("max_calls")
    if (
        tokens is not None
        and run.max_tokens is not None
        and run.tokens_used + tokens > run.max_tokens
    ):
        breached.append("max_tokens")
    if (
        cost_usd is not None
        and run.max_cost_usd is not None
        and run.cost_usd_used + cost_usd > run.max_cost_usd
    ):
        breached.append("max_cost_usd")
    if run.wall_clock_seconds is not None and run.started_at is not None:
        start = as_utc(run.started_at)
        if start is not None and (now - start).total_seconds() > run.wall_clock_seconds:
            breached.append("wall_clock_seconds")
    if run.classification_ceiling is not None and not at_or_below(
        classification, run.classification_ceiling
    ):
        breached.append("classification_ceiling")
    return breached


def _wall_clock_exceeded(run: Run, now: datetime) -> bool:
    if run.wall_clock_seconds is None or run.started_at is None:
        return False
    start = as_utc(run.started_at)
    if start is None:
        return False
    return (now - start).total_seconds() > run.wall_clock_seconds


def _consume(
    run: Run,
    *,
    cost_usd: float | None,
    tokens: int | None,
    now: datetime,
) -> None:
    if run.started_at is None:
        run.started_at = now
    run.calls_used += 1
    if tokens is not None:
        run.tokens_used += tokens
    if cost_usd is not None:
        run.cost_usd_used += float(cost_usd)


def _deny_for_budgets(evaluation: Evaluation, budgets: Sequence[str]) -> Evaluation:
    """Rewrite an evaluation into a DENY that cites each exceeded budget."""
    extra = tuple(
        Check(
            check="BUDGET",
            regime="INTERNAL",
            result=CheckResult.FAIL,
            citation=name,
        )
        for name in budgets
    )
    checks = evaluation.checks + extra
    return Evaluation(
        request_id=evaluation.request_id,
        outcome=Outcome.DENY,
        checks=checks,
        attributes=evaluation.attributes,
        envelope=evaluation.envelope,
        as_of=evaluation.as_of,
        envelope_hash=envelope_hash(evaluation.envelope, evaluation.attributes, checks),
    )


def _refresh_run(session: Session, run: Run) -> None:
    """Load counters from the store so a later step sees what earlier steps spent."""
    row = session.get(RunRecord, run.id)
    if row is None:
        return
    run.calls_used = row.calls_used
    run.tokens_used = row.tokens_used
    run.cost_usd_used = row.cost_usd_used
    run.started_at = as_utc(row.started_at)
    run.approvals_pending = list(row.approvals_pending or [])
    run.continuation_hash = row.continuation_hash
    run.control_state = row.control_state or "live"
    run.tool_denylist = list(row.tool_denylist or [])
    run.override_outcome = row.override_outcome or ""
    if row.classification_ceiling is not None:
        run.classification_ceiling = Classification(row.classification_ceiling)


def _save_run(session: Session, run: Run) -> None:
    row = session.get(RunRecord, run.id)
    if row is None:
        row = RunRecord(id=run.id, approvals_pending=[])
        session.add(row)
    ceiling = run.classification_ceiling
    row.max_calls = run.max_calls
    row.max_tokens = run.max_tokens
    row.max_cost_usd = run.max_cost_usd
    row.wall_clock_seconds = run.wall_clock_seconds
    row.classification_ceiling = (
        None if ceiling is None else Classification(ceiling).value
    )
    row.approvals_pending = list(run.approvals_pending)
    row.calls_used = run.calls_used
    row.tokens_used = run.tokens_used
    row.cost_usd_used = run.cost_usd_used
    row.started_at = run.started_at
    row.continuation_hash = run.continuation_hash
    row.control_state = run.control_state or "live"
    row.tool_denylist = list(run.tool_denylist)
    row.override_outcome = run.override_outcome or ""
    session.add(row)
    session.flush()


def _load_run(session: Session, run_id: str) -> Run:
    row = session.get(RunRecord, run_id)
    if row is None:
        raise ApprovalError(f"no run {run_id!r}")
    ceiling = row.classification_ceiling
    return Run(
        id=row.id,
        max_calls=row.max_calls,
        max_tokens=row.max_tokens,
        max_cost_usd=row.max_cost_usd,
        wall_clock_seconds=row.wall_clock_seconds,
        classification_ceiling=None if ceiling is None else Classification(ceiling),
        approvals_pending=list(row.approvals_pending or []),
        calls_used=row.calls_used,
        tokens_used=row.tokens_used,
        cost_usd_used=row.cost_usd_used,
        started_at=as_utc(row.started_at),
        continuation_hash=row.continuation_hash,
        control_state=row.control_state or "live",
        tool_denylist=list(row.tool_denylist or []),
        override_outcome=row.override_outcome or "",
    )


def _jsonable(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _admit(
    session: Session,
    run: Run | None,
    evaluation: Evaluation,
    *,
    classification: Classification,
    cost_usd: float | None,
    tokens: int | None,
    now: datetime,
) -> tuple[Evaluation, list[str]]:
    """Count the call, or rewrite the evaluation into a budget DENY.

    A denied call does not consume the budget it failed. The snapshot written
    afterwards is the spend that was already on the books.
    """
    if run is None:
        return evaluation, []
    _refresh_run(session, run)
    from kognita.intervention import control_checks

    held = control_checks(
        session,
        run,
        tool=evaluation.envelope.tool,
        classification=classification,
    )
    if held:
        evaluation = _with_checks(evaluation, held)
    breaches = _budget_breaches(
        run,
        classification=classification,
        cost_usd=cost_usd,
        tokens=tokens,
        now=now,
    )
    if breaches:
        return _deny_for_budgets(evaluation, breaches), breaches
    _consume(run, cost_usd=cost_usd, tokens=tokens, now=now)
    return evaluation, []


def _approval_for_decision(session: Session, decision_id: int) -> Approval:
    approval = session.exec(
        select(Approval).where(Approval.decision_id == decision_id)
    ).first()
    if approval is None or approval.id is None:
        raise ApprovalError(f"no approval opened for decision {decision_id}")
    return approval


def _checkpoint(
    session: Session,
    run: Run,
    *,
    envelope: Envelope,
    evaluation: Evaluation,
    approval_id: int,
    evidence: EvidenceWriter,
    kind: str,
    extra: dict[str, Any] | None = None,
    interaction_id: str | None = None,
) -> str:
    """Persist a continuation by content hash and point the run at that hash.

    Checkpoints happen at the policy boundary, once, when the outcome is
    ``HUMAN_APPROVAL``. The run row receives the hash only.
    """
    payload = _jsonable(
        {
            "kind": kind,
            "envelope": envelope.to_dict(),
            "context": dict(envelope.context),
            "request_id": evaluation.request_id,
            "outcome": evaluation.outcome.value,
            "checks": [c.to_dict() for c in evaluation.checks],
            "attributes": evaluation.attributes,
            "as_of": evaluation.as_of.isoformat(),
            "envelope_hash": evaluation.envelope_hash,
            "decision_id": evaluation.decision_id,
            "approval_id": approval_id,
            "policy_ids": _policy_ids_at(session, evaluation.as_of),
            **(extra or {}),
            **({"interaction_id": interaction_id} if interaction_id else {}),
        }
    )
    digest = canonical_hash(payload)
    if session.get(Continuation, digest) is None:
        session.add(Continuation(content_hash=digest, payload=payload))
        session.flush()
    if approval_id not in run.approvals_pending:
        run.approvals_pending.append(approval_id)
    run.continuation_hash = digest
    _save_run(session, run)
    evidence.emit(
        session,
        correlation_id=evaluation.request_id,
        event_type=EventType.APPROVAL,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        classification=Classification.C1,
        payload={
            "action": "REQUESTED",
            "approval_id": approval_id,
            "run_id": run.id,
            "continuation_hash": digest,
        },
    )
    return digest


def _policy_ids_at(session: Session, at: datetime) -> list[int]:
    """Policy ids in force at ``at``, snapshotted when the hold is checkpointed."""
    return [
        policy.id
        for policy in load_snapshot(session, as_of=at).effective(at)
        if policy.id is not None
    ]


def _known_policy_ids(
    session: Session, payload: Mapping[str, Any], at: datetime
) -> set[int]:
    """Ids the hold already saw, or the ids in force at the original instant.

    A continuation written before policy ids were stored has no list. Those
    ids are whatever is effective at the original ``as_of`` when the run
    resumes. A continuation that stored the list, including an empty one,
    uses that list.
    """
    raw = payload.get("policy_ids")
    if isinstance(raw, list):
        return {
            int(item)
            for item in raw
            if isinstance(item, int) and not isinstance(item, bool)
        }
    return set(_policy_ids_at(session, at))


def _envelope_from_continuation(payload: Mapping[str, Any]) -> Envelope:
    data = payload["envelope"]
    return Envelope(
        principal=data["principal"],
        purpose=data["purpose"],
        tool=data["tool"],
        actor_location=data.get("actor_location") or "",
        agent_name=data.get("agent_name"),
        subject_type=data.get("subject_type"),
        subject_id=data.get("subject_id"),
        subjects=dict(data.get("subjects") or {}),
        arguments=dict(data.get("arguments") or {}),
        roles=list(data.get("roles") or []),
        scopes=dict(data.get("scopes") or {}),
        is_admin=bool(data.get("is_admin", False)),
        context=dict(payload.get("context") or {}),
    )


def _evaluation_from_continuation(
    payload: Mapping[str, Any], envelope: Envelope
) -> Evaluation:
    checks = tuple(
        Check(
            check=item["check"],
            regime=item["regime"],
            result=CheckResult(item["result"]),
            citation=item["citation"],
            policy_id=item.get("policy_id"),
            policy_hash=item.get("policy_hash"),
        )
        for item in payload["checks"]
    )
    return Evaluation(
        request_id=payload["request_id"],
        outcome=Outcome(payload["outcome"]),
        checks=checks,
        attributes=dict(payload.get("attributes") or {}),
        envelope=envelope,
        as_of=datetime.fromisoformat(payload["as_of"]),
        envelope_hash=payload["envelope_hash"],
        decision_id=payload.get("decision_id"),
    )


def _load_continuation(session: Session, content_hash: str) -> dict[str, Any]:
    row = session.get(Continuation, content_hash)
    if row is None:
        raise ApprovalError(f"continuation {content_hash} is not in the local store")
    return dict(row.payload)


def _with_checks(evaluation: Evaluation, extra: Sequence[Check]) -> Evaluation:
    """Append checks and resolve the outcome again. An empty addition is a no-op."""
    if not extra:
        return evaluation
    checks = evaluation.checks + tuple(extra)
    return Evaluation(
        request_id=evaluation.request_id,
        outcome=resolve_outcome(checks),
        checks=checks,
        attributes=evaluation.attributes,
        envelope=evaluation.envelope,
        as_of=evaluation.as_of,
        envelope_hash=envelope_hash(evaluation.envelope, evaluation.attributes, checks),
    )


def _call_actor(envelope: Envelope) -> tuple[ActorType, str]:
    """Who the tool evidence names.

    An agent name is an agent. An approved system trigger, carried on the
    envelope context by a gateway after the identity check has passed, is a
    system actor. Anything else stays the human principal the in-process
    runner already records. Context is not an authorization input; the
    decision has already been made.
    """
    if envelope.agent_name:
        return ActorType.AGENT, envelope.agent_name
    trigger = envelope.context.get("system_trigger")
    if isinstance(trigger, str) and trigger:
        return ActorType.SYSTEM, trigger
    return ActorType.HUMAN, envelope.principal


def _release_tool(
    session: Session,
    envelope: Envelope,
    evaluation: Evaluation,
    spec: ToolSpec,
    evidence: EvidenceWriter,
    run: Run | None = None,
) -> Any:
    data = spec.fn(envelope, evaluation, session)
    actor_type, actor_id = _call_actor(envelope)
    response_hash = RetentionStore().retain_value(
        session,
        data,
        kind="tool_response",
        use_case=envelope.purpose,
        correlation_id=evaluation.request_id,
    )
    run_ref = {"run_id": run.id} if run is not None else {}
    evidence.emit(
        session,
        correlation_id=evaluation.request_id,
        event_type=EventType.TOOL_CALL,
        actor_type=actor_type,
        actor_id=actor_id,
        classification=spec.classification,
        payload={
            "tool": spec.name,
            "subjects": envelope.all_subjects(),
            "outcome": evaluation.outcome.value,
            "approval_required": evaluation.approval_required,
            "response_hash": response_hash,
            **run_ref,
        },
    )
    evidence.emit(
        session,
        correlation_id=evaluation.request_id,
        event_type=EventType.EGRESS,
        actor_type=actor_type,
        actor_id=actor_id,
        classification=spec.classification,
        payload={
            "tool": spec.name,
            "request_id": evaluation.request_id,
            "outcome": evaluation.outcome.value,
            "bytes": len(canonical_json(data)),
            "response_hash": response_hash,
            "note": "Payload content is not copied to the evidence plane.",
            **run_ref,
        },
    )
    return data


def _argument_text(envelope: Envelope) -> str:
    """String arguments, in insertion order. Non-strings are not prose to classify."""
    parts: list[str] = []
    for value in envelope.arguments.values():
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts)


def run_governed(
    session: Session,
    envelope: Envelope,
    *,
    registry: ToolRegistry,
    evidence: EvidenceWriter,
    pack: Any,
    snapshot: PolicySnapshot | None = None,
    purposes: Sequence[str] = (),
    as_of: datetime | None = None,
    run: Run | None = None,
    cost_usd: float | None = None,
    tokens: int | None = None,
    now: datetime | None = None,
    classifier: Any | None = None,
    extra_checks: Sequence[Check] = (),
    interaction_id: str | None = None,
) -> ToolRun:
    """Authorise, then execute. The only supported path to a governed tool.

    ``cost_usd`` and ``tokens`` are recorded against ``run`` when the call site
    already knows them. Exceeding a budget is a DENY that cites the budget, and
    the tool body does not run.

    ``classifier`` is the text classifier for free-text arguments. A typed
    classification already on the attributes wins, and this function does not
    decide. ``extra_checks`` are identity checks a gateway already resolved;
    they are part of the same decision, and a failure here means the tool body
    does not run.
    """
    spec = registry.get(envelope.tool)
    at = now or utcnow()
    if interaction_id is not None:
        require_interaction(session, interaction_id)

    # Resolving subjects is a read of the pack's own store, not the governed
    # data: the decision point needs the attributes before it can decide.
    subjects = pack.load_subjects(envelope, session)
    attributes = pack.resolve_attributes(envelope, subjects)
    snap = snapshot if snapshot is not None else load_snapshot(session, as_of=as_of)
    argument_text = _argument_text(envelope)
    if argument_text:
        # Free-text arguments have no typed classification. A caller-supplied
        # classification attribute still wins; identity is not read from the text.
        envelope, attributes = classifier_derived_envelope(
            argument_text,
            envelope,
            attributes=attributes,
            classifier=classifier,
        )

    evaluation = decide(
        envelope,
        snap,
        attributes=attributes,
        subjects=subjects,
        rules=pack.rules(),
        purposes=purposes,
        usecases=_governing_register(session, as_of=as_of or at),
        engages=getattr(pack, "engages", None),
        as_of=as_of,
    )
    evaluation = _with_checks(evaluation, extra_checks)
    evaluation = _with_checks(
        evaluation,
        contact_refusal(session, envelope, interaction_id=interaction_id, now=at),
    )
    evaluation, _breaches = _admit(
        session,
        run,
        evaluation,
        classification=spec.classification,
        cost_usd=cost_usd,
        tokens=tokens,
        now=at,
    )
    if run is not None:
        _save_run(session, run)

    # Evaluation.allowed includes HUMAN_APPROVAL so the work may be prepared.
    # Release is a separate question: an already-granted approval for this
    # exact envelope is the only thing that lets the body run.
    already_granted = (
        find_granted(session, evaluation.envelope_hash, now=at)
        if evaluation.approval_required
        else None
    )
    evaluation = record(
        session,
        evaluation,
        evidence=evidence,
        classification=spec.classification,
        budget=_budget_payload(run) if run is not None else None,
        create_approval=already_granted is None,
        now=at,
        run_id=None if run is None else run.id,
    )

    if not evaluation.allowed:
        if run is not None:
            _save_run(session, run)
        _bind_interaction(session, interaction_id, evaluation, run, evidence, at)
        record_run_decision(session, interaction_id, evaluation, evidence, now=at)
        # Fail closed. The denial is already evidenced by record(); no tool body
        # has run, so there is nothing to leak.
        return ToolRun(
            evaluation=evaluation,
            data=None,
            approval_required=evaluation.approval_required,
        )

    if evaluation.approval_required and already_granted is None:
        # HUMAN_APPROVAL withholds the tool. The continuation, when a run is
        # present, is how continue_run finishes the call after a person decides.
        # There is no final decision until that resume.
        if run is not None:
            approval = _approval_for_decision(session, evaluation.decision_id or 0)
            _checkpoint(
                session,
                run,
                envelope=envelope,
                evaluation=evaluation,
                approval_id=approval.id or 0,
                evidence=evidence,
                kind="tool",
                interaction_id=interaction_id,
            )
        _bind_interaction(session, interaction_id, evaluation, run, evidence, at)
        return ToolRun(evaluation=evaluation, data=None, approval_required=True)

    if run is not None:
        _save_run(session, run)
    _bind_interaction(session, interaction_id, evaluation, run, evidence, at)

    data = _release_tool(session, envelope, evaluation, spec, evidence, run)
    record_run_decision(session, interaction_id, evaluation, evidence, now=at)
    return ToolRun(
        evaluation=evaluation,
        data=data,
        approval_required=False,
    )


def continue_run(
    session: Session,
    run_id: str,
    approvals_resolved: Mapping[int, bool],
    *,
    evidence: EvidenceWriter,
    registry: ToolRegistry | None = None,
    embedder: Any = None,
    subgraph: Callable[..., Any] | None = None,
    approver_name: str = "",
    confirmer: str | None = None,
    now: datetime | None = None,
) -> ToolRun | Any:
    """Resume a run suspended on ``HUMAN_APPROVAL``.

    ``approvals_resolved`` maps an approval id to whether the person granted
    it. ``approver_name`` is the person resolving it. A single-signature grant
    without one, or by the requester, raises
    :class:`~kognita.approvals.ApprovalError` and does not execute. The held call executes only when that approval is actually granted
    and :func:`kognita.approvals.find_granted` still finds it live. A denial
    resumes the record and does not execute.

    A single-signature approval that is still pending is granted or rejected
    here. A two-signature approval is not. ``mark`` is the first signature.
    When that approval is ``MARKED``, a true resolution confirms it and then
    resumes. The confirming party is ``confirmer`` when passed, including an
    empty string, which is a missing signature and is refused. When
    ``confirmer`` is omitted, ``approver_name`` is that party. Either way the
    party must differ from the requester and the marker. A true resolution
    while the approval is still pending raises
    :class:`~kognita.approvals.ApprovalError`, leaves the continuation in
    place, and does not execute.

    A true resolution of a tool that would contact a client under
    do-not-contact raises :class:`~kognita.origination.OriginationError`
    before any signature is applied, and the continuation stays where it is.
    A denial still rejects the hold and does not execute.

    Rehydration fetches the continuation by the hash stored on the run. The
    fetch is evidenced; the payload is not copied onto the run or into the
    resume event. When the continuation names an interaction, the resume
    records a ``FINAL_DECISION``.
    """
    at = now or utcnow()
    run = _load_run(session, run_id)
    if not run.continuation_hash:
        raise ApprovalError(f"run {run_id} has no continuation to resume")

    continuation_hash = run.continuation_hash
    payload = _load_continuation(session, continuation_hash)
    approval_id = int(payload["approval_id"])
    resolved = {int(key): bool(value) for key, value in approvals_resolved.items()}
    if approval_id not in resolved:
        raise ApprovalError(
            f"run {run_id} cannot resume without a resolution for approval {approval_id}"
        )

    correlation_id = str(payload["request_id"])
    envelope = _envelope_from_continuation(payload)
    withheld = _withheld_by_intervention(
        session,
        run,
        envelope,
        payload,
        registry=registry,
        evidence=evidence,
        correlation_id=correlation_id,
        continuation_hash=continuation_hash,
        approval_id=approval_id,
    )
    if withheld is not None:
        return withheld
    blocked = contact_refusal(
        session,
        envelope,
        interaction_id=payload.get("interaction_id")
        if isinstance(payload.get("interaction_id"), str)
        else None,
        now=at,
    )
    if blocked and resolved[approval_id]:
        raise OriginationError(blocked[0].citation)
    ordered = [approval_id] + [aid for aid in resolved if aid != approval_id]
    for aid in ordered:
        is_granted = resolved[aid]
        approval = session.get(Approval, aid)
        if approval is None:
            raise ApprovalError(f"approval {aid} does not exist")
        _resolve_approval(
            session,
            approval,
            granted=is_granted,
            approver_name=approver_name,
            confirmer=confirmer,
            evidence=evidence,
            correlation_id=correlation_id,
            now=at,
        )
        if approval.status != ApprovalStatus.PENDING:
            run.approvals_pending = [
                item for item in run.approvals_pending if item != aid
            ]

    granted_flag = resolved[approval_id]
    evaluation = _evaluation_from_continuation(payload, envelope)
    clock_hit = _wall_clock_exceeded(run, at)
    live = (
        find_granted(session, evaluation.envelope_hash, now=at)
        if granted_flag
        else None
    )
    gating = session.get(Approval, approval_id)
    execute = (
        granted_flag
        and live is not None
        and gating is not None
        and gating.status == ApprovalStatus.APPROVED
        and not clock_hit
    )
    # The approval answers the hold, not what changed while it waited. A kill
    # switch, revocation, breaker, register change, or a policy that came into
    # force since the decision still withholds the call. The stored decision
    # is kept; the recheck is appended.
    recheck: list[Check] = []
    if execute:
        snapshot = load_snapshot(session, as_of=at)
        attributes = dict(evaluation.attributes)
        usecases = _governing_register(session, as_of=at)
        recheck = _standing_checks(
            envelope,
            snapshot,
            attributes=attributes,
            usecases=usecases,
            at=at,
        )
        seen = {(check.check, check.policy_id, check.result) for check in recheck}
        for check in _approval_policy_recheck(
            envelope,
            snapshot,
            attributes=attributes,
            usecases=usecases,
            at=at,
            original=evaluation,
            known_policy_ids=_known_policy_ids(session, payload, evaluation.as_of),
        ):
            key = (check.check, check.policy_id, check.result)
            if key in seen:
                continue
            seen.add(key)
            recheck.append(check)
        if recheck:
            execute = False

    evidence.emit(
        session,
        correlation_id=correlation_id,
        event_type=EventType.APPROVAL,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        classification=Classification.C1,
        payload={
            "action": "RESUMED",
            "approval_id": approval_id,
            "run_id": run.id,
            "continuation_hash": continuation_hash,
            "decision": "approved" if granted_flag else "denied",
            "executed": execute,
            **({"budget": "wall_clock_seconds"} if clock_hit else {}),
            **({"rechecked": [check.check for check in recheck]} if recheck else {}),
        },
    )
    run.continuation_hash = None
    _save_run(session, run)

    if not execute:
        # A granted approval that outlives the run's clock is still a budget DENY.
        # A denied approval stays a denial and does not execute.
        if clock_hit and granted_flag:
            evaluation = _deny_for_budgets(evaluation, ["wall_clock_seconds"])
        if recheck:
            evaluation = _with_checks(evaluation, recheck)
        _record_resume_decision(
            session,
            payload,
            evaluation,
            evidence,
            approver_name=approver_name,
            approval_id=approval_id,
            execute=False,
            granted_flag=granted_flag,
            clock_hit=clock_hit,
            now=at,
        )
        if payload.get("kind") == "retrieval":
            from kognita.broker import BrokerAnswer

            reason = (
                "Request denied by human approval before any data was retrieved."
                if not granted_flag
                else "Request denied before any data was retrieved."
            )
            return BrokerAnswer(
                request_id=evaluation.request_id,
                outcome=evaluation.outcome,
                route=str(payload.get("route") or ""),
                summary=[reason],
                results=[],
                evaluation=evaluation,
            )
        return ToolRun(evaluation=evaluation, data=None, approval_required=False)

    _record_resume_decision(
        session,
        payload,
        evaluation,
        evidence,
        approver_name=approver_name,
        approval_id=approval_id,
        execute=True,
        granted_flag=granted_flag,
        clock_hit=clock_hit,
        now=at,
    )
    if payload.get("kind") == "retrieval":
        from kognita.broker import _resume_retrieval

        return _resume_retrieval(
            session,
            payload,
            envelope=envelope,
            evaluation=evaluation,
            embedder=embedder,
            evidence=evidence,
            subgraph=subgraph,
        )

    if registry is None:
        raise ToolNotRegistered("continuing a tool call requires a registry")
    spec = registry.get(envelope.tool)
    data = _release_tool(session, envelope, evaluation, spec, evidence, run)
    return ToolRun(
        evaluation=evaluation,
        data=data,
        approval_required=False,
    )


def _withheld_by_intervention(
    session: Session,
    run: Run,
    envelope: Envelope,
    payload: Mapping[str, Any],
    *,
    registry: ToolRegistry | None,
    evidence: EvidenceWriter,
    correlation_id: str,
    continuation_hash: str,
    approval_id: int,
) -> ToolRun | None:
    """Keep a held call where it is when an intervention has stopped the run.

    The continuation and the approval stay in place. The stored decision is
    not rewritten. The withhold is a new approval event.
    """
    from kognita.intervention import control_checks

    classification = Classification.C0
    if registry is not None and envelope.tool in registry:
        classification = registry.get(envelope.tool).classification
    elif payload.get("kind") != "retrieval":
        classification = Classification.C3
    held = control_checks(
        session, run, tool=envelope.tool, classification=classification
    )
    if not held:
        return None
    evaluation = _with_checks(_evaluation_from_continuation(payload, envelope), held)
    if evaluation.allowed:
        return None
    evidence.emit(
        session,
        correlation_id=correlation_id,
        event_type=EventType.APPROVAL,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        classification=Classification.C1,
        payload={
            "action": "WITHHELD",
            "approval_id": approval_id,
            "run_id": run.id,
            "continuation_hash": continuation_hash,
            "decision": "withheld",
            "executed": False,
            "citation": held[0].citation,
        },
    )
    return ToolRun(evaluation=evaluation, data=None, approval_required=False)


def _bind_interaction(
    session: Session,
    interaction_id: str | None,
    evaluation: Evaluation,
    run: Run | None,
    evidence: EvidenceWriter,
    now: datetime,
) -> None:
    """Attach this request, and the run when there is one, to the interaction."""
    if not interaction_id:
        return
    if run is not None:
        _save_run(session, run)
    attach(
        session,
        interaction_id,
        evidence=evidence,
        correlation_id=evaluation.request_id,
        run_id=run.id if run is not None else None,
        now=now,
    )


def _rule_citation(evaluation: Evaluation) -> str:
    chosen = evaluation.failures() or evaluation.checks
    texts = [check.citation for check in chosen if check.citation]
    if texts:
        return "; ".join(texts)
    if evaluation.outcome == Outcome.ALLOW:
        return "No policy check failed, and no human approval was required."
    return "The decision failed closed."


def record_run_decision(
    session: Session,
    interaction_id: str | None,
    evaluation: Evaluation,
    evidence: EvidenceWriter,
    *,
    now: datetime,
) -> None:
    """``FINAL_DECISION`` for a run that was not held for approval.

    A hold is not a final decision. The resume records that later. ``OBSERVE``
    is not a client-lifecycle outcome and is left unrecorded.
    """
    if not interaction_id:
        return
    outcome = {
        Outcome.ALLOW: "allow",
        Outcome.DENY: "deny",
        Outcome.ESCALATE: "escalate",
    }.get(evaluation.outcome)
    if outcome is None:
        return
    from kognita.review import record_final_decision

    actor_type, actor_id = _call_actor(evaluation.envelope)
    rm = actor_id if actor_type == ActorType.HUMAN else ""
    record_final_decision(
        session,
        interaction_id,
        outcome=outcome,
        evidence=evidence,
        actor_id=actor_id,
        actor_type=actor_type,
        rm=rm,
        correlation_id=evaluation.request_id,
        citation=_rule_citation(evaluation),
        now=now,
        kind="run",
        checks=[check.to_dict() for check in evaluation.checks],
    )


def _record_resume_decision(
    session: Session,
    payload: Mapping[str, Any],
    evaluation: Evaluation,
    evidence: EvidenceWriter,
    *,
    approver_name: str,
    approval_id: int,
    execute: bool,
    granted_flag: bool,
    clock_hit: bool,
    now: datetime,
) -> None:
    interaction_id = payload.get("interaction_id")
    if not isinstance(interaction_id, str) or not interaction_id:
        return
    if execute:
        outcome = "approve"
    elif not granted_flag:
        outcome = "reject"
    else:
        outcome = "deny"
    del clock_hit
    from kognita.review import record_final_decision

    record_final_decision(
        session,
        interaction_id,
        outcome=outcome,
        evidence=evidence,
        actor_id=approver_name,
        actor_type=ActorType.HUMAN,
        rm=approver_name,
        approver=approver_name,
        correlation_id=str(payload.get("request_id") or interaction_id),
        citation=_rule_citation(evaluation),
        now=now,
        kind="run",
        approval_id=approval_id,
        checks=[check.to_dict() for check in evaluation.checks],
    )


def _second_signature(confirmer: str | None, approver_name: str) -> str:
    """Who confirms a marked two-signature approval.

    ``confirmer`` wins when the caller passed it, and a blank value is a
    missing signature rather than a fall back to ``approver_name``. Omitting
    ``confirmer`` keeps ``approver_name`` as that party.
    """
    if confirmer is not None:
        party = confirmer.strip()
    else:
        party = approver_name.strip()
    if not party:
        raise ApprovalError(
            "the confirming party is missing "
            "(two-signature approval: second signature)"
        )
    return party


def _resolve_approval(
    session: Session,
    approval: Approval,
    *,
    granted: bool,
    approver_name: str,
    confirmer: str | None,
    evidence: EvidenceWriter,
    correlation_id: str,
    now: datetime,
) -> None:
    """Apply one resolution.

    Two-signature approvals that are still pending cannot be granted here.
    Raising leaves the continuation where it is: the caller has not resumed.
    """
    two_signature = approval.confirmation_status is not None
    if approval.status == ApprovalStatus.PENDING and two_signature and granted:
        raise ApprovalError(
            f"approval {approval.id} needs mark() before continue_run() can "
            "confirm it"
        )
    if approval.status == ApprovalStatus.PENDING:
        if granted:
            grant(
                session,
                approval,
                approver_name=approver_name,
                evidence=evidence,
                correlation_id=correlation_id,
                now=now,
            )
        else:
            reject(
                session,
                approval,
                approver_name=approver_name,
                evidence=evidence,
                correlation_id=correlation_id,
                now=now,
            )
        return
    if approval.status == ApprovalStatus.MARKED:
        if granted:
            confirm(
                session,
                approval,
                approver_id=_second_signature(confirmer, approver_name),
                evidence=evidence,
                correlation_id=correlation_id,
                now=now,
            )
        else:
            reject(
                session,
                approval,
                approver_name=approver_name,
                evidence=evidence,
                correlation_id=correlation_id,
                now=now,
            )


__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "ToolRun",
    "ToolNotRegistered",
    "Run",
    "run_governed",
    "continue_run",
]
