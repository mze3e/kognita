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

``EGRESS`` records that data crossed the boundary, by reference: its size and the
request it belongs to, never its content.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence

from sqlmodel import Session, select

from kognita.approvals import (
    ApprovalError,
    find_granted,
    grant,
    reject,
)
from kognita.canonical import canonical_hash, canonical_json
from kognita.envelope import Check, Envelope, Evaluation, envelope_hash
from kognita.evidence import EvidenceWriter
from kognita.governance import (
    PolicySnapshot,
    classifier_derived_envelope,
    decide,
    load_snapshot,
    record,
    resolve_outcome,
)
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
        "classification_ceiling": None if ceiling is None else Classification(ceiling).value,
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
    row.classification_ceiling = None if ceiling is None else Classification(ceiling).value
    row.approvals_pending = list(run.approvals_pending)
    row.calls_used = run.calls_used
    row.tokens_used = run.tokens_used
    row.cost_usd_used = run.cost_usd_used
    row.started_at = run.started_at
    row.continuation_hash = run.continuation_hash
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
            **(extra or {}),
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


def _evaluation_from_continuation(payload: Mapping[str, Any], envelope: Envelope) -> Evaluation:
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
        engages=getattr(pack, "engages", None),
        as_of=as_of,
    )
    evaluation = _with_checks(evaluation, extra_checks)
    evaluation, _breaches = _admit(
        session,
        run,
        evaluation,
        classification=spec.classification,
        cost_usd=cost_usd,
        tokens=tokens,
        now=at,
    )

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
    )

    if not evaluation.allowed:
        if run is not None:
            _save_run(session, run)
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
            )
        return ToolRun(evaluation=evaluation, data=None, approval_required=True)

    if run is not None:
        _save_run(session, run)

    data = _release_tool(session, envelope, evaluation, spec, evidence)
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
    approver_name: str = "Unassigned (duty desk)",
    now: datetime | None = None,
) -> ToolRun | Any:
    """Resume a run suspended on ``HUMAN_APPROVAL``.

    ``approvals_resolved`` maps an approval id to whether the person granted
    it. The held call executes only when that approval is actually granted
    and :func:`kognita.approvals.find_granted` still finds it live. A denial
    resumes the record and does not execute.

    Rehydration fetches the continuation by the hash stored on the run. The
    fetch is evidenced; the payload is not copied onto the run or into the
    resume event.
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
    for aid, is_granted in resolved.items():
        approval = session.get(Approval, aid)
        if approval is None:
            raise ApprovalError(f"approval {aid} does not exist")
        if approval.status == ApprovalStatus.PENDING:
            if is_granted:
                grant(
                    session,
                    approval,
                    approver_name=approver_name,
                    evidence=evidence,
                    correlation_id=correlation_id,
                    now=at,
                )
            else:
                reject(
                    session,
                    approval,
                    approver_name=approver_name,
                    evidence=evidence,
                    correlation_id=correlation_id,
                    now=at,
                )
        if approval.status != ApprovalStatus.PENDING:
            run.approvals_pending = [item for item in run.approvals_pending if item != aid]

    granted_flag = resolved[approval_id]
    envelope = _envelope_from_continuation(payload)
    evaluation = _evaluation_from_continuation(payload, envelope)
    clock_hit = _wall_clock_exceeded(run, at)
    live = (
        find_granted(session, evaluation.envelope_hash, now=at) if granted_flag else None
    )
    gating = session.get(Approval, approval_id)
    execute = (
        granted_flag
        and live is not None
        and gating is not None
        and gating.status == ApprovalStatus.APPROVED
        and not clock_hit
    )

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
        },
    )
    run.continuation_hash = None
    _save_run(session, run)

    if not execute:
        # A granted approval that outlives the run's clock is still a budget DENY.
        # A denied approval stays a denial and does not execute.
        if clock_hit and granted_flag:
            evaluation = _deny_for_budgets(evaluation, ["wall_clock_seconds"])
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
    data = _release_tool(session, envelope, evaluation, spec, evidence)
    return ToolRun(
        evaluation=evaluation,
        data=data,
        approval_required=False,
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
