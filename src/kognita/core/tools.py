"""The governed capability runner.

One ordering, and it is the invariant the whole library rests on::

    envelope → decide → record → (deny: nothing runs) → execute
             → TOOL_CALL evidence → EGRESS evidence

Nothing may reach a tool body before the decision point has allowed it, because
a tool that runs and *then* has its output filtered has already read the data.
Fail-closed means a denied request returns no data at all, not redacted data.

``EGRESS`` records that data crossed the boundary, by reference: its size and the
request it belongs to, never its content.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

from sqlmodel import Session

from kognita.core.canonical import canonical_json
from kognita.core.envelope import Envelope, Evaluation
from kognita.core.evidence import EvidenceWriter
from kognita.core.governance import PolicySnapshot, decide, load_snapshot, record
from kognita.core.vocabulary import ActorType, Classification, EventType, Outcome

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
) -> ToolRun:
    """Authorise, then execute. The only supported path to a governed tool."""
    spec = registry.get(envelope.tool)

    # Resolving subjects is a read of the pack's own store, not the governed
    # data: the decision point needs the attributes before it can decide.
    subjects = pack.load_subjects(envelope, session)
    attributes = pack.resolve_attributes(envelope, subjects)
    snap = snapshot if snapshot is not None else load_snapshot(session, as_of=as_of)

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
    evaluation = record(
        session, evaluation, evidence=evidence, classification=spec.classification
    )

    actor_type = ActorType.AGENT if envelope.agent_name else ActorType.HUMAN
    actor_id = envelope.agent_name or envelope.principal

    if not evaluation.allowed:
        # Fail closed. The denial is already evidenced by record(); no tool body
        # has run, so there is nothing to leak.
        return ToolRun(evaluation=evaluation, data=None, approval_required=False)

    data = spec.fn(envelope, evaluation, session)

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
            "note": "Payload content is not copied to the evidence plane.",
        },
    )

    return ToolRun(
        evaluation=evaluation,
        data=data,
        approval_required=evaluation.approval_required,
    )


__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "ToolRun",
    "ToolNotRegistered",
    "run_governed",
]
