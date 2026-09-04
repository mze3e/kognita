"""Test harness: wires a pack to an in-memory store with a frozen clock.

Time is frozen because effective-dating and approval expiry are decided by
comparisons against "now"; a suite that reads the wall clock is a suite that
fails at midnight on the last day of the month.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, Sequence

from sqlmodel import Session

from kognita.db import create_all, make_engine
from kognita.embedding import HashingEmbedder
from kognita.envelope import Envelope, Evaluation
from kognita.evidence import EvidenceWriter
from kognita.governance import decide, load_snapshot, record
from kognita.tools import ToolRegistry, ToolRun, run_governed

#: A fixed instant every conformance run decides against.
FROZEN_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


class PackUnderTest(Protocol):
    """What the kit needs from a pack to exercise it."""

    name: str

    def load_subjects(self, envelope: Envelope, session: Any) -> dict[str, Any]: ...
    def resolve_attributes(
        self, envelope: Envelope, subjects: dict[str, Any]
    ) -> dict[str, Any]: ...
    def rules(self) -> dict[str, Any]: ...


@dataclass
class Harness:
    """An in-memory store, a pack, and the machinery to decide against them."""

    pack: Any
    purposes: Sequence[str] = ()
    seed: Callable[[Session], None] | None = None
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    now: datetime = FROZEN_NOW
    embedder: Any = field(default_factory=HashingEmbedder)
    engine: Any = field(init=False, default=None)
    evidence: EvidenceWriter = field(init=False)

    def __post_init__(self) -> None:
        self.engine = make_engine()
        create_all(self.engine)
        self.evidence = EvidenceWriter(self.engine, clock=lambda: self.now)
        if self.seed is not None:
            with Session(self.engine) as session:
                self.seed(session)
                session.commit()

    def session(self) -> Session:
        return Session(self.engine)

    def evaluate(
        self,
        envelope: Envelope,
        session: Session,
        *,
        as_of: datetime | None = None,
    ) -> Evaluation:
        """Decide, without recording. The pure path."""
        at = as_of or self.now
        subjects = self.pack.load_subjects(envelope, session)
        attributes = self.pack.resolve_attributes(envelope, subjects)
        snapshot = load_snapshot(session, as_of=at)
        return decide(
            envelope,
            snapshot,
            attributes=attributes,
            subjects=subjects,
            rules=self.pack.rules(),
            purposes=self.purposes,
            engages=getattr(self.pack, "engages", None),
            as_of=at,
        )

    def decide_and_record(
        self,
        envelope: Envelope,
        session: Session,
        *,
        as_of: datetime | None = None,
    ) -> Evaluation:
        """Decide and persist, as a real caller would."""
        evaluation = self.evaluate(envelope, session, as_of=as_of)
        return record(session, evaluation, evidence=self.evidence, now=self.now)

    def run_tool(
        self, envelope: Envelope, session: Session, *, as_of: datetime | None = None
    ) -> ToolRun:
        """Exercise the full governed path, including tool execution."""
        return run_governed(
            session,
            envelope,
            registry=self.registry,
            evidence=self.evidence,
            pack=self.pack,
            purposes=self.purposes,
            as_of=as_of or self.now,
        )

    def run_governed(
        self, envelope: Envelope, session: Session, *, as_of: datetime | None = None
    ) -> ToolRun:
        """Alias for run_tool — authorise and execute the tool."""
        return self.run_tool(envelope, session, as_of=as_of)


__all__ = ["Harness", "PackUnderTest", "FROZEN_NOW"]
