"""Test harness: wires a pack to an in-memory store with a frozen clock.

Time is frozen because effective-dating and approval expiry are decided by
comparisons against "now"; a suite that reads the wall clock is a suite that
fails at midnight on the last day of the month.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, Sequence

from sqlmodel import Session, select

from kognita.db import create_all, make_engine
from kognita.embedding import HashingEmbedder
from kognita.envelope import Envelope, Evaluation
from kognita.evidence import EvidenceWriter
from kognita.governance import decide, load_snapshot, record
from kognita.models import UseCase
from kognita.tools import ToolRegistry, ToolRun, run_governed
from kognita.usecases import (
    ACCOUNTABILITY,
    Appetite,
    UseCaseSpec,
    activate,
    approve_rating,
    load_register,
    model_card_id,
    register as register_usecase,
    register_model_card,
    set_appetite,
)

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
        _ensure_live_purposes(self)

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
            usecases=load_register(session, as_of=at),
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


_LOW_COMPLEXITY = {
    "technology": "established",
    "novelty": "familiar",
    "explainability": "explainable",
    "third_party_visibility": "not_applicable",
}


def _ensure_live_purposes(harness: Harness) -> None:
    """Give each harness purpose a live tier-1 use case when the store has none.

    Execution paths deny an empty register. A purpose the caller already
    stored is left as it is, and an appetite already in force is not replaced.
    """
    purposes = tuple(dict.fromkeys(str(item).strip() for item in harness.purposes if str(item).strip()))
    if not purposes:
        return
    with Session(harness.engine) as session:
        existing = set(session.exec(select(UseCase.usecase_id)).all())
        missing = [item for item in purposes if item not in existing]
        if not missing:
            return
        card = ""
        if "COLLABORATION" in missing:
            # Reconstruction sends this model under COLLABORATION. A registered
            # use case with no approved model blocks that send.
            card = model_card_id(
                "api.openai.com", "gateway-model", "gateway-model-2026-06-01"
            )
            register_model_card(
                session,
                provider="api.openai.com",
                model="gateway-model",
                version="gateway-model-2026-06-01",
                intended_use="Harness model call",
                limitations="Fixture card",
                evaluation_results="Not evaluated",
                approver="Luis Ortega",
                approved_at=harness.now,
                evidence=harness.evidence,
                actor_id="Luis Ortega",
                at=harness.now,
            )
        if load_register(session, as_of=harness.now).appetite is None:
            set_appetite(
                session,
                Appetite(
                    allowed_categories=("research summarization",),
                    prohibited_categories=(),
                    maximum_autonomy=4,
                    permitted_client_impact=("internal productivity",),
                    external_communication=False,
                    transaction_authority=False,
                    maximum_residual_materiality="high",
                ),
                evidence=harness.evidence,
                actor_id="CRO",
                senior_management=True,
                at=harness.now,
            )
        for usecase_id in missing:
            spec = UseCaseSpec(
                usecase_id=usecase_id,
                purpose=f"Harness purpose {usecase_id}",
                category="research summarization",
                business_owner="Jane Smith",
                technology_owner="Omar Aziz",
                risk_owner="Priya Shah",
                model_owner="Luis Ortega",
                knowledge_lead="",
                accountability={seat: "Jane Smith" for seat in ACCOUNTABILITY},
                control_function="Alex Chen",
                builders=("Ann Builder",),
                business_criticality="low",
                client_impact="internal productivity",
                autonomy="none",
                decision_consequence="informational",
                data_sensitivity="internal",
                complexity=_LOW_COMPLEXITY,
                inherent_materiality="medium",
                credited_controls=("entitlement filter",),
                residual_materiality="low",
                approval_gate="none",
                review_sampling_rate=0.1,
                maximum_autonomy=1,
                monitoring=True,
                explanation_depth="citation",
                validators=("Jane Smith",),
                approved_models=(card,) if usecase_id == "COLLABORATION" and card else (),
            )
            register_usecase(
                session, spec, evidence=harness.evidence, actor_id="Jane Smith", at=harness.now
            )
            approve_rating(
                session,
                usecase_id,
                spec.control_function,
                evidence=harness.evidence,
                at=harness.now,
            )
            activate(
                session,
                usecase_id,
                evidence=harness.evidence,
                actor_id=spec.control_function,
                at=harness.now,
                status="live",
            )
        session.commit()


__all__ = ["Harness", "PackUnderTest", "FROZEN_NOW"]
