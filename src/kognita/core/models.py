"""The generic governance tables.

Nothing here knows about any particular business. A domain pack adds its own
system-of-record tables — clients, cases, patients, whatever the domain is — and
Kognita records decisions *about* them by reference (``subject_type`` /
``subject_id``), so the governance plane stays domain-blind.

Timestamps are timezone-aware UTC throughout. Effective-dating comparisons on
naive datetimes fail silently and in the wrong direction, which for a policy
engine means answering "was this allowed?" incorrectly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, TypeDecorator, UniqueConstraint
from sqlalchemy.types import JSON, DateTime
from sqlmodel import Field, SQLModel

from kognita.core.vocabulary import (
    ActorType,
    ApprovalStatus,
    Classification,
    EventType,
    Outcome,
)


def utcnow() -> datetime:
    """Current time, timezone-aware, in UTC."""
    return datetime.now(timezone.utc)


class UtcDateTime(TypeDecorator):
    """A timestamp that is always timezone-aware UTC in Python.

    ``DateTime(timezone=True)`` is not enough. SQLite has no native timestamp
    type and hands back naive datetimes regardless, so a value written as UTC
    returns without its tzinfo and the next comparison raises — or worse, is
    made against a naive "now" and silently answers the wrong way. Since
    effective-dating and approval expiry are both such comparisons, the
    conversion belongs here rather than at every call site.
    """

    impl = DateTime
    cache_ok = True

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("timezone", True)
        super().__init__(*args, **kwargs)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def _utc_column() -> Column:
    return Column(UtcDateTime(), nullable=False)


def _nullable_utc_column() -> Column:
    return Column(UtcDateTime(), nullable=True)


def as_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to a naive datetime read back from a backend that dropped it."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class Agent(SQLModel, table=True):
    """A registered non-human actor, with a named accountable owner.

    An agent absent from this table is unregistered, and an unregistered agent is
    denied — the registry is an allowlist, not a log. ``kill_switch`` is the
    documented way to stop one without a deploy.
    """

    __tablename__ = "agents"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    version: str = ""
    owner_exec: str = ""
    risk_class: str = "MEDIUM"
    materiality_tier: str = "T2"
    kill_switch: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


class Policy(SQLModel, table=True):
    """One effective-dated rule, owned by whoever owns the regime it cites.

    ``rule`` is an open JSON payload interpreted by the evaluator registered for
    ``rule_type``: the core ships primitives, a pack registers the rest. Policies
    are data so they can be edited, versioned and replayed — asking "what did
    this say on the day of the meeting?" is then just ``as_of``.
    """

    __tablename__ = "policies"

    id: int | None = Field(default=None, primary_key=True)
    regime: str = Field(index=True)
    rule_type: str = Field(index=True)
    #: Optional narrowing to one subject kind, e.g. a product type. None = all.
    applies_to: str | None = Field(default=None, index=True)
    rule: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    citation: str = ""
    effective_from: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    effective_to: datetime | None = Field(default=None, sa_column=_nullable_utc_column())

    def is_effective(self, at: datetime) -> bool:
        """Whether this policy is in force at ``at`` (half-open interval)."""
        start = as_utc(self.effective_from)
        end = as_utc(self.effective_to)
        assert start is not None
        return start <= at and (end is None or end > at)


class GovernanceDecision(SQLModel, table=True):
    """The record of one policy evaluation — what a regulator asks to see."""

    __tablename__ = "governance_decisions"

    id: int | None = Field(default=None, primary_key=True)
    request_id: str = Field(index=True, unique=True)
    principal: str = ""
    agent_name: str | None = Field(default=None, index=True)
    purpose: str = ""
    tool: str = ""
    subject_type: str | None = Field(default=None, index=True)
    subject_id: str | None = Field(default=None, index=True)
    #: Domain-resolved attributes the decision turned on (a jurisdiction tuple,
    #: a care relationship, a tenancy) — whatever the pack considers material.
    attributes: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    outcome: Outcome = Field(index=True)
    checks: list[dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    envelope_hash: str = Field(index=True)
    decided_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    #: The instant the policy set was evaluated as of — equal to decided_at for
    #: live decisions, earlier when replaying a historical question.
    as_of: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


class Approval(SQLModel, table=True):
    """A regulated human approval, bound to the exact envelope that was reviewed.

    Binding to ``envelope_hash`` is the point: an approval granted for one request
    cannot be replayed against a different one, because any change to the
    envelope, the resolved attributes or the checks changes the hash.
    """

    __tablename__ = "approvals"

    id: int | None = Field(default=None, primary_key=True)
    decision_id: int = Field(index=True)
    envelope_hash: str = Field(index=True)
    approver_name: str = "Unassigned (duty desk)"
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING, index=True)
    scope: str = ""
    reason: str | None = None
    created_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    expires_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    decided_at: datetime | None = Field(default=None, sa_column=_nullable_utc_column())

    def is_live(self, at: datetime) -> bool:
        """Pending and not yet expired at ``at``."""
        expiry = as_utc(self.expires_at)
        return self.status == ApprovalStatus.PENDING and (
            expiry is None or expiry > at
        )


class EvidenceEvent(SQLModel, table=True):
    """One append-only entry in the evidence plane.

    Each row carries the hash of the row before it, so the log is tamper-evident:
    altering any payload breaks every hash downstream of it and
    ``kognita evidence verify`` says exactly where.

    Payloads default to hashes and references rather than content. An append-only
    log holding personal data collides with erasure rights, so copying content in
    is opt-in per event and not the default.
    """

    __tablename__ = "evidence_events"
    # UNIQUE, not merely indexed. The writer assigns the sequence by reading the
    # current maximum, which is only fork-safe under a single writer. Two writers
    # that computed the same number are rejected here rather than producing a
    # forked chain that verify_chain would only notice later.
    __table_args__ = (UniqueConstraint("sequence", name="uq_evidence_sequence"),)

    id: int | None = Field(default=None, primary_key=True)
    #: Monotonic position in the chain, assigned by the writer under a lock.
    sequence: int = Field(default=0)
    correlation_id: str = Field(index=True)
    event_type: EventType = Field(index=True)
    actor_type: ActorType = ActorType.SYSTEM
    actor_id: str = ""
    classification: Classification = Classification.C1
    payload: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    payload_hash: str = ""
    prev_hash: str = ""
    event_hash: str = Field(default="", index=True)
    recorded_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


class KnowledgeItem(SQLModel, table=True):
    """A retrievable fragment carrying the attributes entitlement is decided on.

    ``classification`` and ``zones`` are not metadata for display — retrieval
    filters on them *before* scoring, so an item outside the caller's entitlement
    is never compared, never ranked, and cannot leak through a relevance score.
    """

    __tablename__ = "knowledge_items"

    id: int | None = Field(default=None, primary_key=True)
    title: str = ""
    body: str = ""
    kind: str = Field(default="DOCUMENT", index=True)
    classification: Classification = Field(default=Classification.C1, index=True)
    #: Zones permitted to hold and serve this item.
    zones: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    source_label: str = ""
    #: L2-normalised embedding, stored as float32 bytes.
    embedding: bytes | None = Field(default=None)
    embedding_dim: int = 0
    embedding_model: str = ""
    published_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


class Entity(SQLModel, table=True):
    """A node in the deterministic mirror of a system of record."""

    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("type", "ref_id", name="uq_entity_ref"),)

    id: int | None = Field(default=None, primary_key=True)
    type: str = Field(index=True)
    ref_id: str = Field(index=True)
    label: str = ""
    classification: Classification = Classification.C1
    properties: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )


class EntityEdge(SQLModel, table=True):
    """A relationship in the deterministic mirror."""

    __tablename__ = "entity_edges"

    id: int | None = Field(default=None, primary_key=True)
    from_entity_id: int = Field(index=True)
    to_entity_id: int = Field(index=True)
    type: str = Field(index=True)
    created_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


__all__ = [
    "Agent",
    "Policy",
    "GovernanceDecision",
    "Approval",
    "EvidenceEvent",
    "KnowledgeItem",
    "Entity",
    "EntityEdge",
    "utcnow",
    "as_utc",
]
