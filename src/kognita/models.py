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

from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    Text,
    TypeDecorator,
    UniqueConstraint,
    event,
    inspect as sa_inspect,
)
from sqlalchemy.types import JSON, DateTime
from sqlmodel import Field, SQLModel

from kognita.canonical import canonical_hash
from kognita.exceptions import PolicyEditError

from kognita.vocabulary import (
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
    #: Hash of the rule as stored. The writer stamps it. Replay does not trust it:
    #: a decision pins its own copy of this hash, and an in-place edit is refused
    #: when the recomputed hash no longer matches the stamp.
    content_hash: str = Field(default="")

    def is_effective(self, at: datetime) -> bool:
        """Whether this policy is in force at ``at`` (half-open interval)."""
        start = as_utc(self.effective_from)
        end = as_utc(self.effective_to)
        assert start is not None
        return start <= at and (end is None or end > at)


def policy_content(policy: Policy) -> dict[str, Any]:
    """The policy row as evaluated, without the window's closing instant.

    ``effective_to`` is how a row is closed. It is not part of the rule that
    ran, so superseding a policy does not change the hash a decision pinned.
    """
    start = as_utc(policy.effective_from)
    return {
        "regime": policy.regime,
        "rule_type": policy.rule_type,
        "applies_to": policy.applies_to,
        "rule": policy.rule,
        "citation": policy.citation,
        "effective_from": start.isoformat() if start else None,
    }


def policy_content_hash(policy: Policy) -> str:
    """Hash of :func:`policy_content`."""
    return canonical_hash(policy_content(policy))


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

    Supports both single-signature and two-signature approvals (ADR 0006):
    - Single: requester asks, approver grants → status=APPROVED
    - Two-signature: requester asks, approver marks, second approver confirms
    """

    __tablename__ = "approvals"

    id: int | None = Field(default=None, primary_key=True)
    decision_id: int = Field(index=True, foreign_key="governance_decisions.id")
    envelope_hash: str = Field(index=True)
    #: Proposal this approval grants (optional; single-sig approvals have none)
    proposal_id: int | None = Field(index=True, foreign_key="proposals.id")
    #: Who requested the approval (the requester)
    requester_id: str = ""
    #: Who will approve/confirm (may differ from requester)
    approver_id: str | None = None
    #: First signature state (PENDING, MARKED, APPROVED, REJECTED, EXPIRED)
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING, index=True)
    #: Second signature state for two-signature gates (None for single-sig)
    confirmation_status: ApprovalStatus | None = None
    scope: str = ""
    reason: str | None = None
    #: When first signature was granted (status=APPROVED)
    approved_at: datetime | None = Field(default=None, sa_column=_nullable_utc_column())
    #: When second signature was granted (confirmation_status=APPROVED)
    confirmed_at: datetime | None = Field(default=None, sa_column=_nullable_utc_column())
    created_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    expires_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    decided_at: datetime | None = Field(default=None, sa_column=_nullable_utc_column())
    #: Deprecated: use requester_id and approver_id instead
    approver_name: str = "Unassigned (duty desk)"

    def is_live(self, at: datetime) -> bool:
        """Pending and not yet expired at ``at``."""
        expiry = as_utc(self.expires_at)
        return self.status == ApprovalStatus.PENDING and (
            expiry is None or expiry > at
        )


class Proposal(SQLModel, table=True):
    """A proposed change awaiting human review and approval (ADR 0007).

    Captures the before-state, the kind of change, rationale and proposer.
    Approvals reference this so the human's signature binds to the exact change.
    Once approved, the proposal applies atomically with no diff substitution.
    """

    __tablename__ = "proposals"

    id: int | None = Field(default=None, primary_key=True)
    #: Kind of change: "document_edit", "policy_update", "criterion_mark", etc.
    kind: str = Field(index=True)
    #: Before-state snapshot (JSON)
    before_snapshot: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    #: Proposed change data (what the proposer is asking for)
    change: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    #: Change description (rationale)
    rationale: str = ""
    #: Subject being changed (optional type)
    subject_type: str | None = None
    #: Subject being changed (optional id)
    subject_id: str | None = None
    #: Proposer's identity
    proposer_id: str = ""
    created_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


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
    #: Row ids the payload cites. Same names as the payload keys. Not part of
    #: the hashed header: the payload already carries them, and the hash covers
    #: that payload. SQLite rejects a value with no target row. Content hashes
    #: are not columns here; erasure removes the bytes and the chain keeps the hash.
    approval_id: int | None = Field(default=None, index=True, foreign_key="approvals.id")
    policy_id: int | None = Field(default=None, index=True, foreign_key="policies.id")
    successor_id: int | None = Field(default=None, index=True, foreign_key="policies.id")
    run_id: str | None = Field(default=None, index=True, foreign_key="runs.id")
    continuation_hash: str | None = Field(
        default=None, index=True, foreign_key="continuations.content_hash"
    )


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


class RunRecord(SQLModel, table=True):
    """One run's budgets and counters.

    The continuation itself is not stored here. ``continuation_hash`` is a
    content-hash handle into :class:`Continuation`; rehydration is a fetch by
    that hash, so the run row stays small.
    """

    __tablename__ = "runs"

    id: str = Field(primary_key=True)
    max_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    wall_clock_seconds: float | None = None
    classification_ceiling: str | None = None
    approvals_pending: list[int] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    calls_used: int = 0
    tokens_used: int = 0
    cost_usd_used: float = 0.0
    started_at: datetime | None = Field(default=None, sa_column=_nullable_utc_column())
    #: Handle into ``continuations``. Null when the run is not suspended.
    #: The evidence payload records the same hash.
    continuation_hash: str | None = Field(
        default=None, foreign_key="continuations.content_hash"
    )


class Continuation(SQLModel, table=True):
    """Content-addressed store for a suspended run.

    Keyed by the hash of the payload. The run record holds only that hash.
    """

    __tablename__ = "continuations"

    content_hash: str = Field(primary_key=True)
    payload: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))


class RetentionPolicy(SQLModel, table=True):
    """How long one use case keeps content in the retention store.

    The use-case register is a later release. This row is only the retention
    period for a use case the caller names. ``retain_days`` of None keeps the
    content until an explicit erasure.
    """

    __tablename__ = "retention_policies"

    use_case: str = Field(primary_key=True)
    retain_days: int | None = None


class RetainedContent(SQLModel, table=True):
    """Content-addressed prompts, responses, and source snapshots.

    Keyed by the same hash the evidence chain records. The body is the content.
    Erasure deletes the row; the chain keeps the hash and records the erasure.
    """

    __tablename__ = "retained_content"

    content_hash: str = Field(primary_key=True)
    kind: str = Field(index=True)
    use_case: str = Field(default="", index=True)
    correlation_id: str = Field(default="", index=True)
    body: str = Field(sa_column=Column(Text, nullable=False))
    retained_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


class EvidenceCheck(SQLModel, table=True):
    """One ``policy_id`` from an evidence payload's ``checks``.

    A decision cites more than one policy, so the id cannot be a single column
    on ``evidence_events``. The payload stays the hashed body. This row is the
    foreign key. Deleting the event removes the citation. Deleting the policy
    does not.
    """

    __tablename__ = "evidence_checks"

    evidence_event_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("evidence_events.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    policy_id: int = Field(foreign_key="policies.id", primary_key=True)


class EvidenceItem(SQLModel, table=True):
    """One knowledge-item id from an evidence payload.

    ``RETRIEVAL`` stores that id on ``items[].id`` and again in ``returned_ids``.
    One row per id. The payload stays the hashed body.
    """

    __tablename__ = "evidence_items"

    evidence_event_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("evidence_events.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    item_id: int = Field(foreign_key="knowledge_items.id", primary_key=True)


class EntityEdge(SQLModel, table=True):
    """A relationship in the deterministic mirror."""

    __tablename__ = "entity_edges"

    id: int | None = Field(default=None, primary_key=True)
    from_entity_id: int = Field(index=True, foreign_key="entities.id")
    to_entity_id: int = Field(index=True, foreign_key="entities.id")
    type: str = Field(index=True)
    created_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


def _committed(policy: Policy, name: str) -> Any:
    """The last flushed value of ``name`` when this update changes it."""
    state = sa_inspect(policy)
    if name in state.committed_state:
        return state.committed_state[name]
    return getattr(policy, name)


@event.listens_for(Policy, "before_insert")
def _stamp_policy_content_hash(_mapper: Any, _connection: Any, target: Policy) -> None:
    """The stamp is the row's content. A caller-supplied hash is not kept."""
    target.content_hash = policy_content_hash(target)


@event.listens_for(Policy, "before_update")
def _reject_in_place_policy_edit(_mapper: Any, _connection: Any, target: Policy) -> None:
    """Refuse to rewrite a policy that is or was in force.

    Setting ``effective_to`` once, from empty, closes the window. A second
    change to that instant, or any change to the rule itself, is an in-place
    edit. The replacement is a new row.
    """
    state = sa_inspect(target)
    if "effective_to" in state.committed_state:
        previous_end = state.committed_state["effective_to"]
        if previous_end is not None:
            raise PolicyEditError(
                "effective_to is already set; changes must be new effective-dated rows"
            )
    fresh = policy_content_hash(target)
    stored = _committed(target, "content_hash") or ""
    if fresh == stored:
        return
    start = as_utc(_committed(target, "effective_from"))
    if start is not None and start <= utcnow():
        raise PolicyEditError(
            "in-place edit of an effective policy; changes must be new effective-dated rows"
        )
    target.content_hash = fresh


__all__ = [
    "Agent",
    "Policy",
    "GovernanceDecision",
    "Approval",
    "EvidenceEvent",
    "EvidenceCheck",
    "EvidenceItem",
    "KnowledgeItem",
    "Entity",
    "EntityEdge",
    "RetentionPolicy",
    "RetainedContent",
    "policy_content",
    "policy_content_hash",
    "utcnow",
    "as_utc",
]
