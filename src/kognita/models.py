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
    Index,
    Integer,
    Text,
    TypeDecorator,
    UniqueConstraint,
    event,
    inspect as sa_inspect,
    or_,
)
from sqlalchemy.types import JSON, DateTime
from sqlmodel import Field, SQLModel, select

from kognita.canonical import canonical_hash
from kognita.exceptions import KognitaError, PolicyEditError

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


def in_force(session: Any, model: type[Any], at: datetime) -> list[Any]:
    """Rows of ``model`` in force at ``at``.

    The predicate is the same half-open window as ``is_effective``: a row
    whose ``effective_to`` equals ``at`` is already closed. Closed history
    stays in the table and is not loaded. Every row the query returns is
    checked with ``is_effective`` again, so a row that is not in force is
    dropped.
    """
    when = as_utc(at)
    if when is None:
        return []
    rows = session.exec(
        select(model).where(
            model.effective_from <= when,
            or_(model.effective_to.is_(None), model.effective_to > when),
        ).execution_options(populate_existing=True)
    ).all()
    return [row for row in rows if row.is_effective(when)]


from pydantic import BaseModel as _BaseModel, ConfigDict as _ConfigDict, Field as _PField
from typing import Literal as _Literal


class BlastRadius(_BaseModel):
    """Independent agent limits; ttl is an absolute UTC expiry."""
    model_config = _ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    max_clients: int | None = _PField(default=None, ge=0)
    max_value: float | None = _PField(default=None, ge=0)
    max_daily_actions: int | None = _PField(default=None, ge=0)
    per_minute: int | None = _PField(default=None, ge=0)
    per_hour: int | None = _PField(default=None, ge=0)
    channels: list[str] | None = None
    jurisdictions: list[str] | None = None
    execution_authority: _Literal["none", "propose", "execute"] = "none"
    ttl: datetime | None = None


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

    purpose: str | None = None
    use_cases: list[str] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    permitted_systems: list[str] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    blast_radius: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    field_allowlist: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    tool_allowlist: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    autonomy_level: int | None = Field(default=0, ge=0, le=6, nullable=True)
    review_date: datetime | None = Field(default=None, sa_column=_nullable_utc_column())
    deployment_hash: str | None = None
    environment: str | None = None
    principal: str | None = None


class Policy(SQLModel, table=True):
    """One effective-dated rule, owned by whoever owns the regime it cites.

    ``rule`` is an open JSON payload interpreted by the evaluator registered for
    ``rule_type``: the core ships primitives, a pack registers the rest. Policies
    are data so they can be edited, versioned and replayed — asking "what did
    this say on the day of the meeting?" is then just ``as_of``.
    """

    __tablename__ = "policies"
    __table_args__ = (
        Index("ix_policies_effective_window", "effective_from", "effective_to"),
    )

    id: int | None = Field(default=None, primary_key=True)
    regime: str = Field(index=True)
    rule_type: str = Field(index=True)
    #: Optional narrowing to one subject kind, e.g. a product type. None = all.
    applies_to: str | None = Field(default=None, index=True)
    rule: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    citation: str = ""
    effective_from: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    effective_to: datetime | None = Field(
        default=None, sa_column=_nullable_utc_column()
    )
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
    confirmed_at: datetime | None = Field(
        default=None, sa_column=_nullable_utc_column()
    )
    created_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    expires_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    decided_at: datetime | None = Field(default=None, sa_column=_nullable_utc_column())
    #: Deprecated: use requester_id and approver_id instead
    approver_name: str = "Unassigned (duty desk)"

    def is_live(self, at: datetime) -> bool:
        """Pending and not yet expired at ``at``."""
        expiry = as_utc(self.expires_at)
        return self.status == ApprovalStatus.PENDING and (expiry is None or expiry > at)


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
    __table_args__ = (
        UniqueConstraint("sequence", name="uq_evidence_sequence"),
        Index("ix_evidence_events_recorded_at", "recorded_at"),
    )

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
    approval_id: int | None = Field(
        default=None, index=True, foreign_key="approvals.id"
    )
    policy_id: int | None = Field(default=None, index=True, foreign_key="policies.id")
    successor_id: int | None = Field(
        default=None, index=True, foreign_key="policies.id"
    )
    run_id: str | None = Field(default=None, index=True, foreign_key="runs.id")
    continuation_hash: str | None = Field(
        default=None, index=True, foreign_key="continuations.content_hash"
    )


class KnowledgeItem(SQLModel, table=True):
    """A retrievable fragment carrying the attributes entitlement is decided on.

    ``classification`` and ``zones`` are not metadata for display — retrieval
    filters on them *before* scoring, so an item outside the caller's entitlement
    is never compared, never ranked, and cannot leak through a relevance score.

    A section ingested from a document also carries a locator: the document
    reference, page, section path, a document-global character range, the
    parser's block span and bounding box. ``source_label`` remains the
    readable citation and quotes the document-global range.
    """

    __tablename__ = "knowledge_items"

    id: int | None = Field(default=None, primary_key=True)
    title: str = ""
    body: str = ""
    kind: str = Field(default="DOCUMENT", index=True)
    classification: Classification = Field(default=Classification.C1, index=True)
    #: The rule that set ``classification`` when ingestion decided it.
    #: Empty when a caller set the classification directly.
    classification_rule: str = ""
    #: Zones permitted to hold and serve this item.
    zones: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    source_label: str = ""
    #: Structural locator. A missing span stays empty; it is not guessed.
    #: ``char_start`` and ``char_end`` are a document-global range into the
    #: source text (the file for markdown and HTML, the extracted paragraph
    #: text for DOCX). ``block_char_start`` and ``block_char_end`` are the
    #: parser's span for the opening block. ``bbox`` uses the parser's keys
    #: ``l``, ``t``, ``r`` and ``b``.
    document_ref: str = ""
    page: int | None = Field(default=None)
    section_path: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    char_start: int | None = Field(default=None)
    char_end: int | None = Field(default=None)
    block_char_start: int | None = Field(default=None)
    block_char_end: int | None = Field(default=None)
    bbox: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
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

    external: bool | None = None
    client_data: bool | None = None
    assurance_level: int | None = None

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
    #: live, paused, recovered, or revoked. A later intervention appends evidence.
    #: This column is the state the next decision reads. It is not a decision.
    control_state: str = "live"
    #: Tools a restrict intervention added. The list only grows.
    tool_denylist: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    #: DENY, ESCALATE, or ALLOW when an override is in force. Empty when none is.
    override_outcome: str = ""


class Continuation(SQLModel, table=True):
    """Content-addressed store for a suspended run.

    Keyed by the hash of the payload. The run record holds only that hash.
    """

    __tablename__ = "continuations"

    content_hash: str = Field(primary_key=True)
    payload: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))


class UseCase(SQLModel, table=True):
    """One version of a registered AI use case.

    Versions are effective-dated. A registry change closes the current row and
    inserts the next version; the closed row is not rewritten. ``record`` is the
    entry. ``effective_to`` is not part of the hash, so closing the window does
    not change what a decision pinned.
    """

    __tablename__ = "use_cases"
    __table_args__ = (
        UniqueConstraint("usecase_id", "version", name="uq_usecase_version"),
        Index("ix_use_cases_effective_window", "effective_from", "effective_to"),
    )

    id: int | None = Field(default=None, primary_key=True)
    usecase_id: str = Field(index=True)
    deployment_mode: str | None = None
    version: int = 1
    record: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    content_hash: str = Field(default="")
    effective_from: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    effective_to: datetime | None = Field(
        default=None, sa_column=_nullable_utc_column()
    )

    def is_effective(self, at: datetime) -> bool:
        """Whether this version is in force at ``at`` (half-open interval)."""
        start = as_utc(self.effective_from)
        end = as_utc(self.effective_to)
        assert start is not None
        return start <= at and (end is None or end > at)


class RiskAppetite(SQLModel, table=True):
    """The AI risk appetite in force, as one versioned policy.

    A use case outside the appetite cannot be registered. Changing the appetite
    is a governed change; the closed row is not rewritten.
    """

    __tablename__ = "risk_appetites"
    __table_args__ = (
        Index("ix_risk_appetites_effective_window", "effective_from", "effective_to"),
    )

    id: int | None = Field(default=None, primary_key=True)
    version: int = 1
    record: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    content_hash: str = Field(default="")
    effective_from: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    effective_to: datetime | None = Field(
        default=None, sa_column=_nullable_utc_column()
    )

    def is_effective(self, at: datetime) -> bool:
        start = as_utc(self.effective_from)
        end = as_utc(self.effective_to)
        assert start is not None
        return start <= at and (end is None or end > at)


class ModelCard(SQLModel, table=True):
    """One version of a model card for an approved model.

    The card id is ``provider/model@version``. A new approval is a new version
    row; the closed row is not rewritten. ``effective_to`` is not part of the
    hash.
    """

    __tablename__ = "model_cards"
    __table_args__ = (
        UniqueConstraint("card_id", "version", name="uq_model_card_version"),
        Index("ix_model_cards_effective_window", "effective_from", "effective_to"),
    )

    id: int | None = Field(default=None, primary_key=True)
    card_id: str = Field(index=True)
    version: int = 1
    record: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    content_hash: str = Field(default="")
    effective_from: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    effective_to: datetime | None = Field(default=None, sa_column=_nullable_utc_column())

    def is_effective(self, at: datetime) -> bool:
        """Whether this version is in force at ``at`` (half-open interval)."""
        start = as_utc(self.effective_from)
        end = as_utc(self.effective_to)
        assert start is not None
        return start <= at and (end is None or end > at)


class RetentionPolicy(SQLModel, table=True):
    """How long one use case keeps content in the retention store.

    The period is keyed by the use case id the register names. ``retain_days``
    of None keeps the content until an explicit erasure. Retiring a use case
    writes this period and erases content that has already aged past it.
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


class RetainedReference(SQLModel, table=True):
    """One use case's hold on a retained body.

    The body is stored once per hash. Each use case that stores the same bytes
    adds its own reference, and a later store by that use case moves its
    ``retained_at`` forward. The body is erased by retention only when every
    reference has passed its use case's period.
    """

    __tablename__ = "retained_references"

    content_hash: str = Field(primary_key=True)
    use_case: str = Field(default="", primary_key=True)
    correlation_id: str = Field(default="", index=True)
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


class Interaction(SQLModel, table=True):
    """One client journey, from trigger through final decision.

    Runs and correlation ids attach to this row. ``reconstruct`` reads the
    journey by this id, not by a single request id.
    """

    __tablename__ = "interactions"

    id: str = Field(primary_key=True)
    client_id: str = Field(index=True)
    use_case: str = Field(default="", index=True)
    #: The stricter of the gate named at open and the gate on the register.
    approval_gate: str = ""
    opened_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


class InteractionLink(SQLModel, table=True):
    """A run or a correlation id that belongs to one interaction."""

    __tablename__ = "interaction_links"

    id: int | None = Field(default=None, primary_key=True)
    interaction_id: str = Field(index=True, foreign_key="interactions.id")
    correlation_id: str = Field(default="", index=True)
    run_id: str | None = Field(default=None, foreign_key="runs.id")


class HumanBoundary(SQLModel, table=True):
    """Where a human must intervene, recorded before a review can approve.

    Stored beside the use-case register. The register already names decision
    points as a list of strings; this row is who reviews, what they see, and
    what a rejection does.
    """

    __tablename__ = "human_boundaries"

    usecase_id: str = Field(primary_key=True)
    intervention: str = ""
    who: str = ""
    must_review: str = ""
    evidence_seen: str = ""
    on_reject: str = ""
    minimum_review_seconds: int = 0


class RmReview(SQLModel, table=True):
    """The recommendation an RM was shown, and the decision still open on it.

    The rendered text, the diff and the packet live in the retention store.
    This row holds their hashes and, for a two-signature gate, the marked
    state that is not yet a final decision.
    """

    __tablename__ = "rm_reviews"

    id: int | None = Field(default=None, primary_key=True)
    interaction_id: str = Field(index=True, foreign_key="interactions.id")
    status: str = Field(default="marked", index=True)
    rm: str = ""
    confirmer: str = ""
    outcome: str = ""
    packet_hash: str = ""
    rendered_hash: str = ""
    diff_hash: str = ""
    provenance: str = ""
    prior_provenance: str = ""
    requested_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    decided_at: datetime | None = Field(default=None, sa_column=_nullable_utc_column())


class RiskCriteria(SQLModel, table=True):
    """How a use case scores a model response for the review queue.

    Versions are effective-dated. A change closes the current row and inserts
    the next version. ``effective_to`` is not part of the hashed record.
    """

    __tablename__ = "risk_criteria"
    __table_args__ = (
        UniqueConstraint("usecase_id", "version", name="uq_risk_criteria_version"),
    )

    id: int | None = Field(default=None, primary_key=True)
    usecase_id: str = Field(index=True)
    version: int = 1
    record: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    content_hash: str = ""
    effective_from: datetime = Field(sa_column=_utc_column())
    effective_to: datetime | None = Field(default=None, sa_column=_nullable_utc_column())

    def is_effective(self, at: datetime) -> bool:
        start = as_utc(self.effective_from)
        end = as_utc(self.effective_to)
        return start is not None and start <= at and (end is None or end > at)


class ReviewQueueItem(SQLModel, table=True):
    """One place on the review queue.

    The evidence event is the record. This row is the queue a reviewer works.
    Closing it sets ``status`` and ``reviewed_at``. The reasons and the citation
    are not rewritten.
    """

    __tablename__ = "review_queue"

    id: int | None = Field(default=None, primary_key=True)
    interaction_id: str = Field(index=True, foreign_key="interactions.id")
    usecase_id: str = Field(default="", index=True)
    risk_tier: int | None = None
    risk: str = ""
    timing: str = ""
    reasons: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    status: str = Field(default="queued", index=True)
    citation: str = ""
    response_hash: str = ""
    #: Set once, when a review outcome is recorded. Empty while the item is queued.
    finding: str = ""
    queued_at: datetime = Field(sa_column=_utc_column())
    reviewed_at: datetime | None = Field(default=None, sa_column=_nullable_utc_column())


class ApprovedChannel(SQLModel, table=True):
    """A channel registered for client communication or for the public lane.

    The row is the channel in force. A change appends a ``POLICY_CHANGE``
    event and then updates the row. Evidence of the earlier registration stays
    on the chain.
    """

    __tablename__ = "approved_channels"

    channel: str = Field(primary_key=True)
    label: str = ""
    archiving: bool = False
    supervision: bool = False
    public: bool = False
    citation: str = ""
    registered_by: str = ""


class CompensatingAction(SQLModel, table=True):
    """One registered step a recover intervention can run for a live run.

    Rows are append-only. Recover reads them in reverse order and records that
    order on the intervention event. It does not delete the rows.
    """

    __tablename__ = "compensating_actions"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, foreign_key="runs.id")
    name: str = ""
    citation: str = ""
    recorded_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


class EntityEdge(SQLModel, table=True):
    """A relationship in the deterministic mirror."""

    __tablename__ = "entity_edges"

    id: int | None = Field(default=None, primary_key=True)
    from_entity_id: int = Field(index=True, foreign_key="entities.id")
    to_entity_id: int = Field(index=True, foreign_key="entities.id")
    type: str = Field(index=True)
    created_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


def usecase_content(row: UseCase) -> dict[str, Any]:
    """The use-case version as evaluated, without the window's closing instant."""
    start = as_utc(row.effective_from)
    return {
        "usecase_id": row.usecase_id,
        "version": row.version,
        "record": row.record,
        "effective_from": start.isoformat() if start else None,
    }


def usecase_content_hash(row: UseCase) -> str:
    """Hash of :func:`usecase_content`."""
    return canonical_hash(usecase_content(row))


def appetite_content(row: RiskAppetite) -> dict[str, Any]:
    """The appetite version as evaluated, without the window's closing instant."""
    start = as_utc(row.effective_from)
    return {
        "version": row.version,
        "record": row.record,
        "effective_from": start.isoformat() if start else None,
    }


def appetite_content_hash(row: RiskAppetite) -> str:
    """Hash of :func:`appetite_content`."""
    return canonical_hash(appetite_content(row))


def model_card_content(row: ModelCard) -> dict[str, Any]:
    """The model-card version as evaluated, without the window's closing instant."""
    start = as_utc(row.effective_from)
    return {
        "card_id": row.card_id,
        "version": row.version,
        "record": row.record,
        "effective_from": start.isoformat() if start else None,
    }


def model_card_content_hash(row: ModelCard) -> str:
    """Hash of :func:`model_card_content`."""
    return canonical_hash(model_card_content(row))


def _committed(row: Any, name: str) -> Any:
    """The last flushed value of ``name`` when this update changes it."""
    state = sa_inspect(row)
    if name in state.committed_state:
        return state.committed_state[name]
    return getattr(row, name)


@event.listens_for(Policy, "before_insert")
def _stamp_policy_content_hash(_mapper: Any, _connection: Any, target: Policy) -> None:
    """The stamp is the row's content. A caller-supplied hash is not kept."""
    target.content_hash = policy_content_hash(target)


@event.listens_for(Policy, "before_update")
def _reject_in_place_policy_edit(
    _mapper: Any, _connection: Any, target: Policy
) -> None:
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


def _refuse_closed_window(row: Any) -> None:
    """A window that is already closed is not moved."""
    state = sa_inspect(row)
    if "effective_to" in state.committed_state:
        previous_end = state.committed_state["effective_to"]
        if previous_end is not None:
            raise KognitaError(
                "effective_to is already set; registry changes are new versions"
            )


@event.listens_for(UseCase, "before_insert")
def _stamp_usecase_content_hash(
    _mapper: Any, _connection: Any, target: UseCase
) -> None:
    """The stamp is the row's content. A caller-supplied hash is not kept."""
    target.content_hash = usecase_content_hash(target)


@event.listens_for(UseCase, "before_update")
def _reject_in_place_usecase_edit(
    _mapper: Any, _connection: Any, target: UseCase
) -> None:
    """Closing the window is the only rewrite an effective use case accepts."""
    _refuse_closed_window(target)
    fresh = usecase_content_hash(target)
    stored = _committed(target, "content_hash") or ""
    if fresh == stored:
        return
    start = as_utc(_committed(target, "effective_from"))
    if start is not None and start <= utcnow():
        raise KognitaError(
            "in-place edit of an effective use case; registry changes are new versions"
        )
    target.content_hash = fresh


@event.listens_for(RiskAppetite, "before_insert")
def _stamp_appetite_content_hash(
    _mapper: Any, _connection: Any, target: RiskAppetite
) -> None:
    target.content_hash = appetite_content_hash(target)


@event.listens_for(RiskAppetite, "before_update")
def _reject_in_place_appetite_edit(
    _mapper: Any, _connection: Any, target: RiskAppetite
) -> None:
    _refuse_closed_window(target)
    fresh = appetite_content_hash(target)
    stored = _committed(target, "content_hash") or ""
    if fresh == stored:
        return
    start = as_utc(_committed(target, "effective_from"))
    if start is not None and start <= utcnow():
        raise KognitaError(
            "in-place edit of an effective risk appetite; changes are new versions"
        )
    target.content_hash = fresh


@event.listens_for(ModelCard, "before_insert")
def _stamp_model_card_content_hash(
    _mapper: Any, _connection: Any, target: ModelCard
) -> None:
    target.content_hash = model_card_content_hash(target)


@event.listens_for(ModelCard, "before_update")
def _reject_in_place_model_card_edit(
    _mapper: Any, _connection: Any, target: ModelCard
) -> None:
    _refuse_closed_window(target)
    fresh = model_card_content_hash(target)
    stored = _committed(target, "content_hash") or ""
    if fresh == stored:
        return
    start = as_utc(_committed(target, "effective_from"))
    if start is not None and start <= utcnow():
        raise KognitaError(
            "in-place edit of an effective model card; changes are new versions"
        )
    target.content_hash = fresh


def _effective(start: datetime | None, end: datetime | None, at: datetime) -> bool:
    """Half-open window. ``end`` equal to ``at`` is already closed."""
    opened = as_utc(start)
    closed = as_utc(end)
    return opened is not None and opened <= at and (closed is None or closed > at)


class Revocation(SQLModel, table=True):
    """One dimension stopped until an accountable owner reinstates it.

    Closing ``effective_to`` is the reinstatement. A provider stays open until
    every accountable owner of every use case that lists it has approved.
    The row is not rewritten into a permission. A later revocation is a new row.
    """

    __tablename__ = "revocations"
    __table_args__ = (
        Index("ix_revocations_effective_window", "effective_from", "effective_to"),
    )

    id: int | None = Field(default=None, primary_key=True)
    dimension: str = Field(index=True)
    value: str = Field(index=True)
    citation: str = ""
    actor_id: str = ""
    accountable_owners: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    #: Provider only. Empty means the provider is not limited to a use-case list.
    limit_usecases: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    #: Provider only. Empty means no classification ceiling. A call at or above
    #: this classification is outside the limit.
    below_classification: str = ""
    effective_from: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    effective_to: datetime | None = Field(
        default=None, sa_column=_nullable_utc_column()
    )

    def is_effective(self, at: datetime) -> bool:
        return _effective(self.effective_from, self.effective_to, at)


class BreakerThreshold(SQLModel, table=True):
    """How many flags, in what window, trip a breaker for one dimension value.

    The roadmap does not set the number. Nothing trips until a threshold is
    recorded. Closing the window retires it; the replacement is a new row.
    """

    __tablename__ = "breaker_thresholds"
    __table_args__ = (
        Index(
            "ix_breaker_thresholds_effective_window",
            "effective_from",
            "effective_to",
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    dimension: str = Field(index=True)
    value: str = Field(index=True)
    threshold: int = 1
    window_seconds: int = 1
    citation: str = ""
    actor_id: str = ""
    accountable_owner: str = ""
    effective_from: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    effective_to: datetime | None = Field(
        default=None, sa_column=_nullable_utc_column()
    )

    def is_effective(self, at: datetime) -> bool:
        return _effective(self.effective_from, self.effective_to, at)


class DrillDue(SQLModel, table=True):
    """When a high-materiality use case must next run a kill-switch drill.

    The evidence of the drill is on the chain. This row is the due date the
    next decision reads. A new due date closes this window and inserts the next.
    """

    __tablename__ = "drill_dues"
    __table_args__ = (
        Index("ix_drill_dues_effective_window", "effective_from", "effective_to"),
    )

    id: int | None = Field(default=None, primary_key=True)
    usecase_id: str = Field(index=True)
    next_due: datetime = Field(sa_column=_utc_column())
    effective_from: datetime = Field(default_factory=utcnow, sa_column=_utc_column())
    effective_to: datetime | None = Field(
        default=None, sa_column=_nullable_utc_column()
    )

    def is_effective(self, at: datetime) -> bool:
        return _effective(self.effective_from, self.effective_to, at)


class ReinstatementApproval(SQLModel, table=True):
    """One owner's cited approval toward closing a revocation. Append-only.

    A provider revocation stays open until every required owner has a row.
    Any other dimension closes on the first approval, which is recorded here too.
    """

    __tablename__ = "reinstatement_approvals"

    id: int | None = Field(default=None, primary_key=True)
    revocation_id: int = Field(foreign_key="revocations.id", index=True)
    actor_id: str = ""
    citation: str = ""
    recorded_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())


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
    "UseCase",
    "Interaction",
    "InteractionLink",
    "HumanBoundary",
    "RmReview",
    "RiskCriteria",
    "ReviewQueueItem",
    "ApprovedChannel",
    "CompensatingAction",
    "RiskAppetite",
    "ModelCard",
    "Revocation",
    "ReinstatementApproval",
    "BreakerThreshold",
    "DrillDue",
    "RetentionPolicy",
    "RetainedContent",
    "policy_content",
    "policy_content_hash",
    "usecase_content",
    "usecase_content_hash",
    "appetite_content",
    "appetite_content_hash",
    "model_card_content",
    "model_card_content_hash",
    "utcnow",
    "as_utc",
]


class _WorkloadCredential(SQLModel, table=True):
    """Only a digest is persisted; each row belongs to exactly one agent."""
    __tablename__ = "_workload_credentials"
    digest: str = Field(primary_key=True)
    agent_name: str = Field(index=True)
    revoked: bool = False


class _Authority(SQLModel, table=True):
    __tablename__ = "authority"
    __table_args__ = (Index("ix_authority_grantee_kind_ended", "grantee", "kind", "ended"),)
    id: str = Field(primary_key=True)
    grantee: str = Field(index=True)
    kind: str = Field(index=True)
    body: dict = Field(default_factory=dict, sa_column=Column(JSON))
    ended: bool = Field(default=False, index=True)


class _ApprovalContinuation(SQLModel, table=True):
    """Exact-request approval holds with a transactional one-use claim."""
    __tablename__ = "_approval_continuations"
    __table_args__ = (Index("ix_approval_hold_expiry", "agent_name", "consumed", "expires_at"),)
    envelope_hash: str = Field(primary_key=True)
    approval_id: int
    consumed: bool = False
    agent_name: str | None = None
    expires_at: datetime | None = Field(default=None, sa_column=_nullable_utc_column())


class _BlastUsage(SQLModel, table=True):
    """Admission reservations; release moves the reservation to execution time."""
    __tablename__ = "_blast_usage"
    __table_args__ = (Index("ix_blast_usage_agent_time", "agent_name", "at"),)
    request_id: str = Field(primary_key=True)
    agent_name: str
    at: datetime = Field(sa_column=_utc_column())
    clients: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    value: float = 0


class _AnomalyDay(SQLModel, table=True):
    __tablename__ = "_anomaly_days"
    __table_args__ = (Index("ix_anomaly_agent_day", "agent_name", "day", unique=True),)
    id: int | None = Field(default=None, primary_key=True)
    agent_name: str
    day: str
    body: dict = Field(default_factory=dict, sa_column=Column(JSON))


class _RunScope(SQLModel, table=True):
    """Trusted host bindings; never populated from message content."""
    __tablename__ = "_run_scopes"
    id: str = Field(primary_key=True, foreign_key="runs.id")
    body: dict = Field(default_factory=dict, sa_column=Column(JSON))


class _AgentTaint(SQLModel, table=True):
    """Sticky provenance for a workload/delegation across run rotations."""
    __tablename__ = "_agent_taint"
    id: str = Field(primary_key=True)
    external: bool = False
    client_data: bool = False

class _StepUpBinding(SQLModel, table=True):
    __tablename__ = "_step_up_bindings"
    event_hash: str = Field(primary_key=True)


class _AnomalyWindow(SQLModel, table=True):
    __tablename__ = "_anomaly_windows"
    agent_name: str = Field(primary_key=True)
    body: list = Field(default_factory=list, sa_column=Column(JSON))
