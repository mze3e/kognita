"""Governed working, relationship and institutional memory.

Working memory lasts only as long as its run. An inference becomes a
relationship record only after a different human verifies the proposal.
Institutional standards are proposed with evidence and take effect only when
the use case Knowledge Lead approves them. That approval issues a new
deployment version. A standard may cite evidence. It may not carry client
content. Outputs record standard_version for the version actually in force.

The caller owns the transaction and supplies authenticated identities.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlmodel import Session, select

from kognita.approvals import ApprovalError, _identity, _proposal_hash, _propose
from kognita.canonical import canonical_json, hash_text
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.models import (
    EvidenceEvent,
    Interaction,
    InteractionLink,
    Proposal,
    RunRecord,
    as_utc,
    utcnow,
)
from kognita.retention import RetentionStore
from kognita.vocabulary import ActorType, EventType


class MemoryScope(str, Enum):
    """Where a memory write is allowed to live."""

    WORKING = "WORKING"
    RELATIONSHIP = "RELATIONSHIP"
    INSTITUTIONAL = "INSTITUTIONAL"


_VERIFIED = "verified fact"
_INFERRED = "inferred preference"
_HYPOTHESIS = "AI hypothesis"
_TYPES = frozenset({_VERIFIED, _INFERRED, _HYPOTHESIS})
_INFERENCES = frozenset({_INFERRED, _HYPOTHESIS})
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LIVE = frozenset({"live", "paused"})
_MEMORY_BODY = frozenset({
    "client_id", "content", "type", "retention_seconds", "expiry",
    "correction_path", "scope", "run_id",
})
_STANDARD_BODY = frozenset({
    "standard_id", "version", "effective", "superseded", "evidence_refs",
    "content", "expiry", "usecase_id",
})


def _code(exc: ValueError) -> str:
    text = str(exc)
    return text if text.startswith("MEMORY_") else "MEMORY_INVALID"


def _now(at: datetime | None) -> datetime:
    if at is None:
        at = utcnow()
    if not isinstance(at, datetime):
        raise ValueError("MEMORY_INVALID")
    stamped = as_utc(at)
    if stamped is None or at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("MEMORY_INVALID")
    return stamped.astimezone(timezone.utc)


def _text(value: object, *, token: bool = False, limit: int = 8000) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("MEMORY_INVALID")
    if any(unicodedata.category(ch) == "Cf" or ord(ch) < 32 for ch in value):
        raise ValueError("MEMORY_INVALID")
    if len(value) > limit:
        raise ValueError("MEMORY_INVALID")
    if token and _TOKEN.fullmatch(value) is None:
        raise ValueError("MEMORY_INVALID")
    return value


def _aware(value: object, code: str = "MEMORY_INVALID") -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(code)
    stamped = as_utc(value)
    if stamped is None:
        raise ValueError(code)
    return stamped.astimezone(timezone.utc)


def _period(value: object) -> timedelta:
    if not isinstance(value, timedelta) or value <= timedelta(0):
        raise ValueError("MEMORY_INVALID")
    seconds = value.total_seconds()
    if seconds != int(seconds) or int(seconds) > 86400 * 3650:
        raise ValueError("MEMORY_INVALID")
    return timedelta(seconds=int(seconds))


def _norm(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return "".join(ch for ch in folded if not ch.isspace() and unicodedata.category(ch) != "Cf")


def _same_person(left: str, right: str) -> bool:
    return _norm(left) == _norm(right)


@dataclass(frozen=True)
class MemoryRecord:
    """One client-scoped memory snapshot. Content is retained, not chained."""

    client_id: str
    content: str
    type: str
    retention_period: timedelta
    expiry: datetime
    correction_path: str
    scope: MemoryScope
    run_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_id", _text(self.client_id, token=True))
        object.__setattr__(self, "content", _text(self.content))
        if not isinstance(self.type, str) or self.type not in _TYPES:
            raise ValueError("MEMORY_INVALID")
        object.__setattr__(self, "retention_period", _period(self.retention_period))
        object.__setattr__(self, "expiry", _aware(self.expiry))
        object.__setattr__(self, "correction_path", _text(self.correction_path, limit=500))
        try:
            scope = MemoryScope(self.scope)
        except ValueError:
            raise ValueError("MEMORY_INVALID") from None
        object.__setattr__(self, "scope", scope)
        run_id = self.run_id
        if scope is MemoryScope.WORKING:
            object.__setattr__(self, "run_id", _text(run_id, token=True))
        elif run_id not in ("", None):
            raise ValueError("MEMORY_SCOPE")
        else:
            object.__setattr__(self, "run_id", "")
        if scope is MemoryScope.INSTITUTIONAL:
            raise ValueError("MEMORY_DIRECT_WRITE")


@dataclass(frozen=True)
class Standard:
    """One effective-dated institutional standard. Versions are not edited."""

    standard_id: str
    version: str
    effective: datetime
    superseded: datetime | None
    evidence_refs: tuple[str, ...]
    content: str
    expiry: datetime | None = None
    usecase_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "standard_id", _text(self.standard_id, token=True))
        object.__setattr__(self, "version", _text(self.version, token=True))
        object.__setattr__(self, "effective", _aware(self.effective, "MEMORY_STANDARD"))
        object.__setattr__(self, "content", _text(self.content))
        object.__setattr__(self, "usecase_id", _text(self.usecase_id, token=True))
        if isinstance(self.evidence_refs, str) or not isinstance(self.evidence_refs, tuple):
            raise ValueError("MEMORY_EVIDENCE")
        refs = tuple(self.evidence_refs)
        if not refs or len(refs) != len(set(refs)):
            raise ValueError("MEMORY_EVIDENCE")
        object.__setattr__(self, "evidence_refs", tuple(_text(ref, token=True) for ref in refs))
        superseded = self.superseded
        if superseded is not None:
            superseded = _aware(superseded, "MEMORY_STANDARD")
        expiry = self.expiry
        if expiry is not None:
            expiry = _aware(expiry, "MEMORY_STANDARD")
            if expiry <= self.effective:
                raise ValueError("MEMORY_STANDARD")
        object.__setattr__(self, "superseded", superseded)
        object.__setattr__(self, "expiry", expiry)


def _iso(at: datetime | None) -> str | None:
    return None if at is None else at.astimezone(timezone.utc).isoformat()


def _parse_time(value: object, code: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(code)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(code) from None
    return _aware(parsed, code)


def _memory_body(record: MemoryRecord) -> dict:
    return {
        "client_id": record.client_id,
        "content": record.content,
        "type": record.type,
        "retention_seconds": int(record.retention_period.total_seconds()),
        "expiry": _iso(record.expiry),
        "correction_path": record.correction_path,
        "scope": record.scope.value,
        "run_id": record.run_id,
    }


def _record_from(body: dict) -> MemoryRecord:
    seconds = body.get("retention_seconds") if isinstance(body, dict) else None
    if not isinstance(body, dict) or set(body) != _MEMORY_BODY or isinstance(seconds, bool) or type(seconds) is not int:
        raise ValueError("MEMORY_INVALID")
    return MemoryRecord(
        client_id=body["client_id"],
        content=body["content"],
        type=body["type"],
        retention_period=timedelta(seconds=seconds),
        expiry=_parse_time(body["expiry"], "MEMORY_INVALID"),
        correction_path=body["correction_path"],
        scope=body["scope"],
        run_id=body["run_id"],
    )


def _standard_body(standard: Standard) -> dict:
    return {
        "standard_id": standard.standard_id,
        "version": standard.version,
        "effective": _iso(standard.effective),
        "superseded": _iso(standard.superseded),
        "evidence_refs": list(standard.evidence_refs),
        "content": standard.content,
        "expiry": _iso(standard.expiry),
        "usecase_id": standard.usecase_id,
    }


def _standard_from(body: dict) -> Standard:
    if not isinstance(body, dict) or set(body) != _STANDARD_BODY:
        raise ValueError("MEMORY_STANDARD")
    refs = body["evidence_refs"]
    if not isinstance(refs, list):
        raise ValueError("MEMORY_EVIDENCE")
    return Standard(
        standard_id=body["standard_id"],
        version=body["version"],
        effective=_parse_time(body["effective"], "MEMORY_STANDARD"),
        superseded=_parse_time(body["superseded"], "MEMORY_STANDARD"),
        evidence_refs=tuple(refs),
        content=body["content"],
        expiry=_parse_time(body["expiry"], "MEMORY_STANDARD"),
        usecase_id=body["usecase_id"],
    )


def _window(record: MemoryRecord, now: datetime) -> None:
    if record.expiry <= now:
        raise ValueError("MEMORY_EXPIRED")
    if record.expiry > now + record.retention_period:
        raise ValueError("MEMORY_INVALID")


def _deny(session: Session, evidence: EvidenceWriter, correlation_id: str, code: str, **payload) -> None:
    body = {"action": "denied", "allowed": False, "reason": code, "standard_version": ""}
    body.update(payload)
    body["standard_version"] = str(body.get("standard_version") or "")
    evidence.emit(
        session, correlation_id=correlation_id, event_type=EventType.MEMORY_WRITE,
        actor_type=ActorType.SYSTEM, actor_id="memory", payload=body,
    )
    raise ValueError(code)


def _person(value: object, code: str) -> str:
    if not isinstance(value, str):
        raise ValueError(code)
    try:
        name = _identity(value)
    except ApprovalError:
        raise ValueError(code) from None
    if any(unicodedata.category(ch) == "Cf" for ch in name):
        raise ValueError(code)
    return name


def _read(session: Session, digest: object, code: str) -> dict:
    if not isinstance(digest, str) or not digest:
        raise ValueError(code)
    raw = RetentionStore().read(session, digest)
    if raw is None or hash_text(raw) != digest:
        raise ValueError(code)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(code) from None
    if not isinstance(body, dict):
        raise ValueError(code)
    return body


def _events(session: Session) -> list[EvidenceEvent]:
    return session.exec(
        select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.MEMORY_WRITE)
        .order_by(EvidenceEvent.sequence)
    ).all()


def _client_pairs(session: Session) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for row in session.exec(select(Interaction)).all():
        if isinstance(row.client_id, str) and row.client_id.strip():
            pairs.append((row.client_id.strip(), ""))
    from kognita.claims import list_claims
    try:
        claims = list_claims(session)
    except Exception:
        raise ValueError("MEMORY_STANDARD") from None
    for claim in claims:
        pairs.append((claim.client_id, claim.claim))
    for proposal in session.exec(select(Proposal).where(Proposal.kind == "relationship_memory")).all():
        body = proposal.change.get("body") if isinstance(proposal.change, dict) else None
        if not isinstance(body, dict):
            raise ValueError("MEMORY_STANDARD")
        client_id = body.get("client_id")
        content = body.get("content")
        if not isinstance(client_id, str) or not isinstance(content, str):
            raise ValueError("MEMORY_STANDARD")
        pairs.append((client_id, content))
    for event in _events(session):
        payload = event.payload or {}
        if payload.get("allowed") is not True or payload.get("scope") == MemoryScope.INSTITUTIONAL.value:
            continue
        if payload.get("action") not in ("write", "verify"):
            raise ValueError("MEMORY_INVALID")
        body = _read(session, payload.get("content_hash"), "MEMORY_INVALID")
        record = _record_from(body)
        pairs.append((record.client_id, record.content))
    return pairs


def _contains_client(text: str, ids: set[str], contents: set[str]) -> bool:
    folded = _norm(text)
    for client_id in ids:
        token = _norm(client_id)
        if token and (folded == token or (len(token) >= 3 and token in folded)):
            return True
    for content in contents:
        token = _norm(content)
        if not token:
            continue
        if folded == token or (len(token) >= 12 and token in folded):
            return True
    return False


def _reject_foreign(session: Session, record: MemoryRecord) -> None:
    ids: set[str] = set()
    contents: set[str] = set()
    for client_id, content in _client_pairs(session):
        if client_id != record.client_id:
            ids.add(client_id)
            if content:
                contents.add(content)
    fields = (record.client_id, record.content, record.correction_path, record.run_id)
    if any(_contains_client(field, ids, contents) for field in fields):
        raise ValueError("MEMORY_SCOPE")


def _reject_client_standard(session: Session, standard: Standard) -> None:
    ids: set[str] = set()
    contents: set[str] = set()
    for client_id, content in _client_pairs(session):
        ids.add(client_id)
        if content:
            contents.add(content)
    fields = [standard.standard_id, standard.version, standard.content, standard.usecase_id, *standard.evidence_refs]
    if any(_contains_client(field, ids, contents) for field in fields):
        raise ValueError("MEMORY_CLIENT_CONTENT")


def _in_force(standard: Standard, now: datetime) -> bool:
    if standard.effective > now:
        return False
    if standard.superseded is not None and now >= standard.superseded:
        return False
    if standard.expiry is not None and now >= standard.expiry:
        return False
    return True


class _Catalog:
    def __init__(self) -> None:
        self.folded: dict[tuple[str, str, str], Standard] = {}
        self.heads: dict[tuple[str, str], Standard] = {}

    def versions(self, usecase_id: str, standard_id: str) -> list[Standard]:
        return [
            standard for (usecase, ident, _version), standard in self.folded.items()
            if usecase == usecase_id and ident == standard_id
        ]

    def in_force(self, usecase_id: str, now: datetime, standard_id: str | None = None) -> list[Standard]:
        found = []
        for (usecase, ident, _version), standard in self.folded.items():
            if usecase == usecase_id or not usecase_id:
                if standard_id is None or ident == standard_id:
                    if _in_force(standard, now):
                        found.append(standard)
        return found


def _catalog(session: Session) -> _Catalog:
    catalog = _Catalog()
    order: dict[tuple[str, str], list[str]] = {}
    for event in _events(session):
        payload = event.payload or {}
        if payload.get("scope") != MemoryScope.INSTITUTIONAL.value or payload.get("action") != "approve":
            continue
        if payload.get("allowed") is not True:
            continue
        body = _read(session, payload.get("content_hash"), "MEMORY_STANDARD")
        standard = _standard_from(body)
        if standard.superseded is not None:
            raise ValueError("MEMORY_STANDARD")
        if payload.get("standard_id") != standard.standard_id or payload.get("standard_version") != standard.version:
            raise ValueError("MEMORY_STANDARD")
        if payload.get("usecase_id") != standard.usecase_id:
            raise ValueError("MEMORY_STANDARD")
        effective = _parse_time(payload.get("effective"), "MEMORY_STANDARD")
        if effective != standard.effective:
            raise ValueError("MEMORY_STANDARD")
        key = (standard.usecase_id, standard.standard_id, standard.version)
        if key in catalog.folded:
            raise ValueError("MEMORY_VERSION")
        supersedes = payload.get("supersedes") or ""
        if not isinstance(supersedes, str):
            raise ValueError("MEMORY_STANDARD")
        if supersedes:
            previous = catalog.folded.get((standard.usecase_id, standard.standard_id, supersedes))
            if previous is None:
                raise ValueError("MEMORY_STANDARD")
            catalog.folded[(standard.usecase_id, standard.standard_id, supersedes)] = Standard(
                standard_id=previous.standard_id,
                version=previous.version,
                effective=previous.effective,
                superseded=standard.effective,
                evidence_refs=previous.evidence_refs,
                content=previous.content,
                expiry=previous.expiry,
                usecase_id=previous.usecase_id,
            )
        catalog.folded[key] = standard
        order.setdefault((standard.usecase_id, standard.standard_id), []).append(standard.version)
        catalog.heads[(standard.usecase_id, standard.standard_id)] = standard
    return catalog


def _version_field(standards: list[Standard]) -> str:
    if not standards:
        return ""
    if len(standards) == 1:
        return standards[0].version
    return canonical_json({item.standard_id: item.version for item in standards})


def _output_version(session: Session, usecase_id: str, cited: str, now: datetime) -> str:
    if usecase_id:
        _text(usecase_id, token=True)
    if not isinstance(cited, str):
        raise ValueError("MEMORY_STANDARD")
    catalog = _catalog(session)
    approved = [item for (usecase, _ident, _version), item in catalog.folded.items() if usecase == usecase_id] if usecase_id else []
    current = [item for item in approved if _in_force(item, now)]
    if approved and not current:
        raise ValueError("MEMORY_STANDARD")
    by_id: dict[str, list[Standard]] = {}
    for item in current:
        by_id.setdefault(item.standard_id, []).append(item)
    if any(len(group) != 1 for group in by_id.values()):
        raise ValueError("MEMORY_STANDARD")
    field = _version_field(current)
    if cited and cited != field:
        raise ValueError("MEMORY_STANDARD")
    if not usecase_id and cited:
        raise ValueError("MEMORY_STANDARD")
    return field


def _run_open(session: Session, record: MemoryRecord, run_id: str, now: datetime) -> None:
    if record.scope is not MemoryScope.WORKING or record.run_id != run_id:
        raise ValueError("MEMORY_SCOPE")
    run = session.get(RunRecord, run_id)
    if run is None or run.control_state not in _LIVE:
        raise ValueError("MEMORY_WORKING_RUN" if run is None else "MEMORY_EXPIRED")
    started = as_utc(run.started_at)
    if started is None:
        raise ValueError("MEMORY_WORKING_RUN")
    if run.wall_clock_seconds is not None:
        span = run.wall_clock_seconds
        if isinstance(span, bool) or not isinstance(span, (int, float)) or span < 0:
            raise ValueError("MEMORY_WORKING_RUN")
        end = started + timedelta(seconds=float(span))
        if now >= end or record.expiry > end:
            raise ValueError("MEMORY_EXPIRED")
    links = session.exec(select(InteractionLink).where(InteractionLink.run_id == run_id)).all()
    clients = []
    for link in links:
        interaction = session.get(Interaction, link.interaction_id)
        if interaction is None or not isinstance(interaction.client_id, str) or not interaction.client_id.strip():
            raise ValueError("MEMORY_SCOPE")
        clients.append(interaction.client_id.strip())
    if len(set(clients)) != 1 or clients[0] != record.client_id:
        raise ValueError("MEMORY_SCOPE")


def _retain(session: Session, body: dict, kind: str, correlation_id: str) -> str:
    return RetentionStore().retain_value(session, body, kind=kind, correlation_id=correlation_id)


def _emit(session, evidence, correlation_id, actor_type, actor_id, payload) -> None:
    evidence.emit(
        session, correlation_id=correlation_id, event_type=EventType.MEMORY_WRITE,
        actor_type=actor_type, actor_id=actor_id,
        payload={"allowed": True, "standard_version": "", **payload},
    )


def _fresh(session: Session, proposal: Proposal, kind: str) -> tuple[Proposal, str]:
    if not isinstance(proposal, Proposal) or proposal.id is None:
        raise ValueError("MEMORY_BINDING")
    row = session.get(Proposal, proposal.id)
    if row is None or row.kind != kind:
        raise ValueError("MEMORY_BINDING")
    session.refresh(row)
    from kognita.models import _ProposalBinding
    binding = session.get(_ProposalBinding, row.id)
    digest = _proposal_hash(row)
    if binding is None or binding.applied or binding.diff_hash != digest:
        raise ValueError("MEMORY_BINDING")
    return row, digest


def _claim(session: Session, proposal_id: int, digest: str) -> None:
    from sqlalchemy import update
    from kognita.models import _ProposalBinding
    claimed = session.execute(update(_ProposalBinding).where(
        _ProposalBinding.proposal_id == proposal_id,
        _ProposalBinding.diff_hash == digest,
        _ProposalBinding.applied == False,  # noqa: E712
    ).values(applied=True).execution_options(synchronize_session=False))
    if claimed.rowcount != 1:
        raise ValueError("MEMORY_BINDING")


def _lead(session: Session, usecase_id: str, approver: str, now: datetime) -> None:
    from kognita.usecases import load_register
    entry = load_register(session, as_of=now).get(usecase_id)
    if entry is None or not entry.knowledge_lead.strip() or not _same_person(approver, entry.knowledge_lead):
        raise ValueError("MEMORY_KNOWLEDGE_LEAD")


def _deployment_version(session, agent_name: str, usecase_id: str, evidence, approver: str, standard_id: str, standard_version: str) -> str:
    from kognita.registry import _deployment_hash, get
    agent = get(session, agent_name)
    if agent is None or not isinstance(agent.use_cases, list) or usecase_id not in agent.use_cases:
        raise ValueError("MEMORY_VERSION")
    current = (agent.tool_allowlist or {}).get("_deployment")
    if not isinstance(current, dict):
        raise ValueError("MEMORY_VERSION")
    try:
        if _deployment_hash(agent, current) != agent.deployment_hash:
            raise ValueError("MEMORY_VERSION")
    except ValueError as exc:
        if _code(exc) == "MEMORY_VERSION":
            raise
        raise ValueError("MEMORY_VERSION") from None
    catalog = _catalog(session)
    items = []
    for usecase in agent.use_cases:
        for (owner, _ident), standard in sorted(catalog.heads.items()):
            if owner == usecase:
                items.append({
                    "standard_id": standard.standard_id,
                    "version": standard.version,
                    "effective": _iso(standard.effective),
                    "usecase_id": usecase,
                })
    behaviour = canonical_json(items)
    ids: set[str] = set()
    contents: set[str] = set()
    for client_id, content in _client_pairs(session):
        ids.add(client_id)
        if content:
            contents.add(content)
    if _contains_client(behaviour, ids, contents):
        raise ValueError("MEMORY_CLIENT_CONTENT")
    if not any(item["standard_id"] == standard_id and item["version"] == standard_version and item["usecase_id"] == usecase_id for item in items):
        raise ValueError("MEMORY_VERSION")
    deployment = dict(current)
    deployment["memory_behaviour"] = behaviour
    deployment = json.loads(canonical_json(deployment))
    allow = {**(agent.tool_allowlist or {}), "_deployment": deployment}
    agent.tool_allowlist = allow
    digest = _deployment_hash(agent, deployment)
    previous = agent.deployment_hash
    agent.deployment_hash = digest
    session.add(agent)
    session.flush()
    evidence.emit(
        session, correlation_id=f"deployment:{agent.name}", event_type=EventType.POLICY_CHANGE,
        actor_type=ActorType.HUMAN, actor_id=approver,
        payload={
            "action": "DEPLOYMENT_APPROVED", "agent": agent.name, "deployment_hash": digest,
            "previous_deployment_hash": previous, "standard_version": standard_version,
            "memory_behaviour": behaviour, "citation": "S1-DEPLOYMENT",
            "_identity": {**agent.model_dump(mode="json"), "deployment_hash": digest},
        },
    )
    return digest


def write_memory(
    session: Session,
    record: MemoryRecord,
    *,
    evidence: EvidenceWriter,
    correlation_id: str,
    run_id: str = "",
    usecase_id: str = "",
    standard_version: str = "",
    now: datetime | None = None,
) -> MemoryRecord:
    """Write working memory, or a verified relationship fact. Inferences are refused."""
    at = _now(now)
    verify_chain(session)
    try:
        if not isinstance(record, MemoryRecord):
            raise ValueError("MEMORY_INVALID")
        if record.scope is MemoryScope.WORKING:
            _run_open(session, record, run_id, at)
        elif record.scope is MemoryScope.RELATIONSHIP:
            if run_id:
                raise ValueError("MEMORY_SCOPE")
            if record.type != _VERIFIED:
                raise ValueError("MEMORY_DIRECT_WRITE")
        else:
            raise ValueError("MEMORY_DIRECT_WRITE")
        _window(record, at)
        _reject_foreign(session, record)
        version = _output_version(session, usecase_id, standard_version, at)
        body = _memory_body(record)
        digest = _retain(session, body, "memory", correlation_id)
        _emit(session, evidence, correlation_id, ActorType.SYSTEM, "memory", {
            "action": "write", "scope": record.scope.value, "client_id": record.client_id,
            "run_id": record.run_id, "type": record.type, "content_hash": digest,
            "standard_version": version, "usecase_id": usecase_id,
        })
        return _record_from(body)
    except ValueError as exc:
        scope = record.scope.value if isinstance(getattr(record, "scope", None), MemoryScope) else ""
        client = record.client_id if isinstance(getattr(record, "client_id", None), str) else ""
        _deny(session, evidence, correlation_id, _code(exc), scope=scope, client_id=client, usecase_id=usecase_id)


def propose_memory(
    session: Session,
    record: MemoryRecord,
    *,
    evidence: EvidenceWriter,
    correlation_id: str,
    proposer_id: str,
    usecase_id: str = "",
    now: datetime | None = None,
) -> Proposal:
    """Hold a relationship inference. It is not a record until verify_memory."""
    at = _now(now)
    verify_chain(session)
    proposer = _person(proposer_id, "MEMORY_INVALID")
    if not isinstance(record, MemoryRecord) or record.scope is not MemoryScope.RELATIONSHIP or record.type not in _INFERENCES:
        raise ValueError("MEMORY_PROPOSAL")
    _window(record, at)
    _reject_foreign(session, record)
    if usecase_id:
        _text(usecase_id, token=True)
    body = _memory_body(record)
    proposal = Proposal(
        kind="relationship_memory",
        before_snapshot={"body": None, "usecase_id": usecase_id},
        change={"body": body, "usecase_id": usecase_id},
        rationale=record.correction_path,
        subject_type="relationship_memory",
        subject_id=record.client_id,
        proposer_id=proposer,
    )
    try:
        return _propose(session, proposal, evidence=evidence, correlation_id=correlation_id)
    except ApprovalError:
        raise ValueError("MEMORY_BINDING") from None


def verify_memory(
    session: Session,
    proposal: Proposal,
    *,
    evidence: EvidenceWriter,
    correlation_id: str,
    verifier_id: str,
    actor_type: ActorType,
    client_id: str,
    usecase_id: str = "",
    standard_version: str = "",
    now: datetime | None = None,
) -> MemoryRecord:
    """Apply a relationship inference after a different human has verified it."""
    at = _now(now)
    verify_chain(session)
    verifier = _person(verifier_id, "MEMORY_VERIFIER")
    if actor_type != ActorType.HUMAN:
        _deny(session, evidence, correlation_id, "MEMORY_VERIFIER", client_id=client_id)
    try:
        row, digest = _fresh(session, proposal, "relationship_memory")
        if _same_person(verifier, row.proposer_id):
            raise ValueError("MEMORY_VERIFIER")
        body = row.change.get("body") if isinstance(row.change, dict) else None
        record = _record_from(body)
        if record.scope is not MemoryScope.RELATIONSHIP or record.type not in _INFERENCES:
            raise ValueError("MEMORY_PROPOSAL")
        if not isinstance(client_id, str) or client_id != record.client_id or row.subject_id != record.client_id:
            raise ValueError("MEMORY_SCOPE")
        if (row.change.get("usecase_id") or "") != usecase_id:
            raise ValueError("MEMORY_BINDING")
        _window(record, at)
        _reject_foreign(session, record)
        version = _output_version(session, usecase_id, standard_version, at)
        with session.begin_nested():
            _claim(session, row.id, digest)
            session.refresh(row)
            if _proposal_hash(row) != digest or _record_from(row.change["body"]) != record:
                raise ValueError("MEMORY_BINDING")
            stored = _memory_body(record)
            content_hash = _retain(session, stored, "memory", correlation_id)
            _emit(session, evidence, correlation_id, ActorType.HUMAN, verifier, {
                "action": "verify", "scope": MemoryScope.RELATIONSHIP.value,
                "client_id": record.client_id, "type": record.type, "content_hash": content_hash,
                "standard_version": version, "usecase_id": usecase_id, "proposal_id": row.id,
            })
        return _record_from(stored)
    except ValueError as exc:
        session.expire_all()
        _deny(session, evidence, correlation_id, _code(exc), client_id=client_id if isinstance(client_id, str) else "", usecase_id=usecase_id)


def propose_standard(
    session: Session,
    standard: Standard,
    *,
    evidence: EvidenceWriter,
    correlation_id: str,
    proposer_id: str,
    usecase_id: str,
    agent_name: str,
    now: datetime | None = None,
) -> Proposal:
    """Propose a learned standard. Client content and direct application are refused."""
    _now(now)
    verify_chain(session)
    try:
        proposer = _person(proposer_id, "MEMORY_INVALID")
        agent = _text(agent_name, token=True)
        if not isinstance(standard, Standard) or standard.usecase_id != usecase_id or standard.superseded is not None:
            raise ValueError("MEMORY_STANDARD")
        _reject_client_standard(session, standard)
        catalog = _catalog(session)
        previous = catalog.heads.get((usecase_id, standard.standard_id))
        if previous is not None and (standard.version == previous.version or standard.effective < previous.effective):
            raise ValueError("MEMORY_VERSION")
        if any(item.version == standard.version for item in catalog.versions(usecase_id, standard.standard_id)):
            raise ValueError("MEMORY_VERSION")
        body = _standard_body(standard)
        proposal = Proposal(
            kind="institutional_standard",
            before_snapshot={"body": None if previous is None else _standard_body(previous), "agent_name": agent, "usecase_id": usecase_id},
            change={"body": body, "agent_name": agent, "usecase_id": usecase_id},
            rationale="proposed institutional standard",
            subject_type="standard",
            subject_id=standard.standard_id,
            proposer_id=proposer,
        )
        try:
            return _propose(session, proposal, evidence=evidence, correlation_id=correlation_id)
        except ApprovalError:
            raise ValueError("MEMORY_BINDING") from None
    except ValueError as exc:
        _deny(session, evidence, correlation_id, _code(exc), scope=MemoryScope.INSTITUTIONAL.value)


def approve_standard(
    session: Session,
    proposal: Proposal,
    *,
    evidence: EvidenceWriter,
    correlation_id: str,
    approver_id: str,
    actor_type: ActorType,
    now: datetime | None = None,
) -> Standard:
    """Knowledge Lead approval creates the next standard and deployment version."""
    at = _now(now)
    verify_chain(session)
    approver = _person(approver_id, "MEMORY_KNOWLEDGE_LEAD")
    if actor_type != ActorType.HUMAN:
        _deny(session, evidence, correlation_id, "MEMORY_KNOWLEDGE_LEAD")
    try:
        row, digest = _fresh(session, proposal, "institutional_standard")
        if _same_person(approver, row.proposer_id):
            raise ValueError("MEMORY_VERIFIER")
        change = row.change if isinstance(row.change, dict) else {}
        standard = _standard_from(change.get("body"))
        usecase_id = change.get("usecase_id")
        agent_name = change.get("agent_name")
        if usecase_id != standard.usecase_id or not isinstance(agent_name, str):
            raise ValueError("MEMORY_BINDING")
        if standard.superseded is not None:
            raise ValueError("MEMORY_STANDARD")
        _lead(session, usecase_id, approver, at)
        _reject_client_standard(session, standard)
        catalog = _catalog(session)
        if any(item.version == standard.version for item in catalog.versions(usecase_id, standard.standard_id)):
            raise ValueError("MEMORY_VERSION")
        previous = catalog.heads.get((usecase_id, standard.standard_id))
        if previous is not None and (standard.version == previous.version or standard.effective < previous.effective):
            raise ValueError("MEMORY_VERSION")
        before = row.before_snapshot.get("body") if isinstance(row.before_snapshot, dict) else None
        if previous is None and before is not None:
            raise ValueError("MEMORY_BINDING")
        if previous is not None and before != _standard_body(previous):
            raise ValueError("MEMORY_BINDING")
        with session.begin_nested():
            _claim(session, row.id, digest)
            session.refresh(row)
            if _proposal_hash(row) != digest or _standard_from(row.change["body"]) != standard:
                raise ValueError("MEMORY_BINDING")
            stored = _standard_body(standard)
            content_hash = _retain(session, stored, "standard", correlation_id)
            _emit(session, evidence, correlation_id, ActorType.HUMAN, approver, {
                "action": "approve", "scope": MemoryScope.INSTITUTIONAL.value,
                "standard_id": standard.standard_id, "standard_version": standard.version,
                "content_hash": content_hash, "usecase_id": usecase_id,
                "effective": _iso(standard.effective),
                "supersedes": "" if previous is None else previous.version,
                "proposal_id": row.id, "agent": agent_name,
            })
            deployment_hash = _deployment_version(
                session, agent_name, usecase_id, evidence, approver, standard.standard_id, standard.version,
            )
            if not deployment_hash:
                raise ValueError("MEMORY_VERSION")
        return _standard_from(stored)
    except ValueError as exc:
        session.expire_all()
        _deny(session, evidence, correlation_id, _code(exc), scope=MemoryScope.INSTITUTIONAL.value)


def current_standard(
    session: Session,
    standard_id: str,
    *,
    usecase_id: str,
    now: datetime | None = None,
) -> Standard:
    """The single in-force version. Superseded, expired and future versions are refused."""
    at = _now(now)
    verify_chain(session)
    ident = _text(standard_id, token=True)
    usecase = _text(usecase_id, token=True)
    try:
        found = _catalog(session).in_force(usecase, at, ident)
    except ValueError as exc:
        raise ValueError(_code(exc)) from None
    if len(found) != 1:
        raise ValueError("MEMORY_STANDARD")
    return found[0]


__all__ = [
    "MemoryRecord",
    "MemoryScope",
    "Standard",
    "write_memory",
    "propose_memory",
    "verify_memory",
    "propose_standard",
    "approve_standard",
    "current_standard",
]
