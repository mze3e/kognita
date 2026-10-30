"""Explicit, evidenced permission to select records for external learning.

This module returns a selection manifest, not a training pipeline. Record ids
are interaction:<id>, review:<id>, or outcome:<evidence sequence>. Approval
identities are supplied by the trusted calling application, like other local
operator identities. An approval applies only to this exact request.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlmodel import Session, select

from kognita.canonical import hash_text
from kognita.evidence import ChainBreak, EvidenceWriter, verify_chain
from kognita.models import EvidenceEvent, Interaction, RmReview, utcnow
from kognita.retention import RetentionStore
from kognita.vocabulary import ActorType, EventType


@dataclass(frozen=True)
class Exclusion:
    """A permanent exclusion, scoped to exactly one client or record."""

    reason: str
    client_id: str = ""
    record_id: str = ""
    recorded_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True)
class ExportRequest:
    """An explicit record selection and the attestations authorising it.

    Anonymisation and validation name the completed controls or their evidence.
    Approval is the approver identity, distinct from requester. Request-local
    exclusions supplement, and cannot replace, permanent exclusions.
    """

    purpose: str = ""
    anonymisation: str = ""
    validation: str = ""
    approval: str = ""
    requester: str = ""
    records: tuple[str, ...] = ()
    exclusions: tuple[Exclusion, ...] = ()


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_exclusion(item: Exclusion) -> bool:
    return (isinstance(item, Exclusion) and _text(item.reason)
            and isinstance(item.client_id, str) and isinstance(item.record_id, str)
            and item.client_id == item.client_id.strip()
            and item.record_id == item.record_id.strip()
            and bool(item.client_id.strip()) != bool(item.record_id.strip())
            and isinstance(item.recorded_at, datetime)
            and item.recorded_at.utcoffset() is not None)


def _record(session: Session, record_id: str) -> Interaction | None:
    kind, _, ident = record_id.partition(":")
    if kind in ("review", "outcome"):
        if (not ident.isascii() or not ident.isdecimal() or len(ident) > 19
                or str(int(ident)) != ident or int(ident) > 2**63 - 1):
            return None
    if kind == "interaction":
        return session.get(Interaction, ident)
    if kind == "review" and ident.isdecimal():
        review = session.get(RmReview, int(ident))
        return session.get(Interaction, review.interaction_id) if review else None
    if kind == "outcome" and ident.isdecimal():
        event = session.exec(select(EvidenceEvent).where(
            EvidenceEvent.sequence == int(ident),
            EvidenceEvent.event_type == EventType.OUTCOME,
        )).first()
        if event:
            return session.get(Interaction, event.payload.get("interaction_id", ""))
    return None


def _evidence(
    session: Session, evidence: EvidenceWriter, action: str, actor: str,
    body: dict[str, Any],
) -> int:
    correlation = f"learning:{uuid4()}"
    digest = RetentionStore().retain_value(
        session, body, kind="learning", correlation_id=correlation,
    )
    event = evidence.emit(
        session, correlation_id=correlation, event_type=EventType.LEARNING_EXPORT,
        actor_type=ActorType.HUMAN, actor_id=actor,
        payload={"action": action, "content_hash": digest},
    )
    return event.sequence


def add_exclusion(
    session: Session, exclusion: Exclusion, *, evidence: EvidenceWriter, actor_id: str,
) -> Exclusion:
    """Persist a permanent exclusion in retained, hash-linked evidence."""
    if not _valid_exclusion(exclusion) or not _text(actor_id):
        raise ValueError("LEARNING_INVALID_EXCLUSION")
    _evidence(session, evidence, "EXCLUDED", actor_id, asdict(exclusion))
    return exclusion


def list_exclusions(session: Session) -> tuple[Exclusion, ...]:
    """Read permanent exclusions, failing closed on damaged or erased evidence."""
    verify_chain(session)
    result = []
    for event in session.exec(select(EvidenceEvent).where(
        EvidenceEvent.event_type == EventType.LEARNING_EXPORT,
    ).order_by(EvidenceEvent.sequence)).all():
        action = event.payload.get("action")
        if action in ("APPROVED", "DENIED"):
            continue
        if action != "EXCLUDED":
            raise ValueError("LEARNING_EXCLUSIONS_UNAVAILABLE")
        digest = event.payload.get("content_hash", "")
        raw = RetentionStore().read(session, digest)
        if raw is None or hash_text(raw) != digest:
            raise ValueError("LEARNING_EXCLUSIONS_UNAVAILABLE")
        body = json.loads(raw)
        body["recorded_at"] = datetime.fromisoformat(body["recorded_at"])
        exclusion = Exclusion(**body)
        if not _valid_exclusion(exclusion):
            raise ValueError("LEARNING_EXCLUSIONS_UNAVAILABLE")
        result.append(exclusion)
    return tuple(result)


def request_export(
    session: Session, request: ExportRequest, *, evidence: EvidenceWriter,
) -> dict[str, Any]:
    """Evidence every decision and return only permitted record identifiers.

    Missing controls deny the whole request. Excluded records are filtered.
    Unknown records deny the whole request, preventing unscoped content export.
    The caller owns the transaction, including committing denied requests.
    """
    reasons = []
    for name in ("purpose", "anonymisation", "validation", "approval", "requester"):
        if not _text(getattr(request, name)):
            reasons.append(f"LEARNING_{name.upper()}_REQUIRED")
    if (_text(request.approval) and _text(request.requester)
            and request.approval.strip().casefold() == request.requester.strip().casefold()):
        reasons.append("LEARNING_SELF_APPROVAL")
    exclusions = ()
    try:
        exclusions = list_exclusions(session)
    except (ValueError, TypeError, KeyError, RuntimeError, ChainBreak):
        reasons.append("LEARNING_EXCLUSIONS_UNAVAILABLE")
    if (not isinstance(request.exclusions, (tuple, list))
            or not all(_valid_exclusion(item) for item in request.exclusions)):
        reasons.append("LEARNING_INVALID_EXCLUSION")
    else:
        exclusions += tuple(request.exclusions)
    selected, excluded = [], []
    if not isinstance(request.records, (tuple, list)) or not request.records:
        reasons.append("LEARNING_SELECTION_REQUIRED")
    else:
        for ident in request.records:
            interaction = _record(session, ident) if _text(ident) else None
            if interaction is None or not _text(interaction.client_id):
                reasons.append("LEARNING_UNKNOWN_RECORD")
                continue
            if any(item.client_id == interaction.client_id or item.record_id in (
                ident, f"interaction:{interaction.id}",
            ) for item in exclusions):
                excluded.append(ident)
            elif ident not in selected:
                selected.append(ident)
    result = {
        "action": "DENIED" if reasons else "APPROVED",
        "reason_codes": sorted(set(reasons)),
        "records": [] if reasons else selected,
        "excluded_records": excluded,
    }
    result["sequence"] = _evidence(
        session, evidence, result["action"],
        request.requester if _text(request.requester) else "unknown",
        {"request": asdict(request), "decision": dict(result)},
    )
    return result


__all__ = ["ExportRequest", "Exclusion", "request_export", "add_exclusion", "list_exclusions"]
