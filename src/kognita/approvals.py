"""Regulated approvals — the human step, bound to what was actually reviewed.

An approval is issued against an ``envelope_hash``, never against a tool name or
a subject id. That hash covers the envelope, the resolved attributes and the
checks, so an approval granted for "discuss product A with client B from
Singapore" cannot be replayed for the same tool from a different location: any
change produces a different hash and no matching approval.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

from sqlmodel import Session, select

from kognita.evidence import EvidenceWriter
from kognita.models import Approval, as_utc, utcnow
from kognita.vocabulary import ActorType, ApprovalStatus, Classification, EventType


class ApprovalError(Exception):
    """An approval could not be transitioned as requested."""


def open_approval(
    session: Session,
    *,
    decision_id: int,
    envelope_hash: str,
    scope: str,
    evidence: EvidenceWriter,
    correlation_id: str,
    ttl: timedelta = timedelta(hours=48),
    approver_name: str = "Unassigned (duty desk)",
    now: datetime | None = None,
) -> Approval:
    """Open a pending approval. Normally called for you by ``record()``."""
    at = now or utcnow()
    approval = Approval(
        decision_id=decision_id,
        envelope_hash=envelope_hash,
        approver_name=approver_name,
        scope=scope,
        created_at=at,
        expires_at=at + ttl,
    )
    session.add(approval)
    session.flush()
    evidence.emit(
        session,
        correlation_id=correlation_id,
        event_type=EventType.APPROVAL,
        actor_id="governance-pdp",
        payload={
            "action": "OPENED",
            "approval_id": approval.id,
            "envelope_hash": envelope_hash,
            "scope": scope,
        },
    )
    return approval


def _transition(
    session: Session,
    approval: Approval,
    *,
    status: ApprovalStatus,
    approver_name: str,
    reason: str | None,
    evidence: EvidenceWriter,
    correlation_id: str,
    now: datetime | None = None,
) -> Approval:
    at = now or utcnow()
    if approval.status != ApprovalStatus.PENDING:
        raise ApprovalError(
            f"approval {approval.id} is already {ApprovalStatus(approval.status).value}"
        )
    expiry = as_utc(approval.expires_at)
    if expiry is not None and expiry <= at:
        approval.status = ApprovalStatus.EXPIRED
        approval.decided_at = at
        session.add(approval)
        session.flush()
        raise ApprovalError(f"approval {approval.id} expired at {expiry.isoformat()}")

    approval.status = status
    approval.approver_name = approver_name
    approval.reason = reason
    approval.decided_at = at
    session.add(approval)
    session.flush()
    evidence.emit(
        session,
        correlation_id=correlation_id,
        event_type=EventType.APPROVAL,
        actor_type=ActorType.HUMAN,
        actor_id=approver_name,
        classification=Classification.C1,
        payload={
            "action": status.value,
            "approval_id": approval.id,
            "envelope_hash": approval.envelope_hash,
            "reason": reason,
        },
    )
    return approval


def grant(
    session: Session,
    approval: Approval,
    *,
    approver_name: str,
    evidence: EvidenceWriter,
    correlation_id: str,
    reason: str | None = None,
    now: datetime | None = None,
) -> Approval:
    """Grant a pending approval. Expiry is enforced here, not at read time."""
    return _transition(
        session,
        approval,
        status=ApprovalStatus.APPROVED,
        approver_name=approver_name,
        reason=reason,
        evidence=evidence,
        correlation_id=correlation_id,
        now=now,
    )


def reject(
    session: Session,
    approval: Approval,
    *,
    approver_name: str,
    evidence: EvidenceWriter,
    correlation_id: str,
    reason: str | None = None,
    now: datetime | None = None,
) -> Approval:
    """Reject a pending approval."""
    return _transition(
        session,
        approval,
        status=ApprovalStatus.REJECTED,
        approver_name=approver_name,
        reason=reason,
        evidence=evidence,
        correlation_id=correlation_id,
        now=now,
    )


def expire_stale(session: Session, *, now: datetime | None = None) -> int:
    """Mark every lapsed pending approval as expired. Returns how many."""
    at = now or utcnow()
    stale = [
        a
        for a in session.exec(
            select(Approval).where(Approval.status == ApprovalStatus.PENDING)
        ).all()
        if (expiry := as_utc(a.expires_at)) is not None and expiry <= at
    ]
    for approval in stale:
        approval.status = ApprovalStatus.EXPIRED
        approval.decided_at = at
        session.add(approval)
    session.flush()
    return len(stale)


def pending(session: Session, *, now: datetime | None = None) -> Sequence[Approval]:
    """Approvals still awaiting a decision and not yet lapsed."""
    at = now or utcnow()
    return [
        a
        for a in session.exec(
            select(Approval).where(Approval.status == ApprovalStatus.PENDING)
        ).all()
        if a.is_live(at)
    ]


def find_granted(
    session: Session, envelope_hash: str, *, now: datetime | None = None
) -> Approval | None:
    """The live granted approval for this exact envelope, if there is one.

    Matching on the hash is what stops an approval being reused for a materially
    different request that merely looks similar.
    """
    at = now or utcnow()
    for approval in session.exec(
        select(Approval).where(Approval.envelope_hash == envelope_hash)
    ).all():
        if approval.status != ApprovalStatus.APPROVED:
            continue
        expiry = as_utc(approval.expires_at)
        if expiry is None or expiry > at:
            return approval
    return None


__all__ = [
    "ApprovalError",
    "open_approval",
    "grant",
    "reject",
    "expire_stale",
    "pending",
    "find_granted",
]
