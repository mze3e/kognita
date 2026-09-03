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
    requester_id: str = "",
    approver_id: str | None = None,
    proposal_id: int | None = None,
    confirmation_required: bool = False,
    now: datetime | None = None,
) -> Approval:
    """Open a pending approval with optional two-signature gating (ADR 0006).

    Args:
        decision_id: The decision this approval guards
        envelope_hash: Hash of the envelope (tamper-evident binding)
        scope: Approval scope (e.g., "LOCAL", "GLOBAL")
        evidence: Evidence writer for recording this event
        correlation_id: Correlation ID for tracing
        ttl: Time-to-live for this approval (default 48 hours)
        approver_name: Deprecated; use approver_id instead
        requester_id: Who requested this approval (ADR 0006)
        approver_id: Who will approve/confirm (may differ from requester)
        proposal_id: Proposal this approval grants (ADR 0007)
        confirmation_required: If True, requires two signatures (mark, then confirm)
        now: Current time (for testing)
    """
    at = now or utcnow()
    approval = Approval(
        decision_id=decision_id,
        envelope_hash=envelope_hash,
        approver_name=approver_name,
        requester_id=requester_id,
        approver_id=approver_id,
        proposal_id=proposal_id,
        scope=scope,
        status=ApprovalStatus.PENDING,
        confirmation_status=ApprovalStatus.PENDING if confirmation_required else None,
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
            "requester_id": requester_id,
            "approver_id": approver_id,
            "confirmation_required": confirmation_required,
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
    """Single-signature approval transition (unchanged behavior).

    Only used for backward compatibility. New code should use mark() and confirm()
    for two-signature approvals.
    """
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
    approval.approved_at = at if status == ApprovalStatus.APPROVED else approval.approved_at
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


def mark(
    session: Session,
    approval: Approval,
    *,
    approver_id: str,
    evidence: EvidenceWriter,
    correlation_id: str,
    reason: str | None = None,
    now: datetime | None = None,
) -> Approval:
    """First signature of a two-signature approval (ADR 0006).

    Transitions PENDING → MARKED. Data is still held; second signature required.
    """
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

    # Validation: requester != approver for separation of duties
    if approval.requester_id == approver_id:
        raise ApprovalError(
            f"approval {approval.id}: requester and approver must differ (separation of duties)"
        )

    approval.status = ApprovalStatus.MARKED
    approval.approver_id = approver_id
    approval.approved_at = at
    session.add(approval)
    session.flush()
    evidence.emit(
        session,
        correlation_id=correlation_id,
        event_type=EventType.APPROVAL,
        actor_type=ActorType.HUMAN,
        actor_id=approver_id,
        classification=Classification.C1,
        payload={
            "action": "MARKED",
            "approval_id": approval.id,
            "envelope_hash": approval.envelope_hash,
            "reason": reason,
        },
    )
    return approval


def confirm(
    session: Session,
    approval: Approval,
    *,
    approver_id: str,
    evidence: EvidenceWriter,
    correlation_id: str,
    reason: str | None = None,
    now: datetime | None = None,
) -> Approval:
    """Second signature of a two-signature approval (ADR 0006).

    Transitions MARKED → APPROVED. Data is now released.
    """
    at = now or utcnow()
    if approval.status != ApprovalStatus.MARKED:
        raise ApprovalError(
            f"approval {approval.id} is not in MARKED state (current: {ApprovalStatus(approval.status).value})"
        )
    expiry = as_utc(approval.expires_at)
    if expiry is not None and expiry <= at:
        approval.status = ApprovalStatus.EXPIRED
        approval.decided_at = at
        session.add(approval)
        session.flush()
        raise ApprovalError(f"approval {approval.id} expired at {expiry.isoformat()}")

    approval.status = ApprovalStatus.APPROVED
    approval.confirmation_status = ApprovalStatus.APPROVED
    approval.confirmed_at = at
    approval.decided_at = at
    session.add(approval)
    session.flush()
    evidence.emit(
        session,
        correlation_id=correlation_id,
        event_type=EventType.APPROVAL,
        actor_type=ActorType.HUMAN,
        actor_id=approver_id,
        classification=Classification.C1,
        payload={
            "action": "CONFIRMED",
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
    """Grant a pending approval. Expiry is enforced here, not at read time.

    For single-signature approvals, transitions PENDING → APPROVED.
    For two-signature approvals already in MARKED state, same as confirm().
    """
    # If already marked, treat as confirm
    if approval.status == ApprovalStatus.MARKED:
        return confirm(
            session,
            approval,
            approver_id=approver_name,
            evidence=evidence,
            correlation_id=correlation_id,
            reason=reason,
            now=now,
        )

    # Single-signature flow
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


def expire_stale(
    session: Session,
    *,
    now: datetime | None = None,
    evidence: EvidenceWriter | None = None,
) -> int:
    """Mark every lapsed pending approval as expired. Returns how many.

    When evidence is supplied, each expiry is recorded in the chain.
    """
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
        if evidence:
            evidence.emit(
                session,
                correlation_id="",
                event_type=EventType.APPROVAL,
                actor_type=ActorType.SYSTEM,
                actor_id="governance-pdp",
                payload={
                    "action": "EXPIRED",
                    "approval_id": approval.id,
                    "envelope_hash": approval.envelope_hash,
                    "reason": "automatic expiry by ttl",
                },
            )
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
    "mark",
    "confirm",
    "grant",
    "reject",
    "expire_stale",
    "pending",
    "find_granted",
]
