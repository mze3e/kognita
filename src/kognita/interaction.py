"""The interaction record.

One ``interaction_id`` spans a client journey. Runs and correlation ids attach
to it, and :func:`kognita.reconstruct.reconstruct` reads that span rather than
one request id.

Opening and attaching are evidenced as ``INTERACTION`` events. The roadmap
names the record; that event type is the chain entry for the span.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlmodel import Session, select

from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.models import Interaction, InteractionLink, RunRecord, utcnow
from kognita.usecases import APPROVAL_GATES, load_register
from kognita.vocabulary import ActorType, EventType


class InteractionError(KognitaError):
    """An interaction could not be opened or extended."""


def open_interaction(
    session: Session,
    *,
    client_id: str,
    evidence: EvidenceWriter,
    use_case: str = "",
    approval_gate: str = "",
    interaction_id: str | None = None,
    now: datetime | None = None,
) -> Interaction:
    """Open a journey for one client.

    When ``use_case`` is on the register, the stored gate is the stricter of
    the register's gate and ``approval_gate``. A use case that is not on the
    register keeps the gate named here. An unknown gate is refused.
    """
    at = now or utcnow()
    client = client_id.strip()
    if not client:
        raise InteractionError(
            "an interaction names the client it is about "
            "(Interaction record: client)"
        )
    ident = (interaction_id or "").strip() or f"interaction-{uuid.uuid4().hex}"
    if session.get(Interaction, ident) is not None:
        raise InteractionError(f"interaction {ident} is already open")
    registered = _registered_gate(session, use_case.strip(), at)
    gate = _stricter_gate(approval_gate.strip(), registered)
    row = Interaction(
        id=ident,
        client_id=client,
        use_case=use_case.strip(),
        approval_gate=gate,
        opened_at=at,
    )
    session.add(row)
    session.flush()
    evidence.emit(
        session,
        correlation_id=ident,
        event_type=EventType.INTERACTION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload={
            "action": "OPENED",
            "interaction_id": ident,
            "client_id": client,
            "use_case": row.use_case,
            "approval_gate": gate,
        },
    )
    return row


def attach(
    session: Session,
    interaction_id: str,
    *,
    evidence: EvidenceWriter,
    correlation_id: str = "",
    run_id: str | None = None,
    now: datetime | None = None,
) -> InteractionLink:
    """Attach a correlation id or a run to an open interaction.

    A second attach of the same pair is the existing link. The run must
    already be stored: the link is a foreign key, not a promise.
    """
    del now
    row = require_interaction(session, interaction_id)
    correlation = correlation_id.strip()
    run = (run_id or "").strip()
    if not correlation and not run:
        raise InteractionError(
            "attach names a correlation id or a run "
            "(Interaction record: runs and correlation ids)"
        )
    if run and session.get(RunRecord, run) is None:
        raise InteractionError(f"run {run} is not recorded")
    existing = _find_link(session, row.id, correlation, run)
    if existing is not None:
        return existing
    link = InteractionLink(
        interaction_id=row.id,
        correlation_id=correlation,
        run_id=run or None,
    )
    session.add(link)
    session.flush()
    evidence.emit(
        session,
        correlation_id=row.id,
        event_type=EventType.INTERACTION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload={
            "action": "ATTACHED",
            "interaction_id": row.id,
            "correlation_id": correlation,
            "run_id": run or None,
        },
    )
    return link


def require_interaction(session: Session, interaction_id: str) -> Interaction:
    """The open interaction, or an error. A missing id is not an empty journey."""
    row = session.get(Interaction, interaction_id)
    if row is None:
        raise InteractionError(
            f"interaction {interaction_id} is not open "
            "(Interaction record: unknown interaction)"
        )
    return row


def approval_gate(session: Session, interaction: Interaction, at: datetime) -> str:
    """The gate in force: the stricter of the row and the register.

    A named use case with no gate on the row and no register entry cannot be
    reviewed, because a two-signature requirement would otherwise be skippable.
    """
    registered = _registered_gate(session, interaction.use_case, at)
    gate = _stricter_gate(interaction.approval_gate, registered)
    if interaction.use_case and not gate:
        raise InteractionError(
            f"use case {interaction.use_case} has no approval gate "
            "(Interaction record: approval gate)"
        )
    if gate != interaction.approval_gate:
        interaction.approval_gate = gate
        session.add(interaction)
        session.flush()
    return gate


def _registered_gate(session: Session, use_case: str, at: datetime) -> str:
    if not use_case:
        return ""
    entry = load_register(session, as_of=at).get(use_case)
    if entry is None:
        return ""
    return entry.approval_gate


def _stricter_gate(*gates: str) -> str:
    best = ""
    best_rank = -1
    for gate in gates:
        if not gate:
            continue
        if gate not in APPROVAL_GATES:
            raise InteractionError(
                f"unknown approval gate {gate!r} " "(Interaction record: approval gate)"
            )
        rank = APPROVAL_GATES.index(gate)
        if rank > best_rank:
            best = gate
            best_rank = rank
    return best


def _find_link(
    session: Session, interaction_id: str, correlation_id: str, run_id: str
) -> InteractionLink | None:
    rows = session.exec(
        select(InteractionLink).where(InteractionLink.interaction_id == interaction_id)
    ).all()
    for link in rows:
        if link.correlation_id != correlation_id:
            continue
        if (link.run_id or "") != run_id:
            continue
        return link
    return None


def linked_correlation_ids(session: Session, interaction_id: str) -> set[str]:
    """Correlation ids attached to ``interaction_id``, including its own id."""
    found = {interaction_id}
    rows = session.exec(
        select(InteractionLink).where(InteractionLink.interaction_id == interaction_id)
    ).all()
    for link in rows:
        if link.correlation_id:
            found.add(link.correlation_id)
    return found


__all__ = [
    "InteractionError",
    "approval_gate",
    "attach",
    "linked_correlation_ids",
    "open_interaction",
    "require_interaction",
]
