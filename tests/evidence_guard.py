"""Stand-in for an attacker with raw write access to the store.

The store refuses UPDATE and DELETE on evidence rows. Tamper tests that need
to rewrite or remove an event first drop that guard, as anyone with raw write
access could, and then check that verification still reports the change.
"""

from __future__ import annotations


def lift_evidence_guard(session) -> None:
    connection = session.connection()
    connection.exec_driver_sql("DROP TRIGGER IF EXISTS evidence_events_no_update")
    connection.exec_driver_sql("DROP TRIGGER IF EXISTS evidence_events_no_delete")


def tamper_for_test(session, sequence: int, payload: dict):
    """Rewrite one evidence payload and leave its hashes stale.

    Not a correction path: ``payload_hash`` and ``event_hash`` are not
    recomputed, so :func:`kognita.evidence.verify_chain` reports that the
    payload does not match its hash. The update guard is lifted for this one
    write and put back.
    """
    from sqlmodel import select

    from kognita.db import EVIDENCE_GUARD_TRIGGERS
    from kognita.models import EvidenceEvent

    event = session.exec(
        select(EvidenceEvent).where(EvidenceEvent.sequence == sequence)
    ).first()
    if event is None:
        raise LookupError(f"no evidence event at sequence {sequence}")
    connection = session.connection()
    connection.exec_driver_sql("DROP TRIGGER IF EXISTS evidence_events_no_update")
    try:
        event.payload = dict(payload)
        session.add(event)
        session.flush()
    finally:
        if connection.dialect.name == "sqlite":
            connection.exec_driver_sql(EVIDENCE_GUARD_TRIGGERS[0])
    return event
