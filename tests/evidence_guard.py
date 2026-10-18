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
