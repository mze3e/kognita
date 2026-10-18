"""Regression tests for finding 13: evidence truncation must not verify clean.

``verify_export`` only checked that whatever rows it was handed were
internally consistent with each other. Dropping trailing events from an
export and leaving ``event_count``/``head_hash`` untouched left a perfectly
self-consistent prefix, so the truncation passed. ``verify_chain`` had the
same shape of bug: a missing or unreadable ``evidence_head`` checkpoint was
treated as "nothing to compare against" rather than a reason to fail, so
deleting the head row (or the whole table) while events still existed also
passed.

These tests build a real engine and evidence store directly with
``kognita.db.create_all`` / ``kognita.evidence.EvidenceWriter`` — no pack,
no governance decisions, no autouse delegation/credential fixtures — so they
exercise the module the way an attacker with raw database access would.
"""
from __future__ import annotations

import copy

import pytest
from sqlmodel import Session

from kognita.db import create_all, make_engine
from kognita.evidence import (
    ChainBreak,
    EvidenceWriter,
    export_chain,
    verify_chain,
    verify_export,
)
from kognita.vocabulary import EventType


def _store(n: int = 4):
    """A fresh engine with ``n`` events and the export of its full chain."""
    engine = make_engine()
    create_all(engine)
    evidence = EvidenceWriter(engine)
    with Session(engine) as session:
        for i in range(n):
            evidence.emit(
                session,
                correlation_id=f"c{i}",
                event_type=EventType.TOOL_CALL,
                payload={"i": i},
            )
        session.commit()
        export = export_chain(session)
    return engine, export


def _drop_head_trigger(session: Session, name: str) -> None:
    session.connection().exec_driver_sql(f"DROP TRIGGER IF EXISTS {name}")


# ── verify_export ────────────────────────────────────────────────────────────


def test_verify_export_accepts_an_untampered_export():
    _, export = _store()
    assert verify_export(export) == export["event_count"] == 4


def test_verify_export_rejects_a_removed_last_event():
    """The Codex repro: an export advertising 4 events, delivering 3, metadata untouched."""
    _, export = _store(4)
    truncated = copy.deepcopy(export)
    truncated["events"].pop()
    assert truncated["event_count"] == 4
    assert len(truncated["events"]) == 3
    with pytest.raises(ChainBreak):
        verify_export(truncated)
    # The untouched export still verifies — the rejection is about the
    # truncation, not some unrelated fixture problem.
    assert verify_export(export) == 4


def test_verify_export_rejects_a_removed_middle_event():
    _, export = _store(4)
    truncated = copy.deepcopy(export)
    truncated["events"].pop(1)
    with pytest.raises(ChainBreak):
        verify_export(truncated)


def test_verify_export_rejects_count_mismatch_even_when_head_hash_is_fixed_up():
    """Isolates the event_count check: head_hash is corrected, event_count is not."""
    _, export = _store(4)
    truncated = copy.deepcopy(export)
    truncated["events"].pop()
    truncated["head_hash"] = truncated["events"][-1]["event_hash"]
    assert truncated["event_count"] != len(truncated["events"])
    with pytest.raises(ChainBreak):
        verify_export(truncated)


def test_verify_export_rejects_head_hash_mismatch_even_when_count_is_fixed_up():
    """Isolates the head_hash check: event_count is corrected, head_hash is not."""
    _, export = _store(4)
    truncated = copy.deepcopy(export)
    truncated["events"].pop()
    truncated["event_count"] = len(truncated["events"])
    assert truncated["head_hash"] != truncated["events"][-1]["event_hash"]
    with pytest.raises(ChainBreak):
        verify_export(truncated)


# ── verify_chain ─────────────────────────────────────────────────────────────


def test_verify_chain_tolerates_a_missing_head_with_no_events():
    engine = make_engine()
    create_all(engine)
    with Session(engine) as session:
        assert verify_chain(session) == 0


def test_verify_chain_fails_when_head_row_deleted_while_events_exist():
    engine, _ = _store(3)
    with Session(engine) as session:
        assert verify_chain(session) == 3
        _drop_head_trigger(session, "evidence_head_no_delete")
        session.connection().exec_driver_sql("DELETE FROM evidence_head WHERE id = 1")
        session.commit()
        with pytest.raises(ChainBreak):
            verify_chain(session)


def test_verify_chain_fails_when_head_table_is_unreadable_while_events_exist():
    """A missing/unreadable head table is a dialect without the checkpoint, not a pass."""
    engine, _ = _store(3)
    with Session(engine) as session:
        assert verify_chain(session) == 3
        session.connection().exec_driver_sql("DROP TABLE evidence_head")
        session.commit()
        with pytest.raises(ChainBreak):
            verify_chain(session)


def test_verify_chain_fails_when_head_points_past_the_last_event():
    engine, _ = _store(3)
    with Session(engine) as session:
        assert verify_chain(session) == 3
        session.connection().exec_driver_sql(
            "UPDATE evidence_head SET sequence = sequence + 100, "
            "event_hash = '" + "ab" * 32 + "' WHERE id = 1"
        )
        session.commit()
        with pytest.raises(ChainBreak):
            verify_chain(session)


def test_verify_chain_fails_when_head_points_behind_the_last_event():
    engine, _ = _store(3)
    with Session(engine) as session:
        assert verify_chain(session) == 3
        _drop_head_trigger(session, "evidence_head_strictly_forward")
        session.connection().exec_driver_sql(
            "UPDATE evidence_head SET sequence = 1, event_hash = '" + "cd" * 32 + "' WHERE id = 1"
        )
        session.commit()
        with pytest.raises(ChainBreak):
            verify_chain(session)
