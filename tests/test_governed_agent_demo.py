"""The flagship demo: scaffold, then the four scenarios, checked from the store.

The script's own prints are not the proof. After ``run_demo.py`` exits, this
reads the SQLite store, the bytes the stand-in received, and a fresh
``kognita evidence verify``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from sqlmodel import select

from kognita import (
    Agent,
    CheckResult,
    EventType,
    EvidenceEvent,
    GovernanceDecision,
    KnowledgeItem,
    Outcome,
    Policy,
    canonical_json,
    make_engine,
    session_scope,
)
from kognita.cli import build_parser

SECRET = "ana@example.org"
CITATION = "Data Sharing Charter s4, Schedule 1"


def _verify(db: Path) -> subprocess.CompletedProcess[str]:
    kognita = shutil.which("kognita")
    command = (
        [kognita, "evidence", "verify", "--db", str(db)]
        if kognita
        else [
            sys.executable,
            "-m",
            "kognita.cli",
            "evidence",
            "verify",
            "--db",
            str(db),
        ]
    )
    return subprocess.run(command, capture_output=True, text=True)


def test_scaffold_command_is_governed_agent():
    args = build_parser().parse_args(
        ["scaffold", "--template", "governed-agent", "--dest", "out"]
    )
    assert args.command == "scaffold"
    assert args.template == "governed-agent"
    assert args.dest == "out"
    assert args.func.__name__ == "cmd_scaffold"


def test_four_scenarios_run_from_scaffold(tmp_path: Path):
    dest = tmp_path / "governed-agent"
    started = time.monotonic()
    scaffold = subprocess.run(
        [
            sys.executable,
            "-m",
            "kognita.cli",
            "scaffold",
            "--template",
            "governed-agent",
            "--dest",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    assert scaffold.returncode == 0, scaffold.stderr
    db = dest / "kognita.db"
    assert db.is_file()

    engine = make_engine(db)
    with session_scope(engine) as session:
        assert session.exec(select(Policy)).all()
        assert session.exec(select(Agent).where(Agent.name == "dossier-agent")).first()
        assert session.exec(select(KnowledgeItem)).all()
        assert session.exec(select(EvidenceEvent)).all() == []
    engine.dispose()

    demo = subprocess.run(
        [sys.executable, str(dest / "run_demo.py")],
        capture_output=True,
        text=True,
        cwd=dest,
    )
    elapsed = time.monotonic() - started
    assert demo.returncode == 0, demo.stderr
    assert elapsed < 180, f"demo took {elapsed:.1f}s"

    engine = make_engine(db)
    with session_scope(engine) as session:
        decisions = list(session.exec(select(GovernanceDecision)).all())
        allowed = [
            row
            for row in decisions
            if row.subject_id == "2" and row.tool == "get_subject_profile"
        ]
        denied = [
            row
            for row in decisions
            if row.subject_id == "1" and row.tool == "get_subject_profile"
        ]
        assert len(allowed) == 1 and allowed[0].outcome is Outcome.ALLOW
        assert len(denied) == 1 and denied[0].outcome is Outcome.DENY
        assert any(
            check["result"] == CheckResult.FAIL.value and check["citation"] == CITATION
            for check in denied[0].checks
        )

        events = list(
            session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
        )
        retrievals = [
            event for event in events if event.event_type is EventType.RETRIEVAL
        ]
        assert len(retrievals) == 1
        assert retrievals[0].correlation_id == allowed[0].request_id
        assert retrievals[0].payload["returned_ids"]
        denied_events = [
            event for event in events if event.correlation_id == denied[0].request_id
        ]
        assert denied_events
        assert all(
            event.event_type is not EventType.RETRIEVAL for event in denied_events
        )

        model_calls = [
            event for event in events if event.event_type is EventType.MODEL_CALL
        ]
        assert model_calls
        manifest = model_calls[-1].payload["manifest_hash"]
        assert len(manifest) == 64
        assert model_calls[-1].payload["redacted_spans"] >= 1
        assert model_calls[-1].payload["decision"] == "REDACT"
        blob = canonical_json([event.payload for event in events])
        assert SECRET not in blob
        assert "Email ana@example.org about the notes" not in blob
        assert any(event.payload.get("tampered") is True for event in events)
    engine.dispose()

    forwarded = (dest / "upstream-request.json").read_bytes()
    assert SECRET.encode() not in forwarded
    assert b"[EMAIL_1]" in forwarded
    assert b"about the notes" in forwarded

    verified = _verify(db)
    assert verified.returncode == 1
    assert "BROKEN" in verified.stderr
    assert "payload does not match its hash" in verified.stderr
