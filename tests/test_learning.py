from dataclasses import replace
import json

import pytest
from sqlmodel import Session, select

from kognita import ExportRequest, Exclusion, add_exclusion, list_exclusions, request_export
from kognita.cli import main
from kognita.evidence import verify_chain
from kognita.models import EvidenceEvent, Interaction, RmReview, RetainedContent
from kognita.retention import RetentionStore
from kognita.vocabulary import ActorType, EventType


def _seed(session, evidence):
    session.add(Interaction(id="a", client_id="client-a"))
    session.add(Interaction(id="b", client_id="client-b"))
    session.flush()
    review = RmReview(interaction_id="a")
    session.add(review)
    session.flush()
    event = evidence.emit(session, correlation_id="a", event_type=EventType.OUTCOME,
                          actor_type=ActorType.HUMAN, actor_id="rm",
                          payload={"interaction_id": "a"})
    return ("interaction:a", f"review:{review.id}", f"outcome:{event.sequence}")


def _request(**kw):
    values = dict(purpose="evaluation", anonymisation="report-1", validation="checks-2",
                  approval="reviewer", requester="analyst", records=("interaction:a",))
    return ExportRequest(**(values | kw))


@pytest.mark.parametrize("purpose", ["training", "fine-tuning", "evaluation"])
def test_positive_and_retained(session, evidence, purpose):
    records = _seed(session, evidence)
    result = request_export(session, _request(purpose=purpose, records=records), evidence=evidence)
    assert result["action"] == "APPROVED"
    assert result["records"] == list(records)
    event = session.exec(select(EvidenceEvent).where(EvidenceEvent.sequence == result["sequence"])).one()
    assert "request" not in event.payload
    body = json.loads(RetentionStore().read(session, event.payload["content_hash"]))
    assert body["request"]["purpose"] == purpose
    assert body["decision"]["action"] == "APPROVED"
    assert verify_chain(session) == 2


@pytest.mark.parametrize("field", ["purpose", "anonymisation", "validation", "approval", "requester"])
@pytest.mark.parametrize("value", ["", "   ", False])
def test_missing_control_denied_and_evidenced(session, evidence, field, value):
    _seed(session, evidence)
    result = request_export(session, replace(_request(), **{field: value}), evidence=evidence)
    assert result["action"] == "DENIED"
    assert f"LEARNING_{field.upper()}_REQUIRED" in result["reason_codes"]
    assert result["records"] == []
    assert verify_chain(session) == 2


def test_default_and_self_approval(session, evidence):
    _seed(session, evidence)
    assert request_export(session, ExportRequest(), evidence=evidence)["action"] == "DENIED"
    result = request_export(session, _request(approval=" ANALYST "), evidence=evidence)
    assert "LEARNING_SELF_APPROVAL" in result["reason_codes"]
    assert result["records"] == []


@pytest.mark.parametrize("client", [True, False])
def test_persistent_exclusion_covers_related_records(engine, session, evidence, client):
    records = _seed(session, evidence)
    exclusion = Exclusion(reason="opt out", **({"client_id": "client-a"} if client else {"record_id": "interaction:a"}))
    add_exclusion(session, exclusion, evidence=evidence, actor_id="operator")
    session.commit()
    with Session(engine) as later:
        assert list_exclusions(later) == (exclusion,)
        for _ in range(2):
            result = request_export(later, _request(records=records + ("interaction:b",)), evidence=evidence)
            assert result["action"] == "APPROVED"
            assert result["records"] == ["interaction:b"]
            assert result["excluded_records"] == list(records)
        later.commit()


def test_record_and_request_local_exclusions(session, evidence):
    records = _seed(session, evidence)
    add_exclusion(session, Exclusion(reason="edit", record_id=records[1]), evidence=evidence, actor_id="rm")
    result = request_export(session, _request(records=records), evidence=evidence)
    assert result["records"] == [records[0], records[2]]
    result = request_export(session, _request(records=records, exclusions=(Exclusion(reason="local", client_id="client-a"),)), evidence=evidence)
    assert result["records"] == []
    assert len(list_exclusions(session)) == 1


@pytest.mark.parametrize("records", [(), ("interaction:unknown",), ("review:01",), ("outcome:999",), ("interaction:a", "unknown")])
def test_selection_bypasses_denied(session, evidence, records):
    _seed(session, evidence)
    result = request_export(session, _request(records=records), evidence=evidence)
    assert result["action"] == "DENIED"
    assert result["records"] == []


@pytest.mark.parametrize("kw", [{}, {"client_id": "a", "record_id": "b"}, {"client_id": " a "}, {"client_id": "a", "reason": ""}])
def test_invalid_exclusion(session, evidence, kw):
    with pytest.raises(ValueError, match="LEARNING_INVALID_EXCLUSION"):
        add_exclusion(session, Exclusion(**({"reason": "opt out"} | kw)), evidence=evidence, actor_id="rm")


def test_erased_exclusion_denies_future_exports(session, evidence):
    _seed(session, evidence)
    add_exclusion(session, Exclusion(reason="opt out", client_id="client-a"), evidence=evidence, actor_id="rm")
    row = session.exec(select(RetainedContent).where(RetainedContent.kind == "learning")).one()
    session.delete(row)
    session.flush()
    result = request_export(session, _request(), evidence=evidence)
    assert result["action"] == "DENIED"
    assert "LEARNING_EXCLUSIONS_UNAVAILABLE" in result["reason_codes"]
    assert verify_chain(session) == 3


def test_cli_persistent_denials_and_exclusion(tmp_path, capsys):
    db = str(tmp_path / "learning.db")
    assert main(["export", "request", "--db", db]) == 2
    assert json.loads(capsys.readouterr().out)["action"] == "DENIED"
    assert main(["export", "exclude", "--db", db, "--client", "client-a", "--reason", "opt out", "--actor", "rm"]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "EXCLUDED"
    from kognita.db import make_engine
    with Session(make_engine(db)) as session:
        session.add(Interaction(id="a", client_id="client-a"))
        session.add(Interaction(id="b", client_id="client-b"))
        session.commit()
    assert main(["export", "request", "--db", db, "--purpose", "training", "--anonymisation", "report", "--validation", "checks", "--requester", "analyst", "--approval", "reviewer", "--record", "interaction:a", "--record", "interaction:b"]) == 0
    assert json.loads(capsys.readouterr().out)["records"] == ["interaction:b"]


def test_redacted_exclusion_cannot_disappear(engine, session, evidence):
    from kognita.evidence import EvidenceWriter
    _seed(session, evidence)
    redacted = EvidenceWriter(engine, redact_payload=lambda kind, body: {})
    add_exclusion(session, Exclusion(reason="opt out", client_id="client-a"), evidence=redacted, actor_id="rm")
    result = request_export(session, _request(), evidence=evidence)
    assert result["action"] == "DENIED"
    assert result["records"] == []
    assert "LEARNING_EXCLUSIONS_UNAVAILABLE" in result["reason_codes"]


def test_corrupt_retained_exclusion_denies(session, evidence):
    _seed(session, evidence)
    add_exclusion(session, Exclusion(reason="opt out", client_id="client-a"), evidence=evidence, actor_id="rm")
    row = session.exec(select(RetainedContent).where(RetainedContent.kind == "learning")).one()
    row.body = '{}'
    session.add(row)
    session.flush()
    assert request_export(session, _request(), evidence=evidence)["action"] == "DENIED"


@pytest.mark.parametrize("exclusions", [None, (Exclusion(reason="invalid"),)])
def test_invalid_local_exclusions_denied(session, evidence, exclusions):
    _seed(session, evidence)
    result = request_export(session, _request(exclusions=exclusions), evidence=evidence)
    assert result["records"] == []
    assert "LEARNING_INVALID_EXCLUSION" in result["reason_codes"]


def test_exclusion_requires_actor_and_aware_timestamp(session, evidence):
    from datetime import datetime
    with pytest.raises(ValueError, match="LEARNING_INVALID_EXCLUSION"):
        add_exclusion(session, Exclusion(reason="opt out", client_id="client-a"), evidence=evidence, actor_id=" ")
    with pytest.raises(ValueError, match="LEARNING_INVALID_EXCLUSION"):
        add_exclusion(session, Exclusion(reason="opt out", client_id="client-a", recorded_at=datetime(2026, 1, 1)), evidence=evidence, actor_id="rm")


def test_record_ids_cannot_overflow_store_or_select_non_outcome_events(session, evidence):
    _seed(session, evidence)
    event = evidence.emit(session, correlation_id="a", event_type=EventType.INTERACTION,
                          payload={"interaction_id": "a"})
    for ident in ("review:999", "review:" + "9" * 100, f"outcome:{event.sequence}"):
        result = request_export(session, _request(records=(ident,)), evidence=evidence)
        assert result["records"] == []
        assert "LEARNING_UNKNOWN_RECORD" in result["reason_codes"]


@pytest.mark.parametrize("missing", ["purpose", "anonymisation", "validation", "approval", "requester", "self-approval"])
def test_cli_controls_cannot_be_bypassed(tmp_path, capsys, missing):
    from kognita.db import create_all, make_engine
    db = str(tmp_path / "controls.db")
    engine = make_engine(db)
    create_all(engine)
    with Session(engine) as session:
        session.add(Interaction(id="a", client_id="client-a"))
        session.commit()
    controls = dict(purpose="evaluation", anonymisation="report", validation="checks",
                    approval="reviewer", requester="analyst")
    if missing == "self-approval":
        controls["approval"] = " ANALYST "
        reason = "LEARNING_SELF_APPROVAL"
    else:
        controls.pop(missing)
        reason = f"LEARNING_{missing.upper()}_REQUIRED"
    args = ["export", "request", "--db", db, "--record", "interaction:a"]
    for name, value in controls.items():
        args.extend([f"--{name}", value])
    assert main(args) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["records"] == []
    assert reason in result["reason_codes"]
    with Session(engine) as session:
        assert verify_chain(session) == 1
        event = session.exec(select(EvidenceEvent)).one()
        body = json.loads(RetentionStore().read(session, event.payload["content_hash"]))
        assert body["decision"]["action"] == "DENIED"


def test_cli_record_exclusion_and_invalid_exclusion(tmp_path, capsys):
    from kognita.db import make_engine
    db = str(tmp_path / "exclusions.db")
    args = ["export", "exclude", "--db", db, "--record", "interaction:a",
            "--reason", "opt out", "--actor", "rm"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["record_id"] == "interaction:a"
    args[-1] = " "
    assert main(args) == 2
    assert json.loads(capsys.readouterr().out)["reason_codes"] == ["LEARNING_INVALID_EXCLUSION"]
    with Session(make_engine(db)) as session:
        assert len(list_exclusions(session)) == 1
        assert verify_chain(session) == 1
