"""Claim provenance and fail-closed label gates."""
from dataclasses import replace
import json

import pytest
from sqlmodel import Session, select

from kognita import Claim, ClaimType, ValidatorStatus
from kognita.claims import record_claim, release_claim, relabel_claim, list_claims
from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.models import EvidenceEvent
from kognita.reconstruct import reconstruct, render_reconstruction
from kognita.retention import RetentionStore
from kognita.vocabulary import CheckResult, EventType


@pytest.fixture
def store(tmp_path):
    path = str(tmp_path / "claims.db")
    engine = make_engine(path)
    create_all(engine)
    with Session(engine) as session:
        yield session, EvidenceWriter(engine), path


def _record(store, claim_type=ClaimType.INFERENCE):
    session, writer, _ = store
    claim = Claim("c1", "client1", "Client may be considering a sale", claim_type)
    record_claim(session, claim, evidence=writer, interaction_id="i1")
    return claim


@pytest.mark.parametrize("audience", ["rm", "client"])
@pytest.mark.parametrize("typed", [True, False])
def test_release(store, audience, typed):
    session, writer, _ = store
    _record(store, ClaimType.INFERENCE if typed else None)
    check = release_claim(session, "c1", evidence=writer, interaction_id="i1", audience=audience)
    assert check.result == (CheckResult.PASS if typed else CheckResult.FAIL)
    assert check.check == ("C1-LABELLED" if typed else "C1-UNLABELLED")
    report = reconstruct(session, "i1")
    assert len(report["claims"]) == int(typed and audience == "rm")
    if typed and audience == "rm":
        assert "Client may be considering a sale" in render_reconstruction(report)
    verify_chain(session)


@pytest.mark.parametrize("original", [ClaimType.INFERENCE, ClaimType.CONVERSATION_HYPOTHESIS])
@pytest.mark.parametrize("source_system,source_record,status,allowed", [
    ("crm", "r1", ValidatorStatus.VERIFIED, True),
    ("", "r1", ValidatorStatus.VERIFIED, False),
    ("crm", "", ValidatorStatus.VERIFIED, False),
    ("crm", "r1", ValidatorStatus.INFERRED, False),
    (" ", "r1", ValidatorStatus.VERIFIED, False),
])
def test_promotion(store, original, source_system, source_record, status, allowed):
    session, writer, _ = store
    _record(store, original)
    def promote():
        return relabel_claim(session, "c1", ClaimType.VERIFIED_FACT, evidence=writer,
                             interaction_id="i1", source_system=source_system,
                             source_record=source_record, validator_status=status)
    if allowed:
        assert promote().claim_type == ClaimType.VERIFIED_FACT
        assert list_claims(session)[0].validator_status == ValidatorStatus.VERIFIED
        assert any(e.event_type == EventType.CLAIM_VERIFIED for e in session.exec(select(EvidenceEvent)))
    else:
        with pytest.raises(ValueError, match="C1-RELABEL"):
            promote()
        assert list_claims(session)[0].claim_type == original


def test_cannot_overwrite_or_release_fabricated_claim(store):
    session, writer, _ = store
    claim = _record(store, None)
    with pytest.raises(ValueError, match="C1-DUPLICATE"):
        record_claim(session, replace(claim, claim_type=ClaimType.VERIFIED_FACT), evidence=writer, interaction_id="i1")
    with pytest.raises(ValueError, match="C1-UNKNOWN"):
        release_claim(session, "invented", evidence=writer, interaction_id="i1")
    assert release_claim(session, "c1", evidence=writer, interaction_id="i1", audience="other").result == CheckResult.FAIL
    assert list_claims(session, client_id="other") == []
    assert list_claims(session, interaction_id="other") == []
    assert list_claims(session, interaction_id="i1") == [claim]


def test_retention_and_cli(store, capsys):
    session, writer, path = store
    claim = _record(store)
    event = session.exec(select(EvidenceEvent)).first()
    assert claim.claim not in json.dumps(event.payload)
    body = RetentionStore().read(session, event.payload["claim_hash"])
    assert json.loads(body)["claim"] == claim.claim
    session.commit()
    assert main(["claims", "show", "c1", "--db", path]) == 0
    assert json.loads(capsys.readouterr().out)["claim_id"] == "c1"
    assert main(["claims", "list", "--client", "client1", "--interaction", "i1", "--db", path]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1
    assert main(["claims", "show", "missing", "--db", path]) == 2
    RetentionStore().erase(session, event.payload["claim_hash"], evidence=writer, actor_id="admin")
    with pytest.raises(ValueError, match="C1-CONTENT"):
        release_claim(session, "c1", evidence=writer, interaction_id="i1")
    verify_chain(session)


def test_reconstruction_keeps_the_label_seen(store):
    session, writer, _ = store
    _record(store)
    release_claim(session, "c1", evidence=writer, interaction_id="i1")
    relabel_claim(session, "c1", ClaimType.VERIFIED_FACT, evidence=writer,
                  interaction_id="i1", source_system="crm", source_record="r1",
                  validator_status=ValidatorStatus.VERIFIED)
    assert list_claims(session)[0].claim_type == ClaimType.VERIFIED_FACT
    assert reconstruct(session, "i1")["claims"][0]["claim_type"] == ClaimType.INFERENCE


def test_changed_retained_content_fails_closed(store):
    from kognita.models import RetainedContent
    session, writer, _ = store
    _record(store)
    event = session.exec(select(EvidenceEvent)).first()
    row = session.get(RetainedContent, event.payload["claim_hash"])
    row.body = row.body.replace("INFERENCE", "VERIFIED_FACT")
    session.add(row)
    session.flush()
    with pytest.raises(ValueError, match="C1-CONTENT"):
        release_claim(session, "c1", evidence=writer, interaction_id="i1")


@pytest.mark.parametrize("field,value", [("claim_type", "invented"), ("validator_status", "invented")])
def test_invalid_labels_are_not_accepted(field, value):
    with pytest.raises(ValueError):
        Claim("c1", "client1", "A statement", **{field: value})


def test_list_includes_claims_released_on_another_interaction(store):
    session, writer, _ = store
    claim = _record(store)
    release_claim(session, "c1", evidence=writer, interaction_id="i2")
    assert list_claims(session, interaction_id="i2") == [claim]
    assert reconstruct(session, "i2")["claims"][0]["claim_id"] == "c1"
