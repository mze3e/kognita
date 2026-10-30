"""Superseding a source invalidates dependents and lists who still has it."""
from datetime import datetime, timedelta, timezone

import json
import pytest
from sqlmodel import Session, select

from kognita import Claim, ClaimType, ValidatorStatus, affected_by_source, supersede_source
from kognita.claims import list_claims, record_claim, relabel_claim, release_claim
from kognita.cli import main
from kognita.communication import send_client_communication
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.interaction import open_interaction
from kognita.models import EvidenceEvent, RetainedContent, UseCase
from kognita.origination import originate
from kognita.review import ReviewError, record_final_decision, review
from kognita.sources import Source, register_source
from kognita.vocabulary import CheckResult, EventType

WHEN = datetime(2020, 8, 1, tzinfo=timezone.utc)
EFFECTIVE = datetime(2020, 1, 1, tzinfo=timezone.utc)
REASON = {
    "fact": "The published view names the sector",
    "detail": "The CIO desk published the current sector stance",
    "citation": "house view register",
}


@pytest.fixture
def store(tmp_path):
    path = str(tmp_path / "house.db")
    engine = make_engine(path)
    create_all(engine)
    with Session(engine) as session:
        writer = EvidenceWriter(engine, clock=lambda: WHEN)
        session.add(UseCase(
            usecase_id="advice",
            effective_from=EFFECTIVE,
            record={"usecase_id": "advice", "approved_sources": ["house-view", "policy-note"]},
        ))
        session.flush()
        register_source(session, _source("neutral"), evidence=writer, interaction_id="setup")
        register_source(session, _source("overweight"), evidence=writer, interaction_id="setup")
        register_source(
            session,
            Source("policy-note", "policy", "1", "Policy owner", EFFECTIVE),
            evidence=writer,
            interaction_id="setup",
        )
        yield session, writer, path


def _source(version):
    return Source("house-view", "house view", version, "CIO desk", EFFECTIVE)


def _claim(claim_id, client_id, version, text="Neutral outlook"):
    return Claim(claim_id, client_id, text, ClaimType.INFERENCE, source_system="house-view", source_record=version)


def _journey(session, writer, name, client_id, version, *, send=False, reject=False, text="Neutral outlook"):
    interaction = open_interaction(
        session, client_id=client_id, use_case="advice", evidence=writer, interaction_id=name, now=WHEN,
    )
    record_claim(session, _claim(f"c-{name}", client_id, version, text), evidence=writer, interaction_id=interaction.id)
    originate(
        session, interaction.id, evidence=writer, trigger="RM request", candidate_set_size=1,
        criteria=["house view"], reasons=[REASON], product_id="fund-1", now=WHEN,
    )
    if send:
        writer.emit(
            session, correlation_id=interaction.id, event_type=EventType.CLIENT_COMMUNICATION,
            payload={"lane": "client", "interaction_id": interaction.id, "client_id": client_id, "sent": True, "channel": "email"},
        )
    if reject:
        record_final_decision(
            session, interaction.id, outcome="reject", evidence=writer, actor_id="rm-1",
            citation="The recommendation is refused", now=WHEN,
        )
    return interaction


def _supersede(store):
    session, writer, _ = store
    return supersede_source(
        session, "house-view", version="neutral", by="overweight", evidence=writer, interaction_id="supersede",
    )


def _ids(rows, key="interaction_id"):
    return {row[key] for row in rows}


def test_supersede_stales_dependents_and_lists_affected(store):
    session, writer, _ = store
    opened = _journey(session, writer, "open", "client-open", "neutral")
    sent = _journey(session, writer, "sent", "client-sent", "neutral", send=True)
    rejected = _journey(session, writer, "rejected", "client-rejected", "neutral", reject=True)
    live = _journey(session, writer, "live", "client-live", "overweight", text="Overweight outlook")
    other = open_interaction(
        session, client_id="client-other", use_case="advice", evidence=writer, interaction_id="other", now=WHEN,
    )
    record_claim(
        session,
        Claim("c-other", "client-other", "A policy statement", ClaimType.INFERENCE, source_system="policy-note", source_record="1"),
        evidence=writer, interaction_id=other.id,
    )
    record_claim(
        session,
        Claim("c-blank", "client-blank", "Unversioned outlook", ClaimType.INFERENCE, source_system="house-view", source_record=""),
        evidence=writer, interaction_id="blank",
    )

    updated = _supersede(store)
    assert updated.superseded_by == "overweight"
    event = next(row for row in session.exec(select(EvidenceEvent)) if row.event_type == EventType.SOURCE_SUPERSEDED)
    assert event.payload["action"] == "supersede"
    assert event.payload["claims"] == ["c-blank", "c-open", "c-rejected", "c-sent"]
    assert event.payload["recommendations"] == [{
        "interaction_id": opened.id, "client_id": "client-open", "validator_status": "STALE",
    }]
    encoded = json.dumps(event.payload)
    assert "Neutral outlook" not in encoded
    assert "CIO desk" not in encoded

    by_id = {claim.claim_id: claim for claim in list_claims(session)}
    for claim_id in ("c-blank", "c-open", "c-rejected", "c-sent"):
        assert by_id[claim_id].validator_status == ValidatorStatus.STALE
    assert by_id["c-live"].validator_status == ValidatorStatus.UNVERIFIED
    assert by_id["c-other"].validator_status == ValidatorStatus.UNVERIFIED

    for claim_id in ("c-open", "c-sent", "c-blank"):
        check = release_claim(session, claim_id, evidence=writer, interaction_id=opened.id, usecase_id="advice")
        assert check.check == "C1-STALE"
        assert check.result == CheckResult.FAIL
    live_claim = by_id["c-live"]
    relabel_claim(
        session, "c-live", live_claim.claim_type, evidence=writer, interaction_id=live.id,
        source_system=live_claim.source_system, source_record=live_claim.source_record,
        validator_status=ValidatorStatus.INFERRED,
    )
    assert release_claim(
        session, "c-live", evidence=writer, interaction_id=live.id, usecase_id="advice",
    ).result == CheckResult.PASS

    report = affected_by_source(session, "house-view")
    assert _ids(report["clients"], "client_id") == {"client-open", "client-sent", "client-rejected", "client-blank"}
    assert "client-live" not in _ids(report["clients"], "client_id")
    assert "client-other" not in _ids(report["clients"], "client_id")
    assert _ids(report["recommendations"]) == {opened.id, sent.id, rejected.id}
    assert _ids(report["communications"]) == {sent.id}
    assert _ids(report["opportunities"]) == {opened.id}
    assert report["opportunities"][0]["validator_status"] == "STALE"
    assert all(row["validator_status"] == "STALE" for row in report["recommendations"])

    quiet = affected_by_source(session, "policy-note")
    assert quiet["clients"] == []
    assert quiet["recommendations"] == []
    assert quiet["communications"] == []
    assert quiet["opportunities"] == []
    verify_chain(session)


def test_release_and_distribution_cannot_bypass_a_stale_view(store):
    session, writer, _ = store
    opened = _journey(session, writer, "open", "client-open", "neutral")
    live = _journey(session, writer, "live", "client-live", "overweight", text="Overweight outlook")
    _supersede(store)

    with pytest.raises(ValueError, match="C1-STALE"):
        relabel_claim(
            session, "c-open", ClaimType.VERIFIED_FACT, evidence=writer, interaction_id=opened.id,
            source_system="house-view", source_record="overweight", validator_status=ValidatorStatus.VERIFIED,
        )
    assert list_claims(session, client_id="client-open")[0].validator_status == ValidatorStatus.STALE

    record_claim(
        session,
        Claim("c-copy", "client-copy", "Neutral outlook", ClaimType.INFERENCE, source_system="house-view", source_record="neutral"),
        evidence=writer, interaction_id="copy",
    )
    copied = release_claim(session, "c-copy", evidence=writer, interaction_id="copy", usecase_id="advice")
    assert copied.check == "C3-SUPERSEDED"
    assert copied.result == CheckResult.FAIL

    for outcome in ("approve", "modify"):
        with pytest.raises(ReviewError, match="superseded source"):
            review(
                session, opened.id, rm="RM", outcome=outcome, packet={}, rendered="note",
                provenance="human", evidence=writer, now=WHEN,
            )
    with pytest.raises(ReviewError, match="human boundary") as refused:
        review(
            session, opened.id, rm="RM", outcome="reject", packet={}, rendered="note",
            provenance="human", evidence=writer, now=WHEN,
        )
    assert "superseded" not in str(refused.value)
    with pytest.raises(ReviewError, match="human boundary") as unrelated:
        review(
            session, live.id, rm="RM", outcome="approve", packet={}, rendered="note",
            provenance="human", evidence=writer, now=WHEN,
        )
    assert "superseded" not in str(unrelated.value)

    denied = send_client_communication(
        session, opened.id, evidence=writer, channel="email", recipient_ref="client-open",
        content="The house view is still neutral.", level="general information", provenance="human",
        actor_id="rm-1", now=WHEN,
    )
    assert denied["sent"] is False
    stale = next(check for check in denied["checks"] if check["check"] == "SOURCE_CURRENT")
    assert stale["result"] == "fail"
    allowed = send_client_communication(
        session, live.id, evidence=writer, channel="email", recipient_ref="client-live",
        content="The house view is overweight.", level="general information", provenance="human",
        actor_id="rm-1", now=WHEN,
    )
    current = next(check for check in allowed["checks"] if check["check"] == "SOURCE_CURRENT")
    assert current["result"] == "pass"
    assert _ids(affected_by_source(session, "house-view")["opportunities"]) == {opened.id}
    assert _ids(affected_by_source(session, "house-view")["communications"]) == set()
    verify_chain(session)


@pytest.mark.parametrize("kwargs,match", [
    ({"source_id": "missing", "version": "neutral", "by": "overweight"}, "C3-UNKNOWN"),
    ({"source_id": "house-view", "version": "missing", "by": "overweight"}, "C3-UNKNOWN"),
    ({"source_id": "house-view", "version": "neutral", "by": "neutral"}, "C3-INVALID"),
    ({"source_id": "house-view", "version": "neutral", "by": "model"}, "C3-INVALID"),
    ({"source_id": " ", "version": "neutral", "by": "overweight"}, "C3-INVALID"),
    ({"source_id": "house-view", "version": "neutral", "by": " "}, "C3-INVALID"),
    ({"source_id": "house-view", "version": "neutral", "by": "overweight", "interaction_id": " "}, "C3-INVALID"),
])
def test_invalid_supersede_writes_nothing(store, kwargs, match):
    session, writer, _ = store
    before = [row.sequence for row in session.exec(select(EvidenceEvent))]
    values = {"evidence": writer, "interaction_id": "supersede"}
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        supersede_source(session, values.pop("source_id"), **values)
    assert [row.sequence for row in session.exec(select(EvidenceEvent))] == before
    assert not any(row.event_type == EventType.SOURCE_SUPERSEDED for row in session.exec(select(EvidenceEvent)))


def test_second_supersede_does_not_revive_or_rewrite(store):
    session, writer, _ = store
    _journey(session, writer, "open", "client-open", "neutral")
    _supersede(store)
    before = session.exec(select(EvidenceEvent)).all()
    with pytest.raises(ValueError, match="C3-IMMUTABLE"):
        supersede_source(
            session, "house-view", version="neutral", by="later", evidence=writer, interaction_id="again",
        )
    assert session.exec(select(EvidenceEvent)).all() == before
    verify_chain(session)


def test_affected_window_fails_closed(store):
    session, writer, _ = store
    opened = _journey(session, writer, "open", "client-open", "neutral")
    _supersede(store)
    included = affected_by_source(session, "house-view", start=WHEN, end=WHEN + timedelta(seconds=1))
    assert _ids(included["clients"], "client_id") == {"client-open"}
    excluded = affected_by_source(session, "house-view", start=WHEN - timedelta(seconds=1), end=WHEN)
    assert excluded["clients"] == []
    assert excluded["recommendations"] == []
    assert excluded["opportunities"] == []
    future = affected_by_source(
        session, "house-view", start=WHEN + timedelta(seconds=1), end=WHEN + timedelta(seconds=2),
    )
    assert future["clients"] == []
    with pytest.raises(ValueError, match="C3-WINDOW"):
        affected_by_source(session, "house-view", start=WHEN)
    with pytest.raises(ValueError, match="C3-WINDOW"):
        affected_by_source(session, "house-view", start=WHEN + timedelta(seconds=1), end=WHEN)
    with pytest.raises(ValueError, match="C3-UNKNOWN"):
        affected_by_source(session, "house")
    assert opened.id in _ids(included["opportunities"])


def test_tampered_source_bytes_fail_closed(store):
    session, writer, _ = store
    _journey(session, writer, "open", "client-open", "neutral")
    _supersede(store)
    event = next(row for row in session.exec(select(EvidenceEvent)) if row.payload.get("action") == "register_source")
    row = session.get(RetainedContent, event.payload["source_hash"])
    row.body = row.body.replace("CIO desk", "changed")
    session.add(row)
    session.flush()
    with pytest.raises(ValueError, match="C3-CONTENT"):
        affected_by_source(session, "house-view")
    with pytest.raises(ValueError, match="C3-CONTENT"):
        supersede_source(
            session, "house-view", version="overweight", by="later", evidence=writer, interaction_id="again",
        )


def test_cli_source_and_affected_source(store, capsys):
    session, writer, path = store
    _journey(session, writer, "open", "client-open", "neutral")
    session.commit()

    assert main(["evidence", "affected", "--db", path, "--model", "1"]) == 2
    assert "--from and --to are required" in capsys.readouterr().err
    assert main([
        "evidence", "affected", "--db", path, "--source", "house-view", "--model", "1",
    ]) == 2
    assert "name one selector" in capsys.readouterr().err
    assert main([
        "evidence", "affected", "--db", path, "--source", "house-view", "--from", WHEN.isoformat(),
    ]) == 2
    assert "both required" in capsys.readouterr().err

    assert main(["source", "list", "--db", path]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {(row["source_id"], row["version"], row["superseded_by"]) for row in listed} == {
        ("house-view", "neutral", ""),
        ("house-view", "overweight", ""),
        ("policy-note", "1", ""),
    }
    assert main([
        "source", "supersede", "--db", path, "--id", "house-view", "--version", "neutral", "--by", "overweight",
    ]) == 0
    superseded = json.loads(capsys.readouterr().out)
    assert superseded["superseded_by"] == "overweight"
    assert superseded["claims"] == ["c-open"]
    assert superseded["recommendations"][0]["validator_status"] == "STALE"

    assert main(["evidence", "affected", "--db", path, "--source", "house-view"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert _ids(report["clients"], "client_id") == {"client-open"}
    assert report["from"] == ""
    assert report["to"] == ""
    assert main(["source", "supersede", "--db", path, "--id", "missing", "--version", "1", "--by", "2"]) == 2
    assert "C3-UNKNOWN" in capsys.readouterr().err
    assert main([
        "source", "register", "--db", path, "--id", "model", "--kind", "house view",
        "--version", "1", "--owner", "CIO desk", "--effective", EFFECTIVE.isoformat(),
    ]) == 2
    assert "C3-INVALID" in capsys.readouterr().err
