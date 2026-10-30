"""Source approval and lifecycle gates resist release bypasses."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest
from sqlmodel import Session, select

from kognita import Claim, ClaimType, Source, SourceRegister, ValidatorStatus
from kognita.claims import record_claim, release_claim
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.models import UseCase, EvidenceEvent, RetainedContent
from kognita.retention import RetentionStore
from kognita.sources import register_source, check_grounding
from kognita.vocabulary import CheckResult

NOW = datetime.now(timezone.utc)


@pytest.fixture
def store(tmp_path):
    engine = make_engine(str(tmp_path / "sources.db"))
    create_all(engine)
    with Session(engine) as session:
        writer = EvidenceWriter(engine)
        for name, approved in (("approved", ["research"]), ("other", [])):
            session.add(UseCase(usecase_id=name, effective_from=NOW - timedelta(days=10), record={"usecase_id": name, "approved_sources": approved}))
        session.flush()
        yield session, writer


def _source(**changes):
    return replace(Source("research", "CIO research", "1", "Research owner", NOW - timedelta(days=1)), **changes)


@pytest.mark.parametrize("audience", ["rm", "client"])
@pytest.mark.parametrize("change,usecase,system,version,reason", [
    ({}, "approved", "research", "1", "C1-LABELLED"),
    ({}, "other", "research", "1", "C3-UNAPPROVED"),
    ({}, "missing", "research", "1", "C3-USECASE"),
    ({}, "", "research", "1", "C3-USECASE"),
    ({}, "approved", "model", "1", "C3-MODEL"),
    ({}, "approved", " MODEL ", "1", "C3-MODEL"),
    ({}, "approved", "", "1", "C3-MODEL"),
    ({}, "approved", "research", "unknown", "C3-UNAPPROVED"),
    ({"superseded_by": "2"}, "approved", "research", "1", "C3-SUPERSEDED"),
    ({"expiry": NOW}, "approved", "research", "1", "C3-EXPIRED"),
    ({"effective": NOW + timedelta(days=1)}, "approved", "research", "1", "C3-NOT-EFFECTIVE"),
])
def test_release(store, audience, change, usecase, system, version, reason):
    session, writer = store
    register_source(session, _source(**change), evidence=writer, interaction_id="setup")
    claim = Claim("c", "client", "Investment outlook", ClaimType.INFERENCE,
                  source_system=system, source_record=version,
                  validator_status=ValidatorStatus.VERIFIED)
    record_claim(session, claim, evidence=writer, interaction_id="i")
    check = release_claim(session, "c", usecase_id=usecase, audience=audience, evidence=writer, interaction_id="i")
    assert check.check == reason
    assert check.result == (CheckResult.PASS if reason == "C1-LABELLED" else CheckResult.FAIL)
    event = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence.desc())).first()
    assert event.payload["allowed"] == (reason == "C1-LABELLED")
    verify_chain(session)


def test_lifecycle_and_evidence(store):
    session, writer = store
    source = _source()
    claim = Claim("c", "client", "Outlook", ClaimType.INFERENCE, source_system="research", source_record="1")
    register_source(session, source, evidence=writer, interaction_id="i")
    assert SourceRegister((source,)).get("research", "1") == source
    assert check_grounding(session, claim, usecase_id="approved", as_of=NOW).result == CheckResult.PASS
    assert check_grounding(session, claim, usecase_id="approved", as_of=source.effective - timedelta(seconds=1)).result == CheckResult.FAIL
    register_source(session, replace(source, expiry=NOW), evidence=writer, interaction_id="i")
    assert check_grounding(session, claim, usecase_id="approved", as_of=NOW - timedelta(seconds=1)).result == CheckResult.PASS
    assert check_grounding(session, claim, usecase_id="approved", as_of=NOW).check == "C3-EXPIRED"
    with pytest.raises(ValueError, match="C3-IMMUTABLE"):
        register_source(session, source, evidence=writer, interaction_id="i")
    register_source(session, replace(source, expiry=NOW, superseded_by="2"), evidence=writer, interaction_id="i")
    with pytest.raises(ValueError, match="C3-IMMUTABLE"):
        register_source(session, replace(source, expiry=NOW), evidence=writer, interaction_id="i")
    event = session.exec(select(EvidenceEvent)).first()
    assert "Research owner" not in json.dumps(event.payload)
    digest = event.payload["source_hash"]
    assert json.loads(RetentionStore().read(session, digest))["owner"] == source.owner
    row = session.get(RetainedContent, digest)
    row.body = row.body.replace("Research owner", "changed")
    session.add(row)
    session.flush()
    with pytest.raises(ValueError, match="C3-CONTENT"):
        check_grounding(session, claim, usecase_id="approved")


@pytest.mark.parametrize("changes", [{"kind": "unknown"}, {"source_id": "model"}, {"owner": " "}, {"version": ""}, {"effective": datetime(2020, 1, 1)}, {"expiry": NOW - timedelta(days=2)}])
def test_invalid_sources(changes):
    with pytest.raises(ValueError, match="C3-INVALID"):
        _source(**changes)


@pytest.mark.parametrize("permitted,cited,allowed", [
    (["research"], "research", True),
    (["research"], "other", False),
    (["research"], None, False),
    ([], "research", False),
    (["model"], "model", False),
])
def test_policy_yaml_sources(permitted, cited, allowed):
    from kognita.envelope import Envelope, RuleContext
    from kognita.models import Policy
    from kognita.policy import parse_yaml, parse_rule
    from kognita.rules import content_policy
    text = """id: approved.sources
regime: INTERNAL
rule_type: CONTENT_POLICY
citation: Approved sources
effective_from: 2026-01-01T00:00:00+00:00
"""
    text += "rule:\n  sources: " + json.dumps(permitted) + "\n"
    document, findings = parse_rule(parse_yaml(text), source="sources.yaml")
    assert document is not None and findings == []
    policy = Policy(regime="INTERNAL", rule_type="CONTENT_POLICY", rule=document.rule,
                    citation="Approved sources")
    context = RuleContext(Envelope("rm", "approved", "release"), attributes={"content": "Outlook", "source_system": cited})
    checks = content_policy(policy, context)
    assert len(checks) == 1
    assert checks[0].result == (CheckResult.PASS if allowed else CheckResult.FAIL)


def test_release_rechecks_lifecycle_and_pins_grounding(store):
    session, writer = store
    source = _source()
    register_source(session, source, evidence=writer, interaction_id="setup")
    claim = Claim("c", "client", "Outlook", ClaimType.INFERENCE,
                  source_system="research", source_record="1",
                  validator_status=ValidatorStatus.VERIFIED)
    record_claim(session, claim, evidence=writer, interaction_id="i")
    assert release_claim(session, "c", usecase_id="approved", evidence=writer, interaction_id="i").result == CheckResult.PASS
    grounding = next(e for e in session.exec(select(EvidenceEvent)) if e.payload.get("action") == "check_grounding")
    pinned = json.loads(RetentionStore().read(session, grounding.payload["grounding_hash"]))
    assert pinned["source"]["version"] == "1"
    assert pinned["approved_sources"] == ["research"]
    assert pinned["usecase_hash"]
    register_source(session, replace(source, superseded_by="2"), evidence=writer, interaction_id="update")
    assert release_claim(session, "c", usecase_id="approved", evidence=writer, interaction_id="i").check == "C1-STALE"
    assert json.loads(RetentionStore().read(session, grounding.payload["grounding_hash"])) == pinned
    verify_chain(session)


@pytest.mark.parametrize("audience", ["rm", "client"])
def test_supersession_propagates_and_cannot_be_revived(store, audience):
    from kognita import supersede_source, affected_by_source, ValidatorStatus
    from kognita.claims import list_claims, relabel_claim
    from kognita.vocabulary import EventType
    session, writer = store
    register_source(session, _source(), evidence=writer, interaction_id="setup")
    record_claim(session, Claim("c", "client", "Outlook", ClaimType.INFERENCE,
                 source_system="research", source_record="1", validator_status=ValidatorStatus.VERIFIED), evidence=writer, interaction_id="journey")
    record_claim(session, Claim("other", "other-client", "Other", ClaimType.INFERENCE,
                 source_system="other", source_record="1"), evidence=writer, interaction_id="other")
    assert release_claim(session, "c", usecase_id="approved", audience=audience, evidence=writer, interaction_id="journey").result == CheckResult.PASS
    supersede_source(session, "research", version="1", by="2", evidence=writer, interaction_id="change")
    assert list_claims(session)[0].validator_status == ValidatorStatus.STALE
    assert list_claims(session)[1].validator_status != ValidatorStatus.STALE
    with pytest.raises(ValueError, match="C1-STALE"):
        relabel_claim(session, "c", ClaimType.INFERENCE, validator_status=ValidatorStatus.VERIFIED, evidence=writer, interaction_id="journey")
    assert release_claim(session, "c", usecase_id="approved", audience=audience, evidence=writer, interaction_id="journey").result == CheckResult.FAIL
    report = affected_by_source(session, "research")
    assert {row["client_id"] for row in report["clients"]} == {"client"}
    assert any(e.event_type == EventType.SOURCE_SUPERSEDED for e in session.exec(select(EvidenceEvent)))
    verify_chain(session)


@pytest.mark.parametrize("source,successor", [("missing", "2"), ("research", ""), ("research", "1")])
def test_supersession_invalid_requests(store, source, successor):
    from kognita import supersede_source
    session, writer = store
    register_source(session, _source(), evidence=writer, interaction_id="setup")
    with pytest.raises(ValueError, match="C3-"):
        supersede_source(session, source, version="1", by=successor, evidence=writer, interaction_id="change")


def test_source_cli(tmp_path, capsys):
    from kognita.cli import main
    path = str(tmp_path / "cli.db")
    assert main(["source", "register", "--source", "research", "--version", "1", "--kind", "house view", "--owner", "Research", "--effective", "2020-01-01T00:00:00+00:00", "--db", path]) == 0
    assert main(["source", "supersede", "--source", "research", "--version", "1", "--superseded-by", "2", "--db", path]) == 0
    assert main(["source", "list", "--db", path]) == 0
    assert '"superseded_by": "2"' in capsys.readouterr().out
    assert main(["evidence", "affected", "--source", "research", "--db", path]) == 0
    assert main(["evidence", "affected", "--source", "missing", "--db", path]) == 2
    assert main(["evidence", "affected", "--model", "x", "--db", path]) == 2


def test_supersession_revival_and_unrelated_claim(store):
    from kognita import supersede_source, affected_by_source, ValidatorStatus
    from kognita.claims import list_claims, relabel_claim
    session, writer = store
    source = _source()
    register_source(session, source, evidence=writer, interaction_id="setup")
    for ident, system in (("dependent", "research"), ("unrelated", "other")):
        record_claim(session, Claim(ident, "client", "Outlook", ClaimType.INFERENCE,
                                   source_system=system, source_record="1",
                                   validator_status=ValidatorStatus.VERIFIED if system == "research" else ValidatorStatus.UNVERIFIED),
                     evidence=writer, interaction_id="journey")
    assert release_claim(session, "dependent", usecase_id="approved", evidence=writer, interaction_id="journey").result == CheckResult.PASS
    supersede_source(session, "research", version="1", by="2", evidence=writer, interaction_id="change")
    statuses = {c.claim_id: c.validator_status for c in list_claims(session)}
    assert statuses["dependent"] == ValidatorStatus.STALE
    assert statuses["unrelated"] == ValidatorStatus.UNVERIFIED
    assert {row["client_id"] for row in affected_by_source(session, "research")["clients"]} == {"client"}
    with pytest.raises(ValueError, match="C1-STALE"):
        relabel_claim(session, "dependent", ClaimType.INFERENCE, validator_status=ValidatorStatus.VERIFIED,
                      evidence=writer, interaction_id="journey")
    assert release_claim(session, "dependent", usecase_id="approved", evidence=writer, interaction_id="journey").result == CheckResult.FAIL
    with pytest.raises(ValueError, match="C3-IMMUTABLE"):
        register_source(session, source, evidence=writer, interaction_id="revive")
    with pytest.raises(ValueError, match="C3-UNKNOWN"):
        supersede_source(session, "unknown", version="1", by="2", evidence=writer, interaction_id="change")
    with pytest.raises(ValueError, match="C3-INVALID"):
        supersede_source(session, "research", version="1", by="1", evidence=writer, interaction_id="change")
    assert any(e.event_type.value == "SOURCE_SUPERSEDED" for e in session.exec(select(EvidenceEvent)))
    verify_chain(session)


def test_affected_records_follow_journey_not_client(store):
    from kognita import affected_by_source
    from kognita.interaction import open_interaction, attach
    from kognita.vocabulary import EventType
    session, writer = store
    register_source(session, _source(), evidence=writer, interaction_id="setup")
    open_interaction(session, client_id="client", interaction_id="journey", evidence=writer)
    attach(session, "journey", correlation_id="analysis", evidence=writer)
    record_claim(session, Claim("c", "client", "Outlook", ClaimType.INFERENCE,
                 source_system="research", source_record="1"), evidence=writer, interaction_id="analysis")
    for journey in ("journey", "unrelated"):
        writer.emit(session, correlation_id=journey, event_type=EventType.ORIGINATION,
                    payload={"interaction_id": journey, "client_id": "client", "outcome": "recommend"})
        writer.emit(session, correlation_id=journey, event_type=EventType.CLIENT_COMMUNICATION,
                    payload={"interaction_id": journey, "client_id": "client", "sent": True})
    from kognita import supersede_source
    supersede_source(session, "research", version="1", by="2", evidence=writer, interaction_id="change")
    report = affected_by_source(session, "research")
    assert len(report["recommendations"]) == len(report["communications"]) == 1
    assert report["opportunities"] == []
    writer.emit(session, correlation_id="journey", event_type=EventType.FINAL_DECISION,
                payload={"interaction_id": "journey", "outcome": "reject"})
    assert affected_by_source(session, "research")["opportunities"] == []
    with pytest.raises(ValueError, match="C3-WINDOW"):
        affected_by_source(session, "research", start=NOW, end=NOW)
