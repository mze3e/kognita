"""Source approval and lifecycle gates resist release bypasses."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest
from sqlmodel import Session, select

from kognita import Claim, ClaimType, Source, SourceRegister
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
                  source_system=system, source_record=version)
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
                  source_system="research", source_record="1")
    record_claim(session, claim, evidence=writer, interaction_id="i")
    assert release_claim(session, "c", usecase_id="approved", evidence=writer, interaction_id="i").result == CheckResult.PASS
    grounding = next(e for e in session.exec(select(EvidenceEvent)) if e.payload.get("action") == "check_grounding")
    pinned = json.loads(RetentionStore().read(session, grounding.payload["grounding_hash"]))
    assert pinned["source"]["version"] == "1"
    assert pinned["approved_sources"] == ["research"]
    assert pinned["usecase_hash"]
    register_source(session, replace(source, superseded_by="2"), evidence=writer, interaction_id="update")
    assert release_claim(session, "c", usecase_id="approved", evidence=writer, interaction_id="i").check == "C3-SUPERSEDED"
    assert json.loads(RetentionStore().read(session, grounding.payload["grounding_hash"])) == pinned
    verify_chain(session)
