"""Governed terms: release cites the version in force, and old work stays listed."""
from datetime import datetime, timezone
import json

import pytest
from sqlmodel import Session, select

from kognita import Claim, ClaimType, GovernedTerm, ValidatorStatus
from kognita.claims import record_claim, relabel_claim, release_claim
from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.definitions import affected_by_definition, check_definition, register_term
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.interaction import open_interaction
from kognita.models import EvidenceEvent, RetainedContent, UseCase
from kognita.retention import RetentionStore
from kognita.review import record_final_decision
from kognita.sources import Source, register_source
from kognita.vocabulary import CheckResult, EventType


EARLIER = datetime(2020, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
T3 = datetime(2025, 1, 1, tzinfo=timezone.utc)
T4 = datetime(2025, 6, 1, tzinfo=timezone.utc)
T5 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _Clock:
    def __init__(self, moment):
        self.now = moment

    def __call__(self):
        return self.now


def _term(
    version="1", superseded_by="", effective=EARLIER, term="AUM",
    ref="catalogue:aum", owner="Finance",
):
    return GovernedTerm(term, ref, owner, version, effective, superseded_by)


def _claim(text, ref=None, claim_id="c1"):
    return Claim(
        claim_id, "client-1", text, ClaimType.INFERENCE,
        source_system="crm", source_record="r1", definition_ref=ref,
        validator_status=ValidatorStatus.VERIFIED,
    )


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "definitions.db"
    engine = make_engine(str(path))
    create_all(engine)
    with Session(engine) as session:
        writer = EvidenceWriter(engine)
        session.add(UseCase(usecase_id="u1", record={"usecase_id": "u1", "approved_sources": ["crm"]}))
        session.flush()
        register_source(
            session, Source("crm", "policy", "r1", "owner", EARLIER),
            evidence=writer, interaction_id="setup",
        )
        register_term(session, _term(), evidence=writer, interaction_id="setup")
        yield session, writer, str(path)


def _release(store, claim, audience="rm"):
    session, writer, _ = store
    record_claim(session, claim, evidence=writer, interaction_id="i1")
    return release_claim(
        session, claim.claim_id, evidence=writer, interaction_id="i1",
        usecase_id="u1", audience=audience,
    )


def _send(session, writer, interaction_id, content):
    digest = RetentionStore().retain_text(
        session, content, kind="client_communication", correlation_id=interaction_id,
    )
    writer.emit(
        session, correlation_id=interaction_id, event_type=EventType.CLIENT_COMMUNICATION,
        payload={
            "lane": "client", "interaction_id": interaction_id, "client_id": "client-1",
            "content_hash": digest, "sent": True,
        },
    )
    return digest


@pytest.mark.parametrize("audience", ["rm", "client"])
def test_current_citation_releases_and_a_missing_one_does_not(store, audience):
    cited = _release(store, _claim("Client AUM is 10", ("AUM", "1"), claim_id=f"ok-{audience}"), audience)
    missing = _release(store, _claim("Client AUM is 10", claim_id=f"no-{audience}"), audience)
    assert cited.result == CheckResult.PASS and cited.check == "C1-LABELLED"
    assert missing.result == CheckResult.FAIL and missing.check == "C12-MISSING"
    session, writer, _ = store
    event = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence.desc())).first()
    assert event.payload["allowed"] is False and event.payload["reason"] == "C12-MISSING"
    assert check_definition(session, _claim("Client AUM is 10", ("AUM", "1"))).check == "C12-CITED"
    verify_chain(session)


@pytest.mark.parametrize("text", [
    "client aum is 10",
    "Client A.U.M is 10",
    "Client A U M is 10",
    "Client A\u200bUM is 10",
    "Client A\u00adUM is 10",
    "Client A\u0301UM is 10",
    "Client \u0410UM is 10",
    "Client \uff21\uff35\uff2d is 10",
])
def test_disguised_terms_cannot_bypass_the_citation(store, text):
    check = _release(store, _claim(text))
    assert check.result == CheckResult.FAIL and check.check == "C12-MISSING"


def test_disguised_term_with_the_current_citation_is_released(store):
    session, _, _ = store
    claim = _claim("Client A\u200bUM is 10", ("aum", "1"))
    assert _release(store, claim).result == CheckResult.PASS
    assert check_definition(session, claim).check == "C12-CITED"


@pytest.mark.parametrize("text", ["BAUM is 10", "the baum chart", "Client balance grew"])
def test_a_longer_word_or_unrelated_text_is_not_blocked(store, text):
    assert _release(store, _claim(text)).result == CheckResult.PASS


def test_term_outside_the_claim_text_does_not_require_a_citation(store):
    claim = Claim(
        "c-calc", "client-1", "Balance grew", ClaimType.INFERENCE,
        source_system="crm", source_record="r1", calculation_method="AUM",
        validator_status=ValidatorStatus.VERIFIED,
    )
    assert _release(store, claim).result == CheckResult.PASS
    session, _, _ = store
    assert check_definition(session, claim).check == "C12-CLEAR"


def test_phrase_spacing_is_not_a_bypass(store):
    session, writer, _ = store
    register_term(
        session, _term(term="net new money", ref="catalogue:nnm"),
        evidence=writer, interaction_id="setup",
    )
    blocked = [
        "Net new money rose",
        "netnewmoney rose",
        "net-new-money rose",
        "Net\nnew   money rose",
    ]
    for index, text in enumerate(blocked):
        check = _release(store, _claim(text, claim_id=f"p{index}"))
        assert check.check == "C12-MISSING"
    cited = _claim("net new money rose", ("net new money", "1"), claim_id="p-ok")
    assert _release(store, cited).result == CheckResult.PASS
    assert _release(store, _claim("network money rose", claim_id="p-word")).result == CheckResult.PASS


def test_longer_word_is_not_the_governed_term(store):
    session, writer, _ = store
    register_term(
        session, _term(term="concentration", ref="catalogue:concentration"),
        evidence=writer, interaction_id="setup",
    )
    assert _release(store, _claim("concentrations rose", claim_id="plural")).result == CheckResult.PASS
    assert _release(store, _claim("concentration rose", claim_id="exact")).check == "C12-MISSING"


def test_superseded_and_not_yet_effective_citations_are_blocked(store):
    session, writer, _ = store
    future = datetime(2099, 1, 1, tzinfo=timezone.utc)
    register_term(session, _term(version="2", effective=future), evidence=writer, interaction_id="setup")
    register_term(session, _term(superseded_by="2"), evidence=writer, interaction_id="setup")
    assert _release(store, _claim("AUM is 10", ("AUM", "1"), claim_id="old")).check == "C12-SUPERSEDED"
    assert _release(store, _claim("AUM is 11", ("AUM", "2"), claim_id="new")).check == "C12-NOT-EFFECTIVE"
    assert _release(store, _claim("AUM is 12", claim_id="none")).check == "C12-MISSING"
    verify_chain(session)


def test_a_later_version_retires_the_older_one_without_a_superseded_flag(store):
    session, writer, _ = store
    register_term(
        session, _term(version="2", effective=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        evidence=writer, interaction_id="setup",
    )
    assert check_definition(session, _claim("AUM is 10", ("AUM", "1"))).check == "C12-SUPERSEDED"
    assert check_definition(session, _claim("AUM is 11", ("AUM", "2"))).check == "C12-CITED"
    assert _release(store, _claim("AUM is 11", ("AUM", "2"), claim_id="current")).result == CheckResult.PASS


def test_release_rechecks_after_the_definition_changes(store):
    session, writer, _ = store
    claim = _claim("AUM is 10", ("AUM", "1"))
    record_claim(session, claim, evidence=writer, interaction_id="i1")
    assert release_claim(
        session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1",
    ).result == CheckResult.PASS
    register_term(
        session, _term(version="2", effective=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        evidence=writer, interaction_id="setup",
    )
    assert release_claim(
        session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1",
    ).check == "C12-SUPERSEDED"


def test_relabel_does_not_bypass_a_missing_definition(store):
    session, writer, _ = store
    record_claim(session, _claim("AUM is 10"), evidence=writer, interaction_id="i1")
    relabel_claim(
        session, "c1", ClaimType.VERIFIED_FACT, evidence=writer, interaction_id="i1",
        validator_status=ValidatorStatus.VERIFIED,
    )
    assert release_claim(
        session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1",
    ).check == "C12-MISSING"


def test_another_terms_current_version_is_not_a_citation(store):
    session, writer, _ = store
    register_term(
        session, _term(term="concentration", ref="catalogue:concentration"),
        evidence=writer, interaction_id="setup",
    )
    wrong_term = _claim("AUM is 10", ("concentration", "1"), claim_id="wrong-term")
    unknown = _claim("AUM is 10", ("AUM", "9"), claim_id="unknown-version")
    assert check_definition(session, wrong_term).check == "C12-MISSING"
    assert check_definition(session, unknown).check == "C12-MISSING"
    assert _release(store, wrong_term).check == "C12-MISSING"
    assert _release(store, unknown).check == "C12-MISSING"


def test_two_governed_terms_need_more_than_one_citation(store):
    session, writer, _ = store
    register_term(
        session, _term(term="concentration", ref="catalogue:concentration"),
        evidence=writer, interaction_id="setup",
    )
    text = "AUM and concentration both rose"
    assert check_definition(session, _claim(text, ("AUM", "1"))).check == "C12-MISSING"
    assert _release(store, _claim(text, ("AUM", "1"), claim_id="both")).check == "C12-MISSING"


def test_ambiguous_current_versions_fail_closed(store):
    session, writer, _ = store
    register_term(
        session, _term(term="VaR", ref="catalogue:var", version="1"),
        evidence=writer, interaction_id="setup",
    )
    register_term(
        session, _term(term="VaR", ref="catalogue:var", version="2"),
        evidence=writer, interaction_id="setup",
    )
    claim = _claim("VaR is 3", ("VaR", "1"), claim_id="var")
    assert check_definition(session, claim).check == "C12-AMBIGUOUS"
    assert check_definition(session, _claim("VaR is 3", ("VaR", "2"), claim_id="var2")).check == "C12-AMBIGUOUS"
    # No figure in the text, so the calculation gate does not answer first.
    assert _release(store, _claim("VaR rose", ("VaR", "1"), claim_id="var3")).check == "C12-AMBIGUOUS"
    with pytest.raises(ValueError, match="C12-AMBIGUOUS"):
        affected_by_definition(session, "VaR")


def test_versions_cannot_be_rewritten_revived_or_spoofed(store):
    session, writer, _ = store
    with pytest.raises(ValueError, match="C12-IMMUTABLE"):
        register_term(session, _term(owner="Other"), evidence=writer, interaction_id="setup")
    register_term(session, _term(superseded_by="2"), evidence=writer, interaction_id="setup")
    with pytest.raises(ValueError, match="C12-IMMUTABLE"):
        register_term(session, _term(), evidence=writer, interaction_id="setup")
    with pytest.raises(ValueError, match="C12-INVALID"):
        register_term(
            session, GovernedTerm("aum", "catalogue:aum", "Finance", "9", EARLIER),
            evidence=writer, interaction_id="setup",
        )


def test_changed_definition_content_fails_closed(store):
    session, writer, _ = store
    record_claim(session, _claim("AUM is 10", ("AUM", "1")), evidence=writer, interaction_id="i1")
    event = next(item for item in session.exec(select(EvidenceEvent)) if item.payload.get("action") == "register_term")
    assert "Finance" not in json.dumps(event.payload)
    body = json.loads(RetentionStore().read(session, event.payload["term_hash"]))
    assert body["owner"] == "Finance"
    assert body["definition_ref"] == "catalogue:aum"
    row = session.get(RetainedContent, event.payload["term_hash"])
    row.body = row.body.replace("Finance", "changed")
    session.add(row)
    session.flush()
    with pytest.raises(ValueError, match="C12-CONTENT"):
        check_definition(session, _claim("AUM is 10", ("AUM", "1")))
    with pytest.raises(ValueError, match="C12-CONTENT"):
        release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1")


@pytest.mark.parametrize("changes", [
    {"term": " "},
    {"term": "..."},
    {"owner": ""},
    {"version": ""},
    {"version": "1 2"},
    {"ref": ""},
    {"effective": datetime(2020, 1, 1)},
    {"superseded_by": "1"},
])
def test_invalid_terms_are_refused(changes):
    with pytest.raises(ValueError, match="C12-INVALID"):
        _term(**changes)


def test_definition_checks_are_retained_by_hash(store):
    session, writer, _ = store
    claim = _claim("Client AUM is 10", ("AUM", "1"))
    assert _release(store, claim).result == CheckResult.PASS
    checked = next(item for item in session.exec(select(EvidenceEvent)) if item.payload.get("action") == "check_definition")
    assert "Client AUM is 10" not in json.dumps(checked.payload)
    pinned = json.loads(RetentionStore().read(session, checked.payload["definition_hash"]))
    assert pinned["reason"] == "C12-CITED"
    assert pinned["current_version"] == "1"
    assert pinned["cited"] == ["AUM", "1"]
    verify_chain(session)


def test_affected_lists_only_previous_versions(tmp_path, capsys):
    path = tmp_path / "affected.db"
    engine = make_engine(str(path))
    create_all(engine)
    clock = _Clock(T1)
    writer = EvidenceWriter(engine, clock=clock)
    with Session(engine) as session:
        session.add(UseCase(usecase_id="u1", record={"usecase_id": "u1", "approved_sources": ["crm"]}))
        session.flush()
        register_source(
            session, Source("crm", "policy", "r1", "owner", T1),
            evidence=writer, interaction_id="setup",
        )
        register_term(session, _term(effective=T1), evidence=writer, interaction_id="setup")
        clock.now = T2
        record_claim(session, _claim("AUM was 10", ("AUM", "1"), claim_id="c-old"), evidence=writer, interaction_id="i-old")
        record_claim(session, _claim("AUM was 9", claim_id="c-bare"), evidence=writer, interaction_id="i-bare")
        open_interaction(session, client_id="client-1", evidence=writer, interaction_id="i-rec", now=T2)
        old_hash = RetentionStore().retain_text(session, "AUM was 10", kind="recommendation", correlation_id="i-rec", now=T2)
        record_final_decision(
            session, "i-rec", outcome="approve", evidence=writer, actor_id="rm-1",
            citation="RM review: final decision", rendered_hash=old_hash, now=T2,
        )
        open_interaction(session, client_id="client-1", evidence=writer, interaction_id="i-plain", now=T2)
        plain_hash = RetentionStore().retain_text(session, "Hello portfolio", kind="recommendation", correlation_id="i-plain", now=T2)
        record_final_decision(
            session, "i-plain", outcome="approve", evidence=writer, actor_id="rm-1",
            citation="RM review: final decision", rendered_hash=plain_hash, now=T2,
        )
        old_communication = _send(session, writer, "i-old", "Please review AUM")
        _send(session, writer, "i-plain", "Hello there")
        clock.now = T3
        register_term(session, _term(version="2", effective=T3), evidence=writer, interaction_id="setup")
        register_term(session, _term(effective=T1, superseded_by="2"), evidence=writer, interaction_id="setup")
        clock.now = T4
        record_claim(session, _claim("AUM is 12", ("AUM", "2"), claim_id="c-new"), evidence=writer, interaction_id="i-new")
        record_claim(session, _claim("AUM still 10", ("AUM", "1"), claim_id="c-stale"), evidence=writer, interaction_id="i-stale")
        open_interaction(session, client_id="client-1", evidence=writer, interaction_id="i-rec-new", now=T4)
        new_hash = RetentionStore().retain_text(session, "AUM is 12", kind="recommendation", correlation_id="i-rec-new", now=T4)
        record_final_decision(
            session, "i-rec-new", outcome="approve", evidence=writer, actor_id="rm-1",
            citation="RM review: final decision", rendered_hash=new_hash, now=T4,
        )
        _send(session, writer, "i-new", "AUM is 12 now")
        report = affected_by_definition(session, "AUM", as_of=T5)
        assert report["term"] == "AUM" and report["current_version"] == "2"
        assert [item["claim_id"] for item in report["claims"]] == ["c-bare", "c-old", "c-stale"]
        assert {item["version"] for item in report["claims"]} == {"1"}
        assert [item["rendered_hash"] for item in report["recommendations"]] == [old_hash]
        assert [item["content_hash"] for item in report["communications"]] == [old_communication]
        early = affected_by_definition(session, "aum", start=T1, end=T2, as_of=T5)
        assert early["claims"] == [] and early["recommendations"] == [] and early["communications"] == []
        during = affected_by_definition(session, "catalogue:aum", start=T2, end=T3, as_of=T5)
        assert [item["claim_id"] for item in during["claims"]] == ["c-bare", "c-old"]
        assert [item["rendered_hash"] for item in during["recommendations"]] == [old_hash]
        later = affected_by_definition(session, "AUM", start=T4, end=T5, as_of=T5)
        assert [item["claim_id"] for item in later["claims"]] == ["c-stale"]
        assert later["recommendations"] == [] and later["communications"] == []
        assert release_claim(
            session, "c-stale", evidence=writer, interaction_id="i-stale", usecase_id="u1",
        ).check == "C12-SUPERSEDED"
        assert release_claim(
            session, "c-bare", evidence=writer, interaction_id="i-bare", usecase_id="u1",
        ).check == "C12-MISSING"
        assert release_claim(
            session, "c-new", evidence=writer, interaction_id="i-new", usecase_id="u1",
        ).result == CheckResult.PASS
        with pytest.raises(ValueError, match="C12-UNKNOWN"):
            affected_by_definition(session, "nope")
        with pytest.raises(ValueError, match="C12-INVALID"):
            affected_by_definition(session, "AUM", start=T3, end=T2)
        verify_chain(session)
        session.commit()

    assert main(["evidence", "affected", "--db", str(path), "--definition", "AUM"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [item["claim_id"] for item in listed["claims"]] == ["c-bare", "c-old", "c-stale"]
    assert len(listed["recommendations"]) == 1
    assert len(listed["communications"]) == 1


def test_cli_register_list_and_affected(store, capsys):
    session, writer, path = store
    record_claim(session, _claim("AUM was 10", ("AUM", "1")), evidence=writer, interaction_id="i1")
    session.commit()
    assert main(["definition", "list", "--db", path]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["term"] == "AUM" and listed[0]["version"] == "1"
    assert main(["evidence", "affected", "--db", path, "--definition", "AUM"]) == 0
    assert json.loads(capsys.readouterr().out)["claims"] == []
    assert main([
        "definition", "register", "--db", path, "--term", "AUM",
        "--definition-ref", "catalogue:aum", "--owner", "Finance", "--version", "2",
        "--effective", "2024-01-01T00:00:00+00:00",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == "2"
    assert main([
        "definition", "register", "--db", path, "--term", "AUM",
        "--definition-ref", "catalogue:aum", "--owner", "Finance", "--version", "1",
        "--effective", "2020-01-01T00:00:00+00:00", "--superseded-by", "2",
    ]) == 0
    capsys.readouterr()
    assert main(["evidence", "affected", "--db", path, "--definition", "catalogue:aum"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [item["claim_id"] for item in report["claims"]] == ["c1"]
    assert report["current_version"] == "2"
    assert main([
        "evidence", "affected", "--db", path, "--definition", "AUM",
        "--from", "2020-01-01T00:00:00+00:00", "--to", "2021-01-01T00:00:00+00:00",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["claims"] == []
    assert main([
        "evidence", "affected", "--db", path, "--definition", "AUM",
        "--from", "2026-01-01T00:00:00+00:00", "--to", "2027-01-01T00:00:00+00:00",
    ]) == 0
    assert [item["claim_id"] for item in json.loads(capsys.readouterr().out)["claims"]] == ["c1"]
    assert main(["evidence", "affected", "--db", path, "--definition", "nope"]) == 2
    assert "C12-UNKNOWN" in capsys.readouterr().err
    assert main(["evidence", "affected", "--db", path, "--definition", "AUM", "--model", "1"]) == 2
    assert main(["evidence", "affected", "--db", path, "--model", "1"]) == 2
    assert main([
        "definition", "register", "--db", path, "--term", "AUM",
        "--definition-ref", "catalogue:aum", "--owner", "Finance", "--version", "3",
        "--effective", "yesterday",
    ]) == 2


def test_governed_term_is_exported():
    import kognita
    from kognita import definitions

    assert kognita.GovernedTerm is GovernedTerm
    assert "GovernedTerm" in kognita.__all__
    assert set(definitions.__all__) >= {
        "GovernedTerm", "register_term", "check_definition", "affected_by_definition",
    }
