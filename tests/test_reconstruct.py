"""Reconstruction report: ten questions, from evidence, with pin findings.

Roadmap 0.3 item 8. Questions answered by 0.4 stay ``not recorded``.
A tampered policy, retrieved item, or retained prompt is a finding.
Erasure keeps the hash and is not a mismatch.
"""
from __future__ import annotations

import contextlib
import json
from datetime import timedelta
from io import StringIO

import pytest
from sqlalchemy import text
from sqlmodel import Session, select

from fixtures import demo_pack as dp

from kognita.canonical import hash_text
from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.egress import EgressGuard
from kognita.envelope import Envelope
from kognita.evidence import EvidenceWriter
from kognita.exceptions import ReplayMismatch
from kognita.governance import decide, load_snapshot, record
from kognita.models import EvidenceEvent, Policy
from kognita.registry import register
from kognita.replay import replay_decision
from kognita.reconstruct import QUESTIONS, reconstruct, render_reconstruction
from kognita.retention import RetentionStore
from kognita.retrieval import index_item, item_content_hash, retrieve
from kognita.tools import ToolRegistry
from kognita.vocabulary import ActorType, Classification, EventType
from kognita.testing.harness import FROZEN_NOW, Harness

PROMPT = "Summarise the eligibility note."
ANSWER = "Eligible under the recorded checks."
ITEM_BODY = "cross-site eligibility for restricted datasets"
NOT_RECORDED = (
    "Why this client?",
    "Why this insight or product?",
    "What did the RM see and change?",
    "Who made the final decision?",
    "What was communicated to the client?",
)


def _seed(session):
    dp.seed_policies(session, now=FROZEN_NOW)
    register(session, name="eligibility-assistant", owner_exec="Head of Data Governance")
    register(session, name="dossier-agent", owner_exec="Head of Research Ops")


@pytest.fixture
def harness():
    return Harness(pack=dp.DemoPack(), purposes=dp.PURPOSES, seed=_seed)


def _envelope() -> Envelope:
    return Envelope(
        principal="test-principal",
        purpose="COLLABORATION",
        tool="get_subject_profile",
        actor_location="SG",
        agent_name="dossier-agent",
        subject_type="subject",
        subject_id="2",
        roles=["OWNER"],
        scopes={"site": "SG"},
    )


def _answer(report: dict, question: str):
    found = [item for item in report["questions"] if item["question"] == question]
    assert len(found) == 1
    return found[0]["answer"]


def _section(document: str, question: str) -> str:
    token = f". {question}"
    start = document.index(token)
    body_start = document.index("\n", start) + 1
    later = []
    for other in QUESTIONS:
        if other == question:
            continue
        pos = document.find(f". {other}", body_start)
        if pos != -1:
            later.append(pos)
    end = min(later) if later else len(document)
    end = document.rfind("\n", body_start, end)
    return document[body_start:end].strip()


def _record_interaction(harness, session):
    """One correlation id with a decision, a retrieval, and a model call."""
    envelope = _envelope()
    evaluation = harness.decide_and_record(envelope, session)
    item = index_item(
        session,
        title="cross-site eligibility",
        body=ITEM_BODY,
        embedder=harness.embedder,
        zones=["SG"],
        classification=Classification.C1,
    )
    hits = retrieve(
        session,
        ITEM_BODY,
        zone="SG",
        embedder=harness.embedder,
        evidence=harness.evidence,
        correlation_id=evaluation.request_id,
        actor_id="dossier-agent",
        actor_type=ActorType.AGENT,
        use_case=envelope.purpose,
    )
    assert hits
    EgressGuard(evidence=harness.evidence).send(
        PROMPT,
        lambda _text: ANSWER,
        classification=Classification.C1,
        destination="api.openai.com",
        destination_is_local=False,
        session=session,
        correlation_id=evaluation.request_id,
        actor_id="dossier-agent",
        actor_type=ActorType.AGENT,
        provider="api.openai.com",
        model="gateway-model",
        model_version="gateway-model-2026-06-01",
        prompt_template_version="template-3",
        use_case=envelope.purpose,
    )
    return evaluation, item


def test_known_interaction_answers_every_question_from_evidence(harness):
    with harness.session() as session:
        evaluation, item = _record_interaction(harness, session)
        session.commit()
        report = reconstruct(session, evaluation.request_id)
        document = render_reconstruction(report)

        assert [item["question"] for item in report["questions"]] == list(QUESTIONS)
        positions = [document.index(question) for question in QUESTIONS]
        assert positions == sorted(positions)

        for question in NOT_RECORDED:
            assert _answer(report, question) == "not recorded"
            assert _section(document, question) == "not recorded"
        for question in QUESTIONS:
            if question in NOT_RECORDED:
                continue
            assert _answer(report, question) != "not recorded"
            assert isinstance(_answer(report, question), dict)

        decision = session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.correlation_id == evaluation.request_id,
                EvidenceEvent.event_type == EventType.POLICY_DECISION,
            )
        ).one()
        recorded_checks = [
            {
                "check": check["check"],
                "regime": check["regime"],
                "result": check["result"],
                "citation": check["citation"],
                "policy_id": check.get("policy_id"),
                "policy_hash": check.get("policy_hash"),
            }
            for check in decision.payload["checks"]
        ]
        retrieval = session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.correlation_id == evaluation.request_id,
                EvidenceEvent.event_type == EventType.RETRIEVAL,
            )
        ).one()
        model_call = session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.correlation_id == evaluation.request_id,
                EvidenceEvent.event_type == EventType.MODEL_CALL,
            )
        ).one()

        info = _answer(report, "What information did the AI use?")
        recorded_item = retrieval.payload["items"][0]
        reported_item = info["retrievals"][0]["items"][0]
        assert reported_item["id"] == recorded_item["id"] == item.id
        assert reported_item["content_hash"] == recorded_item["content_hash"]
        assert reported_item["content_hash"] == item_content_hash(item)
        assert reported_item["embedding_model"] == recorded_item["embedding_model"]
        assert reported_item["embedding_model"] == item.embedding_model
        assert reported_item["content_status"] == "retained"
        assert ITEM_BODY in reported_item["body"]

        produced = _answer(report, "What model or agent produced it?")
        assert produced["agents"][0]["agent_name"] == "dossier-agent"
        assert "dossier-agent" in produced["agents"][0]["registry_citation"]
        assert produced["models"][0]["provider"] == model_call.payload["provider"]
        assert produced["models"][0]["model"] == "gateway-model"
        assert produced["models"][0]["model_version"] == "gateway-model-2026-06-01"
        assert produced["models"][0]["prompt_template_version"] == "template-3"
        assert produced["models"][0]["prompt_hash"] == model_call.payload["prompt_hash"]
        assert produced["models"][0]["prompt_hash"] == hash_text(PROMPT)
        assert produced["models"][0]["prompt_status"] == "retained"
        assert produced["models"][0]["prompt"] == PROMPT
        assert produced["models"][0]["response_hash"] == hash_text(ANSWER)
        assert produced["models"][0]["response_status"] == "retained"

        authority = _answer(report, "What was the agent authorized to do?")
        assert authority["decisions"][0]["roles"] == ["OWNER"]
        assert authority["decisions"][0]["scopes"] == {"site": "SG"}
        assert authority["decisions"][0]["purpose"] == "COLLABORATION"
        assert authority["decisions"][0]["tool"] == "get_subject_profile"
        assert authority["decisions"][0]["checks"] == recorded_checks

        controls = _answer(report, "What suitability or policy controls ran?")
        assert controls["decisions"][0]["checks"] == recorded_checks
        assert any(
            check["policy_id"] is not None
            and check["policy_hash"]
            and check["citation"]
            and check["regime"]
            for check in controls["decisions"][0]["checks"]
        )

        reproduce = _answer(report, "Can the bank reproduce that evidence later?")
        assert report["chain"]["intact"] is True
        assert report["findings"] == []
        assert reproduce["chain_intact"] is True
        assert reproduce["events_checked"] >= 1
        assert any(
            pin["kind"] == "policy" and pin["status"] == "matches" for pin in reproduce["pins"]
        )
        assert any(
            pin["kind"] == "retrieved_item"
            and pin["status"] == "matches"
            and pin["embedding_model_status"] == "matches"
            for pin in reproduce["pins"]
        )
        assert any(
            pin["kind"] == "prompt" and pin["status"] == "retained" for pin in reproduce["pins"]
        )
        assert any(
            pin["kind"] == "response"
            and pin["event_type"] == "EGRESS"
            and pin["status"] == "retained"
            and pin["content_hash"] == model_call.payload["response_hash"]
            for pin in reproduce["pins"]
        )
        assert "A pinned hash does not match" not in document
        assert recorded_item["content_hash"] in document
        assert "gateway-model" in document
        assert "OWNER" in document


def test_reconstruct_does_not_rerun_classifier_model_or_tools(harness, monkeypatch):
    with harness.session() as session:
        evaluation, _item = _record_interaction(harness, session)
        session.commit()

        def _boom(*_args, **_kwargs):
            raise AssertionError("reconstruct re-ran a decision, classifier, or tool")

        monkeypatch.setattr("kognita.governance.decide", _boom)
        monkeypatch.setattr("kognita.classify.classifier_record", _boom)
        monkeypatch.setattr("kognita.replay.replay_decision", _boom)
        monkeypatch.setattr("kognita.tools.run_governed", _boom)
        monkeypatch.setattr("kognita.retrieval.retrieve", _boom)
        report = reconstruct(session, evaluation.request_id)
    assert len(report["questions"]) == 10


def test_editing_a_policy_row_is_a_finding_for_replay_and_reconstruct(harness):
    with harness.session() as session:
        evaluation, _item = _record_interaction(harness, session)
        policy_id = next(check.policy_id for check in evaluation.checks if check.policy_hash)
        session.commit()
        session.execute(
            text("UPDATE policies SET citation = :citation WHERE id = :id"),
            {"citation": "tampered", "id": policy_id},
        )
        session.commit()
        session.expire_all()
        report = reconstruct(session, evaluation.request_id)
        controls = _answer(report, "What suitability or policy controls ran?")
        assert any(
            check["policy_id"] == policy_id for check in controls["decisions"][0]["checks"]
        )
        assert any("edited in place" in finding["message"] for finding in report["findings"])
        assert "edited in place" in render_reconstruction(report)
        assert len(report["questions"]) == len(QUESTIONS)
        with pytest.raises(ReplayMismatch, match="edited in place"):
            replay_decision(session, evaluation.request_id)


def test_editing_a_retrieved_item_is_a_finding_for_replay_and_reconstruct(harness):
    with harness.session() as session:
        evaluation, item = _record_interaction(harness, session)
        session.commit()
        session.execute(
            text("UPDATE knowledge_items SET body = :body WHERE id = :id"),
            {"body": "replaced after retrieval", "id": item.id},
        )
        session.commit()
        session.expire_all()
        report = reconstruct(session, evaluation.request_id)
        info = _answer(report, "What information did the AI use?")
        assert info["retrievals"][0]["items"][0]["id"] == item.id
        assert any(
            "retrieved item" in finding["message"] and "content hash" in finding["message"]
            for finding in report["findings"]
        )
        assert len(report["questions"]) == len(QUESTIONS)
        with pytest.raises(ReplayMismatch, match="retrieved item"):
            replay_decision(session, evaluation.request_id)


def test_editing_a_stored_prompt_is_a_finding_for_replay_and_reconstruct(harness):
    with harness.session() as session:
        evaluation, _item = _record_interaction(harness, session)
        model_call = session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.correlation_id == evaluation.request_id,
                EvidenceEvent.event_type == EventType.MODEL_CALL,
            )
        ).one()
        prompt_hash = model_call.payload["prompt_hash"]
        session.commit()
        session.execute(
            text(
                "UPDATE retained_content SET body = :body WHERE content_hash = :content_hash"
            ),
            {"body": "tampered prompt", "content_hash": prompt_hash},
        )
        session.commit()
        session.expire_all()
        report = reconstruct(session, evaluation.request_id)
        produced = _answer(report, "What model or agent produced it?")
        assert produced["models"][0]["prompt_hash"] == prompt_hash
        assert produced["models"][0]["prompt_status"] == "mismatch"
        assert any(
            "does not match its hash" in finding["message"] for finding in report["findings"]
        )
        assert len(report["questions"]) == len(QUESTIONS)
        with pytest.raises(ReplayMismatch, match="does not match its hash"):
            replay_decision(session, evaluation.request_id)


def test_chain_break_is_a_finding_and_the_report_continues(harness):
    with harness.session() as session:
        evaluation, _item = _record_interaction(harness, session)
        session.commit()
        session.execute(
            text("UPDATE evidence_events SET payload = :payload WHERE sequence = :sequence"),
            {"payload": '{"tampered": true}', "sequence": 1},
        )
        session.commit()
        session.expire_all()
        report = reconstruct(session, evaluation.request_id)
    assert any(finding["kind"] == "chain" for finding in report["findings"])
    assert [item["question"] for item in report["questions"]] == list(QUESTIONS)
    assert _answer(report, "Why this client?") == "not recorded"


def test_erasure_keeps_the_hash_and_is_not_a_mismatch(harness):
    wording = "erase this response"
    registry = ToolRegistry()

    @registry.tool("get_subject_profile", classification=Classification.C2)
    def _profile(envelope, evaluation, session):
        return {"note": wording}

    harness.registry = registry
    envelope = dp.envelope(
        "get_subject_profile",
        site="SG",
        subject="2",
        agent="dossier-agent",
    )
    with harness.session() as session:
        run = harness.run_tool(envelope, session)
        tool_call = session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.correlation_id == run.evaluation.request_id,
                EvidenceEvent.event_type == EventType.TOOL_CALL,
            )
        ).one()
        digest = tool_call.payload["response_hash"]
        RetentionStore().erase(
            session,
            digest,
            evidence=harness.evidence,
            actor_id="privacy-officer",
            reason="erasure request",
        )
        session.commit()

        report = reconstruct(session, run.evaluation.request_id)
        document = render_reconstruction(report)
        reproduce = _answer(report, "Can the bank reproduce that evidence later?")
        erased = [
            pin
            for pin in reproduce["pins"]
            if pin.get("content_hash") == digest and pin["kind"] == "response"
        ]
        assert {pin["event_type"] for pin in erased} == {"TOOL_CALL", "EGRESS"}
        assert all(pin["status"] == "erased" for pin in erased)
        assert report["chain"]["intact"] is True
        assert report["chain"]["events_checked"] >= 1
        assert not any(finding["kind"] == "retained_content" for finding in report["findings"])
        assert not any(finding["kind"] == "chain" for finding in report["findings"])
        assert digest in document
        assert "erased" in document
        assert wording not in document
        assert [item["question"] for item in report["questions"]] == list(QUESTIONS)
        replay_decision(session, run.evaluation.request_id)


def test_empty_response_hash_is_not_a_finding(harness):
    with harness.session() as session:
        harness.evidence.emit(
            session,
            correlation_id="empty-hash",
            event_type=EventType.EGRESS,
            payload={"response_hash": "", "tool": "get_subject_profile"},
        )
        session.commit()
        report = reconstruct(session, "empty-hash")
        pins = _answer(report, "Can the bank reproduce that evidence later?")["pins"]
    assert report["findings"] == []
    assert any(
        pin["kind"] == "response" and pin["status"] == "empty" and pin["content_hash"] == ""
        for pin in pins
    )


def test_missing_interaction_still_lists_every_question(harness):
    with harness.session() as session:
        report = reconstruct(session, "missing-interaction")
    assert [item["question"] for item in report["questions"]] == list(QUESTIONS)
    assert any(finding["kind"] == "interaction" for finding in report["findings"])
    for question in NOT_RECORDED:
        assert _answer(report, question) == "not recorded"


def test_cli_writes_json_and_a_readable_document(tmp_path):
    db = tmp_path / "store.db"
    engine = make_engine(db)
    create_all(engine)
    evidence = EvidenceWriter(engine, clock=lambda: FROZEN_NOW)
    pack = dp.DemoPack()
    with Session(engine) as session:
        _seed(session)
        envelope = _envelope()
        subjects = pack.load_subjects(envelope, session)
        attributes = pack.resolve_attributes(envelope, subjects)
        evaluation = decide(
            envelope,
            load_snapshot(session, as_of=FROZEN_NOW),
            attributes=attributes,
            subjects=subjects,
            rules=pack.rules(),
            purposes=dp.PURPOSES,
            engages=pack.engages,
            as_of=FROZEN_NOW,
            request_id="cli-interaction",
        )
        record(session, evaluation, evidence=evidence, now=FROZEN_NOW)
        session.commit()

    prefix = tmp_path / "report"
    code = main(
        ["evidence", "reconstruct", "cli-interaction", "--db", str(db), "-o", str(prefix)]
    )
    assert code == 0
    report = json.loads((tmp_path / "report.json").read_text())
    document = (tmp_path / "report.txt").read_text()
    assert [item["question"] for item in report["questions"]] == list(QUESTIONS)
    assert _answer(report, "Why this client?") == "not recorded"
    assert "Why this client?" in document
    assert "not recorded" in document
    assert "OWNER" in document

    stdout = StringIO()
    with contextlib.redirect_stdout(stdout):
        code = main(["evidence", "reconstruct", "cli-interaction", "--db", str(db)])
    assert code == 0
    printed = stdout.getvalue()
    assert "Reconstruction report" in printed
    assert "--- json ---" in printed
    printed_json = json.loads(printed.split("--- json ---", 1)[1])
    assert printed_json["interaction_id"] == "cli-interaction"


def test_superseded_policy_still_matches(harness):
    """A successor row is not an in-place edit. The pinned hash still matches."""
    from kognita.governance import supersede_policy

    with harness.session() as session:
        evaluation, _item = _record_interaction(harness, session)
        policy_id = next(check.policy_id for check in evaluation.checks if check.policy_hash)
        policy = session.get(Policy, policy_id)
        supersede_policy(
            session,
            policy,
            at=harness.now + timedelta(days=1),
            citation="revised citation",
            evidence=harness.evidence,
            actor_id="policy-owner",
        )
        session.commit()
        report = reconstruct(session, evaluation.request_id)
    assert not any(
        finding["kind"] == "policy_content_hash" for finding in report["findings"]
    )
