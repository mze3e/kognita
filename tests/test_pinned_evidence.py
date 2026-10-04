"""Pinned evidence: policy, retrieval, and model inputs cannot change quietly.

Roadmap 0.3 item 7. Replay fails when a policy row, a retrieved item, or a
retained prompt no longer matches the hash the decision recorded. Erasure
removes the bytes, appends an event, and leaves the chain verifiable.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlmodel import select

from fixtures import demo_pack as dp

from kognita.canonical import canonical_json, hash_text
from kognita.evidence import verify_chain
from kognita.exceptions import PolicyEditError, ReplayMismatch
from kognita.gateway import ClientConfiguration, Gateway
from kognita.models import EvidenceEvent, Policy, RetainedContent
from kognita.registry import register
from kognita.replay import replay_decision
from kognita.retention import RetentionStore
from kognita.retrieval import index_item, retrieve
from kognita.tools import ToolRegistry
from kognita.vocabulary import Classification, EventType, Outcome
from kognita.testing.harness import FROZEN_NOW, Harness

MODEL = "gateway-model"
MODEL_VERSION = "gateway-model-2026-06-01"
PATH = "/v1/chat/completions"
UPSTREAM = "https://api.openai.com"
SECRET = "ana@example.org"


def _seed(session):
    dp.seed_policies(session, now=FROZEN_NOW)
    register(session, name="eligibility-assistant", owner_exec="Head of Data Governance")
    register(session, name="dossier-agent", owner_exec="Head of Research Ops")


@pytest.fixture
def harness():
    return Harness(pack=dp.DemoPack(), purposes=dp.PURPOSES, seed=_seed)


class _Upstream:
    def __init__(self, response: bytes) -> None:
        self.calls: list[dict] = []
        self.response = response

    def __call__(self, method, url, headers, body):
        self.calls.append({"body": body})
        return 200, {"content-type": "application/json"}, self.response


def _body(content: str) -> bytes:
    return json.dumps(
        {"model": MODEL, "messages": [{"role": "user", "content": content}]}
    ).encode()


def test_checks_pin_the_policy_row_and_replay_accepts_a_successor(harness):
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        evaluation = harness.decide_and_record(envelope, session)
        pinned = [check for check in evaluation.checks if check.policy_id is not None]
        assert pinned
        assert all(check.policy_hash and len(check.policy_hash) == 64 for check in pinned)
        policy = session.get(Policy, pinned[0].policy_id)
        assert policy is not None
        successor = harness_supersede(session, policy, harness)
        session.commit()

        replayed = replay_decision(
            session,
            evaluation.request_id,
            envelope,
            rules=dp.DemoPack().rules(),
            purposes=dp.PURPOSES,
            engages=dp.DemoPack().engages,
            subjects=dp.DemoPack().load_subjects(envelope, session),
        )
    assert replayed.outcome is evaluation.outcome
    assert successor.id != policy.id
    assert policy.effective_to is not None


def harness_supersede(session, policy, harness):
    from kognita.governance import supersede_policy

    return supersede_policy(
        session,
        policy,
        at=harness.now + timedelta(days=1),
        citation="Data Sharing Charter s4, Schedule 1 (revised)",
        evidence=harness.evidence,
        actor_id="policy-owner",
    )


def test_in_place_edit_of_an_effective_policy_is_refused(harness):
    with harness.session() as session:
        policy = session.exec(select(Policy)).first()
        assert policy is not None
        policy.citation = "rewritten after the fact"
        with pytest.raises(PolicyEditError):
            session.flush()
        session.rollback()


def test_a_policy_that_is_not_yet_effective_can_be_edited(harness):
    with harness.session() as session:
        draft = Policy(
            regime="HOME_SITE",
            rule_type="PROHIBITED",
            rule={"description": "not yet", "on_violation": "fail"},
            citation="draft",
            effective_from=datetime(2999, 1, 1, tzinfo=timezone.utc),
        )
        session.add(draft)
        session.commit()
        draft.citation = "still a draft"
        session.commit()
        assert draft.citation == "still a draft"


def test_editing_a_policy_row_after_the_fact_fails_replay(harness):
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        evaluation = harness.decide_and_record(envelope, session)
        policy_id = next(check.policy_id for check in evaluation.checks if check.policy_hash)
        session.commit()
        session.execute(
            text("UPDATE policies SET citation = :citation WHERE id = :id"),
            {"citation": "tampered", "id": policy_id},
        )
        session.commit()
        session.expire_all()
        with pytest.raises(ReplayMismatch, match="edited in place"):
            replay_decision(session, evaluation.request_id)


def test_editing_a_retrieved_item_fails_replay(harness):
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        evaluation = harness.decide_and_record(envelope, session)
        item = index_item(
            session,
            title="cross-site eligibility",
            body="cross-site eligibility for restricted datasets",
            embedder=harness.embedder,
            zones=["SG"],
            classification=Classification.C1,
        )
        hits = retrieve(
            session,
            "cross-site eligibility for restricted datasets",
            zone="SG",
            embedder=harness.embedder,
            evidence=harness.evidence,
            correlation_id=evaluation.request_id,
            use_case=envelope.purpose,
        )
        assert hits
        session.commit()
        session.execute(
            text("UPDATE knowledge_items SET body = :body WHERE id = :id"),
            {"body": "replaced after retrieval", "id": item.id},
        )
        session.commit()
        session.expire_all()
        with pytest.raises(ReplayMismatch, match="retrieved item"):
            replay_decision(session, evaluation.request_id)


def test_reindex_changes_the_embedding_model_and_replay_fails(harness):
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        evaluation = harness.decide_and_record(envelope, session)
        item = index_item(
            session,
            title="cross-site eligibility",
            body="cross-site eligibility for restricted datasets",
            embedder=harness.embedder,
            zones=["SG"],
            classification=Classification.C1,
        )
        retrieve(
            session,
            "cross-site eligibility for restricted datasets",
            zone="SG",
            embedder=harness.embedder,
            evidence=harness.evidence,
            correlation_id=evaluation.request_id,
        )
        session.commit()
        session.execute(
            text("UPDATE knowledge_items SET embedding_model = :model WHERE id = :id"),
            {"model": "other-embedder", "id": item.id},
        )
        session.commit()
        session.expire_all()
        with pytest.raises(ReplayMismatch, match="embedding model"):
            replay_decision(session, evaluation.request_id)


def test_model_call_pins_identity_and_a_stored_prompt_edit_fails_replay(session, evidence):
    register(session, name="dossier-agent", owner_exec="Head of Research Ops")
    session.flush()
    prompt = f"Email {SECRET} about the notes"
    answer = "Reply to [EMAIL_1] tomorrow"
    upstream = _Upstream(
        json.dumps(
            {
                "model": MODEL,
                "model_version": MODEL_VERSION,
                "choices": [{"message": {"content": answer}}],
            }
        ).encode()
    )
    gateway = Gateway(
        engine=session.get_bind(),
        evidence=evidence,
        upstream=UPSTREAM,
        client=ClientConfiguration(
            principal="alice",
            purpose="COLLABORATION",
            agent_names=frozenset({"dossier-agent"}),
            actor_location="SG",
        ),
        purposes=("COLLABORATION",),
        transport=upstream,
    )
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent", "prompt_template_version": "template-3"},
        _body(prompt),
        session=session,
    )
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.ALLOW
    events = session.exec(
        select(EvidenceEvent).where(
            EvidenceEvent.correlation_id == response.evaluation.request_id
        )
    ).all()
    blob = canonical_json([event.payload for event in events])
    assert SECRET not in blob
    assert prompt not in blob
    assert answer not in blob
    model_call = next(event for event in events if event.event_type is EventType.MODEL_CALL)
    egress = next(event for event in events if event.event_type is EventType.EGRESS)
    sent = upstream.calls[0]["body"].decode()
    assert model_call.payload["provider"] == "api.openai.com"
    assert model_call.payload["model"] == MODEL
    assert model_call.payload["model_version"] == MODEL_VERSION
    assert model_call.payload["prompt_template_version"] == "template-3"
    assert model_call.payload["prompt_hash"] == hash_text(sent)
    assert model_call.payload["response_hash"] == hash_text(upstream.response.decode())
    assert egress.payload["response_hash"] == model_call.payload["response_hash"]
    session.execute(
        text("UPDATE retained_content SET body = :body WHERE content_hash = :content_hash"),
        {"body": "tampered prompt", "content_hash": model_call.payload["prompt_hash"]},
    )
    session.commit()
    session.expire_all()
    with pytest.raises(ReplayMismatch, match="does not match its hash"):
        replay_decision(session, response.evaluation.request_id)


def test_tool_response_hash_is_on_the_chain_and_the_body_is_retained(harness):
    wording = "client-specific wording"
    registry = ToolRegistry()

    @registry.tool("get_subject_profile", classification=Classification.C2)
    def _profile(envelope, evaluation, session):
        return {"note": wording}

    harness.registry = registry
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        run = harness.run_tool(envelope, session)
        session.commit()
        events = session.exec(
            select(EvidenceEvent)
            .where(EvidenceEvent.correlation_id == run.evaluation.request_id)
            .order_by(EvidenceEvent.sequence)
        ).all()
        blob = canonical_json([event.payload for event in events])
        assert wording not in blob
        tool_call = next(event for event in events if event.event_type is EventType.TOOL_CALL)
        egress = next(event for event in events if event.event_type is EventType.EGRESS)
        assert tool_call.payload["response_hash"] == egress.payload["response_hash"]
        retained = RetentionStore().read(session, tool_call.payload["response_hash"])
        assert retained is not None
        assert wording in retained


def test_erasing_retained_content_keeps_the_chain_verifiable(harness):
    wording = "erase this response"
    registry = ToolRegistry()

    @registry.tool("get_subject_profile", classification=Classification.C2)
    def _profile(envelope, evaluation, session):
        return {"note": wording}

    harness.registry = registry
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
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

        assert RetentionStore().read(session, digest) is None
        assert session.get(RetainedContent, digest) is None
        checked = verify_chain(session)
        assert checked >= 1
        erasure = session.exec(
            select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.ERASURE)
        ).one()
        assert erasure.payload["content_hash"] == digest
        assert erasure.payload["kind"] == "tool_response"
        assert wording not in canonical_json(erasure.payload)
        replay_decision(session, run.evaluation.request_id)
        assert verify_chain(session) == checked


def test_retention_period_erases_and_records_the_event(harness):
    with harness.session() as session:
        store = RetentionStore()
        store.set_policy(session, "COLLABORATION", retain_days=30)
        digest = store.retain_text(
            session,
            "prompt kept under the use case",
            kind="prompt",
            use_case="COLLABORATION",
            correlation_id="retention-1",
            now=FROZEN_NOW - timedelta(days=31),
        )
        removed = store.enforce(session, evidence=harness.evidence, now=FROZEN_NOW)
        session.commit()
        assert removed == 1
        assert store.read(session, digest) is None
        erasure = session.exec(
            select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.ERASURE)
        ).one()
        assert erasure.payload["content_hash"] == digest
        assert erasure.payload["use_case"] == "COLLABORATION"
        verify_chain(session)
