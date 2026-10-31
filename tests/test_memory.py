"""Memory governance: working expiry, proposal verification, and standards."""
from datetime import datetime, timedelta, timezone
import json

import pytest
from sqlmodel import select

import kognita
from kognita.autonomy import _checks
from kognita.envelope import Envelope
from kognita.evidence import verify_chain
from kognita.models import EvidenceEvent, Interaction, InteractionLink, Proposal, RetainedContent, RunRecord, UseCase
from kognita.registry import _deployment_hash, approve_deployment, register
from kognita.vocabulary import ActorType, CheckResult, EventType
from kognita.memory import (
    MemoryRecord, MemoryScope, Standard, approve_standard, current_standard,
    propose_memory, propose_standard, verify_memory, write_memory,
)

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(days=10)
CLIENT = "client-1"
OTHER = "client-2"
SENTENCE = "Client may be considering a sale of the riverside property"
_DEPLOYMENT = dict(
    system_prompt="prompt", instructions="instructions", tools=[], permissions=[],
    data_sources=[], orchestration="one step", memory_behaviour="none",
)


def _record(scope=MemoryScope.RELATIONSHIP, type_="inferred preference", client=CLIENT, content=SENTENCE, run_id="", expiry=None, retention=timedelta(days=30)):
    if scope is MemoryScope.WORKING and not run_id:
        run_id = "run-1"
    return MemoryRecord(
        client, content, type_, retention, expiry or (NOW + timedelta(hours=1) if scope is MemoryScope.WORKING else NOW + retention),
        "RM desk corrects the record", scope, run_id,
    )


def _standard(version="1", content="Use the current house view", effective=None, expiry=None, refs=("outcome:12",)):
    return Standard("house-view", version, effective or NOW, None, refs, content, expiry, "advice")


@pytest.fixture
def ready(session, evidence):
    session.add(UseCase(
        usecase_id="advice", version=1, effective_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
        record={"usecase_id": "advice", "knowledge_lead": "knowledge-lead", "version": 1, "approved_sources": ["crm"],
                "approved_models": ["house/summary@1"]},
    ))
    session.add(Interaction(id="i1", client_id=CLIENT, use_case="advice"))
    session.add(Interaction(id="i2", client_id=OTHER, use_case="advice"))
    session.add(RunRecord(id="run-1", control_state="live", started_at=NOW, wall_clock_seconds=3600))
    session.flush()
    session.add(InteractionLink(interaction_id="i1", run_id="run-1", correlation_id="i1"))
    session.flush()
    agent = register(session, name="advisor", owner_exec="owner")
    agent.purpose = "Prepare advice"
    agent.use_cases = ["advice"]
    agent.permitted_systems = ["research"]
    agent.tool_allowlist = {"read": {"verb": "read", "required_level": 0, "permissions": ["read"], "system": "research"}}
    agent.autonomy_level = 0
    agent.review_date = NOW + timedelta(days=90)
    agent.principal = "alice"
    agent.environment = "test"
    agent.deployment_date = NOW
    agent.model = "fixture-model"
    approve_deployment(session, agent.name, deployment=dict(_DEPLOYMENT), evidence=evidence, actor_id="owner")
    session.flush()
    return session


def _writes(session):
    return [row for row in session.exec(select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.MEMORY_WRITE)) if row.payload.get("allowed") is True]


def _stored(session):
    return " ".join(row.body for row in session.exec(select(RetainedContent)))


def test_names_are_exported():
    for name in (
        "MemoryRecord", "MemoryScope", "Standard", "write_memory", "propose_memory",
        "verify_memory", "propose_standard", "approve_standard", "current_standard",
    ):
        assert name in kognita.__all__
        assert hasattr(kognita, name)
    assert EventType.MEMORY_WRITE.value == "MEMORY_WRITE"


def test_working_memory_expires_with_the_run(ready, evidence):
    session = ready
    record = write_memory(session, _record(MemoryScope.WORKING), evidence=evidence, correlation_id="run-1", run_id="run-1", now=NOW)
    assert record.scope is MemoryScope.WORKING
    assert record.content not in json.dumps(_writes(session)[-1].payload)
    run = session.get(RunRecord, "run-1")
    run.control_state = "paused"
    session.add(run)
    session.flush()
    write_memory(session, _record(MemoryScope.WORKING, content="A fresh working note"), evidence=evidence, correlation_id="run-1", run_id="run-1", now=NOW)
    run.control_state = "recovered"
    session.add(run)
    session.flush()
    with pytest.raises(ValueError, match="MEMORY_EXPIRED"):
        write_memory(session, _record(MemoryScope.WORKING), evidence=evidence, correlation_id="run-1", run_id="run-1", now=NOW)
    run.control_state = "live"
    session.add(run)
    session.flush()
    with pytest.raises(ValueError, match="MEMORY_EXPIRED"):
        write_memory(session, _record(MemoryScope.WORKING), evidence=evidence, correlation_id="run-1", run_id="run-1", now=NOW + timedelta(seconds=3600))
    with pytest.raises(ValueError, match="MEMORY_EXPIRED"):
        write_memory(session, _record(MemoryScope.WORKING, expiry=NOW + timedelta(days=2)), evidence=evidence, correlation_id="run-1", run_id="run-1", now=NOW)
    with pytest.raises(ValueError, match="MEMORY_WORKING_RUN"):
        write_memory(session, _record(MemoryScope.WORKING, run_id="missing"), evidence=evidence, correlation_id="missing", run_id="missing", now=NOW)
    verify_chain(session)


@pytest.mark.parametrize("memory_type", ["inferred preference", "AI hypothesis"])
@pytest.mark.parametrize("scope", [MemoryScope.RELATIONSHIP, MemoryScope.INSTITUTIONAL])
def test_direct_inference_write_is_denied(ready, evidence, memory_type, scope):
    session = ready
    if scope is MemoryScope.INSTITUTIONAL:
        with pytest.raises(ValueError, match="MEMORY_DIRECT_WRITE"):
            _record(scope, memory_type)
        return
    with pytest.raises(ValueError, match="MEMORY_DIRECT_WRITE"):
        write_memory(session, _record(scope, memory_type), evidence=evidence, correlation_id="i1", now=NOW)
    assert _writes(session) == []
    denied = session.exec(select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.MEMORY_WRITE)).all()[-1]
    assert denied.payload["allowed"] is False
    assert SENTENCE not in json.dumps(denied.payload)
    assert SENTENCE not in _stored(session)


def test_relationship_inference_requires_rm_verification(ready, evidence):
    session = ready
    proposal = propose_memory(session, _record(), evidence=evidence, correlation_id="i1", proposer_id="agent-1", now=NOW)
    assert _writes(session) == []
    assert SENTENCE not in json.dumps(session.exec(select(EvidenceEvent)).all()[-1].payload)
    with pytest.raises(ValueError, match="MEMORY_VERIFIER"):
        verify_memory(session, proposal, evidence=evidence, correlation_id="i1", verifier_id="agent-1", actor_type=ActorType.HUMAN, client_id=CLIENT, now=NOW)
    with pytest.raises(ValueError, match="MEMORY_VERIFIER"):
        verify_memory(session, proposal, evidence=evidence, correlation_id="i1", verifier_id=" agent-1 ", actor_type=ActorType.HUMAN, client_id=CLIENT, now=NOW)
    with pytest.raises(ValueError, match="MEMORY_VERIFIER"):
        verify_memory(session, proposal, evidence=evidence, correlation_id="i1", verifier_id="rm-1", actor_type=ActorType.AGENT, client_id=CLIENT, now=NOW)
    with pytest.raises(ValueError, match="MEMORY_SCOPE"):
        verify_memory(session, proposal, evidence=evidence, correlation_id="i1", verifier_id="rm-1", actor_type=ActorType.HUMAN, client_id=OTHER, now=NOW)
    assert _writes(session) == []
    record = verify_memory(session, proposal, evidence=evidence, correlation_id="i1", verifier_id="rm-1", actor_type=ActorType.HUMAN, client_id=CLIENT, now=NOW)
    assert record.type == "inferred preference" and record.client_id == CLIENT
    event = _writes(session)[-1]
    assert event.payload["action"] == "verify" and event.payload["standard_version"] == ""
    assert SENTENCE not in json.dumps(event.payload)
    assert SENTENCE in _stored(session)
    with pytest.raises(ValueError, match="MEMORY_BINDING"):
        verify_memory(session, proposal, evidence=evidence, correlation_id="i1", verifier_id="rm-2", actor_type=ActorType.HUMAN, client_id=CLIENT, now=NOW)
    verify_chain(session)


def test_verified_fact_can_be_written_for_one_client(ready, evidence):
    session = ready
    _fact(session, evidence, SENTENCE)
    record = write_memory(session, _record(type_="verified fact"), evidence=evidence, correlation_id="i1", now=NOW)
    assert record.type == "verified fact"
    with pytest.raises(ValueError, match="MEMORY_SCOPE"):
        write_memory(session, _record(type_="verified fact", client=OTHER, content=f"see {CLIENT}"), evidence=evidence, correlation_id="i2", now=NOW)
    with pytest.raises(ValueError, match="MEMORY_SCOPE"):
        write_memory(session, _record(MemoryScope.WORKING, client=OTHER), evidence=evidence, correlation_id="run-1", run_id="run-1", now=NOW)
    with pytest.raises(ValueError, match="MEMORY_INVALID"):
        MemoryRecord("client 1", "text", "verified fact", timedelta(days=1), NOW + timedelta(days=1), "desk", MemoryScope.RELATIONSHIP)
    with pytest.raises(ValueError, match="MEMORY_INVALID"):
        MemoryRecord(CLIENT, "text", "verified fact ", timedelta(days=1), NOW + timedelta(days=1), "desk", MemoryScope.RELATIONSHIP)


def test_proposal_tamper_and_wrong_kind_do_not_apply(ready, evidence):
    session = ready
    proposal = propose_memory(session, _record(), evidence=evidence, correlation_id="i1", proposer_id="agent-1", now=NOW)
    proposal.change = {**proposal.change, "body": {**proposal.change["body"], "content": "rewritten after review"}}
    session.add(proposal)
    session.flush()
    with pytest.raises(ValueError, match="MEMORY_BINDING"):
        verify_memory(session, proposal, evidence=evidence, correlation_id="i1", verifier_id="rm-1", actor_type=ActorType.HUMAN, client_id=CLIENT, now=NOW)
    standard = propose_standard(session, _standard(), evidence=evidence, correlation_id="std", proposer_id="agent-1", usecase_id="advice", agent_name="advisor", now=NOW)
    with pytest.raises(ValueError, match="MEMORY_BINDING"):
        verify_memory(session, standard, evidence=evidence, correlation_id="i1", verifier_id="rm-1", actor_type=ActorType.HUMAN, client_id=CLIENT, now=NOW)
    assert _writes(session) == []


def test_single_client_content_cannot_enter_a_standard(ready, evidence):
    session = ready
    propose_memory(session, _record(), evidence=evidence, correlation_id="i1", proposer_id="agent-1", now=NOW)
    for content in (
        SENTENCE,
        f"Learned from {CLIENT}",
        f"Learned from {CLIENT[:6]}\u200b{CLIENT[6:]}",
        "Learned from \uff43lient-1",
        json.dumps({"client_id": CLIENT, "note": SENTENCE}),
    ):
        if "\u200b" in content:
            with pytest.raises(ValueError, match="MEMORY_INVALID"):
                _standard(content=content)
            continue
        with pytest.raises(ValueError, match="MEMORY_CLIENT_CONTENT"):
            propose_standard(session, _standard(content=content), evidence=evidence, correlation_id="std", proposer_id="agent-1", usecase_id="advice", agent_name="advisor", now=NOW)
    with pytest.raises(ValueError, match="MEMORY_CLIENT_CONTENT"):
        propose_standard(session, _standard(refs=(CLIENT,)), evidence=evidence, correlation_id="std", proposer_id="agent-1", usecase_id="advice", agent_name="advisor", now=NOW)
    with pytest.raises(ValueError, match="MEMORY_EVIDENCE"):
        _standard(refs="outcome:12")
    assert not session.exec(select(Proposal).where(Proposal.kind == "institutional_standard")).all()
    assert SENTENCE not in json.dumps([row.payload for row in session.exec(select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.MEMORY_WRITE))])


def test_knowledge_lead_approval_versions_the_deployment(ready, evidence):
    session = ready
    from kognita.registry import get
    agent = get(session, "advisor")
    before = agent.deployment_hash
    proposal = propose_standard(session, _standard(), evidence=evidence, correlation_id="std", proposer_id="agent-1", usecase_id="advice", agent_name="advisor", now=NOW)
    with pytest.raises(ValueError, match="MEMORY_STANDARD"):
        current_standard(session, "house-view", usecase_id="advice", now=NOW)
    with pytest.raises(ValueError, match="MEMORY_KNOWLEDGE_LEAD"):
        approve_standard(session, proposal, evidence=evidence, correlation_id="std", approver_id="owner", actor_type=ActorType.HUMAN, now=NOW)
    with pytest.raises(ValueError, match="MEMORY_KNOWLEDGE_LEAD"):
        approve_standard(session, proposal, evidence=evidence, correlation_id="std", approver_id="knowledge-lead", actor_type=ActorType.AGENT, now=NOW)
    with pytest.raises(ValueError, match="MEMORY_VERIFIER"):
        own = propose_standard(session, _standard(version="9"), evidence=evidence, correlation_id="std", proposer_id="knowledge-lead", usecase_id="advice", agent_name="advisor", now=NOW)
        approve_standard(session, own, evidence=evidence, correlation_id="std", approver_id="knowledge-lead", actor_type=ActorType.HUMAN, now=NOW)
    approved = approve_standard(session, proposal, evidence=evidence, correlation_id="std", approver_id="knowledge-lead", actor_type=ActorType.HUMAN, now=NOW)
    assert approved.version == "1"
    assert current_standard(session, "house-view", usecase_id="advice", now=NOW).version == "1"
    session.refresh(agent)
    assert agent.deployment_hash != before
    assert agent.deployment_hash == _deployment_hash(agent, agent.tool_allowlist["_deployment"])
    behaviour = agent.tool_allowlist["_deployment"]["memory_behaviour"]
    assert "house-view" in behaviour and "\"1\"" in behaviour
    assert agent.tool_allowlist["_deployment"]["instructions"] == "instructions"
    assert SENTENCE not in behaviour
    event = _writes(session)[-1]
    assert event.payload["standard_version"] == "1"
    assert "Use the current house view" not in json.dumps(event.payload)
    checks = _checks(Envelope(principal="alice", purpose="advice", tool="read", agent_name="advisor"), agent, NOW)
    assert next(check for check in checks if check.check == "S1-DEPLOYMENT").result is CheckResult.PASS
    with pytest.raises(ValueError, match="MEMORY_BINDING"):
        approve_standard(session, proposal, evidence=evidence, correlation_id="std", approver_id="knowledge-lead", actor_type=ActorType.HUMAN, now=NOW)
    verify_chain(session)


def test_superseded_and_expired_standards_cannot_be_used(ready, evidence, engine):
    session = ready
    first = propose_standard(session, _standard(expiry=NOW + timedelta(days=5)), evidence=evidence, correlation_id="std", proposer_id="agent-1", usecase_id="advice", agent_name="advisor", now=NOW)
    approve_standard(session, first, evidence=evidence, correlation_id="std", approver_id="knowledge-lead", actor_type=ActorType.HUMAN, now=NOW)
    original = _approved_body(session)
    second = propose_standard(
        session, _standard(version="2", effective=LATER, content="Use the revised house view"),
        evidence=evidence, correlation_id="std", proposer_id="agent-1", usecase_id="advice", agent_name="advisor", now=NOW,
    )
    approve_standard(session, second, evidence=evidence, correlation_id="std", approver_id="knowledge-lead", actor_type=ActorType.HUMAN, now=NOW)
    assert _approved_body(session) == original
    assert original["superseded"] is None
    assert current_standard(session, "house-view", usecase_id="advice", now=NOW).version == "1"
    with pytest.raises(ValueError, match="MEMORY_STANDARD"):
        write_memory(session, _record(type_="verified fact"), evidence=evidence, correlation_id="i1", usecase_id="advice", standard_version="2", now=NOW)
    _fact(session, evidence, "A verified holding")
    recorded = write_memory(session, _record(type_="verified fact", content="A verified holding"), evidence=evidence, correlation_id="i1", usecase_id="advice", now=NOW + timedelta(days=1))
    assert recorded.type == "verified fact"
    assert _writes(session)[-1].payload["standard_version"] == "1"
    assert current_standard(session, "house-view", usecase_id="advice", now=LATER).version == "2"
    with pytest.raises(ValueError, match="MEMORY_STANDARD"):
        write_memory(session, _record(type_="verified fact", content="Another verified holding"), evidence=evidence, correlation_id="i1", usecase_id="advice", standard_version="1", now=LATER)
    with pytest.raises(ValueError, match="MEMORY_VERSION"):
        propose_standard(session, _standard(version="2", effective=LATER, content="edited in place"), evidence=evidence, correlation_id="std", proposer_id="agent-1", usecase_id="advice", agent_name="advisor", now=LATER)
    with pytest.raises(ValueError, match="MEMORY_STANDARD"):
        current_standard(session, "house-view", usecase_id="advice", now=NOW + timedelta(days=5))
    session.commit()
    from sqlmodel import Session
    with Session(engine) as later:
        assert current_standard(later, "house-view", usecase_id="advice", now=LATER).version == "2"
        verify_chain(later)


def _approved_body(session, version="1"):
    from kognita.retention import RetentionStore
    event = next(
        row for row in session.exec(select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.MEMORY_WRITE))
        if row.payload.get("action") == "approve" and row.payload.get("standard_version") == version
    )
    return json.loads(RetentionStore().read(session, event.payload["content_hash"]))


def test_damaged_standard_fails_closed(ready, evidence):
    session = ready
    proposal = propose_standard(session, _standard(), evidence=evidence, correlation_id="std", proposer_id="agent-1", usecase_id="advice", agent_name="advisor", now=NOW)
    approve_standard(session, proposal, evidence=evidence, correlation_id="std", approver_id="knowledge-lead", actor_type=ActorType.HUMAN, now=NOW)
    event = next(row for row in session.exec(select(EvidenceEvent)) if row.payload.get("action") == "approve")
    row = session.get(RetainedContent, event.payload["content_hash"])
    row.body = row.body.replace("house view", "rewritten view")
    session.add(row)
    session.flush()
    with pytest.raises(ValueError, match="MEMORY_STANDARD"):
        current_standard(session, "house-view", usecase_id="advice", now=NOW)


def _fact(session, evidence, content):
    from kognita.claims import Claim, ClaimType, ValidatorStatus, record_claim
    record_claim(session, Claim("fact-" + str(len(content)), CLIENT, content,
                 ClaimType.VERIFIED_FACT, source_system="crm", source_record="note",
                 validator_status=ValidatorStatus.VERIFIED), evidence=evidence, interaction_id="i1")


def test_relationship_fact_requires_sourced_verification(ready, evidence):
    from kognita.claims import Claim, ClaimType, ValidatorStatus, record_claim
    record = _record(type_="verified fact")
    with pytest.raises(ValueError, match="MEMORY_FACT"):
        write_memory(ready, record, evidence=evidence, correlation_id="i1", now=NOW)
    for index, (source, status) in enumerate((("", ValidatorStatus.VERIFIED), ("crm", ValidatorStatus.INFERRED))):
        record_claim(ready, Claim(f"unsupported-{index}", CLIENT, SENTENCE, ClaimType.VERIFIED_FACT,
                     source_system=source, source_record="note", validator_status=status),
                     evidence=evidence, interaction_id="i1")
        with pytest.raises(ValueError, match="MEMORY_FACT"):
            write_memory(ready, record, evidence=evidence, correlation_id="i1", now=NOW)
    _fact(ready, evidence, SENTENCE)
    assert write_memory(ready, record, evidence=evidence, correlation_id="i1", now=NOW) == record


def test_output_standard_checks_deployment_and_lifecycle(ready, evidence):
    from kognita.memory import _output_standard
    from kognita.registry import get
    assert _output_standard(ready, usecase_id="advice", at=NOW) == ""
    with pytest.raises(ValueError, match="MEMORY_STANDARD"):
        _output_standard(ready, usecase_id="advice", standard_version="forged", at=NOW)
    proposal = propose_standard(ready, _standard(expiry=LATER), evidence=evidence,
                                correlation_id="std", proposer_id="agent-1", usecase_id="advice",
                                agent_name="advisor", now=NOW)
    approve_standard(ready, proposal, evidence=evidence, correlation_id="std",
                     approver_id="knowledge-lead", actor_type=ActorType.HUMAN, now=NOW)
    assert _output_standard(ready, agent_id="advisor", usecase_id="advice", at=NOW) == "1"
    with pytest.raises(ValueError, match="MEMORY_STANDARD"):
        _output_standard(ready, agent_id="advisor", standard_version="0", at=NOW)
    with pytest.raises(ValueError, match="MEMORY_STANDARD"):
        _output_standard(ready, agent_id="advisor", at=LATER)
    agent = get(ready, "advisor")
    agent.model = "unapproved-model"
    ready.flush()
    with pytest.raises(ValueError, match="MEMORY_VERSION"):
        _output_standard(ready, agent_id="advisor", at=NOW)


def test_claim_outputs_pin_standard_and_reject_stale_citations(ready, evidence, monkeypatch):
    from dataclasses import replace
    from kognita.claims import Claim, ClaimType, ValidatorStatus, list_claims, record_claim, release_claim
    from kognita.sources import Source, register_source
    register_source(ready, Source("crm", "policy", "note", "owner", NOW),
                    evidence=evidence, interaction_id="setup")
    proposal = propose_standard(ready, _standard(expiry=LATER), evidence=evidence,
                                correlation_id="std", proposer_id="agent-1", usecase_id="advice",
                                agent_name="advisor", now=NOW)
    approve_standard(ready, proposal, evidence=evidence, correlation_id="std",
                     approver_id="knowledge-lead", actor_type=ActorType.HUMAN, now=NOW)
    monkeypatch.setattr("kognita.memory._now", lambda at=None: NOW)
    claim = Claim("output", CLIENT, "A verified current note", ClaimType.VERIFIED_FACT,
                  source_system="crm", source_record="note", agent_id="advisor",
                  validator_status=ValidatorStatus.VERIFIED)
    record_claim(ready, claim, evidence=evidence, interaction_id="i1")
    assert release_claim(ready, claim.claim_id, evidence=evidence, interaction_id="i1",
                         usecase_id="advice").result == CheckResult.PASS
    assert list_claims(ready)[0].standard_version == "1"
    record_claim(ready, replace(claim, claim_id="stale", standard_version="0"),
                 evidence=evidence, interaction_id="i1")
    assert release_claim(ready, "stale", evidence=evidence, interaction_id="i1",
                         usecase_id="advice").check == "MEMORY_STANDARD"


def test_model_egress_pins_standard_and_stops_after_expiry(ready, evidence, monkeypatch):
    from kognita.egress import EgressGuard, EgressDenied
    from kognita.vocabulary import Classification
    proposal = propose_standard(ready, _standard(expiry=LATER), evidence=evidence,
                                correlation_id="std", proposer_id="agent-1", usecase_id="advice",
                                agent_name="advisor", now=NOW)
    approve_standard(ready, proposal, evidence=evidence, correlation_id="std",
                     approver_id="knowledge-lead", actor_type=ActorType.HUMAN, now=NOW)
    monkeypatch.setattr("kognita.memory._now", lambda at=None: NOW)
    from kognita.models import ModelCard
    ready.add(ModelCard(card_id="house/summary@1", effective_from=NOW,
                        record={"card_id": "house/summary@1", "provider": "house",
                                "model": "summary", "version": "1"}))
    ready.flush()
    calls = []
    guard = EgressGuard(evidence=evidence)
    args = dict(classification=Classification.C1, destination="https://api.openai.com",
                destination_is_local=False, session=ready, correlation_id="output", use_case="advice",
                provider="house", model="summary", model_version="1")
    guard.send("instructions", lambda text: calls.append(text) or "response", **args)
    event = ready.exec(select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.MODEL_CALL)
                       .order_by(EvidenceEvent.sequence.desc())).first()
    assert event.payload["standard_version"] == "1"
    monkeypatch.setattr("kognita.memory._now", lambda at=None: LATER)
    with pytest.raises(EgressDenied, match="MEMORY_STANDARD"):
        guard.send("instructions", lambda text: calls.append(text), **args)
    assert calls == ["instructions"]


def test_working_read_enforces_client_run_and_completion(ready, evidence):
    from kognita.memory import _read_memory
    record = _record(MemoryScope.WORKING)
    write_memory(ready, record, evidence=evidence, correlation_id="run-1", run_id="run-1", now=NOW)
    digest = _writes(ready)[-1].payload["content_hash"]
    assert _read_memory(ready, digest, client_id=CLIENT, run_id="run-1", at=NOW) == record
    with pytest.raises(ValueError, match="MEMORY_SCOPE"):
        _read_memory(ready, digest, client_id=OTHER, run_id="run-1", at=NOW)
    with pytest.raises(ValueError, match="MEMORY_SCOPE"):
        _read_memory(ready, digest, client_id=CLIENT, run_id="other-run", at=NOW)
    evidence.emit(ready, correlation_id="run-1", event_type=EventType.FINAL_DECISION, payload={})
    with pytest.raises(ValueError, match="MEMORY_EXPIRED"):
        _read_memory(ready, digest, client_id=CLIENT, run_id="run-1", at=NOW)
    with pytest.raises(ValueError, match="MEMORY_EXPIRED"):
        write_memory(ready, record, evidence=evidence, correlation_id="run-1", run_id="run-1", now=NOW)


def test_unknown_client_identifier_label_is_refused_in_standard(ready, evidence):
    with pytest.raises(ValueError, match="MEMORY_CLIENT_CONTENT"):
        propose_standard(ready, _standard(content="client_id: unrecorded-person"),
                         evidence=evidence, correlation_id="std", proposer_id="agent-1",
                         usecase_id="advice", agent_name="advisor", now=NOW)
