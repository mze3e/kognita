"""S1 identity, deployment integrity, explicit verbs and governed autonomy."""
from dataclasses import replace
from datetime import timedelta
import json

import pytest
from pydantic import ValidationError
from sqlmodel import Session, select

from kognita import autonomy
from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.envelope import Envelope
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.governance import PolicySnapshot, decide, load_snapshot
from kognita.models import Agent, EvidenceEvent, RunRecord
from kognita.registry import register, approve_deployment, _deployment_hash
from kognita.tools import ToolRegistry, Run, run_governed, continue_run
from kognita.vocabulary import ActorType, EventType, Outcome
from fixtures import demo_pack as dp
from test_usecases import NOW, _open, _activate, _spec

_DEPLOYMENT = dict(system_prompt="", instructions="instructions", tools=[], permissions=[],
                   data_sources=[], orchestration="one step", memory_behaviour="none")


def _agent(session, evidence, *, level=0, verb="read", required=0):
    agent = register(session, name="agent", owner_exec="owner")
    agent.purpose = "Review research"
    agent.use_cases = ["research-summary"]
    agent.permitted_systems = ["research"]
    agent.tool_allowlist = {"read": {"verb": verb, "required_level": required,
        "permissions": [verb], "system": "research"}}
    agent.autonomy_level = 0
    agent.review_date = NOW + timedelta(days=90)
    agent.principal = "alice"
    approve_deployment(session, agent.name, deployment=dict(_DEPLOYMENT),
                       evidence=evidence, actor_id="owner")
    if level:
        # Simulate an already validated promotion for tests of downstream gates.
        autonomy._change(session, agent, level, evidence, "owner", "AUTONOMY_PROMOTED",
                         {"validator": "fixture-validator", "metrics": []})
    return agent


def _env(**kwargs):
    return replace(Envelope(principal="alice", purpose="research-summary", tool="read",
                            agent_name="agent"), **kwargs)


def _decide(agent, envelope=None, **kwargs):
    return decide(envelope or _env(), PolicySnapshot(agents={agent.name: agent}),
                  purposes=("research-summary",), as_of=NOW, **kwargs)


def test_old_sqlite_migration_is_nullable_additive_and_repeatable(tmp_path):
    engine = make_engine(tmp_path / "old.db")
    with engine.begin() as connection:
        connection.exec_driver_sql('''CREATE TABLE agents (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, version TEXT NOT NULL,
            owner_exec TEXT NOT NULL, risk_class TEXT NOT NULL,
            materiality_tier TEXT NOT NULL, kill_switch BOOLEAN NOT NULL,
            created_at DATETIME NOT NULL)''')
        connection.exec_driver_sql("INSERT INTO agents VALUES (1, 'agent', '1', 'owner', 'LOW', 'T1', 0, '2026-01-01')")
    create_all(engine)
    create_all(engine)
    with Session(engine) as session:
        agent = session.get(Agent, 1)
        assert agent.version == "1"
        assert agent.autonomy_level is None
        assert agent.permitted_systems is None
        assert agent.blast_radius is None
        assert agent.field_allowlist is None
        evaluation = _decide(agent)
        assert evaluation.outcome is Outcome.DENY
        assert all(check.citation for check in evaluation.checks)
    with engine.connect() as connection:
        columns = {row[1]: row for row in connection.exec_driver_sql("PRAGMA table_info(agents)")}
        assert all(columns[name][3] == 0 for name in (
            "purpose", "tool_allowlist", "autonomy_level", "blast_radius", "field_allowlist"))


@pytest.mark.parametrize("level", [-1, 7, 99])
def test_levels_are_bounded(level):
    with pytest.raises(ValidationError):
        Agent.model_validate(dict(name="a", autonomy_level=level))


def test_approved_deployment_allows_and_chain_is_append_only(session, evidence):
    agent = _agent(session, evidence)
    assert _decide(agent).outcome is Outcome.ALLOW
    assert verify_chain(session) > 0
    events = session.exec(select(EvidenceEvent)).all()
    assert events[-1].payload["deployment_hash"] == agent.deployment_hash
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError, match="append-only"):
        session.connection().exec_driver_sql("UPDATE evidence_events SET actor_id='intruder'")


@pytest.mark.parametrize("field", list(_DEPLOYMENT))
def test_each_behaviour_change_denies_until_owner_reapproves(session, evidence, field):
    agent = _agent(session, evidence)
    changed = {**_DEPLOYMENT, field: ["changed"]}
    envelope = _env(context={"deployment": changed})
    assert _decide(agent, envelope).outcome is Outcome.DENY
    approve_deployment(session, agent.name, deployment=changed, evidence=evidence, actor_id="owner")
    assert _decide(agent, envelope).outcome is Outcome.ALLOW


@pytest.mark.parametrize("change", ["purpose", "use_cases", "permitted_systems", "tool_allowlist", "principal", "environment"])
def test_authority_changes_invalidate_hash(session, evidence, change):
    agent = _agent(session, evidence)
    setattr(agent, change, {**agent.tool_allowlist, "extra": {}} if change == "tool_allowlist" else
            ["changed"] if change in ("use_cases", "permitted_systems") else "changed")
    result = _decide(agent)
    assert result.outcome is Outcome.DENY
    assert any(check.check == "S1-DEPLOYMENT" and check.result.value == "fail" for check in result.checks)


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_review_date_boundary(session, evidence, offset):
    agent = _agent(session, evidence)
    agent.review_date = NOW + timedelta(seconds=offset)
    approve_deployment(session, "agent", deployment=_DEPLOYMENT, evidence=evidence, actor_id="owner")
    assert _decide(agent).outcome is (Outcome.DENY if offset < 0 else Outcome.ALLOW)


def test_approval_requires_owner_and_complete_manifest(session, evidence):
    _agent(session, evidence)
    with pytest.raises(ValueError, match="owner"):
        approve_deployment(session, "agent", deployment=_DEPLOYMENT, evidence=evidence, actor_id="intruder")
    with pytest.raises(ValueError, match="seven"):
        approve_deployment(session, "agent", deployment={}, evidence=evidence, actor_id="owner")


@pytest.mark.parametrize("verb,level", [("read", 0), ("analyse", 1), ("recommend", 2),
    ("draft", 3), ("queue", 4), ("send", 4), ("execute", 4), ("execute", 5), ("execute", 6)])
def test_explicit_verbs_and_levels(session, evidence, verb, level):
    agent = _agent(session, evidence, level=level, verb=verb, required=level)
    expected = Outcome.HUMAN_APPROVAL if level == 4 else Outcome.ALLOW
    assert _decide(agent).outcome is expected
    if level:
        agent.autonomy_level = level - 1
        assert _decide(agent).outcome is Outcome.DENY


def test_recommend_does_not_authorize_execute(session, evidence):
    agent = _agent(session, evidence, level=6, verb="execute", required=5)
    agent.tool_allowlist["read"]["permissions"] = ["recommend"]
    # Even a fresh deployment approval cannot confer a missing verb permission.
    approve_deployment(session, "agent", deployment=_DEPLOYMENT, evidence=evidence, actor_id="owner")
    assert _decide(agent).outcome is Outcome.DENY


def test_usecase_cap_applies_even_when_agent_level_is_higher(session, evidence):
    from kognita.usecases import load_register
    _open(session, evidence)
    _activate(session, evidence, _spec())
    agent = _agent(session, evidence, level=3, verb="draft", required=3)
    result = _decide(agent, usecases=load_register(session, as_of=NOW))
    assert result.outcome is Outcome.DENY
    assert any(check.check == "S1-RISK-CAP" for check in result.failures())


@pytest.mark.parametrize("verb,level", [("bad", 0), ("execute", 2), ("read", True), ("read", 7)])
def test_invalid_tool_declarations_are_rejected(verb, level):
    with pytest.raises(ValueError, match="S1-TOOL"):
        ToolRegistry().register("read", lambda *_: None, verb=verb, required_level=level)


@pytest.mark.parametrize("extension", ["json", "yaml"])
def test_cli_deploy_evidences_manifest(tmp_path, extension):
    manifest = dict(name="agent", owner_exec="owner", purpose="research", use_cases=["research-summary"],
                    permitted_systems=["research"], tool_allowlist={}, principal="alice",
                    environment="test", review_date="2027-01-01T00:00:00+00:00", deployment=_DEPLOYMENT)
    path = tmp_path / f"manifest.{extension}"
    path.write_text(json.dumps(manifest) if extension == "json" else "\n".join(
        f"{key}: {json.dumps(value)}" for key, value in manifest.items()))
    db = tmp_path / "store.db"
    assert main(["agent", "deploy", "--manifest", str(path), "--db", str(db), "--actor", "owner"]) == 0
    with Session(make_engine(db)) as session:
        agent = session.exec(select(Agent)).one()
        assert agent.autonomy_level == 0
        assert agent.environment == "test"
        assert agent.deployment_hash
        assert verify_chain(session) > 0


def test_promotion_refuses_unmeasured_evidence_and_missing_signatures(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec(validators=("Jane Smith", "validator"),
        _autonomy_gates={"1": dict(minimum_sample=2, window_days=7, quality=.95, reliability=.95)}))
    agent = _agent(session, evidence)
    for actor, validator in [("intruder", "validator"), ("owner", "owner"), ("owner", ""),
                             ("owner", "validator")]:
        with pytest.raises(ValueError, match="S1-PROMOTION"):
            autonomy.promote(session, "agent", evidence=evidence, actor_id=actor,
                             validator=validator, now=NOW)
        assert agent.autonomy_level == 0


def _samples(session, engine, agent, at, *, good=4, total=4):
    from kognita.interaction import open_interaction, attach
    from kognita.review import record_final_decision
    writer = EvidenceWriter(engine, clock=lambda: at)
    for index in range(total):
        interaction = open_interaction(session, client_id=f"c{index}", use_case="research-summary",
                                       evidence=writer, now=at)
        correlation = interaction.id
        run_id = f"run:{interaction.id}"
        session.add(RunRecord(id=run_id, approvals_pending=[]))
        session.flush()
        attach(session, interaction.id, correlation_id=correlation, run_id=run_id, evidence=writer)
        writer.emit(session, correlation_id=correlation, event_type=EventType.POLICY_DECISION,
            actor_type=ActorType.SYSTEM, actor_id="pdp", payload={"outcome": "ALLOW",
                "envelope": {"agent_name": agent.name}, "deployment_hash": agent.deployment_hash})
        writer.emit(session, correlation_id=correlation, event_type=EventType.TOOL_CALL,
            actor_type=ActorType.AGENT, actor_id=agent.name,
            payload={"tool": "read", "result_hash": "fixture-output"})
        record_final_decision(session, interaction.id, outcome="allow" if index < good else "reject",
                              evidence=writer, actor_id="reviewer", citation="test", now=at)


def _promotion_setup(session, engine):
    writer = EvidenceWriter(engine, clock=lambda: NOW - timedelta(days=40))
    agent = _agent(session, writer)
    writer = EvidenceWriter(engine, clock=lambda: NOW)
    _open(session, writer)
    _activate(session, writer, _spec(validators=("Jane Smith", "validator"),
        _autonomy_gates={"1": dict(minimum_sample=4, window_days=7, quality=.95, reliability=1)},
        _autonomy_step_down=dict(minimum_sample=4, window_days=7, quality=.7, reliability=.8)))
    return agent, writer


def test_promotion_computes_outcomes_and_evidences_independent_validation(session, engine):
    agent, writer = _promotion_setup(session, engine)
    _samples(session, engine, agent, NOW - timedelta(days=1))
    autonomy.promote(session, "agent", evidence=writer, actor_id="owner", validator="validator", now=NOW)
    assert agent.autonomy_level == 1
    event = session.exec(select(EvidenceEvent).where(EvidenceEvent.correlation_id == "autonomy:agent")).one()
    assert event.payload["validator"] == "validator"
    assert event.payload["metrics"][0]["quality"]["sample"] == 4
    assert verify_chain(session) > 0


def test_below_quality_threshold_cannot_be_promoted(session, engine):
    agent, writer = _promotion_setup(session, engine)
    _samples(session, engine, agent, NOW - timedelta(days=1), good=3)
    with pytest.raises(ValueError, match="below threshold"):
        autonomy.promote(session, "agent", evidence=writer, actor_id="owner", validator="validator", now=NOW)
    assert agent.autonomy_level == 0


def test_sustained_drift_steps_down_on_next_snapshot_once_and_alerts(session, engine):
    agent, writer = _promotion_setup(session, engine)
    agent.autonomy_level = 2
    for days, good in ((18, 4), (11, 3), (4, 2)):
        _samples(session, engine, agent, NOW - timedelta(days=days), good=good)
    snapshot = load_snapshot(session, as_of=NOW, _evidence=writer, _agent_name="agent")
    assert snapshot.agents["agent"].autonomy_level == 1
    load_snapshot(session, as_of=NOW, _evidence=writer, _agent_name="agent")
    assert agent.autonomy_level == 1
    rows = session.exec(select(EvidenceEvent).where(EvidenceEvent.correlation_id == "autonomy:agent")).all()
    assert [row.payload["action"] for row in rows] == ["AUTONOMY_STEP_DOWN", "ALERT"]
    assert "owner" in rows[-1].payload["recipients"]
    assert verify_chain(session) > 0


@pytest.mark.parametrize("change", ["expiry", "deployment", "tool", "step_down", "none"])
def test_l4_hold_rechecks_current_authority_before_execution(session, evidence, change):
    agent = _agent(session, evidence, level=4, verb="queue", required=4)
    from test_usecases import _tier3
    from kognita.breaker import schedule_kill_switch_drill
    _open(session, evidence)
    _activate(session, evidence, _tier3(usecase_id="research-summary", agents=("agent",)), signoff_by="Noel Independent")
    schedule_kill_switch_drill(session, "research-summary", due_at=NOW + timedelta(days=30),
        evidence=evidence, actor_id="owner", citation="test drill", now=NOW)
    registry = ToolRegistry()
    calls = []
    registry.register("read", lambda *_: calls.append(True), verb="queue", required_level=4)
    run = Run()
    from kognita._isolation import _step_up
    authenticated = replace(_env(), assurance_level=2)
    _step_up(session, authenticated, evidence=evidence, actor_id=authenticated.principal,
        authenticated_workflow=True, transaction_context={"task": "test"},
        independent_authorization="independent", expires_at=NOW + timedelta(hours=1))
    held = run_governed(session, authenticated, registry=registry, pack=dp.DemoPack(),
                        evidence=evidence, purposes=("research-summary",), run=run, now=NOW, as_of=NOW)
    assert held.outcome is Outcome.HUMAN_APPROVAL
    assert not calls
    if change == "expiry":
        agent.review_date = NOW - timedelta(seconds=1)
    elif change == "deployment":
        approve_deployment(session, "agent", deployment={**_DEPLOYMENT, "instructions": "new"},
                           evidence=evidence, actor_id="owner")
    elif change == "tool":
        registry.register("read", lambda *_: calls.append(True), verb="execute", required_level=5)
    elif change == "step_down":
        autonomy._change(session, agent, 3, evidence, "autonomy-monitor", "AUTONOMY_STEP_DOWN", {})
    resumed = continue_run(session, run.id, {run.approvals_pending[0]: True},
                           registry=registry, evidence=evidence, approver_name="reviewer", now=NOW)
    assert bool(calls) is (change == "none")
    if change != "none":
        assert resumed.outcome is Outcome.DENY
    assert verify_chain(session) > 0


def test_real_tool_declaration_cannot_be_downgraded_by_agent_permissions(session, evidence):
    _agent(session, evidence)
    registry = ToolRegistry()
    calls = []
    registry.register("read", lambda *_: calls.append(True), verb="execute", required_level=5)
    result = run_governed(session, _env(), registry=registry, pack=dp.DemoPack(),
                          evidence=evidence, purposes=("research-summary",), now=NOW, as_of=NOW)
    assert result.outcome is Outcome.DENY
    assert not calls
    assert any(check.check == "S1-TOOL-DECLARATION" for check in result.evaluation.failures())


def test_metrics_from_another_agent_cannot_promote_this_agent(session, engine):
    agent, writer = _promotion_setup(session, engine)
    other = Agent(name="other")
    _samples(session, engine, other, NOW - timedelta(days=1))
    with pytest.raises(ValueError, match="S1-PROMOTION"):
        autonomy.promote(session, "agent", evidence=writer, actor_id="owner", validator="validator", now=NOW)
    assert agent.autonomy_level == 0


def test_registration_identity_requires_evidence_and_cannot_raise_autonomy(session, evidence):
    with pytest.raises(ValueError, match="evidenced owner"):
        register(session, name="agent", owner_exec="owner", purpose="research")
    agent = register(session, name="agent", owner_exec="owner", purpose="research",
                     evidence=evidence, actor_id="owner")
    assert agent.purpose == "research"
    with pytest.raises(ValueError, match="promote"):
        register(session, name="agent", owner_exec="owner", autonomy_level=1,
                 evidence=evidence, actor_id="owner")
    assert agent.autonomy_level == 0


def test_changed_gateway_system_prompt_is_denied_before_upstream(session, engine, evidence):
    from fixtures.agents import register as fixture_register
    from test_gateway import _gateway, _live, _Upstream, PATH
    fixture_register(session, name="dossier-agent", owner_exec="Head of Research Ops")
    _live(session)
    upstream = _Upstream(b"{}")
    gateway = _gateway(engine, evidence, upstream)
    body = json.dumps({"model": "gateway-model", "messages": [
        {"role": "system", "content": "changed behaviour"}, {"role": "user", "content": "hello"}]}).encode()
    response = gateway.handle("POST", PATH, {"agent_name": "dossier-agent"}, body, session=session)
    assert response.status == 403
    assert not upstream.calls
    assert any(check.check == "S1-DEPLOYMENT" for check in response.evaluation.failures())


def test_replay_pins_identity_before_a_later_deployment_approval(session, evidence):
    from kognita.governance import record
    from kognita.replay import replay_decision
    agent = _agent(session, evidence)
    original = _decide(agent)
    record(session, original, evidence=evidence, now=NOW)
    approve_deployment(session, "agent", deployment={**_DEPLOYMENT, "instructions": "changed"},
                       evidence=evidence, actor_id="owner")
    replayed = replay_decision(session, original.request_id, _env(), purposes=("research-summary",))
    assert replayed.outcome is Outcome.ALLOW
    assert replayed.envelope_hash == original.envelope_hash


def test_promotion_requires_the_full_deployment_observation_window(session, engine):
    agent, writer = _promotion_setup(session, engine)
    approve_deployment(session, "agent", deployment=_DEPLOYMENT, evidence=writer, actor_id="owner")
    # A new approval of identical behaviour retains its digest and history.
    agent.tool_allowlist["read"]["system"] = "changed"
    approve_deployment(session, "agent", deployment=_DEPLOYMENT, evidence=writer, actor_id="owner")
    _samples(session, engine, agent, NOW - timedelta(days=1))
    with pytest.raises(ValueError, match="observation window"):
        autonomy.promote(session, "agent", evidence=writer, actor_id="owner", validator="validator", now=NOW)


def test_promotion_refuses_mixed_model_prompt_versions(session, engine):
    agent, writer = _promotion_setup(session, engine)
    _samples(session, engine, agent, NOW - timedelta(days=1))
    correlations = [row.correlation_id for row in session.exec(select(EvidenceEvent)).all()
                    if row.event_type == EventType.TOOL_CALL]
    for index, correlation in enumerate(correlations):
        writer.emit(session, correlation_id=correlation, event_type=EventType.MODEL_CALL,
                    actor_type=ActorType.AGENT, actor_id=agent.name,
                    payload={"model_version": str(index), "prompt_template_version": "1"})
    with pytest.raises(ValueError, match="mixed model or prompt"):
        autonomy.promote(session, "agent", evidence=writer, actor_id="owner", validator="validator", now=NOW)


def test_changing_transition_gates_clears_usecase_activation(session, engine):
    from kognita.usecases import revise
    agent, writer = _promotion_setup(session, engine)
    revised = revise(session, _spec(validators=("Jane Smith", "validator"),
        _autonomy_gates={"1": dict(minimum_sample=1, window_days=7, quality=.5, reliability=.5)}),
        evidence=writer, actor_id="Jane Smith", at=NOW + timedelta(seconds=1))
    assert revised.record["status"] == ""
    with pytest.raises(ValueError, match="S1-PROMOTION"):
        autonomy.promote(session, "agent", evidence=writer, actor_id="owner", validator="validator",
                         now=NOW + timedelta(seconds=1))


@pytest.mark.parametrize("override", [dict(minimum_sample=0), dict(window_days=-1),
                                      dict(quality=1.1), dict(reliability=True)])
def test_invalid_metric_gates_cannot_be_registered(session, evidence, override):
    from kognita.usecases import register as register_usecase
    from kognita.exceptions import KognitaError
    _open(session, evidence)
    gate = dict(minimum_sample=4, window_days=7, quality=.95, reliability=1)
    with pytest.raises(KognitaError, match="S1-AUTONOMY"):
        register_usecase(session, _spec(_autonomy_gates={"1": {**gate, **override}}),
                         evidence=evidence, actor_id="Jane Smith", at=NOW)


def test_deployment_approval_cannot_bypass_the_promotion_gate(session, evidence):
    agent = _agent(session, evidence)
    original_hash = agent.deployment_hash
    agent.autonomy_level = 1
    with pytest.raises(ValueError, match="S1-PROMOTION"):
        approve_deployment(session, "agent", deployment=_DEPLOYMENT, evidence=evidence, actor_id="owner")
    assert agent.deployment_hash == original_hash
    assert _decide(agent).outcome is Outcome.DENY


def test_initial_deployment_cannot_start_above_l0(session, evidence):
    agent = register(session, name="agent", owner_exec="owner")
    agent.autonomy_level = 6
    with pytest.raises(ValueError, match="S1-PROMOTION"):
        approve_deployment(session, "agent", deployment=_DEPLOYMENT, evidence=evidence, actor_id="owner")


def test_failed_evidence_write_does_not_change_approved_deployment(session, evidence, engine):
    agent = _agent(session, evidence)
    digest = agent.deployment_hash
    original = agent.tool_allowlist
    class _Down(EvidenceWriter):
        def emit(self, *args, **kwargs):
            raise ConnectionError("unavailable")
    with pytest.raises(ConnectionError):
        approve_deployment(session, "agent", deployment={**_DEPLOYMENT, "instructions": "changed"},
                           evidence=_Down(engine), actor_id="owner")
    assert agent.deployment_hash == digest
    assert agent.tool_allowlist == original
    assert _decide(agent).outcome is Outcome.ALLOW


def test_old_deployment_outcomes_cannot_promote_current_behaviour(session, engine):
    agent, writer = _promotion_setup(session, engine)
    old = Agent.model_validate(agent.model_dump())
    old.deployment_hash = "superseded"
    _samples(session, engine, old, NOW - timedelta(days=1))
    with pytest.raises(ValueError, match="S1-PROMOTION"):
        autonomy.promote(session, "agent", evidence=writer, actor_id="owner", validator="validator", now=NOW)
    assert agent.autonomy_level == 0
