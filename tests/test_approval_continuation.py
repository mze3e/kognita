"""Real authority; deliberately outside the legacy fixture allowlists."""
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.approvals import grant, reject, expire_stale
from kognita.authority import Delegation, delegate, grant_to
from kognita.credentials import issue
from kognita.models import Agent, Approval, EvidenceEvent, Policy, _BlastUsage
from kognita.tools import Run, ToolRegistry, continue_run, run_governed
from kognita.vocabulary import Classification, Outcome
from test_authority_s3 import _setup
from test_gateway import MODEL, PATH, _Upstream, _body, _gateway, _live
from test_mcp import _register, _Pack


def _hold(session, tool):
    session.add(Policy(regime="INTERNAL", rule_type="REQUIRES_HUMAN_APPROVAL",
        rule={"tools": [tool]}, citation="APPROVAL-TEST", effective_from=NOW))
    session.flush()


def _approval(session):
    return session.exec(select(Approval).order_by(Approval.id)).first()


def _resolve(session, evidence, approved=True):
    (grant if approved else reject)(session, _approval(session), approver_name="reviewer",
        evidence=evidence, correlation_id="review", now=NOW)


def _tool(session, evidence, pack):
    _, _, _, env = _setup(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == "worker")).one()
    agent.blast_radius = {"per_minute": 10}
    session.add(agent)
    from kognita.registry import approve_deployment
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist["_deployment"],
        evidence=evidence, actor_id="owner")
    from test_usecases import ensure_live_purposes
    ensure_live_purposes(session, evidence, ["READ"], at=NOW)
    _hold(session, "read_note")
    calls = []
    registry = ToolRegistry()
    registry.register("read_note", lambda *_: calls.append(1) or "ok", classification=Classification.C1)
    def invoke(envelope=env, run=None):
        return run_governed(session, envelope, registry=registry, evidence=evidence,
            pack=_Pack(), purposes=["READ"], as_of=NOW, now=NOW, run=run)
    return env, calls, invoke


def test_run_governed_approve_reinvoke_executes_once(session, evidence, pack):
    _, calls, invoke = _tool(session, evidence, pack)
    held = invoke()
    assert held.outcome is Outcome.HUMAN_APPROVAL, held.evaluation.failures()
    _resolve(session, evidence)
    released = invoke()
    assert released.data == "ok"
    assert released.evaluation.envelope_hash == held.evaluation.envelope_hash
    assert invoke().data is None
    assert calls == [1]


@pytest.mark.parametrize("expired", [False, True])
def test_rejected_or_expired_hold_releases_quota(session, evidence, pack, expired):
    _, calls, invoke = _tool(session, evidence, pack)
    invoke()
    assert session.exec(select(_BlastUsage)).all()
    if expired:
        expire_stale(session, now=NOW + timedelta(days=3), evidence=evidence)
    else:
        _resolve(session, evidence, False)
    assert session.exec(select(_BlastUsage)).all() == []
    assert any(e.payload.get("action") == "BLAST_RADIUS_HOLD_RELEASED"
               for e in session.exec(select(EvidenceEvent)).all())
    assert invoke().data is None
    assert calls == []


def test_changed_tool_request_does_not_match(session, evidence, pack):
    env, calls, invoke = _tool(session, evidence, pack)
    invoke()
    _resolve(session, evidence)
    changed = invoke(replace(env, arguments={"query": "different"}))
    assert changed.approval_required
    assert calls == []


def _gateway_call(session, evidence, engine):
    _live(session)
    _register(session)
    scope = {"client": ["123"], "model": [MODEL]}
    grant_to("alice", "read", timedelta(hours=2), scope, session=session,
             evidence=evidence, actor_id="admin", now=NOW)
    did = delegate(session, Delegation(human="alice", agent="dossier-agent",
        purpose="COLLABORATION", client="123", allowed_actions=["read"],
        allowed_data=scope, valid_from=NOW, valid_until=NOW + timedelta(hours=1),
        jurisdiction="SG", task_ref="meeting"), evidence=evidence, actor_id="alice", now=NOW)
    credential = issue(session, "dossier-agent", evidence=evidence, actor_id="Head of Research Ops")
    _hold(session, "model_call")
    backend = _Upstream(b'{"model":"gateway-model","choices":[]}')
    gateway = _gateway(engine, evidence, backend)
    def invoke(text="hello"):
        return gateway.handle("POST", PATH, {"agent_name": "dossier-agent",
            "agent_credential": credential, "delegation_id": did, "assurance_level": "2"}, _body(text), session=session)
    return backend, invoke


def test_gateway_approve_reinvoke_forwards_once(session, evidence, engine):
    backend, invoke = _gateway_call(session, evidence, engine)
    held = invoke()
    assert held.evaluation.outcome is Outcome.HUMAN_APPROVAL
    assert backend.calls == []
    _resolve(session, evidence)
    assert invoke().status == 200
    assert invoke().status == 403
    assert len(backend.calls) == 1


def test_gateway_changed_request_does_not_match(session, evidence, engine):
    backend, invoke = _gateway_call(session, evidence, engine)
    invoke()
    _resolve(session, evidence)
    assert invoke("changed").status == 403
    assert backend.calls == []


@pytest.mark.parametrize("expired", [False, True])
def test_gateway_rejected_or_expired_approval_denies(session, evidence, engine, expired):
    backend, invoke = _gateway_call(session, evidence, engine)
    invoke()
    if expired:
        approval = _approval(session)
        approval.expires_at = NOW - timedelta(seconds=1)
        session.add(approval)
        session.flush()
    else:
        _resolve(session, evidence, False)
    assert invoke().evaluation.outcome is Outcome.DENY
    assert backend.calls == []


def test_usage_counters_are_evidenced_but_do_not_change_hash(session, evidence, pack):
    from kognita.envelope import envelope_hash
    _, _, invoke = _tool(session, evidence, pack)
    held = invoke().evaluation
    changed = {**held.attributes, "_s4_usage": [[NOW.isoformat(), [], 0, "other"]]}
    assert envelope_hash(held.envelope, changed, held.checks) == held.envelope_hash
    event = next(e for e in session.exec(select(EvidenceEvent)).all()
                 if e.correlation_id == held.request_id and "attributes" in e.payload)
    assert "_s4_usage" in event.payload["attributes"]


def test_snapshot_expires_hold_and_releases_quota(session, evidence, pack):
    from kognita.governance import load_snapshot
    _, _, invoke = _tool(session, evidence, pack)
    invoke()
    load_snapshot(session, as_of=NOW + timedelta(days=3), _evidence=evidence, _agent_name="worker")
    assert session.exec(select(_BlastUsage)).all() == []


@pytest.mark.parametrize("continue_first", [False, True])
def test_continue_run_and_reinvoke_share_one_use(session, evidence, pack, continue_first):
    env, calls, invoke = _tool(session, evidence, pack)
    run = Run()
    held = invoke(run=run)
    assert held.outcome is Outcome.HUMAN_APPROVAL, held.evaluation.failures()
    registry = ToolRegistry()
    registry.register("read_note", lambda *_: calls.append(1) or "ok", classification=Classification.C1)
    def resume():
        return continue_run(session, run.id, {_approval(session).id: True},
            approver_name="reviewer", evidence=evidence, registry=registry, now=NOW)
    if continue_first:
        assert resume().data == "ok"
        assert invoke(run=run).outcome is Outcome.DENY
    else:
        _resolve(session, evidence)
        assert invoke(run=run).data == "ok"
        assert resume().outcome is Outcome.DENY
    assert calls == [1]


def test_gateway_l4_model_call_continues_after_approval(session, evidence, engine):
    from copy import deepcopy
    from kognita.autonomy import _change
    from kognita.registry import approve_deployment
    from kognita.usecases import load_register, revise, approve_rating, activate
    from test_usecases import _tier3

    backend, invoke = _gateway_call(session, evidence, engine)
    entry = load_register(session, as_of=NOW).get("COLLABORATION")
    spec = _tier3(usecase_id="COLLABORATION", purpose="COLLABORATION",
        approved_models=tuple(entry.record["approved_models"]))
    revise(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    approve_rating(session, spec.usecase_id, spec.control_function, evidence=evidence, at=NOW)
    activate(session, spec.usecase_id, evidence=evidence, actor_id=spec.control_function, at=NOW, status="live", signoff_by="Independent Risk")
    agent = session.exec(select(Agent).where(Agent.name == "dossier-agent")).one()
    permissions = deepcopy(agent.tool_allowlist)
    permissions["model_call"]["required_level"] = 4
    agent.tool_allowlist = permissions
    session.add(agent)
    _change(session, agent, 4, evidence, agent.owner_exec, "AUTONOMY_PROMOTED", {})
    approve_deployment(session, agent.name, deployment=permissions["_deployment"],
        evidence=evidence, actor_id=agent.owner_exec)
    from kognita.breaker import run_kill_switch_drill
    run_kill_switch_drill(session, "COLLABORATION", agent.name, evidence=evidence,
        actor_id=agent.owner_exec, citation="L4 drill", fallback="manual", now=NOW)
    from kognita._isolation import _step_up
    preliminary = invoke()
    _step_up(session, preliminary.evaluation.envelope, evidence=evidence,
        actor_id="alice", authenticated_workflow=True, transaction_context="model request",
        independent_authorization="risk", expires_at=NOW + timedelta(minutes=5))
    held = invoke()
    assert held.evaluation.outcome is Outcome.HUMAN_APPROVAL, held.evaluation.failures()
    assert any(c.check == "S1-L4-APPROVAL" for c in held.evaluation.checks)
    _resolve(session, evidence)
    assert invoke().status == 200
    assert invoke().status == 403
    assert len(backend.calls) == 1

