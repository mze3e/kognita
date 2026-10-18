"""Review slice regressions using real authority, outside the legacy allowlist."""
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlmodel import Session, select

from conftest import NOW
from fixtures.agents import register
from kognita._isolation import _assurance_checks, _step_up, _tool_checks
from kognita.anomaly import _observe
from kognita.authority import Delegation, delegate, grant_to
from kognita.canonical import canonical_hash
from kognita.credentials import issue, verify
from kognita.envelope import Envelope, Evaluation
from kognita.governance import decide, load_snapshot
from kognita.models import Agent, EvidenceEvent
from kognita.tools import Run, ToolRegistry, run_governed
from kognita.usecases import load_register
from kognita.vocabulary import ActorType, EventType, Outcome
from test_usecases import _live_purpose


def _setup(session, evidence):
    register(session, name='dossier-agent', owner_exec='owner')
    _live_purpose(session, evidence, 'COLLABORATION')
    scope = {'client': ['123']}
    grant_to('alice', 'read', timedelta(hours=2), scope, session=session,
             evidence=evidence, actor_id='admin', now=NOW)
    did = delegate(session, Delegation(human='alice', agent='dossier-agent',
        purpose='COLLABORATION', client='123', allowed_actions=['read'], allowed_data=scope,
        valid_from=NOW, valid_until=NOW + timedelta(hours=1), jurisdiction='SG', task_ref='misc'),
        evidence=evidence, actor_id='alice', now=NOW)
    secret = issue(session, 'dossier-agent', evidence=evidence, actor_id='owner')
    env = Envelope(principal='alice', purpose='COLLABORATION', tool='read_note',
        agent_name='dossier-agent', delegation_id=did, actor_location='SG',
        subject_type='client', subject_id='123', context={'_agent_credential': secret})
    return env, secret


def test_22_decide_requires_register(session, evidence):
    env, _ = _setup(session, evidence)
    snap = load_snapshot(session, as_of=NOW)
    assert decide(env, snap, usecases=load_register(session, as_of=NOW), as_of=NOW).allowed
    result = decide(env, snap, purposes=['COLLABORATION'], as_of=NOW)
    assert result.outcome == Outcome.DENY
    assert any(c.check == 'USE_CASE_REGISTERED' for c in result.failures())


def test_24_forged_step_up_is_not_host_authorization(session, evidence):
    env, _ = _setup(session, evidence)
    env = replace(env, assurance_level=2)
    registry = ToolRegistry()
    registry.register('read_note', lambda *_: None, verb='execute', required_level=5)
    payload = dict(action='STEP_UP', citation='S5-STEP-UP', principal='alice',
        authenticated_workflow=True, transaction_context={'amount': 10},
        independent_authorization='checker', expires_at=(NOW + timedelta(hours=1)).isoformat())
    evidence.emit(session, correlation_id='step-up:' + canonical_hash(env.to_dict()),
        event_type=EventType.POLICY_CHANGE, actor_type=ActorType.SYSTEM,
        actor_id='governance-pdp', payload=payload)
    checks = _tool_checks(session, env, registry.get('read_note'), Run(), evidence, at=NOW)
    assert any(c.citation == 'S5-STEP-UP' for c in checks)
    _step_up(session, env, evidence=evidence, actor_id='alice', authenticated_workflow=True,
        transaction_context={'amount': 10}, independent_authorization='checker',
        expires_at=NOW + timedelta(hours=1))
    assert not _tool_checks(session, env, registry.get('read_note'), Run(), evidence, at=NOW)


def test_26_rolling_volume_crosses_midnight(session, evidence):
    env, _ = _setup(session, evidence)
    start = NOW.replace(hour=23, minute=59)
    for i in range(1001):
        at = start + timedelta(seconds=120 if i >= 500 else 0)
        evaluation = Evaluation(request_id=str(i), outcome=Outcome.ALLOW, checks=(),
            attributes={}, envelope=env, as_of=at, envelope_hash='')
        result = _observe(session, evaluation, evidence, at)
        if i < 1000:
            assert result.allowed
    assert result.outcome == Outcome.DENY
    assert result.checks[-1].citation == 'S6-VOLUME'


def test_28_no_run_cannot_raise_assurance(session, evidence):
    env, _ = _setup(session, evidence)
    assert _assurance_checks(session, env, None) == []
    checks = _assurance_checks(session, replace(env, assurance_level=2), None)
    assert any(c.citation == 'S5-ASSURANCE-RELAY' for c in checks)


@pytest.mark.parametrize('mode', ['retire', 'run'])
def test_29_revoke_credentials_and_run_authority(session, evidence, mode):
    env, secret = _setup(session, evidence)
    if mode == 'retire':
        from kognita.usecases import retire, revise
        from test_usecases import _spec
        revise(session, _spec(usecase_id=env.purpose, approved_models=(), agents=(env.agent_name,)),
               evidence=evidence, actor_id='Jane Smith', at=NOW)
        retire(session, env.purpose, evidence=evidence, actor_id='Jane Smith', at=NOW)
    else:
        from kognita._fleet import _run_checks
        from kognita.interaction import open_interaction, attach
        from kognita.review import define_human_boundary
        from kognita.intervention import revoke_run
        from kognita.models import _Authority
        define_human_boundary(session, env.purpose, intervention='before output', who='alice',
            must_review='facts', evidence_seen='references', on_reject='stop',
            minimum_review_seconds=0, evidence=evidence, now=NOW)
        interaction = open_interaction(session, client_id='123', use_case=env.purpose,
                                       evidence=evidence, now=NOW)
        run = Run()
        assert not any(c.result.value == 'fail' for c in _run_checks(session, run, env, evidence))
        attach(session, interaction.id, run_id=run.id, evidence=evidence)
        revoke_run(session, run.id, evidence=evidence, human='alice', citation='S3-AUTHORITY',
                   agent_name=env.agent_name, now=NOW)
        assert session.get(_Authority, env.delegation_id, populate_existing=True).ended
    assert not verify(session, env.agent_name, secret)
    assert any(e.payload.get('action') == 'CREDENTIAL_REVOKED'
               for e in session.exec(select(EvidenceEvent)))


def test_33_affected_clients_half_open_boundary(session, evidence, engine):
    from kognita.breaker import affected_clients
    from kognita.evidence import EvidenceWriter
    from test_breaker import _received
    for label, at in [('before', NOW - timedelta(microseconds=1)), ('start', NOW),
                      ('inside', NOW + timedelta(seconds=1)), ('end', NOW + timedelta(seconds=2))]:
        _received(session, EvidenceWriter(engine, clock=lambda at=at: at), label, 'COLLABORATION')
    result = affected_clients(session, model='1', start=NOW, end=NOW + timedelta(seconds=2))
    assert {r['client_id'] for r in result['clients']} == {'start', 'inside'}


def test_34_snapshot_refreshes_cached_agent(session, evidence, engine):
    env, _ = _setup(session, evidence)
    session.commit()
    cached = session.exec(select(Agent).where(Agent.name == env.agent_name)).one()
    with Session(engine) as other:
        agent = other.get(Agent, cached.id)
        from kognita.registry import approve_deployment
        agent.tool_allowlist = {k: v for k, v in agent.tool_allowlist.items() if k != 'read_note'}
        approve_deployment(other, agent.name, deployment=agent.tool_allowlist['_deployment'],
                           evidence=evidence, actor_id='owner')
        changed_hash = agent.deployment_hash
        other.commit()
    snap = load_snapshot(session, as_of=NOW)
    assert 'read_note' not in snap.agents[env.agent_name].tool_allowlist
    assert snap.agents[env.agent_name].deployment_hash == changed_hash
    assert decide(env, snap, usecases=load_register(session, as_of=NOW), as_of=NOW).outcome == Outcome.DENY


def test_23_library_requires_issued_credential(session, evidence):
    env, secret = _setup(session, evidence)
    registry = ToolRegistry()
    calls = []
    registry.register('read_note', lambda *_: calls.append(True), required_level=0)
    class _Pack:
        def load_subjects(self, envelope, session): return {}
        def resolve_attributes(self, envelope, subjects): return {}
        def rules(self): return {}
    def call(envelope, snapshot=None):
        return run_governed(session, envelope, registry=registry, evidence=evidence,
                            pack=_Pack(), now=NOW, as_of=NOW, snapshot=snapshot)
    snap = load_snapshot(session, as_of=NOW)
    register = load_register(session, as_of=NOW)
    # decide() stays a pure, replayable PDP; the credential gate is at execution.
    assert decide(env, snap, usecases=register, as_of=NOW).allowed
    result = call(replace(env, context={}))
    assert result.outcome == Outcome.DENY
    assert any(c.citation == 'S2-CREDENTIAL' for c in result.evaluation.failures())
    assert calls == []
    assert call(env).outcome == Outcome.ALLOW
    assert calls == [True]
    assert secret not in str(env.to_dict())
    assert secret not in str([e.payload for e in session.exec(select(EvidenceEvent))])
    from kognita.credentials import revoke
    revoke(session, env.agent_name, secret, evidence=evidence, actor_id='owner')
    assert call(env, snapshot=snap).outcome == Outcome.DENY
    assert calls == [True]
