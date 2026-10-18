"""G1 checks through real HTTP ingresses with explicit authority and credentials."""
import json
from datetime import timedelta
from http.client import HTTPConnection
from threading import Thread

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.authority import Delegation, delegate, grant_to
from kognita.credentials import issue
from kognita.models import EvidenceEvent, GovernanceDecision
from kognita.registry import register, approve_deployment
from kognita.vocabulary import Outcome
from fixtures.agents import _DEPLOYMENT
from test_gateway import MODEL, PATH, _gateway, _body, _live, _Upstream
from test_mcp import _proxy, _call, _Upstream as _McpUpstream


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Use the production ingress handlers and the explicitly issued secret."""
    yield


def _setup(session, evidence):
    agent = register(session, name='dossier-agent', owner_exec='owner')
    agent.purpose = 'Research'
    agent.use_cases = ['COLLABORATION']
    agent.permitted_systems = ['fixture']
    agent.tool_allowlist = {tool: dict(verb='read', required_level=0,
        permissions=['read'], system='fixture') for tool in ('model_call', 'read_note', 'notes/read_note')}
    agent.environment = 'test'
    agent.principal = 'alice'
    agent.review_date = NOW + timedelta(days=90)
    agent.field_allowlist = {'COLLABORATION': ['content.text', 'content.type']}
    _live(session)
    scope = {'client': ['123'], 'model': [MODEL]}
    grant_to('alice', 'read', timedelta(hours=2), scope, session=session,
             evidence=evidence, actor_id='admin', now=NOW)
    did = delegate(session, Delegation(human='alice', agent=agent.name,
        purpose='COLLABORATION', client='123', allowed_actions=['read'], allowed_data=scope,
        valid_from=NOW, valid_until=NOW + timedelta(hours=1), jurisdiction='SG', task_ref='g1'),
        evidence=evidence, actor_id='alice', now=NOW)
    cred = issue(session, agent.name, evidence=evidence, actor_id='owner')
    return agent, {'agent_name': agent.name, 'agent_credential': cred, 'delegation_id': did}


def _approve(session, evidence, agent, deployment=None):
    approve_deployment(session, agent.name, deployment=deployment or dict(_DEPLOYMENT),
                       evidence=evidence, actor_id='owner')
    session.commit()


def _http(ingress, path, headers, body):
    server = ingress._make_server('127.0.0.1', 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection('127.0.0.1', server.server_address[1], timeout=10)
        try:
            connection.request('POST', path, headers=headers, body=body)
            response = connection.getresponse()
            raw = response.read()
            return response.status, raw
        finally:
            connection.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _decision(session):
    session.expire_all()
    return session.exec(select(GovernanceDecision).order_by(GovernanceDecision.id.desc())).first()


@pytest.mark.parametrize('field,value', [('purpose', None), ('purpose', ' '),
    ('use_cases', None), ('permitted_systems', None), ('tool_allowlist', None),
    ('autonomy_level', None), ('review_date', None), ('deployment_hash', None),
    ('environment', None), ('environment', ' '), ('principal', None), ('principal', ' ')])
def test_missing_identity_denied_http(session, evidence, engine, field, value):
    agent, headers = _setup(session, evidence)
    setattr(agent, field, value)
    # Approve the incomplete record so a stale digest cannot hide the gap.
    if field not in ('tool_allowlist', 'deployment_hash'):
        _approve(session, evidence, agent)
    else:
        session.commit()
    backend = _Upstream(b'{}')
    status, _ = _http(_gateway(engine, evidence, backend), PATH, headers, _body('hello'))
    assert status == 403
    assert backend.calls == []
    decision = _decision(session)
    assert decision.outcome is Outcome.DENY
    assert all(c['citation'] for c in decision.checks)
    assert any(c['check'].startswith('S1-') and c['result'] == 'fail' for c in decision.checks)


@pytest.mark.parametrize('kind', ['gateway', 'mcp'])
@pytest.mark.parametrize('change', ['none', 'expiry', 'deployment'])
def test_review_and_deployment_http(session, evidence, engine, kind, change):
    agent, headers = _setup(session, evidence)
    if change == 'expiry':
        agent.review_date = NOW - timedelta(seconds=1)
    _approve(session, evidence, agent)
    if change == 'deployment':
        agent.tool_allowlist = {**agent.tool_allowlist, '_deployment': {
            **_DEPLOYMENT, 'instructions': 'unapproved change'}}
        session.commit()
    backend = _Upstream(json.dumps({'model': MODEL, 'choices': []}).encode()) if kind == 'gateway' else _McpUpstream()
    ingress = _gateway(engine, evidence, backend) if kind == 'gateway' else _proxy(engine, evidence, backend)
    status, raw = _http(ingress, PATH if kind == 'gateway' else '/mcp', headers,
                        _body('hello') if kind == 'gateway' else _call())
    decision = _decision(session)
    if change == 'none':
        assert status == 200, raw
        assert decision.outcome is Outcome.ALLOW
        assert len(backend.calls) == 1
    else:
        assert backend.calls == []
        assert decision.outcome is Outcome.DENY
        rule = 'S1-REVIEW' if change == 'expiry' else 'S1-DEPLOYMENT'
        assert any(c['check'] == rule and c['result'] == 'fail' for c in decision.checks)
        agent.review_date = NOW + timedelta(days=90)
        _approve(session, evidence, agent, agent.tool_allowlist['_deployment'])
        status, raw = _http(ingress, PATH if kind == 'gateway' else '/mcp', headers,
                            _body('hello') if kind == 'gateway' else _call())
        assert status == 200, raw
        assert _decision(session).outcome is Outcome.ALLOW
        assert len(backend.calls) == 1


def test_deployment_environment_provider_key_is_flagged(session, evidence):
    agent, _ = _setup(session, evidence)
    _approve(session, evidence, agent, {**_DEPLOYMENT, 'environment': {
        'OPENAI_API_KEY': 'manifest-private-secret'}})
    events = session.exec(select(EvidenceEvent)).all()
    assert any(e.payload.get('finding') == 'provider-key finding' for e in events)
    assert 'manifest-private-secret' not in json.dumps([e.payload for e in events])
    assert agent.tool_allowlist['_deployment']['environment']['OPENAI_API_KEY'] == '[REDACTED]'


def test_blast_radius_daily_action_deny_http(session, evidence, engine):
    agent, headers = _setup(session, evidence)
    agent.blast_radius = {'max_daily_actions': 1}
    _approve(session, evidence, agent)
    backend = _Upstream(json.dumps({'model': MODEL, 'choices': []}).encode())
    ingress = _gateway(engine, evidence, backend)
    assert _http(ingress, PATH, headers, _body('hello'))[0] == 200
    assert _http(ingress, PATH, headers, _body('hello'))[0] == 403
    assert len(backend.calls) == 1
    decision = _decision(session)
    assert decision.outcome is Outcome.DENY
    assert any(c['check'] == 'S4-DAILY-ACTIONS' and c['result'] == 'fail' for c in decision.checks)
