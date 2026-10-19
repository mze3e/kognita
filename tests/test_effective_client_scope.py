"""Effective client scope uses real authority without legacy fixture shims."""
import pytest
from sqlmodel import select

from kognita.models import Agent, EvidenceEvent, RunRecord, _AnomalyDay, _AnomalyWindow, _BlastUsage
from kognita.registry import approve_deployment
from kognita.tools import Run
from kognita.vocabulary import Outcome
from test_mcp_client_argument_scope import _delegated


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    yield


@pytest.mark.parametrize('limit', [0, 1])
def test_mapped_mcp_client_scope(session, evidence, engine, limit):
    from test_mcp import _proxy, _Upstream, _call
    did, credential = _delegated(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    agent.tool_allowlist = {**agent.tool_allowlist, 'read_note': {
        **agent.tool_allowlist['read_note'], '_subject_paths': {'client': ['account']}}}
    agent.blast_radius = {'max_clients': limit}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    from kognita.envelope import Envelope
    from kognita._fleet import _run_checks
    run = Run()
    _run_checks(session, run, Envelope(principal='alice', purpose='COLLABORATION',
        tool='read_note', agent_name=agent.name, delegation_id=did), evidence)
    backend = _Upstream()
    response = _proxy(engine, evidence, backend).handle('POST', '/mcp',
        {'agent_name': agent.name, 'agent_credential': credential, 'delegation_id': did, 'run_id': run.id},
        _call(arguments={'account': '123'}), session=session)
    ev = response.evaluation
    assert ev.attributes['_s4_clients'] == ['123']
    assert ev.envelope.all_subjects()['client'] == '123'
    assert ev.outcome is (Outcome.DENY if limit == 0 else Outcome.ALLOW)
    assert len(backend.calls) == limit
    _assert_scope(session, ev, '123', limit)
    assert session.get(RunRecord, run.id, populate_existing=True).client_data


@pytest.mark.parametrize('limit', [0, 1])
def test_mapped_gateway_client_scope(session, evidence, limit):
    from test_gateway_run_controls import _setup
    from test_gateway import MODEL, PATH, _body
    from test_classification_floor import _authorize
    gateway, backend, headers = _setup(session, evidence,
        _subject_paths={'client': ['model_version']})
    did, credential = _authorize(session, evidence, client=MODEL, subject_scope={'model': [MODEL]})
    headers.update(delegation_id=did, agent_credential=credential)
    agent = session.exec(select(Agent).where(Agent.name == headers['agent_name'])).one()
    agent.blast_radius = {'max_clients': limit}
    agent.field_allowlist = {'COLLABORATION': []}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    run = Run()
    response = gateway.handle('POST', PATH, {**headers, 'model_version': MODEL},
        _body('hello'), session=session, run=run)
    assert response.evaluation.attributes['_s4_clients'] == [MODEL]
    assert response.evaluation.outcome is (Outcome.DENY if limit == 0 else Outcome.ALLOW), response.evaluation.failures()
    assert len(backend.calls) == limit
    assert run.client_data
    _assert_scope(session, response.evaluation, MODEL, limit)


@pytest.mark.parametrize('limit', [0, 1])
def test_mapped_governed_client_scope(session, evidence, engine, limit):
    from conftest import NOW
    from kognita.envelope import Envelope
    from kognita.tools import ToolRegistry, run_governed
    from test_mcp import _proxy, _Upstream
    did, credential = _delegated(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    agent.tool_allowlist = {**agent.tool_allowlist, 'read_note': {
        **agent.tool_allowlist['read_note'], '_subject_paths': {'client': ['account']}}}
    agent.blast_radius = {'max_clients': limit}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    calls = []
    registry = ToolRegistry()
    registry.register('read_note', lambda *args: calls.append(args) or {})
    run = Run()
    result = run_governed(session, Envelope(principal='alice', purpose='COLLABORATION',
        agent_name=agent.name, delegation_id=did, actor_location='SG', tool='read_note',
        arguments={'account': '123'}, context={'_agent_credential': credential}),
        registry=registry, evidence=evidence, pack=_proxy(engine, evidence, _Upstream()).pack,
        run=run, now=NOW, as_of=NOW)
    assert result.evaluation.attributes['_s4_clients'] == ['123']
    assert result.evaluation.outcome is (Outcome.DENY if limit == 0 else Outcome.ALLOW)
    assert len(calls) == limit
    assert run.client_data
    _assert_scope(session, result.evaluation, '123', limit)


def _assert_scope(session, evaluation, client, limit):
    assert evaluation.envelope.context['_effective_subjects']['client'] == [client]
    assert evaluation.envelope.all_subjects()['client'] == client
    if limit == 0:
        assert 'S4-MAX-CLIENTS' in [c.citation for c in evaluation.failures()]
    day = session.exec(select(_AnomalyDay).where(
        _AnomalyDay.agent_name == evaluation.envelope.agent_name)).one()
    assert day.body['clients'] == [client]
    window = session.get(_AnomalyWindow, evaluation.envelope.agent_name)
    assert {row[3] for row in window.body} == {client}
    assert sum(row[1] for row in window.body) == 1
    usage = session.get(_BlastUsage, evaluation.request_id)
    assert (usage.clients if usage else []) == ([client] if limit else [])
    events = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.correlation_id == evaluation.request_id)).all()
    assert any(e.payload.get('attributes', {}).get('_s4_clients') == [client] for e in events)


def test_mapped_and_named_clients_share_one_scope(session, evidence, engine):
    from conftest import NOW
    from kognita.envelope import Envelope
    from kognita.tools import ToolRegistry, run_governed
    from test_mcp import _proxy, _Upstream
    did, credential = _delegated(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    agent.tool_allowlist = {**agent.tool_allowlist, 'read_note': {
        **agent.tool_allowlist['read_note'], '_subject_paths': {'client': ['account']}}}
    agent.blast_radius = {'max_clients': 1}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    calls = []
    registry = ToolRegistry()
    registry.register('read_note', lambda *args: calls.append(args) or {})
    run = Run()
    result = run_governed(session, Envelope(principal='alice', purpose='COLLABORATION',
        agent_name=agent.name, delegation_id=did, actor_location='SG', tool='read_note',
        arguments={'account': '123', 'client': ['123', '456']},
        context={'_agent_credential': credential, '_effective_subjects': {'client': []},
                 '_subject_resolution_valid': True}),
        registry=registry, evidence=evidence, pack=_proxy(engine, evidence, _Upstream()).pack,
        run=run, now=NOW, as_of=NOW)
    ev = result.evaluation
    assert ev.attributes['_s4_clients'] == ['123', '456']
    assert ev.envelope.context['_effective_subjects']['client'] == ['123', '456']
    assert ev.outcome is Outcome.DENY
    assert {'S3-DELEGATION', 'S4-MAX-CLIENTS'} <= {c.citation for c in ev.failures()}
    assert calls == []
    assert run.client_data
    day = session.exec(select(_AnomalyDay).where(_AnomalyDay.agent_name == agent.name)).one()
    assert day.body['clients'] == ['123', '456']
    window = session.get(_AnomalyWindow, agent.name)
    assert {row[3] for row in window.body} == {'123', '456'}
    assert sum(row[1] for row in window.body) == 1
