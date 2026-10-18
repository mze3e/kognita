"""Gateway delivery regressions using explicit grants and workload credentials."""
import json

import pytest
from sqlmodel import select

from kognita.models import Agent, EvidenceEvent
from kognita.registry import approve_deployment
from kognita.tools import Run
from kognita.vocabulary import Classification, Outcome
from test_gateway import MODEL, PATH, _body
from test_gateway_run_controls import _setup


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Exercise the production handler with only the explicitly supplied secret."""
    yield


def test_response_above_run_ceiling_is_withheld(session, evidence):
    gateway, backend, headers = _setup(session, evidence)
    backend.response = json.dumps({'model': MODEL, 'choices': [
        {'message': {'content': 'alice@example.com'}}]}).encode()
    response = gateway.handle('POST', PATH, headers, _body('hello'),
        session=session, run=Run(classification_ceiling=Classification.C1))
    assert len(backend.calls) == 1
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    assert any(c.citation == 'classification_ceiling' for c in response.evaluation.basis())
    assert b'alice@example.com' not in response.body


@pytest.mark.parametrize('fields,expected', [
    ([], {}),
    (['choices.message.content'], {'choices': [{'message': {'content': 'answer'}}]}),
])
def test_gateway_response_projects_approved_fields(session, evidence, fields, expected):
    gateway, backend, headers = _setup(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == headers['agent_name'])).one()
    agent.field_allowlist = {'COLLABORATION': fields}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
        evidence=evidence, actor_id=agent.owner_exec)
    backend.response = json.dumps({'model': MODEL, 'choices': [
        {'message': {'content': 'answer', 'private': 'secret'}}]}).encode()
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session)
    assert response.status == 200
    assert json.loads(response.body) == expected
    events = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.correlation_id == response.evaluation.request_id)).all()
    assert any(e.payload.get('citation') == 'S4-FIELDS' for e in events)


def test_omitted_approved_instructions_denied_before_forwarding(session, evidence):
    gateway, backend, headers = _setup(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == headers['agent_name'])).one()
    approve_deployment(session, agent.name,
        deployment={**agent.tool_allowlist['_deployment'], 'instructions': 'approved behavior'},
        evidence=evidence, actor_id=agent.owner_exec)
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session)
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    assert any(c.citation == 'S1-DEPLOYMENT' for c in response.evaluation.basis())
    assert backend.calls == []


def test_degraded_gateway_still_minimizes_and_defers_field_evidence(session, evidence):
    from kognita.vocabulary import FailureMode
    from test_gateway import LOCAL_UPSTREAM, _Toggle, _gateway

    _, backend, headers = _setup(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == headers['agent_name'])).one()
    agent.field_allowlist = {'COLLABORATION': []}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
        evidence=evidence, actor_id=agent.owner_exec)
    session.commit()
    writer = _Toggle(session.get_bind())
    writer.clock = evidence.clock
    gateway = _gateway(session.get_bind(), writer, backend,
        upstream_url=LOCAL_UPSTREAM, provider='api.openai.com',
        failure_mode={'COLLABORATION': FailureMode.DEGRADED})
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session)
    assert response.status == 200
    assert json.loads(response.body) == {}
    writer.down = False
    gateway.handle('GET', PATH, {}, b'')
    events = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.correlation_id == response.evaluation.request_id)).all()
    assert any(e.payload.get('citation') == 'S4-FIELDS' for e in events)
