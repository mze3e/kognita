"""Nested content admission with real authority and unwrapped MCP ingress."""
import pytest

from kognita.envelope import Envelope
from kognita.models import Agent
from kognita.registry import approve_deployment
from kognita.tools import Run
from kognita._fleet import _run_checks
from kognita.vocabulary import Outcome
from sqlmodel import select
from test_mcp import _register, _proxy, _Upstream, _call
from test_classification_floor import _authorize


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Use only explicitly issued credentials and production handlers."""


def _arguments(shape):
    if shape == 'object':
        return {'message': {'text': 'alice@example.com'}}
    if shape == 'list':
        return {'message': [{'text': 'alice@example.com'}]}
    if shape == 'depth':
        value = 'alice@example.com'
        for _ in range(40):
            value = [value]
        return {'message': value}
    if shape == 'width':
        return {'message': ['hello'] * 5000 + ['alice@example.com']}
    if shape == 'size':
        return {'message': 'x' * 70000 + ' alice@example.com'}
    return {'message': 'alice@example.com'}


@pytest.mark.parametrize('shape', ['top', 'object', 'list', 'depth', 'width', 'size'])
@pytest.mark.parametrize('with_run', [False, True])
def test_nested_sensitive_mcp_content_is_denied(session, evidence, engine, shape, with_run):
    _register(session)
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    agent.tool_allowlist = {**agent.tool_allowlist,
        'read_note': {**agent.tool_allowlist['read_note'], 'public_content': True}}
    agent.field_allowlist = {'COLLABORATION': ['content']}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
        evidence=evidence, actor_id=agent.owner_exec)
    did, credential = _authorize(session, evidence)
    headers = dict(agent_name=agent.name, agent_credential=credential, delegation_id=did)
    run = Run()
    if with_run:
        _run_checks(session, run, Envelope(principal='alice', purpose='COLLABORATION',
            tool='read_note', agent_name=agent.name, delegation_id=did), evidence)
        headers['run_id'] = run.id
    backend = _Upstream()
    response = _proxy(engine, evidence, backend).handle('POST', '/mcp', headers,
        _call(arguments=_arguments(shape)), session=session)
    assert response.evaluation.outcome is Outcome.DENY
    assert response.status == 403
    assert backend.calls == []
    citations = {c.citation for c in response.evaluation.failures()}
    assert citations & {'S5-TAINT', 'S5-PUBLIC-CONTENT' if with_run else 'S5-RUN-REQUIRED'}
    if with_run:
        from kognita.models import RunRecord
        assert session.get(RunRecord, run.id).client_data
