"""Runless MCP isolation with real delegation and unwrapped ingress."""
import pytest
from sqlmodel import select

from kognita.models import Agent, GovernanceDecision
from kognita.registry import approve_deployment
from kognita.vocabulary import Outcome
from test_mcp import _call, _proxy, _Upstream
from test_mcp_client_argument_scope import _delegated


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Keep the production MCP handler intact."""


@pytest.mark.parametrize('source', ['subjects', 'input', 'typed', 'response'])
def test_runless_client_data_requires_run_before_public_handoff(session, evidence, engine, source):
    did, credential = _delegated(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    permissions = {k: dict(v) for k, v in agent.tool_allowlist.items()}
    permissions['draft_publication']['public_content'] = True
    if source == 'subjects':
        permissions['read_note']['_subject_paths'] = {'client': ['account']}
    agent.tool_allowlist = permissions
    agent.field_allowlist = {'COLLABORATION': ['content.text', 'content.type']}
    approve_deployment(session, agent.name, deployment=permissions['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    headers = dict(agent_name=agent.name, agent_credential=credential, delegation_id=did)
    backend = _Upstream(result={'content': [{'type': 'text', 'text':
        'alice@example.com' if source == 'response' else 'ordinary notes'}]})
    proxy = _proxy(engine, evidence, backend)
    arguments = {'account': '123'} if source == 'subjects' else {
        'query': 'alice@example.com' if source == 'input' else 'hello'}
    response = proxy.handle('POST', '/mcp', {**headers, **(
        {'classification': 'C2'} if source == 'typed' else {})},
        _call(arguments=arguments), session=session)
    # Simulate a later public tool receiving only what ingress actually released.
    public = _Upstream()
    later = _proxy(engine, evidence, public).handle('POST', '/mcp', headers,
        _call('draft_publication', {'query': response.body.decode()}), session=session)
    assert b'alice@example.com' not in b''.join(c['body'] for c in public.calls)
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    assert session.get(GovernanceDecision, response.evaluation.decision_id).outcome is Outcome.DENY
    assert 'S5-RUN-REQUIRED' in {c.citation for c in response.evaluation.failures()}
    assert len(backend.calls) == (1 if source == 'response' else 0)
    assert later.evaluation.outcome is Outcome.ALLOW


def test_bound_mcp_response_taint_blocks_public_tool(session, evidence, engine):
    from kognita._fleet import _run_checks
    from kognita.envelope import Envelope
    from kognita.models import RunRecord
    from kognita.tools import Run

    did, credential = _delegated(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    agent.tool_allowlist = {**agent.tool_allowlist, 'draft_publication': {
        **agent.tool_allowlist['draft_publication'], 'public_content': True}}
    agent.field_allowlist = {'COLLABORATION': ['content.text', 'content.type']}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    run = Run()
    _run_checks(session, run, Envelope(principal='alice', purpose='COLLABORATION',
        tool='read_note', agent_name=agent.name, delegation_id=did), evidence)
    headers = dict(agent_name=agent.name, agent_credential=credential,
                   delegation_id=did, run_id=run.id)
    backend = _Upstream(result={'content': [{'type': 'text', 'text': 'alice@example.com'}]})
    response = _proxy(engine, evidence, backend).handle('POST', '/mcp', headers,
        _call(), session=session)
    assert response.status == 200
    assert session.get(RunRecord, run.id, populate_existing=True).client_data
    public = _Upstream()
    response = _proxy(engine, evidence, public).handle('POST', '/mcp', headers,
        _call('draft_publication', {'query': 'ordinary summary'}), session=session)
    assert response.status == 403
    assert 'S5-PUBLIC-CONTENT' in {c.citation for c in response.evaluation.failures()}
    assert not public.calls
