"""Binding contracts with real authority, outside the legacy fixture allowlist."""
import asyncio
from dataclasses import fields, replace
from datetime import timedelta

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.adapters.binding import Binding, GovernedRefusal
from kognita.adapters.hermes import HermesAdapter
from kognita.adapters.langgraph import LangGraphAdapter
from kognita.adapters.openai_agents import OpenAIAgentsAdapter
from kognita.adapters.pydantic_ai import PydanticAIAdapter
from kognita.authority import Delegation, delegate, grant_to
from kognita.credentials import issue
from kognita.models import Agent
from kognita.registry import approve_deployment
from kognita.tools import Run, ToolRegistry
from kognita.vocabulary import Classification, Outcome
from test_mcp import _Pack, _register


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Disable the global ingress credential shim."""


@pytest.mark.parametrize('path', [
    'hermes', 'openai', 'langgraph', 'pydantic',
    'openai-sdk', 'langgraph-sdk', 'pydantic-sdk',
])
@pytest.mark.parametrize('label', ['external', 'client_data', 'public_content', 'verb'])
def test_adapter_preserves_registered_tool_contract(session, evidence, monkeypatch, path, label):
    if path.endswith('-sdk'):
        pytest.importorskip({'openai-sdk': 'agents', 'langgraph-sdk': 'langchain_core',
                             'pydantic-sdk': 'pydantic_ai'}[path])
    monkeypatch.setattr('kognita.tools.utcnow', lambda: NOW)
    monkeypatch.setattr('kognita.governance.utcnow', lambda: NOW)
    _register(session)
    calls = []

    def read_note(query: str) -> str:
        """Read an approved note."""
        calls.append(query)
        return 'notes'

    def original(*args):
        raise AssertionError('registered placeholder must be replaced')

    def undo(*args):
        pass

    metadata = dict(classification=Classification.C1, description='approved note reader',
                    verb='analyse' if label == 'verb' else 'read',
                    required_level=1 if label == 'verb' else 0, reversible=True,
                    compensation=undo, max_duration=7.0, **({label: True} if label != 'verb' else {}))
    registry = ToolRegistry()
    declared = replace(registry.register('read_note', original, **metadata))
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    agent.field_allowlist = {'COLLABORATION': ['$']}
    permission = {k: v for k, v in metadata.items()
                  if k not in ('classification', 'description', 'compensation')}
    agent.tool_allowlist = {**agent.tool_allowlist, 'read_note': {
        **permission, 'compensation': 'undo', 'permissions': [metadata['verb']], 'system': 'fixture'}}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    scope = {'client': ['123']}
    grant_to('alice', metadata['verb'], timedelta(hours=2), scope, session=session,
             evidence=evidence, actor_id='admin', now=NOW)
    did = delegate(session, Delegation(human='alice', agent=agent.name,
        purpose='COLLABORATION', client='123', allowed_actions=[metadata['verb']], allowed_data=scope,
        valid_from=NOW, valid_until=NOW + timedelta(hours=1), jurisdiction='SG', task_ref='adapter'),
        evidence=evidence, actor_id='alice', now=NOW)
    secret = issue(session, agent.name, evidence=evidence, actor_id=agent.owner_exec)
    binding = Binding(session=session, evidence=evidence, pack=_Pack(), gateway=None,
        registry=registry, principal='alice', purpose='COLLABORATION', agent_name=agent.name,
        purposes=('COLLABORATION',), actor_location='SG', subject_type='client', subject_id='123',
        run=Run(), agent_credential=secret, delegation_id=did)
    decisions = []
    call_tool = binding.call_tool

    def observed_call(*args, **kwargs):
        result = call_tool(*args, **kwargs)
        decisions.append(result)
        return result

    monkeypatch.setattr(binding, 'call_tool', observed_call)
    adapter = {'hermes': HermesAdapter, 'openai': OpenAIAgentsAdapter,
               'langgraph': LangGraphAdapter, 'pydantic': PydanticAIAdapter}[path.split('-')[0]](binding)
    if path == 'hermes':
        directive = adapter.pre_tool_call('read_note', {'query': 'hello'})
        outcome = Outcome.ALLOW if directive is None else Outcome.DENY
        citations = '' if directive is None else directive['message']
    else:
        if path == 'openai-sdk':
            from agents.tool_context import ToolContext
            tool = adapter.function_tool(read_note)
            ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id='one',
                              tool_arguments='{"query":"hello"}')
            invoke = lambda: asyncio.run(tool.on_invoke_tool(ctx, '{"query":"hello"}'))
        elif path == 'langgraph-sdk':
            invoke = lambda: adapter.structured_tool(read_note).invoke({'query': 'hello'})
        elif path == 'pydantic-sdk':
            invoke = lambda: adapter.agent_tool(read_note).function(query='hello')
        else:
            invoke = lambda: adapter.tool(read_note)(query='hello')
        try:
            value = invoke()
            # The Agents SDK renders refusals as error text rather than raising.
            # Assert the real decision below, including its citation and no effect.
            if path == 'openai-sdk' and label in ('verb', 'public_content'):
                assert value != 'notes'
                assert not decisions[-1].released
                outcome = decisions[-1].outcome
                citations = '; '.join(decisions[-1].citations)
            else:
                assert value == 'notes'
                outcome, citations = Outcome.ALLOW, ''
        except GovernedRefusal as exc:
            outcome, citations = exc.outcome, str(exc)
    bound = registry.get('read_note')
    assert {f.name: getattr(bound, f.name) for f in fields(bound) if f.name != 'fn'} == {
        f.name: getattr(declared, f.name) for f in fields(declared) if f.name != 'fn'}
    assert bound.fn is original
    if label == 'verb':
        assert outcome is Outcome.DENY
        assert 'S1-LEVEL' in citations
        assert 'S1-TOOL-DECLARATION' not in citations
        assert calls == []
    elif label == 'public_content':
        assert outcome is Outcome.DENY
        assert 'S5-PUBLIC-CONTENT' in citations
        assert calls == []
    else:
        assert outcome is Outcome.ALLOW, citations
        assert calls == ([] if path == 'hermes' else ['hello'])
        assert binding.run.external is (label == 'external')
