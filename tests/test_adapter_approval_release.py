"""Approved re-invocations use real authority without legacy ingress shims."""
import asyncio

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.adapters.binding import Binding, GovernedRefusal
from kognita.adapters.hermes import HermesAdapter
from kognita.adapters.langgraph import LangGraphAdapter
from kognita.adapters.openai_agents import OpenAIAgentsAdapter
from kognita.adapters.pydantic_ai import PydanticAIAdapter
from kognita.approvals import grant
from kognita.authority import revoke
from kognita.models import Agent, Approval, Policy
from kognita.registry import approve_deployment
from kognita.tools import Run, ToolRegistry
from kognita.vocabulary import Outcome
from test_mcp import _Pack
from test_mcp_client_argument_scope import _delegated


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Use only the explicitly issued workload credential."""


@pytest.mark.parametrize('revoke_before_rerun', [False, True])
@pytest.mark.parametrize('path', [
    'binding', 'hermes', 'openai', 'langgraph', 'pydantic',
    'openai-sdk', 'langgraph-sdk', 'pydantic-sdk',
])
def test_approved_adapter_reinvocation_returns_result_once(
        session, evidence, monkeypatch, path, revoke_before_rerun):
    if path.endswith('-sdk'):
        pytest.importorskip({'openai-sdk': 'agents', 'langgraph-sdk': 'langchain_core',
                             'pydantic-sdk': 'pydantic_ai'}[path])
    monkeypatch.setattr('kognita.tools.utcnow', lambda: NOW)
    monkeypatch.setattr('kognita.governance.utcnow', lambda: NOW)
    did, secret = _delegated(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    agent.field_allowlist = {'COLLABORATION': ['$']}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    session.add(Policy(regime='INTERNAL', rule_type='REQUIRES_HUMAN_APPROVAL',
        rule={'tools': ['read_note']}, citation='adapter-approval', effective_from=NOW))
    session.flush()
    binding = Binding(session=session, evidence=evidence, pack=_Pack(), gateway=None,
        registry=ToolRegistry(), principal='alice', purpose='COLLABORATION',
        agent_name=agent.name, purposes=('COLLABORATION',), actor_location='SG',
        subject_type='client', subject_id='123', run=Run(),
        agent_credential=secret, delegation_id=did)
    calls = []

    def read_note(query: str) -> str:
        """Read the approved note."""
        calls.append(query)
        return 'notes'

    if path == 'hermes':
        adapter = HermesAdapter(binding)
        def invoke():
            directive = adapter.pre_tool_call('read_note', {'query': 'hello'})
            if directive is None:
                return read_note('hello')
            return directive
    elif path == 'binding':
        invoke = lambda: binding.wrap(read_note)(query='hello')
    else:
        adapter = {'openai': OpenAIAgentsAdapter, 'langgraph': LangGraphAdapter,
                   'pydantic': PydanticAIAdapter}[path.split('-')[0]](binding)
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
    if path == 'hermes':
        assert invoke()['action'] == 'approve'
    elif path == 'openai-sdk':
        assert invoke() != 'notes'
    else:
        with pytest.raises(GovernedRefusal) as exc:
            invoke()
        assert exc.value.outcome is Outcome.HUMAN_APPROVAL
        assert 'adapter-approval' in exc.value.citations
    assert calls == []
    approval = session.exec(select(Approval)).one()
    grant(session, approval, approver_name='reviewer@example.org', evidence=evidence,
          correlation_id='adapter-review', now=NOW)
    if revoke_before_rerun:
        revoke(session, did, evidence=evidence, actor_id='alice')
        if path == 'hermes':
            directive = invoke()
            assert directive['action'] == 'block'
            assert 'S3-DELEGATION' in directive['message']
        elif path == 'openai-sdk':
            assert invoke() != 'notes'
        else:
            with pytest.raises(GovernedRefusal) as exc:
                invoke()
            assert exc.value.outcome is Outcome.DENY
            assert 'S3-DELEGATION' in exc.value.citations
        assert calls == []
    else:
        assert invoke() == 'notes'
        assert calls == ['hello']
