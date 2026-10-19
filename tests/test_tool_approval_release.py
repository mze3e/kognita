"""Exact approval release through tools and unwrapped MCP ingress."""
from dataclasses import replace

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.approvals import grant
from kognita.governance import record
from kognita.models import Agent, Approval, _ApprovalContinuation
from kognita.registry import approve_deployment
from kognita.tools import Run
from kognita.vocabulary import Outcome
from test_approval_continuation import _hold, _tool


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Use explicit credentials and production ingress handlers."""


def _invocation(session, evidence, engine, pack, ingress):
    run = Run()
    if ingress == 'tool':
        _, calls, invoke = _tool(session, evidence, pack)
        agent = session.exec(select(Agent).where(Agent.name == 'worker')).one()
        agent.field_allowlist = {'READ': ['content']}
        approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
            evidence=evidence, actor_id=agent.owner_exec)
        return calls, lambda: invoke(run=run)

    from test_mcp_client_argument_scope import _delegated
    from test_mcp import _proxy, _Upstream, _call
    from kognita._fleet import _run_checks
    from kognita.envelope import Envelope
    did, credential = _delegated(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    agent.field_allowlist = {'COLLABORATION': ['content']}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
        evidence=evidence, actor_id=agent.owner_exec)
    _run_checks(session, run, Envelope(principal='alice', purpose='COLLABORATION',
        tool='read_note', agent_name=agent.name, delegation_id=did), evidence)
    _hold(session, 'read_note')
    backend = _Upstream()
    proxy = _proxy(engine, evidence, backend)
    def invoke():
        return proxy.handle('POST', '/mcp', {'agent_name': agent.name,
            'agent_credential': credential, 'delegation_id': did, 'run_id': run.id},
            _call(), session=session)
    return backend.calls, invoke


@pytest.mark.parametrize('ingress', ['tool', 'mcp'])
def test_current_pending_approval_blocks_older_grant(session, evidence, engine, pack, ingress):
    calls, invoke = _invocation(session, evidence, engine, pack, ingress)
    held = invoke().evaluation
    assert held.outcome is Outcome.HUMAN_APPROVAL, held.failures()
    binding = session.get(_ApprovalContinuation, held.envelope_hash)
    old = session.get(Approval, binding.approval_id)
    grant(session, old, approver_name='reviewer', evidence=evidence,
        correlation_id='review', now=NOW)
    record(session, replace(held, request_id='new-decision', decision_id=None),
        evidence=evidence, now=NOW)
    current = session.get(_ApprovalContinuation, held.envelope_hash, populate_existing=True)
    assert current.approval_id != old.id
    result = invoke()
    assert calls == []
    assert result.evaluation.outcome is Outcome.HUMAN_APPROVAL
    assert result.evaluation.basis()
    session.refresh(current)
    assert not current.consumed


@pytest.mark.parametrize('ingress', ['tool', 'mcp'])
def test_approved_continuation_executes_exactly_once(session, evidence, engine, pack, ingress):
    calls, invoke = _invocation(session, evidence, engine, pack, ingress)
    held = invoke().evaluation
    assert held.outcome is Outcome.HUMAN_APPROVAL, held.failures()
    binding = session.get(_ApprovalContinuation, held.envelope_hash)
    grant(session, session.get(Approval, binding.approval_id), approver_name='reviewer',
        evidence=evidence, correlation_id='review', now=NOW)
    invoke()
    assert len(calls) == 1
    session.refresh(binding)
    assert binding.consumed
    repeated = invoke()
    assert repeated.evaluation.outcome is Outcome.DENY
    assert 'APPROVAL-CONTINUATION' in {c.citation for c in repeated.evaluation.basis()}
    assert len(calls) == 1
