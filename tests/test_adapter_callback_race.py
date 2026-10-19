"""Concurrent adapter calls retain their own callback and real authority."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest
from sqlmodel import Session, select

from conftest import NOW
from kognita.adapters.binding import Binding
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter
from kognita.models import Agent
from kognita.registry import approve_deployment
from kognita.tools import Run, ToolRegistry
from kognita.vocabulary import Outcome
from test_mcp import _Pack
from test_mcp_client_argument_scope import _delegated


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Do not install legacy ingress shims."""


@pytest.mark.parametrize('registered', [True, False])
def test_concurrent_adapter_calls_keep_own_callback(monkeypatch, registered):
    import kognita.adapters.binding as module

    monkeypatch.setattr('kognita.tools.utcnow', lambda: NOW)
    monkeypatch.setattr('kognita.governance.utcnow', lambda: NOW)
    registry = ToolRegistry()

    def original(*args):
        raise AssertionError('placeholder executed')

    if registered:
        registry.register('read_note', original)
    ready = Barrier(2)
    database_lock = Lock()
    governed = module.run_governed

    def synchronized(*args, **kwargs):
        # Both calls have bound their callbacks before either looks up the tool.
        ready.wait(timeout=10)
        with database_lock:
            return governed(*args, **kwargs)

    monkeypatch.setattr(module, 'run_governed', synchronized)

    def invoke(label):
        engine = make_engine()
        create_all(engine)
        evidence = EvidenceWriter(engine, clock=lambda: NOW)
        with Session(engine) as session:
            did, secret = _delegated(session, evidence)
            agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
            agent.field_allowlist = {'COLLABORATION': ['$']}
            approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                               evidence=evidence, actor_id=agent.owner_exec)
            binding = Binding(session=session, evidence=evidence, pack=_Pack(), gateway=None,
                registry=registry, principal='alice', purpose='COLLABORATION',
                agent_name=agent.name, purposes=('COLLABORATION',), actor_location='SG',
                subject_type='client', subject_id='123', run=Run(),
                agent_credential=secret, delegation_id=did)
            calls = []

            def callback(query):
                calls.append((label, query))
                return label

            result = binding.call_tool('read_note', {'query': label}, callback)
            return label, result, calls

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, ['first', 'second']))
    for label, result, calls in results:
        assert result.released, result
        assert result.outcome is Outcome.ALLOW
        assert result.citations
        assert result.data == label
        assert calls == [(label, label)]
    if registered:
        assert registry.get('read_note').fn is original
