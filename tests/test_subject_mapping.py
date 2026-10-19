"""Real authority regressions, outside the legacy fixture allowlist."""
from dataclasses import replace

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.envelope import Envelope
from kognita.governance import decide, load_snapshot
from kognita.models import Agent
from kognita.tools import ToolRegistry, run_governed
from kognita.vocabulary import Classification, Outcome
from test_mcp import _Pack, _Upstream, _call, _proxy
from test_mcp_client_argument_scope import _delegated


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Override the global wrapper: exercise the real MCP entry point."""
    yield


def test_conflicting_subject_denied_before_execution(session, evidence):
    did, _ = _delegated(session, evidence)
    env = Envelope(principal="alice", agent_name="dossier-agent", purpose="COLLABORATION",
        tool="read_note", actor_location="SG", delegation_id=did,
        subject_type="client", subject_id="456", subjects={"client": "123"})
    called = []
    registry = ToolRegistry()
    registry.register("read_note", lambda *args: called.append(args), classification=Classification.C0)
    result = run_governed(session, env, registry=registry, evidence=evidence,
        pack=_Pack(), purposes=["COLLABORATION"], now=NOW)
    assert result.evaluation.outcome is Outcome.DENY
    assert "S3-SUBJECT-CONFLICT" in [c.citation for c in result.evaluation.failures()]
    assert called == []
    snapshot = load_snapshot(session, as_of=NOW, _agent_name="dossier-agent")
    assert decide(replace(env, agent_name=None), snapshot, purposes=["COLLABORATION"], as_of=NOW).outcome is Outcome.DENY


@pytest.mark.parametrize("paths,arguments,allowed", [
    ({"client": ["account"]}, {"account": "456"}, False),
    ({"client": ["account"]}, {"account": "123"}, True),
    ({"client": ["account"]}, {}, False),
    ({"client": ["customer.number"]}, {"customer": {"number": "123"}}, True),
    ({"client": ["customer.number"]}, {"customer": {"number": "456"}}, False),
    ({"client": ["account"]}, {"account": {"number": "123"}}, False),
    ({"client": ["account"]}, {"account": ["123"]}, False),
    ({"client": ["account"]}, {"account": True}, False),
    ({"client": ["account"]}, {"account": None}, False),
    ({"client": ["account"]}, {"account": ""}, False),
    ({"client": ["account", "customer.number"]}, {"account": "123", "customer": {"number": "456"}}, False),
    ({"client": ["account"]}, {"account": 123}, True),
    ({"region": ["area"]}, {"area": "SG"}, False),
    ({"client": []}, {"account": "123"}, False),
])
def test_mapped_arguments_enforced(session, evidence, engine, paths, arguments, allowed):
    did, credential = _delegated(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == "dossier-agent")).one()
    manifest = dict(agent.tool_allowlist)
    manifest["read_note"] = {**manifest["read_note"], "_subject_paths": paths}
    agent.tool_allowlist = manifest
    session.add(agent)
    session.flush()
    from kognita.registry import approve_deployment
    from fixtures.agents import _DEPLOYMENT
    from kognita.credentials import issue
    approve_deployment(session, agent.name, deployment=_DEPLOYMENT,
        evidence=evidence, actor_id=agent.owner_exec)
    credential = issue(session, agent.name, evidence=evidence, actor_id=agent.owner_exec)
    # Bind client reads so these cases continue to exercise subject mapping.
    from kognita.tools import Run
    from kognita.envelope import Envelope
    from kognita._fleet import _run_checks
    run = Run()
    _run_checks(session, run, Envelope(principal="alice", purpose="COLLABORATION",
        tool="read_note", agent_name=agent.name, delegation_id=did), evidence)
    backend = _Upstream()
    response = _proxy(engine, evidence, backend).handle("POST", "/mcp",
        {"agent_name": "dossier-agent", "agent_credential": credential, "delegation_id": did,
         "run_id": run.id},
        _call(arguments=arguments), session=session)
    assert response.evaluation.outcome is (Outcome.ALLOW if allowed else Outcome.DENY)
    assert len(backend.calls) == int(allowed)
    assert response.status == (200 if allowed else 403)
    if not allowed:
        assert "S3-DELEGATION" in [c.citation for c in response.evaluation.failures()]
