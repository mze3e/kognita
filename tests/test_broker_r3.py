"""Round-3 broker regressions: real grants, delegations and credentials.

Deliberately outside the ``tests/conftest.py`` ``_legacy_authority_prerequisites``
and the Gateway/McpProxy credential-injection allowlists, so these exercise
production ``decide()``/``run_governed`` behaviour, not the autouse shims.
"""
from copy import deepcopy

import pytest

from conftest import NOW
from kognita.broker import ask
from kognita.credentials import revoke
from kognita.envelope import Envelope
from kognita.registry import get, approve_deployment
from kognita.tools import Run, _save_run, continue_run
from kognita.vocabulary import Outcome
from test_approval_continuation import _hold
from test_mcp import _Pack
from test_mcp_client_argument_scope import _delegated
from test_run_binding_broker import _index


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Use explicit real credentials; do not wrap ingress or inject them."""


def test_broker_ask_denies_revoked_credential(session, evidence):
    """Finding 3: broker.ask must apply the live credential gate run_governed has."""
    did, credential = _delegated(session, evidence)
    agent = get(session, "dossier-agent")
    revoke(session, "dossier-agent", credential, evidence=evidence, actor_id=agent.owner_exec)
    env = Envelope(principal="alice", agent_name="dossier-agent", purpose="COLLABORATION",
        tool="read_note", delegation_id=did, actor_location="SG",
        context={"_agent_credential": credential})
    answer = ask(session, "meeting notes", env, pack=_Pack(), embedder=_index(session),
        evidence=evidence, purposes=["COLLABORATION"], run=Run(), now=NOW, as_of=NOW)
    assert answer.outcome is Outcome.DENY
    assert "S2-CREDENTIAL" in [c.citation for c in answer.evaluation.failures()]
    assert answer.results == []


def test_broker_isolation_uses_resolved_subjects_not_submitted_envelope(session, evidence):
    """Finding 4: broker taint/isolation checks must use decide()'s resolved envelope."""
    did, credential = _delegated(session, evidence)
    agent = get(session, "dossier-agent")
    permissions = deepcopy(agent.tool_allowlist)
    permissions["read_note"] = {**permissions["read_note"],
        "_subject_paths": {"client": ["account"]}, "public_content": True}
    agent.tool_allowlist = permissions
    agent.field_allowlist = {**agent.field_allowlist, "COLLABORATION": ["results.snippet"]}
    approve_deployment(session, agent.name, deployment=permissions["_deployment"],
        evidence=evidence, actor_id=agent.owner_exec)
    # No subject_type/subject_id on the submitted envelope: the client scope
    # only exists once the manifest's argument mapping is resolved by decide().
    env = Envelope(principal="alice", agent_name="dossier-agent", purpose="COLLABORATION",
        tool="read_note", delegation_id=did, actor_location="SG",
        arguments={"account": "123"}, context={"_agent_credential": credential})
    answer = ask(session, "meeting notes", env, pack=_Pack(), embedder=_index(session),
        evidence=evidence, purposes=["COLLABORATION"], run=Run(), now=NOW, as_of=NOW)
    assert answer.outcome is Outcome.DENY, answer.evaluation.basis()
    assert "S5-PUBLIC-CONTENT" in [c.citation for c in answer.evaluation.failures()]
    assert answer.results == []
    assert "confidential fragment" not in " ".join(answer.summary)


def test_broker_resumed_retrieval_rechecks_tool_isolation_after_taint(session, evidence):
    """Finding 5: resumed retrieval must rerun _tool_checks, not just run identity."""
    did, credential = _delegated(session, evidence)
    agent = get(session, "dossier-agent")
    permissions = deepcopy(agent.tool_allowlist)
    permissions["read_note"] = {**permissions["read_note"], "public_content": True}
    agent.tool_allowlist = permissions
    agent.field_allowlist = {**agent.field_allowlist, "COLLABORATION": ["results.snippet"]}
    approve_deployment(session, agent.name, deployment=permissions["_deployment"],
        evidence=evidence, actor_id=agent.owner_exec)
    _hold(session, "read_note")
    env = Envelope(principal="alice", agent_name="dossier-agent", purpose="COLLABORATION",
        tool="read_note", delegation_id=did, actor_location="SG",
        context={"_agent_credential": credential})
    run = Run()
    embedder = _index(session)
    held = ask(session, "meeting notes", env, pack=_Pack(), embedder=embedder,
        evidence=evidence, purposes=["COLLABORATION"], run=run, now=NOW, as_of=NOW)
    assert held.outcome is Outcome.HUMAN_APPROVAL, held.evaluation.basis()
    # The run is tainted with client data while the retrieval sits held, e.g.
    # by another call sharing the same bounded run.
    run.client_data = True
    _save_run(session, run)
    aid = run.approvals_pending[0]
    result = continue_run(session, run.id, {aid: True}, embedder=embedder,
        evidence=evidence, approver_name="reviewer", now=NOW)
    assert result.outcome is Outcome.DENY
    assert "S5-PUBLIC-CONTENT" in [c.citation for c in result.evaluation.failures()]
    assert result.results == []
