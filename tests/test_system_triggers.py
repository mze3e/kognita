"""System trigger admission without legacy authority or credential injection."""
from datetime import timedelta

import pytest

from conftest import NOW
from kognita.authority import Delegation, delegate, grant_to
from kognita.credentials import issue
from kognita.models import Policy
from kognita.vocabulary import CheckResult, Outcome
from test_gateway import MODEL, PATH, _body, _client, _events, _gateway, _live, _Upstream
from test_mcp import _call, _proxy, _register, _Upstream as _Backend


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    # Override the global legacy wrapper: these tests exercise actual headers.
    yield


@pytest.mark.parametrize("mode", ["gateway", "mcp"])
@pytest.mark.parametrize("identity", ["listed", "unlisted", "spoofed", "missing", "policy-denied", "credential", "credential-unlisted"])
def test_system_trigger_admission(session, evidence, engine, mode, identity):
    _live(session)
    grant_to("alice", "read", timedelta(hours=2), {"model": [MODEL]},
             session=session, evidence=evidence, actor_id="admin", now=NOW)
    if identity in {"spoofed", "credential", "credential-unlisted"}:
        _register(session)
        scope = {"client": ["123"], "model": [MODEL]}
        grant_to("alice", "read", timedelta(hours=2), scope,
                 session=session, evidence=evidence, actor_id="admin", now=NOW)
        delegation_id = delegate(session, Delegation(
            human="alice", agent="dossier-agent", purpose="COLLABORATION",
            client="123", allowed_actions=["read"], allowed_data=scope,
            valid_from=NOW, valid_until=NOW + timedelta(hours=1),
            jurisdiction="SG", task_ref="trigger-regression"),
            evidence=evidence, actor_id="alice", now=NOW)
        credential = issue(session, "dossier-agent", evidence=evidence,
                           actor_id="Head of Research Ops")
    if identity == "policy-denied":
        session.add(Policy(regime="INTERNAL", rule_type="PROHIBITED",
            rule={"description": "calls barred", "on_violation": "fail"},
            citation="TEST-TRIGGER-POLICY", effective_from=NOW))
    client = _client(system_triggers=frozenset({"nightly-refresh"}))
    headers = {"system_trigger": "unknown" if identity == "unlisted" else "nightly-refresh"}
    if identity == "spoofed":
        # Even a client-listed agent claim cannot borrow the trigger's admission.
        headers["agent_name"] = "dossier-agent"
    if identity in {"credential", "credential-unlisted"}:
        headers = {"agent_name": "dossier-agent", "agent_credential": credential,
                   "delegation_id": delegation_id}
        if identity == "credential-unlisted":
            headers["system_trigger"] = "unknown"
    if identity == "missing":
        headers = {}
    if mode == "gateway":
        backend = _Upstream(b'{"model":"gateway-model","choices":[]}')
        endpoint = _gateway(engine, evidence, backend, client=client)
        path, body = PATH, _body("hello")
    else:
        backend = _Backend()
        endpoint = _proxy(engine, evidence, backend, client=client)
        path, body = "/mcp", _call()
    response = endpoint.handle("POST", path, headers, body, session=session)
    evaluation = response.evaluation
    assert evaluation is not None
    assert all(check.citation.strip() for check in evaluation.checks)
    if identity in {"listed", "credential"}:
        assert response.status == 200, evaluation.failures()
        assert evaluation.outcome is Outcome.ALLOW
        assert len(backend.calls) == 1
        if identity == "listed":
            assert evaluation.envelope.context["system_trigger"] == "nightly-refresh"
            assert "nightly-refresh" in repr([event.payload for event in _events(session, evaluation.request_id)])
    else:
        assert response.status == 403
        assert evaluation.outcome is Outcome.DENY
        assert backend.calls == []
        expected = "SYSTEM_TRIGGER" if identity in {"unlisted", "credential-unlisted"} else "AGENT_CREDENTIAL"
        if identity == "policy-denied":
            assert any(check.citation == "TEST-TRIGGER-POLICY" for check in evaluation.failures())
        else:
            assert any(check.check == expected and check.result is CheckResult.FAIL
                       for check in evaluation.checks)
