"""Regression test for the MCP client-argument delegation bypass.

A delegation scoped to one client must not authorise a tool call whose
*arguments* name a different client. This module is deliberately named
outside the ``tests/conftest.py`` ``_legacy_authority_prerequisites``
allowlist so its tests run against real credentials and real delegations,
not the autouse fixture authority/credential shims other modules get.
"""
from fixtures.agents import _usecases
from datetime import timedelta

from conftest import NOW
from kognita.authority import Delegation, delegate, grant_to
from kognita.credentials import issue
from kognita.envelope import Envelope
from kognita.governance import decide, load_snapshot
from kognita.vocabulary import Outcome


def _delegated(session, evidence, *, client="123"):
    from test_mcp import _register

    _register(session)
    scope = {"client": [client]}
    grant_to("alice", "read", timedelta(hours=2), scope, session=session,
             evidence=evidence, actor_id="admin", now=NOW)
    did = delegate(session, Delegation(human="alice", agent="dossier-agent", purpose="COLLABORATION",
        client=client, allowed_actions=["read"], allowed_data=scope, valid_from=NOW,
        valid_until=NOW + timedelta(hours=1), jurisdiction="SG", task_ref="meeting"),
        evidence=evidence, actor_id="alice", now=NOW)
    credential = issue(session, "dossier-agent", evidence=evidence, actor_id="Head of Research Ops")
    return did, credential


def test_mismatched_client_argument_is_denied_and_backend_not_called(session, evidence, engine):
    from test_mcp import _proxy, _Upstream, _call

    did, credential = _delegated(session, evidence)
    backend = _Upstream()
    proxy = _proxy(engine, evidence, backend)
    response = proxy.handle("POST", "/mcp",
        {"agent_name": "dossier-agent", "agent_credential": credential, "delegation_id": did},
        _call(arguments={"client_id": "456"}), session=session)

    assert backend.calls == []
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    citations = [check.citation for check in response.evaluation.basis()]
    assert "S3-DELEGATION" in citations


def test_matching_client_argument_is_allowed(session, evidence, engine):
    from test_mcp import _proxy, _Upstream, _call

    did, credential = _delegated(session, evidence)
    # Client-scoped MCP reads now require durable run provenance.
    from kognita.tools import Run
    from kognita.envelope import Envelope
    from kognita._fleet import _run_checks
    run = Run()
    _run_checks(session, run, Envelope(principal="alice", purpose="COLLABORATION",
        tool="read_note", agent_name="dossier-agent", delegation_id=did), evidence)
    backend = _Upstream()
    proxy = _proxy(engine, evidence, backend)
    response = proxy.handle("POST", "/mcp",
        {"agent_name": "dossier-agent", "agent_credential": credential, "delegation_id": did, "run_id": run.id},
        _call(arguments={"client_id": "123"}), session=session)

    assert response.status == 200, response.evaluation.failures()
    assert response.evaluation.outcome is Outcome.ALLOW
    assert len(backend.calls) == 1


def test_conflicting_client_values_in_one_argument_are_denied(session, evidence, engine):
    from test_mcp import _proxy, _Upstream, _call

    did, credential = _delegated(session, evidence)
    backend = _Upstream()
    proxy = _proxy(engine, evidence, backend)
    response = proxy.handle("POST", "/mcp",
        {"agent_name": "dossier-agent", "agent_credential": credential, "delegation_id": did},
        _call(arguments={"client": ["123", "456"]}), session=session)

    assert backend.calls == []
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    citations = [check.citation for check in response.evaluation.basis()]
    assert "S3-DELEGATION" in citations


def test_nested_dict_client_argument_is_denied_and_backend_not_called(session, evidence, engine):
    from test_mcp import _proxy, _Upstream, _call

    did, credential = _delegated(session, evidence)
    backend = _Upstream()
    proxy = _proxy(engine, evidence, backend)
    response = proxy.handle("POST", "/mcp",
        {"agent_name": "dossier-agent", "agent_credential": credential, "delegation_id": did},
        _call(arguments={"filter": {"client_id": "456"}}), session=session)

    assert backend.calls == []
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    citations = [check.citation for check in response.evaluation.basis()]
    assert "S3-DELEGATION" in citations


def test_nested_list_client_argument_is_denied_and_backend_not_called(session, evidence, engine):
    from test_mcp import _proxy, _Upstream, _call

    did, credential = _delegated(session, evidence)
    backend = _Upstream()
    proxy = _proxy(engine, evidence, backend)
    response = proxy.handle("POST", "/mcp",
        {"agent_name": "dossier-agent", "agent_credential": credential, "delegation_id": did},
        _call(arguments={"items": [{"client": "456"}]}), session=session)

    assert backend.calls == []
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    citations = [check.citation for check in response.evaluation.basis()]
    assert "S3-DELEGATION" in citations


def test_subject_type_subject_id_client_argument_is_denied_and_backend_not_called(session, evidence, engine):
    from test_mcp import _proxy, _Upstream, _call

    did, credential = _delegated(session, evidence)
    backend = _Upstream()
    proxy = _proxy(engine, evidence, backend)
    response = proxy.handle("POST", "/mcp",
        {"agent_name": "dossier-agent", "agent_credential": credential, "delegation_id": did},
        _call(arguments={"subject_type": "client", "subject_id": "456"}), session=session)

    assert backend.calls == []
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    citations = [check.citation for check in response.evaluation.basis()]
    assert "S3-DELEGATION" in citations


def test_authority_checks_denies_delegation_on_mismatched_client_argument(session, evidence):
    did, credential = _delegated(session, evidence)
    envelope = Envelope(context={"_agent_credential": credential},principal="alice", agent_name="dossier-agent", purpose="COLLABORATION",
        tool="read_note", actor_location="SG", delegation_id=did, arguments={"client_id": "456"})
    snapshot = load_snapshot(session, as_of=NOW, _agent_name="dossier-agent")
    evaluation = decide(envelope, snapshot, purposes=["COLLABORATION"], as_of=NOW, usecases=_usecases(["COLLABORATION"]))

    assert evaluation.outcome is Outcome.DENY
    assert "S3-DELEGATION" in [check.citation for check in evaluation.failures()]

    matching = Envelope(context={"_agent_credential": credential}, principal="alice", agent_name="dossier-agent", purpose="COLLABORATION",
        tool="read_note", actor_location="SG", delegation_id=did, arguments={"client_id": "123"})
    assert decide(matching, snapshot, purposes=["COLLABORATION"], as_of=NOW, usecases=_usecases(["COLLABORATION"])).outcome is Outcome.ALLOW


def test_structured_value_under_client_key_is_denied(session, evidence, engine):
    from test_mcp import _proxy, _Upstream, _call

    did, credential = _delegated(session, evidence)
    for arguments in ({"client": {"id": "456"}}, {"client_id": [{"id": "456"}]}):
        backend = _Upstream()
        proxy = _proxy(engine, evidence, backend)
        response = proxy.handle("POST", "/mcp",
            {"agent_name": "dossier-agent", "agent_credential": credential, "delegation_id": did},
            _call(arguments=arguments), session=session)

        assert backend.calls == [], arguments
        assert response.status == 403
        assert "S3-DELEGATION" in [check.citation for check in response.evaluation.basis()]
