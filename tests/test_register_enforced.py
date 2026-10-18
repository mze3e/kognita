"""An empty use-case register denies every execution path.

The purpose list does not authorize ``run_governed``, ``ask``, the MCP proxy
or the AI gateway. The denial cites the register rule. No network and no API key.
"""

from __future__ import annotations

import json

from fixtures import demo_pack as dp

from kognita.broker import ask
from kognita.embedding import HashingEmbedder
from kognita.envelope import Envelope
from kognita.gateway import ClientConfiguration, Gateway
from kognita.mcp import BackendServer, McpProxy
from kognita.registry import register
from kognita.tools import ToolRegistry, run_governed
from kognita.vocabulary import CheckResult, Outcome
from test_usecases import NOW

PURPOSE = "COLLABORATION"


class _Upstream:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, method, url, headers, body):
        self.calls.append(url)
        return 200, {"content-type": "application/json"}, b"{}"


def _agent(session) -> None:
    register(session, name="dossier-agent", owner_exec="Head of Research Ops")


def _client() -> ClientConfiguration:
    return ClientConfiguration(
        principal="alice",
        purpose=PURPOSE,
        agent_names=frozenset({"dossier-agent"}),
        actor_location="SG",
    )


def _register_check(evaluation) -> None:
    matches = [check for check in evaluation.checks if check.check == "USE_CASE_REGISTERED"]
    assert len(matches) == 1
    check = matches[0]
    assert check.result is CheckResult.FAIL
    assert "not registered" in check.citation
    assert "Use-case register" in check.citation
    assert not any(item.check == "PURPOSE" for item in evaluation.checks)


def test_empty_register_denies_run_governed(session, evidence):
    executed: list[str] = []
    registry = ToolRegistry()

    @registry.tool("get_subject_profile")
    def _profile(envelope, evaluation, bound):
        executed.append(envelope.tool)
        return {"subject": envelope.subject_id}

    result = run_governed(
        session,
        Envelope(
            principal="alice",
            purpose=PURPOSE,
            tool="get_subject_profile",
            actor_location="SG",
        ),
        registry=registry,
        evidence=evidence,
        pack=dp.DemoPack(),
        purposes=(PURPOSE,),
        as_of=NOW,
        now=NOW,
    )

    assert executed == []
    assert result.outcome is Outcome.DENY
    assert result.data is None
    _register_check(result.evaluation)


def test_empty_register_denies_ask(session, evidence):
    answer = ask(
        session,
        "What does the ethics board require?",
        Envelope(
            principal="alice",
            purpose=PURPOSE,
            tool="get_subject_profile",
            actor_location="SG",
        ),
        pack=dp.DemoPack(),
        embedder=HashingEmbedder(),
        evidence=evidence,
        purposes=(PURPOSE,),
        as_of=NOW,
        now=NOW,
    )

    assert answer.outcome is Outcome.DENY
    assert answer.results == []
    _register_check(answer.evaluation)


def test_empty_register_denies_the_gateway(session, evidence):
    _agent(session)
    session.flush()
    upstream = _Upstream()
    gateway = Gateway(
        engine=session.get_bind(),
        evidence=evidence,
        upstream="https://api.openai.com",
        client=_client(),
        purposes=(PURPOSE,),
        transport=upstream,
    )
    response = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent"},
        json.dumps(
            {
                "model": "gateway-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
        ).encode(),
        session=session,
    )

    assert upstream.calls == []
    assert response.status == 403
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.DENY
    _register_check(response.evaluation)


def test_empty_register_denies_the_mcp_proxy(session, evidence):
    _agent(session)
    session.flush()
    upstream = _Upstream()
    proxy = McpProxy(
        engine=session.get_bind(),
        evidence=evidence,
        servers=(BackendServer(name="notes", url="http://127.0.0.1:9/notes"),),
        client=_client(),
        pack=dp.DemoPack(),
        purposes=(PURPOSE,),
        transport=upstream,
    )
    response = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "read_note", "arguments": {"query": "hello"}},
            }
        ).encode(),
        session=session,
    )

    assert upstream.calls == []
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.DENY
    _register_check(response.evaluation)
