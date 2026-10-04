"""The MCP proxy: authorise with run_governed, then proxy, and evidence it.

Covers roadmap 0.3 item 5. An allow is proxied and evidenced. A denial is not
proxied and returns the outcome and the citations. Free-text arguments are
classified only when a typed classification is missing. A missing agent name
is a denial. An evidence-store failure refuses the call and does not reach
the backend.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path

import pytest
from sqlmodel import select

from kognita.canonical import hash_text
from kognita.classify import INDICATOR_CONFIDENCE, PATTERN_MODEL, PATTERN_VERSION, PatternClassifier
from kognita.cli import build_parser, cmd_serve
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.exceptions import ConfigError
from kognita.gateway import ClientConfiguration
from kognita.mcp import BackendServer, McpProxy, load_root_config
from kognita.models import EvidenceEvent, GovernanceDecision, Policy
from kognita.registry import register
from kognita.rules import build_registry
from kognita.vocabulary import ActorType, EventType, Outcome

SECRET = "ana@example.org"
EFFECTIVE = datetime(2026, 1, 1, tzinfo=timezone.utc)
NOTES = "http://127.0.0.1:9/notes"
CRM = "http://127.0.0.1:9/crm"


class _Pack:
    name = "mcp"

    def load_subjects(self, envelope, session):
        return {}

    def resolve_attributes(self, envelope, subjects):
        return {}

    def rules(self):
        return build_registry()


class _ClassifiedPack(_Pack):
    def resolve_attributes(self, envelope, subjects):
        return {"classification": "C1"}


class _Upstream:
    def __init__(self, result: dict | None = None, *, error: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.result = {"content": [{"type": "text", "text": "rivera notes"}]} if result is None else result
        self.error = error

    def __call__(self, method, url, headers, body):
        self.calls.append(
            {"method": method, "url": url, "headers": dict(headers), "body": body}
        )
        payload: dict = {"jsonrpc": "2.0", "id": 1}
        if self.error is not None:
            payload["error"] = self.error
        else:
            parsed = json.loads(body.decode())
            if parsed["method"] == "tools/list":
                payload["result"] = {
                    "tools": [{"name": "read_note", "description": "Read a note"}]
                }
            else:
                payload["result"] = self.result
        return 200, {"content-type": "application/json"}, json.dumps(payload).encode()


class _Spy(PatternClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def classify(self, text: str, *, hint=None):
        self.seen.append(text)
        return super().classify(text, hint=hint)


class _Down(EvidenceWriter):
    def emit(self, session, **kwargs):
        raise ConnectionError("evidence store unavailable")


def _client(**overrides) -> ClientConfiguration:
    fields: dict = dict(
        principal="alice",
        purpose="COLLABORATION",
        agent_names=frozenset({"dossier-agent"}),
        actor_location="SG",
    )
    fields.update(overrides)
    return ClientConfiguration(**fields)


def _proxy(engine, evidence, upstream, **kwargs) -> McpProxy:
    client = kwargs.pop("client", _client())
    servers = kwargs.pop(
        "servers",
        (BackendServer(name="notes", url=NOTES),),
    )
    pack = kwargs.pop("pack", _Pack())
    return McpProxy(
        engine=engine,
        evidence=evidence,
        servers=servers,
        client=client,
        pack=pack,
        transport=upstream,
        **kwargs,
    )


def _register(session, name: str = "dossier-agent") -> None:
    register(session, name=name, owner_exec="Head of Research Ops")


def _call(name: str = "read_note", arguments: dict | None = None, rpc_id: int = 1) -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments if arguments is not None else {"query": "hello"}},
        }
    ).encode()


def _events(session, request_id: str) -> list[EvidenceEvent]:
    return list(
        session.exec(
            select(EvidenceEvent)
            .where(EvidenceEvent.correlation_id == request_id)
            .order_by(EvidenceEvent.sequence)
        ).all()
    )


def _denial(response) -> dict:
    body = json.loads(response.body)
    return body["error"]["data"]


def test_allowed_call_is_proxied_and_evidenced(session, evidence):
    _register(session)
    session.flush()
    upstream = _Upstream()
    proxy = _proxy(session.get_bind(), evidence, upstream)
    response = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        _call(arguments={"query": "hello"}),
        session=session,
    )

    assert response.status == 200
    assert len(upstream.calls) == 1
    forwarded = json.loads(upstream.calls[0]["body"])
    assert upstream.calls[0]["url"] == NOTES
    assert forwarded["method"] == "tools/call"
    assert forwarded["params"] == {"name": "read_note", "arguments": {"query": "hello"}}
    body = json.loads(response.body)
    assert body["result"]["content"][0]["text"] == "rivera notes"
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.ALLOW
    assert response.evaluation.envelope.tool == "read_note"
    events = _events(session, response.evaluation.request_id)
    kinds = [event.event_type for event in events]
    assert EventType.POLICY_DECISION in kinds
    assert EventType.TOOL_CALL in kinds
    assert EventType.EGRESS in kinds
    tool_call = next(event for event in events if event.event_type is EventType.TOOL_CALL)
    assert tool_call.actor_type is ActorType.AGENT
    assert tool_call.actor_id == "dossier-agent"
    assert tool_call.payload["tool"] == "read_note"
    assert "rivera notes" not in json.dumps(tool_call.payload)
    assert verify_chain(session) >= len(events)


def test_denial_is_not_proxied_and_returns_outcome_and_citations(session, evidence):
    _register(session)
    session.add(
        Policy(
            regime="INTERNAL",
            rule_type="PROHIBITED",
            rule={"description": "notes barred", "on_violation": "fail"},
            citation="Notes stay inside the team",
            effective_from=EFFECTIVE,
        )
    )
    session.flush()
    upstream = _Upstream()
    proxy = _proxy(session.get_bind(), evidence, upstream)
    response = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        _call(arguments={"query": f"Email {SECRET} about the notes"}),
        session=session,
    )

    assert upstream.calls == []
    assert response.status == 403
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.DENY
    denial = _denial(response)
    assert denial["outcome"] == "DENY"
    assert "Notes stay inside the team" in denial["citations"]
    assert SECRET not in response.body.decode()
    events = _events(session, response.evaluation.request_id)
    assert events
    assert all(event.event_type is EventType.POLICY_DECISION for event in events)


def test_free_text_is_classified_only_when_classification_is_missing(session, evidence):
    _register(session)
    session.flush()
    text = f"Email {SECRET} about the notes"
    upstream = _Upstream()
    spy = _Spy()
    proxy = _proxy(session.get_bind(), evidence, upstream, classifier=spy)

    missing = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        _call(arguments={"query": text}),
        session=session,
    )
    assert missing.evaluation is not None
    assert missing.evaluation.outcome is Outcome.ALLOW
    assert missing.evaluation.attributes["classification"] == "C2"
    recorded = missing.evaluation.attributes["classifier"]
    assert recorded["model"] == PATTERN_MODEL
    assert recorded["version"] == PATTERN_VERSION
    assert recorded["label"] == "C2"
    assert recorded["confidence"] == INDICATOR_CONFIDENCE
    assert recorded["input_hash"] == hash_text(text)
    assert spy.seen == [text]
    assert len(upstream.calls) == 1

    typed = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent", "classification": "C0"},
        _call(arguments={"query": text}, rpc_id=2),
        session=session,
    )
    assert typed.status == 200
    assert typed.evaluation is not None
    assert typed.evaluation.outcome is Outcome.ALLOW
    assert typed.evaluation.attributes["classification"] == "C0"
    assert "classifier" not in typed.evaluation.attributes
    assert spy.seen == [text]
    assert len(upstream.calls) == 2

    packed = _proxy(
        session.get_bind(),
        evidence,
        upstream,
        classifier=spy,
        pack=_ClassifiedPack(),
    )
    from_pack = packed.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        _call(arguments={"query": text}, rpc_id=3),
        session=session,
    )
    assert from_pack.evaluation is not None
    assert from_pack.evaluation.attributes["classification"] == "C1"
    assert "classifier" not in from_pack.evaluation.attributes
    assert spy.seen == [text]

    header_wins = packed.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent", "classification": "C0"},
        _call(arguments={"query": text}, rpc_id=4),
        session=session,
    )
    assert header_wins.evaluation is not None
    assert header_wins.evaluation.attributes["classification"] == "C0"
    assert "classifier" not in header_wins.evaluation.attributes
    assert spy.seen == [text]


def test_missing_agent_name_is_deny(session, evidence):
    _register(session)
    session.flush()
    upstream = _Upstream()
    proxy = _proxy(session.get_bind(), evidence, upstream)
    response = proxy.handle("POST", "/mcp", {}, _call(), session=session)

    assert upstream.calls == []
    assert response.status == 403
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.DENY
    assert response.evaluation.envelope.agent_name is None
    denial = _denial(response)
    assert denial["outcome"] == "DENY"
    assert any("no agent name" in citation for citation in denial["citations"])
    events = _events(session, response.evaluation.request_id)
    assert events
    assert all(event.actor_type is not ActorType.HUMAN for event in events)


def test_agent_name_outside_the_client_configuration_is_deny(session, evidence):
    _register(session, "dossier-agent")
    _register(session, "eligibility-assistant")
    session.flush()
    upstream = _Upstream()
    proxy = _proxy(session.get_bind(), evidence, upstream)
    response = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "eligibility-assistant"},
        _call(),
        session=session,
    )

    assert upstream.calls == []
    assert response.status == 403
    denial = _denial(response)
    assert denial["outcome"] == "DENY"
    assert any("authenticated client configuration" in citation for citation in denial["citations"])


def test_approved_system_trigger_is_a_system_actor(session, evidence):
    upstream = _Upstream()
    proxy = _proxy(
        session.get_bind(),
        evidence,
        upstream,
        client=_client(agent_names=frozenset(), system_triggers=frozenset({"nightly-refresh"})),
    )
    response = proxy.handle(
        "POST",
        "/mcp",
        {"system_trigger": "nightly-refresh"},
        _call(),
        session=session,
    )

    assert response.status == 200
    assert len(upstream.calls) == 1
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.ALLOW
    assert response.evaluation.envelope.agent_name is None
    tool_call = next(
        event
        for event in _events(session, response.evaluation.request_id)
        if event.event_type is EventType.TOOL_CALL
    )
    assert tool_call.actor_type is ActorType.SYSTEM
    assert tool_call.actor_id == "nightly-refresh"
    assert tool_call.actor_type is not ActorType.HUMAN


def test_default_actor_context_supplies_the_agent_name(session, evidence):
    _register(session)
    session.flush()
    upstream = _Upstream()
    proxy = _proxy(
        session.get_bind(),
        evidence,
        upstream,
        client=_client(agent_name="dossier-agent"),
    )
    response = proxy.handle("POST", "/mcp", {}, _call(), session=session)

    assert response.status == 200
    assert response.evaluation is not None
    assert response.evaluation.envelope.agent_name == "dossier-agent"
    assert len(upstream.calls) == 1


def test_evidence_store_failure_refuses_without_backend_call(session, engine):
    _register(session)
    session.commit()
    upstream = _Upstream()
    proxy = _proxy(engine, _Down(engine), upstream)
    response = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        _call(arguments={"query": f"Email {SECRET}"}),
        session=session,
    )

    assert response.status == 503
    assert upstream.calls == []
    assert json.loads(response.body)["error"]["message"] == "evidence store unavailable"
    assert session.exec(select(EvidenceEvent)).all() == []
    assert session.exec(select(GovernanceDecision)).all() == []


def test_a_call_reaches_only_the_named_backend(session, evidence):
    _register(session)
    session.flush()
    upstream = _Upstream()
    proxy = _proxy(
        session.get_bind(),
        evidence,
        upstream,
        servers=(
            BackendServer(name="notes", url=NOTES),
            BackendServer(name="crm", url=CRM),
        ),
    )
    denied_list = proxy.handle(
        "POST",
        "/mcp",
        {},
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
        session=session,
    )
    assert denied_list.status == 403
    assert upstream.calls == []

    listed = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).encode(),
        session=session,
    )
    assert listed.status == 200
    names = [tool["name"] for tool in json.loads(listed.body)["result"]["tools"]]
    assert names == ["notes/read_note", "crm/read_note"]
    assert [call["url"] for call in upstream.calls] == [NOTES, CRM]

    response = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        _call(name="crm/read_note", rpc_id=3),
        session=session,
    )
    assert response.status == 200
    assert [call["url"] for call in upstream.calls] == [NOTES, CRM, CRM]
    assert json.loads(upstream.calls[-1]["body"])["params"]["name"] == "read_note"


def test_notification_and_unknown_method_are_not_proxied(session, evidence):
    _register(session)
    session.flush()
    upstream = _Upstream()
    proxy = _proxy(session.get_bind(), evidence, upstream)
    notice = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        json.dumps({"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "read_note"}}).encode(),
        session=session,
    )
    assert notice.status == 202
    assert notice.body == b""
    assert upstream.calls == []

    unknown = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        json.dumps({"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {}}).encode(),
        session=session,
    )
    assert unknown.status == 200
    assert json.loads(unknown.body)["error"]["code"] == -32601
    assert upstream.calls == []


def test_initialize_does_not_call_a_backend(session, evidence):
    _register(session)
    session.flush()
    upstream = _Upstream()
    proxy = _proxy(session.get_bind(), evidence, upstream)
    response = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
            }
        ).encode(),
        session=session,
    )
    assert response.status == 200
    body = json.loads(response.body)
    assert body["result"]["protocolVersion"] == "2025-03-26"
    assert body["result"]["serverInfo"]["name"] == "kognita"
    assert body["result"]["capabilities"]["tools"] == {}
    assert upstream.calls == []
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.ALLOW


def test_root_config_names_servers_pack_evidence_and_actor(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "servers": [
                    {"name": "notes", "url": NOTES},
                    {"name": "crm", "url": CRM},
                ],
                "policy_pack": "fixtures.demo_pack:DemoPack",
                "evidence_database": "kognita.db",
                "actor": {
                    "principal": "alice",
                    "purpose": "COLLABORATION",
                    "actor_location": "SG",
                    "agent_names": ["dossier-agent"],
                    "system_triggers": ["nightly-refresh"],
                },
            }
        )
    )
    config = load_root_config(str(path))
    assert [server.name for server in config.servers] == ["notes", "crm"]
    assert config.servers[0].url == NOTES
    assert config.policy_pack == "fixtures.demo_pack:DemoPack"
    assert config.pack.name == "demo"
    assert config.evidence_database == "kognita.db"
    assert config.actor.principal == "alice"
    assert config.actor.purpose == "COLLABORATION"
    assert "dossier-agent" in config.actor.agent_names
    assert "nightly-refresh" in config.actor.system_triggers

    with pytest.raises(ConfigError):
        load_root_config(str(tmp_path / "missing.json"))


def test_serve_mcp_parses_and_does_not_start_the_model_gateway(tmp_path: Path):
    args = build_parser().parse_args(["serve", "--mcp", "--root-config", "config.json"])
    assert args.command == "serve"
    assert args.mcp is True
    assert args.root_config == "config.json"
    assert args.provider is None
    assert args.func.__name__ == "cmd_serve"

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["serve", "--mcp", "--provider", "openai-compatible", "--upstream", "https://api.openai.com"]
        )

    missing = build_parser().parse_args(
        ["serve", "--mcp", "--root-config", str(tmp_path / "missing.json")]
    )
    assert cmd_serve(missing) == 2


def test_http_server_refuses_a_denial_and_proxies_an_allow(session, evidence, engine):
    _register(session)
    session.commit()
    upstream = _Upstream()
    proxy = _proxy(engine, evidence, upstream)
    server = proxy._make_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        denied = _post(port, {}, _call())
        assert denied.status == 403
        denied_body = json.loads(denied.read())
        assert denied_body["error"]["data"]["outcome"] == "DENY"
        assert upstream.calls == []
        allowed = _post(port, {"agent_name": "dossier-agent"}, _call())
        assert allowed.status == 200
        assert json.loads(allowed.read())["result"]["content"][0]["text"] == "rivera notes"
        assert len(upstream.calls) == 1
        assert upstream.calls[0]["url"] == NOTES
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _post(port: int, headers: dict[str, str], body: bytes):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(
        "POST",
        "/mcp",
        body=body,
        headers={**headers, "Content-Length": str(len(body)), "Content-Type": "application/json"},
    )
    return connection.getresponse()
