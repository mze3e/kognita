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
from sqlmodel import Session, select

from fixtures import demo_pack as dp

from kognita.approvals import grant
from kognita.canonical import canonical_hash, canonical_json, hash_text
from kognita.classify import INDICATOR_CONFIDENCE, PATTERN_MODEL, PATTERN_VERSION, PatternClassifier
from kognita.cli import build_parser, cmd_serve
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.exceptions import ConfigError
from kognita.gateway import ClientConfiguration
from kognita.mcp import BackendServer, McpProxy, load_root_config
from kognita.models import Approval, EvidenceEvent, GovernanceDecision, Policy
from kognita.registry import register
from kognita.rules import build_registry
from kognita.vocabulary import ActorType, CheckResult, EventType, Outcome

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
    purposes = kwargs.pop("purposes", (client.purpose,))
    return McpProxy(
        engine=engine,
        evidence=evidence,
        servers=servers,
        client=client,
        pack=pack,
        purposes=purposes,
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


def test_granted_human_approval_is_proxied_once_and_returned(session, evidence):
    """A live grant releases the call. An ungranted hold does not touch the backend."""
    _register(session)
    session.add(
        Policy(
            regime="INTERNAL",
            rule_type="REQUIRES_HUMAN_APPROVAL",
            rule={"tools": ["read_note"]},
            citation="A person releases this note",
            effective_from=EFFECTIVE,
        )
    )
    session.flush()
    upstream = _Upstream()
    proxy = _proxy(session.get_bind(), evidence, upstream)
    held = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        _call(arguments={"query": "hello"}),
        session=session,
    )

    assert upstream.calls == []
    assert held.status == 403
    assert held.evaluation is not None
    assert held.evaluation.outcome is Outcome.HUMAN_APPROVAL
    assert _denial(held)["outcome"] == "HUMAN_APPROVAL"

    approval = session.exec(select(Approval)).one()
    grant(
        session,
        approval,
        approver_name="reviewer@example.org",
        evidence=evidence,
        correlation_id=held.evaluation.request_id,
        now=evidence.clock(),
    )
    session.flush()

    released = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        _call(arguments={"query": "hello"}, rpc_id=2),
        session=session,
    )

    assert len(upstream.calls) == 1
    assert released.status == 200
    body = json.loads(released.body)
    assert body["result"]["content"][0]["text"] == "rivera notes"
    assert released.evaluation is not None
    assert released.evaluation.outcome is Outcome.HUMAN_APPROVAL


def test_escalation_is_not_proxied(session, evidence):
    _register(session)
    session.add(
        Policy(
            regime="INTERNAL",
            rule_type="ATTRIBUTE_ALLOWLIST",
            rule={"allow": {"actor_location": ["HK"]}, "on_violation": "escalate"},
            citation="Only Hong Kong may call this tool",
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
        _call(),
        session=session,
    )

    assert upstream.calls == []
    assert response.status == 403
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.ESCALATE
    assert _denial(response)["outcome"] == "ESCALATE"


def test_tool_arguments_are_absent_from_evidence(session, evidence):
    """The secret travels to the backend. The evidence chain keeps its hash."""
    _register(session)
    session.flush()
    arguments = {"query": f"Email {SECRET} about the notes"}
    upstream = _Upstream()
    proxy = _proxy(session.get_bind(), evidence, upstream)
    response = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        _call(arguments=arguments),
        session=session,
    )

    assert response.status == 200
    assert SECRET not in response.body.decode()
    forwarded = json.loads(upstream.calls[0]["body"])
    assert forwarded["params"]["arguments"]["query"] == arguments["query"]
    assert response.evaluation is not None
    events = _events(session, response.evaluation.request_id)
    assert events
    for event in events:
        assert SECRET not in canonical_json(event.payload)
    decision = next(event for event in events if event.event_type is EventType.POLICY_DECISION)
    stored = decision.payload["envelope"]["arguments"]
    assert stored == {
        "sha256": canonical_hash(arguments),
        "bytes": len(canonical_json(arguments)),
    }


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
    assert config.purposes == ()
    assert "dossier-agent" in config.actor.agent_names
    assert "nightly-refresh" in config.actor.system_triggers

    with pytest.raises(ConfigError):
        load_root_config(str(tmp_path / "missing.json"))

    claimed_as_list = json.loads(path.read_text())
    claimed_as_list["purposes"] = claimed_as_list["actor"]["purpose"]
    string_path = tmp_path / "purpose-string.json"
    string_path.write_text(json.dumps(claimed_as_list))
    with pytest.raises(ConfigError, match="purposes must be a list"):
        load_root_config(str(string_path))


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


def _seed_agent(path) -> None:
    engine = make_engine(path)
    create_all(engine)
    with Session(engine) as session:
        register(session, name="dossier-agent", owner_exec="Head of Research Ops")
        session.commit()
    engine.dispose()


def _write_root(path, db, purposes) -> None:
    body = {
        "servers": [{"name": "notes", "url": NOTES}],
        "policy_pack": "fixtures.demo_pack:DemoPack",
        "evidence_database": str(db),
        "actor": {
            "principal": "alice",
            "purpose": dp.PURPOSES[0],
            "actor_location": "SG",
            "agent_names": ["dossier-agent"],
        },
    }
    if purposes is not None:
        body["purposes"] = list(purposes)
    path.write_text(json.dumps(body))


def _serve_proxy(monkeypatch, config):
    """Build the proxy the way ``kognita serve --mcp`` does, and do not listen."""
    held = {}

    def serve(self, host="127.0.0.1", port=8080):
        held["proxy"] = self

    monkeypatch.setattr(McpProxy, "serve", serve)
    args = build_parser().parse_args(["serve", "--mcp", "--root-config", str(config)])
    assert cmd_serve(args) == 0
    return held["proxy"]


def _purpose(evaluation):
    matches = [check for check in evaluation.checks if check.check == "PURPOSE"]
    assert len(matches) == 1
    return matches[0]


def test_mcp_serve_allows_a_listed_purpose_and_denies_an_unlisted_one(tmp_path, monkeypatch):
    """The root config ``purposes`` list is the allowlist. ``actor.purpose`` is not."""
    db = tmp_path / "mcp.db"
    _seed_agent(db)
    claimed = dp.PURPOSES[0]
    listed = dp.PURPOSES[1]
    config = tmp_path / "config.json"
    _write_root(config, db, (listed,))
    proxy = _serve_proxy(monkeypatch, config)
    assert proxy.client.purpose == claimed
    assert tuple(proxy.purposes) == (listed,)
    upstream = _Upstream()
    proxy.transport = upstream

    denied = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent", "purpose": claimed},
        _call(arguments={"query": "hello"}),
    )
    assert upstream.calls == []
    assert denied.evaluation is not None
    assert denied.evaluation.envelope.purpose == claimed
    assert denied.evaluation.outcome is Outcome.DENY
    assert _purpose(denied.evaluation).result is CheckResult.FAIL

    allowed = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent", "purpose": listed},
        _call(arguments={"query": "hello"}),
    )
    assert allowed.evaluation is not None
    assert allowed.evaluation.envelope.purpose == listed
    assert allowed.evaluation.outcome is Outcome.ALLOW
    assert _purpose(allowed.evaluation).result is CheckResult.PASS
    assert len(upstream.calls) == 1


def test_mcp_serve_denies_when_no_purpose_list_is_configured(tmp_path, monkeypatch):
    """A root config with only ``actor.purpose`` still denies the call."""
    db = tmp_path / "mcp.db"
    _seed_agent(db)
    claimed = dp.PURPOSES[0]
    config = tmp_path / "config.json"
    _write_root(config, db, None)
    proxy = _serve_proxy(monkeypatch, config)
    assert tuple(proxy.purposes) == ()
    assert proxy.client.purpose == claimed
    upstream = _Upstream()
    proxy.transport = upstream
    response = proxy.handle(
        "POST",
        "/mcp",
        {"agent_name": "dossier-agent"},
        _call(arguments={"query": "hello"}),
    )
    assert upstream.calls == []
    assert response.evaluation is not None
    assert response.evaluation.envelope.purpose == claimed
    assert response.evaluation.outcome is Outcome.DENY
    assert _purpose(response.evaluation).result is not CheckResult.PASS
    assert _purpose(response.evaluation).result is CheckResult.FAIL
