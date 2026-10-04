"""The AI gateway: decide, redact, restore, and refuse when evidence cannot be written.

Covers roadmap 0.3 item 3 and the two gateway identity rules. A denial does not
call the upstream. An allow redacts, forwards, restores, and classifies the
response. Evidence holds hashes, not the prompt or the answer.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from http.client import HTTPConnection

import pytest
from sqlmodel import select

from kognita.canonical import canonical_json, hash_text
from kognita.classify import (
    INDICATOR_CONFIDENCE,
    PATTERN_MODEL,
    PATTERN_VERSION,
    PatternClassifier,
)
from kognita.cli import build_parser
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.gateway import ClientConfiguration, Gateway
from kognita.models import EvidenceEvent, GovernanceDecision, Policy, RunRecord
from kognita.registry import register, set_kill_switch
from kognita.tools import Run
from kognita.vocabulary import ActorType, CheckResult, EventType, Outcome

SECRET = "ana@example.org"
MODEL = "gateway-model"
PATH = "/v1/chat/completions"
EFFECTIVE = datetime(2026, 1, 1, tzinfo=timezone.utc)
UPSTREAM = "https://api.openai.com"


class _Upstream:
    def __init__(self, response: bytes, status: int = 200) -> None:
        self.calls: list[dict] = []
        self.response = response
        self.status = status

    def __call__(self, method, url, headers, body):
        self.calls.append(
            {"method": method, "url": url, "headers": dict(headers), "body": body}
        )
        return self.status, {"content-type": "application/json"}, self.response


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


def _body(content: str) -> bytes:
    return json.dumps(
        {"model": MODEL, "messages": [{"role": "user", "content": content}]}
    ).encode()


def _client(**overrides) -> ClientConfiguration:
    fields: dict = dict(
        principal="alice",
        purpose="COLLABORATION",
        agent_names=frozenset({"dossier-agent"}),
        actor_location="SG",
    )
    fields.update(overrides)
    return ClientConfiguration(**fields)


def _gateway(engine, evidence, upstream, **kwargs) -> Gateway:
    client = kwargs.pop("client", _client())
    return Gateway(
        engine=engine,
        evidence=evidence,
        upstream=UPSTREAM,
        client=client,
        transport=upstream,
        **kwargs,
    )


def _register(session, name: str = "dossier-agent") -> None:
    register(session, name=name, owner_exec="Head of Research Ops")


def _events(session, request_id: str) -> list[EvidenceEvent]:
    return list(
        session.exec(
            select(EvidenceEvent)
            .where(EvidenceEvent.correlation_id == request_id)
            .order_by(EvidenceEvent.sequence)
        ).all()
    )


def test_denial_returns_before_any_upstream_call(session, evidence):
    _register(session)
    session.add(
        Policy(
            regime="INTERNAL",
            rule_type="PROHIBITED",
            rule={"description": "model calls barred", "on_violation": "fail"},
            citation="No external model calls",
            effective_from=EFFECTIVE,
        )
    )
    session.flush()
    upstream = _Upstream(b"{}")
    gateway = _gateway(session.get_bind(), evidence, upstream)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(f"Email {SECRET} about the notes"),
        session=session,
    )

    assert upstream.calls == []
    assert response.status == 403
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.DENY
    body = json.loads(response.body)
    assert body["outcome"] == "DENY"
    assert any(check["citation"] == "No external model calls" for check in body["checks"])
    assert SECRET not in response.body.decode()
    recorded = _events(session, response.evaluation.request_id)
    assert recorded
    assert all(event.event_type is EventType.POLICY_DECISION for event in recorded)
    assert SECRET not in canonical_json([event.payload for event in recorded])


def test_allow_redacts_forwards_restores_and_classifies(session, evidence):
    _register(session)
    session.flush()
    prompt = f"Email {SECRET} about the notes"
    restored_answer = f"Reply to {SECRET} tomorrow"
    upstream = _Upstream(
        json.dumps(
            {
                "model": MODEL,
                "choices": [{"message": {"role": "assistant", "content": "Reply to [EMAIL_1] tomorrow"}}],
            }
        ).encode()
    )
    spy = _Spy()
    gateway = _gateway(session.get_bind(), evidence, upstream, classifier=spy)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(prompt),
        session=session,
    )

    assert response.status == 200
    assert len(upstream.calls) == 1
    forwarded = upstream.calls[0]
    assert forwarded["method"] == "POST"
    assert forwarded["url"] == f"{UPSTREAM}{PATH}"
    assert SECRET.encode() not in forwarded["body"]
    assert b"[EMAIL_1]" in forwarded["body"]
    assert b"about the notes" in forwarded["body"]
    assert SECRET in response.body.decode()
    assert restored_answer in response.body.decode()
    assert response.response_classifier is not None
    assert response.response_classifier["label"] == "C2"
    assert response.response_classifier["model"] == PATTERN_MODEL
    assert response.response_classifier["version"] == PATTERN_VERSION
    assert response.response_classifier["confidence"] == INDICATOR_CONFIDENCE
    assert response.response_classifier["input_hash"] == hash_text(restored_answer)
    assert spy.seen[0] == prompt
    assert spy.seen[-1] == restored_answer
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.ALLOW
    assert response.evaluation.envelope.tool == "model_call"
    assert response.evaluation.envelope.subject_id == MODEL
    assert any(
        check.check == "AGENT_REGISTRY" and check.result is CheckResult.PASS
        for check in response.evaluation.checks
    )


def test_evidence_holds_hashes_not_content(session, evidence):
    _register(session)
    session.flush()
    prompt = f"Email {SECRET} about the notes"
    upstream = _Upstream(
        json.dumps(
            {
                "choices": [
                    {"message": {"content": "Reply to [EMAIL_1] tomorrow"}}
                ]
            }
        ).encode()
    )
    gateway = _gateway(session.get_bind(), evidence, upstream)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(prompt),
        session=session,
    )

    assert response.evaluation is not None
    events = _events(session, response.evaluation.request_id)
    kinds = [event.event_type for event in events]
    assert EventType.POLICY_DECISION in kinds
    assert EventType.MODEL_CALL in kinds
    assert EventType.EGRESS in kinds
    blob = canonical_json([event.payload for event in events])
    assert SECRET not in blob
    assert prompt not in blob
    assert "Reply to" not in blob
    model_call = next(event for event in events if event.event_type is EventType.MODEL_CALL)
    egress = next(event for event in events if event.event_type is EventType.EGRESS)
    assert model_call.payload["manifest_hash"]
    assert len(model_call.payload["manifest_hash"]) == 64
    assert model_call.payload["redacted_spans"] == 1
    assert model_call.payload["sent"] is True
    assert egress.payload["manifest_hash"] == model_call.payload["manifest_hash"]
    assert egress.payload["note"] == "Payload content is not copied to the evidence plane."
    recorded = model_call.payload["response_classifier"]
    assert recorded["input_hash"] == hash_text(f"Reply to {SECRET} tomorrow")
    assert SECRET not in canonical_json(recorded)
    decision = session.exec(select(GovernanceDecision)).one()
    assert SECRET not in canonical_json(decision.attributes)
    assert SECRET not in canonical_json(decision.checks)
    assert verify_chain(session) == len(session.exec(select(EvidenceEvent)).all())


def test_token_usage_counts_against_the_run_budget(session, evidence):
    _register(session)
    session.flush()
    run = Run(max_tokens=10, max_cost_usd=1.0)
    upstream = _Upstream(
        json.dumps(
            {
                "choices": [{"message": {"content": "ack"}}],
                "usage": {"total_tokens": 10, "cost_usd": 0.25},
            }
        ).encode()
    )
    gateway = _gateway(session.get_bind(), evidence, upstream)
    first = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body("quota-marker-phrase"),
        session=session,
        run=run,
    )

    assert first.status == 200
    assert len(upstream.calls) == 1
    assert run.tokens_used == 10
    assert run.cost_usd_used == 0.25
    assert run.calls_used == 1
    stored = session.get(RunRecord, run.id)
    assert stored is not None
    assert stored.tokens_used == 10
    assert stored.cost_usd_used == 0.25
    assert first.evaluation is not None
    model_call = next(
        event
        for event in _events(session, first.evaluation.request_id)
        if event.event_type is EventType.MODEL_CALL
    )
    assert model_call.payload["tokens"] == 10
    assert model_call.payload["cost_usd"] == 0.25
    assert model_call.payload["budget"]["tokens_used"] == 10
    assert "quota-marker-phrase" not in canonical_json(model_call.payload)

    second = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body("once more"),
        session=session,
        run=run,
    )
    assert second.status == 403
    assert second.evaluation is not None
    assert second.evaluation.outcome is Outcome.DENY
    assert any(check.citation == "max_tokens" for check in second.evaluation.basis())
    assert len(upstream.calls) == 1
    assert run.calls_used == 1
    assert run.tokens_used == 10


def test_missing_agent_name_is_deny(session, evidence):
    _register(session)
    session.flush()
    upstream = _Upstream(b"{}")
    gateway = _gateway(session.get_bind(), evidence, upstream)
    response = gateway.handle(
        "POST",
        PATH,
        {},
        _body("hello"),
        session=session,
    )

    assert upstream.calls == []
    assert response.status == 403
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.DENY
    assert response.evaluation.envelope.agent_name is None
    assert any(
        check.check == "AGENT_IDENTITY" and "no agent name" in check.citation
        for check in response.evaluation.checks
    )
    events = _events(session, response.evaluation.request_id)
    assert events
    assert all(event.actor_type is not ActorType.HUMAN for event in events)


def test_agent_name_must_match_the_client_configuration(session, evidence):
    _register(session, "dossier-agent")
    _register(session, "eligibility-assistant")
    session.flush()
    upstream = _Upstream(b"{}")
    gateway = _gateway(session.get_bind(), evidence, upstream)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "eligibility-assistant"},
        _body("hello"),
        session=session,
    )

    assert upstream.calls == []
    assert response.status == 403
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.DENY
    assert any(
        "authenticated client configuration" in check.citation
        for check in response.evaluation.checks
    )


def test_registered_agent_and_approved_system_trigger_are_not_denials(session, evidence):
    """A matching agent is admitted. An approved system trigger is admitted as SYSTEM."""
    _register(session)
    session.flush()
    upstream = _Upstream(
        json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
    )
    agent_gateway = _gateway(session.get_bind(), evidence, upstream)
    agent_response = agent_gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body("hello"),
        session=session,
    )
    assert agent_response.status == 200
    assert agent_response.evaluation is not None
    assert agent_response.evaluation.outcome is Outcome.ALLOW
    agent_call = next(
        event
        for event in _events(session, agent_response.evaluation.request_id)
        if event.event_type is EventType.MODEL_CALL
    )
    assert agent_call.actor_type is ActorType.AGENT
    assert agent_call.actor_id == "dossier-agent"

    trigger_gateway = _gateway(
        session.get_bind(),
        evidence,
        upstream,
        client=_client(agent_names=frozenset(), system_triggers=frozenset({"nightly-refresh"})),
    )
    trigger_response = trigger_gateway.handle(
        "POST",
        PATH,
        {"system_trigger": "nightly-refresh"},
        _body("hello"),
        session=session,
    )
    assert trigger_response.status == 200
    assert len(upstream.calls) == 2
    assert trigger_response.evaluation is not None
    assert trigger_response.evaluation.outcome is Outcome.ALLOW
    assert trigger_response.evaluation.envelope.agent_name is None
    assert any(
        check.check == "SYSTEM_TRIGGER" and check.result is CheckResult.PASS
        for check in trigger_response.evaluation.checks
    )
    trigger_call = next(
        event
        for event in _events(session, trigger_response.evaluation.request_id)
        if event.event_type is EventType.MODEL_CALL
    )
    assert trigger_call.actor_type is ActorType.SYSTEM
    assert trigger_call.actor_id == "nightly-refresh"
    assert trigger_call.actor_type is not ActorType.HUMAN

    bound = _gateway(
        session.get_bind(),
        evidence,
        upstream,
        client=_client(agent_name="dossier-agent"),
    )
    bound_response = bound.handle("POST", PATH, {}, _body("hello"), session=session)
    assert bound_response.status == 200
    assert bound_response.evaluation is not None
    assert bound_response.evaluation.envelope.agent_name == "dossier-agent"
    assert bound_response.evaluation.outcome is Outcome.ALLOW


def test_unregistered_agent_and_kill_switch_still_deny(session, evidence):
    _register(session)
    set_kill_switch(
        session,
        "dossier-agent",
        True,
        evidence=evidence,
        actor_id="owner",
    )
    session.flush()
    upstream = _Upstream(b"{}")
    killed = _gateway(session.get_bind(), evidence, upstream)
    killed_response = killed.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body("hello"),
        session=session,
    )
    assert upstream.calls == []
    assert killed_response.evaluation is not None
    assert killed_response.evaluation.outcome is Outcome.DENY
    assert any(check.check == "KILL_SWITCH" for check in killed_response.evaluation.checks)

    ghost = _gateway(
        session.get_bind(),
        evidence,
        upstream,
        client=_client(agent_names=frozenset({"ghost-agent"})),
    )
    ghost_response = ghost.handle(
        "POST",
        PATH,
        {"agent_name": "ghost-agent"},
        _body("hello"),
        session=session,
    )
    assert upstream.calls == []
    assert ghost_response.evaluation is not None
    assert ghost_response.evaluation.outcome is Outcome.DENY
    assert any(check.check == "AGENT_REGISTRY" for check in ghost_response.evaluation.checks)


def test_typed_classification_wins(session, evidence):
    _register(session)
    session.flush()
    upstream = _Upstream(
        json.dumps({"choices": [{"message": {"content": "noted"}}]}).encode()
    )
    gateway = _gateway(session.get_bind(), evidence, upstream)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent", "classification": "C0"},
        _body(f"Email {SECRET} about the notes"),
        session=session,
    )

    assert response.status == 200
    assert response.evaluation is not None
    assert response.evaluation.attributes["classification"] == "C0"
    assert "classifier" not in response.evaluation.attributes
    assert SECRET.encode() in upstream.calls[0]["body"]


def test_evidence_store_unavailable_refuses_the_call(session, engine):
    _register(session)
    session.commit()
    upstream = _Upstream(b"{}")
    gateway = _gateway(engine, _Down(engine), upstream)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(f"Email {SECRET} about the notes"),
        session=session,
    )

    assert response.status == 503
    assert upstream.calls == []
    assert json.loads(response.body)["error"] == "evidence store unavailable"
    assert session.exec(select(EvidenceEvent)).all() == []
    assert session.exec(select(GovernanceDecision)).all() == []


def test_http_server_refuses_before_upstream_and_returns_an_allow(session, evidence, engine):
    """``serve`` is the same path: the socket does not forward a denial."""
    _register(session)
    session.commit()
    upstream = _Upstream(
        json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
    )
    gateway = _gateway(engine, evidence, upstream)
    server = gateway._make_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        denied = _post(port, {}, _body("hello"))
        assert denied.status == 403
        denied.read()
        assert upstream.calls == []
        allowed = _post(port, {"agent_name": "dossier-agent"}, _body("hello"))
        assert allowed.status == 200
        assert json.loads(allowed.read())["choices"][0]["message"]["content"] == "ok"
        assert len(upstream.calls) == 1
        assert upstream.calls[0]["url"] == f"{UPSTREAM}{PATH}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _post(port: int, headers: dict[str, str], body: bytes):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("POST", PATH, body=body, headers={**headers, "Content-Length": str(len(body))})
    return connection.getresponse()


def test_serve_command_is_the_openai_compatible_gateway():
    args = build_parser().parse_args(
        [
            "serve",
            "--provider",
            "openai-compatible",
            "--upstream",
            "https://api.openai.com",
        ]
    )
    assert args.command == "serve"
    assert args.provider == "openai-compatible"
    assert args.upstream == "https://api.openai.com"
    assert args.func.__name__ == "cmd_serve"

    bound = build_parser().parse_args(
        [
            "serve",
            "--provider",
            "openai-compatible",
            "--upstream",
            "https://api.openai.com",
            "--principal",
            "alice",
            "--purpose",
            "COLLABORATION",
            "--agent",
            "dossier-agent",
            "--system-trigger",
            "nightly-refresh",
        ]
    )
    assert bound.principal == "alice"
    assert bound.agent == ["dossier-agent"]
    assert bound.system_trigger == ["nightly-refresh"]

    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--mcp", "--root-config", "config.json"])
