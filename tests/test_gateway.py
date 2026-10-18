"""The AI gateway: decide, redact, restore, and refuse when evidence cannot be written.

Covers roadmap 0.3 items 3 and 9, and the two gateway identity rules. A denial
does not call the upstream. An allow redacts, forwards, restores, and classifies
the response. Evidence holds hashes, not the prompt or the answer. Fail closed
refuses when the store is down. Degraded proceeds only for a local model with
no client data, and records that call once the store recovers. A classification
header cannot keep that label below C2, and client-identifying content outside
the extracted prompt is client data too, including a unicode escape that the
provider would decode.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from http.client import HTTPConnection

import pytest
from sqlmodel import Session, select

from fixtures import demo_pack as dp

from kognita.canonical import canonical_json, hash_text
from kognita.classify import (
    INDICATOR_CONFIDENCE,
    PATTERN_MODEL,
    PATTERN_VERSION,
    PatternClassifier,
)
from kognita.cli import build_parser, cmd_serve, main
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter, export_chain, verify_chain
from kognita.gateway import ClientConfiguration, Gateway, _prompt_text
from kognita.models import EvidenceEvent, GovernanceDecision, Policy, RunRecord
from kognita.replay import replay_decision
from kognita.registry import register, set_kill_switch
from kognita.tools import Run
from kognita.usecases import retire
from test_usecases import CARD_ID, NOW, _activate, _open, _spec
from kognita.vocabulary import ActorType, CheckResult, EventType, FailureMode, Outcome

SECRET = "ana@example.org"
MODEL = "gateway-model"
PATH = "/v1/chat/completions"
EFFECTIVE = datetime(2026, 1, 1, tzinfo=timezone.utc)
UPSTREAM = "https://api.openai.com"
LOCAL_UPSTREAM = "http://127.0.0.1:11434"
BENIGN = "market hours are public"


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
    purposes = kwargs.pop("purposes", (client.purpose,))
    url = kwargs.pop("upstream_url", UPSTREAM)
    return Gateway(
        engine=engine,
        evidence=evidence,
        upstream=url,
        client=client,
        purposes=purposes,
        transport=upstream,
        **kwargs,
    )


def _register(session, name: str = "dossier-agent") -> None:
    register(session, name=name, owner_exec="Head of Research Ops")


def _live(
    session,
    *,
    provider: str = "api.openai.com",
    purpose: str = "COLLABORATION",
    model: str = MODEL,
    version: str | None = None,
) -> None:
    """A live use case and the one card the gateway's served model must match."""
    from kognita.usecases import load_register, model_card_id, register_model_card
    from test_usecases import ensure_live_purposes

    if load_register(session, as_of=NOW).get(purpose) is not None:
        return
    served = version or model
    evidence = EvidenceWriter(session.get_bind(), clock=lambda: NOW)
    card = model_card_id(provider, model, served)
    register_model_card(
        session,
        provider=provider,
        model=model,
        version=served,
        intended_use="Gateway fixture",
        limitations="Fixture card",
        evaluation_results="Not evaluated",
        approver="Luis Ortega",
        approved_at=NOW,
        evidence=evidence,
        actor_id="Luis Ortega",
        at=NOW,
    )
    ensure_live_purposes(
        session, evidence, (purpose,), at=NOW, approved_models=(card,)
    )


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
    _live(session)
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
    _live(session)
    session.flush()
    prompt = f"Email {SECRET} about the notes"
    upstream = _Upstream(
        json.dumps(
            {
                "model": MODEL,
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
    _live(session)
    session.flush()
    run = Run(max_tokens=10, max_cost_usd=1.0)
    upstream = _Upstream(
        json.dumps(
            {
                "model": MODEL,
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
    _live(session)
    session.flush()
    upstream = _Upstream(
        json.dumps({"model": MODEL, "choices": [{"message": {"content": "ok"}}]}).encode()
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
    _live(session)
    session.flush()
    upstream = _Upstream(
        json.dumps({"model": MODEL, "choices": [{"message": {"content": "noted"}}]}).encode()
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
    _live(session)
    session.commit()
    upstream = _Upstream(
        json.dumps({"model": MODEL, "choices": [{"message": {"content": "ok"}}]}).encode()
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
    assert bound.purpose == "COLLABORATION"
    assert bound.purposes is None
    assert bound.agent == ["dossier-agent"]
    assert bound.system_trigger == ["nightly-refresh"]
    assert bound.mcp is False
    assert bound.provider == "openai-compatible"
    assert args.failure_mode == "FAIL_CLOSED"
    assert bound.failure_mode == "FAIL_CLOSED"

    degraded = build_parser().parse_args(
        [
            "serve",
            "--provider",
            "openai-compatible",
            "--upstream",
            LOCAL_UPSTREAM,
            "--purpose",
            "COLLABORATION",
            "--failure-mode",
            "DEGRADED",
        ]
    )
    assert degraded.failure_mode == "DEGRADED"
    assert degraded.purpose == "COLLABORATION"


class _Toggle(EvidenceWriter):
    """Evidence writes fail until ``down`` is cleared."""

    def __init__(self, engine) -> None:
        super().__init__(engine)
        self.down = True

    def emit(self, session, **kwargs):
        if self.down:
            raise ConnectionError("evidence store unavailable")
        return super().emit(session, **kwargs)


class _Unread:
    """A store that cannot be read, so no decision can be made."""

    def exec(self, *args, **kwargs):
        raise ConnectionError("evidence store unavailable")

    def get(self, *args, **kwargs):
        raise ConnectionError("evidence store unavailable")

    def rollback(self) -> None:
        return None


def _degraded(engine, evidence, transport, **kwargs) -> Gateway:
    return _gateway(
        engine,
        evidence,
        transport,
        failure_mode={"COLLABORATION": FailureMode.DEGRADED},
        **kwargs,
    )


def _stored(engine) -> tuple[list[GovernanceDecision], list[EvidenceEvent]]:
    with Session(engine) as fresh:
        decisions = list(fresh.exec(select(GovernanceDecision)).all())
        events = list(
            fresh.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
        )
    return decisions, events


def test_fail_closed_refuses_when_the_store_is_down_and_does_not_call_the_provider(
    session, engine
):
    """Fail closed is the default and the explicit setting. An unlisted use case is too."""
    _register(session)
    session.commit()
    upstream = _Upstream(b"{}")
    explicit = _gateway(
        engine,
        _Down(engine),
        upstream,
        upstream_url=LOCAL_UPSTREAM,
        failure_mode={"COLLABORATION": FailureMode.FAIL_CLOSED},
    )
    response = explicit.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(BENIGN),
        session=session,
    )
    assert response.status == 503
    assert upstream.calls == []
    assert json.loads(response.body)["error"] == "evidence store unavailable"

    other = _Upstream(b"{}")
    unlisted = _gateway(
        engine,
        _Down(engine),
        other,
        upstream_url=LOCAL_UPSTREAM,
        failure_mode={"OTHER": FailureMode.DEGRADED},
    )
    missed = unlisted.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent", "purpose": "COLLABORATION"},
        _body(BENIGN),
        session=session,
    )
    assert missed.status == 503
    assert other.calls == []


def test_degraded_local_call_without_client_data_is_evidenced_when_the_store_recovers(
    session, engine
):
    _register(session)
    _live(session, provider="127.0.0.1")
    session.commit()
    answer = json.dumps(
        {"model": MODEL, "choices": [{"message": {"content": "the market is open"}}]}
    ).encode()
    upstream = _Upstream(answer)
    writer = _Toggle(engine)
    gateway = _degraded(engine, writer, upstream, upstream_url=LOCAL_UPSTREAM)
    _decisions, seeded = _stored(engine)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(BENIGN),
        session=session,
    )

    assert response.status == 200
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.ALLOW
    assert json.loads(response.body)["choices"][0]["message"]["content"] == "the market is open"
    assert len(upstream.calls) == 1
    assert upstream.calls[0]["url"] == f"{LOCAL_UPSTREAM}{PATH}"
    held_decisions, held_events = _stored(engine)
    assert held_decisions == []
    assert len(held_events) == len(seeded)

    writer.down = False
    recovered = gateway.handle("GET", PATH, {}, b"")
    assert recovered.status == 405
    assert len(upstream.calls) == 1

    decisions, events = _stored(engine)
    request_id = response.evaluation.request_id
    assert [row.request_id for row in decisions] == [request_id]
    assert decisions[0].outcome is Outcome.ALLOW
    assert decisions[0].purpose == "COLLABORATION"
    kinds = [event.event_type for event in events if event.correlation_id == request_id]
    assert EventType.POLICY_DECISION in kinds
    assert EventType.MODEL_CALL in kinds
    model_call = next(event for event in events if event.event_type is EventType.MODEL_CALL)
    assert model_call.payload["destination_is_local"] is True
    assert model_call.payload["prompt_hash"]
    assert model_call.payload["sent"] is True
    stored = canonical_json([event.payload for event in events])
    assert BENIGN not in stored
    assert "the market is open" not in stored
    with Session(engine) as fresh:
        assert verify_chain(fresh) == len(events)


def test_degraded_refuses_a_remote_model_and_client_data_without_calling_the_provider(
    session, engine
):
    _register(session)
    _live(session)
    session.commit()
    modes = {"COLLABORATION": FailureMode.DEGRADED}
    writer = _Down(engine)

    remote_upstream = _Upstream(b"{}")
    remote = _gateway(
        engine,
        writer,
        remote_upstream,
        upstream_url=UPSTREAM,
        failure_mode=modes,
    )
    remote_response = remote.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(BENIGN),
        session=session,
    )
    assert remote_upstream.calls == []
    assert remote_response.status == 403
    assert remote_response.evaluation is not None
    assert remote_response.evaluation.outcome is Outcome.DENY
    assert any(
        check.check == "FAILURE_MODE" and "remote model" in check.citation
        for check in remote_response.evaluation.checks
    )

    local_upstream = _Upstream(b"{}")
    local = _gateway(
        engine,
        writer,
        local_upstream,
        upstream_url=LOCAL_UPSTREAM,
        failure_mode=modes,
    )
    client_data = local.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(f"Email {SECRET} about the notes"),
        session=session,
    )
    assert local_upstream.calls == []
    assert client_data.status == 403
    assert client_data.evaluation is not None
    assert client_data.evaluation.outcome is Outcome.DENY
    assert client_data.evaluation.attributes["classification"] == "C2"
    assert any(
        check.check == "FAILURE_MODE" and check.result is CheckResult.FAIL
        for check in client_data.evaluation.checks
    )
    assert SECRET not in client_data.body.decode()


@pytest.mark.parametrize("header", ["C0", "C1"])
def test_degraded_header_cannot_keep_client_data_below_c2(session, engine, header):
    """A classification header is a floor. It cannot keep an email below C2."""
    _register(session)
    _live(session)
    session.commit()
    upstream = _Upstream(b"{}")
    spy = _Spy()
    gateway = _degraded(
        engine,
        _Down(engine),
        upstream,
        upstream_url=LOCAL_UPSTREAM,
        classifier=spy,
    )
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent", "classification": header},
        _body(f"Email {SECRET} about the notes"),
        session=session,
    )

    assert upstream.calls == []
    assert response.status == 403
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.DENY
    assert response.evaluation.attributes["classification"] == "C2"
    assert any(SECRET in text for text in spy.seen)
    assert any(
        check.check == "FAILURE_MODE"
        and check.result is CheckResult.FAIL
        and "client-identifying" in check.citation
        for check in response.evaluation.checks
    )
    assert SECRET not in response.body.decode()


@pytest.mark.parametrize(
    "payload",
    [
        {"model": MODEL, "messages": [{"role": "user", "content": ""}], "user": SECRET},
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "", "name": SECRET}],
        },
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": BENIGN}],
            "user": SECRET,
        },
    ],
)
def test_degraded_refuses_client_data_outside_the_extracted_prompt(
    session, engine, payload
):
    """The OpenAI user field is forwarded and is not part of the prompt text."""
    _register(session)
    _live(session)
    session.commit()
    assert SECRET not in _prompt_text(payload)
    upstream = _Upstream(b"{}")
    gateway = _degraded(engine, _Down(engine), upstream, upstream_url=LOCAL_UPSTREAM)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        json.dumps(payload).encode(),
        session=session,
    )

    assert upstream.calls == []
    assert response.status == 403
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.DENY
    assert response.evaluation.attributes["classification"] == "C2"
    assert SECRET not in response.body.decode()


def test_degraded_refuses_a_unicode_escaped_email_outside_the_prompt(session, engine):
    """An email that exists only as a JSON escape is still client data."""
    _register(session)
    _live(session)
    session.commit()
    body = (
        '{"model": "' + MODEL + '", "messages": [{"role": "user", "content": ""}], '
        '"user": "ana\\u0040example.org"}'
    ).encode()
    parsed = json.loads(body)
    assert SECRET.encode() not in body
    assert _prompt_text(parsed) == ""
    assert parsed["user"] == SECRET
    upstream = _Upstream(b"{}")
    gateway = _degraded(engine, _Down(engine), upstream, upstream_url=LOCAL_UPSTREAM)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        body,
        session=session,
    )

    assert upstream.calls == []
    assert response.status == 403
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.DENY
    assert response.evaluation.attributes["classification"] == "C2"
    assert SECRET not in response.body.decode()


def test_degraded_low_header_still_calls_a_local_model_without_client_data(
    session, engine
):
    """A header below C2 does not refuse content the classifier leaves below C2."""
    _register(session)
    _live(session, provider="127.0.0.1")
    session.commit()
    upstream = _Upstream(
        json.dumps({"model": MODEL, "choices": [{"message": {"content": "ok"}}]}).encode()
    )
    gateway = _degraded(engine, _Down(engine), upstream, upstream_url=LOCAL_UPSTREAM)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent", "classification": "C0"},
        _body(BENIGN),
        session=session,
    )

    assert response.status == 200
    assert len(upstream.calls) == 1
    assert response.evaluation is not None
    assert response.evaluation.attributes["classification"] == "C1"


def test_degraded_use_case_still_calls_a_remote_model_when_the_store_is_up(session, evidence):
    _register(session)
    _live(session)
    session.commit()
    upstream = _Upstream(
        json.dumps({"model": MODEL, "choices": [{"message": {"content": "ok"}}]}).encode()
    )
    gateway = _degraded(session.get_bind(), evidence, upstream)
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(BENIGN),
        session=session,
    )
    assert response.status == 200
    assert len(upstream.calls) == 1
    assert response.evaluation is not None
    assert _events(session, response.evaluation.request_id)


def test_no_failure_mode_forwards_without_a_decision(session, engine):
    assert {mode.value for mode in FailureMode} == {"FAIL_CLOSED", "DEGRADED"}
    with pytest.raises(ValueError):
        _gateway(
            engine,
            _Down(engine),
            _Upstream(b"{}"),
            failure_mode={"COLLABORATION": "passthrough"},
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "serve",
                "--provider",
                "openai-compatible",
                "--upstream",
                UPSTREAM,
                "--failure-mode",
                "passthrough",
            ]
        )

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
    session.commit()
    denied_upstream = _Upstream(b"{}")
    denied = _degraded(
        engine,
        _Down(engine),
        denied_upstream,
        upstream_url=LOCAL_UPSTREAM,
    )
    denied_response = denied.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(BENIGN),
        session=session,
    )
    assert denied_upstream.calls == []
    assert denied_response.status == 403
    assert denied_response.evaluation is not None
    assert denied_response.evaluation.outcome is Outcome.DENY
    assert any(
        check.citation == "No external model calls" for check in denied_response.evaluation.checks
    )

    unread_upstream = _Upstream(b"{}")
    unread = _degraded(engine, _Down(engine), unread_upstream, upstream_url=LOCAL_UPSTREAM)
    unread_response = unread.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body(BENIGN),
        session=_Unread(),
    )
    assert unread_response.status == 503
    assert unread_upstream.calls == []
    assert json.loads(unread_response.body)["error"] == "evidence store unavailable"


def _seed_agent(path) -> None:
    engine = make_engine(path)
    create_all(engine)
    with Session(engine) as session:
        register(session, name="dossier-agent", owner_exec="Head of Research Ops")
        session.commit()
    engine.dispose()


def _purpose(evaluation):
    matches = [check for check in evaluation.checks if check.check == "PURPOSE"]
    assert len(matches) == 1
    return matches[0]


def _serve_gateway(monkeypatch, db, *purposes: str) -> Gateway:
    """Build the gateway the way ``kognita serve`` does, and do not listen."""
    held: dict[str, Gateway] = {}

    def serve(self, host="127.0.0.1", port=8080):
        held["gateway"] = self

    monkeypatch.setattr(Gateway, "serve", serve)
    argv = [
        "serve",
        "--provider",
        "openai-compatible",
        "--upstream",
        UPSTREAM,
        "--db",
        str(db),
        "--principal",
        "alice",
        "--purpose",
        dp.PURPOSES[0],
        "--actor-location",
        "SG",
        "--agent",
        "dossier-agent",
    ]
    for purpose in purposes:
        argv.extend(["--purposes", purpose])
    assert cmd_serve(build_parser().parse_args(argv)) == 0
    return held["gateway"]


def test_gateway_serve_allows_a_listed_purpose_and_denies_an_unlisted_one(
    tmp_path, monkeypatch
):
    """A registered use case is allowed. A purpose the register does not name is denied.

    ``--purposes`` still limits a purpose header that is not the client's own.
    It does not authorize the call.
    """
    db = tmp_path / "gateway.db"
    _seed_agent(db)
    claimed = dp.PURPOSES[0]
    listed = dp.PURPOSES[1]
    engine = make_engine(db)
    with Session(engine) as session:
        _live(session, purpose=listed)
        session.commit()
    engine.dispose()
    gateway = _serve_gateway(monkeypatch, db, listed)
    assert gateway.client.purpose == claimed
    assert tuple(gateway.purposes) == (listed,)
    upstream = _Upstream(
        json.dumps(
            {
                "model": MODEL,
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        ).encode()
    )
    gateway.transport = upstream

    denied = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent", "purpose": claimed},
        _body("hello"),
    )
    assert upstream.calls == []
    assert denied.evaluation is not None
    assert denied.evaluation.envelope.purpose == claimed
    assert denied.evaluation.outcome is Outcome.DENY
    denied_check = next(
        check for check in denied.evaluation.checks if check.check == "USE_CASE_REGISTERED"
    )
    assert denied_check.result is CheckResult.FAIL
    assert "not registered" in denied_check.citation

    allowed = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent", "purpose": listed},
        _body("hello"),
    )
    assert allowed.evaluation is not None
    assert allowed.evaluation.envelope.purpose == listed
    assert allowed.evaluation.outcome is Outcome.ALLOW
    allowed_check = next(
        check for check in allowed.evaluation.checks if check.check == "USE_CASE"
    )
    assert allowed_check.result is CheckResult.PASS
    assert len(upstream.calls) == 1


def test_gateway_serve_denies_when_no_purpose_list_is_configured(tmp_path, monkeypatch):
    """A claimed purpose with an empty register is a denial."""
    db = tmp_path / "gateway.db"
    _seed_agent(db)
    claimed = dp.PURPOSES[0]
    gateway = _serve_gateway(monkeypatch, db)
    assert gateway.purposes == ()
    assert gateway.client.purpose == claimed
    upstream = _Upstream(b"{}")
    gateway.transport = upstream
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _body("hello"),
    )
    assert upstream.calls == []
    assert response.evaluation is not None
    assert response.evaluation.envelope.purpose == claimed
    assert response.evaluation.outcome is Outcome.DENY
    denied_check = next(
        check for check in response.evaluation.checks if check.check == "USE_CASE_REGISTERED"
    )
    assert denied_check.result is CheckResult.FAIL
    assert "not registered" in denied_check.citation


def _summary_request() -> bytes:
    return json.dumps(
        {"model": "summary", "messages": [{"role": "user", "content": BENIGN}]}
    ).encode()


def _served(version: str, content: str = "ok") -> bytes:
    return json.dumps(
        {
            "model": "summary",
            "model_version": version,
            "choices": [{"message": {"role": "assistant", "content": content}}],
        }
    ).encode()


def _model_call(session, request_id: str) -> EvidenceEvent:
    return next(
        event
        for event in _events(session, request_id)
        if event.event_type is EventType.MODEL_CALL
    )


def test_a_gateway_call_links_its_card_and_flags_an_unnamed_endpoint(session, evidence):
    """An approved model is linked. An endpoint no live use case names is flagged."""
    _register(session)
    _open(session, evidence)
    _activate(session, evidence, _spec())
    upstream = _Upstream(_served("1"))
    gateway = _gateway(
        session.get_bind(),
        evidence,
        upstream,
        client=_client(purpose="research-summary"),
        provider="house",
    )
    unnamed = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _summary_request(),
        session=session,
    )
    assert unnamed.status == 200
    assert unnamed.evaluation is not None
    assert unnamed.evaluation.outcome is Outcome.ALLOW
    assert unnamed.headers["x-kognita-unregistered-ai-endpoint"] == "true"
    assert b"ok" in unnamed.body
    linked = _model_call(session, unnamed.evaluation.request_id)
    assert linked.payload["model_card"] == CARD_ID
    assert linked.payload["model_card_hash"]
    assert linked.payload["unregistered_ai_endpoint"] is True

    _activate(
        session,
        evidence,
        _spec(
            usecase_id="vendor-copilot",
            unmediated=True,
            assurance_gap="The vendor keeps the prompt log",
            compensating_control="Quarterly sample by the control function",
            ai_endpoints=(UPSTREAM,),
        ),
    )
    named = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _summary_request(),
        session=session,
    )
    assert named.status == 200
    assert named.evaluation is not None
    assert named.evaluation.outcome is Outcome.ALLOW
    assert named.headers["x-kognita-unregistered-ai-endpoint"] == "false"
    named_call = _model_call(session, named.evaluation.request_id)
    assert named_call.payload["unregistered_ai_endpoint"] is False
    assert named_call.payload["model_card"] == CARD_ID

    retire(session, "vendor-copilot", evidence=evidence, actor_id="Alex Chen", at=NOW)
    retired_endpoint = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _summary_request(),
        session=session,
    )
    assert retired_endpoint.evaluation is not None
    assert retired_endpoint.evaluation.outcome is Outcome.ALLOW
    assert retired_endpoint.headers["x-kognita-unregistered-ai-endpoint"] == "true"


def test_a_served_model_without_an_approved_card_is_not_returned(
    session, evidence, tmp_path
):
    """The provider body stays withheld when the served version has no card."""
    _register(session)
    _open(session, evidence)
    _activate(session, evidence, _spec(ai_endpoints=(UPSTREAM,)))
    secret = "provider-body-must-not-leave"
    upstream = _Upstream(_served("9", secret))
    gateway = _gateway(
        session.get_bind(),
        evidence,
        upstream,
        client=_client(purpose="research-summary"),
        provider="house",
    )
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _summary_request(),
        session=session,
    )
    assert upstream.calls
    assert response.status == 403
    assert secret.encode() not in response.body
    assert b"no model card" in response.body
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.ALLOW
    assert response.headers["x-kognita-unregistered-ai-endpoint"] == "false"
    recorded = _model_call(session, response.evaluation.request_id)
    assert recorded.payload["model_card"] == ""
    assert recorded.payload["model_card_hash"] == ""
    decisions = session.exec(select(GovernanceDecision)).all()
    assert len(decisions) == 1
    assert decisions[0].outcome is Outcome.ALLOW
    assert decisions[0].id == response.evaluation.decision_id
    assert all(check["check"] != "USE_CASE_MODEL" for check in decisions[0].checks)
    outcomes = [
        event
        for event in session.exec(
            select(EvidenceEvent).order_by(EvidenceEvent.sequence)
        ).all()
        if event.correlation_id == response.evaluation.request_id
        and event.event_type is EventType.POLICY_DECISION
    ]
    assert [event.payload["outcome"] for event in outcomes] == ["ALLOW", "DENY"]
    denied = outcomes[1]
    assert denied.payload["decision_id"] == decisions[0].id
    assert denied.payload["withheld"] is True
    assert denied.payload["model"] == "summary"
    assert denied.payload["model_version"] == "9"
    check = denied.payload["checks"][0]
    assert check["check"] == "USE_CASE_MODEL"
    assert check["result"] == CheckResult.FAIL.value
    assert "summary@9" in check["citation"]
    assert "no model card" in check["citation"]
    assert "accountable owner: Jane Smith" in check["citation"]
    assert secret not in json.dumps(denied.payload)
    assert verify_chain(session) >= 2
    replayed = replay_decision(session, response.evaluation.request_id)
    assert replayed.outcome is Outcome.ALLOW
    exported = export_chain(session)
    path = tmp_path / "chain.json"
    path.write_text(json.dumps(exported))
    assert main(["evidence", "verify", "--file", str(path)]) == 0


def test_a_deferred_withhold_is_appended_when_the_store_recovers(
    session, engine, evidence
):
    """Degraded mode still appends the denial once the queued decision has an id."""
    _register(session)
    _open(session, evidence)
    _activate(
        session,
        evidence,
        _spec(ai_endpoints=(LOCAL_UPSTREAM,)),
    )
    session.commit()
    secret = "deferred-body-must-not-leave"
    upstream = _Upstream(_served("9", secret))
    writer = _Toggle(engine)
    gateway = _gateway(
        engine,
        writer,
        upstream,
        client=_client(purpose="research-summary"),
        provider="house",
        upstream_url=LOCAL_UPSTREAM,
        failure_mode={"research-summary": FailureMode.DEGRADED},
    )
    response = gateway.handle(
        "POST",
        PATH,
        {"agent_name": "dossier-agent"},
        _summary_request(),
        session=session,
    )
    assert response.status == 403
    assert secret.encode() not in response.body
    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.ALLOW
    assert response.evaluation.decision_id is None
    pending_decisions, pending_events = _stored(engine)
    assert pending_decisions == []
    assert all(
        event.correlation_id != response.evaluation.request_id
        for event in pending_events
    )

    writer.down = False
    recovered = gateway.handle("GET", PATH, {}, b"")
    assert recovered.status == 405
    decisions, events = _stored(engine)
    assert len(decisions) == 1
    assert decisions[0].outcome is Outcome.ALLOW
    request_id = response.evaluation.request_id
    outcomes = [
        event.payload["outcome"]
        for event in events
        if event.correlation_id == request_id
        and event.event_type is EventType.POLICY_DECISION
    ]
    assert outcomes == ["ALLOW", "DENY"]
    denied = next(
        event
        for event in events
        if event.correlation_id == request_id
        and event.event_type is EventType.POLICY_DECISION
        and event.payload["outcome"] == "DENY"
    )
    assert denied.payload["decision_id"] == decisions[0].id
    assert denied.payload["checks"][0]["check"] == "USE_CASE_MODEL"
    assert secret not in json.dumps([event.payload for event in events])
    with Session(engine) as fresh:
        assert verify_chain(fresh) == len(events)
