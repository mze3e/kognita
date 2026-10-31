"""R7 compares responses to the card selected at admission, without network calls."""
import json

import pytest

from kognita.evidence import verify_chain
from kognita.usecases import load_register, model_card_id, register_model_card
from kognita.vocabulary import EventType, Outcome
from test_gateway import (
    PATH, _Upstream, _client, _events, _gateway, _register, _served,
    _summary_request,
)
from test_classification_floor import _authorize
from test_usecases import NOW, _activate, _open, _spec


def _setup(session, evidence, *, failover="", second=False):
    _register(session)
    _open(session, evidence)
    cards = [model_card_id("house", "summary", "1")]
    if second:
        register_model_card(session, provider="house", model="backup", version="2",
            intended_use="Fallback", limitations="Test", evaluation_results="Test",
            approver="Luis Ortega", approved_at=NOW, evidence=evidence,
            actor_id="Luis Ortega", at=NOW)
        cards.append(model_card_id("house", "backup", "2"))
    _activate(session, evidence, _spec(usecase_id="COLLABORATION", approved_models=tuple(cards), failover=failover))
    session.info["_r7_delegation"], _ = _authorize(session, evidence, subject_scope={"model": ["summary", "backup"]})
    return _gateway(session.get_bind(), evidence, _Upstream(_served("9")),
        client=_client(), provider="house")


def _call(gateway, session):
    return gateway.handle("POST", PATH, {"agent_name": "dossier-agent", "delegation_id": session.info["_r7_delegation"]},
        _summary_request(), session=session)


@pytest.mark.parametrize("version", ["9", None])
def test_model_change_stops_and_triggers_revalidation(session, evidence, version):
    gateway = _setup(session, evidence)
    gateway.transport = _Upstream(_served(version, "withheld-provider-content"))
    response = _call(gateway, session)
    assert response.status == 403
    assert b"withheld-provider-content" not in response.body
    assert response.evaluation.outcome is Outcome.DENY
    assert any(c.check == "R7-MODEL-CHANGED" for c in response.evaluation.checks)
    triggers = [e for e in _events(session, response.evaluation.request_id)
        if e.event_type is EventType.REVALIDATION_TRIGGERED]
    assert len(triggers) == 1
    assert triggers[0].payload["action"] == "stop"
    assert triggers[0].payload["rule_id"] == "R7-MODEL-CHANGED"
    verify_chain(session)


def test_unchanged_model_has_no_trigger(session, evidence):
    gateway = _setup(session, evidence)
    gateway.transport = _Upstream(_served("1"))
    response = _call(gateway, session)
    assert response.status == 200
    assert not any(e.event_type is EventType.REVALIDATION_TRIGGERED
        for e in _events(session, response.evaluation.request_id))


@pytest.mark.parametrize("changed_again", [False, True])
def test_model_change_falls_back_with_bounded_admission(session, evidence, changed_again):
    fallback = model_card_id("house", "backup", "2")
    gateway = _setup(session, evidence, failover=fallback, second=True)
    calls = []

    def transport(method, url, headers, body):
        payload = json.loads(body)
        calls.append(payload["model"])
        served = {"model": "backup", "model_version": "9" if changed_again else "2",
            "choices": [{"message": {"content": "fallback answer"}}]}
        return 200, {}, _served("9") if len(calls) == 1 else json.dumps(served).encode()

    gateway.transport = transport
    response = _call(gateway, session)
    assert calls == ["summary", "backup"]
    assert response.status == (403 if changed_again else 200)
    assert load_register(session, as_of=NOW).get("COLLABORATION").failover == fallback
    verify_chain(session)


def test_unapproved_fallback_stops(session, evidence):
    gateway = _setup(session, evidence, failover="unknown-card")
    response = _call(gateway, session)
    assert response.status == 403
    assert len(gateway.transport.calls) == 1


def test_other_approved_model_still_triggers_revalidation(session, evidence):
    gateway = _setup(session, evidence, second=True)
    gateway.transport = _Upstream(json.dumps({"model": "backup", "model_version": "2",
        "choices": []}).encode())
    response = _call(gateway, session)
    assert response.status == 403
    assert any(c.check == "R7-MODEL-CHANGED" for c in response.evaluation.checks)
    assert any(e.event_type is EventType.REVALIDATION_TRIGGERED
        for e in _events(session, response.evaluation.request_id))


def test_trigger_write_failure_withholds_response(session, evidence):
    gateway = _setup(session, evidence)
    original = evidence.emit

    def emit(session, **event):
        if event["event_type"] is EventType.REVALIDATION_TRIGGERED:
            raise ConnectionError("store unavailable")
        return original(session, **event)

    evidence.emit = emit
    response = _call(gateway, session)
    assert response.status == 503
    assert len(gateway.transport.calls) == 1


def test_degraded_change_trigger_is_written_on_recovery(session, evidence, engine):
    from sqlmodel import Session
    from test_gateway import _Toggle
    from kognita.vocabulary import FailureMode

    gateway = _setup(session, evidence)
    session.commit()
    writer = _Toggle(engine)
    writer.clock = evidence.clock
    gateway.evidence = writer
    gateway.upstream = "http://127.0.0.1:11434"
    gateway.failure_mode = {"COLLABORATION": FailureMode.DEGRADED}
    response = _call(gateway, session)
    assert response.status == 403
    assert response.evaluation.decision_id is None
    writer.down = False
    gateway.handle("GET", PATH, {}, b"")
    with Session(engine) as fresh:
        triggers = [e for e in _events(fresh, response.evaluation.request_id)
            if e.event_type is EventType.REVALIDATION_TRIGGERED]
        assert len(triggers) == 1
        assert triggers[0].payload["action"] == "stop"
        verify_chain(fresh)
