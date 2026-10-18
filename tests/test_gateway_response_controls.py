"""Response controls with real authority, outside the legacy fixture allowlist."""
import json

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.models import EvidenceEvent, GovernanceDecision, ReviewQueueItem
from kognita.review import define_risk_criteria
from kognita.replay import replay_decision
from kognita.tools import Run, ToolRegistry, run_governed
from kognita.vocabulary import EventType, Outcome
from test_gateway import MODEL, PATH, _body
from test_gateway_run_controls import _setup


def _call(session, evidence, *, model=MODEL, run=None):
    gateway, backend, headers = _setup(session, evidence)
    backend.response = json.dumps({'model': model, 'choices': [{'message': {'content': 'retained answer'}}],
        'usage': {'total_tokens': 11, 'cost_usd': 0.11}}).encode()
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session, run=run)
    return response, gateway


@pytest.mark.parametrize('cap', ['max_tokens', 'max_cost_usd'])
def test_crossing_usage_cap_withholds_body_and_records_usage(session, evidence, cap):
    run = Run(**{cap: 10 if cap == 'max_tokens' else 0.1})
    response, _ = _call(session, evidence, run=run)
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    assert cap in [c.citation for c in response.evaluation.basis()]
    assert b'retained answer' not in response.body
    assert run.tokens_used == 11 and run.cost_usd_used == 0.11
    call = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.correlation_id == response.evaluation.request_id,
        EvidenceEvent.event_type == EventType.MODEL_CALL)).one()
    assert call.payload["tokens"] == 11
    assert call.payload["cost_usd"] == 0.11
    assert replay_decision(session, response.evaluation.request_id).outcome is Outcome.DENY


def test_served_model_card_denial_is_stored_and_replayed(session, evidence):
    response, gateway = _call(session, evidence, model='unapproved-version')
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    row = session.exec(select(GovernanceDecision).where(
        GovernanceDecision.request_id == response.evaluation.request_id)).one()
    assert row.outcome is Outcome.DENY
    assert any(c['check'] == 'USE_CASE_MODEL' for c in row.checks)
    from kognita.usecases import load_register
    for envelope in (None, response.evaluation.envelope):
        replayed = replay_decision(session, row.request_id, envelope=envelope,
            purposes=gateway.purposes, usecases=load_register(session, as_of=NOW))
        assert replayed.outcome is Outcome.DENY


def _criteria(session, evidence, timing):
    define_risk_criteria(session, 'COLLABORATION', product_complexity_at='high',
        vulnerability_indicators=('bereavement',), profile_deviation_is_high_risk=True,
        minimum_classifier_confidence=0.8, timing=timing,
        evidence=evidence, actor_id='reviewer', now=NOW)


@pytest.mark.parametrize('timing', ['before delivery', 'after delivery'])
def test_gateway_response_feeds_review_queue(session, evidence, timing):
    _setup(session, evidence)
    _criteria(session, evidence, timing)
    response, _ = _call(session, evidence)
    queue = session.exec(select(ReviewQueueItem)).all()
    assert len(queue) == 1
    assert queue[0].timing == timing
    assert response.status == (403 if timing == 'before delivery' else 200)
    if timing == 'before delivery':
        assert b'retained answer' not in response.body
        assert any('review' in c.citation.lower() for c in response.evaluation.basis())


@pytest.mark.parametrize("timing", ["before delivery", "after delivery"])
def test_governed_tool_response_feeds_review_queue_and_holds(session, evidence, timing):
    response, gateway = _call(session, evidence)
    assert response.status == 200
    _criteria(session, evidence, timing)
    registry = ToolRegistry()
    registry.register('model_call', lambda *args: {'answer': 'retained answer'})
    result = run_governed(session, response.evaluation.envelope, registry=registry,
        evidence=evidence, pack=gateway.pack, purposes=gateway.purposes, now=NOW, as_of=NOW)
    assert session.exec(select(ReviewQueueItem)).first() is not None
    assert (result.data is None) == (timing == "before delivery")
    assert result.evaluation.outcome is (Outcome.DENY if timing == "before delivery" else Outcome.ALLOW)


def test_deferred_served_model_denial_replays_after_recovery(session, evidence, engine):
    from kognita.vocabulary import FailureMode
    from kognita.usecases import load_register
    from test_gateway import LOCAL_UPSTREAM, _Toggle, _gateway

    _, backend, headers = _setup(session, evidence)
    session.commit()
    writer = _Toggle(engine)
    writer.clock = lambda: NOW
    backend.response = b'{"model":"unapproved-version","choices":[]}'
    gateway = _gateway(engine, writer, backend, upstream_url=LOCAL_UPSTREAM,
        provider='api.openai.com', failure_mode={'COLLABORATION': FailureMode.DEGRADED})
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session)
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    writer.down = False
    assert gateway.handle('GET', PATH, {}, b'').status == 405
    for envelope in (None, response.evaluation.envelope):
        replayed = replay_decision(session, response.evaluation.request_id, envelope=envelope,
            purposes=gateway.purposes, usecases=load_register(session, as_of=NOW))
        assert replayed.outcome is Outcome.DENY
