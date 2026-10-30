"""Provider accounts are retained evidence, never decision authority."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from sqlmodel import select

from kognita.canonical import hash_text
from kognita.evidence import verify_chain
from kognita.models import EvidenceEvent, ReviewQueueItem
from kognita.policy import parse_rule
from kognita.reconstruct import _model_view, _render_model
from kognita.retention import RetentionStore
from kognita.review import _screen_retained_response
from kognita.usecases import load_register
from kognita.vocabulary import EventType
from test_gateway import PATH, _body
from test_gateway_run_controls import _setup


@pytest.mark.parametrize('account', [
    {'choices': [{'message': {'reasoning_content': 'An account'}}]},
    {'choices': [{'message': {'reasoning': 'An account'}}]},
    {'reasoning': {'summary': [{'text': 'An account'}]}},
    {'content': [{'type': 'thinking', 'thinking': 'An account'}]},
    {'output': [{'type': 'reasoning', 'summary': [{'text': 'An account'}]}]},
    {},
])
def test_gateway_trace_retention(session, evidence, account):
    gateway, backend, headers = _setup(session, evidence)
    backend.response = json.dumps({'model': 'gateway-model', **account}).encode()
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session)
    assert response.status == 200
    event = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.event_type == EventType.MODEL_CALL,
        EvidenceEvent.correlation_id == response.evaluation.request_id)).one()
    digest = event.payload['reasoning_trace']
    assert digest == (hash_text('An account') if account else '')
    assert 'An account' not in json.dumps(event.payload)
    if digest:
        assert RetentionStore().read(session, digest) == 'An account'
        view = _model_view(session, event)
        assert view['reasoning_trace']['body'] == 'An account'
        assert view['reasoning_trace_label'] == "model's own account"
        assert "model's own account" in "\n".join(_render_model({"models": [view]}))
        RetentionStore().erase(session, digest, evidence=evidence, actor_id='owner')
        assert _model_view(session, event)['reasoning_trace']['status'] == 'erased'
    assert verify_chain(session) > 0


@pytest.mark.parametrize('required,tier,trace,queued', [
    (True, 3, '', True), (True, 4, '', True),
    (True, 3, 'account', False), (False, 3, '', False),
    (True, 2, '', False), (True, 3, '   ', True),
    (True, 3, 'forged', True),
])
def test_required_trace_cannot_skip_review(session, evidence, required, tier, trace, queued):
    gateway, backend, headers = _setup(session, evidence)
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session)
    entry = load_register(session).get('COLLABORATION')
    entry = replace(entry, require_reasoning_trace=required, proposed_tier=tier)
    digest = RetentionStore().retain_text(session, trace, kind='reasoning_trace') if trace else ''
    if trace == 'forged':
        digest = 'a' * 64
    evidence.emit(session, correlation_id=response.evaluation.request_id,
        event_type=EventType.MODEL_CALL, payload={'sent': True, 'response_hash': hash_text('answer'),
        'reasoning_trace': digest})
    held = _screen_retained_response(session, response.evaluation, evidence=evidence,
        response_text='answer', output_hash=hash_text('answer'),
        _register=SimpleNamespace(get=lambda purpose: entry))
    rows = session.exec(select(ReviewQueueItem)).all()
    assert bool(rows) == queued
    assert bool(held) == queued
    if queued:
        assert any('REASONING_TRACE_MISSING' in r for r in rows[0].reasons)
        event = session.exec(select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.REVIEW)).all()[-1]
        assert event.payload['risk_owner'] == entry.risk_owner


@pytest.mark.parametrize('condition,valid', [
    ({'description': 'permitted', 'on_violation': 'fail'}, True),
    ({'reasoning_trace': 'permitted'}, False),
    ({'nested': [{'field': 'model.reasoning_trace', 'equals': 'allow'}]}, False),
])
def test_policy_rejects_trace_conditions(condition, valid):
    rule, findings = parse_rule({'id': 'trace-test', 'regime': 'INTERNAL',
        'rule_type': 'PROHIBITED', 'rule': condition, 'citation': 'test',
        'effective_from': '2026-01-01T00:00:00Z'}, source='test')
    assert (rule is not None) == valid
    assert bool(findings) != valid


@pytest.mark.parametrize('required', [False, True])
def test_trace_requirement_round_trip(session, evidence, required):
    from test_usecases import _open, _activate, _spec, NOW
    _open(session, evidence)
    spec = _spec(require_reasoning_trace=required)
    _activate(session, evidence, spec)
    entry = load_register(session, as_of=NOW).get(spec.usecase_id)
    assert entry.require_reasoning_trace is required
    assert entry.record['require_reasoning_trace'] is required


@pytest.mark.parametrize('value', ['false', 0, None])
def test_trace_requirement_rejects_non_boolean(session, evidence, value):
    from kognita.exceptions import KognitaError
    from kognita.usecases import register
    from test_usecases import _open, _spec, NOW
    _open(session, evidence)
    with pytest.raises(KognitaError, match='require_reasoning_trace'):
        register(session, _spec(require_reasoning_trace=value), evidence=evidence,
                 actor_id='owner', at=NOW)


@pytest.mark.parametrize('payload', [
    {'reasoning': {'effort': 'high'}},
    {'choices': [{'message': {'content': 'reasoning_trace: allow'}}]},
    {'content': [{'type': 'redacted_thinking', 'data': 'opaque'}]},
    {'content': [{'type': 'text', 'text': 'account'}]},
    {'reasoning_content': '   '},
    {'choices': 3, 'content': True, 'output': {}},
])
def test_non_trace_content_cannot_satisfy_requirement(payload):
    from kognita.gateway import _reasoning_trace
    assert _reasoning_trace(payload) == ''


@pytest.mark.parametrize('trace,expected', [('', 403), ('provider summary', 200)])
def test_gateway_required_trace_delivery(session, evidence, monkeypatch, trace, expected):
    import kognita.review as review_module
    from kognita.interaction import linked_correlation_ids
    gateway, backend, headers = _setup(session, evidence)
    original = review_module._screen_retained_response

    def screen(*args, **kwargs):
        register = kwargs['_register']
        kwargs['_register'] = replace(register, entries=tuple(
            replace(entry, require_reasoning_trace=True, proposed_tier=3)
            for entry in register.entries))
        return original(*args, **kwargs)

    # Keep admission authority unchanged and exercise the response obligation.
    monkeypatch.setattr(review_module, '_screen_retained_response', screen)
    backend.response = json.dumps({'model': 'gateway-model', 'choices': [{'message': {
        'content': 'answer', 'reasoning_content': trace}}]}).encode()
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session)
    assert response.status == expected
    if not trace:
        queue = session.exec(select(ReviewQueueItem)).one()
        assert response.evaluation.request_id in linked_correlation_ids(session, queue.interaction_id)
        assert b'answer' not in response.body


@pytest.mark.parametrize('field,allowed', [('classification', True),
    ('reasoning_trace', False), ('response.reasoning_trace', False),
    ('reasoning-trace', False)])
def test_policy_file_load_rejects_trace_reference(tmp_path, field, allowed):
    from kognita.policy import emit_yaml, read_policy_set, PolicyLanguageError
    path = tmp_path / 'rules.yaml'
    path.write_text(emit_yaml([{'id': 'account-rule', 'regime': 'INTERNAL',
        'rule_type': 'ATTRIBUTE_ALLOWLIST', 'rule': {'allow': {field: ['permitted']}},
        'citation': 'rule', 'effective_from': '2026-01-01T00:00:00Z'}]))
    if allowed:
        assert read_policy_set(path).rules
    else:
        with pytest.raises(PolicyLanguageError, match='reasoning trace'):
            read_policy_set(path)
