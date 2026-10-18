"""Operation authorization with real authority and the production handler."""
from dataclasses import replace

import pytest
from sqlmodel import select

from kognita.canonical import canonical_hash
from kognita.envelope import envelope_hash
from kognita.models import EvidenceEvent
from kognita.tools import Run
from kognita.vocabulary import Outcome
from test_gateway import _body
from test_gateway_run_controls import _setup


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    yield


@pytest.mark.parametrize('path', [
    '/admin/destructive-operation', '/v1/files', '/v1/models',
    '/v1/chat/completions/../files', '/v1/%63hat/completions',
    '//v1/chat/completions', '/v1/chat/completions?operation=delete',
    '/v1/chat/completions#fragment', '/v1/chat/completions/',
    'https://example.com/v1/chat/completions',
])
def test_unsupported_operation_is_denied_before_forwarding(session, evidence, path):
    gateway, backend, headers = _setup(session, evidence)
    response = gateway.handle('POST', path, headers, _body('hello'),
                              session=session, run=Run())
    assert response.evaluation.outcome is Outcome.DENY
    assert any(c.citation == 'S5-MODEL-ENDPOINT' for c in response.evaluation.basis())
    assert response.status == 403
    assert backend.calls == []


@pytest.mark.parametrize('path', [
    '/v1/chat/completions', '/v1/completions', '/v1/responses', '/v1/embeddings',
    'v1/chat/completions',
])
def test_supported_operation_is_bound_to_decision_and_evidence(session, evidence, path):
    gateway, backend, headers = _setup(session, evidence)
    response = gateway.handle('POST', path, headers, _body('hello'),
                              session=session, run=Run())
    assert response.status == 200
    normalized = '/' + path.lstrip('/')
    evaluation = response.evaluation
    assert evaluation.envelope.arguments['_model_path'] == normalized
    assert backend.calls[0]['url'].endswith(normalized)
    changed = replace(evaluation, envelope=replace(evaluation.envelope,
        arguments={**evaluation.envelope.arguments, '_model_path': '/v1/other'}))
    assert envelope_hash(changed.envelope, evaluation.attributes, evaluation.checks) != envelope_hash(
        evaluation.envelope, evaluation.attributes, evaluation.checks)
    events = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.correlation_id == evaluation.request_id)).all()
    assert any(e.payload.get('envelope', {}).get('arguments', {}).get('sha256')
               == canonical_hash(evaluation.envelope.arguments)
               and e.payload.get('envelope_hash') == evaluation.envelope_hash for e in events)
