"""Response release uses real workload authority, outside legacy fixture patches."""
import json

import pytest
from sqlmodel import select

from kognita.canonical import hash_text
from kognita.models import Agent, EvidenceEvent, Policy
from kognita.registry import approve_deployment
from kognita.retention import RetentionStore
from kognita.tools import Run
from kognita.vocabulary import Classification, Outcome
from test_gateway import MODEL, PATH, SECRET, _body
from test_gateway_run_controls import _setup


@pytest.mark.parametrize('field', ['tool_calls', 'function_call', 'extra'])
def test_response_fields_obey_ceiling(session, evidence, field):
    gateway, backend, headers = _setup(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == headers['agent_name'])).one()
    agent.field_allowlist = {'COLLABORATION': [f'choices.message.{field}']}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
        evidence=evidence, actor_id=agent.owner_exec)
    message = {'content': 'hello'}
    function = {'name': 'contact', 'arguments': json.dumps({'email': SECRET})}
    if field == 'tool_calls':
        message[field] = [{'type': 'function', 'function': function}]
    elif field == 'function_call':
        message[field] = function
    else:
        message[field] = SECRET
    backend.response = json.dumps({'model': MODEL, 'choices': [{'message': message}]}).encode()
    run = Run(classification_ceiling=Classification.C1)
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session, run=run)
    assert response.status == 403
    assert any(c.citation == 'classification_ceiling' for c in response.evaluation.basis())
    assert run.client_data


@pytest.mark.parametrize('violation,outcome', [('fail', Outcome.DENY), ('escalate', Outcome.ESCALATE), ('requires_human', Outcome.HUMAN_APPROVAL)])
def test_response_classification_policy_withholds(session, evidence, violation, outcome):
    gateway, backend, headers = _setup(session, evidence)
    session.add(Policy(regime='INTERNAL', rule_type='ATTRIBUTE_ALLOWLIST',
        rule={'allow': {'classification': ['C0', 'C1']}, 'on_violation': violation},
        citation='response-classification', effective_from=evidence.clock()))
    session.flush()
    backend.response = json.dumps({'model': MODEL, 'choices': [{'message': {'content': SECRET}}]}).encode()
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session, run=Run())
    assert response.status == 403
    assert response.evaluation.outcome is outcome
    assert any('response-classification' in c.citation for c in response.evaluation.basis())


def test_runless_sensitive_response_is_withheld(session, evidence):
    gateway, backend, headers = _setup(session, evidence)
    backend.response = json.dumps({'model': MODEL, 'choices': [{'message': {'content': SECRET}}]}).encode()
    response = gateway.handle('POST', PATH, headers, _body('hello'), session=session)
    assert response.status == 403
    assert any(c.citation == 'S5-RUN-REQUIRED' for c in response.evaluation.basis())
    events = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.correlation_id == response.evaluation.request_id)).all()
    assert any(e.payload.get('_delivered') is False and e.payload.get('_delivered_hash') == '' for e in events)


def test_exact_restored_delivery_retained(session, evidence):
    gateway, backend, headers = _setup(session, evidence)
    backend.response = json.dumps({'model': MODEL, 'choices': [{'message': {'content': '[EMAIL_1]', 'extra': 'removed'}}]}).encode()
    response = gateway.handle('POST', PATH, headers, _body(SECRET), session=session, run=Run())
    assert response.status == 200
    assert SECRET.encode() in response.body
    assert b'removed' not in response.body
    events = session.exec(select(EvidenceEvent).where(EvidenceEvent.correlation_id == response.evaluation.request_id)).all()
    delivered = [e for e in events if e.payload.get('_delivered') is True]
    assert delivered
    digest = delivered[-1].payload['_delivered_hash']
    assert digest == hash_text(response.body.decode())
    assert RetentionStore().read(session, digest).encode() == response.body
    assert all(SECRET not in json.dumps(e.payload) for e in events)


def test_response_confidence_policy_uses_recorded_confidence(session, evidence):
    gateway, backend, headers = _setup(session, evidence)
    session.add(Policy(regime='INTERNAL', rule_type='CLASSIFIER_CONFIDENCE',
        rule={'min_confidence': 0.8}, citation='response-confidence',
        effective_from=evidence.clock()))
    session.flush()
    backend.response = json.dumps({'model': MODEL, 'choices': [{'message': {'content': 'hello'}}]}).encode()
    response = gateway.handle('POST', PATH, headers, _body(SECRET), session=session, run=Run())
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.ESCALATE
    assert any('response-confidence' in c.citation and '0.50' in c.citation for c in response.evaluation.basis())
