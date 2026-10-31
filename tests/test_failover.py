"""R5 fallback admission and evidence, using only local register decisions."""
import json

import pytest

from kognita.evidence import verify_chain
from kognita.gateway import _UpstreamUnavailable
from kognita.models import EvidenceEvent
from kognita.usecases import model_card_id
from sqlmodel import select
from test_gateway import _served
from test_model_change import _setup, _call


@pytest.mark.parametrize('failure', ['connection', 'http', 'change'])
def test_approved_failover_is_evidenced(session, evidence, failure):
    gateway = _setup(session, evidence,
        failover=model_card_id('house', 'backup', '2'), second=True)
    calls = []

    def transport(method, url, headers, body):
        calls.append(json.loads(body)['model'])
        if len(calls) == 1:
            if failure == 'connection':
                raise _UpstreamUnavailable('offline')
            return (503 if failure == 'http' else 200), {}, (b'\xff' if failure == 'http' else _served('9'))
        return 200, {}, json.dumps({'model': 'backup', 'model_version': '2',
            'choices': [{'message': {'content': 'backup answer'}}]}).encode()

    gateway.transport = transport
    response = _call(gateway, session)
    assert response.status == 200
    assert calls == ['summary', 'backup']
    events = session.exec(select(EvidenceEvent)).all()
    decisions = [e for e in events if e.payload.get('rule_id') == 'R5-FAILOVER']
    assert len(decisions) == 1
    assert decisions[0].payload['fallback'] == model_card_id('house', 'backup', '2')
    verify_chain(session)


@pytest.mark.parametrize('target', ['unknown', model_card_id('house', 'summary', '1'),
    model_card_id('other', 'backup', '2')])
def test_unapproved_fallback_denied(session, evidence, target):
    gateway = _setup(session, evidence, failover=target, second=True)
    calls = []

    def transport(*args):
        calls.append(args)
        raise _UpstreamUnavailable('offline')

    gateway.transport = transport
    response = _call(gateway, session)
    assert response.status == 403
    assert len(calls) == 1
    assert any(c.check == 'R5-UNAPPROVED-FALLBACK' for c in response.evaluation.checks)
    verify_chain(session)


def test_fallback_failure_is_bounded(session, evidence):
    gateway = _setup(session, evidence,
        failover=model_card_id('house', 'backup', '2'), second=True)
    calls = []

    def transport(*args):
        calls.append(args)
        raise _UpstreamUnavailable('offline')

    gateway.transport = transport
    response = _call(gateway, session)
    assert response.status == 403
    assert len(calls) == 2
    assert b'R5-UNAPPROVED-FALLBACK' in response.body
    verify_chain(session)


def test_failover_evidence_failure_stops(session, evidence):
    gateway = _setup(session, evidence,
        failover=model_card_id('house', 'backup', '2'), second=True)
    calls = []
    original = evidence.emit

    def emit(session, **event):
        if event.get('payload', {}).get('rule_id') == 'R5-FAILOVER':
            raise ConnectionError('offline')
        return original(session, **event)

    def transport(*args):
        calls.append(args)
        raise _UpstreamUnavailable('offline')

    evidence.emit = emit
    gateway.transport = transport
    assert _call(gateway, session).status == 503
    assert len(calls) == 1


def test_unregistered_fallback_provider_denied(session, evidence):
    from kognita.providers import ProviderRecord
    gateway = _setup(session, evidence,
        failover=model_card_id('house', 'backup', '2'), second=True)
    calls = []

    def transport(*args):
        calls.append(args)
        session.delete(session.get(ProviderRecord, 'house'))
        session.flush()
        raise _UpstreamUnavailable('offline')

    gateway.transport = transport
    response = _call(gateway, session)
    assert response.status == 403
    assert len(calls) == 1
    assert b'R5-UNAPPROVED-FALLBACK' in response.body
    verify_chain(session)


def test_fallback_repeats_admission(session, evidence):
    from test_classification_floor import _authorize
    gateway = _setup(session, evidence,
        failover=model_card_id('house', 'backup', '2'), second=True)
    session.info['_r7_delegation'], _ = _authorize(session, evidence,
        subject_scope={'model': ['summary']})
    calls = []

    def transport(*args):
        calls.append(args)
        raise _UpstreamUnavailable('offline')

    gateway.transport = transport
    response = _call(gateway, session)
    assert response.status == 403
    assert len(calls) == 1
    verify_chain(session)


def test_default_stops_without_failover(session, evidence):
    gateway = _setup(session, evidence)
    calls = []

    def transport(*args):
        calls.append(args)
        raise _UpstreamUnavailable('offline')

    gateway.transport = transport
    assert _call(gateway, session).status == 502
    assert len(calls) == 1
    assert not any(e.payload.get('rule_id') == 'R5-FAILOVER'
        for e in session.exec(select(EvidenceEvent)).all())
    verify_chain(session)


def test_existing_card_requires_use_case_approval(session, evidence):
    from kognita.usecases import register_model_card
    from test_usecases import NOW
    gateway = _setup(session, evidence,
        failover=model_card_id('house', 'backup', '2'))
    register_model_card(session, provider='house', model='backup', version='2',
        intended_use='Fallback', limitations='Test', evaluation_results='Test',
        approver='Luis Ortega', approved_at=NOW, evidence=evidence,
        actor_id='Luis Ortega', at=NOW)
    calls = []

    def transport(*args):
        calls.append(args)
        raise _UpstreamUnavailable('offline')

    gateway.transport = transport
    response = _call(gateway, session)
    assert response.status == 403
    assert b'R5-UNAPPROVED-FALLBACK' in response.body
    assert len(calls) == 1
    verify_chain(session)
