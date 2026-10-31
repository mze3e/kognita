"""R13 incident lifecycle, automatic detection and portable evidence."""
import json

import pytest
from sqlmodel import select

from kognita.evidence import ChainBreak, verify_chain, verify_export
from kognita.incidents import IncidentKind, _open, _change, _show, _export, _list
from kognita.models import EvidenceEvent
from kognita.vocabulary import EventType


def test_timeline_export(session, evidence):
    incident_id = _open(session, evidence, IncidentKind._PROVIDER_FAILURE,
        summary='R13: provider failed', source='local')
    assert _open(session, evidence, 'provider_failure',
        summary='R13: still failed', source='local') == incident_id
    _change(session, evidence, incident_id, summary='R13: investigating')
    _change(session, evidence, incident_id, summary='R13: recovered', close=True)
    report = _export(session, incident_id)
    assert [row['event_type'] for row in report['timeline']] == [
        'INCIDENT_OPENED', 'INCIDENT_UPDATED', 'INCIDENT_CLOSED']
    assert verify_export(report['evidence']) == verify_chain(session)
    assert any(event.payload.get('action') == 'INCIDENT'
        for event in session.exec(select(EvidenceEvent)).all())
    with pytest.raises(ValueError, match='R13'):
        _change(session, evidence, incident_id, summary='late')
    with pytest.raises(ValueError, match='R13'):
        _show(session, 'unknown')


def test_chain_break_auto_open(session, evidence):
    event = evidence.emit(session, correlation_id='test', event_type=EventType.MODEL_CALL)
    session.commit()
    from evidence_guard import tamper_for_test
    tamper_for_test(session, event.sequence, {'tampered': True})
    with pytest.raises(ChainBreak):
        verify_chain(session)
    assert _list(session)[0]['kind'] == 'chain_break'
    with pytest.raises(ChainBreak):
        verify_chain(session)
    assert len(_list(session)) == 1


def test_provider_failure_auto_open(session, evidence):
    from test_model_change import _setup, _call
    from kognita.gateway import _UpstreamUnavailable
    gateway = _setup(session, evidence)
    def transport(*args):
        raise _UpstreamUnavailable('offline')
    gateway.transport = transport
    assert _call(gateway, session).status == 502
    assert _list(session)[0]['kind'] == 'provider_failure'
    verify_chain(session)


def test_outage_and_degraded_recovery(session, evidence):
    from test_model_change import _setup
    gateway = _setup(session, evidence)
    session.commit()
    gateway._queue_incident('gateway_outage', 'R13: unavailable')
    gateway._queue_incident('degraded_mode', 'R13: degraded')
    emit = evidence.emit
    def unavailable(*args, **kwargs):
        raise ConnectionError('offline')
    evidence.emit = unavailable
    gateway._flush_incidents()
    assert len(gateway._incident_pending) == 2
    evidence.emit = emit
    gateway._flush_incidents()
    assert not gateway._incident_pending
    session.expire_all()
    assert {item['kind'] for item in _list(session)} == {'gateway_outage', 'degraded_mode'}
    for item in _list(session):
        assert _show(session, item['incident_id'])['timeline'][-1]['event_type'] == 'INCIDENT_CLOSED'
    verify_chain(session)


def test_cli_list_show_export(tmp_path, capsys):
    from kognita.cli import main
    from kognita.db import make_engine, create_all, session_scope
    from kognita.evidence import EvidenceWriter
    path = str(tmp_path / 'incidents.db')
    engine = make_engine(path)
    create_all(engine)
    with session_scope(engine) as session:
        incident_id = _open(session, EvidenceWriter(engine), 'gateway_outage',
            summary='R13: outage', source='gateway')
    for operation in ('list', 'show', 'export'):
        argv = ['incident', operation, '--db', path]
        if operation != 'list':
            argv.append(incident_id)
        assert main(argv) == 0
        result = json.loads(capsys.readouterr().out)
        if operation == 'export':
            verify_export(result['evidence'])


def test_real_degraded_period(session, evidence, engine):
    from test_model_change import _setup, _call
    from test_gateway import _Toggle, PATH
    from kognita.vocabulary import FailureMode
    gateway = _setup(session, evidence)
    session.commit()
    writer = _Toggle(engine)
    writer.clock = evidence.clock
    gateway.evidence = writer
    gateway.upstream = 'http://127.0.0.1:11434'
    gateway.failure_mode = {'COLLABORATION': FailureMode.DEGRADED}
    _call(gateway, session)
    assert {item[0] for item in gateway._incident_pending} >= {'gateway_outage', 'degraded_mode'}
    writer.down = False
    gateway.handle('GET', PATH, {}, b'')
    session.expire_all()
    incidents = _list(session)
    assert {item['kind'] for item in incidents} >= {'gateway_outage', 'degraded_mode'}
    for incident in incidents:
        if incident['kind'] in {'gateway_outage', 'degraded_mode'}:
            assert _show(session, incident['incident_id'])['timeline'][-1]['event_type'] == 'INCIDENT_CLOSED'
    verify_chain(session)


def test_truncation_incident_cannot_heal_chain(session, evidence):
    from evidence_guard import lift_evidence_guard
    for number in range(3):
        evidence.emit(session, correlation_id=str(number), event_type=EventType.MODEL_CALL)
    session.commit()
    lift_evidence_guard(session)
    session.connection().exec_driver_sql('DELETE FROM evidence_events WHERE sequence = 3')
    session.expire_all()
    for _ in range(2):
        with pytest.raises(ChainBreak):
            verify_chain(session)
    assert len(_list(session)) == 1
    with pytest.raises(ChainBreak):
        verify_export(_export(session, _list(session)[0]['incident_id'])['evidence'])


def test_fail_closed_outage_is_recorded_on_recovery(session, evidence, engine):
    from test_model_change import _setup, _call
    from test_gateway import _Toggle, PATH
    gateway = _setup(session, evidence)
    session.commit()
    gateway.evidence = _Toggle(engine)
    assert _call(gateway, session).status == 503
    gateway.evidence.down = False
    gateway.handle('GET', PATH, {}, b'')
    session.expire_all()
    incident = _list(session)[0]
    assert incident['kind'] == 'gateway_outage'
    assert _show(session, incident['incident_id'])['timeline'][-1]['event_type'] == 'INCIDENT_CLOSED'
    verify_chain(session)


def test_signed_chain_break_auto_open(engine):
    from sqlmodel import Session
    from evidence_guard import tamper_for_test
    from kognita.evidence import EvidenceWriter
    from kognita.signing import EvidenceSigner, _generate
    private, public = _generate()
    writer = EvidenceWriter(engine, signing=EvidenceSigner(private))
    with Session(engine) as session:
        event = writer.emit(session, correlation_id='test', event_type=EventType.MODEL_CALL)
        tamper_for_test(session, event.sequence, {'tampered': True})
        with pytest.raises(ChainBreak):
            verify_chain(session, public_key=public)
        assert _list(session)[0]['kind'] == 'chain_break'
        assert all(event.signature for event in session.exec(select(EvidenceEvent)).all())


def test_provider_recovery_closes_incident(session, evidence):
    from test_model_change import _setup, _call
    from test_gateway import _served
    from kognita.gateway import _UpstreamUnavailable
    gateway = _setup(session, evidence)
    def unavailable(*args):
        raise _UpstreamUnavailable('offline')
    gateway.transport = unavailable
    assert _call(gateway, session).status == 502
    incident_id = _list(session)[0]['incident_id']
    gateway.transport = lambda *args: (200, {}, _served('1'))
    assert _call(gateway, session).status == 200
    assert _show(session, incident_id)['timeline'][-1]['event_type'] == 'INCIDENT_CLOSED'
    verify_chain(session)


def test_incident_write_failure_stops(session, evidence):
    from test_model_change import _setup, _call
    from kognita.gateway import _UpstreamUnavailable
    gateway = _setup(session, evidence)
    original = evidence.emit
    def emit(session, **event):
        if event['event_type'] == EventType.INCIDENT_OPENED:
            raise ConnectionError('offline')
        return original(session, **event)
    evidence.emit = emit
    def unavailable(*args):
        raise _UpstreamUnavailable('offline')
    gateway.transport = unavailable
    assert _call(gateway, session).status == 503
    assert not _list(session)
