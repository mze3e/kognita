"""R10 offline faults run through real governance paths."""
import pytest
from kognita.breaker import BreakerError, DrillKind, _simulate
from kognita.evidence import ChainBreak, verify_chain
from kognita.vocabulary import FailureMode
from test_usecases import _open, _activate, _spec, NOW
from kognita.usecases import model_card_id, register_model_card


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    import socket
    def unavailable(*args, **kwargs):
        pytest.fail('R10: a drill attempted a network connection')
    monkeypatch.setattr(socket, 'create_connection', unavailable)


def _setup(session, evidence, mode, fallback=False):
    from kognita.providers import _register
    from test_providers import _details
    _register(session, 'house', _details())
    _open(session, evidence)
    cards = [model_card_id('house', 'summary', '1')]
    if fallback:
        register_model_card(session, provider='house', model='backup', version='2',
            intended_use='Fallback', limitations='Test', evaluation_results='Test',
            approver='Luis Ortega', approved_at=NOW, evidence=evidence,
            actor_id='Luis Ortega', at=NOW)
        cards.append(model_card_id('house', 'backup', '2'))
    _activate(session, evidence, _spec(approved_models=tuple(cards),
        failure_mode=mode.value, failover=cards[-1] if fallback else ''))
    session.commit()


@pytest.mark.parametrize('kind', list(DrillKind), ids=lambda kind: kind.value)
@pytest.mark.parametrize('mode', list(FailureMode))
def test_failure_paths(session, evidence, kind, mode):
    fallback = mode == FailureMode.DEGRADED
    _setup(session, evidence, mode, fallback)
    from sqlalchemy import select as sql_select
    from sqlmodel import SQLModel
    from kognita.evidence import export_chain, verify_export
    tables = [table for table in SQLModel.metadata.sorted_tables
              if table.name not in {'evidence_events', 'evidence_chain_head'}]
    before = {table.name: list(session.execute(sql_select(table)).all()) for table in tables}
    baseline = export_chain(session)
    count = verify_chain(session)
    result = _simulate(session, kind, usecase_id='research-summary', evidence=evidence)
    assert result['failure_mode'] == mode.value
    model_fault = kind in {DrillKind.MODEL_UNAVAILABLE, DrillKind.VENDOR_OUTAGE,
        DrillKind.API_OUTAGE, DrillKind.COMPROMISED_MODEL}
    assert result['released'] == (fallback and model_fault)
    assert result['transcript_complete']
    assert result['chain_events'] == count + 1
    assert {table.name: list(session.execute(sql_select(table)).all()) for table in tables} == before
    events = baseline['events'] + result['transcript']
    assert verify_export(dict(baseline, events=events, event_count=len(events),
                             head_hash=events[-1]['event_hash'])) == len(events)
    assert verify_chain(session) == count + 1


def test_unknown_kind_and_usecase(session, evidence):
    for kind, usecase in [('unknown', 'research-summary'), ('tool_failure', 'unknown')]:
        with pytest.raises(BreakerError, match='R10'):
            _simulate(session, kind, usecase_id=usecase, evidence=evidence)
    assert verify_chain(session) == 0


def test_corrupt_chain_refuses(session, evidence):
    from evidence_guard import tamper_for_test
    from kognita.vocabulary import EventType
    event = evidence.emit(session, correlation_id='old', event_type=EventType.REVIEW)
    session.commit()
    tamper_for_test(session, event.sequence, {'tampered': True})
    with pytest.raises(ChainBreak):
        _simulate(session, 'tool_failure', usecase_id='research-summary', evidence=evidence)


def test_registers_unchanged_and_transcript_verifies(session, evidence):
    from sqlalchemy import select as sql_select
    from sqlmodel import SQLModel
    from kognita.evidence import export_chain, verify_export
    _setup(session, evidence, FailureMode.DEGRADED, True)
    session.flush()
    baseline = export_chain(session)
    tables = [table for table in SQLModel.metadata.sorted_tables
              if table.name not in {'evidence_events', 'evidence_chain_head'}]
    before = {table.name: list(session.execute(sql_select(table)).all()) for table in tables}
    result = _simulate(session, 'compromised_model', usecase_id='research-summary', evidence=evidence)
    after = {table.name: list(session.execute(sql_select(table)).all()) for table in tables}
    assert before == after
    events = baseline['events'] + result['transcript']
    artifact = dict(baseline, events=events, event_count=len(events),
                    head_hash=events[-1]['event_hash'], interest_count=len(events))
    assert verify_export(artifact) == len(events)
    assert any(row['event_type'] == 'REVALIDATION_TRIGGERED' for row in result['transcript'])
    assert any(row['payload'].get('rule_id') == 'R5-FAILOVER' for row in result['transcript'])


def test_unapproved_fallback_refuses(session, evidence):
    from kognita.usecases import revise
    from test_usecases import _spec
    from datetime import timedelta
    _setup(session, evidence, FailureMode.DEGRADED)
    revise(session, _spec(failure_mode='DEGRADED', failover='unknown'),
           evidence=evidence, actor_id='Jane Smith', at=NOW + timedelta(seconds=1))
    evidence.clock = lambda: NOW + timedelta(seconds=2)
    before = verify_chain(session)
    with pytest.raises(BreakerError, match='R10'):
        _simulate(session, 'vendor_outage', usecase_id='research-summary', evidence=evidence)
    assert verify_chain(session) == before


@pytest.mark.parametrize('kind,event_type', [
    ('model_unavailable', 'MODEL_CALL'),
    ('compromised_model', 'REVALIDATION_TRIGGERED'),
    ('retrieval_failure', 'RETRIEVAL'),
    ('identity_service_failure', 'POLICY_DECISION'),
    ('corrupted_context', 'REVIEW'),
    ('tool_failure', 'TOOL_CALL'),
])
def test_missing_path_event_refuses(session, evidence, monkeypatch, kind, event_type):
    from kognita.vocabulary import EventType
    _setup(session, evidence, FailureMode.FAIL_CLOSED)
    original = evidence.emit
    def emit(session, **event):
        if event['event_type'] != EventType(event_type):
            return original(session, **event)
    monkeypatch.setattr(evidence, 'emit', emit)
    before = verify_chain(session)
    with pytest.raises(BreakerError, match='R10: drill evidence is incomplete'):
        _simulate(session, kind, usecase_id='research-summary', evidence=evidence)
    assert verify_chain(session) == before


def test_result_write_failure_refuses(session, evidence, monkeypatch):
    _setup(session, evidence, FailureMode.FAIL_CLOSED)
    original = evidence.emit
    def emit(session, **event):
        if event.get('payload', {}).get('action') == 'simulate':
            raise ConnectionError('store unavailable')
        return original(session, **event)
    monkeypatch.setattr(evidence, 'emit', emit)
    before = verify_chain(session)
    with pytest.raises(ConnectionError):
        _simulate(session, 'tool_failure', usecase_id='research-summary', evidence=evidence)
    assert verify_chain(session) == before


def test_cli_configured_drill(tmp_path, capsys):
    import json
    from sqlmodel import Session
    from kognita.db import make_engine, create_all
    from kognita.evidence import EvidenceWriter
    from kognita.cli import main
    path = tmp_path / 'drills.db'
    engine = make_engine(path)
    create_all(engine)
    with Session(engine) as session:
        writer = EvidenceWriter(engine, clock=lambda: NOW)
        _setup(session, writer, FailureMode.DEGRADED, True)
    assert main(['drill', 'simulate', 'api_outage', '--usecase', 'research-summary', '--db', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['failure_mode'] == 'DEGRADED'
    assert result['released'] is True
    assert main(['drill', 'simulate', 'tool_failure', '--usecase', 'missing', '--db', str(path)]) == 1
    assert 'R10' in capsys.readouterr().err
    engine.dispose()


def test_old_chain_unchanged():
    from sqlmodel import Session
    from test_schema_version import _legacy_store
    from kognita.db import create_all
    from kognita.evidence import EvidenceWriter, export_chain
    engine, artifact = _legacy_store()
    try:
        create_all(engine)
        with Session(engine) as session:
            before = export_chain(session)
            writer = EvidenceWriter(engine, clock=lambda: NOW)
            _setup(session, writer, FailureMode.FAIL_CLOSED)
            _simulate(session, 'tool_failure', usecase_id='research-summary', evidence=writer)
            after = export_chain(session)
            assert after['events'][:2] == before['events']
            assert before['head_hash'] == artifact['head_hash']
            verify_chain(session)
    finally:
        engine.dispose()


def test_signed_drill(engine):
    from sqlmodel import Session
    from kognita.evidence import EvidenceWriter
    from kognita.signing import EvidenceSigner, _generate
    private, public = _generate()
    writer = EvidenceWriter(engine, signing=EvidenceSigner(private), clock=lambda: NOW)
    with Session(engine) as session:
        _setup(session, writer, FailureMode.DEGRADED, True)
        result = _simulate(session, 'compromised_model', usecase_id='research-summary', evidence=writer)
        assert verify_chain(session, public_key=public) == result['chain_events']


@pytest.mark.parametrize('kind', ['model_unavailable', 'vendor_outage', 'api_outage', 'compromised_model'])
def test_degraded_without_fallback_withholds(session, evidence, kind):
    _setup(session, evidence, FailureMode.DEGRADED)
    result = _simulate(session, kind, usecase_id='research-summary', evidence=evidence)
    assert result['outcome'] == 'DENY'
    assert result['released'] is False
    assert len(result['calls']) == 1


def test_conflicting_closed_configuration_refuses(session, evidence):
    _setup(session, evidence, FailureMode.FAIL_CLOSED, True)
    count = verify_chain(session)
    with pytest.raises(BreakerError, match='R10: FAIL_CLOSED'):
        _simulate(session, 'api_outage', usecase_id='research-summary', evidence=evidence)
    assert verify_chain(session) == count


def test_cli_unknown_kind_and_corrupt_chain(tmp_path, capsys):
    from kognita.cli import main
    from kognita.db import make_engine, create_all
    from kognita.evidence import EvidenceWriter
    from kognita.vocabulary import EventType
    from evidence_guard import tamper_for_test
    from sqlmodel import Session
    path = tmp_path / 'drills.db'
    args = ['--usecase', 'research-summary', '--db', str(path)]
    assert main(['drill', 'simulate', 'unknown', *args]) == 1
    assert 'R10: unknown drill kind' in capsys.readouterr().err
    engine = make_engine(path)
    create_all(engine)
    with Session(engine) as session:
        event = EvidenceWriter(engine).emit(session, correlation_id='old', event_type=EventType.REVIEW)
        session.commit()
        tamper_for_test(session, event.sequence, {'tampered': True})
    assert main(['drill', 'simulate', 'tool_failure', *args]) == 1
    assert 'R10: drill refused' in capsys.readouterr().err
    engine.dispose()


def test_missing_result_event_refuses(session, evidence, monkeypatch):
    _setup(session, evidence, FailureMode.FAIL_CLOSED)
    original = evidence.emit
    def emit(session, **event):
        if event.get('payload', {}).get('action') != 'simulate':
            return original(session, **event)
    monkeypatch.setattr(evidence, 'emit', emit)
    before = verify_chain(session)
    with pytest.raises(BreakerError, match='R10: drill result evidence is incomplete'):
        _simulate(session, 'tool_failure', usecase_id='research-summary', evidence=evidence)
    assert verify_chain(session) == before
