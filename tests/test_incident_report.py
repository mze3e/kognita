"""R14 evidence reports and R4 fixed incident guidance."""
import json
from datetime import timedelta

import pytest

from kognita.incidents import _open, _change, _report, _playbook, _PLAYBOOKS
from kognita.models import Interaction
from kognita.vocabulary import EventType


def _scenario(session, evidence):
    at = evidence.clock()
    incident_id = _open(session, evidence, 'provider_failure',
        summary='client-secret private-content', source='private-endpoint', at=at - timedelta(seconds=1))
    session.add(Interaction(id='interaction-secret', client_id='client-secret', use_case='research'))
    session.flush()
    for prompt in ('old private-content', 'new private-content'):
        evidence.emit(session, correlation_id='deployment:agent-secret', event_type=EventType.POLICY_DECISION,
            payload={'action': 'DEPLOYMENT_APPROVED', 'agent': 'agent-secret',
                'deployment': {'system_prompt': prompt, 'tools': ['tool-secret']},
                '_identity': {'model': prompt}})
    evidence.emit(session, correlation_id='interaction-secret', event_type=EventType.MODEL_CALL,
        payload={'sent': True, 'response_hash': 'body-hash', 'model_version': '1',
            'content': 'private-content'})
    evidence.emit(session, correlation_id='interaction-secret', event_type=EventType.POLICY_DECISION,
        payload={'checks': [{'check': 'R3-UNREGISTERED', 'result': 'fail', 'reason': 'client-secret'}]})
    _change(session, evidence, incident_id, summary='private-content recovered', close=True,
        at=at + timedelta(seconds=1))
    return incident_id


def test_report_controls_clients_and_deployment_diff(session, evidence):
    incident_id = _scenario(session, evidence)
    report = _report(session, incident_id)
    assert report['affected_clients'][0]['client_id'] == 'client-secret'
    assert report['controls_failed'][0]['check'] == 'R3-UNREGISTERED'
    assert report['controls_ran'] == report['controls_failed']
    diff = report['deployment_diff'][0]
    assert diff['baseline_available']
    assert diff['changes']['system_prompt'] == {'before': 'old private-content', 'after': 'new private-content'}
    assert diff['identity_after']['model'] == 'new private-content'
    assert report['what_happened']['timeline'][-1]['event_type'] == 'INCIDENT_CLOSED'
    assert 'temporal association' in report['scope']


def test_redacted_has_no_client_ids_or_content(session, evidence):
    incident_id = _scenario(session, evidence)
    report = _report(session, incident_id, redacted=True)
    encoded = json.dumps(report)
    for secret in ('client-secret', 'private-content', 'interaction-secret', 'agent-secret',
                   'tool-secret', 'private-endpoint', incident_id):
        assert secret not in encoded
    assert report['affected_clients'] == 1
    assert report['controls_failed'] == 1
    assert report['deployment_changes'] == 1


def test_missing_evidence_is_explicit(session, evidence):
    incident_id = _open(session, evidence, 'gateway_outage', summary='outage', source='gateway',
        at=evidence.clock() - timedelta(seconds=1))
    report = _report(session, incident_id)
    assert not report['closed']
    assert len(report['controls_missing']) == 3
    assert report['deployment_diff'] == []
    with pytest.raises(ValueError, match='R14'):
        _report(session, 'unknown')


def test_corrupt_chain_refuses_report(session, evidence):
    from evidence_guard import tamper_for_test
    incident_id = _scenario(session, evidence)
    session.commit()
    tamper_for_test(session, 1, {'tampered': True})
    for redacted in (False, True):
        with pytest.raises(ValueError, match='R14'):
            _report(session, incident_id, redacted=redacted)


def test_nine_playbooks():
    expected = {'hallucination', 'data leakage', 'prompt injection', 'unauthorized tool execution',
        'incorrect transaction', 'model outage', 'third-party compromise', 'deepfake instruction',
        'sensitive-data exposure'}
    assert set(_PLAYBOOKS) == expected
    for kind in expected:
        result = _playbook(kind)
        assert result['rule_id'] == 'R4'
        assert all(result[step] for step in ('detect', 'contain', 'investigate'))
    with pytest.raises(ValueError, match='R4'):
        _playbook('unknown')


def test_cli_report_and_playbook(tmp_path, capsys):
    from kognita.cli import main
    from kognita.db import make_engine, create_all, session_scope
    from kognita.evidence import EvidenceWriter
    path = str(tmp_path / 'incident.db')
    engine = make_engine(path)
    create_all(engine)
    with session_scope(engine) as session:
        incident_id = _scenario(session, EvidenceWriter(engine))
    for flags in ([], ['--redacted']):
        assert main(['incident', 'report', incident_id, '--db', path, *flags]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result['rule_id'] == 'R14'
        if flags:
            assert result['redacted']
    assert main(['incident', 'report', 'unknown', '--db', path]) == 1
    assert 'R14' in capsys.readouterr().err
    assert main(['incident', 'playbook', 'prompt injection']) == 0
    assert json.loads(capsys.readouterr().out)['detect']
    assert main(['incident', 'playbook', 'unknown']) == 1
    assert 'R4' in capsys.readouterr().err


def test_deployment_change_before_detection(session, evidence):
    for prompt in ('before', 'after'):
        evidence.emit(session, correlation_id='deployment:agent', event_type=EventType.POLICY_DECISION,
            payload={'action': 'DEPLOYMENT_APPROVED', 'agent': 'agent',
                'deployment': {'system_prompt': prompt}})
    at = evidence.clock() + timedelta(seconds=1)
    incident_id = _open(session, evidence, 'provider_failure', summary='failed', source='provider', at=at)
    _change(session, evidence, incident_id, summary='recovered', close=True, at=at + timedelta(seconds=1))
    diff = _report(session, incident_id)['deployment_diff'][0]
    assert diff['before_detection']
    assert diff['changes']['system_prompt'] == {'before': 'before', 'after': 'after'}


def test_invalid_timeline_refused(session, evidence):
    at = evidence.clock()
    incident_id = _open(session, evidence, 'gateway_outage', summary='outage', source='gateway', at=at)
    _change(session, evidence, incident_id, summary='invalid closure', close=True,
        at=at - timedelta(seconds=1))
    with pytest.raises(ValueError, match='R14'):
        _report(session, incident_id)


def test_single_deployment_marks_missing_baseline(session, evidence):
    evidence.emit(session, correlation_id='deployment:agent', event_type=EventType.POLICY_DECISION,
        payload={'action': 'DEPLOYMENT_APPROVED', 'agent': 'agent', 'deployment': {'model': '1'}})
    incident_id = _open(session, evidence, 'provider_failure', summary='failed', source='provider',
        at=evidence.clock() - timedelta(seconds=1))
    report = _report(session, incident_id)
    assert report['deployment_diff'][0]['baseline_available'] is False
