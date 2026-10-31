"""R8 dependency declarations and observed calls use verified local evidence."""
import json

import pytest

from kognita.dependencies import _map
from kognita.models import Agent
from kognita.usecases import model_card_id
from kognita.vocabulary import EventType
from test_usecases import _open, _activate, _spec


def _setup(session, evidence):
    _open(session, evidence)
    session.add(Agent(name='assistant', use_cases=['research-summary'],
        tool_allowlist={'search': {'verb': 'read'}, 'unused': {'verb': 'read'}}))
    session.flush()
    _activate(session, evidence, _spec(agents=('assistant',),
        approved_models=(model_card_id('house', 'summary', '1'),), approved_sources=('research',), data_sources=()))


def _emit(session, evidence, *, tool='search', agent='assistant', purpose='research-summary'):
    evidence.emit(session, correlation_id='request', event_type=EventType.POLICY_DECISION,
        payload={'envelope': {'purpose': purpose, 'agent_name': agent}})
    evidence.emit(session, correlation_id='request', event_type=EventType.TOOL_CALL,
        actor_id=agent, payload={'tool': tool, 'outcome': 'completed'})


def test_declared_vs_observed(session, evidence):
    _setup(session, evidence)
    _emit(session, evidence)
    evidence.emit(session, correlation_id='request', event_type=EventType.MODEL_CALL,
        actor_id='assistant', payload={'sent': True, 'model_card': model_card_id('house', 'summary', '1'),
            'provider': 'house', 'model': 'summary', 'model_version': '1'})
    evidence.emit(session, correlation_id='request', event_type=EventType.RETRIEVAL,
        actor_id='assistant', payload={'items': [{'source_label': 'research'}]})
    result = _map(session)
    assert not [f for f in result['findings'] if f['rule_id'] == 'R8-UNDECLARED']
    assert [(f['kind'], f['target']) for f in result['findings']] == [('tool', 'unused')]
    assert result['paths'][0]['provider'] == 'house'
    assert result['paths'][0]['cloud'] is None
    assert _map(session) == result


def test_undeclared_call_and_unused(session, evidence):
    _setup(session, evidence)
    _emit(session, evidence, tool='rogue')
    findings = _map(session)['findings']
    assert any(f['rule_id'] == 'R8-UNDECLARED' and f.get('target') == 'rogue' for f in findings)
    assert any(f['rule_id'] == 'R8-UNUSED' and f.get('target') == 'search' for f in findings)


def test_unattributed_and_unsent_calls(session, evidence):
    evidence.emit(session, correlation_id='unknown', event_type=EventType.MODEL_CALL,
        payload={'sent': False, 'model': 'not-called'})
    assert _map(session)['observed'] == []
    evidence.emit(session, correlation_id='unknown', event_type=EventType.TOOL_CALL, payload={'tool': 'rogue'})
    assert any('attribution' in f.get('reason', '') for f in _map(session)['findings'])


def test_corrupt_chain_refuses_map(session, evidence):
    from evidence_guard import tamper_for_test
    _emit(session, evidence)
    session.commit()
    tamper_for_test(session, 1, {'changed': True})
    with pytest.raises(ValueError, match='R8-UNDECLARED'):
        _map(session)


def test_cli_map_check(tmp_path, capsys):
    from kognita.cli import main
    from kognita.db import create_all, make_engine, session_scope
    from kognita.evidence import EvidenceWriter
    path = str(tmp_path / 'deps.db')
    engine = make_engine(path)
    create_all(engine)
    assert main(['deps', 'check', '--db', path]) == 0
    capsys.readouterr()
    with session_scope(engine) as session:
        _emit(session, EvidenceWriter(engine), tool='undeclared')
    assert main(['deps', 'map', '--db', path]) == 0
    assert json.loads(capsys.readouterr().out)['findings']
    assert main(['deps', 'check', '--db', path]) == 1
    engine.dispose()


def test_ambiguous_correlation_is_a_finding(session, evidence):
    _emit(session, evidence)
    evidence.emit(session, correlation_id='request', event_type=EventType.POLICY_DECISION,
        payload={'envelope': {'purpose': 'other', 'agent_name': 'assistant'}})
    assert any('ambiguous' in f.get('reason', '') for f in _map(session)['findings'])


def test_served_model_does_not_hide_behind_card(session, evidence):
    _setup(session, evidence)
    _emit(session, evidence)
    evidence.emit(session, correlation_id='request', event_type=EventType.MODEL_CALL,
        actor_id='assistant', payload={'sent': True, 'model_card': model_card_id('house', 'summary', '1'),
            'provider': 'house', 'model': 'summary', 'model_version': '9'})
    assert any(f['rule_id'] == 'R8-UNDECLARED' and f.get('target') == 'house/summary/9'
        for f in _map(session)['findings'])


def test_old_chain_unchanged():
    from sqlmodel import Session
    from test_schema_version import _legacy_store
    from kognita.db import create_all
    from kognita.evidence import export_chain, verify_chain
    engine, artifact = _legacy_store()
    create_all(engine)
    with Session(engine) as session:
        before = export_chain(session)
        assert _map(session)['findings']
        assert verify_chain(session) == 2
        after = export_chain(session)
        assert after['events'] == before['events']
        assert after['head_hash'] == before['head_hash']
        assert before['head_hash'] == artifact['head_hash']
    engine.dispose()
