"""Machine-speed containment, bounded baseline reads, and human release."""
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlmodel import select

from conftest import NOW
from fixtures.agents import register
from kognita.anomaly import _observe, _retrieved
from kognita.credentials import issue, verify
from kognita.envelope import Envelope, Evaluation
from kognita.governance import record
from kognita.models import _AnomalyDay, EvidenceEvent
from kognita.registry import get, set_kill_switch
from kognita.testing._anomaly import _attacks
from kognita.vocabulary import Outcome, Classification


def _eval(tool='read', outcome=Outcome.ALLOW, client='1'):
    return Evaluation(request_id=str(uuid4()), outcome=outcome, checks=(), attributes={},
        envelope=Envelope(principal='human', purpose='test', tool=tool, agent_name='a',
                          subject_type='client', subject_id=client), as_of=NOW, envelope_hash='')


def _agent(session, evidence):
    agent = register(session, name='a', owner_exec='owner')
    agent.tool_allowlist = {'read': {'verb': 'read'}, 'other': {'verb': 'read'}}
    session.add(agent)
    session.flush()
    return agent


def _day(session, body, days=0):
    session.add(_AnomalyDay(agent_name='a', day=(NOW-timedelta(days=days)).date().isoformat(), body=body))
    session.flush()


@pytest.mark.parametrize('metric,body', [('actions', {'actions':1000}), ('clients', {'clients':[str(i) for i in range(100)]})])
def test_spike_contains_before_release(session, evidence, metric, body):
    _agent(session, evidence)
    secret = issue(session, 'a', evidence=evidence, actor_id='owner')
    second = issue(session, 'a', evidence=evidence, actor_id='owner')
    _day(session, body)
    result = record(session, _eval(client='new'), evidence=evidence, now=NOW)
    assert result.outcome == Outcome.DENY
    assert result.checks[-1].citation == 'S6-VOLUME'
    assert get(session, 'a').kill_switch
    assert not verify(session, 'a', secret) and not verify(session, 'a', second)
    events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    actions = [e.payload.get('action') for e in events]
    assert actions.index('ANOMALY_QUARANTINED') < actions.index('CREDENTIAL_REVOKED') < actions.index('ANOMALY_TOOLS_STOPPED') < actions.index('ANOMALY_ALERT')
    assert actions.count('INCIDENT') == 1
    record(session, _eval(), evidence=evidence, now=NOW)
    assert sum(e.payload.get('action') == 'INCIDENT' for e in session.exec(select(EvidenceEvent))) == 1
    with pytest.raises(ValueError, match='S6-RELEASE'):
        set_kill_switch(session, 'a', False, evidence=evidence, actor_id='intruder')
    set_kill_switch(session, 'a', False, evidence=evidence, actor_id='owner')
    assert not verify(session, 'a', secret)


@pytest.mark.parametrize('tool,count', [('forbidden',3), ('read',10)])
def test_repeated_denials_and_probing(session, evidence, tool, count):
    _agent(session, evidence)
    for i in range(count):
        result = _observe(session, _eval(tool, Outcome.DENY), evidence, NOW)
        assert get(session, 'a').kill_switch == (i == count-1)
    assert result.checks[-1].citation == 'S6-PRIVILEGE-PROBING'


def test_historical_sequence_and_classes(session, evidence):
    _agent(session, evidence)
    for days in range(1,4):
        _day(session, {'actions':20,'clients':['1'], 'tools':['read','other'],
             'sequences':[['read','other']], 'data_classes':['C1']}, days)
    _day(session, {'last_tool':'other'})
    result = _observe(session, _eval(), evidence, NOW)
    assert result.outcome == Outcome.DENY
    assert result.checks[-1].citation == 'S6-TOOL-SEQUENCE'


def test_document_spike_withheld(session, evidence):
    _agent(session, evidence)
    _day(session, {'documents':1000})
    hits, checked = _retrieved(session, _eval(), evidence, [SimpleNamespace(classification=Classification.C1)])
    assert hits == [] and checked.outcome == Outcome.DENY
    assert get(session, 'a').kill_switch


def test_bounded_indexed_history(session, evidence, engine):
    _agent(session, evidence)
    for days in range(1,101):
        _day(session, {'actions':20, 'tools':['read']}, days)
    statements=[]
    def capture(conn, cursor, statement, parameters, context, many):
        if 'FROM _anomaly_days' in statement:
            statements.append(statement)
    event.listen(engine, 'before_cursor_execute', capture)
    try:
        assert _observe(session, _eval(), evidence, NOW).outcome == Outcome.ALLOW
    finally:
        event.remove(engine, 'before_cursor_execute', capture)
    assert len(statements) == 1 and 'LIMIT' in statements[0]
    plan = session.connection().exec_driver_sql("EXPLAIN QUERY PLAN SELECT * FROM _anomaly_days WHERE agent_name='a' AND day <= '2026-06-01' ORDER BY day DESC LIMIT 8").all()
    assert 'ix_anomaly_agent_day' in str(plan)


@pytest.mark.parametrize('name,body,client', _attacks())
def test_adversarial_traffic(session, evidence, name, body, client):
    _agent(session, evidence)
    _day(session, body)
    assert record(session, _eval(client=client), evidence=evidence, now=NOW).outcome == Outcome.DENY


def test_data_class_drift_is_blocked(session, evidence):
    _agent(session, evidence)
    for days in range(1,4):
        _day(session, {'actions':20,'tools':['read'], 'data_classes':['C1']}, days)
    hits, checked = _retrieved(session, _eval(), evidence,
        [SimpleNamespace(classification=Classification.C3)])
    assert hits == [] and checked.checks[-1].citation == 'S6-DATA-CLASS'


def test_current_day_cannot_raise_baseline(session, evidence):
    _agent(session, evidence)
    for days in range(1,4):
        _day(session, {'actions':2, 'tools':['read']}, days)
    _day(session, {'actions':10})
    assert record(session, _eval(), evidence=evidence, now=NOW).outcome == Outcome.DENY


def test_containment_rolls_back_with_evidence(session, evidence):
    _agent(session, evidence)
    secret = issue(session, 'a', evidence=evidence, actor_id='owner')
    _day(session, {'actions':1000})
    session.commit()
    before = len(session.exec(select(EvidenceEvent)).all())
    assert record(session, _eval(), evidence=evidence, now=NOW).outcome == Outcome.DENY
    session.rollback()
    assert not get(session, 'a').kill_switch and verify(session, 'a', secret)
    assert len(session.exec(select(EvidenceEvent)).all()) == before


def test_anomaly_counts_once_toward_breaker(session, evidence):
    from kognita.breaker import set_breaker_threshold, _flags
    from kognita.models import Policy
    _agent(session, evidence)
    set_breaker_threshold(session, 'usecase', 'test', threshold=1, window_seconds=60,
        evidence=evidence, actor_id='owner', owner='owner', citation='test', now=NOW)
    _day(session, {'actions':1000})
    record(session, _eval(), evidence=evidence, now=NOW)
    assert any(p.rule_type == 'CIRCUIT_BREAKER' for p in session.exec(select(Policy)))
    record(session, _eval(), evidence=evidence, now=NOW)
    assert len(_flags(session, NOW, window_seconds=60)) == 1


@pytest.mark.parametrize('mode', ['gateway', 'mcp'])
def test_threshold_crossing_never_reaches_backend(session, evidence, engine, mode):
    from test_gateway import _gateway, _Upstream, _body, _live, PATH
    from test_mcp import _proxy, _Upstream as _Backend, _call, _register
    _live(session)
    _register(session)
    session.add(_AnomalyDay(agent_name='dossier-agent', day=NOW.date().isoformat(),
                            body={'actions':1000}))
    secret = issue(session, 'dossier-agent', evidence=evidence, actor_id='Head of Research Ops')
    headers = {'agent_name':'dossier-agent', 'agent_credential':secret}
    if mode == 'gateway':
        backend = _Upstream(b'{}')
        endpoint, path, body = _gateway(engine, evidence, backend), PATH, _body('hello')
    else:
        backend = _Backend()
        endpoint, path, body = _proxy(engine, evidence, backend), '/mcp', _call()
    response = endpoint.handle('POST', path, headers, body, session=session)
    assert response.status == 403 and backend.calls == []
    assert response.evaluation.outcome == Outcome.DENY
    assert any(c.citation == 'S6-VOLUME' for c in response.evaluation.checks)
    assert not verify(session, 'dossier-agent', secret)
