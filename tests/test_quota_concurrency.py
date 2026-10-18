"""Real authority, file-backed databases, and independent concurrent sessions."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4
from dataclasses import replace

import pytest
from sqlmodel import Session, select

from conftest import NOW
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter
from kognita.governance import record
from kognita.anomaly import _observe
from kognita.models import _AnomalyDay, _BlastUsage, RunRecord
from kognita.tools import Run
from kognita.vocabulary import Outcome
from test_limits_s4 import _configure, _decide
from test_fleet_s6b import _family, _scope, _child


def _store(tmp_path):
    engine = make_engine(tmp_path / 'concurrent.db')
    create_all(engine)
    return engine, EvidenceWriter(engine, clock=lambda: NOW)


def _parallel(count, work):
    barrier = Barrier(count)
    def worker(index):
        return work(index, barrier)
    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(worker, range(count)))


@pytest.mark.parametrize('limit,cap', [('per_minute', 1), ('max_daily_actions', 2)])
def test_quota_admission_atomic(tmp_path, limit, cap):
    engine, evidence = _store(tmp_path)
    with Session(engine) as session:
        _, env = _configure(session, evidence, **{limit: cap})
        # Isolate quota over-admission from the first anomaly-row insert race.
        session.add(_AnomalyDay(agent_name='worker', day=NOW.date().isoformat(), body={'actions': 0}))
        evaluation = _decide(session, env)
        assert evaluation.outcome is Outcome.ALLOW
        session.commit()
    def work(index, barrier):
        with Session(engine) as session:
            stale = replace(evaluation, request_id=str(uuid4()))
            barrier.wait(timeout=10)
            result = record(session, stale, evidence=evidence, now=NOW)
            session.commit()
            assert all(check.citation for check in result.checks)
            return result.outcome
    results = _parallel(6, work)
    assert results.count(Outcome.ALLOW) == cap
    assert results.count(Outcome.DENY) == 6 - cap
    with Session(engine) as session:
        assert len(session.exec(select(_BlastUsage)).all()) == cap


def test_delegation_reservations_atomic(tmp_path):
    engine, evidence = _store(tmp_path)
    with Session(engine) as session:
        _, _, _, parent = _family(session, evidence)
        session.commit()
    def work(index, barrier):
        with Session(engine) as session:
            scope = _scope()
            scope['max_calls'] = 6
            barrier.wait(timeout=10)
            try:
                child = _child(session, evidence, Run(id=parent.id), scope=scope)
                session.commit()
                return child.max_calls
            except ValueError as error:
                assert 'S6B-ATTENUATION' in str(error)
                session.rollback()
                return 0
    reservations = _parallel(4, work)
    assert sum(reservations) == 6
    with Session(engine) as session:
        assert session.get(RunRecord, parent.id).max_calls == 4


@pytest.mark.parametrize('existing', [False, True])
def test_anomaly_counter_atomic(tmp_path, existing):
    engine, evidence = _store(tmp_path)
    with Session(engine) as session:
        _, env = _configure(session, evidence)
        evaluation = _decide(session, env)
        assert evaluation.allowed
        if existing:
            session.add(_AnomalyDay(agent_name='worker', day=NOW.date().isoformat(), body={'actions': 7}))
        session.commit()
    def work(index, barrier):
        with Session(engine) as session:
            barrier.wait(timeout=10)
            result = _observe(session, replace(evaluation, request_id=str(uuid4())), evidence, NOW)
            session.commit()
            assert result.allowed
    _parallel(6, work)
    with Session(engine) as session:
        rows = session.exec(select(_AnomalyDay)).all()
        assert len(rows) == 1
        assert rows[0].body['actions'] == (7 if existing else 0) + 6
