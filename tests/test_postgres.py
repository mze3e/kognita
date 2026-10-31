"""T2 integration checks, enabled only with a disposable Postgres URL."""
import multiprocessing as _mp
import os as _os
import uuid as _uuid

import pytest
from sqlalchemy import text as _text
from sqlalchemy.engine import make_url as _make_url
from sqlmodel import Session as _Session, select as _select

from kognita.db import create_all as _create_all, make_engine as _make_engine
from kognita.evidence import EvidenceWriter as _Writer, verify_chain as _verify
from kognita.models import EvidenceEvent as _Event, RunRecord as _Run
from kognita.postgres import _apply_rls
from kognita.vocabulary import EventType as _Type

pytestmark = pytest.mark.skipif(
    not _os.environ.get('KOGNITA_TEST_POSTGRES_URL'),
    reason='KOGNITA_TEST_POSTGRES_URL is not set',
)


@pytest.fixture
def _store():
    url = _make_url(_os.environ['KOGNITA_TEST_POSTGRES_URL'])
    if url.drivername in ('postgres', 'postgresql'):
        url = url.set(drivername='postgresql+psycopg')
    admin = _make_engine(url.render_as_string(hide_password=False))
    schema = '_t2_' + _uuid.uuid4().hex
    with admin.begin() as connection:
        connection.exec_driver_sql(f'CREATE SCHEMA {schema}')
    scoped = url.update_query_dict({'options': f'-csearch_path={schema}'})
    scoped = scoped.render_as_string(hide_password=False)
    engine = _make_engine(scoped)
    try:
        _create_all(engine)
        yield engine, scoped
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA {schema} CASCADE')
        admin.dispose()


def _write(url, ready, start):
    engine = _make_engine(url)
    try:
        ready.put(True)
        assert start.wait(30)
        for index in range(8):
            with _Session(engine) as session:
                _Writer(engine).emit(session, correlation_id=str(index), event_type=_Type.TOOL_CALL)
                session.commit()
    finally:
        engine.dispose()


def test_concurrent_writers_no_gaps(_store):
    engine, url = _store
    context = _mp.get_context('spawn')
    ready, start = context.Queue(), context.Event()
    writers = [context.Process(target=_write, args=(url, ready, start)) for _ in range(4)]
    try:
        for writer in writers:
            writer.start()
        for _ in writers:
            assert ready.get(timeout=30)
        start.set()
        for writer in writers:
            writer.join(60)
            assert writer.exitcode == 0
        with _Session(engine) as session:
            assert _verify(session) == 32
            assert list(session.exec(_select(_Event.sequence).order_by(_Event.sequence))) == list(range(1, 33))
            _Writer(engine).emit(session, correlation_id='rolled-back', event_type=_Type.TOOL_CALL)
            session.rollback()
            _Writer(engine).emit(session, correlation_id='committed', event_type=_Type.TOOL_CALL)
            session.commit()
            assert _verify(session) == 33
    finally:
        for writer in writers:
            if writer.is_alive():
                writer.terminate()
                writer.join()
        ready.close()


def test_agent_a_cannot_read_agent_b_runs(_store):
    engine, _ = _store
    role = '_t2_' + _uuid.uuid4().hex
    with engine.begin() as connection:
        _apply_rls(connection)
        _apply_rls(connection)
        schema = connection.exec_driver_sql('SELECT current_schema()').scalar_one()
        connection.exec_driver_sql(f'CREATE ROLE {role} NOSUPERUSER NOBYPASSRLS')
        connection.exec_driver_sql(f'GRANT USAGE ON SCHEMA {schema} TO {role}')
        connection.exec_driver_sql(f'GRANT SELECT, INSERT, UPDATE, DELETE ON runs TO {role}')
    try:
        for agent in ('A', 'B'):
            with _Session(engine) as session:
                session.execute(_text(f'SET LOCAL ROLE {role}'))
                session.execute(_text("SELECT set_config('run.agent_id', :agent, true)"), {'agent': agent})
                session.add(_Run(id=agent))
                session.commit()
        with _Session(engine) as session:
            session.execute(_text(f'SET LOCAL ROLE {role}'))
            assert session.exec(_select(_Run)).all() == []
            session.execute(_text("SELECT set_config('run.agent_id', 'A', true)"))
            assert [row.id for row in session.exec(_select(_Run))] == ['A']
            assert session.get(_Run, 'B') is None
            assert session.execute(_text("UPDATE runs SET control_state='revoked' WHERE id='B'")).rowcount == 0
            assert session.execute(_text("DELETE FROM runs WHERE id='B'")).rowcount == 0
            with pytest.raises(Exception, match='row-level security'):
                session.execute(_text("UPDATE runs SET _agent_id='B' WHERE id='A'"))
            session.rollback()
        with _Session(engine) as session:
            session.execute(_text(f'SET LOCAL ROLE {role}'))
            session.execute(_text("SELECT set_config('run.agent_id', '', true)"))
            session.add(_Run(id='missing-owner'))
            with pytest.raises(Exception, match='row-level security'):
                session.flush()
    finally:
        with engine.begin() as connection:
            connection.exec_driver_sql(f'DROP OWNED BY {role}')
            connection.exec_driver_sql(f'DROP ROLE {role}')
