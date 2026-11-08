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


def test_upgrade_existing_event_enum_preserves_chain_and_accepts_assurance(_store):
    """Upgrade a 0.7 native enum twice, then write assurance without rebuilding rows."""
    engine, _ = _store
    from kognita.evidence import export_chain
    with _Session(engine) as session:
        _Writer(engine).emit(session, correlation_id='before-upgrade', event_type=_Type.TOOL_CALL)
        session.commit()
        before = export_chain(session)['events']
    with engine.begin() as connection:
        schema = connection.exec_driver_sql('SELECT current_schema()').scalar_one()
        quote = connection.dialect.identifier_preparer.quote
        enum = f'{quote(schema)}.{quote("eventtype")}'
        connection.exec_driver_sql('ALTER TABLE evidence_events ALTER COLUMN event_type TYPE TEXT USING event_type::text')
        connection.exec_driver_sql(f'DROP TYPE {enum}')
        labels = ', '.join("'" + value.value + "'" for value in _Type if value != _Type.ASSURANCE)
        connection.exec_driver_sql(f'CREATE TYPE {enum} AS ENUM ({labels})')
        connection.exec_driver_sql(f'ALTER TABLE evidence_events ALTER COLUMN event_type TYPE {enum} USING event_type::{enum}')
    _create_all(engine)
    _create_all(engine)
    with _Session(engine) as session:
        assert export_chain(session)['events'] == before
        _Writer(engine).emit(session, correlation_id='after-upgrade', event_type=_Type.ASSURANCE,
                            payload={'action': 'UPGRADE_CHECK'})
        session.commit()
        assert _verify(session) == len(before) + 1


def _outcome(url, ready, start, results, index, mode, private):
    """Submit from a separate process with a deliberately retained stale identity."""
    from kognita.assurance.lifecycle import ProductionOutcome, record_outcome
    from kognita.assurance.records import agent_for, policy
    from kognita.signing import EvidenceSigner
    engine = _make_engine(url)
    try:
        with _Session(engine) as session:
            retained = agent_for(session, 'outcome-worker')
            policy(session, retained.name)
            from kognita.models import Interaction
            retained_interaction = session.get(Interaction, 'concurrent-outcomes')
            pid = session.execute(_text('SELECT pg_backend_pid()')).scalar_one()
            ready.put(pid)
            assert start.wait(20)
            writer = _Writer(engine, signing=EvidenceSigner(private))
            if mode == 'rollback' and index == 0:
                original = writer.emit
                def omit_trigger(session, **kwargs):
                    """Force an incomplete required write inside the savepoint."""
                    if kwargs['event_type'] != _Type.REVALIDATION_TRIGGERED:
                        return original(session, **kwargs)
                writer.emit = omit_trigger
            try:
                result = record_outcome(session, retained.name, ProductionOutcome(
                    interaction_id=retained_interaction.id, intended='correct', actual='adverse',
                    client_impact='loss', adverse=True, source_reference=f'bank:{index}',
                    outcome_id='shared' if mode == 'duplicate' else str(index)),
                    evidence=writer, actor_id='owner')
                session.commit()
                results.put(('ok', result))
            except ValueError as error:
                # Preserve the outer transaction to exercise savepoint rollback.
                session.commit()
                results.put(('refused', str(error)))
    finally:
        engine.dispose()


def _await_advisory_waiters(engine, pids):
    """Observe actual database lock contention, rather than assume scheduling."""
    import time
    from threading import Event
    deadline = time.monotonic() + 20
    with engine.connect() as connection:
        while time.monotonic() < deadline:
            waiting = connection.execute(_text(
                "SELECT pid FROM pg_stat_activity WHERE pid = ANY(:pids) "
                "AND wait_event_type = 'Lock' AND wait_event = 'advisory'"), {'pids': pids}).scalars().all()
            connection.rollback()
            if set(waiting) == set(pids):
                return
            Event().wait(0.01)
    pytest.fail('outcome writers did not all contend on the database lock')


@pytest.mark.parametrize('mode', ['threshold', 'duplicate', 'rollback', 'policy', 'interaction'])
def test_concurrent_outcomes_serialize_safety_state_and_preserve_signed_chain(_store, mode):
    """Contending processes refresh safety state and leave exact, signed evidence."""
    from kognita.assurance import AssurancePolicy, configure
    from kognita.assurance.lifecycle import ProductionOutcome, record_outcome
    from kognita.assurance.records import agent_for
    from kognita.autonomy import _change
    from kognita.models import Interaction
    from kognita.registry import register, approve_deployment
    from kognita.signing import EvidenceSigner, _generate
    engine, url = _store
    private, public = _generate()
    writer = _Writer(engine, signing=EvidenceSigner(private))
    with _Session(engine) as session:
        agent = register(session, name='outcome-worker', owner_exec='owner',
                         use_cases=['p'], evidence=writer, actor_id='owner')
        approve_deployment(session, agent.name, deployment=dict(system_prompt='', instructions='',
            tools=[], permissions=[], data_sources=[], orchestration='', memory_behaviour=''),
            evidence=writer, actor_id='owner')
        _change(session, agent, 3, writer, 'owner', 'AUTONOMY_PROMOTED', {})
        configure(session, agent.name, AssurancePolicy(use_case='p', audience='bank',
            max_adverse_outcomes=2, outcome_step_down=True), evidence=writer, actor_id='owner')
        session.add(Interaction(id='concurrent-outcomes', client_id='client', use_case='p'))
        session.commit()
    context = _mp.get_context('spawn')
    ready, results, start = context.Queue(), context.Queue(), context.Event()
    processes = [context.Process(target=_outcome, args=(url, ready, start, results, i, mode, private))
                 for i in range(3)]
    try:
        # The blocker is the same transaction-level lock used by evidence writes.
        with _Session(engine) as blocker:
            blocker.execute(_text('SELECT pg_advisory_xact_lock(7070002)'))
            for process in processes:
                process.start()
            pids = [ready.get(timeout=20) for _ in processes]
            start.set()
            _await_advisory_waiters(engine, pids)
            if mode == 'interaction':
                blocker.get(Interaction, 'concurrent-outcomes').use_case = 'other'
            elif mode == 'policy':
                # State changed while all callers held old identities/policies.
                configure(blocker, 'outcome-worker', AssurancePolicy(use_case='p', audience='bank',
                    max_adverse_outcomes=1, outcome_step_down=True), evidence=writer, actor_id='owner')
            else:
                record_outcome(blocker, 'outcome-worker', ProductionOutcome(
                    interaction_id='concurrent-outcomes', intended='correct', actual='adverse',
                    client_impact='loss', adverse=True, source_reference='bank:seed', outcome_id='seed'),
                    evidence=writer, actor_id='owner')
            blocker.commit()
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        responses = [results.get(timeout=5) for _ in processes]
        accepted = [result for status, result in responses if status == 'ok']
        refused = [result for status, result in responses if status == 'refused']
        if mode == 'interaction':
            assert accepted == [] and len(refused) == 3
            assert all('outcome must link to an interaction of the governed use case' in error for error in refused)
        elif mode == 'duplicate':
            assert len(accepted) == 1 and len(refused) == 2
            assert all('duplicate authoritative outcome' in error for error in refused)
        elif mode == 'rollback':
            assert len(accepted) == 2 and refused == ['A3: revalidation evidence write incomplete']
        else:
            assert len(accepted) == 3 and refused == []
        assert all(result['breached'] for result in accepted)
        assert sorted(result['adverse_count'] for result in accepted) == list(
            range(1 if mode == 'policy' else 2, (1 if mode == 'policy' else 2) + len(accepted)))
        with _Session(engine) as session:
            rows = session.exec(_select(_Event).order_by(_Event.sequence)).all()
            actions = [row.payload.get('action') for row in rows]
            outcomes = [row.payload['body']['outcome']['outcome_id'] for row in rows
                        if row.payload.get('action') == 'PRODUCTION_OUTCOME']
            assert len(outcomes) == len(accepted) + (mode not in ('policy', 'interaction'))
            assert len(set(outcomes)) == len(outcomes)
            if mode == 'rollback':
                assert '0' not in outcomes
            assert actions.count('FINDING') == len(accepted)
            assert sum(row.event_type == _Type.REVALIDATION_TRIGGERED for row in rows) == len(accepted)
            reductions = [row.payload for row in rows if row.payload.get('action') == 'AUTONOMY_STEP_DOWN']
            assert [(row['from'], row['to']) for row in reductions] == [(3-i, 2-i) for i in range(len(accepted))]
            assert agent_for(session, 'outcome-worker').autonomy_level == 3 - len(accepted)
            assert [row.sequence for row in rows] == list(range(1, len(rows) + 1))
            assert _verify(session, public_key=public) == len(rows)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        ready.close()
        results.close()


@pytest.mark.parametrize('isolation', ['REPEATABLE READ', 'SERIALIZABLE'])
def test_outcomes_refuse_nonrefreshing_transaction_isolation(_store, isolation):
    """A transaction with an old snapshot must refuse before any safety decision."""
    from kognita.assurance.lifecycle import record_outcome
    engine, _ = _store
    with _Session(engine) as session:
        session.execute(_text(f'SET TRANSACTION ISOLATION LEVEL {isolation}'))
        with pytest.raises(ValueError, match='T2: evidence writes require READ COMMITTED'):
            record_outcome(session, 'unknown', {}, evidence=_Writer(engine), actor_id='owner')
        session.rollback()
        assert _verify(session) == 0
