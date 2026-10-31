"""T5: atomic upgrades preserve legacy evidence and enforcement."""
import pytest
from sqlmodel import Session, select

import kognita.migrations as _migrations
from kognita.cli import main
from kognita.evidence import EvidenceWriter, verify_chain, verify_export, export_chain
from kognita.models import EvidenceEvent
from kognita.vocabulary import EventType
from test_schema_version import _legacy_store


def _snapshot(engine):
    with engine.connect() as connection:
        return (
            connection.exec_driver_sql('SELECT type, name, sql FROM sqlite_master ORDER BY type, name').all(),
            connection.exec_driver_sql('SELECT * FROM evidence_events ORDER BY sequence').all(),
            connection.exec_driver_sql('SELECT * FROM evidence_head').all(),
        )


def test_interrupted_migration_leaves_db_unchanged(monkeypatch):
    engine, _ = _legacy_store()
    before = _snapshot(engine)
    original = _migrations._copy

    def _interrupt(connection, columns):
        original(connection, columns)
        raise KeyboardInterrupt('interrupted after copy')

    monkeypatch.setattr(_migrations, '_copy', _interrupt)
    with pytest.raises(KeyboardInterrupt):
        _migrations._migrate(engine)
    assert _snapshot(engine) == before
    monkeypatch.setattr(_migrations, '_copy', original)
    assert _migrations._migrate(engine) == 2


def test_v1_events_readable_after():
    engine, artifact = _legacy_store()
    with engine.connect() as connection:
        assert _migrations._status(connection) == 1
    assert _migrations._migrate(engine) == 2
    before = _snapshot(engine)
    assert _migrations._migrate(engine) == 2
    assert _snapshot(engine) == before
    with Session(engine) as session:
        events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
        assert [event.schema_version for event in events] == [1, 1]
        assert [event.event_hash for event in events] == [row['event_hash'] for row in artifact['events']]
        assert verify_chain(session) == verify_export(export_chain(session)) == 2
        EvidenceWriter(engine).emit(session, correlation_id='new', event_type=EventType.TOOL_CALL)
        session.commit()
        assert verify_chain(session) == 3
    with pytest.raises(Exception, match='append-only'):
        with engine.begin() as connection:
            connection.exec_driver_sql('DELETE FROM evidence_events')


def test_cli_status_and_migrate(tmp_path, capsys):
    engine, _ = _legacy_store()
    path = tmp_path / 'store.db'
    import sqlite3
    with engine.connect() as connection, sqlite3.connect(path) as target:
        connection.connection.driver_connection.backup(target)
    assert main(['db', 'status', '--db', str(path)]) == 0
    assert 'schema_version=1' in capsys.readouterr().out
    assert main(['db', 'migrate', '--db', str(path)]) == 0
    assert main(['db', 'status', '--db', str(path)]) == 0
    assert 'schema_version=2' in capsys.readouterr().out


def test_unknown_schema_fails_closed():
    engine, _ = _legacy_store()
    with engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE evidence_events ADD COLUMN _unknown TEXT')
    before = _snapshot(engine)
    with pytest.raises(ValueError, match='T5: unsupported'):
        _migrations._migrate(engine)
    assert _snapshot(engine) == before


def test_failure_after_drop_rolls_back():
    from sqlalchemy import event

    engine, _ = _legacy_store()
    before = _snapshot(engine)

    def _interrupt(connection, cursor, statement, parameters, context, executemany):
        if statement == 'DROP TABLE _evidence_events_migration':
            raise RuntimeError('failure after drop')

    event.listen(engine, 'after_cursor_execute', _interrupt)
    try:
        with pytest.raises(RuntimeError, match='failure after drop'):
            _migrations._migrate(engine)
    finally:
        event.remove(engine, 'after_cursor_execute', _interrupt)
    assert _snapshot(engine) == before
    assert _migrations._migrate(engine) == 2
