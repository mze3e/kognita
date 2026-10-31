"""Transactional SQLite evidence schema upgrades (T5).

No migration ledger is needed: the evidence table describes the supported
versions. Helpers stay private; operators use ``kognita db migrate|status``.
"""
from __future__ import annotations

from sqlalchemy import inspect as _inspect
from sqlalchemy.engine import Engine as _Engine

from kognita.db import EVIDENCE_GUARD_TRIGGERS as _GUARDS, _EVIDENCE_HEAD
from kognita.models import EvidenceEvent as _EvidenceEvent


def _status(connection) -> int:
    if connection.dialect.name != 'sqlite':
        raise ValueError('T5: migration runner supports SQLite only')
    inspector = _inspect(connection)
    if not inspector.has_table('evidence_events'):
        raise ValueError('T5: evidence_events is absent; initialise the store first')
    expected = set(_EvidenceEvent.__table__.columns.keys())
    actual = {column['name'] for column in inspector.get_columns('evidence_events')}
    if actual == expected - {'schema_version'}:
        return 1
    if actual != expected:
        raise ValueError('T5: unsupported evidence schema')
    if connection.exec_driver_sql(
        'SELECT 1 FROM evidence_events WHERE schema_version IS NULL '
        'OR typeof(schema_version) != \'integer\' '
        'OR schema_version NOT IN (1, 2) LIMIT 1'
    ).first():
        raise ValueError('T5: unsupported schema_version')
    return 2


def _copy(connection, columns) -> None:
    names = ', '.join(connection.dialect.identifier_preparer.quote(name) for name in columns)
    connection.exec_driver_sql(
        f'INSERT INTO evidence_events ({names}, schema_version) '
        f'SELECT {names}, 1 FROM _evidence_events_migration'
    )


def _migrate(engine: _Engine) -> int:
    if engine.dialect.name != 'sqlite':
        raise ValueError('T5: migration runner supports SQLite only')
    with engine.connect() as connection:
        # Explicit BEGIN is essential: sqlite3 legacy transaction control does
        # not begin a transaction for DDL. A rename must roll back too.
        connection.exec_driver_sql('BEGIN IMMEDIATE')
        try:
            version = _status(connection)
            if version == 1:
                if _inspect(connection).has_table('_evidence_events_migration'):
                    raise ValueError('T5: migration staging table already exists')
                triggers = connection.exec_driver_sql(
                    "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
                    "AND tbl_name = 'evidence_events'"
                ).scalars().all()
                index_sql = connection.exec_driver_sql(
                    "SELECT sql FROM sqlite_master WHERE type = 'index' "
                    "AND tbl_name = 'evidence_events' AND sql IS NOT NULL"
                ).scalars().all()
                indexes = _inspect(connection).get_indexes('evidence_events')
                columns = [column['name'] for column in _inspect(connection).get_columns('evidence_events')]
                connection.exec_driver_sql('ALTER TABLE evidence_events RENAME TO _evidence_events_migration')
                for index in indexes:
                    name = connection.dialect.identifier_preparer.quote(index['name'])
                    connection.exec_driver_sql(f'DROP INDEX {name}')
                _EvidenceEvent.__table__.create(connection)
                _copy(connection, columns)
                connection.exec_driver_sql('DROP TABLE _evidence_events_migration')
                for statement in index_sql:
                    prefix = 'CREATE UNIQUE INDEX ' if statement.startswith('CREATE UNIQUE INDEX ') else 'CREATE INDEX '
                    statement = statement.replace(prefix, prefix + 'IF NOT EXISTS ', 1)
                    connection.exec_driver_sql(statement)
                for statement in triggers:
                    connection.exec_driver_sql(statement)
            # Reinstall standard guards only after copying, so the checkpoint
            # never moves backwards or gets rebuilt from a truncated chain.
            for statement in (*_EVIDENCE_HEAD, *_GUARDS):
                connection.exec_driver_sql(statement)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return 2
