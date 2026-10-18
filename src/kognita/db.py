"""Database engine and session management for the governance store."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

# Importing the models registers them on SQLModel.metadata, which create_all needs.
# Spelled as a submodule import rather than ``from kognita import models``: this
# module is imported *by* ``kognita/__init__.py``, so the parent package is only
# partially initialised at this point and may not carry the attribute yet.
import kognita.models as _models  # noqa: F401,E402


def make_engine(
    url_or_path: str | Path = ":memory:",
    *,
    echo: bool = False,
) -> Engine:
    """Create an engine from a SQLAlchemy URL or a plain filesystem path.

    ``":memory:"`` gives a shared in-memory database — useful for tests, where a
    per-connection database would appear empty on the next checkout.
    """
    url = str(url_or_path)
    if "://" not in url:
        if url == ":memory:":
            url = "sqlite://"
        else:
            Path(url).parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{url}"

    connect_args: dict = {}
    kwargs: dict = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if url in ("sqlite://", "sqlite:///:memory:"):
            # Without a shared pool every connection gets its own empty database.
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool

    engine = create_engine(url, echo=echo, connect_args=connect_args, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            # Foreign keys are off by default in SQLite; WAL keeps a reader from
            # blocking the single writer this store assumes.
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


# create_all does not add an index to a table that already exists. These
# statements cover a store created before the effective-window indexes.
_WINDOW_INDEXES: tuple[str, ...] = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_agents_name ON agents (name)",
    "CREATE INDEX IF NOT EXISTS ix_policies_effective_window ON policies (effective_from, effective_to)",
    "CREATE INDEX IF NOT EXISTS ix_use_cases_effective_window ON use_cases (effective_from, effective_to)",
    "CREATE INDEX IF NOT EXISTS ix_risk_appetites_effective_window ON risk_appetites (effective_from, effective_to)",
    "CREATE INDEX IF NOT EXISTS ix_model_cards_effective_window ON model_cards (effective_from, effective_to)",
    "CREATE INDEX IF NOT EXISTS ix_revocations_effective_window ON revocations (effective_from, effective_to)",
    "CREATE INDEX IF NOT EXISTS ix_breaker_thresholds_effective_window ON breaker_thresholds (effective_from, effective_to)",
    "CREATE INDEX IF NOT EXISTS ix_drill_dues_effective_window ON drill_dues (effective_from, effective_to)",
    "CREATE INDEX IF NOT EXISTS ix_evidence_events_correlation_window ON evidence_events (correlation_id, recorded_at, sequence)",
    "CREATE INDEX IF NOT EXISTS ix_evidence_events_recorded_at ON evidence_events (recorded_at)",
)


# Evidence is append-only in the store, not only by convention. An UPDATE or
# DELETE on an evidence row is refused by SQLite. A one-row head checkpoint
# records the last sequence and its hash on every insert; it can only move
# forward and cannot be deleted. verify_chain compares the chain's tail with it,
# so events removed from the end are reported, not silently accepted.
EVIDENCE_GUARD_TRIGGERS: tuple[str, ...] = (
    """CREATE TRIGGER IF NOT EXISTS evidence_events_no_update
    BEFORE UPDATE ON evidence_events
    BEGIN SELECT RAISE(ABORT, 'evidence is append-only: an evidence row cannot be updated'); END""",
    """CREATE TRIGGER IF NOT EXISTS evidence_events_no_delete
    BEFORE DELETE ON evidence_events
    BEGIN SELECT RAISE(ABORT, 'evidence is append-only: an evidence row cannot be deleted'); END""",
)
_EVIDENCE_HEAD: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS evidence_head (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        sequence INTEGER NOT NULL,
        event_hash TEXT NOT NULL
    )""",
    """INSERT INTO evidence_head (id, sequence, event_hash)
    SELECT 1, sequence, event_hash FROM evidence_events
    WHERE NOT EXISTS (SELECT 1 FROM evidence_head)
    ORDER BY sequence DESC LIMIT 1""",
    """CREATE TRIGGER IF NOT EXISTS evidence_head_advance
    AFTER INSERT ON evidence_events
    BEGIN
        UPDATE evidence_head SET sequence = NEW.sequence, event_hash = NEW.event_hash
        WHERE id = 1;
        INSERT INTO evidence_head (id, sequence, event_hash)
        SELECT 1, NEW.sequence, NEW.event_hash
        WHERE NOT EXISTS (SELECT 1 FROM evidence_head);
    END""",
    # The first release allowed an equal sequence, so a store that lost its
    # last event could append a new one under the same number. The head now
    # only moves strictly forward; that trigger is replaced on older stores.
    "DROP TRIGGER IF EXISTS evidence_head_forward_only",
    # RAISE's message must be a string literal on SQLite before 3.47; an
    # expression with || is a syntax error on GitHub Actions runners.
    """CREATE TRIGGER IF NOT EXISTS evidence_head_strictly_forward
    BEFORE UPDATE ON evidence_head
    WHEN NEW.sequence <= OLD.sequence
    BEGIN SELECT RAISE(ABORT, 'the evidence head only moves forward'); END""",
    """CREATE TRIGGER IF NOT EXISTS evidence_head_no_delete
    BEFORE DELETE ON evidence_head
    BEGIN SELECT RAISE(ABORT, 'the evidence head cannot be deleted'); END""",
)


def create_all(engine: Engine) -> None:
    """Create every registered table that does not yet exist.

    A domain pack's models must be imported before this runs, or its tables are
    simply absent from the metadata and silently skipped. On SQLite, indexes
    that a later release added are created on a store that already has the
    tables. Evidence rows are protected from UPDATE and DELETE, and the head
    checkpoint is created and backfilled from the last event.
    """
    SQLModel.metadata.create_all(engine)
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(agents)")}
        for name, kind in {
            "purpose": "TEXT", "use_cases": "JSON", "permitted_systems": "JSON",
            "tool_allowlist": "JSON", "autonomy_level": "INTEGER", "review_date": "DATETIME",
            "deployment_hash": "TEXT", "environment": "TEXT", "principal": "TEXT",
        }.items():
            if name not in columns:
                connection.exec_driver_sql(f'ALTER TABLE agents ADD COLUMN "{name}" {kind}')
        use_columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(use_cases)")}
        if "deployment_mode" not in use_columns:
            connection.exec_driver_sql("ALTER TABLE use_cases ADD COLUMN deployment_mode TEXT")
        for statement in _WINDOW_INDEXES:
            connection.exec_driver_sql(statement)
        for statement in (*_EVIDENCE_HEAD, *EVIDENCE_GUARD_TRIGGERS):
            connection.exec_driver_sql(statement)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """A transactional session: commit on success, roll back on error."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = ["make_engine", "create_all", "session_scope", "Session"]
