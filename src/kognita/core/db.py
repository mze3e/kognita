"""Database engine and session management for the governance store."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

# Importing the models registers them on SQLModel.metadata, which create_all needs.
from kognita.core import models as _models  # noqa: F401


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


def create_all(engine: Engine) -> None:
    """Create every registered table that does not yet exist.

    A domain pack's models must be imported before this runs, or its tables are
    simply absent from the metadata and silently skipped.
    """
    SQLModel.metadata.create_all(engine)


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
