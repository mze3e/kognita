"""The single serialized Kuzu accessor.

Kuzu takes an *exclusive lock per database directory*, so a process may hold only
one :class:`kuzu.Database` for a given path. Graphiti wants one, the deterministic
SoR mirror wants one, and ad-hoc Cypher wants one — which is three too many.

:class:`KuzuSession` owns the single handle and lends connections to all three.
Two graphs then co-tenant one database: Graphiti's own tables carry the
LLM-extracted knowledge, and ``SoR_*`` tables carry the deterministic mirror of a
system of record, so a single traversal can cross both planes.

``mode="two-db"`` keeps the identical interface but places the SoR tables in a
separate database directory. It exists as the escape hatch if a future
graphiti-core release stops tolerating foreign tables in its database; nothing
above this module changes when it is used, only cross-plane Cypher is lost.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal

from kognita.exceptions import ConfigError, KognitaError

SessionMode = Literal["single-db", "two-db"]

#: Prefix marking tables owned by the deterministic mirror rather than Graphiti.
SOR_PREFIX = "SoR_"

_READ_ONLY_PREFIXES = ("match", "call", "return", "with")


class KuzuSession:
    """Owns one :class:`kuzu.Database` per path and serializes access to it.

    Use as a context manager so the handle is always released::

        with KuzuSession(".kognita_db") as session:
            session.cypher("MATCH (n:SoR_Client) RETURN n.label")
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        mode: SessionMode = "single-db",
        sor_db_path: str | Path | None = None,
    ) -> None:
        if not db_path:
            raise ConfigError("db_path is required.")
        if mode == "two-db" and sor_db_path is None:
            sor_db_path = f"{db_path}_sor"

        self.db_path = str(db_path)
        self.mode: SessionMode = mode
        self.sor_db_path = str(sor_db_path) if sor_db_path else self.db_path
        self._lock = threading.RLock()
        self._db: Any = None
        self._sor_db: Any = None

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self) -> "KuzuSession":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> "KuzuSession":
        with self._lock:
            if self._db is None:
                kuzu = _require_kuzu()
                self._db = kuzu.Database(self.db_path)
                if self.mode == "two-db":
                    self._sor_db = kuzu.Database(self.sor_db_path)
        return self

    def close(self) -> None:
        with self._lock:
            self._db = None
            self._sor_db = None

    # -- handles ------------------------------------------------------------

    @property
    def database(self) -> Any:
        """The live ``kuzu.Database``. Hand this to a driver, never a path."""
        self.open()
        return self._db

    @property
    def sor_database(self) -> Any:
        """The database holding ``SoR_*`` tables — the same one in single-db mode."""
        self.open()
        return self._sor_db if self.mode == "two-db" else self._db

    def connection(self, *, sor: bool = False) -> Any:
        """Open a fresh connection on the shared handle. Caller closes it."""
        kuzu = _require_kuzu()
        return kuzu.Connection(self.sor_database if sor else self.database)

    # -- queries ------------------------------------------------------------

    def cypher(
        self,
        statement: str,
        *,
        allow_writes: bool = False,
        sor: bool = False,
    ) -> list[dict]:
        """Run Cypher on the shared handle and return rows as dicts.

        Defaults to read-only, matching :func:`kognita.graph.query.execute_cypher`.
        Unlike that function this never opens a second ``kuzu.Database``, so it is
        safe to call while Graphiti holds the same path.
        """
        normalized = statement.strip().rstrip(";")
        if not normalized:
            return []
        if not allow_writes and not normalized.lower().startswith(_READ_ONLY_PREFIXES):
            raise KognitaError(
                "Only read-only MATCH / CALL / RETURN / WITH queries are allowed. "
                "Pass allow_writes=True to override."
            )

        with self._lock:
            conn = self.connection(sor=sor)
            try:
                result = conn.execute(normalized)
                if result is None:
                    return []
                return list(result.rows_as_dict())
            finally:
                conn.close()

    def execute_many(self, statements: list[str], *, sor: bool = False) -> None:
        """Run a batch of write statements on one connection, in order."""
        with self._lock:
            conn = self.connection(sor=sor)
            try:
                for statement in statements:
                    normalized = statement.strip().rstrip(";")
                    if normalized:
                        conn.execute(normalized)
            finally:
                conn.close()

    # -- introspection ------------------------------------------------------

    def table_names(self, *, sor: bool = False) -> list[str]:
        """Names of every node and rel table in the database."""
        rows = self.cypher("CALL SHOW_TABLES() RETURN *", sor=sor)
        return [str(row.get("name", "")) for row in rows]

    def sor_table_names(self) -> list[str]:
        """Names of the tables owned by the deterministic mirror."""
        return [n for n in self.table_names(sor=True) if n.startswith(SOR_PREFIX)]


def _require_kuzu() -> Any:
    try:
        import kuzu
    except ImportError as exc:  # pragma: no cover - exercised by the bare-install test
        raise ConfigError(
            "Kuzu is not installed. Install the graph extra: pip install kognita[graph]"
        ) from exc
    return kuzu
