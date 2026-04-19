"""Direct Kuzu Cypher passthrough for advanced callers."""
from __future__ import annotations

from pathlib import Path

import kuzu

from kognita.exceptions import ConfigError, KognitaError

_READ_ONLY_PREFIXES = ("match", "call", "return", "with")


def execute_cypher(
    db_path: str | Path,
    cypher: str,
    *,
    allow_writes: bool = False,
) -> list[dict]:
    """Run a Cypher statement against a Kuzu database and return rows as dicts.

    Defaults to read-only: statements must begin with ``MATCH``, ``CALL``,
    ``RETURN`` or ``WITH``. Pass ``allow_writes=True`` to bypass the check for
    admin / migration scripts. Trailing semicolons are tolerated.
    """
    if not db_path:
        raise ConfigError("db_path is required.")

    normalized = cypher.strip().rstrip(";")
    if not normalized:
        return []

    if not allow_writes and not normalized.lower().startswith(_READ_ONLY_PREFIXES):
        raise KognitaError(
            "Only read-only MATCH / CALL / RETURN / WITH queries are allowed. "
            "Pass allow_writes=True to override."
        )

    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)
    try:
        result = conn.execute(normalized)
        if result is None:
            return []
        return list(result.rows_as_dict())
    finally:
        conn.close()
