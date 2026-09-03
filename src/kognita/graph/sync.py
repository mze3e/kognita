"""Declarative System of Record mirror: deterministic Kuzu tables synced from SQLite.

The SoR (System of Record) holds a deterministic, LLM-free copy of application
state — clients, accounts, policies, whatever a domain pack declares. It lives
in ``SoR_*`` tables in the same Kuzu database as the knowledge graph, so one
Cypher traversal can cross both planes.

Phase 6 is in progress. The sync machinery exists in proof-of-concept form but
awaits a real domain pack to drive implementation. The warehouse pattern is:

- A pack defines its subjects as SQLModel tables (clients, accounts, policies, ...)
- sync() scans those tables and mirrors them into SoR_* Kuzu tables
- One Cypher traversal then crosses knowledge graph + SoR (future)
- REFERENCES edges link knowledge items to the subjects that triggered them

For now, this module is a placeholder. A real implementation will:

1. Take a SQLAlchemy table or declarative mapping
2. Pull data from SQLModel session
3. Create Kuzu node tables with the same schema
4. Optionally create edge tables for named relationships
5. Emit evidence for the sync operation

The actual integration is pending a real domain pack.
"""
from __future__ import annotations

from typing import Any

from kognita.exceptions import ConfigError


def sync(
    session: Any,
    mapping: dict[str, Any],
    kuzu_session: Any,
    *,
    dry_run: bool = False,
) -> int:
    """Sync a pack's SoR from SQLite into Kuzu.

    Idempotent: drops all SoR_* tables and recreates them. Run once at startup
    after Graphiti's schema setup.

    Args:
        session: SQLModel session with the application data
        mapping: SoR shape declaration (subject_type -> config)
        kuzu_session: KuzuSession holding the target Kuzu database
        dry_run: if True, print Cypher but do not execute

    Returns:
        Total number of nodes created

    Raises:
        ConfigError: if mapping is invalid or sync fails
    """
    raise ConfigError(
        "The SoR mirror is not yet implemented. "
        "Phase 6 work is in progress. "
        "See src/kognita/graph/sync.py."
    )


__all__ = ["sync"]
