# 0001 — Two graphs in one Kuzu database

**Status:** accepted · **Date:** 2026-09-01 · **Verified against:** `kuzu==0.11.3`, `graphiti-core==0.28.2`

## Context

Kognita holds two graphs:

- a **knowledge graph** that Graphiti extracts from documents with an LLM (bi-temporal, deduplicated), and
- a **deterministic SoR mirror** of a system of record, synced from SQLite by code with no LLM involved.

The prize for putting both in one database is a single Cypher traversal that crosses
them — *subject → policy → the document that justifies it*. The risk is that Graphiti
owns its schema in that database and re-runs `build_indices_and_constraints()` /
`setup_schema()` whenever it is constructed.

## What the spike found

1. **Two `kuzu.Database` handles on one path do not share a consistent view.** A table
   created through the first handle was invisible to a second handle opened before the
   write; a handle opened afterwards did see it. No exception is raised either way.
   Silent, timing-dependent divergence is worse than a lock error, because nothing
   fails loudly enough to notice.
2. **`SoR_*` tables co-tenant cleanly.** `SoR_Client`, `SoR_Policy` and a
   `SoR_GOVERNED_BY` relationship all survived a subsequent `setup_schema()` and
   full-text index installation, and a traversal over them returned correct rows
   alongside Graphiti's own `Entity` / `Episodic` / `RELATES_TO` tables.

## Decision

**Single database, `SoR_`-prefixed tables, one shared handle.**

`kognita.graph.session.KuzuSession` owns the only `kuzu.Database` for a path and lends
connections to Graphiti, the mirror, and ad-hoc Cypher.
`KuzuDriver(database=session.database)` binds Graphiti to that handle by
intercepting `kuzu.Database(...)` for the duration of the parent constructor — the
narrowest available hook, depending only on the constructor calling
`kuzu.Database(path)` rather than on its body, which shifts between releases.

Finding 1 makes this mandatory rather than merely tidy, and it condemns one existing
API: `kognita.graph.query.execute_cypher` opens its own handle and so may read a stale
view while a session holds the same path. It stays for standalone library use;
anything inside a session must call `KuzuSession.cypher()`.

## Consequences and the escape hatch

Re-run this spike on any `graphiti-core` or `kuzu` bump — the versions above are pinned
exactly for that reason.

If a future release stops tolerating foreign tables, `KuzuSession(mode="two-db")`
places `SoR_*` in a separate database behind an unchanged interface. Nothing above
`session.py` changes; only cross-plane Cypher is lost.
