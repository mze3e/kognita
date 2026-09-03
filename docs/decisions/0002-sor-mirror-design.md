# ADR 0002: System of Record Mirror Design

**Status:** Pending (Phase 6)  
**Decision:** Defer SoR mirror implementation until a real domain pack drives requirements

## Context

Phase 6 involves building a deterministic graph mirror (`SoR_*` tables in Kuzu) that:

1. Reflects application state (clients, accounts, policies, anything a pack declares)
2. Lives in the same Kuzu database as the knowledge graph
3. Enables one Cypher traversal to cross both LLM-extracted knowledge and deterministic application state
4. Is linked to knowledge items via `REFERENCES` edges

The earlier design assumed a simple query-based mapping where a pack provides SQL queries to pull data. However, that assumes:

- Queries return data already shaped for Kuzu
- Relationships are simple (foreign keys map directly to Cypher edges)
- A pack knows and declares everything in advance

These assumptions break when:

- A pack's subject schema evolves
- Relationships are complex (many-to-many, time-dependent)
- A pack's "system of record" is actually several sources (databases, APIs, spreadsheets)

## Decision

Defer Phase 6 until a real domain pack (e.g., the wealth management pack) drives requirements:

1. The pack defines its subjects as SQLModel tables (or SQLAlchemy ORM)
2. We see what "subject" and "relationship" actually mean in practice
3. We design sync with that shape in mind
4. We know what evidence should be emitted

Until then:

- `kognita.graph.sync` is a placeholder that raises `ConfigError`
- The core conformance kit tests pass without the SoR (tests use only knowledge graph)
- `KuzuSession` is ready and tested (can be used standalone)
- Documentation explains the intended architecture

## Consequences

**Pro:**

- No half-baked sync that breaks when a pack's schema shifts
- The real pack can drive design rather than the reverse
- Phases 0–5 are complete, tested, and shippable
- The architecture is clear: deterministic SoR + LLM knowledge graph

**Con:**

- Knowledge graph stands alone for now; cross-plane traversal is future work
- Domain packs cannot use the SoR until Phase 6 ships
- `kognita[graph]` can ingest and query Graphiti but not mirror application state

## Implementation

When Phase 6 begins:

1. Take the domain pack's SQLModel schema (e.g., `Client`, `Account`, `Policy`)
2. Define a `SoRMapping` protocol that describes which tables to sync
3. Implement `sync()` to:
   - Iterate over mapping definitions
   - Read tables from the SQLModel session
   - Create Kuzu node and relationship tables
   - Emit `SOR_SYNC` evidence events
4. Add `kognita graph sync` CLI command
5. Test round-trip: SQLite → Kuzu → Cypher query

The wealth management pack will be the first real test.
