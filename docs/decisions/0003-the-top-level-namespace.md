# ADR 0003: `kognita` is the decision engine, not a pointer to one

**Status:** Accepted (0.2.0)
**Supersedes:** the lazy graph re-export introduced in Phase 0

## Context

Through 0.1.x and the first pass of the 0.2 port, `kognita/__init__.py` exported
fifteen names — `Kognita`, `KognitaConfig`, `KognitaKuzuDriver`, `Node`, `Edge`,
`SearchResult`, … — every one of them from the graph engine, bound lazily so
that importing the package did not drag in Kuzu. The decision engine lived one
level down in `kognita.core` and was exported from the top level not at all.

That arrangement was defended as backwards compatibility. What it actually did
was advertise the wrong product. The positioning is that Kognita occupies the
row *"prove an answer was permitted, and evidence it"* — a row LlamaIndex,
LangChain, CrewAI and GraphRAG do not occupy — and that the graph is one
optional backend behind a protocol. A reader who typed `dir(kognita)` saw a
GraphRAG competitor with a governance module bolted on the side.

Three smaller problems followed from the same root:

- `from kognita import Kognita` names a class after its own package, so
  `kognita.graph.Kognita` reads as a stutter and keeps implying the graph is
  the identity.
- The `kognita.core` subpackage implied a `kognita` that was something more
  than its core, when in fact `kognita` was *only* a lazy pointer at the graph.
- `kognita.graph.core` and `kognita.graph.engine` were two modules whose names
  gave no hint which held the user-facing class.

## Decision

1. **Flatten the engine into `kognita`.** `kognita/core/*.py` moves up to
   `kognita/*.py`; the `core` subpackage is deleted. `kognita/__init__.py`
   eagerly re-exports the engine, so `from kognita import Envelope, decide`
   works on the four hard dependencies.

2. **No graph name is reachable from the top level.** The graph is imported
   from the module that owns it: `from kognita.graph import GraphEngine`.

3. **Rename the graph's public surface** so nothing is named after the package:
   `Kognita` → `GraphEngine`, `KognitaConfig` → `GraphConfig`,
   `KognitaKuzuDriver` → `KuzuDriver`. The modules follow:
   `graph/core.py` → `graph/engine.py` (holds `GraphEngine`) and the old
   `graph/engine.py` → `graph/driver.py` (holds `KuzuDriver`, `make_graphiti`).

4. **Clean break, no shim.** Touching a retired name raises `AttributeError`
   naming the module that now owns it. 0.1.0 is a pre-release alpha; a
   deprecation shim would keep advertising exactly the namespace this ADR
   exists to remove.

## Consequences

**What this buys.** `import kognita` now loads the thing the README is about,
on four dependencies, with no lazy-binding machinery in the path. An import
statement tells a reader whether a graph database is about to be loaded, which
is the property that made the layering worth enforcing in the first place.

**What it costs — and the mitigations.**

- *The import-linter engine contract must name its modules one by one.* When
  the engine was a subpackage, `source_modules = ["kognita.core"]` covered it
  and everything added to it. Flat, a new module that nobody adds to the
  contract is not merely undocumented but unconstrained.
  → `tests/test_packaging.py::test_every_engine_module_is_covered_by_the_contract`
  diffs the contract against the filesystem, so the drift fails loudly.

- *`kognita/__init__.py` itself sits outside that contract*, because naming
  `kognita` as a source module would sweep in `kognita.graph`.
  → `test_importing_kognita_loads_no_optional_dependency` runs `import kognita`
  in a subprocess and asserts nothing from `graphiti_core`, `kuzu`, `openai` or
  `anthropic` reached `sys.modules`. That is the stronger guarantee anyway, and
  it is the one the README claims.

- *Losing the lazy top-level binding lost its good error message.* Without the
  extra, `import kognita.graph` would have raised a bare
  `ModuleNotFoundError: No module named 'kuzu'` — true, but it names a
  transitive dependency rather than the extra that supplies it.
  → `kognita/graph/__init__.py` catches `ImportError` and re-raises
  `ConfigError` naming `pip install kognita[graph]`.

- *`kognita.load_snapshot` and `kognita.graph.load_snapshot` are different
  functions* — the policy snapshot and a saved graph. They no longer collide
  because the graph is not re-exported upward, which is itself a small piece of
  evidence for the decision. The distinction is documented on
  `kognita.__getattr__`.

- *Two modules named `engine.py` existed for one commit.* Resolved by the
  rename in decision 3; `graph/engine.py` is now unambiguously the engine and
  `graph/driver.py` unambiguously the Kuzu plumbing.

## Migration

| 0.1.x | 0.2 |
|---|---|
| `from kognita.core import decide` | `from kognita import decide` |
| `from kognita import Kognita` | `from kognita.graph import GraphEngine` |
| `from kognita import KognitaConfig` | `from kognita.graph import GraphConfig` |
| `from kognita import KognitaKuzuDriver` | `from kognita.graph import KuzuDriver` |
| `from kognita import KuzuSession, Node, Edge` | `from kognita.graph import KuzuSession, Node, Edge` |
