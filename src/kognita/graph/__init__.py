"""The Graphiti + Kuzu knowledge engine — one optional backend, not the product.

``kognita`` itself is the governed decision engine and installs on four
dependencies. This package is where documents become a bi-temporal,
auto-deduplicated knowledge graph, and it requires the graph extra::

    pip install kognita[graph]

    from kognita.graph import GraphEngine, GraphConfig

Nothing here is re-exported from the top-level :mod:`kognita` namespace: the
graph is reached by its own name so that reading an import tells you whether a
graph database is in play.
"""
from kognita.exceptions import ConfigError

try:
    from kognita.graph.chunking import chunk_text
    from kognita.graph.config import GraphConfig
    from kognita.graph.driver import KuzuDriver, make_graphiti
    from kognita.graph.engine import GraphEngine
    from kognita.graph.query import execute_cypher
    from kognita.graph.session import KuzuSession
    from kognita.graph.storage import (
        GraphSnapshot,
        content_hash,
        load_snapshot,
        save_snapshot,
        saved_kuzu_db_path,
    )
    from kognita.graph.types import Edge, EpisodeResult, Node, SearchResult
except ImportError as exc:  # pragma: no cover - exercised by the bare-install test
    # A bare `ModuleNotFoundError: No module named 'kuzu'` is true but useless:
    # it names a transitive dependency rather than the extra that supplies it.
    raise ConfigError(
        f"Kognita's graph engine is not installed ({exc.name or exc} is missing).\n"
        f"    pip install kognita[graph]\n"
        f"The decision engine in `kognita` itself does not require it."
    ) from exc

__all__ = [
    # the engine
    "GraphEngine",
    "GraphConfig",
    # Kuzu access
    "KuzuSession",
    "KuzuDriver",
    "make_graphiti",
    "execute_cypher",
    # ingestion
    "chunk_text",
    # snapshots
    "GraphSnapshot",
    "load_snapshot",
    "save_snapshot",
    "saved_kuzu_db_path",
    "content_hash",
    # value types
    "Node",
    "Edge",
    "SearchResult",
    "EpisodeResult",
]
