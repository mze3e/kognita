"""Kognita — prove an AI answer was permitted, and evidence it.

Two layers ship in one distribution:

``kognita.core``
    The governed decision engine: envelopes, a deterministic policy decision
    point, entitlement-filtered retrieval, an egress guard, and a hash-chained
    evidence plane. Installs and runs on four dependencies, no LLM required::

        from kognita.core import Envelope, decide

``kognita.graph``
    The Graphiti + Kuzu knowledge engine — documents to a bi-temporal graph.
    Requires the graph extra::

        pip install kognita[graph]

Names from the graph layer are re-exported here for backwards compatibility and
bound lazily, so importing :mod:`kognita` never drags in a graph database.
Touching one without the extra installed raises :class:`ConfigError` naming the
extra to install, rather than an ``ImportError`` traceback.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kognita.config import (
    EmbedderConfig,
    EmbedderProvider,
    LLMConfig,
    LLMProvider,
    list_models,
)
from kognita.exceptions import ConfigError, KognitaError, ProviderError

__version__ = "0.2.0"

# Attribute name -> the module it lives in and the extra that provides it.
_LAZY: dict[str, tuple[str, str]] = {
    "Kognita": ("kognita.graph.core", "graph"),
    "KognitaConfig": ("kognita.graph.config", "graph"),
    "KognitaKuzuDriver": ("kognita.graph.engine", "graph"),
    "KuzuSession": ("kognita.graph.session", "graph"),
    "make_graphiti": ("kognita.graph.engine", "graph"),
    "execute_cypher": ("kognita.graph.query", "graph"),
    "chunk_text": ("kognita.graph.chunking", "graph"),
    "GraphSnapshot": ("kognita.graph.storage", "graph"),
    "load_snapshot": ("kognita.graph.storage", "graph"),
    "save_snapshot": ("kognita.graph.storage", "graph"),
    "content_hash": ("kognita.graph.storage", "graph"),
    "Node": ("kognita.graph.types", "graph"),
    "Edge": ("kognita.graph.types", "graph"),
    "SearchResult": ("kognita.graph.types", "graph"),
    "EpisodeResult": ("kognita.graph.types", "graph"),
}


def __getattr__(name: str) -> Any:
    """Resolve a graph-layer export on first use.

    PEP 562 module ``__getattr__``: only called for names not already bound, so
    a resolved attribute costs nothing on subsequent access.
    """
    try:
        module_path, extra = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'kognita' has no attribute {name!r}") from None

    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigError(
            f"'{name}' needs Kognita's graph engine, which is not installed. "
            f"Install it with:  pip install kognita[{extra}]\n"
            f"(the core decision engine in kognita.core does not require it)"
        ) from exc

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # pragma: no cover - import-time cost is the whole point
    from kognita.graph.chunking import chunk_text
    from kognita.graph.config import KognitaConfig
    from kognita.graph.core import Kognita
    from kognita.graph.engine import KognitaKuzuDriver, make_graphiti
    from kognita.graph.query import execute_cypher
    from kognita.graph.session import KuzuSession
    from kognita.graph.storage import (
        GraphSnapshot,
        content_hash,
        load_snapshot,
        save_snapshot,
    )
    from kognita.graph.types import Edge, EpisodeResult, Node, SearchResult

__all__ = [
    # provider configuration (always available)
    "LLMConfig",
    "EmbedderConfig",
    "LLMProvider",
    "EmbedderProvider",
    "list_models",
    # errors (always available)
    "KognitaError",
    "ConfigError",
    "ProviderError",
    # graph engine (lazy; requires kognita[graph])
    "Kognita",
    "KognitaConfig",
    "KognitaKuzuDriver",
    "KuzuSession",
    "make_graphiti",
    "execute_cypher",
    "chunk_text",
    "GraphSnapshot",
    "load_snapshot",
    "save_snapshot",
    "content_hash",
    "Node",
    "Edge",
    "SearchResult",
    "EpisodeResult",
    "__version__",
]
