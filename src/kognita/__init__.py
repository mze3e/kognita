"""Kognita — prove an AI answer was permitted, and evidence it.

``kognita`` **is** the governed decision engine. Everything exported here —
envelopes, a deterministic policy decision point, entitlement-filtered
retrieval, an egress guard, a hash-chained evidence plane — installs and runs on
four dependencies, with no LLM, no graph database and no network::

    from kognita import Envelope, decide, load_snapshot

    evaluation = decide(envelope, load_snapshot(session))
    evaluation.outcome        # ALLOW / DENY / ESCALATE / HUMAN_APPROVAL
    evaluation.basis()        # the checks that decided it, each with a citation

Deciding whether a request is permitted, and proving afterwards that it was,
should not require the machinery that answers it. That is the whole claim, and
this namespace is where it is kept: nothing reachable from ``import kognita``
touches a provider or a database engine.

Optional subpackages sit alongside, reached by their own names so that reading
an import tells you what a call will actually load:

``kognita.graph``
    The Graphiti + Kuzu knowledge engine — ``pip install kognita[graph]``::

        from kognita.graph import GraphEngine, GraphConfig

``kognita.adapters``
    Provider-backed embedders and clients — ``pip install kognita[openai]``.

``kognita.testing``
    The conformance kit a domain pack runs against itself.
"""
from __future__ import annotations

from typing import Any

from kognita.approvals import (
    ApprovalError,
    expire_stale,
    find_granted,
    grant,
    reject,
)
from kognita.broker import BrokerAnswer, ask, default_route_resolver
from kognita.canonical import canonical_hash, canonical_json
from kognita.classify import FixedClassifier, PatternClassifier, most_sensitive
from kognita.config import (
    EmbedderConfig,
    EmbedderProvider,
    LLMConfig,
    LLMProvider,
    list_models,
)
from kognita.db import create_all, make_engine, session_scope
from kognita.egress import (
    EgressDenied,
    EgressGuard,
    EgressPolicy,
    EgressResult,
    NullRedactor,
    PatternRedactor,
)
from kognita.embedding import HashingEmbedder, cosine, lexical_overlap
from kognita.envelope import Check, Envelope, Evaluation, RuleContext, envelope_hash
from kognita.evidence import (
    ChainBreak,
    EvidenceWriter,
    export_chain,
    hashes_only,
    verify_chain,
    verify_export,
)
from kognita.exceptions import ConfigError, KognitaError, ProviderError
from kognita.governance import (
    PolicySnapshot,
    decide,
    load_snapshot,
    record,
    resolve_outcome,
)
from kognita.models import (
    Agent,
    Approval,
    Entity,
    EntityEdge,
    EvidenceEvent,
    GovernanceDecision,
    KnowledgeItem,
    Policy,
    utcnow,
)
from kognita.retrieval import Retrieved, index_item, reindex, retrieve
from kognita.rules import CORE_RULES, build_registry, rule
from kognita.tools import ToolRegistry, ToolRun, run_governed
from kognita.vectors import NumpyVectorIndex, SqliteVecIndex, default_index
from kognita.vocabulary import (
    ActorType,
    ApprovalStatus,
    CheckResult,
    Classification,
    EgressDecision,
    EventType,
    Outcome,
)

__version__ = "0.2.0"

#: Names 0.1.x exposed here that 0.2 does not. The graph engine is one optional
#: backend behind a protocol, so it is reached at ``kognita.graph`` rather than
#: advertised in the namespace that has to stay dependency-light.
_MOVED: dict[str, str] = {
    "Kognita": "kognita.graph.GraphEngine",
    "KognitaConfig": "kognita.graph.GraphConfig",
    "KognitaKuzuDriver": "kognita.graph.KuzuDriver",
    "KuzuSession": "kognita.graph.KuzuSession",
    "make_graphiti": "kognita.graph.make_graphiti",
    "execute_cypher": "kognita.graph.execute_cypher",
    "chunk_text": "kognita.graph.chunk_text",
    "GraphSnapshot": "kognita.graph.GraphSnapshot",
    "save_snapshot": "kognita.graph.save_snapshot",
    "content_hash": "kognita.graph.content_hash",
    "Node": "kognita.graph.Node",
    "Edge": "kognita.graph.Edge",
    "SearchResult": "kognita.graph.SearchResult",
    "EpisodeResult": "kognita.graph.EpisodeResult",
}


def __getattr__(name: str) -> Any:
    """Explain the retired 0.1.x graph names rather than resolving them.

    These are not lazily bound aliases. The graph engine deliberately no longer
    reaches into this namespace, so the error names the module that owns it.

    ``load_snapshot`` is absent from ``_MOVED`` on purpose: it still lives here
    and means the *policy* snapshot. :func:`kognita.graph.load_snapshot` is a
    different function that rehydrates a saved graph.
    """
    destination = _MOVED.get(name)
    if destination is None:
        raise AttributeError(f"module 'kognita' has no attribute {name!r}")
    module, _, attribute = destination.rpartition(".")
    raise AttributeError(
        f"'{name}' is no longer exported from 'kognita'. The graph engine is an "
        f"optional backend rather than the top-level namespace, so import it "
        f"from the module that owns it:\n"
        f"    from {module} import {attribute}\n"
        f"which needs the graph extra:  pip install kognita[graph]"
    )


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    # decisions
    "Envelope",
    "Check",
    "Evaluation",
    "RuleContext",
    "envelope_hash",
    "decide",
    "record",
    "resolve_outcome",
    "PolicySnapshot",
    "load_snapshot",
    # rules
    "rule",
    "build_registry",
    "CORE_RULES",
    # evidence
    "EvidenceWriter",
    "verify_chain",
    "export_chain",
    "verify_export",
    "hashes_only",
    "ChainBreak",
    # approvals
    "grant",
    "reject",
    "expire_stale",
    "find_granted",
    "ApprovalError",
    # retrieval
    "retrieve",
    "index_item",
    "reindex",
    "Retrieved",
    "HashingEmbedder",
    "cosine",
    "lexical_overlap",
    "NumpyVectorIndex",
    "SqliteVecIndex",
    "default_index",
    # egress
    "EgressGuard",
    "EgressPolicy",
    "EgressResult",
    "EgressDenied",
    "PatternRedactor",
    "NullRedactor",
    # classification
    "PatternClassifier",
    "FixedClassifier",
    "most_sensitive",
    # tools and broker
    "ToolRegistry",
    "ToolRun",
    "run_governed",
    "ask",
    "BrokerAnswer",
    "default_route_resolver",
    # storage
    "make_engine",
    "create_all",
    "session_scope",
    "Agent",
    "Policy",
    "Approval",
    "GovernanceDecision",
    "EvidenceEvent",
    "KnowledgeItem",
    "Entity",
    "EntityEdge",
    "utcnow",
    # vocabulary
    "Outcome",
    "CheckResult",
    "Classification",
    "ActorType",
    "EventType",
    "ApprovalStatus",
    "EgressDecision",
    # hashing
    "canonical_hash",
    "canonical_json",
    # provider configuration (no dependencies of its own)
    "LLMConfig",
    "EmbedderConfig",
    "LLMProvider",
    "EmbedderProvider",
    "list_models",
    # errors
    "KognitaError",
    "ConfigError",
    "ProviderError",
    "__version__",
]
