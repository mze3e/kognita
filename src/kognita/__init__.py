"""Kognita — prove an AI answer was permitted, and evidence it.

``kognita`` **is** the governed decision engine. Everything exported here —
envelopes, a deterministic policy decision point, entitlement-filtered
retrieval, an egress guard, a hash-chained evidence plane — installs and runs on
four dependencies, with no LLM, no database server and no network::

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
from kognita.exceptions import (
    ConfigError,
    KognitaError,
    PolicyEditError,
    ProviderError,
    ReplayMismatch,
    RetentionError,
)
from kognita.governance import (
    PolicySnapshot,
    classifier_derived_envelope,
    decide,
    load_snapshot,
    record,
    resolve_outcome,
    supersede_policy,
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
    policy_content_hash,
    utcnow,
)
from kognita.replay import replay_decision
from kognita.reconstruct import reconstruct, render_reconstruction
from kognita.retention import RetentionStore
from kognita.retrieval import Retrieved, index_item, reindex, retrieve
from kognita.rules import CORE_RULES, build_registry, rule
from kognita.gateway import ClientConfiguration, Gateway, GatewayResponse
from kognita.mcp import McpProxy, McpResponse
from kognita.tools import Run, ToolRegistry, ToolRun, continue_run, run_governed
from kognita.vectors import NumpyVectorIndex, SqliteVecIndex, default_index
from kognita.vocabulary import (
    ActorType,
    ApprovalStatus,
    CheckResult,
    Classification,
    EgressDecision,
    EventType,
    FailureMode,
    Outcome,
)

__version__ = "0.3.0"

#: Names from the Graphiti + Kuzu graph engine, which 0.1.x exported here and
#: which left the package entirely after 0.3 (ADR 0008).
_REMOVED: frozenset[str] = frozenset({
    "Kognita",
    "KognitaConfig",
    "KognitaKuzuDriver",
    "KuzuSession",
    "make_graphiti",
    "execute_cypher",
    "chunk_text",
    "GraphSnapshot",
    "save_snapshot",
    "content_hash",
    "Node",
    "Edge",
    "SearchResult",
    "EpisodeResult",
    "GraphEngine",
    "GraphConfig",
})


def __getattr__(name: str) -> Any:
    """Explain the removed graph names rather than failing with a bare error.

    ``load_snapshot`` is not in ``_REMOVED``: it lives here and means the
    *policy* snapshot.
    """
    if name not in _REMOVED:
        raise AttributeError(f"module 'kognita' has no attribute {name!r}")
    raise AttributeError(
        f"'{name}' belonged to the Graphiti + Kuzu graph engine, which is no "
        f"longer part of kognita (removed after 0.3.0, see ADR 0008). Kognita "
        f"governs policies, guidelines and evidence; it does not ship a "
        f"knowledge graph. Pin kognita[graph]==0.3.0 if you still need it."
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
    "classifier_derived_envelope",
    "PolicySnapshot",
    "load_snapshot",
    "supersede_policy",
    "replay_decision",
    "reconstruct",
    "render_reconstruction",
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
    "RetentionStore",
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
    "Run",
    "run_governed",
    "continue_run",
    "ask",
    "BrokerAnswer",
    "default_route_resolver",
    # gateway
    "ClientConfiguration",
    "Gateway",
    "GatewayResponse",
    "McpProxy",
    "McpResponse",
    # storage
    "make_engine",
    "create_all",
    "session_scope",
    "Agent",
    "Policy",
    "policy_content_hash",
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
    "FailureMode",
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
    "PolicyEditError",
    "ReplayMismatch",
    "RetentionError",
    "ProviderError",
    "__version__",
]
