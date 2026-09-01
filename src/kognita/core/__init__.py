"""The governed decision engine.

Everything here installs and runs on Kognita's four hard dependencies — no LLM,
no graph database, no network. That is the point: deciding whether a request is
permitted, and proving afterwards that it was, should not require the machinery
that answers it.

A minimal use::

    from kognita.core import Envelope, decide, PolicySnapshot

    evaluation = decide(
        Envelope(principal="alice", purpose="SUPPORT", tool="get_profile"),
        PolicySnapshot(policies=..., agents=...),
    )
    evaluation.outcome        # ALLOW / DENY / ESCALATE / HUMAN_APPROVAL
    evaluation.basis()        # the checks that decided it, each with a citation

See ``kognita.testing`` for the conformance kit a domain pack runs against
itself.
"""
from kognita.core.approvals import ApprovalError, expire_stale, find_granted, grant, reject
from kognita.core.broker import BrokerAnswer, ask, default_route_resolver
from kognita.core.canonical import canonical_hash, canonical_json
from kognita.core.classify import FixedClassifier, PatternClassifier, most_sensitive
from kognita.core.db import create_all, make_engine, session_scope
from kognita.core.egress import (
    EgressDenied,
    EgressGuard,
    EgressPolicy,
    EgressResult,
    NullRedactor,
    PatternRedactor,
)
from kognita.core.embedding import HashingEmbedder, cosine, lexical_overlap
from kognita.core.envelope import Check, Envelope, Evaluation, RuleContext, envelope_hash
from kognita.core.evidence import (
    ChainBreak,
    EvidenceWriter,
    export_chain,
    hashes_only,
    verify_chain,
    verify_export,
)
from kognita.core.governance import (
    PolicySnapshot,
    decide,
    load_snapshot,
    record,
    resolve_outcome,
)
from kognita.core.models import (
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
from kognita.core.retrieval import Retrieved, index_item, reindex, retrieve
from kognita.core.rules import CORE_RULES, build_registry, rule
from kognita.core.tools import ToolRegistry, ToolRun, run_governed
from kognita.core.vectors import NumpyVectorIndex, SqliteVecIndex, default_index
from kognita.core.vocabulary import (
    ActorType,
    ApprovalStatus,
    CheckResult,
    Classification,
    EgressDecision,
    EventType,
    Outcome,
)

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
]
