"""Closed vocabularies.

These are deliberately closed sets rather than free strings: a purpose outside
the registry, or an outcome nobody defined, is the kind of thing that turns an
audit into an argument. Domain packs supply their own *values* for regimes and
rule types, but never new members of these enums.
"""
from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    """What the policy decision point concluded.

    ``OBSERVE`` exists for monitor-only deployments, where a decision is recorded
    without gating anything. Nothing in the core produces it; a pack must opt in
    explicitly.
    """

    DENY = "DENY"
    ESCALATE = "ESCALATE"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"
    ALLOW = "ALLOW"
    OBSERVE = "OBSERVE"


class CheckResult(str, Enum):
    """The verdict of a single policy check."""

    PASS = "pass"
    FAIL = "fail"
    ESCALATE = "escalate"
    REQUIRES_HUMAN = "requires_human"


#: Which outcome each failing check result forces. Order matters: see
#: :func:`kognita.governance.resolve_outcome`.
_RESULT_TO_OUTCOME: dict[CheckResult, Outcome] = {
    CheckResult.FAIL: Outcome.DENY,
    CheckResult.ESCALATE: Outcome.ESCALATE,
    CheckResult.REQUIRES_HUMAN: Outcome.HUMAN_APPROVAL,
}

#: Severity order, most severe first. Resolution is fail-closed: the most severe
#: result present decides the outcome, so one FAIL among a hundred passes still
#: denies.
OUTCOME_PRECEDENCE: tuple[Outcome, ...] = (
    Outcome.DENY,
    Outcome.ESCALATE,
    Outcome.HUMAN_APPROVAL,
    Outcome.ALLOW,
)


class Classification(str, Enum):
    """Sensitivity of a piece of data, ordered from least to most restricted.

    Compare with :func:`classification_rank` rather than by string: the ordering
    is what entitlement filtering and the egress guard are built on.
    """

    C0 = "C0"  # public
    C1 = "C1"  # internal
    C2 = "C2"  # confidential — client-identifying
    C3 = "C3"  # restricted — need-to-know


_CLASSIFICATION_ORDER = {c: i for i, c in enumerate(Classification)}


def classification_rank(value: Classification | str) -> int:
    """Position of ``value`` in the sensitivity ordering (C0 = 0 … C3 = 3)."""
    return _CLASSIFICATION_ORDER[Classification(value)]


def at_or_below(value: Classification | str, ceiling: Classification | str) -> bool:
    """True when ``value`` is no more sensitive than ``ceiling``."""
    return classification_rank(value) <= classification_rank(ceiling)


class ActorType(str, Enum):
    """Who or what took an action."""

    HUMAN = "HUMAN"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"


class EventType(str, Enum):
    """The kinds of event the evidence plane records."""

    POLICY_DECISION = "POLICY_DECISION"
    TOOL_CALL = "TOOL_CALL"
    RETRIEVAL = "RETRIEVAL"
    MODEL_CALL = "MODEL_CALL"
    APPROVAL = "APPROVAL"
    EGRESS = "EGRESS"
    POLICY_CHANGE = "POLICY_CHANGE"


class ApprovalStatus(str, Enum):
    """Lifecycle of a regulated approval."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class EgressDecision(str, Enum):
    """What the egress guard concluded about an outbound model call."""

    ALLOW = "ALLOW"
    REDACT = "REDACT"
    DENY = "DENY"
