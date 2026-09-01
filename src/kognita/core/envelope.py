"""The governance envelope — what is asked, by whom, for what purpose.

Authorise before discovery: an envelope describes an *intent* and is evaluated
before any data is fetched. It deliberately carries references to subjects
(``subject_type`` / ``subject_id``) rather than the subjects themselves, so that
constructing one leaks nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kognita.core.canonical import canonical_hash
from kognita.core.vocabulary import CheckResult, Outcome


@dataclass(frozen=True)
class Envelope:
    """A request for permission, evaluated before anything is retrieved."""

    principal: str
    purpose: str
    tool: str
    #: Where the actor is acting from — the zone used for entitlement filtering.
    actor_location: str = ""
    agent_name: str | None = None
    #: The subject of the request, by reference. ``("client", "1")`` and so on.
    subject_type: str | None = None
    subject_id: str | None = None
    #: Additional referenced objects a pack needs, e.g. {"instrument": "3"}.
    subjects: dict[str, str] = field(default_factory=dict)
    #: True when the principal holds an elevated entitlement ceiling.
    is_admin: bool = False
    #: Free-form request context. Never trusted for authorisation decisions.
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal": self.principal,
            "purpose": self.purpose,
            "tool": self.tool,
            "actor_location": self.actor_location,
            "agent_name": self.agent_name,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "subjects": dict(self.subjects),
            "is_admin": self.is_admin,
        }

    def all_subjects(self) -> dict[str, str]:
        """Every referenced subject, including the primary one."""
        merged = dict(self.subjects)
        if self.subject_type and self.subject_id is not None:
            merged.setdefault(self.subject_type, self.subject_id)
        return merged


@dataclass(frozen=True)
class Check:
    """One policy check and its basis.

    ``citation`` is required by convention throughout: a decision a regulator
    cannot trace to a rule is not a decision, it is an assertion. The conformance
    kit enforces that every check carries one.
    """

    check: str
    regime: str
    result: CheckResult
    citation: str
    policy_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "regime": self.regime,
            "result": self.result.value,
            "citation": self.citation,
            "policy_id": self.policy_id,
        }


@dataclass(frozen=True)
class RuleContext:
    """Everything a rule evaluator may look at.

    ``attributes`` are resolved by the domain pack before evaluation — a
    jurisdiction tuple, a care relationship, a tenancy. ``subjects`` holds the
    loaded rows those attributes were derived from.
    """

    envelope: Envelope
    attributes: dict[str, Any] = field(default_factory=dict)
    subjects: dict[str, Any] = field(default_factory=dict)
    as_of: datetime | None = None


@dataclass(frozen=True)
class Evaluation:
    """The result of :func:`kognita.core.governance.decide` — pure, unrecorded.

    Holds no database identity: ``decision_id`` is filled in only once
    :func:`kognita.core.governance.record` has persisted it.
    """

    request_id: str
    outcome: Outcome
    checks: tuple[Check, ...]
    attributes: dict[str, Any]
    envelope: Envelope
    as_of: datetime
    envelope_hash: str
    decision_id: int | None = None

    @property
    def allowed(self) -> bool:
        """Whether data may be released at all.

        ``HUMAN_APPROVAL`` is included: the work may be prepared, but it must not
        leave the boundary until the approval is granted.
        """
        return self.outcome in (Outcome.ALLOW, Outcome.HUMAN_APPROVAL, Outcome.OBSERVE)

    @property
    def approval_required(self) -> bool:
        return self.outcome == Outcome.HUMAN_APPROVAL

    def failures(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.result == CheckResult.FAIL)

    def escalations(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.result == CheckResult.ESCALATE)

    def human_reviews(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.result == CheckResult.REQUIRES_HUMAN)

    def basis(self) -> tuple[Check, ...]:
        """The checks that determined the outcome — what to show a user."""
        if self.outcome == Outcome.DENY:
            return self.failures()
        if self.outcome == Outcome.ESCALATE:
            return self.escalations()
        if self.outcome == Outcome.HUMAN_APPROVAL:
            return self.human_reviews()
        return self.checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "outcome": self.outcome.value,
            "checks": [c.to_dict() for c in self.checks],
            "attributes": dict(self.attributes),
            "envelope": self.envelope.to_dict(),
            "as_of": self.as_of.isoformat(),
            "envelope_hash": self.envelope_hash,
        }


def envelope_hash(
    envelope: Envelope, attributes: dict[str, Any], checks: tuple[Check, ...]
) -> str:
    """Bind an envelope, its resolved attributes and its checks into one hash.

    Approvals are issued against this value, so changing any of the three
    invalidates an approval rather than silently widening it.
    """
    return canonical_hash(
        {
            "envelope": envelope.to_dict(),
            "attributes": attributes,
            "checks": [c.to_dict() for c in checks],
        }
    )


__all__ = ["Envelope", "Check", "RuleContext", "Evaluation", "envelope_hash"]
