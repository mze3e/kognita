"""The rule registry and the generic primitives it ships with.

A policy row names a ``rule_type``; the evaluator registered for that type reads
the row's JSON ``rule`` payload and returns checks. The core ships primitives
general enough to express most access rules — allowlists, denylists, required
flags, mandatory human review — and a domain pack registers evaluators for
anything its regimes need beyond them.

Every evaluator must be a pure function of ``(policy, context)``. No I/O, no
clock reads, no randomness: replaying a decision must give the same answer, or
the evidence plane is recording a story rather than a fact.
"""
from __future__ import annotations

from typing import Any, Callable

from kognita.envelope import Check, RuleContext
from kognita.vocabulary import CheckResult

Evaluator = Callable[[Any, RuleContext], list[Check]]

#: Evaluators shipped by the core, keyed by ``rule_type``.
CORE_RULES: dict[str, Evaluator] = {}


def rule(rule_type: str) -> Callable[[Evaluator], Evaluator]:
    """Register an evaluator for ``rule_type`` in the core registry."""

    def register(fn: Evaluator) -> Evaluator:
        CORE_RULES[rule_type] = fn
        return fn

    return register


def build_registry(*overlays: dict[str, Evaluator]) -> dict[str, Evaluator]:
    """Merge pack registries over the core primitives; later overlays win."""
    registry = dict(CORE_RULES)
    for overlay in overlays:
        registry.update(overlay)
    return registry


def _lookup(context: RuleContext, key: str) -> Any:
    """Read an attribute a rule refers to.

    Attributes resolved by the pack take precedence; envelope fields are the
    fallback so simple packs need not restate ``actor_location`` and friends.
    """
    if key in context.attributes:
        return context.attributes[key]
    return getattr(context.envelope, key, None)


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(v) for v in value}
    return {str(value)}


# ── Primitives ───────────────────────────────────────────────────────────────


@rule("ATTRIBUTE_ALLOWLIST")
def attribute_allowlist(policy: Any, context: RuleContext) -> list[Check]:
    """Every named attribute must be in its allowlist.

    Rule payload::

        {"allow": {"actor_location": ["SG", "HK"]},
         "on_violation": "fail" | "escalate" | "requires_human"}

    An attribute the request does not carry is a violation, not a pass: an
    allowlist that cannot be evaluated has not been satisfied.
    """
    allow: dict[str, Any] = policy.rule.get("allow", {})
    on_violation = CheckResult(policy.rule.get("on_violation", "fail"))
    checks: list[Check] = []
    for key, permitted in allow.items():
        actual = _lookup(context, key)
        permitted_set = _as_set(permitted)
        ok = actual is not None and str(actual) in permitted_set
        checks.append(
            Check(
                check=f"{policy.rule_type}: {key} {actual!s}",
                regime=policy.regime,
                result=CheckResult.PASS if ok else on_violation,
                citation=policy.citation,
                policy_id=policy.id,
            )
        )
    return checks


@rule("ATTRIBUTE_DENYLIST")
def attribute_denylist(policy: Any, context: RuleContext) -> list[Check]:
    """No named attribute may match its denylist.

    Rule payload::

        {"deny": {"actor_location": ["RU"]}, "on_violation": "fail"}
    """
    deny: dict[str, Any] = policy.rule.get("deny", {})
    on_violation = CheckResult(policy.rule.get("on_violation", "fail"))
    checks: list[Check] = []
    for key, forbidden in deny.items():
        actual = _lookup(context, key)
        hit = actual is not None and str(actual) in _as_set(forbidden)
        checks.append(
            Check(
                check=f"{policy.rule_type}: {key} {actual!s}",
                regime=policy.regime,
                result=on_violation if hit else CheckResult.PASS,
                citation=policy.citation,
                policy_id=policy.id,
            )
        )
    return checks


@rule("REQUIRES_FLAG")
def requires_flag(policy: Any, context: RuleContext) -> list[Check]:
    """Named attributes must all be truthy.

    Rule payload::

        {"flags": ["accredited_investor"], "on_violation": "fail"}
    """
    flags: list[str] = list(policy.rule.get("flags", []))
    on_violation = CheckResult(policy.rule.get("on_violation", "fail"))
    checks: list[Check] = []
    for flag in flags:
        ok = bool(_lookup(context, flag))
        checks.append(
            Check(
                check=f"{policy.rule_type}: {flag}",
                regime=policy.regime,
                result=CheckResult.PASS if ok else on_violation,
                citation=policy.citation,
                policy_id=policy.id,
            )
        )
    return checks


@rule("REQUIRES_HUMAN_APPROVAL")
def requires_human_approval(policy: Any, context: RuleContext) -> list[Check]:
    """Mandate human review, optionally only for certain tools or purposes.

    Rule payload::

        {"tools": ["draft_meeting_pack"], "purposes": ["ADVICE"]}

    Empty or absent lists mean "any". This is how an automated-decision regime is
    expressed: the work may be prepared, but a person signs it off.
    """
    tools = _as_set(policy.rule.get("tools"))
    purposes = _as_set(policy.rule.get("purposes"))
    if tools and context.envelope.tool not in tools:
        return []
    if purposes and context.envelope.purpose not in purposes:
        return []
    return [
        Check(
            check=policy.rule_type,
            regime=policy.regime,
            result=CheckResult.REQUIRES_HUMAN,
            citation=policy.citation,
            policy_id=policy.id,
        )
    ]


@rule("PROHIBITED")
def prohibited(policy: Any, context: RuleContext) -> list[Check]:
    """An unconditional bar, once the policy is engaged at all.

    Engagement is decided by ``applies_to`` and the pack's ``engages`` predicate,
    so reaching this evaluator already means the rule is in scope.
    """
    return [
        Check(
            check=policy.rule.get("description", policy.rule_type),
            regime=policy.regime,
            result=CheckResult(policy.rule.get("on_violation", "fail")),
            citation=policy.citation,
            policy_id=policy.id,
        )
    ]


__all__ = ["rule", "build_registry", "CORE_RULES", "Evaluator"]
