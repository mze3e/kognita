"""Fact admission and fail-closed freshness checks.

RETRIEVAL events with action FACT_ADMISSION record outcomes, with content
retained separately. Admission returns (fact or None, warning flag).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from kognita.vocabulary import Classification, EventType, StaleAction


@dataclass(frozen=True)
class ResearchLifecycle:
    author: str
    publication_date: datetime
    passage: str
    version: str
    created: datetime
    effective: datetime
    last_reviewed: datetime
    superseded: datetime | None = None
    expiry: datetime | None = None


@dataclass(frozen=True)
class Fact:
    value: Any
    source: str
    owner: str
    as_of: datetime | None
    classification: Classification
    entitlement: str
    freshness_threshold: timedelta | None


@dataclass(frozen=True)
class FreshnessRule:
    data_type: str
    max_age: timedelta | None = None
    rule: Callable[[Fact, datetime], bool] | None = None

    def __post_init__(self):
        if not self.data_type.strip() or (self.max_age is None) == (self.rule is None):
            raise ValueError("freshness requires a data type and exactly one threshold or rule")
        if self.max_age is not None and (
            not isinstance(self.max_age, timedelta) or self.max_age < timedelta(0)
        ):
            raise ValueError("max_age must be a nonnegative timedelta")
        if self.rule is not None and not callable(self.rule):
            raise ValueError("rule must be callable and return True for fresh facts")


def _aware(at):
    if not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("freshness requires timezone-aware dates")
    return at.astimezone(timezone.utc)


def _research_blocked(research, now):
    if research is None:
        return False
    try:
        return (research.superseded is not None
                or (research.expiry is not None and _aware(research.expiry) <= now)
                or _aware(research.effective) > now)
    except (ValueError, TypeError, AttributeError):
        return True


def is_stale(fact: Fact, data_type: str, freshness: Mapping[str, FreshnessRule], *,
             now: datetime | None = None, research: ResearchLifecycle | None = None) -> bool:
    """Missing rules, invalid dates and failed custom rules count as stale.

    A custom rule returns exactly True when fresh. The fact's own threshold
    may tighten the policy threshold, but cannot loosen or replace it.
    """
    try:
        at = _aware(now or datetime.now(timezone.utc))
        if _research_blocked(research, at):
            return True
        age = at - _aware(fact.as_of)
        policy = freshness.get(data_type)
        if not isinstance(policy, FreshnessRule) or policy.data_type != data_type or age < timedelta(0):
            return True
        if fact.freshness_threshold is not None:
            if not isinstance(fact.freshness_threshold, timedelta) or fact.freshness_threshold < timedelta(0):
                return True
            if age > fact.freshness_threshold:
                return True
        if policy.max_age is not None:
            return age > policy.max_age
        return policy.rule(fact, at) is not True
    except Exception:
        return True


def admit_fact(fact: Fact, use_case, data_type: str, freshness: Mapping[str, FreshnessRule], *,
               session, evidence, retention, correlation_id: str,
               refresher: Callable[[Fact], Fact] | None = None,
               now: datetime | None = None, research: ResearchLifecycle | None = None
               ) -> tuple[Fact | None, bool]:
    """Admit only after evidence succeeds. Invalid contracts always suppress.

    The caller supplies the authoritative data type and research lifecycle.
    Refreshed facts must preserve source, owner, classification and entitlement.
    """
    at = _aware(now or datetime.now(timezone.utc))
    from dataclasses import asdict

    result, warning, reason = fact, False, "FACT_FRESH"
    valid = (isinstance(fact, Fact) and isinstance(fact.classification, Classification)
             and all(isinstance(v, str) and v.strip() for v in (fact.source, fact.owner, fact.entitlement)))
    if not valid:
        result, reason = None, "FACT_CONTRACT_INVALID"
    elif _research_blocked(research, at):
        result, reason = None, "FACT_RESEARCH_INACTIVE"
    elif is_stale(fact, data_type, freshness, now=at, research=research):
        action = getattr(use_case, "stale_action", StaleAction.SUPPRESS)
        result, reason = None, "FACT_STALE_SUPPRESSED"
        if action == StaleAction.WARN:
            result, warning, reason = fact, True, "FACT_STALE_WARNING"
        elif action == StaleAction.REFRESH:
            reason = "FACT_REFRESH_FAILED"
            try:
                updated = refresher(fact) if refresher else None
                if (isinstance(updated, Fact)
                    and all(getattr(updated, key) == getattr(fact, key)
                            for key in ("source", "owner", "classification", "entitlement"))
                    and not is_stale(updated, data_type, freshness, now=at, research=research)):
                    result, reason = updated, "FACT_REFRESHED"
            except Exception:
                pass
    content_hash = retention.retain_value(
        session, {"fact": asdict(fact), "research": asdict(research) if research else None,
                  "admitted": asdict(result) if result else None}, kind="fact",
        use_case=use_case.usecase_id, correlation_id=correlation_id, now=at)
    evidence.emit(session, correlation_id=correlation_id, event_type=EventType.RETRIEVAL,
                  classification=fact.classification if valid else Classification.C3,
                  payload={"action": "FACT_ADMISSION", "reason_code": reason,
                           "content_hash": content_hash, "use_case": use_case.usecase_id,
                           "data_type": data_type, "warning": warning, "admitted": result is not None})
    return result, warning


__all__ = ["Fact", "FreshnessRule", "StaleAction", "ResearchLifecycle", "admit_fact", "is_stale"]
