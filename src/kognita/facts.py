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


_SUPPRESSED = object()
_FACT_MARKERS = {"value", "source", "as_of", "owner"}
_FACT_REQUIRED = {"value", "source", "owner", "as_of", "classification", "entitlement"}


def _is_fact(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    keys = set(node)
    if "data_type" in keys and keys & _FACT_MARKERS:
        return True
    return _FACT_REQUIRED <= keys


def _duration(value: Any) -> timedelta | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, timedelta):
        return value if value >= timedelta(0) else None
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return timedelta(seconds=float(value))
    if isinstance(value, str):
        import re
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([smhd])", value.strip())
        if match is None:
            return None
        scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
        return timedelta(seconds=float(match.group(1)) * scale)
    return None


def _moment(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _classification(value: Any) -> Classification:
    try:
        return Classification(value)
    except (ValueError, TypeError):
        return Classification.C1


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _build_fact(node: Mapping[str, Any]) -> Fact:
    threshold = node.get("freshness_threshold")
    return Fact(
        value=node.get("value"),
        source=_text(node.get("source")),
        owner=_text(node.get("owner")),
        as_of=_moment(node.get("as_of")),
        classification=_classification(node.get("classification", Classification.C1)),
        entitlement=_text(node.get("entitlement")),
        freshness_threshold=_duration(threshold) if threshold is not None else None,
    )


def _research(node: Mapping[str, Any]) -> ResearchLifecycle | None:
    raw = node.get("research")
    if not isinstance(raw, dict):
        return None
    try:
        return ResearchLifecycle(
            author=_text(raw.get("author")),
            publication_date=_moment(raw.get("publication_date")) or datetime(2000, 1, 1, tzinfo=timezone.utc),
            passage=_text(raw.get("passage")),
            version=_text(raw.get("version")),
            created=_moment(raw.get("created")) or datetime(2000, 1, 1, tzinfo=timezone.utc),
            effective=_moment(raw.get("effective")) or datetime(2099, 1, 1, tzinfo=timezone.utc),
            last_reviewed=_moment(raw.get("last_reviewed")) or datetime(2000, 1, 1, tzinfo=timezone.utc),
            superseded=_moment(raw.get("superseded")) if raw.get("superseded") else None,
            expiry=_moment(raw.get("expiry")) if raw.get("expiry") else None,
        )
    except (TypeError, ValueError):
        return ResearchLifecycle(
            author="invalid",
            publication_date=datetime(2000, 1, 1, tzinfo=timezone.utc),
            passage="invalid",
            version="invalid",
            created=datetime(2000, 1, 1, tzinfo=timezone.utc),
            effective=datetime(2099, 1, 1, tzinfo=timezone.utc),
            last_reviewed=datetime(2000, 1, 1, tzinfo=timezone.utc),
            superseded=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )


def _freshness_rules(entry: Any) -> dict[str, FreshnessRule]:
    raw = getattr(entry, "record", {}) or {}
    mapping = raw.get("freshness") if isinstance(raw, dict) else None
    if not isinstance(mapping, dict):
        return {}
    rules: dict[str, FreshnessRule] = {}
    for name, threshold in mapping.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(threshold, dict) and "max_age" in threshold:
            threshold = threshold["max_age"]
        age = _duration(threshold)
        if age is None:
            continue
        rules[name] = FreshnessRule(name, max_age=age)
    return rules


def _case(session: Any, use_case: str, now: datetime) -> Any:
    from types import SimpleNamespace
    from kognita.usecases import load_register
    entry = load_register(session, as_of=now).get(use_case or "")
    if entry is not None:
        return entry
    return SimpleNamespace(usecase_id=use_case or "", stale_action=StaleAction.SUPPRESS, record={})


def _dump_fact(fact: Fact, data_type: str, warning: bool) -> dict[str, Any]:
    return {
        "value": fact.value,
        "source": fact.source,
        "owner": fact.owner,
        "as_of": fact.as_of.isoformat() if isinstance(fact.as_of, datetime) else None,
        "classification": fact.classification.value,
        "entitlement": fact.entitlement,
        "freshness_threshold": (
            fact.freshness_threshold.total_seconds() if fact.freshness_threshold is not None else None
        ),
        "data_type": data_type,
        "warning": warning,
    }


def _admit_node(session: Any, node: dict[str, Any], *, use_case: str, evidence: Any,
                correlation_id: str, now: datetime) -> Any:
    case = _case(session, use_case, now)
    data_type = node.get("data_type") if isinstance(node.get("data_type"), str) else ""
    refreshed = node.get("refreshed") if isinstance(node.get("refreshed"), dict) else None

    def refresher(_fact: Fact) -> Fact:
        if refreshed is None:
            raise RuntimeError("no refresh")
        return _build_fact(refreshed)

    from kognita.retention import RetentionStore
    admitted, warning = admit_fact(
        _build_fact(node),
        case,
        data_type,
        _freshness_rules(case),
        session=session,
        evidence=evidence,
        retention=RetentionStore(),
        correlation_id=correlation_id or "fact",
        refresher=refresher if refreshed is not None else None,
        now=now,
        research=_research(node),
    )
    if admitted is None:
        return _SUPPRESSED
    return _dump_fact(admitted, data_type, warning)


def _admit_observed(session: Any, payload: Any, *, use_case: str, evidence: Any,
                    correlation_id: str, now: datetime | None = None) -> Any:
    """Admit fact-shaped values on a retrieval, tool, or gateway result.

    A fact with no freshness rule is stale. The use case's ``stale_action``
    refreshes, flags, or removes it. Removed facts are not returned.
    """
    at = now or datetime.now(timezone.utc)
    if at.tzinfo is None or at.utcoffset() is None:
        at = at.replace(tzinfo=timezone.utc)

    def walk(node: Any) -> Any:
        if _is_fact(node):
            if evidence is None:
                return _SUPPRESSED
            return _admit_node(
                session, node, use_case=use_case, evidence=evidence,
                correlation_id=correlation_id, now=at,
            )
        if isinstance(node, dict):
            kept = {}
            for key, item in node.items():
                walked = walk(item)
                if walked is not _SUPPRESSED:
                    kept[key] = walked
            return kept
        if isinstance(node, list):
            return [item for item in (walk(item) for item in node) if item is not _SUPPRESSED]
        if isinstance(node, tuple):
            return tuple(item for item in (walk(item) for item in node) if item is not _SUPPRESSED)
        return node

    walked = walk(payload)
    if walked is _SUPPRESSED:
        return None
    return walked


__all__ = ["Fact", "FreshnessRule", "StaleAction", "ResearchLifecycle", "admit_fact", "is_stale"]
