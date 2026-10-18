"""Origination evidence.

The step that selects a client or a product records why, before anything is
shown to an RM. The rationale is structured facts with citations. A bare score
is refused. "Do not contact" is an outcome, with a closed set of reasons, and
a later recommendation on that interaction is refused.

Suitability stays a separate check. Origination says why something was
proposed. It does not say why it was permitted.

Contact-frequency limits are the ``CONTACT_FREQUENCY`` policy rows in force.
Crossing one records do-not-contact with reason "excessive frequency" and
cites that policy. It does not record the recommendation the caller asked for.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from sqlmodel import Session, select

from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.interaction import require_interaction
from kognita.models import EvidenceEvent, Policy, as_utc, policy_content_hash, utcnow
from kognita.vocabulary import ActorType, CheckResult, EventType

#: How the selection started. The roadmap's three triggers.
TRIGGERS: tuple[str, ...] = ("event", "schedule", "RM request")

#: Why no communication is recommended. Wording is the roadmap list.
DO_NOT_CONTACT_REASONS: tuple[str, ...] = (
    "low relevance",
    "poor timing",
    "vulnerability",
    "excessive frequency",
    "a recent rejection",
    "a relationship circumstance",
)

_BARE_SCORE = re.compile(r"(?i)^(?:relevance\s+)?score\s*[:=]?\s*\d+(?:\.\d+)?\s*%?$")
_BARE_NUMBER = re.compile(r"(?i)^\d+(?:\.\d+)?\s*%?$")
_SCORE_FACTS = frozenset({"score", "relevance", "relevance score"})


class OriginationError(KognitaError):
    """An origination was refused and not recorded."""


def originate(
    session: Session,
    interaction_id: str,
    *,
    evidence: EvidenceWriter,
    trigger: str,
    candidate_set_size: int,
    criteria: Sequence[str],
    reasons: Sequence[Mapping[str, str]] | None = None,
    do_not_contact: Sequence[str] | None = None,
    product_id: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record why this client, and why this product when one was selected.

    ``do_not_contact`` names reasons from :data:`DO_NOT_CONTACT_REASONS`.
    Passing it together with ``reasons`` is refused: a recommendation and a
    refusal are not the same outcome. A recommendation after do-not-contact
    is already in force on this interaction is refused and not recorded.

    A contact-frequency policy that this recommendation would cross is
    honoured by recording do-not-contact instead of the recommendation.
    """
    at = now or utcnow()
    interaction = require_interaction(session, interaction_id)
    _trigger(trigger)
    _candidate_set(candidate_set_size)
    criteria_list = _criteria(criteria)
    standing = do_not_contact_reasons(session, interaction_id)
    recommending = do_not_contact is None
    if recommending and standing:
        joined = ", ".join(standing)
        raise OriginationError(
            f"do not contact is in force for {interaction_id} ({joined}). "
            "A recommendation is not recorded. "
            "(Origination: do not contact)"
        )
    if recommending:
        stated = _recommendation_reasons(reasons or ())
        forced = _frequency_refusal(session, interaction.client_id, at)
        if forced is not None:
            outcome = "do not contact"
            contact_reasons = ["excessive frequency"]
            recorded_reasons: list[dict[str, str]] = []
            checks = [_frequency_check(forced)]
        else:
            outcome = "recommend"
            contact_reasons = []
            recorded_reasons = stated
            checks = [_reason_check(item) for item in stated]
    else:
        if reasons:
            raise OriginationError(
                "origination cannot recommend and refuse contact in one record "
                "(Origination: do not contact)"
            )
        contact_reasons = _contact_reasons(do_not_contact)
        outcome = "do not contact"
        recorded_reasons = []
        checks = [_contact_check(reason) for reason in contact_reasons]

    payload: dict[str, Any] = {
        "interaction_id": interaction.id,
        "client_id": interaction.client_id,
        "product_id": product_id.strip(),
        "trigger": trigger,
        "candidate_set_size": candidate_set_size,
        "criteria": criteria_list,
        "outcome": outcome,
        "reasons": recorded_reasons,
        "do_not_contact_reasons": contact_reasons,
        "checks": checks,
    }
    evidence.emit(
        session,
        correlation_id=interaction.id,
        event_type=EventType.ORIGINATION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload=payload,
    )
    return payload


def do_not_contact_reasons(session: Session, interaction_id: str) -> tuple[str, ...]:
    """Reasons on the latest origination, when that outcome is do not contact.

    An empty tuple means this interaction has not refused contact. The name
    follows the roadmap outcome; callers use it to honour that outcome.
    """
    latest = _latest(session, interaction_id)
    if latest is None:
        return ()
    if latest.get("outcome") != "do not contact":
        return ()
    reasons = latest.get("do_not_contact_reasons") or []
    return tuple(str(item) for item in reasons)


def _latest(session: Session, interaction_id: str) -> dict[str, Any] | None:
    rows = session.exec(
        select(EvidenceEvent)
        .where(EvidenceEvent.event_type == EventType.ORIGINATION)
        .order_by(EvidenceEvent.sequence)
    ).all()
    found: dict[str, Any] | None = None
    for event in rows:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("interaction_id") == interaction_id:
            found = payload
    return found


def _trigger(trigger: str) -> None:
    if trigger not in TRIGGERS:
        raise OriginationError(
            f"trigger {trigger!r} is not event, schedule, or RM request "
            "(Origination: trigger)"
        )


def _candidate_set(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise OriginationError(
            "candidate_set_size must be a non-negative integer "
            "(Origination: candidate set)"
        )


def _criteria(criteria: Sequence[str]) -> list[str]:
    cleaned = [
        item.strip() for item in criteria if isinstance(item, str) and item.strip()
    ]
    if not cleaned or len(cleaned) != len(list(criteria)):
        raise OriginationError(
            "origination records the selection criteria "
            "(Origination: selection criteria)"
        )
    return cleaned


def _recommendation_reasons(
    reasons: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if not reasons:
        raise OriginationError(
            "a recommendation records the facts behind it, not an empty rationale "
            "(Origination: reasons, not scores)"
        )
    recorded: list[dict[str, str]] = []
    for reason in reasons:
        recorded.append(_one_reason(reason))
    return recorded


def _one_reason(reason: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(reason, Mapping):
        raise OriginationError(
            "a reason is a fact, a detail, and a citation "
            "(Origination: reasons, not scores)"
        )
    keys = {str(key).lower() for key in reason}
    if "score" in keys or "relevance_score" in keys:
        raise OriginationError(
            "a bare score is not a rationale " "(Origination: reasons, not scores)"
        )
    fact = str(reason.get("fact") or "").strip()
    detail = str(reason.get("detail") or "").strip()
    citation = str(reason.get("citation") or "").strip()
    if not fact or not detail or not citation:
        raise OriginationError(
            "a reason needs a fact, a detail, and a citation "
            "(Origination: reasons, not scores)"
        )
    if fact.lower() in _SCORE_FACTS or _is_bare_score(fact) or _is_bare_score(detail):
        raise OriginationError(
            "a bare score is not a rationale. Origination records the client "
            "facts and the evidence behind them. "
            "(Origination: reasons, not scores)"
        )
    return {"fact": fact, "detail": detail, "citation": citation}


def _is_bare_score(text: str) -> bool:
    return bool(_BARE_SCORE.match(text) or _BARE_NUMBER.match(text))


def _contact_reasons(reasons: Sequence[str]) -> list[str]:
    if not reasons:
        raise OriginationError(
            "do not contact records at least one reason "
            "(Origination: do not contact)"
        )
    cleaned: list[str] = []
    for reason in reasons:
        if reason not in DO_NOT_CONTACT_REASONS:
            raise OriginationError(
                f"do not contact reason {reason!r} is not in the closed set "
                "(Origination: do not contact)"
            )
        if reason not in cleaned:
            cleaned.append(reason)
    return cleaned


def _reason_check(reason: Mapping[str, str]) -> dict[str, Any]:
    return {
        "check": "ORIGINATION",
        "regime": "ORIGINATION",
        "result": CheckResult.PASS.value,
        "citation": reason["citation"],
    }


def _contact_check(reason: str) -> dict[str, Any]:
    return {
        "check": "DO_NOT_CONTACT",
        "regime": "ORIGINATION",
        "result": CheckResult.PASS.value,
        "citation": f"Origination: do not contact ({reason})",
    }


def _frequency_check(policy: Policy) -> dict[str, Any]:
    return {
        "check": "CONTACT_FREQUENCY",
        "regime": policy.regime,
        "result": CheckResult.FAIL.value,
        "citation": policy.citation or "Origination: contact frequency",
        "policy_id": policy.id,
        "policy_hash": policy.content_hash or policy_content_hash(policy),
    }


def _frequency_refusal(session: Session, client_id: str, at: datetime) -> Policy | None:
    """The first frequency policy this recommendation would cross, if any.

    A policy that does not say how many contacts, or over what window, is
    refused rather than ignored.
    """
    policies = _effective_frequency(session, at)
    for policy in policies:
        limit, window = _limit(policy)
        count = _recent_recommendations(session, client_id, at, window)
        if count >= limit:
            return policy
    return None


def _effective_frequency(session: Session, at: datetime) -> list[Policy]:
    rows = session.exec(
        select(Policy).where(Policy.rule_type == "CONTACT_FREQUENCY")
    ).all()
    found: list[Policy] = []
    for policy in rows:
        if not policy.is_effective(at):
            continue
        _limit(policy)
        found.append(policy)
    return found


def _limit(policy: Policy) -> tuple[int, int]:
    rule = policy.rule if isinstance(policy.rule, dict) else {}
    raw_limit = rule.get("max_contacts")
    raw_window = rule.get("window_days")
    if (
        isinstance(raw_limit, bool)
        or not isinstance(raw_limit, int)
        or raw_limit < 0
        or isinstance(raw_window, bool)
        or not isinstance(raw_window, int)
        or raw_window < 1
    ):
        raise OriginationError(
            f"contact-frequency policy {policy.id} cannot be evaluated "
            "(Origination: contact frequency)"
        )
    return raw_limit, raw_window


def _recent_recommendations(
    session: Session, client_id: str, at: datetime, window_days: int
) -> int:
    start = at - timedelta(days=window_days)
    rows = session.exec(
        select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.ORIGINATION)
    ).all()
    count = 0
    for event in rows:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("client_id") != client_id:
            continue
        if payload.get("outcome") != "recommend":
            continue
        recorded = as_utc(event.recorded_at)
        if recorded is not None and recorded >= start:
            count += 1
    return count


__all__ = [
    "DO_NOT_CONTACT_REASONS",
    "OriginationError",
    "TRIGGERS",
    "do_not_contact_reasons",
    "originate",
]
