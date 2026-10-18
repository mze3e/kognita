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

Do-not-contact is per client. While it stands, a later recommendation, an RM
approval, and any tool call a ``CONTACT_FREQUENCY`` policy marks in ``tools``
are refused. A policy with no ``tools`` list does not mark tool calls. The
bar lifts only when :func:`reverse_do_not_contact` records a cited reversal
by an authorised human. The name is the roadmap outcome plus reversal.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from sqlmodel import Session, select

from kognita.envelope import Check, Envelope
from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.interaction import require_interaction
from kognita.models import (
    Agent,
    EvidenceEvent,
    HumanBoundary,
    Policy,
    as_utc,
    policy_content_hash,
    utcnow,
)
from kognita.usecases import load_register
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
    is already in force for this client is refused and not recorded.

    A contact-frequency policy that this recommendation would cross is
    honoured by recording do-not-contact instead of the recommendation.
    """
    at = now or utcnow()
    interaction = require_interaction(session, interaction_id)
    _trigger(trigger)
    _candidate_set(candidate_set_size)
    criteria_list = _criteria(criteria)
    standing = do_not_contact_reasons(session, interaction.id)
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
    """Reasons do-not-contact is in force for this interaction's client.

    The latest do-not-contact or reversal for that client decides. A reversal
    clears the bar. An earlier recommendation does not. An empty tuple means
    contact is not refused. The name follows the roadmap outcome.
    """
    interaction = require_interaction(session, interaction_id)
    return _client_standing(session, interaction.client_id)


def reverse_do_not_contact(
    session: Session,
    interaction_id: str,
    *,
    evidence: EvidenceWriter,
    human: str,
    citation: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Lift do-not-contact for this interaction's client.

    The reversal is recorded, and it cites ``citation``. ``human`` must be an
    authorised person for the use case: the human boundary's ``who``, or a
    person named on the use-case register. A registered agent is not a human.
    A missing citation, a missing human, an unauthorised human, or a client
    who is not under do-not-contact is refused and not recorded.
    """
    at = now or utcnow()
    interaction = require_interaction(session, interaction_id)
    party = human.strip()
    basis = citation.strip()
    if not party:
        raise OriginationError(
            "a reversal names the human who makes it "
            "(Origination: do not contact reversal)"
        )
    if not basis:
        raise OriginationError(
            "a reversal cites the basis for lifting do not contact "
            "(Origination: do not contact reversal)"
        )
    if not _client_standing(session, interaction.client_id):
        raise OriginationError(
            f"do not contact is not in force for client {interaction.client_id} "
            "(Origination: do not contact reversal)"
        )
    if _is_agent(session, party):
        raise OriginationError(
            f"{party} is a registered agent and cannot reverse do not contact "
            "(Origination: do not contact reversal)"
        )
    authorised = _authorised_humans(session, interaction.use_case, at)
    if not authorised:
        raise OriginationError(
            f"use case {interaction.use_case or '(none)'} names no authorised "
            "human who can reverse do not contact "
            "(Origination: do not contact reversal)"
        )
    if party not in authorised:
        raise OriginationError(
            f"{party} is not an authorised human for use case "
            f"{interaction.use_case} (Origination: do not contact reversal)"
        )
    payload: dict[str, Any] = {
        "interaction_id": interaction.id,
        "client_id": interaction.client_id,
        "outcome": "reversal",
        "human": party,
        "citation": basis,
        "checks": [
            {
                "check": "DO_NOT_CONTACT",
                "regime": "ORIGINATION",
                "result": CheckResult.PASS.value,
                "citation": basis,
            }
        ],
    }
    evidence.emit(
        session,
        correlation_id=interaction.id,
        event_type=EventType.ORIGINATION,
        actor_type=ActorType.HUMAN,
        actor_id=party,
        payload=payload,
    )
    return payload


def contact_refusal(
    session: Session,
    envelope: Envelope,
    *,
    interaction_id: str | None = None,
    now: datetime | None = None,
) -> list[Check]:
    """Failing checks when this tool call would contact a barred client.

    An empty list means the call is not a marked contact, or the named client
    is not under do-not-contact. A marked tool with no client, or with two
    different clients, fails closed.
    """
    at = now or utcnow()
    policies = _marking_policies(session, envelope.tool, at)
    if not policies:
        return []
    client_id, conflict = _call_client(session, envelope, interaction_id)
    citation = _mark_citation(policies, envelope.tool)
    policy = policies[0]
    if conflict or not client_id:
        return [
            _refusal_check(
                policy,
                f"{citation}, and the client could not be determined "
                "(Origination: do not contact)",
            )
        ]
    standing = _client_standing(session, client_id)
    if not standing:
        return []
    reasons = ", ".join(standing)
    return [
        _refusal_check(
            policy,
            f"Origination: do not contact ({reasons}). {citation}",
        )
    ]


def _client_standing(session: Session, client_id: str) -> tuple[str, ...]:
    """Reasons on the latest do-not-contact, unless a later reversal cleared it."""
    rows = session.exec(
        select(EvidenceEvent)
        .where(EvidenceEvent.event_type == EventType.ORIGINATION)
        .order_by(EvidenceEvent.sequence)
    ).all()
    standing: tuple[str, ...] = ()
    for event in rows:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("client_id") != client_id:
            continue
        outcome = payload.get("outcome")
        if outcome == "do not contact":
            reasons = payload.get("do_not_contact_reasons") or []
            standing = tuple(str(item) for item in reasons)
        elif outcome == "reversal":
            standing = ()
    return standing


def _is_agent(session: Session, name: str) -> bool:
    row = session.exec(select(Agent).where(Agent.name == name)).first()
    return row is not None


def _authorised_humans(session: Session, use_case: str, at: datetime) -> set[str]:
    """People who may reverse do-not-contact for this use case.

    The human boundary names who must intervene. The register names the
    owners, the Knowledge Lead, the control function, and the accountability
    seats. An empty set means nobody is authorised, and a reversal is refused.
    """
    names: set[str] = set()
    if not use_case:
        return names
    boundary = session.get(HumanBoundary, use_case)
    if boundary is not None and boundary.who.strip():
        names.add(boundary.who.strip())
    entry = load_register(session, as_of=at).get(use_case)
    if entry is None:
        return names
    for person in (
        entry.business_owner,
        entry.technology_owner,
        entry.risk_owner,
        entry.model_owner,
        entry.knowledge_lead,
        entry.control_function,
    ):
        if person and person.strip():
            names.add(person.strip())
    for _label, person in entry.accountability:
        if person and str(person).strip():
            names.add(str(person).strip())
    return names


def _marking_policies(session: Session, tool: str, at: datetime) -> list[Policy]:
    """Effective contact-frequency policies that list ``tool`` as a contact."""
    rows = session.exec(
        select(Policy).where(Policy.rule_type == "CONTACT_FREQUENCY")
    ).all()
    found: list[Policy] = []
    for policy in rows:
        if not policy.is_effective(at):
            continue
        rule = policy.rule if isinstance(policy.rule, dict) else {}
        tools = rule.get("tools")
        if not isinstance(tools, list):
            continue
        names = {
            item.strip() for item in tools if isinstance(item, str) and item.strip()
        }
        if tool in names:
            found.append(policy)
    return found


def _call_client(
    session: Session, envelope: Envelope, interaction_id: str | None
) -> tuple[str, bool]:
    """The client this call is about, and whether the names conflict.

    A conflict, or no name at all, is ``("", True)`` only for the conflict.
    No name is ``("", False)``.
    """
    names: list[str] = []
    if interaction_id:
        interaction = require_interaction(session, interaction_id)
        if interaction.client_id.strip():
            names.append(interaction.client_id.strip())
    if envelope.subject_type == "client" and envelope.subject_id:
        client = str(envelope.subject_id).strip()
        if client and client not in names:
            names.append(client)
    named = envelope.subjects.get("client")
    if isinstance(named, str) and named.strip() and named.strip() not in names:
        names.append(named.strip())
    if len(names) > 1:
        return "", True
    if not names:
        return "", False
    return names[0], False


def _mark_citation(policies: Sequence[Policy], tool: str) -> str:
    texts = [policy.citation or "Origination: contact frequency" for policy in policies]
    joined = "; ".join(texts)
    return f"{joined} marks {tool} as contacting the client"


def _refusal_check(policy: Policy, citation: str) -> Check:
    return Check(
        check="DO_NOT_CONTACT",
        regime="ORIGINATION",
        result=CheckResult.FAIL,
        citation=citation,
        policy_id=policy.id,
        policy_hash=policy.content_hash or policy_content_hash(policy),
    )


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


authorised_humans = _authorised_humans


__all__ = [
    "DO_NOT_CONTACT_REASONS",
    "OriginationError",
    "TRIGGERS",
    "authorised_humans",
    "contact_refusal",
    "do_not_contact_reasons",
    "originate",
    "reverse_do_not_contact",
]
