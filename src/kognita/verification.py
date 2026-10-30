"""Registered verifiers and fail-closed claim verification.

A use case at tier 2 or above cannot release material output until its
``verifier`` field names a registered verifier. The verifier records one
validator status per claim. Independence is decided from the models pinned on
that event: the same model, model version, or provider is denied unless the
use case is tier 1 and its record sets ``permit_same_model`` to true. A
verifier cannot verify its own claims in any case.
"""
from __future__ import annotations

import json as _json
from dataclasses import asdict as _asdict, dataclass as _dataclass, replace as _replace

from sqlmodel import select as _select

from kognita.canonical import hash_text as _hash_text
from kognita.claims import Claim as _Claim, ValidatorStatus as _ValidatorStatus
from kognita.envelope import Check as _Check
from kognita.evidence import verify_chain as _verify_chain
from kognita.models import EvidenceEvent as _EvidenceEvent
from kognita.retention import RetentionStore as _RetentionStore
from kognita.vocabulary import CheckResult as _CheckResult, EventType as _EventType

_CITATION = (
    "Claims: a verifier independent of the producer records validator status, "
    "and policy decides which status may reach an RM or a client"
)
_AUDIENCES = frozenset({"rm", "client"})
_DEFAULT_AUDIENCES = {
    "VERIFIED": frozenset({"client", "rm"}),
    "INFERRED": frozenset({"rm"}),
    "UNVERIFIED": frozenset(),
    "CONFLICTING": frozenset(),
    "STALE": frozenset(),
    "BLOCKED": frozenset(),
}


def _norm(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().casefold()


def _text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _check(reason: str, *, passed: bool) -> _Check:
    return _Check(
        check=reason,
        regime="INTERNAL",
        result=_CheckResult.PASS if passed else _CheckResult.FAIL,
        citation=_CITATION,
    )


@_dataclass(frozen=True)
class Verifier:
    """A registered agent or service that checks another agent's claims."""

    verifier_id: str
    agent_id: str
    model: str
    model_version: str
    provider: str

    def __post_init__(self) -> None:
        fields = (self.verifier_id, self.agent_id, self.model, self.model_version, self.provider)
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise ValueError(
                "C7-INVALID: verifier id, agent identity, model, model version and provider are required"
            )
        object.__setattr__(self, "verifier_id", self.verifier_id.strip())
        object.__setattr__(self, "agent_id", self.agent_id.strip())
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "model_version", self.model_version.strip())
        object.__setattr__(self, "provider", self.provider.strip())


def _events(session):
    _verify_chain(session)
    return session.exec(_select(_EvidenceEvent).order_by(_EvidenceEvent.sequence)).all()


def _read_json(session, digest: str, message: str) -> dict:
    body = _RetentionStore().read(session, digest)
    if body is None or _hash_text(body) != digest:
        raise ValueError(message)
    loaded = _json.loads(body)
    if not isinstance(loaded, dict):
        raise ValueError(message)
    return loaded


def _verifiers(session) -> dict[str, Verifier]:
    found: dict[str, Verifier] = {}
    for event in _events(session):
        if event.payload.get("action") != "register_verifier":
            continue
        loaded = _read_json(
            session,
            str(event.payload.get("verifier_hash") or ""),
            "C7-CONTENT: verifier record missing or changed",
        )
        verifier = Verifier(**loaded)
        found[verifier.verifier_id] = verifier
    return found


def register_verifier(session, verifier: Verifier, *, evidence, interaction_id: str) -> Verifier:
    """Retain a verifier identity. A later registration cannot rewrite it."""
    if not isinstance(verifier, Verifier):
        raise ValueError("C7-INVALID: verifier id, agent identity, model, model version and provider are required")
    current = _verifiers(session).get(verifier.verifier_id)
    if current is not None:
        if current != verifier:
            raise ValueError("C7-IMMUTABLE: a registered verifier cannot be rewritten")
        return current
    digest = _RetentionStore().retain_value(
        session, _asdict(verifier), kind="verifier", correlation_id=interaction_id
    )
    evidence.emit(
        session,
        correlation_id=interaction_id,
        event_type=_EventType.CLAIM,
        actor_id=verifier.agent_id,
        payload={
            "action": "register_verifier",
            "verifier_id": verifier.verifier_id,
            "verifier_hash": digest,
        },
    )
    return verifier


def check_independence(
    verifier: Verifier,
    *,
    agent_id: str,
    model: str,
    model_version: str,
    provider: str,
    tier: int = 4,
    permit: bool = False,
) -> _Check:
    """Deny shared models and self-verification.

    A shared model, model version, or provider is allowed only when ``tier``
    is the integer 1 and ``permit`` is exactly True. Self-verification is
    never allowed. Blank fields are a denial. The default tier denies overlap.
    """
    if not isinstance(verifier, Verifier):
        return _check("C7-PIN", passed=False)
    if any(not _norm(value) for value in (agent_id, model, model_version, provider)):
        return _check("C7-PIN", passed=False)
    if _norm(verifier.agent_id) == _norm(agent_id):
        return _check("C7-SELF", passed=False)
    exception = type(tier) is int and tier == 1 and permit is True
    if not exception:
        if _norm(verifier.model) == _norm(model):
            return _check("C7-MODEL", passed=False)
        if _norm(verifier.model_version) == _norm(model_version):
            return _check("C7-VERSION", passed=False)
        if _norm(verifier.provider) == _norm(provider):
            return _check("C7-PROVIDER", passed=False)
    return _check("C7-INDEPENDENT", passed=True)


def _usecase(session, usecase_id: str):
    if not _text(usecase_id):
        return None
    from kognita.usecases import load_register as _load_register
    return _load_register(session).get(usecase_id)


def _tier_of(entry) -> int:
    """Integer tiers 1 to 4 are honoured. A missing tier is not tier 2 or above.

    Any other stored value fails closed as tier 4, so a non-integer cannot
    skip the verifier or unlock the tier 1 exception.
    """
    if entry is None:
        return 0
    raw = entry.record.get("proposed_tier")
    if type(raw) is int and raw in (1, 2, 3, 4):
        return raw
    if raw in (None, "", 0):
        return 0
    return 4


def _permit_of(entry) -> bool:
    return entry is not None and entry.record.get("permit_same_model") is True


def _named_verifier(entry) -> str:
    if entry is None:
        return ""
    return _text(entry.record.get("verifier"))


def _claim(session, claim_id: str) -> _Claim:
    from kognita.claims import list_claims as _list_claims
    for claim in _list_claims(session):
        if claim.claim_id == claim_id:
            return claim
    raise ValueError("C1-UNKNOWN: claim is not recorded")


def _producer(agent_id: object, model: object, model_version: object, provider: object) -> dict[str, str]:
    return {
        "agent_id": _text(agent_id),
        "model": _text(model),
        "model_version": _text(model_version),
        "provider": _text(provider),
    }


def _decide(
    claim: _Claim,
    verifier: Verifier | None,
    producer: dict[str, str],
    *,
    verifier_id: object,
    entry,
    tier: int,
) -> str:
    named = _named_verifier(entry)
    supplied = _text(verifier_id)
    if tier >= 2 and not named:
        return "C7-TIER"
    if named and named != supplied:
        return "C7-MISMATCH"
    if verifier is None:
        return "C7-UNREGISTERED"
    if not _norm(claim.agent_id) or not _norm(claim.model_version):
        return "C7-PIN"
    if any(not value for value in producer.values()):
        return "C7-PIN"
    if _norm(claim.agent_id) != _norm(producer["agent_id"]) or _norm(claim.model_version) != _norm(producer["model_version"]):
        return "C7-PRODUCER"
    independence = check_independence(
        verifier,
        agent_id=producer["agent_id"],
        model=producer["model"],
        model_version=producer["model_version"],
        provider=producer["provider"],
        tier=tier,
        permit=_permit_of(entry),
    )
    if independence.result != _CheckResult.PASS:
        return independence.check
    return "C7-RECORDED"


def verify_claim(
    session,
    claim_id: str,
    status: _ValidatorStatus,
    *,
    verifier_id: str,
    agent_id: str,
    model: str,
    model_version: str,
    provider: str,
    evidence,
    interaction_id: str,
    usecase_id: str = "",
) -> _Check:
    """Record one validator status when the verifier is independent.

    The tier and any shared-model permit come from the stored use case. Both
    models are written into the ``CLAIM_VERIFIED`` event and the retention
    store. A denial is evidenced and does not change the claim.
    """
    try:
        parsed = _ValidatorStatus(status)
    except ValueError:
        raise ValueError("C7-STATUS: validator status is not recognised") from None
    claim = _claim(session, claim_id)
    registered = _verifiers(session)
    supplied = _text(verifier_id)
    verifier = registered.get(supplied)
    entry = _usecase(session, usecase_id)
    tier = _tier_of(entry)
    producer = _producer(agent_id, model, model_version, provider)
    reason = _decide(
        claim,
        verifier,
        producer,
        verifier_id=verifier_id,
        entry=entry,
        tier=tier,
    )
    allowed = reason == "C7-RECORDED"
    body = {
        "claim_id": claim.claim_id,
        "verifier": _asdict(verifier) if verifier is not None else None,
        "producer": producer,
        "status": parsed.value,
        "tier": tier,
        "permit_same_model": _permit_of(entry),
        "reason": reason,
        "allowed": allowed,
    }
    digest = _RetentionStore().retain_value(
        session, body, kind="verification", correlation_id=interaction_id
    )
    payload = {
        "action": "verify",
        "claim_id": claim.claim_id,
        "verifier_id": supplied,
        "status": parsed.value,
        "allowed": allowed,
        "reason": reason,
        "verification_hash": digest,
        "producer_agent_id": producer["agent_id"],
        "producer_model": producer["model"],
        "producer_model_version": producer["model_version"],
        "producer_provider": producer["provider"],
        "verifier_agent_id": verifier.agent_id if verifier is not None else "",
        "verifier_model": verifier.model if verifier is not None else "",
        "verifier_model_version": verifier.model_version if verifier is not None else "",
        "verifier_provider": verifier.provider if verifier is not None else "",
    }
    if allowed:
        updated = _replace(claim, validator_status=parsed)
        payload["claim_hash"] = _RetentionStore().retain_value(
            session, _asdict(updated), kind="claim", correlation_id=interaction_id
        )
    evidence.emit(
        session,
        correlation_id=interaction_id,
        event_type=_EventType.CLAIM_VERIFIED,
        actor_id=verifier.agent_id if verifier is not None else "system",
        payload=payload,
    )
    return _check(reason, passed=allowed)


def _normalize_map(mapping: object) -> dict[str, frozenset[str]]:
    """A broken map allows nothing. Missing statuses allow nothing."""
    blocked = {key: frozenset() for key in _DEFAULT_AUDIENCES}
    if not isinstance(mapping, dict):
        return blocked
    table: dict[str, frozenset[str]] = {}
    for key in _DEFAULT_AUDIENCES:
        raw = mapping.get(key, [])
        if not isinstance(raw, list):
            return blocked
        allowed = {
            item.strip().casefold()
            for item in raw
            if isinstance(item, str) and item.strip().casefold() in _AUDIENCES
        }
        table[key] = frozenset(allowed)
    return table


def _reaches(status: object, audience: object, table: dict[str, frozenset[str]]) -> bool:
    if not isinstance(audience, str) or audience.strip().casefold() not in _AUDIENCES:
        return False
    try:
        name = _ValidatorStatus(status).value
    except ValueError:
        return False
    return audience.strip().casefold() in table.get(name, frozenset())


def _table_for(session) -> dict[str, frozenset[str]]:
    from kognita.models import Policy as _Policy, in_force as _in_force, utcnow as _utcnow
    rows = [
        row
        for row in _in_force(session, _Policy, _utcnow())
        if row.rule_type == "CONTENT_POLICY"
        and isinstance(row.rule, dict)
        and "validator_audiences" in row.rule
    ]
    if not rows:
        return dict(_DEFAULT_AUDIENCES)
    combined = None
    for row in rows:
        current = _normalize_map(row.rule.get("validator_audiences"))
        combined = current if combined is None else {
            key: combined[key] & current[key] for key in combined
        }
    return combined


def _latest_verification(session, claim_id: str):
    found = None
    for event in _events(session):
        payload = event.payload or {}
        if (
            event.event_type == _EventType.CLAIM_VERIFIED
            and payload.get("action") == "verify"
            and payload.get("allowed") is True
            and payload.get("claim_id") == claim_id
        ):
            found = event
    return found


def _payload_agrees(payload: dict, body: dict, verifier: Verifier, producer: dict) -> bool:
    return (
        payload.get("producer_model") == producer.get("model")
        and payload.get("producer_model_version") == producer.get("model_version")
        and payload.get("producer_provider") == producer.get("provider")
        and payload.get("producer_agent_id") == producer.get("agent_id")
        and payload.get("verifier_model") == verifier.model
        and payload.get("verifier_model_version") == verifier.model_version
        and payload.get("verifier_provider") == verifier.provider
        and payload.get("verifier_agent_id") == verifier.agent_id
        and body.get("verifier") == _asdict(verifier)
    )


def _recorded_release(session, claim: _Claim, entry, tier: int, event) -> str | None:
    """Re-check the pinned verification. A later status edit does not count."""
    named = _named_verifier(entry)
    payload = event.payload or {}
    recorded_id = _text(payload.get("verifier_id"))
    if named and recorded_id != named:
        return "C7-MISMATCH"
    if not recorded_id:
        return "C7-UNVERIFIED"
    verifier = _verifiers(session).get(recorded_id)
    if verifier is None or (named and verifier.verifier_id != named):
        return "C7-UNREGISTERED"
    body = _read_json(
        session,
        str(payload.get("verification_hash") or ""),
        "C7-CONTENT: verification record missing or changed",
    )
    producer = body.get("producer")
    if not isinstance(producer, dict) or not _payload_agrees(payload, body, verifier, producer):
        return "C7-PIN"
    if _norm(claim.agent_id) != _norm(producer.get("agent_id")) or _norm(claim.model_version) != _norm(producer.get("model_version")):
        return "C7-PRODUCER"
    independence = check_independence(
        verifier,
        agent_id=str(producer.get("agent_id") or ""),
        model=str(producer.get("model") or ""),
        model_version=str(producer.get("model_version") or ""),
        provider=str(producer.get("provider") or ""),
        tier=tier,
        permit=_permit_of(entry),
    )
    if independence.result != _CheckResult.PASS:
        return independence.check
    try:
        recorded = _ValidatorStatus(body.get("status"))
    except ValueError:
        return "C7-STATUS"
    if claim.validator_status != recorded:
        return "C7-STATUS"
    return None


def _release_reason(session, claim: _Claim, *, audience: str, usecase_id: str) -> str | None:
    """None when the stored status may reach ``audience``."""
    entry = _usecase(session, usecase_id)
    tier = _tier_of(entry)
    named = _named_verifier(entry)
    if tier >= 2 and not named:
        return "C7-TIER"
    if tier >= 2 and _verifiers(session).get(named) is None:
        return "C7-UNREGISTERED"
    event = _latest_verification(session, claim.claim_id)
    if tier >= 2 and event is None:
        return "C7-UNVERIFIED"
    if event is not None:
        denial = _recorded_release(session, claim, entry, tier, event)
        if denial:
            return denial
    if not _reaches(claim.validator_status, audience, _table_for(session)):
        return "C7-AUDIENCE"
    return None


__all__ = ["Verifier", "register_verifier", "verify_claim", "check_independence"]
