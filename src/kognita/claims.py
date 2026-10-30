"""Typed claim snapshots, retained content and evidenced release checks."""
from __future__ import annotations

import json as _json
from dataclasses import dataclass as _dataclass, asdict as _asdict, replace as _replace
from datetime import datetime as _datetime
from enum import Enum as _Enum

from sqlmodel import Session as _Session, select as _select

from kognita.canonical import hash_text as _hash_text
from kognita.envelope import Check as _Check
from kognita.evidence import EvidenceWriter as _EvidenceWriter, verify_chain as _verify_chain
from kognita.models import EvidenceEvent as _EvidenceEvent
from kognita.retention import RetentionStore as _RetentionStore
from kognita.vocabulary import EventType as _EventType, CheckResult as _CheckResult


class ClaimType(str, _Enum):
    VERIFIED_FACT = "VERIFIED_FACT"
    INFERENCE = "INFERENCE"
    CONVERSATION_HYPOTHESIS = "CONVERSATION_HYPOTHESIS"
    POSSIBLE_BANK_CAPABILITY = "POSSIBLE_BANK_CAPABILITY"


class ValidatorStatus(str, _Enum):
    VERIFIED = "VERIFIED"
    INFERRED = "INFERRED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICTING = "CONFLICTING"
    STALE = "STALE"
    BLOCKED = "BLOCKED"


@_dataclass(frozen=True)
class Claim:
    claim_id: str
    client_id: str
    claim: str
    claim_type: ClaimType | None = None
    source_system: str = ""
    source_record: str = ""
    source_timestamp: str | _datetime | None = None
    retrieval_timestamp: str | _datetime | None = None
    calculation_method: str = ""
    agent_id: str = ""
    model_version: str = ""
    prompt_version: str = ""
    confidence: float | None = None
    validator_status: ValidatorStatus = ValidatorStatus.UNVERIFIED
    policy_status: str = ""
    definition_ref: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.client_id.strip() or not self.claim.strip():
            raise ValueError("C1-INVALID: claim id, client id and content are required")
        if self.claim_type is not None:
            object.__setattr__(self, "claim_type", ClaimType(self.claim_type))
        object.__setattr__(self, "validator_status", ValidatorStatus(self.validator_status))
        if self.definition_ref is not None:
            if len(self.definition_ref) != 2 or any(not isinstance(v, str) or not v.strip() for v in self.definition_ref):
                raise ValueError("C1-INVALID: definition_ref requires name and version")
            object.__setattr__(self, "definition_ref", tuple(self.definition_ref))


def _events(session):
    _verify_chain(session)
    return session.exec(_select(_EvidenceEvent).order_by(_EvidenceEvent.sequence)).all()


def _read(session, event):
    digest = event.payload.get("claim_hash", "")
    body = _RetentionStore().read(session, digest)
    if body is None or _hash_text(body) != digest:
        raise ValueError("C1-CONTENT: claim content missing or changed")
    return Claim(**_json.loads(body))


_SNAPSHOT_ACTIONS = ("record", "relabel", "release", "invalidate")


def _latest_events(session, interaction_id=None):
    events = _events(session)
    if interaction_id is not None:
        from kognita.reconstruct import _events_for
        events = _events_for(session, interaction_id, events)
    latest = {}
    for event in events:
        payload = event.payload or {}
        action = payload.get("action")
        verified = action == "verify" and payload.get("allowed") is True
        if (
            event.event_type in (_EventType.CLAIM, _EventType.CLAIM_VERIFIED)
            and payload.get("claim_id")
            and payload.get("claim_hash")
            and (action in _SNAPSHOT_ACTIONS or verified)
        ):
            latest[payload["claim_id"]] = event
    return latest


def list_claims(
    session: _Session,
    *,
    client_id: str | None = None,
    interaction_id: str | None = None,
) -> list[Claim]:
    """Latest recorded snapshots, optionally scoped to a client or journey."""
    claims = [_read(session, event) for event in _latest_events(session, interaction_id).values()]
    return [c for c in claims if client_id is None or c.client_id == client_id]


def _invalidate(session, claim, evidence):
    """Retain a stale snapshot. A claim that is already stale stays stale."""
    if claim.validator_status == ValidatorStatus.STALE:
        return claim
    event = _latest_events(session).get(claim.claim_id)
    if event is None:
        raise ValueError("C1-UNKNOWN: claim is not recorded")
    return _write(session, _replace(claim, validator_status=ValidatorStatus.STALE), evidence, event.correlation_id, "invalidate")


def _get(session, claim_id):
    for claim in list_claims(session):
        if claim.claim_id == claim_id:
            return claim
    raise ValueError("C1-UNKNOWN: claim is not recorded")


def _write(session, claim, evidence, interaction_id, action):
    digest = _RetentionStore().retain_value(session, _asdict(claim), kind="claim", correlation_id=interaction_id)
    evidence.emit(session, correlation_id=interaction_id,
                  event_type=_EventType.CLAIM_VERIFIED if action == "relabel" and claim.validator_status == ValidatorStatus.VERIFIED else _EventType.CLAIM,
                  payload={"claim_id": claim.claim_id, "claim_hash": digest, "action": action})
    return claim


def record_claim(
    session: _Session,
    claim: Claim,
    *,
    evidence: _EvidenceWriter,
    interaction_id: str,
) -> Claim:
    """Retain a new claim and append its hash in the caller's transaction."""
    if any(c.claim_id == claim.claim_id for c in list_claims(session)):
        raise ValueError("C1-DUPLICATE: use relabel_claim for an existing claim")
    return _write(session, claim, evidence, interaction_id, "record")


def release_claim(
    session: _Session,
    claim_id: str,
    *,
    evidence: _EvidenceWriter,
    interaction_id: str,
    audience: str = "rm",
    usecase_id: str = "",
    authoritative_figure: bool = False,
) -> _Check:
    """Check the stored label, validator status, source, definition, and evidence an RM or client release.

    ``authoritative_figure`` marks a figure the caller is presenting. A number
    and a figure keyword in the stored claim text mark one as well.
    """
    if type(authoritative_figure) is not bool:
        raise ValueError("C6-INVALID: authoritative_figure must be a boolean")
    claim = _get(session, claim_id)
    citation = "Claims: every released claim must be typed and cite a current approved source"
    if claim.validator_status == ValidatorStatus.STALE:
        reason = "C1-STALE"
    elif audience not in ("rm", "client"):
        reason = "C1-AUDIENCE"
    elif claim.claim_type is None:
        reason = "C1-UNLABELLED"
    else:
        reason = "C1-LABELLED"
        from kognita.sources import check_grounding
        grounding = check_grounding(session, claim, usecase_id=usecase_id,
                                    evidence=evidence, interaction_id=interaction_id)
        if grounding.result != _CheckResult.PASS:
            reason = grounding.check
    if reason == "C1-LABELLED":
        from kognita.calculations import check_calculation
        calculation = check_calculation(
            session, claim, authoritative_figure=authoritative_figure,
            evidence=evidence, interaction_id=interaction_id,
        )
        if calculation.result != _CheckResult.PASS:
            reason = calculation.check
            citation = calculation.citation
    if reason == "C1-LABELLED":
        from kognita.verification import _release_reason
        denial = _release_reason(session, claim, audience=audience, usecase_id=usecase_id)
        if denial:
            reason = denial
    definition = None
    if reason == "C1-LABELLED":
        from kognita.definitions import check_definition
        definition = check_definition(session, claim, evidence=evidence, interaction_id=interaction_id)
        if definition.result != _CheckResult.PASS:
            reason = definition.check
    passed = reason == "C1-LABELLED"
    if definition is not None and definition.result != _CheckResult.PASS:
        check = definition
    else:
        check = _Check(check=reason, regime="INTERNAL", result=_CheckResult.PASS if passed else _CheckResult.FAIL, citation=citation)
    digest = _RetentionStore().retain_value(session, _asdict(claim), kind="claim", correlation_id=interaction_id)
    evidence.emit(session, correlation_id=interaction_id, event_type=_EventType.CLAIM,
                  payload={"action": "release", "claim_id": claim_id, "claim_hash": digest, "audience": audience, "usecase_id": usecase_id, "authoritative_figure": authoritative_figure, "allowed": passed, "reason": reason})
    return check


def relabel_claim(
    session: _Session,
    claim_id: str,
    claim_type: ClaimType,
    *,
    evidence: _EvidenceWriter,
    interaction_id: str,
    source_system: str | None = None,
    source_record: str | None = None,
    validator_status: ValidatorStatus | None = None,
) -> Claim:
    """Append a successor label; promotion to fact needs source and verification."""
    old = _get(session, claim_id)
    next_status = old.validator_status if validator_status is None else ValidatorStatus(validator_status)
    if old.validator_status == ValidatorStatus.STALE and next_status != ValidatorStatus.STALE:
        raise ValueError("C1-STALE: a stale claim cannot be relabelled into circulation")
    new = _replace(old, claim_type=ClaimType(claim_type),
                   source_system=old.source_system if source_system is None else source_system,
                   source_record=old.source_record if source_record is None else source_record,
                   validator_status=old.validator_status if validator_status is None else validator_status)
    if new.claim_type == ClaimType.VERIFIED_FACT and (
        not new.source_system.strip()
        or not new.source_record.strip()
        or new.validator_status != ValidatorStatus.VERIFIED
    ):
        raise ValueError("C1-RELABEL: a fact requires a cited source and VERIFIED status")
    return _write(session, new, evidence, interaction_id, "relabel")


# Keys that name a transport field, not a statement an RM or client could rely on.
_STRUCTURAL_KEYS = frozenset({
    "model",
    "id",
    "object",
    "role",
    "finish_reason",
    "system_fingerprint",
    "type",
    "status",
    "index",
    "logprobs",
    "created",
})
_RELEASE_CITATION = (
    "Claims: material content is released only with a recorded claim that was released"
)


def _material_texts(payload: object) -> list[str]:
    """Prose values in a model response. Structural fields are not claims."""
    found: list[str] = []

    def walk(value: object, key: str | None) -> None:
        if isinstance(value, str):
            if key not in _STRUCTURAL_KEYS and value.strip():
                found.append(value.strip())
            return
        if isinstance(value, dict):
            for name, item in value.items():
                walk(item, name if isinstance(name, str) else None)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item, key)

    if isinstance(payload, str):
        if payload.strip():
            found.append(payload.strip())
        return found
    walk(payload, None)
    return found


def _release_covers(session: _Session, content: str, *, audience: str) -> _Check:
    """Fail closed unless this text has a recorded claim released to ``audience``.

    A blank body is not a claim. A recorded claim that was not released, a
    release for a different audience, or a release whose retained text differs
    does not cover the content.
    """
    def finish(reason: str, passed: bool) -> _Check:
        return _Check(
            check=reason,
            regime="INTERNAL",
            result=_CheckResult.PASS if passed else _CheckResult.FAIL,
            citation=_RELEASE_CITATION,
        )

    if audience not in ("rm", "client"):
        return finish("C1-AUDIENCE", False)
    if not isinstance(content, str) or not content.strip():
        return finish("C1-LABELLED", True)
    wanted = content.strip()
    # Only claim events are read, and the chain is not re-verified here: this
    # runs on every material response, and chain verification belongs to
    # reconstruct, replay and `evidence verify`. A forged event still fails
    # because the retained claim must exist and hash to the recorded digest.
    try:
        events = session.exec(
            _select(_EvidenceEvent)
            .where(_EvidenceEvent.event_type.in_((_EventType.CLAIM, _EventType.CLAIM_VERIFIED)))
            .order_by(_EvidenceEvent.sequence)
        ).all()
    except Exception:
        return finish("C1-UNKNOWN", False)
    recorded: set[str] = set()
    latest_release: dict[str, _EvidenceEvent] = {}
    latest_snapshot: dict[str, _EvidenceEvent] = {}
    for event in events:
        payload = event.payload or {}
        claim_id = payload.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            continue
        action = payload.get("action")
        if action == "record":
            recorded.add(claim_id)
        if action == "release" and payload.get("audience") == audience:
            latest_release[claim_id] = event
        if payload.get("claim_hash") and (
            action in _SNAPSHOT_ACTIONS or (action == "verify" and payload.get("allowed") is True)
        ):
            latest_snapshot[claim_id] = event
    for claim_id, event in latest_release.items():
        payload = event.payload or {}
        if claim_id not in recorded:
            continue
        if payload.get("allowed") is not True or payload.get("reason") != "C1-LABELLED":
            continue
        try:
            released = _read(session, event)
            current = _read(session, latest_snapshot.get(claim_id, event))
        except (ValueError, TypeError, KeyError):
            continue
        # A claim invalidated or blocked after its release no longer covers text.
        if current.validator_status in (ValidatorStatus.STALE, ValidatorStatus.BLOCKED):
            continue
        if released.claim_type is not None and released.claim.strip() == wanted:
            return finish("C1-LABELLED", True)
    return finish("C1-UNKNOWN", False)


__all__ = ["Claim", "ClaimType", "ValidatorStatus", "record_claim", "release_claim", "relabel_claim", "list_claims"]
