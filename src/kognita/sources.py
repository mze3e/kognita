"""Approved source snapshots and deterministic claim grounding."""
from __future__ import annotations

import json as _json
from dataclasses import dataclass as _dataclass, asdict as _asdict
from datetime import datetime as _datetime, timezone as _timezone

from kognita.canonical import hash_text as _hash_text
from kognita.envelope import Check as _Check
from kognita.evidence import verify_chain as _verify_chain
from kognita.models import EvidenceEvent as _EvidenceEvent
from kognita.retention import RetentionStore as _RetentionStore
from kognita.vocabulary import CheckResult as _CheckResult, EventType as _EventType
from sqlmodel import select as _select


def _time(value):
    moment = _datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(moment, _datetime) or moment.tzinfo is None:
        raise ValueError("C3-INVALID: source dates require a timezone")
    return moment.astimezone(_timezone.utc)


@_dataclass(frozen=True)
class Source:
    source_id: str
    kind: str
    version: str
    owner: str
    effective: _datetime | str
    superseded_by: str = ""
    expiry: _datetime | str | None = None

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.source_id, self.version, self.owner)):
            raise ValueError("C3-INVALID: source id, version and owner are required")
        if self.source_id.strip().casefold() == "model" or self.kind not in (
            "CIO research", "house view", "product documentation", "market data", "policy"
        ) or not isinstance(self.superseded_by, str):
            raise ValueError("C3-INVALID: invalid source kind or identity")
        object.__setattr__(self, "effective", _time(self.effective))
        if self.expiry is not None:
            object.__setattr__(self, "expiry", _time(self.expiry))
            if self.expiry <= self.effective:
                raise ValueError("C3-INVALID: expiry must follow effective date")


@_dataclass(frozen=True)
class SourceRegister:
    entries: tuple[Source, ...] = ()

    def get(self, source_id: str, version: str) -> Source | None:
        return next((s for s in self.entries if (s.source_id, s.version) == (source_id, version)), None)


def _load(session):
    _verify_chain(session)
    latest = {}
    for event in session.exec(_select(_EvidenceEvent).order_by(_EvidenceEvent.sequence)):
        if event.payload.get("action") != "register_source":
            continue
        digest = event.payload["source_hash"]
        body = _RetentionStore().read(session, digest)
        if body is None or _hash_text(body) != digest:
            raise ValueError("C3-CONTENT: source content missing or changed")
        source = Source(**_json.loads(body))
        latest[source.source_id, source.version] = source
    return SourceRegister(tuple(latest.values()))


def register_source(session, source: Source, *, evidence, interaction_id: str) -> Source:
    """Retain an approved source version, or append its lifecycle update."""
    register = _load(session)
    old = register.get(source.source_id, source.version)
    if old is not None and old != source:
        # A lifecycle update may only tighten an existing version.
        if (old.kind, old.owner, old.effective) != (source.kind, source.owner, source.effective) or (
            old.superseded_by and old.superseded_by != source.superseded_by
        ) or (old.expiry is not None and (source.expiry is None or source.expiry > old.expiry)):
            raise ValueError("C3-IMMUTABLE: source versions cannot be rewritten or revived")
    digest = _RetentionStore().retain_value(session, _asdict(source), kind="source", correlation_id=interaction_id)
    evidence.emit(session, correlation_id=interaction_id, event_type=_EventType.CLAIM,
                  payload={"action": "register_source", "source_hash": digest})
    return source


def check_grounding(session, claim, *, usecase_id: str, as_of=None, evidence=None, interaction_id: str = "") -> _Check:
    """Require an exact, live source version approved by the stored use case."""
    from kognita.usecases import load_register as _load_usecases
    moment = _time(as_of or _datetime.now(_timezone.utc))
    entry = _load_usecases(session, as_of=moment).get(usecase_id)
    source = _load(session).get(claim.source_system, claim.source_record)
    reason = "C3-GROUNDED"
    if entry is None:
        reason = "C3-USECASE"
    elif not claim.source_system.strip() or claim.source_system.strip().casefold() == "model":
        reason = "C3-MODEL"
    elif source is None or source.source_id not in entry.approved_sources:
        reason = "C3-UNAPPROVED"
    elif source.superseded_by:
        reason = "C3-SUPERSEDED"
    elif source.effective > moment:
        reason = "C3-NOT-EFFECTIVE"
    elif source.expiry is not None and source.expiry <= moment:
        reason = "C3-EXPIRED"
    if evidence is not None:
        digest = _RetentionStore().retain_value(
            session, {"source": _asdict(source) if source else None,
                      "usecase_id": usecase_id,
                      "usecase_hash": entry.content_hash if entry else "",
                      "approved_sources": list(entry.approved_sources) if entry else [],
                      "at": moment, "reason": reason},
            kind="source", correlation_id=interaction_id)
        evidence.emit(session, correlation_id=interaction_id, event_type=_EventType.CLAIM,
                      payload={"action": "check_grounding", "claim_id": claim.claim_id,
                               "grounding_hash": digest, "reason": reason})
    return _Check(check=reason, regime="INTERNAL", result=_CheckResult.PASS if reason == "C3-GROUNDED" else _CheckResult.FAIL,
                  citation="Claims: released claims require a current source approved for the use case")


__all__ = ["Source", "SourceRegister", "register_source", "check_grounding"]
