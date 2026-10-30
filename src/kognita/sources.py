"""Approved source snapshots and deterministic claim grounding."""
from __future__ import annotations

import json as _json
from dataclasses import dataclass as _dataclass, asdict as _asdict, replace as _replace
from datetime import datetime as _datetime, timezone as _timezone

from kognita.canonical import canonical_json as _canonical_json, hash_text as _hash_text
from kognita.envelope import Check as _Check
from kognita.evidence import ChainBreak as _ChainBreak, verify_chain as _verify_chain
from kognita.models import (
    EvidenceEvent as _EvidenceEvent,
    Interaction as _Interaction,
    RmReview as _RmReview,
    as_utc as _as_utc,
)
from kognita.retention import RetentionStore as _RetentionStore
from kognita.vocabulary import (
    ActorType as _ActorType,
    CheckResult as _CheckResult,
    EventType as _EventType,
)
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


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _required(value, message):
    text = _text(value)
    if not text:
        raise ValueError(message)
    return text


def _payload(event):
    body = event.payload
    return body if isinstance(body, dict) else {}


def _events(session):
    return list(session.exec(_select(_EvidenceEvent).order_by(_EvidenceEvent.sequence)))


def _find(register, system, record):
    system, record = _text(system), _text(record)
    found = register.get(system, record)
    if found is not None:
        return found
    return next((item for item in register.entries if _text(item.source_id) == system and _text(item.version) == record), None)


def _depends(system, record, source_id, version):
    """A blank version cites the source without naming a live snapshot."""
    return _text(system) == source_id and (_text(record) in ("", version))


def _cites_stale(register, source_id, system, record):
    """True when this citation is a superseded or unknown version of source_id."""
    if _text(system) != source_id:
        return False
    record = _text(record)
    if not record:
        return any(_text(item.source_id) == source_id and item.superseded_by for item in register.entries)
    source = _find(register, source_id, record)
    if source is None:
        return True
    return bool(source.superseded_by)


def _explicit(body, source_id, version):
    if "source_system" not in body and "source_id" not in body:
        return False
    system = body.get("source_system", body.get("source_id"))
    record = body.get("source_record", body.get("version", ""))
    return _depends(system, record, source_id, version)


def _explicit_stale(register, body):
    if "source_system" not in body and "source_id" not in body:
        return False
    system = _text(body.get("source_system", body.get("source_id")))
    record = body.get("source_record", body.get("version", ""))
    return bool(system) and _cites_stale(register, system, system, record)


def _was_sent(body):
    if body.get("lane") == "public":
        return False
    if body.get("sent") is True:
        return True
    if body.get("sent") is False:
        return False
    return str(body.get("outcome") or "").upper() == "ALLOW"


def _named(event, body):
    interaction_id = body.get("interaction_id")
    if isinstance(interaction_id, str) and interaction_id.strip():
        return interaction_id.strip()
    return event.correlation_id if isinstance(event.correlation_id, str) else ""


def _touches(event, body, interaction_id):
    return _named(event, body) == interaction_id or event.correlation_id == interaction_id


def _open(events, interaction_id):
    outcome = ""
    sent = False
    for event in events:
        body = _payload(event)
        if not _touches(event, body, interaction_id):
            continue
        if event.event_type == _EventType.FINAL_DECISION and body.get("interaction_id") in (None, "", interaction_id):
            outcome = str(body.get("outcome") or "")
        if event.event_type == _EventType.CLIENT_COMMUNICATION and _was_sent(body):
            sent = True
    return not sent and outcome not in ("reject", "deny")


def _links(events):
    found = {}
    for event in events:
        if event.event_type not in (_EventType.CLAIM, _EventType.CLAIM_VERIFIED):
            continue
        body = _payload(event)
        claim_id = body.get("claim_id")
        if body.get("action") in ("record", "relabel", "release", "invalidate") and isinstance(claim_id, str) and claim_id:
            found.setdefault(claim_id, set()).add(event.correlation_id)
    return found


def _client_id(session, interaction_id, fallback=""):
    row = session.get(_Interaction, interaction_id) if interaction_id else None
    if row is not None and isinstance(row.client_id, str) and row.client_id.strip():
        return row.client_id
    return fallback if isinstance(fallback, str) else ""


def _moment(value):
    if not isinstance(value, _datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=_timezone.utc)
    return value.astimezone(_timezone.utc)


def _window(start, end):
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError("C3-WINDOW: --from and --to are both required when either is set")
    opened, closed = _moment(start), _moment(end)
    if opened is None or closed is None:
        raise ValueError("C3-WINDOW: --from and --to must be timestamps")
    if closed <= opened:
        raise ValueError("C3-WINDOW: --to is after --from")
    return opened, closed


def _within(moment, window):
    if window is None:
        return True
    if moment is None:
        return True
    opened, closed = window
    return opened <= moment < closed


def _iso(moment):
    stamped = _moment(moment)
    return stamped.isoformat() if stamped is not None else ""


def _open_recommendations(session, events, source_id, version, claim_ids):
    """Open recommendations and reviews that still depend on this version."""
    links = _links(events)
    interactions = set()
    for claim_id in claim_ids:
        interactions.update(links.get(claim_id, ()))
    found, seen = [], set()
    for event in events:
        if event.event_type != _EventType.ORIGINATION:
            continue
        body = _payload(event)
        if body.get("outcome") != "recommend":
            continue
        interaction_id = _named(event, body)
        if not interaction_id:
            continue
        if interaction_id not in interactions and not _explicit(body, source_id, version):
            continue
        if not _open(events, interaction_id) or interaction_id in seen:
            continue
        seen.add(interaction_id)
        found.append({
            "interaction_id": interaction_id,
            "client_id": _client_id(session, interaction_id, _text(body.get("client_id"))),
            "validator_status": "STALE",
        })
    for row in session.exec(_select(_RmReview)):
        if not row.interaction_id or row.interaction_id in seen or row.interaction_id not in interactions:
            continue
        if not _open(events, row.interaction_id):
            continue
        seen.add(row.interaction_id)
        found.append({
            "interaction_id": row.interaction_id,
            "client_id": _client_id(session, row.interaction_id),
            "validator_status": "STALE",
        })
    found.sort(key=lambda item: item["interaction_id"])
    return found


def supersede_source(session, source_id, *, version, by, evidence, interaction_id):
    """Supersede one source version and invalidate what still depends on it.

    Records ``SOURCE_SUPERSEDED``. Every claim citing that version, including a
    citation that names the source and no version, is retained again with
    ``validator_status`` STALE. Every open recommendation on those interactions
    is invalidated the same way. Release of a stale claim fails.
    """
    source_id = _required(source_id, "C3-INVALID: source id is required")
    version = _required(version, "C3-INVALID: source version is required")
    by = _required(by, "C3-INVALID: superseded_by must name a different version")
    interaction_id = _required(interaction_id, "C3-INVALID: interaction id is required")
    if by.casefold() == "model":
        raise ValueError("C3-INVALID: model knowledge cannot supersede a source")
    if by == version:
        raise ValueError("C3-INVALID: superseded_by must name a different version")
    current = _load(session).get(source_id, version)
    if current is None:
        current = _find(_load(session), source_id, version)
    if current is None:
        raise ValueError("C3-UNKNOWN: source version is not registered")
    if current.superseded_by:
        raise ValueError("C3-IMMUTABLE: source version is already superseded")
    updated = _replace(current, superseded_by=by)
    register_source(session, updated, evidence=evidence, interaction_id=interaction_id)
    from kognita.claims import _invalidate, list_claims
    stale_claims = []
    for claim in sorted(list_claims(session), key=lambda item: item.claim_id):
        if _depends(claim.source_system, claim.source_record, source_id, version):
            _invalidate(session, claim, evidence)
            stale_claims.append(claim.claim_id)
    recommendations = _open_recommendations(session, _events(session), source_id, version, stale_claims)
    evidence.emit(
        session,
        correlation_id=interaction_id,
        event_type=_EventType.SOURCE_SUPERSEDED,
        actor_type=_ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload={
            "action": "supersede",
            "source_id": source_id,
            "version": version,
            "superseded_by": by,
            "source_hash": _hash_text(_canonical_json(_asdict(updated))),
            "claims": stale_claims,
            "recommendations": recommendations,
        },
    )
    return updated


def _distribution_block(session, interaction_id):
    """A reason code when this journey must not be approved or sent, else None."""
    if not isinstance(interaction_id, str) or not interaction_id.strip():
        return "C1-STALE"
    interaction_id = interaction_id.strip()
    events = _events(session)
    interesting = False
    for event in events:
        body = _payload(event)
        if event.event_type == _EventType.SOURCE_SUPERSEDED:
            interesting = True
            break
        if event.event_type in (_EventType.CLAIM, _EventType.CLAIM_VERIFIED) and body.get("action") in (
            "record", "relabel", "release", "invalidate"
        ):
            interesting = True
            break
    if not interesting:
        return None
    try:
        from kognita.claims import ValidatorStatus, _latest_events, _read
        register = _load(session)
        links = _links(events)
        for claim_id, event in _latest_events(session).items():
            if interaction_id not in links.get(claim_id, ()):
                continue
            claim = _read(session, event)
            if claim.validator_status == ValidatorStatus.STALE:
                return "C1-STALE"
            source = _find(register, claim.source_system, claim.source_record)
            if source is not None and source.superseded_by:
                return "C3-SUPERSEDED"
        for event in events:
            body = _payload(event)
            if event.event_type == _EventType.SOURCE_SUPERSEDED:
                for item in body.get("recommendations") or []:
                    if isinstance(item, dict) and item.get("interaction_id") == interaction_id:
                        return "C1-STALE"
            if _touches(event, body, interaction_id) and event.event_type == _EventType.ORIGINATION and _explicit_stale(register, body):
                return "C3-SUPERSEDED"
    except (ValueError, _ChainBreak):
        return "C1-STALE"
    return None


def affected_by_source(session, source_id, *, start=None, end=None):
    """Clients, previous recommendations, previous communications and open opportunities.

    Only a superseded or unknown version of ``source_id`` contributes. A live
    version does not. ``start`` and ``end`` form an optional half-open window.
    A source that was never registered is an error, not an empty result.
    """
    source_id = _required(source_id, "C3-INVALID: source id is required")
    window = _window(start, end)
    register = _load(session)
    if not any(_text(item.source_id) == source_id for item in register.entries):
        raise ValueError("C3-UNKNOWN: source is not registered")
    from kognita.claims import _latest_events, list_claims
    events = _events(session)
    links = _links(events)
    latest = _latest_events(session)
    claims = [claim for claim in list_claims(session) if _cites_stale(register, source_id, claim.source_system, claim.source_record)]
    interactions = set()
    for claim in claims:
        interactions.update(links.get(claim.claim_id, ()))
    for event in events:
        body = _payload(event)
        if event.event_type == _EventType.SOURCE_SUPERSEDED and _text(body.get("source_id")) == source_id:
            for item in body.get("recommendations") or []:
                if isinstance(item, dict) and isinstance(item.get("interaction_id"), str) and item["interaction_id"]:
                    interactions.add(item["interaction_id"])
        elif _explicit_stale(register, body):
            named = _named(event, body)
            if named:
                interactions.add(named)
    clients, recommendations, communications, opportunities = [], [], [], []
    for claim in claims:
        event = latest.get(claim.claim_id)
        moment = _as_utc(event.recorded_at) if event is not None else None
        if not _within(_moment(moment), window):
            continue
        for correlation in sorted(links.get(claim.claim_id, ())):
            clients.append({"client_id": claim.client_id, "interaction_id": correlation})
    seen_open = set()
    for event in events:
        if event.event_type != _EventType.ORIGINATION:
            continue
        body = _payload(event)
        if body.get("outcome") != "recommend":
            continue
        interaction_id = _named(event, body)
        if interaction_id not in interactions and not _explicit_stale(register, body):
            continue
        moment = _moment(_as_utc(event.recorded_at))
        if not _within(moment, window):
            continue
        client_id = _client_id(session, interaction_id, _text(body.get("client_id")))
        recommendations.append({
            "interaction_id": interaction_id,
            "client_id": client_id,
            "sequence": event.sequence,
            "product_id": _text(body.get("product_id")),
            "recorded_at": _iso(moment),
            "validator_status": "STALE",
        })
        clients.append({"client_id": client_id, "interaction_id": interaction_id})
        if _open(events, interaction_id) and interaction_id not in seen_open:
            seen_open.add(interaction_id)
            opportunities.append({
                "interaction_id": interaction_id,
                "client_id": client_id,
                "product_id": _text(body.get("product_id")),
                "recorded_at": _iso(moment),
                "validator_status": "STALE",
            })
    for row in session.exec(_select(_RmReview).order_by(_RmReview.id)):
        if not row.interaction_id or row.interaction_id not in interactions:
            continue
        moment = _moment(_as_utc(row.requested_at))
        if not _within(moment, window):
            continue
        client_id = _client_id(session, row.interaction_id)
        recommendations.append({
            "interaction_id": row.interaction_id,
            "client_id": client_id,
            "sequence": row.id or 0,
            "product_id": "",
            "recorded_at": _iso(moment),
            "validator_status": "STALE",
            "outcome": row.outcome,
        })
        clients.append({"client_id": client_id, "interaction_id": row.interaction_id})
        if _open(events, row.interaction_id) and row.interaction_id not in seen_open:
            seen_open.add(row.interaction_id)
            opportunities.append({
                "interaction_id": row.interaction_id,
                "client_id": client_id,
                "product_id": "",
                "recorded_at": _iso(moment),
                "validator_status": "STALE",
            })
    for event in events:
        if event.event_type != _EventType.CLIENT_COMMUNICATION:
            continue
        body = _payload(event)
        if not _was_sent(body):
            continue
        interaction_id = _named(event, body)
        if interaction_id not in interactions and not _explicit_stale(register, body):
            continue
        moment = _moment(_as_utc(event.recorded_at))
        if not _within(moment, window):
            continue
        client_id = _client_id(session, interaction_id, _text(body.get("client_id")))
        communications.append({
            "interaction_id": interaction_id,
            "client_id": client_id,
            "sequence": event.sequence,
            "recorded_at": _iso(moment),
            "channel": _text(body.get("channel")),
            "sent": True,
        })
        clients.append({"client_id": client_id, "interaction_id": interaction_id})

    def _unique(rows, *keys):
        seen, result = set(), []
        for row in rows:
            key = tuple(row.get(key) for key in keys)
            if key in seen or not any(key):
                continue
            seen.add(key)
            result.append(row)
        result.sort(key=lambda row: tuple(str(row.get(key) or "") for key in keys))
        return result

    opened = window[0] if window else None
    closed = window[1] if window else None
    return {
        "source_id": source_id,
        "from": _iso(opened) if opened else "",
        "to": _iso(closed) if closed else "",
        "clients": _unique(clients, "client_id", "interaction_id"),
        "recommendations": _unique(recommendations, "interaction_id", "sequence"),
        "communications": _unique(communications, "interaction_id", "sequence"),
        "opportunities": _unique(opportunities, "interaction_id"),
        "citation": (
            "Claims: superseding a source lists affected clients, previous "
            "recommendations, previous communications and outstanding opportunities"
        ),
    }


__all__ = [
    "Source",
    "SourceRegister",
    "affected_by_source",
    "check_grounding",
    "register_source",
    "supersede_source",
]
