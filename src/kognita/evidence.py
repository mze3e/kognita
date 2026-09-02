"""The evidence plane — append-only, hash-chained, infrastructure-owned.

Two properties make this worth having rather than just logging:

**Tamper-evidence.** Each row carries the hash of the row before it, so altering
any payload breaks every hash after it. :func:`verify_chain` says precisely where.

**Erasure-compatibility.** Payloads default to hashes and references rather than
content. An append-only log holding personal data collides with erasure rights,
so copying content in is opt-in per event — and a ``redact_payload`` hook lets a
deployment strip fields on the way in.

Evidence is written by the infrastructure, never by an agent: nothing in this
module takes a caller-supplied hash or sequence number.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any
from weakref import WeakKeyDictionary

from sqlalchemy.engine import Engine
from sqlmodel import Session, desc, select

from kognita.canonical import canonical_hash, canonical_json
from kognita.models import EvidenceEvent, as_utc, utcnow
from kognita.vocabulary import ActorType, Classification, EventType

#: Hook applied to every payload before it is hashed and stored. Return the
#: payload to keep, or a reduced version. The default keeps payloads as given —
#: callers decide what to put in them.
PayloadFilter = Callable[[EventType, dict[str, Any]], dict[str, Any]]

#: The prev_hash of the first event in a chain.
GENESIS_HASH = "0" * 64


#: One lock per engine, shared across writers. Two ``EvidenceWriter`` objects
#: over the same store are ordinary — one with a redaction hook, one without —
#: and a per-instance lock would let them interleave and fork the chain.
_ENGINE_LOCKS: "WeakKeyDictionary[Engine, threading.Lock]" = WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()


def _lock_for(engine: Engine) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _ENGINE_LOCKS.get(engine)
        if lock is None:
            lock = threading.Lock()
            _ENGINE_LOCKS[engine] = lock
        return lock


class EvidenceWriter:
    """Appends events to the chain for one database.

    The sequence number and both hashes are assigned under a lock shared by every
    writer over the same engine, so concurrent writers cannot fork the chain.

    That lock guards writers *in one process*. Across processes the store
    assumes a single writer; if that assumption is broken, the UNIQUE constraint
    on ``sequence`` rejects the second insert rather than letting the chain fork
    silently, and ``verify_chain`` catches anything that slips past.
    """

    def __init__(
        self,
        engine: Engine,
        *,
        redact_payload: PayloadFilter | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.engine = engine
        self.redact_payload = redact_payload
        self.clock = clock
        self._lock = _lock_for(engine)

    def emit(
        self,
        session: Session,
        *,
        correlation_id: str,
        event_type: EventType,
        actor_type: ActorType = ActorType.SYSTEM,
        actor_id: str = "system",
        classification: Classification = Classification.C1,
        payload: dict[str, Any] | None = None,
    ) -> EvidenceEvent:
        """Append one event and return it, with its hashes assigned.

        Takes the caller's ``session`` so an evidence write lands in the same
        transaction as the thing it evidences: a decision that is rolled back
        must not leave a record claiming it happened.
        """
        body = dict(payload or {})
        if self.redact_payload is not None:
            body = self.redact_payload(event_type, body)

        with self._lock:
            previous = session.exec(
                select(EvidenceEvent).order_by(desc(EvidenceEvent.sequence)).limit(1)
            ).first()
            sequence = (previous.sequence + 1) if previous else 1
            prev_hash = previous.event_hash if previous else GENESIS_HASH

            payload_hash = canonical_hash(body)
            header = {
                "sequence": sequence,
                "correlation_id": correlation_id,
                "event_type": event_type.value,
                "actor_type": actor_type.value,
                "actor_id": actor_id,
                "classification": classification.value,
                "payload_hash": payload_hash,
                "prev_hash": prev_hash,
            }
            event = EvidenceEvent(
                sequence=sequence,
                correlation_id=correlation_id,
                event_type=event_type,
                actor_type=actor_type,
                actor_id=actor_id,
                classification=classification,
                payload=body,
                payload_hash=payload_hash,
                prev_hash=prev_hash,
                event_hash=canonical_hash(header),
                recorded_at=self.clock(),
            )
            session.add(event)
            session.flush()
        return event


def recompute_event_hash(event: EvidenceEvent) -> str:
    """Recompute an event's hash from its stored fields."""
    return canonical_hash(
        {
            "sequence": event.sequence,
            "correlation_id": event.correlation_id,
            "event_type": EventType(event.event_type).value,
            "actor_type": ActorType(event.actor_type).value,
            "actor_id": event.actor_id,
            "classification": Classification(event.classification).value,
            "payload_hash": event.payload_hash,
            "prev_hash": event.prev_hash,
        }
    )


class ChainBreak(Exception):
    """The evidence chain does not verify."""

    def __init__(self, sequence: int, reason: str) -> None:
        super().__init__(f"evidence chain broken at sequence {sequence}: {reason}")
        self.sequence = sequence
        self.reason = reason


def verify_chain(session: Session) -> int:
    """Verify every event in order; return how many were checked.

    Raises :class:`ChainBreak` naming the first sequence that fails and why —
    a rewritten payload, a recomputed hash that no longer matches, or a link
    that points at the wrong predecessor.
    """
    events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    expected_prev = GENESIS_HASH
    for index, event in enumerate(events, start=1):
        if event.sequence != index:
            raise ChainBreak(event.sequence, f"expected sequence {index}")
        if canonical_hash(event.payload) != event.payload_hash:
            raise ChainBreak(event.sequence, "payload does not match its hash")
        if event.prev_hash != expected_prev:
            raise ChainBreak(event.sequence, "prev_hash does not match the previous event")
        if recompute_event_hash(event) != event.event_hash:
            raise ChainBreak(event.sequence, "event hash does not match its contents")
        expected_prev = event.event_hash
    return len(events)


def export_chain(
    session: Session,
    *,
    since: datetime | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Export the chain as a portable, self-verifying artifact.

    The export always covers the chain from the beginning up to the last event in
    range, because a slice starting mid-chain cannot be verified on its own — the
    ``since`` and ``correlation_id`` filters mark which events are *of interest*
    rather than truncating the evidence that proves them.
    """
    events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()

    def of_interest(event: EvidenceEvent) -> bool:
        if since is not None:
            recorded = as_utc(event.recorded_at)
            if recorded is not None and recorded < since:
                return False
        if correlation_id is not None and event.correlation_id != correlation_id:
            return False
        return True

    rows = [
        {
            "sequence": e.sequence,
            "correlation_id": e.correlation_id,
            "event_type": EventType(e.event_type).value,
            "actor_type": ActorType(e.actor_type).value,
            "actor_id": e.actor_id,
            "classification": Classification(e.classification).value,
            "payload": e.payload,
            "payload_hash": e.payload_hash,
            "prev_hash": e.prev_hash,
            "event_hash": e.event_hash,
            "recorded_at": (as_utc(e.recorded_at) or utcnow()).isoformat(),
            "of_interest": of_interest(e),
        }
        for e in events
    ]
    return {
        "kognita_evidence_export": 1,
        "exported_at": utcnow().isoformat(),
        "event_count": len(rows),
        "interest_count": sum(1 for r in rows if r["of_interest"]),
        "filters": {
            "since": since.isoformat() if since else None,
            "correlation_id": correlation_id,
        },
        "head_hash": rows[-1]["event_hash"] if rows else GENESIS_HASH,
        "events": rows,
    }


def verify_export(export: dict[str, Any]) -> int:
    """Verify an exported artifact without a database. Returns events checked."""
    expected_prev = GENESIS_HASH
    rows = export.get("events", [])
    for index, row in enumerate(rows, start=1):
        if row["sequence"] != index:
            raise ChainBreak(row["sequence"], f"expected sequence {index}")
        if canonical_hash(row["payload"]) != row["payload_hash"]:
            raise ChainBreak(row["sequence"], "payload does not match its hash")
        if row["prev_hash"] != expected_prev:
            raise ChainBreak(row["sequence"], "prev_hash does not match the previous event")
        header = {
            "sequence": row["sequence"],
            "correlation_id": row["correlation_id"],
            "event_type": row["event_type"],
            "actor_type": row["actor_type"],
            "actor_id": row["actor_id"],
            "classification": row["classification"],
            "payload_hash": row["payload_hash"],
            "prev_hash": row["prev_hash"],
        }
        if canonical_hash(header) != row["event_hash"]:
            raise ChainBreak(row["sequence"], "event hash does not match its contents")
        expected_prev = row["event_hash"]
    return len(rows)


def hashes_only(_event_type: EventType, payload: dict[str, Any]) -> dict[str, Any]:
    """A ``redact_payload`` hook that replaces every value with its hash.

    The strictest setting: the log proves what happened and that the content was
    what it claims, while holding none of it.
    """
    return {
        key: {"sha256": canonical_hash(value), "bytes": len(canonical_json(value))}
        for key, value in payload.items()
    }


__all__ = [
    "EvidenceWriter",
    "ChainBreak",
    "verify_chain",
    "export_chain",
    "verify_export",
    "hashes_only",
    "recompute_event_hash",
    "GENESIS_HASH",
]
