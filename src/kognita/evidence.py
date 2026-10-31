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

The append-only guard and the ``evidence_head`` checkpoint (see
:mod:`kognita.db`) are installed on SQLite only; this project ships no other
dialect driver. Deploying this store on a dialect where ``create_all`` does
not install them leaves raw DELETE/UPDATE unguarded at the database layer.
:func:`verify_chain` still fails closed in that case — a missing or unreadable
head checkpoint is treated as a chain break whenever events exist, rather than
as "nothing to compare against" — so truncation is reported instead of passing
silently, but there is nothing stopping the truncation itself without the
SQLite triggers.
"""
from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any
from weakref import WeakKeyDictionary

from sqlalchemy.engine import Engine
from sqlmodel import Session, desc, select

from kognita.canonical import canonical_hash, canonical_json
from kognita.models import EvidenceCheck, EvidenceEvent, EvidenceItem, as_utc, utcnow
from kognita.vocabulary import ActorType, Classification, EventType

#: Hook applied to every payload before it is hashed and stored. Return the
#: payload to keep, or a reduced version. The default keeps payloads as given —
#: callers decide what to put in them.
PayloadFilter = Callable[[EventType, dict[str, Any]], dict[str, Any]]

#: Called with a copy of an event after that event is in the chain. A failure
#: here is recorded and does not change the event.
TelemetryHook = Callable[[dict[str, Any]], None]

#: The prev_hash of the first event in a chain.
GENESIS_HASH = "0" * 64


#: One lock per engine, shared across writers. Two ``EvidenceWriter`` objects
#: over the same store are ordinary — one with a redaction hook, one without —
#: and a per-instance lock would let them interleave and fork the chain.
_ENGINE_LOCKS: "WeakKeyDictionary[Engine, threading.Lock]" = WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()


def _int_ref(value: Any) -> int | None:
    """An integer row id. Booleans and redacted hashes are not ids."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _str_ref(value: Any) -> str | None:
    """A non-empty string id. A redacted ``{sha256, bytes}`` value is not one."""
    if isinstance(value, str) and value:
        return value
    return None


def _apply_payload_links(event: EvidenceEvent, payload: dict[str, Any]) -> None:
    """Copy the payload's single-valued row ids onto the foreign-key columns.

    The names are the payload keys. A missing key, or a value that is not an
    id, leaves the column null. SQLite checks the column on insert.
    """
    approval_id = _int_ref(payload.get("approval_id"))
    if approval_id is not None:
        event.approval_id = approval_id
    policy_id = _int_ref(payload.get("policy_id"))
    if policy_id is not None:
        event.policy_id = policy_id
    successor_id = _int_ref(payload.get("successor_id"))
    if successor_id is not None:
        event.successor_id = successor_id
    run_id = _str_ref(payload.get("run_id"))
    if run_id is not None:
        event.run_id = run_id
    continuation_hash = _str_ref(payload.get("continuation_hash"))
    if continuation_hash is not None:
        event.continuation_hash = continuation_hash


def _cite_payload_links(session: Session, event: EvidenceEvent, payload: dict[str, Any]) -> None:
    """One foreign-key row per repeated id in ``checks``, ``items``, and ``returned_ids``.

    Those lists can cite more than one row, so they are not columns on the
    event. Content hashes are left in the payload: erasure deletes the bytes
    and the chain keeps the hash.
    """
    if event.id is None:
        return
    cited = False
    seen_policies: set[int] = set()
    checks = payload.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            policy_id = _int_ref(check.get("policy_id"))
            if policy_id is None or policy_id in seen_policies:
                continue
            seen_policies.add(policy_id)
            session.add(EvidenceCheck(evidence_event_id=event.id, policy_id=policy_id))
            cited = True
    seen_items: set[int] = set()
    returned = payload.get("returned_ids")
    if isinstance(returned, list):
        for value in returned:
            item_id = _int_ref(value)
            if item_id is not None:
                seen_items.add(item_id)
    items = payload.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = _int_ref(item.get("id"))
            if item_id is not None:
                seen_items.add(item_id)
    for item_id in seen_items:
        session.add(EvidenceItem(evidence_event_id=event.id, item_id=item_id))
        cited = True
    if cited:
        session.flush()


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
        telemetry: TelemetryHook | None = None,
        signing=None,
    ) -> None:
        self._signing = signing
        self.engine = engine
        self.redact_payload = redact_payload
        self.clock = clock
        self.telemetry = telemetry
        #: Export failures, in order. The decision that was already appended is
        #: not in this list and is not rewritten when an export fails.
        self.telemetry_failures: list[str] = []
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

            recorded_at = self.clock()
            payload_hash = canonical_hash(body)
            header = {
                "schema_version": 2,
                "sequence": sequence,
                "correlation_id": correlation_id,
                "event_type": event_type.value,
                "actor_type": actor_type.value,
                "actor_id": actor_id,
                "classification": classification.value,
                "payload_hash": payload_hash,
                "prev_hash": prev_hash,
                "recorded_at": recorded_at.isoformat(),
            }
            signature = None
            if self._signing is not None:
                if previous is not None and previous.signature is None:
                    raise ValueError('T1: signing requires a wholly signed chain')
                signature = self._signing._sign(header)
                header['signature'] = signature
            elif previous is not None and previous.signature is not None:
                raise ValueError('T1: unsigned write would break a signed chain')
            event = EvidenceEvent(
                signature=signature,
                key_id=header.get('key_id'),
                schema_version=2,
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
                recorded_at=recorded_at,
            )
            _apply_payload_links(event, body)
            session.add(event)
            session.flush()
            _cite_payload_links(session, event, body)
        self._export_snapshot(event)
        return event

    def observe(self, snapshot: dict[str, Any]) -> None:
        """Hand a copy to the telemetry hook.

        The hook is not called when none is set. An exception is appended to
        :attr:`telemetry_failures` and written to stderr. It is not raised,
        and this method does not write evidence.
        """
        if self.telemetry is None:
            return
        try:
            self.telemetry(dict(snapshot))
        except Exception as exc:
            self._note_export_failure(f"{exc}")

    def _export_snapshot(self, event: EvidenceEvent) -> None:
        """Export a copy of ``event`` after it has been appended.

        The hook receives a plain dict, not the row. A failure leaves the row
        as it was flushed.
        """
        if self.telemetry is None:
            return
        try:
            recorded = as_utc(event.recorded_at)
            snapshot = {
                "kind": "evidence",
                "signature": event.signature,
                "key_id": event.key_id,
                "schema_version": event.schema_version,
                "sequence": event.sequence,
                "correlation_id": event.correlation_id,
                "event_type": EventType(event.event_type).value,
                "actor_type": ActorType(event.actor_type).value,
                "actor_id": event.actor_id,
                "classification": Classification(event.classification).value,
                "payload": json.loads(canonical_json(event.payload)),
                "event_hash": event.event_hash,
                "recorded_at": recorded.isoformat() if recorded else None,
            }
            self.telemetry(snapshot)
        except Exception as exc:
            self._note_export_failure(f"sequence {event.sequence}: {exc}")

    def _note_export_failure(self, message: str) -> None:
        text = f"OpenTelemetry export failed: {message}"
        self.telemetry_failures.append(text)
        sys.stderr.write(text + "\n")


def recompute_event_hash(event: EvidenceEvent) -> str:
    """Recompute an event's hash from its stored fields.

    Timestamps are included in the hash so they are tamper-evident:
    altering recorded_at breaks the hash and is caught by verify_chain.
    """
    return canonical_hash(_event_header(event))


def _event_header(event):
    recorded_at = as_utc(event.recorded_at)
    header = {
        "sequence": event.sequence,
        "correlation_id": event.correlation_id,
        "event_type": EventType(event.event_type).value,
        "actor_type": ActorType(event.actor_type).value,
        "actor_id": event.actor_id,
        "classification": Classification(event.classification).value,
        "payload_hash": event.payload_hash,
        "prev_hash": event.prev_hash,
        "recorded_at": recorded_at.isoformat() if recorded_at else None,
    }
    _version_header(header, event.schema_version, event.sequence)
    _signature_header(header, event.signature, event.key_id, event.schema_version, event.sequence)
    return header


def _signature_header(header, signature, key_id, version, sequence):
    if signature is None and key_id is None:
        return
    if version != 2 or not isinstance(signature, str) or not isinstance(key_id, str) or not signature or not key_id:
        raise ChainBreak(sequence, 'T1: invalid signature metadata')
    header['key_id'] = key_id
    header['signature'] = signature


def _check_signature(header, public_key, required, sequence):
    signature = header.get('signature')
    if signature is None:
        if required:
            raise ChainBreak(sequence, 'T1: unsigned event in a signed chain')
        return False
    if public_key is not None:
        from kognita.signing import _verify
        unsigned = dict(header)
        del unsigned['signature']
        _verify(unsigned, signature, header['key_id'], public_key, sequence)
    return True


class ChainBreak(Exception):
    """The evidence chain does not verify."""

    def __init__(self, sequence: int, reason: str) -> None:
        super().__init__(f"evidence chain broken at sequence {sequence}: {reason}")
        self.sequence = sequence
        self.reason = reason


def _version_header(header: dict[str, Any], schema_version: Any, sequence: int) -> None:
    """Keep v1 hashes unchanged and reject unsupported schemas (T5)."""
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ChainBreak(sequence, "T5: unsupported schema_version")
    if schema_version >= 2:
        header["schema_version"] = schema_version


def verify_chain(session: Session, *, public_key: bytes | None = None) -> int:
    """Verify every event in order; return how many were checked.

    With ``public_key`` (PEM bytes), require and verify every signature.
    Without it, check hash integrity and signature continuity only.

    Raises :class:`ChainBreak` naming the first sequence that fails and why —
    a rewritten payload, a recomputed hash that no longer matches, a link that
    points at the wrong predecessor, or a head checkpoint that is missing,
    unreadable, or does not match the chain's tail. A missing or unreadable
    head is only tolerated when there are no events to check against it —
    once events exist, the absence of a checkpoint is itself a reason to fail
    closed rather than silently skip the tail check.
    """
    events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    required = public_key is not None or any(event.signature is not None for event in events)
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
        required = _check_signature(
            _event_header(event),
            public_key, required, event.sequence,
        ) or required
        expected_prev = event.event_hash
    head = evidence_head(session)
    if head is None:
        if events:
            raise ChainBreak(
                len(events),
                "the evidence_head checkpoint is missing or unreadable, but "
                f"{len(events)} event(s) exist: the tail cannot be verified",
            )
        return 0
    sequence, head_hash = head
    if sequence != len(events) or head_hash != expected_prev:
        raise ChainBreak(
            sequence,
            f"the chain ends at sequence {len(events)}, but the head checkpoint "
            f"records sequence {sequence}: events were removed from the end",
        )
    return len(events)


def evidence_head(session: Session) -> tuple[int, str] | None:
    """The head checkpoint ``(sequence, event_hash)``, or ``None`` if the store has none.

    ``create_all`` keeps it on SQLite. A store without the table has no
    checkpoint to compare against.
    """
    try:
        row = session.connection().exec_driver_sql(
            "SELECT sequence, event_hash FROM evidence_head WHERE id = 1"
        ).first()
    except Exception:
        return None
    if row is None:
        return None
    return int(row[0]), str(row[1])


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
            "signature": e.signature,
            "key_id": e.key_id,
            "schema_version": e.schema_version,
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


def verify_export(export: dict[str, Any], *, public_key: bytes | None = None) -> int:
    """Verify an exported artifact without a database. Returns events checked.

    With ``public_key`` (PEM bytes), require and verify every signature.
    Without it, check hash integrity and signature continuity only.

    Internal chain consistency alone is not enough: dropping trailing rows
    from ``events`` leaves the remaining prefix internally consistent, so the
    advertised ``event_count`` and ``head_hash`` are checked against what was
    actually present and verified, and a mismatch — including a short export
    whose metadata was left unchanged — is reported as a chain break.
    """
    expected_prev = GENESIS_HASH
    rows = export.get("events", [])
    required = public_key is not None or any(row.get("signature") is not None for row in rows)
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
            "recorded_at": row.get("recorded_at"),
        }
        _version_header(header, row.get("schema_version", 1), row["sequence"])
        _signature_header(header, row.get('signature'), row.get('key_id'), row.get('schema_version', 1), row['sequence'])
        required = _check_signature(header, public_key, required, row['sequence']) or required
        if canonical_hash(header) != row["event_hash"]:
            raise ChainBreak(row["sequence"], "event hash does not match its contents")
        expected_prev = row["event_hash"]
    advertised_count = export.get("event_count")
    if type(advertised_count) is not int or len(rows) != advertised_count:
        raise ChainBreak(
            len(rows),
            f"the export contains {len(rows)} event(s), but advertises "
            f"event_count {advertised_count}: events were removed from the end",
        )
    advertised_head = export.get("head_hash", GENESIS_HASH)
    if advertised_head != expected_prev:
        raise ChainBreak(
            len(rows),
            "the export's head_hash does not match the hash of its last "
            "verified event: events were removed from the end",
        )
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
    "TelemetryHook",
    "ChainBreak",
    "verify_chain",
    "export_chain",
    "verify_export",
    "hashes_only",
    "recompute_event_hash",
    "GENESIS_HASH",
]
