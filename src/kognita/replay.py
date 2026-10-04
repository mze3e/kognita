"""Replay a recorded decision against the inputs it pinned.

:func:`kognita.governance.decide` is pure: the same envelope and the same
snapshot give the same outcome. That is not enough when a policy row, a
retrieved item, or a retained prompt can change after the decision. Replay
reads the hashes the decision recorded and fails if the current row, item, or
stored bytes are not those hashes.

Erasure is not a mismatch. The bytes are gone, the hash remains, and an
``ERASURE`` event says the removal was recorded.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlmodel import Session, select

from kognita.canonical import canonical_hash, hash_text
from kognita.envelope import Check, Envelope, Evaluation
from kognita.evidence import verify_chain
from kognita.exceptions import ReplayMismatch
from kognita.governance import decide, load_snapshot
from kognita.retrieval import item_content_hash
from kognita.models import (
    EvidenceEvent,
    GovernanceDecision,
    KnowledgeItem,
    Policy,
    RetainedContent,
    as_utc,
    policy_content_hash,
)
from kognita.vocabulary import CheckResult, EventType, Outcome


def replay_decision(
    session: Session,
    request_id: str,
    envelope: Envelope | None = None,
    *,
    rules: dict[str, Any] | None = None,
    purposes: tuple[str, ...] | list[str] = (),
    engages: Any = None,
    subjects: dict[str, Any] | None = None,
) -> Evaluation:
    """Replay ``request_id`` and raise :class:`ReplayMismatch` when a pin fails.

    The chain is verified first. A policy hash is compared to the policy row
    with that id, and the row must still cover the decision's ``as_of``. A
    retrieved item's content hash and embedding model are compared to the item.
    A retained prompt or response must still hash to the value on the chain,
    unless an erasure event records that it was removed.

    Pass the original ``envelope`` to decide again. The chain stores a hash of
    the tool arguments, not the arguments, so a replay without the envelope
    returns the recorded evaluation after the pins have been checked.
    """
    verify_chain(session)
    decision = session.exec(
        select(GovernanceDecision).where(GovernanceDecision.request_id == request_id)
    ).first()
    if decision is None:
        raise ReplayMismatch(f"no decision recorded for {request_id}")
    event = _decision_event(session, request_id)
    payload = _trusted_payload(event)
    at = as_utc(decision.as_of)
    if at is None:
        raise ReplayMismatch(f"decision {request_id} has no as_of")
    _verify_policy_pins(session, payload.get("checks") or [], at)
    events = _events(session, request_id)
    for recorded in events:
        _trusted_payload(recorded)
    _verify_retrieval_pins(session, events)
    _verify_retained(session, events)
    if envelope is None:
        return _recorded_evaluation(decision, payload, at)
    evaluation = decide(
        envelope,
        load_snapshot(session, as_of=at),
        attributes=dict(payload.get("attributes") or {}),
        subjects=subjects or {},
        rules=rules,
        purposes=purposes,
        engages=engages,
        as_of=at,
        request_id=request_id,
    )
    _same_policy_hashes(payload.get("checks") or [], evaluation)
    return evaluation


def _events(session: Session, request_id: str) -> list[EvidenceEvent]:
    return list(
        session.exec(
            select(EvidenceEvent)
            .where(EvidenceEvent.correlation_id == request_id)
            .order_by(EvidenceEvent.sequence)
        ).all()
    )


def _decision_event(session: Session, request_id: str) -> EvidenceEvent:
    event = session.exec(
        select(EvidenceEvent)
        .where(EvidenceEvent.correlation_id == request_id)
        .where(EvidenceEvent.event_type == EventType.POLICY_DECISION)
        .order_by(EvidenceEvent.sequence)
    ).first()
    if event is None:
        raise ReplayMismatch(f"no policy decision evidence for {request_id}")
    return event


def _trusted_payload(event: EvidenceEvent) -> dict[str, Any]:
    payload = dict(event.payload or {})
    if canonical_hash(payload) != event.payload_hash:
        raise ReplayMismatch(
            f"evidence payload at sequence {event.sequence} does not match its hash"
        )
    return payload


def _verify_policy_pins(
    session: Session, checks: list[dict[str, Any]], at: datetime
) -> None:
    for check in checks:
        policy_id = check.get("policy_id")
        recorded = check.get("policy_hash")
        if policy_id is None or not isinstance(recorded, str) or not recorded:
            continue
        policy = session.get(Policy, policy_id)
        if policy is None:
            raise ReplayMismatch(
                f"policy {policy_id} recorded on the decision is gone"
            )
        current = policy_content_hash(policy)
        if current != recorded:
            raise ReplayMismatch(
                f"policy {policy_id} content hash does not match the decision; "
                "the row was edited in place"
            )
        start = as_utc(policy.effective_from)
        end = as_utc(policy.effective_to)
        # Closing the window at the decision instant still covers that decision.
        # Moving the close to before as_of does not.
        if start is None or start > at or (end is not None and end < at):
            raise ReplayMismatch(
                f"policy {policy_id} is not effective at {at.isoformat()}"
            )


def _verify_retrieval_pins(session: Session, events: list[EvidenceEvent]) -> None:
    for event in events:
        if event.event_type != EventType.RETRIEVAL:
            continue
        items = event.payload.get("items")
        if not isinstance(items, list):
            continue
        for entry in items:
            if not isinstance(entry, dict):
                continue
            item_id = entry.get("id")
            recorded = entry.get("content_hash")
            if not isinstance(recorded, str) or not recorded:
                continue
            item = session.get(KnowledgeItem, item_id)
            if item is None:
                raise ReplayMismatch(f"retrieved item {item_id} recorded on the decision is gone")
            if item_content_hash(item) != recorded:
                raise ReplayMismatch(
                    f"retrieved item {item_id} content hash does not match the decision"
                )
            recorded_model = entry.get("embedding_model")
            if isinstance(recorded_model, str) and item.embedding_model != recorded_model:
                raise ReplayMismatch(
                    f"retrieved item {item_id} embedding model does not match the decision"
                )


def _verify_retained(session: Session, events: list[EvidenceEvent]) -> None:
    for digest in _content_hashes(events):
        row = session.get(RetainedContent, digest)
        if row is None:
            if not _erasure_recorded(session, digest):
                raise ReplayMismatch(
                    f"retained content {digest} is missing and no erasure was recorded"
                )
            continue
        if hash_text(row.body) != digest:
            raise ReplayMismatch(f"retained content {digest} does not match its hash")


def _content_hashes(events: list[EvidenceEvent]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for event in events:
        payload = event.payload or {}
        for key in ("prompt_hash", "response_hash"):
            value = payload.get(key)
            if isinstance(value, str) and value and value not in seen:
                seen.add(value)
                found.append(value)
        items = payload.get("items")
        if isinstance(items, list):
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                value = entry.get("content_hash")
                if isinstance(value, str) and value and value not in seen:
                    seen.add(value)
                    found.append(value)
    return found


def _erasure_recorded(session: Session, content_hash: str) -> bool:
    events = session.exec(
        select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.ERASURE)
    ).all()
    return any(
        isinstance(event.payload, dict) and event.payload.get("content_hash") == content_hash
        for event in events
    )


def _same_policy_hashes(recorded: list[dict[str, Any]], evaluation: Evaluation) -> None:
    fresh = {
        check.policy_id: check.policy_hash
        for check in evaluation.checks
        if check.policy_id is not None and check.policy_hash
    }
    for check in recorded:
        policy_id = check.get("policy_id")
        pinned = check.get("policy_hash")
        if policy_id is None or not isinstance(pinned, str) or not pinned:
            continue
        if policy_id in fresh and fresh[policy_id] != pinned:
            raise ReplayMismatch(
                f"policy {policy_id} content hash does not match the decision; "
                "the row was edited in place"
            )


def _recorded_evaluation(
    decision: GovernanceDecision, payload: dict[str, Any], at: datetime
) -> Evaluation:
    return Evaluation(
        request_id=decision.request_id,
        outcome=Outcome(payload["outcome"]),
        checks=tuple(_check_from_dict(item) for item in payload.get("checks") or []),
        attributes=dict(payload.get("attributes") or {}),
        envelope=_envelope_from_evidence(payload.get("envelope") or {}),
        as_of=at,
        envelope_hash=str(payload.get("envelope_hash") or decision.envelope_hash),
        decision_id=decision.id,
    )


def _check_from_dict(item: dict[str, Any]) -> Check:
    return Check(
        check=item["check"],
        regime=item["regime"],
        result=CheckResult(item["result"]),
        citation=item["citation"],
        policy_id=item.get("policy_id"),
        policy_hash=item.get("policy_hash"),
    )


def _envelope_from_evidence(data: dict[str, Any]) -> Envelope:
    arguments = data.get("arguments") or {}
    if _arguments_are_a_hash(arguments):
        arguments = {}
    return Envelope(
        principal=data.get("principal") or "",
        purpose=data.get("purpose") or "",
        tool=data.get("tool") or "",
        actor_location=data.get("actor_location") or "",
        agent_name=data.get("agent_name"),
        subject_type=data.get("subject_type"),
        subject_id=data.get("subject_id"),
        subjects=dict(data.get("subjects") or {}),
        arguments=dict(arguments),
        roles=list(data.get("roles") or []),
        scopes=dict(data.get("scopes") or {}),
        is_admin=bool(data.get("is_admin", False)),
    )


def _arguments_are_a_hash(arguments: Any) -> bool:
    return (
        isinstance(arguments, dict)
        and set(arguments) == {"sha256", "bytes"}
    )


__all__ = ["replay_decision"]
