"""Reconstruction report for one recorded interaction.

``kognita evidence reconstruct`` answers the ten reconstruction-test questions
from the evidence chain and the retention store. It does not decide again, and
it does not call a classifier, a model, or a tool.

There is no interaction table in this release. The identifier is the
``correlation_id`` already written on evidence events, which is the decision's
``request_id``. An interaction record that groups several of those ids is a
later release; this report does not invent one.

Questions answered by origination, RM review capture, or governed client
communication are present and marked ``not recorded``. Approval events can
exist on the chain when a policy forced one. That is not a final decision on
every path, so "Who made the final decision?" stays ``not recorded``.

Pin checks follow replay, including two residuals left as they are:

- A policy window closed exactly at ``as_of`` still covers that decision.
  :meth:`kognita.models.Policy.is_effective` is half-open and would exclude it.
- An empty ``response_hash`` or ``prompt_hash`` is not a pin. Replay skips it.

A mismatch is a finding. The report still lists the row and the other questions.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlmodel import Session, select

from kognita.canonical import canonical_hash, hash_text
from kognita.evidence import GENESIS_HASH, recompute_event_hash
from kognita.models import (
    EvidenceEvent,
    KnowledgeItem,
    Policy,
    RetainedContent,
    as_utc,
    policy_content_hash,
)
from kognita.retrieval import item_content_hash
from kognita.vocabulary import EventType

#: Machine-readable marker. Not a status enum and not a sentence.
NOT_RECORDED = "not recorded"

#: The reconstruction test, in order. Wording is the roadmap table.
QUESTIONS: tuple[str, ...] = (
    "Why this client?",
    "Why this insight or product?",
    "What information did the AI use?",
    "What model or agent produced it?",
    "What was the agent authorized to do?",
    "What suitability or policy controls ran?",
    "What did the RM see and change?",
    "Who made the final decision?",
    "What was communicated to the client?",
    "Can the bank reproduce that evidence later?",
)

#: Delivered by 0.4: origination, RM review capture, governed client communication.
_NOT_RECORDED_QUESTIONS = frozenset(
    {
        "Why this client?",
        "Why this insight or product?",
        "What did the RM see and change?",
        "Who made the final decision?",
        "What was communicated to the client?",
    }
)

_PIN_HASH_KEYS = ("prompt_hash", "response_hash")
_OK_PIN_STATUSES = frozenset({"matches", "retained", "erased", "empty", "recorded"})


def reconstruct(session: Session, interaction_id: str) -> dict[str, Any]:
    """Build the report for ``interaction_id`` from stored evidence.

    The chain is checked in full. Pins are checked for events whose
    ``correlation_id`` is ``interaction_id``. Nothing is raised: a broken link
    or a hash mismatch is a finding, and the rest of the report is still built.
    """
    events = list(
        session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    )
    findings, checked = _chain_findings(events)
    selected = [event for event in events if event.correlation_id == interaction_id]
    if not selected:
        findings.append(
            {
                "kind": "interaction",
                "message": f"no evidence recorded for {interaction_id}",
            }
        )

    decisions = [_decision_view(event) for event in selected if _is(event, EventType.POLICY_DECISION)]
    retrievals = [_retrieval_view(session, event) for event in selected if _is(event, EventType.RETRIEVAL)]
    models = [_model_view(session, event) for event in selected if _is(event, EventType.MODEL_CALL)]
    findings.extend(_policy_findings(session, decisions))
    findings.extend(_retrieval_findings(session, selected))
    findings.extend(_retained_findings(session, selected))
    findings.extend(_response_hash_findings(selected))

    pins = _pins(session, decisions, selected)
    chain_intact = not any(item["kind"] == "chain" for item in findings)
    report = {
        "kognita_reconstruction": 1,
        "interaction_id": interaction_id,
        "chain": {"intact": chain_intact, "events_checked": checked},
        "findings": findings,
        "questions": _questions(
            decisions=decisions,
            retrievals=retrievals,
            models=models,
            pins=pins,
            chain_intact=chain_intact,
            events_checked=checked,
        ),
    }
    return report


def render_reconstruction(report: dict[str, Any]) -> str:
    """The regulator-readable form of a :func:`reconstruct` report."""
    lines: list[str] = [
        "Reconstruction report",
        f"Interaction: {report.get('interaction_id', '')}",
        "",
    ]
    chain = report.get("chain") or {}
    if chain.get("intact"):
        lines.append(f"Evidence chain: intact, {chain.get('events_checked', 0)} events checked.")
    else:
        lines.append(
            f"Evidence chain: not intact, {chain.get('events_checked', 0)} events checked."
        )
    lines.append("")
    lines.append("Findings")
    findings = report.get("findings") or []
    if not findings:
        lines.append("none")
    else:
        for finding in findings:
            lines.append(f"- {finding.get('message', '')}")
    lines.append("")
    questions = report.get("questions") or []
    for index, item in enumerate(questions, start=1):
        question = str(item.get("question", ""))
        lines.append(f"{index}. {question}")
        lines.append("")
        lines.extend(_render_answer(question, item.get("answer")))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _questions(
    *,
    decisions: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    models: list[dict[str, Any]],
    pins: list[dict[str, Any]],
    chain_intact: bool,
    events_checked: int,
) -> list[dict[str, Any]]:
    answers: dict[str, Any] = {
        "What information did the AI use?": {
            "retrievals": retrievals,
            "argument_hashes": _argument_hashes(decisions),
        },
        "What model or agent produced it?": {
            "agents": _agents(decisions),
            "models": models,
        },
        "What was the agent authorized to do?": {"decisions": _authority(decisions)},
        "What suitability or policy controls ran?": {"decisions": _controls(decisions)},
        "Can the bank reproduce that evidence later?": {
            "chain_intact": chain_intact,
            "events_checked": events_checked,
            "pins": pins,
        },
    }
    return [
        {
            "question": question,
            "answer": NOT_RECORDED if question in _NOT_RECORDED_QUESTIONS else answers[question],
        }
        for question in QUESTIONS
    ]


def _chain_findings(events: list[EvidenceEvent]) -> tuple[list[dict[str, Any]], int]:
    """Every break in the chain. One break does not stop the walk."""
    findings: list[dict[str, Any]] = []
    expected_prev = GENESIS_HASH
    for index, event in enumerate(events, start=1):
        sequence = event.sequence
        if sequence != index:
            findings.append(
                _chain(sequence, f"expected sequence {index}")
            )
        payload = event.payload if isinstance(event.payload, dict) else event.payload
        if not isinstance(payload, dict) or canonical_hash(payload) != event.payload_hash:
            findings.append(_chain(sequence, "payload does not match its hash"))
        if event.prev_hash != expected_prev:
            findings.append(
                _chain(sequence, "prev_hash does not match the previous event")
            )
        try:
            recomputed = recompute_event_hash(event)
        except (TypeError, ValueError) as exc:
            findings.append(_chain(sequence, f"event hash does not match its contents ({exc})"))
        else:
            if recomputed != event.event_hash:
                findings.append(_chain(sequence, "event hash does not match its contents"))
        expected_prev = event.event_hash
    return findings, len(events)


def _chain(sequence: int, reason: str) -> dict[str, Any]:
    return {
        "kind": "chain",
        "sequence": sequence,
        "message": f"evidence chain broken at sequence {sequence}: {reason}",
    }


def _decision_view(event: EvidenceEvent) -> dict[str, Any]:
    payload = _payload(event)
    envelope = payload.get("envelope")
    if not isinstance(envelope, dict):
        envelope = {}
    as_of = _parse_time(payload.get("as_of"))
    return {
        "sequence": event.sequence,
        "request_id": event.correlation_id,
        "as_of": as_of.isoformat() if as_of else None,
        "outcome": payload.get("outcome"),
        "agent_name": envelope.get("agent_name"),
        "principal": envelope.get("principal") or "",
        "purpose": envelope.get("purpose") or "",
        "tool": envelope.get("tool") or "",
        "roles": list(envelope.get("roles") or []),
        "scopes": dict(envelope.get("scopes") or {}),
        "arguments": envelope.get("arguments") if isinstance(envelope.get("arguments"), dict) else {},
        "checks": _checks(payload.get("checks")),
        "registry_citation": _registry_citation(payload.get("checks")),
    }


def _retrieval_view(session: Session, event: EvidenceEvent) -> dict[str, Any]:
    payload = _payload(event)
    items: list[dict[str, Any]] = []
    raw_items = payload.get("items")
    if isinstance(raw_items, list):
        for entry in raw_items:
            if not isinstance(entry, dict):
                continue
            recorded = entry.get("content_hash")
            content_hash = recorded if isinstance(recorded, str) else ""
            retained = _content_record(session, content_hash)
            items.append(
                {
                    "id": entry.get("id"),
                    "content_hash": content_hash,
                    "embedding_model": entry.get("embedding_model"),
                    "content_status": retained["status"],
                    "body": retained.get("body"),
                }
            )
    return {
        "sequence": event.sequence,
        "zone": payload.get("zone"),
        "embedder": payload.get("embedder"),
        "items": items,
    }


def _model_view(session: Session, event: EvidenceEvent) -> dict[str, Any]:
    payload = _payload(event)
    prompt = _content_record(session, _hash_field(payload, "prompt_hash"))
    response = _content_record(session, _hash_field(payload, "response_hash"))
    return {
        "sequence": event.sequence,
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "model_version": payload.get("model_version"),
        "prompt_template_version": payload.get("prompt_template_version"),
        "prompt_hash": prompt["content_hash"],
        "prompt_status": prompt["status"],
        "prompt": prompt.get("body"),
        "response_hash": response["content_hash"],
        "response_status": response["status"],
        "response": response.get("body"),
    }


def _authority(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": decision["request_id"],
            "agent_name": decision["agent_name"],
            "principal": decision["principal"],
            "purpose": decision["purpose"],
            "tool": decision["tool"],
            "roles": decision["roles"],
            "scopes": decision["scopes"],
            "outcome": decision["outcome"],
            "checks": decision["checks"],
        }
        for decision in decisions
    ]


def _controls(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": decision["request_id"],
            "as_of": decision["as_of"],
            "outcome": decision["outcome"],
            "checks": decision["checks"],
        }
        for decision in decisions
    ]


def _agents(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agents: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for decision in decisions:
        name = decision.get("agent_name")
        key = (name, decision.get("registry_citation"))
        if key in seen:
            continue
        seen.add(key)
        if not name and not decision.get("registry_citation"):
            continue
        agents.append(
            {
                "agent_name": name,
                "registry_citation": decision.get("registry_citation"),
            }
        )
    return agents


def _argument_hashes(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for decision in decisions:
        arguments = decision.get("arguments") or {}
        digest = arguments.get("sha256")
        if isinstance(digest, str) and digest:
            found.append(
                {
                    "sha256": digest,
                    "bytes": arguments.get("bytes"),
                    "sequence": decision["sequence"],
                }
            )
    return found


def _policy_findings(session: Session, decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()
    for decision in decisions:
        at = _parse_time(decision.get("as_of"))
        for check in decision["checks"]:
            policy_id = check.get("policy_id")
            recorded = check.get("policy_hash")
            if policy_id is None or not isinstance(recorded, str) or not recorded:
                continue
            key = (policy_id, recorded)
            if key in seen:
                continue
            seen.add(key)
            policy = session.get(Policy, policy_id)
            if policy is None:
                findings.append(
                    {
                        "kind": "policy_content_hash",
                        "policy_id": policy_id,
                        "policy_hash": recorded,
                        "sequence": decision["sequence"],
                        "message": (
                            f"policy {policy_id} recorded on the decision is gone"
                        ),
                    }
                )
                continue
            if policy_content_hash(policy) != recorded:
                findings.append(
                    {
                        "kind": "policy_content_hash",
                        "policy_id": policy_id,
                        "policy_hash": recorded,
                        "sequence": decision["sequence"],
                        "message": (
                            f"policy {policy_id} content hash does not match the decision; "
                            "the row was edited in place"
                        ),
                    }
                )
                continue
            if at is not None and not _window_covers(policy, at):
                findings.append(
                    {
                        "kind": "policy_content_hash",
                        "policy_id": policy_id,
                        "policy_hash": recorded,
                        "sequence": decision["sequence"],
                        "message": (
                            f"policy {policy_id} is not effective at {at.isoformat()}"
                        ),
                    }
                )
    return findings


def _window_covers(policy: Policy, at: datetime) -> bool:
    """Whether replay would still treat ``policy`` as covering ``at``.

    A window closed exactly at ``at`` still covers. :meth:`Policy.is_effective`
    is half-open and would not. Reconstruct uses the replay comparison so a pin
    replay accepts is not reported as a finding.
    """
    start = as_utc(policy.effective_from)
    end = as_utc(policy.effective_to)
    if start is None or start > at or (end is not None and end < at):
        return False
    return True


def _retrieval_findings(session: Session, events: list[EvidenceEvent]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for event in events:
        if not _is(event, EventType.RETRIEVAL):
            continue
        items = _payload(event).get("items")
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
                findings.append(
                    {
                        "kind": "retrieved_item_content_hash",
                        "item_id": item_id,
                        "content_hash": recorded,
                        "sequence": event.sequence,
                        "message": (
                            f"retrieved item {item_id} recorded on the decision is gone"
                        ),
                    }
                )
                continue
            if item_content_hash(item) != recorded:
                findings.append(
                    {
                        "kind": "retrieved_item_content_hash",
                        "item_id": item_id,
                        "content_hash": recorded,
                        "sequence": event.sequence,
                        "message": (
                            f"retrieved item {item_id} content hash does not match "
                            "the decision"
                        ),
                    }
                )
            recorded_model = entry.get("embedding_model")
            if isinstance(recorded_model, str) and item.embedding_model != recorded_model:
                findings.append(
                    {
                        "kind": "embedding_model",
                        "item_id": item_id,
                        "sequence": event.sequence,
                        "message": (
                            f"retrieved item {item_id} embedding model does not match "
                            "the decision"
                        ),
                    }
                )
    return findings


def _retained_findings(session: Session, events: list[EvidenceEvent]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for digest in _content_hashes(events):
        if digest in seen:
            continue
        seen.add(digest)
        record = _content_record(session, digest)
        if record["status"] == "missing":
            findings.append(
                {
                    "kind": "retained_content",
                    "content_hash": digest,
                    "message": (
                        f"retained content {digest} is missing and no erasure was recorded"
                    ),
                }
            )
        elif record["status"] == "mismatch":
            findings.append(
                {
                    "kind": "retained_content",
                    "content_hash": digest,
                    "message": f"retained content {digest} does not match its hash",
                }
            )
    return findings


def _response_hash_findings(events: list[EvidenceEvent]) -> list[dict[str, Any]]:
    """A tool or model call and the egress event after it must share a response hash.

    An empty hash is not compared. Replay skips those, and so does this.
    """
    findings: list[dict[str, Any]] = []
    pending: list[tuple[int, str]] = []
    for event in events:
        payload = _payload(event)
        recorded = _hash_field(payload, "response_hash")
        if _is(event, EventType.TOOL_CALL) or _is(event, EventType.MODEL_CALL):
            pending.append((event.sequence, recorded))
            continue
        if not _is(event, EventType.EGRESS) or not pending:
            continue
        call_sequence, call_hash = pending.pop(0)
        if not call_hash or not recorded or call_hash == recorded:
            continue
        findings.append(
            {
                "kind": "response_hash",
                "sequence": event.sequence,
                "message": (
                    f"call and egress response hashes differ at sequence {event.sequence}"
                    f" (call sequence {call_sequence})"
                ),
            }
        )
    return findings


def _pins(
    session: Session,
    decisions: list[dict[str, Any]],
    events: list[EvidenceEvent],
) -> list[dict[str, Any]]:
    pins: list[dict[str, Any]] = []
    for decision in decisions:
        at = _parse_time(decision.get("as_of"))
        for check in decision["checks"]:
            policy_id = check.get("policy_id")
            recorded = check.get("policy_hash")
            if policy_id is None or not isinstance(recorded, str) or not recorded:
                continue
            pins.append(
                {
                    "kind": "policy",
                    "policy_id": policy_id,
                    "policy_hash": recorded,
                    "status": _policy_status(session, policy_id, recorded, at),
                    "sequence": decision["sequence"],
                }
            )
    for event in events:
        if _is(event, EventType.RETRIEVAL):
            items = _payload(event).get("items")
            if not isinstance(items, list):
                continue
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                pins.append(_item_pin(session, event.sequence, entry))
        if _is(event, EventType.MODEL_CALL):
            payload = _payload(event)
            pins.append(
                {
                    "kind": "model",
                    "sequence": event.sequence,
                    "event_type": EventType.MODEL_CALL.value,
                    "provider": payload.get("provider"),
                    "model": payload.get("model"),
                    "model_version": payload.get("model_version"),
                    "prompt_template_version": payload.get("prompt_template_version"),
                    "status": "recorded",
                }
            )
        for key, kind in (("prompt_hash", "prompt"), ("response_hash", "response")):
            if key == "prompt_hash" and not _is(event, EventType.MODEL_CALL):
                continue
            if key == "response_hash" and not (
                _is(event, EventType.MODEL_CALL)
                or _is(event, EventType.EGRESS)
                or _is(event, EventType.TOOL_CALL)
            ):
                continue
            recorded = _hash_field(_payload(event), key)
            retained = _content_record(session, recorded)
            pins.append(
                {
                    "kind": kind,
                    "content_hash": recorded,
                    "status": retained["status"],
                    "sequence": event.sequence,
                    "event_type": _event_type_value(event),
                }
            )
    return pins


def _policy_status(
    session: Session, policy_id: Any, recorded: str, at: datetime | None
) -> str:
    policy = session.get(Policy, policy_id)
    if policy is None:
        return "missing"
    if policy_content_hash(policy) != recorded:
        return "mismatch"
    if at is not None and not _window_covers(policy, at):
        return "window"
    return "matches"


def _item_pin(session: Session, sequence: int, entry: dict[str, Any]) -> dict[str, Any]:
    item_id = entry.get("id")
    recorded = entry.get("content_hash")
    content_hash = recorded if isinstance(recorded, str) else ""
    recorded_model = entry.get("embedding_model")
    status = "empty"
    model_status = "recorded"
    if content_hash:
        item = session.get(KnowledgeItem, item_id)
        if item is None:
            status = "missing"
            model_status = "missing"
        else:
            status = "matches" if item_content_hash(item) == content_hash else "mismatch"
            if isinstance(recorded_model, str):
                model_status = (
                    "matches" if item.embedding_model == recorded_model else "mismatch"
                )
            if model_status == "mismatch":
                status = "mismatch"
    retained = _content_record(session, content_hash)
    return {
        "kind": "retrieved_item",
        "item_id": item_id,
        "content_hash": content_hash,
        "embedding_model": recorded_model if isinstance(recorded_model, str) else None,
        "embedding_model_status": model_status,
        "status": status,
        "content_status": retained["status"],
        "sequence": sequence,
    }


def _content_hashes(events: list[EvidenceEvent]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for event in events:
        payload = _payload(event)
        for key in _PIN_HASH_KEYS:
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


def _content_record(session: Session, content_hash: str) -> dict[str, Any]:
    """What the retention store holds for a pinned hash.

    Erasure keeps the hash and is not a mismatch. An empty hash is not a pin.
    """
    if not content_hash:
        return {"content_hash": "", "status": "empty"}
    row = session.get(RetainedContent, content_hash)
    if row is None:
        erasure = _erasure_event(session, content_hash)
        if erasure is None:
            return {"content_hash": content_hash, "status": "missing"}
        erasure_payload = _payload(erasure)
        return {
            "content_hash": content_hash,
            "status": "erased",
            "kind": erasure_payload.get("kind"),
            "reason": erasure_payload.get("reason"),
            "erasure_sequence": erasure.sequence,
        }
    if hash_text(row.body) != content_hash:
        return {
            "content_hash": content_hash,
            "status": "mismatch",
            "kind": row.kind,
        }
    return {
        "content_hash": content_hash,
        "status": "retained",
        "kind": row.kind,
        "body": row.body,
    }


def _erasure_event(session: Session, content_hash: str) -> EvidenceEvent | None:
    events = session.exec(
        select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.ERASURE)
    ).all()
    for event in events:
        payload = event.payload
        if isinstance(payload, dict) and payload.get("content_hash") == content_hash:
            return event
    return None


def _checks(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    checks: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        checks.append(
            {
                "check": item.get("check"),
                "regime": item.get("regime"),
                "result": item.get("result"),
                "citation": item.get("citation"),
                "policy_id": item.get("policy_id"),
                "policy_hash": item.get("policy_hash"),
            }
        )
    return checks


def _registry_citation(raw: Any) -> str | None:
    if not isinstance(raw, list):
        return None
    for item in raw:
        if isinstance(item, dict) and item.get("check") == "AGENT_REGISTRY":
            citation = item.get("citation")
            return citation if isinstance(citation, str) else None
    return None


def _payload(event: EvidenceEvent) -> dict[str, Any]:
    if isinstance(event.payload, dict):
        return event.payload
    return {}


def _hash_field(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return as_utc(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return as_utc(parsed)
    return None


def _is(event: EvidenceEvent, event_type: EventType) -> bool:
    return event.event_type == event_type


def _event_type_value(event: EvidenceEvent) -> str:
    event_type = event.event_type
    return event_type.value if isinstance(event_type, EventType) else str(event_type)


def _render_answer(question: str, answer: Any) -> list[str]:
    if answer == NOT_RECORDED:
        return [NOT_RECORDED]
    if question == "What information did the AI use?":
        return _render_information(answer)
    if question == "What model or agent produced it?":
        return _render_model(answer)
    if question == "What was the agent authorized to do?":
        return _render_authority(answer)
    if question == "What suitability or policy controls ran?":
        return _render_controls(answer)
    if question == "Can the bank reproduce that evidence later?":
        return _render_reproduce(answer)
    return [str(answer)]


def _render_information(answer: Any) -> list[str]:
    if not isinstance(answer, dict):
        return []
    lines: list[str] = []
    retrievals = answer.get("retrievals") or []
    if not retrievals:
        lines.append("No retrieval was recorded.")
    for retrieval in retrievals:
        lines.append(
            f"Retrieval at sequence {retrieval.get('sequence')}, "
            f"zone {retrieval.get('zone')}, embedder {retrieval.get('embedder')}."
        )
        items = retrieval.get("items") or []
        if not items:
            lines.append("No item was returned.")
        for item in items:
            lines.append(
                f"Item {item.get('id')}, content hash {item.get('content_hash')}, "
                f"embedding model {item.get('embedding_model')}, "
                f"content {item.get('content_status')}."
            )
            if item.get("content_status") == "retained" and item.get("body"):
                lines.append(str(item["body"]))
            if item.get("content_status") == "erased":
                lines.append(
                    f"Content erased; hash {item.get('content_hash')} kept."
                )
    for argument in answer.get("argument_hashes") or []:
        lines.append(
            f"Tool arguments hash {argument.get('sha256')} "
            f"({argument.get('bytes')} bytes)."
        )
    return lines


def _render_model(answer: Any) -> list[str]:
    if not isinstance(answer, dict):
        return []
    lines: list[str] = []
    agents = answer.get("agents") or []
    if not agents:
        lines.append("No agent was recorded on the decision.")
    for agent in agents:
        lines.append(f"Agent: {agent.get('agent_name')}.")
        if agent.get("registry_citation"):
            lines.append(f"Registry: {agent['registry_citation']}.")
    models = answer.get("models") or []
    if not models:
        lines.append("No model call was recorded.")
    for model in models:
        lines.append(
            f"Model: {model.get('model')}, version {model.get('model_version')}, "
            f"provider {model.get('provider')}."
        )
        lines.append(f"Prompt template: {model.get('prompt_template_version')}.")
        lines.append(
            f"Prompt hash {model.get('prompt_hash')}: {model.get('prompt_status')}."
        )
        if model.get("prompt_status") == "retained" and model.get("prompt"):
            lines.append(str(model["prompt"]))
        if model.get("prompt_status") == "erased":
            lines.append(f"Prompt content erased; hash {model.get('prompt_hash')} kept.")
        lines.append(
            f"Response hash {model.get('response_hash')}: {model.get('response_status')}."
        )
        if model.get("response_status") == "retained" and model.get("response"):
            lines.append(str(model["response"]))
        if model.get("response_status") == "erased":
            lines.append(
                f"Response content erased; hash {model.get('response_hash')} kept."
            )
    return lines


def _render_authority(answer: Any) -> list[str]:
    if not isinstance(answer, dict):
        return []
    lines: list[str] = []
    decisions = answer.get("decisions") or []
    if not decisions:
        lines.append("No decision was recorded.")
    for decision in decisions:
        lines.append(
            f"Agent {decision.get('agent_name')} for principal {decision.get('principal')}, "
            f"purpose {decision.get('purpose')}, tool {decision.get('tool')}. "
            f"Outcome recorded: {decision.get('outcome')}."
        )
        lines.append(f"Roles: {_join(decision.get('roles'))}.")
        lines.append(f"Scopes: {_scopes(decision.get('scopes'))}.")
        lines.extend(_render_check_lines(decision.get("checks") or []))
    return lines


def _render_controls(answer: Any) -> list[str]:
    if not isinstance(answer, dict):
        return []
    lines: list[str] = []
    decisions = answer.get("decisions") or []
    if not decisions:
        lines.append("No control was recorded.")
    for decision in decisions:
        lines.append(
            f"Decision {decision.get('request_id')} as of {decision.get('as_of')}, "
            f"outcome {decision.get('outcome')}."
        )
        lines.extend(_render_check_lines(decision.get("checks") or []))
    return lines


def _render_check_lines(checks: list[dict[str, Any]]) -> list[str]:
    if not checks:
        return ["No check was recorded."]
    lines: list[str] = []
    for check in checks:
        policy = ""
        if check.get("policy_id") is not None:
            policy = f" Policy {check.get('policy_id')}, hash {check.get('policy_hash')}."
        lines.append(
            f"- {check.get('result')} {check.get('check')} ({check.get('regime')}): "
            f"{check.get('citation')}.{policy}"
        )
    return lines


def _render_reproduce(answer: Any) -> list[str]:
    if not isinstance(answer, dict):
        return []
    lines: list[str] = []
    checked = answer.get("events_checked", 0)
    if answer.get("chain_intact"):
        lines.append(f"The evidence chain is intact ({checked} events checked).")
    else:
        lines.append(f"The evidence chain is not intact ({checked} events checked).")
    pins = answer.get("pins") or []
    broken = [pin for pin in pins if pin.get("status") not in _OK_PIN_STATUSES]
    if broken:
        lines.append("A pinned hash does not match the record it was taken from.")
    elif pins:
        lines.append(
            "Each pinned hash matches, or the content was erased and the hash remains."
        )
    else:
        lines.append("No pinned hash was recorded on this interaction.")
    for pin in pins:
        lines.append(_render_pin(pin))
    return lines


def _render_pin(pin: dict[str, Any]) -> str:
    kind = pin.get("kind")
    status = pin.get("status")
    if kind == "policy":
        return (
            f"- Policy {pin.get('policy_id')} hash {pin.get('policy_hash')}: {status}."
        )
    if kind == "retrieved_item":
        text = (
            f"- Retrieved item {pin.get('item_id')} content hash {pin.get('content_hash')}, "
            f"embedding model {pin.get('embedding_model')}: {status}."
        )
        if pin.get("content_status") == "erased":
            text += f" Content erased; hash {pin.get('content_hash')} kept."
        return text
    if kind == "model":
        return (
            f"- Model {pin.get('model')} version {pin.get('model_version')} "
            f"from {pin.get('provider')}, template {pin.get('prompt_template_version')}: "
            f"{status}."
        )
    label = "Prompt" if kind == "prompt" else "Response"
    event_type = pin.get("event_type") or ""
    text = (
        f"- {label} hash {pin.get('content_hash')} ({event_type}, "
        f"sequence {pin.get('sequence')}): {status}."
    )
    if status == "erased":
        text += f" Content erased; hash {pin.get('content_hash')} kept."
    return text


def _join(values: Any) -> str:
    if isinstance(values, list) and values:
        return ", ".join(str(value) for value in values)
    return "none recorded"


def _scopes(values: Any) -> str:
    if isinstance(values, dict) and values:
        return ", ".join(f"{key}={value}" for key, value in values.items())
    return "none recorded"


__all__ = [
    "NOT_RECORDED",
    "QUESTIONS",
    "reconstruct",
    "render_reconstruction",
]
