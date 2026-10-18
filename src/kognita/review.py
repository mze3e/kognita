"""RM review capture.

What the RM saw is the hash of the rendered recommendation, with the text in
the retention store. What they changed is a structured diff, also retained by
hash. A final decision is recorded on every completed path.

An approval of a recommendation needs a complete packet: proposed action,
reason, evidence, risks, alternatives, the policy checks that ran, and the
agent's confidence. Approve and modify are refused while any of those are
missing. Reject and escalate are still recorded, so a bad packet can be refused.

A two-signature gate marks on ``review`` and records ``FINAL_DECISION`` only
from ``confirm_review``. The confirming party is not the RM who marked.

Approvals faster than the human boundary's threshold, or an approver whose
approval rate reaches 99.9 percent over at least
:data:`RUBBER_STAMP_MIN_DECISIONS` RM reviews, are flagged to the accountable
owner named on the use-case register. The flag is evidence. It does not itself
block the decision. The sample of 10 is the decided floor.

Content keeps a provenance label. AI-generated content that an RM edits is
``AI then human-edited``. Relabelling it as human is refused.

``FINAL_DECISION`` outcomes are the approval outcomes (approve, modify, reject,
escalate) plus ``allow`` and ``deny``, so a path that never opened an approval
packet still has a decision.
"""

from __future__ import annotations

import difflib
from datetime import datetime
from typing import Any, Mapping, Sequence

from sqlmodel import Session, select

from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.interaction import approval_gate, require_interaction
from kognita.models import (
    EvidenceEvent,
    HumanBoundary,
    Interaction,
    RmReview,
    as_utc,
    utcnow,
)
from kognita.origination import do_not_contact_reasons
from kognita.retention import RetentionStore
from kognita.usecases import load_register
from kognita.vocabulary import ActorType, EventType

PACKET_FIELDS: tuple[str, ...] = (
    "proposed_action",
    "reason",
    "evidence",
    "risks",
    "alternatives",
    "policy_checks",
    "agent_confidence",
)

PROVENANCE: tuple[str, ...] = (
    "human",
    "AI",
    "AI then human-edited",
    "human with AI assistance",
)

REVIEW_OUTCOMES: tuple[str, ...] = ("approve", "modify", "reject", "escalate")

#: Paths that complete without an approval packet.
RUN_OUTCOMES: tuple[str, ...] = ("allow", "deny")

FINAL_OUTCOMES: tuple[str, ...] = REVIEW_OUTCOMES + RUN_OUTCOMES

#: The roadmap names 99.9 percent. Ten is the decided sample: a rate over fewer
#: RM reviews is not treated as a rubber stamp.
RUBBER_STAMP_APPROVAL_RATE = 0.999
RUBBER_STAMP_MIN_DECISIONS = 10

_AI_LABELS = frozenset({"AI", "AI then human-edited", "human with AI assistance"})
_ACCEPTING = frozenset({"approve", "modify"})


class ReviewError(KognitaError):
    """An RM review could not be recorded as requested."""


def define_human_boundary(
    session: Session,
    usecase_id: str,
    *,
    intervention: str,
    who: str,
    must_review: str,
    evidence_seen: str,
    on_reject: str,
    minimum_review_seconds: int,
    evidence: EvidenceWriter,
    now: datetime | None = None,
) -> HumanBoundary:
    """Record where a human intervenes, before any approval of that use case.

    ``intervention`` is where they must intervene. ``who`` is that person.
    ``must_review`` is what they review. ``evidence_seen`` is what they see.
    ``on_reject`` is what happens when they reject.
    """
    del now
    usecase = usecase_id.strip()
    fields = {
        "intervention": intervention,
        "who": who,
        "must_review": must_review,
        "evidence_seen": evidence_seen,
        "on_reject": on_reject,
    }
    cleaned = {key: value.strip() for key, value in fields.items()}
    if not usecase or any(not value for value in cleaned.values()):
        raise ReviewError(
            "a human boundary names the use case, where the human intervenes, "
            "who they are, what they review, what they see, and what a "
            "rejection does (RM review: human review is defined before deployment)"
        )
    if (
        isinstance(minimum_review_seconds, bool)
        or not isinstance(minimum_review_seconds, int)
        or minimum_review_seconds < 0
    ):
        raise ReviewError(
            "minimum_review_seconds must be a non-negative integer "
            "(RM review: rubber-stamp approvals are flagged)"
        )
    row = session.get(HumanBoundary, usecase)
    if row is None:
        row = HumanBoundary(usecase_id=usecase)
    row.intervention = cleaned["intervention"]
    row.who = cleaned["who"]
    row.must_review = cleaned["must_review"]
    row.evidence_seen = cleaned["evidence_seen"]
    row.on_reject = cleaned["on_reject"]
    row.minimum_review_seconds = minimum_review_seconds
    session.add(row)
    session.flush()
    evidence.emit(
        session,
        correlation_id=f"boundary:{usecase}",
        event_type=EventType.POLICY_CHANGE,
        actor_type=ActorType.HUMAN,
        actor_id=cleaned["who"],
        payload={
            "action": "HUMAN_BOUNDARY",
            "usecase_id": usecase,
            **cleaned,
            "minimum_review_seconds": minimum_review_seconds,
            "citation": ("RM review: human review is defined before deployment"),
        },
    )
    return row


def review(
    session: Session,
    interaction_id: str,
    *,
    rm: str,
    outcome: str,
    packet: Mapping[str, Any],
    rendered: str,
    provenance: str,
    evidence: EvidenceWriter,
    approved: str | None = None,
    requested_at: datetime | None = None,
    now: datetime | None = None,
    retention: RetentionStore | None = None,
) -> dict[str, Any]:
    """Capture what the RM saw and, when the gate allows, the final decision.

    Approve and modify need a complete packet and a human boundary. On a
    two-signature gate they mark the review and do not record
    ``FINAL_DECISION`` until :func:`confirm_review`.
    """
    at = now or utcnow()
    interaction = require_interaction(session, interaction_id)
    reviewer = rm.strip()
    if not reviewer:
        raise ReviewError("RM review names the RM (RM review: who decided)")
    if outcome not in REVIEW_OUTCOMES:
        raise ReviewError(
            f"review outcome {outcome!r} is not approve, modify, reject, or escalate "
            "(RM review: outcome)"
        )
    if outcome in _ACCEPTING:
        _refuse_contact(interaction_id, session)
    boundary = _boundary(session, interaction)
    gate = approval_gate(session, interaction, at)
    shown = _shown_at(requested_at, at)
    text = _rendered(rendered)
    source_label = _provenance(provenance)
    edited, approved_text = _edited(outcome, text, approved)
    # The label is the one on the rendered recommendation. An edit does not
    # replace an AI label with human.
    label = _surviving_label(source_label, edited=edited)
    gaps = _packet_gaps(packet)
    if outcome in _ACCEPTING and gaps:
        missing = ", ".join(gaps)
        raise ReviewError(
            f"an incomplete approval packet cannot be approved (missing {missing}) "
            "(RM review: approval packet)"
        )
    store = retention or RetentionStore()
    rendered_hash = store.retain_text(
        session,
        text,
        kind="recommendation",
        use_case=interaction.use_case,
        correlation_id=interaction.id,
        now=at,
    )
    diff = structured_diff(text, approved_text)
    diff_hash = store.retain_value(
        session,
        diff,
        kind="rm_diff",
        use_case=interaction.use_case,
        correlation_id=interaction.id,
        now=at,
    )
    packet_body = _packet_body(packet)
    packet_hash = store.retain_value(
        session,
        packet_body,
        kind="approval_packet",
        use_case=interaction.use_case,
        correlation_id=interaction.id,
        now=at,
    )
    marked = gate == "two_signature" and outcome in _ACCEPTING
    row = RmReview(
        interaction_id=interaction.id,
        status="marked" if marked else "final",
        rm=reviewer,
        outcome=outcome,
        packet_hash=packet_hash,
        rendered_hash=rendered_hash,
        diff_hash=diff_hash,
        provenance=label,
        prior_provenance=source_label,
        requested_at=shown,
        decided_at=None if marked else at,
    )
    session.add(row)
    session.flush()
    _emit_seen(
        session,
        evidence,
        interaction=interaction,
        row=row,
        diff=diff,
        two_signature=marked,
    )
    if marked:
        return _view(row, diff, final=False, rubber_stamp=None)
    stamp = _rubber_stamp(
        session,
        approver=reviewer,
        outcome=outcome,
        seconds=(at - shown).total_seconds(),
        threshold=boundary.minimum_review_seconds,
        owner=_owner(session, interaction.use_case, at),
    )
    record_final_decision(
        session,
        interaction.id,
        outcome=outcome,
        evidence=evidence,
        actor_id=reviewer,
        actor_type=ActorType.HUMAN,
        rm=reviewer,
        correlation_id=interaction.id,
        citation=_packet_citation(packet_body),
        now=at,
        kind="rm_review",
        rendered_hash=rendered_hash,
        diff_hash=diff_hash,
        packet_hash=packet_hash,
        provenance=label,
        prior_provenance=source_label,
        rubber_stamp=stamp,
        checks=_packet_checks(packet_body),
    )
    return _view(row, diff, final=True, rubber_stamp=stamp)


def confirm_review(
    session: Session,
    interaction_id: str,
    *,
    confirmer: str,
    evidence: EvidenceWriter,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Second signature on a marked RM review. Records ``FINAL_DECISION``.

    The confirming party must differ from the RM who marked the review. A
    review that is not marked is not confirmed.
    """
    at = now or utcnow()
    interaction = require_interaction(session, interaction_id)
    party = confirmer.strip()
    row = _marked(session, interaction.id)
    if row is None:
        raise ReviewError(
            f"interaction {interaction_id} has no marked review to confirm "
            "(RM review: two-signature approval)"
        )
    if not party or party == row.rm:
        raise ReviewError(
            "the confirming party must differ from the RM who marked the review "
            "(RM review: two-signature approval)"
        )
    boundary = _boundary(session, interaction)
    shown = as_utc(row.requested_at) or at
    row.status = "final"
    row.confirmer = party
    row.decided_at = at
    session.add(row)
    session.flush()
    stamp = _rubber_stamp(
        session,
        approver=party,
        outcome=row.outcome,
        seconds=(at - shown).total_seconds(),
        threshold=boundary.minimum_review_seconds,
        owner=_owner(session, interaction.use_case, at),
    )
    record_final_decision(
        session,
        interaction.id,
        outcome=row.outcome,
        evidence=evidence,
        actor_id=party,
        actor_type=ActorType.HUMAN,
        rm=row.rm,
        correlation_id=interaction.id,
        citation="RM review: two-signature approval",
        now=at,
        kind="rm_review",
        rendered_hash=row.rendered_hash,
        diff_hash=row.diff_hash,
        packet_hash=row.packet_hash,
        provenance=row.provenance,
        prior_provenance=row.prior_provenance,
        rubber_stamp=stamp,
        approver=party,
    )
    return _view(row, [], final=True, rubber_stamp=stamp)


def record_final_decision(
    session: Session,
    interaction_id: str,
    *,
    outcome: str,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    correlation_id: str = "",
    actor_type: ActorType = ActorType.HUMAN,
    rm: str = "",
    now: datetime | None = None,
    kind: str = "run",
    rendered_hash: str = "",
    diff_hash: str = "",
    packet_hash: str = "",
    provenance: str = "",
    prior_provenance: str = "",
    rubber_stamp: Mapping[str, Any] | None = None,
    checks: Sequence[Mapping[str, Any]] | None = None,
    approval_id: int | None = None,
    approver: str = "",
) -> EvidenceEvent:
    """Record who decided, the outcome, and when.

    ``allow`` and ``deny`` cover paths that never opened an approval packet.
    The citation is required: a final decision with no rule is not recorded.
    """
    at = now or utcnow()
    require_interaction(session, interaction_id)
    if outcome not in FINAL_OUTCOMES:
        raise ReviewError(
            f"final decision outcome {outcome!r} is not a recorded outcome "
            "(RM review: final decision)"
        )
    basis = citation.strip()
    if not basis:
        raise ReviewError(
            "a final decision cites the rule that produced it "
            "(RM review: final decision)"
        )
    payload: dict[str, Any] = {
        "interaction_id": interaction_id,
        "outcome": outcome,
        "rm": rm,
        "actor_id": actor_id,
        "approver": approver or rm or actor_id,
        "decided_at": at.isoformat(),
        "citation": basis,
        "kind": kind,
    }
    if rendered_hash:
        payload["rendered_hash"] = rendered_hash
    if diff_hash:
        payload["diff_hash"] = diff_hash
    if packet_hash:
        payload["packet_hash"] = packet_hash
    if provenance:
        payload["provenance"] = provenance
    if prior_provenance:
        payload["prior_provenance"] = prior_provenance
    if rubber_stamp is not None:
        payload["rubber_stamp"] = dict(rubber_stamp)
    if checks:
        payload["checks"] = [dict(item) for item in checks]
    if approval_id is not None:
        payload["approval_id"] = approval_id
    return evidence.emit(
        session,
        correlation_id=correlation_id or interaction_id,
        event_type=EventType.FINAL_DECISION,
        actor_type=actor_type,
        actor_id=actor_id,
        payload=payload,
    )


def structured_diff(before: str, after: str) -> list[dict[str, str]]:
    """Line changes between the recommendation and what was approved.

    Equal lines are omitted. The result is the change, not a prose summary.
    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    changes: list[dict[str, str]] = []
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changes.append(
            {
                "op": tag,
                "before": "\n".join(before_lines[i1:i2]),
                "after": "\n".join(after_lines[j1:j2]),
            }
        )
    return changes


def _refuse_contact(interaction_id: str, session: Session) -> None:
    reasons = do_not_contact_reasons(session, interaction_id)
    if not reasons:
        return
    joined = ", ".join(reasons)
    raise ReviewError(
        f"do not contact is in force ({joined}). The recommendation cannot "
        "be approved. (Origination: do not contact)"
    )


def _boundary(session: Session, interaction: Interaction) -> HumanBoundary:
    if not interaction.use_case:
        raise ReviewError(
            "RM review names a use case whose human boundary was defined "
            "before deployment (RM review: human review is defined before deployment)"
        )
    row = session.get(HumanBoundary, interaction.use_case)
    if row is None:
        raise ReviewError(
            f"use case {interaction.use_case} has no human boundary "
            "(RM review: human review is defined before deployment)"
        )
    return row


def _shown_at(requested_at: datetime | None, at: datetime) -> datetime:
    shown = as_utc(requested_at) if requested_at is not None else at
    if shown is None:
        return at
    if shown > at:
        raise ReviewError(
            "the review was requested after it was decided "
            "(RM review: rubber-stamp approvals are flagged)"
        )
    return shown


def _rendered(rendered: str) -> str:
    if not isinstance(rendered, str) or not rendered.strip():
        raise ReviewError(
            "RM review records the recommendation as rendered "
            "(RM review: what the RM saw)"
        )
    return rendered


def _provenance(label: str) -> str:
    if label not in PROVENANCE:
        raise ReviewError(
            f"content provenance {label!r} is not a recorded label "
            "(RM review: content provenance)"
        )
    return label


def _edited(outcome: str, rendered: str, approved: str | None) -> tuple[bool, str]:
    if outcome not in _ACCEPTING:
        return False, rendered
    text = rendered if approved is None else approved
    edited = text != rendered
    if outcome == "approve" and edited:
        raise ReviewError(
            "approve records the recommendation unchanged; a change is modify "
            "(RM review: what the RM changed)"
        )
    if outcome == "modify" and not edited:
        raise ReviewError(
            "modify records a change; the text is unchanged "
            "(RM review: what the RM changed)"
        )
    if outcome == "modify" and not isinstance(approved, str):
        raise ReviewError(
            "modify records the text the RM approved "
            "(RM review: what the RM changed)"
        )
    return edited, text


def _surviving_label(source: str, *, edited: bool) -> str:
    """The label after an edit. AI content is not rewritten as human."""
    if not edited or source == "human":
        return source
    if source in _AI_LABELS:
        return "AI then human-edited"
    return source


def _packet_gaps(packet: Mapping[str, Any]) -> list[str]:
    if not isinstance(packet, Mapping):
        return list(PACKET_FIELDS)
    gaps: list[str] = []
    for field in PACKET_FIELDS:
        if field not in packet:
            gaps.append(field)
            continue
        value = packet[field]
        if field == "agent_confidence":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                gaps.append(field)
            continue
        if field == "policy_checks":
            if not _policy_checks_ok(value):
                gaps.append(field)
            continue
        if isinstance(value, str):
            if not value.strip():
                gaps.append(field)
        elif isinstance(value, list):
            if not value:
                gaps.append(field)
        elif value is None:
            gaps.append(field)
    return gaps


def _policy_checks_ok(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        citation = item.get("citation")
        if not isinstance(citation, str) or not citation.strip():
            return False
    return True


def _packet_body(packet: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(packet, Mapping):
        return {}
    return {key: packet[key] for key in packet}


def _packet_checks(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = packet.get("policy_checks")
    if not isinstance(raw, list):
        return []
    checks: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("citation"):
            checks.append(dict(item))
    return checks


def _packet_citation(packet: Mapping[str, Any]) -> str:
    citations = [
        str(item.get("citation"))
        for item in _packet_checks(packet)
        if item.get("citation")
    ]
    if citations:
        return "; ".join(citations)
    return "RM review: final decision"


def _emit_seen(
    session: Session,
    evidence: EvidenceWriter,
    *,
    interaction: Interaction,
    row: RmReview,
    diff: list[dict[str, str]],
    two_signature: bool,
) -> None:
    evidence.emit(
        session,
        correlation_id=interaction.id,
        event_type=EventType.INTERACTION,
        actor_type=ActorType.HUMAN,
        actor_id=row.rm,
        payload={
            "action": "RM_REVIEW",
            "interaction_id": interaction.id,
            "rm": row.rm,
            "outcome": row.outcome,
            "rendered_hash": row.rendered_hash,
            "diff_hash": row.diff_hash,
            "packet_hash": row.packet_hash,
            "provenance": row.provenance,
            "prior_provenance": row.prior_provenance,
            "diff_ops": len(diff),
            "two_signature": two_signature,
            "status": row.status,
        },
    )


def _marked(session: Session, interaction_id: str) -> RmReview | None:
    rows = session.exec(
        select(RmReview).where(RmReview.interaction_id == interaction_id)
    ).all()
    for row in rows:
        if row.status == "marked":
            return row
    return None


def _owner(session: Session, use_case: str, at: datetime) -> str:
    if not use_case:
        return ""
    entry = load_register(session, as_of=at).get(use_case)
    if entry is None:
        return ""
    return entry.business_owner


def _rubber_stamp(
    session: Session,
    *,
    approver: str,
    outcome: str,
    seconds: float,
    threshold: int,
    owner: str,
) -> dict[str, Any]:
    outcomes = _prior_outcomes(session, approver)
    outcomes.append(outcome)
    total = len(outcomes)
    approvals = outcomes.count("approve")
    edits = outcomes.count("modify")
    rejections = outcomes.count("reject")
    escalations = outcomes.count("escalate")
    approval_rate = approvals / total
    reasons: list[str] = []
    if seconds < threshold:
        reasons.append("faster than the review threshold")
    if (
        total >= RUBBER_STAMP_MIN_DECISIONS
        and approval_rate >= RUBBER_STAMP_APPROVAL_RATE
    ):
        reasons.append("approval rate")
    return {
        "flagged": bool(reasons),
        "reasons": reasons,
        "accountable_owner": owner,
        "approval_rate": approval_rate,
        "edit_rate": edits / total,
        "rejection_rate": rejections / total,
        "escalation_rate": escalations / total,
        "decisions": total,
        "seconds_to_decision": seconds,
        "threshold_seconds": threshold,
        "citation": "RM review: rubber-stamp approvals are flagged",
    }


def _prior_outcomes(session: Session, approver: str) -> list[str]:
    rows = session.exec(
        select(EvidenceEvent).where(
            EvidenceEvent.event_type == EventType.FINAL_DECISION
        )
    ).all()
    found: list[str] = []
    for event in rows:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if payload.get("kind") != "rm_review":
            continue
        if payload.get("approver") != approver:
            continue
        outcome = payload.get("outcome")
        if outcome in REVIEW_OUTCOMES:
            found.append(str(outcome))
    return found


def _view(
    row: RmReview,
    diff: list[dict[str, str]],
    *,
    final: bool,
    rubber_stamp: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "interaction_id": row.interaction_id,
        "status": row.status,
        "final": final,
        "rm": row.rm,
        "confirmer": row.confirmer,
        "outcome": row.outcome,
        "rendered_hash": row.rendered_hash,
        "diff_hash": row.diff_hash,
        "packet_hash": row.packet_hash,
        "provenance": row.provenance,
        "prior_provenance": row.prior_provenance,
        "diff": diff,
    }
    if rubber_stamp is not None:
        body["rubber_stamp"] = dict(rubber_stamp)
    return body


__all__ = [
    "FINAL_OUTCOMES",
    "PACKET_FIELDS",
    "PROVENANCE",
    "REVIEW_OUTCOMES",
    "RUBBER_STAMP_APPROVAL_RATE",
    "RUBBER_STAMP_MIN_DECISIONS",
    "ReviewError",
    "confirm_review",
    "define_human_boundary",
    "record_final_decision",
    "review",
    "structured_diff",
]
