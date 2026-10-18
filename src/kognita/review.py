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

Risk-based review scores each model response on an interaction against the use
case's risk criteria. A high-risk response, and a sampled low-risk response,
go on the review queue. Coverage is the share screened, queued and reviewed.
A confirmed problem records a flag. It does not insert a prohibiting policy.
"""

from __future__ import annotations

import difflib
import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from sqlmodel import Session, select

from kognita.canonical import canonical_hash
from kognita.classify import PatternClassifier, classifier_record
from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.interaction import approval_gate, linked_correlation_ids, require_interaction
from kognita.models import (
    EvidenceEvent,
    HumanBoundary,
    Interaction,
    ReviewQueueItem,
    RiskCriteria,
    RmReview,
    as_utc,
    utcnow,
)
from kognita.origination import do_not_contact_reasons
from kognita.retention import RetentionStore
from kognita.usecases import MATERIALITY, load_register
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


#: When a response is placed on the queue relative to delivery.
QUEUE_TIMING: tuple[str, ...] = ("before delivery", "after delivery")

#: What a reviewer can record. "confirmed problem" is the finding that raises a flag.
REVIEW_FINDINGS: tuple[str, ...] = ("confirmed problem", "cleared")

CITE_CRITERIA = "Risk-based review: risk criteria are defined per use case"
CITE_HIGH = "Risk-based review: high-risk interactions are queued"
CITE_SAMPLE = "Risk-based review: low-risk interactions are sampled"
CITE_MISSING = "Risk-based review: a missing measurement is not a pass"
CITE_FINDING = "Risk-based review: a review outcome is linked to the interaction"
CITE_FLAG = "Outcomes and near misses: a confirmed problem counts toward the circuit breaker"
CITE_COVERAGE = "Risk-based review: coverage is reported per use case and risk tier"
CITE_HOLD = "Risk-based review: a before-delivery review is still open"


def define_risk_criteria(
    session: Session,
    usecase_id: str,
    *,
    product_complexity_at: str,
    vulnerability_indicators: Sequence[str],
    profile_deviation_is_high_risk: bool,
    minimum_classifier_confidence: float,
    timing: str,
    evidence: EvidenceWriter,
    actor_id: str,
    now: datetime | None = None,
) -> RiskCriteria:
    """Record the risk criteria for one use case.

    A later call closes the current version and appends the next. The closed
    row's record is not edited.
    """
    at = now or utcnow()
    usecase = usecase_id.strip()
    who = actor_id.strip()
    if not usecase or not who:
        raise ReviewError(
            "risk criteria name the use case and who defined them "
            f"({CITE_CRITERIA})"
        )
    if load_register(session, as_of=at).get(usecase) is None:
        raise ReviewError(
            f"'{usecase}' is not registered ({CITE_CRITERIA})"
        )
    complexity = product_complexity_at.strip()
    if complexity not in MATERIALITY:
        raise ReviewError(
            "product complexity is low, medium, high, or critical "
            f"({CITE_CRITERIA})"
        )
    if not isinstance(profile_deviation_is_high_risk, bool):
        raise ReviewError(
            "profile deviation is explicitly high-risk or not "
            f"({CITE_CRITERIA})"
        )
    if (
        isinstance(minimum_classifier_confidence, bool)
        or not isinstance(minimum_classifier_confidence, (int, float))
        or not 0 <= float(minimum_classifier_confidence) <= 1
    ):
        raise ReviewError(
            "minimum classifier confidence is a number from 0 to 1 "
            f"({CITE_CRITERIA})"
        )
    when = timing.strip()
    if when not in QUEUE_TIMING:
        raise ReviewError(
            "the queue is before delivery or after delivery "
            f"({CITE_CRITERIA})"
        )
    indicators: list[str] = []
    for item in vulnerability_indicators:
        name = str(item).strip()
        if not name:
            raise ReviewError(
                "a vulnerability indicator has a name "
                f"({CITE_CRITERIA})"
            )
        if name not in indicators:
            indicators.append(name)
    if not indicators:
        raise ReviewError(
            "risk criteria name the vulnerability indicators they watch "
            f"({CITE_CRITERIA})"
        )
    record = {
        "product_complexity_at": complexity,
        "vulnerability_indicators": indicators,
        "profile_deviation_is_high_risk": profile_deviation_is_high_risk,
        "minimum_classifier_confidence": float(minimum_classifier_confidence),
        "timing": when,
    }
    current = _criteria_at(session, usecase, at)
    version = 1 if current is None else current.version + 1
    if current is not None:
        current.effective_to = at
        session.add(current)
    row = RiskCriteria(
        usecase_id=usecase,
        version=version,
        record=record,
        content_hash=canonical_hash(record),
        effective_from=at,
    )
    session.add(row)
    session.flush()
    evidence.emit(
        session,
        correlation_id=f"usecase:{usecase}",
        event_type=EventType.POLICY_CHANGE,
        actor_type=ActorType.HUMAN,
        actor_id=who,
        payload={
            "action": "RISK_CRITERIA",
            "usecase_id": usecase,
            "version": version,
            "content_hash": row.content_hash,
            "record": record,
            "citation": CITE_CRITERIA,
        },
    )
    return row


def screen_interaction(
    session: Session,
    interaction_id: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    product_complexity: str | None = None,
    vulnerability_indicators: Sequence[str] | None = None,
    deviates_from_profile: bool | None = None,
    classifier: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Score every model response on the interaction and queue the high-risk ones.

    A response the gateway already classified is not classified again. A
    response with no classifier record is classified from the retained text.
    A response that cannot be classified is high risk. Low-risk interactions
    are sampled at the use case's review sampling rate. The sample is a
    hash of the interaction and the criteria version, so the same interaction
    is sampled the same way every time.
    """
    at = now or utcnow()
    interaction = require_interaction(session, interaction_id)
    who = actor_id.strip()
    if not who:
        raise ReviewError(
            f"screening names who screened the interaction ({CITE_HIGH})"
        )
    calls = _model_responses(session, interaction, at)
    if not calls:
        return {
            "interaction_id": interaction.id,
            "screened": [],
            "queued": False,
            "citation": CITE_COVERAGE,
        }
    screens: list[dict[str, Any]] = []
    queued = False
    for event in calls:
        screen = _screen_one(
            session,
            interaction,
            event,
            evidence=evidence,
            actor_id=who,
            product_complexity=product_complexity,
            vulnerability_indicators=vulnerability_indicators,
            deviates_from_profile=deviates_from_profile,
            classifier=classifier,
            now=at,
        )
        screens.append(screen)
        queued = queued or bool(screen["queued"])
    return {
        "interaction_id": interaction.id,
        "screened": screens,
        "queued": queued,
        "citation": screens[-1]["citation"],
    }


def screen_response(
    session: Session,
    interaction_id: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    output_hash: str = "",
    response_text: str | None = None,
    product_complexity: str | None = None,
    vulnerability_indicators: Sequence[str] | None = None,
    deviates_from_profile: bool | None = None,
    classifier: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Score one model response. See :func:`screen_interaction`."""
    at = now or utcnow()
    interaction = require_interaction(session, interaction_id)
    who = actor_id.strip()
    if not who:
        raise ReviewError(
            f"screening names who screened the interaction ({CITE_HIGH})"
        )
    calls = _model_responses(session, interaction, at)
    named = output_hash.strip()
    if named:
        calls = [
            event
            for event in calls
            if str((_payload(event).get("response_hash") or "")).strip() == named
        ]
        if not calls:
            raise ReviewError(
                "the output is not a model response on the interaction "
                f"({CITE_MISSING})"
            )
    elif len(calls) > 1:
        raise ReviewError(
            "the interaction has more than one model response; name the output "
            f"({CITE_MISSING})"
        )
    event = calls[0] if calls else None
    if event is None and not response_text:
        raise ReviewError(
            "there is no model response to screen "
            f"({CITE_MISSING})"
        )
    return _screen_one(
        session,
        interaction,
        event,
        evidence=evidence,
        actor_id=who,
        product_complexity=product_complexity,
        vulnerability_indicators=vulnerability_indicators,
        deviates_from_profile=deviates_from_profile,
        classifier=classifier,
        response_text=response_text,
        now=at,
    )


def enqueue_for_review(
    session: Session,
    interaction_id: str,
    *,
    evidence: EvidenceWriter,
    reasons: Sequence[str],
    risk: str,
    citation: str,
    actor_id: str,
    timing: str | None = None,
    response_hash: str = "",
    now: datetime | None = None,
) -> ReviewQueueItem:
    """Place an interaction on the queue, or return the open item.

    An interaction that is already queued is not queued again. A reviewed
    item does not block a later place on the queue. The open item's reasons
    are not rewritten.
    """
    at = now or utcnow()
    interaction = require_interaction(session, interaction_id)
    cleaned = [str(item).strip() for item in reasons if str(item).strip()]
    basis = citation.strip()
    who = actor_id.strip()
    if not cleaned or not basis or not who:
        raise ReviewError(
            f"a review queue item names why it is there and the rule ({CITE_HIGH})"
        )
    if risk not in ("high", "sampled"):
        raise ReviewError(
            f"queue risk {risk!r} is not high or sampled ({CITE_HIGH})"
        )
    open_item = _open_queue_item(session, interaction.id)
    if open_item is not None:
        return open_item
    criteria = _criteria_at(session, interaction.use_case, at) if interaction.use_case else None
    when = (timing or "").strip()
    if not when and criteria is not None:
        when = str((criteria.record or {}).get("timing") or "")
    if when not in QUEUE_TIMING:
        when = "before delivery"
    entry = (
        load_register(session, as_of=at).get(interaction.use_case)
        if interaction.use_case
        else None
    )
    tier = entry.proposed_tier if entry is not None else None
    row = ReviewQueueItem(
        interaction_id=interaction.id,
        usecase_id=interaction.use_case,
        risk_tier=tier,
        risk=risk,
        timing=when,
        reasons=list(cleaned),
        status="queued",
        citation=basis,
        response_hash=response_hash.strip(),
        queued_at=at,
    )
    session.add(row)
    session.flush()
    evidence.emit(
        session,
        correlation_id=interaction.id,
        event_type=EventType.REVIEW,
        actor_type=ActorType.SYSTEM,
        actor_id=who,
        payload={
            "action": "QUEUED",
            "interaction_id": interaction.id,
            "usecase_id": interaction.use_case,
            "risk_tier": tier,
            "risk": risk,
            "timing": when,
            "reasons": list(cleaned),
            "response_hash": response_hash.strip(),
            "queued_at": at.isoformat(),
            "citation": basis,
        },
    )
    return row


def queue_oversight_flags(
    session: Session,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    now: datetime | None = None,
) -> tuple[str, ...]:
    """Queue interactions whose evidence already asks for risk-based review.

    Rubber-stamp flags, outcome events, recorded near misses and RM
    rejections are included. An interaction that is already queued is left
    on the queue.
    """
    at = now or utcnow()
    who = actor_id.strip()
    if not who:
        raise ReviewError(
            f"queueing oversight flags names who queued them ({CITE_HIGH})"
        )
    events = _events_until(session, at)
    reasons_for: dict[str, list[str]] = {}
    for event in events:
        payload = _payload(event)
        interaction_id = str(payload.get("interaction_id") or "")
        if not interaction_id:
            continue
        reason = _oversight_reason(event)
        if not reason:
            continue
        bucket = reasons_for.setdefault(interaction_id, [])
        if reason not in bucket:
            bucket.append(reason)
    queued: list[str] = []
    for interaction_id, reasons in sorted(reasons_for.items()):
        if session.get(Interaction, interaction_id) is None:
            continue
        before = _open_queue_item(session, interaction_id)
        enqueue_for_review(
            session,
            interaction_id,
            evidence=evidence,
            reasons=reasons,
            risk="high",
            citation=CITE_HIGH,
            actor_id=who,
            now=at,
        )
        if before is None:
            queued.append(interaction_id)
    return tuple(queued)


def record_review_finding(
    session: Session,
    interaction_id: str,
    *,
    reviewer: str,
    outcome: str,
    finding: str,
    evidence: EvidenceWriter,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record the review outcome and, for a confirmed problem, a flag.

    The flag is evidence a later circuit breaker can count. This does not
    insert a prohibiting policy. The screening event is not rewritten.
    """
    at = now or utcnow()
    interaction = require_interaction(session, interaction_id)
    who = reviewer.strip()
    text = finding.strip()
    if outcome not in REVIEW_FINDINGS:
        raise ReviewError(
            f"review outcome {outcome!r} is not confirmed problem or cleared "
            f"({CITE_FINDING})"
        )
    if not who or not text:
        raise ReviewError(
            "a review outcome names the reviewer and the finding "
            f"({CITE_FINDING})"
        )
    item = _open_queue_item(session, interaction.id)
    if item is None:
        raise ReviewError(
            f"interaction {interaction.id} is not on the review queue "
            f"({CITE_FINDING})"
        )
    item.status = "reviewed"
    item.reviewed_at = at
    item.finding = outcome
    session.add(item)
    session.flush()
    payload = {
        "action": "REVIEWED",
        "interaction_id": interaction.id,
        "usecase_id": interaction.use_case,
        "queue_id": item.id,
        "outcome": outcome,
        "reviewer": who,
        "finding": text,
        "queued_at": _iso(as_utc(item.queued_at)),
        "reviewed_at": at.isoformat(),
        "risk_tier": item.risk_tier,
        "citation": CITE_FINDING,
    }
    evidence.emit(
        session,
        correlation_id=interaction.id,
        event_type=EventType.REVIEW,
        actor_type=ActorType.HUMAN,
        actor_id=who,
        payload=payload,
    )
    if outcome == "confirmed problem":
        model_version, prompt_version = _versions_for(session, interaction, at)
        evidence.emit(
            session,
            correlation_id=interaction.id,
            event_type=EventType.REVIEW,
            actor_type=ActorType.HUMAN,
            actor_id=who,
            payload={
                "action": "REVIEW_FLAG",
                "interaction_id": interaction.id,
                "usecase_id": interaction.use_case,
                "risk_tier": item.risk_tier,
                "model_version": model_version,
                "model_version_recorded": bool(model_version),
                "prompt_template_version": prompt_version,
                "prompt_template_version_recorded": bool(prompt_version),
                "citation": CITE_FLAG,
            },
        )
    return payload


def review_blocks_delivery(session: Session, interaction_id: str) -> str | None:
    """The citation that holds delivery, or None when delivery is not held.

    A before-delivery item that is still queued holds delivery. A confirmed
    problem on a before-delivery item holds it as well. An after-delivery
    item does not. A cleared before-delivery item does not.
    """
    interaction = session.get(Interaction, interaction_id)
    if interaction is None:
        return (
            f"interaction {interaction_id} is not open ({CITE_HOLD})"
        )
    rows = session.exec(
        select(ReviewQueueItem).where(ReviewQueueItem.interaction_id == interaction.id)
    ).all()
    for row in rows:
        if row.timing != "before delivery":
            continue
        if row.status == "queued" or row.finding == "confirmed problem":
            return f"{CITE_HOLD}. {row.citation}"
    return None


def review_flags(
    session: Session, *, as_of: datetime | None = None
) -> tuple[dict[str, Any], ...]:
    """Flags a confirmed problem raised. The circuit breaker is not tripped here."""
    at = _aware(as_of or utcnow())
    found: list[dict[str, Any]] = []
    for event in _events_until(session, at):
        payload = _payload(event)
        if event.event_type == EventType.REVIEW and payload.get("action") == "REVIEW_FLAG":
            found.append(dict(payload))
    return tuple(found)


def coverage_report(
    session: Session, *, as_of: datetime | None = None
) -> dict[str, Any]:
    """Screened, queued and reviewed shares, per use case and per risk tier.

    The denominator is interactions that have a model response. A share with
    an empty denominator is unmeasured, not zero and not complete. Time to
    review is the mean seconds from queue to review outcome. No completed
    review leaves that time unmeasured.
    """
    at = _aware(as_of or utcnow())
    register = load_register(session, as_of=at)
    events = _events_until(session, at)
    interactions = [
        row
        for row in session.exec(select(Interaction)).all()
        if (opened := as_utc(row.opened_at)) is not None and opened <= at
    ]
    by_use: dict[str, list[Interaction]] = {}
    for row in interactions:
        by_use.setdefault(row.use_case, []).append(row)
    rows = [
        _coverage_row(session, name, group, events, register, at)
        for name, group in sorted(by_use.items())
    ]
    tiers: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        tiers.setdefault(row["risk_tier"], []).append(row)
    by_tier = [
        _sum_coverage(tier, group) for tier, group in sorted(tiers.items(), key=_tier_key)
    ]
    return {
        "at": at.isoformat(),
        "by_use_case": rows,
        "by_risk_tier": by_tier,
        "citation": CITE_COVERAGE,
    }


def _screen_one(
    session: Session,
    interaction: Interaction,
    event: EvidenceEvent | None,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    product_complexity: str | None,
    vulnerability_indicators: Sequence[str] | None,
    deviates_from_profile: bool | None,
    classifier: Any | None,
    now: datetime,
    response_text: str | None = None,
) -> dict[str, Any]:
    payload = _payload(event) if event is not None else {}
    response_hash = str(payload.get("response_hash") or "").strip()
    recorded, classify_reasons = _classifier_reading(
        session, payload, response_text=response_text, classifier=classifier
    )
    criteria = _criteria_at(session, interaction.use_case, now) if interaction.use_case else None
    reasons = list(classify_reasons)
    reasons.extend(
        _criterion_reasons(
            criteria,
            product_complexity=product_complexity,
            vulnerability_indicators=vulnerability_indicators,
            deviates_from_profile=deviates_from_profile,
            confidence=None if recorded is None else recorded.get("confidence"),
        )
    )
    reasons.extend(_oversight_reasons(session, interaction.id, now))
    entry = (
        load_register(session, as_of=now).get(interaction.use_case)
        if interaction.use_case
        else None
    )
    rate = float(entry.review_sampling_rate) if entry is not None else None
    criteria_version = criteria.version if criteria is not None else 0
    high = bool(reasons)
    sampled = False
    if high:
        citation = CITE_MISSING if any(
            "not recorded" in reason or reason == "no risk criteria" or "not classified" in reason
            for reason in reasons
        ) else CITE_HIGH
        # A recognised high-risk hit still cites the queue rule when criteria exist
        # and the only gaps are not measurement gaps.
        if criteria is not None and not any(
            "not recorded" in reason or "not classified" in reason or reason == "no risk criteria"
            for reason in reasons
        ):
            citation = CITE_HIGH
        queued = True
        risk = "high"
    else:
        citation = CITE_SAMPLE
        if rate is None:
            reasons = ["review sampling rate was not recorded"]
            citation = CITE_MISSING
            queued = True
            risk = "high"
        else:
            sampled = _sampled(interaction.id, criteria_version, rate)
            queued = sampled
            risk = "sampled" if sampled else "low"
    screen_key = canonical_hash(
        {
            "interaction_id": interaction.id,
            "response_hash": response_hash,
            "classifier": recorded or {},
            "product_complexity": product_complexity,
            "vulnerability_indicators": list(vulnerability_indicators or []),
            "deviates_from_profile": deviates_from_profile,
            "criteria_version": criteria_version,
            "reasons": reasons,
        }
    )
    prior = _prior_screen(session, interaction.id, screen_key, now)
    if prior is not None:
        return prior
    body: dict[str, Any] = {
        "action": "SCREENED",
        "interaction_id": interaction.id,
        "usecase_id": interaction.use_case,
        "risk_tier": entry.proposed_tier if entry is not None else None,
        "response_hash": response_hash,
        "classifier": recorded or {},
        "reasons": reasons,
        "risk": risk,
        "queued": queued,
        "sampled": sampled,
        "sampling_rate": rate,
        "criteria_version": criteria.version if criteria is not None else None,
        "screen_key": screen_key,
        "timing": str((criteria.record or {}).get("timing") or "") if criteria else "",
        "citation": citation,
    }
    evidence.emit(
        session,
        correlation_id=interaction.id,
        event_type=EventType.REVIEW,
        actor_type=ActorType.SYSTEM,
        actor_id=actor_id,
        payload=body,
    )
    if queued:
        enqueue_for_review(
            session,
            interaction.id,
            evidence=evidence,
            reasons=reasons or ["sampled for review"],
            risk="high" if risk == "high" else "sampled",
            citation=citation,
            actor_id=actor_id,
            response_hash=response_hash,
            now=now,
        )
    return body


def _classifier_reading(
    session: Session,
    payload: Mapping[str, Any],
    *,
    response_text: str | None,
    classifier: Any | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    recorded = payload.get("response_classifier")
    if isinstance(recorded, dict) and _classifier_complete(recorded):
        return dict(recorded), []
    text = response_text
    if text is None:
        response_hash = str(payload.get("response_hash") or "").strip()
        if response_hash:
            text = RetentionStore().read(session, response_hash)
    if text is None:
        return None, ["response was not classified"]
    chosen = classifier if classifier is not None else PatternClassifier()
    return classifier_record(chosen, text), []


def _classifier_complete(recorded: Mapping[str, Any]) -> bool:
    confidence = recorded.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False
    return all(
        str(recorded.get(key) or "").strip()
        for key in ("model", "version", "label", "input_hash")
    )


def _criterion_reasons(
    criteria: RiskCriteria | None,
    *,
    product_complexity: str | None,
    vulnerability_indicators: Sequence[str] | None,
    deviates_from_profile: bool | None,
    confidence: Any,
) -> list[str]:
    if criteria is None:
        return ["no risk criteria"]
    record = criteria.record or {}
    reasons: list[str] = []
    threshold = str(record.get("product_complexity_at") or "")
    supplied = (product_complexity or "").strip()
    if not supplied:
        reasons.append("product complexity was not recorded")
    elif supplied not in MATERIALITY or threshold not in MATERIALITY:
        reasons.append("product complexity was not recognised")
    elif MATERIALITY.index(supplied) >= MATERIALITY.index(threshold):
        reasons.append("product complexity")
    watched = [str(item) for item in record.get("vulnerability_indicators") or []]
    if vulnerability_indicators is None:
        reasons.append("vulnerability indicators were not recorded")
    else:
        for item in vulnerability_indicators:
            name = str(item).strip()
            if name and name in watched:
                reasons.append(f"vulnerability indicator {name}")
    if deviates_from_profile is None:
        reasons.append("deviation from the client profile was not recorded")
    elif record.get("profile_deviation_is_high_risk") and deviates_from_profile:
        reasons.append("deviation from the client profile")
    minimum = record.get("minimum_classifier_confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        reasons.append("classifier confidence was not recorded")
    elif isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        if float(confidence) < float(minimum):
            reasons.append("classifier confidence")
    return reasons


def _oversight_reasons(session: Session, interaction_id: str, at: datetime) -> list[str]:
    reasons: list[str] = []
    for event in _events_until(session, at):
        payload = _payload(event)
        if str(payload.get("interaction_id") or "") != interaction_id:
            continue
        reason = _oversight_reason(event)
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons


def _oversight_reason(event: EvidenceEvent) -> str:
    payload = _payload(event)
    if event.event_type == EventType.FINAL_DECISION:
        stamp = payload.get("rubber_stamp")
        if isinstance(stamp, dict) and stamp.get("flagged"):
            return "rubber-stamp approval"
        if payload.get("outcome") == "reject":
            return "rm rejection"
    if event.event_type == EventType.OUTCOME and payload.get("action") == "RECORDED":
        return f"outcome event: {payload.get('kind')}"
    if event.event_type == EventType.NEAR_MISS and payload.get("action") == "RECORDED":
        return f"near miss: {payload.get('kind')}"
    return ""


def _sampled(interaction_id: str, criteria_version: int, rate: float) -> bool:
    if rate >= 1:
        return True
    if rate <= 0:
        return False
    digest = hashlib.sha256(f"{interaction_id}\n{criteria_version}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return bucket < int(round(rate * 10_000))


def _prior_screen(
    session: Session, interaction_id: str, screen_key: str, at: datetime
) -> dict[str, Any] | None:
    for event in _events_until(session, at):
        if event.event_type != EventType.REVIEW:
            continue
        payload = _payload(event)
        if payload.get("action") != "SCREENED":
            continue
        if payload.get("interaction_id") != interaction_id:
            continue
        if payload.get("screen_key") == screen_key:
            return dict(payload)
    return None


def _model_responses(
    session: Session, interaction: Interaction, at: datetime
) -> list[EvidenceEvent]:
    correlations = linked_correlation_ids(session, interaction.id)
    found: list[EvidenceEvent] = []
    for event in _events_until(session, at):
        if event.event_type != EventType.MODEL_CALL:
            continue
        if event.correlation_id not in correlations:
            continue
        payload = _payload(event)
        if payload.get("sent") is False:
            continue
        if not str(payload.get("response_hash") or "").strip():
            continue
        found.append(event)
    return found


def _criteria_at(session: Session, usecase_id: str, at: datetime) -> RiskCriteria | None:
    if not usecase_id:
        return None
    best: RiskCriteria | None = None
    for row in session.exec(
        select(RiskCriteria).where(RiskCriteria.usecase_id == usecase_id)
    ).all():
        if not row.is_effective(at):
            continue
        if best is None or row.version > best.version:
            best = row
    return best


def _open_queue_item(session: Session, interaction_id: str) -> ReviewQueueItem | None:
    rows = session.exec(
        select(ReviewQueueItem).where(ReviewQueueItem.interaction_id == interaction_id)
    ).all()
    for row in rows:
        if row.status == "queued":
            return row
    return None


def _versions_for(
    session: Session, interaction: Interaction, at: datetime
) -> tuple[str, str]:
    versions: set[str] = set()
    prompts: set[str] = set()
    for event in _model_responses(session, interaction, at):
        payload = _payload(event)
        version = str(payload.get("model_version") or "").strip()
        prompt = str(payload.get("prompt_template_version") or "").strip()
        if version:
            versions.add(version)
        if prompt:
            prompts.add(prompt)
    return (
        next(iter(versions)) if len(versions) == 1 else "",
        next(iter(prompts)) if len(prompts) == 1 else "",
    )


def _coverage_row(
    session: Session,
    usecase_id: str,
    interactions: Sequence[Interaction],
    events: Sequence[EvidenceEvent],
    register: Any,
    at: datetime,
) -> dict[str, Any]:
    entry = register.get(usecase_id) if usecase_id else None
    tier = entry.proposed_tier if entry is not None else None
    population: list[Interaction] = []
    screened: list[Interaction] = []
    for item in interactions:
        hashes = _population_hashes(
            events, item, linked_correlation_ids(session, item.id)
        )
        if not hashes:
            continue
        population.append(item)
        if hashes <= _screened_hashes(events, item):
            screened.append(item)
    queued = [item for item in population if _was_queued(events, item, at)]
    reviewed = [item for item in population if _was_reviewed(events, item, at)]
    spans = _review_spans(events, population, at)
    return _coverage_body(
        usecase_id=usecase_id,
        risk_tier=tier,
        interactions=len(interactions),
        with_model_response=len(population),
        screened=len(screened),
        queued=len(queued),
        reviewed=len(reviewed),
        spans=spans,
    )


def _sum_coverage(tier: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spans: list[float] = []
    for row in rows:
        spans.extend(row.get("review_spans") or [])
    return _coverage_body(
        usecase_id="",
        risk_tier=tier,
        interactions=sum(int(row["interactions"]) for row in rows),
        with_model_response=sum(int(row["with_model_response"]) for row in rows),
        screened=sum(int(row["screened"]) for row in rows),
        queued=sum(int(row["queued"]) for row in rows),
        reviewed=sum(int(row["reviewed"]) for row in rows),
        spans=spans,
    )


def _coverage_body(
    *,
    usecase_id: str,
    risk_tier: Any,
    interactions: int,
    with_model_response: int,
    screened: int,
    queued: int,
    reviewed: int,
    spans: Sequence[float],
) -> dict[str, Any]:
    def share(numerator: int) -> float | None:
        if with_model_response == 0:
            return None
        return numerator / with_model_response

    return {
        "usecase_id": usecase_id,
        "risk_tier": risk_tier,
        "interactions": interactions,
        "with_model_response": with_model_response,
        "screened": screened,
        "queued": queued,
        "reviewed": reviewed,
        "share_screened": share(screened),
        "share_queued": share(queued),
        "share_reviewed": share(reviewed),
        "time_to_review_seconds": _mean(spans),
        "review_spans": list(spans),
        "unmeasured": [
            name
            for name, value in (
                ("share_screened", share(screened)),
                ("share_queued", share(queued)),
                ("share_reviewed", share(reviewed)),
                ("time_to_review_seconds", _mean(spans)),
            )
            if value is None
        ],
        "citation": CITE_COVERAGE,
    }


def _screened_hashes(events: Sequence[EvidenceEvent], interaction: Interaction) -> set[str]:
    found: set[str] = set()
    for event in events:
        if event.event_type != EventType.REVIEW:
            continue
        payload = _payload(event)
        if payload.get("action") != "SCREENED":
            continue
        if payload.get("interaction_id") != interaction.id:
            continue
        response_hash = str(payload.get("response_hash") or "").strip()
        if response_hash:
            found.add(response_hash)
    return found


def _population_hashes(
    events: Sequence[EvidenceEvent],
    interaction: Interaction,
    correlations: set[str],
) -> set[str]:
    found: set[str] = set()
    for event in events:
        if event.event_type != EventType.MODEL_CALL:
            continue
        if event.correlation_id not in correlations:
            continue
        payload = _payload(event)
        if payload.get("sent") is False:
            continue
        response_hash = str(payload.get("response_hash") or "").strip()
        if response_hash:
            found.add(response_hash)
    return found


def _was_queued(events: Sequence[EvidenceEvent], interaction: Interaction, at: datetime) -> bool:
    for event in events:
        if event.event_type != EventType.REVIEW:
            continue
        payload = _payload(event)
        if payload.get("action") != "QUEUED":
            continue
        if payload.get("interaction_id") != interaction.id:
            continue
        when = _parse_time(payload.get("queued_at")) or as_utc(event.recorded_at)
        if when is not None and when <= at:
            return True
    return False


def _was_reviewed(events: Sequence[EvidenceEvent], interaction: Interaction, at: datetime) -> bool:
    for event in events:
        if event.event_type != EventType.REVIEW:
            continue
        payload = _payload(event)
        if payload.get("action") != "REVIEWED":
            continue
        if payload.get("interaction_id") != interaction.id:
            continue
        when = _parse_time(payload.get("reviewed_at")) or as_utc(event.recorded_at)
        if when is not None and when <= at:
            return True
    return False


def _review_spans(
    events: Sequence[EvidenceEvent],
    interactions: Sequence[Interaction],
    at: datetime,
) -> list[float]:
    ids = {item.id for item in interactions}
    spans: list[float] = []
    for event in events:
        payload = _payload(event)
        if event.event_type != EventType.REVIEW or payload.get("action") != "REVIEWED":
            continue
        if payload.get("interaction_id") not in ids:
            continue
        end = _parse_time(payload.get("reviewed_at"))
        start = _parse_time(payload.get("queued_at"))
        if start is None or end is None or end > at or end < start:
            continue
        spans.append((end - start).total_seconds())
    return spans


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _tier_key(item: tuple[Any, Any]) -> tuple[int, int]:
    tier = item[0]
    if tier is None:
        return (1, 0)
    return (0, int(tier))


def _events_until(session: Session, at: datetime) -> list[EvidenceEvent]:
    rows = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    kept: list[EvidenceEvent] = []
    for row in rows:
        recorded = as_utc(row.recorded_at)
        if recorded is not None and recorded <= at:
            kept.append(row)
    return kept


def _payload(event: EvidenceEvent) -> dict[str, Any]:
    raw = event.payload
    return raw if isinstance(raw, dict) else {}


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _aware(value)
    try:
        return _aware(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


def _iso(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.isoformat()


__all__ = [
    "CITE_COVERAGE",
    "CITE_CRITERIA",
    "CITE_FINDING",
    "CITE_FLAG",
    "CITE_HIGH",
    "CITE_HOLD",
    "CITE_MISSING",
    "CITE_SAMPLE",
    "FINAL_OUTCOMES",
    "PACKET_FIELDS",
    "PROVENANCE",
    "QUEUE_TIMING",
    "REVIEW_FINDINGS",
    "REVIEW_OUTCOMES",
    "RUBBER_STAMP_APPROVAL_RATE",
    "RUBBER_STAMP_MIN_DECISIONS",
    "ReviewError",
    "confirm_review",
    "coverage_report",
    "define_human_boundary",
    "define_risk_criteria",
    "enqueue_for_review",
    "queue_oversight_flags",
    "record_final_decision",
    "record_review_finding",
    "review",
    "review_blocks_delivery",
    "review_flags",
    "screen_interaction",
    "screen_response",
    "structured_diff",
]
