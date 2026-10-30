"""Governed client communication.

Sending to a client is its own governed action. The outcome is recorded as a
``CLIENT_COMMUNICATION`` event and is not written back onto the final decision
or the recommendation it came from. A denial is still that event: the draft is
retained by hash and ``sent`` is false.

The control floors below are constants.

- Every send names a communication level and a content provenance label.
- A channel carries client communication only when it is registered with
  archiving and supervision. Personal WhatsApp, Telegram, and personal email
  cannot be registered. A public-lane channel cannot carry a client send.
- A send that does not trace back to a final decision is a denial.
- House view and above need a research attribution.
- Investment discussion and above need a cross-border rule for the recipient,
  and a use case whose tier can be read. An unknown tier is a denial.
- Product recommendation and above need a recorded suitability pass, an RM
  approval (approve or modify), and investment-risk language. That is the step
  from insight to recommendation.
- A transaction is sent only when the final decision is approve and the
  content is the approved text. A modify stays a denial: the full approved
  text is not retained separately from the diff.
- Tier 3 and above, at investment discussion and above, needs approve or
  modify rather than a plain allow. Tier 4, at product recommendation and
  above, needs a confirmer on that decision.
- AI provenance needs a disclosure rule to have decided whether the client
  must be told. The content provenance label selects the trigger.

A public send uses the public lane. It carries a level and a provenance
label, uses a channel registered for that lane, and is denied when a client
is named. A run that has touched client data is denied too. Client data is
an interaction that names a client, or evidence on the run at classification
C2 or above. A run that is not recorded is a denial. External-content taint
and every other public-content tool stay a later slice.

A complaint is recorded by delegating to ``kognita.outcomes.record_outcome``
with kind ``client complaint``. That is the one ``OUTCOME`` event and the
one complaint payload. A client send is held when ``review_blocks_delivery``
returns a citation.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping, Sequence

from sqlmodel import Session, select

from kognita.envelope import Check, Envelope, RuleContext
from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.governance import resolve_outcome
from kognita.interaction import linked_correlation_ids, require_interaction
from kognita.models import (
    Agent,
    ApprovedChannel,
    EvidenceEvent,
    Interaction,
    InteractionLink,
    Policy,
    RmReview,
    RunRecord,
    utcnow,
)
from kognita.origination import do_not_contact_reasons
from kognita.outcomes import record_outcome
from kognita.retention import RetentionStore
from kognita.review import CITE_HOLD, PROVENANCE, review_blocks_delivery
from kognita.rules import CORE_RULES
from kognita.usecases import load_register
from kognita.vocabulary import (
    COMMUNICATION_LEVELS,
    ActorType,
    CheckResult,
    Classification,
    EventType,
    Outcome,
    classification_rank,
)

#: Channels the roadmap says cannot be registered, compared after normalisation.
UNREGISTRABLE_CHANNELS: tuple[str, ...] = (
    "personal whatsapp",
    "telegram",
    "personal email",
)

#: Public-lane names. They are not client channels.
PUBLIC_LANE_CHANNELS: tuple[str, ...] = (
    "linkedin",
    "newsletter",
    "event material",
)

_LEVEL_RANK = {level: index for index, level in enumerate(COMMUNICATION_LEVELS)}
_HOUSE_VIEW = _LEVEL_RANK["house view"]
_DISCUSSION = _LEVEL_RANK["investment discussion"]
_RECOMMENDATION = _LEVEL_RANK["product recommendation"]
_TRANSACTION = _LEVEL_RANK["transaction"]

_ACCEPTING = frozenset({"allow", "approve", "modify"})
_RM_APPROVAL = frozenset({"approve", "modify"})
_PROVENANCE_TRIGGER = {
    "AI": "AI generated content",
    "human with AI assistance": "assisted analysis",
    "AI then human-edited": "materially contributed to a decision",
}
_PUBLIC_LEVELS = frozenset({"general information", "house view"})


class CommunicationError(KognitaError):
    """A communication or a complaint could not be recorded as requested."""


def register_channel(
    session: Session,
    channel: str,
    *,
    evidence: EvidenceWriter,
    human: str,
    citation: str,
    archiving: bool,
    supervision: bool,
    public: bool = False,
    now: datetime | None = None,
) -> ApprovedChannel:
    """Register a channel that meets archiving and supervision.

    Personal WhatsApp, Telegram, and personal email are refused and not
    recorded. A public-lane name is refused as a client channel. An agent
    cannot register a channel.
    """
    del now
    label = channel.strip()
    key = _channel_key(label)
    party = human.strip()
    basis = citation.strip()
    if not label or not key:
        raise CommunicationError(
            "a channel has a name (Governed client communication: approved channel)"
        )
    if not party or not basis:
        raise CommunicationError(
            "a channel registration names the human and the rule "
            "(Governed client communication: approved channel)"
        )
    if not isinstance(archiving, bool) or not isinstance(supervision, bool):
        raise CommunicationError(
            "archiving and supervision are booleans "
            "(Governed client communication: approved channel)"
        )
    if not isinstance(public, bool):
        raise CommunicationError(
            "public is a boolean (Governed client communication: public lane)"
        )
    if _is_agent(session, party):
        raise CommunicationError(
            f"{party} is a registered agent and cannot register a channel "
            "(Governed client communication: approved channel)"
        )
    if key in UNREGISTRABLE_CHANNELS:
        raise CommunicationError(
            f"{label} is not registrable "
            "(Governed client communication: approved channel)"
        )
    if not public and key in PUBLIC_LANE_CHANNELS:
        raise CommunicationError(
            f"{label} is a public-lane channel, not a client channel "
            "(Governed client communication: public lane)"
        )
    if not archiving or not supervision:
        raise CommunicationError(
            "a channel is registered only when it meets archiving and "
            "supervision (Governed client communication: approved channel)"
        )
    row = session.get(ApprovedChannel, key)
    if row is None:
        row = ApprovedChannel(channel=key)
    row.label = label
    row.archiving = archiving
    row.supervision = supervision
    row.public = public
    row.citation = basis
    row.registered_by = party
    session.add(row)
    session.flush()
    evidence.emit(
        session,
        correlation_id=f"channel:{key}",
        event_type=EventType.POLICY_CHANGE,
        actor_type=ActorType.HUMAN,
        actor_id=party,
        payload={
            "action": "APPROVED_CHANNEL",
            "channel": key,
            "label": label,
            "archiving": archiving,
            "supervision": supervision,
            "public": public,
            "citation": basis,
        },
    )
    return row


def send_client_communication(
    session: Session,
    interaction_id: str,
    *,
    evidence: EvidenceWriter,
    channel: str,
    recipient_ref: str,
    content: str,
    level: str,
    provenance: str,
    actor_id: str,
    tone: str = "",
    research_attribution: str = "",
    risk_language: bool = False,
    recipient_jurisdiction: str = "",
    chatbot: bool = False,
    retention: RetentionStore | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Authorise a client send and record the ``CLIENT_COMMUNICATION`` event.

    The return value is the recorded payload, including the outcome. Only an
    ``ALLOW`` was sent.
    """
    at = now or utcnow()
    interaction = require_interaction(session, interaction_id)
    actor = actor_id.strip()
    recipient = recipient_ref.strip()
    if not actor:
        raise CommunicationError(
            "a send names who is sending it "
            "(Governed client communication: client communication)"
        )
    if not recipient:
        raise CommunicationError(
            "a send names the recipient reference "
            "(Governed client communication: client communication)"
        )
    if not isinstance(content, str) or not content.strip():
        raise CommunicationError(
            "a send has content (Governed client communication: client communication)"
        )
    if level not in _LEVEL_RANK:
        raise CommunicationError(
            f"communication level {level!r} is not a recorded level "
            "(Governed client communication: communication level)"
        )
    if provenance not in PROVENANCE:
        raise CommunicationError(
            f"content provenance {provenance!r} is not a recorded label "
            "(Governed client communication: content provenance)"
        )
    if not isinstance(risk_language, bool) or not isinstance(chatbot, bool):
        raise CommunicationError(
            "risk_language and chatbot are booleans "
            "(Governed client communication: content policy)"
        )
    rank = _LEVEL_RANK[level]
    standing_label, claimed = _standing_provenance(session, interaction.id, provenance)
    triggers = _triggers(standing_label, chatbot)
    attributes = {
        "communication_level": level,
        "lane": "client",
        "content": content,
        "tone": tone.strip(),
        "channel": _channel_key(channel),
        "recipient_jurisdiction": recipient_jurisdiction.strip(),
        "risk_language": risk_language,
        "research_attribution": research_attribution.strip(),
        "disclosure_triggers": triggers,
        "use_case": interaction.use_case,
        "principal": actor,
        "client_id": interaction.client_id,
    }
    checks = _client_checks(
        session,
        interaction_id=interaction.id,
        client_id=interaction.client_id,
        use_case=interaction.use_case,
        channel_key=attributes["channel"],
        channel_label=channel.strip(),
        level=level,
        rank=rank,
        provenance=standing_label,
        claimed_provenance=claimed,
        content=content,
        attributes=attributes,
        at=at,
    )
    checks.extend(_policy_checks(session, attributes, at, tool="send_client_communication"))
    checks.extend(_coverage_checks(checks, attributes, rank=rank, lane="client"))
    return _emit_send(
        session,
        evidence,
        retention=retention or RetentionStore(),
        interaction_id=interaction.id,
        client_id=interaction.client_id,
        use_case=interaction.use_case,
        actor=actor,
        channel=attributes["channel"],
        recipient_ref=recipient,
        content=content,
        level=level,
        provenance=standing_label,
        claimed_provenance=claimed,
        lane="client",
        checks=checks,
        at=at,
    )


def send_public_content(
    session: Session,
    *,
    evidence: EvidenceWriter,
    channel: str,
    content: str,
    level: str,
    provenance: str,
    actor_id: str,
    approved_research: str,
    tone: str = "",
    client_id: str = "",
    research_attribution: str = "",
    risk_language: bool = False,
    recipient_jurisdiction: str = "",
    chatbot: bool = False,
    run_id: str = "",
    retention: RetentionStore | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Authorise a public-lane send. A named client is a denial.

    ``run_id`` is the run the content came from. A run that has touched
    client data, and a run that is not recorded, are denials.

    The event is ``CLIENT_COMMUNICATION`` with ``lane`` ``public``. It does
    not answer what was communicated to a client.
    """
    at = now or utcnow()
    actor = actor_id.strip()
    if not actor:
        raise CommunicationError(
            "a public send names who is sending it "
            "(Governed client communication: public lane)"
        )
    if not isinstance(content, str) or not content.strip():
        raise CommunicationError(
            "a public send has content (Governed client communication: public lane)"
        )
    if level not in _LEVEL_RANK:
        raise CommunicationError(
            f"communication level {level!r} is not a recorded level "
            "(Governed client communication: communication level)"
        )
    if provenance not in PROVENANCE:
        raise CommunicationError(
            f"content provenance {provenance!r} is not a recorded label "
            "(Governed client communication: content provenance)"
        )
    if not isinstance(risk_language, bool) or not isinstance(chatbot, bool):
        raise CommunicationError(
            "risk_language and chatbot are booleans "
            "(Governed client communication: content policy)"
        )
    attributes = {
        "communication_level": level,
        "lane": "public",
        "content": content,
        "tone": tone.strip(),
        "channel": _channel_key(channel),
        "recipient_jurisdiction": recipient_jurisdiction.strip(),
        "risk_language": risk_language,
        "research_attribution": research_attribution.strip(),
        "disclosure_triggers": _triggers(provenance, chatbot),
        "use_case": "",
        "principal": actor,
        "client_id": client_id.strip(),
        "approved_research": approved_research.strip(),
    }
    checks = _public_checks(
        session,
        channel_key=attributes["channel"],
        channel_label=channel.strip(),
        level=level,
        provenance=provenance,
        client_id=attributes["client_id"],
        approved_research=attributes["approved_research"],
        run_id=run_id.strip(),
        at=at,
    )
    checks.extend(_policy_checks(session, attributes, at, tool="send_public_content"))
    checks.extend(
        _coverage_checks(checks, attributes, rank=_LEVEL_RANK[level], lane="public")
    )
    return _emit_send(
        session,
        evidence,
        retention=retention or RetentionStore(),
        interaction_id="",
        client_id="",
        use_case="",
        actor=actor,
        channel=attributes["channel"],
        recipient_ref="",
        content=content,
        level=level,
        provenance=provenance,
        claimed_provenance="",
        lane="public",
        checks=checks,
        at=at,
    )


def record_complaint(
    session: Session,
    interaction_id: str,
    *,
    evidence: EvidenceWriter,
    output_hash: str,
    summary: str,
    actor_id: str,
    citation: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record a complaint by delegating to the outcome register.

    The event and the payload are the ones ``record_outcome`` writes for
    kind ``client complaint``. A complaint that cannot name the output, the
    model version, and the approver is refused and nothing is appended.
    """
    return record_outcome(
        session,
        interaction_id,
        kind="client complaint",
        summary=summary,
        evidence=evidence,
        actor_id=actor_id,
        citation=citation,
        output_hash=output_hash,
        now=now,
    )


def _emit_send(
    session: Session,
    evidence: EvidenceWriter,
    *,
    retention: RetentionStore,
    interaction_id: str,
    client_id: str,
    use_case: str,
    actor: str,
    channel: str,
    recipient_ref: str,
    content: str,
    level: str,
    provenance: str,
    claimed_provenance: str,
    lane: str,
    checks: list[Check],
    at: datetime,
) -> dict[str, Any]:
    outcome = resolve_outcome(checks)
    sent = outcome is Outcome.ALLOW
    content_hash = retention.retain_text(
        session,
        content,
        kind="client_communication" if lane == "client" else "public_content",
        use_case=use_case,
        correlation_id=interaction_id or f"public:{channel}",
        now=at,
    )
    final = _latest_final(session, interaction_id) if interaction_id else None
    model = _latest_model(session, interaction_id) if interaction_id else None
    recommendation = _recommendation_link(session, interaction_id) if interaction_id else {}
    payload: dict[str, Any] = {
        "lane": lane,
        "interaction_id": interaction_id,
        "client_id": client_id,
        "channel": channel,
        "recipient_ref": recipient_ref,
        "content_hash": content_hash,
        "communication_level": level,
        "provenance": provenance,
        "outcome": outcome.value,
        "sent": sent,
        "actor_id": actor,
        "recorded_at": at.isoformat(),
        "citation": _basis(checks, outcome),
        "checks": [check.to_dict() for check in checks],
    }
    if claimed_provenance and claimed_provenance != provenance:
        payload["claimed_provenance"] = claimed_provenance
    if final is not None:
        payload["final_decision_sequence"] = final.sequence
        payload["final_decision_outcome"] = _payload(final).get("outcome")
    if recommendation:
        payload.update(recommendation)
    if model is not None:
        body = _payload(model)
        payload["model_call_sequence"] = model.sequence
        payload["model_version"] = body.get("model_version") or ""
        payload["model"] = body.get("model") or ""
        payload["provider"] = body.get("provider") or ""
        payload["response_hash"] = body.get("response_hash") or ""
    correlation = interaction_id or f"public:{channel}"
    evidence.emit(
        session,
        correlation_id=correlation,
        event_type=EventType.CLIENT_COMMUNICATION,
        actor_type=ActorType.AGENT if _is_agent(session, actor) else ActorType.HUMAN,
        actor_id=actor,
        payload=payload,
    )
    return payload


def _client_checks(
    session: Session,
    *,
    interaction_id: str,
    client_id: str,
    use_case: str,
    channel_key: str,
    channel_label: str,
    level: str,
    rank: int,
    provenance: str,
    claimed_provenance: str,
    content: str,
    attributes: Mapping[str, Any],
    at: datetime,
) -> list[Check]:
    checks: list[Check] = [
        _structural(
            "COMMUNICATION_LEVEL",
            True,
            "Governed client communication: communication level",
            f"level {level}",
        )
    ]
    reasons = do_not_contact_reasons(session, interaction_id)
    if reasons:
        joined = ", ".join(reasons)
        checks.append(
            _structural(
                "DO_NOT_CONTACT",
                False,
                f"Origination: do not contact ({joined})",
                "do not contact is in force",
            )
        )
    else:
        checks.append(
            _structural(
                "DO_NOT_CONTACT",
                True,
                "Origination: do not contact",
                "do not contact is not in force",
            )
        )
    from kognita.sources import _distribution_block
    blocked = _distribution_block(session, interaction_id)
    citation = "Claims: a superseded source invalidates open recommendations"
    if blocked:
        citation = f"{citation}: {blocked}"
    checks.append(_structural("SOURCE_CURRENT", blocked is None, citation, ""))
    hold = review_blocks_delivery(session, interaction_id)
    if hold:
        checks.append(
            _structural(
                "REVIEW_HOLD",
                False,
                hold,
                "a before-delivery review holds this send",
            )
        )
    else:
        checks.append(
            _structural(
                "REVIEW_HOLD",
                True,
                CITE_HOLD,
                "no before-delivery review holds this send",
            )
        )
    checks.append(_channel_check(session, channel_key, channel_label, public=False))
    if claimed_provenance and claimed_provenance != provenance:
        checks.append(
            _structural(
                "CONTENT_PROVENANCE",
                False,
                "Governed client communication: content provenance",
                f"the label that stands is {provenance}",
            )
        )
    else:
        checks.append(
            _structural(
                "CONTENT_PROVENANCE",
                True,
                "Governed client communication: content provenance",
                provenance,
            )
        )
    final = _latest_final(session, interaction_id)
    checks.extend(
        _decision_checks(
            session,
            final,
            rank=rank,
            use_case=use_case,
            content=content,
            at=at,
        )
    )
    if rank >= _HOUSE_VIEW and not str(attributes.get("research_attribution") or ""):
        checks.append(
            _structural(
                "RESEARCH_ATTRIBUTION",
                False,
                "Governed client communication: research attribution",
                "research attribution is missing",
            )
        )
    elif rank >= _HOUSE_VIEW:
        checks.append(
            _structural(
                "RESEARCH_ATTRIBUTION",
                True,
                "Governed client communication: research attribution",
                "research attribution present",
            )
        )
    if rank >= _RECOMMENDATION:
        checks.append(_suitability_check(session, interaction_id))
        present = attributes.get("risk_language") is True
        checks.append(
            _structural(
                "RISK_LANGUAGE",
                present,
                "Governed client communication: investment-risk language",
                "present" if present else "investment-risk language is missing",
            )
        )
    return checks


def _decision_checks(
    session: Session,
    final: EvidenceEvent | None,
    *,
    rank: int,
    use_case: str,
    content: str,
    at: datetime,
) -> list[Check]:
    if final is None:
        return [
            _structural(
                "FINAL_DECISION",
                False,
                "Governed client communication: a communication traces back to a final decision",
                "no final decision was recorded",
            )
        ]
    body = _payload(final)
    outcome = str(body.get("outcome") or "")
    checks = [
        _structural(
            "FINAL_DECISION",
            True,
            "Governed client communication: a communication traces back to a final decision",
            f"final decision {final.sequence} is {outcome}",
        )
    ]
    accepting = outcome in _ACCEPTING
    rm = outcome in _RM_APPROVAL
    if outcome == "escalate":
        checks.append(
            Check(
                check="RM_APPROVAL",
                regime="COMMUNICATION",
                result=CheckResult.ESCALATE,
                citation=(
                    "Governed client communication: RM approval — "
                    "the final decision was escalate"
                ),
            )
        )
    elif rank >= _RECOMMENDATION or _tier_requires_rm(session, use_case, rank, at):
        checks.append(
            _structural(
                "RM_APPROVAL",
                rm,
                "Governed client communication: RM approval",
                f"final decision is {outcome}",
            )
        )
    else:
        checks.append(
            _structural(
                "RM_APPROVAL",
                accepting,
                "Governed client communication: RM approval",
                f"final decision is {outcome}",
            )
        )
    entry = load_register(session, as_of=at).get(use_case) if use_case else None
    if rank >= _DISCUSSION and entry is None:
        checks.append(
            _structural(
                "USE_CASE_TIER",
                False,
                "Governed client communication: use-case tier",
                "the use case tier could not be read",
            )
        )
    elif entry is not None and entry.status not in ("pilot", "live"):
        checks.append(
            _structural(
                "USE_CASE_TIER",
                False,
                "Governed client communication: use-case tier",
                f"use case status is {entry.status or 'unset'}",
            )
        )
    elif entry is not None:
        checks.append(
            _structural(
                "USE_CASE_TIER",
                True,
                "Governed client communication: use-case tier",
                f"tier {entry.proposed_tier}",
            )
        )
        if (
            entry.proposed_tier >= 4
            and rank >= _RECOMMENDATION
            and not str(body.get("confirmer") or "").strip()
            and not _confirmed(body)
        ):
            checks.append(
                _structural(
                    "TWO_SIGNATURE",
                    False,
                    "Governed client communication: two-signature approval",
                    "the final decision names no confirmer",
                )
            )
        elif entry.proposed_tier >= 4 and rank >= _RECOMMENDATION:
            checks.append(
                _structural(
                    "TWO_SIGNATURE",
                    True,
                    "Governed client communication: two-signature approval",
                    "a confirmer is named",
                )
            )
    if rank >= _TRANSACTION:
        checks.append(_transaction_check(session, body, content))
    return checks


def _confirmed(body: Mapping[str, Any]) -> bool:
    """A two-signature final decision records the confirming party."""
    if str(body.get("confirmer") or "").strip():
        return True
    approver = str(body.get("approver") or "").strip()
    rm = str(body.get("rm") or "").strip()
    return bool(approver and rm and approver != rm and body.get("kind") == "rm_review")


def _tier_requires_rm(session: Session, use_case: str, rank: int, at: datetime) -> bool:
    if rank < _DISCUSSION or not use_case:
        return False
    entry = load_register(session, as_of=at).get(use_case)
    if entry is None:
        return False
    return entry.proposed_tier >= 3


def _transaction_check(
    session: Session, body: Mapping[str, Any], content: str
) -> Check:
    citation = "Governed client communication: transaction text"
    if body.get("outcome") != "approve":
        return _structural(
            "TRANSACTION_TEXT",
            False,
            citation,
            "a transaction is the approved text",
        )
    rendered_hash = str(body.get("rendered_hash") or "")
    if not rendered_hash:
        return _structural(
            "TRANSACTION_TEXT",
            False,
            citation,
            "the approved text was not retained",
        )
    stored = RetentionStore().read(session, rendered_hash)
    if stored is None or stored != content:
        return _structural(
            "TRANSACTION_TEXT",
            False,
            citation,
            "the content is not the approved text",
        )
    return _structural("TRANSACTION_TEXT", True, citation, "the content is the approved text")


def _suitability_check(session: Session, interaction_id: str) -> Check:
    citation = "Governed client communication: suitability"
    if _suitability_passed(session, interaction_id):
        return _structural("SUITABILITY", True, citation, "a suitability check passed")
    return _structural(
        "SUITABILITY",
        False,
        citation,
        "no suitability check passed on this interaction",
    )


def _suitability_passed(session: Session, interaction_id: str) -> bool:
    ids = linked_correlation_ids(session, interaction_id)
    events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    for event in events:
        payload = _payload(event)
        if event.correlation_id not in ids and payload.get("interaction_id") != interaction_id:
            continue
        if _checks_mention_suitability(payload.get("checks")):
            return True
        packet_hash = payload.get("packet_hash")
        if isinstance(packet_hash, str) and packet_hash:
            raw = RetentionStore().read(session, packet_hash)
            if raw and _packet_mentions_suitability(raw):
                return True
    return False


def _checks_mention_suitability(checks: Any) -> bool:
    if not isinstance(checks, list):
        return False
    for item in checks:
        if not isinstance(item, dict):
            continue
        result = str(item.get("result") or "")
        if result and result != CheckResult.PASS.value:
            continue
        text = f"{item.get('check') or ''} {item.get('citation') or ''}".casefold()
        if "suitability" in text:
            return True
    return False


def _packet_mentions_suitability(raw: str) -> bool:
    try:
        packet = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(packet, dict):
        return False
    return _checks_mention_suitability(packet.get("policy_checks"))


def _public_checks(
    session: Session,
    *,
    channel_key: str,
    channel_label: str,
    level: str,
    provenance: str,
    client_id: str,
    approved_research: str,
    run_id: str,
    at: datetime,
) -> list[Check]:
    del at
    checks = [
        _structural(
            "COMMUNICATION_LEVEL",
            level in _PUBLIC_LEVELS,
            "Governed client communication: communication level",
            f"public lane level {level}",
        ),
        _structural(
            "CONTENT_PROVENANCE",
            True,
            "Governed client communication: content provenance",
            provenance,
        ),
        _channel_check(session, channel_key, channel_label, public=True),
    ]
    if client_id:
        checks.append(
            _structural(
                "PUBLIC_LANE",
                False,
                "Governed client communication: public lane",
                "public content does not carry a client",
            )
        )
    else:
        checks.append(
            _structural(
                "PUBLIC_LANE",
                True,
                "Governed client communication: public lane",
                "no client is named",
            )
        )
    checks.append(
        _structural(
            "APPROVED_RESEARCH",
            bool(approved_research),
            "Governed client communication: public lane",
            "approved research" if approved_research else "approved research is missing",
        )
    )
    if run_id:
        checks.append(_public_run_check(session, run_id))
    return checks


def _public_run_check(session: Session, run_id: str) -> Check:
    """A run that has touched client data cannot reach the public lane."""
    citation = "Governed client communication: public lane"
    if session.get(RunRecord, run_id) is None:
        return _structural(
            "PUBLIC_LANE",
            False,
            citation,
            "the run is not recorded, so the public lane is denied",
        )
    if _run_touched_client_data(session, run_id):
        return _structural(
            "PUBLIC_LANE",
            False,
            citation,
            "a run that has touched client data cannot reach the public lane",
        )
    return _structural(
        "PUBLIC_LANE",
        True,
        citation,
        "the run has not touched client data",
    )


def _run_touched_client_data(session: Session, run_id: str) -> bool:
    """Whether evidence shows this run met a client or client-identifying data.

    An interaction names a client, so a run on one has touched client data.
    Classification C2 and above is client-identifying. An event counts when
    it names the run or sits on a correlation attached to the run.
    """
    links = session.exec(
        select(InteractionLink).where(InteractionLink.run_id == run_id)
    ).all()
    correlations: set[str] = set()
    for link in links:
        if link.correlation_id:
            correlations.add(link.correlation_id)
        if not link.interaction_id:
            continue
        interaction = session.get(Interaction, link.interaction_id)
        if interaction is not None and interaction.client_id.strip():
            return True
    events = session.exec(select(EvidenceEvent)).all()
    for event in events:
        named = event.run_id == run_id or event.correlation_id in correlations
        if not named:
            continue
        if _client_classification(event.classification):
            return True
        payload = _payload(event)
        items = payload.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and _client_classification(item.get("classification")):
                    return True
        client = payload.get("client_id")
        if isinstance(client, str) and client.strip():
            return True
    return False


def _client_classification(value: Any) -> bool:
    """C2 and above identify a client. An unreadable class fails closed."""
    if value is None or value == "":
        return False
    try:
        return classification_rank(value) >= classification_rank(Classification.C2)
    except (KeyError, ValueError):
        return True


def _coverage_checks(
    checks: Sequence[Check],
    attributes: Mapping[str, Any],
    *,
    rank: int,
    lane: str,
) -> list[Check]:
    names = {check.check for check in checks}
    extra: list[Check] = []
    if not any(name.startswith("CONTENT_POLICY") for name in names):
        extra.append(
            _structural(
                "CONTENT_POLICY",
                False,
                "Governed client communication: content policy",
                "no content policy is in force for this communication level",
            )
        )
    if lane == "client" and rank >= _DISCUSSION and "CONTENT_POLICY: jurisdiction" not in names:
        extra.append(
            _structural(
                "CROSS_BORDER",
                False,
                "Governed client communication: cross-border",
                "no cross-border rule was evaluated for this recipient",
            )
        )
    triggers = attributes.get("disclosure_triggers") or []
    if triggers and "AI_DISCLOSURE" not in names:
        extra.append(
            _structural(
                "AI_DISCLOSURE",
                False,
                "Governed client communication: AI disclosure",
                "no disclosure rule decided whether the client must be told",
            )
        )
    return extra


def _policy_checks(
    session: Session,
    attributes: Mapping[str, Any],
    at: datetime,
    *,
    tool: str,
) -> list[Check]:
    envelope = Envelope(
        principal=str(attributes.get("principal") or ""),
        purpose=str(attributes.get("use_case") or "client communication"),
        tool=tool,
        subject_type="client" if attributes.get("client_id") else None,
        subject_id=str(attributes.get("client_id") or "") or None,
    )
    context = RuleContext(envelope=envelope, attributes=dict(attributes), as_of=at)
    from kognita.sensitive import check_sensitive
    entry = load_register(session, as_of=at).get(envelope.purpose)
    found: list[Check] = [check_sensitive(attributes.get("content"), entry)]
    rows = session.exec(select(Policy)).all()
    for policy in rows:
        if policy.rule_type not in ("CONTENT_POLICY", "AI_DISCLOSURE"):
            continue
        if not policy.is_effective(at):
            continue
        evaluator = CORE_RULES.get(policy.rule_type)
        if evaluator is None:
            found.append(
                Check(
                    check=f"UNEVALUABLE_POLICY: {policy.rule_type}",
                    regime=policy.regime,
                    result=CheckResult.ESCALATE,
                    citation=policy.citation
                    or f"No evaluator registered for rule type '{policy.rule_type}'",
                    policy_id=policy.id,
                    policy_hash=policy.content_hash or None,
                )
            )
            continue
        found.extend(evaluator(policy, context))
    return found


def _channel_check(
    session: Session, channel_key: str, label: str, *, public: bool
) -> Check:
    citation = (
        "Governed client communication: public lane"
        if public
        else "Governed client communication: approved channel"
    )
    if channel_key in UNREGISTRABLE_CHANNELS:
        return _structural(
            "APPROVED_CHANNEL",
            False,
            citation,
            f"{label or channel_key} is not registrable",
        )
    row = session.get(ApprovedChannel, channel_key)
    if row is None or not row.archiving or not row.supervision:
        return _structural(
            "APPROVED_CHANNEL",
            False,
            citation,
            f"{label or channel_key} is not a registered channel",
        )
    if public != bool(row.public):
        lane = "public" if row.public else "client"
        return _structural(
            "APPROVED_CHANNEL",
            False,
            citation,
            f"{label or channel_key} is registered for the {lane} lane",
        )
    return _structural(
        "APPROVED_CHANNEL",
        True,
        row.citation or citation,
        f"{row.label or channel_key} is registered",
    )


def _standing_provenance(
    session: Session, interaction_id: str, claimed: str
) -> tuple[str, str]:
    rows = session.exec(
        select(RmReview).where(RmReview.interaction_id == interaction_id)
    ).all()
    if not rows:
        return claimed, ""
    latest = max(rows, key=lambda row: row.id or 0)
    standing = latest.provenance or claimed
    if standing != claimed:
        return standing, claimed
    return standing, ""


def _triggers(provenance: str, chatbot: bool) -> list[str]:
    found: list[str] = []
    trigger = _PROVENANCE_TRIGGER.get(provenance)
    if trigger:
        found.append(trigger)
    if chatbot:
        found.append("powers a chatbot")
    return found


def _latest_final(session: Session, interaction_id: str) -> EvidenceEvent | None:
    if not interaction_id:
        return None
    ids = linked_correlation_ids(session, interaction_id)
    rows = session.exec(
        select(EvidenceEvent)
        .where(EvidenceEvent.event_type == EventType.FINAL_DECISION)
        .order_by(EvidenceEvent.sequence)
    ).all()
    chosen: EvidenceEvent | None = None
    for event in rows:
        payload = _payload(event)
        if event.correlation_id in ids or payload.get("interaction_id") == interaction_id:
            chosen = event
    return chosen


def _latest_model(session: Session, interaction_id: str) -> EvidenceEvent | None:
    return _models(session, interaction_id)[-1] if _models(session, interaction_id) else None


def _models(session: Session, interaction_id: str) -> list[EvidenceEvent]:
    ids = linked_correlation_ids(session, interaction_id)
    rows = session.exec(
        select(EvidenceEvent)
        .where(EvidenceEvent.event_type == EventType.MODEL_CALL)
        .order_by(EvidenceEvent.sequence)
    ).all()
    return [
        event
        for event in rows
        if event.correlation_id in ids
        or _payload(event).get("interaction_id") == interaction_id
    ]


def _recommendation_link(session: Session, interaction_id: str) -> dict[str, Any]:
    rows = session.exec(
        select(EvidenceEvent)
        .where(EvidenceEvent.event_type == EventType.ORIGINATION)
        .order_by(EvidenceEvent.sequence)
    ).all()
    link: dict[str, Any] = {}
    for event in rows:
        payload = _payload(event)
        if payload.get("interaction_id") != interaction_id:
            continue
        if payload.get("outcome") == "recommend":
            link["origination_sequence"] = event.sequence
            product = payload.get("product_id") or ""
            if product:
                link["product_id"] = product
    reviews = session.exec(
        select(RmReview).where(RmReview.interaction_id == interaction_id)
    ).all()
    if reviews:
        latest = max(reviews, key=lambda row: row.id or 0)
        link["recommendation_hash"] = latest.rendered_hash
    return link


def _structural(check: str, ok: bool, citation: str, detail: str) -> Check:
    text = citation if not detail else f"{citation} — {detail}"
    return Check(
        check=check,
        regime="COMMUNICATION",
        result=CheckResult.PASS if ok else CheckResult.FAIL,
        citation=text,
    )


def _basis(checks: Sequence[Check], outcome: Outcome) -> str:
    if outcome is Outcome.DENY:
        chosen = [check for check in checks if check.result is CheckResult.FAIL]
    elif outcome is Outcome.ESCALATE:
        chosen = [check for check in checks if check.result is CheckResult.ESCALATE]
    elif outcome is Outcome.HUMAN_APPROVAL:
        chosen = [check for check in checks if check.result is CheckResult.REQUIRES_HUMAN]
    else:
        chosen = list(checks)
    texts = [check.citation for check in chosen if check.citation]
    return "; ".join(texts) if texts else "Governed client communication"


def _channel_key(channel: str) -> str:
    return " ".join(channel.strip().casefold().split())


def _is_agent(session: Session, name: str) -> bool:
    return session.exec(select(Agent).where(Agent.name == name)).first() is not None


def _payload(event: EvidenceEvent) -> dict[str, Any]:
    payload = event.payload
    if isinstance(payload, dict):
        return payload
    return {}


__all__ = [
    "PUBLIC_LANE_CHANNELS",
    "UNREGISTRABLE_CHANNELS",
    "CommunicationError",
    "record_complaint",
    "register_channel",
    "send_client_communication",
    "send_public_content",
]
