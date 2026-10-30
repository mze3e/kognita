"""Special-category controls and human review of vulnerability signals."""
from __future__ import annotations

import json as _json
import re as _re
import unicodedata as _unicodedata
from dataclasses import dataclass as _dataclass
from typing import Callable as _Callable, Iterable as _Iterable
from uuid import uuid4 as _uuid4

from kognita.envelope import Check
from kognita.exceptions import KognitaError
from kognita.vocabulary import ActorType, CheckResult, Classification, EventType

_PATTERNS = {
    "religion": r"religio\w*|muslim|islam\w*|christian\w*|jewish|judaism|hindu\w*|buddhis\w*|church|mosque|synagogue",
    "health": r"health|dementia|alzheimer\w*|cancer|diagnos\w*|diabet\w*|depress\w*|medical|illness|disab\w*",
    "political_views": r"political\w*|republican|democrat\w*|socialist|conservative|liberal|vot(?:e|es|ed|ing)",
    "ethnicity": r"ethnic\w*|racial|race|ancestry|caucasian|african|asian",
    "sexuality": r"sexual\w*|gay|lesbian|bisexual|homosexual|heterosexual|lgbt\w*",
    "vulnerability": r"vulnerab\w*|dementia|alzheimer\w*|confus\w*|coerci\w*|bereave\w*|financial hardship|cognitive\s+decline|cannot understand|struggling to understand|incapacitated|lacks capacity",
}
_DIAGNOSIS = _re.compile(
    r"\b(?:has|have|suffers? from|diagnosed with|is afflicted with)\s+(?:\w+\s+){0,2}"
    r"(?:dementia|alzheimer\w*|cancer|diabetes|depression)\b", _re.I,
)

_DETERMINATION = _re.compile(r"\b(?:client|customer)\s+(?:is\s+(?:vulnerable|incapacitated)|lacks\s+capacity)\b", _re.I)


def _normalize(text: str) -> str:
    return ''.join(c for c in _unicodedata.normalize('NFKC', text)
                   if _unicodedata.category(c) != 'Cf')


def _categories(value: object) -> set[str]:
    if not isinstance(value, (set, frozenset, list, tuple)) or any(
        not isinstance(c, str) or c not in _PATTERNS for c in value
    ):
        raise KognitaError("SENSITIVE_INVALID_CATEGORIES")
    return set(value)


def detect_sensitive(
    text: str, *, classifier: _Callable[[str], _Iterable[str]] | None = None,
    extend: bool = True,
) -> set[str]:
    """Label categories. A bank callable can extend or replace the patterns.

    Classifier errors and unknown labels fail closed. Replacement is an explicit
    deployment choice; the diagnosis guard remains independent.
    """
    if not isinstance(text, str):
        raise KognitaError("SENSITIVE_INVALID_CONTENT")
    normalized = _normalize(text)
    result = {category for category, pattern in _PATTERNS.items()
              if _re.search(r'\b(?:' + pattern + r')\b', normalized, _re.I)}
    if classifier is not None:
        try:
            labels = classifier(normalized)
            if isinstance(labels, str):
                raise ValueError("labels must be a collection")
            extra = _categories(list(labels))
        except Exception as exc:
            raise KognitaError("SENSITIVE_CLASSIFIER_FAILED") from exc
        result = result | extra if extend else extra
    return result


def _check_sensitive(text: str, usecase: object = None, *, classifier=None,
                     extend: bool = True, _flag: bool = False) -> Check:
    """Return an explicit pass or denial, including a stable reason code."""
    reason = "SENSITIVE_PERMITTED"
    result = CheckResult.PASS
    try:
        permitted = _categories(getattr(usecase, 'permitted_sensitive', set()))
        categories = detect_sensitive(text, classifier=classifier, extend=extend)
        if _DIAGNOSIS.search(_normalize(text)):
            reason = "SENSITIVE_UNSUPPORTED_DIAGNOSIS"
        elif _DETERMINATION.search(_normalize(text)):
            reason = "VULNERABILITY_DETERMINATION_BLOCKED"
        elif categories - permitted:
            reason = "SENSITIVE_NOT_PERMITTED"
        elif 'vulnerability' in categories and not _flag:
            reason = "VULNERABILITY_RM_REVIEW_REQUIRED"
    except KognitaError as exc:
        reason = str(exc)
    if reason != "SENSITIVE_PERMITTED":
        result = CheckResult.FAIL
    return Check(check=reason, regime="INTERNAL", result=result,
                 citation="Sensitive attribute controls: " + reason)


def check_sensitive(text: str, usecase: object = None, *, classifier=None,
                    extend: bool = True) -> Check:
    """Gate relationship content, even permitted vulnerability needs RM review."""
    return _check_sensitive(text, usecase, classifier=classifier, extend=extend)


@_dataclass(frozen=True)
class VulnerabilityFlag:
    """A potential indicator routed to the RM, never a determination."""
    flag_id: str
    interaction_id: str
    client_id: str
    signal_hash: str
    review_required: bool = True


def _record(session, interaction, content, action, evidence, *, actor_id,
            actor_type=ActorType.AGENT, **links):
    from kognita.retention import RetentionStore
    digest = RetentionStore().retain_value(
        session, content, kind="sensitive", use_case=interaction.use_case,
        correlation_id=interaction.id,
    )
    event = evidence.emit(
        session, correlation_id=interaction.id, event_type=EventType.REVIEW,
        actor_type=actor_type, actor_id=actor_id, classification=Classification.C3,
        payload=dict(action=action, interaction_id=interaction.id,
                     client_id=interaction.client_id, content_hash=digest, **links),
    )
    return digest, event


def raise_vulnerability_flag(session, interaction_id: str, signal: str, *,
                             usecase, evidence, actor_id: str, classifier=None) -> VulnerabilityFlag:
    """Retain an explained signal and queue it for RM review on the chain."""
    from kognita.models import Interaction
    interaction = session.get(Interaction, interaction_id)
    if interaction is None or interaction.use_case != usecase.usecase_id:
        raise KognitaError("VULNERABILITY_SCOPE_MISMATCH")
    categories = detect_sensitive(signal, classifier=classifier)
    check = _check_sensitive(signal, usecase, classifier=lambda _: categories,
                             extend=False, _flag=True)
    if check.result != CheckResult.PASS:
        raise KognitaError(check.check)
    if not signal.strip() or 'vulnerability' not in categories:
        raise KognitaError("VULNERABILITY_SIGNAL_REQUIRED")
    flag_id = str(_uuid4())
    digest, _ = _record(session, interaction, {"signal": signal},
                         "VULNERABILITY_FLAG", evidence, actor_id=actor_id,
                         flag_id=flag_id, review_required=True, route="RM")
    return VulnerabilityFlag(flag_id, interaction.id, interaction.client_id, digest)


def change_treatment(session, flag: VulnerabilityFlag, treatment: str, *,
                     review_id: int | None = None, usecase, evidence,
                     actor_id: str) -> Check:
    """Authorize only the exact treatment approved in a completed RM packet.

    The existing review API must record a packet whose proposed_action is the
    canonical JSON text of {flag_id, treatment}. This function records authorization; the bank
    applies the approved treatment through its own system.
    """
    from sqlmodel import select
    from kognita.models import EvidenceEvent, Interaction, RmReview
    from kognita.retention import RetentionStore
    from kognita.canonical import canonical_json, hash_text
    from kognita.evidence import verify_chain
    interaction = session.get(Interaction, flag.interaction_id)
    review = session.get(RmReview, review_id) if type(review_id) is int else None
    reason = "VULNERABILITY_RM_REVIEW_REQUIRED"
    events = session.exec(select(EvidenceEvent)).all()
    valid_flag = next((e for e in events if e.payload.get('action') == 'VULNERABILITY_FLAG'
                       and e.payload.get('flag_id') == flag.flag_id
                       and e.payload.get('interaction_id') == flag.interaction_id
                       and e.payload.get('client_id') == flag.client_id
                       and e.payload.get('content_hash') == flag.signal_hash), None)
    try:
        verify_chain(session)
        permitted = 'vulnerability' in _categories(usecase.permitted_sensitive)
        raw = RetentionStore().read(session, review.packet_hash) if review else None
        packet = _json.loads(raw) if raw else {}
        final = next((e for e in events if review and e.event_type == EventType.FINAL_DECISION
                      and e.actor_type == ActorType.HUMAN
                      and e.actor_id == (review.confirmer or review.rm)
                      and e.payload.get('rm') == review.rm
                      and e.payload.get('kind') == 'rm_review'
                      and e.payload.get('interaction_id') == flag.interaction_id
                      and e.payload.get('packet_hash') == review.packet_hash
                      and e.payload.get('outcome') in ('approve', 'modify')), None)
        if (permitted and interaction and interaction.client_id == flag.client_id
                and interaction.use_case == usecase.usecase_id and valid_flag
                and review and review.interaction_id == interaction.id
                and review.status == 'final' and review.outcome in ('approve', 'modify')
                and review.rm and review.decided_at and final
                and final.sequence > valid_flag.sequence
                and raw and hash_text(raw) == review.packet_hash
                and isinstance(treatment, str) and treatment.strip()
                and packet.get('proposed_action') == canonical_json({'flag_id': flag.flag_id, 'treatment': treatment})):
            reason = 'VULNERABILITY_TREATMENT_PERMITTED'
    except Exception:
        pass
    allowed = reason == 'VULNERABILITY_TREATMENT_PERMITTED'
    if interaction:
        _record(session, interaction, {'treatment': treatment},
                'VULNERABILITY_TREATMENT', evidence, actor_id=actor_id,
                flag_id=flag.flag_id, review_id=review_id, reason_code=reason)
    return Check(check=reason, regime='INTERNAL',
                 result=CheckResult.PASS if allowed else CheckResult.FAIL,
                 citation='Vulnerable-client safeguards: ' + reason)


__all__ = ['VulnerabilityFlag', 'detect_sensitive', 'check_sensitive',
           'raise_vulnerability_flag', 'change_treatment']


def _checks(envelope, usecases, attributes):
    from kognita.canonical import canonical_json
    usecase = usecases.get(envelope.purpose) if usecases is not None else None
    text = _json.dumps(_json.loads(canonical_json(envelope.arguments)), ensure_ascii=False)
    if "content" in attributes:
        text += " " + _json.dumps(_json.loads(canonical_json(attributes["content"])), ensure_ascii=False)
    return [check_sensitive(text, usecase)]
