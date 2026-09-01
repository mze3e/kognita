"""The egress guard — what may cross the boundary to an external model.

A binary rule ("classified content may only reach a local provider, otherwise
refuse") is safe but useless: it confines a governed system to whatever model
runs on the box. The guard adds a third outcome.

``ALLOW``
    The destination is permitted for this classification as it stands.
``REDACT``
    Sensitive spans are replaced with tokens, the provider sees only the
    sanitised text, and the tokens are restored in the response.
``DENY``
    No sanitisation makes this lawful for this destination.

Every call emits ``MODEL_CALL`` and ``EGRESS`` evidence carrying the redaction
manifest hash, so "no raw sensitive data left the boundary" is a claim that can
be checked rather than asserted.

.. warning::
   :class:`PatternRedactor` is a floor, not a guarantee. Regexes miss
   unstructured identifiers, names in prose, and anything the pattern set does
   not anticipate. A redactor trusted beyond its accuracy is worse than none, so
   deployments handling real personal data should supply a proper NER-based
   :class:`~kognita.core.protocols.Redactor`. The conformance suite tests the
   *plumbing* — that nothing unredacted escapes the guard — never detection
   recall.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from kognita.core.canonical import canonical_hash
from kognita.core.evidence import EvidenceWriter
from kognita.core.vocabulary import (
    ActorType,
    Classification,
    EgressDecision,
    EventType,
    classification_rank,
)


class EgressDenied(Exception):
    """The guard refused to send this content to this destination."""

    def __init__(self, message: str, *, classification: Classification, destination: str):
        super().__init__(message)
        self.classification = classification
        self.destination = destination


# ── Redaction ────────────────────────────────────────────────────────────────

#: Ordered patterns; earlier entries win where matches overlap.
DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("EMAIL", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("IBAN", r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}\b"),
    ("ISIN", r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b"),
    ("CARD", r"\b(?:\d[ -]*?){13,16}\b"),
    ("PHONE", r"\+\d[\d\s().-]{7,}\d"),
    ("IP", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("ACCOUNT", r"\b[A-Z]{2}-[A-Z]{2}-\d{4,}\b"),
)


class PatternRedactor:
    """Deterministic pattern-based redaction. No dependencies, no model.

    Tokens are stable within a call (``[EMAIL_1]`` always maps back to the same
    address) so a model can reason about "the same person" without being told
    who they are.
    """

    def __init__(
        self,
        patterns: Iterable[tuple[str, str]] = DEFAULT_PATTERNS,
        *,
        extra_terms: Iterable[str] = (),
    ) -> None:
        self._patterns = [(label, re.compile(p)) for label, p in patterns]
        #: Literal strings that must never leave — names, project code names.
        self._extra = [t for t in extra_terms if t]

    def redact(self, text: str) -> tuple[str, dict[str, str]]:
        token_map: dict[str, str] = {}
        seen: dict[str, str] = {}
        counters: dict[str, int] = {}
        result = text

        def substitute(label: str, value: str) -> str:
            if value in seen:
                return seen[value]
            counters[label] = counters.get(label, 0) + 1
            token = f"[{label}_{counters[label]}]"
            seen[value] = token
            token_map[token] = value
            return token

        for term in sorted(self._extra, key=len, reverse=True):
            if term in result:
                result = result.replace(term, substitute("TERM", term))

        for label, pattern in self._patterns:
            result = pattern.sub(lambda m, _l=label: substitute(_l, m.group(0)), result)

        return result, token_map

    def restore(self, text: str, token_map: dict[str, str]) -> str:
        result = text
        for token, value in token_map.items():
            result = result.replace(token, value)
        return result


class NullRedactor:
    """Redacts nothing. For destinations inside the boundary."""

    def redact(self, text: str) -> tuple[str, dict[str, str]]:
        return text, {}

    def restore(self, text: str, token_map: dict[str, str]) -> str:
        return text


# ── Policy ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EgressPolicy:
    """Which classifications may reach a destination, and on what terms.

    Defaults are conservative: internal content may go out as-is, confidential
    content only redacted, restricted content not at all. A deployment that wants
    something else states it explicitly.
    """

    #: Highest classification that may leave unmodified.
    allow_plain_up_to: Classification = Classification.C1
    #: Highest classification that may leave after redaction.
    allow_redacted_up_to: Classification = Classification.C2
    #: Local destinations skip redaction — nothing crosses a boundary.
    trust_local: bool = True

    def evaluate(
        self, classification: Classification, *, destination_is_local: bool
    ) -> EgressDecision:
        if destination_is_local and self.trust_local:
            return EgressDecision.ALLOW
        rank = classification_rank(classification)
        if rank <= classification_rank(self.allow_plain_up_to):
            return EgressDecision.ALLOW
        if rank <= classification_rank(self.allow_redacted_up_to):
            return EgressDecision.REDACT
        return EgressDecision.DENY


@dataclass
class EgressResult:
    """What the guard did, and what came back."""

    decision: EgressDecision
    classification: Classification
    destination: str
    destination_is_local: bool
    sent_text: str = ""
    response: Any = None
    token_map: dict[str, str] = field(default_factory=dict)
    manifest_hash: str = ""

    @property
    def redacted(self) -> bool:
        return self.decision == EgressDecision.REDACT

    @property
    def redacted_span_count(self) -> int:
        return len(self.token_map)


class EgressGuard:
    """Wraps every outbound model call in a classification decision.

    ``send`` takes the call itself as a function, so the guard is the only path
    to the provider — there is no way to reach it that skips the evidence.
    """

    def __init__(
        self,
        *,
        policy: EgressPolicy | None = None,
        redactor: Any = None,
        evidence: EvidenceWriter | None = None,
    ) -> None:
        self.policy = policy or EgressPolicy()
        self.redactor = redactor or PatternRedactor()
        self.evidence = evidence

    def evaluate(
        self, classification: Classification, *, destination_is_local: bool
    ) -> EgressDecision:
        return self.policy.evaluate(
            classification, destination_is_local=destination_is_local
        )

    def send(
        self,
        text: str,
        call: Callable[[str], Any],
        *,
        classification: Classification,
        destination: str,
        destination_is_local: bool,
        session: Any = None,
        correlation_id: str = "",
        actor_id: str = "",
        actor_type: ActorType = ActorType.SYSTEM,
        restore_response: bool = True,
    ) -> EgressResult:
        """Send ``text`` to ``call``, redacting or refusing as policy requires."""
        decision = self.evaluate(
            classification, destination_is_local=destination_is_local
        )

        if decision == EgressDecision.DENY:
            result = EgressResult(
                decision=decision,
                classification=classification,
                destination=destination,
                destination_is_local=destination_is_local,
            )
            self._evidence(session, result, correlation_id, actor_id, actor_type, sent=False)
            raise EgressDenied(
                f"{Classification(classification).value} content may not be sent to "
                f"'{destination}': no redaction makes this destination permissible.",
                classification=classification,
                destination=destination,
            )

        if decision == EgressDecision.REDACT:
            sent_text, token_map = self.redactor.redact(text)
        else:
            sent_text, token_map = text, {}

        response = call(sent_text)

        if token_map and restore_response and isinstance(response, str):
            response = self.redactor.restore(response, token_map)

        result = EgressResult(
            decision=decision,
            classification=classification,
            destination=destination,
            destination_is_local=destination_is_local,
            sent_text=sent_text,
            response=response,
            token_map=token_map,
            # The manifest binds *which* spans were redacted without recording
            # the spans themselves.
            manifest_hash=canonical_hash(sorted(token_map)),
        )
        self._evidence(session, result, correlation_id, actor_id, actor_type, sent=True)
        return result

    def _evidence(
        self,
        session: Any,
        result: EgressResult,
        correlation_id: str,
        actor_id: str,
        actor_type: ActorType,
        *,
        sent: bool,
    ) -> None:
        if self.evidence is None or session is None:
            return
        common = {
            "destination": result.destination,
            "destination_is_local": result.destination_is_local,
            "classification": Classification(result.classification).value,
            "decision": result.decision.value,
            "redacted_spans": result.redacted_span_count,
            "manifest_hash": result.manifest_hash,
        }
        self.evidence.emit(
            session,
            correlation_id=correlation_id,
            event_type=EventType.MODEL_CALL,
            actor_type=actor_type,
            actor_id=actor_id or "egress-guard",
            classification=result.classification,
            payload={**common, "sent": sent, "sent_bytes": len(result.sent_text)},
        )
        self.evidence.emit(
            session,
            correlation_id=correlation_id,
            event_type=EventType.EGRESS,
            actor_type=actor_type,
            actor_id=actor_id or "egress-guard",
            classification=result.classification,
            payload={
                **common,
                "sent": sent,
                # Content is never copied into the evidence plane.
                "note": "Payload content is not copied to the evidence plane.",
            },
        )


__all__ = [
    "EgressGuard",
    "EgressPolicy",
    "EgressResult",
    "EgressDenied",
    "PatternRedactor",
    "NullRedactor",
    "DEFAULT_PATTERNS",
]
