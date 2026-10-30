"""Registered calculation services and fail-closed figure citation.

Kognita does not perform financial math. A bank service does. This module
records that service and blocks a release that presents an authoritative
figure without citing one that covers it.
"""
from __future__ import annotations

import html as _html
import json as _json
import re as _re
import unicodedata as _unicodedata
from dataclasses import asdict as _asdict, dataclass as _dataclass
from urllib.parse import unquote as _unquote

from kognita.canonical import hash_text as _hash_text
from kognita.envelope import Check as _Check
from kognita.evidence import verify_chain as _verify_chain
from kognita.models import EvidenceEvent as _EvidenceEvent
from kognita.retention import RetentionStore as _RetentionStore
from kognita.vocabulary import CheckResult as _CheckResult, EventType as _EventType
from sqlmodel import select as _select


# Canonical figure ids. The patterns are the words that present them.
_FIGURES = {
    "allocation": r"allocations?",
    "concentration": r"concentrations?",
    "exposure": r"exposures?",
    "fx": r"fx|foreign\s+exchange",
    "interest": r"interests?",
    "leverage": r"leveraged?",
    "maturity": r"maturity(?:\s+values?)?",
    "nav": r"nav|n\.a\.v\.?|net\s+asset\s+values?",
    "performance": r"performances?",
    "pnl": r"p\s*[&/]\s*l|pnl|profit\s*(?:and|&|/)\s*loss(?:es)?",
    "suitability": r"suitability(?:\s+scores?)?",
    "tax": r"tax(?:es)?",
    "var": r"var|value[\s-]*at[\s-]*risk",
}
_FIGURE_RE = {
    name: _re.compile(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])")
    for name, pattern in _FIGURES.items()
}
# Letters that look like the keywords after casefold, so a lookalike cannot hide a figure.
_CONFUSABLES = str.maketrans({
    "\u0430": "a",
    "\u0435": "e",
    "\u043a": "k",
    "\u043c": "m",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0442": "t",
    "\u0443": "y",
    "\u0445": "x",
    "\u0455": "s",
    "\u0456": "i",
    "\u03b1": "a",
    "\u03b5": "e",
    "\u03b9": "i",
    "\u03bd": "v",
    "\u03bf": "o",
    "\u03c1": "p",
    "\u03c4": "t",
    "\u03c5": "u",
    "\u03c7": "x",
})
_CITATION = (
    "Claims: an authoritative figure must cite a registered calculation service. "
    "Kognita performs no financial math."
)


def _decode(text: str) -> str:
    current = text
    for _ in range(2):
        decoded = _html.unescape(_unquote(_unquote(current)))
        if decoded == current:
            break
        current = decoded
    return current


def _strip(text: str) -> str:
    normalized = _unicodedata.normalize("NFKC", text)
    kept = "".join(
        char for char in normalized if _unicodedata.category(char) not in {"Cf", "Mn", "Me"}
    )
    return kept.casefold().translate(_CONFUSABLES)


def _form_has_number(text: str) -> bool:
    forms = (text, _unicodedata.normalize("NFKC", text))
    return any(_unicodedata.category(char).startswith("N") for form in forms for char in form)


def _form_kinds(text: str) -> frozenset[str]:
    normalized = _strip(text)
    return frozenset(name for name, pattern in _FIGURE_RE.items() if pattern.search(normalized))


def _scan(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """Keywords tied to a number, and every keyword on the raw and decoded claim.

    Decoding can hide a digit inside a percent-escape or reveal a keyword
    written as an entity. A form contributes its keywords to the numeric set
    only when that same form also contains a number.
    """
    numeric: set[str] = set()
    found: set[str] = set()
    forms = [text]
    decoded = _decode(text)
    if decoded != text:
        forms.append(decoded)
    for form in forms:
        kinds = _form_kinds(form)
        found.update(kinds)
        if _form_has_number(form):
            numeric.update(kinds)
    return frozenset(numeric), frozenset(found)


def _identity(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "@" in value
        or any(char.isspace() or _unicodedata.category(char).startswith("C") for char in value)
    ):
        raise ValueError("C6-INVALID: service id, name, version and owner are required")
    return value


def _prose(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(_unicodedata.category(char).startswith("C") for char in value)
    ):
        raise ValueError("C6-INVALID: service id, name, version and owner are required")
    return value


def _covered(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set, frozenset)) or not value:
        raise ValueError("C6-INVALID: figures_covered must name known figures")
    names = []
    for item in value:
        if not isinstance(item, str) or item not in _FIGURES:
            raise ValueError("C6-INVALID: figures_covered must name known figures")
        names.append(item)
    return tuple(sorted(set(names)))


def _cited(method: object):
    """Return None when no method is cited, False when the citation is malformed."""
    if method is None:
        return None
    if not isinstance(method, str):
        return False
    if method.strip() == "":
        return None
    if method != method.strip() or method.count("@") != 1:
        return False
    service_id, version = method.split("@", 1)
    if (
        not service_id
        or not version
        or any(char.isspace() for char in service_id)
        or any(char.isspace() for char in version)
    ):
        return False
    return service_id, version


@_dataclass(frozen=True)
class CalculationService:
    """A registered service that produces authoritative figures.

    ``figures_covered`` uses the canonical ids: performance, pnl, nav,
    concentration, allocation, fx, leverage, var, maturity, interest,
    exposure, tax, suitability.
    """

    service_id: str
    name: str
    version: str
    owner: str
    figures_covered: tuple[str, ...]

    def __post_init__(self) -> None:
        service_id = _identity(self.service_id)
        if service_id.casefold() == "model":
            raise ValueError("C6-INVALID: model is not a calculation service")
        object.__setattr__(self, "service_id", service_id)
        object.__setattr__(self, "version", _identity(self.version))
        object.__setattr__(self, "name", _prose(self.name))
        object.__setattr__(self, "owner", _prose(self.owner))
        object.__setattr__(self, "figures_covered", _covered(self.figures_covered))


def _load(session) -> dict[tuple[str, str], CalculationService]:
    _verify_chain(session)
    latest: dict[tuple[str, str], CalculationService] = {}
    for event in session.exec(_select(_EvidenceEvent).order_by(_EvidenceEvent.sequence)):
        if event.payload.get("action") != "register_calculation_service":
            continue
        digest = event.payload.get("service_hash", "")
        body = _RetentionStore().read(session, digest) if isinstance(digest, str) else None
        if not isinstance(body, str) or _hash_text(body) != digest:
            raise ValueError("C6-CONTENT: calculation service content missing or changed")
        try:
            service = CalculationService(**_json.loads(body))
        except (TypeError, ValueError, _json.JSONDecodeError) as exc:
            raise ValueError("C6-CONTENT: calculation service content missing or changed") from exc
        latest[(service.service_id, service.version)] = service
    return latest


def register_calculation_service(
    session, service: CalculationService, *, evidence, interaction_id: str,
) -> CalculationService:
    """Retain one calculation service version. The same snapshot may be repeated."""
    if not isinstance(service, CalculationService):
        raise ValueError("C6-INVALID: a calculation service is required")
    current = _load(session)
    old = current.get((service.service_id, service.version))
    if old is not None and old != service:
        raise ValueError("C6-IMMUTABLE: calculation services cannot be rewritten")
    digest = _RetentionStore().retain_value(
        session, _asdict(service), kind="calculation", correlation_id=interaction_id,
    )
    evidence.emit(
        session, correlation_id=interaction_id, event_type=_EventType.CLAIM,
        payload={"action": "register_calculation_service", "service_hash": digest},
    )
    return service


def check_calculation(
    session, claim, *, authoritative_figure: bool = False, evidence=None, interaction_id: str = "",
) -> _Check:
    """Require a covering service when a claim presents an authoritative figure.

    ``authoritative_figure`` is the explicit release flag. A number anywhere
    in the claim together with a figure keyword is the same gate. The check
    cites a service; it does not compute the figure.
    """
    if type(authoritative_figure) is not bool:
        raise ValueError("C6-INVALID: authoritative_figure must be a boolean")
    text = getattr(claim, "claim", None)
    if not isinstance(text, str):
        raise ValueError("C6-INVALID: claim text is required")
    services = _load(session)
    numeric_kinds, all_kinds = _scan(text)
    presented = authoritative_figure or bool(numeric_kinds)
    kinds = all_kinds if authoritative_figure else numeric_kinds
    method = getattr(claim, "calculation_method", "")
    cited = _cited(method)
    service = services.get(cited) if isinstance(cited, tuple) else None
    if not presented:
        reason = "C6-CLEAR"
        service = None
    elif cited is None:
        reason = "C6-UNCITED"
        service = None
    elif service is None:
        reason = "C6-UNREGISTERED"
    elif kinds and not kinds <= set(service.figures_covered):
        reason = "C6-UNCOVERED"
    else:
        reason = "C6-CITED"
    if evidence is not None:
        digest = _RetentionStore().retain_value(
            session,
            {
                "service": _asdict(service) if service is not None else None,
                "calculation_method": method if isinstance(method, str) else "",
                "figures": sorted(kinds if presented else ()),
                "authoritative_figure": authoritative_figure,
                "reason": reason,
            },
            kind="calculation",
            correlation_id=interaction_id,
        )
        claim_id = getattr(claim, "claim_id", "")
        evidence.emit(
            session, correlation_id=interaction_id, event_type=_EventType.CLAIM,
            payload={
                "action": "check_calculation",
                "claim_id": claim_id if isinstance(claim_id, str) else "",
                "calculation_hash": digest,
                "reason": reason,
            },
        )
    passed = reason in ("C6-CLEAR", "C6-CITED")
    return _Check(
        check=reason, regime="INTERNAL",
        result=_CheckResult.PASS if passed else _CheckResult.FAIL,
        citation=_CITATION,
    )


__all__ = ["CalculationService", "register_calculation_service", "check_calculation"]
