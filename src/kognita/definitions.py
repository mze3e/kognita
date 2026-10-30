"""Governed business definitions.

The bank's catalogue owns what a term means. Kognita stores the reference,
owner, version and effective date, and will not release a claim that uses a
registered term unless the claim cites the version in force.
"""
from __future__ import annotations

import json as _json
import re as _re
import unicodedata as _unicodedata
from dataclasses import asdict as _asdict, dataclass as _dataclass
from datetime import datetime as _datetime, timezone as _timezone
from types import SimpleNamespace as _SimpleNamespace

from sqlmodel import Session as _Session, select as _select

from kognita.canonical import hash_text as _hash_text
from kognita.envelope import Check as _Check
from kognita.evidence import verify_chain as _verify_chain
from kognita.models import (
    EvidenceEvent as _EvidenceEvent,
    RetainedContent as _RetainedContent,
    RmReview as _RmReview,
    as_utc as _as_utc,
)
from kognita.retention import RetentionStore as _RetentionStore
from kognita.vocabulary import CheckResult as _CheckResult, EventType as _EventType


_CITE = "Claims: a governed term must cite the current registered definition"
_PASS = frozenset({"C12-CLEAR", "C12-CITED"})

# Letters that look like Latin but are not. Matching folds them so a term
# cannot be hidden by swapping in a lookalike character.
_CONFUSABLES = str.maketrans(
    {
        "\u0410": "A", "\u0430": "a", "\u0412": "B", "\u0415": "E", "\u0435": "e",
        "\u041d": "H", "\u043d": "h", "\u041a": "K", "\u043a": "k", "\u041c": "M",
        "\u043c": "m", "\u041e": "O", "\u043e": "o", "\u0420": "P", "\u0440": "p",
        "\u0422": "T", "\u0442": "t", "\u0425": "X", "\u0445": "x", "\u0406": "I",
        "\u0456": "i", "\u0391": "A", "\u03b1": "a", "\u0392": "B", "\u0395": "E",
        "\u03b5": "e", "\u0397": "H", "\u03b7": "h", "\u039a": "K", "\u03ba": "k",
        "\u039c": "M", "\u03bc": "m", "\u039f": "O", "\u03bf": "o", "\u03a1": "P",
        "\u03c1": "p", "\u03a4": "T", "\u03c4": "t", "\u03a7": "X", "\u03c7": "x",
        "\u0399": "I", "\u03b9": "i",
    }
)


class _Ambiguous:
    pass


_AMBIGUOUS = _Ambiguous()


def _strip_marks(text: str) -> str:
    return "".join(ch for ch in text if _unicodedata.category(ch) not in {"Cf", "Mn", "Me"})


def _normalize(value: str) -> str:
    """Fold case, width, marks and lookalikes so a disguise still matches."""
    text = _strip_marks(_unicodedata.normalize("NFKD", value))
    text = text.translate(_CONFUSABLES).casefold()
    text = _strip_marks(_unicodedata.normalize("NFKD", text))
    return _re.sub(r"\s+", " ", text).strip()


def _term_key(value: str) -> str:
    return _normalize(value)


def _version_key(value: str) -> str:
    return _normalize(value)


def _token(value: str) -> bool:
    return any(ch.isalnum() for ch in _normalize(value))


def _plain(value, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"C12-INVALID: {label} is required")
    text = _re.sub(r"\s+", " ", _strip_marks(_unicodedata.normalize("NFKC", value))).strip()
    if not text:
        raise ValueError(f"C12-INVALID: {label} is required")
    return text


def _time(value, label: str = "dates"):
    try:
        moment = _datetime.fromisoformat(value) if isinstance(value, str) else value
    except ValueError as exc:
        raise ValueError(f"C12-INVALID: {label} require a timezone") from exc
    if not isinstance(moment, _datetime) or moment.tzinfo is None:
        raise ValueError(f"C12-INVALID: {label} require a timezone")
    return moment.astimezone(_timezone.utc)


@_dataclass(frozen=True)
class GovernedTerm:
    """One registered version of a business term.

    ``definition_ref`` is the catalogue's reference for this version, not the
    ``(term, version)`` citation a claim carries.
    """

    term: str
    definition_ref: str
    owner: str
    version: str
    effective: _datetime | str
    superseded_by: str = ""

    def __post_init__(self) -> None:
        term = _plain(self.term, "term")
        version = _plain(self.version, "version")
        if not _token(term) or not _token(version) or any(ch.isspace() for ch in version):
            raise ValueError("C12-INVALID: term and version are required")
        successor = _plain(self.superseded_by, "superseded_by") if self.superseded_by else ""
        if self.superseded_by and (not _token(successor) or any(ch.isspace() for ch in successor)):
            raise ValueError("C12-INVALID: superseded_by must name a version")
        if successor and _version_key(successor) == _version_key(version):
            raise ValueError("C12-INVALID: a version cannot supersede itself")
        object.__setattr__(self, "term", term)
        object.__setattr__(self, "definition_ref", _plain(self.definition_ref, "definition reference"))
        object.__setattr__(self, "owner", _plain(self.owner, "owner"))
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "effective", _time(self.effective))
        object.__setattr__(self, "superseded_by", successor)


def _events(session: _Session) -> list[_EvidenceEvent]:
    _verify_chain(session)
    return list(session.exec(_select(_EvidenceEvent).order_by(_EvidenceEvent.sequence)))


def _read_term(session: _Session, event: _EvidenceEvent) -> GovernedTerm:
    digest = event.payload.get("term_hash") or ""
    body = _RetentionStore().read(session, digest)
    if not isinstance(body, str) or _hash_text(body) != digest:
        raise ValueError("C12-CONTENT: definition content missing or changed")
    try:
        return GovernedTerm(**_json.loads(body))
    except (TypeError, ValueError, _json.JSONDecodeError) as exc:
        raise ValueError("C12-CONTENT: definition content missing or changed") from exc


def _history(session: _Session, events: list[_EvidenceEvent] | None = None):
    rows = events if events is not None else _events(session)
    found = []
    for event in rows:
        if event.payload.get("action") == "register_term":
            found.append((_as_utc(event.recorded_at), _read_term(session, event)))
    return found


def _latest(history) -> tuple[GovernedTerm, ...]:
    chosen: dict[tuple[str, str], GovernedTerm] = {}
    for _, term in history:
        chosen[(_term_key(term.term), _version_key(term.version))] = term
    return tuple(chosen.values())


def _registered_terms(session: _Session) -> tuple[GovernedTerm, ...]:
    return _latest(_history(session))


def _group(entries) -> dict[str, list[GovernedTerm]]:
    grouped: dict[str, list[GovernedTerm]] = {}
    for term in entries:
        grouped.setdefault(_term_key(term.term), []).append(term)
    return grouped


def _pattern(normalized: str) -> _re.Pattern[str]:
    parts = [part for part in normalized.split(" ") if part]
    single = len(parts) == 1

    def word(token: str) -> str:
        gap = r"[\-._\s]*" if single else r"[\-._]*"
        return gap.join(_re.escape(ch) for ch in token)

    body = (r"[\s\-._]*").join(word(part) for part in parts)
    return _re.compile(rf"(?<![0-9a-z]){body}(?![0-9a-z])")


def _spans(text: str, term: str) -> list[tuple[int, int]]:
    normalized = _normalize(text)
    token = _normalize(term)
    if not normalized or not token:
        return []
    return [match.span() for match in _pattern(token).finditer(normalized)]


def _matching_groups(text: str, groups: dict[str, list[GovernedTerm]]):
    hits = []
    for key, versions in groups.items():
        spans: list[tuple[int, int]] = []
        seen: set[str] = set()
        for version in versions:
            if version.term in seen:
                continue
            seen.add(version.term)
            spans.extend(_spans(text, version.term))
        if spans:
            hits.append((key, versions, spans))
    kept = []
    for key, versions, spans in hits:
        residual = []
        for start, end in spans:
            covered = False
            for other_key, _, other_spans in hits:
                if other_key == key:
                    continue
                if any(os <= start and end <= oe and (oe - os) > (end - start) for os, oe in other_spans):
                    covered = True
                    break
            if not covered:
                residual.append((start, end))
        if residual:
            kept.append((key, versions))
    return kept


def _current(versions: list[GovernedTerm], at: _datetime):
    live = [item for item in versions if item.effective <= at and not item.superseded_by]
    if not live:
        return None
    latest = max(item.effective for item in live)
    chosen = [item for item in live if item.effective == latest]
    if len(chosen) != 1:
        return _AMBIGUOUS
    return chosen[0]


def _find_version(versions: list[GovernedTerm], version: str) -> GovernedTerm | None:
    key = _version_key(version)
    found = [item for item in versions if _version_key(item.version) == key]
    return found[0] if len(found) == 1 else None


def _cited(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    name, version = value
    if not isinstance(name, str) or not isinstance(version, str):
        return None
    if not name.strip() or not version.strip():
        return None
    return name, version


def _citation_reason(matched, cited, at: _datetime) -> str:
    if len(matched) != 1:
        return "C12-MISSING"
    key, versions = matched[0]
    pair = _cited(cited)
    if pair is None or _term_key(pair[0]) != key:
        return "C12-MISSING"
    chosen = _find_version(versions, pair[1])
    if chosen is None:
        return "C12-MISSING"
    if chosen.effective > at:
        return "C12-NOT-EFFECTIVE"
    current = _current(versions, at)
    if current is _AMBIGUOUS:
        return "C12-AMBIGUOUS"
    if current is None or _version_key(chosen.version) != _version_key(current.version):
        return "C12-SUPERSEDED"
    return "C12-CITED"


def _versions_at(history, term_key: str, at: _datetime) -> list[GovernedTerm]:
    chosen: dict[str, GovernedTerm] = {}
    for recorded_at, term in history:
        if recorded_at is not None and recorded_at <= at and _term_key(term.term) == term_key:
            chosen[_version_key(term.version)] = term
    return list(chosen.values())


def _in_force(history, term_key: str, at: _datetime):
    current = _current(_versions_at(history, term_key, at), at)
    if current is _AMBIGUOUS:
        raise ValueError("C12-AMBIGUOUS: more than one current definition version")
    return current


def _resolve(entries: tuple[GovernedTerm, ...], definition_id: str) -> list[GovernedTerm]:
    if not isinstance(definition_id, str) or not definition_id.strip():
        raise ValueError("C12-UNKNOWN: definition is not registered")
    by_term = [item for item in entries if _term_key(item.term) == _term_key(definition_id)]
    if by_term:
        return by_term
    needle = _plain(definition_id, "definition reference")
    by_ref = [item for item in entries if item.definition_ref == needle]
    if not by_ref:
        raise ValueError("C12-UNKNOWN: definition is not registered")
    keys = {_term_key(item.term) for item in by_ref}
    if len(keys) != 1:
        raise ValueError("C12-AMBIGUOUS: definition id matches more than one term")
    chosen = keys.pop()
    return [item for item in entries if _term_key(item.term) == chosen]


def _same_body(old: GovernedTerm, new: GovernedTerm) -> bool:
    return (
        old.term == new.term
        and old.definition_ref == new.definition_ref
        and old.owner == new.owner
        and old.version == new.version
        and old.effective == new.effective
    )


def _reject_spelling(entries, term: GovernedTerm) -> None:
    for item in entries:
        same_term = _term_key(item.term) == _term_key(term.term)
        if same_term and item.term != term.term:
            raise ValueError("C12-INVALID: term identity is already registered with a different spelling")
        if same_term and _version_key(item.version) == _version_key(term.version) and item.version != term.version:
            raise ValueError("C12-INVALID: version identity is already registered with a different spelling")


def register_term(session: _Session, term: GovernedTerm, *, evidence, interaction_id: str) -> GovernedTerm:
    """Retain one governed term version, or append a supersession of that version."""
    if not isinstance(term, GovernedTerm):
        raise ValueError("C12-INVALID: a governed term is required")
    if evidence is None or not isinstance(interaction_id, str) or not interaction_id.strip():
        raise ValueError("C12-INVALID: registration requires evidence and an interaction id")
    entries = _latest(_history(session))
    _reject_spelling(entries, term)
    old = next(
        (
            item for item in entries
            if _term_key(item.term) == _term_key(term.term) and _version_key(item.version) == _version_key(term.version)
        ),
        None,
    )
    if old is not None and old != term:
        if not _same_body(old, term) or (old.superseded_by and old.superseded_by != term.superseded_by):
            raise ValueError("C12-IMMUTABLE: definition versions cannot be rewritten or revived")
        if not term.superseded_by:
            raise ValueError("C12-IMMUTABLE: definition versions cannot be rewritten or revived")
    digest = _RetentionStore().retain_value(
        session, _asdict(term), kind="definition", correlation_id=interaction_id,
    )
    evidence.emit(
        session, correlation_id=interaction_id, event_type=_EventType.CLAIM,
        payload={"action": "register_term", "term_hash": digest},
    )
    return term


def _check_reason(entries, claim, at: _datetime) -> tuple[str, list[str], str]:
    raw = getattr(claim, "claim", None)
    groups = _group(entries)
    if not isinstance(raw, str):
        return ("C12-MISSING" if entries else "C12-CLEAR"), [], ""
    matched = _matching_groups(raw, groups)
    if not matched:
        return "C12-CLEAR", [], ""
    reason = _citation_reason(matched, getattr(claim, "definition_ref", None), at)
    names = [versions[0].term for _, versions in matched]
    current = ""
    if len(matched) == 1:
        live = _current(matched[0][1], at)
        if live is not None and live is not _AMBIGUOUS:
            current = live.version
    return reason, names, current


def check_definition(session: _Session, claim, *, evidence=None, interaction_id: str = "", as_of=None) -> _Check:
    """Pass only when every governed term in the claim cites the version in force."""
    if evidence is not None and (not isinstance(interaction_id, str) or not interaction_id.strip()):
        raise ValueError("C12-INVALID: an evidenced check requires an interaction id")
    moment = _time(as_of or _datetime.now(_timezone.utc))
    entries = _latest(_history(session))
    reason, matched, current = _check_reason(entries, claim, moment)
    if evidence is not None:
        claim_id = getattr(claim, "claim_id", "")
        cited = _cited(getattr(claim, "definition_ref", None))
        digest = _RetentionStore().retain_value(
            session,
            {
                "claim_id": claim_id if isinstance(claim_id, str) else "",
                "cited": list(cited) if cited else None,
                "matched": matched,
                "current_version": current,
                "reason": reason,
                "at": moment.isoformat(),
            },
            kind="definition",
            correlation_id=interaction_id,
        )
        evidence.emit(
            session, correlation_id=interaction_id, event_type=_EventType.CLAIM,
            payload={
                "action": "check_definition",
                "claim_id": claim_id if isinstance(claim_id, str) else "",
                "definition_hash": digest,
                "reason": reason,
            },
        )
    return _Check(
        check=reason, regime="INTERNAL",
        result=_CheckResult.PASS if reason in _PASS else _CheckResult.FAIL,
        citation=_CITE,
    )


def _window(moment: _datetime | None, start, end) -> bool:
    if moment is None:
        raise ValueError("C12-CONTENT: affected content missing or changed")
    if start is not None and moment < start:
        return False
    if end is not None and moment >= end:
        return False
    return True


def _body(session: _Session, digest: str, label: str) -> str:
    body = _RetentionStore().read(session, digest)
    if not isinstance(body, str) or _hash_text(body) != digest:
        raise ValueError(f"C12-CONTENT: {label} content missing or changed")
    return body


def _produced_claim(history, group, term_key: str, claim, when: _datetime):
    known = {_version_key(item.version): item for item in group}
    pair = _cited(getattr(claim, "definition_ref", None))
    if pair is not None and _term_key(pair[0]) == term_key and _version_key(pair[1]) in known:
        return known[_version_key(pair[1])]
    if _matching_groups(claim.claim, {term_key: group}):
        return _in_force(history, term_key, when)
    return None


def _is_previous(produced: GovernedTerm | None, current) -> bool:
    if produced is None:
        return False
    if current is None:
        return True
    return _version_key(produced.version) != _version_key(current.version)


def _claim_when(events, claim_id: str):
    fallback = None
    for event in events:
        payload = event.payload
        if payload.get("claim_id") != claim_id or payload.get("action") not in ("record", "relabel", "release"):
            continue
        if payload.get("action") == "record":
            return event
        if fallback is None:
            fallback = event
    return fallback


def affected_by_definition(
    session: _Session,
    definition_id: str,
    *,
    start=None,
    end=None,
    as_of=None,
) -> dict:
    """List claims, recommendations and communications produced under a previous version."""
    window_start = _time(start, "window dates") if start is not None else None
    window_end = _time(end, "window dates") if end is not None else None
    if window_start is not None and window_end is not None and window_start >= window_end:
        raise ValueError("C12-INVALID: window end must follow the start")
    moment = _time(as_of or _datetime.now(_timezone.utc))
    events = _events(session)
    history = _history(session, events)
    group = _resolve(_latest(history), definition_id)
    current = _current(group, moment)
    if current is _AMBIGUOUS:
        raise ValueError("C12-AMBIGUOUS: more than one current definition version")
    term_key = _term_key(group[0].term)
    from kognita.claims import list_claims as _list_claims

    claims = []
    for claim in _list_claims(session):
        record = _claim_when(events, claim.claim_id)
        if record is None:
            raise ValueError("C12-CONTENT: claim evidence is missing")
        when = _as_utc(record.recorded_at)
        if not _window(when, window_start, window_end):
            continue
        produced = _produced_claim(history, group, term_key, claim, when)
        if not _is_previous(produced, current):
            continue
        claims.append(
            {
                "claim_id": claim.claim_id,
                "client_id": claim.client_id,
                "interaction_id": record.correlation_id,
                "version": produced.version,
                "recorded_at": when.isoformat(),
            }
        )

    recommendations = []
    seen: dict[str, _EvidenceEvent | _SimpleNamespace] = {}
    for event in events:
        rendered = event.payload.get("rendered_hash")
        if not isinstance(rendered, str) or not rendered:
            continue
        row = session.get(_RetainedContent, rendered)
        expected = event.event_type == _EventType.FINAL_DECISION or event.payload.get("action") == "RM_REVIEW"
        if row is None or _hash_text(row.body) != rendered:
            if expected or (row is not None and row.kind == "recommendation"):
                raise ValueError("C12-CONTENT: recommendation content missing or changed")
            continue
        if not expected and row.kind != "recommendation":
            continue
        previous = seen.get(rendered)
        if previous is None or _as_utc(event.recorded_at) < _as_utc(previous.recorded_at):
            seen[rendered] = event
    for row in session.exec(_select(_RmReview)):
        if row.rendered_hash and row.rendered_hash not in seen:
            seen[row.rendered_hash] = _SimpleNamespace(
                recorded_at=row.requested_at,
                correlation_id=row.interaction_id,
                payload={"interaction_id": row.interaction_id},
            )
    for rendered, event in seen.items():
        body = _body(session, rendered, "recommendation")
        when = _as_utc(event.recorded_at)
        if not _matching_groups(body, {term_key: group}) or not _window(when, window_start, window_end):
            continue
        produced = _in_force(history, term_key, when)
        if not _is_previous(produced, current):
            continue
        payload = event.payload if isinstance(event.payload, dict) else {}
        recommendations.append(
            {
                "interaction_id": payload.get("interaction_id") or event.correlation_id,
                "rendered_hash": rendered,
                "version": produced.version,
                "recorded_at": when.isoformat(),
            }
        )

    communications = []
    for event in events:
        if event.event_type != _EventType.CLIENT_COMMUNICATION:
            continue
        digest = event.payload.get("content_hash")
        if not isinstance(digest, str) or not digest:
            raise ValueError("C12-CONTENT: communication content missing or changed")
        body = _body(session, digest, "communication")
        when = _as_utc(event.recorded_at)
        if not _matching_groups(body, {term_key: group}) or not _window(when, window_start, window_end):
            continue
        produced = _in_force(history, term_key, when)
        if not _is_previous(produced, current):
            continue
        communications.append(
            {
                "interaction_id": event.payload.get("interaction_id") or event.correlation_id,
                "client_id": event.payload.get("client_id") or "",
                "content_hash": digest,
                "version": produced.version,
                "recorded_at": when.isoformat(),
                "sequence": event.sequence,
            }
        )

    claims.sort(key=lambda item: (item["recorded_at"], item["claim_id"]))
    recommendations.sort(key=lambda item: (item["recorded_at"], item["rendered_hash"]))
    communications.sort(key=lambda item: (item["recorded_at"], item["sequence"]))
    return {
        "definition_id": definition_id,
        "term": current.term if current is not None else group[0].term,
        "current_version": "" if current is None else current.version,
        "claims": claims,
        "recommendations": recommendations,
        "communications": communications,
    }


__all__ = ["GovernedTerm", "register_term", "check_definition", "affected_by_definition"]
