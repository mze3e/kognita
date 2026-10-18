"""Revocation of one dimension, effective on the next decision.

``kognita revoke`` stops an agent, a model, a provider, a tool, a source, a
client, a use case, or an action without stopping the rest of the platform.
The agent dimension is the kill switch that already exists: this module
engages it and records who may turn it back on. A live-run revoke that only
needs that switch calls :func:`kognita.registry.set_kill_switch` and does not
open a second mechanism.

A revoked dimension denies until an accountable owner reinstates it and cites
the rule. Closing the row is that reinstatement. A provider stays revoked
until every accountable owner of every use case that lists it has cited one;
one owner records a pending approval and leaves the row open. The evidence
is appended; the old row is not rewritten into a permission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from sqlmodel import Session, select

from kognita.envelope import Check, Envelope
from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.models import Agent, ReinstatementApproval, Revocation, as_utc, utcnow
from kognita.registry import set_kill_switch
from kognita.rules import request_field_keys, request_model_keys
from kognita.usecases import load_register
from kognita.vocabulary import (
    ActorType,
    CheckResult,
    Classification,
    EventType,
    classification_rank,
)

DIMENSIONS: tuple[str, ...] = (
    "agent",
    "model",
    "provider",
    "tool",
    "source",
    "client",
    "usecase",
    "action",
)

CITE = "Revocation"


class RevocationError(KognitaError):
    """A revocation or a reinstatement could not be recorded."""


@dataclass(frozen=True)
class OpenRevocation:
    """A revocation in force at one instant. Pure input to ``decide``."""

    revocation_id: int
    dimension: str
    value: str
    citation: str
    accountable_owners: tuple[str, ...]
    limit_usecases: tuple[str, ...]
    below_classification: str


def revoke(
    session: Session,
    dimension: str,
    value: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    owner: str = "",
    limit_usecases: Sequence[str] = (),
    below_classification: str = "",
    now: datetime | None = None,
) -> Revocation:
    """Revoke one dimension. The next decision that matches it is a denial.

    ``limit_usecases`` and ``below_classification`` apply only to a provider.
    ``below_classification`` is exclusive: ``C2`` still permits C0 and C1.
    An agent revocation engages that agent's kill switch.
    """
    at = _aware(now or utcnow())
    kind = _dimension(dimension)
    named = value.strip()
    if not named:
        raise RevocationError(f"{CITE}: a {kind} revocation names the {kind}")
    actor = _human(session, actor_id, "revoke")
    basis = _citation(citation, "revoke")
    limited = tuple(_names(limit_usecases))
    ceiling = below_classification.strip()
    if kind != "provider" and (limited or ceiling):
        raise RevocationError(
            f"{CITE}: a limit to use cases or a classification applies to a provider"
        )
    if ceiling:
        _classification(ceiling)
    if _open_row(session, kind, named, at) is not None:
        raise RevocationError(
            f"{CITE}: {kind} {named} is already revoked"
        )
    if kind == "agent" and _agent(session, named) is None:
        raise RevocationError(f"{CITE}: agent {named} is not registered")
    owners = accountable_owners(
        session, kind, named, explicit=owner, at=at
    )
    row = Revocation(
        dimension=kind,
        value=named,
        citation=basis,
        actor_id=actor,
        accountable_owners=list(owners),
        limit_usecases=list(limited),
        below_classification=ceiling,
        effective_from=at,
    )
    session.add(row)
    session.flush()
    if kind == "agent":
        set_kill_switch(
            session,
            named,
            True,
            evidence=evidence,
            actor_id=actor,
            correlation_id=f"revocation:{kind}:{named}",
            reason=basis,
            now=at,
        )
    _emit(
        session,
        evidence,
        actor=actor,
        action="REVOKE",
        body={
            "dimension": kind,
            "value": named,
            "citation": basis,
            "accountable_owners": list(owners),
            "revocation_id": row.id,
            "limit_usecases": list(limited),
            "below_classification": ceiling,
            "at": at.isoformat(),
        },
    )
    return row


def reinstate(
    session: Session,
    revocation_id: int,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    citation: str,
    now: datetime | None = None,
) -> Revocation:
    """Record one owner's cited reinstatement.

    Any one accountable owner closes an agent, model, tool, source, client,
    use case, or action. A provider stays open until every accountable owner
    of every use case that lists it has cited a reinstatement. One owner
    records ``REINSTATE_PENDING`` and the next decision still denies.
    """
    at = _aware(now or utcnow())
    actor = _human(session, actor_id, "reinstate")
    basis = _citation(citation, "reinstate")
    row = session.get(Revocation, revocation_id)
    if row is None or not row.is_effective(at):
        raise RevocationError(
            f"{CITE}: revocation {revocation_id} is not in force"
        )
    required = _required_owners(session, row, at)
    if actor not in required:
        named = ", ".join(required) or "none"
        raise RevocationError(
            f"{CITE}: {actor} cannot reinstate {row.dimension} {row.value} "
            f"· accountable owner: {named}"
        )
    session.add(
        ReinstatementApproval(
            revocation_id=row.id or 0,
            actor_id=actor,
            citation=basis,
            recorded_at=at,
        )
    )
    session.flush()
    approved = _approved_actors(session, row.id or 0)
    if row.dimension == "provider" and not set(required).issubset(approved):
        still = [name for name in required if name not in approved]
        _emit(
            session,
            evidence,
            actor=actor,
            action="REINSTATE_PENDING",
            body={
                "dimension": row.dimension,
                "value": row.value,
                "citation": basis,
                "revocation_citation": row.citation,
                "accountable_owners": list(required),
                "approved_by": [name for name in required if name in approved],
                "still_required": still,
                "revocation_id": row.id,
                "at": at.isoformat(),
            },
        )
        return row
    row.effective_to = at
    session.add(row)
    session.flush()
    if row.dimension == "agent":
        set_kill_switch(
            session,
            row.value,
            False,
            evidence=evidence,
            actor_id=actor,
            correlation_id=f"revocation:{row.dimension}:{row.value}",
            reason=basis,
            now=at,
        )
    _emit(
        session,
        evidence,
        actor=actor,
        action="REINSTATE",
        body={
            "dimension": row.dimension,
            "value": row.value,
            "citation": basis,
            "revocation_citation": row.citation,
            "accountable_owners": list(required),
            "revocation_id": row.id,
            "at": at.isoformat(),
        },
    )
    return row


def load_revocations(session: Session, at: datetime) -> tuple[OpenRevocation, ...]:
    """Revocations in force at ``at``."""
    when = _aware(at)
    rows = session.exec(select(Revocation)).all()
    open_rows = [row for row in rows if row.is_effective(when) and row.id is not None]
    return tuple(_view(row) for row in open_rows)


def revocation_checks(
    envelope: Envelope,
    revocations: Sequence[OpenRevocation],
    *,
    attributes: dict[str, Any] | None = None,
    register: Any = None,
) -> list[Check]:
    """Failing checks for revocations that match this request.

    A request that matches nothing gains no check. A matching provider that
    the revocation still permits, by use case or by classification, gains no
    check either.
    """
    if not revocations:
        return []
    attrs = attributes or {}
    keys = cited_values(envelope, attrs, register)
    classification = _request_classification(attrs)
    checks: list[Check] = []
    for item in revocations:
        if not _blocks(item, keys, envelope.purpose, classification):
            continue
        owners = ", ".join(item.accountable_owners) or "none"
        checks.append(
            Check(
                check="REVOCATION",
                regime="INTERNAL",
                result=CheckResult.FAIL,
                citation=(
                    f"{CITE} — {item.dimension} {item.value} is revoked · "
                    f"{item.citation} · accountable owner: {owners}"
                ),
            )
        )
    return checks


def cited_values(
    envelope: Envelope,
    attributes: dict[str, Any],
    register: Any = None,
) -> dict[str, set[str]]:
    """Values this request carries, per revocation dimension, plus prompt."""
    keys: dict[str, set[str]] = {name: set() for name in (*DIMENSIONS, "prompt")}
    if envelope.agent_name and envelope.agent_name.strip():
        keys["agent"].add(envelope.agent_name.strip())
    keys["model"] = request_model_keys(envelope, attributes)
    keys["provider"] = request_field_keys(envelope, attributes, "provider")
    if register is not None and envelope.purpose:
        entry = register.get(envelope.purpose)
        if entry is not None:
            for provider in entry.model_providers:
                if str(provider).strip():
                    keys["provider"].add(str(provider).strip())
    if envelope.tool.strip():
        keys["tool"].add(envelope.tool.strip())
    keys["source"] = request_field_keys(envelope, attributes, "source")
    keys["action"] = request_field_keys(envelope, attributes, "action")
    keys["client"] = request_field_keys(envelope, attributes, "client_id", "client")
    keys["prompt"] = request_field_keys(
        envelope, attributes, "prompt_template_version"
    )
    if envelope.purpose.strip():
        keys["usecase"].add(envelope.purpose.strip())
    if envelope.subject_type and envelope.subject_id:
        kind = envelope.subject_type.strip()
        subject = str(envelope.subject_id).strip()
        if kind in keys and subject:
            keys[kind].add(subject)
    for kind, subject in (envelope.subjects or {}).items():
        if kind in keys and str(subject).strip():
            keys[kind].add(str(subject).strip())
    return keys


def accountable_owners(
    session: Session,
    dimension: str,
    value: str,
    *,
    explicit: str,
    at: datetime,
) -> tuple[str, ...]:
    """Who may reinstate this revocation.

    A use case names its business owner. An agent names ``owner_exec``. A
    model names the business owners of the use cases that depend on it, and
    an explicit owner may narrow that set to one of them. A provider names
    every business owner of every non-retired use case that lists it. Naming
    one of them does not drop the others. Where none of those exist,
    ``explicit`` must already be a business owner or an agent's accountable
    owner. A revocation with no such person is refused.
    """
    derived = _derived_owners(session, dimension, value, at)
    named = explicit.strip()
    if named and _is_agent(session, named):
        raise RevocationError(
            f"{CITE}: {named} is a registered agent and cannot be the accountable owner"
        )
    if derived:
        if named and named not in derived:
            raise RevocationError(
                f"{CITE}: {named} is not an accountable owner for {dimension} {value}"
            )
        if dimension == "provider":
            return derived
        return (named,) if named else derived
    if not named or not _authorised_human(session, named, at):
        raise RevocationError(
            f"{CITE}: {dimension} {value} has no accountable owner who can reinstate it"
        )
    return (named,)


def _derived_owners(
    session: Session, dimension: str, value: str, at: datetime
) -> tuple[str, ...]:
    register = load_register(session, as_of=at)
    if dimension == "usecase":
        entry = register.get(value)
        if entry is None or not entry.owner.strip():
            return ()
        return (entry.owner.strip(),)
    if dimension == "agent":
        agent = _agent(session, value)
        if agent is None or not agent.owner_exec.strip():
            return ()
        return (agent.owner_exec.strip(),)
    if dimension == "provider":
        owners = [
            entry.owner.strip()
            for entry in register.entries
            if entry.status != "retired" and value in set(entry.model_providers) and entry.owner.strip()
        ]
        return tuple(dict.fromkeys(owners))
    if dimension == "model":
        owners = []
        cards = {card.card_id: card for card in register.model_cards}
        for entry in register.entries:
            if entry.status == "retired" or not entry.owner.strip():
                continue
            if _entry_uses_model(entry, cards, value):
                owners.append(entry.owner.strip())
        return tuple(dict.fromkeys(owners))
    return ()


def _required_owners(session: Session, row: Revocation, at: datetime) -> tuple[str, ...]:
    """Who must cite a reinstatement before the row can close.

    A provider is the union of the owners stored at revocation and the
    business owners of non-retired use cases that list it now. A use case
    that starts listing the provider while the revocation is open adds its
    owner. Other dimensions keep the owners stored on the row, and any one
    of them can close it.
    """
    stored = tuple(
        str(item).strip() for item in (row.accountable_owners or []) if str(item).strip()
    )
    if row.dimension != "provider":
        return stored
    derived = _derived_owners(session, "provider", row.value, at)
    return tuple(dict.fromkeys((*stored, *derived)))


def _approved_actors(session: Session, revocation_id: int) -> set[str]:
    rows = session.exec(
        select(ReinstatementApproval).where(
            ReinstatementApproval.revocation_id == revocation_id
        )
    ).all()
    return {row.actor_id.strip() for row in rows if row.actor_id and row.actor_id.strip()}


def _entry_uses_model(entry: Any, cards: dict[str, Any], value: str) -> bool:
    for card_id in entry.approved_models:
        if card_id == value:
            return True
        card = cards.get(card_id)
        if card is None:
            continue
        if value in {card.version, card.model, card.card_id}:
            return True
    return False


def _authorised_human(session: Session, name: str, at: datetime) -> bool:
    if _is_agent(session, name):
        return False
    register = load_register(session, as_of=at)
    if any(entry.owner.strip() == name for entry in register.entries):
        return True
    agents = session.exec(select(Agent)).all()
    return any(agent.owner_exec.strip() == name for agent in agents)


def _blocks(
    item: OpenRevocation,
    keys: dict[str, set[str]],
    purpose: str,
    classification: str | None,
) -> bool:
    if item.dimension == "provider":
        return _provider_blocks(item, keys, purpose, classification)
    values = keys.get(item.dimension)
    if values is None:
        return True
    return item.value in values


def _provider_blocks(
    item: OpenRevocation,
    keys: dict[str, set[str]],
    purpose: str,
    classification: str | None,
) -> bool:
    if item.value not in keys.get("provider", set()):
        return False
    if item.limit_usecases and purpose not in set(item.limit_usecases):
        return True
    if item.below_classification:
        if classification is None:
            return True
        try:
            if classification_rank(classification) >= classification_rank(
                item.below_classification
            ):
                return True
        except KeyError:
            return True
    if item.limit_usecases or item.below_classification:
        return False
    return True


def _open_row(
    session: Session, dimension: str, value: str, at: datetime
) -> Revocation | None:
    rows = session.exec(
        select(Revocation).where(Revocation.dimension == dimension, Revocation.value == value)
    ).all()
    for row in rows:
        if row.is_effective(at):
            return row
    return None


def _view(row: Revocation) -> OpenRevocation:
    return OpenRevocation(
        revocation_id=row.id or 0,
        dimension=row.dimension,
        value=row.value,
        citation=row.citation,
        accountable_owners=tuple(
            str(item).strip() for item in (row.accountable_owners or []) if str(item).strip()
        ),
        limit_usecases=tuple(
            str(item).strip() for item in (row.limit_usecases or []) if str(item).strip()
        ),
        below_classification=row.below_classification or "",
    )


def _emit(
    session: Session,
    evidence: EvidenceWriter,
    *,
    actor: str,
    action: str,
    body: dict[str, Any],
) -> None:
    evidence.emit(
        session,
        correlation_id=f"revocation:{body.get('dimension')}:{body.get('value')}",
        event_type=EventType.POLICY_CHANGE,
        actor_type=ActorType.HUMAN,
        actor_id=actor,
        payload={"action": action, **body},
    )


def _dimension(value: str) -> str:
    kind = value.strip()
    if kind not in DIMENSIONS:
        raise RevocationError(f"{CITE}: {value!r} is not a revocation dimension")
    return kind


def _human(session: Session, actor_id: str, action: str) -> str:
    actor = actor_id.strip()
    if not actor:
        raise RevocationError(f"{CITE}: a {action} names the person")
    if _is_agent(session, actor):
        raise RevocationError(
            f"{CITE}: {actor} is a registered agent and cannot {action}"
        )
    return actor


def _citation(citation: str, action: str) -> str:
    basis = citation.strip()
    if not basis:
        raise RevocationError(f"{CITE}: a {action} cites its rule")
    return basis


def _classification(value: str) -> str:
    try:
        return Classification(value).value
    except ValueError as exc:
        raise RevocationError(
            f"{CITE}: {value!r} is not a classification"
        ) from exc


def _request_classification(attributes: dict[str, Any]) -> str | None:
    value = attributes.get("classification")
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return Classification(value.strip()).value
    except ValueError:
        return None


def _names(values: Sequence[str]) -> list[str]:
    found: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise RevocationError(f"{CITE}: a limited use case has a name")
        name = value.strip()
        if name not in found:
            found.append(name)
    return found


def _agent(session: Session, name: str) -> Agent | None:
    return session.exec(select(Agent).where(Agent.name == name)).first()


def _is_agent(session: Session, name: str) -> bool:
    return _agent(session, name) is not None


def _aware(value: datetime) -> datetime:
    when = as_utc(value)
    if when is None:
        return utcnow()
    return when


__all__ = [
    "DIMENSIONS",
    "OpenRevocation",
    "RevocationError",
    "accountable_owners",
    "cited_values",
    "load_revocations",
    "reinstate",
    "revoke",
    "revocation_checks",
]
