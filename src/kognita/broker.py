"""The Context Broker — one governed front door.

A question arrives, is normalised into an envelope, and is authorised *before*
anything is retrieved. What comes back is composed deterministically from
entitled, cited fragments: there is no free-text generation here, so an answer
cannot assert something no source supports.

``HUMAN_APPROVAL`` withholds retrieval the same way a tool call is withheld:
nothing is fetched and nothing is returned until the approval is actually
granted. Resuming that held question is :func:`kognita.tools.continue_run`.

Routing is domain-specific — which question shapes map to which tool and
retrieval mode is a business judgement — so a pack supplies a resolver. The core
provides a default that routes on how many subjects are in scope, matching the
usual case: a question about a subject *and* a related object is an eligibility
question, one about a subject alone is a context question, and one about neither
is a knowledge question.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Sequence

from sqlmodel import Session

from kognita.approvals import ApprovalError, find_granted
from kognita.envelope import Envelope, Evaluation
from kognita.evidence import EvidenceWriter
from kognita.governance import (
    PolicySnapshot,
    classifier_derived_envelope,
    decide,
    load_snapshot,
    record,
)
from kognita.models import utcnow
from kognita.protocols import Embedder
from kognita.retrieval import Retrieved, ceiling_for, retrieve
from kognita.tools import (
    Run,
    _admit,
    _approval_for_decision,
    _budget_payload,
    _checkpoint,
    _save_run,
)
from kognita.vocabulary import ActorType, Classification, Outcome


@dataclass(frozen=True)
class Route:
    """Where a question is going, and the tool whose permission it needs."""

    name: str
    tool: str


#: Routes the default resolver produces.
ELIGIBILITY = "ELIGIBILITY"
SUBJECT_CONTEXT = "SUBJECT_CONTEXT"
KNOWLEDGE = "KNOWLEDGE"

RouteResolver = Callable[[str, Envelope], Route]


def default_route_resolver(question: str, envelope: Envelope) -> Route:
    """Route by what is in scope, never by keywords in the question.

    Keyword routing is the trap here: a knowledge question that merely *mentions*
    subjects would otherwise authorise subject-profile access. Scope is a fact
    about the request; wording is not.
    """
    subjects = envelope.all_subjects()
    has_subject = envelope.subject_id is not None
    has_other = any(k != envelope.subject_type for k in subjects)

    if has_subject and has_other:
        return Route(ELIGIBILITY, envelope.tool or "check_eligibility")
    if has_subject:
        return Route(SUBJECT_CONTEXT, envelope.tool or "get_subject_profile")
    return Route(KNOWLEDGE, envelope.tool or "get_knowledge")


@dataclass
class BrokerAnswer:
    """A deterministic answer: a summary, its basis, and its citations."""

    request_id: str
    outcome: Outcome
    route: str
    summary: list[str] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)
    results: list[Retrieved] = field(default_factory=list)
    evaluation: Evaluation | None = None
    graph: dict[str, int] | None = None

    @property
    def denied(self) -> bool:
        return self.outcome in (Outcome.DENY, Outcome.ESCALATE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "outcome": self.outcome.value,
            "route": self.route,
            "summary": list(self.summary),
            "citations": list(self.citations),
            "results": [r.to_dict() for r in self.results],
            "graph": self.graph,
        }


def _policy_citations(evaluation: Evaluation) -> list[dict[str, str]]:
    return [
        {"label": c.citation, "classification": Classification.C1.value}
        for c in evaluation.checks
        if c.citation
    ]


def _refused_answer(evaluation: Evaluation, route_name: str) -> BrokerAnswer:
    """The basis for refusing, and nothing retrieved."""
    verb = "denied" if evaluation.outcome == Outcome.DENY else "escalated"
    summary = [
        f"Request {verb} by governance before any data was retrieved."
    ] + [f"{c.regime}: {c.check} — {c.citation}" for c in evaluation.basis()]
    return BrokerAnswer(
        request_id=evaluation.request_id,
        outcome=evaluation.outcome,
        route=route_name,
        summary=summary,
        citations=_policy_citations(evaluation),
        results=[],
        evaluation=evaluation,
    )


def _compose_answer(
    evaluation: Evaluation,
    route_name: str,
    results: list[Retrieved],
    envelope: Envelope,
    *,
    subgraph: Callable[[Envelope, Session], dict[str, int]] | None,
    session: Session,
) -> BrokerAnswer:
    """Build the answer from sources that have already been retrieved."""
    summary: list[str] = []
    graph: dict[str, int] | None = None
    citations = _policy_citations(evaluation)

    if route_name == ELIGIBILITY:
        human = evaluation.human_reviews()
        if human:
            summary.append("Permitted subject to human review before any communication:")
            summary.extend(f"{c.regime} — {c.citation}" for c in human)
        else:
            summary.append("Permitted under all engaged regimes. Citations below.")
    elif route_name == SUBJECT_CONTEXT:
        if subgraph is not None:
            graph = subgraph(envelope, session)
            summary.append(
                f"Governed context: {graph.get('nodes', 0)} connected objects "
                f"across {graph.get('edges', 0)} relationships."
            )
        else:
            summary.append("Governed subject context.")
    else:
        summary.append(
            f"{len(results)} entitled sources answer this in zone "
            f"{envelope.actor_location}."
            if results
            else f"No entitled sources answer this in zone {envelope.actor_location}."
        )

    citations.extend(
        {"label": r.source_label, "classification": Classification(r.classification).value}
        for r in results
        if r.source_label
    )
    return BrokerAnswer(
        request_id=evaluation.request_id,
        outcome=evaluation.outcome,
        route=route_name,
        summary=summary,
        citations=citations,
        results=results,
        evaluation=evaluation,
        graph=graph,
    )


def _retrieve_for(
    session: Session,
    question: str,
    envelope: Envelope,
    evaluation: Evaluation,
    *,
    embedder: Embedder,
    evidence: EvidenceWriter,
    top_k: int,
) -> list[Retrieved]:
    return retrieve(
        session,
        question,
        zone=envelope.actor_location,
        embedder=embedder,
        evidence=evidence,
        correlation_id=evaluation.request_id,
        actor_id=envelope.agent_name or envelope.principal,
        actor_type=ActorType.AGENT if envelope.agent_name else ActorType.HUMAN,
        is_admin=envelope.is_admin,
        top_k=top_k,
        use_case=envelope.purpose,
    )


def ask(
    session: Session,
    question: str,
    envelope: Envelope,
    *,
    pack: Any,
    embedder: Embedder,
    evidence: EvidenceWriter,
    snapshot: PolicySnapshot | None = None,
    purposes: Sequence[str] = (),
    route_resolver: RouteResolver | None = None,
    subgraph: Callable[[Envelope, Session], dict[str, int]] | None = None,
    as_of: datetime | None = None,
    top_k: int = 5,
    run: Run | None = None,
    cost_usd: float | None = None,
    tokens: int | None = None,
    now: datetime | None = None,
) -> BrokerAnswer:
    """Authorise a question, then answer it from entitled sources only.

    A ``HUMAN_APPROVAL`` outcome returns no results and does not retrieve.
    Passing ``run`` checkpoints that hold so :func:`kognita.tools.continue_run`
    can retrieve after the approval is granted.
    """
    resolver = route_resolver or default_route_resolver
    route = resolver(question, envelope)
    routed = envelope if envelope.tool else replace(envelope, tool=route.tool)
    at = now or utcnow()

    subjects = pack.load_subjects(routed, session)
    attributes = pack.resolve_attributes(routed, subjects)
    snap = snapshot if snapshot is not None else load_snapshot(session, as_of=as_of)
    # The question is free text. Typed envelope fields stay as the caller set
    # them; classification is filled only when the pack did not supply one.
    # decide() below reads that recorded label and does not classify again.
    routed, attributes = classifier_derived_envelope(
        question,
        routed,
        attributes=attributes,
    )

    evaluation = decide(
        routed,
        snap,
        attributes=attributes,
        subjects=subjects,
        rules=pack.rules(),
        purposes=purposes,
        engages=getattr(pack, "engages", None),
        as_of=as_of,
    )
    # The ceiling the caller would otherwise search at. A run that sits below
    # it is a budget breach, not a silent narrowing of the result set.
    evaluation, _breaches = _admit(
        session,
        run,
        evaluation,
        classification=ceiling_for(routed.is_admin),
        cost_usd=cost_usd,
        tokens=tokens,
        now=at,
    )
    already_granted = (
        find_granted(session, evaluation.envelope_hash, now=at)
        if evaluation.approval_required
        else None
    )
    evaluation = record(
        session,
        evaluation,
        evidence=evidence,
        budget=_budget_payload(run) if run is not None else None,
        create_approval=already_granted is None,
        now=at,
    )

    holding = evaluation.approval_required and already_granted is None
    if run is not None and holding:
        approval = _approval_for_decision(session, evaluation.decision_id or 0)
        _checkpoint(
            session,
            run,
            envelope=routed,
            evaluation=evaluation,
            approval_id=approval.id or 0,
            evidence=evidence,
            kind="retrieval",
            extra={"question": question, "route": route.name, "top_k": top_k},
        )
    elif run is not None:
        _save_run(session, run)

    if not evaluation.allowed:
        # Fail closed, but never silently: the basis for refusing is itself the
        # answer, and it is the part a user can act on.
        return _refused_answer(evaluation, route.name)

    if holding:
        # Same rule as the tool runner: prepare the decision, release nothing.
        summary = [
            "Request held for human approval before any data was retrieved."
        ] + [f"{c.regime}: {c.check} — {c.citation}" for c in evaluation.basis()]
        return BrokerAnswer(
            request_id=evaluation.request_id,
            outcome=evaluation.outcome,
            route=route.name,
            summary=summary,
            citations=_policy_citations(evaluation),
            results=[],
            evaluation=evaluation,
        )

    results = _retrieve_for(
        session,
        question,
        routed,
        evaluation,
        embedder=embedder,
        evidence=evidence,
        top_k=top_k,
    )
    return _compose_answer(
        evaluation,
        route.name,
        results,
        routed,
        subgraph=subgraph,
        session=session,
    )


def _resume_retrieval(
    session: Session,
    payload: dict[str, Any],
    *,
    envelope: Envelope,
    evaluation: Evaluation,
    embedder: Embedder | None,
    evidence: EvidenceWriter,
    subgraph: Callable[[Envelope, Session], dict[str, int]] | None,
) -> BrokerAnswer:
    """Fetch entitled sources for a continuation that was held before retrieval."""
    if embedder is None:
        raise ApprovalError("continuing a retrieval requires an embedder")
    results = _retrieve_for(
        session,
        str(payload["question"]),
        envelope,
        evaluation,
        embedder=embedder,
        evidence=evidence,
        top_k=int(payload.get("top_k") or 5),
    )
    return _compose_answer(
        evaluation,
        str(payload.get("route") or KNOWLEDGE),
        results,
        envelope,
        subgraph=subgraph,
        session=session,
    )


__all__ = [
    "ask",
    "BrokerAnswer",
    "Route",
    "RouteResolver",
    "default_route_resolver",
    "ELIGIBILITY",
    "SUBJECT_CONTEXT",
    "KNOWLEDGE",
]
