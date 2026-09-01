"""The Context Broker — one governed front door.

A question arrives, is normalised into an envelope, and is authorised *before*
anything is retrieved. What comes back is composed deterministically from
entitled, cited fragments: there is no free-text generation here, so an answer
cannot assert something no source supports.

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

from kognita.core.envelope import Envelope, Evaluation
from kognita.core.evidence import EvidenceWriter
from kognita.core.governance import PolicySnapshot, decide, load_snapshot, record
from kognita.core.protocols import Embedder
from kognita.core.retrieval import Retrieved, retrieve
from kognita.core.vocabulary import ActorType, Classification, Outcome


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
) -> BrokerAnswer:
    """Authorise a question, then answer it from entitled sources only."""
    resolver = route_resolver or default_route_resolver
    route = resolver(question, envelope)
    routed = envelope if envelope.tool else replace(envelope, tool=route.tool)

    subjects = pack.load_subjects(routed, session)
    attributes = pack.resolve_attributes(routed, subjects)
    snap = snapshot if snapshot is not None else load_snapshot(session, as_of=as_of)

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
    evaluation = record(session, evaluation, evidence=evidence)

    citations = [
        {"label": c.citation, "classification": Classification.C1.value}
        for c in evaluation.checks
        if c.citation
    ]

    if not evaluation.allowed:
        # Fail closed, but never silently: the basis for refusing is itself the
        # answer, and it is the part a user can act on.
        verb = "denied" if evaluation.outcome == Outcome.DENY else "escalated"
        summary = [
            f"Request {verb} by governance before any data was retrieved."
        ] + [f"{c.regime}: {c.check} — {c.citation}" for c in evaluation.basis()]
        return BrokerAnswer(
            request_id=evaluation.request_id,
            outcome=evaluation.outcome,
            route=route.name,
            summary=summary,
            citations=citations,
            results=[],
            evaluation=evaluation,
        )

    results = retrieve(
        session,
        question,
        zone=routed.actor_location,
        embedder=embedder,
        evidence=evidence,
        correlation_id=evaluation.request_id,
        actor_id=routed.agent_name or routed.principal,
        actor_type=ActorType.AGENT if routed.agent_name else ActorType.HUMAN,
        is_admin=routed.is_admin,
        top_k=top_k,
    )

    summary: list[str] = []
    graph: dict[str, int] | None = None

    if route.name == ELIGIBILITY:
        human = evaluation.human_reviews()
        if human:
            summary.append("Permitted subject to human review before any communication:")
            summary.extend(f"{c.regime} — {c.citation}" for c in human)
        else:
            summary.append("Permitted under all engaged regimes. Citations below.")
    elif route.name == SUBJECT_CONTEXT:
        if subgraph is not None:
            graph = subgraph(routed, session)
            summary.append(
                f"Governed context: {graph.get('nodes', 0)} connected objects "
                f"across {graph.get('edges', 0)} relationships."
            )
        else:
            summary.append("Governed subject context.")
    else:
        summary.append(
            f"{len(results)} entitled sources answer this in zone "
            f"{routed.actor_location}."
            if results
            else f"No entitled sources answer this in zone {routed.actor_location}."
        )

    if evaluation.approval_required:
        summary.append(
            "A regulated approval has been opened and must be granted before "
            "acting on this answer."
        )

    citations.extend(
        {"label": r.source_label, "classification": Classification(r.classification).value}
        for r in results
        if r.source_label
    )

    return BrokerAnswer(
        request_id=evaluation.request_id,
        outcome=evaluation.outcome,
        route=route.name,
        summary=summary,
        citations=citations,
        results=results,
        evaluation=evaluation,
        graph=graph,
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
