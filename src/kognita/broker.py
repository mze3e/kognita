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
from kognita.usecases import _governing_register
from kognita.models import utcnow
from kognita.protocols import Embedder
from kognita.retrieval import Retrieved, ceiling_for, retrieve
from kognita.interaction import require_interaction
from kognita.tools import (
    Run,
    ToolSpec,
    _admit,
    _approval_for_decision,
    _bind_interaction,
    _budget_payload,
    _checkpoint,
    _remember_pack,
    _save_run,
    _with_checks,
    record_run_decision,
)
from kognita.vocabulary import ActorType, Classification, Outcome, CheckResult


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
    summary = [f"Request {verb} by governance before any data was retrieved."] + [
        f"{c.regime}: {c.check} — {c.citation}" for c in evaluation.basis()
    ]
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
            summary.append(
                "Permitted subject to human review before any communication:"
            )
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
        {
            "label": r.source_label,
            "classification": Classification(r.classification).value,
        }
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
    run: Run | None = None,
) -> tuple[list[Retrieved], Evaluation]:
    hits = retrieve(
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
        run=run,
    )

    from kognita.anomaly import _retrieved
    return _retrieved(session, evaluation, evidence, hits)


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
    interaction_id: str | None = None,
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
    if interaction_id is not None:
        require_interaction(session, interaction_id)

    subjects = pack.load_subjects(routed, session)
    attributes = pack.resolve_attributes(routed, subjects)
    _remember_pack(pack)
    snap = snapshot if snapshot is not None else load_snapshot(session, as_of=as_of or at, _evidence=evidence, _agent_name=envelope.agent_name or "")
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
        usecases=_governing_register(session, as_of=as_of or at),
        engages=getattr(pack, "engages", None),
        as_of=as_of,
    )
    from kognita._fleet import _run_checks
    from kognita.tools import _refresh_run
    if run is not None:
        _refresh_run(session, run)
    isolation = _run_checks(session, run, routed, evidence)
    evaluation = _with_checks(evaluation, isolation)
    if any(c.result == CheckResult.FAIL for c in isolation):
        run = None
    from kognita._isolation import _ingest, _tool_checks
    _ingest(session, run, evidence, text=question, correlation_id=evaluation.request_id,
        _agent_name=routed.agent_name, client_data='client' in routed.all_subjects()
        or attributes.get('classification') in (Classification.C2, Classification.C3, 'C2', 'C3'))
    agent = snap.agents.get(routed.agent_name)
    permission = (agent.tool_allowlist or {}).get(routed.tool, {}) if agent else {}
    permission = permission if isinstance(permission, dict) else {}
    level = permission.get('required_level', 0)
    retrieval_spec = ToolSpec(routed.tool, lambda *_: None,
        required_level=level if type(level) is int else 0,
        verb=permission.get('verb', 'read'),
        external=permission.get('external', False),
        client_data=permission.get('client_data', False),
        public_content=permission.get('public_content', False))
    evaluation = _with_checks(evaluation, _tool_checks(session, routed, retrieval_spec, run, evidence, at=at))
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
    if run is not None:
        _save_run(session, run)
    evaluation = record(
        session,
        evaluation,
        evidence=evidence,
        budget=_budget_payload(run) if run is not None else None,
        create_approval=already_granted is None,
        now=at,
        run_id=None if run is None else run.id,
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
            interaction_id=interaction_id,
            pack=pack,
        )
    elif run is not None:
        _save_run(session, run)

    if not evaluation.allowed:
        _bind_interaction(session, interaction_id, evaluation, run, evidence, at)
        record_run_decision(session, interaction_id, evaluation, evidence, now=at)
        # Fail closed, but never silently: the basis for refusing is itself the
        # answer, and it is the part a user can act on.
        return _refused_answer(evaluation, route.name)

    _bind_interaction(session, interaction_id, evaluation, run, evidence, at)
    if holding:
        # Same rule as the tool runner: prepare the decision, release nothing.
        # The final decision waits until the approval is resolved.
        summary = ["Request held for human approval before any data was retrieved."] + [
            f"{c.regime}: {c.check} — {c.citation}" for c in evaluation.basis()
        ]
        return BrokerAnswer(
            request_id=evaluation.request_id,
            outcome=evaluation.outcome,
            route=route.name,
            summary=summary,
            citations=_policy_citations(evaluation),
            results=[],
            evaluation=evaluation,
        )

    results, evaluation = _retrieve_for(
        session,
        question,
        routed,
        evaluation,
        embedder=embedder,
        evidence=evidence,
        top_k=top_k,
        run=run,
    )
    record_run_decision(session, interaction_id, evaluation, evidence, now=at)
    return _minimized_answer(session, evidence, routed, _compose_answer(
        evaluation,
        route.name,
        results,
        routed,
        subgraph=subgraph,
        session=session,
    ))


def _resume_retrieval(
    session: Session,
    payload: dict[str, Any],
    *,
    envelope: Envelope,
    evaluation: Evaluation,
    embedder: Embedder | None,
    evidence: EvidenceWriter,
    subgraph: Callable[[Envelope, Session], dict[str, int]] | None,
    run: Run | None = None,
) -> BrokerAnswer:
    """Fetch entitled sources for a continuation that was held before retrieval."""
    if embedder is None:
        raise ApprovalError("continuing a retrieval requires an embedder")
    from kognita._fleet import _run_checks
    isolation = _run_checks(session, run, envelope, evidence)
    evaluation = _with_checks(evaluation, isolation)
    if not evaluation.allowed:
        return _refused_answer(evaluation, str(payload.get("route") or KNOWLEDGE))
    results, evaluation = _retrieve_for(
        session,
        str(payload["question"]),
        envelope,
        evaluation,
        embedder=embedder,
        evidence=evidence,
        top_k=int(payload.get("top_k") or 5),
        run=run,
    )
    return _minimized_answer(session, evidence, envelope, _compose_answer(
        evaluation,
        str(payload.get("route") or KNOWLEDGE),
        results,
        envelope,
        subgraph=subgraph,
        session=session,
    ))


def _minimized_answer(session, evidence, envelope, answer):
    if not envelope.agent_name:
        return answer
    from kognita._limits import _minimize
    from dataclasses import fields
    data = _minimize(session, envelope, answer.to_dict(), evidence,
                     _request_id=answer.request_id)
    answer.summary = [v for v in data.get('summary', []) if v is not None]
    answer.citations = [v for v in data.get('citations', []) if v]
    answer.graph = data.get('graph')
    baseline = Retrieved(0, '', '', '', Classification.C0, '', None, 0)
    names = {f.name for f in fields(Retrieved)}
    answer.results = []
    for item in data.get('results', []):
        if not item:
            continue
        values = {k: v for k, v in item.items() if k in names}
        if 'document' in item:
            values['document_ref'] = item['document']
        if 'section_path' in values:
            values['section_path'] = tuple(v for v in values['section_path'] if v is not None)
        for key, start, end in [('charspan', 'char_start', 'char_end'),
                                ('block_charspan', 'block_char_start', 'block_char_end')]:
            span = item.get(key)
            if isinstance(span, list) and len(span) == 2:
                values[start], values[end] = span
        answer.results.append(replace(baseline, **values))
    return answer


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
