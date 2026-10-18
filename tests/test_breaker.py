"""Circuit breaker, revocation, affected clients, and kill-switch drills.

Roadmap 0.4 item 14. Decisions name a registered use case when the test
supplies the register. No network and no API key.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from sqlmodel import Session, select

from kognita.breaker import (
    BreakerError,
    affected_clients,
    evaluate_breakers,
    reset_breaker,
    run_kill_switch_drill,
    schedule_kill_switch_drill,
    set_breaker_threshold,
)
from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.envelope import Envelope
from evidence_guard import tamper_for_test
from kognita.evidence import ChainBreak, EvidenceWriter, verify_chain
from kognita.gateway import ClientConfiguration, Gateway
from kognita.governance import decide, load_snapshot
from kognita.interaction import open_interaction
from kognita.models import EvidenceEvent, InteractionLink, Policy, Revocation
from kognita.outcomes import record_near_miss
from kognita.policy import parse_rule
from kognita.registry import get as get_agent
from fixtures.agents import register as register_agent
from kognita.registry import set_kill_switch
from kognita.review import enqueue_for_review, record_review_finding
from kognita.revocation import RevocationError, reinstate, revoke
from kognita.usecases import load_register
from kognita.vocabulary import ActorType, EventType, Outcome
from test_gateway import _live
from test_usecases import NOW, _activate, _open, _spec

LATER = NOW + timedelta(hours=1)
BEFORE = NOW - timedelta(hours=1)


def _checks(evaluation, name: str):
    return [check for check in evaluation.checks if check.check == name]


def _decide(
    session,
    purpose: str = "research-summary",
    tool: str = "get_subject_profile",
    *,
    agent_name: str | None = None,
    attributes: dict | None = None,
    arguments: dict | None = None,
    subject_type: str | None = None,
    subject_id: str | None = None,
    purposes: tuple[str, ...] = (),
    engages=None,
    with_register: bool = True,
):
    from test_usecases import ensure_live_purposes
    from kognita.evidence import EvidenceWriter
    if with_register:
        ensure_live_purposes(session, EvidenceWriter(session.get_bind(), clock=lambda: NOW), [purpose], at=NOW)
    register = load_register(session, as_of=NOW)
    return decide(
        Envelope(
            principal="alice",
            purpose=purpose,
            tool=tool,
            agent_name=agent_name,
            subject_type=subject_type,
            subject_id=subject_id,
            arguments=arguments or {},
        ),
        load_snapshot(session, as_of=NOW),
        attributes=attributes or {},
        usecases=register if with_register and register.entries else None,
        purposes=purposes,
        engages=engages,
        as_of=NOW,
    )


def _owners(session, evidence) -> None:
    _open(session, evidence)
    _activate(session, evidence, _spec())
    _activate(
        session,
        evidence,
        _spec(
            usecase_id="briefing",
            purpose="Prepare a briefing",
            model_providers=("house",),
        ),
    )


def _near(session, evidence, summary: str) -> None:
    record_near_miss(
        session,
        kind="unsafe draft",
        summary=summary,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Outcomes and near misses: an unsafe draft counts toward the breaker",
        usecase_id="research-summary",
        model_version="1",
        now=NOW,
    )


def _received(session, evidence, client: str, usecase: str, **payload):
    interaction = open_interaction(
        session,
        client_id=client,
        use_case=usecase,
        evidence=evidence,
        now=NOW,
    )
    body = {
        "sent": True,
        "response_hash": f"hash-{client}",
        "model": "summary",
        "model_version": "1",
        "provider": "house",
    }
    body.update(payload)
    evidence.emit(
        session,
        correlation_id=interaction.id,
        event_type=EventType.MODEL_CALL,
        actor_type=ActorType.AGENT,
        actor_id="dossier-agent",
        payload=body,
    )
    return interaction


def test_near_misses_trip_the_breaker_on_the_next_decision(session, evidence):
    _owners(session, evidence)
    set_breaker_threshold(
        session,
        "usecase",
        "research-summary",
        threshold=2,
        window_seconds=3600,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Flags: two near misses trip the use case",
        now=NOW,
    )
    _near(session, evidence, "The draft named the wrong issuer")
    assert (
        session.exec(select(Policy).where(Policy.rule_type == "CIRCUIT_BREAKER")).all()
        == []
    )

    _near(session, evidence, "The draft omitted the rating")
    policies = session.exec(
        select(Policy).where(Policy.rule_type == "CIRCUIT_BREAKER")
    ).all()
    assert len(policies) == 1
    assert policies[0].rule["dimension"] == "usecase"
    assert policies[0].rule["value"] == "research-summary"
    assert policies[0].effective_from <= NOW
    incident = [
        event
        for event in session.exec(select(EvidenceEvent)).all()
        if event.payload.get("action") == "INCIDENT"
    ]
    assert incident
    assert incident[-1].payload["accountable_owner"] == "Jane Smith"
    assert incident[-1].payload["impact"] == 1
    trip = [
        event
        for event in session.exec(select(EvidenceEvent)).all()
        if event.payload.get("action") == "CIRCUIT_BREAKER"
    ]
    assert trip[-1].payload["escalated_to"] == "Jane Smith"

    _near(session, evidence, "A third draft does not insert a second policy")
    assert (
        len(session.exec(select(Policy).where(Policy.rule_type == "CIRCUIT_BREAKER")).all())
        == 1
    )

    denied = _decide(session)
    assert denied.outcome is Outcome.DENY
    breaker = _checks(denied, "CIRCUIT_BREAKER")
    assert len(breaker) == 1
    assert "research-summary" in breaker[0].citation
    assert "Jane Smith" in breaker[0].citation
    assert breaker[0].policy_id == policies[0].id

    skipped = _decide(session, engages=lambda _policy, _context: False)
    assert skipped.outcome is Outcome.DENY
    assert _checks(skipped, "CIRCUIT_BREAKER")

    assert _decide(session, purpose="briefing").outcome is Outcome.ALLOW

    with pytest.raises(BreakerError, match="cannot reset"):
        reset_breaker(
            session,
            "usecase",
            "research-summary",
            evidence=evidence,
            actor_id="Omar Aziz",
            citation="Technology owner resetting a breaker they do not own",
            now=NOW,
        )
    assert _decide(session).outcome is Outcome.DENY

    reset_breaker(
        session,
        "usecase",
        "research-summary",
        evidence=evidence,
        actor_id="Jane Smith",
        citation="The owner closed the breaker after the drafts were corrected",
        now=NOW,
    )
    assert _decide(session).outcome is Outcome.ALLOW
    assert verify_chain(session) > 0


def test_a_confirmed_problem_trips_the_breaker(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec())
    set_breaker_threshold(
        session,
        "usecase",
        "research-summary",
        threshold=1,
        window_seconds=3600,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Flags: one confirmed problem trips the use case",
        now=NOW,
    )
    interaction = open_interaction(
        session,
        client_id="client-1",
        use_case="research-summary",
        evidence=evidence,
        now=NOW,
    )
    enqueue_for_review(
        session,
        interaction.id,
        evidence=evidence,
        reasons=("the note names the wrong bond",),
        risk="high",
        citation="Risk-based review: a high-risk response is queued",
        actor_id="Jane Smith",
        now=NOW,
    )
    record_review_finding(
        session,
        interaction.id,
        reviewer="Jane Smith",
        outcome="confirmed problem",
        finding="The note names the wrong bond",
        evidence=evidence,
        now=NOW,
    )
    assert _decide(session).outcome is Outcome.DENY
    assert _checks(_decide(session), "CIRCUIT_BREAKER")


def test_a_problem_flagged_a_day_after_its_decision_trips_a_policy_breaker(
    session, engine
):
    """A flag inside the window cites a decision from the day before.

    The breaker counts the flag. The policy id comes from that older decision,
    loaded by the interaction link, not from evidence inside the flag window.
    """
    from kognita.governance import record
    from kognita.interaction import attach

    clock = {"at": NOW}
    evidence = EvidenceWriter(engine, clock=lambda: clock["at"])
    _open(session, evidence)
    _activate(session, evidence, _spec())
    policy = Policy(
        regime="INTERNAL",
        rule_type="ATTRIBUTE_ALLOWLIST",
        rule={
            "allow": {"actor_location": ["SG"]},
            "on_violation": "fail",
            "description": "Requests come from SG",
        },
        citation="Site rule",
        effective_from=NOW - timedelta(days=1),
    )
    session.add(policy)
    session.flush()
    register = load_register(session, as_of=NOW)
    evaluation = decide(
        Envelope(
            principal="alice",
            purpose="research-summary",
            tool="get_subject_profile",
            actor_location="SG",
        ),
        load_snapshot(session, as_of=NOW),
        usecases=register,
        as_of=NOW,
    )
    assert evaluation.outcome is Outcome.ALLOW
    assert any(check.policy_id == policy.id for check in evaluation.checks)
    record(session, evaluation, evidence=evidence, now=NOW)
    interaction = open_interaction(
        session,
        client_id="client-1",
        use_case="research-summary",
        evidence=evidence,
        now=NOW,
    )
    attach(
        session,
        interaction.id,
        evidence=evidence,
        correlation_id=evaluation.request_id,
    )
    enqueue_for_review(
        session,
        interaction.id,
        evidence=evidence,
        reasons=("the note names the wrong bond",),
        risk="high",
        citation="Risk-based review: a high-risk response is queued",
        actor_id="Jane Smith",
        now=NOW,
    )
    later = NOW + timedelta(days=1)
    clock["at"] = later
    set_breaker_threshold(
        session,
        "policy",
        str(policy.id),
        threshold=1,
        window_seconds=3600,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="One confirmed problem on this policy trips the breaker",
        owner="Jane Smith",
        now=later,
    )
    record_review_finding(
        session,
        interaction.id,
        reviewer="Jane Smith",
        outcome="confirmed problem",
        finding="The note names the wrong bond",
        evidence=evidence,
        now=later,
    )
    tripped = session.exec(
        select(Policy).where(Policy.rule_type == "CIRCUIT_BREAKER")
    ).all()
    assert len(tripped) == 1
    assert tripped[0].rule["dimension"] == "policy"
    assert tripped[0].rule["value"] == str(policy.id)


def test_affected_clients_are_exactly_those_who_received_the_output(session, evidence, engine):
    _owners(session, evidence)
    first = _received(session, evidence, "client-a", "research-summary")
    evidence.emit(
        session,
        correlation_id=first.id,
        event_type=EventType.MODEL_CALL,
        actor_type=ActorType.AGENT,
        actor_id="dossier-agent",
        payload={
            "sent": True,
            "response_hash": "hash-client-a-again",
            "model": "summary",
            "model_version": "1",
            "provider": "house",
        },
    )
    second = _received(session, evidence, "client-b", "research-summary")
    _received(session, evidence, "client-c", "research-summary", model_version="2")
    _received(session, evidence, "client-d", "research-summary", sent=False, response_hash="")
    _received(session, evidence, "client-e", "briefing", model_version="9")
    evidence.emit(
        session,
        correlation_id="orphan-call",
        event_type=EventType.MODEL_CALL,
        actor_type=ActorType.AGENT,
        actor_id="dossier-agent",
        payload={
            "sent": True,
            "response_hash": "hash-orphan",
            "model_version": "1",
        },
    )
    late = EvidenceWriter(engine, clock=lambda: NOW + timedelta(days=2))
    late_interaction = open_interaction(
        session,
        client_id="client-late",
        use_case="research-summary",
        evidence=late,
        now=NOW + timedelta(days=2),
    )
    late.emit(
        session,
        correlation_id=late_interaction.id,
        event_type=EventType.MODEL_CALL,
        actor_type=ActorType.AGENT,
        actor_id="dossier-agent",
        payload={"sent": True, "response_hash": "hash-late", "model_version": "1"},
    )
    first_policy = Policy(
        regime="INTERNAL",
        rule_type="PROHIBITED",
        rule={"description": "first", "on_violation": "fail"},
        citation="The policy on client-a's decision",
        effective_from=NOW,
    )
    second_policy = Policy(
        regime="INTERNAL",
        rule_type="PROHIBITED",
        rule={"description": "second", "on_violation": "fail"},
        citation="The policy on client-b's decision",
        effective_from=NOW,
    )
    session.add(first_policy)
    session.add(second_policy)
    session.flush()
    evidence.emit(
        session,
        correlation_id=first.id,
        event_type=EventType.POLICY_DECISION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload={
            "outcome": "ALLOW",
            "checks": [
                {"check": "PURPOSE", "result": "pass", "policy_id": first_policy.id}
            ],
        },
    )
    evidence.emit(
        session,
        correlation_id=second.id,
        event_type=EventType.POLICY_DECISION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload={
            "outcome": "ALLOW",
            "checks": [
                {"check": "PURPOSE", "result": "pass", "policy_id": second_policy.id}
            ],
        },
    )

    by_model = affected_clients(
        session, model="1", start=BEFORE, end=LATER
    )
    assert {row["client_id"] for row in by_model["clients"]} == {"client-a", "client-b"}
    assert {row["interaction_id"] for row in by_model["clients"]} == {first.id, second.id}
    assert len(by_model["clients"]) == 2

    by_usecase = affected_clients(
        session, usecase="research-summary", start=BEFORE, end=LATER
    )
    assert {row["client_id"] for row in by_usecase["clients"]} == {
        "client-a",
        "client-b",
        "client-c",
    }

    by_policy = affected_clients(
        session, policy_id=str(first_policy.id), start=BEFORE, end=LATER
    )
    assert [row["client_id"] for row in by_policy["clients"]] == ["client-a"]
    assert by_policy["clients"][0]["interaction_id"] == first.id

    with pytest.raises(BreakerError, match="name one of"):
        affected_clients(session, model="1", usecase="research-summary", start=BEFORE, end=LATER)


def test_affected_clients_include_released_tool_output(session, evidence):
    _owners(session, evidence)
    released = open_interaction(
        session,
        client_id="client-tool",
        use_case="research-summary",
        evidence=evidence,
        now=NOW,
    )
    session.add(
        InteractionLink(interaction_id=released.id, correlation_id="tool-released")
    )
    session.flush()
    evidence.emit(
        session,
        correlation_id="tool-released",
        event_type=EventType.TOOL_CALL,
        actor_type=ActorType.AGENT,
        actor_id="dossier-agent",
        payload={
            "tool": "get_subject_profile",
            "subjects": {},
            "outcome": "ALLOW",
            "approval_required": False,
            "response_hash": "hash-tool",
        },
    )
    denied = open_interaction(
        session,
        client_id="client-denied-tool",
        use_case="research-summary",
        evidence=evidence,
        now=NOW,
    )
    evidence.emit(
        session,
        correlation_id=denied.id,
        event_type=EventType.TOOL_CALL,
        actor_type=ActorType.AGENT,
        actor_id="dossier-agent",
        payload={
            "tool": "get_subject_profile",
            "subjects": {},
            "outcome": "DENY",
            "approval_required": False,
            "response_hash": "hash-denied-tool",
        },
    )
    evidence.emit(
        session,
        correlation_id="tool-orphan",
        event_type=EventType.TOOL_CALL,
        actor_type=ActorType.AGENT,
        actor_id="dossier-agent",
        payload={
            "tool": "get_subject_profile",
            "subjects": {},
            "outcome": "ALLOW",
            "approval_required": False,
            "response_hash": "hash-orphan-tool",
        },
    )
    both = _received(session, evidence, "client-both", "research-summary")
    evidence.emit(
        session,
        correlation_id=both.id,
        event_type=EventType.TOOL_CALL,
        actor_type=ActorType.AGENT,
        actor_id="dossier-agent",
        payload={
            "tool": "get_subject_profile",
            "subjects": {},
            "outcome": "ALLOW",
            "approval_required": False,
            "response_hash": "hash-both-tool",
        },
    )
    egress_only = open_interaction(
        session,
        client_id="client-egress",
        use_case="research-summary",
        evidence=evidence,
        now=NOW,
    )
    evidence.emit(
        session,
        correlation_id=egress_only.id,
        event_type=EventType.EGRESS,
        actor_type=ActorType.AGENT,
        actor_id="dossier-agent",
        payload={"response_hash": "hash-egress", "sent": True},
    )

    listed = affected_clients(
        session, usecase="research-summary", start=BEFORE, end=LATER
    )
    assert {row["client_id"] for row in listed["clients"]} == {
        "client-tool",
        "client-both",
    }
    assert sum(1 for row in listed["clients"] if row["interaction_id"] == both.id) == 1
    by_model = affected_clients(session, model="1", start=BEFORE, end=LATER)
    assert {row["client_id"] for row in by_model["clients"]} == {"client-both"}


def test_revocation_of_each_dimension_denies_the_next_matching_decision(session, evidence):
    _owners(session, evidence)
    register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    register_agent(session, name="eligibility-assistant", owner_exec="Head of Data Governance")

    cases = [
        (
            "tool",
            "send_note",
            "Jane Smith",
            dict(tool="send_note"),
            dict(tool="get_subject_profile"),
        ),
        (
            "action",
            "send",
            "Jane Smith",
            dict(arguments={"action": "send"}),
            dict(tool="send", arguments={"action": "read"}),
        ),
        (
            "source",
            "archive",
            "Jane Smith",
            dict(attributes={"source": "archive"}),
            dict(attributes={"source": "catalogue"}),
        ),
        (
            "client",
            "client-9",
            "Jane Smith",
            dict(attributes={"client_id": "client-9"}),
            dict(subject_type="client", subject_id="client-8"),
        ),
        (
            "model",
            "1",
            "",
            dict(attributes={"model_version": "1"}),
            dict(attributes={"model_version": "2"}),
        ),
        (
            "usecase",
            "research-summary",
            "",
            dict(purpose="research-summary"),
            dict(purpose="briefing"),
        ),
        (
            "agent",
            "dossier-agent",
            "",
            dict(agent_name="dossier-agent"),
            dict(agent_name="eligibility-assistant"),
        ),
    ]
    for dimension, value, owner, match, other in cases:
        row = revoke(
            session,
            dimension,
            value,
            evidence=evidence,
            actor_id="Jane Smith",
            citation=f"Revocation of {dimension} {value}",
            owner=owner,
            now=NOW,
        )
        denied = _decide(session, **match)
        assert denied.outcome is Outcome.DENY, dimension
        revocation = _checks(denied, "REVOCATION")
        assert revocation, dimension
        assert value in revocation[0].citation
        assert row.citation in revocation[0].citation
        allowed = _decide(session, **other)
        assert allowed.outcome is Outcome.ALLOW, dimension
        assert not _checks(allowed, "REVOCATION")
        reinstate(
            session,
            row.id,
            evidence=evidence,
            actor_id=row.accountable_owners[0],
            citation=f"Reinstated {dimension} {value}",
            now=NOW,
        )
        assert _decide(session, **match).outcome is Outcome.ALLOW, dimension

    with pytest.raises(RevocationError, match="registered agent"):
        revoke(
            session,
            "tool",
            "send_note",
            evidence=evidence,
            actor_id="dossier-agent",
            citation="An agent cannot revoke a tool",
            now=NOW,
        )


def test_provider_revocation_stops_every_listed_use_case_and_honours_limits(session, evidence):
    _owners(session, evidence)
    row = revoke(
        session,
        "provider",
        "house",
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Provider house is outside appetite",
        now=NOW,
    )
    denied = _decide(session, purpose="briefing")
    assert denied.outcome is Outcome.DENY
    assert "house" in _checks(denied, "REVOCATION")[0].citation
    assert _decide(session).outcome is Outcome.ALLOW
    named = _decide(session, attributes={"provider": "house"})
    assert named.outcome is Outcome.DENY
    reinstate(
        session,
        row.id,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Provider house is back inside appetite",
        now=NOW,
    )

    revoke(
        session,
        "provider",
        "house",
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Provider house is limited to briefing below C2",
        limit_usecases=("briefing",),
        below_classification="C2",
        now=NOW,
    )
    assert (
        _decide(
            session,
            purpose="briefing",
            attributes={"provider": "house", "classification": "C1"},
        ).outcome
        is Outcome.ALLOW
    )
    assert (
        _decide(
            session,
            purpose="briefing",
            attributes={"provider": "house", "classification": "C2"},
        ).outcome
        is Outcome.DENY
    )
    assert (
        _decide(session, purpose="briefing", attributes={"provider": "house"}).outcome
        is Outcome.DENY
    )
    assert (
        _decide(
            session,
            attributes={"provider": "house", "classification": "C1"},
        ).outcome
        is Outcome.DENY
    )
    assert _decide(session, purpose="briefing").outcome is Outcome.DENY


def test_reinstatement_requires_the_owner_and_a_citation(session, evidence):
    _owners(session, evidence)
    row = revoke(
        session,
        "usecase",
        "research-summary",
        evidence=evidence,
        actor_id="Omar Aziz",
        citation="Stop research-summary",
        now=NOW,
    )
    with pytest.raises(RevocationError, match="cannot reinstate"):
        reinstate(
            session,
            row.id,
            evidence=evidence,
            actor_id="Omar Aziz",
            citation="Technology owner reinstating a use case they do not own",
            now=NOW,
        )
    with pytest.raises(RevocationError, match="cites its rule"):
        reinstate(
            session,
            row.id,
            evidence=evidence,
            actor_id="Jane Smith",
            citation="  ",
            now=NOW,
        )
    assert _decide(session).outcome is Outcome.DENY
    reinstate(
        session,
        row.id,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="The owner reinstated research-summary",
        now=NOW,
    )
    assert _decide(session).outcome is Outcome.ALLOW


def test_reinstating_a_provider_needs_every_owner(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec(model_providers=("house",)))
    _activate(
        session,
        evidence,
        _spec(
            usecase_id="briefing",
            purpose="Prepare a briefing",
            business_owner="Priya Shah",
            validators=("Priya Shah",),
            model_providers=("house",),
        ),
    )
    row = revoke(
        session,
        "provider",
        "house",
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Provider house is outside appetite",
        owner="Jane Smith",
        now=NOW,
    )
    assert set(row.accountable_owners) == {"Jane Smith", "Priya Shah"}
    with pytest.raises(RevocationError, match="cannot reinstate"):
        reinstate(
            session,
            row.id,
            evidence=evidence,
            actor_id="Omar Aziz",
            citation="Technology owner reinstating a provider they do not own",
            now=NOW,
        )
    pending = reinstate(
        session,
        row.id,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Jane reinstates house",
        now=NOW,
    )
    assert pending.effective_to is None
    assert pending.is_effective(NOW)
    again = reinstate(
        session,
        row.id,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Jane reinstates house again",
        now=NOW,
    )
    assert again.effective_to is None
    assert _decide(session, purpose="briefing").outcome is Outcome.DENY
    _activate(
        session,
        evidence,
        _spec(
            usecase_id="notes",
            purpose="Draft a note",
            business_owner="Luis Ortega",
            validators=("Luis Ortega",),
            model_providers=("house",),
        ),
    )
    still = reinstate(
        session,
        row.id,
        evidence=evidence,
        actor_id="Priya Shah",
        citation="Priya reinstates house",
        now=NOW,
    )
    assert still.effective_to is None
    assert _decide(session, purpose="notes").outcome is Outcome.DENY
    closed = reinstate(
        session,
        row.id,
        evidence=evidence,
        actor_id="Luis Ortega",
        citation="Luis reinstates house",
        now=NOW,
    )
    assert closed.effective_to == NOW
    assert not closed.is_effective(NOW)
    assert _decide(session, purpose="briefing").outcome is Outcome.ALLOW
    assert _decide(session, purpose="notes").outcome is Outcome.ALLOW
    actions = [
        event.payload.get("action")
        for event in session.exec(select(EvidenceEvent)).all()
        if isinstance(event.payload, dict)
    ]
    assert actions.count("REINSTATE_PENDING") == 3
    assert actions.count("REINSTATE") == 1


def test_agent_revocation_uses_the_kill_switch(session, evidence):
    _owners(session, evidence)
    register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    row = revoke(
        session,
        "agent",
        "dossier-agent",
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Stop dossier-agent",
        now=NOW,
    )
    agent = get_agent(session, "dossier-agent")
    assert agent is not None and agent.kill_switch is True
    with pytest.raises(RevocationError, match="Stop dossier-agent"):
        set_kill_switch(
            session,
            "dossier-agent",
            False,
            evidence=evidence,
            actor_id="Jane Smith",
            reason="A direct release does not reinstate the revocation",
            now=NOW,
        )
    held = get_agent(session, "dossier-agent")
    assert held is not None and held.kill_switch is True
    denied = _decide(session, agent_name="dossier-agent")
    assert denied.outcome is Outcome.DENY
    assert _checks(denied, "REVOCATION")
    assert _checks(denied, "KILL_SWITCH")
    assert "Stop dossier-agent" in _checks(denied, "REVOCATION")[0].citation
    reinstate(
        session,
        row.id,
        evidence=evidence,
        actor_id="Head of Research Ops",
        citation="The owner reinstated dossier-agent",
        now=NOW,
    )
    released = get_agent(session, "dossier-agent")
    assert released is not None and released.kill_switch is False
    assert _decide(session, agent_name="dossier-agent").outcome is Outcome.ALLOW


def test_kill_switch_drill_is_evidenced_and_an_overdue_drill_denies(session, evidence):
    _open(session, evidence)
    register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    _activate(
        session,
        evidence,
        _spec(
            residual_materiality="high",
            model_providers=("house",),
            agents=("dossier-agent",),
        ),
    )
    _activate(
        session,
        evidence,
        _spec(usecase_id="briefing", purpose="Prepare a briefing"),
    )
    unscheduled = _decide(session, agent_name="dossier-agent")
    assert unscheduled.outcome is Outcome.DENY
    assert "no drill scheduled" in _checks(unscheduled, "KILL_SWITCH_DRILL")[0].citation
    with pytest.raises(BreakerError, match="not high-materiality"):
        schedule_kill_switch_drill(
            session,
            "briefing",
            due_at=NOW,
            evidence=evidence,
            actor_id="Jane Smith",
            citation="A low-materiality use case has no drill",
            now=NOW,
        )

    drilled = run_kill_switch_drill(
        session,
        "research-summary",
        "dossier-agent",
        evidence=evidence,
        actor_id="Head of Research Ops",
        citation="Quarterly kill-switch drill",
        fallback="The RM used the house procedure manual",
        now=NOW,
    )
    assert drilled["activated_by"] == "Head of Research Ops"
    assert drilled["fallback"] == "The RM used the house procedure manual"
    assert drilled["latency_seconds"] >= 0
    assert "Kill switch" in drilled["denied_citation"]
    assert get_agent(session, "dossier-agent").kill_switch is False
    assert _decide(session, agent_name="dossier-agent").outcome is Outcome.ALLOW

    schedule_kill_switch_drill(
        session,
        "research-summary",
        due_at=NOW,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="The drill is due now",
        now=NOW,
    )
    overdue = _decide(session, agent_name="dossier-agent")
    assert overdue.outcome is Outcome.DENY
    assert "past its drill date" in _checks(overdue, "KILL_SWITCH_DRILL")[0].citation
    assert "no drill scheduled" not in _checks(overdue, "KILL_SWITCH_DRILL")[0].citation
    assert "Jane Smith" in _checks(overdue, "KILL_SWITCH_DRILL")[0].citation
    assert _decide(session, purpose="briefing").outcome is Outcome.ALLOW

    without_register = decide(
        Envelope(
            principal="alice",
            purpose="research-summary",
            tool="get_subject_profile",
            agent_name="dossier-agent",
        ),
        load_snapshot(session, as_of=NOW),
        purposes=("research-summary",),
        as_of=NOW,
    )
    assert without_register.outcome is Outcome.DENY
    assert _checks(without_register, "KILL_SWITCH_DRILL")
    assert verify_chain(session) > 0


def test_a_high_materiality_pilot_or_live_use_case_with_no_drill_denies(session, evidence):
    _open(session, evidence)
    _activate(
        session,
        evidence,
        _spec(residual_materiality="high", model_providers=("house",)),
    )
    _activate(
        session,
        evidence,
        _spec(
            usecase_id="pilot-summary",
            purpose="Pilot the summary with a named user list",
            residual_materiality="high",
            model_providers=("house",),
            pilot_users=("alice",),
            pilot_end=NOW + timedelta(days=10),
            success_criteria="RM acceptance above the baseline",
        ),
        status="pilot",
    )
    _activate(
        session,
        evidence,
        _spec(usecase_id="briefing", purpose="Prepare a briefing"),
    )
    live = _decide(session)
    assert live.outcome is Outcome.DENY
    assert "no drill scheduled" in _checks(live, "KILL_SWITCH_DRILL")[0].citation
    assert "Jane Smith" in _checks(live, "KILL_SWITCH_DRILL")[0].citation
    pilot = _decide(session, purpose="pilot-summary")
    assert pilot.outcome is Outcome.DENY
    assert "no drill scheduled" in _checks(pilot, "KILL_SWITCH_DRILL")[0].citation
    assert _decide(session, purpose="briefing").outcome is Outcome.ALLOW

    schedule_kill_switch_drill(
        session,
        "research-summary",
        due_at=NOW + timedelta(days=1),
        evidence=evidence,
        actor_id="Jane Smith",
        citation="The live drill is tomorrow",
        now=NOW,
    )
    assert _decide(session).outcome is Outcome.ALLOW
    assert _decide(session, purpose="pilot-summary").outcome is Outcome.DENY

    without_register = decide(
        Envelope(
            principal="alice",
            purpose="pilot-summary",
            tool="get_subject_profile",
        ),
        load_snapshot(session, as_of=NOW),
        purposes=("pilot-summary",),
        as_of=NOW,
    )
    assert without_register.outcome is Outcome.DENY
    assert not _checks(without_register, "KILL_SWITCH_DRILL")

    schedule_kill_switch_drill(
        session,
        "pilot-summary",
        due_at=NOW,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="The pilot drill is due now",
        now=NOW,
    )
    overdue = _decide(session, purpose="pilot-summary")
    assert overdue.outcome is Outcome.DENY
    assert "is past its drill date" in _checks(overdue, "KILL_SWITCH_DRILL")[0].citation
    assert "no drill scheduled" not in _checks(overdue, "KILL_SWITCH_DRILL")[0].citation


def test_a_drill_does_not_release_a_switch_an_open_revocation_holds(session, evidence):
    _open(session, evidence)
    register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    _activate(
        session,
        evidence,
        _spec(
            residual_materiality="high",
            model_providers=("house",),
            agents=("dossier-agent",),
        ),
    )
    revoke(
        session,
        "agent",
        "dossier-agent",
        evidence=evidence,
        actor_id="Jane Smith",
        citation="The agent stays revoked through the drill",
        now=NOW,
    )
    run_kill_switch_drill(
        session,
        "research-summary",
        "dossier-agent",
        evidence=evidence,
        actor_id="Head of Research Ops",
        citation="Drill while the agent is revoked",
        fallback="The RM used the house procedure manual",
        now=NOW,
    )
    assert get_agent(session, "dossier-agent").kill_switch is True
    assert _checks(_decide(session, agent_name="dossier-agent"), "REVOCATION")


def test_a_pack_cannot_skip_a_breaker_and_can_still_skip_another_policy(session, evidence):
    _owners(session, evidence)
    session.add(
        Policy(
            regime="INTERNAL",
            rule_type="PROHIBITED",
            rule={"description": "model calls barred", "on_violation": "fail"},
            citation="No external model calls",
            effective_from=NOW,
        )
    )
    session.add(
        Policy(
            regime="CIRCUIT_BREAKER",
            rule_type="CIRCUIT_BREAKER",
            rule={
                "dimension": "usecase",
                "value": "research-summary",
                "on_violation": "fail",
                "accountable_owner": "Jane Smith",
            },
            citation="Circuit breaker — usecase research-summary",
            effective_from=NOW,
        )
    )
    session.flush()
    evaluation = _decide(session, engages=lambda _policy, _context: False)
    assert evaluation.outcome is Outcome.DENY
    assert _checks(evaluation, "CIRCUIT_BREAKER")
    assert not any(check.check == "model calls barred" for check in evaluation.checks)
    assert _decide(session, purpose="briefing", engages=lambda _policy, _context: False).outcome is Outcome.ALLOW


def test_containment_denials_do_not_count_as_flags(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec())
    evidence.emit(
        session,
        correlation_id="contained",
        event_type=EventType.POLICY_DECISION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload={
            "outcome": "DENY",
            "envelope": {"purpose": "research-summary"},
            "checks": [{"check": "REVOCATION", "result": "fail"}],
        },
    )
    set_breaker_threshold(
        session,
        "usecase",
        "research-summary",
        threshold=1,
        window_seconds=3600,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Flags: one denial trips the use case",
        now=NOW,
    )
    assert evaluate_breakers(session, evidence=evidence, now=NOW) == ()
    evidence.emit(
        session,
        correlation_id="denied",
        event_type=EventType.POLICY_DECISION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload={
            "outcome": "DENY",
            "envelope": {
                "purpose": "research-summary",
                "arguments": {"model_version": "9", "prompt_template_version": "prompt-9"},
            },
            "checks": [{"check": "PURPOSE", "result": "fail"}],
        },
    )
    tripped = evaluate_breakers(session, evidence=evidence, now=NOW)
    assert len(tripped) == 1
    assert tripped[0]["dimension"] == "usecase"

    set_breaker_threshold(
        session,
        "model",
        "9",
        threshold=1,
        window_seconds=3600,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Flags: the denied model version trips",
        owner="Jane Smith",
        now=NOW,
    )
    model_trip = evaluate_breakers(session, evidence=evidence, now=NOW)
    assert any(item["dimension"] == "model" and item["value"] == "9" for item in model_trip)


def test_tamper_for_test_breaks_the_chain_without_fixing_the_hashes(session, evidence):
    evidence.emit(
        session,
        correlation_id="one",
        event_type=EventType.POLICY_CHANGE,
        actor_type=ActorType.HUMAN,
        actor_id="Jane Smith",
        payload={"action": "NOTE", "citation": "a row to edit"},
    )
    event = session.exec(select(EvidenceEvent)).one()
    payload_hash = event.payload_hash
    event_hash = event.event_hash
    tamper_for_test(session, event.sequence, {**event.payload, "tampered": True})
    assert event.payload["tampered"] is True
    assert event.payload_hash == payload_hash
    assert event.event_hash == event_hash
    with pytest.raises(ChainBreak, match="payload does not match its hash"):
        verify_chain(session)


def test_a_circuit_breaker_rule_names_its_dimension(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec())
    accepted, findings = parse_rule(
        {
            "id": "breaker.usecase",
            "regime": "CIRCUIT_BREAKER",
            "rule_type": "CIRCUIT_BREAKER",
            "citation": "Circuit breaker",
            "effective_from": "2026-06-01T12:00:00+00:00",
            "rule": {
                "dimension": "usecase",
                "value": "research-summary",
                "accountable_owner": "Jane Smith",
            },
        },
        source="breaker.yaml",
    )
    assert findings == []
    assert accepted is not None
    _rejected, rejected = parse_rule(
        {
            "id": "breaker.bad",
            "regime": "CIRCUIT_BREAKER",
            "rule_type": "CIRCUIT_BREAKER",
            "citation": "Circuit breaker",
            "effective_from": "2026-06-01T12:00:00+00:00",
            "rule": {},
        },
        source="breaker.yaml",
    )
    messages = " ".join(item.message for item in rejected)
    assert "dimension must be" in messages
    assert "value must be" in messages

    session.add(
        Policy(
            regime="CIRCUIT_BREAKER",
            rule_type="CIRCUIT_BREAKER",
            rule={},
            citation="Circuit breaker — unscoped",
            effective_from=NOW,
        )
    )
    session.add(
        Revocation(
            dimension="nope",
            value="x",
            citation="unknown dimension",
            actor_id="Jane Smith",
            accountable_owners=["Jane Smith"],
            effective_from=NOW,
        )
    )
    session.flush()
    denied = _decide(session, purpose="briefing")
    assert denied.outcome is Outcome.DENY
    assert _checks(denied, "CIRCUIT_BREAKER")
    assert _checks(denied, "REVOCATION")


def test_a_model_threshold_with_two_owners_names_one(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec(model_providers=("house",)))
    _activate(
        session,
        evidence,
        _spec(
            usecase_id="briefing",
            purpose="Prepare a briefing",
            business_owner="Priya Shah",
            validators=("Priya Shah",),
            model_providers=("house",),
        ),
    )
    with pytest.raises(BreakerError, match="name the accountable owner"):
        set_breaker_threshold(
            session,
            "model",
            "1",
            threshold=2,
            window_seconds=3600,
            evidence=evidence,
            actor_id="Jane Smith",
            citation="Flags: the model version needs one owner",
            now=NOW,
        )
    set_breaker_threshold(
        session,
        "model",
        "1",
        threshold=2,
        window_seconds=3600,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Flags: Jane owns this model threshold",
        owner="Jane Smith",
        now=NOW,
    )


def test_cli_affected_revoke_and_breaker(tmp_path, capsys):
    path = tmp_path / "kognita.db"
    engine = make_engine(path)
    create_all(engine)
    with Session(engine) as file_session:
        file_evidence = EvidenceWriter(engine, clock=lambda: NOW)
        _owners(file_session, file_evidence)
        _received(file_session, file_evidence, "client-a", "research-summary")
        _received(file_session, file_evidence, "client-b", "briefing", model_version="2")
        file_session.commit()

    start = BEFORE.isoformat()
    end = LATER.isoformat()
    assert (
        main(
            [
                "evidence",
                "affected",
                "--db",
                str(path),
                "--model",
                "1",
                "--from",
                start,
                "--to",
                end,
            ]
        )
        == 0
    )
    listed = json.loads(capsys.readouterr().out)
    assert [row["client_id"] for row in listed["clients"]] == ["client-a"]

    assert (
        main(
            [
                "evidence",
                "affected",
                "--db",
                str(path),
                "--model",
                "1",
                "--usecase",
                "research-summary",
                "--from",
                start,
                "--to",
                end,
            ]
        )
        == 2
    )

    assert (
        main(
            [
                "revoke",
                "--db",
                str(path),
                "--usecase",
                "research-summary",
                "--actor",
                "Omar Aziz",
                "--citation",
                "Stop research-summary",
            ]
        )
        == 0
    )
    revoked = json.loads(capsys.readouterr().out)
    assert revoked["dimension"] == "usecase"
    assert revoked["accountable_owners"] == ["Jane Smith"]

    with Session(engine) as file_session:
        register = load_register(file_session)
        denied = decide(
            Envelope(principal="alice", purpose="research-summary", tool="get_subject_profile"),
            load_snapshot(file_session),
            usecases=register,
        )
        allowed = decide(
            Envelope(principal="alice", purpose="briefing", tool="get_subject_profile"),
            load_snapshot(file_session),
            usecases=register,
        )
        assert denied.outcome is Outcome.DENY
        assert allowed.outcome is Outcome.ALLOW

    assert (
        main(
            [
                "reinstate",
                "--db",
                str(path),
                "--id",
                str(revoked["revocation_id"]),
                "--actor",
                "Jane Smith",
                "--citation",
                "The owner reinstated research-summary",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "breaker",
                "threshold",
                "--db",
                str(path),
                "--dimension",
                "usecase",
                "--value",
                "research-summary",
                "--threshold",
                "2",
                "--window-seconds",
                "3600",
                "--actor",
                "Jane Smith",
                "--citation",
                "Flags: two near misses trip the use case",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["accountable_owner"] == "Jane Smith"

    register_agent_path = path
    with Session(engine) as file_session:
        register_agent(file_session, name="dossier-agent", owner_exec="Head of Research Ops")
        file_session.commit()
    assert (
        main(
            [
                "revoke",
                "--db",
                str(register_agent_path),
                "--tool",
                "send_note",
                "--actor",
                "dossier-agent",
                "--citation",
                "An agent cannot revoke a tool",
            ]
        )
        == 1
    )
    assert "registered agent" in capsys.readouterr().err


def test_gateway_records_the_provider_and_a_revoked_provider_stops_the_next_call(
    session, evidence
):
    register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    _live(session, model="summary", version="1")
    session.commit()
    upstream_calls: list[bytes] = []

    def transport(method, url, headers, body):
        upstream_calls.append(body)
        return (
            200,
            {"content-type": "application/json"},
            b'{"model":"summary","model_version":"1","choices":[{"message":{"content":"ok"}}]}',
        )

    gateway = Gateway(
        engine=session.get_bind(),
        evidence=evidence,
        upstream="https://api.openai.com",
        client=ClientConfiguration(
            principal="alice",
            purpose="COLLABORATION",
            agent_names=frozenset({"dossier-agent"}),
            actor_location="SG",
        ),
        purposes=("COLLABORATION",),
        transport=transport,
    )
    allowed = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent", "prompt_template_version": "prompt-9"},
        json.dumps(
            {"model": "summary", "messages": [{"role": "user", "content": "market hours are public"}]}
        ).encode(),
        session=session,
    )
    assert allowed.evaluation is not None
    assert allowed.evaluation.attributes["provider"] == "api.openai.com"
    assert allowed.evaluation.envelope.arguments["prompt_template_version"] == "prompt-9"
    assert allowed.evaluation.outcome is Outcome.ALLOW
    assert len(upstream_calls) == 1

    revoke(
        session,
        "provider",
        "api.openai.com",
        evidence=evidence,
        actor_id="Head of Research Ops",
        citation="Stop the upstream provider",
        owner="Head of Research Ops",
        now=NOW,
    )
    session.commit()
    denied = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent"},
        json.dumps(
            {"model": "summary", "messages": [{"role": "user", "content": "market hours are public"}]}
        ).encode(),
        session=session,
    )
    assert denied.evaluation is not None
    assert denied.evaluation.outcome is Outcome.DENY
    assert _checks(denied.evaluation, "REVOCATION")
    assert len(upstream_calls) == 1


def test_prompt_revocation_is_not_a_dimension_and_prompt_breaker_matches(session, evidence):
    """Prompt is a breaker dimension. It is not a revocation dimension."""
    _open(session, evidence)
    _activate(session, evidence, _spec())
    set_breaker_threshold(
        session,
        "prompt",
        "prompt-3",
        threshold=1,
        window_seconds=3600,
        evidence=evidence,
        actor_id="Jane Smith",
        citation="Flags: the prompt version trips",
        owner="Jane Smith",
        now=NOW,
    )
    evidence.emit(
        session,
        correlation_id="prompt-flag",
        event_type=EventType.NEAR_MISS,
        actor_type=ActorType.HUMAN,
        actor_id="Jane Smith",
        payload={
            "action": "RECORDED",
            "usecase_id": "research-summary",
            "prompt_template_version": "prompt-3",
        },
    )
    evaluate_breakers(session, evidence=evidence, now=NOW)
    denied = _decide(session, arguments={"prompt_template_version": "prompt-3"})
    assert denied.outcome is Outcome.DENY
    assert _decide(session, arguments={"prompt_template_version": "prompt-4"}).outcome is Outcome.ALLOW
    with pytest.raises(RevocationError, match="not a revocation dimension"):
        revoke(
            session,
            "prompt",
            "prompt-3",
            evidence=evidence,
            actor_id="Jane Smith",
            citation="Prompt is not a revocation dimension",
            now=NOW,
        )


def test_gateway_withholds_a_revoked_model_version_the_provider_served(session, evidence):
    """The caller sends no model_version header; the response names a revoked one."""
    register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    # Revoke before the use case lists the card, so the named owner stands.
    # The request names only "summary"; the response is what matches the card.
    revoke(
        session,
        "model",
        "api.openai.com/summary@1",
        evidence=evidence,
        actor_id="Head of Research Ops",
        citation="Stop summary version 1",
        owner="Head of Research Ops",
        now=NOW,
    )
    _live(session, model="summary", version="1")
    session.commit()
    upstream_calls: list[bytes] = []

    def transport(method, url, headers, body):
        upstream_calls.append(body)
        return (
            200,
            {"content-type": "application/json"},
            b'{"model":"summary","model_version":"1","choices":[{"message":{"content":"v1 body"}}]}',
        )

    gateway = Gateway(
        engine=session.get_bind(),
        evidence=evidence,
        upstream="https://api.openai.com",
        client=ClientConfiguration(
            principal="alice",
            purpose="COLLABORATION",
            agent_names=frozenset({"dossier-agent"}),
            actor_location="SG",
        ),
        purposes=("COLLABORATION",),
        transport=transport,
    )
    response = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent"},
        json.dumps(
            {"model": "summary", "messages": [{"role": "user", "content": "market hours are public"}]}
        ).encode(),
        session=session,
    )
    session.commit()
    assert response.status == 403
    assert b"v1 body" not in response.body
    withheld = [
        event
        for event in session.exec(select(EvidenceEvent)).all()
        if event.payload.get("withheld") is True
    ]
    assert withheld and withheld[-1].payload["checks"][0]["check"] == "REVOCATION"
    # The client was given the denial, not the output: a withheld response is
    # not receipt, even though the model call was recorded with its hash.
    calls = [
        event
        for event in session.exec(select(EvidenceEvent)).all()
        if event.event_type == EventType.MODEL_CALL
        and event.correlation_id == withheld[-1].correlation_id
    ]
    assert calls and calls[-1].payload.get("response_hash")
    at = calls[-1].recorded_at
    at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    window = dict(start=at - timedelta(days=1), end=at + timedelta(days=1))
    affected = affected_clients(session, model="summary", **window)
    assert affected["clients"] == []
    assert affected["unlinked"] == []
    assert affected_clients(session, usecase="COLLABORATION", **window)["unlinked"] == []


def test_gateway_denies_a_principal_header_that_differs_from_the_bound_client(
    session, evidence
):
    """A caller cannot borrow another principal's name through a header."""
    register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    _live(session, model="summary", version="1")
    session.commit()
    upstream_calls: list[bytes] = []

    def transport(method, url, headers, body):
        upstream_calls.append(body)
        return (
            200,
            {"content-type": "application/json"},
            b'{"model":"summary","model_version":"1","choices":[{"message":{"content":"ok"}}]}',
        )

    gateway = Gateway(
        engine=session.get_bind(),
        evidence=evidence,
        upstream="https://api.openai.com",
        client=ClientConfiguration(
            principal="bob",
            purpose="COLLABORATION",
            agent_names=frozenset({"dossier-agent"}),
            actor_location="SG",
        ),
        purposes=("COLLABORATION",),
        transport=transport,
    )
    body = json.dumps(
        {"model": "summary", "messages": [{"role": "user", "content": "market hours are public"}]}
    ).encode()
    borrowed = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent", "principal": "alice"},
        body,
        session=session,
    )
    assert borrowed.evaluation is not None
    assert borrowed.evaluation.outcome is Outcome.DENY
    assert borrowed.evaluation.envelope.principal == "bob"
    assert any(
        check.check == "AGENT_IDENTITY" and "principal header" in check.citation
        for check in borrowed.evaluation.failures()
    )
    other_purpose = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent", "purpose": "DOSSIER_PREP"},
        body,
        session=session,
    )
    assert other_purpose.evaluation.outcome is Outcome.DENY
    same = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent", "principal": "bob"},
        body,
        session=session,
    )
    assert same.evaluation.outcome is Outcome.ALLOW
    assert len(upstream_calls) == 1


def test_a_gateway_bound_to_a_run_refuses_a_foreign_run_id(session, evidence):
    """A run_id header cannot move a bound client onto another budget."""
    from kognita.tools import Run

    register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    _live(session, model="summary", version="1")
    session.commit()
    upstream_calls: list[bytes] = []

    def transport(method, url, headers, body):
        upstream_calls.append(body)
        return (
            200,
            {"content-type": "application/json"},
            b'{"model":"summary","model_version":"1","choices":[{"message":{"content":"ok"}}]}',
        )

    bound = Run(max_calls=5)
    gateway = Gateway(
        engine=session.get_bind(),
        evidence=evidence,
        upstream="https://api.openai.com",
        client=ClientConfiguration(
            principal="alice",
            purpose="COLLABORATION",
            agent_names=frozenset({"dossier-agent"}),
            actor_location="SG",
        ),
        purposes=("COLLABORATION",),
        transport=transport,
        run=bound,
    )
    body = json.dumps(
        {"model": "summary", "messages": [{"role": "user", "content": "market hours are public"}]}
    ).encode()
    foreign = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent", "run_id": "someone-elses-run"},
        body,
        session=session,
    )
    assert foreign.status == 403
    assert b"not bound" in foreign.body
    assert upstream_calls == []
    own = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent", "run_id": bound.id},
        body,
        session=session,
    )
    assert own.status == 200
    assert len(upstream_calls) == 1


def test_gateway_output_with_no_interaction_is_listed_as_client_unknown(session, evidence):
    """A gateway call has no interaction; the query says so instead of omitting it."""
    from datetime import datetime, timezone

    register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    _live(session, model="summary", version="1")
    session.commit()

    def transport(method, url, headers, body):
        return (
            200,
            {"content-type": "application/json"},
            b'{"model":"summary","model_version":"1","choices":[{"message":{"content":"ok"}}]}',
        )

    gateway = Gateway(
        engine=session.get_bind(),
        evidence=evidence,
        upstream="https://api.openai.com",
        client=ClientConfiguration(
            principal="alice",
            purpose="COLLABORATION",
            agent_names=frozenset({"dossier-agent"}),
            actor_location="SG",
        ),
        purposes=("COLLABORATION",),
        transport=transport,
    )
    response = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent"},
        json.dumps(
            {"model": "summary", "messages": [{"role": "user", "content": "market hours are public"}]}
        ).encode(),
        session=session,
    )
    session.commit()
    assert response.status == 200
    calls = [
        event
        for event in session.exec(select(EvidenceEvent)).all()
        if event.event_type == EventType.MODEL_CALL
    ]
    assert calls, "the gateway recorded a model call"
    at = calls[-1].recorded_at
    at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    window = dict(start=at - timedelta(days=1), end=at + timedelta(days=1))
    by_model = affected_clients(session, model="summary", **window)
    assert by_model["clients"] == []
    assert [row["client"] for row in by_model["unlinked"]] == ["client unknown"]
    by_usecase = affected_clients(session, usecase="COLLABORATION", **window)
    assert len(by_usecase["unlinked"]) == 1
    assert affected_clients(session, usecase="OTHER", **window)["unlinked"] == []
