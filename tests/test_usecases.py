"""The use-case register replaces the free-text purpose list when a caller supplies one.

Decisions in this file name a registered use case and cite its accountable
owner. They use no network and no API key.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.envelope import Envelope
from kognita.evidence import verify_chain
from kognita.exceptions import KognitaError
from kognita.governance import PolicySnapshot, decide
from kognita.models import EvidenceEvent, UseCase
from kognita.registry import register as register_agent
from kognita.usecases import (
    ACCOUNTABILITY,
    Appetite,
    UseCaseRegister,
    UseCaseSpec,
    activate,
    approve_rating,
    derived_tier,
    list_usecases,
    load_register,
    register,
    retire,
    revise,
    set_appetite,
    suspend,
)
from kognita.vocabulary import CheckResult, EventType, Outcome

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

LOW_COMPLEXITY = {
    "technology": "established",
    "novelty": "familiar",
    "explainability": "explainable",
    "third_party_visibility": "not_applicable",
}


def _appetite(**overrides) -> Appetite:
    base = Appetite(
        allowed_categories=(
            "research summarization",
            "client intelligence",
            "meeting preparation",
            "kyc",
            "suitability",
            "recommendations",
            "financial execution",
        ),
        prohibited_categories=(),
        maximum_autonomy=4,
        permitted_client_impact=(
            "internal productivity",
            "rm assistance",
            "client influence",
            "client communication",
            "advice",
            "execution",
        ),
        external_communication=True,
        transaction_authority=True,
        maximum_residual_materiality="high",
    )
    return replace(base, **overrides)


def _spec(**overrides) -> UseCaseSpec:
    base = UseCaseSpec(
        usecase_id="research-summary",
        purpose="Summarise approved research for an RM",
        category="research summarization",
        business_owner="Jane Smith",
        technology_owner="Omar Aziz",
        risk_owner="Priya Shah",
        model_owner="Luis Ortega",
        knowledge_lead="",
        accountability={seat: "Jane Smith" for seat in ACCOUNTABILITY},
        control_function="Alex Chen",
        builders=("Ann Builder",),
        business_criticality="low",
        client_impact="internal productivity",
        autonomy="none",
        decision_consequence="informational",
        data_sensitivity="internal",
        complexity=LOW_COMPLEXITY,
        inherent_materiality="medium",
        credited_controls=("entitlement filter",),
        residual_materiality="low",
        approval_gate="none",
        review_sampling_rate=0.1,
        maximum_autonomy=1,
        monitoring=True,
        explanation_depth="citation",
        validators=("Jane Smith",),
        human_decision_points=(),
        client_segments=("private banking",),
        investor_impact="none; internal research only",
        permitted_data_classes=("internal",),
        approved_models=("house-summary@1",),
        agent_authority="retrieve",
        retention_days=365,
        business_process="RM research",
        data_sources=("data-inventory:research",),
        vendors=("outsourcing:none",),
    )
    return replace(base, **overrides)


def _tier2(**overrides) -> UseCaseSpec:
    values = dict(
        usecase_id="client-intelligence",
        purpose="Prepare a client briefing",
        category="client intelligence",
        client_impact="rm assistance",
        knowledge_lead="Jane Smith",
        approval_gate="human_review",
        review_sampling_rate=0.2,
        maximum_autonomy=3,
        explanation_depth="sources",
        validators=("Jane Smith",),
    )
    values.update(overrides)
    return _spec(**values)


def _tier3(**overrides) -> UseCaseSpec:
    values = dict(
        usecase_id="suitability-check",
        purpose="Check suitability before a recommendation",
        category="suitability",
        knowledge_lead="Jane Smith",
        business_criticality="medium",
        client_impact="advice",
        autonomy="assisted",
        decision_consequence="suitability",
        data_sensitivity="confidential",
        approval_gate="human_approval",
        review_sampling_rate=0.5,
        maximum_autonomy=4,
        explanation_depth="suitability",
        validators=("Jane Smith", "Noel Independent"),
        human_decision_points=("RM approves the recommendation",),
    )
    values.update(overrides)
    return _spec(**values)


def _open(session, evidence, appetite: Appetite | None = None) -> None:
    set_appetite(
        session,
        appetite or _appetite(),
        evidence=evidence,
        actor_id="CRO",
        senior_management=True,
        at=NOW,
    )


def _activate(session, evidence, spec: UseCaseSpec, **kwargs) -> None:
    register(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    approve_rating(
        session,
        spec.usecase_id,
        spec.control_function,
        evidence=evidence,
        at=NOW,
    )
    activate(
        session,
        spec.usecase_id,
        evidence=evidence,
        actor_id=kwargs.pop("actor_id", spec.control_function),
        at=NOW,
        status=kwargs.pop("status", "live"),
        **kwargs,
    )


def _envelope(purpose: str, principal: str = "alice") -> Envelope:
    return Envelope(principal=principal, purpose=purpose, tool="get_subject_profile")


def _named(evaluation, check: str):
    matches = [item for item in evaluation.checks if item.check == check]
    assert matches, [item.check for item in evaluation.checks]
    return matches


def test_derived_tier_is_the_highest_rank():
    """A research use case on highly sensitive data is not left at tier 1."""
    assert derived_tier(_spec()) == 1
    assert derived_tier(_tier2()) == 2
    assert derived_tier(_tier3()) == 3
    lifted = derived_tier(_spec(data_sensitivity="highly sensitive"))
    assert lifted == 3


def test_register_replaces_the_purpose_list():
    """A purpose on the free-text list is still denied when no use case is registered."""
    evaluation = decide(
        _envelope("COLLABORATION"),
        PolicySnapshot(),
        purposes=("COLLABORATION",),
        usecases=UseCaseRegister(),
        as_of=NOW,
    )
    assert evaluation.outcome is Outcome.DENY
    assert not any(check.check == "PURPOSE" for check in evaluation.checks)
    check = _named(evaluation, "USE_CASE_REGISTERED")[0]
    assert check.result is CheckResult.FAIL
    assert "not registered" in check.citation
    assert "Use-case register" in check.citation


def test_a_live_use_case_is_allowed_and_cites_its_owner(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec())
    register_snapshot = load_register(session, as_of=NOW)
    evaluation = decide(
        _envelope("research-summary", principal="alice"),
        PolicySnapshot(),
        purposes=("research-summary",),
        usecases=register_snapshot,
        as_of=NOW,
    )
    assert evaluation.outcome is Outcome.ALLOW
    check = _named(evaluation, "USE_CASE")[0]
    assert check.result is CheckResult.PASS
    assert "accountable owner: Jane Smith" in check.citation
    assert "tier 1" in check.citation
    assert not any(item.check == "PURPOSE" for item in evaluation.checks)


def test_tier_2_without_a_knowledge_lead_is_denied(session, evidence):
    """The role may be the business owner, but it has to be named."""
    _open(session, evidence)
    spec = _tier2(knowledge_lead="")
    register(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    with pytest.raises(KognitaError, match="no Knowledge Lead"):
        activate(
            session,
            spec.usecase_id,
            status="live",
            evidence=evidence,
            actor_id="Alex Chen",
            at=NOW,
        )
    held = replace(spec, knowledge_lead="Jane Smith")
    assert derived_tier(held) == 2


def test_a_decision_denies_a_tier_2_use_case_with_no_knowledge_lead():
    from kognita.usecases import build_record, from_record

    spec = _tier2(knowledge_lead="")
    record = build_record(
        spec,
        version=1,
        status="live",
        rating_approved=True,
        rating_approved_by="Alex Chen",
        review_date=NOW + timedelta(days=30),
    )
    entry = from_record(record)
    evaluation = decide(
        _envelope(spec.usecase_id),
        PolicySnapshot(),
        usecases=UseCaseRegister(entries=(entry,), appetite=_appetite()),
        as_of=NOW,
    )
    assert evaluation.outcome is Outcome.DENY
    check = _named(evaluation, "USE_CASE_KNOWLEDGE_LEAD")[0]
    assert "no Knowledge Lead" in check.citation
    assert "accountable owner: Jane Smith" in check.citation


def test_an_agent_cannot_be_an_owner(session, evidence):
    _open(session, evidence)
    register_agent(session, name="dossier-agent", owner_exec="Jane Smith")
    spec = _spec(business_owner="dossier-agent")
    with pytest.raises(KognitaError, match="names agent 'dossier-agent'"):
        register(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    assert list_usecases(session, as_of=NOW) == ()


def test_a_decision_denies_an_agent_owner():
    from kognita.usecases import build_record, from_record

    spec = _spec(knowledge_lead="dossier-agent")
    record = build_record(
        spec,
        version=1,
        status="live",
        rating_approved=True,
        review_date=NOW + timedelta(days=30),
    )
    entry = from_record(record)
    evaluation = decide(
        _envelope(spec.usecase_id),
        PolicySnapshot(),
        usecases=UseCaseRegister(
            entries=(entry,),
            appetite=_appetite(),
            agents=frozenset({"dossier-agent"}),
        ),
        as_of=NOW,
    )
    assert evaluation.outcome is Outcome.DENY
    check = _named(evaluation, "USE_CASE_AGENT_OWNER")[0]
    assert "dossier-agent" in check.citation
    assert "Knowledge Lead" in check.citation


def test_use_case_outside_the_appetite_is_not_registered(session, evidence):
    allowed = tuple(
        category
        for category in _appetite().allowed_categories
        if category != "financial execution"
    )
    _open(
        session,
        evidence,
        _appetite(
            allowed_categories=allowed,
            prohibited_categories=("financial execution",),
            transaction_authority=False,
        ),
    )
    spec = _spec(
        category="financial execution",
        client_impact="execution",
        decision_consequence="financial action",
    )
    with pytest.raises(KognitaError, match="outside the risk appetite"):
        register(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    assert session.exec(select(UseCase)).all() == []


def test_residual_materiality_outside_the_appetite_cannot_be_registered(
    session, evidence
):
    _open(session, evidence, _appetite(maximum_residual_materiality="medium"))
    with pytest.raises(
        KognitaError, match="residual materiality is outside the risk appetite"
    ):
        register(
            session,
            _spec(residual_materiality="high"),
            evidence=evidence,
            actor_id="Jane Smith",
            at=NOW,
        )
    assert session.exec(select(UseCase)).all() == []


def test_activation_rechecks_residual_materiality(session, evidence):
    """Registered inside the appetite, then the ceiling drops before activation."""
    _open(session, evidence, _appetite(maximum_residual_materiality="high"))
    spec = _spec(residual_materiality="medium")
    register(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    approve_rating(session, spec.usecase_id, "Alex Chen", evidence=evidence, at=NOW)
    later = NOW + timedelta(days=1)
    set_appetite(
        session,
        _appetite(maximum_residual_materiality="low"),
        evidence=evidence,
        actor_id="CRO",
        senior_management=True,
        at=later,
    )
    with pytest.raises(
        KognitaError, match="residual materiality is outside the risk appetite"
    ):
        activate(
            session,
            spec.usecase_id,
            status="live",
            evidence=evidence,
            actor_id="Alex Chen",
            at=later,
        )


def test_unapproved_rating_cannot_be_activated(session, evidence):
    _open(session, evidence)
    spec = _spec()
    register(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    with pytest.raises(KognitaError, match="has not approved cannot be activated"):
        activate(
            session,
            spec.usecase_id,
            status="live",
            evidence=evidence,
            actor_id="Alex Chen",
            at=NOW,
        )


def test_control_function_cannot_approve_a_use_case_it_owns(session, evidence):
    _open(session, evidence)
    with pytest.raises(KognitaError, match="cannot approve a use case it owns"):
        register(
            session,
            _spec(control_function="Jane Smith"),
            evidence=evidence,
            actor_id="Jane Smith",
            at=NOW,
        )
    spec = _spec()
    register(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    with pytest.raises(KognitaError, match="cannot approve a use case it owns"):
        approve_rating(session, spec.usecase_id, "Omar Aziz", evidence=evidence, at=NOW)
    with pytest.raises(KognitaError, match="designated control function"):
        approve_rating(
            session, spec.usecase_id, "Priya Shah", evidence=evidence, at=NOW
        )
    approve_rating(session, spec.usecase_id, "Alex Chen", evidence=evidence, at=NOW)
    shown = load_register(session, as_of=NOW).get(spec.usecase_id)
    assert shown is not None
    assert shown.rating_approved is True
    assert shown.rating_approved_by == "Alex Chen"
    assert shown.rating_approved_at == NOW


def test_tier_3_cannot_be_signed_off_by_its_builder(session, evidence):
    _open(session, evidence)
    spec = _tier3()
    assert derived_tier(spec) == 3
    register(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    approve_rating(session, spec.usecase_id, "Alex Chen", evidence=evidence, at=NOW)
    with pytest.raises(KognitaError, match="cannot be signed off by its builder"):
        activate(
            session,
            spec.usecase_id,
            status="live",
            evidence=evidence,
            actor_id="Ann Builder",
            signoff_by="Noel Independent",
            at=NOW,
        )
    with pytest.raises(KognitaError, match="cannot be signed off by its builder"):
        activate(
            session,
            spec.usecase_id,
            status="live",
            evidence=evidence,
            actor_id="Alex Chen",
            signoff_by="Ann Builder",
            at=NOW,
        )
    activate(
        session,
        spec.usecase_id,
        status="live",
        evidence=evidence,
        actor_id="Alex Chen",
        signoff_by="Noel Independent",
        at=NOW,
    )
    assert load_register(session, as_of=NOW).get(spec.usecase_id).status == "live"


def test_tier_1_may_be_activated_by_its_builder(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec(), actor_id="Ann Builder")
    assert load_register(session, as_of=NOW).get("research-summary").status == "live"


def test_a_use_case_below_its_tier_minimum_fails_validation(session, evidence):
    _open(session, evidence)
    spec = _spec(review_sampling_rate=0.0, monitoring=False)
    register(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    from kognita.usecases import validate_usecase

    findings = validate_usecase(session, spec.usecase_id, as_of=NOW)
    assert any("minimum controls" in item for item in findings)
    approve_rating(session, spec.usecase_id, "Alex Chen", evidence=evidence, at=NOW)
    with pytest.raises(KognitaError, match="minimum controls"):
        activate(
            session,
            spec.usecase_id,
            status="live",
            evidence=evidence,
            actor_id="Alex Chen",
            at=NOW,
        )


def test_a_decision_denies_controls_below_the_tier_minimum():
    """The tier floor is a decision rule, not only a validation message."""
    from kognita.usecases import build_record, from_record

    spec = _spec(review_sampling_rate=0.0)
    entry = from_record(
        build_record(
            spec,
            version=1,
            status="live",
            rating_approved=True,
            review_date=NOW + timedelta(days=30),
        )
    )
    evaluation = decide(
        _envelope(spec.usecase_id),
        PolicySnapshot(),
        usecases=UseCaseRegister(entries=(entry,), appetite=_appetite()),
        as_of=NOW,
    )
    assert evaluation.outcome is Outcome.DENY
    citation = _named(evaluation, "USE_CASE_MINIMUM_CONTROLS")[0].citation
    assert "minimum controls" in citation
    assert "accountable owner: Jane Smith" in citation


def test_a_use_case_past_its_review_date_is_denied(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec())
    entry = load_register(session, as_of=NOW).get("research-summary")
    assert entry is not None
    assert entry.review_date is not None
    on_the_day = load_register(session, as_of=entry.review_date)
    denied = decide(
        _envelope("research-summary"),
        PolicySnapshot(),
        usecases=on_the_day,
        as_of=entry.review_date,
    )
    assert denied.outcome is Outcome.DENY
    assert "past its review date" in _named(denied, "USE_CASE_REVIEW")[0].citation
    before = entry.review_date - timedelta(microseconds=1)
    allowed = decide(
        _envelope("research-summary"),
        PolicySnapshot(),
        usecases=load_register(session, as_of=before),
        as_of=before,
    )
    assert allowed.outcome is Outcome.ALLOW


def test_a_missed_review_needs_the_control_function_to_approve_again(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec())
    entry = load_register(session, as_of=NOW).get("research-summary")
    assert entry is not None
    assert entry.review_date is not None
    assert entry.rating_approved_at == NOW
    deadline = entry.review_date
    with pytest.raises(
        KognitaError, match="approve the materiality rating again"
    ) as denied:
        activate(
            session,
            "research-summary",
            status="live",
            evidence=evidence,
            actor_id="Alex Chen",
            at=deadline,
        )
    assert "accountable owner: Jane Smith" in str(denied.value)
    stale = deadline - timedelta(microseconds=1)
    approve_rating(
        session, "research-summary", "Alex Chen", evidence=evidence, at=stale
    )
    with pytest.raises(KognitaError, match="approve the materiality rating again"):
        activate(
            session,
            "research-summary",
            status="live",
            evidence=evidence,
            actor_id="Alex Chen",
            at=deadline,
        )
    approve_rating(
        session, "research-summary", "Alex Chen", evidence=evidence, at=deadline
    )
    activate(
        session,
        "research-summary",
        status="live",
        evidence=evidence,
        actor_id="Alex Chen",
        at=deadline,
    )
    renewed = load_register(session, as_of=deadline).get("research-summary")
    assert renewed is not None
    assert renewed.rating_approved_at == deadline
    assert renewed.review_date == deadline + timedelta(days=365)
    allowed = decide(
        _envelope("research-summary"),
        PolicySnapshot(),
        usecases=load_register(session, as_of=deadline),
        as_of=deadline,
    )
    assert allowed.outcome is Outcome.ALLOW


def test_a_pilot_denies_users_outside_its_list_and_calls_after_its_end(
    session, evidence
):
    _open(session, evidence)
    end = NOW + timedelta(days=10)
    spec = _spec(
        pilot_users=("alice", "carol"),
        pilot_end=end,
        success_criteria="RM acceptance above the baseline",
    )
    _activate(session, evidence, spec, status="pilot")
    snapshot = load_register(session, as_of=NOW)
    outsider = decide(
        _envelope("research-summary", principal="bob"),
        PolicySnapshot(),
        usecases=snapshot,
        as_of=NOW,
    )
    assert outsider.outcome is Outcome.DENY
    assert (
        "not on the user list" in _named(outsider, "USE_CASE_PILOT_USERS")[0].citation
    )
    insider = decide(
        _envelope("research-summary", principal="alice"),
        PolicySnapshot(),
        usecases=snapshot,
        as_of=NOW,
    )
    assert insider.outcome is Outcome.ALLOW
    after = end + timedelta(seconds=1)
    late = decide(
        _envelope("research-summary", principal="alice"),
        PolicySnapshot(),
        usecases=load_register(session, as_of=after),
        as_of=after,
    )
    assert late.outcome is Outcome.DENY
    assert "past its end date" in _named(late, "USE_CASE_PILOT_END")[0].citation


def test_a_pilot_without_a_user_list_or_end_date_cannot_be_activated(session, evidence):
    _open(session, evidence)
    spec = _spec(success_criteria="a recorded criterion")
    register(session, spec, evidence=evidence, actor_id="Jane Smith", at=NOW)
    approve_rating(session, spec.usecase_id, "Alex Chen", evidence=evidence, at=NOW)
    with pytest.raises(KognitaError, match="no named user list"):
        activate(
            session,
            spec.usecase_id,
            status="pilot",
            evidence=evidence,
            actor_id="Alex Chen",
            at=NOW,
        )


def test_registry_changes_are_policy_change_events(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec())
    suspend(
        session, "research-summary", evidence=evidence, actor_id="Alex Chen", at=NOW
    )
    retire(session, "research-summary", evidence=evidence, actor_id="Alex Chen", at=NOW)
    assert verify_chain(session) >= 1
    events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    assert events
    assert all(event.event_type == EventType.POLICY_CHANGE for event in events)
    actions = [event.payload["action"] for event in events]
    assert actions == [
        "SET_APPETITE",
        "REGISTER",
        "APPROVE_RATING",
        "ACTIVATE",
        "SUSPEND",
        "RETIRE",
    ]
    denied = decide(
        _envelope("research-summary"),
        PolicySnapshot(),
        usecases=load_register(session, as_of=NOW),
        as_of=NOW,
    )
    assert denied.outcome is Outcome.DENY
    assert "retired" in _named(denied, "USE_CASE_STATUS")[0].citation


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jurisdiction", "HK"),
        ("purpose", "Summarise research for a different desk"),
        ("category", "meeting preparation"),
    ],
)
def test_a_material_change_clears_activation_until_reapproved(
    session, evidence, field, value
):
    _open(session, evidence)
    spec = _spec(jurisdiction="SG")
    _activate(session, evidence, spec)
    revise(
        session,
        replace(spec, **{field: value}),
        evidence=evidence,
        actor_id="Jane Smith",
        at=NOW,
    )
    entry = load_register(session, as_of=NOW).get(spec.usecase_id)
    assert entry is not None
    assert entry.status == ""
    assert entry.rating_approved is False
    assert entry.rating_approved_at is None
    evaluation = decide(
        _envelope(spec.usecase_id),
        PolicySnapshot(),
        usecases=load_register(session, as_of=NOW),
        as_of=NOW,
    )
    assert evaluation.outcome is Outcome.DENY
    assert "not activated" in _named(evaluation, "USE_CASE_STATUS")[0].citation


def test_appetite_change_requires_senior_management(session, evidence):
    with pytest.raises(KognitaError, match="senior-management"):
        set_appetite(
            session,
            _appetite(),
            evidence=evidence,
            actor_id="Jane Smith",
            senior_management=False,
            at=NOW,
        )
    from kognita.models import RiskAppetite

    assert session.exec(select(RiskAppetite)).all() == []


def test_in_place_edit_of_an_effective_use_case_is_refused(session, evidence):
    _open(session, evidence)
    register(session, _spec(), evidence=evidence, actor_id="Jane Smith", at=NOW)
    row = session.exec(select(UseCase)).one()
    row.record = {**row.record, "purpose": "rewritten after the fact"}
    with pytest.raises(KognitaError, match="in-place edit"):
        session.flush()


def test_cli_output(tmp_path, capsys):
    path = tmp_path / "kognita.db"
    engine = make_engine(path)
    create_all(engine)
    from sqlmodel import Session

    from kognita.evidence import EvidenceWriter

    with Session(engine) as file_session:
        file_evidence = EvidenceWriter(engine)
        _open(file_session, file_evidence)
        _activate(file_session, file_evidence, _spec())
        register(
            file_session,
            _tier2(usecase_id="briefing", knowledge_lead=""),
            evidence=file_evidence,
            actor_id="Jane Smith",
            at=NOW,
        )
        file_session.commit()

    assert main(["usecase", "list", "--db", str(path)]) == 0
    listed = capsys.readouterr().out
    assert "research-summary" in listed
    assert "accountable owner: Jane Smith" in listed
    assert "briefing" in listed
    assert "not activated" in listed

    assert main(["usecase", "show", "research-summary", "--db", str(path)]) == 0
    shown = capsys.readouterr().out
    assert "Summarise approved research" in shown

    assert main(["usecase", "validate", "briefing", "--db", str(path)]) == 1
    findings = capsys.readouterr().out
    assert "no Knowledge Lead" in findings

    assert main(["usecase", "validate", "research-summary", "--db", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "valid"
    assert main(["usecase", "show", "missing", "--db", str(path)]) == 2


def test_moving_a_pilot_to_live_needs_the_control_function(session, evidence):
    _open(session, evidence)
    spec = _spec(
        pilot_users=("alice",),
        pilot_end=NOW + timedelta(days=10),
        success_criteria="RM acceptance above the baseline",
    )
    _activate(session, evidence, spec, status="pilot")
    later = NOW + timedelta(days=1)
    with pytest.raises(KognitaError, match="control function's approval"):
        activate(
            session,
            spec.usecase_id,
            status="live",
            evidence=evidence,
            actor_id="Jane Smith",
            criteria_met=True,
            at=later,
        )
    activate(
        session,
        spec.usecase_id,
        status="live",
        evidence=evidence,
        actor_id="Alex Chen",
        criteria_met=True,
        at=later,
    )
    assert load_register(session, as_of=later).get(spec.usecase_id).status == "live"
