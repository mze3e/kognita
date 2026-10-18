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
from kognita.egress import EgressDenied, EgressGuard
from kognita.envelope import Envelope
from kognita.evidence import verify_chain
from kognita.exceptions import KognitaError
from kognita.governance import PolicySnapshot, decide, load_snapshot
from kognita.models import EvidenceEvent, ModelCard, RetentionPolicy, UseCase
from kognita.registry import get as get_agent
from kognita.registry import register as register_agent
from kognita.retention import RetentionStore
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
    model_card_id,
    quantitative_measures,
    record_incident,
    record_primary_metric,
    register,
    register_model_card,
    retire,
    revise,
    senior_management_report,
    set_appetite,
    suspend,
    validate_usecase,
)
from kognita.vocabulary import Classification
from kognita.vocabulary import CheckResult, EventType, Outcome

NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
CARD_ID = model_card_id("house", "summary", "1")

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
        approved_models=(CARD_ID,),
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
    register_model_card(
        session,
        provider="house",
        model="summary",
        version="1",
        intended_use="Summarise approved research for an RM",
        limitations="Does not read tables in exhibits",
        evaluation_results="Reviewed against the research fixture, May 2026",
        approver="Luis Ortega",
        approved_at=NOW,
        evidence=evidence,
        actor_id="Luis Ortega",
        at=NOW,
    )
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
        "MODEL_CARD",
        "SET_APPETITE",
        "REGISTER",
        "APPROVE_RATING",
        "ACTIVATE",
        "SUSPEND",
        "RETIRE",
        "RETENTION_APPLIED",
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


def _actions(session) -> list[str]:
    events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    return [str((event.payload or {}).get("action") or event.event_type.value) for event in events]


def _model_envelope(purpose: str, model: str, version: str = "") -> Envelope:
    arguments = {"model_version": version} if version else {}
    return Envelope(
        principal="alice",
        purpose=purpose,
        tool="model_call",
        subject_type="model",
        subject_id=model,
        arguments=arguments,
    )


def test_an_approved_model_without_a_card_cannot_be_registered(session, evidence):
    _open(session, evidence)
    with pytest.raises(KognitaError, match="has no model card"):
        register(
            session,
            _spec(approved_models=("house/other@1",)),
            evidence=evidence,
            actor_id="Jane Smith",
            at=NOW,
        )
    assert session.exec(select(UseCase)).all() == []


def test_a_model_call_cites_the_one_approved_card(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec())
    register = load_register(session, as_of=NOW)
    pinned = decide(
        _model_envelope("research-summary", "summary", "1"),
        PolicySnapshot(),
        usecases=register,
        as_of=NOW,
    )
    assert pinned.outcome is Outcome.ALLOW
    assert f"model card: {CARD_ID}" in _named(pinned, "USE_CASE")[0].citation
    missing = decide(
        _model_envelope("research-summary", "other", "1"),
        PolicySnapshot(),
        usecases=register,
        as_of=NOW,
    )
    assert missing.outcome is Outcome.DENY
    assert "no model card" in _named(missing, "USE_CASE_MODEL")[0].citation


def test_two_cards_for_one_model_name_must_be_pinned(session, evidence):
    _open(session, evidence)
    second = model_card_id("house", "summary", "2")
    register_model_card(
        session,
        provider="house",
        model="summary",
        version="2",
        intended_use="Summarise approved research for an RM",
        limitations="Does not read tables in exhibits",
        evaluation_results="Reviewed against the research fixture, May 2026",
        approver="Luis Ortega",
        approved_at=NOW,
        evidence=evidence,
        actor_id="Luis Ortega",
        at=NOW,
    )
    _activate(session, evidence, _spec(approved_models=(CARD_ID, second)))
    register = load_register(session, as_of=NOW)
    loose = decide(
        _model_envelope("research-summary", "summary"),
        PolicySnapshot(),
        usecases=register,
        as_of=NOW,
    )
    assert loose.outcome is Outcome.DENY
    assert "not pinned" in _named(loose, "USE_CASE_MODEL")[0].citation
    pinned = decide(
        _model_envelope("research-summary", "summary", "2"),
        PolicySnapshot(),
        usecases=register,
        as_of=NOW,
    )
    assert pinned.outcome is Outcome.ALLOW
    assert f"model card: {second}" in _named(pinned, "USE_CASE")[0].citation


def test_in_place_edit_of_an_effective_model_card_is_refused(session, evidence):
    _open(session, evidence)
    row = session.exec(select(ModelCard)).one()
    row.record = {**row.record, "limitations": "rewritten after the fact"}
    with pytest.raises(KognitaError, match="in-place edit"):
        session.flush()


def test_a_new_model_card_approval_is_a_new_version(session, evidence):
    _open(session, evidence)
    later = NOW + timedelta(days=1)
    register_model_card(
        session,
        provider="house",
        model="summary",
        version="1",
        intended_use="Summarise approved research for an RM",
        limitations="Also misses footnotes",
        evaluation_results="Reviewed against the research fixture, May 2026",
        approver="Luis Ortega",
        approved_at=later,
        evidence=evidence,
        actor_id="Luis Ortega",
        at=later,
    )
    current = load_register(session, as_of=later).model_cards
    assert len(current) == 1
    assert current[0].card_id == CARD_ID
    assert current[0].registry_version == 2
    assert current[0].limitations == "Also misses footnotes"
    previous = load_register(session, as_of=NOW).model_cards
    assert previous[0].registry_version == 1
    assert "footnotes" not in previous[0].limitations


def test_egress_links_the_approved_card_and_refuses_a_mismatch(session, evidence):
    _open(session, evidence)
    _activate(session, evidence, _spec())
    guard = EgressGuard(evidence=evidence)
    result = guard.send(
        "hello",
        lambda text: text,
        classification=Classification.C1,
        destination="https://api.openai.com",
        destination_is_local=False,
        session=session,
        correlation_id="card-link",
        provider="house",
        model="summary",
        model_version="1",
        use_case="research-summary",
    )
    assert result.decision.value == "ALLOW"
    event = next(
        item
        for item in session.exec(
            select(EvidenceEvent).where(EvidenceEvent.correlation_id == "card-link")
        ).all()
        if item.event_type is EventType.MODEL_CALL
    )
    assert event.payload["model_card"] == CARD_ID
    assert event.payload["model_card_hash"]
    called = {"sent": False}

    def _call(_text: str) -> str:
        called["sent"] = True
        return "no"

    with pytest.raises(EgressDenied, match="no model card"):
        guard.send(
            "hello",
            _call,
            classification=Classification.C1,
            destination="https://api.openai.com",
            destination_is_local=False,
            session=session,
            correlation_id="card-block",
            provider="house",
            model="summary",
            model_version="9",
            use_case="research-summary",
        )
    assert called["sent"] is False


def test_unmediated_entries_record_the_gap_and_do_not_authorise_a_call(
    session, evidence
):
    _open(session, evidence)
    with pytest.raises(KognitaError, match="assurance gap"):
        register(
            session,
            _spec(
                usecase_id="vendor-copilot",
                unmediated=True,
                compensating_control="Quarterly sample by the control function",
            ),
            evidence=evidence,
            actor_id="Jane Smith",
            at=NOW,
        )
    with pytest.raises(KognitaError, match="compensating control"):
        register(
            session,
            _spec(
                usecase_id="vendor-copilot",
                unmediated=True,
                assurance_gap="The vendor keeps the prompt log",
            ),
            evidence=evidence,
            actor_id="Jane Smith",
            at=NOW,
        )
    spec = _spec(
        usecase_id="vendor-copilot",
        unmediated=True,
        assurance_gap="The vendor keeps the prompt log",
        compensating_control="Quarterly sample by the control function",
        ai_endpoints=("https://copilot.example",),
    )
    _activate(session, evidence, spec)
    assert validate_usecase(session, spec.usecase_id, as_of=NOW) == ()
    denied = decide(
        _envelope(spec.usecase_id),
        PolicySnapshot(),
        usecases=load_register(session, as_of=NOW),
        as_of=NOW,
    )
    assert denied.outcome is Outcome.DENY
    citation = _named(denied, "USE_CASE_UNMEDIATED")[0].citation
    assert "does not authorise a call" in citation
    assert "accountable owner: Jane Smith" in citation


def test_a_material_use_case_records_its_model_providers(session, evidence):
    _open(session, evidence)
    with pytest.raises(KognitaError, match="no model provider"):
        register(
            session,
            _spec(residual_materiality="high"),
            evidence=evidence,
            actor_id="Jane Smith",
            at=NOW,
        )
    assert session.exec(select(UseCase)).all() == []


def test_a_negative_appetite_threshold_is_refused(session, evidence):
    _open(session, evidence)
    with pytest.raises(KognitaError, match="non-negative"):
        set_appetite(
            session,
            _appetite(incident_count_threshold=-1),
            evidence=evidence,
            actor_id="CRO",
            senior_management=True,
            at=NOW,
        )


def test_quantitative_measures_escalate_breaches_to_the_owner(session, evidence):
    _open(
        session,
        evidence,
        _appetite(
            incident_count_threshold=0,
            incident_impact_threshold=1.0,
            single_provider_threshold=1,
            performance_breach_threshold=0,
        ),
    )
    summary = _spec(
        residual_materiality="high",
        model_providers=("house",),
        jurisdiction="HK",
        primary_metric="reliability",
        baseline=0.9,
        target=0.95,
    )
    briefing = _spec(
        usecase_id="client-brief",
        purpose="Prepare a short internal briefing",
        residual_materiality="high",
        model_providers=("house",),
        jurisdiction="SG",
        primary_metric="cycle time",
        baseline=12.0,
        target=8.0,
    )
    notes = _spec(
        usecase_id="notes-draft",
        purpose="Draft internal meeting notes",
        residual_materiality="high",
        model_providers=("other",),
        jurisdiction="HK",
        primary_metric="decision quality",
        baseline=0.8,
        target=0.9,
    )
    _activate(session, evidence, summary)
    _activate(session, evidence, briefing)
    _activate(session, evidence, notes)
    record_incident(
        session,
        "research-summary",
        impact=2.0,
        summary="The summary cited a superseded note",
        evidence=evidence,
        actor_id="Priya Shah",
        at=NOW,
    )
    record_primary_metric(
        session,
        "research-summary",
        0.5,
        evidence=evidence,
        actor_id="Jane Smith",
        at=NOW,
    )
    record_primary_metric(
        session,
        "client-brief",
        8.0,
        evidence=evidence,
        actor_id="Jane Smith",
        at=NOW,
    )
    measures = {
        item.measure: item
        for item in quantitative_measures(session, as_of=NOW)
    }
    assert measures["AI incidents"].value == 1
    assert measures["AI incidents"].breached is True
    assert measures["AI incident impact"].value == 2.0
    assert measures["AI incident impact"].breached is True
    assert measures["single-provider dependencies"].value == 2
    assert measures["single-provider dependencies"].detail == "house"
    assert measures["single-provider dependencies"].breached is True
    assert measures["performance breaches"].value == 1
    assert measures["performance breaches"].usecase_ids == ("research-summary",)
    assert measures["performance breaches"].breached is True
    assert measures["performance breaches"].accountable_owners == ("Jane Smith",)

    hong_kong = {
        item.measure: item
        for item in quantitative_measures(session, as_of=NOW, jurisdiction="HK")
    }
    assert hong_kong["AI incidents"].value == 1
    assert hong_kong["single-provider dependencies"].value == 1
    assert hong_kong["single-provider dependencies"].breached is False
    assert hong_kong["performance breaches"].value == 1
    singapore = quantitative_measures(session, as_of=NOW, jurisdiction="SG")
    assert {item.measure: item.value for item in singapore}["AI incidents"] == 0

    preview = senior_management_report(session, as_of=NOW, jurisdiction="HK")
    assert {row["usecase_id"] for row in preview["material_use_cases"]} == {
        "research-summary",
        "notes-draft",
    }
    assert preview["material_use_cases"][0]["open_findings"] == []
    with pytest.raises(KognitaError, match="who published it"):
        senior_management_report(session, as_of=NOW, evidence=evidence)
    senior_management_report(
        session,
        as_of=NOW,
        evidence=evidence,
        actor_id="CRO",
    )
    breaches = [
        event
        for event in session.exec(select(EvidenceEvent)).all()
        if (event.payload or {}).get("action") == "APPETITE_BREACH"
    ]
    assert len(breaches) == 4
    assert {event.payload["accountable_owner"] for event in breaches} == {"Jane Smith"}
    assert any(
        (event.payload or {}).get("action") == "SENIOR_MANAGEMENT_REPORT"
        for event in session.exec(select(EvidenceEvent)).all()
    )


def test_an_unset_threshold_and_an_equal_value_are_inside_appetite(session, evidence):
    _open(session, evidence, _appetite(incident_count_threshold=1))
    _activate(session, evidence, _spec())
    record_incident(
        session,
        "research-summary",
        impact=1,
        summary="A draft was withdrawn before it was sent",
        evidence=evidence,
        actor_id="Priya Shah",
        at=NOW,
    )
    measures = {
        item.measure: item for item in quantitative_measures(session, as_of=NOW)
    }
    assert measures["AI incidents"].value == 1
    assert measures["AI incidents"].breached is False
    assert measures["AI incident impact"].threshold is None
    assert measures["AI incident impact"].breached is False
    assert measures["single-provider dependencies"].breached is False


def test_retirement_stops_exclusive_agents_applies_retention_and_notifies(
    session, evidence
):
    register_agent(session, name="only-agent", owner_exec="Jane Smith")
    register_agent(session, name="shared-agent", owner_exec="Omar Aziz")
    _open(session, evidence)
    _activate(
        session,
        evidence,
        _spec(agents=("only-agent", "shared-agent"), retention_days=365),
    )
    _activate(
        session,
        evidence,
        _spec(
            usecase_id="briefing",
            purpose="Prepare a briefing from the summary",
            agents=("shared-agent",),
            depends_on=("research-summary",),
            retention_days=None,
        ),
    )
    store = RetentionStore()
    aged = store.retain_text(
        session,
        "aged prompt",
        kind="prompt",
        use_case="research-summary",
        now=NOW - timedelta(days=400),
    )
    kept = store.retain_text(
        session,
        "recent prompt",
        kind="prompt",
        use_case="research-summary",
        now=NOW - timedelta(days=1),
    )
    other = store.retain_text(
        session,
        "other prompt",
        kind="prompt",
        use_case="briefing",
        now=NOW - timedelta(days=400),
    )
    retire(
        session, "research-summary", evidence=evidence, actor_id="Alex Chen", at=NOW
    )
    assert get_agent(session, "only-agent").kill_switch is True
    assert get_agent(session, "shared-agent").kill_switch is False
    assert store.read(session, aged) is None
    assert store.read(session, kept) == "recent prompt"
    assert store.read(session, other) == "other prompt"
    policy = session.get(RetentionPolicy, "research-summary")
    assert policy is not None
    assert policy.retain_days == 365
    actions = _actions(session)
    assert "KILL_SWITCH_ENGAGED" in actions
    assert "AGENT_KEPT" in actions
    assert "RETENTION_APPLIED" in actions
    assert "DEPENDENT_NOTIFIED" in actions
    assert "ERASURE" in actions
    register = load_register(session, as_of=NOW)
    snapshot = load_snapshot(session, as_of=NOW)
    stopped = decide(
        Envelope(
            principal="alice",
            purpose="research-summary",
            tool="get_subject_profile",
            agent_name="only-agent",
        ),
        snapshot,
        usecases=register,
        as_of=NOW,
    )
    assert stopped.outcome is Outcome.DENY
    assert "Kill switch engaged" in _named(stopped, "KILL_SWITCH")[0].citation
    dependent = decide(
        Envelope(
            principal="alice",
            purpose="briefing",
            tool="get_subject_profile",
            agent_name="shared-agent",
        ),
        snapshot,
        usecases=register,
        as_of=NOW,
    )
    assert dependent.outcome is Outcome.DENY
    assert not any(check.check == "KILL_SWITCH" for check in dependent.checks)
    dependency = _named(dependent, "USE_CASE_DEPENDENCY")[0].citation
    assert "depends on 'research-summary', which is retired" in dependency
    assert "accountable owner: Jane Smith" in dependency


def test_a_use_case_cannot_depend_on_itself_or_a_missing_agent(session, evidence):
    _open(session, evidence)
    with pytest.raises(KognitaError, match="depends on itself"):
        register(
            session,
            _spec(depends_on=("research-summary",)),
            evidence=evidence,
            actor_id="Jane Smith",
            at=NOW,
        )
    with pytest.raises(KognitaError, match="is not registered"):
        register(
            session,
            _spec(agents=("ghost-agent",)),
            evidence=evidence,
            actor_id="Jane Smith",
            at=NOW,
        )
