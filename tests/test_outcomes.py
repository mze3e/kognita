"""Outcome events, near misses, and outcome metrics from the interaction record.

Roadmap 0.4 items 7 and 15. No network and no API key.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from sqlmodel import select

from kognita.cli import main
from kognita.evidence import verify_chain
from kognita.interaction import attach, open_interaction
from kognita.models import Continuation, EvidenceEvent, RunRecord
from kognita.outcomes import (
    METRIC_DEFINITION_VERSION,
    NOT_RECORDED,
    OutcomeError,
    latest_metric,
    metric_trend,
    near_miss_register,
    outcome_metrics,
    publish_outcome_metrics,
    record_exception,
    record_human_recovery,
    record_near_miss,
    record_outcome,
)
from kognita.review import record_final_decision
from kognita.usecases import quantitative_measures, record_primary_metric
from kognita.vocabulary import ActorType, EventType
from test_usecases import NOW, _activate, _appetite, _open, _spec

LATER = NOW + timedelta(minutes=5)


def _live(session, evidence, **overrides):
    _open(session, evidence, _appetite(performance_breach_threshold=0))
    values = {
        "residual_materiality": "high",
        "model_providers": ("house",),
        "primary_metric": "cycle time",
        "baseline": 40.0,
        "target": 25.0,
    }
    values.update(overrides)
    spec = _spec(**values)
    _activate(session, evidence, spec)
    return spec


def _journey(session, evidence, *, client="client-1", use_case="research-summary", opened=NOW):
    return open_interaction(
        session,
        client_id=client,
        use_case=use_case,
        evidence=evidence,
        now=opened,
    )


def _model_call(session, evidence, interaction, correlation, *, response_hash, version="1"):
    attach(
        session,
        interaction.id,
        evidence=evidence,
        correlation_id=correlation,
        now=NOW,
    )
    evidence.emit(
        session,
        correlation_id=correlation,
        event_type=EventType.MODEL_CALL,
        actor_type=ActorType.AGENT,
        actor_id="dossier-agent",
        payload={
            "sent": True,
            "response_hash": response_hash,
            "model": "summary",
            "model_version": version,
            "provider": "house",
            "prompt_template_version": "prompt-1",
            "response_classifier": {
                "model": "pattern",
                "version": "1",
                "label": "C1",
                "confidence": 0.99,
                "input_hash": "abc",
            },
        },
    )


def _run(session, evidence, interaction, run_id, correlation, outcome, checks=None):
    session.add(RunRecord(id=run_id, approvals_pending=[]))
    session.flush()
    attach(
        session,
        interaction.id,
        evidence=evidence,
        correlation_id=correlation,
        run_id=run_id,
        now=NOW,
    )
    evidence.emit(
        session,
        correlation_id=correlation,
        event_type=EventType.POLICY_DECISION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload={
            "outcome": outcome,
            "checks": checks or [],
            "envelope": {
                "purpose": interaction.use_case,
                "agent_name": "dossier-agent",
            },
        },
    )


def _outcomes(session):
    return session.exec(
        select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.OUTCOME)
    ).all()


def test_a_complaint_links_the_output_model_version_and_approver(session, evidence):
    _live(session, evidence)
    interaction = _journey(session, evidence)
    _model_call(session, evidence, interaction, "req-1", response_hash="output-1")
    record_final_decision(
        session,
        interaction.id,
        outcome="allow",
        evidence=evidence,
        actor_id="RM One",
        approver="RM One",
        citation="No policy check failed, and no human approval was required.",
        now=LATER,
    )
    recorded = record_outcome(
        session,
        interaction.id,
        kind="client complaint",
        summary="The client says the note named the wrong bond",
        evidence=evidence,
        actor_id="Priya Shah",
        citation="Complaint 441",
        now=LATER,
    )
    assert recorded["output_hash"] == "output-1"
    assert recorded["model_version"] == "1"
    assert recorded["approver"] == "RM One"
    assert recorded["rule"]
    stored = _outcomes(session)
    assert len(stored) == 1
    assert stored[0].payload["output_hash"] == "output-1"
    assert stored[0].payload["approver"] == "RM One"
    assert verify_chain(session) >= 1


def test_a_complaint_without_one_output_is_refused_and_not_appended(session, evidence):
    _live(session, evidence)
    interaction = _journey(session, evidence)
    record_final_decision(
        session,
        interaction.id,
        outcome="allow",
        evidence=evidence,
        actor_id="RM One",
        approver="RM One",
        citation="No policy check failed, and no human approval was required.",
        now=LATER,
    )
    before = len(_outcomes(session))
    with pytest.raises(OutcomeError, match="does not link"):
        record_outcome(
            session,
            interaction.id,
            kind="client complaint",
            summary="The client complained",
            evidence=evidence,
            actor_id="Priya Shah",
            citation="Complaint 441",
            now=LATER,
        )
    _model_call(session, evidence, interaction, "req-a", response_hash="output-a")
    _model_call(session, evidence, interaction, "req-b", response_hash="output-b")
    with pytest.raises(OutcomeError, match="does not link"):
        record_outcome(
            session,
            interaction.id,
            kind="client complaint",
            summary="The client complained",
            evidence=evidence,
            actor_id="Priya Shah",
            citation="Complaint 441",
            now=LATER,
        )
    with pytest.raises(OutcomeError, match="does not link"):
        record_outcome(
            session,
            interaction.id,
            kind="correction",
            summary="The bond name was corrected",
            evidence=evidence,
            actor_id="Priya Shah",
            citation="Correction",
            output_hash="missing",
            now=LATER,
        )
    assert len(_outcomes(session)) == before


def test_near_misses_aggregate_decisions_and_recorded_drafts(session, evidence):
    _live(session, evidence)
    interaction = _journey(session, evidence)
    _model_call(session, evidence, interaction, "req-1", response_hash="output-1")
    evidence.emit(
        session,
        correlation_id="req-1",
        event_type=EventType.POLICY_DECISION,
        payload={
            "outcome": "DENY",
            "checks": [],
            "envelope": {
                "purpose": "research-summary",
                "agent_name": "dossier-agent",
            },
        },
    )
    record_final_decision(
        session,
        interaction.id,
        outcome="reject",
        evidence=evidence,
        actor_id="RM One",
        approver="RM One",
        citation="RM review: final decision",
        now=LATER,
    )
    with pytest.raises(OutcomeError, match="taken from decisions"):
        record_near_miss(
            session,
            kind="rm rejection",
            summary="again",
            evidence=evidence,
            actor_id="Priya Shah",
            citation="Near misses: blocked actions, verifier blocks, RM rejections, unsafe drafts and unusual agent behaviour",
            interaction_id=interaction.id,
            now=LATER,
        )
    record_near_miss(
        session,
        kind="unsafe draft",
        summary="The draft named a product the client cannot hold",
        evidence=evidence,
        actor_id="Priya Shah",
        citation="Near misses: blocked actions, verifier blocks, RM rejections, unsafe drafts and unusual agent behaviour",
        interaction_id=interaction.id,
        agent="dossier-agent",
        model_version="1",
        now=LATER,
    )
    rows = {(item.kind, item.agent, item.model_version): item for item in near_miss_register(session, as_of=LATER)}
    assert rows[("blocked action", "dossier-agent", "1")].count == 1
    assert rows[("rm rejection", "dossier-agent", "1")].count == 1
    assert rows[("unsafe draft", "dossier-agent", "1")].model_version_recorded is True
    loose = evidence.emit(
        session,
        correlation_id="loose",
        event_type=EventType.POLICY_DECISION,
        payload={
            "outcome": "DENY",
            "envelope": {"purpose": "research-summary", "agent_name": ""},
            "checks": [],
        },
    )
    del loose
    again = near_miss_register(session, as_of=LATER)
    missing = [
        item
        for item in again
        if item.kind == "blocked action" and item.model_version == NOT_RECORDED
    ]
    assert missing
    assert missing[0].model_version_recorded is False


def test_cycle_time_decision_quality_and_reliability_come_from_evidence(session, evidence):
    _live(session, evidence)
    first = _journey(session, evidence, client="client-1")
    second = _journey(session, evidence, client="client-2")
    _run(session, evidence, first, "run-clean", "req-clean", "ALLOW")
    _run(
        session,
        evidence,
        second,
        "run-deny",
        "req-deny",
        "DENY",
        checks=[{"check": "BUDGET", "result": "fail", "citation": "max_calls", "regime": "INTERNAL"}],
    )
    record_final_decision(
        session,
        first.id,
        outcome="allow",
        evidence=evidence,
        actor_id="RM One",
        approver="RM One",
        citation="No policy check failed, and no human approval was required.",
        correlation_id="req-clean",
        now=NOW + timedelta(seconds=10),
    )
    record_final_decision(
        session,
        second.id,
        outcome="modify",
        evidence=evidence,
        actor_id="RM One",
        approver="RM One",
        citation="RM review: final decision",
        correlation_id="req-deny",
        now=NOW + timedelta(seconds=30),
        provenance="AI then human-edited",
    )
    report = outcome_metrics(session, as_of=LATER, usecase_id="research-summary")[0]
    assert report.definition_version == METRIC_DEFINITION_VERSION
    assert report.primary_metric == "cycle time"
    assert report.baseline == 40.0
    assert report.target == 25.0
    assert report.measured is True
    assert report.value == 20.0
    assert report.breached is False
    cycle = report.metrics["cycle time"]
    assert cycle.unit == "seconds"
    assert cycle.components["rm_effort_measured"] is False
    quality = report.metrics["decision quality"]
    assert quality.value == 0.5
    assert quality.components["edit_rate"] == 0.5
    assert quality.components["rejection_rate"] == 0.0
    reliability = report.metrics["reliability"]
    assert reliability.value == 0.5
    assert reliability.components["denials"] == 1
    assert reliability.components["budget_breaches"] == 1
    assert report.citation


def test_a_missing_primary_measurement_is_a_breach(session, evidence):
    _live(session, evidence, primary_metric="reliability", baseline=0.9, target=0.95)
    _journey(session, evidence)
    report = outcome_metrics(session, as_of=LATER, usecase_id="research-summary")[0]
    assert report.measured is False
    assert report.breached is True
    assert report.detail == "unmeasured"
    assert report.value is None


def test_definitions_are_not_averaged_across_versions(session, evidence):
    _live(session, evidence)
    interaction = _journey(session, evidence)
    record_final_decision(
        session,
        interaction.id,
        outcome="allow",
        evidence=evidence,
        actor_id="RM One",
        approver="RM One",
        citation="No policy check failed, and no human approval was required.",
        now=NOW + timedelta(seconds=8),
    )
    publish_outcome_metrics(session, evidence=evidence, actor_id="Jane Smith", at=LATER)
    evidence.emit(
        session,
        correlation_id="usecase:research-summary",
        event_type=EventType.OUTCOME,
        actor_type=ActorType.HUMAN,
        actor_id="Jane Smith",
        payload={
            "action": "METRIC",
            "usecase_id": "research-summary",
            "primary_metric": "cycle time",
            "definition_version": METRIC_DEFINITION_VERSION + 1,
            "value": 1.0,
            "measured": True,
            "breached": False,
        },
    )
    trend = metric_trend(session, "research-summary", as_of=LATER)
    assert [item["definition_version"] for item in trend["versions"]] == [
        METRIC_DEFINITION_VERSION,
        METRIC_DEFINITION_VERSION + 1,
    ]
    assert "value" not in trend
    with pytest.raises(OutcomeError, match="not averaged"):
        latest_metric(session, "research-summary", as_of=LATER)


def test_reliability_excludes_an_exception_a_recovery_and_a_suspended_run(session, evidence):
    _live(session, evidence, primary_metric="reliability", baseline=0.5, target=0.9)
    interaction = _journey(session, evidence)
    _run(session, evidence, interaction, "run-ok", "req-ok", "ALLOW")
    _run(session, evidence, interaction, "run-ex", "req-ex", "ALLOW")
    record_exception(
        session,
        interaction.id,
        run_id="run-ex",
        summary="The tool raised before the note was stored",
        evidence=evidence,
        actor_id="Priya Shah",
        citation="Outcome metrics: reliability excludes an exception",
        now=LATER,
    )
    _run(session, evidence, interaction, "run-re", "req-re", "ALLOW")
    record_human_recovery(
        session,
        interaction.id,
        run_id="run-re",
        summary="An operator resumed the run by hand",
        evidence=evidence,
        actor_id="Priya Shah",
        citation="Outcome metrics: reliability excludes human recovery",
        now=LATER,
    )
    session.add(Continuation(content_hash="held-hash", payload={"state": "suspended"}))
    session.add(
        RunRecord(id="run-held", approvals_pending=[], continuation_hash="held-hash")
    )
    session.flush()
    attach(
        session,
        interaction.id,
        evidence=evidence,
        correlation_id="req-held",
        run_id="run-held",
        now=NOW,
    )
    evidence.emit(
        session,
        correlation_id="req-held",
        event_type=EventType.POLICY_DECISION,
        payload={"outcome": "ALLOW", "checks": [], "envelope": {"agent_name": "dossier-agent"}},
    )
    session.add(RunRecord(id="run-missing", approvals_pending=[]))
    session.flush()
    with pytest.raises(OutcomeError, match="not attached"):
        record_exception(
            session,
            interaction.id,
            run_id="run-missing",
            summary="no",
            evidence=evidence,
            actor_id="Priya Shah",
            citation="Outcome metrics: reliability excludes an exception",
            now=LATER,
        )
    report = outcome_metrics(session, as_of=LATER, usecase_id="research-summary")[0]
    reliability = report.metrics["reliability"]
    assert reliability.sample == 4
    assert reliability.components["clean"] == 1
    assert reliability.components["exceptions"] == 1
    assert reliability.components["human_recoveries"] == 1
    assert reliability.value == 0.25
    assert report.breached is True


def test_an_interaction_replaces_a_self_reported_primary_metric(session, evidence):
    """A rosy recorded value does not pass once an interaction exists and is unmeasured."""
    _live(session, evidence, primary_metric="reliability", baseline=0.9, target=0.95)
    record_primary_metric(
        session,
        "research-summary",
        0.99,
        evidence=evidence,
        actor_id="Jane Smith",
        at=NOW,
    )
    _journey(session, evidence)
    performance = {
        item.measure: item for item in quantitative_measures(session, as_of=LATER)
    }["performance breaches"]
    assert performance.usecase_ids == ("research-summary",)
    assert performance.detail == "unmeasured"
    assert performance.breached is True


def test_a_computed_pass_ignores_a_failing_self_report(session, evidence):
    _live(session, evidence, primary_metric="reliability", baseline=0.5, target=0.5)
    interaction = _journey(session, evidence)
    _run(session, evidence, interaction, "run-ok", "req-ok", "ALLOW")
    record_primary_metric(
        session,
        "research-summary",
        0.1,
        evidence=evidence,
        actor_id="Jane Smith",
        at=NOW,
    )
    performance = {
        item.measure: item for item in quantitative_measures(session, as_of=LATER)
    }["performance breaches"]
    assert performance.value == 0
    assert performance.usecase_ids == ()


def test_a_pilot_with_no_measurement_stays_out_of_the_appetite_count(session, evidence):
    _open(session, evidence, _appetite(performance_breach_threshold=0))
    spec = _spec(
        residual_materiality="high",
        model_providers=("house",),
        primary_metric="reliability",
        baseline=0.9,
        target=0.95,
        pilot_users=("alice",),
        pilot_end=NOW + timedelta(days=10),
        success_criteria="RM acceptance above the baseline",
    )
    _activate(session, evidence, spec, status="pilot")
    _journey(session, evidence)
    performance = {
        item.measure: item for item in quantitative_measures(session, as_of=LATER)
    }["performance breaches"]
    assert performance.value == 0
    report = outcome_metrics(session, as_of=LATER, usecase_id="research-summary")[0]
    assert report.breached is True
    assert report.detail == "unmeasured"


def test_usecase_metrics_prints_the_report(tmp_path, capsys):
    database = tmp_path / "metrics.db"
    assert main(["usecase", "metrics", "--db", str(database)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "no outcome metrics"
    assert main(["usecase", "coverage", "--db", str(database)]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["by_use_case"] == []
    assert body["citation"]
