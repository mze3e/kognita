"""Evaluation evidence gate and fairness report. Missing or partial evidence denies."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from sqlmodel import Session, select

from kognita import (
    EvaluationResult,
    FairnessReport,
    check_activation,
    fairness_report,
    record_evaluation,
)
from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.evaluation import METRICS
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.interaction import open_interaction
from kognita.models import EvidenceEvent, UseCase
from kognita.origination import originate
from kognita.policy import PolicyLanguageError, read_policy_set
from kognita.registry import approve_deployment, register as register_agent
from kognita.retention import RetentionStore
from kognita.usecases import UseCaseSpec, build_record, from_record, load_register, vocabulary_citations
from kognita.vocabulary import ActorType, EventType
from test_usecases import NOW, _activate, _open, _spec, _tier2

_DEPLOYMENT = dict(
    system_prompt="", instructions="instructions", tools=[], permissions=[],
    data_sources=[], orchestration="one step", memory_behaviour="none",
)
_REASON = {"fact": "house view", "detail": "overweight equities", "citation": "cio-2026"}


def _thresholds(**overrides):
    values = {name: (0.05 if name == "hallucination_rate" else 0.9) for name in METRICS}
    values.update(overrides)
    return values


def _metrics(**overrides):
    values = {name: (0.01 if name == "hallucination_rate" else 0.95) for name in METRICS}
    values.update(overrides)
    return values


def _result(**overrides):
    values = dict(
        agent_id="briefing-agent", version="1.2.0", model_version="house-v3",
        level="system", metrics=_metrics(),
    )
    values.update(overrides)
    return EvaluationResult(**values)


def _arm(session, evidence, spec=None):
    _open(session, evidence)
    chosen = spec or _tier2(evaluation_thresholds=_thresholds())
    _activate(session, evidence, chosen)
    return chosen


def _agent(session, *, model="house-v3", version="1.2.0", use_cases=("client-intelligence",)):
    agent = register_agent(session, name="briefing-agent", owner_exec="owner", version=version)
    agent.use_cases = list(use_cases)
    agent.model = model
    agent.purpose = "Prepare a client briefing"
    return agent


def _approve(session, evidence, agent=None):
    agent = agent or _agent(session)
    return approve_deployment(
        session, agent.name, deployment=dict(_DEPLOYMENT), evidence=evidence, actor_id="owner",
    )


def _record(session, evidence, **overrides):
    return record_evaluation(session, _result(**overrides), evidence=evidence)


def test_system_and_agent_results_open_the_gate(session, evidence):
    _arm(session, evidence)
    agent = _agent(session)
    for level in ("system", "agent"):
        _record(session, evidence, level=level)
        assert check_activation(session, agent) is None
        approved = _approve(session, evidence, agent)
        assert approved.deployment_hash
    assert verify_chain(session) > 0


def test_tier1_activation_does_not_require_results(session, evidence):
    _arm(session, evidence, _spec())
    agent = _agent(session, use_cases=("research-summary",))
    assert check_activation(session, agent) is None
    assert _approve(session, evidence, agent).deployment_hash


def test_missing_results_and_thresholds_deny(session, evidence):
    _arm(session, evidence)
    agent = _agent(session)
    with pytest.raises(ValueError, match="EVALUATION_REQUIRED"):
        _approve(session, evidence, agent)
    assert load_register(session).get("client-intelligence").evaluation_thresholds["accuracy"] == 0.9


def test_tier2_without_thresholds_is_denied(session, evidence):
    _arm(session, evidence, _tier2(evaluation_thresholds={}))
    agent = _agent(session)
    with pytest.raises(ValueError, match="EVALUATION_THRESHOLDS_MISSING"):
        check_activation(session, agent)


@pytest.mark.parametrize("overrides", [
    {"level": "model"},
    {"model_version": "other-model"},
    {"version": "9.9.9"},
    {"agent_id": "other-agent"},
])
def test_non_applicable_results_do_not_open_the_gate(session, evidence, overrides):
    _arm(session, evidence)
    agent = _agent(session)
    _record(session, evidence, **overrides)
    with pytest.raises(ValueError, match="EVALUATION_REQUIRED"):
        check_activation(session, agent)


@pytest.mark.parametrize("metric,value,match", [
    ("hallucination_rate", 0.2, "hallucination_rate"),
    ("accuracy", 0.5, "accuracy"),
    ("hallucination_rate", 0.05, None),
    ("accuracy", 0.9, None),
])
def test_threshold_direction_and_equality(session, evidence, metric, value, match):
    _arm(session, evidence)
    agent = _agent(session)
    _record(session, evidence, metrics=_metrics(**{metric: value}))
    if match:
        with pytest.raises(ValueError, match=f"EVALUATION_BELOW_THRESHOLD: {match}"):
            check_activation(session, agent)
    else:
        assert check_activation(session, agent) is None


def test_latest_applicable_result_wins_and_results_are_not_stitched(session, evidence):
    _arm(session, evidence)
    agent = _agent(session)
    _record(session, evidence, level="system", metrics=_metrics())
    _record(session, evidence, level="model", metrics=_metrics(accuracy=0.0, hallucination_rate=1.0))
    assert check_activation(session, agent) is None
    _record(session, evidence, level="agent", metrics=_metrics(accuracy=0.1))
    with pytest.raises(ValueError, match="EVALUATION_BELOW_THRESHOLD: accuracy"):
        check_activation(session, agent)
    _record(session, evidence, level="system", metrics=_metrics(hallucination_rate=0.0))
    _record(session, evidence, level="system", metrics=_metrics(accuracy=0.1, hallucination_rate=0.0))
    with pytest.raises(ValueError, match="EVALUATION_BELOW_THRESHOLD"):
        check_activation(session, agent)


def test_stricter_threshold_across_linked_use_cases(session, evidence):
    _arm(session, evidence)
    _activate(session, evidence, _tier2(
        usecase_id="meeting-prep", purpose="Prepare the meeting",
        category="meeting preparation", evaluation_thresholds=_thresholds(accuracy=0.99),
    ))
    agent = _agent(session, use_cases=("client-intelligence", "meeting-prep"))
    _record(session, evidence, metrics=_metrics(accuracy=0.95))
    with pytest.raises(ValueError, match="accuracy"):
        check_activation(session, agent)
    _record(session, evidence, metrics=_metrics(accuracy=0.99))
    assert check_activation(session, agent) is None


def test_model_change_requires_a_new_result(session, evidence):
    _arm(session, evidence)
    agent = _agent(session)
    _record(session, evidence)
    approved = _approve(session, evidence, agent)
    first = approved.deployment_hash
    agent.model = "house-v4"
    with pytest.raises(ValueError, match="EVALUATION_REQUIRED"):
        approve_deployment(
            session, agent.name, deployment={**_DEPLOYMENT, "instructions": "changed"},
            evidence=evidence, actor_id="owner",
        )
    assert agent.deployment_hash == first
    _record(session, evidence, model_version="house-v4")
    again = approve_deployment(
        session, agent.name, deployment={**_DEPLOYMENT, "instructions": "changed"},
        evidence=evidence, actor_id="owner",
    )
    assert again.deployment_hash != first


@pytest.mark.parametrize("kwargs", [
    {"level": "System"},
    {"metrics": {}},
    {"metrics": {**_metrics(), "extra": 1.0}},
    {"metrics": _metrics(accuracy=True)},
    {"metrics": _metrics(accuracy=1.1)},
    {"metrics": _metrics(hallucination_rate=-0.01)},
    {"agent_id": " "},
    {"model_version": ""},
    {"version": " "},
])
def test_invalid_results_are_not_recorded(kwargs):
    with pytest.raises(ValueError, match="EVALUATION_"):
        _result(**kwargs)


def test_tampered_retention_denies(session, evidence):
    from kognita.models import RetainedContent

    _arm(session, evidence)
    agent = _agent(session)
    _record(session, evidence)
    row = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.event_type == EventType.EVALUATION_RECORDED)).all()[-1]
    retained = session.get(RetainedContent, row.payload["content_hash"])
    retained.body = "{}"
    session.add(retained)
    session.flush()
    with pytest.raises(ValueError, match="EVALUATION_UNREADABLE"):
        check_activation(session, agent)


def test_redacted_results_deny(session, evidence, engine):
    _arm(session, evidence)
    agent = _agent(session)
    redacted = EvidenceWriter(engine, redact_payload=lambda kind, body: {})
    record_evaluation(session, _result(), evidence=redacted)
    with pytest.raises(ValueError, match="EVALUATION_REQUIRED"):
        check_activation(session, agent)
    partial = EvidenceWriter(engine, redact_payload=lambda kind, body: {
        key: body[key] for key in ("action", "agent_id", "version", "model_version", "level", "content_hash")
        if key in body
    })
    record_evaluation(session, _result(), evidence=partial)
    with pytest.raises(ValueError, match="EVALUATION_UNREADABLE"):
        check_activation(session, agent)


def test_manual_partial_event_cannot_bypass(session, evidence):
    _arm(session, evidence)
    agent = _agent(session)
    body = {"agent_id": agent.name, "version": agent.version, "model_version": agent.model,
            "level": "system", "metrics": {"accuracy": 1}}
    digest = RetentionStore().retain_value(session, body, kind="evaluation", correlation_id="evaluation:x")
    evidence.emit(
        session, correlation_id=f"evaluation:{agent.name}", event_type=EventType.EVALUATION_RECORDED,
        actor_type=ActorType.HUMAN, actor_id=agent.name,
        payload={"action": "RECORDED", **body, "content_hash": digest},
    )
    with pytest.raises(ValueError, match="EVALUATION_UNREADABLE"):
        check_activation(session, agent)


def test_threshold_vocabulary_and_record_round_trip():
    spec = _tier2(evaluation_thresholds=_thresholds())
    record = build_record(spec, version=1)
    assert record["evaluation_thresholds"]["hallucination_rate"] == 0.05
    assert from_record(record).evaluation_thresholds["accuracy"] == 0.9
    assert not any("EVALUATION_THRESHOLDS_INVALID" in item for item in vocabulary_citations(spec))
    for bad in ({"accuracy": True}, {"accuracy": 2}, {"unknown": 0.5}, [], "accuracy"):
        cited = vocabulary_citations(replace(spec, evaluation_thresholds=bad))
        assert any("EVALUATION_THRESHOLDS_INVALID" in item for item in cited)


def test_policy_evaluation_key(tmp_path):
    path = tmp_path / "evaluation.yaml"
    path.write_text("evaluation:\n  accuracy: 0.9\n  hallucination_rate: 0.05\n")
    loaded = read_policy_set(path)
    assert loaded.evaluation == {"accuracy": 0.9, "hallucination_rate": 0.05}
    for bad in ("true", "2", "nope", "null"):
        path.write_text(f"evaluation:\n  accuracy: {bad}\n")
        with pytest.raises(PolicyLanguageError):
            read_policy_set(path)
    path.write_text("evaluation:\n  not_a_metric: 0.5\n")
    with pytest.raises(PolicyLanguageError, match="EVALUATION_THRESHOLDS_INVALID"):
        read_policy_set(path)
    path.write_text("""evaluation:
  accuracy: 0.9
freshness:
  positions: 5m
rules:
  - id: prohibit
    regime: INTERNAL
    rule_type: PROHIBITED
    citation: No execution
    effective_from: 2026-01-01T00:00:00Z
    rule:
      on_violation: fail
""")
    current = read_policy_set(path)
    assert current.evaluation["accuracy"] == 0.9
    assert current.freshness and current.rules
    from kognita.policy import PolicyChange, apply_change
    changed = apply_change(current, PolicyChange(removed=("prohibit",)), directory=tmp_path)
    assert changed.evaluation == current.evaluation
    assert read_policy_set(tmp_path).evaluation == current.evaluation
    (tmp_path / "again.yaml").write_text("evaluation:\n  accuracy: 0.8\n")
    with pytest.raises(PolicyLanguageError, match="duplicate evaluation threshold"):
        read_policy_set(tmp_path)


def _populate(session, evidence):
    _arm(session, evidence)
    clients = {}
    for client_id, outcome, product in (
        ("c1", "recommend", "fund-a"),
        ("c2", "recommend", "fund-a"),
        ("c3", "do not contact", ""),
        ("c4", "do not contact", ""),
    ):
        interaction = open_interaction(
            session, client_id=client_id, evidence=evidence, use_case="client-intelligence", now=NOW,
        )
        clients[client_id] = interaction.id
        if outcome == "recommend":
            originate(
                session, interaction.id, evidence=evidence, trigger="RM request",
                candidate_set_size=4, criteria=["house view"], reasons=[_REASON],
                product_id=product, now=NOW,
            )
        else:
            originate(
                session, interaction.id, evidence=evidence, trigger="RM request",
                candidate_set_size=4, criteria=["house view"], do_not_contact=["low relevance"],
                now=NOW,
            )
    return clients


def _segments():
    return {
        "nationality": {"c1": "SG", "c2": "SG", "c3": "CH", "c4": "CH"},
        "postcode": {"c1": "018956", "c2": "018956", "c3": "8001", "c4": "8001"},
    }


def test_fairness_flags_disparities_including_proxies(session, evidence):
    _populate(session, evidence)
    before = session.exec(select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.OUTCOME)).all()
    report = fairness_report(
        session, segments=_segments(), disparity_threshold=0.2, proxy_attributes=("postcode",),
        evidence=evidence, usecase_id="client-intelligence", register_monitored=True,
    )
    assert isinstance(report, FairnessReport)
    assert report.flagged
    assert report.monitored_metric is None
    assert report.risk_owners == ("Priya Shah",)
    assert report.proxy_attributes == ("postcode",)
    rates = {(row["segment"], row["value"]): row for row in report.rates}
    assert rates[("nationality", "SG")]["recommendation_rate"] == 1
    assert rates[("nationality", "CH")]["contact_rate"] == 0
    assert rates[("nationality", "SG")]["prioritisation_rate"] == 1
    assert rates[("postcode", "018956")]["proxy"] is True
    assert any(item["proxy"] and item["segment"] == "postcode" for item in report.disparities)
    assert any(item["rate"] == "recommendation_rate" and item["value"] == "CH" for item in report.disparities)
    assert not any(item["rate"] == "prioritisation_rate" for item in report.disparities)
    flag = session.exec(select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.REVIEW)).all()[-1]
    assert flag.payload["action"] == "REVIEW_FLAG"
    assert flag.payload["reason"] == "FAIRNESS_DISPARITY"
    assert flag.payload["risk_owner"] == "Priya Shah"
    after = session.exec(select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.OUTCOME)).all()
    assert after == list(before)


def test_even_rates_are_not_flagged(session, evidence):
    _populate(session, evidence)
    balanced = {
        "nationality": {"c1": "SG", "c2": "CH", "c3": "SG", "c4": "CH"},
    }
    report = fairness_report(
        session, segments=balanced, disparity_threshold=0.2, evidence=evidence,
        usecase_id="client-intelligence",
    )
    assert report.flagged is False
    assert report.disparities == ()
    assert not any(
        event.payload.get("reason") == "FAIRNESS_DISPARITY"
        for event in session.exec(select(EvidenceEvent)).all()
    )


def test_omitted_clients_stay_visible_as_unassigned(session, evidence):
    _populate(session, evidence)
    report = fairness_report(
        session, segments={"nationality": {"c3": "CH", "c4": "CH"}}, disparity_threshold=0.1,
        evidence=evidence, usecase_id="client-intelligence",
    )
    values = {row["value"] for row in report.rates if row["segment"] == "nationality"}
    assert "(unassigned)" in values
    assert report.flagged
    assert any(item["value"] == "(unassigned)" for item in report.disparities)


def test_fairness_rejects_bad_inputs_and_mismatched_evidence(session, evidence):
    _populate(session, evidence)
    with pytest.raises(ValueError, match="FAIRNESS_SEGMENTS_REQUIRED"):
        fairness_report(session, segments={}, disparity_threshold=0.1, evidence=evidence)
    with pytest.raises(ValueError, match="FAIRNESS_PROXY_UNKNOWN"):
        fairness_report(
            session, segments=_segments(), disparity_threshold=0.1, proxy_attributes=("religion",),
            evidence=evidence,
        )
    with pytest.raises(ValueError, match="FAIRNESS_THRESHOLD_INVALID"):
        fairness_report(session, segments=_segments(), disparity_threshold=True, evidence=evidence)
    interaction = open_interaction(
        session, client_id="c9", evidence=evidence, use_case="client-intelligence", now=NOW,
    )
    evidence.emit(
        session, correlation_id=interaction.id, event_type=EventType.ORIGINATION,
        actor_type=ActorType.SYSTEM, actor_id="governance-pdp",
        payload={"interaction_id": interaction.id, "client_id": "someone-else", "outcome": "recommend",
                 "product_id": "fund-a"},
    )
    with pytest.raises(ValueError, match="FAIRNESS_EVIDENCE_MISMATCH"):
        fairness_report(
            session, segments=_segments(), disparity_threshold=0.2, evidence=evidence,
            usecase_id="client-intelligence",
        )


def test_redacted_fairness_flag_is_refused(session, evidence, engine):
    _populate(session, evidence)
    redacted = EvidenceWriter(engine, redact_payload=lambda kind, body: {})
    with pytest.raises(ValueError, match="FAIRNESS_FLAG_REQUIRED"):
        fairness_report(
            session, segments=_segments(), disparity_threshold=0.2, evidence=redacted,
            usecase_id="client-intelligence",
        )
    session.rollback()


def test_cli_record_deploy_and_report(tmp_path, capsys):
    db = str(tmp_path / "eval.db")
    engine = make_engine(db)
    create_all(engine)
    evidence = EvidenceWriter(engine, clock=lambda: NOW)
    with Session(engine) as session:
        _arm(session, evidence)
        session.commit()
    manifest = {
        "name": "briefing-agent", "owner_exec": "owner", "version": "1.2.0",
        "purpose": "Prepare a client briefing", "use_cases": ["client-intelligence"],
        "model": "house-v3", "deployment": _DEPLOYMENT,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert main(["agent", "deploy", "--manifest", str(path), "--db", db, "--actor", "owner"]) == 2
    denied = json.loads(capsys.readouterr().out)
    assert denied["action"] == "DENIED"
    assert denied["reason"].startswith("EVALUATION_")
    with Session(engine) as session:
        assert session.exec(select(UseCase).where(UseCase.usecase_id == "client-intelligence")).first()
        from kognita.models import Agent
        assert session.exec(select(Agent)).first() is None
    record_args = [
        "eval", "record", "--db", db, "--agent", "briefing-agent", "--version", "1.2.0",
        "--model-version", "house-v3", "--level", "model",
    ]
    assert main(record_args) == 2
    assert json.loads(capsys.readouterr().out)["action"] == "DENIED"
    record_args = [
        "eval", "record", "--db", db, "--agent", "briefing-agent", "--version", "1.2.0",
        "--model-version", "house-v3", "--level", "system",
    ]
    for name, value in _metrics().items():
        record_args.extend(["--metric", f"{name}={value}"])
    assert main(record_args) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "RECORDED"
    assert main(["agent", "deploy", "--manifest", str(path), "--db", db, "--actor", "owner"]) == 0
    assert json.loads(capsys.readouterr().out)["deployment_hash"]
    with Session(engine) as session:
        for client_id, product in (("c1", "fund-a"), ("c2", "")):
            interaction = open_interaction(
                session, client_id=client_id, evidence=evidence, use_case="client-intelligence", now=NOW,
            )
            originate(
                session, interaction.id, evidence=evidence, trigger="event", candidate_set_size=2,
                criteria=["house view"], reasons=[_REASON], product_id=product, now=NOW,
            )
        session.commit()
    segments = tmp_path / "segments.json"
    segments.write_text(json.dumps({"nationality": {"c1": "SG", "c2": "CH"}, "postcode": {"c1": "018956", "c2": "8001"}}))
    assert main([
        "eval", "report", "--db", db, "--usecase", "client-intelligence", "--threshold", "0.1",
        "--proxy", "postcode", "--segments", str(segments), "--register-metric",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["flagged"] is True
    assert report["risk_owners"] == ["Priya Shah"]
    assert report["monitored_metric"] is None
    assert any(item["proxy"] for item in report["disparities"])
    assert main(["eval", "report", "--db", db, "--threshold", "0.1", "--proxy", "missing"]) == 2
    assert json.loads(capsys.readouterr().out)["action"] == "DENIED"


def test_public_exports():
    import kognita
    for name in ("EvaluationResult", "FairnessReport", "record_evaluation", "check_activation", "fairness_report"):
        assert name in kognita.__all__
    assert EventType.EVALUATION_RECORDED.value == "EVALUATION_RECORDED"
    spec = UseCaseSpec(
        "research-summary", "Summarise approved research for an RM", "research summarization",
        "Jane Smith", "Omar Aziz", "Priya Shah", "Luis Ortega", "Alex Chen",
        evaluation_thresholds={"accuracy": 0.5},
    )
    assert spec.evaluation_thresholds["accuracy"] == 0.5
