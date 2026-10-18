"""The flight recorder reads the evidence chain.

A missing run is refused. A missing rate is unmeasured. An export verifies
with the same chain check as the store. A telemetry failure leaves the
decision event as it was appended.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from fixtures import demo_pack as dp

from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.evidence import ChainBreak, EvidenceWriter, verify_chain, verify_export
from kognita.models import EvidenceEvent, RunRecord
from kognita.recorder import (
    REPEATED_DENIALS,
    RecorderError,
    alerts,
    effectiveness,
    export_run,
    recent_runs,
    show_run,
    telemetry_signals,
)
from kognita.registry import register
from kognita.testing.harness import FROZEN_NOW, Harness
from kognita.tools import Run, ToolRegistry, run_governed
from kognita.vocabulary import ActorType, EventType, Outcome


def _decision(
    evidence,
    session,
    *,
    correlation: str,
    outcome: str,
    principal: str = "alice",
    agent: str = "",
    tool: str = "read_note",
    run_id: str = "",
    checks: list | None = None,
    confidence: float | None = None,
    model: str = "pattern",
    version: str = "1",
) -> None:
    if run_id:
        if session.get(RunRecord, run_id) is None:
            session.add(RunRecord(id=run_id, approvals_pending=[]))
            session.flush()
    attributes: dict = {}
    if confidence is not None:
        attributes["classifier"] = {
            "model": model,
            "version": version,
            "label": "C1",
            "confidence": confidence,
            "input_hash": "abc",
        }
    payload = {
        "envelope": {
            "principal": principal,
            "agent_name": agent,
            "tool": tool,
            "purpose": "research-summary",
        },
        "attributes": attributes,
        "outcome": outcome,
        "checks": checks
        or [
            {
                "check": "PURPOSE",
                "regime": "INTERNAL",
                "result": "pass" if outcome == "ALLOW" else "fail",
                "citation": "purpose rule",
            }
        ],
    }
    if run_id:
        payload["run_id"] = run_id
    evidence.emit(
        session,
        correlation_id=correlation,
        event_type=EventType.POLICY_DECISION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload=payload,
    )


def test_recent_runs_filter_by_actor_tool_and_outcome(session, evidence):
    _decision(
        evidence,
        session,
        correlation="c-1",
        outcome="ALLOW",
        principal="alice",
        tool="read_note",
        run_id="run-a",
    )
    _decision(
        evidence,
        session,
        correlation="c-2",
        outcome="DENY",
        principal="bob",
        tool="send_note",
        run_id="run-b",
    )
    session.commit()

    denied = recent_runs(session, outcome="DENY")
    assert [row["run_id"] for row in denied] == ["run-b"]
    assert recent_runs(session, actor="alice")[0]["run_id"] == "run-a"
    assert recent_runs(session, tool="send_note")[0]["tools"] == ["send_note"]
    assert recent_runs(session, actor="nobody") == ()


def test_show_run_lists_citations_and_refuses_an_unknown_run(session, evidence):
    _decision(
        evidence,
        session,
        correlation="c-1",
        outcome="DENY",
        run_id="run-a",
        checks=[
            {
                "check": "BUDGET",
                "regime": "INTERNAL",
                "result": "fail",
                "citation": "max_calls",
            }
        ],
    )
    session.commit()

    report = show_run(session, "run-a")
    assert report["decisions"][0]["citations"][0]["citation"] == "max_calls"
    assert report["decisions"][0]["citations_recorded"] is True
    assert report["evidence"][0]["event_hash"]
    with pytest.raises(RecorderError, match="not in the evidence"):
        show_run(session, "missing")


def test_export_run_verifies_and_marks_only_that_run(session, evidence):
    _decision(evidence, session, correlation="c-1", outcome="ALLOW", run_id="run-a")
    _decision(evidence, session, correlation="c-2", outcome="DENY", run_id="run-b")
    session.commit()

    artifact = export_run(session, "run-a")
    assert verify_export(artifact) == artifact["event_count"]
    interest = [row for row in artifact["events"] if row["of_interest"]]
    assert interest
    assert all(row["payload"].get("run_id") == "run-a" for row in interest)
    assert artifact["filters"]["run_id"] == "run-a"
    tampered = json.loads(json.dumps(artifact))
    tampered["events"][0]["payload"]["outcome"] = "DENY"
    with pytest.raises(ChainBreak):
        verify_export(tampered)
    with pytest.raises(RecorderError):
        export_run(session, "missing")


def test_a_limit_below_one_is_refused(session):
    with pytest.raises(RecorderError):
        recent_runs(session, limit=0)


def test_budget_and_repeated_denials_and_an_intact_chain(session, evidence):
    _decision(
        evidence,
        session,
        correlation="c-budget",
        outcome="DENY",
        principal="alice",
        run_id="run-budget",
        checks=[
            {
                "check": "BUDGET",
                "regime": "INTERNAL",
                "result": "fail",
                "citation": "max_calls",
            }
        ],
    )
    for index in range(REPEATED_DENIALS):
        _decision(
            evidence,
            session,
            correlation=f"c-deny-{index}",
            outcome="DENY",
            principal="carol",
            run_id=f"run-deny-{index}",
        )
    session.commit()

    report = alerts(session)
    by_kind = {item["kind"]: item for item in report["alerts"]}
    assert by_kind["budget exceeded"]["fired"] is True
    assert by_kind["budget exceeded"]["events"][0]["citations"] == ["max_calls"]
    denials = by_kind["repeated denials"]
    assert denials["fired"] is True
    assert denials["measured"] is True
    assert denials["actors"] == [
        {"actor": "carol", "denials": REPEATED_DENIALS, "measured": True}
    ]
    assert by_kind["chain break"]["fired"] is False
    assert by_kind["chain break"]["measured"] is True
    assert verify_chain(session) == by_kind["chain break"]["checked"]


def test_a_denial_with_no_actor_is_unmeasured(session, evidence):
    evidence.emit(
        session,
        correlation_id="c-blank",
        event_type=EventType.POLICY_DECISION,
        actor_type=ActorType.SYSTEM,
        actor_id="governance-pdp",
        payload={
            "envelope": {"principal": "", "agent_name": "", "tool": "read_note"},
            "outcome": "DENY",
            "checks": [
                {
                    "check": "PURPOSE",
                    "regime": "INTERNAL",
                    "result": "fail",
                    "citation": "purpose rule",
                }
            ],
        },
    )
    session.commit()
    denials = next(item for item in alerts(session)["alerts"] if item["kind"] == "repeated denials")
    assert denials["fired"] is True
    assert denials["measured"] is False
    assert denials["detail"] == "unmeasured"
    assert denials["unattributed"] == 1


def test_confidence_drift_is_unmeasured_until_the_sample_exists(session, evidence):
    _decision(
        evidence,
        session,
        correlation="c-1",
        outcome="ALLOW",
        run_id="run-1",
        confidence=0.99,
    )
    session.commit()
    drift = next(
        item
        for item in alerts(session)["alerts"]
        if item["kind"] == "classifier confidence drift"
    )
    assert drift["fired"] is False
    assert drift["measured"] is False
    assert drift["detail"] == "unmeasured"

    for index, confidence in enumerate((0.99, 0.40, 0.40)):
        _decision(
            evidence,
            session,
            correlation=f"c-more-{index}",
            outcome="ALLOW",
            run_id=f"run-more-{index}",
            confidence=confidence,
        )
    session.commit()
    drift = next(
        item
        for item in alerts(session)["alerts"]
        if item["kind"] == "classifier confidence drift"
    )
    assert drift["measured"] is True
    assert drift["fired"] is True
    assert drift["series"][0]["earlier_mean"] == pytest.approx(0.99)
    assert drift["series"][0]["later_mean"] == pytest.approx(0.40)


def test_two_classifier_versions_are_not_one_series(session, evidence):
    for index in range(4):
        _decision(
            evidence,
            session,
            correlation=f"v1-{index}",
            outcome="ALLOW",
            run_id=f"run-v1-{index}",
            confidence=0.9,
            version="1",
        )
        _decision(
            evidence,
            session,
            correlation=f"v2-{index}",
            outcome="ALLOW",
            run_id=f"run-v2-{index}",
            confidence=0.2,
            version="2",
        )
    session.commit()
    drift = next(
        item
        for item in alerts(session)["alerts"]
        if item["kind"] == "classifier confidence drift"
    )
    assert drift["fired"] is False
    assert {row["version"] for row in drift["series"]} == {"1", "2"}


def test_a_chain_break_is_an_alert(session, evidence):
    _decision(evidence, session, correlation="c-1", outcome="ALLOW", run_id="run-a")
    session.commit()
    event = session.exec(select(EvidenceEvent)).one()
    event.payload = {"tampered": True}
    session.add(event)
    session.commit()
    chain = next(item for item in alerts(session)["alerts"] if item["kind"] == "chain break")
    assert chain["fired"] is True
    assert chain["measured"] is True
    assert "payload" in chain["reason"]


def test_injection_and_entitlement_are_unmeasured_without_a_record(session, evidence):
    _decision(evidence, session, correlation="c-1", outcome="ALLOW", run_id="run-a")
    session.commit()
    report = alerts(session)
    signals = {item["signal"] for item in report["unmeasured"]}
    assert signals == {"injection", "entitlement"}
    assert all(item["detail"] == "unmeasured" for item in report["unmeasured"])


def test_a_recorded_injection_is_not_reported_as_unmeasured(session, evidence):
    evidence.emit(
        session,
        correlation_id="c-inject",
        event_type=EventType.POLICY_DECISION,
        payload={
            "outcome": "DENY",
            "envelope": {"principal": "alice", "tool": "read_note"},
            "injection": True,
            "checks": [
                {
                    "check": "INJECTION",
                    "regime": "INTERNAL",
                    "result": "fail",
                    "citation": "injection attempt",
                }
            ],
        },
    )
    session.commit()
    report = alerts(session)
    assert report["injection_attempts"]
    assert "injection" not in {item["signal"] for item in report["unmeasured"]}


def test_effectiveness_rates_come_from_evidence_and_a_gap_is_unmeasured(session, evidence):
    evidence.emit(
        session,
        correlation_id="interaction-1",
        event_type=EventType.INTERACTION,
        payload={
            "action": "OPENED",
            "interaction_id": "interaction-1",
            "use_case": "research-summary",
        },
    )
    for outcome in ("approve", "modify", "reject"):
        evidence.emit(
            session,
            correlation_id="interaction-1",
            event_type=EventType.FINAL_DECISION,
            actor_type=ActorType.HUMAN,
            actor_id="rm-1",
            payload={
                "kind": "rm_review",
                "interaction_id": "interaction-1",
                "outcome": outcome,
                "citation": "RM review: final decision",
            },
        )
    evidence.emit(
        session,
        correlation_id="interaction-1",
        event_type=EventType.OUTCOME,
        payload={
            "action": "RECORDED",
            "kind": "wrong-client targeting",
            "interaction_id": "interaction-1",
            "usecase_id": "research-summary",
        },
    )
    session.commit()

    report = effectiveness(session, usecase_id="research-summary")
    rates = report["use_cases"][0]["rates"]
    assert rates["acceptance"]["value"] == pytest.approx(1 / 3)
    assert rates["edit"]["value"] == pytest.approx(1 / 3)
    assert rates["rejection"]["value"] == pytest.approx(1 / 3)
    assert rates["wrong_client"]["value"] == pytest.approx(1)
    assert rates["acceptance"]["measured"] is True

    empty = effectiveness(session, usecase_id="no-such-use-case")
    assert empty["detail"] == "unmeasured"
    assert empty["use_cases"][0]["rates"]["acceptance"]["measured"] is False
    assert empty["use_cases"][0]["rates"]["acceptance"]["value"] is None


def test_signals_do_not_treat_a_missing_token_count_as_zero():
    decision = telemetry_signals(
        {
            "kind": "evidence",
            "event_type": "POLICY_DECISION",
            "correlation_id": "c-1",
            "sequence": 1,
            "payload": {"outcome": "DENY"},
        }
    )
    by_name = {item["name"]: item for item in decision}
    assert by_name["decision"]["measured"] is True
    assert by_name["denial"]["value"] == 1
    assert by_name["injection"]["measured"] is False
    assert by_name["injection"]["value"] is None
    assert by_name["entitlement"]["detail"] == "unmeasured"
    tokens = telemetry_signals(
        {
            "kind": "evidence",
            "event_type": "MODEL_CALL",
            "payload": {"sent": True},
        }
    )[0]
    assert tokens["name"] == "tokens"
    assert tokens["measured"] is False
    assert tokens["detail"] == "unmeasured"
    assert tokens["value"] is None
    latency = telemetry_signals({"kind": "latency", "seconds": 0.01, "correlation_id": "c-1"})[0]
    assert latency["measured"] is True
    assert latency["value"] == pytest.approx(0.01)


def test_a_failing_export_does_not_change_the_decision(session, engine):
    seen: list[dict] = []

    def hook(snapshot):
        seen.append(snapshot)
        raise RuntimeError("collector down")

    writer = EvidenceWriter(engine, telemetry=hook)
    event = writer.emit(
        session,
        correlation_id="c-1",
        event_type=EventType.POLICY_DECISION,
        payload={"outcome": "DENY", "checks": [{"citation": "purpose rule"}]},
    )
    session.commit()
    stored = session.exec(select(EvidenceEvent)).one()
    assert stored.payload["outcome"] == "DENY"
    assert stored.payload_hash == event.payload_hash
    assert stored.event_hash == event.event_hash
    assert verify_chain(session) == 1
    assert writer.telemetry_failures
    assert "collector down" in writer.telemetry_failures[0]
    assert seen[0]["payload"]["outcome"] == "DENY"
    seen[0]["payload"]["outcome"] = "ALLOW"
    assert stored.payload["outcome"] == "DENY"


def test_the_gateway_keeps_the_decision_when_export_fails(session, evidence, engine):
    from kognita.gateway import ClientConfiguration, Gateway

    def hook(snapshot):
        raise RuntimeError("collector down")

    evidence.telemetry = hook
    gateway = Gateway(
        engine=engine,
        evidence=evidence,
        upstream="https://api.openai.com",
        client=ClientConfiguration(
            principal="alice",
            purpose="research-summary",
            agent_names=frozenset({"dossier-agent"}),
        ),
        purposes=(),
        transport=lambda *args: (_ for _ in ()).throw(AssertionError("upstream")),
    )
    response = gateway.handle(
        "POST",
        "/v1/chat/completions",
        {"agent_name": "dossier-agent"},
        b'{"model":"gateway-model","messages":[{"role":"user","content":"hello"}]}',
        session=session,
    )
    session.commit()
    body = json.loads(response.body)
    assert body["outcome"] == Outcome.DENY.value
    assert response.headers["x-kognita-telemetry-export"] == "failed"
    assert verify_chain(session) >= 1
    stored = session.exec(
        select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.POLICY_DECISION)
    ).all()
    assert stored
    assert all(event.payload["outcome"] == "DENY" for event in stored)


def test_run_governed_names_the_run_on_the_decision():
    def seed(current):
        dp.seed_policies(current, now=FROZEN_NOW)
        register(current, name="dossier-agent", owner_exec="Head of Research Ops")

    harness = Harness(pack=dp.DemoPack(), purposes=dp.PURPOSES, seed=seed)
    registry = ToolRegistry()

    @registry.tool("get_subject_profile")
    def _profile(envelope, evaluation, current):
        return {"subject": envelope.subject_id}

    run = Run(id="run-recorded")
    with harness.session() as current:
        result = run_governed(
            current,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=run,
            now=harness.now,
        )
        current.commit()
        assert result.evaluation.outcome is Outcome.ALLOW
        rows = recent_runs(current)
        assert rows[0]["run_id"] == "run-recorded"
        assert "get_subject_profile" in rows[0]["tools"]


def test_recorder_cli_exports_a_run(tmp_path: Path):
    path = tmp_path / "store.db"
    engine = make_engine(path)
    create_all(engine)
    writer = EvidenceWriter(engine)
    with Session(engine) as current:
        current.add(RunRecord(id="run-cli", approvals_pending=[]))
        current.flush()
        writer.emit(
            current,
            correlation_id="c-1",
            event_type=EventType.POLICY_DECISION,
            payload={
                "run_id": "run-cli",
                "outcome": "ALLOW",
                "envelope": {"principal": "alice", "tool": "read_note"},
                "checks": [
                    {
                        "check": "PURPOSE",
                        "regime": "INTERNAL",
                        "result": "pass",
                        "citation": "purpose rule",
                    }
                ],
            },
        )
        current.commit()
    output = tmp_path / "run.json"
    assert main(["recorder", "export", "run-cli", "--db", str(path), "-o", str(output)]) == 0
    artifact = json.loads(output.read_text())
    assert verify_export(artifact) == artifact["event_count"]
    assert main(["recorder", "export", "missing", "--db", str(path)]) == 1
    assert main(["recorder", "runs", "--db", str(path), "--actor", "alice"]) == 0


def test_attach_refuses_without_an_endpoint(monkeypatch, engine):
    from kognita.otel import OtelError, attach

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    writer = EvidenceWriter(engine)
    with pytest.raises(OtelError, match="OTEL_EXPORTER_OTLP_ENDPOINT"):
        attach(writer)
    assert writer.telemetry is None


def test_otlp_export_failure_is_visible_and_a_decision_is_not_a_zero(monkeypatch):
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import opentelemetry.exporter.otlp.proto.http.trace_exporter as exporter_mod
    from kognita.otel import OtelError, build_hook

    memory = InMemorySpanExporter()

    class _Memory:
        def export(self, spans):
            return memory.export(spans)

        def shutdown(self) -> None:
            memory.shutdown()

    class _Broken:
        def export(self, spans):
            raise RuntimeError("collector down")

        def shutdown(self) -> None:
            return None

    monkeypatch.setattr(exporter_mod, "OTLPSpanExporter", lambda: _Memory())
    hook = build_hook()
    hook(
        {
            "kind": "evidence",
            "event_type": "POLICY_DECISION",
            "correlation_id": "c-1",
            "sequence": 1,
            "payload": {"outcome": "ALLOW"},
        }
    )
    spans = {span.name: span for span in memory.get_finished_spans()}
    assert spans["kognita.denial"].attributes["kognita.value"] == 0
    assert spans["kognita.injection"].attributes["kognita.measured"] is False
    assert "kognita.value" not in spans["kognita.injection"].attributes

    monkeypatch.setattr(exporter_mod, "OTLPSpanExporter", lambda: _Broken())
    failing = build_hook()
    with pytest.raises(OtelError, match="collector down"):
        failing(
            {
                "kind": "evidence",
                "event_type": "POLICY_DECISION",
                "payload": {"outcome": "DENY"},
            }
        )


def test_build_hook_names_the_extra_when_it_is_absent():
    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        from kognita.otel import OtelError, build_hook

        with pytest.raises(OtelError, match=r"kognita\[otel\]"):
            build_hook()
    else:
        pytest.skip("the otel extra is installed")
