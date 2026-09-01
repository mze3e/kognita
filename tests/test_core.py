"""Core machinery tests: replay, evidence integrity, egress, tool ordering.

These cover properties the ported suite does not, because the TypeScript
original had no equivalent — a pure decision function, a hash chain, and an
egress guard are all additions.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pytest
from sqlmodel import select

from fixtures import demo_pack as dp

from kognita.core.approvals import ApprovalError, expire_stale, find_granted, grant, reject
from kognita.core.canonical import canonical_hash, canonical_json
from kognita.core.classify import PatternClassifier
from kognita.core.egress import (
    EgressDenied,
    EgressGuard,
    EgressPolicy,
    NullRedactor,
    PatternRedactor,
)
from kognita.core.evidence import (
    ChainBreak,
    EvidenceWriter,
    export_chain,
    hashes_only,
    verify_chain,
    verify_export,
)
from kognita.core.models import Approval, EvidenceEvent
from kognita.core.registry import register
from kognita.core.tools import ToolNotRegistered, ToolRegistry
from kognita.core.vocabulary import (
    ApprovalStatus,
    Classification,
    EgressDecision,
    EventType,
    Outcome,
)
from kognita.testing.harness import FROZEN_NOW, Harness


def _seed(session):
    dp.seed_policies(session, now=FROZEN_NOW)
    register(session, name="eligibility-assistant", owner_exec="Head of Data Governance")
    register(session, name="dossier-agent", owner_exec="Head of Research Ops")


@pytest.fixture
def harness():
    return Harness(pack=dp.DemoPack(), purposes=dp.PURPOSES, seed=_seed)


# ── Canonical hashing ────────────────────────────────────────────────────────


def test_hash_is_independent_of_key_order():
    """Insertion order must not change a hash.

    Python dicts preserve insertion order and json.dumps follows it, so without
    sort_keys the same envelope built two different ways would produce two
    different hashes — and an approval bound to one would not match the other.
    """
    assert canonical_hash({"b": 1, "a": [3, 2]}) == canonical_hash({"a": [3, 2], "b": 1})
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_hash_covers_nested_and_typed_values():
    from datetime import datetime, timezone
    from decimal import Decimal

    payload = {
        "when": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "amount": Decimal("42.50"),
        "level": Classification.C2,
    }
    assert canonical_hash(payload) == canonical_hash(dict(payload))


# ── as_of replay ─────────────────────────────────────────────────────────────


def test_as_of_replays_a_superseded_policy(harness):
    """A withdrawn rule must still bind for decisions dated before its withdrawal.

    This is the question an audit actually asks — not "is this allowed?" but
    "was it allowed then?". It only has an answer because decide() is pure and
    takes the instant as a parameter.
    """
    envelope = dp.envelope("get_subject_profile", site="AE", subject="2")

    with harness.session() as session:
        assert harness.evaluate(envelope, session).outcome is Outcome.ALLOW

        dp.seed_superseded_policy(session, now=FROZEN_NOW)
        session.commit()

        # Still allowed today: the rule lapsed 30 days ago.
        assert harness.evaluate(envelope, session).outcome is Outcome.ALLOW

        # Denied as of 60 days ago, when it was in force.
        historic = harness.evaluate(
            envelope, session, as_of=FROZEN_NOW - timedelta(days=60)
        )
        assert historic.outcome is Outcome.DENY
        assert any("superseded" in c.citation for c in historic.failures())


def test_as_of_is_recorded_alongside_the_decision(harness):
    from kognita.core.models import GovernanceDecision

    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    at = FROZEN_NOW - timedelta(days=5)
    with harness.session() as session:
        evaluation = harness.decide_and_record(envelope, session, as_of=at)
        session.commit()
        decision = session.exec(
            select(GovernanceDecision).where(
                GovernanceDecision.request_id == evaluation.request_id
            )
        ).one()
        assert decision.as_of == at
        assert decision.decided_at == harness.now


def test_unevaluable_policy_escalates_rather_than_being_ignored(harness):
    """A rule nobody can evaluate is not a rule that does not apply.

    Silently skipping it would let a policy be neutralised by deleting its
    evaluator, which is the sort of change nobody reviews.
    """
    from kognita.core.models import Policy

    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        session.add(
            Policy(
                regime="HOME_SITE",
                rule_type="A_RULE_TYPE_WITH_NO_EVALUATOR",
                rule={},
                citation="Charter s12",
                effective_from=FROZEN_NOW - timedelta(days=10),
            )
        )
        session.commit()
        evaluation = harness.evaluate(envelope, session)

    assert evaluation.outcome is Outcome.ESCALATE
    assert any("UNEVALUABLE_POLICY" in c.check for c in evaluation.checks)


# ── Evidence ─────────────────────────────────────────────────────────────────


def test_chain_detects_a_rewritten_payload(harness):
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        for _ in range(3):
            harness.decide_and_record(envelope, session)
        session.commit()
        assert verify_chain(session) == 3

        event = session.exec(
            select(EvidenceEvent).where(EvidenceEvent.sequence == 2)
        ).one()
        event.payload = {**event.payload, "outcome": "ALLOW-ish"}
        session.add(event)
        session.commit()

        with pytest.raises(ChainBreak) as excinfo:
            verify_chain(session)
    assert excinfo.value.sequence == 2


def test_chain_detects_a_deleted_event(harness):
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        for _ in range(3):
            harness.decide_and_record(envelope, session)
        session.commit()

        victim = session.exec(
            select(EvidenceEvent).where(EvidenceEvent.sequence == 2)
        ).one()
        session.delete(victim)
        session.commit()

        with pytest.raises(ChainBreak):
            verify_chain(session)


def test_export_verifies_without_a_database(harness):
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        harness.decide_and_record(envelope, session)
        session.commit()
        payload = export_chain(session)

    # Survives a JSON round-trip: the artifact is portable, not just in-process.
    reloaded = json.loads(json.dumps(payload))
    assert verify_export(reloaded) == payload["event_count"]

    reloaded["events"][0]["payload"]["tampered"] = True
    with pytest.raises(ChainBreak):
        verify_export(reloaded)


def test_export_marks_events_of_interest_but_keeps_the_whole_chain(harness):
    """A slice starting mid-chain cannot be verified, so the export never truncates."""
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        first = harness.decide_and_record(envelope, session)
        second = harness.decide_and_record(envelope, session)
        session.commit()
        payload = export_chain(session, correlation_id=second.request_id)

    assert payload["event_count"] > payload["interest_count"]
    assert verify_export(payload) == payload["event_count"]
    interesting = [e for e in payload["events"] if e["of_interest"]]
    assert {e["correlation_id"] for e in interesting} == {second.request_id}
    assert first.request_id != second.request_id


def test_redact_payload_hook_keeps_content_out_of_the_log(harness):
    """The strictest setting proves what happened while holding none of it."""
    writer = EvidenceWriter(
        harness.engine, redact_payload=hashes_only, clock=lambda: harness.now
    )
    with harness.session() as session:
        writer.emit(
            session,
            correlation_id="r1",
            event_type=EventType.TOOL_CALL,
            payload={"subject_name": "Rivera Cohort", "notes": "sensitive detail"},
        )
        session.commit()
        event = session.exec(select(EvidenceEvent)).one()

    assert "Rivera Cohort" not in canonical_json(event.payload)
    assert set(event.payload) == {"subject_name", "notes"}
    assert event.payload["subject_name"]["sha256"]
    with harness.session() as session:
        assert verify_chain(session) == 1


# ── Approvals ────────────────────────────────────────────────────────────────


def test_approval_grant_and_hash_binding(harness):
    envelope = dp.envelope("draft_publication", purpose="DOSSIER_PREP", site="SG", subject="1")
    with harness.session() as session:
        evaluation = harness.decide_and_record(envelope, session)
        session.commit()

        approval = session.exec(select(Approval)).one()
        assert find_granted(session, evaluation.envelope_hash, now=harness.now) is None

        grant(
            session,
            approval,
            approver_name="reviewer@example.org",
            evidence=harness.evidence,
            correlation_id=evaluation.request_id,
            now=harness.now,
        )
        session.commit()

        found = find_granted(session, evaluation.envelope_hash, now=harness.now)
        assert found is not None and found.status is ApprovalStatus.APPROVED
        # A different envelope has a different hash and no approval.
        assert find_granted(session, "0" * 64, now=harness.now) is None


def test_an_approval_cannot_be_decided_twice(harness):
    envelope = dp.envelope("draft_publication", purpose="DOSSIER_PREP", site="SG", subject="1")
    with harness.session() as session:
        evaluation = harness.decide_and_record(envelope, session)
        session.commit()
        approval = session.exec(select(Approval)).one()
        reject(
            session,
            approval,
            approver_name="reviewer@example.org",
            evidence=harness.evidence,
            correlation_id=evaluation.request_id,
            now=harness.now,
        )
        session.commit()

        with pytest.raises(ApprovalError):
            grant(
                session,
                approval,
                approver_name="someone.else@example.org",
                evidence=harness.evidence,
                correlation_id=evaluation.request_id,
                now=harness.now,
            )


def test_expired_approvals_cannot_be_granted(harness):
    envelope = dp.envelope("draft_publication", purpose="DOSSIER_PREP", site="SG", subject="1")
    with harness.session() as session:
        evaluation = harness.decide_and_record(envelope, session)
        session.commit()
        approval = session.exec(select(Approval)).one()
        later = harness.now + timedelta(days=3)

        with pytest.raises(ApprovalError):
            grant(
                session,
                approval,
                approver_name="late.reviewer@example.org",
                evidence=harness.evidence,
                correlation_id=evaluation.request_id,
                now=later,
            )
        session.commit()
        assert session.exec(select(Approval)).one().status is ApprovalStatus.EXPIRED


def test_expire_stale_sweeps_lapsed_approvals(harness):
    envelope = dp.envelope("draft_publication", purpose="DOSSIER_PREP", site="SG", subject="1")
    with harness.session() as session:
        harness.decide_and_record(envelope, session)
        session.commit()
        assert expire_stale(session, now=harness.now) == 0
        assert expire_stale(session, now=harness.now + timedelta(days=3)) == 1


# ── Egress guard ─────────────────────────────────────────────────────────────


def test_redaction_round_trip_hides_content_from_the_provider(harness):
    """The provider must never see the sensitive spans; the caller must get them back."""
    guard = EgressGuard(
        redactor=PatternRedactor(extra_terms=["Rivera Cohort"]),
        evidence=harness.evidence,
    )
    seen: dict[str, str] = {}

    def provider(text: str) -> str:
        seen["text"] = text
        return f"Considered: {text}"

    secret = (
        "Rivera Cohort lead is ana@example.org, account SG-PB-100231, "
        "dataset ISIN HK0000521847."
    )
    with harness.session() as session:
        result = guard.send(
            secret,
            provider,
            classification=Classification.C2,
            destination="api.example-llm.com",
            destination_is_local=False,
            session=session,
            correlation_id="egress-1",
        )
        session.commit()

    assert result.decision is EgressDecision.REDACT
    for leaked in ("Rivera Cohort", "ana@example.org", "SG-PB-100231", "HK0000521847"):
        assert leaked not in seen["text"], f"{leaked} reached the provider"
    # The caller still gets a useful answer.
    assert "Rivera Cohort" in result.response
    assert "ana@example.org" in result.response

    with harness.session() as session:
        events = session.exec(
            select(EvidenceEvent).where(EvidenceEvent.correlation_id == "egress-1")
        ).all()
    kinds = {e.event_type for e in events}
    assert EventType.MODEL_CALL in kinds and EventType.EGRESS in kinds
    assert all(e.payload["manifest_hash"] for e in events)
    assert all(e.payload["redacted_spans"] == 4 for e in events)
    # Content is never copied into the evidence plane.
    assert all("ana@example.org" not in canonical_json(e.payload) for e in events)


def test_restricted_content_is_denied_outright(harness):
    guard = EgressGuard(evidence=harness.evidence)
    with harness.session() as session:
        with pytest.raises(EgressDenied):
            guard.send(
                "restricted detail",
                lambda t: t,
                classification=Classification.C3,
                destination="api.example-llm.com",
                destination_is_local=False,
                session=session,
                correlation_id="egress-deny",
            )
        session.commit()
        events = session.exec(
            select(EvidenceEvent).where(EvidenceEvent.correlation_id == "egress-deny")
        ).all()
    # A refusal is evidenced too — "we did not send it" is itself a fact to prove.
    assert events and all(e.payload["sent"] is False for e in events)


def test_local_destinations_skip_redaction():
    guard = EgressGuard(redactor=PatternRedactor())
    seen: dict[str, str] = {}
    result = guard.send(
        "ana@example.org",
        lambda t: seen.setdefault("text", t),
        classification=Classification.C3,
        destination="http://localhost:11434",
        destination_is_local=True,
    )
    assert result.decision is EgressDecision.ALLOW
    assert seen["text"] == "ana@example.org"


def test_egress_policy_is_configurable():
    strict = EgressPolicy(
        allow_plain_up_to=Classification.C0, allow_redacted_up_to=Classification.C1
    )
    assert strict.evaluate(Classification.C0, destination_is_local=False) is EgressDecision.ALLOW
    assert strict.evaluate(Classification.C1, destination_is_local=False) is EgressDecision.REDACT
    assert strict.evaluate(Classification.C2, destination_is_local=False) is EgressDecision.DENY


def test_null_redactor_is_a_no_op():
    redactor = NullRedactor()
    text, token_map = redactor.redact("ana@example.org")
    assert text == "ana@example.org" and token_map == {}


# ── Classification ───────────────────────────────────────────────────────────


def test_classifier_fails_upward():
    """Ambiguity must resolve to more sensitive, never less."""
    classifier = PatternClassifier()
    assert classifier.classify("Nothing notable here.") is Classification.C1
    assert classifier.classify("Reach me at ana@example.org") is Classification.C2
    assert classifier.classify("RESTRICTED — need-to-know") is Classification.C3


def test_a_hint_can_raise_but_never_lower():
    classifier = PatternClassifier()
    assert classifier.classify("public notice", hint=Classification.C3) is Classification.C3
    # An indicator still wins over a lower hint.
    assert (
        classifier.classify("ana@example.org", hint=Classification.C0)
        is Classification.C2
    )


# ── Tool runner ──────────────────────────────────────────────────────────────


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool("get_subject_profile", classification=Classification.C2)
    def _profile(envelope, evaluation, session):
        return {"subject": envelope.subject_id, "checks": len(evaluation.checks)}

    @registry.tool("check_dataset_eligibility", classification=Classification.C2)
    def _eligibility(envelope, evaluation, session):
        return {"verdicts": [c.to_dict() for c in evaluation.checks]}

    @registry.tool("draft_publication", classification=Classification.C2)
    def _draft(envelope, evaluation, session):
        return {"status": "DRAFT — not for release"}

    return registry


def test_denied_tool_never_executes(harness):
    """Fail closed means the body does not run, not that its output is filtered."""
    executed: list[str] = []
    registry = ToolRegistry()

    @registry.tool("check_dataset_eligibility")
    def _tool(envelope, evaluation, session):
        executed.append("ran")
        return {"secret": "should never be produced"}

    harness.registry = registry
    envelope = dp.envelope(
        "check_dataset_eligibility",
        purpose="ELIGIBILITY_CHECK",
        site="AE",
        subject="1",
        dataset="1",
    )
    with harness.session() as session:
        run = harness.run_tool(envelope, session)
        session.commit()

    assert run.outcome is Outcome.DENY
    assert run.data is None
    assert executed == [], "the tool body ran despite a denial"


def test_allowed_tool_emits_tool_call_then_egress(harness):
    """The ordering is the invariant: decision, execution, then both evidence events."""
    harness.registry = _registry()
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        run = harness.run_tool(envelope, session)
        session.commit()

        events = session.exec(
            select(EvidenceEvent)
            .where(EvidenceEvent.correlation_id == run.evaluation.request_id)
            .order_by(EvidenceEvent.sequence)
        ).all()

    assert run.outcome is Outcome.ALLOW
    assert run.data["subject"] == "2"
    assert [e.event_type for e in events] == [
        EventType.POLICY_DECISION,
        EventType.TOOL_CALL,
        EventType.EGRESS,
    ]
    egress = events[-1]
    assert egress.payload["bytes"] > 0
    assert "should never" not in canonical_json(egress.payload)


def test_human_approval_still_produces_a_draft(harness):
    """Review gates release, not preparation — the work is done, marked, and held."""
    harness.registry = _registry()
    envelope = dp.envelope("draft_publication", purpose="DOSSIER_PREP", site="SG", subject="1")
    with harness.session() as session:
        run = harness.run_tool(envelope, session)
        session.commit()

    assert run.outcome is Outcome.HUMAN_APPROVAL
    assert run.approval_required
    assert run.data["status"].startswith("DRAFT")


def test_unregistered_tool_is_rejected(harness):
    harness.registry = ToolRegistry()
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        with pytest.raises(ToolNotRegistered):
            harness.run_tool(envelope, session)


def test_a_forked_sequence_is_rejected_by_the_database(harness):
    """A second writer that computed the same sequence must fail loudly.

    The writer assigns sequence numbers by reading the current maximum, which is
    only fork-safe under a single writer. The UNIQUE constraint turns a silently
    forked chain — which verify_chain would only notice much later — into an
    immediate error.
    """
    import sqlalchemy

    from kognita.core.models import EvidenceEvent

    with harness.session() as session:
        harness.evidence.emit(
            session,
            correlation_id="first",
            event_type=EventType.TOOL_CALL,
            payload={},
        )
        session.commit()

    with harness.session() as session:
        session.add(
            EvidenceEvent(
                sequence=1,
                correlation_id="forged",
                event_type=EventType.TOOL_CALL,
            )
        )
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            session.commit()


def test_writers_over_one_engine_share_a_lock(harness):
    """A redacting writer and a plain one are ordinary; they must not interleave."""
    plain = EvidenceWriter(harness.engine)
    redacting = EvidenceWriter(harness.engine, redact_payload=hashes_only)
    assert plain._lock is redacting._lock

    with harness.session() as session:
        plain.emit(session, correlation_id="a", event_type=EventType.TOOL_CALL, payload={"v": 1})
        redacting.emit(session, correlation_id="b", event_type=EventType.TOOL_CALL, payload={"v": 2})
        session.commit()
        assert verify_chain(session) == 2
