"""Interaction record, origination, and RM review.

Roadmap 0.4 items 9 to 11. Decisions are offline: no network and no API key.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from sqlmodel import select

from fixtures import demo_pack as dp
from fixtures import governance_pack as gp

from kognita import confirm, mark
from kognita.approvals import ApprovalError, grant
from kognita.envelope import Check, Envelope, RuleContext
from kognita.evidence import verify_chain
from kognita.interaction import InteractionError, attach, open_interaction
from kognita.models import Approval, EvidenceEvent, Policy, RunRecord, UseCase
from kognita.origination import (
    OriginationError,
    contact_refusal,
    do_not_contact_reasons,
    originate,
    reverse_do_not_contact,
)
from kognita.policy import parse_rule
from kognita.reconstruct import reconstruct, render_reconstruction
from kognita.review import (
    RUBBER_STAMP_MIN_DECISIONS,
    ReviewError,
    confirm_review,
    define_human_boundary,
    review,
)
from kognita.rules import CORE_RULES
from kognita.tools import Run, continue_run, run_governed
from kognita.vocabulary import (
    ApprovalStatus,
    CheckResult,
    Classification,
    EventType,
    Outcome,
)
from kognita.testing.harness import FROZEN_NOW, Harness


def _seed(session):
    dp.seed_policies(session, now=FROZEN_NOW)
    from kognita.registry import register

    register(
        session, name="eligibility-assistant", owner_exec="Head of Data Governance"
    )


@pytest.fixture
def harness():
    return Harness(pack=dp.DemoPack(), purposes=dp.PURPOSES, seed=_seed)


def _gov_seed(session):
    gp.seed_policies(session, now=FROZEN_NOW)
    from kognita.registry import register

    register(session, name="marker-agent", owner_exec="Risk Team")


@pytest.fixture
def gov():
    h = Harness(pack=gp.GovernancePack(), purposes=gp.PURPOSES, seed=_gov_seed)

    def body(envelope, evaluation, session):
        return {"marked": True}

    h.registry.register("mark_criterion", body, classification=Classification.C1)
    return h


def _boundary(session, evidence, usecase="portfolio-review", seconds=0):
    define_human_boundary(
        session,
        usecase,
        intervention="before the note is sent",
        who="RM",
        must_review="the recommendation and the cited facts",
        evidence_seen="the rendered note and the policy checks",
        on_reject="the note is not sent",
        minimum_review_seconds=seconds,
        evidence=evidence,
        now=FROZEN_NOW,
    )


def _open(session, evidence, **kwargs):
    values = {
        "client_id": "client-1",
        "use_case": "portfolio-review",
        "approval_gate": "human_approval",
        "evidence": evidence,
        "now": FROZEN_NOW,
    }
    values.update(kwargs)
    return open_interaction(session, **values)


def _packet(**overrides):
    body = {
        "proposed_action": "send the banks note",
        "reason": "the house view changed and the client is exposed",
        "evidence": "house view 2026-10-01",
        "risks": "the client may already have acted",
        "alternatives": "wait until the next meeting",
        "policy_checks": [
            {
                "check": "SUITABILITY",
                "citation": "Suitability: the product is permitted",
            }
        ],
        "agent_confidence": 0.62,
    }
    body.update(overrides)
    return body


def _reason():
    return {
        "fact": "exposure",
        "detail": "The client holds 12 percent of the book in the banks the house view downgraded.",
        "citation": "house view 2026-10-01",
    }


def _events(session, event_type):
    return session.exec(
        select(EvidenceEvent).where(EvidenceEvent.event_type == event_type)
    ).all()


def test_mark_and_confirm_are_exported_from_kognita():
    import kognita
    from kognita.approvals import confirm as confirm_fn
    from kognita.approvals import mark as mark_fn

    assert kognita.mark is mark_fn
    assert kognita.confirm is confirm_fn
    assert mark is mark_fn
    assert confirm is confirm_fn
    assert "mark" in kognita.__all__
    assert "confirm" in kognita.__all__


def test_a_bare_score_is_refused_and_not_recorded(harness):
    with harness.session() as session:
        interaction = _open(session, harness.evidence)
        with pytest.raises(OriginationError, match="reasons, not scores"):
            originate(
                session,
                interaction.id,
                evidence=harness.evidence,
                trigger="RM request",
                candidate_set_size=40,
                criteria=["house-view change"],
                reasons=[
                    {
                        "fact": "relevance",
                        "detail": "Relevance score 89%",
                        "citation": "model",
                    }
                ],
                now=FROZEN_NOW,
            )
        session.commit()
        assert _events(session, EventType.ORIGINATION) == []


def test_origination_records_facts_and_reconstruct_answers_both_questions(harness):
    with harness.session() as session:
        interaction = _open(session, harness.evidence)
        payload = originate(
            session,
            interaction.id,
            evidence=harness.evidence,
            trigger="event",
            candidate_set_size=40,
            criteria=["house-view change", "existing exposure"],
            reasons=[_reason()],
            product_id="banks-note",
            now=FROZEN_NOW,
        )
        session.commit()
        report = reconstruct(session, interaction.id)
        document = render_reconstruction(report)

    assert payload["outcome"] == "recommend"
    client = _answer(report, "Why this client?")
    product = _answer(report, "Why this insight or product?")
    assert client != "not recorded"
    assert "12 percent" in document
    assert "house view 2026-10-01" in document
    assert product != "not recorded"
    assert "banks-note" in document
    assert _answer(report, "What was communicated to the client?") == "not recorded"
    assert report["chain"]["intact"] is True


def test_do_not_contact_is_honoured(harness):
    with harness.session() as session:
        interaction = _open(session, harness.evidence)
        _boundary(session, harness.evidence)
        payload = originate(
            session,
            interaction.id,
            evidence=harness.evidence,
            trigger="schedule",
            candidate_set_size=12,
            criteria=["recent rejection"],
            do_not_contact=["a recent rejection", "vulnerability"],
            now=FROZEN_NOW,
        )
        assert payload["outcome"] == "do not contact"
        assert do_not_contact_reasons(session, interaction.id) == (
            "a recent rejection",
            "vulnerability",
        )
        with pytest.raises(OriginationError, match="do not contact"):
            originate(
                session,
                interaction.id,
                evidence=harness.evidence,
                trigger="RM request",
                candidate_set_size=12,
                criteria=["house-view change"],
                reasons=[_reason()],
                now=FROZEN_NOW,
            )
        with pytest.raises(ReviewError, match="do not contact"):
            review(
                session,
                interaction.id,
                rm="rm-1",
                outcome="approve",
                packet=_packet(),
                rendered="Send the banks note.",
                provenance="AI",
                evidence=harness.evidence,
                requested_at=FROZEN_NOW - timedelta(minutes=5),
                now=FROZEN_NOW,
            )
        session.commit()
        assert len(_events(session, EventType.ORIGINATION)) == 1
        assert _events(session, EventType.FINAL_DECISION) == []


def test_contact_frequency_records_do_not_contact_and_cites_the_policy(harness):
    with harness.session() as session:
        session.add(
            Policy(
                regime="ORIGINATION",
                rule_type="CONTACT_FREQUENCY",
                rule={"max_contacts": 1, "window_days": 30},
                citation="Contact frequency: one recommendation in 30 days",
                effective_from=FROZEN_NOW - timedelta(days=1),
            )
        )
        session.flush()
        first = _open(session, harness.evidence, interaction_id="int-freq-1")
        second = _open(
            session,
            harness.evidence,
            interaction_id="int-freq-2",
            client_id="client-1",
        )
        originate(
            session,
            first.id,
            evidence=harness.evidence,
            trigger="event",
            candidate_set_size=4,
            criteria=["exposure"],
            reasons=[_reason()],
            now=FROZEN_NOW,
        )
        forced = originate(
            session,
            second.id,
            evidence=harness.evidence,
            trigger="event",
            candidate_set_size=4,
            criteria=["exposure"],
            reasons=[_reason()],
            now=FROZEN_NOW,
        )
        session.commit()

    assert forced["outcome"] == "do not contact"
    assert forced["do_not_contact_reasons"] == ["excessive frequency"]
    assert forced["reasons"] == []
    assert any(
        check["citation"] == "Contact frequency: one recommendation in 30 days"
        for check in forced["checks"]
    )


def test_an_incomplete_packet_cannot_be_approved(harness):
    with harness.session() as session:
        interaction = _open(session, harness.evidence)
        _boundary(session, harness.evidence)
        packet = _packet()
        del packet["alternatives"]
        with pytest.raises(ReviewError, match="incomplete approval packet"):
            review(
                session,
                interaction.id,
                rm="rm-1",
                outcome="approve",
                packet=packet,
                rendered="Send the banks note.",
                provenance="AI",
                evidence=harness.evidence,
                now=FROZEN_NOW,
            )
        rejected = review(
            session,
            interaction.id,
            rm="rm-1",
            outcome="reject",
            packet=packet,
            rendered="Send the banks note.",
            provenance="AI",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        session.commit()
        finals = _events(session, EventType.FINAL_DECISION)

    assert rejected["outcome"] == "reject"
    assert len(finals) == 1
    assert finals[0].payload["outcome"] == "reject"


def test_review_keeps_the_provenance_label_and_the_diff(harness):
    with harness.session() as session:
        interaction = _open(session, harness.evidence)
        _boundary(session, harness.evidence)
        result = review(
            session,
            interaction.id,
            rm="rm-1",
            outcome="modify",
            packet=_packet(),
            rendered="The house view downgraded banks.",
            approved="The house view downgraded banks. I removed the product link.",
            provenance="AI",
            evidence=harness.evidence,
            requested_at=FROZEN_NOW - timedelta(minutes=10),
            now=FROZEN_NOW,
        )
        session.commit()
        report = reconstruct(session, interaction.id)
        document = render_reconstruction(report)

    assert result["provenance"] == "AI then human-edited"
    assert result["prior_provenance"] == "AI"
    assert result["diff"]
    assert result["final"] is True
    assert _answer(report, "What did the RM see and change?") != "not recorded"
    assert "AI then human-edited" in document
    assert _answer(report, "Who made the final decision?") != "not recorded"
    assert "rm-1 decided modify" in document
    assert _answer(report, "What was communicated to the client?") == "not recorded"


def test_two_signature_review_waits_for_a_different_confirmer(harness):
    with harness.session() as session:
        interaction = _open(session, harness.evidence, approval_gate="two_signature")
        _boundary(session, harness.evidence)
        marked = review(
            session,
            interaction.id,
            rm="rm-1",
            outcome="approve",
            packet=_packet(),
            rendered="Send the banks note.",
            provenance="human with AI assistance",
            evidence=harness.evidence,
            requested_at=FROZEN_NOW - timedelta(minutes=10),
            now=FROZEN_NOW,
        )
        assert marked["final"] is False
        assert _events(session, EventType.FINAL_DECISION) == []
        with pytest.raises(ReviewError, match="confirming party"):
            confirm_review(
                session,
                interaction.id,
                confirmer="rm-1",
                evidence=harness.evidence,
                now=FROZEN_NOW,
            )
        confirmed = confirm_review(
            session,
            interaction.id,
            confirmer="ada-owner",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        session.commit()
        report = reconstruct(session, interaction.id)

    assert confirmed["final"] is True
    assert confirmed["confirmer"] == "ada-owner"
    decided = _answer(report, "Who made the final decision?")
    assert decided["decisions"][0]["outcome"] == "approve"
    assert decided["decisions"][0]["actor_id"] == "ada-owner"


def test_a_fast_approval_is_flagged_to_the_accountable_owner(harness):
    with harness.session() as session:
        session.add(
            UseCase(
                usecase_id="portfolio-review",
                version=1,
                record={
                    "usecase_id": "portfolio-review",
                    "version": 1,
                    "business_owner": "Ada Owner",
                    "approval_gate": "human_approval",
                },
                effective_from=FROZEN_NOW - timedelta(days=1),
            )
        )
        session.flush()
        interaction = _open(session, harness.evidence, approval_gate="none")
        assert interaction.approval_gate == "human_approval"
        _boundary(session, harness.evidence, seconds=120)
        result = review(
            session,
            interaction.id,
            rm="rm-1",
            outcome="approve",
            packet=_packet(),
            rendered="Send the banks note.",
            provenance="human",
            evidence=harness.evidence,
            requested_at=FROZEN_NOW - timedelta(seconds=1),
            now=FROZEN_NOW,
        )
        session.commit()

    stamp = result["rubber_stamp"]
    assert stamp["flagged"] is True
    assert "faster than the review threshold" in stamp["reasons"]
    assert stamp["accountable_owner"] == "Ada Owner"


def test_an_approver_who_approves_everything_is_flagged(harness):
    with harness.session() as session:
        _boundary(session, harness.evidence, seconds=0)
        last = None
        for index in range(RUBBER_STAMP_MIN_DECISIONS):
            interaction = _open(
                session,
                harness.evidence,
                interaction_id=f"int-rate-{index}",
                client_id=f"client-{index}",
            )
            last = review(
                session,
                interaction.id,
                rm="rm-1",
                outcome="approve",
                packet=_packet(),
                rendered="Send the banks note.",
                provenance="human",
                evidence=harness.evidence,
                requested_at=FROZEN_NOW - timedelta(hours=1),
                now=FROZEN_NOW,
            )
        session.commit()

    assert last is not None
    assert last["rubber_stamp"]["decisions"] == RUBBER_STAMP_MIN_DECISIONS
    assert last["rubber_stamp"]["approval_rate"] == 1
    assert last["rubber_stamp"]["flagged"] is True
    assert "approval rate" in last["rubber_stamp"]["reasons"]


def test_final_decision_on_allow_and_deny_and_across_correlation_ids(harness):
    registry, executed = _registry()
    with harness.session() as session:
        interaction = _open(session, harness.evidence, use_case="", approval_gate="")
        allowed = run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            interaction_id=interaction.id,
        )
        denied = run_governed(
            session,
            dp.envelope(
                "get_subject_profile", purpose="NOT_A_PURPOSE", site="SG", subject="2"
            ),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            interaction_id=interaction.id,
        )
        session.commit()
        report = reconstruct(session, interaction.id)
        plain = run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
        )
        session.commit()
        plain_report = reconstruct(session, plain.evaluation.request_id)
        assert verify_chain(session) > 0

    assert allowed.evaluation.outcome is Outcome.ALLOW
    assert denied.evaluation.outcome is Outcome.DENY
    assert executed == ["profile", "profile"]
    outcomes = [
        item["outcome"]
        for item in _answer(report, "Who made the final decision?")["decisions"]
    ]
    assert outcomes == ["allow", "deny"]
    authority = _answer(report, "What was the agent authorized to do?")
    assert len(authority["decisions"]) == 2
    assert _answer(plain_report, "Who made the final decision?") == "not recorded"


def test_continue_run_confirms_only_a_marked_two_signature_approval(gov):
    envelope = gp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="marker_alice",
        criterion="crit_1",
        principal="marker_alice",
        agent="marker-agent",
    )
    run = Run()
    with gov.session() as session:
        held = run_governed(
            session,
            envelope,
            registry=gov.registry,
            evidence=gov.evidence,
            pack=gov.pack,
            purposes=gp.PURPOSES,
            as_of=FROZEN_NOW,
            run=run,
            now=FROZEN_NOW,
        )
        session.commit()
        approval = session.exec(select(Approval)).one()
        assert held.data is None
        assert approval.confirmation_status is ApprovalStatus.PENDING
        with pytest.raises(ApprovalError, match="needs mark"):
            grant(
                session,
                approval,
                approver_name="owner_bob",
                evidence=gov.evidence,
                correlation_id=held.evaluation.request_id,
                now=FROZEN_NOW,
            )
        with pytest.raises(ApprovalError, match="needs mark"):
            continue_run(
                session,
                run.id,
                {approval.id: True},
                evidence=gov.evidence,
                registry=gov.registry,
                approver_name="owner_bob",
                now=FROZEN_NOW,
            )
        session.commit()
        stored = session.get(RunRecord, run.id)
        assert stored is not None
        assert stored.continuation_hash
        assert session.get(Approval, approval.id).status is ApprovalStatus.PENDING

        mark(
            session,
            approval,
            approver_id="reviewer_rm",
            evidence=gov.evidence,
            correlation_id=held.evaluation.request_id,
            now=FROZEN_NOW,
        )
        with pytest.raises(ApprovalError, match="confirming party"):
            continue_run(
                session,
                run.id,
                {approval.id: True},
                evidence=gov.evidence,
                registry=gov.registry,
                approver_name="reviewer_rm",
                now=FROZEN_NOW,
            )
        session.commit()
        assert session.get(Approval, approval.id).status is ApprovalStatus.MARKED
        assert session.get(RunRecord, run.id).continuation_hash

        resumed = continue_run(
            session,
            run.id,
            {approval.id: True},
            evidence=gov.evidence,
            registry=gov.registry,
            approver_name="owner_bob",
            now=FROZEN_NOW,
        )
        session.commit()
        actions = [
            event.payload.get("action")
            for event in session.exec(
                select(EvidenceEvent)
                .where(EvidenceEvent.correlation_id == held.evaluation.request_id)
                .order_by(EvidenceEvent.sequence)
            ).all()
            if event.event_type is EventType.APPROVAL
        ]

    assert resumed.data == {"marked": True}
    assert "MARKED" in actions
    assert "CONFIRMED" in actions
    assert (
        actions.index("MARKED") < actions.index("CONFIRMED") < actions.index("RESUMED")
    )


def test_continue_run_rejects_a_marked_approval_without_executing(gov):
    envelope = gp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="marker_alice",
        criterion="crit_1",
        principal="marker_alice",
        agent="marker-agent",
    )
    run = Run()
    with gov.session() as session:
        held = run_governed(
            session,
            envelope,
            registry=gov.registry,
            evidence=gov.evidence,
            pack=gov.pack,
            purposes=gp.PURPOSES,
            as_of=FROZEN_NOW,
            run=run,
            now=FROZEN_NOW,
        )
        approval = session.exec(select(Approval)).one()
        mark(
            session,
            approval,
            approver_id="reviewer_rm",
            evidence=gov.evidence,
            correlation_id=held.evaluation.request_id,
            now=FROZEN_NOW,
        )
        resumed = continue_run(
            session,
            run.id,
            {approval.id: False},
            evidence=gov.evidence,
            registry=gov.registry,
            approver_name="owner_bob",
            now=FROZEN_NOW,
        )
        session.commit()
        stored = session.get(Approval, approval.id)

    assert resumed.data is None
    assert stored.status is ApprovalStatus.REJECTED


def test_do_not_contact_blocks_later_client_contact_until_reversal(harness):
    executed: list[str] = []

    def _body(envelope, evaluation, session):
        executed.append(envelope.tool)
        return {"ok": True}

    harness.registry.register("send_note", _body, classification=Classification.C1)
    harness.registry.register(
        "get_subject_profile", _body, classification=Classification.C2
    )
    with harness.session() as session:
        session.add(
            Policy(
                regime="ORIGINATION",
                rule_type="CONTACT_FREQUENCY",
                rule={"max_contacts": 5, "window_days": 30, "tools": ["send_note"]},
                citation="Contact frequency: send_note contacts the client",
                effective_from=FROZEN_NOW - timedelta(days=1),
            )
        )
        session.add(
            UseCase(
                usecase_id="portfolio-review",
                version=1,
                record={
                    "usecase_id": "portfolio-review",
                    "version": 1,
                    "business_owner": "Ada Owner",
                    "approval_gate": "human_approval",
                },
                effective_from=FROZEN_NOW - timedelta(days=1),
            )
        )
        session.flush()
        _boundary(session, harness.evidence)
        barred = _open(session, harness.evidence, interaction_id="int-barred")
        same = _open(session, harness.evidence, interaction_id="int-same")
        other = _open(
            session,
            harness.evidence,
            interaction_id="int-other",
            client_id="client-2",
        )
        unnamed = _open(
            session,
            harness.evidence,
            interaction_id="int-unnamed",
            client_id="client-3",
            use_case="",
            approval_gate="human_approval",
        )
        originate(
            session,
            barred.id,
            evidence=harness.evidence,
            trigger="schedule",
            candidate_set_size=4,
            criteria=["vulnerability"],
            do_not_contact=["vulnerability"],
            now=FROZEN_NOW,
        )
        originate(
            session,
            unnamed.id,
            evidence=harness.evidence,
            trigger="schedule",
            candidate_set_size=1,
            criteria=["poor timing"],
            do_not_contact=["poor timing"],
            now=FROZEN_NOW,
        )
        with pytest.raises(OriginationError, match="do not contact"):
            originate(
                session,
                same.id,
                evidence=harness.evidence,
                trigger="RM request",
                candidate_set_size=4,
                criteria=["exposure"],
                reasons=[_reason()],
                now=FROZEN_NOW,
            )
        other_recommendation = originate(
            session,
            other.id,
            evidence=harness.evidence,
            trigger="event",
            candidate_set_size=2,
            criteria=["exposure"],
            reasons=[_reason()],
            now=FROZEN_NOW,
        )
        with pytest.raises(ReviewError, match="do not contact"):
            review(
                session,
                same.id,
                rm="rm-1",
                outcome="approve",
                packet=_packet(),
                rendered="Send the banks note.",
                provenance="human",
                evidence=harness.evidence,
                requested_at=FROZEN_NOW - timedelta(minutes=5),
                now=FROZEN_NOW,
            )
        rejected = review(
            session,
            same.id,
            rm="rm-1",
            outcome="reject",
            packet=_packet(),
            rendered="Send the banks note.",
            provenance="human",
            evidence=harness.evidence,
            requested_at=FROZEN_NOW - timedelta(minutes=5),
            now=FROZEN_NOW,
        )
        denied = run_governed(
            session,
            Envelope(
                principal="test-principal",
                purpose="COLLABORATION",
                tool="send_note",
                actor_location="SG",
            ),
            registry=harness.registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            interaction_id=barred.id,
        )
        profile = run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=harness.registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            interaction_id=barred.id,
        )
        other_send = run_governed(
            session,
            Envelope(
                principal="test-principal",
                purpose="COLLABORATION",
                tool="send_note",
                actor_location="SG",
            ),
            registry=harness.registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            interaction_id=other.id,
        )
        unknown = run_governed(
            session,
            Envelope(
                principal="test-principal",
                purpose="COLLABORATION",
                tool="send_note",
                actor_location="SG",
            ),
            registry=harness.registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
        )
        conflict = contact_refusal(
            session,
            Envelope(
                principal="test-principal",
                purpose="COLLABORATION",
                tool="send_note",
                subject_type="client",
                subject_id="client-2",
            ),
            interaction_id=barred.id,
            now=FROZEN_NOW,
        )
        with pytest.raises(OriginationError, match="cites the basis"):
            reverse_do_not_contact(
                session,
                barred.id,
                evidence=harness.evidence,
                human="RM",
                citation="  ",
                now=FROZEN_NOW,
            )
        with pytest.raises(OriginationError, match="not an authorised human"):
            reverse_do_not_contact(
                session,
                barred.id,
                evidence=harness.evidence,
                human="stranger",
                citation="RM instruction 2026-06-01",
                now=FROZEN_NOW,
            )
        with pytest.raises(OriginationError, match="registered agent"):
            reverse_do_not_contact(
                session,
                barred.id,
                evidence=harness.evidence,
                human="eligibility-assistant",
                citation="RM instruction 2026-06-01",
                now=FROZEN_NOW,
            )
        with pytest.raises(OriginationError, match="no authorised human"):
            reverse_do_not_contact(
                session,
                unnamed.id,
                evidence=harness.evidence,
                human="RM",
                citation="RM instruction 2026-06-01",
                now=FROZEN_NOW,
            )
        assert do_not_contact_reasons(session, same.id) == ("vulnerability",)
        lifted = reverse_do_not_contact(
            session,
            barred.id,
            evidence=harness.evidence,
            human="Ada Owner",
            citation="RM instruction 2026-06-01",
            now=FROZEN_NOW,
        )
        assert do_not_contact_reasons(session, same.id) == ()
        resumed_recommendation = originate(
            session,
            same.id,
            evidence=harness.evidence,
            trigger="RM request",
            candidate_set_size=4,
            criteria=["exposure"],
            reasons=[_reason()],
            now=FROZEN_NOW,
        )
        sent = run_governed(
            session,
            Envelope(
                principal="test-principal",
                purpose="COLLABORATION",
                tool="send_note",
                actor_location="SG",
            ),
            registry=harness.registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            interaction_id=barred.id,
        )
        session.commit()

    assert other_recommendation["outcome"] == "recommend"
    assert rejected["outcome"] == "reject"
    assert denied.evaluation.outcome is Outcome.DENY
    assert denied.data is None
    assert any(
        "vulnerability" in check.citation for check in denied.evaluation.failures()
    )
    assert any("send_note" in check.citation for check in denied.evaluation.failures())
    assert profile.evaluation.outcome is Outcome.ALLOW
    assert profile.data == {"ok": True}
    assert other_send.evaluation.outcome is Outcome.ALLOW
    assert unknown.evaluation.outcome is Outcome.DENY
    assert any(
        "could not be determined" in check.citation
        for check in unknown.evaluation.failures()
    )
    assert conflict and "could not be determined" in conflict[0].citation
    assert lifted["outcome"] == "reversal"
    assert lifted["citation"] == "RM instruction 2026-06-01"
    assert resumed_recommendation["outcome"] == "recommend"
    assert sent.evaluation.outcome is Outcome.ALLOW
    assert executed == ["get_subject_profile", "send_note", "send_note"]


def test_continue_run_leaves_a_barred_contact_held(harness):
    executed: list[str] = []

    def _body(envelope, evaluation, session):
        executed.append(envelope.tool)
        return {"ok": True}

    harness.registry.register("send_note", _body, classification=Classification.C1)
    run = Run()
    with harness.session() as session:
        session.add(
            Policy(
                regime="ORIGINATION",
                rule_type="CONTACT_FREQUENCY",
                rule={"max_contacts": 5, "window_days": 30, "tools": ["send_note"]},
                citation="Contact frequency: send_note contacts the client",
                effective_from=FROZEN_NOW - timedelta(days=1),
            )
        )
        session.flush()
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence, interaction_id="int-held")
        held = run_governed(
            session,
            Envelope(
                principal="test-principal",
                purpose="COLLABORATION",
                tool="send_note",
                actor_location="SG",
            ),
            registry=harness.registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            run=run,
            interaction_id=interaction.id,
            extra_checks=[
                Check(
                    check="HOLD",
                    regime="INTERNAL",
                    result=CheckResult.REQUIRES_HUMAN,
                    citation="test hold",
                )
            ],
        )
        approval = session.exec(select(Approval)).one()
        assert held.data is None
        assert approval.status is ApprovalStatus.PENDING
        originate(
            session,
            interaction.id,
            evidence=harness.evidence,
            trigger="schedule",
            candidate_set_size=2,
            criteria=["vulnerability"],
            do_not_contact=["vulnerability"],
            now=FROZEN_NOW,
        )
        with pytest.raises(OriginationError, match="do not contact"):
            continue_run(
                session,
                run.id,
                {approval.id: True},
                evidence=harness.evidence,
                registry=harness.registry,
                approver_name="RM",
                now=FROZEN_NOW,
            )
        session.commit()
        stored = session.get(RunRecord, run.id)
        assert stored is not None and stored.continuation_hash
        assert session.get(Approval, approval.id).status is ApprovalStatus.PENDING
        assert executed == []
        reverse_do_not_contact(
            session,
            interaction.id,
            evidence=harness.evidence,
            human="RM",
            citation="The client asked to be contacted",
            now=FROZEN_NOW,
        )
        resumed = continue_run(
            session,
            run.id,
            {approval.id: True},
            evidence=harness.evidence,
            registry=harness.registry,
            approver_name="RM",
            now=FROZEN_NOW,
        )
        session.commit()

    assert resumed.data == {"ok": True}
    assert executed == ["send_note"]


def test_continue_run_confirmer_must_be_a_different_party(gov):
    envelope = gp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="marker_alice",
        criterion="crit_1",
        principal="marker_alice",
        agent="marker-agent",
    )
    run = Run()
    with gov.session() as session:
        held = run_governed(
            session,
            envelope,
            registry=gov.registry,
            evidence=gov.evidence,
            pack=gov.pack,
            purposes=gp.PURPOSES,
            as_of=FROZEN_NOW,
            run=run,
            now=FROZEN_NOW,
        )
        approval = session.exec(select(Approval)).one()
        mark(
            session,
            approval,
            approver_id="reviewer_rm",
            evidence=gov.evidence,
            correlation_id=held.evaluation.request_id,
            now=FROZEN_NOW,
        )
        with pytest.raises(ApprovalError, match="confirming party is missing"):
            continue_run(
                session,
                run.id,
                {approval.id: True},
                evidence=gov.evidence,
                registry=gov.registry,
                approver_name="owner_bob",
                confirmer="",
                now=FROZEN_NOW,
            )
        with pytest.raises(ApprovalError, match="confirming party is missing"):
            continue_run(
                session,
                run.id,
                {approval.id: True},
                evidence=gov.evidence,
                registry=gov.registry,
                approver_name="",
                now=FROZEN_NOW,
            )
        with pytest.raises(ApprovalError, match="must differ"):
            continue_run(
                session,
                run.id,
                {approval.id: True},
                evidence=gov.evidence,
                registry=gov.registry,
                approver_name="owner_bob",
                confirmer="reviewer_rm",
                now=FROZEN_NOW,
            )
        with pytest.raises(ApprovalError, match="must differ"):
            continue_run(
                session,
                run.id,
                {approval.id: True},
                evidence=gov.evidence,
                registry=gov.registry,
                confirmer="marker_alice",
                now=FROZEN_NOW,
            )
        session.commit()
        assert session.get(Approval, approval.id).status is ApprovalStatus.MARKED
        assert session.get(RunRecord, run.id).continuation_hash
        resumed = continue_run(
            session,
            run.id,
            {approval.id: True},
            evidence=gov.evidence,
            registry=gov.registry,
            approver_name="reviewer_rm",
            confirmer="owner_bob",
            now=FROZEN_NOW,
        )
        session.commit()
        confirmed = session.exec(
            select(EvidenceEvent)
            .where(EvidenceEvent.event_type == EventType.APPROVAL)
            .where(EvidenceEvent.actor_id == "owner_bob")
        ).all()

    assert resumed.data == {"marked": True}
    assert any(event.payload.get("action") == "CONFIRMED" for event in confirmed)


def test_unknown_interaction_and_unknown_gate_fail_closed(harness):
    with harness.session() as session:
        with pytest.raises(InteractionError, match="unknown interaction"):
            attach(
                session,
                "missing",
                evidence=harness.evidence,
                correlation_id="req-1",
            )
        with pytest.raises(InteractionError, match="approval gate"):
            _open(session, harness.evidence, approval_gate="rubber-stamp")


def test_contact_frequency_rule_fails_closed_without_a_count():
    policy = SimpleNamespace(
        id=1,
        regime="ORIGINATION",
        rule_type="CONTACT_FREQUENCY",
        citation="Contact frequency limit",
        rule={"max_contacts": 2, "window_days": 30, "tools": ["send_note"]},
    )
    missing = RuleContext(
        envelope=Envelope(principal="a", purpose="p", tool="send_note"),
        attributes={},
    )
    checks = CORE_RULES["CONTACT_FREQUENCY"](policy, missing)
    assert checks[0].result is CheckResult.FAIL
    assert "could not be evaluated" in checks[0].citation
    other = RuleContext(
        envelope=Envelope(principal="a", purpose="p", tool="read_note"),
        attributes={},
    )
    assert CORE_RULES["CONTACT_FREQUENCY"](policy, other) == []
    over = RuleContext(
        envelope=Envelope(principal="a", purpose="p", tool="send_note"),
        attributes={"contact_count": 2},
    )
    refused = CORE_RULES["CONTACT_FREQUENCY"](policy, over)
    assert refused[0].result is CheckResult.FAIL
    assert "Contact frequency limit" in refused[0].citation


def test_policy_language_accepts_a_contact_frequency_rule():
    rule, findings = parse_rule(
        {
            "id": "contact-frequency",
            "regime": "ORIGINATION",
            "rule_type": "CONTACT_FREQUENCY",
            "citation": "Contact frequency: one recommendation in 30 days",
            "effective_from": "2026-01-01T00:00:00+00:00",
            "rule": {"max_contacts": 1, "window_days": 30},
        },
        source="contact.yaml",
    )
    assert findings == []
    assert rule is not None
    broken, problems = parse_rule(
        {
            "id": "contact-frequency",
            "regime": "ORIGINATION",
            "rule_type": "CONTACT_FREQUENCY",
            "citation": "Contact frequency: one recommendation in 30 days",
            "effective_from": "2026-01-01T00:00:00+00:00",
            "rule": {"max_contacts": "many"},
        },
        source="contact.yaml",
    )
    assert broken is None
    assert any("max_contacts" in item.message for item in problems)
    refused, type_problems = parse_rule([], source="contact.yaml")
    assert refused is None
    assert any("must be a mapping" in item.message for item in type_problems)


def _registry():
    executed: list[str] = []
    from kognita.tools import ToolRegistry

    registry = ToolRegistry()

    @registry.tool("get_subject_profile", classification=Classification.C2)
    def _profile(envelope, evaluation, session):
        executed.append("profile")
        return {"subject": envelope.subject_id}

    return registry, executed


def _answer(report, question):
    for item in report["questions"]:
        if item["question"] == question:
            return item["answer"]
    raise AssertionError(question)
