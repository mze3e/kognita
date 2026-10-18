"""Governed client communication, live-run intervention, and complaints.

Roadmap 0.4 item 12, the intervention controls on a live run, and the
complaint link. Decisions are offline: no network and no API key.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlmodel import select

from fixtures import demo_pack as dp
from fixtures import governance_pack as gp
from test_usecases import _activate, _open as open_register, _spec, _tier3

from kognita.communication import (
    CommunicationError,
    record_complaint,
    register_channel,
    send_client_communication,
    send_public_content,
)
from kognita.egress import EgressGuard
from kognita.evidence import verify_chain
from kognita.interaction import open_interaction
from kognita.intervention import (
    InterventionError,
    observe_run,
    override_run,
    pause_run,
    recover_run,
    register_compensating_action,
    restrict_run,
    revoke_run,
)
from kognita.models import (
    Approval,
    ApprovedChannel,
    EvidenceEvent,
    Policy,
    RunRecord,
)
from kognita.origination import originate
from kognita.outcomes import OutcomeError
from kognita.policy import parse_rule
from kognita.reconstruct import QUESTIONS, reconstruct, render_reconstruction
from kognita.registry import get as get_agent
from kognita.retrieval import index_item, retrieve
from kognita.review import (
    CITE_HOLD,
    confirm_review,
    define_human_boundary,
    enqueue_for_review,
    record_final_decision,
    record_review_finding,
    review,
)
from kognita.tools import Run, ToolRegistry, continue_run, run_governed
from kognita.vocabulary import (
    ApprovalStatus,
    ActorType,
    CheckResult,
    Classification,
    EventType,
    Outcome,
)
from kognita.testing.harness import FROZEN_NOW, Harness

NOTE = "The house view changed. Capital is at risk."
PROMPT = "Summarise the eligibility note."
ANSWER = "Eligible under the recorded checks."
ITEM_BODY = "cross-site eligibility for restricted datasets"


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


def _boundary(session, evidence, usecase="portfolio-review"):
    define_human_boundary(
        session,
        usecase,
        intervention="before the note is sent",
        who="RM",
        must_review="the recommendation and the cited facts",
        evidence_seen="the rendered note and the policy checks",
        on_reject="the note is not sent",
        minimum_review_seconds=0,
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
            {"check": "SUITABILITY", "citation": "Suitability: the product is permitted"}
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


def _content_policy(session, *, lanes=("client",), **rule):
    body = {
        "levels": [
            "general information",
            "house view",
            "contextualised insight",
            "investment discussion",
            "product recommendation",
            "transaction",
        ],
        "lanes": list(lanes),
        "prohibited_claims": ["guaranteed"],
    }
    body.update(rule)
    session.add(
        Policy(
            regime="COMMUNICATION",
            rule_type="CONTENT_POLICY",
            rule=body,
            citation="Content policy: prohibited claims and the constraints named here",
            effective_from=FROZEN_NOW - timedelta(days=1),
        )
    )
    session.flush()


def _disclosure(session):
    session.add(
        Policy(
            regime="COMMUNICATION",
            rule_type="AI_DISCLOSURE",
            rule={
                "triggers": [
                    "AI generated content",
                    "assisted analysis",
                    "powers a chatbot",
                    "materially contributed to a decision",
                ],
                "jurisdictions": ["SG"],
                "text": "This message used AI.",
            },
            citation="AI disclosure: tell the client when AI was involved",
            effective_from=FROZEN_NOW - timedelta(days=1),
        )
    )
    session.flush()


def _channel(session, evidence, name="Secure Message", **kwargs):
    values = {
        "archiving": True,
        "supervision": True,
        "human": "RM",
        "citation": "Approved channel: archiving and supervision",
        "now": FROZEN_NOW,
    }
    values.update(kwargs)
    return register_channel(session, name, evidence=evidence, **values)


def _allow(session, evidence, interaction_id):
    record_final_decision(
        session,
        interaction_id,
        outcome="allow",
        evidence=evidence,
        actor_id="RM",
        citation="Final decision: the call was allowed",
        now=FROZEN_NOW,
    )


def _send(session, evidence, interaction_id, **kwargs):
    values = {
        "channel": "Secure Message",
        "recipient_ref": "client-1-ref",
        "content": "Market hours are 9 to 5.",
        "level": "general information",
        "provenance": "human",
        "actor_id": "RM",
        "now": FROZEN_NOW,
    }
    values.update(kwargs)
    return send_client_communication(
        session, interaction_id, evidence=evidence, **values
    )


def _answer(report, question):
    found = [item for item in report["questions"] if item["question"] == question]
    assert len(found) == 1
    return found[0]["answer"]


def _live(session, evidence, spec, **kwargs):
    open_register(session, evidence)
    _activate(session, evidence, spec, **kwargs)


def _registry():
    executed: list[str] = []
    registry = ToolRegistry()

    @registry.tool("get_subject_profile", classification=Classification.C2)
    def _profile(envelope, evaluation, session):
        executed.append(envelope.tool)
        return {"subject": envelope.subject_id}

    @registry.tool("lookup_note", classification=Classification.C1)
    def _note(envelope, evaluation, session):
        executed.append(envelope.tool)
        return {"note": True}

    return registry, executed


def _rule(**overrides):
    body = {
        "id": "content.claims",
        "regime": "COMMUNICATION",
        "rule_type": "CONTENT_POLICY",
        "citation": "Content policy: no guaranteed claims",
        "effective_from": "2026-06-01T00:00:00+00:00",
        "rule": {"prohibited_claims": ["guaranteed"]},
    }
    body.update(overrides)
    return body


def test_content_policy_and_disclosure_rules_parse():
    document, findings = parse_rule(_rule(), source="content.yaml")
    assert findings == []
    assert document is not None
    assert document.rule_type == "CONTENT_POLICY"
    disclosure, disclosure_findings = parse_rule(
        {
            "id": "disclosure.ai",
            "regime": "COMMUNICATION",
            "rule_type": "AI_DISCLOSURE",
            "citation": "AI disclosure: tell the client",
            "effective_from": "2026-06-01T00:00:00+00:00",
            "rule": {
                "triggers": ["AI generated content"],
                "text": "This message used AI.",
            },
        },
        source="disclosure.yaml",
    )
    assert disclosure_findings == []
    assert disclosure is not None
    empty, empty_findings = parse_rule(
        _rule(id="content.empty", rule={}), source="empty.yaml"
    )
    assert empty is None
    assert any("at least one constraint" in item.message for item in empty_findings)
    unknown, unknown_findings = parse_rule(
        _rule(rule={"prohibited_claims": ["guaranteed"], "levels": ["broadcast"]}),
        source="level.yaml",
    )
    assert unknown is None
    assert any("unknown communication level" in item.message for item in unknown_findings)
    missing, missing_findings = parse_rule(
        {
            "id": "disclosure.blank",
            "regime": "COMMUNICATION",
            "rule_type": "AI_DISCLOSURE",
            "citation": "AI disclosure: tell the client",
            "effective_from": "2026-06-01T00:00:00+00:00",
            "rule": {"triggers": ["AI generated content"]},
        },
        source="disclosure.yaml",
    )
    assert missing is None
    assert any("text must be a non-empty string" in item.message for item in missing_findings)


def test_personal_and_unsupervised_channels_are_not_registered(harness):
    with harness.session() as session:
        for name in ("Personal WhatsApp", "Telegram", "personal email"):
            with pytest.raises(CommunicationError, match="not registrable"):
                _channel(session, harness.evidence, name)
        with pytest.raises(CommunicationError, match="public-lane"):
            _channel(session, harness.evidence, "LinkedIn")
        with pytest.raises(CommunicationError, match="archiving and supervision"):
            _channel(session, harness.evidence, "Secure Message", archiving=False)
        with pytest.raises(CommunicationError, match="registered agent"):
            _channel(session, harness.evidence, "Secure Message", human="eligibility-assistant")
        session.commit()
        assert session.exec(select(ApprovedChannel)).all() == []
        assert _events(session, EventType.POLICY_CHANGE) == []


def test_a_general_information_send_carries_level_and_provenance(harness):
    with harness.session() as session:
        interaction = _open(session, harness.evidence)
        _content_policy(session)
        _channel(session, harness.evidence)
        _allow(session, harness.evidence, interaction.id)
        sent = _send(session, harness.evidence, interaction.id)
        session.commit()
        report = reconstruct(session, interaction.id)
        assert verify_chain(session) > 0

    assert sent["outcome"] == "ALLOW"
    assert sent["sent"] is True
    assert sent["communication_level"] == "general information"
    assert sent["provenance"] == "human"
    assert sent["content_hash"]
    assert all(check["citation"] for check in sent["checks"])
    answer = _answer(report, "What was communicated to the client?")
    assert answer["communications"][0]["communication_level"] == "general information"
    assert answer["communications"][0]["provenance"] == "human"
    assert answer["communications"][0]["content_status"] == "retained"
    document = render_reconstruction(report)
    assert "Communication level: general information." in document
    assert "Content provenance: human." in document


def test_an_unregistered_channel_is_denied_and_still_recorded(harness):
    with harness.session() as session:
        interaction = _open(session, harness.evidence)
        _content_policy(session)
        _allow(session, harness.evidence, interaction.id)
        denied = _send(session, harness.evidence, interaction.id, channel="unknown desk")
        with pytest.raises(CommunicationError, match="not a recorded level"):
            _send(session, harness.evidence, interaction.id, level="broadcast")
        session.commit()
        events = _events(session, EventType.CLIENT_COMMUNICATION)

    assert denied["outcome"] == "DENY"
    assert denied["sent"] is False
    assert "not a registered channel" in denied["citation"]
    assert len(events) == 1
    assert events[0].payload["content_hash"] == denied["content_hash"]


def test_missing_final_decision_house_view_and_do_not_contact_deny(harness):
    with harness.session() as session:
        missing = _open(session, harness.evidence, interaction_id="int-missing")
        house = _open(session, harness.evidence, interaction_id="int-house")
        barred = _open(session, harness.evidence, interaction_id="int-barred")
        _content_policy(session)
        _channel(session, harness.evidence)
        no_decision = _send(session, harness.evidence, missing.id)
        _allow(session, harness.evidence, house.id)
        no_attribution = _send(
            session, harness.evidence, house.id, level="house view"
        )
        attributed = _send(
            session,
            harness.evidence,
            house.id,
            level="house view",
            research_attribution="house view 2026-10-01",
            content="The house view downgraded the banks.",
        )
        originate(
            session,
            barred.id,
            evidence=harness.evidence,
            trigger="event",
            candidate_set_size=4,
            criteria=["vulnerability"],
            do_not_contact=["vulnerability"],
            now=FROZEN_NOW,
        )
        _allow(session, harness.evidence, barred.id)
        blocked = _send(session, harness.evidence, barred.id)
        session.commit()

    assert no_decision["outcome"] == "DENY"
    assert no_decision["sent"] is False
    assert "final decision" in no_decision["citation"]
    assert no_attribution["outcome"] == "DENY"
    assert "research attribution" in no_attribution["citation"]
    assert attributed["outcome"] == "ALLOW"
    assert blocked["outcome"] == "DENY"
    assert blocked["sent"] is False
    assert "do not contact" in blocked["citation"]


def test_investment_discussion_needs_a_tier_and_a_cross_border_rule(harness):
    with harness.session() as session:
        unknown = _open(session, harness.evidence, interaction_id="int-unknown")
        known = _open(
            session,
            harness.evidence,
            interaction_id="int-known",
            use_case="research-summary",
        )
        _live(session, harness.evidence, _spec())
        _content_policy(session)
        _content_policy(
            session,
            levels=["investment discussion", "product recommendation", "transaction"],
            permitted_jurisdictions=["SG"],
            prohibited_claims=["guaranteed return"],
        )
        _channel(session, harness.evidence)
        _allow(session, harness.evidence, unknown.id)
        _allow(session, harness.evidence, known.id)
        unread = _send(
            session,
            harness.evidence,
            unknown.id,
            level="investment discussion",
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            content="We can discuss the house view.",
        )
        abroad = _send(
            session,
            harness.evidence,
            known.id,
            level="investment discussion",
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="DE",
            content="We can discuss the house view.",
        )
        discussed = _send(
            session,
            harness.evidence,
            known.id,
            level="investment discussion",
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            content="We can discuss the house view.",
        )
        session.commit()

    assert unread["outcome"] == "DENY"
    assert any(check["check"] == "USE_CASE_TIER" and check["result"] == "fail" for check in unread["checks"])
    assert abroad["outcome"] == "DENY"
    assert any("cross-border" in check["citation"] for check in abroad["checks"])
    assert discussed["outcome"] == "ALLOW"


def test_a_recommendation_needs_suitability_risk_language_and_rm_approval(harness):
    with harness.session() as session:
        _live(session, harness.evidence, _spec(usecase_id="portfolio-review"))
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence)
        _content_policy(session)
        _content_policy(
            session,
            levels=["investment discussion", "product recommendation", "transaction"],
            permitted_jurisdictions=["SG"],
            prohibited_claims=["guaranteed return"],
        )
        _channel(session, harness.evidence)
        _allow(session, harness.evidence, interaction.id)
        plain = _send(
            session,
            harness.evidence,
            interaction.id,
            level="product recommendation",
            content=NOTE,
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            risk_language=True,
        )
        review(
            session,
            interaction.id,
            rm="RM",
            outcome="approve",
            packet=_packet(
                policy_checks=[{"check": "TONE", "citation": "Tone: neutral"}]
            ),
            rendered=NOTE,
            provenance="human",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        unsuitable = _send(
            session,
            harness.evidence,
            interaction.id,
            level="product recommendation",
            content=NOTE,
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            risk_language=True,
        )
        review(
            session,
            interaction.id,
            rm="RM",
            outcome="approve",
            packet=_packet(),
            rendered=NOTE,
            provenance="human",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        no_risk = _send(
            session,
            harness.evidence,
            interaction.id,
            level="product recommendation",
            content=NOTE,
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            risk_language=False,
        )
        recommended = _send(
            session,
            harness.evidence,
            interaction.id,
            level="product recommendation",
            content=NOTE,
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            risk_language=True,
        )
        other = "Buy 10 units. Capital is at risk."
        wrong_text = _send(
            session,
            harness.evidence,
            interaction.id,
            level="transaction",
            content=other,
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            risk_language=True,
        )
        transacted = _send(
            session,
            harness.evidence,
            interaction.id,
            level="transaction",
            content=NOTE,
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            risk_language=True,
        )
        session.commit()

    assert plain["outcome"] == "DENY"
    assert "RM approval" in plain["citation"]
    assert unsuitable["outcome"] == "DENY"
    assert "suitability" in unsuitable["citation"]
    assert no_risk["outcome"] == "DENY"
    assert "investment-risk language" in no_risk["citation"]
    assert recommended["outcome"] == "ALLOW"
    assert wrong_text["outcome"] == "DENY"
    assert "approved text" in wrong_text["citation"]
    assert transacted["outcome"] == "ALLOW"
    assert transacted["sent"] is True


def test_tier_3_does_not_accept_a_plain_allow(harness):
    with harness.session() as session:
        _live(
            session,
            harness.evidence,
            _tier3(usecase_id="portfolio-review"),
            signoff_by="Noel Independent",
        )
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence)
        _content_policy(session)
        _content_policy(
            session,
            levels=["investment discussion", "product recommendation", "transaction"],
            permitted_jurisdictions=["SG"],
            prohibited_claims=["guaranteed return"],
        )
        _channel(session, harness.evidence)
        _allow(session, harness.evidence, interaction.id)
        denied = _send(
            session,
            harness.evidence,
            interaction.id,
            level="investment discussion",
            content="We can discuss the position.",
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
        )
        review(
            session,
            interaction.id,
            rm="RM",
            outcome="approve",
            packet=_packet(),
            rendered="We can discuss the position.",
            provenance="human",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        allowed = _send(
            session,
            harness.evidence,
            interaction.id,
            level="investment discussion",
            content="We can discuss the position.",
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
        )
        session.commit()

    assert denied["outcome"] == "DENY"
    assert any(check["check"] == "RM_APPROVAL" and check["result"] == "fail" for check in denied["checks"])
    assert allowed["outcome"] == "ALLOW"


def test_tier_4_recommendation_needs_a_confirmer(harness):
    spec = _spec(
        usecase_id="portfolio-review",
        purpose="Recommend a product the RM has approved",
        category="financial execution",
        knowledge_lead="Jane Smith",
        client_impact="execution",
        autonomy="assisted",
        decision_consequence="financial action",
        data_sensitivity="confidential",
        approval_gate="two_signature",
        review_sampling_rate=1.0,
        maximum_autonomy=4,
        explanation_depth="full",
        validators=("Jane Smith", "Noel Independent"),
        human_decision_points=("RM approves the recommendation",),
        model_providers=("house",),
    )
    with harness.session() as session:
        _live(session, harness.evidence, spec, signoff_by="Noel Independent")
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence, approval_gate="two_signature")
        _content_policy(session)
        _content_policy(
            session,
            levels=["investment discussion", "product recommendation", "transaction"],
            permitted_jurisdictions=["SG"],
            prohibited_claims=["guaranteed return"],
        )
        _channel(session, harness.evidence)
        review(
            session,
            interaction.id,
            rm="RM",
            outcome="approve",
            packet=_packet(),
            rendered=NOTE,
            provenance="human",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        unmarked = _send(
            session,
            harness.evidence,
            interaction.id,
            level="product recommendation",
            content=NOTE,
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            risk_language=True,
        )
        confirm_review(
            session,
            interaction.id,
            confirmer="Alex Chen",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        confirmed = _send(
            session,
            harness.evidence,
            interaction.id,
            level="product recommendation",
            content=NOTE,
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            risk_language=True,
        )
        session.commit()

    assert unmarked["outcome"] == "DENY"
    assert "final decision" in unmarked["citation"]
    assert confirmed["outcome"] == "ALLOW"
    assert any(check["check"] == "TWO_SIGNATURE" and check["result"] == "pass" for check in confirmed["checks"])


def test_ai_provenance_needs_a_disclosure_decision(harness):
    with harness.session() as session:
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence)
        _content_policy(session)
        _disclosure(session)
        _channel(session, harness.evidence)
        review(
            session,
            interaction.id,
            rm="RM",
            outcome="approve",
            packet=_packet(),
            rendered="Draft.",
            provenance="AI",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        relabelled = _send(
            session,
            harness.evidence,
            interaction.id,
            provenance="human",
            recipient_jurisdiction="SG",
            content="Draft.",
        )
        missing = _send(
            session,
            harness.evidence,
            interaction.id,
            provenance="AI",
            recipient_jurisdiction="SG",
            content="Draft.",
        )
        disclosed = _send(
            session,
            harness.evidence,
            interaction.id,
            provenance="AI",
            recipient_jurisdiction="SG",
            content="Draft. This message used AI.",
        )
        session.commit()

    assert relabelled["outcome"] == "DENY"
    assert relabelled["provenance"] == "AI"
    assert relabelled["claimed_provenance"] == "human"
    assert missing["outcome"] == "DENY"
    assert "disclosure" in missing["citation"]
    assert disclosed["outcome"] == "ALLOW"


def test_public_lane_refuses_a_named_client(harness):
    with harness.session() as session:
        _content_policy(session, lanes=["public"], prohibited_claims=["guaranteed"])
        _channel(session, harness.evidence, "LinkedIn", public=True)
        named = send_public_content(
            session,
            evidence=harness.evidence,
            channel="LinkedIn",
            content="The published house view.",
            level="house view",
            provenance="human",
            actor_id="RM",
            approved_research="house view 2026-10-01",
            research_attribution="house view 2026-10-01",
            client_id="client-1",
            now=FROZEN_NOW,
        )
        published = send_public_content(
            session,
            evidence=harness.evidence,
            channel="LinkedIn",
            content="The published house view.",
            level="house view",
            provenance="human",
            actor_id="RM",
            approved_research="house view 2026-10-01",
            research_attribution="house view 2026-10-01",
            now=FROZEN_NOW,
        )
        session.commit()
        report = reconstruct(session, "not-an-interaction")

    assert named["outcome"] == "DENY"
    assert named["sent"] is False
    assert "client" in named["citation"]
    assert published["outcome"] == "ALLOW"
    assert published["lane"] == "public"
    assert _answer(report, "What was communicated to the client?") == "not recorded"


def _publish(session, evidence, **kwargs):
    values = {
        "channel": "LinkedIn",
        "content": "The published house view.",
        "level": "house view",
        "provenance": "human",
        "actor_id": "RM",
        "approved_research": "house view 2026-10-01",
        "research_attribution": "house view 2026-10-01",
        "now": FROZEN_NOW,
    }
    values.update(kwargs)
    return send_public_content(session, evidence=evidence, **values)


def test_a_run_that_touched_client_data_cannot_publish(harness):
    registry, executed = _registry()
    client_run = Run()
    identifying = Run()
    with harness.session() as session:
        _content_policy(session, lanes=["public"], prohibited_claims=["guaranteed"])
        _channel(session, harness.evidence, "LinkedIn", public=True)
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence)
        run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            run=client_run,
            interaction_id=interaction.id,
        )
        run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            run=identifying,
        )
        session.add(
            RunRecord(id="run-clear", approvals_pending=[], tool_denylist=[])
        )
        session.flush()
        from_client = _publish(session, harness.evidence, run_id=client_run.id)
        from_class = _publish(session, harness.evidence, run_id=identifying.id)
        missing = _publish(session, harness.evidence, run_id="run-missing")
        clear = _publish(session, harness.evidence, run_id="run-clear")
        session.commit()

    assert from_client["outcome"] == "DENY"
    assert from_client["sent"] is False
    assert "touched client data" in from_client["citation"]
    assert from_class["outcome"] == "DENY"
    assert "touched client data" in from_class["citation"]
    assert missing["outcome"] == "DENY"
    assert "not recorded" in missing["citation"]
    assert clear["outcome"] == "ALLOW"
    assert clear["sent"] is True
    assert executed == ["get_subject_profile", "get_subject_profile"]


def test_a_transaction_after_modify_stays_denied(harness):
    approved = "Buy 10 units of the note. Capital is at risk."
    with harness.session() as session:
        _live(session, harness.evidence, _spec(usecase_id="portfolio-review"))
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence)
        _content_policy(session)
        _content_policy(
            session,
            levels=["investment discussion", "product recommendation", "transaction"],
            permitted_jurisdictions=["SG"],
            prohibited_claims=["guaranteed return"],
        )
        _channel(session, harness.evidence)
        review(
            session,
            interaction.id,
            rm="RM",
            outcome="modify",
            packet=_packet(),
            rendered=NOTE,
            approved=approved,
            provenance="human",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        sent = _send(
            session,
            harness.evidence,
            interaction.id,
            level="transaction",
            content=approved,
            research_attribution="house view 2026-10-01",
            recipient_jurisdiction="SG",
            risk_language=True,
        )
        session.commit()

    assert sent["outcome"] == "DENY"
    assert sent["sent"] is False
    assert "approved text" in sent["citation"]


def test_an_allow_override_does_not_outrank_a_later_policy_denial(harness):
    registry, executed = _registry()
    run = Run()
    with harness.session() as session:
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence)
        allowed = run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            run=run,
            interaction_id=interaction.id,
        )
        decision = session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.event_type == EventType.POLICY_DECISION
            )
        ).one()
        digest = decision.payload_hash
        overridden = override_run(
            session,
            run.id,
            evidence=harness.evidence,
            human="RM",
            citation="Intervention: override",
            outcome="allow",
            now=FROZEN_NOW,
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
            run=run,
            interaction_id=interaction.id,
        )
        session.expire_all()
        stored = session.get(EvidenceEvent, decision.id)
        stored_outcome = stored.payload["outcome"]
        stored_hash = stored.payload_hash
        override_state = session.get(RunRecord, run.id).override_outcome
        session.commit()

    assert allowed.evaluation.outcome is Outcome.ALLOW
    assert overridden["outcome"] == "allow"
    assert denied.evaluation.outcome is Outcome.DENY
    assert executed == ["get_subject_profile"]
    assert override_state == "ALLOW"
    assert stored_outcome == "ALLOW"
    assert stored_hash == digest
    assert all(check.result is not CheckResult.PASS or "override" not in check.citation for check in denied.evaluation.checks)


def test_a_complaint_links_the_output_model_version_and_approver(harness):
    with harness.session() as session:
        interaction = _open(session, harness.evidence)
        _allow(session, harness.evidence, interaction.id)
        EgressGuard(evidence=harness.evidence).send(
            PROMPT,
            lambda _text: ANSWER,
            classification=Classification.C1,
            destination="api.openai.com",
            destination_is_local=False,
            session=session,
            correlation_id=interaction.id,
            actor_id="RM",
            actor_type=ActorType.HUMAN,
            provider="api.openai.com",
            model="gateway-model",
            model_version="gateway-model-2026-06-01",
            prompt_template_version="template-3",
            use_case=interaction.use_case,
        )
        model_call = session.exec(
            select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.MODEL_CALL)
        ).one()
        response_hash = model_call.payload["response_hash"]
        with pytest.raises(OutcomeError, match="does not link"):
            record_complaint(
                session,
                interaction.id,
                evidence=harness.evidence,
                output_hash="0" * 64,
                summary="The note named the wrong client.",
                actor_id="RM",
                citation="Complaint: the note named the wrong client",
            )
        assert _events(session, EventType.OUTCOME) == []
        complaint = record_complaint(
            session,
            interaction.id,
            evidence=harness.evidence,
            output_hash=response_hash,
            summary="The note named the wrong client.",
            actor_id="RM",
            citation="Complaint: the note named the wrong client",
        )
        stored = _events(session, EventType.OUTCOME)
        assert len(stored) == 1
        recorded = dict(stored[0].payload)
        session.commit()

    assert complaint["action"] == "RECORDED"
    assert complaint["kind"] == "client complaint"
    assert complaint["output_hash"] == response_hash
    assert complaint["model_version"] == "gateway-model-2026-06-01"
    assert complaint["approver"] == "RM"
    assert complaint["sequence"]
    assert recorded["kind"] == "client complaint"
    assert recorded["action"] == "RECORDED"
    assert recorded["output_hash"] == response_hash
    assert recorded["rule"].startswith("Outcomes: a complaint links")
    assert "checks" not in recorded


def test_a_before_delivery_review_holds_the_client_send(harness):
    with harness.session() as session:
        _content_policy(session)
        _channel(session, harness.evidence)
        held_interaction = _open(session, harness.evidence)
        _allow(session, harness.evidence, held_interaction.id)
        enqueue_for_review(
            session,
            held_interaction.id,
            evidence=harness.evidence,
            reasons=["the response was screened high"],
            risk="high",
            citation="Risk-based review: high risk",
            actor_id="RM",
            timing="before delivery",
            now=FROZEN_NOW,
        )
        held = _send(session, harness.evidence, held_interaction.id)
        record_review_finding(
            session,
            held_interaction.id,
            reviewer="RM",
            outcome="cleared",
            finding="The draft is clear.",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        released = _send(session, harness.evidence, held_interaction.id)
        confirmed = _open(session, harness.evidence, client_id="client-2")
        _allow(session, harness.evidence, confirmed.id)
        enqueue_for_review(
            session,
            confirmed.id,
            evidence=harness.evidence,
            reasons=["the response was screened high"],
            risk="high",
            citation="Risk-based review: high risk",
            actor_id="RM",
            timing="before delivery",
            now=FROZEN_NOW,
        )
        record_review_finding(
            session,
            confirmed.id,
            reviewer="RM",
            outcome="confirmed problem",
            finding="The draft names the wrong client.",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        blocked = _send(
            session,
            harness.evidence,
            confirmed.id,
            recipient_ref="client-2-ref",
        )
        later = _open(session, harness.evidence, client_id="client-3")
        _allow(session, harness.evidence, later.id)
        enqueue_for_review(
            session,
            later.id,
            evidence=harness.evidence,
            reasons=["reviewed after the note went out"],
            risk="high",
            citation="Risk-based review: high risk",
            actor_id="RM",
            timing="after delivery",
            now=FROZEN_NOW,
        )
        after = _send(
            session,
            harness.evidence,
            later.id,
            recipient_ref="client-3-ref",
        )
        session.commit()

    assert held["outcome"] == "DENY"
    assert held["sent"] is False
    assert any(
        check["check"] == "REVIEW_HOLD" and check["result"] == "fail" and CITE_HOLD in check["citation"]
        for check in held["checks"]
    )
    assert released["outcome"] == "ALLOW"
    assert released["sent"] is True
    assert any(
        check["check"] == "REVIEW_HOLD" and check["result"] == "pass"
        for check in released["checks"]
    )
    assert blocked["outcome"] == "DENY"
    assert any(CITE_HOLD in check["citation"] for check in blocked["checks"])
    assert after["outcome"] == "ALLOW"


def test_one_interaction_answers_all_ten_questions(harness):
    with harness.session() as session:
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence)
        originate(
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
        item = index_item(
            session,
            title="cross-site eligibility",
            body=ITEM_BODY,
            embedder=harness.embedder,
            zones=["SG"],
            classification=Classification.C1,
        )
        hits = retrieve(
            session,
            ITEM_BODY,
            zone="SG",
            embedder=harness.embedder,
            evidence=harness.evidence,
            correlation_id=interaction.id,
            actor_id="RM",
            use_case=interaction.use_case,
        )
        assert hits
        EgressGuard(evidence=harness.evidence).send(
            PROMPT,
            lambda _text: ANSWER,
            classification=Classification.C1,
            destination="api.openai.com",
            destination_is_local=False,
            session=session,
            correlation_id=interaction.id,
            actor_id="RM",
            actor_type=ActorType.HUMAN,
            provider="api.openai.com",
            model="gateway-model",
            model_version="gateway-model-2026-06-01",
            prompt_template_version="template-3",
            use_case=interaction.use_case,
        )
        review(
            session,
            interaction.id,
            rm="RM",
            outcome="approve",
            packet=_packet(),
            rendered=NOTE,
            provenance="human",
            evidence=harness.evidence,
            now=FROZEN_NOW,
        )
        _content_policy(session)
        _channel(session, harness.evidence)
        sent = _send(
            session,
            harness.evidence,
            interaction.id,
            content="Market hours are 9 to 5.",
        )
        session.commit()
        report = reconstruct(session, interaction.id)
        assert verify_chain(session) > 0

    assert sent["outcome"] == "ALLOW"
    assert item.id
    assert [item["question"] for item in report["questions"]] == list(QUESTIONS)
    for question in QUESTIONS:
        assert _answer(report, question) != "not recorded"
    communicated = _answer(report, "What was communicated to the client?")
    assert communicated["communications"][0]["sent"] is True
    assert report["chain"]["intact"] is True
    assert report["findings"] == []


def test_pause_override_restrict_and_recover_are_evidenced(harness):
    registry, executed = _registry()
    run = Run()
    with harness.session() as session:
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence)
        allowed = run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            run=run,
            interaction_id=interaction.id,
        )
        decision = session.exec(
            select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.POLICY_DECISION)
        ).one()
        digest = decision.payload_hash
        observed = observe_run(
            session,
            run.id,
            evidence=harness.evidence,
            human="RM",
            citation="Intervention: observe",
            now=FROZEN_NOW,
        )
        with pytest.raises(InterventionError, match="not an authorised human"):
            pause_run(
                session,
                run.id,
                evidence=harness.evidence,
                human="stranger",
                citation="Intervention: pause",
                now=FROZEN_NOW,
            )
        restrict_run(
            session,
            run.id,
            evidence=harness.evidence,
            human="RM",
            citation="Intervention: restrict",
            ceiling="C1",
            now=FROZEN_NOW,
        )
        with pytest.raises(InterventionError, match="does not raise"):
            restrict_run(
                session,
                run.id,
                evidence=harness.evidence,
                human="RM",
                citation="Intervention: restrict",
                ceiling="C2",
                now=FROZEN_NOW,
            )
        restricted = restrict_run(
            session,
            run.id,
            evidence=harness.evidence,
            human="RM",
            citation="Intervention: restrict",
            tools=["get_subject_profile"],
            ceiling="C0",
            now=FROZEN_NOW,
        )
        narrowed = run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            run=run,
            interaction_id=interaction.id,
        )
        session.expire_all()
        assert session.get(EvidenceEvent, decision.id).payload_hash == digest
        overridden = override_run(
            session,
            run.id,
            evidence=harness.evidence,
            human="RM",
            citation="Intervention: override",
            outcome="deny",
            decision_sequence=decision.sequence,
            now=FROZEN_NOW,
        )
        session.expire_all()
        assert session.get(EvidenceEvent, decision.id).payload["outcome"] == "ALLOW"
        paused = pause_run(
            session,
            run.id,
            evidence=harness.evidence,
            human="RM",
            citation="Intervention: pause",
            now=FROZEN_NOW,
        )
        stopped = run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            run=run,
            interaction_id=interaction.id,
        )
        register_compensating_action(
            session,
            run.id,
            evidence=harness.evidence,
            human="RM",
            citation="Intervention: recover",
            name="withdraw the draft",
            now=FROZEN_NOW,
        )
        register_compensating_action(
            session,
            run.id,
            evidence=harness.evidence,
            human="RM",
            citation="Intervention: recover",
            name="cancel the queued note",
            now=FROZEN_NOW,
        )
        recovered = recover_run(
            session,
            run.id,
            evidence=harness.evidence,
            human="RM",
            citation="Intervention: recover",
            now=FROZEN_NOW,
        )
        after = run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            run=run,
            interaction_id=interaction.id,
        )
        session.commit()
        stored = session.get(RunRecord, run.id)
        assert verify_chain(session) > 0

    assert allowed.evaluation.outcome is Outcome.ALLOW
    assert observed["action"] == "observe"
    assert observed["snapshot"]["control_state"] == "live"
    assert restricted["ceiling_after"] == "C0"
    assert "get_subject_profile" in restricted["tools_after"]
    assert narrowed.evaluation.outcome is Outcome.DENY
    assert any("restricted" in check.citation for check in narrowed.evaluation.checks)
    assert overridden["original_outcome"] == "ALLOW"
    assert overridden["outcome"] == "deny"
    assert paused["action"] == "pause"
    assert stopped.evaluation.outcome is Outcome.DENY
    assert any("paused" in check.citation for check in stopped.evaluation.checks)
    assert [step["name"] for step in recovered["steps"]] == [
        "cancel the queued note",
        "withdraw the draft",
    ]
    assert after.evaluation.outcome is Outcome.DENY
    assert any("recovered" in check.citation for check in after.evaluation.checks)
    assert stored.control_state == "recovered"
    assert executed == ["get_subject_profile"]


def test_revoke_engages_the_kill_switch_and_stops_the_run(harness):
    registry, executed = _registry()
    run = Run()
    with harness.session() as session:
        _boundary(session, harness.evidence)
        interaction = _open(session, harness.evidence)
        run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            run=run,
            interaction_id=interaction.id,
        )
        with pytest.raises(InterventionError, match="not a registered agent"):
            revoke_run(
                session,
                run.id,
                evidence=harness.evidence,
                human="RM",
                citation="Intervention: revoke",
                agent_name="missing-agent",
                now=FROZEN_NOW,
            )
        revoked = revoke_run(
            session,
            run.id,
            evidence=harness.evidence,
            human="RM",
            citation="Intervention: revoke",
            agent_name="eligibility-assistant",
            now=FROZEN_NOW,
        )
        denied = run_governed(
            session,
            dp.envelope("get_subject_profile", site="SG", subject="2"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=FROZEN_NOW,
            now=FROZEN_NOW,
            run=run,
            interaction_id=interaction.id,
        )
        agent = get_agent(session, "eligibility-assistant")
        killed = agent.kill_switch
        session.commit()

    assert killed is True
    assert revoked["agent"] == "eligibility-assistant"
    assert denied.evaluation.outcome is Outcome.DENY
    assert any("revoked" in check.citation for check in denied.evaluation.checks)
    assert executed == ["get_subject_profile"]


def test_a_paused_run_does_not_resume_a_held_call(gov):
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
        _boundary(session, gov.evidence)
        interaction = _open(session, gov.evidence)
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
            interaction_id=interaction.id,
        )
        approval = session.exec(select(Approval)).one()
        pause_run(
            session,
            run.id,
            evidence=gov.evidence,
            human="RM",
            citation="Intervention: pause",
            now=FROZEN_NOW,
        )
        withheld = continue_run(
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
        approval_row = session.get(Approval, approval.id)
        finals = _events(session, EventType.FINAL_DECISION)
        actions = [
            event.payload.get("action")
            for event in _events(session, EventType.APPROVAL)
        ]

    assert held.data is None
    assert withheld.data is None
    assert withheld.evaluation.outcome is Outcome.DENY
    assert stored.continuation_hash
    assert stored.control_state == "paused"
    assert approval_row.status is ApprovalStatus.PENDING
    assert finals == []
    assert "WITHHELD" in actions
    assert "RESUMED" not in actions
