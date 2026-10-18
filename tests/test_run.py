"""Run budgets, and suspend/resume for HUMAN_APPROVAL.

These cover the first slice of roadmap 0.3: a run that can refuse a call for a
named budget, and a hold that neither executes a tool nor retrieves until the
approval is actually granted.
"""
from __future__ import annotations

from datetime import timedelta

import pytest
from sqlmodel import select

from fixtures import demo_pack as dp

from kognita.approvals import ApprovalError
from kognita.broker import ask
from kognita.canonical import canonical_hash
from kognita.evidence import verify_chain
from kognita.models import Continuation, EvidenceEvent, GovernanceDecision, Policy, RunRecord
from kognita.retrieval import index_item
from kognita.tools import Run, ToolRegistry, continue_run, run_governed
from kognita.vocabulary import Classification, EventType, Outcome
from kognita.testing.harness import FROZEN_NOW, Harness


def _seed(session):
    dp.seed_policies(session, now=FROZEN_NOW)
    from kognita.registry import register

    register(session, name="eligibility-assistant", owner_exec="Head of Data Governance")
    register(session, name="dossier-agent", owner_exec="Head of Research Ops")


@pytest.fixture
def harness():
    return Harness(pack=dp.DemoPack(), purposes=dp.PURPOSES, seed=_seed)


def _registry(classification: Classification = Classification.C2) -> tuple[ToolRegistry, list[str]]:
    executed: list[str] = []
    registry = ToolRegistry()

    @registry.tool("get_subject_profile", classification=classification)
    def _profile(envelope, evaluation, session):
        executed.append("profile")
        return {"subject": envelope.subject_id}

    @registry.tool("draft_publication", classification=classification)
    def _draft(envelope, evaluation, session):
        executed.append("draft")
        return {"status": "DRAFT — not for release"}

    return registry, executed


def _allow_envelope():
    return dp.envelope("get_subject_profile", site="SG", subject="2")


def _hold_envelope():
    return dp.envelope(
        "draft_publication", purpose="DOSSIER_PREP", site="SG", subject="1"
    )


def _approval_actions(session, request_id: str) -> list[str]:
    events = session.exec(
        select(EvidenceEvent)
        .where(EvidenceEvent.correlation_id == request_id)
        .order_by(EvidenceEvent.sequence)
    ).all()
    return [
        event.payload.get("action")
        for event in events
        if event.event_type == EventType.APPROVAL and event.payload.get("action")
    ]


def test_budget_deny_cites_the_budget_and_records_consumption(harness):
    """Exceeding a cap is a DENY whose citation is that cap, and the chain shows the spend."""
    registry, executed = _registry()
    run = Run(max_calls=1)
    envelope = _allow_envelope()

    with harness.session() as session:
        first = run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=run,
            now=harness.now,
        )
        second = run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=run,
            now=harness.now,
        )
        session.commit()

        denial = session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.correlation_id == second.evaluation.request_id,
                EvidenceEvent.event_type == EventType.POLICY_DECISION,
            )
        ).one()

    assert first.outcome is Outcome.ALLOW
    assert first.data is not None
    assert second.outcome is Outcome.DENY
    assert second.data is None
    assert executed == ["profile"]
    assert any(check.citation == "max_calls" for check in second.evaluation.basis())
    assert denial.payload["budget"]["calls_used"] == 1
    assert denial.payload["budget"]["max_calls"] == 1
    assert any(check["citation"] == "max_calls" for check in denial.payload["checks"])


def test_wall_clock_cost_classification_and_tokens(harness):
    """The caps this slice can enforce, plus token spend recorded for a later gateway."""
    registry, executed = _registry()
    envelope = _allow_envelope()

    with harness.session() as session:
        clock = Run(wall_clock_seconds=30)
        run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=clock,
            now=harness.now,
        )
        late = run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=clock,
            now=harness.now + timedelta(seconds=31),
        )

        priced = Run(max_cost_usd=1.0)
        run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=priced,
            cost_usd=1.0,
            now=harness.now,
        )
        over_cost = run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=priced,
            cost_usd=0.01,
            now=harness.now,
        )

        metered = Run(max_tokens=10)
        run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=metered,
            tokens=4,
            now=harness.now,
        )
        over_tokens = run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=metered,
            tokens=7,
            now=harness.now,
        )
        recorded = session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.correlation_id == over_tokens.evaluation.request_id,
                EvidenceEvent.event_type == EventType.POLICY_DECISION,
            )
        ).one()
        recorded_tokens = recorded.payload["budget"]["tokens_used"]
        session.commit()

    sensitive, sensitive_calls = _registry(Classification.C3)
    with harness.session() as session:
        ceiling = Run(classification_ceiling=Classification.C1)
        above = run_governed(
            session,
            envelope,
            registry=sensitive,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=ceiling,
            now=harness.now,
        )
        session.commit()

    assert late.outcome is Outcome.DENY
    assert any(check.citation == "wall_clock_seconds" for check in late.evaluation.basis())
    assert over_cost.outcome is Outcome.DENY
    assert any(check.citation == "max_cost_usd" for check in over_cost.evaluation.basis())
    assert priced.cost_usd_used == 1.0
    assert over_tokens.outcome is Outcome.DENY
    assert any(check.citation == "max_tokens" for check in over_tokens.evaluation.basis())
    assert metered.tokens_used == 4
    assert recorded_tokens == 4
    assert above.outcome is Outcome.DENY
    assert any(check.citation == "classification_ceiling" for check in above.evaluation.basis())
    assert sensitive_calls == []
    assert executed == ["profile", "profile", "profile"]


def test_human_approval_executes_neither_tool_nor_retrieval(harness):
    """A HUMAN_APPROVAL outcome releases no tool result and retrieves nothing."""
    registry, executed = _registry()
    envelope = _hold_envelope()
    seen: list[str] = []

    def subgraph(_envelope, _session):
        seen.append("subgraph")
        return {"nodes": 1, "edges": 1}

    with harness.session() as session:
        index_item(
            session,
            title="Ethics board standing order",
            body="Restricted-region subjects require a person to sign the release.",
            embedder=harness.embedder,
            zones=["SG"],
            source_label="Ethics Board Standing Order 22",
        )
        held = run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=Run(),
            now=harness.now,
        )
        answer = ask(
            session,
            "What does the ethics board require before release?",
            envelope,
            pack=harness.pack,
            embedder=harness.embedder,
            evidence=harness.evidence,
            purposes=dp.PURPOSES,
            subgraph=subgraph,
            as_of=harness.now,
            run=Run(),
            now=harness.now,
        )
        session.commit()
        events = session.exec(select(EvidenceEvent)).all()

    assert held.outcome is Outcome.HUMAN_APPROVAL
    assert held.data is None
    assert executed == []
    assert answer.outcome is Outcome.HUMAN_APPROVAL
    assert answer.results == []
    assert answer.graph is None
    assert seen == []
    assert not any(event.event_type == EventType.RETRIEVAL for event in events)
    assert not any(event.event_type == EventType.TOOL_CALL for event in events)


def test_continue_run_executes_only_after_grant(harness):
    """The held tool runs on resume, and only once the approval is granted.

    The continuation is a content hash on the run, not the payload itself, and
    it survives a new session.
    """
    registry, executed = _registry()
    run = Run()

    with harness.session() as session:
        held = run_governed(
            session,
            _hold_envelope(),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=run,
            now=harness.now,
        )
        session.commit()
        run_id = run.id
        approval_id = run.approvals_pending[0]
        request_id = held.evaluation.request_id

    assert executed == []

    with harness.session() as session:
        row = session.get(RunRecord, run_id)
        assert row is not None
        assert row.continuation_hash
        stored = session.get(Continuation, row.continuation_hash)
        assert stored is not None
        assert canonical_hash(stored.payload) == row.continuation_hash
        assert "payload" not in row.model_dump()
        assert stored.payload["envelope"]["tool"] == "draft_publication"
        assert "tool" not in row.model_dump()

        resumed = continue_run(
            session,
            run_id,
            {approval_id: True},
            approver_name="Jane Approver",
            evidence=harness.evidence,
            registry=registry,
            now=harness.now,
        )
        session.commit()
        actions = _approval_actions(session, request_id)
        assert verify_chain(session) > 0
        resumed_event = next(
            event
            for event in session.exec(select(EvidenceEvent)).all()
            if event.event_type == EventType.APPROVAL and event.payload.get("action") == "RESUMED"
        )

    assert executed == ["draft"]
    assert resumed.data == {"status": "DRAFT — not for release"}
    assert actions.index("REQUESTED") < actions.index("APPROVED") < actions.index("RESUMED")
    assert "rechecked" not in resumed_event.payload
    assert resumed_event.payload["executed"] is True


def test_continue_run_withholds_a_policy_added_while_the_call_was_held(harness):
    """A prohibition added after the hold withholds the release.

    A fresh decision on the same call is a denial. Approving the original
    hold does not execute the tool. The stored decision stays the hold, and
    the resume event lists the new check.
    """
    registry, executed = _registry()
    run = Run()
    envelope = _hold_envelope()

    with harness.session() as session:
        held = run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=run,
            now=harness.now,
        )
        session.add(
            Policy(
                regime="HOME_SITE",
                rule_type="PROHIBITED",
                rule={"description": "release barred", "on_violation": "fail"},
                citation="Ethics board withdrew the release",
                effective_from=harness.now,
            )
        )
        session.commit()
        fresh = run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            now=harness.now,
        )
        run_id = run.id
        approval_id = run.approvals_pending[0]
        request_id = held.evaluation.request_id
        resumed = continue_run(
            session,
            run_id,
            {approval_id: True},
            approver_name="Jane Approver",
            evidence=harness.evidence,
            registry=registry,
            now=harness.now,
        )
        session.commit()
        stored = session.exec(
            select(GovernanceDecision).where(GovernanceDecision.request_id == request_id)
        ).one()
        resumed_event = next(
            event
            for event in session.exec(select(EvidenceEvent)).all()
            if event.correlation_id == request_id
            and event.event_type == EventType.APPROVAL
            and event.payload.get("action") == "RESUMED"
        )

    assert held.outcome is Outcome.HUMAN_APPROVAL
    assert fresh.evaluation.outcome is Outcome.DENY
    assert executed == []
    assert resumed.data is None
    assert resumed.evaluation.outcome is Outcome.DENY
    assert any(check.check == "release barred" for check in resumed.evaluation.checks)
    assert "release barred" in resumed_event.payload["rechecked"]
    assert resumed_event.payload["executed"] is False
    assert stored.outcome is Outcome.HUMAN_APPROVAL


def test_denied_approval_does_not_execute(harness):
    """A false resolution rejects the approval and leaves the tool unrun."""
    registry, executed = _registry()
    run = Run()

    with harness.session() as session:
        held = run_governed(
            session,
            _hold_envelope(),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=run,
            now=harness.now,
        )
        session.commit()
        request_id = held.evaluation.request_id
        resumed = None

    with harness.session() as session:
        resumed = continue_run(
            session,
            run.id,
            {run.approvals_pending[0]: False},
            evidence=harness.evidence,
            registry=registry,
            now=harness.now,
        )
        session.commit()
        actions = _approval_actions(session, request_id)
        tool_calls = session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.correlation_id == request_id,
                EvidenceEvent.event_type == EventType.TOOL_CALL,
            )
        ).all()

    assert resumed.data is None
    assert executed == []
    assert tool_calls == []
    assert actions.index("REQUESTED") < actions.index("REJECTED") < actions.index("RESUMED")
    assert "APPROVED" not in actions


def test_continue_run_retrieves_only_after_grant(harness):
    """A held question retrieves on resume, and not before the grant."""
    envelope = _hold_envelope()
    run = Run()
    question = "What does the ethics board require before release?"

    with harness.session() as session:
        index_item(
            session,
            title="Ethics board standing order",
            body="Restricted-region subjects require a person to sign the release.",
            embedder=harness.embedder,
            zones=["SG"],
            source_label="Ethics Board Standing Order 22",
        )
        held = ask(
            session,
            question,
            envelope,
            pack=harness.pack,
            embedder=harness.embedder,
            evidence=harness.evidence,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=run,
            now=harness.now,
        )
        session.commit()
        assert held.results == []
        assert not session.exec(
            select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.RETRIEVAL)
        ).all()

    with harness.session() as session:
        resumed = continue_run(
            session,
            run.id,
            {run.approvals_pending[0]: False},
            evidence=harness.evidence,
            embedder=harness.embedder,
            now=harness.now,
        )
        session.commit()
        assert resumed.results == []
        assert not session.exec(
            select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.RETRIEVAL)
        ).all()

    run = Run()
    with harness.session() as session:
        ask(
            session,
            question,
            envelope,
            pack=harness.pack,
            embedder=harness.embedder,
            evidence=harness.evidence,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=run,
            now=harness.now,
        )
        session.commit()

    with harness.session() as session:
        resumed = continue_run(
            session,
            run.id,
            {run.approvals_pending[0]: True},
            approver_name="Jane Approver",
            evidence=harness.evidence,
            embedder=harness.embedder,
            now=harness.now,
        )
        session.commit()
        retrievals = session.exec(
            select(EvidenceEvent).where(
                EvidenceEvent.correlation_id == resumed.request_id,
                EvidenceEvent.event_type == EventType.RETRIEVAL,
            )
        ).all()

    assert retrievals
    assert isinstance(resumed.results, list)


def _held(harness, registry):
    run = Run()
    with harness.session() as session:
        run_governed(
            session,
            _hold_envelope(),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=run,
            now=harness.now,
        )
        session.commit()
    return run


def test_continue_run_refuses_a_grant_with_no_named_approver(harness):
    """Holding a run id is not an approval: a grant must name the approver."""
    registry, executed = _registry()
    run = _held(harness, registry)

    with harness.session() as session:
        with pytest.raises(ApprovalError, match="must name the approver"):
            continue_run(
                session,
                run.id,
                {run.approvals_pending[0]: True},
                evidence=harness.evidence,
                registry=registry,
                now=harness.now,
            )
        session.rollback()

    with harness.session() as session:
        row = session.get(RunRecord, run.id)
        assert row is not None and row.continuation_hash
    assert executed == []


def test_continue_run_refuses_self_approval_by_the_requester(harness):
    """The requesting principal cannot sign their own single-signature approval."""
    registry, executed = _registry()
    run = _held(harness, registry)
    requester = _hold_envelope().principal

    with harness.session() as session:
        with pytest.raises(ApprovalError, match="differ from the requester"):
            continue_run(
                session,
                run.id,
                {run.approvals_pending[0]: True},
                evidence=harness.evidence,
                registry=registry,
                approver_name=requester,
                now=harness.now,
            )
        session.rollback()
    assert executed == []


def test_resume_withholds_a_call_revoked_while_it_waited(harness):
    """A revocation recorded after the hold wins over the later grant."""
    from kognita.models import Agent
    from kognita.registry import register
    from kognita.revocation import revoke

    registry, executed = _registry()
    run = _held(harness, registry)

    with harness.session() as session:
        if session.exec(select(Agent).where(Agent.name == "dossier-agent")).first() is None:
            register(session, name="dossier-agent", owner_exec="Head of Research Ops")
        revoke(
            session,
            "tool",
            "draft_publication",
            evidence=harness.evidence,
            actor_id="Head of Research Ops",
            citation="Tool withdrawn pending review",
            owner="Head of Research Ops",
            now=harness.now,
        )
        resumed = continue_run(
            session,
            run.id,
            {run.approvals_pending[0]: True},
            approver_name="Jane Approver",
            evidence=harness.evidence,
            registry=registry,
            now=harness.now,
        )
        session.commit()

    assert executed == []
    assert resumed.data is None
    assert resumed.evaluation.outcome is Outcome.DENY
    assert any("Revocation" in check.citation for check in resumed.evaluation.failures())


def test_resume_withholds_a_call_after_the_agent_kill_switch(harness):
    """Engaging the agent's kill switch after the hold stops the release."""
    from kognita.models import Agent
    from kognita.registry import register, set_kill_switch

    registry, executed = _registry()
    run = Run()
    envelope = dp.envelope(
        "draft_publication",
        purpose="DOSSIER_PREP",
        site="SG",
        subject="1",
        agent="dossier-agent",
    )
    with harness.session() as session:
        if session.exec(select(Agent).where(Agent.name == "dossier-agent")).first() is None:
            register(session, name="dossier-agent", owner_exec="Head of Research Ops")
        held = run_governed(
            session,
            envelope,
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES,
            as_of=harness.now,
            run=run,
            now=harness.now,
        )
        session.commit()
    assert held.approval_required

    with harness.session() as session:
        set_kill_switch(
            session,
            "dossier-agent",
            True,
            evidence=harness.evidence,
            actor_id="Head of Research Ops",
            now=harness.now,
        )
        resumed = continue_run(
            session,
            run.id,
            {run.approvals_pending[0]: True},
            approver_name="Jane Approver",
            evidence=harness.evidence,
            registry=registry,
            now=harness.now,
        )
        session.commit()

    assert executed == []
    assert resumed.evaluation.outcome is Outcome.DENY


def test_retrieval_excludes_a_revoked_source_the_request_never_named(harness):
    """Revoking a source removes it from the candidates, not only from named requests."""
    from kognita.models import Agent
    from kognita.registry import register
    from kognita.retrieval import retrieve
    from kognita.revocation import revoke

    question = "What does the ethics board require before release?"
    with harness.session() as session:
        if session.exec(select(Agent).where(Agent.name == "dossier-agent")).first() is None:
            register(session, name="dossier-agent", owner_exec="Head of Research Ops")
        index_item(
            session,
            title="Ethics board standing order",
            body="Restricted-region subjects require a person to sign the release.",
            embedder=harness.embedder,
            zones=["SG"],
            source_label="Ethics Board Standing Order 22",
            document_ref="revoked-doc",
        )
        before = retrieve(
            session, question, zone="SG", embedder=harness.embedder, min_score=0.0
        )
        revoke(
            session,
            "source",
            "revoked-doc",
            evidence=harness.evidence,
            actor_id="Head of Research Ops",
            citation="Source withdrawn",
            owner="Head of Research Ops",
            now=harness.now,
        )
        after = retrieve(
            session, question, zone="SG", embedder=harness.embedder, min_score=0.0
        )
        session.commit()

    assert [hit.document_ref for hit in before] == ["revoked-doc"]
    assert after == []


def test_run_governed_denies_a_purpose_the_register_does_not_name(harness):
    """A purpose the register does not name is denied, including one on the purpose list."""
    registry, executed = _registry()
    with harness.session() as session:
        result = run_governed(
            session,
            dp.envelope("draft_publication", purpose="NOT-REGISTERED", site="SG", subject="1"),
            registry=registry,
            evidence=harness.evidence,
            pack=harness.pack,
            purposes=dp.PURPOSES + ("NOT-REGISTERED",),
            as_of=harness.now,
            now=harness.now,
        )
        session.commit()

    assert executed == []
    assert result.evaluation.outcome is Outcome.DENY
    assert any(
        check.check == "USE_CASE_REGISTERED" for check in result.evaluation.failures()
    )
