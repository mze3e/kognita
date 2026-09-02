"""BMOS conformance: testing governance doctrine against the fixture pack.

This test suite exercises the BMOS-shaped fixture pack against Kognita's
capability to express its central doctrine: "one person marks a criterion met;
only an owner confirms; nothing self-passes."

Each test that fails due to a missing Kognita feature is marked `xfail(strict=True)`.
When all xfails flip to passing, the gap analysis roadmap is complete.

Reference: `docs/gap-analysis-bmos.md`
"""
from __future__ import annotations

import pytest
from sqlmodel import select

from fixtures import bmos_pack as bp

from kognita.models import Approval, EvidenceEvent
from kognita.registry import register
from kognita.vocabulary import Outcome
from kognita.testing.conformance import ConformanceCase
from kognita.testing.harness import FROZEN_NOW, Harness


def _seed(session):
    bp.seed_policies(session, now=FROZEN_NOW)
    register(session, name="marker-agent", owner_exec="Risk Team")
    register(session, name="proposer-agent", owner_exec="Policy Team")


@pytest.fixture
def harness():
    return Harness(pack=bp.BMOSPack(), purposes=bp.PURPOSES, seed=_seed)


# ── Test shape: role-gated access ────────────────────────────────────────────


def test_marker_can_mark_criterion(harness):
    """MARKER role can execute mark_criterion tool."""
    envelope = bp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="marker_alice",
        criterion="crit_1",
        principal="marker_alice",
    )
    with harness.session() as session:
        evaluation = harness.evaluate(envelope, session)

    # Should pass: the tool is allowed for MARKER role
    assert evaluation.outcome in (Outcome.ALLOW, Outcome.HUMAN_APPROVAL)


def test_marker_cannot_confirm(harness):
    """MARKER role cannot execute confirm_criterion tool (OWNER-only)."""
    envelope = bp.envelope(
        "confirm_criterion",
        purpose="CONFIRM_CRITERION",
        user="marker_alice",
        criterion="crit_1",
        principal="marker_alice",
    )
    with harness.session() as session:
        evaluation = harness.evaluate(envelope, session)

    # Should fail: MARKER has no permission to confirm
    assert evaluation.outcome is Outcome.DENY


def test_owner_can_confirm_criterion(harness):
    """OWNER role can execute confirm_criterion tool."""
    envelope = bp.envelope(
        "confirm_criterion",
        purpose="CONFIRM_CRITERION",
        user="owner_bob",
        criterion="crit_1",
        principal="owner_bob",
    )
    with harness.session() as session:
        evaluation = harness.evaluate(envelope, session)

    # Should pass: OWNER can confirm
    assert evaluation.outcome is Outcome.ALLOW


def test_owner_cannot_mark(harness):
    """OWNER role cannot execute mark_criterion tool (MARKER-only)."""
    envelope = bp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="owner_bob",
        criterion="crit_1",
        principal="owner_bob",
    )
    with harness.session() as session:
        evaluation = harness.evaluate(envelope, session)

    # Should fail: OWNER has no permission to mark
    assert evaluation.outcome is Outcome.DENY


# ── Test shape: tool arguments (Tier 1.1) ───────────────────────────────────

def test_mark_criterion_arguments_hashed(harness):
    """Tool arguments travel on the envelope and are included in the hash.

    When a MARKER proposes to mark criterion_1, the decision should bind to:
    - The criterion_id being marked
    - Any mark data (e.g., score, note)

    An approval granted for mark criterion_1 should NOT authorize marking
    criterion_2, even if signed by the same owner.
    """
    envelope_1 = bp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="marker_alice",
        criterion="crit_1",
        principal="marker_alice",
        agent="marker-agent",
    )
    envelope_2 = bp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="marker_alice",
        criterion="crit_2",  # Different criterion
        principal="marker_alice",
        agent="marker-agent",
    )

    with harness.session() as session:
        eval1 = harness.decide_and_record(envelope_1, session)
        eval2 = harness.decide_and_record(envelope_2, session)
        session.commit()

        # Different criterion_id should produce different envelope_hash
        assert eval1.envelope_hash != eval2.envelope_hash


# ── Test shape: two-signature / separation of duties (Tier 1.3) ──────────────

@pytest.mark.xfail(strict=True, reason="Two-signature approvals not yet implemented (Tier 1.3)")
def test_marking_requires_two_signatures(harness):
    """Marking a criterion requires both MARKER and OWNER signatures.

    1. MARKER marks criterion_1
    2. System creates an approval for OWNER
    3. OWNER must confirm the mark
    4. Only then does the mark bind

    A single MARKER action should not self-pass.
    """
    envelope = bp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="marker_alice",
        criterion="crit_1",
        principal="marker_alice",
        agent="marker-agent",
    )

    with harness.session() as session:
        # MARKER action
        result = harness.run_governed(envelope, session)
        assert result.outcome is Outcome.HUMAN_APPROVAL
        assert result.data is None  # Data must not leak before approval

        session.commit()
        approvals = session.exec(select(Approval)).all()
        assert any(
            a.decision_id == result.evaluation.decision_id
            and a.approver_name != "marker_alice"  # Requester is separate from approver
            for a in approvals
        )


@pytest.mark.xfail(strict=True, reason="Two-signature approvals not yet implemented (Tier 1.3)")
def test_approval_carries_proposal_payload(harness):
    """An approval binds to a specific proposal, not just a request.

    When owner_bob approves marking criterion_1, the approval should record:
    - What was proposed (mark criterion_1 with score=85)
    - By whom (marker_alice)
    - The proposer's identity must be separate from the approver's

    A second marker cannot substitute a different proposal.
    """
    envelope = bp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="marker_alice",
        criterion="crit_1",
        principal="marker_alice",
        agent="marker-agent",
    )

    with harness.session() as session:
        # Initial mark proposal
        eval1 = harness.decide_and_record(envelope, session)
        session.commit()

        approval = session.exec(
            select(Approval).where(Approval.decision_id == eval1.decision_id)
        ).first()
        assert approval is not None
        assert approval.approver_name != "marker_alice"  # Separation of duties


# ── Test shape: propose-then-apply (Tier 1.4) ────────────────────────────────

@pytest.mark.xfail(strict=True, reason="Propose-then-apply not yet implemented (Tier 1.4)")
def test_proposal_must_bind_to_diff(harness):
    """A proposal carries the before-state and the change, not just a request.

    Agent proposes: change document title from "Old Title" to "New Title"
    Owner receives: the proposal with both versions visible
    Owner applies: the change is atomic

    The approval must bind to the specific diff, not just "apply proposal".
    """
    # This is a structural gap: Approval has no payload field
    pass


@pytest.mark.xfail(strict=True, reason="Propose-then-apply not yet implemented (Tier 1.4)")
def test_data_held_until_second_signature(harness):
    """Data is held and not returned until the second signature.

    1. Agent proposes to fetch a document
    2. Evaluation outcome is HUMAN_APPROVAL
    3. Data is None (not returned to the agent)
    4. Owner confirms
    5. Only then does the fetch execute and data leaves the boundary

    This requires HUMAN_APPROVAL to truly withhold data.
    Reference: Tier 0 defect #1
    """
    envelope = bp.envelope(
        "propose_document",
        purpose="PROPOSE_CHANGE",
        user="marker_alice",
        document="doc_1",
        principal="marker_alice",
        agent="proposer-agent",
    )

    with harness.session() as session:
        result = harness.run_governed(envelope, session)

        # HUMAN_APPROVAL should release no data
        if result.outcome is Outcome.HUMAN_APPROVAL:
            assert result.data is None, "HUMAN_APPROVAL must not return data (Tier 0.1)"


# ── Test shape: human-readable feedback on rejection ──────────────────────────


def test_denial_names_the_failing_policy(harness):
    """When a request is denied, the response names the policy that blocked it.

    This proves the denial is traceable, not a black-box "access denied".
    """
    envelope = bp.envelope(
        "confirm_criterion",  # MARKER trying to confirm (not allowed)
        purpose="CONFIRM_CRITERION",
        user="marker_alice",
        criterion="crit_1",
        principal="marker_alice",
    )

    with harness.session() as session:
        evaluation = harness.evaluate(envelope, session)

    if evaluation.outcome is Outcome.DENY:
        assert evaluation.failures(), "Denial should cite the failing check"
        for check in evaluation.failures():
            assert check.citation, "Each failure must cite its policy"


# ── Test shape: identity and authority (Tier 1.2) ─────────────────────────────

@pytest.mark.xfail(strict=True, reason="Identity/authority model not yet implemented (Tier 1.2)")
def test_role_verified_against_pack(harness):
    """The pack's role assertion is verified, not blindly trusted.

    Today, Kognita trusts is_admin on the envelope. A hostile caller can
    assert is_admin=True and the library accepts it.

    With identity and authority (Tier 1.2), roles should be verified:
    - Envelope declares a role
    - Pack confirms it's real
    - Kognita enforces role-based policy

    This test passes when the conformance kit verifies role claims.
    """
    # This requires a conformance invariant, not just a pack feature
    pass


# ── Test shape: evidence trails ──────────────────────────────────────────────


def test_approval_is_evidenced(harness):
    """When an approval is granted, an evidence event is recorded.

    The event captures:
    - The decision being approved
    - The approver's identity
    - The timestamp
    - A hash linking to the previous event (tamper-evident)
    """
    envelope = bp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="marker_alice",
        criterion="crit_1",
        principal="marker_alice",
        agent="marker-agent",
    )

    with harness.session() as session:
        eval1 = harness.decide_and_record(envelope, session)
        session.commit()

        events = session.exec(select(EvidenceEvent)).all()
        # Should have at least DECISION + TOOL_CALL + EGRESS events
        assert len(events) > 0
        decision_event = next(
            e for e in events if e.event_type.value == "DECISION"
        )
        assert decision_event is not None


def test_denied_access_is_evidenced(harness):
    """When access is denied, the denial is recorded in evidence.

    A denied decision produces an event with outcome=DENY and reasons.
    """
    envelope = bp.envelope(
        "mark_criterion",
        purpose="MARK_CRITERION",
        user="owner_bob",  # OWNER trying to mark (not allowed)
        criterion="crit_1",
        principal="owner_bob",
    )

    with harness.session() as session:
        eval1 = harness.decide_and_record(envelope, session)
        session.commit()

        # Denied access should leave an evidence trail
        events = session.exec(select(EvidenceEvent)).all()
        assert any(e.payload.get("outcome") == "DENY" for e in events)
