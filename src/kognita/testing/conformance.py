"""The conformance invariants themselves.

Subclass :class:`ConformanceCase` in a pack's own test suite, set ``harness``,
and these run against it. They assert *properties of the machinery*, never
domain outcomes — a pack decides what its regimes say; the kit decides that
whatever they say is decided fail-closed, cited, and evidenced.

Each case documents what breaks in the real world when it fails, because a
conformance test nobody understands gets deleted the first time it is
inconvenient.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
from sqlmodel import select

from kognita.envelope import Envelope
from kognita.evidence import verify_chain
from kognita.governance import resolve_outcome
from kognita.models import Approval, EvidenceEvent, GovernanceDecision
from kognita.vocabulary import ApprovalStatus, CheckResult, EventType, Outcome


class ConformanceCase:
    """Invariants every domain pack must satisfy.

    Provide ``harness`` (a :class:`kognita.testing.harness.Harness`) and
    ``allow_envelope`` / ``deny_envelope`` as fixtures or class attributes.
    """

    harness = None

    #: An envelope the pack should permit.
    allow_envelope: Envelope | None = None
    #: An envelope the pack should refuse.
    deny_envelope: Envelope | None = None
    #: An envelope the pack should send to human review. Optional.
    human_envelope: Envelope | None = None

    # -- resolution ----------------------------------------------------------

    def test_outcome_precedence_is_fail_closed(self):
        """One failure among any number of passes must still deny.

        If precedence were most-common or last-wins, a policy set could be
        widened simply by adding permissive rules.
        """
        from kognita.envelope import Check

        def check(result):
            return Check("c", "R", result, "citation")

        assert resolve_outcome([check(CheckResult.PASS)]) is Outcome.ALLOW
        assert (
            resolve_outcome([check(CheckResult.PASS), check(CheckResult.FAIL)])
            is Outcome.DENY
        )
        assert (
            resolve_outcome(
                [check(CheckResult.FAIL), check(CheckResult.REQUIRES_HUMAN)]
            )
            is Outcome.DENY
        )
        assert (
            resolve_outcome(
                [check(CheckResult.ESCALATE), check(CheckResult.REQUIRES_HUMAN)]
            )
            is Outcome.ESCALATE
        )
        assert (
            resolve_outcome([check(CheckResult.PASS), check(CheckResult.REQUIRES_HUMAN)])
            is Outcome.HUMAN_APPROVAL
        )

    # -- purity --------------------------------------------------------------

    def test_decide_is_pure(self):
        """``decide`` must write nothing.

        A decision function with side effects cannot be replayed to answer
        "what would this have decided?", which is the question an audit asks.
        """
        with self.harness.session() as session:
            before = len(session.exec(select(GovernanceDecision)).all())
            evidence_before = len(session.exec(select(EvidenceEvent)).all())
            self.harness.evaluate(self.allow_envelope, session)
            self.harness.evaluate(self.deny_envelope, session)
            assert len(session.exec(select(GovernanceDecision)).all()) == before
            assert len(session.exec(select(EvidenceEvent)).all()) == evidence_before

    def test_decide_is_deterministic(self):
        """The same envelope and snapshot must give the same hash every time.

        The envelope hash is what approvals bind to; if it drifted, an approval
        would stop matching the request it was granted for.
        """
        with self.harness.session() as session:
            first = self.harness.evaluate(self.allow_envelope, session)
            second = self.harness.evaluate(self.allow_envelope, session)
            assert first.envelope_hash == second.envelope_hash
            assert first.outcome == second.outcome
            assert [c.to_dict() for c in first.checks] == [
                c.to_dict() for c in second.checks
            ]

    # -- citations -----------------------------------------------------------

    def test_every_check_carries_a_citation(self):
        """A decision that cannot be traced to a rule is an assertion, not a decision."""
        with self.harness.session() as session:
            for envelope in filter(
                None, [self.allow_envelope, self.deny_envelope, self.human_envelope]
            ):
                evaluation = self.harness.evaluate(envelope, session)
                assert evaluation.checks, "a decision with no checks explains nothing"
                for check in evaluation.checks:
                    assert check.citation.strip(), f"{check.check} has no citation"

    def test_denial_explains_itself(self):
        """A refusal must name what refused it, or a user cannot act on it."""
        with self.harness.session() as session:
            evaluation = self.harness.evaluate(self.deny_envelope, session)
            assert evaluation.outcome in (Outcome.DENY, Outcome.ESCALATE)
            assert evaluation.basis(), "a denial with no basis is unactionable"

    # -- registry ------------------------------------------------------------

    def test_unregistered_agent_is_denied(self):
        """The agent registry is an allowlist, not a log.

        An unknown agent acting and being recorded afterwards is exactly what a
        registry exists to prevent.
        """
        envelope = replace(self.allow_envelope, agent_name="unregistered-rogue-agent")
        with self.harness.session() as session:
            evaluation = self.harness.evaluate(envelope, session)
            assert evaluation.outcome is Outcome.DENY
            assert any(
                c.check == "AGENT_REGISTRY" and c.result is CheckResult.FAIL
                for c in evaluation.checks
            )

    # -- recording -----------------------------------------------------------

    def test_decision_is_recorded_and_evidenced(self):
        """Every recorded decision leaves a POLICY_DECISION event."""
        with self.harness.session() as session:
            evaluation = self.harness.decide_and_record(self.allow_envelope, session)
            session.commit()
            assert evaluation.decision_id is not None

            decision = session.exec(
                select(GovernanceDecision).where(
                    GovernanceDecision.request_id == evaluation.request_id
                )
            ).one()
            assert decision.envelope_hash == evaluation.envelope_hash
            assert decision.attributes, "the attributes a decision turned on must persist"

            events = session.exec(
                select(EvidenceEvent).where(
                    EvidenceEvent.correlation_id == evaluation.request_id
                )
            ).all()
            assert any(e.event_type == EventType.POLICY_DECISION for e in events)

    def test_denied_request_is_still_evidenced(self):
        """A refusal is a decision and must be as auditable as a permission."""
        with self.harness.session() as session:
            evaluation = self.harness.decide_and_record(self.deny_envelope, session)
            session.commit()
            events = session.exec(
                select(EvidenceEvent).where(
                    EvidenceEvent.correlation_id == evaluation.request_id
                )
            ).all()
            assert any(e.event_type == EventType.POLICY_DECISION for e in events)

    def test_evidence_chain_stays_intact(self):
        """Recording several decisions must leave a verifiable chain."""
        with self.harness.session() as session:
            for envelope in filter(
                None, [self.allow_envelope, self.deny_envelope, self.human_envelope]
            ):
                self.harness.decide_and_record(envelope, session)
            session.commit()
            assert verify_chain(session) > 0

    # -- approvals -----------------------------------------------------------

    def test_human_approval_opens_a_bound_approval(self):
        """Human review must open an approval bound to this exact envelope.

        Binding to the hash is what stops an approval for one request being
        replayed against a materially different one.
        """
        if self.human_envelope is None:
            pytest.skip("pack declares no human-approval scenario")
        with self.harness.session() as session:
            before = len(session.exec(select(Approval)).all())
            evaluation = self.harness.decide_and_record(self.human_envelope, session)
            session.commit()
            assert evaluation.outcome is Outcome.HUMAN_APPROVAL

            approvals = session.exec(select(Approval)).all()
            assert len(approvals) == before + 1
            opened = next(
                a for a in approvals if a.decision_id == evaluation.decision_id
            )
            assert opened.status is ApprovalStatus.PENDING
            assert opened.envelope_hash == evaluation.envelope_hash
            assert opened.expires_at > self.harness.now


# ── Bundled fixture pack and suite (for pytest --pyargs kognita.testing.conformance)
from datetime import timedelta

from kognita.models import Policy
from kognita.registry import register
from kognita.rules import build_registry
from kognita.testing.harness import FROZEN_NOW, Harness

_START = FROZEN_NOW - timedelta(days=365)

_SUBJECTS = {
    "open": {"id": "open", "tier": "standard", "cleared": True},
    "sealed": {"id": "sealed", "tier": "sealed", "cleared": False},
}


class _MinimalPack:
    """The smallest pack that exercises every invariant in the kit."""

    name = "minimal"

    def load_subjects(self, envelope: Envelope, session=None) -> dict:
        subjects = {}
        for kind, ref in envelope.all_subjects().items():
            row = _SUBJECTS.get(str(ref))
            if row is None:
                raise LookupError(f"{kind} {ref!r} not found")
            subjects[kind] = row
        return subjects

    def resolve_attributes(self, envelope: Envelope, subjects: dict) -> dict:
        record = subjects.get("record", {})
        return {
            "site": envelope.actor_location,
            "tier": record.get("tier", "standard"),
            "cleared": bool(record.get("cleared")),
        }

    def rules(self) -> dict:
        return build_registry()

    def engages(self, policy: Policy, context) -> bool:
        return True


def _seed(session) -> None:
    session.add(
        Policy(
            regime="HOUSE",
            rule_type="ATTRIBUTE_ALLOWLIST",
            rule={"allow": {"tier": ["standard"]}, "on_violation": "fail"},
            citation="House Rules s1",
            effective_from=_START,
        )
    )
    session.add(
        Policy(
            regime="HOUSE",
            rule_type="REQUIRES_HUMAN_APPROVAL",
            rule={"tools": ["publish"]},
            citation="House Rules s2",
            effective_from=_START,
        )
    )
    register(session, name="demo-agent", owner_exec="Head of Nothing")


def _envelope(tool: str, record: str) -> Envelope:
    return Envelope(
        principal="tester",
        purpose="REVIEW",
        tool=tool,
        actor_location="SG",
        subject_type="record",
        subject_id=record,
    )


@pytest.fixture
def _harness():
    return Harness(pack=_MinimalPack(), purposes=("REVIEW",), seed=_seed)


class TestBundledConformance(ConformanceCase):
    """The kit, run against the minimal pack.

    Run via::

        pytest --pyargs kognita.testing.conformance
    """

    @pytest.fixture(autouse=True)
    def _bind(self, _harness):
        self.harness = _harness
        self.allow_envelope = _envelope("read", "open")
        self.deny_envelope = _envelope("read", "sealed")
        self.human_envelope = _envelope("publish", "open")


__all__ = ["ConformanceCase"]
