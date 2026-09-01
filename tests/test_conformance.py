"""The 15 conformance shapes, ported onto the neutral fixture pack.

The TypeScript original's suite was written against a wealth-management domain
that now lives in another repository. Each of its assertions is reproduced here
by *shape*, so the machinery stays proven without the domain being present. The
mapping is documented in ``tests/fixtures/demo_pack.py``.
"""
from __future__ import annotations

import pytest
from sqlmodel import select

from fixtures import demo_pack as dp

from kognita.core.envelope import Envelope
from kognita.core.evidence import verify_chain
from kognita.core.models import Approval, EvidenceEvent
from kognita.core.registry import register, set_kill_switch
from kognita.core.vocabulary import (
    ApprovalStatus,
    CheckResult,
    EventType,
    Outcome,
)
from kognita.testing.conformance import ConformanceCase
from kognita.testing.harness import FROZEN_NOW, Harness


def _seed(session):
    dp.seed_policies(session, now=FROZEN_NOW)
    register(session, name="eligibility-assistant", owner_exec="Head of Data Governance")
    register(session, name="dossier-agent", owner_exec="Head of Research Ops")


@pytest.fixture
def harness():
    return Harness(pack=dp.DemoPack(), purposes=dp.PURPOSES, seed=_seed)


# ── PDP shapes (ported from engine.test.ts) ──────────────────────────────────


def test_flagship_two_regimes_deny_one_request(harness):
    """The flagship: two independent regimes each fail the same request.

    Ported from "denies the tri-jurisdiction case". The point is not that one
    rule fired but that *both* did and neither was masked by the other — a
    decision is the conjunction of every engaged regime, not the first hit.
    """
    envelope = dp.envelope(
        "check_dataset_eligibility",
        purpose="ELIGIBILITY_CHECK",
        site="AE",
        subject="1",
        dataset="1",
    )
    with harness.session() as session:
        evaluation = harness.evaluate(envelope, session)

    assert evaluation.outcome is Outcome.DENY
    failing = {c.regime for c in evaluation.failures()}
    assert "ORIGIN_SITE" in failing
    assert "REQUESTER_SITE" in failing
    for check in evaluation.checks:
        assert check.citation


def test_control_case_allows(harness):
    """The control: same tool, unengaged regimes, permitted.

    Without this the DENY above proves nothing — a decision point that refuses
    everything is trivially "safe" and entirely useless.
    """
    envelope = dp.envelope(
        "check_dataset_eligibility",
        purpose="ELIGIBILITY_CHECK",
        site="SG",
        subject="2",
        dataset="2",
    )
    with harness.session() as session:
        assert harness.evaluate(envelope, session).outcome is Outcome.ALLOW


def test_human_review_forced_for_restricted_region_subject(harness):
    """Drafting is permitted; releasing needs a person. Ported from the Art 22 case."""
    envelope = dp.envelope(
        "draft_publication", purpose="DOSSIER_PREP", site="SG", subject="1"
    )
    with harness.session() as session:
        before = len(session.exec(select(Approval)).all())
        evaluation = harness.decide_and_record(envelope, session)
        session.commit()

        assert evaluation.outcome is Outcome.HUMAN_APPROVAL
        assert any(
            c.result is CheckResult.REQUIRES_HUMAN and c.regime == "ETHICS_BOARD"
            for c in evaluation.checks
        )

        approvals = session.exec(select(Approval)).all()
        assert len(approvals) == before + 1
        opened = next(a for a in approvals if a.decision_id == evaluation.decision_id)
        assert opened.status is ApprovalStatus.PENDING
        assert opened.envelope_hash == evaluation.envelope_hash
        assert opened.expires_at > harness.now


def test_unregistered_agent_fails_closed(harness):
    envelope = dp.envelope(
        "get_subject_profile", site="SG", subject="2", agent="unregistered-rogue-agent"
    )
    with harness.session() as session:
        evaluation = harness.evaluate(envelope, session)
    assert evaluation.outcome is Outcome.DENY
    assert any(
        c.check == "AGENT_REGISTRY" and c.result is CheckResult.FAIL
        for c in evaluation.checks
    )


def test_kill_switch_denies_every_action(harness):
    """An engaged kill switch stops the agent before any policy is consulted."""
    envelope = dp.envelope(
        "get_subject_profile", site="SG", subject="2", agent="dossier-agent"
    )
    with harness.session() as session:
        assert harness.evaluate(envelope, session).outcome is Outcome.ALLOW

        set_kill_switch(
            session,
            "dossier-agent",
            True,
            evidence=harness.evidence,
            actor_id="risk.officer",
            reason="conformance test",
        )
        session.commit()

        evaluation = harness.evaluate(envelope, session)
        assert evaluation.outcome is Outcome.DENY
        assert any(c.check == "KILL_SWITCH" for c in evaluation.checks)


def test_attributes_recorded_on_every_decision(harness):
    """The attributes a decision turned on must survive it.

    Ported from "records the four-tuple". A decision whose inputs were not kept
    cannot be re-examined later, which is most of the value of keeping it.
    """
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        evaluation = harness.evaluate(envelope, session)
    attrs = evaluation.attributes
    assert attrs["requester_site"] == "SG"
    assert attrs["subject_region"] == "SG"
    assert attrs["home_site"] == "SG"
    assert attrs["processing_site"]


# ── Retrieval shapes (ported from retrieval.test.ts) ─────────────────────────


@pytest.fixture
def indexed(harness):
    from kognita.core.retrieval import index_item
    from kognita.core.vocabulary import Classification

    with harness.session() as session:
        for item in dp.KNOWLEDGE:
            index_item(
                session,
                title=item["title"],
                body=item["body"],
                embedder=harness.embedder,
                kind=item["kind"],
                classification=Classification(item["classification"]),
                zones=item["zones"],
                source_label=item["source_label"],
            )
        session.commit()
    return harness


def test_restricted_items_never_reach_a_non_admin(indexed):
    """Whatever the query, C3 must not surface below the ceiling."""
    from kognita.core.retrieval import retrieve

    with indexed.session() as session:
        results = retrieve(
            session,
            "Al-Rashid register consolidation linkage keys",
            zone="AE",
            embedder=indexed.embedder,
            is_admin=False,
        )
    assert all(r.classification.value != "C3" for r in results)


def test_the_same_item_reaches_an_admin(indexed):
    """The ceiling is the only thing excluding it — not relevance."""
    from kognita.core.retrieval import retrieve

    with indexed.session() as session:
        results = retrieve(
            session,
            "Al-Rashid register consolidation linkage keys",
            zone="AE",
            embedder=indexed.embedder,
            is_admin=True,
        )
    assert any(r.classification.value == "C3" for r in results)


def test_zone_partition_is_enforced(indexed):
    """An AE-scoped document is invisible from SG even to an admin."""
    from kognita.core.retrieval import retrieve

    with indexed.session() as session:
        results = retrieve(
            session,
            "Requester Site conduct classification inducements",
            zone="SG",
            embedder=indexed.embedder,
            is_admin=True,
        )
    assert all(not r.title.startswith("Requester Site") for r in results)


def test_entitlement_filtering_precedes_scoring(indexed):
    """The candidate count recorded in evidence must already be filtered.

    Filtering after ranking would mean the ranking had seen the item, and a
    top-k that silently drops entries leaks the fact that they exist.
    """
    from kognita.core.retrieval import entitled_items, retrieve
    from kognita.core.vocabulary import Classification

    with indexed.session() as session:
        all_items = len(session.exec(select(__import__(
            "kognita.core.models", fromlist=["KnowledgeItem"]
        ).KnowledgeItem)).all())
        entitled = entitled_items(session, zone="SG", ceiling=Classification.C2)
        assert len(entitled) < all_items

        retrieve(
            session,
            "anything at all",
            zone="SG",
            embedder=indexed.embedder,
            evidence=indexed.evidence,
            correlation_id="probe",
            is_admin=False,
        )
        session.commit()
        event = session.exec(
            select(EvidenceEvent).where(EvidenceEvent.correlation_id == "probe")
        ).one()
        assert event.payload["candidate_count"] == len(entitled)


def test_embedding_sanity(harness):
    """Identical text scores ~1, unrelated text well below, vectors normalised."""
    import numpy as np

    from kognita.core.embedding import cosine

    a = harness.embedder.embed("cross-site eligibility for restricted datasets")
    b = harness.embedder.embed("cross-site eligibility for restricted datasets")
    c = harness.embedder.embed("quarterly fee reconciliation ledger")

    assert cosine(a, b) > 0.99
    assert cosine(a, c) < 0.3
    assert float(np.linalg.norm(a)) == pytest.approx(1.0, abs=1e-6)
    assert len(a) == 256


def test_lexical_overlap_rewards_shared_terms():
    from kognita.core.embedding import lexical_overlap

    high = lexical_overlap("genomic linkage set", "genomic linkage set keys registry")
    low = lexical_overlap("genomic linkage set", "ledger reconciliation fee")
    assert high > low


# ── Broker shapes (ported from broker.test.ts) ───────────────────────────────


def test_question_mentioning_subjects_stays_on_the_knowledge_route(indexed):
    """Routing is by scope, never by wording.

    Ported directly. Keyword routing would let a knowledge question that merely
    says "subjects" authorise subject-profile access — the exact confusion the
    envelope exists to prevent.
    """
    from kognita.core.broker import KNOWLEDGE, ask

    envelope = dp.envelope("get_house_guidance", purpose="PUBLICATION", site="SG")
    with indexed.session() as session:
        answer = ask(
            session,
            "What do the rules say about automated release for restricted-region subjects?",
            envelope,
            pack=indexed.pack,
            embedder=indexed.embedder,
            evidence=indexed.evidence,
            purposes=dp.PURPOSES,
            as_of=indexed.now,
        )
        session.commit()

    assert answer.route == KNOWLEDGE
    assert answer.outcome is Outcome.ALLOW
    assert answer.results


def test_subject_and_dataset_in_scope_routes_to_eligibility(indexed):
    from kognita.core.broker import ELIGIBILITY, ask

    envelope = dp.envelope(
        "check_dataset_eligibility",
        purpose="ELIGIBILITY_CHECK",
        site="AE",
        subject="1",
        dataset="1",
    )
    with indexed.session() as session:
        answer = ask(
            session,
            "Can I share this dataset with the subject?",
            envelope,
            pack=indexed.pack,
            embedder=indexed.embedder,
            evidence=indexed.evidence,
            purposes=dp.PURPOSES,
            as_of=indexed.now,
        )
        session.commit()
    assert answer.route == ELIGIBILITY


def test_denial_is_fail_closed_with_its_basis(indexed):
    """Zero results, and an explanation. Both halves matter."""
    from kognita.core.broker import ask

    envelope = dp.envelope(
        "check_dataset_eligibility",
        purpose="ELIGIBILITY_CHECK",
        site="AE",
        subject="1",
        dataset="1",
    )
    with indexed.session() as session:
        answer = ask(
            session,
            "Share the genomic linkage set?",
            envelope,
            pack=indexed.pack,
            embedder=indexed.embedder,
            evidence=indexed.evidence,
            purposes=dp.PURPOSES,
            as_of=indexed.now,
        )
        session.commit()

    assert answer.outcome is Outcome.DENY
    assert answer.results == []
    assert len(answer.summary) > 1
    assert answer.citations


def test_subject_context_route_reports_graph_anchoring(indexed):
    from kognita.core.broker import SUBJECT_CONTEXT, ask

    envelope = dp.envelope("get_subject_profile", purpose="DOSSIER_PREP", site="SG", subject="2")

    def subgraph(_envelope, _session):
        return {"nodes": 7, "edges": 9}

    with indexed.session() as session:
        answer = ask(
            session,
            "Prepare the dossier",
            envelope,
            pack=indexed.pack,
            embedder=indexed.embedder,
            evidence=indexed.evidence,
            purposes=dp.PURPOSES,
            subgraph=subgraph,
            as_of=indexed.now,
        )
        session.commit()

    assert answer.route == SUBJECT_CONTEXT
    assert answer.graph is not None
    assert answer.graph["nodes"] > 0


# ── The kit, run against this pack ───────────────────────────────────────────


class TestDemoPackConformance(ConformanceCase):
    """The portable kit, exercised against the fixture pack."""

    @pytest.fixture(autouse=True)
    def _bind(self, harness):
        self.harness = harness
        self.allow_envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
        self.deny_envelope = dp.envelope(
            "check_dataset_eligibility",
            purpose="ELIGIBILITY_CHECK",
            site="AE",
            subject="1",
            dataset="1",
        )
        self.human_envelope = dp.envelope(
            "draft_publication", purpose="DOSSIER_PREP", site="SG", subject="1"
        )
