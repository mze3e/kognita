"""Risk-based review queue and coverage.

Roadmap 0.4 item 13. No network and no API key.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest
from sqlmodel import select

from kognita.evidence import recompute_event_hash, verify_chain
from kognita.interaction import attach, open_interaction
from kognita.models import EvidenceEvent, ReviewQueueItem
from kognita.review import (
    REVIEW_FINDINGS,
    ReviewError,
    coverage_report,
    define_risk_criteria,
    queue_oversight_flags,
    record_final_decision,
    record_review_finding,
    review_blocks_delivery,
    review_flags,
    screen_interaction,
    screen_response,
)
from kognita.vocabulary import EventType
from test_usecases import NOW, _activate, _open, _spec, _tier2

LATER = NOW + timedelta(minutes=5)


def _ready(session, evidence, *, rate=0.05, usecase_id="research-summary", tier2=False):
    _open(session, evidence)
    if tier2:
        spec = _tier2(review_sampling_rate=rate, usecase_id=usecase_id)
    else:
        spec = _spec(review_sampling_rate=rate, usecase_id=usecase_id)
    _activate(session, evidence, spec)
    define_risk_criteria(
        session,
        spec.usecase_id,
        product_complexity_at="high",
        vulnerability_indicators=("bereavement",),
        profile_deviation_is_high_risk=True,
        minimum_classifier_confidence=0.8,
        timing="before delivery",
        evidence=evidence,
        actor_id="Priya Shah",
        now=NOW,
    )
    return spec


def _journey(session, evidence, interaction_id, use_case="research-summary"):
    return open_interaction(
        session,
        interaction_id=interaction_id,
        client_id="client-1",
        use_case=use_case,
        evidence=evidence,
        now=NOW,
    )


def _respond(session, evidence, interaction, correlation, response_hash, confidence=0.99):
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
        payload={
            "sent": True,
            "response_hash": response_hash,
            "model": "summary",
            "model_version": "1",
            "prompt_template_version": "prompt-1",
            "response_classifier": {
                "model": "pattern",
                "version": "1",
                "label": "C1",
                "confidence": confidence,
                "input_hash": response_hash,
            },
        },
    )


def _low():
    return {
        "product_complexity": "low",
        "vulnerability_indicators": (),
        "deviates_from_profile": False,
    }


def _bucket(interaction_id: str, version: int = 1) -> int:
    digest = hashlib.sha256(f"{interaction_id}\n{version}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


def _ids_for_rate(rate: float) -> tuple[str, str]:
    threshold = int(round(rate * 10_000))
    inside = outside = None
    index = 0
    while inside is None or outside is None:
        name = f"interaction-{index}"
        bucket = _bucket(name)
        if bucket < threshold and inside is None:
            inside = name
        if bucket >= threshold and outside is None:
            outside = name
        index += 1
    return inside, outside


def _screens(session):
    return [
        event
        for event in session.exec(select(EvidenceEvent)).all()
        if event.event_type == EventType.REVIEW
        and (event.payload or {}).get("action") == "SCREENED"
    ]


def test_missing_criteria_and_missing_confidence_are_queued(session, evidence):
    _open(session, evidence)
    spec = _spec()
    _activate(session, evidence, spec)
    interaction = _journey(session, evidence, "interaction-gap")
    attach(
        session,
        interaction.id,
        evidence=evidence,
        correlation_id="req-gap",
        now=NOW,
    )
    evidence.emit(
        session,
        correlation_id="req-gap",
        event_type=EventType.MODEL_CALL,
        payload={
            "sent": True,
            "response_hash": "output-gap",
            "model": "summary",
            "model_version": "1",
        },
    )
    screened = screen_interaction(
        session,
        interaction.id,
        evidence=evidence,
        actor_id="Priya Shah",
        now=NOW,
        **_low(),
    )
    assert screened["queued"] is True
    assert "no risk criteria" in screened["screened"][0]["reasons"]
    assert screened["screened"][0]["citation"]
    assert review_blocks_delivery(session, interaction.id)


def test_high_risk_is_queued_and_low_risk_is_sampled(session, evidence):
    _ready(session, evidence, rate=0.05)
    inside, outside = _ids_for_rate(0.05)
    low = _journey(session, evidence, outside)
    high = _journey(session, evidence, inside)
    _respond(session, evidence, low, "req-low", "output-low")
    _respond(session, evidence, high, "req-high", "output-high")
    quiet = screen_interaction(
        session, low.id, evidence=evidence, actor_id="Priya Shah", now=NOW, **_low()
    )
    assert quiet["queued"] is False
    assert quiet["screened"][0]["citation"].startswith("Risk-based review: low-risk")
    loud = screen_interaction(
        session,
        high.id,
        evidence=evidence,
        actor_id="Priya Shah",
        now=NOW,
        product_complexity="critical",
        vulnerability_indicators=(),
        deviates_from_profile=False,
    )
    assert loud["queued"] is True
    assert "product complexity" in loud["screened"][0]["reasons"]
    watched = screen_response(
        session,
        low.id,
        evidence=evidence,
        actor_id="Priya Shah",
        output_hash="output-low",
        vulnerability_indicators=("bereavement",),
        product_complexity="low",
        deviates_from_profile=False,
        now=NOW,
    )
    assert watched["queued"] is True
    assert any("bereavement" in reason for reason in watched["reasons"])
    ignored = screen_response(
        session,
        high.id,
        evidence=evidence,
        actor_id="Priya Shah",
        output_hash="output-high",
        vulnerability_indicators=("language",),
        product_complexity="low",
        deviates_from_profile=False,
        now=NOW + timedelta(seconds=1),
    )
    # The first screen queued this interaction for product complexity. A later
    # low-risk screen does not take it off the queue, and an indicator the
    # criteria do not watch is not itself a reason.
    assert "vulnerability indicator language" not in ignored["reasons"]
    row = session.exec(select(ReviewQueueItem)).all()
    assert len([item for item in row if item.interaction_id == high.id]) == 1
    assert row[0].reasons == ["product complexity"] or any(
        item.reasons == ["product complexity"] for item in row
    )


def test_sampling_is_deterministic_and_a_full_rate_queues(session, evidence):
    _ready(session, evidence, rate=1.0)
    inside, _outside = _ids_for_rate(0.05)
    interaction = _journey(session, evidence, inside)
    _respond(session, evidence, interaction, "req-1", "output-1")
    first = screen_interaction(
        session, interaction.id, evidence=evidence, actor_id="Priya Shah", now=NOW, **_low()
    )
    second = screen_interaction(
        session, interaction.id, evidence=evidence, actor_id="Priya Shah", now=NOW, **_low()
    )
    assert first["queued"] is True
    assert first["screened"][0]["risk"] == "sampled"
    assert second["screened"][0] == first["screened"][0]
    assert len(_screens(session)) == 1


def test_a_review_outcome_is_linked_and_a_confirmed_problem_is_flagged(session, evidence):
    _ready(session, evidence, rate=1.0)
    interaction = _journey(session, evidence, "interaction-review")
    _respond(session, evidence, interaction, "req-1", "output-1")
    screen_interaction(
        session, interaction.id, evidence=evidence, actor_id="Priya Shah", now=NOW, **_low()
    )
    screen_event = _screens(session)[0]
    screen_hash = screen_event.event_hash
    other = _journey(session, evidence, "interaction-cold")
    with pytest.raises(ReviewError, match="not on the review queue"):
        record_review_finding(
            session,
            other.id,
            reviewer="Noel Independent",
            outcome="cleared",
            finding="nothing to review",
            evidence=evidence,
            now=LATER,
        )
    assert review_blocks_delivery(session, interaction.id)
    cleared_hold = record_review_finding(
        session,
        interaction.id,
        reviewer="Noel Independent",
        outcome="cleared",
        finding="The note matches the house view",
        evidence=evidence,
        now=NOW + timedelta(seconds=30),
    )
    assert cleared_hold["outcome"] == "cleared"
    assert cleared_hold["citation"]
    assert review_blocks_delivery(session, interaction.id) is None
    assert review_flags(session, as_of=LATER) == ()
    # A later confirmed problem needs a new place on the queue.
    screen_interaction(
        session,
        interaction.id,
        evidence=evidence,
        actor_id="Priya Shah",
        now=NOW + timedelta(seconds=40),
        product_complexity="critical",
        vulnerability_indicators=(),
        deviates_from_profile=False,
    )
    record_review_finding(
        session,
        interaction.id,
        reviewer="Noel Independent",
        outcome="confirmed problem",
        finding="The note recommends a product outside the profile",
        evidence=evidence,
        now=NOW + timedelta(seconds=50),
    )
    flags = review_flags(session, as_of=LATER)
    assert len(flags) == 1
    assert flags[0]["model_version"] == "1"
    assert flags[0]["citation"]
    assert review_blocks_delivery(session, interaction.id)
    assert recompute_event_hash(screen_event) == screen_hash
    assert screen_event.payload["action"] == "SCREENED"
    assert "confirmed problem" in REVIEW_FINDINGS
    assert verify_chain(session) >= 1


def test_coverage_counts_unscreened_responses_and_time_to_review(session, evidence):
    _ready(session, evidence, rate=1.0)
    done = _journey(session, evidence, "interaction-done")
    waiting = _journey(session, evidence, "interaction-wait")
    skipped = _journey(session, evidence, "interaction-skip")
    _respond(session, evidence, done, "req-done", "output-done")
    _respond(session, evidence, waiting, "req-wait", "output-wait")
    _respond(session, evidence, skipped, "req-skip", "output-skip")
    _respond(session, evidence, skipped, "req-skip-2", "output-skip-2")
    screen_interaction(
        session, done.id, evidence=evidence, actor_id="Priya Shah", now=NOW, **_low()
    )
    screen_interaction(
        session, waiting.id, evidence=evidence, actor_id="Priya Shah", now=NOW, **_low()
    )
    screen_response(
        session,
        skipped.id,
        evidence=evidence,
        actor_id="Priya Shah",
        output_hash="output-skip",
        now=NOW,
        **_low(),
    )
    record_review_finding(
        session,
        done.id,
        reviewer="Noel Independent",
        outcome="cleared",
        finding="Reviewed",
        evidence=evidence,
        now=NOW + timedelta(seconds=30),
    )
    report = coverage_report(session, as_of=LATER)
    row = report["by_use_case"][0]
    assert row["risk_tier"] == 1
    assert row["with_model_response"] == 3
    assert row["screened"] == 2
    assert row["share_screened"] == pytest.approx(2 / 3)
    assert row["queued"] == 3
    assert row["reviewed"] == 1
    assert row["share_reviewed"] == pytest.approx(1 / 3)
    assert row["time_to_review_seconds"] == 30
    assert row["citation"]
    tier = report["by_risk_tier"][0]
    assert tier["risk_tier"] == 1
    assert tier["screened"] == row["screened"]
    empty = coverage_report(session, as_of=NOW - timedelta(days=1))
    assert empty["by_use_case"] == []


def test_time_to_review_is_unmeasured_when_nothing_has_been_reviewed(session, evidence):
    _ready(session, evidence, rate=1.0)
    interaction = _journey(session, evidence, "interaction-open")
    _respond(session, evidence, interaction, "req-1", "output-1")
    screen_interaction(
        session, interaction.id, evidence=evidence, actor_id="Priya Shah", now=NOW, **_low()
    )
    row = coverage_report(session, as_of=LATER)["by_use_case"][0]
    assert row["queued"] == 1
    assert row["reviewed"] == 0
    assert row["time_to_review_seconds"] is None
    assert "time_to_review_seconds" in row["unmeasured"]


def test_a_rubber_stamp_flag_joins_the_queue(session, evidence):
    _ready(session, evidence, rate=0.05)
    _outside = _ids_for_rate(0.05)[1]
    interaction = _journey(session, evidence, _outside)
    record_final_decision(
        session,
        interaction.id,
        outcome="approve",
        evidence=evidence,
        actor_id="RM One",
        approver="RM One",
        citation="RM review: final decision",
        now=NOW,
        rubber_stamp={
            "flagged": True,
            "reasons": ["faster than the review threshold"],
            "citation": "RM review: rubber-stamp approvals are flagged",
        },
    )
    queued = queue_oversight_flags(
        session, evidence=evidence, actor_id="Priya Shah", now=NOW
    )
    assert queued == (interaction.id,)
    again = queue_oversight_flags(
        session, evidence=evidence, actor_id="Priya Shah", now=NOW
    )
    assert again == ()
    assert review_blocks_delivery(session, interaction.id)


def test_after_delivery_does_not_hold_a_send(session, evidence):
    _open(session, evidence)
    spec = _spec(review_sampling_rate=1.0)
    _activate(session, evidence, spec)
    define_risk_criteria(
        session,
        spec.usecase_id,
        product_complexity_at="high",
        vulnerability_indicators=("bereavement",),
        profile_deviation_is_high_risk=True,
        minimum_classifier_confidence=0.8,
        timing="after delivery",
        evidence=evidence,
        actor_id="Priya Shah",
        now=NOW,
    )
    interaction = _journey(session, evidence, "interaction-after")
    _respond(session, evidence, interaction, "req-1", "output-1")
    screen_interaction(
        session,
        interaction.id,
        evidence=evidence,
        actor_id="Priya Shah",
        now=NOW,
        product_complexity="critical",
        vulnerability_indicators=(),
        deviates_from_profile=False,
    )
    assert review_blocks_delivery(session, interaction.id) is None


def test_an_omitted_vulnerability_answer_is_not_a_pass(session, evidence):
    _ready(session, evidence, rate=0.05)
    outside = _ids_for_rate(0.05)[1]
    interaction = _journey(session, evidence, outside)
    _respond(session, evidence, interaction, "req-1", "output-1")
    screened = screen_interaction(
        session,
        interaction.id,
        evidence=evidence,
        actor_id="Priya Shah",
        now=NOW,
        product_complexity="low",
        vulnerability_indicators=None,
        deviates_from_profile=False,
    )
    assert screened["queued"] is True
    assert "vulnerability indicators were not recorded" in screened["screened"][0]["reasons"]


def test_coverage_groups_two_risk_tiers(session, evidence):
    _ready(session, evidence, rate=1.0)
    _ready(session, evidence, rate=1.0, usecase_id="client-intelligence", tier2=True)
    first = _journey(session, evidence, "interaction-t1", use_case="research-summary")
    second = _journey(session, evidence, "interaction-t2", use_case="client-intelligence")
    _respond(session, evidence, first, "req-t1", "output-t1")
    _respond(session, evidence, second, "req-t2", "output-t2")
    for interaction in (first, second):
        screen_interaction(
            session,
            interaction.id,
            evidence=evidence,
            actor_id="Priya Shah",
            now=NOW,
            **_low(),
        )
    tiers = {
        row["risk_tier"]: row for row in coverage_report(session, as_of=LATER)["by_risk_tier"]
    }
    assert tiers[1]["with_model_response"] == 1
    assert tiers[2]["with_model_response"] == 1
    assert tiers[1]["share_screened"] == 1
    assert tiers[2]["share_screened"] == 1
