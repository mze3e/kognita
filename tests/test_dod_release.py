"""Claim release, fact freshness, and standard version on the output path.

Positive cases pass the gate. Bypass attempts (skip the record, skip the
release, swap the audience or the text, hide a stale fact, or name a standard
version the caller prefers) fail closed.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from conftest import NOW
from fixtures import demo_pack as dp
from fixtures.agents import register
from kognita.claims import Claim, ClaimType, ValidatorStatus, record_claim, release_claim
from kognita.communication import send_client_communication
from kognita.embedding import HashingEmbedder
from kognita.evidence import EvidenceWriter
from kognita.models import EvidenceEvent, UseCase
from kognita.retrieval import index_item, retrieve
from kognita.sources import Source, register_source
from kognita.tools import ToolRegistry, run_governed
from kognita.vocabulary import Classification, EventType, Outcome, StaleAction
from test_communication import (
    _allow,
    _channel,
    _content_policy,
    _cover,
    _open,
)
from test_gateway import MODEL, PATH, _Upstream, _body, _gateway
from test_gateway_run_controls import _setup
from test_library_execution_gates import _tier_two
from test_usecases import ensure_live_purposes
from kognita.testing.harness import FROZEN_NOW, Harness

ANSWER = "The briefing is ready."
PAST = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _response(content: str) -> bytes:
    return json.dumps({
        "model": MODEL,
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }).encode()


def _events(session, event_type):
    return session.exec(
        select(EvidenceEvent).where(EvidenceEvent.event_type == event_type).order_by(EvidenceEvent.sequence)
    ).all()


def _fresh_fact(as_of, *, refreshed=None):
    body = {
        "value": 42,
        "source": "positions-system",
        "owner": "operations",
        "as_of": as_of.isoformat(),
        "classification": "C2",
        "entitlement": "positions:read",
        "data_type": "positions",
    }
    if refreshed is not None:
        body["refreshed"] = refreshed
    return body


def _with_policy(session, *, action=StaleAction.SUPPRESS, standard_version=""):
    """Put freshness, and an optional standard version, on the live use case."""
    ensure_live_purposes(session, EvidenceWriter(session.get_bind(), clock=lambda: NOW), ("COLLABORATION",), at=NOW)
    row = session.exec(
        select(UseCase).where(UseCase.usecase_id == "COLLABORATION", UseCase.effective_to.is_(None))
    ).one()
    record = dict(row.record)
    record["version"] = row.version + 1
    record["freshness"] = {"positions": "1h"}
    record["stale_action"] = action.value
    if standard_version:
        record["standard_version"] = standard_version
    row.effective_to = NOW
    session.add(UseCase(
        usecase_id="COLLABORATION",
        version=row.version + 1,
        record=record,
        stale_action=action,
        effective_from=NOW,
    ))
    session.flush()
    return row.version + 1


def _seed(session):
    dp.seed_policies(session, now=NOW)
    register(session, name="eligibility-assistant", owner_exec="Head of Data Governance")


@pytest.fixture
def harness():
    return Harness(pack=dp.DemoPack(), purposes=dp.PURPOSES, seed=_seed)


def test_gateway_refuses_material_text_until_a_claim_is_released(session, evidence):
    _ignored, _backend, headers = _setup(session, evidence)
    _tier_two(session, evidence)
    upstream = _Upstream(_response(ANSWER))
    gateway = _gateway(session.get_bind(), evidence, upstream)
    headers = {**headers, "standard_version": "999"}
    denied = gateway.handle("POST", PATH, headers, _body("hello"), session=session)
    assert denied.status == 403
    assert denied.evaluation.outcome is Outcome.DENY
    assert any(check.check == "C1-UNKNOWN" for check in denied.evaluation.checks)
    assert ANSWER.encode() not in denied.body

    _cover(session, evidence, ANSWER, audience="rm", usecase_id="COLLABORATION")
    allowed = gateway.handle("POST", PATH, headers, _body("hello"), session=session)
    assert allowed.status == 200, allowed.body
    assert ANSWER.encode() in allowed.body
    model = _events(session, EventType.MODEL_CALL)[-1]
    assert model.payload["standard_version"] != "999"
    assert model.payload["standard_version"]
    assert any(
        event.payload.get("action") == "release" and event.payload.get("allowed") is True
        for event in _events(session, EventType.CLAIM)
    )


@pytest.mark.parametrize("bypass", ["unrecorded", "unreleased", "wrong audience", "other text", "forged"])
def test_gateway_claim_bypasses_fail_closed(session, evidence, bypass):
    _ignored, _backend, headers = _setup(session, evidence)
    _tier_two(session, evidence)
    if bypass == "unreleased":
        record_claim(
            session,
            Claim("c-open", "client-1", ANSWER, ClaimType.INFERENCE, validator_status=ValidatorStatus.VERIFIED),
            evidence=evidence,
            interaction_id="i",
        )
    elif bypass == "wrong audience":
        _cover(session, evidence, ANSWER, audience="client", usecase_id="COLLABORATION")
    elif bypass == "other text":
        _cover(session, evidence, "A different sentence.", audience="rm", usecase_id="COLLABORATION")
    elif bypass == "forged":
        evidence.emit(
            session,
            correlation_id="i",
            event_type=EventType.CLAIM,
            payload={
                "action": "release",
                "claim_id": "forged",
                "audience": "rm",
                "allowed": True,
                "reason": "C1-LABELLED",
                "claim_hash": "0" * 64,
            },
        )
    upstream = _Upstream(_response(ANSWER))
    gateway = _gateway(session.get_bind(), evidence, upstream)
    response = gateway.handle("POST", PATH, headers, _body("hello"), session=session)
    assert response.status == 403
    assert any(check.check == "C1-UNKNOWN" for check in response.evaluation.checks)
    assert ANSWER.encode() not in response.body


def test_communication_refuses_a_house_view_without_a_released_claim(harness):
    with harness.session() as session:
        interaction = _open(session, harness.evidence)
        _content_policy(session)
        _channel(session, harness.evidence)
        _allow(session, harness.evidence, interaction.id)
        denied = send_client_communication(
            session,
            interaction.id,
            evidence=harness.evidence,
            channel="Secure Message",
            recipient_ref="client-1-ref",
            content="The house view downgraded the banks.",
            level="house view",
            provenance="human",
            actor_id="RM",
            research_attribution="house view 2026-10-01",
            now=FROZEN_NOW,
        )
        _cover(session, harness.evidence, "The house view downgraded the banks.", interaction_id=interaction.id)
        allowed = send_client_communication(
            session,
            interaction.id,
            evidence=harness.evidence,
            channel="Secure Message",
            recipient_ref="client-1-ref",
            content="The house view downgraded the banks.",
            level="house view",
            provenance="human",
            actor_id="RM",
            research_attribution="house view 2026-10-01",
            now=FROZEN_NOW,
        )
    assert denied["sent"] is False
    assert denied["outcome"] == "DENY"
    assert any(check["check"] == "C1-UNKNOWN" and check["result"] == "fail" for check in denied["checks"])
    assert allowed["sent"] is True
    assert allowed["outcome"] == "ALLOW"


def test_tool_and_retrieval_do_not_use_stale_facts(session, evidence):
    _with_policy(session)
    _seed(session)
    session.flush()
    stale = _fresh_fact(NOW - timedelta(hours=2))
    fresh = _fresh_fact(NOW)
    registry = ToolRegistry()
    seen = {}

    def _read(envelope, evaluation, _session):
        body = stale if envelope.arguments.get("which") == "stale" else fresh
        seen["raw"] = dict(body)
        return body

    registry.register("get_subject_profile", _read, classification=Classification.C1)
    denied = run_governed(
        session,
        dp.envelope("get_subject_profile", arguments={"which": "stale"}),
        registry=registry,
        evidence=evidence,
        pack=dp.DemoPack(),
        purposes=dp.PURPOSES,
        as_of=NOW,
        now=NOW,
    )
    assert denied.evaluation.outcome is Outcome.ALLOW
    assert seen["raw"]["value"] == 42
    assert denied.data is None or denied.data.get("value") != 42
    admitted = _events(session, EventType.RETRIEVAL)
    assert any(event.payload.get("reason_code") == "FACT_STALE_SUPPRESSED" for event in admitted)

    allowed = run_governed(
        session,
        dp.envelope("get_subject_profile", arguments={"which": "fresh"}),
        registry=registry,
        evidence=evidence,
        pack=dp.DemoPack(),
        purposes=dp.PURPOSES,
        as_of=NOW,
        now=NOW,
    )
    assert allowed.data["value"] == 42
    assert allowed.data["warning"] is False

    embedder = HashingEmbedder()
    index_item(
        session, title="book", body=json.dumps(stale), embedder=embedder,
        zones=["SG"], kind="FACT", source_label="positions-system",
    )
    index_item(
        session, title="note", body="positions are discussed in the note", embedder=embedder,
        zones=["SG"], kind="DOCUMENT", source_label="minutes",
    )
    hits = retrieve(
        session, "positions", zone="SG", embedder=embedder, evidence=evidence,
        correlation_id="r", use_case="COLLABORATION",
    )
    assert [hit.kind for hit in hits] == ["DOCUMENT"]
    assert all(hit.warning is False for hit in hits)


def test_stale_fact_refresh_and_warning_are_explicit(session, evidence):
    _with_policy(session, action=StaleAction.WARN)
    _seed(session)
    registry = ToolRegistry()
    stale = _fresh_fact(NOW - timedelta(hours=2))

    def _read(_envelope, _evaluation, _session):
        return dict(stale)

    registry.register("get_subject_profile", _read, classification=Classification.C1)
    warned = run_governed(
        session, dp.envelope("get_subject_profile"), registry=registry, evidence=evidence,
        pack=dp.DemoPack(), purposes=dp.PURPOSES, as_of=NOW, now=NOW,
    )
    assert warned.data["warning"] is True
    assert warned.data["value"] == 42
    assert any(event.payload.get("reason_code") == "FACT_STALE_WARNING" for event in _events(session, EventType.RETRIEVAL))

    _with_policy(session, action=StaleAction.REFRESH)
    refreshed = _fresh_fact(NOW)
    refreshed.pop("data_type", None)

    def _refresh(_envelope, _evaluation, _session):
        return _fresh_fact(NOW - timedelta(hours=2), refreshed=refreshed)

    registry.register("get_dataset_snapshot", _refresh, classification=Classification.C1)
    updated = run_governed(
        session, dp.envelope("get_dataset_snapshot"), registry=registry, evidence=evidence,
        pack=dp.DemoPack(), purposes=dp.PURPOSES, as_of=NOW, now=NOW,
    )
    assert updated.data["value"] == 42
    assert updated.data["warning"] is False
    assert datetime.fromisoformat(updated.data["as_of"]) == NOW
    assert any(event.payload.get("reason_code") == "FACT_REFRESHED" for event in _events(session, EventType.RETRIEVAL))


def _stamp_standard(session, standard_version: str) -> None:
    row = session.exec(
        select(UseCase).where(UseCase.usecase_id == "COLLABORATION", UseCase.effective_to.is_(None))
    ).one()
    record = dict(row.record)
    record["version"] = row.version + 1
    record["standard_version"] = standard_version
    row.effective_to = NOW
    session.add(UseCase(
        usecase_id="COLLABORATION",
        version=row.version + 1,
        record=record,
        stale_action=row.stale_action,
        effective_from=NOW,
    ))
    session.flush()


def test_outputs_record_the_in_force_standard_not_a_caller_version(session, evidence):
    version = _with_policy(session, standard_version="std-2")
    _seed(session)
    registry = ToolRegistry()
    registry.register("get_subject_profile", lambda *_a: {"marked": True}, classification=Classification.C1)
    run_governed(
        session,
        dp.envelope("get_subject_profile", arguments={"standard_version": "1"}),
        registry=registry,
        evidence=evidence,
        pack=dp.DemoPack(),
        purposes=dp.PURPOSES,
        as_of=NOW,
        now=NOW,
    )
    tool = _events(session, EventType.TOOL_CALL)[-1]
    assert tool.payload["standard_version"] == "std-2"
    assert tool.payload["standard_version"] != "1"
    assert version > 1


def test_a_closed_standard_is_not_the_one_recorded(session, evidence):
    _ignored, _backend, headers = _setup(session, evidence)
    _stamp_standard(session, "std-old")
    _stamp_standard(session, "std-new")
    upstream = _Upstream(b'{"model":"gateway-model","choices":[]}')
    gateway = _gateway(session.get_bind(), evidence, upstream)
    response = gateway.handle("POST", PATH, headers, _body("hello"), session=session)
    assert response.status == 200, response.body
    model = [event for event in _events(session, EventType.MODEL_CALL) if event.payload.get("sent")][-1]
    assert model.payload["standard_version"] == "std-new"
    assert model.payload["standard_version"] != "std-old"


def test_a_superseded_source_stops_the_gateway_sending_the_released_text(session, evidence):
    from kognita.sources import supersede_source

    _ignored, _backend, headers = _setup(session, evidence)
    _tier_two(session, evidence)
    _cover(session, evidence, ANSWER, audience="rm", usecase_id="COLLABORATION")
    gateway = _gateway(session.get_bind(), evidence, _Upstream(_response(ANSWER)))
    assert gateway.handle("POST", PATH, headers, _body("hello"), session=session).status == 200
    supersede_source(session, "desk", version="v1", by="v2", evidence=evidence, interaction_id="hv")
    after = gateway.handle("POST", PATH, headers, _body("hello"), session=session)
    assert after.status == 403
    assert ANSWER.encode() not in after.body


def test_release_cover_does_not_verify_the_whole_chain(session, evidence, monkeypatch):
    import kognita.claims as claims

    _cover(session, evidence, ANSWER, audience="rm", usecase_id="COLLABORATION")

    def boom(_session):
        raise AssertionError("the request path must not verify the chain")

    monkeypatch.setattr(claims, "_verify_chain", boom)
    assert claims._release_covers(session, ANSWER, audience="rm", usecase_id="COLLABORATION").check == "C1-LABELLED"


def test_a_value_with_an_as_of_date_but_no_owner_is_not_passed_through(session, evidence):
    from kognita.facts import _admit_observed

    payload = {"value": 42, "source": "positions-system", "as_of": PAST.isoformat()}
    ensure_live_purposes(session, evidence, ("COLLABORATION",), at=NOW)
    result = _admit_observed(session, payload, use_case="COLLABORATION", evidence=evidence,
                             correlation_id="c", now=NOW)
    assert result is None


def _scoped_cover(session, evidence, scope):
    """A tier 1 use case with ``scope``, one claim for client-1 released in interaction i-1."""
    from datetime import datetime, timezone

    from kognita.models import UseCase

    session.add(UseCase(
        usecase_id="scoped", version=1,
        record={"usecase_id": "scoped", "approved_sources": ["desk"], "proposed_tier": 1,
                "version": 1, "release_scope": scope},
        effective_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
    ))
    session.flush()
    register_source(session, Source("desk", "policy", "v1", "owner", PAST), evidence=evidence,
                    interaction_id="i-1")
    _cover(session, evidence, "Scoped text.", audience="rm", usecase_id="scoped",
           interaction_id="i-1", client_id="client-1")


@pytest.mark.parametrize(
    "scope, asked, covered",
    [
        ("usecase", dict(usecase_id="scoped"), True),
        ("usecase", dict(usecase_id="other"), False),
        ("usecase", dict(), False),
        ("client", dict(usecase_id="scoped", client_id="client-1"), True),
        ("client", dict(usecase_id="scoped", client_id="client-2"), False),
        ("client", dict(usecase_id="scoped"), False),
        ("interaction", dict(usecase_id="scoped", interaction_id="i-1"), True),
        ("interaction", dict(usecase_id="scoped", interaction_id="i-2"), False),
        ("interaction", dict(usecase_id="scoped"), False),
        ("bogus", dict(usecase_id="scoped", client_id="client-1"), False),
    ],
)
def test_release_scope_modes(session, evidence, scope, asked, covered):
    from kognita.claims import _release_covers

    _scoped_cover(session, evidence, scope)
    check = _release_covers(session, "Scoped text.", audience="rm", **asked)
    assert (check.check == "C1-LABELLED") is covered, check.check


def test_release_scope_must_be_a_known_value():
    from test_usecases import _spec
    from kognita.usecases import vocabulary_citations

    assert any("RELEASE_SCOPE_INVALID" in item for item in vocabulary_citations(_spec(release_scope="everyone")))
    assert not any("RELEASE_SCOPE_INVALID" in item for item in vocabulary_citations(_spec(release_scope="client")))


def test_gateway_interaction_scope_takes_the_client_from_the_record(session, evidence):
    from datetime import datetime, timezone

    from kognita.interaction import open_interaction

    _ignored, _backend, headers = _setup(session, evidence)
    _tier_two(session, evidence)
    open_interaction(session, client_id="client-1", use_case="COLLABORATION",
                     interaction_id="i-1", evidence=evidence)
    open_interaction(session, client_id="client-2", use_case="COLLABORATION",
                     interaction_id="i-2", evidence=evidence)
    _cover(session, evidence, ANSWER, audience="rm", usecase_id="COLLABORATION",
           interaction_id="i-1", client_id="client-1")
    live = session.exec(
        select(UseCase).where(UseCase.usecase_id == "COLLABORATION", UseCase.effective_to.is_(None))
    ).one()
    now = datetime.now(timezone.utc)
    live.effective_to = now
    session.add(live)
    session.add(UseCase(
        usecase_id="COLLABORATION", version=live.version + 1,
        record={**live.record, "version": live.version + 1, "release_scope": "interaction"},
        effective_from=now,
    ))
    session.flush()
    gateway = _gateway(session.get_bind(), evidence, _Upstream(_response(ANSWER)))

    def send(**extra):
        return gateway.handle("POST", PATH, {**headers, **extra}, _body("hello"), session=session)

    assert send(interaction_id="i-1").status == 200
    assert send().status == 403
    assert send(interaction_id="i-2").status == 403
    assert send(interaction_id="does-not-exist").status == 403
