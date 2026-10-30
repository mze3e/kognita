"""Verifier independence and fail-closed material release."""
from datetime import datetime, timedelta, timezone
import json

import pytest
from sqlmodel import Session, select

import kognita
from kognita import Claim, ClaimType, ValidatorStatus, Verifier
from kognita.claims import list_claims, record_claim, relabel_claim, release_claim
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.models import EvidenceEvent, Policy, RetainedContent, UseCase
from kognita.policy import parse_rule, parse_yaml
from kognita.retention import RetentionStore
from kognita.rules import content_policy
from kognita.sources import Source, register_source
from kognita.envelope import Envelope, RuleContext
from kognita.usecases import UseCaseSpec, build_record, from_record
from kognita.verification import (
    __all__ as verification_exports,
    check_independence,
    register_verifier,
    verify_claim,
)
from kognita.vocabulary import CheckResult, EventType

NOW = datetime.now(timezone.utc)
PRODUCER = {
    "agent_id": "producer-agent",
    "model": "prod-model",
    "model_version": "prod-2026.10",
    "provider": "prod-provider",
}


def _verifier(**changes):
    base = dict(
        verifier_id="v-check",
        agent_id="verifier-agent",
        model="check-model",
        model_version="check-2026.10",
        provider="check-provider",
    )
    base.update(changes)
    return Verifier(**base)


@pytest.fixture
def store(tmp_path):
    engine = make_engine(str(tmp_path / "verification.db"))
    create_all(engine)
    with Session(engine) as session:
        writer = EvidenceWriter(engine)
        yield session, writer


def _prepare(store, *, tier=2, verifier="v-check", permit=False, tier_value=None):
    session, writer = store
    record = {
        "usecase_id": "u1",
        "approved_sources": ["crm"],
        "proposed_tier": tier if tier_value is None else tier_value,
        "verifier": verifier,
        "permit_same_model": permit,
    }
    session.add(UseCase(usecase_id="u1", record=record))
    session.flush()
    register_source(
        session,
        Source("crm", "policy", "r1", "owner", NOW - timedelta(days=1)),
        evidence=writer,
        interaction_id="setup",
    )
    claim = Claim(
        "c1",
        "client1",
        "The position is within the mandate",
        ClaimType.INFERENCE,
        source_system="crm",
        source_record="r1",
        agent_id=PRODUCER["agent_id"],
        model_version=PRODUCER["model_version"],
    )
    record_claim(session, claim, evidence=writer, interaction_id="i1")
    return session, writer


def _verify(store, verifier=None, status=ValidatorStatus.VERIFIED, **producer):
    session, writer = store
    fields = dict(PRODUCER)
    fields.update(producer)
    return verify_claim(
        session,
        "c1",
        status,
        verifier_id=(verifier or _verifier()).verifier_id,
        evidence=writer,
        interaction_id="i1",
        usecase_id="u1",
        **fields,
    )


@pytest.mark.parametrize("field", ["verifier_id", "agent_id", "model", "model_version", "provider"])
@pytest.mark.parametrize("value", ["", " ", None])
def test_verifier_fields_are_required(field, value):
    with pytest.raises(ValueError, match="C7-INVALID"):
        _verifier(**{field: value})


def test_exports_and_usecase_field():
    assert set(verification_exports) == {
        "Verifier", "register_verifier", "verify_claim", "check_independence",
    }
    for name in verification_exports:
        assert name in kognita.__all__
    assert EventType.CLAIM_VERIFIED.value == "CLAIM_VERIFIED"
    row = UseCase(usecase_id="u1", verifier=" v-check ", permit_same_model=True)
    assert row.verifier == "v-check"
    assert row.permit_same_model is True
    with pytest.raises(ValueError, match="C7-INVALID"):
        UseCase(usecase_id="u1", permit_same_model="true")
    spec = UseCaseSpec(
        usecase_id="u",
        purpose="Check claims",
        category="client intelligence",
        business_owner="a",
        technology_owner="b",
        risk_owner="c",
        model_owner="d",
        control_function="e",
        verifier=" v-check ",
        permit_same_model=True,
    )
    entry = from_record(build_record(spec, version=1))
    assert entry.verifier == "v-check"
    assert entry.permit_same_model is True
    assert entry.proposed_tier >= 2


def test_independent_verifier_pins_both_models_and_releases(store):
    session, writer = _prepare(store)
    verifier = _verifier()
    register_verifier(session, verifier, evidence=writer, interaction_id="i1")
    assert register_verifier(session, verifier, evidence=writer, interaction_id="i1") == verifier
    check = _verify(store)
    assert check.result == CheckResult.PASS
    assert check.check == "C7-RECORDED"
    assert list_claims(session)[0].validator_status == ValidatorStatus.VERIFIED
    event = next(row for row in session.exec(select(EvidenceEvent)) if row.payload.get("action") == "verify")
    assert event.event_type == EventType.CLAIM_VERIFIED
    assert event.payload["producer_model"] == PRODUCER["model"]
    assert event.payload["producer_model_version"] == PRODUCER["model_version"]
    assert event.payload["producer_provider"] == PRODUCER["provider"]
    assert event.payload["verifier_model"] == verifier.model
    assert event.payload["verifier_model_version"] == verifier.model_version
    assert event.payload["verifier_provider"] == verifier.provider
    pinned = json.loads(RetentionStore().read(session, event.payload["verification_hash"]))
    assert pinned["producer"]["model"] == PRODUCER["model"]
    assert pinned["verifier"]["model"] == verifier.model
    assert "prod-model" not in json.dumps({key: value for key, value in event.payload.items() if key == "verification_hash"})
    for audience in ("rm", "client"):
        released = release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience=audience)
        assert released.result == CheckResult.PASS
        assert released.check == "C1-LABELLED"
    verify_chain(session)


@pytest.mark.parametrize("audience,status,allowed", [
    ("rm", ValidatorStatus.VERIFIED, True),
    ("client", ValidatorStatus.VERIFIED, True),
    ("rm", ValidatorStatus.INFERRED, True),
    ("client", ValidatorStatus.INFERRED, False),
    ("rm", ValidatorStatus.UNVERIFIED, False),
    ("client", ValidatorStatus.UNVERIFIED, False),
    ("rm", ValidatorStatus.CONFLICTING, False),
    ("client", ValidatorStatus.STALE, False),
    ("rm", ValidatorStatus.BLOCKED, False),
    ("client", ValidatorStatus.BLOCKED, False),
])
def test_recorded_status_limits_audience(store, audience, status, allowed):
    session, writer = _prepare(store)
    register_verifier(session, _verifier(), evidence=writer, interaction_id="i1")
    assert _verify(store, status=status).result == CheckResult.PASS
    released = release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience=audience)
    assert released.result == (CheckResult.PASS if allowed else CheckResult.FAIL)
    blocked = "C1-STALE" if status == ValidatorStatus.STALE else "C7-AUDIENCE"
    assert released.check == ("C1-LABELLED" if allowed else blocked)


@pytest.mark.parametrize("tier,permit,changes,reason", [
    (2, False, {"model": "prod-model"}, "C7-MODEL"),
    (2, True, {"model": "prod-model"}, "C7-MODEL"),
    (4, False, {"model_version": "prod-2026.10"}, "C7-VERSION"),
    (2, False, {"provider": "prod-provider"}, "C7-PROVIDER"),
    (2, False, {"provider": " PROD-PROVIDER "}, "C7-PROVIDER"),
    (2, False, {"model": " prod-model "}, "C7-MODEL"),
    (1, False, {"provider": "prod-provider"}, "C7-PROVIDER"),
    (1, True, {"model": "prod-model", "model_version": "prod-2026.10", "provider": "prod-provider"}, "C7-RECORDED"),
    (1, True, {"agent_id": "producer-agent"}, "C7-SELF"),
    (2, True, {"agent_id": " PRODUCER-AGENT "}, "C7-SELF"),
    (1, True, {"agent_id": "producer-agent", "model": "prod-model"}, "C7-SELF"),
])
def test_overlap_and_self_verification_fail_closed(store, tier, permit, changes, reason):
    session, writer = _prepare(store, tier=tier, permit=permit)
    register_verifier(session, _verifier(**changes), evidence=writer, interaction_id="i1")
    check = _verify(store, verifier=_verifier(**changes))
    assert check.result == (CheckResult.PASS if reason == "C7-RECORDED" else CheckResult.FAIL)
    assert check.check == reason
    assert list_claims(session)[0].validator_status == (
        ValidatorStatus.VERIFIED if reason == "C7-RECORDED" else ValidatorStatus.UNVERIFIED
    )
    if reason != "C7-RECORDED":
        released = release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience="rm")
        assert released.result == CheckResult.FAIL
        # Tier 2 and above require a successful verification. Below that, the
        # claim stays unverified, so the audience rule is what denies release.
        assert released.check == ("C7-UNVERIFIED" if tier >= 2 else "C7-AUDIENCE")


@pytest.mark.parametrize("tier,permit,same,expected", [
    (1, True, True, "C7-INDEPENDENT"),
    (1, False, True, "C7-MODEL"),
    (1, True, False, "C7-SELF"),
    (2, True, True, "C7-MODEL"),
    (4, False, False, "C7-SELF"),
])
def test_check_independence_does_not_accept_lookalike_permits(tier, permit, same, expected):
    verifier = _verifier(agent_id="producer-agent") if not same else _verifier(model="prod-model")
    if expected == "C7-SELF":
        verifier = _verifier(agent_id="producer-agent")
    elif expected == "C7-MODEL":
        verifier = _verifier(model="prod-model")
    else:
        verifier = _verifier(model="prod-model", model_version="prod-2026.10", provider="prod-provider")
    check = check_independence(verifier, tier=tier, permit=permit, **PRODUCER)
    assert check.check == expected
    assert check_independence(verifier, tier=1, permit=1, **PRODUCER).check != "C7-INDEPENDENT" or expected == "C7-SELF"
    assert check_independence(
        _verifier(model="prod-model"), tier=1, permit="true", **PRODUCER
    ).check == "C7-MODEL"


def test_blank_and_conflicting_producer_is_not_independent():
    verifier = _verifier()
    assert check_independence(verifier, agent_id=" ", model="m", model_version="v", provider="p").check == "C7-PIN"
    assert check_independence(verifier, agent_id="a", model="", model_version="v", provider="p").check == "C7-PIN"
    assert check_independence(
        _verifier(model="prod-model"),
        tier=True,
        permit=True,
        agent_id="producer-agent",
        model="prod-model",
        model_version="v",
        provider="p",
    ).check == "C7-MODEL"


@pytest.mark.parametrize("tier_value,verifier,expected", [
    (2, "", "C7-TIER"),
    (2, "missing", "C7-UNREGISTERED"),
    ("2", "", "C7-TIER"),
    (True, "", "C7-TIER"),
])
def test_tier_gate_denies_release_without_a_real_verifier(store, tier_value, verifier, expected):
    session, writer = _prepare(store, verifier=verifier, tier_value=tier_value)
    released = release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience="client")
    assert released.check == expected
    assert released.result == CheckResult.FAIL


def test_labelled_verified_without_verification_event_is_denied(store):
    session, writer = _prepare(store)
    register_verifier(session, _verifier(), evidence=writer, interaction_id="i1")
    relabel_claim(
        session, "c1", ClaimType.VERIFIED_FACT, evidence=writer, interaction_id="i1",
        source_system="crm", source_record="r1", validator_status=ValidatorStatus.VERIFIED,
    )
    released = release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience="client")
    assert released.check == "C7-UNVERIFIED"


def test_wrong_verifier_and_producer_spoof_do_not_record(store):
    session, writer = _prepare(store, verifier="v-check")
    register_verifier(session, _verifier(), evidence=writer, interaction_id="i1")
    register_verifier(session, _verifier(verifier_id="other"), evidence=writer, interaction_id="i1")
    other = _verify(store, verifier=_verifier(verifier_id="other"))
    assert other.check == "C7-MISMATCH"
    spoofed = verify_claim(
        session, "c1", ValidatorStatus.VERIFIED, verifier_id="v-check",
        agent_id="someone-else", model="other-model", model_version="other-version",
        provider="other-provider", evidence=writer, interaction_id="i1", usecase_id="u1",
    )
    assert spoofed.check == "C7-PRODUCER"
    assert list_claims(session)[0].validator_status == ValidatorStatus.UNVERIFIED
    with pytest.raises(TypeError):
        verify_claim(
            session, "c1", ValidatorStatus.VERIFIED, verifier_id="v-check",
            evidence=writer, interaction_id="i1", usecase_id="u1", permit=True, tier=1,
            **PRODUCER,
        )


def test_status_relabel_and_tier_increase_do_not_widen_release(store):
    session, writer = _prepare(store, tier=1, permit=True)
    register_verifier(session, _verifier(model="prod-model"), evidence=writer, interaction_id="i1")
    assert _verify(store, verifier=_verifier(model="prod-model"), status=ValidatorStatus.INFERRED).check == "C7-RECORDED"
    relabel_claim(
        session, "c1", ClaimType.INFERENCE, evidence=writer, interaction_id="i1",
        validator_status=ValidatorStatus.VERIFIED,
    )
    assert release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience="client").check == "C7-STATUS"
    row = session.exec(select(UseCase)).one()
    when = datetime.now(timezone.utc)
    row.effective_to = when
    session.add(row)
    session.flush()
    # Restore the verified snapshot by verifying again at tier 1 before the close takes effect.
    # The closed row is already closed, so re-verify against the replacement below.
    record = dict(row.record)
    record["proposed_tier"] = 2
    session.add(UseCase(usecase_id="u1", version=2, record=record, effective_from=when))
    session.flush()
    denied = release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience="rm")
    assert denied.result == CheckResult.FAIL
    assert denied.check in {"C7-MODEL", "C7-STATUS"}


def test_release_rechecks_independence_when_tier_rises(store):
    session, writer = _prepare(store, tier=1, permit=True)
    register_verifier(session, _verifier(provider="prod-provider"), evidence=writer, interaction_id="i1")
    assert _verify(store, verifier=_verifier(provider="prod-provider")).result == CheckResult.PASS
    assert release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1").result == CheckResult.PASS
    row = session.exec(select(UseCase)).one()
    when = datetime.now(timezone.utc)
    row.effective_to = when
    session.add(row)
    session.flush()
    record = dict(row.record)
    record["proposed_tier"] = 2
    session.add(UseCase(usecase_id="u1", version=2, record=record, effective_from=when))
    session.flush()
    denied = release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience="client")
    assert denied.check == "C7-PROVIDER"
    verify_chain(session)


def test_tampered_records_fail_closed(store):
    session, writer = _prepare(store)
    register_verifier(session, _verifier(), evidence=writer, interaction_id="i1")
    registered = next(row for row in session.exec(select(EvidenceEvent)) if row.payload.get("action") == "register_verifier")
    retained = session.get(RetainedContent, registered.payload["verifier_hash"])
    original = retained.body
    retained.body = original.replace("check-model", "prod-model")
    session.add(retained)
    session.flush()
    with pytest.raises(ValueError, match="C7-CONTENT"):
        _verify(store)
    retained.body = original
    session.add(retained)
    session.flush()
    assert _verify(store).result == CheckResult.PASS
    verified = next(row for row in session.exec(select(EvidenceEvent)) if row.payload.get("action") == "verify" and row.payload.get("allowed") is True)
    body = session.get(RetainedContent, verified.payload["verification_hash"])
    body.body = body.body.replace("VERIFIED", "INFERRED")
    session.add(body)
    session.flush()
    with pytest.raises(ValueError, match="C7-CONTENT"):
        release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1")


def test_verifier_registration_is_immutable(store):
    session, writer = store
    register_verifier(session, _verifier(), evidence=writer, interaction_id="i1")
    with pytest.raises(ValueError, match="C7-IMMUTABLE"):
        register_verifier(session, _verifier(model="other-model"), evidence=writer, interaction_id="i1")


@pytest.mark.parametrize("audience,status,allowed", [
    ("rm", ValidatorStatus.INFERRED, True),
    ("client", ValidatorStatus.INFERRED, False),
    ("client", ValidatorStatus.VERIFIED, True),
    ("rm", ValidatorStatus.UNVERIFIED, False),
    ("client", ValidatorStatus.STALE, False),
])
def test_default_audience_without_a_verifier_at_tier_1(store, audience, status, allowed):
    session, writer = _prepare(store, tier=1, verifier="")
    claim = list_claims(session)[0]
    if status != ValidatorStatus.UNVERIFIED:
        relabel_claim(
            session, "c1", claim.claim_type, evidence=writer, interaction_id="i1",
            validator_status=status,
        )
    released = release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience=audience)
    assert released.result == (CheckResult.PASS if allowed else CheckResult.FAIL)
    blocked = "C1-STALE" if status == ValidatorStatus.STALE else "C7-AUDIENCE"
    assert released.check == ("C1-LABELLED" if allowed else blocked)


def test_policy_maps_are_intersected_and_unknown_keys_grant_nothing(store):
    session, writer = _prepare(store, tier=1, verifier="")
    relabel_claim(
        session, "c1", ClaimType.INFERENCE, evidence=writer, interaction_id="i1",
        validator_status=ValidatorStatus.INFERRED,
    )
    session.add(Policy(
        regime="INTERNAL", rule_type="CONTENT_POLICY", citation="Wide map",
        rule={"validator_audiences": {"INFERRED": ["client", "rm"], "VERIFIED": ["client", "rm"]}},
    ))
    session.flush()
    assert release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience="client").result == CheckResult.PASS
    session.add(Policy(
        regime="INTERNAL", rule_type="CONTENT_POLICY", citation="Narrow map",
        rule={"validator_audiences": {"INFERRED": ["rm"], "VERIFIED": ["rm"]}},
    ))
    session.flush()
    assert release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience="client").check == "C7-AUDIENCE"
    assert release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience="rm").result == CheckResult.PASS
    for row in session.exec(select(Policy)):
        row.effective_to = datetime.now(timezone.utc)
        session.add(row)
    session.flush()
    session.add(Policy(
        regime="INTERNAL", rule_type="CONTENT_POLICY", citation="Wrong case",
        rule={"validator_audiences": {"verified": ["client"], "INFERRED": "client"}},
    ))
    session.flush()
    assert release_claim(session, "c1", evidence=writer, interaction_id="i1", usecase_id="u1", audience="client").check == "C7-AUDIENCE"
    verify_chain(session)


def test_policy_language_and_content_rule():
    text = """
id: status.map
regime: INTERNAL
rule_type: CONTENT_POLICY
citation: Status audiences
effective_from: 2026-01-01T00:00:00+00:00
rule:
  validator_audiences:
    VERIFIED: [client, rm]
    INFERRED: [RM]
"""
    document, findings = parse_rule(parse_yaml(text), source="status.yaml")
    assert findings == [] and document is not None
    broken = text.replace("INFERRED: [RM]", "MADE_UP: [everyone]")
    _, findings = parse_rule(parse_yaml(broken), source="status.yaml")
    assert any("unknown validator status" in item.message for item in findings)
    policy = Policy(regime="INTERNAL", rule_type="CONTENT_POLICY", rule=document.rule, citation="Status audiences")
    allowed = content_policy(policy, RuleContext(
        Envelope("rm", "u1", "release"),
        attributes={"validator_status": "VERIFIED", "audience": "client"},
    ))
    blocked = content_policy(policy, RuleContext(
        Envelope("rm", "u1", "release"),
        attributes={"validator_status": "BLOCKED", "audience": "client"},
    ))
    missing = content_policy(policy, RuleContext(Envelope("rm", "u1", "release"), attributes={"content": "text"}))
    assert allowed[0].result == CheckResult.PASS
    assert blocked[0].result == CheckResult.FAIL
    assert missing[0].result == CheckResult.FAIL
