from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json

import pytest
from sqlmodel import select

from kognita import Fact, FreshnessRule, ResearchLifecycle, StaleAction, admit_fact, is_stale
from kognita.evidence import verify_chain
from kognita.models import EvidenceEvent, UseCase
from kognita.policy import PolicyLanguageError, read_policy_set
from kognita.retention import RetentionStore
from kognita.vocabulary import Classification

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
FACT = Fact(42, "positions-system", "operations", NOW, Classification.C2, "positions:read", timedelta(hours=1))
RULES = {"positions": FreshnessRule("positions", timedelta(minutes=5))}
RESEARCH = ResearchLifecycle("analyst", NOW, "passage", "1", NOW, NOW, NOW)


def _admit(session, evidence, fact=FACT, action=StaleAction.SUPPRESS, **kwargs):
    return admit_fact(fact, SimpleNamespace(usecase_id="meeting", stale_action=action),
                      "positions", RULES, session=session, evidence=evidence,
                      retention=RetentionStore(), correlation_id="interaction-1", now=NOW, **kwargs)


def test_fresh_and_stale_defaults(session, evidence):
    assert _admit(session, evidence) == (FACT, False)
    stale = replace(FACT, as_of=NOW - timedelta(minutes=6))
    assert _admit(session, evidence, stale) == (None, False)
    events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    assert [e.payload["reason_code"] for e in events] == ["FACT_FRESH", "FACT_STALE_SUPPRESSED"]
    assert "42" not in json.dumps(events[-1].payload)
    retained = json.loads(RetentionStore().read(session, events[-1].payload["content_hash"]))
    assert retained["fact"]["value"] == 42
    assert verify_chain(session) == 2
    assert UseCase(usecase_id="meeting").stale_action == StaleAction.SUPPRESS
    assert UseCase(usecase_id="meeting", stale_action=StaleAction.WARN).stale_action == StaleAction.WARN


@pytest.mark.parametrize("as_of", [None, NOW + timedelta(seconds=1), NOW.replace(tzinfo=None)])
def test_bad_timestamp(as_of):
    assert is_stale(replace(FACT, as_of=as_of), "positions", RULES, now=NOW)


def test_thresholds_and_unknown_types():
    assert not is_stale(replace(FACT, as_of=NOW-timedelta(minutes=5)), "positions", RULES, now=NOW)
    assert is_stale(FACT, "unknown", RULES, now=NOW)
    assert is_stale(FACT, "positions", {}, now=NOW)
    assert is_stale(replace(FACT, as_of=NOW-timedelta(seconds=2), freshness_threshold=timedelta(seconds=1)), "positions", RULES, now=NOW)
    assert is_stale(replace(FACT, as_of=NOW-timedelta(minutes=6), freshness_threshold=timedelta(days=100)), "positions", RULES, now=NOW)


def test_warning_and_invalid_contract(session, evidence):
    stale = replace(FACT, as_of=None)
    assert _admit(session, evidence, stale, StaleAction.WARN) == (stale, True)
    assert _admit(session, evidence, replace(stale, owner=""), StaleAction.WARN) == (None, False)
    assert _admit(session, evidence, stale, "invalid") == (None, False)


@pytest.mark.parametrize("updated", [FACT, None, replace(FACT, as_of=None), replace(FACT, source="other"), replace(FACT, entitlement="other")])
def test_refresh_rechecks(session, evidence, updated):
    result = _admit(session, evidence, replace(FACT, as_of=None), StaleAction.REFRESH,
                    refresher=lambda _: updated)
    assert result == ((FACT, False) if updated == FACT else (None, False))
    assert verify_chain(session) == 1


def test_refresh_failure(session, evidence):
    def broken(_):
        raise RuntimeError("unavailable")
    for refresher in (None, broken):
        assert _admit(session, evidence, replace(FACT, as_of=None), StaleAction.REFRESH,
                      refresher=refresher) == (None, False)


@pytest.mark.parametrize("action", list(StaleAction))
@pytest.mark.parametrize("lifecycle", [RESEARCH, replace(RESEARCH, expiry=NOW), replace(RESEARCH, superseded=NOW), replace(RESEARCH, effective=NOW+timedelta(days=1))])
def test_research_hard_gate(session, evidence, action, lifecycle):
    expected = (FACT, False) if lifecycle == RESEARCH else (None, False)
    assert _admit(session, evidence, action=action, research=lifecycle, refresher=lambda _: FACT) == expected


def test_custom_rules():
    for answer in (True, False, 1, "yes", None):
        rules = {"positions": FreshnessRule("positions", rule=lambda f, at: answer)}
        assert is_stale(FACT, "positions", rules, now=NOW) == (answer is not True)
    def broken(f, at):
        raise ValueError()
    assert is_stale(FACT, "positions", {"positions": FreshnessRule("positions", rule=broken)}, now=NOW)
    for kwargs in ({}, {"max_age": timedelta(days=-1)}, {"max_age": timedelta(0), "rule": broken}):
        with pytest.raises(ValueError):
            FreshnessRule("positions", **kwargs)


def test_policy_freshness(tmp_path):
    path = tmp_path / "freshness.yaml"
    path.write_text("freshness:\n  positions: 5m\n  kyc: {max_age: 365d}\n")
    policy = read_policy_set(path)
    assert dict((r.data_type, r.max_age) for r in policy.freshness) == {"positions": timedelta(minutes=5), "kyc": timedelta(days=365)}
    assert read_policy_set(tmp_path).freshness == policy.freshness
    for bad in ("-1", "true", "unlimited", "{}", "null"):
        path.write_text(f"freshness:\n  positions: {bad}\n")
        with pytest.raises(PolicyLanguageError):
            read_policy_set(path)


def test_evidence_failure_withholds(session, evidence, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("evidence unavailable")
    monkeypatch.setattr(evidence, "emit", fail)
    with pytest.raises(RuntimeError):
        _admit(session, evidence, replace(FACT, as_of=None), StaleAction.WARN)


def test_usecase_persistence(session):
    from kognita.usecases import UseCaseSpec, build_record, from_record
    spec = UseCaseSpec("meeting", "prepare meeting", "meeting preparation",
                       "business", "technology", "risk", "model", "control",
                       stale_action=StaleAction.WARN)
    record = build_record(spec, version=1)
    assert record["stale_action"] == "WARN"
    assert from_record(record).stale_action == StaleAction.WARN
    assert from_record({k: v for k, v in record.items() if k != "stale_action"}).stale_action == StaleAction.SUPPRESS
    row = UseCase(usecase_id="meeting", record=record, stale_action=spec.stale_action)
    session.add(row)
    session.commit()
    session.expire_all()
    assert session.get(UseCase, row.id).stale_action == StaleAction.WARN


def test_policy_changes_preserve_freshness(tmp_path):
    from kognita.policy import PolicyChange, apply_change
    path = tmp_path / "policy.yaml"
    path.write_text('''freshness:
  positions: 5m
rules:
  - id: prohibit
    regime: INTERNAL
    rule_type: PROHIBITED
    citation: No execution
    effective_from: 2026-01-01T00:00:00Z
    rule:
      on_violation: fail
''')
    current = read_policy_set(path)
    changed = apply_change(current, PolicyChange(removed=("prohibit",)), directory=tmp_path)
    assert changed.freshness == current.freshness
    assert read_policy_set(tmp_path).freshness == current.freshness
    assert not read_policy_set(tmp_path).rules
    second = tmp_path / "duplicate.yaml"
    second.write_text("freshness:\n  positions: 1d\n")
    with pytest.raises(PolicyLanguageError, match="duplicate freshness"):
        read_policy_set(tmp_path)


def test_usecase_stale_action_validation():
    from kognita.usecases import UseCaseSpec, vocabulary_citations
    spec = UseCaseSpec("meeting", "prepare meeting", "meeting preparation",
                       "business", "technology", "risk", "model", "control")
    assert not any("FACT_STALE_ACTION_INVALID" in c for c in vocabulary_citations(spec))
    assert any("FACT_STALE_ACTION_INVALID" in c for c in vocabulary_citations(replace(spec, stale_action="skip")))
