"""Calculation citation gates resist release bypasses and perform no math."""
import ast
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlmodel import Session, select

from kognita import CalculationService, Claim, ClaimType, ValidatorStatus, check_calculation, register_calculation_service
from kognita.claims import record_claim, release_claim
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.models import EvidenceEvent, RetainedContent, UseCase
from kognita.retention import RetentionStore
from kognita.sources import Source, register_source
from kognita.vocabulary import CheckResult


@pytest.fixture
def store(tmp_path):
    engine = make_engine(str(tmp_path / "calculations.db"))
    create_all(engine)
    with Session(engine) as session:
        writer = EvidenceWriter(engine)
        session.add(UseCase(usecase_id="u1", record={"usecase_id": "u1", "approved_sources": ["crm"]}))
        session.flush()
        register_source(
            session,
            Source("crm", "policy", "r1", "owner", datetime(2020, 1, 1, tzinfo=timezone.utc)),
            evidence=writer,
            interaction_id="setup",
        )
        yield session, writer


def _service(*figures, service_id="book", version="1", owner="Finance owner"):
    return CalculationService(service_id, "Book service", version, owner, figures)


def _claim(text, method="", claim_id="c1", **changes):
    claim = Claim(
        claim_id, "client", text, ClaimType.INFERENCE,
        source_system="crm", source_record="r1", calculation_method=method,
        validator_status=ValidatorStatus.VERIFIED,
    )
    return replace(claim, **changes) if changes else claim


def _register(store, service=None):
    session, writer = store
    service = _service("nav") if service is None else service
    register_calculation_service(session, service, evidence=writer, interaction_id="setup")
    return service


def _release(store, text, method="", *, flag=False, audience="rm", claim_id="c1"):
    session, writer = store
    record_claim(session, _claim(text, method, claim_id), evidence=writer, interaction_id="i")
    return release_claim(
        session, claim_id, evidence=writer, interaction_id="i", usecase_id="u1",
        audience=audience, authoritative_figure=flag,
    )


@pytest.mark.parametrize("audience", ["rm", "client"])
@pytest.mark.parametrize("text,method,figures,flag,reason", [
    ("NAV is 1.25", "book@1", ("nav",), False, "C1-LABELLED"),
    ("NAV is 1.25", "", ("nav",), False, "C6-UNCITED"),
    ("NAV is 1.25", "book@1", ("tax",), False, "C6-UNCOVERED"),
    ("see the attached note", "book@1", ("nav",), True, "C1-LABELLED"),
    ("see the attached note", "", ("nav",), True, "C6-UNCITED"),
    ("NAV is twelve", "book@1", ("tax",), True, "C6-UNCOVERED"),
    ("NAV is twelve", "book@1", ("nav",), True, "C1-LABELLED"),
])
def test_release_blocks_uncited_figures(store, audience, text, method, figures, flag, reason):
    _register(store, _service(*figures))
    check = _release(store, text, method, flag=flag, audience=audience)
    assert check.check == reason
    assert check.result == (CheckResult.PASS if reason == "C1-LABELLED" else CheckResult.FAIL)
    if reason.startswith("C6-"):
        assert "no financial math" in check.citation
    session, writer = store
    event = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence.desc())).first()
    assert event.payload["allowed"] is (reason == "C1-LABELLED")
    assert event.payload["authoritative_figure"] is flag
    assert event.payload["reason"] == reason
    verify_chain(session)


@pytest.mark.parametrize("text", [
    "Client may be considering a sale",
    "NAV is stable",
    "The balance is 12",
    "NAV is twelve",
    "NAV is XII",
    "Performance improved",
])
def test_ordinary_claims_are_not_figures(store, text):
    session, writer = store
    claim = _claim(text, method="not-a-service")
    check = check_calculation(session, claim, evidence=writer, interaction_id="i")
    assert check.check == "C6-CLEAR"
    assert check.result == CheckResult.PASS
    assert _release(store, text).result == CheckResult.PASS


@pytest.mark.parametrize("text,kinds", [
    ("NAV is 1.25", ["nav"]),
    ("nav: 1.25", ["nav"]),
    ("Net asset value is 10", ["nav"]),
    ("N.A.V. is 5", ["nav"]),
    ("P&L of -3.2", ["pnl"]),
    ("p & l (4)", ["pnl"]),
    ("profit and loss: 4", ["pnl"]),
    ("profit/loss 4", ["pnl"]),
    ("VaR 2.5%", ["var"]),
    ("value-at-risk equals 0.02", ["var"]),
    ("value at risk 0.02", ["var"]),
    ("allocation 40%", ["allocation"]),
    ("FX rate 1.08", ["fx"]),
    ("foreign exchange 0.92", ["fx"]),
    ("leverage 2x", ["leverage"]),
    ("concentration is 15", ["concentration"]),
    ("maturity value 1000", ["maturity"]),
    ("interest 3.5%", ["interest"]),
    ("3 clients expressed interest", ["interest"]),
    ("exposure: 8", ["exposure"]),
    ("tax is 1200", ["tax"]),
    ("suitability score 72", ["suitability"]),
    ("performance 8.1%", ["performance"]),
    ("The 3 clients called. NAV is unchanged.", ["nav"]),
    ("NAV is 1.2 and tax is 4", ["nav", "tax"]),
    ("ＮＡＶ is １２．５", ["nav"]),
    ("NA\u200bV is 10", ["nav"]),
    ("NA\u00adV is 10", ["nav"]),
    ("NAV is &#49;2", ["nav"]),
    ("N&#65;V is &#49;2", ["nav"]),
    ("NAV%20is%2012", ["nav"]),
    ("NAV%2520is%252012", ["nav"]),
    ("NAV is ½", ["nav"]),
    ("NAV is Ⅳ", ["nav"]),
    ("NAV is ①", ["nav"]),
    ("n\u0430v is 10", ["nav"]),
    ("performance is 2+2", ["performance"]),
])
def test_obfuscated_figures_fail_closed(store, text, kinds):
    session, writer = store
    _register(store, _service("nav", "tax"))
    missed = check_calculation(session, _claim(text), evidence=writer, interaction_id="i")
    assert missed.check == "C6-UNCITED"
    assert missed.result == CheckResult.FAIL
    covered = set(kinds) <= {"nav", "tax"}
    cited = check_calculation(session, _claim(text, "book@1"), evidence=writer, interaction_id="i")
    assert cited.check == ("C6-CITED" if covered else "C6-UNCOVERED")
    event = next(
        item for item in session.exec(select(EvidenceEvent))
        if item.payload.get("action") == "check_calculation" and item.payload.get("reason") == "C6-UNCITED"
    )
    assert text not in json.dumps(event.payload)
    pinned = json.loads(RetentionStore().read(session, event.payload["calculation_hash"]))
    assert pinned["figures"] == sorted(kinds)
    assert "4" not in missed.check


@pytest.mark.parametrize("method", [
    "book",
    "book@2",
    "book@1 ",
    " book@1",
    "BOOK@1",
    "book@1@extra",
    "prefix-book@1",
    "Book service@1",
    "model",
    "model@1",
    "book@１",
    12,
])
def test_citation_must_match_one_registered_version(store, method):
    session, writer = store
    _register(store)
    check = check_calculation(session, _claim("NAV is 1.25", method), evidence=writer, interaction_id="i")
    assert check.check == "C6-UNREGISTERED"
    assert check.result == CheckResult.FAIL


def test_flag_and_keyword_require_coverage_without_a_digit(store):
    session, writer = store
    _register(store, _service("tax"))
    blocked = check_calculation(
        session, _claim("NAV is twelve", "book@1"), authoritative_figure=True,
        evidence=writer, interaction_id="i",
    )
    assert blocked.check == "C6-UNCOVERED"
    _register(store, _service("nav", service_id="navbook"))
    allowed = check_calculation(
        session, _claim("NAV is twelve", "navbook@1"), authoritative_figure=True,
    )
    assert allowed.check == "C6-CITED"


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None, "False"])
def test_only_a_real_boolean_can_set_the_release_flag(store, flag):
    session, writer = store
    record_claim(session, _claim("NAV is 1.25"), evidence=writer, interaction_id="i")
    with pytest.raises(ValueError, match="C6-INVALID"):
        check_calculation(session, _claim("NAV is 1.25"), authoritative_figure=flag)
    with pytest.raises(ValueError, match="C6-INVALID"):
        release_claim(
            session, "c1", evidence=writer, interaction_id="i", usecase_id="u1",
            authoritative_figure=flag,
        )


def test_registration_evidence_and_immutability(store):
    session, writer = store
    service = _service("nav", "tax", owner="Finance owner")
    assert register_calculation_service(session, service, evidence=writer, interaction_id="i") == service
    reordered = _service("tax", "nav", owner="Finance owner")
    assert register_calculation_service(session, reordered, evidence=writer, interaction_id="i") == service
    with pytest.raises(ValueError, match="C6-IMMUTABLE"):
        register_calculation_service(session, replace(service, owner="Other owner"), evidence=writer, interaction_id="i")
    with pytest.raises(ValueError, match="C6-IMMUTABLE"):
        register_calculation_service(session, _service("tax"), evidence=writer, interaction_id="i")
    event = next(item for item in session.exec(select(EvidenceEvent)) if item.payload.get("action") == "register_calculation_service")
    assert "Finance owner" not in json.dumps(event.payload)
    body = json.loads(RetentionStore().read(session, event.payload["service_hash"]))
    assert body["owner"] == "Finance owner"
    assert set(body["figures_covered"]) == {"nav", "tax"}
    row = session.get(RetainedContent, event.payload["service_hash"])
    row.body = row.body.replace("Finance owner", "changed")
    session.add(row)
    session.flush()
    with pytest.raises(ValueError, match="C6-CONTENT"):
        check_calculation(session, _claim("hello"))
    verify_chain(session)


@pytest.mark.parametrize("changes", [
    {"service_id": "model"},
    {"service_id": "Model"},
    {"service_id": ""},
    {"service_id": " book"},
    {"service_id": "bo ok"},
    {"service_id": "book@1"},
    {"version": ""},
    {"version": "1 0"},
    {"name": " "},
    {"owner": ""},
    {"owner": " owner"},
    {"figures_covered": ()},
    {"figures_covered": ("P&L",)},
    {"figures_covered": ("all",)},
    {"figures_covered": ("*",)},
    {"figures_covered": ("",)},
    {"figures_covered": "nav"},
    {"figures_covered": ("nav", "unknown")},
])
def test_invalid_services_are_rejected(changes):
    base = {"service_id": "book", "name": "Book", "version": "1", "owner": "Finance", "figures_covered": ("nav",)}
    base.update(changes)
    with pytest.raises(ValueError, match="C6-INVALID"):
        CalculationService(**base)


def test_non_service_registration_is_rejected(store):
    session, writer = store
    with pytest.raises(ValueError, match="C6-INVALID"):
        register_calculation_service(session, object(), evidence=writer, interaction_id="i")


def test_release_does_not_let_a_citation_skip_other_gates(store):
    session, writer = store
    _register(store)
    record_claim(session, _claim("NAV is 1.25", "book@1", claim_type=None), evidence=writer, interaction_id="i")
    unlabelled = release_claim(session, "c1", evidence=writer, interaction_id="i", usecase_id="u1")
    assert unlabelled.result == CheckResult.FAIL
    assert unlabelled.check == "C1-UNLABELLED"
    record_claim(
        session,
        _claim("NAV is 1.25", "book@1", claim_id="c2", source_system="model"),
        evidence=writer,
        interaction_id="i",
    )
    ungrounded = release_claim(session, "c2", evidence=writer, interaction_id="i", usecase_id="u1")
    assert ungrounded.check == "C3-MODEL"
    record_claim(session, _claim("NAV is 1.25", claim_id="c3"), evidence=writer, interaction_id="i")
    with pytest.raises(ValueError, match="C1-DUPLICATE"):
        record_claim(session, _claim("NAV is 1.25", "book@1", claim_id="c3"), evidence=writer, interaction_id="i")
    assert release_claim(session, "c3", evidence=writer, interaction_id="i", usecase_id="u1").check == "C6-UNCITED"


def test_claim_text_is_required(store):
    session, _ = store
    with pytest.raises(ValueError, match="C6-INVALID"):
        check_calculation(session, object())


def test_calculation_module_performs_no_math():
    source = (Path(__file__).resolve().parents[1] / "src" / "kognita" / "calculations.py").read_text()
    assert "\u2014" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.BinOp, ast.AugAssign))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec", "compile", "float", "int", "complex"}
    import kognita
    from kognita import calculations
    assert set(calculations.__all__) == {
        "CalculationService", "register_calculation_service", "check_calculation",
    }
    for name in calculations.__all__:
        assert name in kognita.__all__
