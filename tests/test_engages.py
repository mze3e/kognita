"""DomainPack.engages is on the protocol, and the decision path uses it.

The gap note says the method is absent and that a pack without it still
passes isinstance(pack, DomainPack). That note has drifted. The method is
declared on the protocol, a pack that omits it is not a DomainPack, and
decide skips a policy the pack's engages rejects. The call sites still read
the method with getattr. When it is absent, every effective policy is still
evaluated. This test adds no method and changes no decision.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from fixtures import bmos_pack as bp
from fixtures import demo_pack as dp

from kognita.broker import ask
from kognita.envelope import Envelope
from kognita.models import Policy
from kognita.protocols import DomainPack
from kognita.rules import build_registry
from kognita.testing.harness import FROZEN_NOW, Harness
from kognita.vocabulary import CheckResult, Outcome

_PATHS = ("harness", "run_governed", "ask")


class _Pack:
    """A pack whose engages answer is fixed for one decision."""

    name = "probe"

    def __init__(self, engaged: bool) -> None:
        self.engaged = engaged
        self.calls = 0

    def load_subjects(self, envelope: Envelope, session: Any) -> dict[str, Any]:
        return {}

    def resolve_attributes(self, envelope: Envelope, subjects: dict[str, Any]) -> dict[str, Any]:
        return {}

    def rules(self) -> dict[str, Any]:
        return build_registry()

    def engages(self, policy: Any, context: Any) -> bool:
        self.calls += 1
        return self.engaged


class _PackWithoutEngages:
    """The same shape, without the method the protocol requires."""

    name = "probe"

    def load_subjects(self, envelope: Envelope, session: Any) -> dict[str, Any]:
        return {}

    def resolve_attributes(self, envelope: Envelope, subjects: dict[str, Any]) -> dict[str, Any]:
        return {}

    def rules(self) -> dict[str, Any]:
        return build_registry()


def _seed(session) -> None:
    session.add(
        Policy(
            regime="HOUSE",
            rule_type="PROHIBITED",
            rule={"on_violation": "fail"},
            citation="House Rules s1",
            effective_from=FROZEN_NOW - timedelta(days=1),
        )
    )


def _envelope() -> Envelope:
    return Envelope(
        principal="tester",
        purpose="REVIEW",
        tool="get_subject_profile",
        actor_location="SG",
    )


def _harness(pack: Any) -> tuple[Harness, list[str]]:
    ran: list[str] = []

    def tool(envelope, evaluation, session):
        ran.append("released")
        return "released"

    harness = Harness(pack=pack, purposes=("REVIEW",), seed=_seed)
    harness.registry.register("get_subject_profile", tool)
    return harness, ran


def _evaluation(path: str, harness: Harness):
    envelope = _envelope()
    with harness.session() as session:
        if path == "harness":
            return harness.evaluate(envelope, session)
        if path == "run_governed":
            return harness.run_tool(envelope, session).evaluation
        answer = ask(
            session,
            "record status",
            envelope,
            pack=harness.pack,
            embedder=harness.embedder,
            evidence=harness.evidence,
            purposes=harness.purposes,
            as_of=harness.now,
            now=harness.now,
        )
        return answer.evaluation


def _prohibited(evaluation):
    return [check for check in evaluation.checks if check.check == "PROHIBITED"]


def test_engages_is_on_the_domain_pack_protocol():
    """The protocol names engages, and the packs the suite already runs implement it.

    ``__protocol_attrs__`` is the member set ``isinstance`` consults on Python 3.12.
    """
    assert "engages" in DomainPack.__protocol_attrs__
    assert isinstance(_Pack(True), DomainPack)
    assert isinstance(dp.DemoPack(), DomainPack)
    assert isinstance(bp.BMOSPack(), DomainPack)


def test_a_pack_without_engages_is_not_a_domain_pack():
    """Omitting the method no longer passes the protocol check."""
    pack = _PackWithoutEngages()
    assert not hasattr(pack, "engages")
    assert not isinstance(pack, DomainPack)


@pytest.mark.parametrize("path", _PATHS)
def test_an_unengaged_policy_is_skipped(path):
    """A policy the pack does not engage is not evaluated, on each call site."""
    pack = _Pack(False)
    harness, ran = _harness(pack)
    evaluation = _evaluation(path, harness)

    assert pack.calls == 1
    assert evaluation.outcome is Outcome.ALLOW
    assert _prohibited(evaluation) == []
    if path == "run_governed":
        assert ran == ["released"]


@pytest.mark.parametrize("path", _PATHS)
def test_an_engaged_policy_is_evaluated(path):
    """The same policy is a denial when the pack engages it."""
    pack = _Pack(True)
    harness, ran = _harness(pack)
    evaluation = _evaluation(path, harness)

    assert pack.calls == 1
    assert evaluation.outcome is Outcome.DENY
    failed = _prohibited(evaluation)
    assert len(failed) == 1
    assert failed[0].result is CheckResult.FAIL
    assert failed[0].citation == "House Rules s1"
    if path == "run_governed":
        assert ran == []


@pytest.mark.parametrize("path", _PATHS)
def test_a_missing_engages_method_still_evaluates_every_policy(path):
    """getattr still defaults to evaluating every effective policy.

    The protocol rejects this pack. The call sites were not changed to require
    the method, so a pack that never meets isinstance still runs every policy.
    """
    pack = _PackWithoutEngages()
    harness, ran = _harness(pack)
    evaluation = _evaluation(path, harness)

    assert evaluation.outcome is Outcome.DENY
    failed = _prohibited(evaluation)
    assert len(failed) == 1
    assert failed[0].result is CheckResult.FAIL
    if path == "run_governed":
        assert ran == []
