"""Runnable conformance suite over the bundled fixture pack.

    pytest --pyargs kognita.testing

Proves the kit itself works. A domain pack subclasses
:class:`kognita.testing.conformance.ConformanceCase` in its own suite rather
than running this one.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from kognita.core.envelope import Envelope
from kognita.core.models import Policy
from kognita.core.registry import register
from kognita.core.rules import build_registry
from kognita.testing.conformance import ConformanceCase
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
    """The kit, run against the minimal pack."""

    @pytest.fixture(autouse=True)
    def _bind(self, _harness):
        self.harness = _harness
        self.allow_envelope = _envelope("read", "open")
        self.deny_envelope = _envelope("read", "sealed")
        self.human_envelope = _envelope("publish", "open")
