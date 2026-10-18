"""The purpose check is a closed list.

A missing or empty list declares no purposes, so the check fails and the
decision denies. A list that is present still admits only its own members.
The names below are the demo pack's existing purposes; this test adds none.
"""
from __future__ import annotations

from fixtures import demo_pack as dp

from kognita.envelope import Envelope
from kognita.governance import PolicySnapshot, decide
from kognita.vocabulary import CheckResult, Outcome


def _envelope(purpose: str) -> Envelope:
    return Envelope(principal="alice", purpose=purpose, tool="get_subject_profile")


def _purpose(evaluation):
    matches = [check for check in evaluation.checks if check.check == "PURPOSE"]
    assert len(matches) == 1
    return matches[0]


def test_missing_purpose_list_does_not_pass():
    """Omitting the list, or passing none, is not permission to use any purpose."""
    envelope = _envelope(dp.PURPOSES[0])
    omitted = decide(envelope, PolicySnapshot())
    explicit_none = decide(envelope, PolicySnapshot(), purposes=None)

    for evaluation in (omitted, explicit_none):
        check = _purpose(evaluation)
        assert check.result is not CheckResult.PASS
        assert check.result is CheckResult.FAIL
        assert evaluation.outcome is Outcome.DENY


def test_empty_purpose_list_does_not_pass():
    """An empty tuple or list is the same closed failure as a missing list."""
    envelope = _envelope(dp.PURPOSES[0])
    for purposes in ((), []):
        evaluation = decide(envelope, PolicySnapshot(), purposes=purposes)
        check = _purpose(evaluation)
        assert check.result is not CheckResult.PASS
        assert check.result is CheckResult.FAIL
        assert evaluation.outcome is Outcome.DENY


def test_configured_list_allows_a_listed_purpose():
    """A purpose the deployment listed is still allowed, and cites the principal."""
    listed = dp.PURPOSES[0]
    evaluation = decide(_envelope(listed), PolicySnapshot(), purposes=(listed,))
    check = _purpose(evaluation)
    assert check.result is CheckResult.PASS
    assert evaluation.outcome is Outcome.ALLOW
    assert "accountable owner: alice" in check.citation


def test_configured_list_denies_an_unlisted_purpose():
    """A purpose outside the declared list is still denied, and still cites the owner."""
    listed = dp.PURPOSES[0]
    unlisted = next(purpose for purpose in dp.PURPOSES if purpose != listed)
    evaluation = decide(_envelope(unlisted), PolicySnapshot(), purposes=(listed,))
    check = _purpose(evaluation)
    assert check.result is not CheckResult.PASS
    assert check.result is CheckResult.FAIL
    assert evaluation.outcome is Outcome.DENY
    assert "accountable owner: alice" in check.citation


def test_a_listed_purpose_with_no_owner_fails_closed():
    """No agent owner and no principal is not a decision that names nobody."""
    listed = dp.PURPOSES[0]
    envelope = Envelope(principal="", purpose=listed, tool="get_subject_profile")
    evaluation = decide(envelope, PolicySnapshot(), purposes=(listed,))
    check = _purpose(evaluation)
    assert check.result is CheckResult.FAIL
    assert evaluation.outcome is Outcome.DENY
    assert "no accountable owner can be cited" in check.citation


def test_an_agent_decision_cites_the_agent_owner():
    """The purpose list stays; the owner is the agent's, not the principal's."""
    from kognita.models import Agent

    listed = dp.PURPOSES[0]
    snapshot = PolicySnapshot(
        agents={"dossier-agent": Agent(name="dossier-agent", owner_exec="Jane Smith")}
    )
    envelope = Envelope(
        principal="alice",
        purpose=listed,
        tool="get_subject_profile",
        agent_name="dossier-agent",
    )
    evaluation = decide(envelope, snapshot, purposes=(listed,))
    check = _purpose(evaluation)
    assert check.result is CheckResult.PASS
    assert "accountable owner: Jane Smith" in check.citation
    assert "alice" not in check.citation
