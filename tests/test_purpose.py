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
    """A purpose the deployment listed is still allowed."""
    listed = dp.PURPOSES[0]
    evaluation = decide(_envelope(listed), PolicySnapshot(), purposes=(listed,))
    check = _purpose(evaluation)
    assert check.result is CheckResult.PASS
    assert evaluation.outcome is Outcome.ALLOW


def test_configured_list_denies_an_unlisted_purpose():
    """A purpose outside the declared list is still denied."""
    listed = dp.PURPOSES[0]
    unlisted = next(purpose for purpose in dp.PURPOSES if purpose != listed)
    evaluation = decide(_envelope(unlisted), PolicySnapshot(), purposes=(listed,))
    check = _purpose(evaluation)
    assert check.result is not CheckResult.PASS
    assert check.result is CheckResult.FAIL
    assert evaluation.outcome is Outcome.DENY
