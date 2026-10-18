"""Approval and citation regressions, outside the legacy authority allowlist."""
from fixtures.agents import _usecases
from dataclasses import replace

import pytest

from conftest import NOW
from kognita.approvals import ApprovalError, confirm, grant, mark, open_approval, reject
from kognita.credentials import issue
from kognita.governance import decide, load_snapshot, record
from kognita.models import Policy
from kognita.review import PACKET_FIELDS, ReviewError, review
from kognita.vocabulary import ApprovalStatus, CheckResult, Outcome
from test_authority_s3 import _setup
from test_interaction import _boundary, _open, _packet


def _approval(session, evidence, requester="alice"):
    _, _, _, envelope = _setup(session, evidence)
    issue(session, "worker", evidence=evidence, actor_id="owner")
    evaluation = decide(envelope, load_snapshot(session, as_of=NOW), purposes=["READ"], as_of=NOW, usecases=_usecases(["READ"]))
    decision = record(session, evaluation, evidence=evidence, now=NOW)
    return open_approval(session, decision_id=decision.decision_id, envelope_hash="reviewed", scope="LOCAL",
                         requester_id=requester, confirmation_required=True,
                         evidence=evidence, correlation_id="integrity", now=NOW)


@pytest.mark.parametrize("requester,signer", [
    ("alice", ""), ("alice", " \t"), ("alice", " alice "),
    ("", "bob"), ("  ", "bob"),
])
def test_mark_requires_nonblank_distinct_identities(session, evidence, requester, signer):
    approval = _approval(session, evidence, requester)
    with pytest.raises(ApprovalError):
        mark(session, approval, approver_id=signer, evidence=evidence,
             correlation_id="integrity", now=NOW)
    assert approval.status == ApprovalStatus.PENDING


@pytest.mark.parametrize("requester,marker,confirmer", [
    ("alice", "", "bob"), ("alice", " ", "bob"),
    ("alice", "bob", " "), ("alice", "bob", " bob "),
    (" alice ", "bob", "alice"), ("", "bob", "carol"),
    ("alice", " alice ", "carol"),
])
def test_confirm_revalidates_all_three_identities(session, evidence, requester, marker, confirmer):
    approval = _approval(session, evidence, requester)
    # A pre-existing marked row must not bypass validation on confirmation.
    approval.status = ApprovalStatus.MARKED
    approval.approver_id = marker
    with pytest.raises(ApprovalError):
        confirm(session, approval, approver_id=confirmer, evidence=evidence,
                correlation_id="integrity", now=NOW)
    assert approval.status == ApprovalStatus.MARKED


def test_signature_identities_are_stripped(session, evidence):
    approval = _approval(session, evidence, " alice ")
    mark(session, approval, approver_id=" bob ", evidence=evidence,
         correlation_id="integrity", now=NOW)
    assert approval.approver_id == "bob"
    confirm(session, approval, approver_id=" carol ", evidence=evidence,
            correlation_id="integrity", now=NOW)
    assert approval.status == ApprovalStatus.APPROVED


@pytest.mark.parametrize("citation", ["", " \t"])
@pytest.mark.parametrize("result,expected", [
    ("pass", Outcome.ESCALATE), ("requires_human", Outcome.ESCALATE),
    ("fail", Outcome.DENY),
])
def test_blank_policy_citation_fails_closed(session, evidence, citation, result, expected):
    _, _, _, envelope = _setup(session, evidence)
    issue(session, "worker", evidence=evidence, actor_id="owner")
    baseline = decide(envelope, load_snapshot(session, as_of=NOW), purposes=["READ"], as_of=NOW, usecases=_usecases(["READ"]))
    assert baseline.outcome is Outcome.ALLOW
    policy = Policy(regime="INTERNAL", rule_type="PROHIBITED", citation=citation,
                    rule={"on_violation": result}, effective_from=NOW)
    session.add(policy)
    session.flush()
    snapshot = load_snapshot(session, as_of=NOW)
    evaluation = decide(envelope, snapshot, purposes=["READ"], as_of=NOW, usecases=_usecases(["READ"]))
    assert evaluation.outcome is expected
    check = next(c for c in evaluation.checks if c.policy_id == policy.id)
    assert check.citation == "CITATION-REQUIRED"
    assert check.result is (CheckResult.FAIL if result == "fail" else CheckResult.ESCALATE)
    assert check.policy_hash
    assert decide(replace(envelope, delegation_id=None), snapshot,
                  purposes=["READ"], as_of=NOW, usecases=_usecases(["READ"])).outcome is Outcome.DENY


@pytest.mark.parametrize("field", [f for f in PACKET_FIELDS if f not in {"agent_confidence", "policy_checks"}])
@pytest.mark.parametrize("value", [[""], [" "], 0, {}, ["valid", " "], [0], [None]])
def test_approval_packet_requires_text(session, evidence, field, value):
    interaction = _open(session, evidence)
    _boundary(session, evidence)
    with pytest.raises(ReviewError, match="incomplete approval packet"):
        review(session, interaction.id, rm="reviewer", outcome="approve",
               packet=_packet(**{field: value}), rendered="Send the note.",
               provenance="AI", evidence=evidence, now=NOW)


@pytest.mark.parametrize("signer", ["", " \t"])
def test_reject_requires_named_signer(session, evidence, signer):
    approval = _approval(session, evidence)
    with pytest.raises(ApprovalError):
        reject(session, approval, approver_name=signer, evidence=evidence,
               correlation_id="integrity", now=NOW)
    assert approval.status == ApprovalStatus.PENDING


def test_single_signature_is_stripped(session, evidence):
    approval = _approval(session, evidence)
    approval.confirmation_status = None
    grant(session, approval, approver_name=" bob ", evidence=evidence,
          correlation_id="integrity", now=NOW)
    assert approval.approver_name == "bob"
