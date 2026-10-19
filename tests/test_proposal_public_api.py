"""Public proposal workflow, outside the legacy authority injection allowlist."""
from datetime import timedelta

import pytest

import kognita
from kognita import approvals
from kognita.canonical import canonical_hash
from kognita.credentials import issue
from kognita.governance import decide, load_snapshot, record
from kognita.models import KnowledgeItem, Proposal
from kognita.vocabulary import ActorType, Outcome
from conftest import NOW
from fixtures.agents import _usecases
from test_authority_s3 import _setup


def _draft(session, evidence):
    _, _, _, envelope = _setup(session, evidence)
    issue(session, 'worker', evidence=evidence, actor_id='owner')
    evaluation = decide(envelope, load_snapshot(session, as_of=NOW),
                        purposes=['READ'], as_of=NOW, usecases=_usecases(['READ']))
    assert evaluation.outcome is Outcome.ALLOW
    decision = record(session, evaluation, evidence=evidence, now=NOW)
    target = KnowledgeItem(title='Old', body='Original')
    session.add(target)
    session.flush()
    proposal = Proposal(kind='document_edit', before_snapshot={'title': 'Old'},
                        change={'title': 'New'}, proposer_id='agent',
                        subject_type='knowledge_items', subject_id=str(target.id))
    assert kognita.propose_document is approvals.propose_document
    assert kognita.apply_proposal_tool is approvals.apply_proposal_tool
    kognita.propose_document(session, proposal, evidence=evidence, correlation_id='public')
    digest = canonical_hash({key: getattr(proposal, key) for key in (
        'kind', 'before_snapshot', 'change', 'rationale', 'subject_type',
        'subject_id', 'proposer_id')})
    approval = approvals.open_approval(
        session, decision_id=decision.decision_id, envelope_hash=digest,
        proposal_id=proposal.id, requester_id='agent', scope='LOCAL',
        evidence=evidence, correlation_id='public', now=NOW)
    session.commit()
    return target, proposal, digest, approval


def _apply(session, evidence, target, proposal, digest, approval, now=NOW):
    return kognita.apply_proposal_tool(
        session, proposal, target, approval, diff_hash=digest, approver_id='owner',
        actor_type=ActorType.HUMAN, evidence=evidence, correlation_id='public', now=now)


def _grant(session, evidence, approval):
    approvals.grant(session, approval, approver_name='owner', evidence=evidence,
                    correlation_id='public', now=NOW)
    session.commit()


def test_public_propose_approve_apply(session, evidence):
    target, proposal, digest, approval = _draft(session, evidence)
    assert target.title == 'Old'
    _grant(session, evidence, approval)
    _apply(session, evidence, target, proposal, digest, approval)
    session.commit()
    assert target.title == 'New'
    with pytest.raises(approvals.ApprovalError, match='PROPOSAL-BINDING'):
        _apply(session, evidence, target, proposal, digest, approval)


def test_public_proposal_tampered_diff_denied(session, evidence):
    target, proposal, digest, approval = _draft(session, evidence)
    _grant(session, evidence, approval)
    proposal.change = {'title': 'Substituted'}
    session.add(proposal)
    session.commit()
    with pytest.raises(approvals.ApprovalError, match='PROPOSAL-BINDING'):
        _apply(session, evidence, target, proposal, digest, approval)
    assert target.title == 'Old'


@pytest.mark.parametrize('invalid', ['pending', 'expired', 'wrong_hash', 'wrong_proposal'])
def test_public_proposal_requires_exact_live_approval(session, evidence, invalid):
    target, proposal, digest, approval = _draft(session, evidence)
    if invalid != 'pending':
        _grant(session, evidence, approval)
    if invalid == 'wrong_hash':
        approval.envelope_hash = 'another diff'
    if invalid == 'wrong_proposal':
        approval.proposal_id = None
    session.commit()
    at = NOW + timedelta(days=3) if invalid == 'expired' else NOW
    with pytest.raises(approvals.ApprovalError, match='PROPOSAL-APPROVAL'):
        _apply(session, evidence, target, proposal, digest, approval, now=at)
    assert target.title == 'Old'
