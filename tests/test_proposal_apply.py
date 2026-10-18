"""Proposal regression tests use real local transactions, without workload fixtures."""
import pytest
from sqlmodel import select

from kognita import approvals
from kognita.models import EvidenceEvent, KnowledgeItem, Proposal
from kognita.vocabulary import ActorType


def _draft(session, evidence):
    target = KnowledgeItem(title='Old Title', body='Original')
    session.add(target)
    session.flush()
    proposal = Proposal(kind='document_edit', before_snapshot={'title': 'Old Title'},
                        change={'title': 'New Title'}, proposer_id='agent',
                        subject_type='knowledge_items', subject_id=str(target.id))
    approvals._propose(session, proposal, evidence=evidence, correlation_id='review')
    session.commit()
    return target, proposal, approvals._proposal_hash(proposal)


def _apply(session, evidence, target, proposal, digest, **kwargs):
    return approvals._apply_proposal(
        session, proposal, target, diff_hash=digest,
        evidence=evidence, correlation_id='review',
        approver_id=kwargs.get('approver_id', 'owner'),
        actor_type=kwargs.get('actor_type', ActorType.HUMAN))


def test_proposal_must_bind_to_diff(session, evidence):
    target, proposal, digest = _draft(session, evidence)
    assert target.title == 'Old Title'
    proposal.change = {'title': 'Substituted'}
    session.add(proposal)
    session.commit()
    with pytest.raises(approvals.ApprovalError, match='PROPOSAL-BINDING'):
        _apply(session, evidence, target, proposal, digest)
    # Even a newly computed hash cannot replace the original review binding.
    with pytest.raises(approvals.ApprovalError, match='PROPOSAL-BINDING'):
        _apply(session, evidence, target, proposal, approvals._proposal_hash(proposal))
    assert session.get(KnowledgeItem, target.id).title == 'Old Title'


def test_human_applies_exact_diff_once_with_evidence(session, evidence):
    target, proposal, digest = _draft(session, evidence)
    _apply(session, evidence, target, proposal, digest)
    session.commit()
    assert target.title == 'New Title'
    events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    assert [e.payload['action'] for e in events] == ['PROPOSED', 'APPLIED']
    assert [e.actor_type for e in events] == [ActorType.AGENT, ActorType.HUMAN]
    assert [e.actor_id for e in events] == ['agent', 'owner']
    assert all(e.payload['_diff_hash'] == digest for e in events)
    assert all(e.payload['rule_id'] == 'PROPOSAL-BINDING' for e in events)
    with pytest.raises(approvals.ApprovalError, match='PROPOSAL-BINDING'):
        _apply(session, evidence, target, proposal, digest)


@pytest.mark.parametrize('kwargs', [
    {'approver_id': ' agent '}, {'approver_id': ''},
    {'actor_type': ActorType.AGENT}, {'actor_type': ActorType.SYSTEM},
])
def test_proposal_requires_distinct_human(session, evidence, kwargs):
    target, proposal, digest = _draft(session, evidence)
    with pytest.raises(approvals.ApprovalError):
        _apply(session, evidence, target, proposal, digest, **kwargs)
    assert session.get(KnowledgeItem, target.id).title == 'Old Title'
    _apply(session, evidence, target, proposal, digest)
    assert target.title == 'New Title'


def test_proposal_rejects_changed_before_state(session, evidence):
    target, proposal, digest = _draft(session, evidence)
    target.title = 'Concurrent edit'
    session.add(target)
    session.commit()
    with pytest.raises(approvals.ApprovalError, match='PROPOSAL-TARGET'):
        _apply(session, evidence, target, proposal, digest)
    assert target.title == 'Concurrent edit'


def test_proposal_apply_and_evidence_roll_back_together(session, evidence):
    target, proposal, digest = _draft(session, evidence)

    class _BrokenEvidence:
        def emit(self, *args, **kwargs):
            raise RuntimeError('evidence unavailable')

    with pytest.raises(RuntimeError, match='evidence unavailable'):
        _apply(session, _BrokenEvidence(), target, proposal, digest)
    assert session.get(KnowledgeItem, target.id).title == 'Old Title'
    _apply(session, evidence, target, proposal, digest)
    assert target.title == 'New Title'
