"""Durable gateway holds with real workload authority."""
import pytest
from sqlmodel import select

from conftest import NOW
from kognita.approvals import ApprovalError
from kognita.models import Approval, RunRecord
from kognita.tools import Run, continue_run
from kognita.vocabulary import Outcome
from test_approval_continuation import _hold
from test_gateway import PATH, _body
from test_gateway_run_controls import _setup


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    yield


@pytest.mark.parametrize('change', ['none', 'credential', 'delegation', 'control'])
def test_gateway_run_approval_continuation(session, evidence, change):
    gateway, backend, headers = _setup(session, evidence)
    _hold(session, 'model_call')
    run = Run()
    body = _body('exact held question')
    held = gateway.handle('POST', PATH, headers, body, session=session, run=run)
    assert held.evaluation.outcome is Outcome.HUMAN_APPROVAL
    assert backend.calls == []
    row = session.get(RunRecord, run.id)
    approval = session.exec(select(Approval).where(
        Approval.decision_id == held.evaluation.decision_id)).one()
    assert row.continuation_hash
    assert approval.id in row.approvals_pending
    session.commit()
    if change == 'credential':
        from kognita.credentials import revoke
        revoke(session, headers['agent_name'], headers['agent_credential'],
               evidence=evidence, actor_id='Head of Research Ops')
    elif change == 'delegation':
        from kognita.authority import revoke as revoke_delegation
        revoke_delegation(session, headers['delegation_id'], evidence=evidence, actor_id='alice')
    elif change == 'control':
        row.control_state = 'paused'
        session.add(row)
        session.flush()
    result = continue_run(session, run.id, {approval.id: True}, evidence=evidence,
                          approver_name='reviewer', now=NOW)
    if change == 'none':
        assert result.status == 200
        assert backend.calls[0]['body'] == body
        assert len(backend.calls) == 1
        with pytest.raises(ApprovalError, match='no continuation'):
            continue_run(session, run.id, {approval.id: True}, evidence=evidence,
                         approver_name='reviewer', now=NOW)
        retry = gateway.handle('POST', PATH, headers, body, session=session, run=Run())
        assert retry.evaluation.outcome is Outcome.DENY
        assert len(backend.calls) == 1
    else:
        assert result.evaluation.outcome in (Outcome.DENY, Outcome.ESCALATE)
        assert backend.calls == []
