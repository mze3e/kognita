"""Release regressions using real credentials and delegated authority."""
from copy import deepcopy
from datetime import timedelta
from dataclasses import replace

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.approvals import grant
from kognita.broker import ask
from kognita.credentials import issue, revoke
from kognita.models import Approval, Continuation, EvidenceEvent, ReviewQueueItem, _ApprovalContinuation
from kognita.registry import get, approve_deployment
from kognita.review import define_risk_criteria
from kognita.tools import Run, ToolRegistry, continue_run, run_governed
from kognita.vocabulary import Outcome
from test_approval_continuation import _hold
from test_authority_s3 import _setup
from test_mcp import _Pack
from test_run_binding_broker import _index
from test_usecases import _tier2, _tier3
from kognita.usecases import revise, approve_rating, activate


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Do not wrap ingress or inject credentials in this module."""


def _review_required(session, evidence, execution=False):
    from fixtures.evaluation import evaluation_thresholds
    spec = (_tier3 if execution else _tier2)(
        usecase_id='READ', purpose='READ', approved_models=(),
        evaluation_thresholds=evaluation_thresholds(),
    )
    revise(session, spec, evidence=evidence, actor_id='Jane Smith', at=NOW)
    approve_rating(session, 'READ', spec.control_function, evidence=evidence, at=NOW)
    activate(session, 'READ', evidence=evidence, actor_id=spec.control_function, at=NOW, status='live', **({'signoff_by': 'Independent Risk'} if execution else {}))


def _criteria(session, evidence, timing):
    define_risk_criteria(session, 'READ', product_complexity_at='high',
        vulnerability_indicators=('bereavement',), profile_deviation_is_high_risk=True,
        minimum_classifier_confidence=0.8, timing=timing,
        evidence=evidence, actor_id='reviewer', now=NOW)


def _tool(session, evidence, *, execute=False, review=False):
    _, _, _, env = _setup(session, evidence, verb='execute' if execute else 'read',
                           threshold='execute' if execute else 'read')
    credential = issue(session, 'worker', evidence=evidence, actor_id='owner')
    # Exercise resumed tier-2+ releases after trusted ingress admission.
    env = replace(env, context={'_agent_credential': credential, '_governed_ingress': True})
    if execute:
        _review_required(session, evidence, execution=True)
        _criteria(session, evidence, 'after delivery')
        agent = get(session, 'worker')
        permissions = deepcopy(agent.tool_allowlist)
        permissions['read_note']['required_level'] = 4
        from kognita.autonomy import _change
        _change(session, agent, 4, evidence, 'owner', 'AUTONOMY_PROMOTED', {})
        permissions['read_note']['_sandbox'] = {'network': ['api.example.org:443']}
        agent.tool_allowlist = permissions
        from fixtures.evaluation import satisfy_evaluation
        satisfy_evaluation(session, evidence, agent)
        approve_deployment(session, 'worker', deployment=permissions['_deployment'], evidence=evidence, actor_id='owner')
    if execute:
        from kognita.breaker import schedule_kill_switch_drill
        schedule_kill_switch_drill(session, 'READ', due_at=NOW + timedelta(days=30), evidence=evidence,
            actor_id='owner', citation='test drill', now=NOW)
        from kognita._isolation import _step_up
        env = replace(env, assurance_level=2)
        _step_up(session, env, evidence=evidence, actor_id='alice', authenticated_workflow=True,
            transaction_context='execute job', independent_authorization='risk', expires_at=NOW + timedelta(minutes=5))
    if review:
        _review_required(session, evidence)
    _hold(session, env.tool)
    calls = []
    registry = ToolRegistry()
    registry.register(env.tool, lambda e, v, s: calls.append(v.attributes.get('_sandbox_constraints')) or 'unreviewed answer',
                      verb='execute' if execute else 'read', required_level=4 if execute else 0)
    run = Run()
    def invoke():
        return run_governed(session, env, registry=registry, evidence=evidence,
                            pack=_Pack(), purposes=['READ'], run=run, now=NOW, as_of=NOW)
    held = invoke()
    assert held.outcome is Outcome.HUMAN_APPROVAL, held.evaluation.basis()
    return env, credential, registry, run, calls, invoke


def _resume(session, evidence, registry, run):
    return continue_run(session, run.id, {run.approvals_pending[0]: True},
        registry=registry, evidence=evidence, approver_name='reviewer', now=NOW)


@pytest.mark.parametrize('continuation', [True, False])
def test_approved_execution_restores_and_evidences_sandbox(session, evidence, continuation):
    env, _, registry, run, calls, invoke = _tool(session, evidence, execute=True)
    if continuation:
        result = _resume(session, evidence, registry, run)
    else:
        approval = session.exec(select(Approval)).one()
        grant(session, approval, approver_name='reviewer', evidence=evidence, correlation_id='review', now=NOW)
        result = invoke()
    assert result.data == 'unreviewed answer'
    from kognita._sandbox import _constraints
    expected, _ = _constraints(env, get(session, 'worker'))
    assert calls == [expected]
    assert result.evaluation.attributes['_sandbox_constraints'] == expected
    assert any(e.payload.get('attributes', {}).get('_sandbox_constraints') == expected
               for e in session.exec(select(EvidenceEvent)).all())


def test_continuation_rechecks_exact_credential_revocation(session, evidence):
    _, credential, registry, run, calls, _ = _tool(session, evidence)
    revoke(session, 'worker', credential, evidence=evidence, actor_id='owner')
    # Another live credential must not authorize the held identity.
    issue(session, 'worker', evidence=evidence, actor_id='owner')
    result = _resume(session, evidence, registry, run)
    assert result.outcome is Outcome.DENY
    assert any(c.citation == 'S2-CREDENTIAL' for c in result.evaluation.failures())
    assert calls == []
    payload = session.exec(select(Continuation)).one().payload
    assert credential not in str(payload)
    from kognita.credentials import _digest
    assert payload["_credential_digest"] == _digest(credential)


@pytest.mark.parametrize('continuation, criteria', [(True, True), (True, False), (False, False)])
def test_resumed_delivery_is_screened_and_missing_required_criteria_hold(session, evidence, criteria, continuation):
    _, _, registry, run, calls, invoke = _tool(session, evidence, review=True)
    if criteria:
        _criteria(session, evidence, 'before delivery')
    if continuation:
        result = _resume(session, evidence, registry, run)
    else:
        grant(session, session.get(Approval, run.approvals_pending[0]), approver_name='reviewer',
            evidence=evidence, correlation_id='review', now=NOW)
        result = invoke()
    assert len(calls) == 1
    assert result.data is None
    assert result.outcome is Outcome.DENY
    queue = session.exec(select(ReviewQueueItem)).one()
    assert queue.timing == 'before delivery'
    if not criteria:
        assert 'no risk criteria' in queue.reasons
    assert any(c.check == 'REVIEW' and c.citation for c in result.evaluation.failures())


@pytest.mark.parametrize('continue_first', [False, True])
def test_broker_approval_is_consumed_once_across_release_paths(session, evidence, continue_first):
    _, _, _, env = _setup(session, evidence)
    agent = get(session, 'worker')
    agent.field_allowlist = {'READ': ['results.snippet']}
    approve_deployment(session, 'worker', deployment=agent.tool_allowlist['_deployment'], evidence=evidence, actor_id='owner')
    _hold(session, env.tool)
    run = Run()
    embedder = _index(session)
    def invoke():
        return ask(session, 'meeting notes', env, pack=_Pack(), embedder=embedder,
            evidence=evidence, purposes=['READ'], run=run, now=NOW, as_of=NOW)
    held = invoke()
    assert held.outcome is Outcome.HUMAN_APPROVAL
    aid = run.approvals_pending[0]
    if continue_first:
        released = continue_run(session, run.id, {aid: True}, embedder=embedder,
            evidence=evidence, approver_name='reviewer', now=NOW)
    else:
        grant(session, session.get(Approval, aid), approver_name='reviewer',
              evidence=evidence, correlation_id='review', now=NOW)
        released = invoke()
    assert released.results
    binding = session.get(_ApprovalContinuation, held.evaluation.envelope_hash, populate_existing=True)
    assert binding.consumed
    repeated = invoke()
    assert repeated.outcome is Outcome.DENY
    assert repeated.results == []
    if not continue_first:
        assert continue_run(session, run.id, {aid: True}, embedder=embedder,
            evidence=evidence, approver_name='reviewer', now=NOW).outcome is Outcome.DENY


def test_broker_cannot_release_an_older_grant_over_the_exact_pending_approval(session, evidence):
    from kognita.governance import record
    _, _, _, env = _setup(session, evidence)
    _hold(session, env.tool)
    embedder = _index(session)
    run = Run()
    def invoke():
        return ask(session, 'meeting notes', env, pack=_Pack(), embedder=embedder,
            evidence=evidence, purposes=['READ'], run=run, now=NOW, as_of=NOW)
    held = invoke()
    old = session.get(Approval, run.approvals_pending[0])
    grant(session, old, approver_name='reviewer', evidence=evidence, correlation_id='review', now=NOW)
    # A newer decision places this exact envelope behind a fresh pending hold.
    record(session, replace(held.evaluation, request_id="new-decision", decision_id=None), evidence=evidence, now=NOW)
    current = session.get(_ApprovalContinuation, held.evaluation.envelope_hash, populate_existing=True)
    assert current.approval_id != old.id
    answer = invoke()
    assert answer.results == []
    assert not any(e.event_type.value == 'RETRIEVAL' for e in session.exec(select(EvidenceEvent)).all())
