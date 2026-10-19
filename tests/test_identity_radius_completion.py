"""DoD 1/4: real credentials and authority, outside legacy fixture allowlists."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
from threading import Barrier

import pytest
from sqlmodel import Session, select

from conftest import NOW
from kognita.approvals import grant
from kognita.authority import Delegation, delegate, grant_to
from kognita.credentials import issue
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter
from kognita.models import Agent, Approval, _BlastUsage
from kognita.registry import approve_deployment, register
from kognita.vocabulary import Outcome
from test_gateway import MODEL, PATH, _body, _gateway, _live, _Upstream as _ModelBackend
from test_mcp import _register, _proxy, _call, _Upstream as _ToolBackend


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Exercise the original handlers with explicit credentials."""
    yield


def _setup(session, evidence, engine, path, *, transaction=False, limits=None):
    if path == 'gateway':
        _live(session)
    _register(session)
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    # Explicitly set approved identity rather than relying on credential injection.
    object.__setattr__(agent, 'deployment_date', NOW)
    object.__setattr__(agent, 'model', 'approved-model')
    agent.blast_radius = limits
    if transaction:
        from test_usecases import _tier3
        from kognita.usecases import load_register, revise, approve_rating, activate
        entry = load_register(session, as_of=NOW).get('COLLABORATION')
        spec = _tier3(usecase_id='COLLABORATION', purpose='COLLABORATION',
            approved_models=entry.approved_models)
        revise(session, spec, evidence=evidence, actor_id='Jane Smith', at=NOW)
        approve_rating(session, spec.usecase_id, spec.control_function, evidence=evidence, at=NOW)
        activate(session, spec.usecase_id, evidence=evidence, actor_id=spec.control_function,
            at=NOW, status='live', signoff_by='Independent Risk')
        from kognita.review import define_risk_criteria
        define_risk_criteria(session, spec.usecase_id, product_complexity_at='high',
            vulnerability_indicators=['bereavement'], profile_deviation_is_high_risk=True,
            minimum_classifier_confidence=0.8, timing='after delivery',
            evidence=evidence, actor_id='Independent Risk', now=NOW)
        from kognita.breaker import schedule_kill_switch_drill
        schedule_kill_switch_drill(session, spec.usecase_id, due_at=NOW + timedelta(days=30),
            evidence=evidence, actor_id='Independent Risk', citation='TEST-DRILL', now=NOW)
        from kognita.autonomy import _change
        permissions = deepcopy(agent.tool_allowlist)
        tool = 'model_call' if path == 'gateway' else 'read_note'
        permissions[tool].update(verb='execute', required_level=4,
            permissions=['execute'], reversible=True, compensation='undo')
        agent.tool_allowlist = permissions
        _change(session, agent, 4, evidence, agent.owner_exec, 'AUTONOMY_PROMOTED', {})
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
        evidence=evidence, actor_id=agent.owner_exec)
    scope = {'client': ['123'], 'product': ['permitted', 'other'], 'model': [MODEL]}
    verb = 'execute' if transaction else 'read'
    grant_to('alice', verb, timedelta(hours=2), scope, session=session,
        evidence=evidence, actor_id='admin', now=NOW)
    did = delegate(session, Delegation(human='alice', agent=agent.name,
        purpose='COLLABORATION', client='123', allowed_actions=[verb], allowed_data=scope,
        valid_from=NOW, valid_until=NOW + timedelta(hours=1), jurisdiction='SG',
        approval_threshold='execute', task_ref='completion'), evidence=evidence, actor_id='alice', now=NOW)
    credential = issue(session, agent.name, evidence=evidence, actor_id=agent.owner_exec)
    headers = {'agent_name': agent.name, 'agent_credential': credential, 'delegation_id': did}
    if transaction:
        from kognita.tools import Run, _save_run
        run = Run(max_calls=100, assurance_level=2)
        _save_run(session, run)
        from kognita.models import _RunScope
        session.add(_RunScope(id=run.id, body=dict(agent=agent.name, human='alice',
            purpose='COLLABORATION', delegation_id=did)))
        session.flush()
        headers.update(assurance_level='2', run_id=run.id)
    return agent, headers


def _invoke(engine, evidence, session, path, headers, arguments=None):
    from kognita.tools import Run
    run_args = {'run': Run(id=headers['run_id'])} if 'run_id' in headers else {}
    if path == 'gateway':
        backend = _ModelBackend(b'{"model":"gateway-model","choices":[]}')
        response = _gateway(engine, evidence, backend, **run_args).handle('POST', PATH, headers,
            _body('hello'), session=session)
    else:
        backend = _ToolBackend()
        response = _proxy(engine, evidence, backend).handle('POST', '/mcp', headers,
            _call(arguments=arguments or {}), session=session)
    if response.evaluation and 'S5-STEP-UP' in [c.citation for c in response.evaluation.failures()]:
        from kognita._isolation import _step_up
        _step_up(session, response.evaluation.envelope, evidence=evidence, actor_id='alice',
            authenticated_workflow=True, transaction_context='completion',
            independent_authorization='reviewer', expires_at=NOW + timedelta(hours=1))
        return _invoke(engine, evidence, session, path, headers, arguments)
    return response, backend.calls


@pytest.mark.parametrize('path', ['gateway', 'mcp'])
@pytest.mark.parametrize('field', ['deployment_date', 'model'])
def test_missing_identity_denies_before_transport(session, evidence, engine, path, field):
    agent, headers = _setup(session, evidence, engine, path)
    assert _invoke(engine, evidence, session, path, headers)[0].status == 200
    object.__setattr__(agent, field, None)
    session.flush()
    response, calls = _invoke(engine, evidence, session, path, headers)
    assert response.evaluation.outcome is Outcome.DENY
    assert 'S1-IDENTITY' in [c.citation for c in response.evaluation.failures()]
    assert calls == []


@pytest.mark.parametrize('path', ['gateway', 'mcp'])
def test_changed_model_requires_reapproval(session, evidence, engine, path):
    agent, headers = _setup(session, evidence, engine, path)
    object.__setattr__(agent, 'model', 'changed-model')
    session.flush()
    response, calls = _invoke(engine, evidence, session, path, headers)
    assert response.evaluation.outcome is Outcome.DENY
    assert 'S1-DEPLOYMENT' in [c.citation for c in response.evaluation.failures()]
    assert calls == []
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
        evidence=evidence, actor_id=agent.owner_exec)
    assert _invoke(engine, evidence, session, path, headers)[0].status == 200


@pytest.mark.parametrize('arguments', [
    {'product': 'other'}, {'filter': {'product_id': 'other'}},
    {'items': [{'subject_type': 'product', 'subject_id': 'other'}]},
    {'product': ['permitted', 'other']},
])
def test_product_arguments_checked_against_allowlist(session, evidence, engine, arguments):
    _, headers = _setup(session, evidence, engine, 'mcp', limits={'products': ['permitted']})
    response, calls = _invoke(engine, evidence, session, 'mcp', headers, arguments)
    assert response.evaluation.outcome is Outcome.DENY
    assert 'S4-PRODUCTS' in [c.citation for c in response.evaluation.failures()]
    assert calls == []
    assert _invoke(engine, evidence, session, 'mcp', headers, {'product': 'permitted'})[0].status == 200


@pytest.mark.parametrize('path', ['gateway', 'mcp'])
def test_daily_transactions_released_only_and_read_calls_exempt(session, evidence, engine, path):
    _, headers = _setup(session, evidence, engine, path, transaction=True,
        limits={'max_daily_transactions': 1, 'execution_authority': 'execute'})
    held, calls = _invoke(engine, evidence, session, path, headers)
    assert held.evaluation.outcome is Outcome.HUMAN_APPROVAL, held.evaluation.failures()
    assert calls == []
    assert not any(row.is_transaction for row in session.exec(select(_BlastUsage)).all())
    approval = session.exec(select(Approval)).one()
    grant(session, approval, approver_name='reviewer', evidence=evidence,
        correlation_id='completion', now=NOW)
    released, calls = _invoke(engine, evidence, session, path, headers)
    assert released.status == 200, released.evaluation.failures()
    assert len(calls) == 1
    blocked, calls = _invoke(engine, evidence, session, path, headers)
    assert blocked.evaluation.outcome is Outcome.DENY
    assert 'S4-DAILY-TRANSACTIONS' in [c.citation for c in blocked.evaluation.failures()]
    assert calls == []


def test_transaction_cap_atomic_on_approved_concurrent_release(tmp_path):
    engine = make_engine(tmp_path / 'transactions.db')
    create_all(engine)
    evidence = EvidenceWriter(engine, clock=lambda: NOW)
    # Separate approved envelopes all contend for the same agent/day quota.
    with Session(engine) as session:
        _, headers = _setup(session, evidence, engine, 'mcp', transaction=True,
            limits={'max_daily_transactions': 2, 'execution_authority': 'execute'})
        held = []
        for index in range(6):
            response, _ = _invoke(engine, evidence, session, 'mcp', headers, {'query': str(index)})
            assert response.evaluation.outcome is Outcome.HUMAN_APPROVAL
            held.append(response.evaluation)
        for approval in session.exec(select(Approval)).all():
            grant(session, approval, approver_name='reviewer', evidence=evidence,
                correlation_id='completion', now=NOW)
        session.commit()
    barrier = Barrier(6)
    def work(index):
        from dataclasses import replace
        from uuid import uuid4
        from kognita._limits import _recheck
        from kognita.approvals import _resume
        from kognita.governance import record
        with Session(engine) as session:
            barrier.wait(timeout=10)
            # Real approved decisions race at the production admission lock.
            checked = _recheck(session, held[index], NOW)
            released = _resume(session, checked, evidence, NOW)
            result = record(session, replace(released, request_id=str(uuid4())),
                evidence=evidence, now=NOW)
            session.commit()
            return result
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(work, range(6)))
    assert sum(result.outcome is Outcome.ALLOW for result in results) == 2
    for result in results:
        if result.outcome is not Outcome.ALLOW:
            assert result.outcome is Outcome.DENY
            assert 'S4-DAILY-TRANSACTIONS' in [c.citation for c in result.failures()]
    with Session(engine) as session:
        assert sum(bool(row.is_transaction) for row in session.exec(select(_BlastUsage)).all()) == 2


def test_registry_accepts_approved_identity_fields(session, evidence):
    agent = register(session, name='new-agent', owner_exec='owner',
        deployment_date=NOW, model='registered-model', evidence=evidence, actor_id='owner')
    assert agent.deployment_date == NOW
    assert agent.model == 'registered-model'


@pytest.mark.parametrize('path', ['gateway', 'mcp'])
def test_read_calls_do_not_spend_transaction_cap(session, evidence, engine, path):
    _, headers = _setup(session, evidence, engine, path, limits={'max_daily_transactions': 0})
    for _ in range(2):
        response, calls = _invoke(engine, evidence, session, path, headers)
        assert response.status == 200, response.evaluation.failures()
        assert len(calls) == 1
    assert not any(row.is_transaction for row in session.exec(select(_BlastUsage)).all())


@pytest.mark.parametrize('prior_day', [True, False])
def test_transaction_day_starts_at_utc_midnight(session, evidence, engine, prior_day):
    _, headers = _setup(session, evidence, engine, 'mcp', transaction=True,
        limits={'max_daily_transactions': 1, 'execution_authority': 'execute'})
    midnight = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    session.add(_BlastUsage(request_id='prior', agent_name='dossier-agent',
        at=midnight - timedelta(microseconds=1) if prior_day else midnight,
        is_transaction=True))
    session.flush()
    response, calls = _invoke(engine, evidence, session, 'mcp', headers)
    assert response.evaluation.outcome is (Outcome.HUMAN_APPROVAL if prior_day else Outcome.DENY)
    if not prior_day:
        assert 'S4-DAILY-TRANSACTIONS' in [c.citation for c in response.evaluation.failures()]
    assert calls == []


@pytest.mark.parametrize('path', ['gateway', 'mcp'])
def test_mapped_product_uses_resolved_subjects(session, evidence, engine, path):
    agent, headers = _setup(session, evidence, engine, path, limits={'products': ['permitted']})
    tool = 'model_call' if path == 'gateway' else 'read_note'
    permissions = deepcopy(agent.tool_allowlist)
    permissions[tool]['_subject_paths'] = {'product': ['model_version' if path == 'gateway' else 'sku']}
    agent.tool_allowlist = permissions
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
        evidence=evidence, actor_id=agent.owner_exec)
    if path == 'gateway':
        headers['model_version'] = 'other'
    response, calls = _invoke(engine, evidence, session, path, headers, {'sku': 'other'})
    assert response.evaluation.outcome is Outcome.DENY
    assert 'S4-PRODUCTS' in [c.citation for c in response.evaluation.failures()]
    assert calls == []


def test_additive_identity_migration_preserves_null_existing_rows(tmp_path):
    engine = make_engine(tmp_path / 'legacy.db')
    with engine.begin() as connection:
        connection.exec_driver_sql('CREATE TABLE agents (id INTEGER PRIMARY KEY, name TEXT)')
        connection.exec_driver_sql("INSERT INTO agents (name) VALUES ('legacy')")
    create_all(engine)
    create_all(engine)
    with engine.connect() as connection:
        assert connection.exec_driver_sql('SELECT deployment_date, model FROM agents').one() == (None, None)
    engine.dispose()
