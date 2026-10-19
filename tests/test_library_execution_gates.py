"""Library execution gates with real authority and no legacy injection."""
import pytest
from sqlmodel import select

from conftest import NOW
from kognita.broker import ask
from kognita.envelope import Envelope
from kognita.models import UseCase, _WorkloadCredential
from kognita.tools import Run, ToolRegistry, run_governed
from kognita.vocabulary import Outcome
from test_gateway import PATH, _body
from test_gateway_run_controls import _setup
from test_mcp import _Pack, _Upstream, _call, _proxy
from test_run_binding_broker import _index


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Keep production ingress and explicit credentials intact."""


def _tier_two(session, evidence):
    from kognita.usecases import revise, approve_rating, activate
    from test_usecases import _spec

    session.flush()
    row = session.exec(select(UseCase).where(UseCase.usecase_id == 'COLLABORATION',
                                             UseCase.effective_to.is_(None))).one()
    spec = _spec(usecase_id='COLLABORATION', data_sensitivity='confidential',
        deployment_mode='gateway', knowledge_lead='Jane Smith',
        approval_gate='human_review', explanation_depth='sources',
        approved_models=tuple(row.record['approved_models']))
    revise(session, spec, evidence=evidence, actor_id='Jane Smith', at=NOW)
    approve_rating(session, spec.usecase_id, spec.control_function, evidence=evidence, at=NOW)
    activate(session, spec.usecase_id, evidence=evidence, actor_id=spec.control_function,
             at=NOW, status='live')
    from kognita.review import define_risk_criteria
    define_risk_criteria(session, spec.usecase_id, product_complexity_at='high',
        vulnerability_indicators=('bereavement',), profile_deviation_is_high_risk=False,
        minimum_classifier_confidence=0, timing='after delivery', evidence=evidence,
        actor_id='Priya Shah', now=NOW)


def _library(session, evidence, headers, path):
    env = Envelope(principal='alice', agent_name=headers['agent_name'],
        purpose='COLLABORATION', tool='read_note', actor_location='SG',
        delegation_id=headers['delegation_id'],
        context={'_agent_credential': headers.get('agent_credential')})
    if path == 'broker':
        result = ask(session, 'meeting notes', env, pack=_Pack(), embedder=_index(session),
            evidence=evidence, purposes=['COLLABORATION'], run=Run(), now=NOW, as_of=NOW)
        if result.evaluation.outcome is Outcome.DENY:
            assert result.results == []
        return result.evaluation
    registry = ToolRegistry()
    calls = []
    registry.register('read_note', lambda *args: calls.append(True) or {'marked': True})
    result = run_governed(session, env, registry=registry, pack=_Pack(), evidence=evidence,
        purposes=['COLLABORATION'], run=Run(), now=NOW, as_of=NOW)
    if result.evaluation.outcome is Outcome.DENY:
        assert calls == []
    return result.evaluation


@pytest.mark.parametrize('path', ['broker', 'tool'])
def test_library_requires_a_live_credential_even_without_issued_rows(session, evidence, path):
    _, _, headers = _setup(session, evidence)
    for row in session.exec(select(_WorkloadCredential)).all():
        session.delete(row)
    session.flush()
    headers.pop('agent_credential')
    evaluation = _library(session, evidence, headers, path)
    assert evaluation.outcome is Outcome.DENY
    assert 'S2-CREDENTIAL' in [c.citation for c in evaluation.failures()]


@pytest.mark.parametrize('path', ['broker', 'tool'])
def test_tier_two_library_execution_is_denied(session, evidence, path):
    _, _, headers = _setup(session, evidence)
    _tier_two(session, evidence)
    evaluation = _library(session, evidence, headers, path)
    assert evaluation.outcome is Outcome.DENY
    assert 'S2-DEPLOYMENT-MODE' in [c.citation for c in evaluation.failures()]


@pytest.mark.parametrize('ingress', ['gateway', 'mcp'])
def test_tier_two_ingress_execution_is_allowed(session, evidence, ingress):
    gateway, backend, headers = _setup(session, evidence)
    _tier_two(session, evidence)
    if ingress == 'gateway':
        response = gateway.handle('POST', PATH, headers, _body('hello'), session=session, run=Run())
    else:
        backend = _Upstream()
        proxy = _proxy(session.get_bind(), evidence, backend)
        from kognita._fleet import _run_checks
        run = Run()
        env = Envelope(principal='alice', purpose='COLLABORATION', tool='read_note',
            agent_name=headers['agent_name'], delegation_id=headers['delegation_id'])
        _run_checks(session, run, env, evidence)
        response = proxy.handle('POST', '/mcp', {**headers, 'run_id': run.id},
                                _call(), session=session)
    assert response.evaluation.outcome is Outcome.ALLOW, response.evaluation.failures()
    assert response.status == 200
    assert len(backend.calls) == 1


def test_tier_two_adapter_tool_execution_is_denied(session, evidence, monkeypatch):
    from kognita.adapters.binding import Binding

    monkeypatch.setattr("kognita.tools.utcnow", lambda: NOW)
    monkeypatch.setattr("kognita.governance.utcnow", lambda: NOW)
    gateway, _, headers = _setup(session, evidence)
    _tier_two(session, evidence)
    calls = []
    binding = Binding(session=session, evidence=evidence, pack=_Pack(), gateway=gateway,
        registry=ToolRegistry(), principal='alice', purpose='COLLABORATION',
        agent_name=headers['agent_name'], agent_credential=headers['agent_credential'],
        delegation_id=headers['delegation_id'], purposes=['COLLABORATION'],
        actor_location='SG', run=Run())
    result = binding.call_tool('read_note', {}, lambda: calls.append(True) or {'marked': True})
    assert result.outcome is Outcome.DENY
    assert 'S2-DEPLOYMENT-MODE' in result.citations
    assert not result.released
    assert calls == []
