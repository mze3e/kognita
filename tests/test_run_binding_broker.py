"""Real authority regressions; deliberately outside the legacy authority allowlist."""
import pytest

from conftest import NOW
from kognita.broker import ask, _resume_retrieval
from kognita.embedding import HashingEmbedder
from kognita.models import RunRecord
from kognita.retrieval import index_item
from kognita.tools import Run
from kognita.vocabulary import Classification, Outcome
from test_fleet_s6b import _family, _child, _env
from test_mcp import _Pack, _Upstream, _proxy, _call
from test_mcp_client_argument_scope import _delegated


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Use explicit real credentials without wrapping ingress methods."""


def _index(session):
    embedder = HashingEmbedder()
    index_item(session, title='meeting notes', body='meeting notes confidential fragment',
               embedder=embedder, kind='note', classification=Classification.C1,
               zones=['SG'], source_label='private-source')
    return embedder


def _ask(session, evidence, env, run, embedder):
    return ask(session, 'meeting notes', env, pack=_Pack(), embedder=embedder,
               evidence=evidence, purposes=['READ'], run=run, now=NOW, as_of=NOW)


@pytest.mark.parametrize('replacement', ['none', 'fresh', 'parent'])
def test_broker_child_run_isolation(session, evidence, replacement):
    _, _, env, parent = _family(session, evidence)
    child = _child(session, evidence, parent)
    embedder = _index(session)
    before = parent.calls_used
    answer = _ask(session, evidence, _env(env, child),
                  {'none': None, 'fresh': Run(), 'parent': parent}[replacement], embedder)
    assert answer.outcome is Outcome.DENY
    assert 'S6B-RUN-ISOLATION' in [c.citation for c in answer.evaluation.failures()]
    assert not answer.results
    assert not session.get(RunRecord, parent.id).external
    assert parent.calls_used == before


def test_broker_retrieval_taints_run_external(session, evidence):
    _, _, env, run = _family(session, evidence)
    answer = _ask(session, evidence, env, run, _index(session))
    assert answer.outcome is Outcome.ALLOW, answer.evaluation.failures()
    assert session.get(RunRecord, run.id).external
    from kognita._fleet import _run_checks
    rotated = Run()
    _run_checks(session, rotated, env, evidence)
    assert rotated.external


def test_broker_empty_field_allowlist_releases_no_content(session, evidence):
    from kognita.registry import get, approve_deployment
    _, _, env, run = _family(session, evidence)
    agent = get(session, env.agent_name)
    agent.field_allowlist = {'READ': []}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id='owner')
    answer = _ask(session, evidence, env, run, _index(session))
    assert answer.outcome is Outcome.ALLOW, answer.evaluation.failures()
    assert answer.summary == []
    assert answer.results == []
    assert answer.citations == []
    assert answer.graph is None


def test_broker_resumed_response_is_minimized_and_tainted(session, evidence):
    from kognita.governance import decide, load_snapshot
    _, _, env, run = _family(session, evidence)
    evaluation = decide(env, load_snapshot(session, as_of=NOW, _agent_name='worker'),
                        purposes=['READ'], as_of=NOW)
    assert evaluation.outcome is Outcome.ALLOW
    answer = _resume_retrieval(session, {'question': 'meeting notes'}, envelope=env,
        evaluation=evaluation, embedder=_index(session), evidence=evidence, subgraph=None, run=run)
    assert answer.summary == [] and answer.results == []
    assert session.get(RunRecord, run.id).external


def test_broker_resumed_retrieval_rechecks_run(session, evidence):
    from kognita.governance import decide, load_snapshot
    _, _, env, parent = _family(session, evidence)
    child = _child(session, evidence, parent)
    env = _env(env, child)
    evaluation = decide(env, load_snapshot(session, as_of=NOW, _agent_name='child'),
                        purposes=['READ'], as_of=NOW)
    assert evaluation.outcome is Outcome.ALLOW
    answer = _resume_retrieval(session, {'question': 'meeting notes'}, envelope=env,
        evaluation=evaluation, embedder=_index(session), evidence=evidence, subgraph=None, run=parent)
    assert answer.outcome is Outcome.DENY
    assert not answer.results
    assert not session.get(RunRecord, parent.id).external


@pytest.mark.parametrize('run_id', ['unknown', 'unbound', 'foreign'])
def test_mcp_requires_existing_owned_run(session, evidence, engine, run_id):
    from kognita._fleet import _run_checks
    from kognita.envelope import Envelope
    from kognita.tools import _save_run
    did, credential = _delegated(session, evidence)
    run = Run()
    if run_id == 'unbound':
        _save_run(session, run)
    elif run_id == 'foreign':
        _run_checks(session, run, Envelope(principal='mallory', agent_name='dossier-agent',
                    purpose='COLLABORATION', tool='read_note', delegation_id=did), evidence)
    upstream = _Upstream()
    response = _proxy(engine, evidence, upstream).handle('POST', '/mcp',
        dict(agent_name='dossier-agent', agent_credential=credential, delegation_id=did,
             run_id=run.id), _call(), session=session)
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    assert 'S6B-RUN-ISOLATION' in [c.citation for c in response.evaluation.failures()]
    assert upstream.calls == []


@pytest.mark.parametrize('label', ['external', 'client_data'])
@pytest.mark.parametrize('replacement', ['none', 'unknown', 'fresh'])
def test_mcp_run_rotation_preserves_taint(session, evidence, engine, label, replacement):
    from kognita._fleet import _run_checks
    from kognita._isolation import _ingest
    from kognita.envelope import Envelope
    from kognita.registry import get, approve_deployment
    did, credential = _delegated(session, evidence)
    agent = get(session, 'dossier-agent')
    permissions = {k: dict(v) if isinstance(v, dict) else v for k, v in agent.tool_allowlist.items()}
    # An external-tainted workload may read, but cannot take L2 authority.
    permissions['read_note'].update(public_content=label == 'client_data',
                                    required_level=2 if label == 'external' else 0)
    agent.tool_allowlist = permissions
    approve_deployment(session, agent.name, deployment=permissions['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    if label == 'external':
        from kognita.autonomy import _change
        _change(session, agent, 2, evidence, agent.owner_exec, 'AUTONOMY_PROMOTED', {})
    env = Envelope(principal='alice', agent_name=agent.name, purpose='COLLABORATION', tool='read_note', delegation_id=did)
    old = Run()
    _run_checks(session, old, env, evidence)
    _ingest(session, old, evidence, external=label == 'external', client_data=label == 'client_data',
            _agent_name=agent.name)
    fresh = Run()
    if replacement == 'fresh':
        _run_checks(session, fresh, env, evidence)
    headers = dict(agent_name=agent.name, agent_credential=credential, delegation_id=did)
    if replacement != 'none':
        headers['run_id'] = fresh.id
    upstream = _Upstream()
    response = _proxy(engine, evidence, upstream).handle('POST', '/mcp', headers, _call(), session=session)
    assert response.evaluation.outcome is Outcome.DENY
    assert upstream.calls == []
    assert any(c.citation in ('S5-RUN-REQUIRED', 'S6B-RUN-ISOLATION',
                              'S5-PUBLIC-CONTENT', 'S5-EXTERNAL-AUTHORITY')
               for c in response.evaluation.failures())


def test_broker_selective_fields_are_minimized(session, evidence):
    from kognita.registry import get, approve_deployment
    _, _, env, run = _family(session, evidence)
    agent = get(session, env.agent_name)
    agent.field_allowlist = {'READ': ['results.snippet']}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id='owner')
    answer = _ask(session, evidence, env, run, _index(session))
    assert answer.outcome is Outcome.ALLOW
    assert answer.results and 'confidential fragment' in answer.results[0].snippet
    assert answer.results[0].title == ''
    assert answer.results[0].source_label == ''
    assert answer.summary == [] and answer.citations == []


def test_existing_taint_is_backfilled_on_schema_upgrade(session, evidence, engine):
    from sqlalchemy import inspect, text
    from kognita.db import create_all
    from kognita.tools import _save_run
    _, _, env, run = _family(session, evidence)
    run.external = True
    _save_run(session, run)
    session.commit()
    # Emulate a 0.5 store that predates the new private provenance table.
    if inspect(engine).has_table('_agent_taint'):
        session.execute(text('DROP TABLE _agent_taint'))
        session.commit()
    create_all(engine)
    answer = _ask(session, evidence, env, None, _index(session))
    assert answer.outcome is Outcome.DENY
    assert 'S5-RUN-REQUIRED' in [c.citation for c in answer.evaluation.failures()]
