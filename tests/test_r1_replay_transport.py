"""Real authority regressions, outside the legacy fixture allowlist."""
from datetime import timedelta
from threading import Event

from conftest import NOW
from kognita.authority import Delegation, delegate, grant_to
from kognita.credentials import issue
from kognita.vocabulary import Outcome


def _authority(session, evidence, verb='read'):
    from test_mcp import _register
    _register(session)
    scope = {'client': ['123'], 'model': ['gateway-model']}
    grant_to('alice', verb, timedelta(hours=2), scope, session=session,
             evidence=evidence, actor_id='admin', now=NOW)
    did = delegate(session, Delegation(human='alice', agent='dossier-agent',
        purpose='COLLABORATION', client='123', allowed_actions=[verb], allowed_data=scope,
        valid_from=NOW, valid_until=NOW + timedelta(hours=1), jurisdiction='SG', task_ref='meeting'),
        evidence=evidence, actor_id='alice', now=NOW)
    return {'agent_name': 'dossier-agent', 'delegation_id': did,
            'agent_credential': issue(session, 'dossier-agent', evidence=evidence, actor_id='Head of Research Ops')}


def test_successful_gateway_replays_with_admission_checks(session, evidence, engine):
    from test_gateway import _live, _gateway, _Upstream, _body, PATH
    from kognita.replay import replay_decision
    from kognita.usecases import load_register
    _live(session)
    headers = _authority(session, evidence)
    response = _gateway(engine, evidence, _Upstream(b'{"model":"gateway-model","choices":[]}')).handle(
        'POST', PATH, headers, _body('hello'), session=session)
    original = response.evaluation
    assert response.status == 200
    assert original.outcome is Outcome.ALLOW
    replayed = replay_decision(session, original.request_id, original.envelope,
        purposes=['COLLABORATION'], usecases=load_register(session, as_of=NOW))
    assert replayed.checks == original.checks
    assert replayed.envelope_hash == original.envelope_hash
    assert replayed.outcome is original.outcome


def test_default_http_mutation_requires_compensation(session, evidence, engine):
    from kognita.models import Agent
    from sqlmodel import select
    from test_mcp import _proxy, _call
    from kognita.mcp import _urllib_transport
    headers = _authority(session, evidence, 'send')
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    agent.tool_allowlist = {**agent.tool_allowlist, 'read_note': {
        **agent.tool_allowlist['read_note'], 'verb': 'send'}, 'notes/read_note': {
        **agent.tool_allowlist['notes/read_note'], 'verb': 'send'}}
    session.add(agent)
    from kognita.registry import approve_deployment
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
        evidence=evidence, actor_id=agent.owner_exec)
    session.commit()
    response = _proxy(engine, evidence, _urllib_transport).handle(
        'POST', '/mcp', headers, _call(), session=session)
    assert response.evaluation.outcome is Outcome.DENY
    assert 'S6C-UNCANCELLABLE' in [c.citation for c in response.evaluation.failures()]


def test_http_local_abort_is_not_confirmed_cancellation():
    from kognita.mcp import _HttpCall
    from kognita._actions import _bounded
    call = _HttpCall()
    release = Event()
    try:
        outcome, _ = _bounded(lambda: release.wait(2), seconds=1,
            agent='agent', run=None, delegation=None, _check=lambda: False,
            cancel=lambda: call._cancel(1))
        assert outcome == 'connection_aborted'
        assert call._stopped
    finally:
        release.set()
