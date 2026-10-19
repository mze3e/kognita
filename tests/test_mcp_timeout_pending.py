"""Timeout recovery uses real workload authority and a durable action barrier."""
import json
from threading import Event, Thread

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.envelope import Envelope
from kognita.evidence import verify_chain
from kognita.interaction import attach, open_interaction
from kognita.intervention import recover_run
from kognita.models import Agent, EvidenceEvent, _PendingAction
from kognita.registry import approve_deployment
from kognita.tools import Run
from test_actions_s6c import _manifest
from test_mcp import _call, _proxy
from test_mcp_client_argument_scope import _delegated


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    yield


@pytest.mark.parametrize('stop', ['timeout', 'unconfirmed', 'aborted'])
@pytest.mark.parametrize('fails', [False, True])
def test_uncertain_backend_blocks_recovery_until_late_completion(session, evidence, stop, fails):
    did, secret = _delegated(session, evidence)
    _manifest(session, evidence, reversible=True, compensation='undo_note',
              max_duration=0.1 if stop == 'timeout' else 3)
    agent = session.exec(select(Agent).where(Agent.name == 'dossier-agent')).one()
    agent.field_allowlist = {'COLLABORATION': ['marked']}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    from kognita.review import define_human_boundary
    define_human_boundary(session, 'COLLABORATION', intervention='before output',
        who='alice', must_review='facts', evidence_seen='references', on_reject='stop',
        minimum_review_seconds=0, evidence=evidence, now=NOW)
    run = Run(started_at=NOW)
    from kognita._fleet import _run_checks
    _run_checks(session, run, Envelope(principal='alice', agent_name=agent.name,
        purpose='COLLABORATION', tool='read_note', delegation_id=did), evidence)
    interaction = open_interaction(session, client_id='123', use_case='COLLABORATION',
                                   evidence=evidence, now=NOW)
    attach(session, interaction.id, run_id=run.id, evidence=evidence)
    session.commit()
    entered, release, finished = Event(), Event(), Event()
    calls = []

    class Transport:
        def __call__(self, *args):
            payload = json.loads(args[-1])
            if payload['params']['name'] == 'undo_note':
                calls.append('undo')
                return 200, {}, b'{"result":{}}'
            entered.set()
            try:
                assert release.wait(5)
                calls.append('effect')
                if fails:
                    raise RuntimeError('failure after effect')
                return 200, {}, b'{"result":{"marked":"late"}}'
            finally:
                finished.set()

        def _cancel(self, _):
            return 'connection_aborted' if stop == 'aborted' else False

    def signal():
        assert entered.wait(2)
        from kognita._actions import _stop
        _stop(run=run.id)

    thread = Thread(target=signal) if stop != 'timeout' else None
    if thread:
        thread.start()
    proxy = _proxy(session.get_bind(), evidence, Transport())
    try:
        response = proxy.handle('POST', '/mcp', {'agent_name': agent.name,
            'agent_credential': secret, 'delegation_id': did, 'run_id': run.id},
            _call(), session=session)
        assert entered.is_set(), response.body
        assert json.loads(response.body)['error']['message'] == {
            'timeout': 'timeout', 'unconfirmed': 'cancellation_requested',
            'aborted': 'connection_aborted'}[stop]
        result = recover_run(session, run.id, evidence=evidence, human='alice',
                             citation='S6C-RECOVER', now=NOW)
        assert result['control_state'] == 'paused'
        assert result['_pending_actions']
        assert calls == []
        assert session.exec(select(_PendingAction)).first() is not None
    finally:
        release.set()
        assert finished.wait(2)
        if thread:
            thread.join(2)
    result = recover_run(session, run.id, evidence=evidence, human='alice',
                         citation='S6C-RECOVER', now=NOW)
    session.commit()
    assert result['control_state'] == 'recovered'
    assert calls == ['effect', 'undo']
    assert session.exec(select(_PendingAction)).first() is None
    actions = [e.payload.get('action') for e in session.exec(select(EvidenceEvent)).all()]
    assert '_action_pending' in actions
    assert '_action_uncertain' in actions
    assert '_action_finished' in actions
    assert 'compensation' in actions
    assert verify_chain(session) > 0
