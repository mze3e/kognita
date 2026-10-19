"""Real system-trigger admission must survive live MCP polling."""
from dataclasses import replace
from datetime import timedelta
from threading import Event
from time import sleep

import pytest
from sqlmodel import Session

from conftest import NOW
from kognita._fleet import _run_checks
from kognita.authority import grant_to
from kognita.envelope import Envelope
from kognita.models import RunRecord
from kognita.tools import Run
from kognita.vocabulary import Outcome
from test_gateway import _live
from test_mcp import _call, _client, _proxy, _Upstream


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Leave production ingress unwrapped and supply the actual trigger identity."""
    yield


@pytest.mark.parametrize('change', ['none', 'remove-trigger', 'stop-run'])
def test_system_trigger_live_polling(session, evidence, engine, change):
    _live(session)
    grant_to('alice', 'read', timedelta(hours=2), {}, session=session,
             evidence=evidence, actor_id='admin', now=NOW)
    run = Run()
    _run_checks(session, run, Envelope(principal='alice', purpose='COLLABORATION',
        tool='read_note', context={'system_trigger': 'nightly-refresh'}), evidence)
    session.commit()
    changed, finished = Event(), Event()
    cancellations = []

    class _Delayed(_Upstream):
        def __call__(self, *args):
            try:
                sleep(0.06)  # Several production polls must succeed first.
                if change == 'remove-trigger':
                    proxy.client = replace(proxy.client, system_triggers=frozenset())
                elif change == 'stop-run':
                    with Session(engine) as control:
                        row = control.get(RunRecord, run.id)
                        row.control_state = 'paused'
                        control.add(row)
                        control.commit()
                changed.set()
                sleep(0.09)
                return super().__call__(*args)
            finally:
                finished.set()

        def _cancel(self, rpc_id):
            cancellations.append(changed.is_set())
            return False  # Backend cannot confirm cancellation; withhold its result.

    backend = _Delayed()
    proxy = _proxy(engine, evidence, backend,
        client=_client(system_triggers=frozenset({'nightly-refresh'})))
    try:
        response = proxy.handle('POST', '/mcp',
            {'system_trigger': 'nightly-refresh', 'run_id': run.id},
            _call(), session=session)
        assert response.evaluation is not None
        assert all(check.citation for check in response.evaluation.checks)
        assert any('S2-SYSTEM-TRIGGER' in check.citation
                   for check in response.evaluation.checks)
        if change == 'none':
            assert response.status == 200, response.body
            assert response.evaluation.outcome is Outcome.ALLOW
            assert b'rivera notes' in response.body
            assert cancellations == []
        else:
            assert response.status == 502, response.body
            assert b'cancellation_requested' in response.body
            assert b'rivera notes' not in response.body
            assert cancellations == [True], 'cancel only after the live identity/control changes'
    finally:
        assert finished.wait(2)
