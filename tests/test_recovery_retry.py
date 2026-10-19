"""Recovery retries use real governed actions and durable compensation evidence."""
import pytest
from sqlmodel import select

from conftest import NOW
from kognita._actions import _compensations
from kognita.envelope import Envelope
from kognita.evidence import verify_chain
from kognita.interaction import attach, open_interaction
from kognita.intervention import control_checks, recover_run
from kognita.models import EvidenceEvent, RunRecord
from kognita.review import define_human_boundary
from kognita.tools import Run, ToolRegistry, run_governed
from kognita.vocabulary import CheckResult, Classification, Outcome
from test_isolation_s5 import _Pack
from test_usecases import _live_purpose


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Disable the ingress shim; these are direct human library calls."""


@pytest.mark.parametrize('unavailable', [False, True], ids=['failed', 'unavailable'])
def test_recovery_retries_only_unfinished_compensations(session, evidence, unavailable):
    # Human invocation needs no agent credential or delegated workload authority.
    _live_purpose(session, evidence, 'READ')
    define_human_boundary(session, 'READ', intervention='before output', who='human',
        must_review='facts', evidence_seen='references', on_reject='stop',
        minimum_review_seconds=0, evidence=evidence, now=NOW)
    run = Run()
    interaction = open_interaction(session, client_id='123', use_case='READ',
                                   evidence=evidence, now=NOW)
    registry = ToolRegistry()
    calls = []
    broken = True

    def undo(name):
        def callback(*_):
            calls.append(name)
            if name == 'second' and broken:
                raise RuntimeError('temporary backend outage')
        return callback

    for name in ('first', 'second', 'third'):
        registry.register(name, lambda *_: 'ok', reversible=True, compensation=undo(name))
        result = run_governed(session, Envelope(principal='human', purpose='READ', tool=name),
            registry=registry, run=run, evidence=evidence, pack=_Pack(), purposes=['READ'],
            as_of=NOW, now=NOW)
        assert result.outcome is Outcome.ALLOW
    attach(session, interaction.id, run_id=run.id, evidence=evidence)
    session.commit()
    if unavailable:
        # Simulate losing one callback at restart; restore from retained data on retry.
        callbacks = _compensations[session.get_bind()]
        key = next(key for key, value in callbacks.items()
                   if value[0] is registry.get('second').compensation)
        callbacks.pop(key)

    kwargs = dict(evidence=evidence, human='human', citation='S6C-RECOVER', now=NOW)
    first = recover_run(session, run.id, **kwargs)
    session.commit()
    assert [s['name'] for s in first['steps']] == ['third', 'second', 'first']
    assert [s['outcome'] for s in first['steps']] == [
        'completed', 'unavailable' if unavailable else 'failed', 'completed']
    assert first['control_state'] == 'paused'
    assert session.get(RunRecord, run.id).control_state == 'paused'
    assert first['checks'][0]['result'] == CheckResult.FAIL.value
    assert first['checks'][0]['citation'] == 'S6C-RECOVER'
    checks = control_checks(session, session.get(RunRecord, run.id),
                            tool='first', classification=Classification.C0)
    assert checks and checks[0].result is CheckResult.FAIL and checks[0].citation
    assert calls == (['third', 'first'] if unavailable else ['third', 'second', 'first'])

    broken = False
    second = recover_run(session, run.id, _registry=registry, **kwargs)
    session.commit()
    assert [s['outcome'] for s in second['steps']] == [
        'already_completed', 'completed', 'already_completed']
    assert calls == (['third', 'first', 'second'] if unavailable
                     else ['third', 'second', 'first', 'second'])
    assert second['control_state'] == 'recovered'
    assert session.get(RunRecord, run.id).control_state == 'recovered'
    assert second['checks'][0]['result'] == CheckResult.PASS.value
    events = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.correlation_id == interaction.id)).all()
    completed = [e.payload['name'] for e in events
                 if e.payload.get('action') == 'compensation'
                 and e.payload.get('outcome') == 'completed']
    assert completed == ['third', 'first', 'second']
    assert verify_chain(session) > 0
