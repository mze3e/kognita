"""Recovery must account for actions whose bodies have already been released."""
import pytest
from sqlmodel import Session, select

from conftest import NOW
from kognita.db import create_all, make_engine
from kognita.evidence import verify_chain
from kognita.envelope import Envelope
from kognita.interaction import attach, open_interaction
from kognita.intervention import recover_run
from kognita.models import CompensatingAction, EvidenceEvent, RunRecord
from kognita.review import define_human_boundary
from kognita.tools import Run, ToolRegistry, run_governed
from kognita.vocabulary import Outcome
from test_isolation_s5 import _Pack
from test_usecases import _live_purpose


@pytest.fixture
def engine(tmp_path):
    engine = make_engine(tmp_path / 'recovery.db')
    create_all(engine)
    yield engine
    engine.dispose()


@pytest.mark.parametrize('fails', [False, True])
def test_recovery_stays_incomplete_until_released_action_finishes(session, evidence, fails):
    # Human library invocation: no delegated identity or synthetic authority.
    _live_purpose(session, evidence, 'READ')
    define_human_boundary(session, 'READ', intervention='before output', who='human',
        must_review='facts', evidence_seen='references', on_reject='stop',
        minimum_review_seconds=0, evidence=evidence, now=NOW)
    run = Run()
    from kognita.tools import _save_run
    _save_run(session, run)
    interaction = open_interaction(session, client_id='123', use_case='READ',
                                   evidence=evidence, now=NOW)
    attach(session, interaction.id, run_id=run.id, evidence=evidence)
    session.commit()
    calls = []
    registry = ToolRegistry()

    def body(*_):
        # A fresh session must see the declared compensation before side effects.
        with Session(session.get_bind()) as observer:
            assert observer.exec(select(CompensatingAction).where(
                CompensatingAction.run_id == run.id)).first() is not None
            result = recover_run(observer, run.id, evidence=evidence, human='human',
                                 citation='S6C-RECOVER', now=NOW, _registry=registry)
            observer.commit()
            assert result['control_state'] == 'paused'
            assert result['_pending_actions']
            assert calls == []
        calls.append('effect')
        if fails:
            raise RuntimeError('failure after side effect')
        return 'ok'

    registry.register('tool', body, reversible=True,
                      compensation=lambda *_: calls.append('undo'))
    kwargs = dict(registry=registry, run=run, evidence=evidence, pack=_Pack(),
                  purposes=['READ'], as_of=NOW, now=NOW)
    if fails:
        with pytest.raises(RuntimeError, match='failure after side effect'):
            run_governed(session, Envelope(principal='human', purpose='READ', tool='tool'), **kwargs)
    else:
        assert run_governed(session, Envelope(principal='human', purpose='READ', tool='tool'), **kwargs).outcome is Outcome.ALLOW
    session.commit()
    with Session(session.get_bind()) as recovery:
        result = recover_run(recovery, run.id, evidence=evidence, human='human',
                             citation='S6C-RECOVER', now=NOW, _registry=registry)
        recovery.commit()
        assert result['control_state'] == 'recovered'
        assert calls == ['effect', 'undo']
        events = recovery.exec(select(EvidenceEvent)).all()
        actions = [e.payload.get('action') for e in events]
        assert '_action_pending' in actions
        assert '_action_finished' in actions
        assert 'compensation' in actions
        assert recovery.get(RunRecord, run.id).control_state == 'recovered'
        assert verify_chain(recovery) > 0
