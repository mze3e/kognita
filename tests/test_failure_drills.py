"""R10 offline faults run through real governance paths."""
import pytest
from sqlmodel import select
from kognita.breaker import BreakerError, DrillKind, _simulate
from kognita.evidence import ChainBreak, verify_chain
from kognita.models import EvidenceEvent, UseCase
from kognita.vocabulary import FailureMode
from test_usecases import _open, _activate, _spec, NOW
from kognita.usecases import model_card_id, register_model_card


def _setup(session, evidence, mode, fallback=False):
    from kognita.providers import _register
    from test_providers import _details
    _register(session, 'house', _details())
    _open(session, evidence)
    cards = [model_card_id('house', 'summary', '1')]
    if fallback:
        register_model_card(session, provider='house', model='backup', version='2',
            intended_use='Fallback', limitations='Test', evaluation_results='Test',
            approver='Luis Ortega', approved_at=NOW, evidence=evidence,
            actor_id='Luis Ortega', at=NOW)
        cards.append(model_card_id('house', 'backup', '2'))
    _activate(session, evidence, _spec(approved_models=tuple(cards),
        failure_mode=mode.value, failover=cards[-1] if fallback else ''))
    session.commit()


@pytest.mark.parametrize('kind', list(DrillKind), ids=lambda kind: kind.value)
@pytest.mark.parametrize('mode', list(FailureMode))
def test_failure_paths(session, evidence, kind, mode):
    fallback = mode == FailureMode.DEGRADED
    _setup(session, evidence, mode, fallback)
    before = [row.model_dump() for row in session.exec(select(UseCase)).all()]
    count = verify_chain(session)
    result = _simulate(session, kind, usecase_id='research-summary', evidence=evidence)
    assert result['failure_mode'] == mode.value
    model_fault = kind in {DrillKind.MODEL_UNAVAILABLE, DrillKind.VENDOR_OUTAGE,
        DrillKind.API_OUTAGE, DrillKind.COMPROMISED_MODEL}
    assert result['released'] == (fallback and model_fault)
    assert result['transcript_complete']
    assert result['chain_events'] == count + 1
    assert [row.model_dump() for row in session.exec(select(UseCase)).all()] == before
    assert verify_chain(session) == count + 1


def test_unknown_kind_and_usecase(session, evidence):
    for kind, usecase in [('unknown', 'research-summary'), ('tool_failure', 'unknown')]:
        with pytest.raises(BreakerError, match='R10'):
            _simulate(session, kind, usecase_id=usecase, evidence=evidence)
    assert verify_chain(session) == 0


def test_corrupt_chain_refuses(session, evidence):
    from evidence_guard import tamper_for_test
    from kognita.vocabulary import EventType
    event = evidence.emit(session, correlation_id='old', event_type=EventType.REVIEW)
    session.commit()
    tamper_for_test(session, event.sequence, {'tampered': True})
    with pytest.raises(ChainBreak):
        _simulate(session, 'tool_failure', usecase_id='research-summary', evidence=evidence)
