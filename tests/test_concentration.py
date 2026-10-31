"""R9 distinct critical use-case shares from local declarations."""
import json

import pytest

from kognita.dependencies import _concentration, _map
from kognita.usecases import register, load_register
from test_usecases import _open, _spec, NOW, CARD_ID


def _register(session, evidence, name, **fields):
    fields.setdefault('model_providers', ('house',))
    register(session, _spec(usecase_id=name, business_criticality='critical', **fields),
             evidence=evidence, actor_id='Jane Smith', at=NOW)


def test_concentration_shares_and_register_fields(session, evidence):
    _open(session, evidence)
    _register(session, evidence, 'first', cloud='cloud-a', framework='framework-a',
              vector_platform='vector-a', orchestration='orchestrator-a',
              model_providers=('house', 'second'), approved_models=(CARD_ID,))
    _register(session, evidence, 'second', cloud='cloud-b', framework='framework-a',
              vector_platform='vector-b', orchestration='orchestrator-a')
    register(session, _spec(usecase_id='low', cloud='excluded'), evidence=evidence, actor_id='Jane Smith', at=NOW)
    from kognita.models import Agent
    session.add(Agent(name='one', use_cases=['first']))
    session.add(Agent(name='two', use_cases=['first']))
    session.flush()
    result = _concentration(session)
    assert result['total_critical_use_cases'] == 2
    dimensions = result['dimensions']
    assert dimensions['provider']['dependencies'] == [
        {'dependency': 'house', 'use_cases': ['first', 'second'], 'count': 2, 'share': 1.0},
        {'dependency': 'second', 'use_cases': ['first'], 'count': 1, 'share': 0.5}]
    assert [row['share'] for row in dimensions['cloud']['dependencies']] == [0.5, 0.5]
    assert dimensions['framework']['dependencies'][0]['share'] == 1.0
    assert dimensions['vector_platform']['unknown_count'] == 0
    assert dimensions['orchestration']['dependencies'][0]['count'] == 2
    entry = next(e for e in load_register(session).entries if e.usecase_id == 'first')
    assert (entry.cloud, entry.framework, entry.vector_platform, entry.orchestration) == (
        'cloud-a', 'framework-a', 'vector-a', 'orchestrator-a')
    assert _concentration(session) == result
    assert any(e['cloud'] == 'cloud-a' for e in _map(session)['use_cases'])


def test_missing_metadata_is_unknown(session, evidence):
    _open(session, evidence)
    _register(session, evidence, 'missing')
    result = _concentration(session)
    assert result['dimensions']['provider']['dependencies'][0]['share'] == 1.0
    for name, dimension in result['dimensions'].items():
        if name == 'provider':
            continue
        assert dimension['dependencies'] == []
        assert dimension['unknown_use_cases'] == ['missing']
        assert dimension['unknown_share'] == 1.0


def test_empty_report(session):
    result = _concentration(session)
    assert result['total_critical_use_cases'] == 0
    assert all(d['dependencies'] == [] and d['unknown_share'] is None
               for d in result['dimensions'].values())


def test_cli_concentration_and_corrupt_evidence(tmp_path, capsys):
    from kognita.cli import main
    from kognita.db import create_all, make_engine, session_scope
    from kognita.evidence import EvidenceWriter
    from evidence_guard import tamper_for_test
    path = str(tmp_path / 'concentration.db')
    engine = make_engine(path)
    create_all(engine)
    with session_scope(engine) as session:
        evidence = EvidenceWriter(engine)
        _open(session, evidence)
        _register(session, evidence, 'critical')
    assert main(['deps', 'concentration', '--db', path]) == 0
    assert json.loads(capsys.readouterr().out)['total_critical_use_cases'] == 1
    with session_scope(engine) as session:
        tamper_for_test(session, 1, {'changed': True})
    assert main(['deps', 'concentration', '--db', path]) == 1
    assert 'R9' in capsys.readouterr().err
    engine.dispose()


def test_old_chain_unchanged():
    from sqlmodel import Session
    from test_schema_version import _legacy_store
    from kognita.db import create_all
    from kognita.evidence import export_chain, verify_chain
    engine, artifact = _legacy_store()
    create_all(engine)
    with Session(engine) as session:
        before = export_chain(session)
        _concentration(session)
        assert verify_chain(session) == 2
        after = export_chain(session)
        assert after['events'] == before['events']
        assert after['head_hash'] == before['head_hash']
        assert before['head_hash'] == artifact['head_hash']
    engine.dispose()


@pytest.mark.parametrize('name', ['cloud', 'framework', 'vector_platform', 'orchestration'])
def test_invalid_register_metadata_refused(session, evidence, name):
    from kognita.exceptions import KognitaError
    _open(session, evidence)
    with pytest.raises(KognitaError, match='R9'):
        _register(session, evidence, 'invalid', **{name: ['invalid']})
