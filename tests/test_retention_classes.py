"""R11 class holds compose with use-case holds without changing old evidence."""
from datetime import timedelta
import json

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.exceptions import RetentionError
from kognita.models import RecordClassRetention, RetainedReference
from kognita.retention import RetentionStore, _set_class, _RECORD_CLASSES
from kognita.evidence import verify_chain


@pytest.mark.parametrize('kind', _RECORD_CLASSES)
@pytest.mark.parametrize('use_days,class_days', [(30, 10), (10, 30)])
def test_longest_period(session, evidence, kind, use_days, class_days):
    store = RetentionStore()
    store.set_policy(session, 'case', retain_days=use_days)
    _set_class(session, kind, class_days)
    digest = store.retain_text(session, 'body', kind='prompt', use_case='case',
                               record_classes=(kind,), now=NOW)
    with pytest.raises(RetentionError, match='R11-RECORD-LIVE'):
        store.erase(session, digest, evidence=evidence, actor_id='operator', now=NOW+timedelta(days=29))
    assert store.read(session, digest) == 'body'
    assert len(session.exec(select(RetainedReference)).all()) == 1
    assert store.enforce(session, evidence=evidence, now=NOW+timedelta(days=29)) == 0
    assert store.enforce(session, evidence=evidence, now=NOW+timedelta(days=30)) == 1
    assert store.read(session, digest) is None
    assert verify_chain(session) == 1


def test_multiple_classes_and_refreshed_shared_content(session, evidence):
    store = RetentionStore()
    for case in ('a', 'b'):
        store.set_policy(session, case, retain_days=0)
    _set_class(session, 'advice', 5)
    _set_class(session, 'transactions', 10)
    digest = store.retain_text(session, 'shared', kind='advice', use_case='a', now=NOW)
    assert store.retain_text(session, 'shared', kind='transactions', use_case='b',
                             now=NOW+timedelta(days=4)) == digest
    assert store.enforce(session, evidence=evidence, now=NOW+timedelta(days=10)) == 0
    store.erase(session, digest, evidence=evidence, actor_id='operator', now=NOW+timedelta(days=14))
    assert verify_chain(session) == 1


@pytest.mark.parametrize('missing', ['class', 'usecase', 'indefinite'])
def test_missing_or_indefinite_policy_refuses(session, evidence, missing):
    store = RetentionStore()
    if missing != 'usecase':
        store.set_policy(session, 'case', retain_days=0)
    if missing != 'class':
        _set_class(session, 'advice', None if missing == 'indefinite' else 0)
    digest = store.retain_value(session, {'text': 'record'}, kind='prompt', use_case='case',
                                record_classes=('advice',), now=NOW)
    with pytest.raises(RetentionError, match='R11-RECORD-LIVE'):
        store.erase(session, digest, evidence=evidence, actor_id='operator', now=NOW+timedelta(days=365))
    assert verify_chain(session) == 0


def test_invalid_configuration_refuses(session):
    for kind, days in [('unknown', 10), ('advice', -1), ('advice', 1.5)]:
        with pytest.raises(RetentionError, match='R11-RECORD-LIVE'):
            _set_class(session, kind, days)
    with pytest.raises(RetentionError, match='R11-RECORD-LIVE'):
        RetentionStore().retain_text(session, 'x', kind='prompt', record_classes=('unknown',))
    assert session.exec(select(RecordClassRetention)).all() == []


def test_cli_set_list(tmp_path, capsys):
    from kognita.cli import main
    db = str(tmp_path / 'retention.db')
    assert main(['retention', 'class', 'set', 'advice', '30', '--db', db]) == 0
    assert json.loads(capsys.readouterr().out) == [{'kind': 'advice', 'retain_days': 30}]
    assert main(['retention', 'class', 'list', '--db', db]) == 0
    assert json.loads(capsys.readouterr().out)[0]['retain_days'] == 30
    assert main(['retention', 'class', 'set', 'advice', 'none', '--db', db]) == 0
    capsys.readouterr()
    assert main(['retention', 'class', 'set', 'unknown', '10', '--db', db]) == 1
    assert 'R11-RECORD-LIVE' in capsys.readouterr().err


def test_old_chain_unchanged():
    from sqlmodel import Session
    from test_schema_version import _legacy_store
    from kognita.db import create_all
    from kognita.evidence import export_chain
    engine, artifact = _legacy_store()
    create_all(engine)
    with Session(engine) as session:
        _set_class(session, 'advice', 30)
        assert verify_chain(session) == 2
        assert export_chain(session)['head_hash'] == artifact['head_hash']
    engine.dispose()


def test_backup_preserves_class_holds(session, engine, evidence, tmp_path):
    from kognita.backup import _create, _restore
    from kognita.db import make_engine, session_scope
    store = RetentionStore()
    store.set_policy(session, 'case', retain_days=0)
    _set_class(session, 'advice', 30)
    digest = store.retain_text(session, 'record', kind='prompt', use_case='case',
                               record_classes=('advice',), now=NOW)
    session.commit()
    bundle = _create(session)
    path = str(tmp_path / 'restored.db')
    _restore(bundle, path)
    restored = make_engine(path)
    with session_scope(restored) as target:
        with pytest.raises(RetentionError, match='R11-RECORD-LIVE'):
            store.erase(target, digest, evidence=evidence, actor_id='operator', now=NOW)
    restored.dispose()
