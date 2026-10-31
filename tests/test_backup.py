"""R1 recovery drills, integrity refusals and legacy compatibility."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, select

from kognita.backup import _create, _restore, _verify
from kognita.canonical import canonical_hash
from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.evidence import EvidenceWriter, export_chain, verify_chain
from kognita.models import RetainedContent
from kognita.retention import RetentionStore
from kognita.signing import EvidenceSigner, _generate
from kognita.vocabulary import EventType


def _source(path=':memory:', signing=None):
    engine = make_engine(path)
    create_all(engine)
    at = datetime.now(timezone.utc) - timedelta(days=1)
    with Session(engine) as session:
        digest = RetentionStore().retain_text(session, 'recover me', kind='prompt', now=at)
        writer = EvidenceWriter(engine, signing=signing)
        for _ in range(3):
            writer.emit(session, correlation_id='r1', event_type=EventType.TOOL_CALL,
                                payload={'content_hash': digest})
        session.commit()
    return engine


def test_restore_drill(tmp_path):
    private, public = _generate()
    source = _source(signing=EvidenceSigner(private))
    with Session(source) as session:
        retention = RetentionStore()
        retention.set_policy(session, 'recover', retain_days=90)
        retention.retain_text(session, 'recover me', kind='prompt', use_case='recover')
        session.commit()
        before = export_chain(session)
        anchor = {'sequence': 1, 'event_hash': before['events'][0]['event_hash']}
        bundle = _create(session, _anchors=[anchor], _public_key=public)
    destination = tmp_path / 'restored.db'
    assert _restore(bundle, destination, _public_key=public) == 3
    engine = make_engine(destination)
    with Session(engine) as session:
        assert verify_chain(session, public_key=public) == 3
        assert export_chain(session)['events'] == before['events']
        assert session.exec(select(RetainedContent)).one().body == 'recover me'
        recovered = _create(session, _anchors=[anchor], _public_key=public)
        assert recovered['_retention'] == bundle['_retention']
        assert recovered['_registry'] == bundle['_registry']
        assert recovered['_anchors'] == bundle['_anchors']
    engine.dispose()


def test_point_in_time_and_sequence(tmp_path):
    source = _source()
    with Session(source) as session:
        bundle = _create(session)
    for name, boundary in [('sequence', {'_to_sequence': 2}), ('time', {'_to_time': bundle['_chain']['events'][1]['recorded_at']})]:
        path = tmp_path / name
        assert _restore(bundle, path, **boundary) == 2
        engine = make_engine(path)
        with Session(engine) as session:
            assert verify_chain(session) == 2
            assert session.exec(select(RetainedContent)).one().body == 'recover me'
            EvidenceWriter(engine).emit(session, correlation_id='continued', event_type=EventType.TOOL_CALL)
            session.commit()
            assert verify_chain(session) == 3
        engine.dispose()


@pytest.mark.parametrize('member', ['_chain', '_retention', '_registry', '_anchors'])
def test_tamper_refused_without_destination(tmp_path, member):
    source = _source()
    with Session(source) as session:
        bundle = _create(session)
    if member == '_chain':
        bundle[member]['events'][0]['payload'] = {}
    elif member == '_retention':
        bundle[member]['retained_content'][0]['body'] = 'changed'
    elif member == '_registry':
        bundle[member]['unknown'] = {}
    else:
        bundle[member] = [{'sequence': 1, 'event_hash': '0' * 64}]
    bundle['_hash'] = canonical_hash({k: v for k, v in bundle.items() if k != '_hash'})
    with pytest.raises(ValueError, match='R1'):
        _restore(bundle, tmp_path / 'refused.db')
    assert not (tmp_path / 'refused.db').exists()


def test_existing_destination_and_invalid_boundary_refused(tmp_path):
    source = _source()
    with Session(source) as session:
        bundle = _create(session)
    path = tmp_path / 'existing'
    path.write_text('preserve')
    with pytest.raises(ValueError, match='R1'):
        _restore(bundle, path)
    assert path.read_text() == 'preserve'
    for boundary in (-1, 4):
        with pytest.raises(ValueError, match='R1'):
            _restore(bundle, tmp_path / 'bad', _to_sequence=boundary)


def test_erased_history_refused(tmp_path):
    source = _source()
    with Session(source) as session:
        row = session.exec(select(RetainedContent)).one()
        RetentionStore().erase(session, row.content_hash, evidence=EvidenceWriter(source), actor_id='operator')
        session.commit()
        bundle = _create(session)
    with pytest.raises(ValueError, match='R1.*erased'):
        _restore(bundle, tmp_path / 'historical', _to_sequence=2)
    assert _restore(bundle, tmp_path / 'current') == 4


def test_cli_restore_drill(tmp_path):
    source = tmp_path / 'source.db'
    _source(source).dispose()
    bundle = tmp_path / 'bundle.json'
    restored = tmp_path / 'restored.db'
    assert main(['backup', 'create', str(bundle), '--db', str(source)]) == 0
    assert main(['backup', 'verify', str(bundle)]) == 0
    assert main(['backup', 'restore', str(bundle), '--db', str(restored), '--to-sequence', '2']) == 0
    assert main(['evidence', 'verify', '--db', str(restored)]) == 0
    assert main(['backup', 'restore', str(bundle), '--db', str(restored)]) == 1
    bundle.write_text('{}')
    assert main(['backup', 'verify', str(bundle)]) == 1


def test_legacy_chain_recovery(tmp_path):
    from test_schema_version import _legacy_store
    source, original = _legacy_store()
    create_all(source)
    with Session(source) as session:
        bundle = _create(session)
    destination = tmp_path / 'legacy.db'
    assert _restore(bundle, destination) == 2
    engine = make_engine(destination)
    with Session(engine) as session:
        assert verify_chain(session) == 2
        rows = export_chain(session)['events']
        assert [r['event_hash'] for r in rows] == [r['event_hash'] for r in original['events']]
        assert all(r['schema_version'] == 1 for r in rows)
    engine.dispose()


def test_empty_recovery(tmp_path):
    engine = make_engine()
    create_all(engine)
    with Session(engine) as session:
        bundle = _create(session)
    assert _restore(bundle, tmp_path / 'empty.db') == 0


def test_rotated_registry_recovery(tmp_path):
    from kognita.signing import _key_id, _public_pem
    initial, public = _generate()
    replacement, _ = _generate()
    source = _source(signing=EvidenceSigner(initial))
    with Session(source) as session:
        writer = EvidenceWriter(source, signing=EvidenceSigner(initial))
        writer.emit(session, correlation_id='rotation', event_type=EventType.KEY_ROTATED,
                    payload={'key_id': _key_id(EvidenceSigner(replacement)._load().public_key()),
                             '_public_key': _public_pem(EvidenceSigner(replacement)).decode('ascii'), '_escrow': 'bank custody'})
        session.commit()
        EvidenceWriter(source, signing=EvidenceSigner(replacement)).emit(
            session, correlation_id='after', event_type=EventType.TOOL_CALL)
        session.commit()
        bundle = _create(session, _public_key=public)
    assert any(v['_public_key'] is not None for v in bundle['_registry'].values())
    assert _restore(bundle, tmp_path / 'rotated.db', _public_key=public) == 5


def test_failed_restore_leaves_no_destination(tmp_path, monkeypatch):
    import kognita.backup as backup
    source = _source()
    with Session(source) as session:
        bundle = _create(session)
    def _fail(*args, **kwargs):
        raise RuntimeError('injected storage failure')
    monkeypatch.setattr(backup, '_verify_chain', _fail)
    with pytest.raises(RuntimeError, match='storage failure'):
        _restore(bundle, tmp_path / 'failed.db')
    assert list(tmp_path.iterdir()) == []


def test_checksum_and_independent_anchor_refused(tmp_path):
    source = _source()
    with Session(source) as session:
        bundle = _create(session)
    with pytest.raises(ValueError, match='R1.*anchor'):
        _restore(bundle, tmp_path / 'bad', _anchors=[{'sequence': 3, 'event_hash': '0' * 64}])
    bundle['_retention']['retained_content'] = []
    with pytest.raises(ValueError, match='R1.*checksum'):
        _verify(bundle)
    assert list(tmp_path.iterdir()) == []


def test_published_anchors_recover(tmp_path):
    from kognita.anchoring import FileAnchorSink, _anchor, _verify_anchors
    source = _source()
    checkpoint = tmp_path / 'checkpoint.json'
    with Session(source) as session:
        _anchor(session, EvidenceWriter(source), FileAnchorSink(checkpoint))
        session.commit()
        bundle = _create(session)
    assert bundle['_anchors'][0]['sequence'] == 3
    path = tmp_path / 'anchored.db'
    assert _restore(bundle, path) == 4
    engine = make_engine(path)
    with Session(engine) as session:
        _verify_anchors(export_chain(session), [checkpoint])
    engine.dispose()
