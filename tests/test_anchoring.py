"""T6 independent checkpoints detect internally consistent rewrites."""
import copy
import json

import pytest
from sqlmodel import Session

from kognita.anchoring import FileAnchorSink, _anchor, _verify_anchors
from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.evidence import ChainBreak, EvidenceWriter, export_chain, verify_chain, verify_export
from kognita.vocabulary import EventType


def _store(payload=None):
    engine = make_engine()
    create_all(engine)
    with Session(engine) as session:
        EvidenceWriter(engine).emit(session, correlation_id='t6', event_type=EventType.TOOL_CALL,
                                    payload=payload)
        session.commit()
    return engine


def test_rewrite_after_anchor_detected(tmp_path):
    engine = _store({'_value': 'original'})
    path = tmp_path / 'checkpoint.json'
    with Session(engine) as session:
        _anchor(session, EvidenceWriter(engine), FileAnchorSink(path))
        session.commit()
        assert verify_chain(session) == 2
        _verify_anchors(export_chain(session), [path])
        assert export_chain(session)['events'][-1]['event_type'] == 'ANCHOR_PUBLISHED'
    # A replacement database has a self-consistent chain and head checkpoint.
    replacement = _store({'_value': 'rewritten'})
    with Session(replacement) as session:
        assert verify_chain(session) == 1
        artifact = export_chain(session)
        assert verify_export(artifact) == 1
        with pytest.raises(ChainBreak, match='T6'):
            _verify_anchors(artifact, [path])
    with pytest.raises(ValueError, match='T6'):
        FileAnchorSink(path)._publish({'sequence': 1, 'event_hash': '0' * 64})


def test_extension_and_truncation(tmp_path):
    engine = _store()
    path = tmp_path / 'head'
    with Session(engine) as session:
        writer = EvidenceWriter(engine)
        writer.emit(session, correlation_id='second', event_type=EventType.TOOL_CALL)
        session.commit()
        _anchor(session, writer, FileAnchorSink(path))
        session.commit()
        writer.emit(session, correlation_id='later', event_type=EventType.TOOL_CALL)
        session.commit()
        artifact = export_chain(session)
        assert verify_export(artifact) == 4
        _verify_anchors(artifact, [path])
        shortened = copy.deepcopy(artifact)
        shortened['events'] = shortened['events'][:1]
        shortened['event_count'] = 1
        shortened['head_hash'] = shortened['events'][0]['event_hash']
        assert verify_export(shortened) == 1
        with pytest.raises(ChainBreak, match='T6'):
            _verify_anchors(shortened, [path])


@pytest.mark.parametrize('contents', ['', '[]', '{}', '{', '{"sequence":true,"event_hash":"abc"}'])
def test_invalid_checkpoint_fails_closed(tmp_path, contents):
    path = tmp_path / 'bad'
    path.write_text(contents)
    with pytest.raises(ChainBreak, match='T6'):
        _verify_anchors({'events': []}, [path])
    with pytest.raises(ChainBreak, match='T6'):
        _verify_anchors({'events': []}, [tmp_path / 'missing'])


def test_publication_failure_rolls_back(tmp_path):
    engine = _store()
    with Session(engine) as session:
        with pytest.raises(ValueError, match='T6'):
            _anchor(session, EvidenceWriter(engine), FileAnchorSink(tmp_path / 'missing' / 'head'))
        session.rollback()
        assert verify_chain(session) == 1
    empty = make_engine()
    create_all(empty)
    with Session(empty) as session:
        with pytest.raises(ValueError, match='T6'):
            _anchor(session, EvidenceWriter(empty), FileAnchorSink(tmp_path / 'empty'))
    assert not (tmp_path / 'empty').exists()


def test_cli_anchor_and_verify(tmp_path):
    from kognita.signing import EvidenceSigner, _generate
    private, public = _generate()
    key = tmp_path / 'key'
    key.write_bytes(private)
    pub = tmp_path / 'public'
    pub.write_bytes(public)
    db = str(tmp_path / 'store.db')
    engine = make_engine(db)
    create_all(engine)
    with Session(engine) as session:
        EvidenceWriter(engine, signing=EvidenceSigner(private)).emit(
            session, correlation_id='signed', event_type=EventType.TOOL_CALL)
        session.commit()
    path = str(tmp_path / 'anchor')
    assert main(['evidence', 'anchor', path, '--db', db]) == 1
    assert not (tmp_path / 'anchor').exists()
    assert main(['evidence', 'anchor', path, str(key), '--db', db]) == 0
    assert main(['evidence', 'verify', '--db', db, '--anchors', path, '--public-key', str(pub)]) == 0
    artifact = tmp_path / 'export.json'
    with Session(engine) as session:
        artifact.write_text(json.dumps(export_chain(session)))
    assert main(['evidence', 'verify', '--file', str(artifact), '--anchors', path]) == 0
    assert main(['evidence', 'anchor', path, str(key), '--db', db]) == 1
    with Session(engine) as session:
        assert verify_chain(session, public_key=public) == 2


def test_legacy_chain_with_anchor(tmp_path):
    from test_schema_version import _legacy_store
    engine, artifact = _legacy_store()
    path = tmp_path / 'legacy'
    FileAnchorSink(path)._publish({'sequence': 2, 'event_hash': artifact['head_hash']})
    assert verify_export(artifact) == 2
    _verify_anchors(artifact, [path])
    create_all(engine)
    with Session(engine) as session:
        _anchor(session, EvidenceWriter(engine), FileAnchorSink(tmp_path / 'new'))
        session.commit()
        assert verify_chain(session) == 3
        _verify_anchors(export_chain(session), [path, tmp_path / 'new'])
