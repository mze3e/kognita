"""T1: independent signature verification and fail-closed writes."""
import copy
import builtins
import os

import pytest
from sqlmodel import Session, select

from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.evidence import ChainBreak, EvidenceWriter, export_chain, verify_chain, verify_export
from kognita.models import EvidenceEvent
from kognita.signing import EvidenceSigner, _generate
from kognita.vocabulary import EventType


def _chain(path=':memory:'):
    private, public = _generate()
    engine = make_engine(path)
    create_all(engine)
    with Session(engine) as session:
        writer = EvidenceWriter(engine, signing=EvidenceSigner(private))
        for _ in range(2):
            writer.emit(session, correlation_id='signed', event_type=EventType.TOOL_CALL)
        session.commit()
        assert verify_chain(session, public_key=public) == 2
        artifact = export_chain(session)
    return engine, artifact, public


def test_tamper():
    _, artifact, public = _chain()
    assert verify_export(artifact, public_key=public) == 2
    for field, value in [('payload', {'changed': True}), ('signature', '00' * 64), ('key_id', 'wrong')]:
        changed = copy.deepcopy(artifact)
        changed['events'][0][field] = value
        with pytest.raises(ChainBreak):
            verify_export(changed, public_key=public)
    # Even a self-consistent rewritten payload and chain cannot forge a signature.
    from kognita.canonical import canonical_hash
    changed = copy.deepcopy(artifact)
    row = changed['events'][0]
    row['payload'] = {'changed': True}
    row['payload_hash'] = canonical_hash(row['payload'])
    with pytest.raises(ChainBreak, match='T1'):
        verify_export(changed, public_key=public)


def test_wrong_key():
    engine, artifact, _ = _chain()
    _, wrong = _generate()
    with Session(engine) as session, pytest.raises(ChainBreak, match='T1'):
        verify_chain(session, public_key=wrong)
    with pytest.raises(ChainBreak, match='T1'):
        verify_export(artifact, public_key=wrong)
    with pytest.raises(ChainBreak, match='T1'):
        verify_export(artifact, public_key=b'invalid PEM')


def test_unsigned_event_in_signed_chain_is_a_break():
    engine, artifact, public = _chain()
    from kognita.canonical import canonical_hash
    row = artifact['events'][1]
    row['signature'] = row['key_id'] = None
    header = {key: row[key] for key in ('sequence', 'schema_version', 'correlation_id', 'event_type', 'actor_type', 'actor_id', 'classification', 'payload_hash', 'prev_hash', 'recorded_at')}
    row['event_hash'] = artifact['head_hash'] = canonical_hash(header)
    for key in (None, public):
        with pytest.raises(ChainBreak, match='T1: unsigned'):
            verify_export(artifact, public_key=key)
    with Session(engine) as session, pytest.raises(ValueError, match='T1'):
        EvidenceWriter(engine).emit(session, correlation_id='unsigned', event_type=EventType.TOOL_CALL)


def test_missing_extra_refuses_writes(monkeypatch):
    private, _ = _generate()
    engine = make_engine()
    create_all(engine)
    original = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith('cryptography'):
            raise ImportError('blocked')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _blocked)
    with Session(engine) as session:
        with pytest.raises(ValueError, match='T1: signing requires'):
            EvidenceWriter(engine, signing=EvidenceSigner(private)).emit(
                session, correlation_id='blocked', event_type=EventType.TOOL_CALL)
        assert not session.exec(select(EvidenceEvent)).all()


def test_cli_generate_and_verify(tmp_path):
    private, public = tmp_path / 'private.pem', tmp_path / 'public.pem'
    assert main(['keys', 'generate', str(private), str(public)]) == 0
    assert os.stat(private).st_mode & 0o777 == 0o600
    saved = private.read_bytes()
    assert main(['keys', 'generate', str(private), str(public)]) == 1
    assert private.read_bytes() == saved
    db = tmp_path / 'evidence.db'
    engine = make_engine(db)
    create_all(engine)
    with Session(engine) as session:
        EvidenceWriter(engine, signing=EvidenceSigner(saved)).emit(
            session, correlation_id='cli', event_type=EventType.TOOL_CALL)
        session.commit()
        artifact = export_chain(session)
    import json
    artifact_path = tmp_path / 'export.json'
    artifact_path.write_text(json.dumps(artifact))
    for args in (['--db', str(db)], ['--file', str(artifact_path)]):
        assert main(['evidence', 'verify', *args, '--public-key', str(public)]) == 0
        public.write_bytes(b'invalid')
        assert main(['evidence', 'verify', *args, '--public-key', str(public)]) == 1
        # Restore the original matching public key for the next path.
        from kognita.signing import _crypto
        serialization, _, _ = _crypto()
        key = serialization.load_pem_private_key(saved, password=None).public_key()
        public.write_bytes(key.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))


def test_signing_columns_upgrade_preserves_hashes():
    from kognita.migrations import _migrate
    engine = make_engine()
    create_all(engine)
    with Session(engine) as session:
        EvidenceWriter(engine).emit(session, correlation_id='old-v2', event_type=EventType.TOOL_CALL)
        session.commit()
        original = export_chain(session)['head_hash']
    with engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE evidence_events DROP COLUMN signature')
        connection.exec_driver_sql('ALTER TABLE evidence_events DROP COLUMN key_id')
    assert _migrate(engine) == 2
    with Session(engine) as session:
        assert verify_chain(session) == 1
        assert export_chain(session)['head_hash'] == original


def test_signing_import_is_dependency_light():
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c',
                    "import sys; import kognita.signing; import kognita.evidence; "
                    "assert not any(m.split('.')[0] in {'cryptography', 'psycopg'} for m in sys.modules)"],
                   check=True)


def test_unsigned_prefix_and_malformed_metadata_refused():
    _, artifact, public = _chain()
    for field, value in [('signature', None), ('key_id', None), ('signature', 42)]:
        changed = copy.deepcopy(artifact)
        changed['events'][0][field] = value
        with pytest.raises(ChainBreak, match='T1'):
            verify_export(changed, public_key=public)
    changed = copy.deepcopy(artifact)
    changed['events'][0]['signature'] = changed['events'][0]['key_id'] = None
    with pytest.raises(ChainBreak, match='T1: unsigned'):
        verify_export(changed)


def test_invalid_private_key_refuses_write():
    engine = make_engine()
    create_all(engine)
    with Session(engine) as session:
        with pytest.raises(ValueError, match='T1: invalid'):
            EvidenceWriter(engine, signing=EvidenceSigner(b'invalid')).emit(
                session, correlation_id='invalid', event_type=EventType.TOOL_CALL)
        assert not session.exec(select(EvidenceEvent)).all()


def test_signing_refuses_unsigned_prefix_write():
    private, _ = _generate()
    engine = make_engine()
    create_all(engine)
    with Session(engine) as session:
        EvidenceWriter(engine).emit(session, correlation_id='legacy', event_type=EventType.TOOL_CALL)
        session.commit()
        with pytest.raises(ValueError, match='T1: signing requires'):
            EvidenceWriter(engine, signing=EvidenceSigner(private)).emit(
                session, correlation_id='signed', event_type=EventType.TOOL_CALL)
        assert verify_chain(session) == 1
