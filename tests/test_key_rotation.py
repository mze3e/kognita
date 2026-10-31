"""R2 authenticated lifecycle replay and local CLI operations."""
import copy
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session

from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.evidence import ChainBreak, EvidenceWriter, export_chain, verify_chain, verify_export
from kognita.signing import EvidenceSigner, _generate, _key_id, _public_pem, _registry
from kognita.vocabulary import EventType


def test_verify_across_rotations_and_escrow():
    engine = make_engine()
    create_all(engine)
    first, root = _generate()
    second, _ = _generate()
    third, _ = _generate()
    signer = EvidenceSigner(first)
    with Session(engine) as session:
        for replacement in (second, third):
            writer = EvidenceWriter(engine, signing=signer)
            writer.emit(session, correlation_id='r2', event_type=EventType.TOOL_CALL)
            new = EvidenceSigner(replacement)
            writer.emit(session, correlation_id='r2', event_type=EventType.KEY_ROTATED,
                        payload={'key_id': _key_id(new._load().public_key()),
                                 '_public_key': _public_pem(new).decode(),
                                 '_escrow': 'Bank custody, vault reference 123'})
            signer = new
        EvidenceWriter(engine, signing=signer).emit(session, correlation_id='r2', event_type=EventType.TOOL_CALL)
        session.commit()
        assert verify_chain(session, public_key=root) == 5
        artifact = export_chain(session)
        assert verify_export(artifact, public_key=root) == 5
        assert len(_registry(artifact['events'], root)) == 3
        changed = copy.deepcopy(artifact)
        changed['events'][1]['payload']['_public_key'] = _generate()[1].decode()
        with pytest.raises(ChainBreak):
            verify_export(changed, public_key=root)
        with pytest.raises(ValueError, match='R2'):
            EvidenceWriter(engine, signing=EvidenceSigner(first)).emit(
                session, correlation_id='old', event_type=EventType.TOOL_CALL)


def test_revocation_time_and_revoked_writes():
    private, public = _generate()
    signer = EvidenceSigner(private)
    engine = make_engine()
    create_all(engine)
    now = datetime.now(timezone.utc)
    key_id = _key_id(signer._load().public_key())
    with Session(engine) as session:
        writer = EvidenceWriter(engine, signing=signer, clock=lambda: now)
        writer.emit(session, correlation_id='r2', event_type=EventType.TOOL_CALL)
        for cutoff in (now, now - timedelta(seconds=1)):
            with pytest.raises(ValueError, match='R2'):
                writer.emit(session, correlation_id='r2', event_type=EventType.KEY_REVOKED,
                            payload={'key_id': key_id, 'recorded_at': cutoff.isoformat()})
        writer.emit(session, correlation_id='r2', event_type=EventType.KEY_REVOKED,
                    payload={'key_id': key_id, 'recorded_at': (now + timedelta(seconds=1)).isoformat()})
        session.commit()
        assert verify_chain(session, public_key=public) == 2
        # A correctly signed revocation still invalidates signatures after its cutoff.
        from kognita.canonical import canonical_hash
        artifact = export_chain(session)
        row = artifact['events'][-1]
        row['payload']['recorded_at'] = now.isoformat()
        row['payload_hash'] = canonical_hash(row['payload'])
        header = {k: row[k] for k in ('schema_version', 'sequence', 'correlation_id',
                  'event_type', 'actor_type', 'actor_id', 'classification', 'payload_hash',
                  'prev_hash', 'recorded_at')}
        row['signature'] = signer._sign(header)
        header['signature'] = row['signature']
        row['event_hash'] = artifact['head_hash'] = canonical_hash(header)
        with pytest.raises(ChainBreak, match='R2'):
            verify_export(artifact, public_key=public)
        with pytest.raises(ValueError, match='R2'):
            writer.emit(session, correlation_id='r2', event_type=EventType.TOOL_CALL)


def test_cli_rotate_revoke_list(tmp_path, capsys):
    db = str(tmp_path / 'chain.db')
    paths = [tmp_path / name for name in ('old', 'old.pub', 'new', 'new.pub')]
    for private, public in (paths[:2], paths[2:]):
        assert main(['keys', 'generate', str(private), str(public)]) == 0
    assert main(['keys', 'rotate', str(paths[0]), str(paths[2]), 'Bank vault 123', '--db', db]) == 0
    assert main(['keys', 'list', '--db', db]) == 0
    assert 'Bank vault 123' in capsys.readouterr().out
    old_id = _key_id(EvidenceSigner(paths[0].read_bytes())._load().public_key())
    assert main(['keys', 'revoke', str(paths[2]), old_id,
                 datetime.now(timezone.utc).isoformat(), '--db', db]) == 0
    assert main(['evidence', 'verify', '--db', db, '--public-key', str(paths[1])]) == 0
    assert main(['keys', 'rotate', str(paths[0]), str(paths[2]), 'vault', '--db', db]) == 1


def test_unsigned_lifecycle_refused():
    engine = make_engine()
    create_all(engine)
    with Session(engine) as session, pytest.raises(ValueError, match='R2'):
        EvidenceWriter(engine).emit(session, correlation_id='r2', event_type=EventType.KEY_ROTATED)


def test_invalid_rotation_and_revocation_refused():
    private, _ = _generate()
    signer = EvidenceSigner(private)
    engine = make_engine()
    create_all(engine)
    replacement = EvidenceSigner(_generate()[0])
    with Session(engine) as session:
        writer = EvidenceWriter(engine, signing=signer)
        writer.emit(session, correlation_id='r2', event_type=EventType.TOOL_CALL)
        for kind, body in (
            (EventType.KEY_ROTATED, {'key_id': _key_id(signer._load().public_key()),
                                    '_public_key': _public_pem(signer).decode(), '_escrow': 'vault'}),
            (EventType.KEY_ROTATED, {'key_id': 'unknown', '_public_key': 'invalid', '_escrow': ''}),
            (EventType.KEY_ROTATED, {'key_id': _key_id(replacement._load().public_key()),
                                    '_public_key': _public_pem(replacement).decode(), '_escrow': ''}),
            (EventType.KEY_REVOKED, {'key_id': 'unknown', 'recorded_at': 'invalid'}),
        ):
            with pytest.raises(ValueError, match='R2'):
                writer.emit(session, correlation_id='r2', event_type=kind, payload=body)
        session.commit()
        assert verify_chain(session) == 1
