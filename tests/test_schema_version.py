"""Schema versions preserve original evidence hashes and bind new headers."""
import copy

import pytest
from sqlmodel import Session, select

from kognita.canonical import canonical_hash
from kognita.db import create_all, make_engine
from kognita.evidence import (
    ChainBreak, EvidenceWriter, GENESIS_HASH, export_chain,
    recompute_event_hash, verify_chain, verify_export,
)
from kognita.models import EvidenceEvent
from kognita.vocabulary import EventType


def _legacy_store():
    engine = make_engine()
    create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE evidence_events DROP COLUMN schema_version')
        previous = GENESIS_HASH
        rows = []
        for sequence in (1, 2):
            header = {
                'sequence': sequence, 'correlation_id': 'legacy',
                'event_type': EventType.TOOL_CALL.value, 'actor_type': 'SYSTEM',
                'actor_id': 'system', 'classification': 'C1',
                'payload_hash': canonical_hash({}), 'prev_hash': previous,
                'recorded_at': '2026-01-01T00:00:00+00:00',
            }
            previous = canonical_hash(header)
            row = dict(header, payload={}, event_hash=previous)
            rows.append(row)
            connection.exec_driver_sql(
                'INSERT INTO evidence_events '
                '(sequence, correlation_id, event_type, actor_type, actor_id, classification, '
                'payload, payload_hash, prev_hash, event_hash, recorded_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (sequence, 'legacy', header['event_type'], 'SYSTEM', 'system', 'C1',
                 '{}', header['payload_hash'], header['prev_hash'], previous,
                 header['recorded_at']),
            )
    artifact = {'events': rows, 'event_count': 2, 'head_hash': previous}
    return engine, artifact


def test_old_chain_verifies():
    engine, legacy_export = _legacy_store()
    assert verify_export(legacy_export) == 2
    create_all(engine)
    create_all(engine)
    with Session(engine) as session:
        events = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
        assert [event.schema_version for event in events] == [1, 1]
        assert [event.event_hash for event in events] == [row['event_hash'] for row in legacy_export['events']]
        assert verify_chain(session) == 2
        assert verify_export(export_chain(session)) == 2
        EvidenceWriter(engine).emit(session, correlation_id='new', event_type=EventType.TOOL_CALL)
        session.commit()
        assert verify_chain(session) == 3
        assert verify_export(export_chain(session)) == 3


def test_new_events_carry_version():
    engine = make_engine()
    create_all(engine)
    snapshots = []
    with Session(engine) as session:
        event = EvidenceWriter(engine, telemetry=snapshots.append).emit(
            session, correlation_id='new', event_type=EventType.TOOL_CALL,
        )
        session.commit()
        session.refresh(event)
        assert event.schema_version == 2
        assert snapshots[0]['schema_version'] == 2
        assert recompute_event_hash(event) == event.event_hash
        artifact = export_chain(session)
        assert artifact['events'][0]['schema_version'] == 2
        assert verify_chain(session) == verify_export(artifact) == 1
        for version in (1, 3):
            changed = copy.deepcopy(artifact)
            changed['events'][0]['schema_version'] = version
            with pytest.raises(ChainBreak):
                verify_export(changed)
        del artifact['events'][0]['schema_version']
        with pytest.raises(ChainBreak):
            verify_export(artifact)
        event.schema_version = 1
        assert recompute_event_hash(event) != event.event_hash
        session.rollback()


@pytest.mark.parametrize('version', [None, True, '2', 0, -1, 3])
def test_invalid_schema_version_fails_closed(version):
    engine, artifact = _legacy_store()
    artifact['events'][0]['schema_version'] = version
    with pytest.raises(ChainBreak, match='T5: unsupported schema_version'):
        verify_export(artifact)
    create_all(engine)
    with Session(engine) as session:
        event = session.exec(select(EvidenceEvent)).first()
        event.schema_version = version
        with pytest.raises(ChainBreak, match='T5: unsupported schema_version'):
            recompute_event_hash(event)
        session.rollback()
