"""R1 portable evidence recovery bundles.

Bundle members are private pending public format approval. Registry state is
replayed from lifecycle events. Embedded anchors preserve checkpoints but an
independent checkpoint or public key is still the operator's trust input.
Recovery restores evidence and retention, not operational configuration.
PIT recovery cannot resurrect bodies erased before the backup was taken.

CLI: backup create PATH --db SOURCE; backup verify PATH; backup restore PATH
--db NEW_SQLITE_PATH [--to-sequence N | --to-time ISO_TIMESTAMP]. Existing
--public-key and repeatable --anchors accept independently held trust inputs.
Creation reads a consistent database snapshot. Full recovery preserves current
retention policies; earlier cutoffs refuse mutable policy or hold ambiguity.
The bundle checksum detects corruption, not malicious replacement of a bundle.
Keep external checkpoints independently; embedded copies are recovery material.
"""
from __future__ import annotations

import json as _json
import os as _os
import tempfile as _tempfile
from datetime import datetime as _datetime
from pathlib import Path as _Path

from sqlmodel import Session as _Session, select as _select

from kognita.canonical import canonical_hash as _hash, hash_text as _hash_text
from kognita.db import make_engine as _make_engine, create_all as _create_all
from kognita.evidence import (
    ChainBreak as _ChainBreak, export_chain as _export_chain,
    verify_chain as _verify_chain, verify_export as _verify_export,
    GENESIS_HASH as _GENESIS_HASH,
)
from kognita.models import (
    EvidenceEvent as _EvidenceEvent, RetainedContent as _RetainedContent,
    RetainedReference as _RetainedReference, RetentionPolicy as _RetentionPolicy,
    RecordClassRetention as _RecordClassRetention, _RecordClassHold,
    as_utc as _as_utc,
)
from kognita.signing import _registry

_MODELS = (_RetainedContent, _RetainedReference, _RetentionPolicy,
           _RecordClassRetention, _RecordClassHold)


def _registry_json(_events):
    return _json.loads(_json.dumps(_registry(_events), default=lambda _v: _v.isoformat() if isinstance(_v, _datetime) else _v.decode('ascii')))


def _check_anchors(_chain, _anchors):
    for _anchor in _anchors:
        if set(_anchor) != {'sequence', 'event_hash'}:
            raise ValueError('R1: invalid anchor fields')
        _n = _anchor['sequence']
        if type(_n) is not int or not 1 <= _n <= len(_chain['events']):
            raise ValueError('R1: anchor outside chain')
        if _chain['events'][_n - 1]['event_hash'] != _anchor['event_hash']:
            raise ValueError('R1: anchor does not match chain')


def _verify(_bundle, *, _public_key=None, _anchors=()):
    try:
        if set(_bundle) != {'_chain', '_retention', '_registry', '_anchors', '_hash'}:
            raise ValueError('invalid bundle members')
        if _hash({k: v for k, v in _bundle.items() if k != '_hash'}) != _bundle['_hash']:
            raise ValueError('bundle checksum mismatch')
        _count = _verify_export(_bundle['_chain'], public_key=_public_key)
        if _bundle['_registry'] != _registry_json(_bundle['_chain']['events']):
            raise ValueError('key registry does not match lifecycle evidence')
        _check_anchors(_bundle['_chain'], [*_bundle['_anchors'], *_anchors])
        if set(_bundle['_retention']) not in (
            {_m.__tablename__ for _m in _MODELS},
            {_m.__tablename__ for _m in _MODELS[:3]},
        ):
            raise ValueError('missing retention table')
        _bodies = {}
        for _model in _MODELS:
            _seen = set()
            for _row in _bundle['_retention'].get(_model.__tablename__, []):
                if set(_row) != set(_model.model_fields):
                    raise ValueError('invalid retention fields')
                _obj = _model.model_validate(_row)
                _pk = tuple(getattr(_obj, _c.name) for _c in _model.__table__.primary_key)
                if _pk in _seen:
                    raise ValueError('duplicate retention row')
                _seen.add(_pk)
                if _model is _RetainedContent:
                    if _hash_text(_obj.body) != _obj.content_hash:
                        raise ValueError('retention body hash mismatch')
                    _bodies[_obj.content_hash] = _obj
                if _model in (_RetentionPolicy, _RecordClassRetention) and _obj.retain_days is not None and _obj.retain_days < 0:
                    raise ValueError('invalid retention period')
        for _name in ('retained_references', '_record_class_holds'):
            for _row in _bundle['_retention'].get(_name, []):
                if _row['content_hash'] not in _bodies:
                    raise ValueError('retention reference has no body')
        return _count
    except (ValueError, TypeError, KeyError, AttributeError, _ChainBreak) as _exc:
        raise ValueError(f'R1: backup verification refused: {_exc}') from _exc


def _create(_session, *, _anchors=(), _public_key=None):
    _verify_chain(_session, public_key=_public_key)
    _chain = _export_chain(_session)
    _bundle = {
        '_chain': _chain,
        '_retention': {_m.__tablename__: [_r.model_dump(mode='json') for _r in _session.exec(_select(_m)).all()] for _m in _MODELS},
        '_registry': _registry_json(_chain['events']),
        '_anchors': [dict(_r['payload']) for _r in _chain['events'] if _r['event_type'] == 'ANCHOR_PUBLISHED'] + list(_anchors),
    }
    _bundle['_hash'] = _hash(_bundle)
    _verify(_bundle, _public_key=_public_key)
    return _bundle


def _prefix(_bundle, _to_sequence=None, _to_time=None):
    if _to_sequence is not None and _to_time is not None:
        raise ValueError('R1: choose one recovery boundary')
    _rows = _bundle['_chain']['events']
    _n = len(_rows)
    if _to_sequence is not None:
        if type(_to_sequence) is not int or not 0 <= _to_sequence <= _n:
            raise ValueError('R1: recovery sequence outside backup')
        _n = _to_sequence
    if _to_time is not None:
        _at = _as_utc(_datetime.fromisoformat(_to_time))
        _n = 0
        for _row in _rows:
            if _as_utc(_datetime.fromisoformat(_row['recorded_at'])) > _at:
                break
            _n += 1
        if any(_as_utc(_datetime.fromisoformat(_r['recorded_at'])) <= _at for _r in _rows[_n:]):
            raise ValueError('R1: time boundary is not a contiguous prefix')
    _result = _json.loads(_json.dumps(_bundle))
    if _n < len(_rows):
        # Mutable retention policies and refreshed holds have no historical log.
        # Refuse ambiguity instead of claiming to reconstruct unavailable state.
        if any(_r['event_type'] == 'ERASURE' for _r in _rows[_n:]):
            raise ValueError('R1: erased historical retention cannot be recovered')
        _cutoff = _as_utc(_datetime.fromisoformat(_rows[_n - 1]['recorded_at'])) if _n else None
        if (_bundle['_retention']['retention_policies'] or
                _bundle['_retention'].get('RecordClassRetention')):
            raise ValueError('R1: historical retention policies unavailable')
        for _name in ('retained_content', 'retained_references', '_record_class_holds'):
            if any(_cutoff is None or _as_utc(_datetime.fromisoformat(_r['retained_at'])) > _cutoff for _r in _bundle['_retention'].get(_name, [])):
                raise ValueError('R1: historical retention state unavailable; use an earlier backup')
    _chain = _result['_chain']
    _chain['events'] = _rows[:_n]
    _chain['event_count'] = _n
    _chain['interest_count'] = sum(bool(_r['of_interest']) for _r in _rows[:_n])
    _chain['head_hash'] = _rows[_n - 1]['event_hash'] if _n else _GENESIS_HASH
    _result['_registry'] = _registry_json(_rows[:_n])
    _result['_anchors'] = [_a for _a in _result['_anchors'] if _a['sequence'] <= _n]
    _result['_hash'] = _hash({k: v for k, v in _result.items() if k != '_hash'})
    return _result


def _restore(_bundle, _destination, *, _to_sequence=None, _to_time=None, _public_key=None, _anchors=()):
    _verify(_bundle, _public_key=_public_key, _anchors=_anchors)
    _bundle = _prefix(_bundle, _to_sequence, _to_time)
    _verify(_bundle, _public_key=_public_key)
    _path = _Path(_destination)
    if '://' in str(_destination) or str(_destination) == ':memory:':
        raise ValueError('R1: restore requires a new SQLite file destination')
    if _path.exists():
        raise ValueError('R1: restore destination already exists')
    _fd, _tmp = _tempfile.mkstemp(prefix='.kognita-restore-', dir=_path.parent)
    _os.close(_fd)
    _engine = None
    try:
        _engine = _make_engine(_tmp)
        _create_all(_engine)
        with _Session(_engine) as _session:
            for _row in _bundle['_chain']['events']:
                _fields = {k: v for k, v in _row.items() if k in _EvidenceEvent.model_fields}
                _session.add(_EvidenceEvent.model_validate(_fields))
                _session.flush()
            for _model in _MODELS:
                for _row in _bundle['_retention'].get(_model.__tablename__, []):
                    _session.add(_model.model_validate(_row))
            _session.flush()
            _verify_chain(_session, public_key=_public_key)
            _session.commit()
        with _engine.connect() as _connection:
            _connection.exec_driver_sql('PRAGMA wal_checkpoint(TRUNCATE)')
        _engine.dispose()
        with open(_tmp, 'rb') as _file:
            _os.fsync(_file.fileno())
        _os.link(_tmp, _path)
        _fd = _os.open(_path.parent, _os.O_RDONLY)
        try:
            _os.fsync(_fd)
        finally:
            _os.close(_fd)
    finally:
        if _engine is not None:
            _engine.dispose()
        for _suffix in ('', '-wal', '-shm'):
            _Path(_tmp + _suffix).unlink(missing_ok=True)
    return _bundle['_chain']['event_count']
