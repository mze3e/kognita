"""T6 checkpoints kept independently of the evidence database.

Keep the file in separately controlled storage. A checkpoint cannot protect
against an actor who can replace both the database and the checkpoint.
Sink operations are private pending approval of public method names.
"""
from __future__ import annotations

import json as _json
import os as _os
from pathlib import Path as _Path
from typing import Protocol as _Protocol

from kognita.evidence import (
    ChainBreak as _ChainBreak, EvidenceWriter as _EvidenceWriter,
    export_chain as _export_chain, verify_chain as _verify_chain,
)
from kognita.vocabulary import EventType as _EventType


class AnchorSink(_Protocol):
    """A sink must durably publish a checkpoint or raise an exception."""

    def _publish(self, _record: dict) -> None: ...


class FileAnchorSink:
    """Create a durable checkpoint file exclusively; never overwrite one."""

    def __init__(self, _path: str | _Path) -> None:
        self._path = _Path(_path)

    def _publish(self, _record: dict) -> None:
        try:
            with self._path.open('x', encoding='utf-8') as _file:
                _file.write(_json.dumps(_record, sort_keys=True) + '\n')
                _file.flush()
                _os.fsync(_file.fileno())
            _fd = _os.open(self._path.parent, _os.O_RDONLY)
            try:
                _os.fsync(_fd)
            finally:
                _os.close(_fd)
        except OSError as _exc:
            raise ValueError(f'T6: cannot publish checkpoint: {_exc}') from _exc


def _anchor(_session, _writer: _EvidenceWriter, _sink: AnchorSink):
    """Publish the verified prefix and evidence publication in the transaction.

    The caller commits. A later rollback can leave an external checkpoint of
    the preceding committed prefix, which remains valid and useful.
    """
    try:
        _verify_chain(_session)
    except _ChainBreak as _exc:
        raise ValueError(f'T6: refusing to anchor an invalid chain: {_exc}') from _exc
    _export = _export_chain(_session)
    if not _export['events']:
        raise ValueError('T6: cannot anchor an empty chain')
    _record = { 'sequence': _export['event_count'], 'event_hash': _export['head_hash'] }
    # Validate signing and event creation before publishing to the external sink.
    _event = _writer.emit(
        _session, correlation_id='anchoring', event_type=_EventType.ANCHOR_PUBLISHED,
        payload=_record,
    )
    try:
        _sink._publish(_record)
    except Exception as _exc:
        # The caller must roll back the transaction on refusal.
        raise ValueError(f'T6: checkpoint publication failed: {_exc}') from _exc
    return _event


def _verify_anchors(_export: dict, _paths) -> None:
    """Compare every independently supplied checkpoint with a verified export."""
    for _path in _paths:
        try:
            _record = _json.loads(_Path(_path).read_text(encoding='utf-8'))
            if not isinstance(_record, dict) or set(_record) != {'sequence', 'event_hash'}:
                raise ValueError('invalid checkpoint fields')
            _sequence = _record['sequence']
            _hash = _record['event_hash']
            if type(_sequence) is not int or _sequence < 1:
                raise ValueError('invalid checkpoint sequence')
            if not isinstance(_hash, str) or len(_hash) != 64 or any(
                _c not in '0123456789abcdef' for _c in _hash
            ):
                raise ValueError('invalid checkpoint hash')
            _rows = _export['events']
            if _sequence > len(_rows) or _rows[_sequence - 1]['event_hash'] != _hash:
                raise ValueError('checkpoint does not match chain prefix')
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as _exc:
            raise _ChainBreak(0, f'T6: cannot verify checkpoint {_path}: {_exc}') from _exc
