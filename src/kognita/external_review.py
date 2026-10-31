"""T3 local external review API with authenticated reviewer submissions.

The caller supplies an independent root public PEM and commits the session.
Reviewers use keys already enrolled through R2 rotation. A retired key can
review, but a revoked key cannot. Review signatures are separate from the
writer signature, so reviewer and evidence writer need not share a key.
"""
from dataclasses import dataclass

from kognita.canonical import canonical_hash
from kognita.evidence import export_chain, verify_chain
from kognita.signing import _registry, _verify, _key_id as signing_key_id
from kognita.vocabulary import EventType, ActorType


@dataclass(frozen=True)
class ReviewPeriod:
    """Immutable review snapshot; annotations return a replacement snapshot."""

    _body: str


def _decode(period):
    import json
    try:
        return json.loads(period._body)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('T3: invalid review snapshot') from exc


def _period(body):
    import json
    return ReviewPeriod(json.dumps(body, sort_keys=True))


def request_review_period(session, *, start, end, key_id, public_key):
    """Select an inclusive sequence range from an independently verified chain."""
    try:
        verify_chain(session, public_key=public_key)
        rows = export_chain(session)['events']
        keys = _registry(rows, public_key)
        if not public_key or key_id not in keys or keys[key_id]['_revoked'] is not None:
            raise ValueError('R2: reviewer key is unknown or revoked')
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(rows):
            raise ValueError('invalid sequence period')
        return _period({'key_id': key_id, 'start': start, 'end': end,
                        'head': rows[end - 1]['event_hash'], 'annotations': []})
    except Exception as exc:
        raise ValueError(f'T3: review period refused: {exc}') from exc


def annotate(period, *, sequence, text):
    body = _decode(period)
    if (type(sequence) is not int or not body['start'] <= sequence <= body['end']
            or not isinstance(text, str) or not text.strip()):
        raise ValueError('T3: annotation requires a target in the period and nonempty text')
    body['annotations'].append({'sequence': sequence, 'text': text})
    return _period(body)


def _verify_review(row, keys, rows):
    """Authenticate reviewer content and bind it to the preceding chain prefix."""
    try:
        payload = row['payload']
        body = payload['review']
        key = keys[body['key_id']]
        if key['_revoked'] is not None:
            from datetime import datetime
            from kognita.models import as_utc
            if as_utc(datetime.fromisoformat(row['recorded_at'])) >= key['_revoked']:
                raise ValueError('R2: reviewer signature at or after revocation time')
        start, end = body['start'], body['end']
        if (type(start) is not int or type(end) is not int
                or not 1 <= start <= end < row['sequence']
                or rows[end - 1]['event_hash'] != body['head']):
            raise ValueError('review prefix changed')
        if not isinstance(body['annotations'], list):
            raise ValueError('invalid annotations')
        for item in body['annotations']:
            if (type(item['sequence']) is not int or not start <= item['sequence'] <= end
                    or not isinstance(item['text'], str) or not item['text'].strip()):
                raise ValueError('invalid annotation')
        _verify(body, payload['signature'], body['key_id'],
                key['_public_key'], row['sequence'])
    except Exception as exc:
        raise ValueError(f'T3: reviewer signature or period refused: {exc}') from exc


def submit_review(session, period, *, signing, evidence, public_key):
    """Sign the complete period and annotations, verify registry, then append."""
    try:
        if not public_key or evidence.redact_payload is not None:
            raise ValueError('independent trust root and unfiltered review evidence required')
        verify_chain(session, public_key=public_key)
        rows = export_chain(session)['events']
        keys = _registry(rows, public_key)
        body = _decode(period)
        if body['key_id'] not in keys or keys[body['key_id']]['_revoked'] is not None:
            raise ValueError('R2: reviewer key is unknown or revoked')
        if signing_key_id(signing._load().public_key()) != body['key_id']:
            raise ValueError('reviewer key mismatch')
        payload = {'review': body, 'signature': signing._sign(body)}
        _verify_review({'payload': payload, 'sequence': len(rows) + 1}, keys, rows)
        if any(row['event_type'] == 'EXTERNAL_REVIEW_SUBMITTED'
               and row['payload'].get('review') == body for row in rows):
            raise ValueError('review already submitted')
        return evidence.emit(session, correlation_id=canonical_hash(body),
                              event_type=EventType.EXTERNAL_REVIEW_SUBMITTED,
                              actor_type=ActorType.HUMAN, actor_id=body['key_id'], payload=payload)
    except Exception as exc:
        raise ValueError(f'T3: review submission refused: {exc}') from exc
