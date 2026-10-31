"""Optional local Ed25519 signing (T1).

EvidenceSigner accepts a PKCS8 PEM private key as bytes. Signing starts
with an empty chain; legacy unsigned chains continue using hash verification. Configure a writer
with ``EvidenceWriter(engine, signing=EvidenceSigner(pem))``. Generate PEM
files with ``kognita keys generate PRIVATE_PATH PUBLIC_PATH``. The public
file is the independent trust input for ``evidence verify --public-key``.

R2 lifecycle commands use positional inputs (and the existing --db flag)::

    kognita keys rotate PRIVATE_PATH REPLACEMENT_PRIVATE_PATH ESCROW_RECORD
    kognita keys revoke PRIVATE_PATH KEY_ID REVOCATION_TIME
    kognita keys list

Generate the replacement first with ``keys generate``. Rotation is signed by
the active key and activates the replacement for subsequent writes. Keep the
initial public PEM as the verification trust root. The registry is replayed
from lifecycle evidence, with no separate mutable table. Private payload
members ``_public_key`` and ``_escrow`` record the replacement public PEM and
bank custody reference; private keys are never stored. Actual escrow and
custody remain the bank's responsibility. Revocation uses an ISO timestamp in
``recorded_at`` and rejects signatures at or after that cutoff, including
earlier chain entries. A revoked key is immediately unavailable for writes;
rotate before revoking the old key to preserve write availability. ``list``
checks local hash integrity; independent signature verification requires the
initial public PEM through ``evidence verify --public-key``.
"""
from hashlib import sha256 as _sha256


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    except ImportError as exc:
        raise ValueError('T1: signing requires the signing extra') from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


def _key_id(key):
    serialization, _, _ = _crypto()
    return _sha256(key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).hexdigest()


class EvidenceSigner:
    """A local PEM private key, loaded lazily when a write is attempted."""

    def __init__(self, signing: bytes):
        self._pem = signing

    def _load(self):
        serialization, private, _ = _crypto()
        try:
            key = serialization.load_pem_private_key(self._pem, password=None)
            if not isinstance(key, private):
                raise ValueError('expected Ed25519')
            return key
        except (ValueError, TypeError) as exc:
            raise ValueError('T1: invalid Ed25519 private key') from exc

    def _sign(self, header):
        from kognita.canonical import canonical_hash
        key = self._load()
        header['key_id'] = _key_id(key.public_key())
        return key.sign(bytes.fromhex(canonical_hash(header))).hex()


def _verify(header, signature, key_id, public_key, sequence):
    from kognita.canonical import canonical_hash
    from kognita.evidence import ChainBreak
    try:
        serialization, _, public = _crypto()
        key = serialization.load_pem_public_key(public_key)
        if not isinstance(key, public) or key_id != _key_id(key):
            raise ValueError('wrong key')
        key.verify(bytes.fromhex(signature), bytes.fromhex(canonical_hash(header)))
    except Exception as exc:
        raise ChainBreak(sequence, 'T1: signature verification failed') from exc


def _generate():
    serialization, private, _ = _crypto()
    key = private.generate()
    return (key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                              serialization.NoEncryption()),
            key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))


__all__ = ['EvidenceSigner']


def _public_pem(signing):
    serialization, _, _ = _crypto()
    return signing._load().public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def _registry(rows, public_key=None):
    """Replay the private registry from authenticated lifecycle evidence."""
    from datetime import datetime
    from kognita.evidence import ChainBreak, _check_signature, _signature_header, _version_header
    from kognita.models import as_utc
    registry = {}
    active = None
    signed = []
    for row in rows:
        sequence = row['sequence']
        try:
            key_id = row.get('key_id')
            kind = row['event_type']
            lifecycle = kind in ('KEY_ROTATED', 'KEY_REVOKED')
            if key_id is None:
                if lifecycle:
                    raise ValueError('unsigned lifecycle event')
                continue
            if active is None:
                active = key_id
                registry[active] = {'_public_key': public_key, '_revoked': None, '_escrow': None}
            if key_id != active or registry[key_id]['_revoked'] is not None:
                raise ValueError('inactive or revoked signing key')
            header = {k: row[k] for k in ('sequence', 'correlation_id', 'event_type',
                      'actor_type', 'actor_id', 'classification', 'payload_hash', 'prev_hash', 'recorded_at')}
            _version_header(header, row.get('schema_version', 1), sequence)
            _signature_header(header, row.get('signature'), key_id, row.get('schema_version', 1), sequence)
            if public_key is not None:
                _check_signature(header, registry[key_id]['_public_key'], True, sequence)
            moment = as_utc(datetime.fromisoformat(row['recorded_at']))
            signed.append((key_id, moment, sequence))
            if not lifecycle:
                continue
            body = row['payload']
            target = body['key_id']
            if kind == 'KEY_ROTATED':
                pem = body['_public_key'].encode('ascii')
                serialization, _, public = _crypto()
                key = serialization.load_pem_public_key(pem)
                if not isinstance(key, public) or _key_id(key) != target or target in registry:
                    raise ValueError('invalid or reused rotation key')
                escrow = body['_escrow']
                if not isinstance(escrow, str) or not escrow.strip():
                    raise ValueError('escrow record required')
                registry[target] = {'_public_key': pem, '_revoked': None, '_escrow': escrow}
                active = target
            else:
                if target not in registry or registry[target]['_revoked'] is not None:
                    raise ValueError('unknown or already revoked key')
                cutoff = as_utc(datetime.fromisoformat(body['recorded_at']))
                registry[target]['_revoked'] = cutoff
                if any(k == target and t >= cutoff for k, t, _ in signed):
                    raise ValueError('signature at or after revocation time')
        except ChainBreak:
            raise
        except Exception as exc:
            raise ChainBreak(sequence, 'R2: key lifecycle refused: ' + str(exc)) from exc
    if public_key is not None:
        from kognita.external_review import _verify_review
        for row in rows:
            if row['event_type'] == 'EXTERNAL_REVIEW_SUBMITTED':
                try:
                    _verify_review(row, registry, rows)
                except ValueError as exc:
                    raise ChainBreak(row['sequence'], str(exc)) from exc
    return registry
