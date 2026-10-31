"""Optional local Ed25519 signing (T1).

EvidenceSigner accepts a PKCS8 PEM private key as bytes. Signing starts
with an empty chain; legacy unsigned chains continue using hash verification. Configure a writer
with ``EvidenceWriter(engine, signing=EvidenceSigner(pem))``. Generate PEM
files with ``kognita keys generate PRIVATE_PATH PUBLIC_PATH``. The public
file is the independent trust input for ``evidence verify --public-key``.
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
