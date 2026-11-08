"""Small protocol-neutral Ed25519 assertions with pinned trust roots."""
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from kognita.canonical import canonical_hash
from kognita.models import as_utc
from kognita.assurance.records import Record


class Assertion(Record):
    """Protocol-neutral external identity with exact attenuated scopes."""
    issuer: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    principal: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    audience: str = Field(min_length=1)
    scopes: dict[str, list[str]]
    issued_at: datetime
    expires_at: datetime
    delegation_depth: int = Field(default=0, ge=0, le=16, strict=True)
    parent_digest: str | None = None

    @model_validator(mode='after')
    def bounded(self):
        """Reject unbounded validity windows and incomplete signed scope."""
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError('A2: timezone required')
        if self.expires_at <= self.issued_at or not self.scopes:
            raise ValueError('A2: bounded assertion required')
        for key, values in self.scopes.items():
            if not key.strip() or not values or any(not v.strip() or v == '*' for v in values):
                raise ValueError('A2: exact nonempty scopes required')
        if bool(self.parent_digest) != bool(self.delegation_depth):
            raise ValueError('A2: parent and depth must agree')
        return self


class Attestation(Record):
    """Independent signed boundary evidence for a specific policy and deployment."""
    kind: Literal['privacy', 'environment', 'monitoring', 'instruction', 'transfer']
    issuer: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    issued_at: datetime
    expires_at: datetime
    deployment: str = Field(min_length=1)
    policy: str = Field(min_length=1)
    claims: dict
    reference: str = Field(min_length=1)

    @model_validator(mode='after')
    def bounded(self):
        """Reject unbounded validity windows and incomplete signed scope."""
        if (self.issued_at.tzinfo is None or self.expires_at.tzinfo is None
                or self.expires_at <= self.issued_at):
            raise ValueError('A7: bounded timezone-aware attestation required')
        return self


def sign_record(record, signer):
    """Sign a domain-separated document; the key itself never enters evidence."""
    document = {'domain': 'kognita-assurance-v1', 'type': type(record).__name__,
                'body': record.model_dump(mode='json')}
    signature = signer._sign(document)
    return {'document': document, 'signature': signature}


def verify_record(packet, model, keys, at):
    """Authenticate a domain-separated record and enforce its validity window."""
    from kognita.signing import _verify
    try:
        if set(packet) != {'document', 'signature'}:
            raise ValueError('packet fields')
        doc = dict(packet['document'])
        if set(doc) != {'domain', 'type', 'body', 'key_id'} or (
                doc['domain'] != 'kognita-assurance-v1' or doc['type'] != model.__name__):
            raise ValueError('domain')
        record = model.model_validate(doc['body'])
        pem = keys[record.issuer].encode()
        _verify(doc, packet['signature'], doc['key_id'], pem, 0)
        if not as_utc(record.issued_at) <= as_utc(at) < as_utc(record.expires_at):
            raise ValueError('time')
        return record
    except Exception as exc:
        raise ValueError('A2/A7: untrusted, invalid or expired signed record') from exc


def verify_assertion(packet, p, envelope, at, *, parent=None):
    """Verify identity, audience, scopes and every independently trusted parent."""
    a = verify_record(packet, Assertion, p.issuers, at)
    if (a.subject != envelope.agent_name or a.principal != envelope.principal
            or a.purpose != envelope.purpose or a.audience != p.audience):
        raise ValueError('A2: subject/principal/purpose/audience mismatch')
    requested = {k: [v] for k, v in envelope.all_subjects().items()}
    requested['tools'] = sorted({*requested.get('tools', []), envelope.tool})
    for dimension, values in envelope.context.get('_effective_subjects', {}).items():
        requested[dimension] = sorted(set(requested.get(dimension, [])) | set(values))
    if not all(set(v) <= set(a.scopes.get(k, [])) for k, v in requested.items()):
        raise ValueError('A2: authority expansion')
    if a.delegation_depth:
        if not isinstance(parent, list) or len(parent) != a.delegation_depth:
            raise ValueError('A2: complete independently trusted parent chain required')
        child = a
        for packet_parent in parent:
            pa = verify_record(packet_parent, Assertion, p.issuers, at)
            if (child.parent_digest != canonical_hash(packet_parent)
                    or child.delegation_depth != pa.delegation_depth + 1
                    or (child.principal, child.purpose, child.audience) !=
                       (pa.principal, pa.purpose, pa.audience)
                    or child.issued_at < pa.issued_at or child.expires_at > pa.expires_at
                    or not all(set(v) <= set(pa.scopes.get(k, [])) for k, v in child.scopes.items())):
                raise ValueError('A2: delegation expansion')
            child = pa
        if child.delegation_depth:
            raise ValueError('A2: incomplete chain')
    elif parent:
        raise ValueError('A2: unexpected parent')
    return a
