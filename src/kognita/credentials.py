"""Per-agent workload secrets. Callers own the transaction, including evidence."""
from __future__ import annotations

import hashlib as _hashlib
import hmac as _hmac
import secrets as _secrets

from sqlmodel import Session as _Session
from kognita.models import _WorkloadCredential
from kognita.registry import get as _get
from kognita.evidence import EvidenceWriter as _EvidenceWriter
from kognita.vocabulary import ActorType as _ActorType, EventType as _EventType


def _digest(secret: str) -> str:
    return _hashlib.sha256(secret.encode()).hexdigest()


def issue(session: _Session, agent_name: str, *, evidence: _EvidenceWriter,
          actor_id: str) -> str:
    """Return a new secret once; persist only its SHA-256 digest."""
    agent = _get(session, agent_name)
    if agent is None or not actor_id.strip() or actor_id != agent.owner_exec:
        raise ValueError("S2-CREDENTIAL: accountable owner required")
    secret = _secrets.token_urlsafe(32)
    digest = _digest(secret)
    evidence.emit(session, correlation_id=f"credentials:{agent_name}",
        event_type=_EventType.POLICY_CHANGE, actor_type=_ActorType.HUMAN,
        actor_id=actor_id, payload={"action": "CREDENTIAL_ISSUED",
        "agent": agent_name, "digest": digest, "citation": "S2-CREDENTIAL"})
    session.add(_WorkloadCredential(digest=digest, agent_name=agent_name))
    session.flush()
    return secret


def verify(session: _Session, agent_name: str, credential: str | None) -> bool:
    """Fresh indexed lookup on every call, including after revocation."""
    if not isinstance(credential, str) or not credential or len(credential) > 256:
        return False
    digest = _digest(credential)
    row = session.get(_WorkloadCredential, digest, populate_existing=True)
    return bool(row and not row.revoked and
                _hmac.compare_digest(row.agent_name.encode(), agent_name.encode()) and
                _hmac.compare_digest(row.digest, digest))


def revoke(session: _Session, agent_name: str, credential: str, *,
           evidence: _EvidenceWriter, actor_id: str) -> None:
    """Revoke one secret with append-only evidence."""
    agent = _get(session, agent_name)
    row = session.get(_WorkloadCredential, _digest(credential), populate_existing=True)
    if agent is None or actor_id != agent.owner_exec or not actor_id.strip():
        raise ValueError("S2-CREDENTIAL: accountable owner required")
    if row is None or row.agent_name != agent_name:
        raise ValueError("S2-CREDENTIAL: credential does not belong to agent")
    if row.revoked:
        return
    evidence.emit(session, correlation_id=f"credentials:{agent_name}",
        event_type=_EventType.POLICY_CHANGE, actor_type=_ActorType.HUMAN,
        actor_id=actor_id, payload={"action": "CREDENTIAL_REVOKED",
        "agent": agent_name, "digest": row.digest, "citation": "S2-CREDENTIAL"})
    row.revoked = True
    session.add(row)
    session.flush()


__all__ = ["issue", "verify", "revoke"]
