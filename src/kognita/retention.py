"""Content retention — prompts, responses, and source snapshots by hash.

The evidence chain holds the hash. This store holds the bytes that hash, under
a retention period set per use case. Erasure deletes the bytes and appends an
``ERASURE`` event, so the chain still shows that the content existed and was
removed. The chain stays verifiable because nothing already written is rewritten.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlmodel import Session, select

from kognita.canonical import canonical_json, hash_text
from kognita.evidence import EvidenceWriter
from kognita.exceptions import RetentionError
from kognita.models import RetainedContent, RetentionPolicy, as_utc, utcnow
from kognita.vocabulary import ActorType, Classification, EventType


class RetentionStore:
    """Content-addressed store in the same database as the evidence chain.

    The table is not the chain. A hash already present is not overwritten: the
    bytes that produced the hash are the bytes that stay until erasure.
    """

    def retain_text(
        self,
        session: Session,
        text: str,
        *,
        kind: str,
        use_case: str = "",
        correlation_id: str = "",
        now: datetime | None = None,
    ) -> str:
        """Store ``text`` and return its hash. A second store of the same bytes is a no-op."""
        digest = hash_text(text)
        self._keep(
            session,
            digest,
            text,
            kind=kind,
            use_case=use_case,
            correlation_id=correlation_id,
            now=now,
        )
        return digest

    def retain_value(
        self,
        session: Session,
        value: Any,
        *,
        kind: str,
        use_case: str = "",
        correlation_id: str = "",
        now: datetime | None = None,
    ) -> str:
        """Store the canonical JSON of ``value`` and return that document's hash."""
        return self.retain_text(
            session,
            canonical_json(value),
            kind=kind,
            use_case=use_case,
            correlation_id=correlation_id,
            now=now,
        )

    def read(self, session: Session, content_hash: str) -> str | None:
        """The retained bytes, or None when the hash was never stored or has been erased."""
        row = session.get(RetainedContent, content_hash)
        if row is None:
            return None
        return row.body

    def set_policy(
        self,
        session: Session,
        use_case: str,
        *,
        retain_days: int | None,
    ) -> RetentionPolicy:
        """Set how long ``use_case`` keeps content. None keeps it until erasure."""
        if retain_days is not None and retain_days < 0:
            raise RetentionError("retain_days cannot be negative")
        row = session.get(RetentionPolicy, use_case)
        if row is None:
            row = RetentionPolicy(use_case=use_case)
        row.retain_days = retain_days
        session.add(row)
        session.flush()
        return row

    def erase(
        self,
        session: Session,
        content_hash: str,
        *,
        evidence: EvidenceWriter,
        actor_id: str,
        correlation_id: str | None = None,
        reason: str | None = None,
        actor_type: ActorType = ActorType.HUMAN,
        now: datetime | None = None,
    ) -> None:
        """Remove one retained body and record the erasure on the chain.

        The chain keeps the hash. The event records that the content was removed.
        A second erasure of the same hash is an error: there is nothing left to remove.
        """
        row = session.get(RetainedContent, content_hash)
        if row is None:
            raise RetentionError(f"retained content {content_hash} is not in the store")
        kind = row.kind
        use_case = row.use_case
        linked = correlation_id or row.correlation_id or f"erasure:{content_hash[:12]}"
        session.delete(row)
        session.flush()
        evidence.emit(
            session,
            correlation_id=linked,
            event_type=EventType.ERASURE,
            actor_type=actor_type,
            actor_id=actor_id,
            classification=Classification.C1,
            payload={
                "content_hash": content_hash,
                "kind": kind,
                "use_case": use_case,
                "reason": reason,
            },
        )

    def enforce(
        self,
        session: Session,
        *,
        evidence: EvidenceWriter,
        now: datetime | None = None,
        actor_id: str = "retention",
    ) -> int:
        """Erase content whose use-case retention period has elapsed. Returns how many."""
        at = now or utcnow()
        policies = {
            row.use_case: row for row in session.exec(select(RetentionPolicy)).all()
        }
        expired: list[str] = []
        for row in session.exec(select(RetainedContent)).all():
            policy = policies.get(row.use_case)
            if policy is None or policy.retain_days is None:
                continue
            retained_at = as_utc(row.retained_at)
            if retained_at is None:
                continue
            if retained_at + timedelta(days=policy.retain_days) <= at:
                expired.append(row.content_hash)
        for content_hash in expired:
            self.erase(
                session,
                content_hash,
                evidence=evidence,
                actor_id=actor_id,
                actor_type=ActorType.SYSTEM,
                reason="retention period elapsed",
                now=at,
            )
        return len(expired)

    def _keep(
        self,
        session: Session,
        digest: str,
        body: str,
        *,
        kind: str,
        use_case: str,
        correlation_id: str,
        now: datetime | None,
    ) -> None:
        if session.get(RetainedContent, digest) is not None:
            return
        session.add(
            RetainedContent(
                content_hash=digest,
                kind=kind,
                use_case=use_case,
                correlation_id=correlation_id,
                body=body,
                retained_at=now or utcnow(),
            )
        )
        session.flush()


__all__ = ["RetentionStore"]
