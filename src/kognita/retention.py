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
from kognita.models import (
    RetainedContent,
    RetainedReference,
    RetentionPolicy,
    as_utc,
    utcnow,
)
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
        for reference in session.exec(
            select(RetainedReference).where(RetainedReference.content_hash == content_hash)
        ).all():
            session.delete(reference)
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
        holds: dict[str, list[tuple[str, datetime | None]]] = {}
        for reference in session.exec(select(RetainedReference)).all():
            holds.setdefault(reference.content_hash, []).append(
                (reference.use_case, as_utc(reference.retained_at))
            )
        expired: list[str] = []
        for row in session.exec(select(RetainedContent)).all():
            # The first store is a hold too, so content kept before references
            # existed is still governed by its own use case.
            references = [(row.use_case, as_utc(row.retained_at))]
            references.extend(holds.get(row.content_hash, []))
            if all(
                _elapsed(policies.get(use_case), retained_at, at)
                for use_case, retained_at in references
            ):
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
        at = now or utcnow()
        self._hold(session, digest, use_case, correlation_id, at)
        if session.get(RetainedContent, digest) is not None:
            session.flush()
            return
        session.add(
            RetainedContent(
                content_hash=digest,
                kind=kind,
                use_case=use_case,
                correlation_id=correlation_id,
                body=body,
                retained_at=at,
            )
        )
        session.flush()

    def _hold(
        self,
        session: Session,
        digest: str,
        use_case: str,
        correlation_id: str,
        at: datetime,
    ) -> None:
        """Record or extend this use case's hold on ``digest``."""
        reference = session.get(RetainedReference, (digest, use_case))
        if reference is None:
            session.add(
                RetainedReference(
                    content_hash=digest,
                    use_case=use_case,
                    correlation_id=correlation_id,
                    retained_at=at,
                )
            )
            return
        previous = as_utc(reference.retained_at)
        if previous is None or previous < as_utc(at):
            reference.retained_at = at
            session.add(reference)


def _elapsed(policy: RetentionPolicy | None, retained_at: datetime | None, at: datetime) -> bool:
    """Whether one hold has passed its use case's period. No period keeps it."""
    if policy is None or policy.retain_days is None or retained_at is None:
        return False
    return retained_at + timedelta(days=policy.retain_days) <= at


__all__ = ["RetentionStore"]
