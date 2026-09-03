"""Governed retrieval — entitlement first, scoring second.

The ordering is the whole design. Filtering happens in SQL *before* anything is
embedded or scored, so an item outside the caller's zone or above their
classification ceiling is never compared, never ranked, and cannot surface
through a relevance score. Filtering after retrieval would mean the ranking had
already seen it — and a top-k that quietly drops entries is a leak of the fact
that they exist.

Every search emits a ``RETRIEVAL`` evidence event recording what was asked, how
large the entitled candidate set was, and which items came back.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from sqlmodel import Session, select

from kognita.embedding import lexical_overlap
from kognita.evidence import EvidenceWriter
from kognita.models import KnowledgeItem, as_utc
from kognita.protocols import Embedder
from kognita.vectors import NumpyVectorIndex
from kognita.vocabulary import (
    ActorType,
    Classification,
    EventType,
    at_or_below,
)

#: Weighting between semantic similarity and exact-term overlap.
SEMANTIC_WEIGHT = 0.6
LEXICAL_WEIGHT = 0.4
#: Below this, a hit is noise rather than an answer.
MIN_SCORE = 0.08
TOP_K = 5


@dataclass(frozen=True)
class Retrieved:
    """One entitled, cited fragment."""

    id: int
    title: str
    snippet: str
    kind: str
    classification: Classification
    source_label: str
    published_at: datetime | None
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "snippet": self.snippet,
            "kind": self.kind,
            "classification": Classification(self.classification).value,
            "source_label": self.source_label,
            "score": self.score,
        }


def entitled_items(
    session: Session,
    *,
    zone: str,
    ceiling: Classification,
) -> list[KnowledgeItem]:
    """Items this caller may see at all — the candidate set, before any scoring.

    Fail-closed: an item with empty zones is not visible in any zone. An item
    with zones=['SG', 'HK'] is visible only in those zones.
    """
    items = session.exec(select(KnowledgeItem)).all()
    return [
        item
        for item in items
        if item.zones and zone in item.zones
        and at_or_below(item.classification, ceiling)
    ]


def ceiling_for(is_admin: bool) -> Classification:
    """The classification ceiling a principal carries.

    Deliberately coarse in the core: a deployment with real roles supplies its
    own ceiling rather than a boolean.
    """
    return Classification.C3 if is_admin else Classification.C2


def retrieve(
    session: Session,
    query: str,
    *,
    zone: str,
    embedder: Embedder,
    evidence: EvidenceWriter | None = None,
    correlation_id: str = "",
    actor_id: str = "",
    actor_type: ActorType = ActorType.HUMAN,
    is_admin: bool = False,
    ceiling: Classification | None = None,
    index: Any = None,
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
) -> list[Retrieved]:
    """Search the knowledge store within the caller's entitlement."""
    limit = ceiling if ceiling is not None else ceiling_for(is_admin)
    candidates = entitled_items(session, zone=zone, ceiling=limit)

    hits: list[Retrieved] = []
    if candidates:
        vector_index = index or NumpyVectorIndex()
        query_vector = embedder.embed(query)
        scored = vector_index.search(
            query_vector,
            [(item, item.embedding) for item in candidates],
            top_k=len(candidates),
        )
        semantic_by_id = {id(item): score for item, score in scored}

        ranked: list[tuple[KnowledgeItem, float]] = []
        for item in candidates:
            semantic = semantic_by_id.get(id(item), 0.0)
            lexical = lexical_overlap(query, f"{item.title} {item.body}")
            score = SEMANTIC_WEIGHT * semantic + LEXICAL_WEIGHT * lexical
            if score >= min_score:
                ranked.append((item, score))
        ranked.sort(key=lambda pair: (-pair[1], pair[0].id or 0))

        hits = [
            Retrieved(
                id=item.id or 0,
                title=item.title,
                snippet=item.body[:280],
                kind=item.kind,
                classification=Classification(item.classification),
                source_label=item.source_label,
                published_at=as_utc(item.published_at),
                score=round(score, 4),
            )
            for item, score in ranked[:top_k]
        ]

    if evidence is not None:
        evidence.emit(
            session,
            correlation_id=correlation_id,
            event_type=EventType.RETRIEVAL,
            actor_type=actor_type,
            actor_id=actor_id,
            classification=Classification.C1,
            payload={
                "query": query,
                "zone": zone,
                "ceiling": Classification(limit).value,
                "candidate_count": len(candidates),
                "returned_ids": [h.id for h in hits],
                "top_score": hits[0].score if hits else 0.0,
                "embedder": embedder.model,
            },
        )
    return hits


def index_item(
    session: Session,
    *,
    title: str,
    body: str,
    embedder: Embedder,
    kind: str = "DOCUMENT",
    classification: Classification = Classification.C1,
    zones: Sequence[str] = (),
    source_label: str = "",
    published_at: datetime | None = None,
) -> KnowledgeItem:
    """Embed and store one item with the attributes entitlement is decided on."""
    from kognita.embedding import to_bytes
    from kognita.models import utcnow

    vector = embedder.embed(f"{title} {body}")
    item = KnowledgeItem(
        title=title,
        body=body,
        kind=kind,
        classification=classification,
        zones=list(zones),
        source_label=source_label,
        embedding=to_bytes(vector),
        embedding_dim=embedder.dimension,
        embedding_model=embedder.model,
        published_at=published_at or utcnow(),
    )
    session.add(item)
    session.flush()
    return item


def reindex(session: Session, embedder: Embedder) -> int:
    """Re-embed every item, e.g. after switching embedder. Returns how many."""
    from kognita.embedding import to_bytes

    items = session.exec(select(KnowledgeItem)).all()
    for item in items:
        item.embedding = to_bytes(embedder.embed(f"{item.title} {item.body}"))
        item.embedding_dim = embedder.dimension
        item.embedding_model = embedder.model
        session.add(item)
    session.flush()
    return len(items)


__all__ = [
    "Retrieved",
    "retrieve",
    "index_item",
    "reindex",
    "entitled_items",
    "ceiling_for",
    "MIN_SCORE",
    "TOP_K",
]
