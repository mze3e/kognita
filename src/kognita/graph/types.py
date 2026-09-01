"""Lightweight dataclasses mirroring the shapes surfaced by Graphiti."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Node:
    """A graph entity extracted from ingested text."""

    uuid: str
    name: str
    summary: str = ""
    labels: list[str] = field(default_factory=list)

    @classmethod
    def from_graphiti(cls, node: Any) -> "Node":
        return cls(
            uuid=getattr(node, "uuid", ""),
            name=getattr(node, "name", "") or "",
            summary=getattr(node, "summary", "") or "",
            labels=list(getattr(node, "labels", None) or []),
        )


@dataclass(frozen=True)
class Edge:
    """A relationship between two entities."""

    uuid: str
    source_uuid: str
    target_uuid: str
    fact: str = ""
    name: str = ""

    @classmethod
    def from_graphiti(cls, edge: Any) -> "Edge":
        return cls(
            uuid=getattr(edge, "uuid", ""),
            source_uuid=getattr(edge, "source_node_uuid", ""),
            target_uuid=getattr(edge, "target_node_uuid", ""),
            fact=getattr(edge, "fact", "") or "",
            name=getattr(edge, "name", "") or "",
        )


@dataclass(frozen=True)
class SearchResult:
    """A single hit returned by Kognita.search()."""

    fact: str
    source_node: Node | None = None
    target_node: Node | None = None
    score: float | None = None
    raw: Any = None


@dataclass(frozen=True)
class EpisodeResult:
    """Summary of a single ingested chunk."""

    chunk_index: int
    preview: str
    node_count: int
    edge_count: int
    error: str | None = None
