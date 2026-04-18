"""The main :class:`Kognita` class — ingest text, search, export, persist."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from graphiti_core.nodes import EpisodeType

from kognita.chunking import chunk_text
from kognita.config import KognitaConfig
from kognita.exceptions import ProviderError, extract_api_error
from kognita.graph import make_graphiti
from kognita.query import execute_cypher
from kognita.storage import GraphSnapshot, load_snapshot, save_snapshot
from kognita.types import Edge, EpisodeResult, Node, SearchResult

ProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[str], None]


class Kognita:
    """High-level API for turning text into a queryable knowledge graph.

    A :class:`Kognita` wraps a single Graphiti instance backed by a Kuzu
    database. Use it as an async context manager so the underlying driver is
    always closed::

        async with Kognita(config) as kg:
            await kg.ingest_text("...", source="doc1")
            hits = await kg.search("...")
    """

    def __init__(self, config: KognitaConfig) -> None:
        self.config = config
        self._graphiti: Any = None
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._episodes: list[dict[str, Any]] = []

    async def __aenter__(self) -> "Kognita":
        await self._ensure_graphiti()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # -- lifecycle ----------------------------------------------------------

    async def _ensure_graphiti(self) -> Any:
        if self._graphiti is None:
            self._graphiti = make_graphiti(
                self.config.llm,
                self.config.embedder,
                str(self.config.db_path),
            )
            await self._graphiti.build_indices_and_constraints()
        return self._graphiti

    async def close(self) -> None:
        """Close the underlying Graphiti / Kuzu driver."""
        if self._graphiti is not None:
            try:
                await self._graphiti.close()
            finally:
                self._graphiti = None

    # -- ingestion ----------------------------------------------------------

    async def ingest_text(
        self,
        text: str,
        source: str,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        on_progress: ProgressCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> list[EpisodeResult]:
        """Chunk ``text`` and feed each chunk through Graphiti as an episode.

        Raises :class:`ProviderError` on the first LLM/embedder API failure;
        chunks already processed remain in the graph and snapshot state.
        """
        size = chunk_size if chunk_size is not None else self.config.chunk_size_words
        overlap = (
            chunk_overlap if chunk_overlap is not None else self.config.chunk_overlap_words
        )
        chunks = chunk_text(text, size=size, overlap=overlap)
        return await self.ingest_chunks(
            chunks,
            source,
            on_progress=on_progress,
            on_status=on_status,
        )

    async def ingest_chunks(
        self,
        chunks: list[str],
        source: str,
        *,
        on_progress: ProgressCallback | None = None,
        on_status: StatusCallback | None = None,
    ) -> list[EpisodeResult]:
        """Ingest already-chunked text.

        Lower-level than :meth:`ingest_text`; use this when you want to control
        chunking yourself (e.g. respecting document structure).
        """
        graphiti = await self._ensure_graphiti()
        total = len(chunks)
        results: list[EpisodeResult] = []

        for idx, chunk in enumerate(chunks):
            label = f"Chunk {idx + 1}/{total}"
            if on_status:
                on_status(f"Processing {label}")
            try:
                episode = await graphiti.add_episode(
                    name=f"{source}__chunk_{idx + 1:04d}",
                    episode_body=chunk,
                    source=EpisodeType.text,
                    source_description=f"source: {source}",
                    reference_time=datetime.now(),
                )
            except Exception as exc:
                message = extract_api_error(exc)
                preview = chunk[:130].replace("\n", " ") + "..."
                results.append(
                    EpisodeResult(
                        chunk_index=idx + 1,
                        preview=preview,
                        node_count=0,
                        edge_count=0,
                        error=message,
                    )
                )
                self._episodes.append(
                    {
                        "chunk": idx + 1,
                        "source": source,
                        "preview": preview,
                        "error": message,
                        "nodes": 0,
                        "edges": 0,
                    }
                )
                if on_status:
                    on_status(f"Stopped at chunk {idx + 1}/{total}: {message}")
                raise ProviderError(message) from exc

            for node in episode.nodes:
                self._nodes[node.uuid] = Node.from_graphiti(node)
            for edge in episode.edges:
                self._edges.append(Edge.from_graphiti(edge))

            preview = chunk[:130].replace("\n", " ") + "..."
            self._episodes.append(
                {
                    "chunk": idx + 1,
                    "source": source,
                    "preview": preview,
                    "nodes": len(episode.nodes),
                    "edges": len(episode.edges),
                }
            )
            results.append(
                EpisodeResult(
                    chunk_index=idx + 1,
                    preview=preview,
                    node_count=len(episode.nodes),
                    edge_count=len(episode.edges),
                )
            )
            if on_progress:
                on_progress(idx + 1, total)

        return results

    # -- retrieval ----------------------------------------------------------

    async def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Semantic search over the graph.

        Returns lightweight :class:`SearchResult` dataclasses. The original
        Graphiti result object is preserved on ``SearchResult.raw`` for callers
        that need richer fields.
        """
        graphiti = await self._ensure_graphiti()
        try:
            raw_results = await graphiti.search(query)
        except Exception as exc:
            raise ProviderError(extract_api_error(exc)) from exc

        hits: list[SearchResult] = []
        for item in raw_results[:limit]:
            source_uuid = getattr(item, "source_node_uuid", "")
            target_uuid = getattr(item, "target_node_uuid", "")
            hits.append(
                SearchResult(
                    fact=getattr(item, "fact", "") or "",
                    source_node=self._nodes.get(source_uuid),
                    target_node=self._nodes.get(target_uuid),
                    score=getattr(item, "score", None),
                    raw=item,
                )
            )
        return hits

    def query_cypher(self, cypher: str, *, allow_writes: bool = False) -> list[dict]:
        """Read-only Cypher passthrough against the Kuzu database.

        Synchronous because Kuzu's Python driver is synchronous.
        """
        return execute_cypher(
            self.config.db_path, cypher, allow_writes=allow_writes
        )

    # -- export / persist ---------------------------------------------------

    def snapshot(self) -> GraphSnapshot:
        """Build an in-memory :class:`GraphSnapshot` of the current graph state."""
        return GraphSnapshot(
            nodes=dict(self._nodes),
            edges=list(self._edges),
            episodes=list(self._episodes),
            metadata={
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "embedder_provider": self.config.embedder.provider,
                "embedder_model": self.config.embedder.model,
                "db_path": str(self.config.db_path),
            },
        )

    def export(self) -> dict[str, Any]:
        """Return the current graph state as a JSON-serializable dict."""
        return self.snapshot().to_dict()

    async def save(
        self,
        path: str | Path,
        *,
        include_kuzu_db: bool = True,
    ) -> str:
        """Persist the graph to disk.

        Writes ``snapshot.json`` + ``metadata.json`` under ``path`` and, when
        ``include_kuzu_db`` is true, copies the underlying Kuzu database folder
        alongside them.
        """
        kuzu_src = self.config.db_path if include_kuzu_db else None
        return save_snapshot(self.snapshot(), path, include_kuzu_db=kuzu_src)

    @classmethod
    def from_snapshot(
        cls, snapshot: GraphSnapshot, config: KognitaConfig
    ) -> "Kognita":
        """Re-hydrate a :class:`Kognita` from an in-memory snapshot.

        Only nodes/edges/episodes are restored — the underlying Kuzu store is
        determined by ``config.db_path`` and will be loaded lazily on the next
        async call.
        """
        kg = cls(config)
        kg._nodes = dict(snapshot.nodes)
        kg._edges = list(snapshot.edges)
        kg._episodes = list(snapshot.episodes)
        return kg

    @classmethod
    def load(cls, path: str | Path, config: KognitaConfig) -> "Kognita":
        """Load a saved graph from ``path`` and return a configured Kognita.

        The returned instance shares no state with the original; it's your
        responsibility to point ``config.db_path`` at the copied Kuzu folder if
        you want search to work against the persisted embeddings.
        """
        snapshot = load_snapshot(path)
        return cls.from_snapshot(snapshot, config)
