"""Nearest-neighbour search over an entitlement-filtered candidate set.

NumPy brute force is the default, deliberately. At the scale a governed corpus
actually reaches — thousands of policy fragments, not millions of web pages — a
matrix multiply is sub-millisecond, and it works everywhere. ``sqlite-vec`` is
available behind the same interface for when that stops being true, but it is
not the load-bearing path: ``enable_load_extension`` is compiled out of many
stock Python builds, so a deployment that depended on it would fail on exactly
the locked-down machines this library is meant for.

Both backends search a candidate list the caller has *already* filtered. That
ordering is the point: an item outside the caller's entitlement is never scored,
so it cannot surface through a relevance ranking.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from kognita.embedding import from_bytes


class NumpyVectorIndex:
    """Brute-force cosine search. Always available."""

    name = "numpy"

    def search(
        self,
        query_vector: Sequence[float],
        candidates: Sequence[tuple[Any, Sequence[float] | bytes | None]],
        *,
        top_k: int = 5,
    ) -> list[tuple[Any, float]]:
        """Score ``candidates`` against ``query_vector``, best first.

        Candidates are ``(item, vector)`` pairs; a candidate with no vector, or
        one of a different dimension, scores zero rather than raising — a corpus
        part-way through a re-index should degrade, not break.
        """
        if not candidates:
            return []

        query = np.asarray(query_vector, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return []
        query = query / query_norm

        items: list[Any] = []
        rows: list[np.ndarray] = []
        for item, vector in candidates:
            if vector is None:
                continue
            array = np.asarray(
                from_bytes(vector) if isinstance(vector, (bytes, bytearray)) else vector,
                dtype=np.float64,
            )
            if array.shape != query.shape:
                continue
            norm = float(np.linalg.norm(array))
            items.append(item)
            rows.append(array / norm if norm else array)

        if not items:
            return []

        scores = np.asarray(rows) @ query
        order = np.argsort(-scores)[: max(0, top_k)]
        return [(items[i], float(scores[i])) for i in order]


class SqliteVecIndex:
    """``sqlite-vec``-backed search, for corpora large enough to need it.

    Constructing this raises if the extension cannot be loaded, so a deployment
    opts in explicitly rather than silently falling back and wondering later why
    a query is slow.
    """

    name = "sqlite-vec"

    def __init__(self, connection: Any, *, table: str = "knowledge_vec") -> None:
        try:
            import sqlite_vec
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise RuntimeError(
                "sqlite-vec is not installed. Install it with: pip install kognita[vec]"
            ) from exc
        try:
            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
            connection.enable_load_extension(False)
        except Exception as exc:  # pragma: no cover - depends on the build
            raise RuntimeError(
                "This Python's sqlite3 cannot load extensions, so sqlite-vec is "
                "unavailable. Use NumpyVectorIndex (the default)."
            ) from exc
        self.connection = connection
        self.table = table

    def search(
        self,
        query_vector: Sequence[float],
        candidates: Sequence[tuple[Any, Sequence[float] | bytes | None]],
        *,
        top_k: int = 5,
    ) -> list[tuple[Any, float]]:
        # The entitlement filter has already reduced the candidate set, so the
        # KNN runs over ids the caller is permitted to see and nothing else.
        import struct

        by_id = {item.id: item for item, _ in candidates if hasattr(item, "id")}
        if not by_id:
            return []
        packed = struct.pack(f"{len(query_vector)}f", *query_vector)
        rows = self.connection.execute(
            f"SELECT rowid, distance FROM {self.table} "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (packed, top_k),
        ).fetchall()
        results: list[tuple[Any, float]] = []
        for rowid, distance in rows:
            item = by_id.get(rowid)
            if item is not None:
                results.append((item, 1.0 - float(distance)))
        return results


def default_index() -> NumpyVectorIndex:
    """The vector index used unless a deployment chooses otherwise."""
    return NumpyVectorIndex()


__all__ = ["NumpyVectorIndex", "SqliteVecIndex", "default_index"]
