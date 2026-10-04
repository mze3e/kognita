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

    ``index_item`` and ``reindex`` store the embedding bytes and, when given
    this index, write them here. ``upsert`` deletes that item's previous vec
    row before inserting the new one. SQLite assigns the vec ``rowid``. The
    knowledge item id is stored beside the vector as ``knowledge_id``; it is
    not the rowid.

    ``search`` only reads. The KNN is unfiltered. A neighbour whose
    ``knowledge_id`` is not in the candidate set is dropped. No rows, or only
    outside neighbours, yields ``[]``. A hit is ``(item, 1 - distance)`` for
    the candidate with that ``knowledge_id``, so two items that share embedding
    bytes stay distinct.
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
        self._dimension: int | None = None

    def _table_exists(self) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?",
            (self.table,),
        ).fetchone()
        return row is not None

    def vector_dimension(self) -> int | None:
        """Dimension of the vec table, if it exists and holds a row."""
        if self._dimension is not None:
            return self._dimension
        if not self._table_exists():
            return None
        row = self.connection.execute(
            f"SELECT embedding FROM {self.table} LIMIT 1"
        ).fetchone()
        if not row or row[0] is None:
            return None
        return len(row[0]) // 4

    def reset(self) -> None:
        """Drop the vec table. The next ``upsert`` creates it again."""
        self.connection.execute(f"DROP TABLE IF EXISTS {self.table}")
        self._dimension = None

    def _ensure_table(self, dimension: int) -> None:
        if self._table_exists():
            return
        self.connection.execute(
            f"CREATE VIRTUAL TABLE {self.table} USING vec0("
            f"embedding float[{int(dimension)}], +knowledge_id integer)"
        )
        self._dimension = dimension

    def upsert(self, item_id: int, embedding: bytes) -> None:
        """Replace the vec row for ``item_id`` with ``embedding``.

        Any previous vector for this item is deleted first, so it cannot
        remain in a later KNN. The new ``rowid`` is assigned by SQLite.
        """
        blob = bytes(embedding)
        if len(blob) == 0 or len(blob) % 4 != 0:
            raise RuntimeError("stored embedding must be float32 bytes")
        self._ensure_table(len(blob) // 4)
        self.connection.execute(
            f"DELETE FROM {self.table} WHERE knowledge_id = ?",
            (int(item_id),),
        )
        self.connection.execute(
            f"INSERT INTO {self.table}(embedding, knowledge_id) VALUES (?, ?)",
            (blob, int(item_id)),
        )

    def search(
        self,
        query_vector: Sequence[float],
        candidates: Sequence[tuple[Any, Sequence[float] | bytes | None]],
        *,
        top_k: int = 5,
    ) -> list[tuple[Any, float]]:
        """Return candidate hits, nearest first. This method does not write.

        ``[]`` means the KNN returned no rows, or every returned neighbour
        is outside this candidate set. A returned ``knowledge_id`` is the
        candidate with that item id, scored ``1 - distance``.
        """
        import struct

        if not candidates or not self._table_exists():
            return []

        by_id: dict[int, Any] = {}
        for item, _vector in candidates:
            item_id = getattr(item, "id", None)
            if item_id is None:
                continue
            by_id.setdefault(int(item_id), item)
        if not by_id:
            return []

        packed = struct.pack(f"{len(query_vector)}f", *query_vector)
        rows = self.connection.execute(
            f"SELECT knowledge_id, distance FROM {self.table} "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (packed, max(0, top_k)),
        ).fetchall()
        if not rows:
            return []

        results: list[tuple[Any, float]] = []
        seen: set[int] = set()
        for knowledge_id, distance in rows:
            item_id = int(knowledge_id)
            item = by_id.get(item_id)
            if item is None or item_id in seen:
                continue
            seen.add(item_id)
            results.append((item, 1.0 - float(distance)))
        return results


def default_index() -> NumpyVectorIndex:
    """The vector index used unless a deployment chooses otherwise."""
    return NumpyVectorIndex()


__all__ = ["NumpyVectorIndex", "SqliteVecIndex", "default_index"]
