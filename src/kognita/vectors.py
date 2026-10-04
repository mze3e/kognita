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


def _stored_embedding(vector: Sequence[float] | bytes | None) -> bytes | None:
    """Embedding bytes in the form ``index_item`` stores on the row.

    A ``bytes`` value is kept as stored. A sequence is packed as float32,
    which is what ``to_bytes`` writes.
    """
    if vector is None:
        return None
    if isinstance(vector, (bytes, bytearray)):
        return bytes(vector) or None
    array = np.asarray(vector, dtype=np.float32)
    if array.size == 0:
        return None
    return np.ascontiguousarray(array).tobytes()


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

    Production stores an embedding as float32 bytes on the knowledge row.
    Nothing else writes ``knowledge_vec``. ``search`` copies those bytes into
    the vec table so the KNN can see them. SQLite assigns the vec ``rowid``.
    That rowid is not ``item.id``. A hit joins back to the candidate by the
    stored embedding bytes, or by the rowid this index wrote for those bytes.

    The KNN itself is unfiltered. A neighbour that is not in the candidate
    set is dropped. If none of the returned rows are candidates, or the query
    returns no rows, ``search`` returns ``[]``. A candidate that was stored
    and comes back is still ``(item, 1 - distance)``.
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
        # Blob to the vec rowid SQLite assigned. Not item.id.
        self._rowid_by_blob: dict[bytes, int] = {}

    def _ensure_table(self, dimension: int) -> None:
        self.connection.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} "
            f"USING vec0(embedding float[{dimension}])"
        )

    def _store_embedding(self, blob: bytes) -> int:
        """Write one stored embedding, if this table does not already hold it.

        The rowid comes from SQLite, or from a row already stored under those
        bytes. Callers do not pass ``item.id``.
        """
        if not hasattr(self, "_rowid_by_blob"):
            self._rowid_by_blob = {}
        known = self._rowid_by_blob.get(blob)
        if known is not None:
            return known
        found = self.connection.execute(
            f"SELECT rowid FROM {self.table} WHERE embedding = ?",
            (blob,),
        ).fetchall()
        if found:
            rowid = int(found[0][0])
            self._rowid_by_blob[blob] = rowid
            return rowid
        cursor = self.connection.execute(
            f"INSERT INTO {self.table}(embedding) VALUES (?)",
            (blob,),
        )
        rowid = cursor.lastrowid
        if rowid is None:
            raise RuntimeError(
                "sqlite-vec did not assign a rowid for a stored embedding. "
                "This search failed; it is not an empty result."
            )
        rowid = int(rowid)
        self._rowid_by_blob[blob] = rowid
        return rowid

    def search(
        self,
        query_vector: Sequence[float],
        candidates: Sequence[tuple[Any, Sequence[float] | bytes | None]],
        *,
        top_k: int = 5,
    ) -> list[tuple[Any, float]]:
        """Return candidate hits, nearest first.

        ``[]`` means the KNN returned no rows, or every returned neighbour
        sits outside this candidate set. Both are empty results. A stored
        embedding that the KNN does return comes back as that candidate,
        with score ``1 - distance``.
        """
        import struct

        if not candidates:
            return []

        stored: list[tuple[Any, bytes]] = []
        for item, vector in candidates:
            blob = _stored_embedding(vector)
            if blob is None or len(blob) % 4 != 0:
                continue
            stored.append((item, blob))
        if not stored:
            return []

        dimension = len(stored[0][1]) // 4
        self._ensure_table(dimension)
        rowid_to_item: dict[int, Any] = {}
        blob_to_item: dict[bytes, Any] = {}
        for item, blob in stored:
            if len(blob) != dimension * 4:
                continue
            rowid = self._store_embedding(blob)
            rowid_to_item.setdefault(rowid, item)
            blob_to_item.setdefault(blob, item)
        if not rowid_to_item:
            return []

        packed = struct.pack(f"{len(query_vector)}f", *query_vector)
        rows = self.connection.execute(
            f"SELECT rowid, embedding, distance FROM {self.table} "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (packed, top_k),
        ).fetchall()
        if not rows:
            return []

        results: list[tuple[Any, float]] = []
        seen: set[int] = set()
        for rowid, embedding, distance in rows:
            blob = bytes(embedding) if embedding is not None else b""
            item = rowid_to_item.get(rowid)
            if item is None:
                item = blob_to_item.get(blob)
            if item is None:
                continue
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            results.append((item, 1.0 - float(distance)))
        return results


def default_index() -> NumpyVectorIndex:
    """The vector index used unless a deployment chooses otherwise."""
    return NumpyVectorIndex()


__all__ = ["NumpyVectorIndex", "SqliteVecIndex", "default_index"]
