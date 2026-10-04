"""SqliteVecIndex must return a stored embedding, and must not invent a failure.

Production writes the embedding as float32 bytes on the knowledge row
(``index_item``). Nothing writes ``knowledge_vec``. ``item.id`` is not the
vec ``rowid``. The gap note still describes a map keyed by Python ``id()``.

These tests force that identity: a KNN row may carry the stored embedding
under a rowid that is not ``item.id`` and not the rowid SQLite would assign
for a fresh insert. The candidate still comes back as ``(item, 1 - distance)``.
A neighbour outside the candidate set, and a query that returns no rows, are
empty results. ``retrieve`` may still rank an honest empty vector result
lexically. A hit that is the stored embedding does not collapse to that
lexical score.
"""
from __future__ import annotations

import sqlite3

import pytest

from kognita.embedding import HashingEmbedder
from kognita.retrieval import index_item, retrieve
from kognita.vectors import SqliteVecIndex
from kognita.vocabulary import Classification


TITLE = "genomic linkage set"
BODY = "genomic linkage set keys registry"
QUERY = "genomic linkage set"


class _Result:
    def __init__(self, rows: list, lastrowid: int | None = None) -> None:
        self._rows = rows
        self.lastrowid = lastrowid

    def fetchall(self) -> list:
        return list(self._rows)


class _Connection:
    """Records vec writes and returns a scripted KNN result.

    Inserted rowids start at 1000 so they are not a knowledge-item id.
    """

    def __init__(self, knn) -> None:
        self.knn = knn
        self.calls: list[tuple[str, tuple]] = []
        self.rows: list[tuple[int, bytes]] = []
        self.next_rowid = 1000

    def execute(self, sql: str, params: tuple = ()) -> _Result:
        self.calls.append((sql, params))
        compact = " ".join(sql.split())
        if compact.startswith("CREATE"):
            return _Result([])
        if compact.startswith("INSERT"):
            blob = params[0]
            rowid = self.next_rowid
            self.next_rowid += 1
            self.rows.append((rowid, blob))
            return _Result([], lastrowid=rowid)
        if "MATCH" in compact:
            rows = self.knn(self) if callable(self.knn) else self.knn
            return _Result(rows)
        if "embedding = ?" in compact:
            blob = params[0]
            found = [(rowid,) for rowid, stored in self.rows if stored == blob]
            return _Result(found)
        raise AssertionError(compact)


def _index(knn) -> tuple[SqliteVecIndex, _Connection]:
    connection = _Connection(knn)
    index = SqliteVecIndex.__new__(SqliteVecIndex)
    index.connection = connection
    index.table = "knowledge_vec"
    index._rowid_by_blob = {}
    return index, connection


def _seed(session):
    embedder = HashingEmbedder()
    item = index_item(
        session,
        title=TITLE,
        body=BODY,
        embedder=embedder,
        zones=["AE"],
        classification=Classification.C1,
    )
    session.commit()
    return embedder, item


def test_indexed_embedding_is_returned_when_rowid_is_not_item_id(session):
    """The bytes ``index_item`` stored come back as that candidate.

    The vec rowid the connection assigns is not ``item.id``. The score is
    ``1 - distance``.
    """
    _embedder, item = _seed(session)

    def knn(conn: _Connection):
        rowid, blob = conn.rows[-1]
        assert rowid != item.id
        assert blob == item.embedding
        return [(rowid, blob, 0.25)]

    index, conn = _index(knn)
    found = index.search([0.0, 1.0], [(item, item.embedding)])

    assert found == [(item, 0.75)]
    inserts = [call for call in conn.calls if call[0].lstrip().upper().startswith("INSERT")]
    assert len(inserts) == 1
    sql, params = inserts[0]
    assert "rowid" not in sql.lower()
    assert params == (item.embedding,)
    assert conn.rows[0][0] != item.id


def test_knn_rows_outside_the_candidate_set_are_an_empty_result(session):
    """A global top-k that misses the candidates is empty, and does not raise."""
    _embedder, item = _seed(session)

    def knn(conn: _Connection):
        blob = conn.rows[-1][1]
        other = bytes([blob[0] ^ 0xFF]) + blob[1:]
        return [(4242, other, 0.1), (4243, other, 0.2)]

    index, _conn = _index(knn)
    assert index.search([0.0, 1.0], [(item, item.embedding)]) == []


def test_knn_that_returns_no_rows_is_empty(session):
    _embedder, item = _seed(session)
    index, _conn = _index(lambda _conn: [])
    assert index.search([0.0, 1.0], [(item, item.embedding)]) == []


def test_partial_knn_keeps_the_candidate_and_drops_the_rest(session):
    """A nearer neighbour outside the set is dropped. The candidate stays."""
    _embedder, item = _seed(session)

    def knn(conn: _Connection):
        rowid, blob = conn.rows[-1]
        other = bytes([blob[0] ^ 0xFF]) + blob[1:]
        return [(4242, other, 0.0), (rowid, blob, 0.25)]

    index, _conn = _index(knn)
    assert index.search([0.0, 1.0], [(item, item.embedding)], top_k=2) == [(item, 0.75)]


def test_stored_embedding_under_python_id_is_still_that_candidate(session):
    """The gap's key is Python ``id()``. The stored bytes still name the item.

    ``retrieve`` must apply that semantic score. The same text with an empty
    KNN ranks lexically, below this score.
    """
    _embedder, item = _seed(session)

    def knn(_conn: _Connection):
        return [(id(item), item.embedding, 0.0)]

    index, _conn = _index(knn)
    hits = retrieve(session, QUERY, zone="AE", embedder=_embedder, index=index)

    assert [hit.title for hit in hits] == [TITLE]
    assert hits[0].score == 1.0


def test_empty_knn_still_ranks_lexically(session):
    embedder, _item = _seed(session)
    index, _conn = _index(lambda _conn: [])
    hits = retrieve(session, QUERY, zone="AE", embedder=embedder, index=index)

    assert [hit.title for hit in hits] == [TITLE]
    assert hits[0].score == 0.4


def test_neighbours_outside_the_set_still_rank_lexically(session):
    embedder, item = _seed(session)

    def knn(conn: _Connection):
        blob = conn.rows[-1][1]
        other = bytes([blob[0] ^ 0xFF]) + blob[1:]
        return [(4242, other, 0.0)]

    index, _conn = _index(knn)
    hits = retrieve(session, QUERY, zone="AE", embedder=embedder, index=index)

    assert [hit.title for hit in hits] == [TITLE]
    assert hits[0].score == 0.4


def test_no_candidates_does_not_query():
    index, conn = _index(lambda _conn: [(1, b"", 0.0)])
    assert index.search([0.0, 1.0], []) == []
    assert conn.calls == []


def _real_index() -> tuple[SqliteVecIndex, sqlite3.Connection]:
    sqlite_vec = pytest.importorskip("sqlite_vec")
    connection = sqlite3.connect(":memory:")
    try:
        index = SqliteVecIndex(connection)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return index, connection


def test_sqlite_vec_returns_the_stored_embedding_under_its_own_rowid(session):
    """Against the extension: the indexed candidate comes back, rowid != item.id."""
    embedder, item = _seed(session)
    index, connection = _real_index()
    text = f"{TITLE} {BODY}"
    vector = embedder.embed(text)

    spacer = [0.0] * embedder.dimension
    spacer[0] = 1.0
    index.search(spacer, [(object(), spacer)], top_k=1)

    found = index.search(vector, [(item, item.embedding)])
    rows = connection.execute("SELECT rowid, embedding FROM knowledge_vec").fetchall()
    owned = [rowid for rowid, blob in rows if blob == item.embedding]

    assert found[0][0] is item
    assert found[0][1] == pytest.approx(1.0, abs=1e-5)
    assert owned
    assert owned[0] != item.id

    missed = index.search(spacer, [(item, item.embedding)], top_k=1)
    assert missed == []

    assert index.search(vector, [(item, item.embedding)], top_k=0) == []
