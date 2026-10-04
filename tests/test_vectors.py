"""SqliteVecIndex on a real sqlite-vec connection.

``index_item`` and ``reindex`` write ``knowledge_vec``. ``search`` only reads.
The vec ``rowid`` is not ``item.id``. A hit is the candidate that was indexed,
including when another candidate has the same embedding bytes. No KNN rows,
and neighbours outside the candidate set, are empty results. An empty vector
result may still rank lexically.
"""
from __future__ import annotations

import sqlite3

import pytest
import sqlite_vec

from kognita.embedding import HashingEmbedder
from kognita.retrieval import index_item, reindex, retrieve
from kognita.vectors import SqliteVecIndex
from kognita.vocabulary import Classification


TITLE = "genomic linkage set"
BODY = "genomic linkage set keys registry"
QUERY = "genomic linkage set"


class _Vec:
    """Embedder with caller-chosen vectors. ``index_item`` stores ``to_bytes``."""

    def __init__(self) -> None:
        self.dimension = 4
        self.model = "vec-test"
        self.vectors: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        return list(self.vectors[text])


def _connection() -> tuple[SqliteVecIndex, sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
    except Exception as exc:  # pragma: no cover - the suite requires the extension
        raise RuntimeError(
            "sqlite-vec must load for these tests. Install the vec extra."
        ) from exc
    return SqliteVecIndex(connection), connection


def _row_count(connection: sqlite3.Connection) -> int:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'knowledge_vec'"
    ).fetchone()
    if exists is None:
        return 0
    return int(connection.execute("SELECT count(*) FROM knowledge_vec").fetchone()[0])


def _vec_rows(connection: sqlite3.Connection) -> list[tuple[int, int]]:
    return list(
        connection.execute("SELECT rowid, knowledge_id FROM knowledge_vec").fetchall()
    )


def test_indexed_item_is_returned_when_rowid_is_not_item_id(session):
    """``index_item`` writes the embedding. After replace, rowid != item.id.

    ``search`` reads that row and returns ``(item, 1 - distance)``. It does
    not insert.
    """
    index, connection = _connection()
    embedder = _Vec()
    embedder.vectors["alpha one"] = [1.0, 0.0, 0.0, 0.0]
    item = index_item(
        session,
        title="alpha",
        body="one",
        embedder=embedder,
        zones=["AE"],
        classification=Classification.C1,
        index=index,
    )
    session.commit()
    # Replace the row so SQLite assigns a new rowid and the old one is gone.
    reindex(session, embedder, index=index)
    session.commit()

    rows = _vec_rows(connection)
    assert len(rows) == 1
    rowid, knowledge_id = rows[0]
    assert knowledge_id == item.id
    assert rowid != item.id

    statements: list[str] = []
    before = _row_count(connection)
    connection.set_trace_callback(statements.append)
    found = index.search(
        [1.0, 0.0, 0.0, 0.0],
        [(item, item.embedding)],
    )
    connection.set_trace_callback(None)

    assert found[0][0] is item
    assert found[0][1] == pytest.approx(1.0)
    assert _row_count(connection) == before
    assert not any(statement.lstrip().upper().startswith("INSERT") for statement in statements)
    assert not any(statement.lstrip().upper().startswith("DELETE") for statement in statements)


def test_same_embedding_bytes_keep_each_candidate(session):
    """Two indexed items with the same bytes are both returned as themselves."""
    index, _vec = _connection()
    embedder = _Vec()
    embedder.vectors["alpha one"] = [1.0, 0.0, 0.0, 0.0]
    embedder.vectors["beta two"] = [1.0, 0.0, 0.0, 0.0]
    first = index_item(
        session,
        title="alpha",
        body="one",
        embedder=embedder,
        zones=["AE"],
        index=index,
    )
    second = index_item(
        session,
        title="beta",
        body="two",
        embedder=embedder,
        zones=["AE"],
        index=index,
    )
    session.commit()
    assert first.embedding == second.embedding
    assert first.id != second.id

    found = index.search(
        [1.0, 0.0, 0.0, 0.0],
        [(first, first.embedding), (second, second.embedding)],
        top_k=2,
    )

    assert {id(item) for item, _score in found} == {id(first), id(second)}
    assert [score for _item, score in found] == pytest.approx([1.0, 1.0])


def test_reindex_drops_the_previous_vector(session):
    """The old vector must not remain a neighbour that can fill top-k."""
    index, connection = _connection()
    embedder = _Vec()
    embedder.vectors["alpha one"] = [1.0, 0.0, 0.0, 0.0]
    embedder.vectors["beta two"] = [0.0, 1.0, 0.0, 0.0]
    alpha = index_item(
        session,
        title="alpha",
        body="one",
        embedder=embedder,
        zones=["AE"],
        index=index,
    )
    beta = index_item(
        session,
        title="beta",
        body="two",
        embedder=embedder,
        zones=["AE"],
        index=index,
    )
    embedder.vectors["alpha one"] = [0.0, 0.0, 1.0, 0.0]
    reindex(session, embedder, index=index)
    session.commit()

    assert _row_count(connection) == 2
    old = index.search(
        [1.0, 0.0, 0.0, 0.0],
        [(alpha, alpha.embedding), (beta, beta.embedding)],
        top_k=2,
    )
    assert old
    assert all(score < 0.99 for _item, score in old)

    live = index.search([0.0, 0.0, 1.0, 0.0], [(alpha, alpha.embedding)], top_k=2)
    assert live[0][0] is alpha
    assert live[0][1] == pytest.approx(1.0)


def test_knn_with_no_rows_returns_empty(session):
    index, _vec = _connection()
    embedder = _Vec()
    embedder.vectors["alpha one"] = [1.0, 0.0, 0.0, 0.0]
    item = index_item(
        session,
        title="alpha",
        body="one",
        embedder=embedder,
        zones=["AE"],
    )
    session.commit()
    fresh, _vec = _connection()
    assert fresh.search([1.0, 0.0, 0.0, 0.0], [(item, item.embedding)]) == []

    embedder.vectors["beta two"] = [0.0, 1.0, 0.0, 0.0]
    index, _vec = _connection()
    index_item(
        session,
        title="beta",
        body="two",
        embedder=embedder,
        zones=["AE"],
        index=index,
    )
    session.commit()
    assert index.search([1.0, 0.0, 0.0, 0.0], [(item, item.embedding)], top_k=0) == []


def test_partial_knn_keeps_the_candidate(session):
    """A nearer neighbour outside the set is dropped. The candidate stays."""
    index, _vec = _connection()
    embedder = _Vec()
    embedder.vectors["alpha one"] = [1.0, 0.0, 0.0, 0.0]
    embedder.vectors["beta two"] = [0.0, 1.0, 0.0, 0.0]
    alpha = index_item(
        session,
        title="alpha",
        body="one",
        embedder=embedder,
        zones=["AE"],
        index=index,
    )
    index_item(
        session,
        title="beta",
        body="two",
        embedder=embedder,
        zones=["AE"],
        index=index,
    )
    session.commit()

    found = index.search([0.0, 1.0, 0.0, 0.0], [(alpha, alpha.embedding)], top_k=2)

    assert [item for item, _score in found] == [alpha]
    assert found[0][1] < 0.99


def test_no_candidates_does_not_write():
    index, connection = _connection()
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    assert index.search([1.0, 0.0, 0.0, 0.0], []) == []
    connection.set_trace_callback(None)
    assert statements == []
    assert _row_count(connection) == 0


def test_neighbours_outside_the_candidate_set_return_empty(session):
    index, _vec = _connection()
    embedder = _Vec()
    embedder.vectors["alpha one"] = [1.0, 0.0, 0.0, 0.0]
    embedder.vectors["beta two"] = [0.0, 1.0, 0.0, 0.0]
    alpha = index_item(
        session,
        title="alpha",
        body="one",
        embedder=embedder,
        zones=["AE"],
        index=index,
    )
    index_item(
        session,
        title="beta",
        body="two",
        embedder=embedder,
        zones=["AE"],
        index=index,
    )
    session.commit()

    missed = index.search([0.0, 1.0, 0.0, 0.0], [(alpha, alpha.embedding)], top_k=1)
    assert missed == []


def test_empty_vector_result_still_ranks_lexically(session):
    embedder = HashingEmbedder()
    index_item(
        session,
        title=TITLE,
        body=BODY,
        embedder=embedder,
        zones=["AE"],
        classification=Classification.C1,
    )
    session.commit()
    index, _vec = _connection()

    hits = retrieve(session, QUERY, zone="AE", embedder=embedder, index=index)

    assert [hit.title for hit in hits] == [TITLE]
    assert hits[0].score == 0.4


def test_outside_neighbour_still_ranks_lexically(session):
    """A nearer vec row outside the entitled set is an empty vector result.

    ``retrieve`` asks for ``top_k`` equal to the candidate count. That neighbour
    can fill the only slot. The entitled item still ranks on lexical overlap.
    """
    embedder = _Vec()
    embedder.vectors[QUERY] = [1.0, 0.0, 0.0, 0.0]
    embedder.vectors[f"{TITLE} {BODY}"] = [0.0, 1.0, 0.0, 0.0]
    embedder.vectors["other item"] = [1.0, 0.0, 0.0, 0.0]
    index, _vec = _connection()
    index_item(
        session,
        title=TITLE,
        body=BODY,
        embedder=embedder,
        zones=["AE"],
        classification=Classification.C1,
        index=index,
    )
    index_item(
        session,
        title="other",
        body="item",
        embedder=embedder,
        zones=["XX"],
        index=index,
    )
    session.commit()

    hits = retrieve(session, QUERY, zone="AE", embedder=embedder, index=index)

    assert [hit.title for hit in hits] == [TITLE]
    assert hits[0].score == 0.4
