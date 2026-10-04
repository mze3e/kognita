"""SqliteVecIndex must not report a failed search as an empty one.

The gap note says ``search`` keys candidates by Python ``id()`` and looks them
up by SQLite ``rowid``, so a hit can never join and the method returns ``[]``.
The code keys by ``item.id``. What is still silent, and what these tests force,
is that same report: the index returned rows, none of those rowids are
candidate ids, and ``search`` used to return ``[]``.
"""
from __future__ import annotations

import struct

import pytest

from kognita.vectors import SqliteVecIndex, VectorSearchError


class _Candidate:
    def __init__(self, id: int) -> None:
        self.id = id


class _NoId:
    pass


class _VecQuery:
    def __init__(self, rows: list[tuple[int, float]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[int, float]]:
        return list(self._rows)


class _Connection:
    """Stand-in for the sqlite-vec connection ``search`` queries.

    Constructing ``SqliteVecIndex`` loads the optional extension. The failure
    under test happens after that query returns, so the connection only has to
    hand back rowids and distances.
    """

    def __init__(self, rows: list[tuple[int, float]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple) -> _VecQuery:
        self.calls.append((sql, params))
        return _VecQuery(self.rows)


def _index(rows: list[tuple[int, float]]) -> tuple[SqliteVecIndex, _Connection]:
    connection = _Connection(rows)
    index = SqliteVecIndex.__new__(SqliteVecIndex)
    index.connection = connection
    index.table = "knowledge_vec"
    return index, connection


def test_unjoined_rowids_are_not_an_empty_success():
    """A KNN hit that joins no candidate is a failed search, not ``[]``.

    The rowid is the candidate's Python ``id()``, the key the gap analysis
    names. That value is not ``item.id``, so the join misses. The index did
    run: it returned a row.
    """
    item = _Candidate(1)
    index, connection = _index([(id(item), 0.25)])

    with pytest.raises(VectorSearchError, match="not an empty result") as caught:
        result = index.search([1.0, 0.0], [(item, [1.0, 0.0])])
        assert result != [], "a failed search must not be reported as an empty success"

    assert caught.value.__class__ is VectorSearchError
    assert connection.calls, "the index must run before a missed join is a failure"
    sql, params = connection.calls[0]
    assert sql == (
        "SELECT rowid, distance FROM knowledge_vec "
        "WHERE embedding MATCH ? ORDER BY distance LIMIT ?"
    )
    assert params[0] == struct.pack("2f", 1.0, 0.0)
    assert params[1] == 5


def test_a_search_that_matches_nothing_stays_empty():
    """No KNN rows means the index ran and matched nothing."""
    item = _Candidate(1)
    index, connection = _index([])

    assert index.search([1.0, 0.0], [(item, [1.0, 0.0])]) == []
    assert connection.calls


def test_no_candidates_stays_empty_without_querying():
    index, connection = _index([(1, 0.0)])

    assert index.search([1.0, 0.0], []) == []
    assert connection.calls == []


def test_candidates_that_cannot_be_keyed_are_not_an_empty_success():
    index, connection = _index([(1, 0.0)])

    with pytest.raises(VectorSearchError, match="not an empty result"):
        result = index.search([1.0, 0.0], [(_NoId(), [1.0, 0.0])])
        assert result != [], "a failed search must not be reported as an empty success"

    assert connection.calls == []


def test_joined_hits_keep_the_existing_score():
    """A rowid equal to ``item.id`` still returns ``(item, 1 - distance)``.

    A neighbour row whose rowid is outside the candidate set is still dropped.
    The pairs that do join are unchanged.
    """
    first = _Candidate(7)
    second = _Candidate(8)
    index, connection = _index([(8, 0.25), (7, 0.5), (999, 0.0)])

    assert index.search(
        [1.0, 0.0],
        [(first, [1.0, 0.0]), (second, [0.0, 1.0])],
        top_k=3,
    ) == [
        (second, 0.75),
        (first, 0.5),
    ]
    assert connection.calls[0][1][1] == 3


def test_retrieve_does_not_hide_a_sqlite_vec_failure_behind_lexical_hits(session):
    """The gap's visible failure: ``[]`` from search became a lexical ranking.

    The indexed text shares every query token, so a semantic miss would still
    clear the score floor. A missed rowid join must not come back as those hits.
    """
    from kognita.embedding import HashingEmbedder
    from kognita.retrieval import index_item, retrieve
    from kognita.vocabulary import Classification

    embedder = HashingEmbedder()
    item = index_item(
        session,
        title="genomic linkage set",
        body="genomic linkage set keys registry",
        embedder=embedder,
        zones=["AE"],
        classification=Classification.C1,
    )
    session.commit()
    index, _connection = _index([(id(item), 0.1)])

    with pytest.raises(VectorSearchError, match="not an empty result"):
        result = retrieve(
            session,
            "genomic linkage set",
            zone="AE",
            embedder=embedder,
            index=index,
        )
        assert result != [], "a failed search must not be reported as an empty success"


def test_retrieve_keeps_lexical_hits_when_sqlite_vec_matches_nothing(session):
    """An honest empty KNN result still leaves lexical ranking alone."""
    from kognita.embedding import HashingEmbedder
    from kognita.retrieval import index_item, retrieve
    from kognita.vocabulary import Classification

    embedder = HashingEmbedder()
    index_item(
        session,
        title="genomic linkage set",
        body="genomic linkage set keys registry",
        embedder=embedder,
        zones=["AE"],
        classification=Classification.C1,
    )
    session.commit()
    index, _connection = _index([])

    hits = retrieve(
        session,
        "genomic linkage set",
        zone="AE",
        embedder=embedder,
        index=index,
    )

    assert [hit.title for hit in hits] == ["genomic linkage set"]
