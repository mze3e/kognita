"""Zone entitlement fails closed.

A missing zone list is not visibility in every zone. A caller with no zone
does not match a zone an item lists. A classification or ceiling the filter
cannot evaluate does not admit the item. A zone the item already lists, at
or below the ceiling, still does. The names are the demo pack's sites and
the classifications already in the vocabulary. This test adds none.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from fixtures import demo_pack as dp

from kognita.embedding import HashingEmbedder, to_bytes
from kognita.models import KnowledgeItem
from kognita.retrieval import entitled_items, index_item, retrieve
from kognita.vocabulary import Classification, at_or_below


def _source(zones: list[str], classification: str) -> dict:
    return next(
        item
        for item in dp.KNOWLEDGE
        if item["zones"] == zones and item["classification"] == classification
    )


def _index(session, source: dict, *, zones, embedder: HashingEmbedder) -> KnowledgeItem:
    return index_item(
        session,
        title=source["title"],
        body=source["body"],
        embedder=embedder,
        kind=source["kind"],
        classification=Classification(source["classification"]),
        zones=zones,
        source_label=source["source_label"],
    )


def _ids(session, zone: str, ceiling: Classification) -> set[int]:
    return {item.id for item in entitled_items(session, zone=zone, ceiling=ceiling)}


def _hits(session, source: dict, zone: str, embedder: HashingEmbedder, **kwargs):
    return retrieve(
        session,
        source["title"],
        zone=zone,
        embedder=embedder,
        min_score=-1.0,
        top_k=10,
        **kwargs,
    )


def test_empty_zones_are_not_visible_in_any_zone(session):
    """An item with an empty zone list is absent at every site and ceiling.

    The same text, stored with the zones it already lists, is still returned
    where that list and the ceiling allow it.
    """
    embedder = HashingEmbedder()
    source = _source(["SG", "HK"], "C1")
    missing = _index(session, source, zones=(), embedder=embedder)
    granted = _index(session, source, zones=list(source["zones"]), embedder=embedder)
    session.commit()

    for zone in dp.SITES:
        for ceiling in Classification:
            found = _ids(session, zone, ceiling)
            assert missing.id not in found
            if zone in source["zones"] and at_or_below(source["classification"], ceiling):
                assert granted.id in found
            else:
                assert granted.id not in found

        hits = _hits(session, source, zone, embedder, ceiling=Classification.C3)
        assert all(hit.id != missing.id for hit in hits)
        if zone in source["zones"]:
            assert any(hit.id == granted.id for hit in hits)
        else:
            assert all(hit.id != granted.id for hit in hits)


def test_null_zones_are_not_visible(session):
    """A zone list that was never set is the same closed failure as an empty one."""
    embedder = HashingEmbedder()
    source = _source(["AE"], "C1")
    vector = embedder.embed(f"{source['title']} {source['body']}")
    missing = KnowledgeItem(
        title=source["title"],
        body=source["body"],
        kind=source["kind"],
        classification=Classification(source["classification"]),
        zones=None,
        source_label=source["source_label"],
        embedding=to_bytes(vector),
        embedding_dim=embedder.dimension,
        embedding_model=embedder.model,
    )
    session.add(missing)
    session.commit()
    session.expire_all()

    for zone in (*dp.SITES, ""):
        assert missing.id not in _ids(session, zone, Classification.C3)
        hits = _hits(session, source, zone, embedder, ceiling=Classification.C3)
        assert all(hit.id != missing.id for hit in hits)


def test_missing_caller_zone_does_not_match_a_listed_zone(session):
    """The envelope's default location is no zone, so a listed site does not match."""
    embedder = HashingEmbedder()
    source = _source(["SG", "HK", "AE"], "C1")
    granted = _index(session, source, zones=list(source["zones"]), embedder=embedder)
    session.commit()

    assert granted.id not in _ids(session, "", Classification.C3)
    hits = _hits(session, source, "", embedder, ceiling=Classification.C3)
    assert all(hit.id != granted.id for hit in hits)

    site = source["zones"][0]
    assert granted.id in _ids(session, site, Classification.C1)
    hits = _hits(session, source, site, embedder, ceiling=Classification.C1)
    assert any(hit.id == granted.id for hit in hits)


def test_listed_zone_within_ceiling_is_still_returned(session):
    """A ceiling retrieve already grants, when none is passed, still admits the item.

    A non-admin ceiling is C2. An admin ceiling is C3. An item above the
    ceiling stays out.
    """
    embedder = HashingEmbedder()
    ordinary = _source(["SG", "HK"], "C1")
    restricted = _source(["AE", "HK"], "C3")
    visible = _index(session, ordinary, zones=list(ordinary["zones"]), embedder=embedder)
    held = _index(session, restricted, zones=list(restricted["zones"]), embedder=embedder)
    session.commit()

    # Both items list HK. The non-admin ceiling is C2, so the C1 item is
    # returned and the C3 item is not.
    non_admin = _hits(session, ordinary, "HK", embedder, is_admin=False)
    assert any(hit.id == visible.id for hit in non_admin)
    assert all(hit.id != held.id for hit in non_admin)

    admin = _hits(session, restricted, "HK", embedder, is_admin=True)
    assert any(hit.id == visible.id for hit in admin)
    assert any(hit.id == held.id for hit in admin)

    below = _hits(session, restricted, "HK", embedder, ceiling=Classification.C2)
    assert any(hit.id == visible.id for hit in below)
    assert all(hit.id != held.id for hit in below)


def test_ceiling_that_cannot_be_evaluated_does_not_return_the_item(session):
    """A ceiling outside the classification vocabulary does not admit a listed zone."""
    embedder = HashingEmbedder()
    source = _source(["SG", "HK"], "C1")
    granted = _index(session, source, zones=list(source["zones"]), embedder=embedder)
    session.commit()
    site = source["zones"][0]

    with pytest.raises(ValueError):
        entitled_items(session, zone=site, ceiling=None)
    assert granted.id in _ids(session, site, Classification.C1)


def test_classification_that_cannot_be_evaluated_does_not_return_the_item(session):
    """A stored classification outside C0–C3 does not come back as a hit.

    ``SG`` is a site, not a classification. The row cannot be ranked, and the
    search does not return it.
    """
    embedder = HashingEmbedder()
    source = _source(["AE"], "C1")
    item = _index(session, source, zones=list(source["zones"]), embedder=embedder)
    session.commit()
    session.execute(
        text("UPDATE knowledge_items SET classification = 'SG' WHERE id = :id"),
        {"id": item.id},
    )
    session.commit()
    session.expire_all()

    with pytest.raises(LookupError):
        entitled_items(session, zone="AE", ceiling=Classification.C3)
    with pytest.raises(LookupError):
        _hits(session, source, "AE", embedder, ceiling=Classification.C3)
