"""Structured ingestion: section locators, per-section classification, passages.

The chunker, the classifier and the locator are exercised with a fake parsed
tree, so they need no Docling and no network. One test parses real markdown,
HTML and DOCX files when the ``docling`` extra is installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import select

from kognita.embedding import HashingEmbedder
from kognita.exceptions import ConfigError, KognitaError
from kognita.ingest import chunk, classify_section, ingest
from kognita.ingest.docling_backend import (
    PDF_REFUSAL,
    convert_path,
    require_success,
    tree_from_document,
)
from kognita.ingest.locator import source_text
from kognita.ingest.tree import (
    BoundingBox,
    ParsedBlock,
    Provenance,
    blocks_to_tree,
)
from kognita.models import KnowledgeItem
from kognita.retrieval import index_item, passage_snippet, retrieve
from kognita.vocabulary import Classification, EventType, classification_rank


def _box() -> BoundingBox:
    return BoundingBox(left=1.0, top=2.0, right=30.0, bottom=4.0)


def _prov(page: int, start: int, end: int) -> Provenance:
    return Provenance(page_no=page, bbox=_box(), charspan=(start, end))


def _handbook() -> ParsedBlock:
    return blocks_to_tree(
        [
            ParsedBlock(
                "Public chapter",
                "section_header",
                1,
                provenance=(_prov(1, 0, 20),),
            ),
            ParsedBlock(
                "This chapter is public and explains the review process.",
                "text",
            ),
            ParsedBlock("Steps", "section_header", 2),
            ParsedBlock("An analyst reads the file and writes a note.", "text"),
            ParsedBlock(
                "Restricted annex",
                "section_header",
                1,
                provenance=(_prov(4, 100, 180),),
            ),
            ParsedBlock(
                "Restricted. The client file names ana@example.org.",
                "text",
            ),
            ParsedBlock("Public note", "section_header", 2),
            ParsedBlock("This note is public.", "text"),
        ]
    )


def _handbook_source() -> str:
    return "\n\n".join(
        [
            "Public chapter",
            "This chapter is public and explains the review process.",
            "Steps",
            "An analyst reads the file and writes a note.",
            "Restricted annex",
            "Restricted. The client file names ana@example.org.",
            "Public note",
            "This note is public.",
        ]
    )


def _write(tmp_path: Path, name: str = "handbook.md") -> Path:
    path = tmp_path / name
    path.write_text(_handbook_source(), encoding="utf-8")
    return path


def _ingest(session, path: Path, *, zones=("SG",), document="Handbook"):
    return ingest(
        session,
        path,
        embedder=HashingEmbedder(),
        zones=list(zones),
        document=document,
    )


def test_chunk_splits_on_section_boundaries_and_keeps_nested_text_out():
    passages = chunk(_handbook(), document="Handbook")
    by_path = {passage.locator.section_path: passage.text for passage in passages}

    public = by_path[("Public chapter",)]
    steps = by_path[("Public chapter", "Steps")]
    annex = by_path[("Restricted annex",)]
    note = by_path[("Restricted annex", "Public note")]

    assert "review process" in public
    assert "An analyst" not in public
    assert "ana@example.org" not in public
    assert "An analyst" in steps
    assert "ana@example.org" in annex
    assert "This note is public" not in annex
    assert note == "Public note\n\nThis note is public."

    public_locator = next(
        passage.locator
        for passage in passages
        if passage.locator.section_path == ("Public chapter",)
    )
    assert public_locator.document == "Handbook"
    assert public_locator.page == 1
    assert public_locator.charspan is None
    assert public_locator.block_charspan == (0, 20)
    assert public_locator.bbox == (1.0, 2.0, 30.0, 4.0)
    assert "Handbook" in public_locator.as_source_label()
    assert "Public chapter" in public_locator.as_source_label()
    assert "p. 1" in public_locator.as_source_label()
    assert "chars" not in public_locator.as_source_label()

    steps_locator = next(
        passage.locator
        for passage in passages
        if passage.locator.section_path == ("Public chapter", "Steps")
    )
    assert steps_locator.page is None
    assert steps_locator.charspan is None
    assert steps_locator.block_charspan is None
    assert steps_locator.bbox is None


def test_chunk_refuses_an_empty_document_reference():
    with pytest.raises(KognitaError, match="document reference"):
        chunk(_handbook(), document="  ")


def test_ingestion_markings_cannot_lower_a_section_to_c0():
    """An explicit public marking stays at the classifier floor."""
    for text in (
        "This chapter is public.",
        "Unclassified.",
        "For publication.",
    ):
        label, rule = classify_section(text)
        assert label == Classification.C1
        assert label != Classification.C0
        assert "floor" in rule
        assert "does not lower the section" in rule


def test_preamble_is_a_document_wide_floor(session, tmp_path, monkeypatch):
    """Text before the first heading raises every later section to its level."""
    root = blocks_to_tree(
        [
            ParsedBlock("This draft is restricted.", "text"),
            ParsedBlock("Public chapter", "section_header", 1),
            ParsedBlock("This chapter is public.", "text"),
            ParsedBlock("Steps", "section_header", 2),
            ParsedBlock("An analyst reads the file.", "text"),
            ParsedBlock("Restricted annex", "section_header", 1),
            ParsedBlock("Restricted. The client file is here.", "text"),
        ]
    )
    monkeypatch.setattr(
        "kognita.ingest.docling_backend.convert_path", lambda _path: root
    )
    source = "\n\n".join(
        [
            "This draft is restricted.",
            "Public chapter",
            "This chapter is public.",
            "Steps",
            "An analyst reads the file.",
            "Restricted annex",
            "Restricted. The client file is here.",
        ]
    )
    path = tmp_path / "handbook.md"
    path.write_text(source, encoding="utf-8")
    items = _ingest(session, path)
    by_path = {tuple(item.section_path): item for item in items}
    preamble = by_path[()]
    chapter = by_path[("Public chapter",)]
    steps = by_path[("Public chapter", "Steps")]
    annex = by_path[("Restricted annex",)]
    assert preamble.classification == Classification.C3
    assert "marking" in preamble.classification_rule
    assert chapter.classification == Classification.C3
    assert "preamble" in chapter.classification_rule
    assert "cannot be classified below the preamble" in chapter.classification_rule
    assert "restricted" not in chapter.body.lower()
    assert steps.classification == Classification.C3
    assert "preamble" in steps.classification_rule
    assert "Public chapter" in steps.classification_rule
    assert annex.classification == Classification.C3
    assert "marking" in annex.classification_rule
    assert source[chapter.char_start : chapter.char_end].startswith("Public chapter")
    assert (
        "This draft is restricted." not in source[chapter.char_start : chapter.char_end]
    )


def test_preamble_floor_does_not_lower_a_later_section(session, tmp_path, monkeypatch):
    root = blocks_to_tree(
        [
            ParsedBlock("This note is public.", "text"),
            ParsedBlock("Restricted annex", "section_header", 1),
            ParsedBlock("Restricted. The client file is here.", "text"),
        ]
    )
    monkeypatch.setattr(
        "kognita.ingest.docling_backend.convert_path", lambda _path: root
    )
    path = tmp_path / "handbook.md"
    path.write_text(
        "This note is public.\n\nRestricted annex\n\nRestricted. The client file is here.",
        encoding="utf-8",
    )
    items = _ingest(session, path)
    by_path = {tuple(item.section_path): item for item in items}
    assert by_path[()].classification == Classification.C1
    annex = by_path[("Restricted annex",)]
    assert annex.classification == Classification.C3
    assert "marking" in annex.classification_rule
    assert "preamble" not in annex.classification_rule


def test_a_passage_missing_from_the_source_indexes_nothing(
    session, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "kognita.ingest.docling_backend.convert_path", lambda _path: _handbook()
    )
    path = tmp_path / "handbook.md"
    path.write_text("this file does not contain the parsed passages", encoding="utf-8")
    with pytest.raises(KognitaError, match="does not invent a character span"):
        _ingest(session, path)
    assert session.exec(select(KnowledgeItem)).all() == []


def test_a_non_utf8_document_indexes_nothing(session, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kognita.ingest.docling_backend.convert_path", lambda _path: _handbook()
    )
    path = tmp_path / "handbook.md"
    path.write_bytes(b"\xff\xfe not utf-8")
    with pytest.raises(KognitaError, match="UTF-8"):
        _ingest(session, path)
    assert session.exec(select(KnowledgeItem)).all() == []


def test_classification_cites_a_rule_and_does_not_fall_below_a_parent():
    public, public_rule = classify_section(
        "Public chapter\n\nThis chapter is public and explains the review process."
    )
    assert public == Classification.C1
    assert "floor" in public_rule
    assert "does not lower the section" in public_rule
    assert "C1" in public_rule

    unmarked, unmarked_rule = classify_section("An analyst reads the file.")
    assert unmarked == Classification.C1
    assert "no handling marking" in unmarked_rule
    assert "not a lower classification" in unmarked_rule

    identified, identified_rule = classify_section(
        "Call ana@example.org about the notes."
    )
    assert identified == Classification.C2
    assert "identifier" in identified_rule

    annex, annex_rule = classify_section("Restricted annex\n\nThe client file is here.")
    assert annex == Classification.C3
    assert "marking" in annex_rule
    assert "C3" in annex_rule

    note, note_rule = classify_section(
        "This note is public.",
        parent=Classification.C3,
        parent_section="Restricted annex",
    )
    assert note == Classification.C3
    assert "inherited" in note_rule
    assert "Restricted annex" in note_rule
    assert "cannot be classified below its parent" in note_rule
    assert classification_rank(note) >= classification_rank(public)


def test_empty_zones_index_nothing(session, tmp_path, monkeypatch):
    def explode(_path):
        raise AssertionError("conversion must not run when zones are empty")

    monkeypatch.setattr("kognita.ingest.docling_backend.convert_path", explode)
    with pytest.raises(ConfigError, match="empty zone list"):
        _ingest(session, _write(tmp_path), zones=())
    assert session.exec(select(KnowledgeItem)).all() == []


def test_missing_file_indexes_nothing(session, tmp_path):
    missing = tmp_path / "absent.md"
    with pytest.raises(KognitaError, match="document not found"):
        _ingest(session, missing)
    assert session.exec(select(KnowledgeItem)).all() == []


def test_unsupported_format_indexes_nothing(session, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(KognitaError, match="unsupported document format"):
        _ingest(session, path)
    assert session.exec(select(KnowledgeItem)).all() == []


def test_a_failed_conversion_indexes_nothing(session, tmp_path, monkeypatch):
    def fail(_path):
        raise KognitaError(
            "document conversion failed (PARTIAL_SUCCESS); "
            "ingestion indexes nothing from a partial parse"
        )

    monkeypatch.setattr("kognita.ingest.docling_backend.convert_path", fail)
    with pytest.raises(KognitaError, match="PARTIAL_SUCCESS"):
        _ingest(session, _write(tmp_path))
    assert session.exec(select(KnowledgeItem)).all() == []


def test_require_success_rejects_a_partial_parse():
    class Status:
        def __init__(self, name):
            self.name = name

    class Result:
        def __init__(self, status):
            self.status = status

    require_success(Result(Status("SUCCESS")))
    with pytest.raises(KognitaError, match="PARTIAL_SUCCESS"):
        require_success(Result(Status("PARTIAL_SUCCESS")))
    with pytest.raises(KognitaError, match="indexes nothing"):
        require_success(Result(None))


def test_ingest_classifies_each_section_and_cites_the_passage(
    session, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "kognita.ingest.docling_backend.convert_path", lambda _path: _handbook()
    )
    path = _write(tmp_path)
    items = _ingest(session, path)
    session.commit()
    by_title = {item.title: item for item in items}

    public = by_title["Public chapter"]
    steps = by_title["Steps"]
    annex = by_title["Restricted annex"]
    note = by_title["Public note"]

    assert public.classification == Classification.C1
    assert "floor" in public.classification_rule
    assert steps.classification == Classification.C1
    assert annex.classification == Classification.C3
    assert "marking" in annex.classification_rule
    assert note.classification == Classification.C3
    assert "inherited" in note.classification_rule
    assert "Restricted annex" in note.classification_rule
    assert "ana@example.org" not in public.body
    assert "An analyst" not in public.body
    assert "ana@example.org" in annex.body

    source = path.read_text(encoding="utf-8")
    assert public.document_ref == "Handbook"
    assert public.page == 1
    assert public.section_path == ["Public chapter"]
    assert public.block_char_start == 0
    assert public.block_char_end == 20
    public_slice = source[public.char_start : public.char_end]
    assert public_slice.startswith("Public chapter")
    assert public_slice.endswith("review process.")
    assert "An analyst" not in public_slice
    assert f"chars {public.char_start}-{public.char_end}" in public.source_label
    assert public.bbox == {"l": 1.0, "t": 2.0, "r": 30.0, "b": 4.0}
    assert public.source_label.startswith("Handbook, Public chapter")
    assert steps.page is None
    assert steps.block_char_start is None
    assert "An analyst" in source[steps.char_start : steps.char_end]
    assert "ana@example.org" not in source[steps.char_start : steps.char_end]
    assert annex.page == 4
    assert annex.block_char_start == 100
    assert annex.block_char_end == 180
    assert source[annex.char_start : annex.char_end].startswith("Restricted annex")
    assert annex.section_path == ["Restricted annex"]
    assert note.section_path == ["Restricted annex", "Public note"]
    assert "This note is public." in source[note.char_start : note.char_end]

    visible = retrieve(
        session,
        "client file",
        zone="SG",
        embedder=HashingEmbedder(),
        ceiling=Classification.C3,
        min_score=-1.0,
    )
    annex_hits = [hit for hit in visible if hit.id == annex.id]
    assert annex_hits
    assert annex_hits[0].snippet == "Restricted. The client file names ana@example.org."
    assert annex_hits[0].section_path == ("Restricted annex",)
    assert annex_hits[0].page == 4
    assert annex_hits[0].char_start == annex.char_start
    assert annex_hits[0].char_end == annex.char_end
    assert annex_hits[0].block_char_start == 100
    assert annex_hits[0].block_char_end == 180
    assert annex_hits[0].document_ref == "Handbook"
    assert "Handbook" in annex_hits[0].source_label

    hidden = retrieve(
        session,
        "client file",
        zone="SG",
        embedder=HashingEmbedder(),
        ceiling=Classification.C2,
        min_score=-1.0,
    )
    assert all(hit.id != annex.id for hit in hidden)
    assert all(hit.id != note.id for hit in hidden)
    assert any(hit.id == public.id for hit in hidden)


def test_retrieval_evidence_records_the_locator(
    session, evidence, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "kognita.ingest.docling_backend.convert_path", lambda _path: _handbook()
    )
    _ingest(session, _write(tmp_path))
    session.commit()
    retrieve(
        session,
        "client file",
        zone="SG",
        embedder=HashingEmbedder(),
        evidence=evidence,
        correlation_id="ingest-locator",
        ceiling=Classification.C3,
        min_score=-1.0,
    )
    from kognita.models import EvidenceEvent

    event = session.exec(
        select(EvidenceEvent).where(EvidenceEvent.correlation_id == "ingest-locator")
    ).one()
    assert event.event_type == EventType.RETRIEVAL
    annex = next(item for item in event.payload["items"] if item["page"] == 4)
    source = _handbook_source()
    assert annex["section_path"] == ["Restricted annex"]
    assert annex["block_charspan"] == [100, 180]
    start, end = annex["charspan"]
    assert source[start:end].startswith("Restricted annex")
    assert "ana@example.org" in source[start:end]
    assert "This note is public." not in source[start:end]
    assert annex["document"] == "Handbook"
    assert annex["classification"] == "C3"
    assert "marking" in annex["classification_rule"]
    assert annex["content_hash"]
    assert annex["bbox"] == {"l": 1.0, "t": 2.0, "r": 30.0, "b": 4.0}


def test_snippet_is_the_matching_passage_not_a_prefix(session):
    prefix = "alpha " * 80
    body = prefix.strip() + "\n\n" + "the client mandate is restricted to this passage"
    assert len(prefix) > 280
    item = index_item(
        session,
        title="Charter",
        body=body,
        embedder=HashingEmbedder(),
        zones=["SG"],
        source_label="Charter",
    )
    session.commit()
    hits = retrieve(
        session,
        "mandate",
        zone="SG",
        embedder=HashingEmbedder(),
        min_score=-1.0,
    )
    hit = next(found for found in hits if found.id == item.id)
    assert hit.snippet == "the client mandate is restricted to this passage"
    assert not hit.snippet.startswith("alpha")
    whole = "mandate " + ("word " * 200)
    assert len(whole) > 280
    assert passage_snippet(whole, "mandate") == whole.strip()


def test_tree_from_document_keeps_only_complete_provenance():
    box = type("Box", (), {})()
    for name, value in (("l", 5.0), ("t", 6.0), ("r", 7.0), ("b", 8.0)):
        setattr(box, name, value)

    class Complete:
        page_no = 2
        bbox = box
        charspan = (8, 16)

    class MissingBox:
        page_no = 9
        bbox = None
        charspan = (1, 2)

    class Item:
        def __init__(self, text, label, level=None, prov=()):
            self.text = text
            self.label = type("Label", (), {"value": label})()
            self.level = level
            self.prov = list(prov)

    class Document:
        def __init__(self, items):
            self._items = items

        def iterate_items(self):
            return [(item, 0) for item in self._items]

    root = tree_from_document(
        Document(
            [
                Item("Chapter", "section_header", 1, prov=(Complete(), MissingBox())),
                Item("A paragraph.", "text", prov=(MissingBox(),)),
            ]
        )
    )
    passages = chunk(root, document="Guide")
    assert passages[0].locator.page == 2
    assert passages[0].locator.block_charspan == (8, 16)
    assert passages[0].locator.charspan is None
    assert passages[0].locator.bbox == (5.0, 6.0, 7.0, 8.0)
    assert "A paragraph." in passages[0].text


class _Cref:
    def __init__(self, cref: str) -> None:
        self.cref = cref


def _doc_item(
    text: str,
    label: str,
    *,
    level: int | None = None,
    self_ref: str = "",
    parent: str = "#/body",
):
    """A stand-in for one Docling item. No Docling import."""
    item = type("Item", (), {})()
    item.text = text
    item.label = type("Label", (), {"value": label})()
    item.level = level
    item.prov = []
    item.self_ref = self_ref
    item.parent = _Cref(parent)
    return item


def _document(items: list[object]):
    document = type("Document", (), {})()
    document.iterate_items = lambda: [(item, 0) for item in items]
    return document


def _paths(root) -> dict[tuple[str, ...], str]:
    return {
        passage.locator.section_path: passage.text
        for passage in chunk(root, document="Handbook")
    }


def test_flat_heading_levels_nest_under_the_title():
    """Markdown stores an h2 as section_header level 1, sibling of the title.

    Parent refs all name the body, so nesting has to come from the level.
    Docling's stored level is one below the title; the adapter adds one.
    """
    root = tree_from_document(
        _document(
            [
                _doc_item("Public chapter", "title", self_ref="#/texts/0"),
                _doc_item(
                    "This chapter is public and explains the review process.",
                    "text",
                    self_ref="#/texts/1",
                ),
                _doc_item("Steps", "section_header", level=1, self_ref="#/texts/2"),
                _doc_item(
                    "An analyst reads the file and writes a note.",
                    "text",
                    self_ref="#/texts/3",
                ),
                _doc_item("Restricted annex", "title", self_ref="#/texts/4"),
                _doc_item(
                    "Restricted. The client file names ana@example.org.",
                    "text",
                    self_ref="#/texts/5",
                ),
            ]
        )
    )
    by_path = _paths(root)
    assert "An analyst" not in by_path[("Public chapter",)]
    assert "ana@example.org" not in by_path[("Public chapter",)]
    assert "An analyst" in by_path[("Public chapter", "Steps")]
    assert "ana@example.org" in by_path[("Restricted annex",)]


def test_parent_links_nest_when_the_level_numbers_match():
    """HTML parents an h2 to the h1 and stores both at level 1.

    Level stacking alone would make them siblings. The parent link is what
    puts Steps under the public chapter.
    """
    root = tree_from_document(
        _document(
            [
                _doc_item(
                    "Public chapter", "title", self_ref="#/texts/0", parent="#/body"
                ),
                _doc_item(
                    "This chapter is public and explains the review process.",
                    "text",
                    self_ref="#/texts/1",
                    parent="#/texts/0",
                ),
                _doc_item(
                    "Steps",
                    "title",
                    self_ref="#/texts/2",
                    parent="#/texts/0",
                ),
                _doc_item(
                    "An analyst reads the file and writes a note.",
                    "text",
                    self_ref="#/texts/3",
                    parent="#/texts/2",
                ),
                _doc_item(
                    "Restricted annex",
                    "title",
                    self_ref="#/texts/4",
                    parent="#/body",
                ),
                _doc_item(
                    "Restricted. The client file names ana@example.org.",
                    "text",
                    self_ref="#/texts/5",
                    parent="#/texts/4",
                ),
            ]
        )
    )
    by_path = _paths(root)
    assert "An analyst" not in by_path[("Public chapter",)]
    assert "An analyst" in by_path[("Public chapter", "Steps")]
    assert "ana@example.org" in by_path[("Restricted annex",)]
    assert "ana@example.org" not in by_path[("Public chapter", "Steps")]


def test_missing_docling_names_the_extra(session, tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "docling" or name.startswith("docling."):
            raise ImportError("simulated")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    path = _write(tmp_path)
    with pytest.raises(ConfigError, match=r"kognita\[docling\]"):
        _ingest(session, path)
    assert session.exec(select(KnowledgeItem)).all() == []


def test_pdf_is_refused_and_indexes_nothing(session, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kognita.ingest.docling_backend._load_docling",
        lambda: (object, object),
    )
    pdf = tmp_path / "policy.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(KognitaError, match="layout model") as excinfo:
        _ingest(session, pdf)
    message = str(excinfo.value)
    assert message == PDF_REFUSAL
    assert "torch" in message
    assert "does not guess" in message
    assert session.exec(select(KnowledgeItem)).all() == []
    with pytest.raises(KognitaError, match="layout model"):
        convert_path(pdf)


_HANDBOOK = "\n".join(
    [
        "# Public chapter",
        "",
        "This chapter is public and explains the review process.",
        "",
        "## Steps",
        "",
        "An analyst reads the file and writes a note.",
        "",
        "# Restricted annex",
        "",
        "Restricted. The client file names ana@example.org and stays in this annex.",
        "",
    ]
)


def _assert_handbook(items, source: str) -> None:
    assert items
    by_path = {tuple(item.section_path): item for item in items}
    public = by_path[("Public chapter",)]
    steps = by_path[("Public chapter", "Steps")]
    annex = by_path[("Restricted annex",)]
    assert "An analyst" not in public.body
    assert "ana@example.org" not in public.body
    assert "ana@example.org" not in steps.body
    assert "An analyst" in steps.body
    assert annex.classification == Classification.C3
    assert "marking" in annex.classification_rule
    assert "ana@example.org" in annex.body
    assert public.classification != Classification.C3
    assert steps.classification != Classification.C3
    assert public.classification != Classification.C0
    assert steps.classification != Classification.C0
    assert all(item.document_ref == "Handbook" for item in items)
    assert all(item.classification_rule for item in items)
    for item in (public, steps, annex):
        window = source[item.char_start : item.char_end]
        assert item.title in window
        assert window in source


def test_docling_parses_markdown_html_and_docx(session, tmp_path):
    """Runs only when the extra is installed. No network and no model weights."""
    pytest.importorskip("docling")
    markdown = tmp_path / "handbook.md"
    markdown.write_text(_HANDBOOK, encoding="utf-8")
    html = tmp_path / "handbook.html"
    html.write_text(
        "\n".join(
            [
                "<h1>Public chapter</h1>",
                "<p>This chapter is public and explains the review process.</p>",
                "<h2>Steps</h2>",
                "<p>An analyst reads the file and writes a note.</p>",
                "<h1>Restricted annex</h1>",
                "<p>Restricted. The client file names ana@example.org "
                "and stays in this annex.</p>",
            ]
        ),
        encoding="utf-8",
    )
    docx = pytest.importorskip("docx")
    word = tmp_path / "handbook.docx"
    document = docx.Document()
    document.add_heading("Public chapter", level=1)
    document.add_paragraph("This chapter is public and explains the review process.")
    document.add_heading("Steps", level=2)
    document.add_paragraph("An analyst reads the file and writes a note.")
    document.add_heading("Restricted annex", level=1)
    document.add_paragraph(
        "Restricted. The client file names ana@example.org and stays in this annex."
    )
    document.save(word)

    for path in (markdown, html, word):
        items = ingest(
            session,
            path,
            embedder=HashingEmbedder(),
            zones=["SG"],
            document="Handbook",
        )
        session.commit()
        _assert_handbook(items, source_text(path))


def test_a_decoded_entity_or_escape_is_located_in_the_raw_source():
    """``Tom &amp; Jerry`` parses to ``Tom & Jerry``; the span is the raw range."""
    from kognita.ingest.tree import ParsedBlock

    for source, raw in (
        ("<h1>Cartoons</h1>\n<p>Tom &amp; Jerry\n   chase</p>", "Tom &amp; Jerry\n   chase"),
        ("# Cartoons\n\nTom \\& Jerry chase \\*daily\\*\n", "Tom \\& Jerry chase \\*daily\\*"),
    ):
        body = "Tom & Jerry chase" if "&amp;" in source else "Tom & Jerry chase *daily*"
        root = ParsedBlock(
            text="",
            label="document",
            children=(
                ParsedBlock(
                    text="Cartoons",
                    label="section_header",
                    level=1,
                    children=(ParsedBlock(text=body, label="text"),),
                ),
            ),
        )
        passages = chunk(root, document="Guide", source=source)
        start, end = passages[0].locator.charspan
        assert source[start:end].endswith(raw)
        assert source[start:end].startswith("Cartoons")


def test_a_passage_absent_from_the_source_is_still_refused():
    from kognita.ingest.tree import ParsedBlock

    root = ParsedBlock(
        text="",
        label="document",
        children=(ParsedBlock(text="Not in the file", label="text"),),
    )
    with pytest.raises(KognitaError, match="not found in the source text"):
        chunk(root, document="Guide", source="<p>Something else &amp; more</p>")
