"""Structured document ingestion.

``ingest`` parses a markdown, HTML or DOCX file into one knowledge item per
section. Classification is per section: a nested section cannot be lower
than the heading it sits under. The chunker is Kognita's. Docling is
imported only while a file is being parsed, and only when the ``docling``
extra is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from sqlmodel import Session

from kognita.classify import PatternClassifier
from kognita.exceptions import ConfigError, KognitaError
from kognita.ingest.chunker import Passage, chunk
from kognita.ingest.locator import SectionLocator
from kognita.ingest.sections import classify_section
from kognita.models import KnowledgeItem
from kognita.protocols import Embedder
from kognita.retrieval import index_item

__all__ = [
    "Passage",
    "SectionLocator",
    "chunk",
    "classify_section",
    "ingest",
]


def _title(locator: SectionLocator, reference: str) -> str:
    if locator.section_path:
        return locator.section_path[-1]
    return reference


def _bbox(locator: SectionLocator) -> dict[str, float] | None:
    if locator.bbox is None:
        return None
    left, top, right, bottom = locator.bbox
    return {"l": left, "t": top, "r": right, "b": bottom}


def ingest(
    session: Session,
    source: str | Path,
    *,
    embedder: Embedder,
    zones: Sequence[str],
    document: str = "",
    kind: str = "DOCUMENT",
    classifier: Any = None,
    index: Any = None,
) -> list[KnowledgeItem]:
    """Parse ``source`` and store one item per section.

    ``zones`` is required. An empty zone list is not visibility everywhere,
    so ingestion refuses it and writes nothing. The caller commits.

    ``document`` is the document reference. When it is empty, the file name
    is used. A reference that is still empty is refused.
    """
    cleaned = [str(zone).strip() for zone in zones if str(zone).strip()]
    if not cleaned:
        raise ConfigError(
            "ingestion refuses an empty zone list: an item with no zones "
            "is not visible in any zone"
        )
    path = Path(source)
    if not path.is_file():
        raise KognitaError(f"document not found: {path}")
    reference = (document or path.name).strip()
    if not reference:
        raise KognitaError(
            "a document reference is required; ingestion does not invent one"
        )

    from kognita.ingest.docling_backend import convert_path

    root = convert_path(path)
    passages = chunk(root, document=reference)
    chosen = classifier if classifier is not None else PatternClassifier()
    decided: list[tuple[Passage, Any, str]] = []
    by_path: dict[tuple[str, ...], Any] = {}
    for passage in passages:
        parent = None
        parent_section = ""
        section_path = passage.locator.section_path
        if len(section_path) >= 2:
            parent_path = section_path[:-1]
            parent = by_path.get(parent_path)
            parent_section = " / ".join(parent_path)
        label, rule = classify_section(
            passage.text,
            parent=parent,
            parent_section=parent_section,
            classifier=chosen,
        )
        if not rule.strip():
            raise KognitaError(
                "section classification produced no rule; ingestion indexes nothing"
            )
        by_path[section_path] = label
        decided.append((passage, label, rule))

    items: list[KnowledgeItem] = []
    for passage, label, rule in decided:
        locator = passage.locator
        start = end = None
        if locator.charspan is not None:
            start, end = locator.charspan
        items.append(
            index_item(
                session,
                title=_title(locator, reference),
                body=passage.text,
                embedder=embedder,
                kind=kind,
                classification=label,
                classification_rule=rule,
                zones=cleaned,
                source_label=locator.as_source_label(),
                index=index,
                document_ref=locator.document,
                page=locator.page,
                section_path=locator.section_path,
                char_start=start,
                char_end=end,
                bbox=_bbox(locator),
            )
        )
    return items
