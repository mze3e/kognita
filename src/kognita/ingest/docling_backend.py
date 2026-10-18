"""The only module that imports Docling.

The import sits inside a function. Importing ``kognita.ingest`` does not
import Docling, and a missing extra raises ``ConfigError`` naming
``kognita[docling]`` rather than a bare import error.

PDF files are refused. ``docling-parse`` is MIT and the extra pins
``format-pdf``, but the headings a PDF section tree needs come from
Docling's layout model, and that model pulls torch. This module does not
guess headings and does not download weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kognita.exceptions import ConfigError, KognitaError
from kognita.ingest.tree import (
    BoundingBox,
    ParsedBlock,
    Provenance,
    blocks_to_tree,
    is_heading,
    link_by_parent,
)

#: Formats whose backends emit a document tree without a layout model.
_SIMPLE_SUFFIXES = frozenset({".md", ".markdown", ".html", ".htm", ".docx"})

PDF_REFUSAL = (
    "PDF section structure comes from Docling's layout model, which is not "
    "part of kognita[docling]. That model ships in docling-ibm-models and "
    "pulls torch. format-pdf is pinned because docling-parse is MIT, but "
    "ingestion does not guess headings from a PDF and does not download "
    "model weights. Markdown, HTML, and DOCX are parsed into sections."
)


def _load_docling() -> tuple[Any, Any]:
    """Import Docling, or say which extra supplies it."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ConfigError(
            "Docling is not installed. Install it with: pip install 'kognita[docling]'"
        ) from exc
    return DocumentConverter, InputFormat


def require_success(result: Any) -> None:
    """Refuse anything other than a completed conversion.

    A partial parse is not indexed. A missing status is not success.
    """
    status = getattr(result, "status", None)
    name = getattr(status, "name", None)
    if not isinstance(name, str):
        name = str(status)
    if name != "SUCCESS":
        raise KognitaError(
            f"document conversion failed ({name}); "
            "ingestion indexes nothing from a partial parse"
        )


def _label_name(item: Any) -> str:
    label = getattr(item, "label", "")
    value = getattr(label, "value", label)
    return str(value).strip().lower()


def _text_of(item: Any) -> str:
    text = getattr(item, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    exporter = getattr(item, "export_to_markdown", None)
    if not callable(exporter):
        return ""
    try:
        rendered = exporter()
    except TypeError:
        return ""
    if isinstance(rendered, str) and rendered.strip():
        return rendered.strip()
    return ""


def _provenance_of(item: Any) -> tuple[Provenance, ...]:
    """Keep a pointer only when page, box and character span are all present."""
    found: list[Provenance] = []
    for prov in getattr(item, "prov", None) or []:
        page = getattr(prov, "page_no", None)
        bbox = getattr(prov, "bbox", None)
        span = getattr(prov, "charspan", None)
        if page is None or bbox is None or span is None:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
            box = BoundingBox(
                left=float(bbox.l),
                top=float(bbox.t),
                right=float(bbox.r),
                bottom=float(bbox.b),
            )
            page_no = int(page)
        except (TypeError, ValueError, IndexError, AttributeError):
            continue
        found.append(Provenance(page_no=page_no, bbox=box, charspan=(start, end)))
    return tuple(found)


def _level_of(item: Any, label: str) -> int | None:
    level = getattr(item, "level", None)
    if level is None:
        return 1 if label == "title" else None
    try:
        return int(level)
    except (TypeError, ValueError):
        return None


def block_from_item(item: Any) -> ParsedBlock | None:
    """Copy one Docling item into a ``ParsedBlock``. Empty text is dropped."""
    label = _label_name(item)
    text = _text_of(item)
    if not text:
        return None
    return ParsedBlock(
        text=text,
        label=label,
        level=_level_of(item, label),
        provenance=_provenance_of(item),
    )


def _self_ref(item: Any) -> str:
    return str(getattr(item, "self_ref", "") or "")


def _parent_ref(item: Any) -> str:
    parent = getattr(item, "parent", None)
    return str(getattr(parent, "cref", "") or "")


def _headings_are_nested(nodes: list[tuple[str, str, ParsedBlock]]) -> bool:
    """True when a heading's parent is itself a heading.

    HTML and DOCX do this. Markdown puts every block under the body and
    does not.
    """
    headings = {ref for ref, _parent, block in nodes if ref and is_heading(block)}
    return any(parent in headings for _ref, parent, block in nodes if is_heading(block))


def _flat_heading_level(block: ParsedBlock) -> ParsedBlock:
    """Restore a heading level Docling stored one below the title.

    In the flat markdown tree, an h1 is a title and an h2 is a section
    header whose ``level`` is 1. Adding one puts the h2 under the h1.
    Callers use this only when parent links did not already nest headings.
    """
    if block.label == "section_header" and block.level is not None:
        return ParsedBlock(
            text=block.text,
            label=block.label,
            level=block.level + 1,
            provenance=block.provenance,
            children=block.children,
        )
    return block


def tree_from_document(document: Any) -> ParsedBlock:
    """Copy a Docling document into Kognita's tree.

    Parent links are kept when they already nest headings. A flat reading
    order is nested from heading levels instead.
    """
    nodes: list[tuple[str, str, ParsedBlock]] = []
    for entry in document.iterate_items():
        item = entry[0] if isinstance(entry, tuple) else entry
        block = block_from_item(item)
        if block is None:
            continue
        nodes.append((_self_ref(item), _parent_ref(item), block))
    if _headings_are_nested(nodes):
        return link_by_parent(nodes)
    return blocks_to_tree(
        [_flat_heading_level(block) for _ref, _parent, block in nodes]
    )


def convert_path(path: Path) -> ParsedBlock:
    """Convert one file. Docling is imported here, not at module import."""
    suffix = path.suffix.lower()
    if suffix not in _SIMPLE_SUFFIXES and suffix != ".pdf":
        raise KognitaError(
            f"unsupported document format {suffix or '(none)'}; "
            "ingestion parses markdown, HTML, and DOCX into sections"
        )
    DocumentConverter, InputFormat = _load_docling()
    if suffix == ".pdf":
        raise KognitaError(PDF_REFUSAL)
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD, InputFormat.HTML, InputFormat.DOCX]
    )
    result = converter.convert(str(path))
    require_success(result)
    document = getattr(result, "document", None)
    if document is None:
        raise KognitaError(
            "document conversion returned no document; ingestion indexes nothing"
        )
    return tree_from_document(document)


__all__ = [
    "PDF_REFUSAL",
    "block_from_item",
    "convert_path",
    "require_success",
    "tree_from_document",
]
