"""Kognita's chunker.

It walks the parsed tree and splits on section boundaries. A section's
passage is its own heading and body, not the text of a nested section.
Nested sections are their own passages, so a restricted annex is not
copied into the public chapter above it.
"""

from __future__ import annotations

from dataclasses import dataclass

from kognita.exceptions import KognitaError
from kognita.ingest.locator import SectionLocator
from kognita.ingest.tree import ParsedBlock, is_heading


@dataclass(frozen=True)
class Passage:
    """One section's own text, and the locator that cites it."""

    text: str
    locator: SectionLocator
    level: int | None


def _parts(node: ParsedBlock, *, include_self: bool) -> list[str]:
    parts: list[str] = []
    if include_self and node.text.strip():
        parts.append(node.text.strip())
    for child in node.children:
        if is_heading(child):
            continue
        parts.extend(_parts(child, include_self=True))
    return parts


def _blocks(node: ParsedBlock, *, include_self: bool) -> list[ParsedBlock]:
    found: list[ParsedBlock] = []
    if include_self:
        found.append(node)
    for child in node.children:
        if is_heading(child):
            continue
        found.extend(_blocks(child, include_self=True))
    return found


def _locator(
    document: str,
    path: tuple[str, ...],
    blocks: list[ParsedBlock],
    global_span: tuple[int, int] | None,
) -> SectionLocator:
    """Parser provenance for the opening block, plus the global range."""
    page = None
    bbox = None
    block_charspan = None
    for block in blocks:
        if not block.provenance:
            continue
        provenance = block.provenance[0]
        box = provenance.bbox
        page = provenance.page_no
        block_charspan = provenance.charspan
        bbox = (box.left, box.top, box.right, box.bottom)
        break
    return SectionLocator(
        document=document,
        page=page,
        section_path=path,
        charspan=global_span,
        bbox=bbox,
        block_charspan=block_charspan,
    )


def _block_spans(root: ParsedBlock, source: str) -> dict[int, tuple[int, int]]:
    """Locate every block in ``source``, in reading order.

    The search starts after the previous block. A block the source does not
    contain is refused: the span is not guessed, and nothing is indexed.
    """
    spans: dict[int, tuple[int, int]] = {}
    cursor = 0

    def walk(node: ParsedBlock) -> None:
        nonlocal cursor
        needle = node.text.strip()
        if needle and node.label != "document":
            at = source.find(needle, cursor)
            if at < 0:
                raise KognitaError(
                    "a passage was not found in the source text; "
                    "ingestion does not invent a character span and indexes nothing"
                )
            spans[id(node)] = (at, at + len(needle))
            cursor = at + len(needle)
        for child in node.children:
            walk(child)

    walk(root)
    return spans


def _global_span(
    blocks: list[ParsedBlock], spans: dict[int, tuple[int, int]]
) -> tuple[int, int] | None:
    found = [spans[id(block)] for block in blocks if id(block) in spans]
    if not found:
        return None
    return min(span[0] for span in found), max(span[1] for span in found)


def chunk(
    root: ParsedBlock, *, document: str, source: str | None = None
) -> list[Passage]:
    """Split ``root`` into one passage per section.

    ``document`` is the document reference stored on every locator. An empty
    reference is refused: the chunker does not invent one.

    ``source`` is the text :func:`kognita.ingest.locator.source_text` returns.
    When it is given, each passage's ``charspan`` is the document-global
    range covering that passage's own blocks. A block that cannot be found
    is refused.
    """
    reference = document.strip()
    if not reference:
        raise KognitaError(
            "a document reference is required; ingestion does not invent one"
        )
    spans = _block_spans(root, source) if source is not None else None

    passages: list[Passage] = []

    def emit(node: ParsedBlock, path: tuple[str, ...], level: int | None) -> None:
        include_self = node.label != "document"
        own_blocks = _blocks(node, include_self=include_self)
        parts = _parts(node, include_self=include_self)
        text = "\n\n".join(parts).strip()
        if text:
            global_span = _global_span(own_blocks, spans) if spans is not None else None
            if source is not None and global_span is None:
                raise KognitaError(
                    "a passage was not found in the source text; "
                    "ingestion does not invent a character span and indexes nothing"
                )
            passages.append(
                Passage(
                    text=text,
                    locator=_locator(reference, path, own_blocks, global_span),
                    level=level,
                )
            )
        for child in node.children:
            if is_heading(child):
                emit(child, path + (child.text.strip(),), child.level)

    emit(root, (), None)
    return passages


__all__ = ["Passage", "chunk"]
