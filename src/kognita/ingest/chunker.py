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
    document: str, path: tuple[str, ...], blocks: list[ParsedBlock]
) -> SectionLocator:
    """The first provenance the parser actually supplied for this passage."""
    for block in blocks:
        if not block.provenance:
            continue
        provenance = block.provenance[0]
        box = provenance.bbox
        return SectionLocator(
            document=document,
            page=provenance.page_no,
            section_path=path,
            charspan=provenance.charspan,
            bbox=(box.left, box.top, box.right, box.bottom),
        )
    return SectionLocator(
        document=document,
        page=None,
        section_path=path,
        charspan=None,
        bbox=None,
    )


def chunk(root: ParsedBlock, *, document: str) -> list[Passage]:
    """Split ``root`` into one passage per section.

    ``document`` is the document reference stored on every locator. An empty
    reference is refused: the chunker does not invent one.
    """
    reference = document.strip()
    if not reference:
        raise KognitaError(
            "a document reference is required; ingestion does not invent one"
        )

    passages: list[Passage] = []

    def emit(node: ParsedBlock, path: tuple[str, ...], level: int | None) -> None:
        include_self = node.label != "document"
        parts = _parts(node, include_self=include_self)
        text = "\n\n".join(parts).strip()
        if text:
            passages.append(
                Passage(
                    text=text,
                    locator=_locator(
                        reference, path, _blocks(node, include_self=include_self)
                    ),
                    level=level,
                )
            )
        for child in node.children:
            if is_heading(child):
                emit(child, path + (child.text.strip(),), child.level)

    emit(root, (), None)
    return passages


__all__ = ["Passage", "chunk"]
