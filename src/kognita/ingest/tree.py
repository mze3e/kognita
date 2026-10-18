"""The parsed tree the chunker walks.

Nodes are Kognita's. A Docling document is copied into this shape at the
edge of the optional extra, so the chunker never imports Docling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BoundingBox:
    """One parser bounding box, in the parser's coordinates."""

    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class Provenance:
    """One parser pointer back into the source file.

    ``page_no``, ``bbox`` and ``charspan`` are the parser's. A pointer that
    is missing any of them is dropped by the adapter rather than completed.
    """

    page_no: int
    bbox: BoundingBox
    charspan: tuple[int, int]


@dataclass(frozen=True)
class ParsedBlock:
    """One node. Headings carry ``level``; body blocks do not.

    ``children`` holds the blocks that belong to this node. A heading's
    children are the body under it and any nested headings. The chunker
    splits on those heading boundaries.
    """

    text: str
    label: str
    level: int | None = None
    provenance: tuple[Provenance, ...] = ()
    children: tuple[ParsedBlock, ...] = ()


#: Labels that open a section. Docling's names, plus ``title``.
HEADING_LABELS = frozenset({"section_header", "title"})


def is_heading(block: ParsedBlock) -> bool:
    """A heading with a level and text. Anything else is body content."""
    return (
        block.label in HEADING_LABELS
        and block.level is not None
        and block.level >= 1
        and bool(block.text.strip())
    )


@dataclass
class _Build:
    text: str
    label: str
    level: int | None
    provenance: tuple[Provenance, ...]
    children: list[_Build]

    def freeze(self) -> ParsedBlock:
        return ParsedBlock(
            text=self.text,
            label=self.label,
            level=self.level,
            provenance=self.provenance,
            children=tuple(child.freeze() for child in self.children),
        )


def link_by_parent(nodes: Sequence[tuple[str, str, ParsedBlock]]) -> ParsedBlock:
    """Attach each block to the block its parent ref names.

    ``nodes`` is ``(self_ref, parent_ref, block)`` in reading order. A parent
    ref that names no block hangs off the document, in that same order. This
    is the tree the parser already built.
    """
    builds: dict[str, _Build] = {}
    ordered: list[tuple[str, _Build]] = []
    for self_ref, parent_ref, block in nodes:
        node = _Build(
            text=block.text,
            label=block.label,
            level=block.level,
            provenance=block.provenance,
            children=[],
        )
        ordered.append((parent_ref, node))
        if self_ref and self_ref not in builds:
            builds[self_ref] = node
    root = _Build(text="", label="document", level=0, provenance=(), children=[])
    for parent_ref, node in ordered:
        parent = builds.get(parent_ref)
        if parent is None or parent is node or _contains(node, parent):
            root.children.append(node)
        else:
            parent.children.append(node)
    return root.freeze()


def _contains(node: _Build, target: _Build) -> bool:
    """True when ``target`` is ``node`` or already under it. Breaks a cycle."""
    if node is target:
        return True
    return any(_contains(child, target) for child in node.children)


def blocks_to_tree(blocks: Sequence[ParsedBlock]) -> ParsedBlock:
    """Nest a reading-order list of blocks under their headings.

    A heading of level N closes open headings at level N or deeper and
    becomes a child of the heading still open above it. Body blocks belong
    to the heading that is open. Text before the first heading belongs to
    the document node.
    """
    root = _Build(text="", label="document", level=0, provenance=(), children=[])
    stack: list[_Build] = [root]
    for block in blocks:
        node = _Build(
            text=block.text,
            label=block.label,
            level=block.level,
            provenance=block.provenance,
            children=[],
        )
        if is_heading(block):
            while len(stack) > 1 and (stack[-1].level or 0) >= (block.level or 0):
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].children.append(node)
    return root.freeze()


__all__ = [
    "BoundingBox",
    "HEADING_LABELS",
    "ParsedBlock",
    "Provenance",
    "blocks_to_tree",
    "is_heading",
    "link_by_parent",
]
