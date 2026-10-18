"""A structural locator for one ingested passage.

The fields are the ones a parser can actually report: a document reference,
a page, a section path, a character span, and a bounding box. Nothing here
fills a missing page or span with a guess.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionLocator:
    """Where a passage sits in its source document.

    ``charspan`` is the parser's ``(start, end)`` pair. ``bbox`` is
    ``(l, t, r, b)`` in the parser's own coordinates. Either is ``None``
    when the parser did not report it.
    """

    document: str
    page: int | None
    section_path: tuple[str, ...]
    charspan: tuple[int, int] | None
    bbox: tuple[float, float, float, float] | None

    def as_source_label(self) -> str:
        """The readable citation stored on ``KnowledgeItem.source_label``."""
        parts = [self.document]
        if self.section_path:
            parts.append(" / ".join(self.section_path))
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.charspan is not None:
            start, end = self.charspan
            parts.append(f"chars {start}-{end}")
        return ", ".join(parts)


__all__ = ["SectionLocator"]
