"""A structural locator for one ingested passage.

``charspan`` is a document-global range into the source text, so a reader
can slice that text and land on the passage. ``block_charspan`` is the
parser's own span for the opening block, kept when the parser reported one.
Nothing here fills a missing page or span with a guess.
"""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from kognita.exceptions import KognitaError

_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".html", ".htm"})
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass(frozen=True)
class SectionLocator:
    """Where a passage sits in its source document.

    ``charspan`` counts into :func:`source_text`. ``block_charspan`` is the
    parser's ``(start, end)`` pair for one block, into that block's own
    text, and is ``None`` when the parser did not report a complete pointer.
    ``bbox`` is ``(l, t, r, b)`` in the parser's coordinates.
    """

    document: str
    page: int | None
    section_path: tuple[str, ...]
    charspan: tuple[int, int] | None
    bbox: tuple[float, float, float, float] | None
    block_charspan: tuple[int, int] | None = None

    def as_source_label(self) -> str:
        """The readable citation stored on ``KnowledgeItem.source_label``.

        The character numbers are the document-global range.
        """
        parts = [self.document]
        if self.section_path:
            parts.append(" / ".join(self.section_path))
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.charspan is not None:
            start, end = self.charspan
            parts.append(f"chars {start}-{end}")
        return ", ".join(parts)


def source_text(path: Path) -> str:
    """The text a citation's character range is counted in.

    Markdown and HTML are the file, decoded as UTF-8. A DOCX range is into
    the paragraph text of ``word/document.xml`` in reading order, paragraphs
    joined by newlines, because the package is not a character file. A file
    that is not UTF-8, or a docx without that part, is refused.
    """
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _docx_text(path)
    if suffix in _TEXT_SUFFIXES:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise KognitaError(
                "document is not UTF-8; ingestion does not guess an encoding "
                "and indexes nothing"
            ) from exc
    raise KognitaError(
        "no character coordinate space for "
        f"{suffix or '(none)'}; ingestion indexes nothing"
    )


def _docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            try:
                payload = archive.read("word/document.xml")
            except KeyError as exc:
                raise KognitaError(
                    "docx has no word/document.xml; ingestion does not invent "
                    "a character span and indexes nothing"
                ) from exc
    except zipfile.BadZipFile as exc:
        raise KognitaError("docx could not be read; ingestion indexes nothing") from exc
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise KognitaError(
            "docx document.xml could not be read; ingestion indexes nothing"
        ) from exc
    paragraphs = [
        "".join(node.text or "" for node in paragraph.iter(_W + "t"))
        for paragraph in root.iter(_W + "p")
    ]
    return "\n".join(paragraphs)


__all__ = ["SectionLocator", "source_text"]
