"""Word-based text chunking for graph ingestion."""
from __future__ import annotations


def chunk_text(
    text: str,
    size: int = 220,
    overlap: int = 25,
    *,
    min_chars: int = 60,
) -> list[str]:
    """Split ``text`` into overlapping word-windows.

    Chunks shorter than ``min_chars`` (after stripping) are discarded so that
    trailing whitespace/fragment windows don't produce empty LLM episodes.
    """
    words = text.split()
    step = max(1, size - overlap)
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if len(chunk.strip()) > min_chars:
            chunks.append(chunk)
        i += step
    return chunks
