"""Embedders.

:class:`HashingEmbedder` is the default and is not a placeholder. It is
deterministic, needs no model, no network and no API key, and produces the same
vector on every machine — which makes it the right fixture for a conformance
suite and a reasonable offline default for small corpora. A real embedder is a
configuration swap, not a rewrite: both satisfy
:class:`kognita.protocols.Embedder`.

Vectors are L2-normalised, so cosine similarity is a dot product.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

#: Matches the tokenisation used for both embedding and lexical overlap.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of length two or more."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


def lexical_overlap(query: str, text: str) -> float:
    """Share of the query's distinct tokens that appear in ``text``.

    Complements the semantic score in hybrid retrieval: it rewards exact term
    matches that a low-dimensional embedding blurs together.
    """
    q = set(tokenize(query))
    if not q:
        return 0.0
    t = set(tokenize(text))
    return len(q & t) / len(q)


class HashingEmbedder:
    """Deterministic feature-hashing embedder over unigrams and bigrams.

    Each feature hashes to one dimension and a sign, so unrelated texts collide
    only by accident and identical texts always produce identical vectors. No
    text leaves the process.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return f"hashing-{self._dimension}"

    def _features(self, tokens: list[str]) -> Iterable[str]:
        yield from tokens
        for a, b in zip(tokens, tokens[1:]):
            yield f"{a} {b}"

    def embed(self, text: str) -> list[float]:
        vector = np.zeros(self._dimension, dtype=np.float64)
        for feature in self._features(tokenize(text)):
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Cosine similarity, safe for zero vectors."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denominator)


def to_bytes(vector: list[float]) -> bytes:
    """Pack a vector for storage as a BLOB."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_bytes(blob: bytes) -> list[float]:
    """Unpack a stored BLOB."""
    return np.frombuffer(blob, dtype=np.float32).astype(np.float64).tolist()


__all__ = [
    "HashingEmbedder",
    "tokenize",
    "lexical_overlap",
    "cosine",
    "to_bytes",
    "from_bytes",
]
