"""Embedders backed by a hosted or self-hosted model.

Every supported backend — OpenAI, Ollama, a local server, anything
OpenAI-shaped — speaks the same ``/v1/embeddings`` contract, so one adapter
covers all of them and switching is a base URL rather than a code change.

This satisfies the same :class:`kognita.core.protocols.Embedder` protocol as
:class:`~kognita.core.embedding.HashingEmbedder`, so retrieval, the vector index
and the conformance suite are unaware of which is in use.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from kognita.config import EmbedderConfig, openai_compatible_base_url
from kognita.exceptions import ProviderError


class OpenAICompatibleEmbedder:
    """Calls any OpenAI-compatible ``/v1/embeddings`` endpoint.

    Uses :mod:`urllib` rather than an SDK: the request is one POST, and adding a
    provider SDK to reach it would put a dependency in the path of every
    deployment that only wanted embeddings.
    """

    def __init__(self, config: EmbedderConfig, *, timeout: float = 30.0) -> None:
        if config.provider != "openai" and not config.base_url:
            raise ProviderError(
                f"EmbedderConfig.base_url is required for provider '{config.provider}'."
            )
        self.config = config
        self.timeout = timeout
        base = config.base_url or "https://api.openai.com/v1"
        self._url = openai_compatible_base_url(base) + "/embeddings"

    @property
    def dimension(self) -> int:
        return self.config.dimension

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def is_local(self) -> bool:
        """Whether this endpoint sits inside the trust boundary.

        The egress guard uses this to decide whether text may be sent unredacted.
        """
        return self.config.is_local()

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ProviderError(
                f"HTTP {exc.code} from {self._url}: {detail}", status_code=exc.code
            ) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ProviderError(f"Could not reach {self._url}: {exc}") from exc

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body = self._post({"model": self.config.model, "input": texts})
        try:
            rows = sorted(body["data"], key=lambda d: d.get("index", 0))
            vectors = [list(row["embedding"]) for row in rows]
        except (KeyError, TypeError) as exc:
            raise ProviderError(
                f"Unexpected embeddings response from {self._url}: {str(body)[:300]}"
            ) from exc

        if vectors and len(vectors[0]) != self.config.dimension:
            # A dimension mismatch silently poisons a shared index: some vectors
            # would score against the query and others would be skipped.
            raise ProviderError(
                f"Model '{self.config.model}' returned {len(vectors[0])}-dimensional "
                f"vectors but EmbedderConfig declares {self.config.dimension}. "
                "Fix the dimension and re-index."
            )
        return vectors


__all__ = ["OpenAICompatibleEmbedder"]
