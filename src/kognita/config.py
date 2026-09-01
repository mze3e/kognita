"""Provider configuration dataclasses and model-listing helpers.

Deliberately dependency-free: these types are shared by ``kognita.core`` (which
must import on the four hard dependencies alone), ``kognita.graph`` and
``kognita.adapters``. Model discovery uses :mod:`urllib.request` rather than
``requests`` so that nothing here pulls a transitive dependency tree.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from kognita.exceptions import ConfigError

LLMProvider = Literal["anthropic", "openai", "groq", "gemini", "ollama", "custom"]
EmbedderProvider = Literal["openai", "ollama", "local", "custom"]


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the LLM used to extract entities and relationships."""

    provider: LLMProvider
    api_key: str
    model: str
    base_url: str | None = None
    use_reranker: bool | None = None  # None = provider default; True/False overrides

    def resolved_use_reranker(self) -> bool:
        if self.use_reranker is not None:
            return self.use_reranker
        return self.provider in ("groq", "ollama", "custom")

    def is_local(self) -> bool:
        """True when this provider runs inside the trust boundary.

        Used by :mod:`kognita.core.egress` to decide whether classified content
        may be sent to it unredacted. Only providers reached over loopback (or an
        explicitly declared local host) count; a cloud endpoint never does.
        """
        return _is_local_base_url(self.provider, self.base_url)


@dataclass(frozen=True)
class EmbedderConfig:
    """Configuration for the embedding model used for vector search."""

    provider: EmbedderProvider
    model: str
    dimension: int
    api_key: str = ""
    base_url: str | None = None

    def is_local(self) -> bool:
        """True when this embedder runs inside the trust boundary."""
        return _is_local_base_url(self.provider, self.base_url)


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal")


def _is_local_base_url(provider: str, base_url: str | None) -> bool:
    """Classify a provider endpoint as inside or outside the trust boundary.

    ``local`` is local by definition. Everything else is judged by its host:
    absent a base URL the provider is a hosted API, so it is *not* local. This
    fails closed on purpose — a misconfigured endpoint must never be mistaken
    for one that keeps data in-zone.
    """
    if provider == "local":
        return True
    if not base_url:
        return False
    return (urlparse(base_url).hostname or "").lower() in _LOCAL_HOSTS


def openai_compatible_base_url(base_url: str) -> str:
    """Return a base URL ending at the OpenAI-compatible ``/v1`` path."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return normalized + "/v1"


def _get_json(url: str, headers: dict[str, str], timeout: float) -> dict | None:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def list_models(
    provider: LLMProvider,
    api_key: str = "",
    base_url: str | None = None,
    *,
    timeout: float = 10.0,
) -> list[str]:
    """Return the list of model IDs advertised by ``provider``.

    Returns an empty list on network / auth errors — callers that need to
    distinguish missing credentials from real failures should call the
    underlying HTTP endpoint directly.
    """
    if provider == "anthropic":
        if not api_key:
            return []
        data = _get_json(
            "https://api.anthropic.com/v1/models",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout,
        )
        return [m["id"] for m in (data or {}).get("data", [])]

    if provider == "openai":
        if not api_key:
            return []
        data = _get_json(
            "https://api.openai.com/v1/models",
            {"Authorization": f"Bearer {api_key}"},
            timeout,
        )
        return sorted(
            m["id"]
            for m in (data or {}).get("data", [])
            if m["id"].startswith(("gpt-", "chatgpt-"))
        )

    if provider == "groq":
        if not api_key:
            return []
        data = _get_json(
            "https://api.groq.com/openai/v1/models",
            {"Authorization": f"Bearer {api_key}"},
            timeout,
        )
        return [m["id"] for m in (data or {}).get("data", [])]

    if provider == "gemini":
        if not api_key:
            return []
        data = _get_json(
            "https://generativelanguage.googleapis.com/v1beta/models",
            {"x-goog-api-key": api_key},
            timeout,
        )
        return [
            m["name"].replace("models/", "")
            for m in (data or {}).get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]

    if provider == "ollama":
        if not base_url:
            return []
        data = _get_json(openai_compatible_base_url(base_url) + "/models", {}, timeout)
        return [m["id"] for m in (data or {}).get("data", [])]

    if provider == "custom":
        if not base_url:
            return []
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        data = _get_json(base_url.rstrip("/") + "/models", headers, timeout)
        return [m["id"] for m in (data or {}).get("data", [])]

    raise ConfigError(f"Unsupported provider: {provider}")
