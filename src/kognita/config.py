"""Configuration dataclasses and provider model-listing helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import requests

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


@dataclass(frozen=True)
class EmbedderConfig:
    """Configuration for the embedding model used for vector search."""

    provider: EmbedderProvider
    model: str
    dimension: int
    api_key: str = ""
    base_url: str | None = None


@dataclass(frozen=True)
class KognitaConfig:
    """Top-level configuration for a Kognita instance."""

    llm: LLMConfig
    embedder: EmbedderConfig
    db_path: str | Path = ".kognita_db"
    chunk_size_words: int = 220
    chunk_overlap_words: int = 25

    @classmethod
    def from_env(cls) -> "KognitaConfig":
        """Best-effort construction from environment variables.

        Picks the first LLM provider with a matching ``*_API_KEY`` env var and
        pairs it with OpenAI embeddings when ``OPENAI_API_KEY`` is set.
        """
        providers: list[tuple[LLMProvider, str, str]] = [
            ("anthropic", "ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022"),
            ("openai", "OPENAI_API_KEY", "gpt-4o-mini"),
            ("groq", "GROQ_API_KEY", "llama-3.1-70b-versatile"),
            ("gemini", "GOOGLE_API_KEY", "gemini-1.5-flash"),
        ]
        llm: LLMConfig | None = None
        for provider, env_var, default_model in providers:
            key = os.environ.get(env_var, "")
            if key:
                llm = LLMConfig(provider=provider, api_key=key, model=default_model)
                break
        if llm is None:
            raise ConfigError(
                "No LLM provider API key found in env. Set one of "
                "ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY."
            )

        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key:
            raise ConfigError(
                "OPENAI_API_KEY is required for the default embedder; "
                "either set it or construct EmbedderConfig manually."
            )
        embedder = EmbedderConfig(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
            api_key=openai_key,
        )
        return cls(llm=llm, embedder=embedder)


def _openai_compatible_base_url(base_url: str) -> str:
    """Return a base URL ending at the OpenAI-compatible ``/v1`` path."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return normalized + "/v1"


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
    underlying HTTP client directly.
    """
    try:
        if provider == "anthropic":
            if not api_key:
                return []
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            r = requests.get("https://api.anthropic.com/v1/models", headers=headers, timeout=timeout)
            if r.status_code != 200:
                return []
            return [m["id"] for m in r.json().get("data", [])]

        if provider == "openai":
            if not api_key:
                return []
            r = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
            if r.status_code != 200:
                return []
            return sorted(
                m["id"]
                for m in r.json().get("data", [])
                if m["id"].startswith(("gpt-", "chatgpt-"))
            )

        if provider == "groq":
            if not api_key:
                return []
            r = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
            if r.status_code != 200:
                return []
            return [m["id"] for m in r.json().get("data", [])]

        if provider == "gemini":
            if not api_key:
                return []
            r = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": api_key},
                timeout=timeout,
            )
            if r.status_code != 200:
                return []
            return [
                m["name"].replace("models/", "")
                for m in r.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]

        if provider == "ollama":
            if not base_url:
                return []
            url = _openai_compatible_base_url(base_url) + "/models"
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                return []
            return [m["id"] for m in r.json().get("data", [])]

        if provider == "custom":
            if not base_url:
                return []
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            r = requests.get(base_url.rstrip("/") + "/models", headers=headers, timeout=timeout)
            if r.status_code != 200:
                return []
            return [m["id"] for m in r.json().get("data", [])]

        raise ConfigError(f"Unsupported provider: {provider}")
    except requests.RequestException:
        return []
