"""Configuration for the Graphiti + Kuzu knowledge engine."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from kognita.config import EmbedderConfig, LLMConfig, LLMProvider
from kognita.exceptions import ConfigError


@dataclass(frozen=True)
class GraphConfig:
    """Top-level configuration for a :class:`~kognita.graph.engine.GraphEngine` instance."""

    llm: LLMConfig
    embedder: EmbedderConfig
    db_path: str | Path = ".kognita_db"
    chunk_size_words: int = 220
    chunk_overlap_words: int = 25

    @classmethod
    def from_env(cls) -> "GraphConfig":
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
