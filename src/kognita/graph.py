"""Graphiti instance construction and Kognita's custom Kuzu driver."""
from __future__ import annotations

import os

import kuzu
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.graph_queries import get_fulltext_indices
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from kognita.config import EmbedderConfig, LLMConfig
from kognita.exceptions import ConfigError


class KognitaKuzuDriver(KuzuDriver):
    """Kuzu driver that also installs Graphiti's full-text indexes."""

    def setup_schema(self) -> None:
        super().setup_schema()
        conn = kuzu.Connection(self.db)
        try:
            for query in get_fulltext_indices(GraphProvider.KUZU):
                try:
                    conn.execute(query)
                except RuntimeError as exc:
                    if "already exists" not in str(exc):
                        raise
        finally:
            conn.close()


def make_graphiti(
    llm: LLMConfig,
    embedder: EmbedderConfig,
    db_path: str,
) -> Graphiti:
    """Construct a Graphiti instance wired to the configured LLM and embedder.

    The embedder is always an ``OpenAIEmbedder`` — we rely on every supported
    backend (OpenAI, Ollama, local server, custom) exposing an OpenAI-compatible
    ``/v1/embeddings`` endpoint.
    """
    embed_api_key = embedder.api_key or os.environ.get("OPENAI_API_KEY", "")
    graphiti_embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=embed_api_key,
            embedding_model=embedder.model,
            embedding_dim=embedder.dimension,
            base_url=embedder.base_url or None,
        )
    )

    model = llm.model
    if llm.provider == "anthropic":
        llm_client = AnthropicClient(
            config=GraphitiLLMConfig(api_key=llm.api_key, model=model)
        )
        cross_encoder = None
    elif llm.provider == "openai":
        llm_client = OpenAIClient(
            config=GraphitiLLMConfig(api_key=llm.api_key, model=model)
        )
        cross_encoder = None
    elif llm.provider == "groq":
        cfg = GraphitiLLMConfig(
            api_key=llm.api_key,
            model=model,
            small_model=model,
            base_url=llm.base_url or "https://api.groq.com/openai/v1",
        )
        llm_client = OpenAIGenericClient(config=cfg)
        cross_encoder = OpenAIRerankerClient(client=llm_client.client, config=cfg)
    elif llm.provider == "gemini":
        llm_client = GeminiClient(
            config=GraphitiLLMConfig(api_key=llm.api_key, model=model)
        )
        cross_encoder = None
    elif llm.provider in ("ollama", "custom"):
        if not llm.base_url:
            raise ConfigError(
                f"LLMConfig.base_url is required for provider '{llm.provider}'."
            )
        cfg = GraphitiLLMConfig(
            api_key=llm.api_key or "ollama",
            model=model,
            small_model=model,
            base_url=llm.base_url,
        )
        llm_client = OpenAIGenericClient(config=cfg)
        cross_encoder = OpenAIRerankerClient(client=llm_client.client, config=cfg)
    else:
        raise ConfigError(f"Unsupported LLM provider: {llm.provider}")

    if not llm.resolved_use_reranker():
        cross_encoder = None

    kwargs = {}
    if cross_encoder is not None:
        kwargs["cross_encoder"] = cross_encoder

    return Graphiti(
        graph_driver=KognitaKuzuDriver(db=str(db_path)),
        llm_client=llm_client,
        embedder=graphiti_embedder,
        **kwargs,
    )
