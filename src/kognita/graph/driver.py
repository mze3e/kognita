"""Graphiti instance construction and Kognita's custom Kuzu driver."""
from __future__ import annotations

import os
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Iterator

import kuzu
from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.kuzu_driver import KuzuDriver as _GraphitiKuzuDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.graph_queries import get_fulltext_indices
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig

from kognita.config import EmbedderConfig, LLMConfig
from kognita.exceptions import ConfigError


def _llm_client_for(provider: str) -> Any:
    """Import a Graphiti LLM client on demand.

    Provider SDKs are extras, so importing them all at module scope would make
    ``kognita[graph]`` useless without also installing every provider. Each is
    resolved only when it is actually selected, and a missing one names the
    extra that supplies it.
    """
    extras = {
        "anthropic": ("graphiti_core.llm_client.anthropic_client", "AnthropicClient", "anthropic"),
        "openai": ("graphiti_core.llm_client.openai_client", "OpenAIClient", "openai"),
        "gemini": ("graphiti_core.llm_client.gemini_client", "GeminiClient", "gemini"),
        "groq": ("graphiti_core.llm_client.openai_generic_client", "OpenAIGenericClient", "groq"),
        "ollama": ("graphiti_core.llm_client.openai_generic_client", "OpenAIGenericClient", "openai"),
        "custom": ("graphiti_core.llm_client.openai_generic_client", "OpenAIGenericClient", "openai"),
    }
    if provider not in extras:
        raise ConfigError(f"Unsupported LLM provider: {provider}")
    module_path, attr, extra = extras[provider]
    try:
        return getattr(import_module(module_path), attr)
    except ImportError as exc:
        raise ConfigError(
            f"The '{provider}' provider is not installed. "
            f"Install it with:  pip install kognita[{extra}]"
        ) from exc


def _reranker_client() -> Any:
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

    return OpenAIRerankerClient


@contextmanager
def _database_factory_returning(database: Any) -> Iterator[None]:
    """Make ``kuzu.Database(...)`` yield ``database`` for the duration of the block.

    Graphiti's own ``KuzuDriver.__init__`` constructs a ``kuzu.Database`` from a path,
    and two handles on one path do **not** share state — a table created through
    one is invisible to the other, silently, with no error. Sharing a single
    handle is therefore mandatory, not merely tidy.

    Intercepting the constructor is deliberately the narrowest possible hook: it
    depends only on ``kuzu.Database(path)`` being called, not on the rest of the
    driver's constructor body, which changes between graphiti-core releases.
    """
    original = kuzu.Database
    kuzu.Database = lambda *args, **kwargs: database  # type: ignore[assignment]
    try:
        yield
    finally:
        kuzu.Database = original  # type: ignore[assignment]


class KuzuDriver(_GraphitiKuzuDriver):
    """Kuzu driver that installs Graphiti's full-text indexes and can share a handle.

    Pass ``database=`` to bind the driver to a :class:`~kognita.graph.session.KuzuSession`'s
    live handle, so Graphiti's tables and the deterministic ``SoR_*`` mirror
    co-tenant one database and a single Cypher traversal can cross both planes.
    Pass ``db=`` (a path) for standalone use, exactly as before.
    """

    def __init__(
        self,
        db: str = ":memory:",
        max_concurrent_queries: int = 1,
        *,
        database: Any = None,
    ) -> None:
        if database is None:
            super().__init__(db=str(db), max_concurrent_queries=max_concurrent_queries)
            return
        # Construction is a one-shot at startup and callers reach it through
        # KuzuSession, whose lock serializes it.
        with _database_factory_returning(database):
            super().__init__(db=str(db), max_concurrent_queries=max_concurrent_queries)

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
    *,
    session: Any = None,
) -> Graphiti:
    """Construct a Graphiti instance wired to the configured LLM and embedder.

    The embedder is always an ``OpenAIEmbedder`` — we rely on every supported
    backend (OpenAI, Ollama, local server, custom) exposing an OpenAI-compatible
    ``/v1/embeddings`` endpoint.

    Pass ``session=`` (a :class:`~kognita.graph.session.KuzuSession`) to share its
    single database handle, so the LLM-extracted knowledge graph and the
    deterministic ``SoR_*`` mirror live in one database. Without it the driver
    opens ``db_path`` itself — correct only when nothing else holds that path.
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
        llm_client = _llm_client_for("anthropic")(
            config=GraphitiLLMConfig(api_key=llm.api_key, model=model)
        )
        cross_encoder = None
    elif llm.provider == "openai":
        llm_client = _llm_client_for("openai")(
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
        llm_client = _llm_client_for(llm.provider)(config=cfg)
        cross_encoder = _reranker_client()(client=llm_client.client, config=cfg)
    elif llm.provider == "gemini":
        llm_client = _llm_client_for("gemini")(
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
        llm_client = _llm_client_for(llm.provider)(config=cfg)
        cross_encoder = _reranker_client()(client=llm_client.client, config=cfg)
    else:
        raise ConfigError(f"Unsupported LLM provider: {llm.provider}")

    if not llm.resolved_use_reranker():
        cross_encoder = None

    kwargs = {}
    if cross_encoder is not None:
        kwargs["cross_encoder"] = cross_encoder

    driver = KuzuDriver(
        db=str(db_path),
        database=session.database if session is not None else None,
    )
    return Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=graphiti_embedder,
        **kwargs,
    )
