"""Tests for config dataclasses — no network calls required."""
import pytest

from kognita import EmbedderConfig, KognitaConfig, LLMConfig
from kognita.exceptions import ConfigError


def test_llm_config_basic():
    cfg = LLMConfig(provider="anthropic", api_key="sk-test", model="claude-3-5-sonnet-20241022")
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "sk-test"
    assert cfg.model == "claude-3-5-sonnet-20241022"
    assert cfg.base_url is None
    assert cfg.use_reranker is None


def test_llm_config_resolved_reranker_defaults():
    groq = LLMConfig(provider="groq", api_key="gsk-test", model="llama3-70b")
    assert groq.resolved_use_reranker() is True

    anthropic = LLMConfig(provider="anthropic", api_key="sk-test", model="claude-3-5-sonnet-20241022")
    assert anthropic.resolved_use_reranker() is False


def test_llm_config_reranker_override():
    cfg = LLMConfig(provider="anthropic", api_key="sk-test", model="claude-3-5-sonnet-20241022", use_reranker=True)
    assert cfg.resolved_use_reranker() is True


def test_embedder_config_basic():
    cfg = EmbedderConfig(provider="openai", model="text-embedding-3-small", dimension=1536, api_key="sk-test")
    assert cfg.provider == "openai"
    assert cfg.dimension == 1536
    assert cfg.base_url is None


def test_embedder_config_local():
    cfg = EmbedderConfig(
        provider="local",
        model="bge-small-en-v1.5",
        dimension=384,
        base_url="http://localhost:8000/v1",
    )
    assert cfg.provider == "local"
    assert cfg.dimension == 384


def test_kognita_config_defaults():
    llm = LLMConfig(provider="openai", api_key="sk-test", model="gpt-4o-mini")
    embedder = EmbedderConfig(provider="openai", model="text-embedding-3-small", dimension=1536, api_key="sk-test")
    cfg = KognitaConfig(llm=llm, embedder=embedder)
    assert cfg.chunk_size_words == 220
    assert cfg.chunk_overlap_words == 25
    assert str(cfg.db_path) == ".kognita_db"


def test_kognita_config_custom_path():
    llm = LLMConfig(provider="openai", api_key="sk-test", model="gpt-4o-mini")
    embedder = EmbedderConfig(provider="openai", model="text-embedding-3-small", dimension=1536, api_key="sk-test")
    cfg = KognitaConfig(llm=llm, embedder=embedder, db_path="/tmp/my_graph", chunk_size_words=300)
    assert str(cfg.db_path) == "/tmp/my_graph"
    assert cfg.chunk_size_words == 300


def test_config_is_frozen():
    llm = LLMConfig(provider="openai", api_key="sk-test", model="gpt-4o-mini")
    with pytest.raises((TypeError, AttributeError)):
        llm.api_key = "other"  # type: ignore[misc]


def test_from_env_no_keys(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ConfigError):
        KognitaConfig.from_env()


def test_from_env_llm_but_no_openai(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    for key in ("OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        KognitaConfig.from_env()


def test_from_env_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    cfg = KognitaConfig.from_env()
    assert cfg.llm.provider == "anthropic"
    assert cfg.embedder.provider == "openai"
