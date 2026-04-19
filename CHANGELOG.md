# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-19

### Added

- `Kognita` async context manager for ingesting text into a knowledge graph and running semantic search
- `KognitaConfig`, `LLMConfig`, `EmbedderConfig` dataclasses for type-safe configuration
- Multi-provider LLM support: Anthropic (Claude), OpenAI (GPT), Groq, Google Gemini, Ollama, and any OpenAI-compatible custom endpoint
- Multi-provider embedding support: OpenAI, Ollama, and a local CPU embedding server
- `KuzuDB` embedded graph database — no Docker or external services required
- Automatic entity deduplication and merging via Graphiti across text chunks
- `GraphSnapshot` with save/load and MD5-based PDF deduplication
- `execute_cypher()` for direct Cypher queries against the KuzuDB instance
- `list_models()` helper to fetch available model IDs from each provider's API
- `Node`, `Edge`, `SearchResult`, `EpisodeResult` dataclasses for structured graph data
- `KognitaError`, `ProviderError`, `ConfigError` exception hierarchy
- Streamlit demo application (`examples/streamlit_app/`) with interactive graph visualisation, LLM playground, Cypher explorer, and export
- Local CPU embedding server (`examples/local_embedding_server/`) powered by FastAPI and `sentence-transformers`

[Unreleased]: https://github.com/mze3e/kognita/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mze3e/kognita/releases/tag/v0.1.0
