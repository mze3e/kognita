# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-04

### Added

- **Decision engine core**: `decide()` function that determines whether an AI request is permitted before any data is retrieved
- `Envelope` dataclass for structured access requests (principal, purpose, tool, subject, etc.)
- Policy-based access control with multi-regime support (HK SFC, DIFC DFSA, etc.)
- Evidence recording: every decision is logged with its basis (which policy/regime allowed or denied it)
- `EvidenceWriter` and `EvidenceReader` for decision auditability
- Two-signature approvals (ADR 0006): separation of duties for high-risk actions
- Proposal-then-apply pattern (ADR 0007): propose changes, verify, then execute atomically
- Identity and authority model (ADR 0005) with role-based access control
- Tool arguments channel (ADR 0004): structured, hashable arguments for reproducibility
- Governance module with rule sets, regimes, and policy evaluation
- Retrieval integration: controlled access to knowledge graphs (optional `[graph]` extra)
- SoR (System of Record) mirror architecture for safe separation of concerns
- Import layering contracts via `import-linter` to prevent architectural drift
- CLI with `kognita` command and subcommands
- Comprehensive test suite (75+ tests) with BMOS conformance validation

### Changed

- Refactored from PDF→graph library to decision engine for AI governance
- Core now depends only on pydantic, sqlmodel, numpy, python-dotenv
- Graph, embeddings, and LLM features moved to optional extras: `[graph]`, `[vec]`, `[openai]`, etc.

## [0.1.0] - 2026-04-19

### Added

- Initial release of Kognita as a PDF→knowledge graph library
- Graphiti-based entity and relationship extraction
- KuzuDB embedded graph persistence
- Multi-provider LLM support (Anthropic, OpenAI, Groq, Gemini, Ollama)
- Streamlit demo application

[Unreleased]: https://github.com/mze3e/kognita/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mze3e/kognita/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mze3e/kognita/releases/tag/v0.1.0
