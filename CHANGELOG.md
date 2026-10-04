# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pinned evidence for replay. Every policy check records the hash of the policy row as evaluated, and replay fails if that row's content no longer matches. An in-place edit of an effective policy is refused; a change is a new effective-dated row via `supersede_policy`. `RETRIEVAL` records a content hash and embedding model for each returned item. `MODEL_CALL` records the provider, the model name and version reported by the provider, the prompt template version, and hashes of the prompt as sent and the response as received. `TOOL_CALL` and `EGRESS` record a hash of the tool response. Prompts, responses, and source snapshots live in a content-addressed retention store keyed by those hashes, with a retention period per use case. Erasure deletes the bytes and appends an `ERASURE` event; the chain keeps the hash and still verifies.

- Flagship demo: `kognita scaffold --template governed-agent` creates a small app and a SQLite policy and evidence store. `run_demo.py` runs four scenarios: an allowed read with evidence, a denial with citations and no retrieval, redaction of `ana@example.org` through the AI gateway into a local stand-in (evidence stores the manifest hash, not the prompt), and an edited evidence row that `kognita evidence verify` reports as a break.

- MCP proxy: `kognita serve --mcp --root-config <config>`. The config names the backend MCP servers, the policy pack, the evidence database, and the default actor context. Every MCP call becomes an envelope and is authorised with `run_governed()`. The backend is contacted only when that call is released: an allow, or a human approval that already has a live grant. A denial, an escalation, and an ungranted human approval return the outcome and the citations and do not call the backend. A released call returns the proxied result. `POLICY_DECISION` evidence stores a hash and size of the tool arguments, not the arguments. Free-text tool arguments are classified only when a typed classification is missing; the classifier does not decide. A call with no agent name is denied. An agent name is accepted only when the bound client configuration lists it. An approved system trigger is a system actor. If the evidence store cannot record the call, the proxy refuses it and does not forward.

- AI gateway: `kognita serve --provider openai-compatible --upstream <url>`. Agents point `base_url` at the local gateway. The proxy speaks the OpenAI-compatible wire format, decides before any byte is forwarded, redacts through the egress guard, restores redacted spans, and classifies the response. `MODEL_CALL` and `EGRESS` evidence record hashes and references, not content. Token totals and cost reported by the provider count against the Run budget. A call with no agent name is denied, and an agent name is accepted only when the bound client configuration lists it. An approved system trigger is admitted and is not treated as a human. If the evidence store cannot record the call, the gateway refuses it.

- Classifier-derived envelopes at the decision boundary. `ask` classifies the question, and `run_governed` classifies free-text arguments, using the pattern classifier already in core. Typed attributes stay authoritative. Identity, purpose and subject references are not taken from the text. The recorded model, version, label, calibrated confidence and input hash are what `decide()` replays, so replay does not run the classifier. Below a `CLASSIFIER_CONFIDENCE` policy threshold the outcome is ESCALATE, never ALLOW. A citation that acted on a classifier label names both the policy rule and that label.

- `Run` budgets on `run_governed` and `ask`: call count, wall clock, classification ceiling, and cost when the caller already knows it. Token spend is recorded when the call site supplies it. Exceeding a budget is a DENY that cites that budget, and the consumption is written to the evidence chain.
- Durable suspend and resume for `HUMAN_APPROVAL`. A hold checkpoints at the policy decision and stores a content-hash continuation; `continue_run(run_id, approvals_resolved={approval_id: True/False})` resumes it. Evidence records the decision requested, then approved or rejected, then resumed.

### Fixed

- `HUMAN_APPROVAL` no longer retrieves or returns data from `ask`. A held tool call or retrieval runs only after the approval is actually granted; a denied approval does not execute.

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
