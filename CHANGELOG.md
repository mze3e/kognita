# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- The Graphiti + Kuzu graph engine: `kognita.graph`, the `[graph]` extra, and the `[anthropic]`, `[groq]` and `[gemini]` extras, which installed only Graphiti. The Streamlit graph demo under `examples/` is removed with it. Kognita holds policies, guidelines and governance records only (ADR 0008). Touching a removed name such as `kognita.Kognita` or `kognita.GraphEngine` raises an `AttributeError` that names ADR 0008; anyone who still needs the engine can pin `kognita[graph]==0.3.0`. `kognita doctor` no longer lists graph, Anthropic, Groq or Gemini packages. `graphiti_core` and `kuzu` stay on the core's forbidden-import list.

### Changed

- The `openai` extra no longer caps the SDK below version 2. The cap existed for Graphiti. Kognita does not import the SDK (the embedder adapter uses `urllib`); the extra is for agents that call the AI gateway with it. openai 3.28 was checked through `kognita serve`: the provider received the redacted prompt, the client received the restored values, and a call with no agent name was denied without reaching the provider.
- The README and the package description were brought up to date for 0.3.0. The roadmap status line now says v0.3.0 is shipped.
- `docs/control-frameworks.md` maps the MAS Guidelines on Artificial Intelligence Risk Management (7 October 2026) paragraph by paragraph, with timing against the MAS deadlines. "Covered" now means shipped in v0.3.0 or earlier.
- The roadmap adopts the twelve gaps that mapping found. 0.4's use-case register gains a designated control function, complexity as a sixth materiality axis, inherent and residual materiality, quantitative risk-appetite measures, a single-provider dependency count, shared inventory identifiers, unmediated AI entries, pilot status and complete retirement; 0.4 revocation gains `--provider` and kill-switch drills; 0.6 fairness reporting covers proxy attributes. No release dates changed.

## [0.3.0] - 2026-10-06

### Added

- Gateway overhead on `kognita serve`. The benchmark times an allowed `POST /v1/chat/completions` through the gateway built by `kognita serve --provider openai-compatible --upstream https://api.openai.com`, with `--purpose COLLABORATION`, `--purposes COLLABORATION`, `--agent dossier-agent`, and principal `alice`. The prompt is `Email ana@example.org about the notes`. `_urllib_transport` forwards the call to a local stand-in that returns a fixed completion, so the provider is not called. Overhead for a call is its wall time minus time inside `PatternClassifier.classify` and `PatternClassifier.calibrated_confidence` (the prompt and the response) minus time inside the stand-in. The decision path is the one already on main. The test fails when a call's overhead is not under 50 ms. On Linux 6.12.94+ x86_64, 4 CPUs, Intel(R) Xeon(R) Processor, MemTotal 16398384 kB, a sample of 30 sequential calls, none dropped, measured a maximum overhead of 37.010 ms and a mean of 9.469 ms. Classifier inference on that run averaged 0.041 ms per call, and the stand-in averaged 0.001 ms per call.

- Gateway failure mode, per use case. A use case is the purpose string already recorded as `use_case`. The default is `FAIL_CLOSED`: if the evidence store cannot record the call, the gateway refuses it and does not call the provider. `DEGRADED` may proceed only for a local model and content below classification C2 (C2 is client-identifying), and writes the decision and the model evidence once the store accepts writes again. A call is never forwarded without a decision. `kognita serve --failure-mode` sets the mode for `--purpose`.

- Reconstruction report: `kognita evidence reconstruct <interaction_id>` answers the ten reconstruction-test questions from the evidence chain and the retention store. The same command writes JSON and a readable document. It checks the chain and every pinned hash while building the report; a mismatch is a finding and the rest of the report is still written. Questions that belong to origination, RM review capture, and governed client communication are present and marked "not recorded". The report does not re-run the classifier, the model, or a tool.

- Pinned evidence for replay. Every policy check records the hash of the policy row as evaluated, and replay fails if that row's content no longer matches. An in-place edit of an effective policy is refused; a change is a new effective-dated row via `supersede_policy`. `RETRIEVAL` records a content hash and embedding model for each returned item. `MODEL_CALL` records the provider, the model name and version reported by the provider, the prompt template version, and hashes of the prompt as sent and the response as received. `TOOL_CALL` and `EGRESS` record a hash of the tool response. Prompts, responses, and source snapshots live in a content-addressed retention store keyed by those hashes, with a retention period per use case. Erasure deletes the bytes and appends an `ERASURE` event; the chain keeps the hash and still verifies.

- Flagship demo: `kognita scaffold --template governed-agent` creates a small app and a SQLite policy and evidence store. `run_demo.py` runs four scenarios: an allowed read with evidence, a denial with citations and no retrieval, redaction of `ana@example.org` through the AI gateway into a local stand-in (evidence stores the manifest hash, not the prompt), and an edited evidence row that `kognita evidence verify` reports as a break.

- MCP proxy: `kognita serve --mcp --root-config <config>`. The config names the backend MCP servers, the policy pack, the evidence database, and the default actor context. Every MCP call becomes an envelope and is authorised with `run_governed()`. The backend is contacted only when that call is released: an allow, or a human approval that already has a live grant. A denial, an escalation, and an ungranted human approval return the outcome and the citations and do not call the backend. A released call returns the proxied result. `POLICY_DECISION` evidence stores a hash and size of the tool arguments, not the arguments. Free-text tool arguments are classified only when a typed classification is missing; the classifier does not decide. A call with no agent name is denied. An agent name is accepted only when the bound client configuration lists it. An approved system trigger is a system actor. If the evidence store cannot record the call, the proxy refuses it and does not forward.

- AI gateway: `kognita serve --provider openai-compatible --upstream <url>`. Agents point `base_url` at the local gateway. The proxy speaks the OpenAI-compatible wire format, decides before any byte is forwarded, redacts through the egress guard, restores redacted spans, and classifies the response. `MODEL_CALL` and `EGRESS` evidence record hashes and references, not content. Token totals and cost reported by the provider count against the Run budget. A call with no agent name is denied, and an agent name is accepted only when the bound client configuration lists it. An approved system trigger is admitted and is not treated as a human. If the evidence store cannot record the call, the gateway refuses it.

- Classifier-derived envelopes at the decision boundary. `ask` classifies the question, and `run_governed` classifies free-text arguments, using the pattern classifier already in core. Typed attributes stay authoritative. Identity, purpose and subject references are not taken from the text. The recorded model, version, label, calibrated confidence and input hash are what `decide()` replays, so replay does not run the classifier. Below a `CLASSIFIER_CONFIDENCE` policy threshold the outcome is ESCALATE, never ALLOW. A citation that acted on a classifier label names both the policy rule and that label.

- `Run` budgets on `run_governed` and `ask`: call count, wall clock, classification ceiling, and cost when the caller already knows it. Token spend is recorded when the call site supplies it. Exceeding a budget is a DENY that cites that budget, and the consumption is written to the evidence chain.
- Durable suspend and resume for `HUMAN_APPROVAL`. A hold checkpoints at the policy decision and stores a content-hash continuation; `continue_run(run_id, approvals_resolved={approval_id: True/False})` resumes it. Evidence records the decision requested, then approved or rejected, then resumed.

### Fixed

- The 0.3 rollup "All Tier 0 defects closed" is ticked. Each box under "Tier 0 Defect Closure" was already ticked, and each claim still matches this commit. `HUMAN_APPROVAL` withholds the tool and the retrieval until a live grant. `continue_run` closes the approval loop. An empty or missing zone list is not visible. `SqliteVecIndex.search` returns the candidate stored with the embedding. Classifiers run at the decision boundary, and a typed classification still wins. `DomainPack.engages` is on the protocol. Evidence references are foreign keys. A missing or empty purpose list fails closed. Through the AI gateway and the MCP proxy, a call with no agent name is a denial, a registered agent and an approved system trigger are not, and an agent name outside the bound client configuration is a denial. The decision path was not changed.

- The conformance kit already passes. `TestDemoPackConformance` runs every `ConformanceCase` against the demo fixture pack in the normal test suite: fail-closed outcome precedence, a pure and deterministic `decide`, a citation on every check, a denial that names its basis, an unregistered agent denied, recorded decisions and refusals, an intact evidence chain, and a human approval bound to the envelope hash. None of those cases is skipped or expected-fail. The decision path was not changed.

- Text that relabels itself cannot widen permission. A classification written into the text, including every value from `C0` to `C3`, is not applied as a typed classification. The pattern classifier's hint may raise a classification and may not lower it, and `decide` on that label is not more permissive than the unhinted label under the same policy. A typed classification still replaces the label, including a lower one. The decision path was not changed.

- `DomainPack.engages` is already on the protocol, and the gap note that it is missing is stale. `run_governed`, `ask`, and the test harness pass the pack's `engages` into `decide`. A policy the pack does not engage is skipped. A policy it engages is evaluated. A pack that omits the method fails `isinstance(pack, DomainPack)`. The call sites still use `getattr`, so a pack that never meets that check still evaluates every effective policy. The decision path was not changed.

- Entitlement filtering already fails closed, and the gap note that describes an empty `zones` list as visible in every zone is stale. A missing or empty zone list is not visible in any zone. A caller with no zone does not match a listed zone. A classification or ceiling the filter cannot evaluate does not return the item. A zone the item lists, at or below the ceiling, is still returned, including the ceiling `retrieve` already grants when none is passed. The decision path is unchanged.

- Evidence payload references are foreign keys, enforced on the SQLite connection `make_engine` opens. `approval_id`, `policy_id`, `successor_id`, `run_id`, and `continuation_hash` are columns on `evidence_events`. Each `checks[].policy_id` is a row in `evidence_checks`, and each `items[].id` / `returned_ids` entry is a row in `evidence_items`. A dangling id is rejected. `runs.continuation_hash` references `continuations`. Content hashes stay in the payload: erasure deletes the bytes and the chain keeps the hash. `approvals.decision_id` and `entity_edges.from_entity_id` / `to_entity_id` already enforced; they are unchanged.

- `index_item` and `reindex` maintain `knowledge_vec` when given a `SqliteVecIndex`. Replacing an embedding deletes that item's previous vec row before the new one is inserted. `search` only reads. A hit is the candidate whose id was stored on the row, so two items that share embedding bytes stay distinct, and the vec `rowid` is not `item.id`. The score is `1 - distance`. A KNN with no rows, or only neighbours outside the candidate set, returns `[]`. `retrieve` still ranks that empty vector result lexically.

- The purpose check fails closed when the purpose list is missing or empty. A configured list still allows a listed purpose and denies an unlisted one. `kognita serve` takes a repeatable `--purposes` allowlist for the AI gateway, separate from `--purpose`. The MCP root config takes a `purposes` list, separate from `actor.purpose`. With neither list set, those processes deny the call.

- Degraded mode, while the evidence store is down, no longer lets a classification header keep a call below C2. The header is a floor. The classifier runs on the JSON values that would be forwarded after parsing, so client-identifying content outside the extracted prompt, including the OpenAI `user` field, is refused and the provider is not called. A unicode escape of that content cannot stay below C2.

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
- Comprehensive test suite (75+ tests) with governance conformance validation

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

[Unreleased]: https://github.com/mze3e/kognita/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/mze3e/kognita/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mze3e/kognita/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mze3e/kognita/releases/tag/v0.1.0
