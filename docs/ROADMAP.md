# Kognita Roadmap: From Governance Engine to Agent Harness

**Vision:** Transform Kognita from a policy decision engine into a complete agent harness: a framework anyone can adopt to build governed AI agents with proof of permission and tamper-evident audit logs.

**Current Status:** v0.2.0 (core governance engine complete). Starting from a foundation of fail-closed decisions, cited policies, and tamper-evident evidence, we build outward to become the standard way organizations safely deploy agentic systems.

**Last revised:** September 2026. See [Decision Log](#decision-log) for what changed and why.

---

## Strategic Direction

Kognita's core strength is **gating**: authorization before execution, with every decision citable to policy and every decision chain tamper-evident. Current gaps prevent adoption as a harness:

- No **Run context** to group and budget tool calls across agent steps
- No **gateway** in front of model providers; the egress guard exists but callers must remember to use it, and any code importing a provider SDK directly bypasses it
- No way to govern **free text**: `decide()` needs a typed `Envelope`, but traffic through a proxy arrives as raw prompts
- No **MCP integration** for agent-agnostic governance (Claude, OpenAI, Cursor, Claude Desktop agents all speak MCP)
- No **adapters** for existing frameworks (Claude Agent SDK, LangGraph, Pydantic AI, LangChain)
- No **policy language** or CLI for declaring and versioning rules without Python code
- **Weak citations for retrieved content**: `KnowledgeItem` carries provenance as a flat `source_label` string, the returned snippet is the document's first 280 characters rather than the passage that matched, and classification is one value per document

The wedge to adoption: explicit proxies. `kognita serve` fronts an MCP server or a model provider, and every call is authorized and evidenced with zero changes to agent code.

### Scope boundary

Kognita governs **AI and agent traffic**: agent-to-tool calls, agent-to-model calls, and agent-to-agent messages. It is **not** a general enterprise integration platform. Protocol mediation, data transformation, and connector catalogs (the MuleSoft, Kong, Apigee space) are out of scope. In that market the differentiator is connector count; in ours it is fail-closed decisions with citations and a tamper-evident chain.

---

## Release Timeline

```
Q4 2026     Q1 2027     Q2 2027     Q3 2027     Q4 2027
│           │           │           │           │
├─ 0.3 ─────┼─ 0.4 ─────┼─ 0.5 ─────┼─ 0.6 ─────┼─► 1.0
  Gateways    Ingestion   Fleets      Trust       Production
  & the Run   & Policy                            ready
              Language
```

---

## 0.3 "Gateways and the Run" (Critical Launch Release)

**Timeline:** Q4 2026
**Goal:** Make Kognita the default harness for agentic governance. Every call an agent makes, to a tool or to a model, passes through a Kognita gateway first.

### 1. Run Context and Budgets

- `Run` dataclass grouping calls across agent steps, with:
  - `max_calls`: deny beyond this count (prevents runaway loops)
  - `max_tokens`: LLM token spend
  - `max_cost_usd`: hard stop on spend
  - `wall_clock_seconds`: timeout for the whole run
  - `classification_ceiling`: calls may only touch data up to this sensitivity
  - `approvals_pending`: open HUMAN_APPROVAL decisions awaiting action
- `run_governed(run=…)` accepts a Run; budget consumption is written to the evidence chain
- Exceeding any budget is a DENY citing the budget, not a warning

**Rationale:** Agents make many calls; budgets prevent runaway spend and recursive loops. Organizations require cost limits before deployment.

### 2. Durable Suspend/Resume for HUMAN_APPROVAL

- A HUMAN_APPROVAL decision checkpoints the run and persists a continuation
- **Checkpoints happen synchronously at policy evaluation boundaries**, not continuously. The points where a run must be able to pause are exactly the points where `decide()` runs, so those are the only persistence points. (Pattern observed in cavemem's lifecycle-hook capture; idea only, no code taken.)
- **The continuation is stored as a content-hash handle into a local store, not inlined** in the run record. The run stays small and diffable, and rehydration is an explicit, evidenced fetch. This extends the existing evidence principle that payloads hold hashes and references rather than content.
- Caller resumes with `continue_run(run_id, approvals_resolved={approval_id: True/False})`
- Evidence shows: decision requested → approved or denied → resumed
- Closes the Tier 0 defect "HUMAN_APPROVAL withholds nothing" and "approval loop unclosed"

**Rationale:** The approval flow is currently open-loop. Fixing it unblocks regulated use cases where an action must have explicit human sign-off.

### 3. AI Gateway: Explicit Proxy in Front of Model Providers

Replaces the previously planned in-process "governed model wrapper." One component, same shape as the MCP proxy.

- `kognita serve --provider openai-compatible --upstream https://api.openai.com`
- Agents set `base_url=http://localhost:PORT/v1` instead of the provider URL. **Explicit, not transparent:** no TLS interception, no certificates to distribute. It governs everything configured to use it.
- **OpenAI-compatible wire format first.** OpenAI, Groq, Ollama, vLLM and LiteLLM-fronted models all speak it, so one adapter covers most deployments. A native Anthropic Messages adapter follows when a deployment needs it.
- Request path, reusing existing machinery:
  1. Parse the request body just enough to build an `Envelope`: principal and purpose from authenticated headers or a bound session, `tool="model_call"`, subject is the model name
  2. Derive remaining attributes from the prompt text (see item 4)
  3. `decide()` against the caller's policy pack; a denial returns before any bytes leave the gateway
  4. On allow, redact through the existing egress guard
  5. Forward to the real provider; restore redacted spans in the response via `Redactor.restore()`
  6. Classify the **response** too before returning it; answers are governed, not only questions
  7. Emit `MODEL_CALL` and `EGRESS` evidence by hash and reference, never content
- Token usage feeds the Run budget

**Rationale:** The egress guard and redaction already exist but are opt-in per call site. A gateway makes governance the path of least resistance and removes the "forgot to wrap it" bypass. The only genuinely new code is the wire-format adapter.

**Landscape:** LiteLLM proxy, Portkey, Cloudflare AI Gateway and Kong AI Gateway occupy the "gateway in front of model providers" category. They compete on routing, caching and observability. None make fail-closed decisions with citations and a tamper-evident chain; that is the opening.

### 4. Classifier-Derived Envelopes

Traffic through a gateway arrives as free text, with no typed purpose, subject or classification. A fast calibrated text classifier fills those in so policy applies dynamically to questions and answers, without callers constructing typed requests.

**The classifier builds the envelope. It never makes the decision.** Rules that must hold:

- **Typed envelopes stay authoritative.** When a caller supplies typed attributes, they win. Classification fills gaps only.
- **Classifier output is evidence.** Record model identifier and version, label, calibrated confidence, and input hash. `decide()` runs deterministically on the recorded attributes, and replay re-uses the recorded label rather than re-running the model. This keeps decisions pure and replayable.
- **Low confidence escalates.** Confidence thresholds are policy. Below threshold, the outcome is ESCALATE, never ALLOW. This is the existing fail-closed rule applied to uncertainty.
- **Classifier attributes may only narrow permission, never widen it.** Text can be written to fool a classifier (prompt injection). Anything tied to identity comes from authentication and cannot be overridden by what the text says.
- **Citations have two parts:** the policy rule, and the classifier label it acted on. "Denied under rule X because the prompt was classified as client PII at 0.94" is auditable; "denied because the model said so" is not.

**Implementation:** behind the existing `Classifier` protocol, as an optional extra, never in the core. Candidate backends:

| Backend | Where it runs | Concern |
|---|---|---|
| Rule and pattern classifier (ships in core) | In process | Low recall; a floor, not a guarantee |
| Laya (Apache 2.0, local BERT-family model) | In process | Requires torch, transformers and a model download; days old at time of writing |
| Jev (TypeSafe AI, hosted API) | Third-party service | Sends the content being governed to a third party *before* deciding whether it may leave; circular for most regulated deployments |

### 5. MCP Proxy Server

- `kognita serve --mcp --root-config config.json` fronts one or more MCP servers
- Config names backend server(s), policy pack, evidence database, and default actor context
- Every MCP call is translated to an `Envelope`, authorized via `run_governed()`, proxied if allowed, and evidenced; denials return the outcome and citations
- MCP tool arguments are free text too, so item 4 applies here as well

**Rationale:** MCP is the lingua franca for agent integrations. A proxy makes governance transparent to every MCP-speaking agent without framework lock-in. Note that many popular agent tools now ship MCP servers themselves, which makes them governance *targets* rather than dependencies.

### 6. Flagship Demo

- `kognita scaffold --template governed-agent` creates a small app, a SQLite policy and evidence store, and three scenarios:
  1. Allowed query: agent reads permitted data, evidence logged
  2. Denied query: agent asks for another client's data, denied with citations, nothing retrieved
  3. Tampering: user edits a row in the evidence database; `kognita evidence verify` reports the break
- A fourth scenario through the AI gateway: a prompt containing client PII is redacted before reaching the provider, and the evidence shows the manifest hash, not the content

### Tier 0 Defect Closure

From `docs/gap-analysis-bmos.md`:
- [ ] HUMAN_APPROVAL withholds nothing (item 2)
- [ ] Approval loop unclosed (item 2)
- [ ] Entitlements fail open
- [ ] SqliteVecIndex silent failure
- [ ] Classifiers never invoked (item 4 makes them load-bearing)
- [ ] `engages` missing from protocol
- [ ] No foreign keys on evidence references

### Definition of Done

- [ ] All tests pass, including new coverage for Run, suspend/resume, AI gateway, classifier-derived envelopes, MCP proxy
- [ ] Replay test: a decision made from a classifier-derived envelope replays identically without calling the classifier
- [ ] Injection test: text crafted to relabel itself cannot widen permission
- [ ] Conformance kit passes
- [ ] Gateway overhead benchmarked (target: under 50 ms per call, excluding classifier inference)
- [ ] Flagship demo runs end-to-end in under 3 minutes from scaffold
- [ ] Import contracts pass: the core still has no optional dependencies
- [ ] All Tier 0 defects closed

---

## 0.4 "Ingestion and Policy Language"

**Timeline:** Q1 2027
**Goal:** Make citations real down to the passage, let non-engineers author and review policy, and meet developers in the frameworks they already use.

### 1. Structured Document Ingestion (Docling)

Fixes the weak-citation gap. Docling's `ProvenanceItem` carries `page_no`, `bbox` and `charspan` on every item, plus `parent`/`children` references and section heading levels. Provenance is derived by the parser, not guessed by an LLM, so it is verifiable against the source file.

- `KnowledgeItem` gains a structural locator (document reference, page, section path, character span) replacing the flat `source_label`
- **Classification per section subtree**, not per document: a handbook's public chapters and restricted annex can be indexed with different ceilings
- The returned snippet becomes the passage that matched, not `body[:280]`
- A Kognita-owned chunker walks the parsed tree and splits on section boundaries (replaces the 26-line word-window chunker)

**Dependency discipline:**
- Depend on `docling-slim` with named format extras only (e.g. `format-docx`, `format-html`, `format-markdown`, `format-pdf`). Never the `docling` metapackage, which pulls torch.
- Do **not** take `feat-chunking`; it pulls transformers and tree-sitter grammars. Chunking is ours.
- Pin exactly. Docling releases very frequently and has shipped regressions that broke all conversions across several versions.
- Verify licenses of `docling-parse` and its model packages before shipping the PDF extra.

### 2. The Graph Extra: Decide Its Fate

`kognita.graph` (Graphiti + Kuzu) has no functional test coverage, its headline SoR mirror is unimplemented, and its only integration into a governed answer is a node and edge count in a summary string. It also hard-pins `graphiti-core==0.28.2` and forces `openai<2` on users' environments.

This release decides between:
- **Finish it:** functional tests, implement the SoR mirror, lift the pins; or
- **Extract it** into a separate package so its pins stop travelling with Kognita, with structured ingestion (item 1) becoming the primary retrieval path.

The default, absent a concrete deployment needing cross-plane Cypher, is extraction.

### 3. Policy Language: YAML and CLI

`Policy` rows are already declarative data (`regime`, `rule_type`, `applies_to`, JSON `rule`, `citation`, `effective_from`). The YAML format is a **serialization of rows that already exist**, not a new rule engine.

Design borrowed from OpenSpec (ideas only):

- **Change files contain deltas, not whole policies.** A policy change is a document of `## ADDED`, `## MODIFIED` and `## REMOVED` rules. This is the primitive behind `kognita policy diff`, and two concurrent changes to one policy set do not conflict.
- **Lifecycle:** propose → review → `validate --strict` → apply, which merges the delta into current policy and archives the change with its date
- **Referential integrity:** a MODIFIED or REMOVED rule whose identifier matches nothing in the current policy set is a validation error. (OpenSpec's own validator misses this case.)

Commands:
- `kognita policy load` — parse, type-check, compile to rows
- `kognita policy diff` — show the delta between two versions
- `kognita policy validate [--strict]` — fails on uncited rules, untested rules, dangling references, and unresolved clarification markers
- `kognita policy explain --envelope …` — which rules engage and why

**Clarification markers** (from spec-kit): a rule may contain `[NEEDS CLARIFICATION: question]`. `validate --strict` refuses to pass while any remain, so an ambiguous rule cannot silently ship.

**Rule expressions:** evaluate using the existing `rule_type` registry. A future `rule_type` may embed a restricted expression language such as Cedar (principal/action/resource shape, designed to be analyzable) for conditions; the surrounding outcome ordering, citation and evidence stay Kognita's. Replacing `decide()` with a general policy engine is not planned (see [Evaluated and Rejected](#evaluated-and-rejected)).

### 4. Policy Test Format

Colocated with policy files, in a scenario shape under a named rule:

```yaml
rule: client-email-isolation
scenarios:
  - name: client can read own email
    given: { principal: "alice@corp", subject_type: client, subject_id: alice }
    when:  { tool: read_email }
    then:  { decision: ALLOW }

  - name: client cannot read another client's email
    given: { principal: "alice@corp", subject_type: client, subject_id: bob }
    when:  { tool: read_email }
    then:  { decision: DENY, cites: client-email-isolation }
```

- `kognita policy test` runs all scenarios
- A rule with no scenarios fails `validate --strict`

### 5. Starter Policy Packs

Templates: role-based access, geo-fencing, data classification, time-gated access, two-signature approval chains. Each pack is policy files, scenarios, and a fixture pack.

**Distribution** (pattern from Fabric; idea only):
- One directory per pack; name-based lookup; no central registry
- `kognita packs update` fetches from a configurable git repository
- A user overlay directory shadows built-in packs on name collision and is never touched by updates, so private packs coexist with public ones
- **Unlike Fabric, packs are versioned**: each carries a version, and `policy diff` works across pack versions

### 6. Framework Adapters (under 200 lines each)

- **hermes-agent first.** It is MIT licensed and already has an approvals mode, a dangerous-command list, and local security logs, but no rule citation, no escalation tier and no tamper-evident chain. Its approval hook is the natural injection point.
- Claude Agent SDK, OpenAI Agents SDK, LangGraph, Pydantic AI, LangChain

### 7. Flight Recorder

- Local dashboard: recent runs filtered by actor, tool and outcome; drill-down into each decision, its citations and evidence
- Export a run as self-verifying JSON
- Alerts: budget exceeded, repeated denials from one actor, chain break detected, classifier confidence drift

### Definition of Done

- [ ] Docling-backed ingestion with per-section classification and passage-level citations
- [ ] Graph extra either tested and unpinned, or extracted
- [ ] Policy language: load, diff, validate, explain, test
- [ ] Five starter packs with scenarios
- [ ] hermes-agent adapter plus at least three others
- [ ] Flight recorder with export

---

## 0.5 "Fleets" (Multi-Agent Governance)

**Timeline:** Q2 2027
**Goal:** Deploy multiple governed agents with shared policies and governed agent-to-agent communication.

### 1. Delegation with Attenuation
- `Run.delegate_to(agent_name, attenuated_scope=…)`; the child's budget and scope are carved from the parent's and can only narrow
- Every delegation is evidenced with the authority transferred

### 2. Capability Grants
- `grant_to(grantee, capability, duration, subject_scope)`; issue, use, revoke and expiry are all evidenced
- Revocation takes effect on the next decision

### 3. Agent Manifests
- Declarative YAML: identity, capabilities, dependencies, budgets
- `kognita fleet deploy --manifest …`; signed once 0.6 lands

### 4. Fleet Controls
- Per-agent quotas, run isolation, and authorization of agent-to-agent calls through the same gateway path as agent-to-tool calls
- Agent-to-agent message bodies are free text, so classifier-derived envelopes (0.3 item 4) apply

### 5. Sandbox Constraints as Decision Output
An ALLOW for code execution can carry constraints the executing sandbox must honor: network block-all, a network allow-list, an auto-stop interval, and CPU, memory and disk limits. The vocabulary is borrowed from existing sandbox APIs; Kognita emits constraints and evidences them, and does not ship a sandbox.

### Definition of Done
- [ ] Delegation, grants, manifests, fleet controls
- [ ] Agent-to-agent calls authorized and evidenced
- [ ] Sandbox constraints emitted on ALLOW for execution tools
- [ ] Scenarios covering delegation chains, grant expiry, quota exhaustion

---

## 0.6 "Trust" (Cryptographic Proof)

**Timeline:** Q3 2027
**Goal:** Make evidence tamper-proof and verifiable by outside parties.

- **Ed25519 signing:** each evidence event signed; `kognita evidence verify --public-key` walks the chain and verifies every signature
- **Postgres backend:** concurrency for many writers, replication, row-level security so agents cannot read each other's runs
- **External review API:** a reviewer requests a period, annotates decisions, and submits a signed review into the chain
- **TypeScript evidence verifier:** `@kognita/verify`, so auditors can verify without installing Python
- **Schema versioning:** evidence events carry a schema version; migrations rename, recreate, copy intersecting columns and drop inside one transaction, and old events stay readable

---

## 1.0 "Production Ready"

**Timeline:** Q4 2027

- **Benchmarks:** decision latency p50 and p99, evidence write throughput, gateway overhead, resume time; published with hardware specs
- **Docs rewrite:** lead with the problem ("prove an AI request was allowed before any data moved"), scenarios by industry, glossary, honest comparison with content guardrails and AI gateways
- **Ten single-file examples**, each runnable in under five minutes and tested in CI, including the AI gateway, the MCP proxy, a policy-only YAML deployment, an approval workflow, and an evidence audit
- **Graduation checklist:** coverage above 85 percent, published benchmarks, external security review, at least one production deployment in a regulated domain

---

## Evaluated and Rejected

Projects assessed in September 2026 and the reason each was not adopted. "Idea only" means a design was borrowed with no code or dependency taken.

| Project | Considered for | Verdict |
|---|---|---|
| OpenViking | Replacing the knowledge graph | Rejected. A server-only agent context database, not a graph: no query language, no bi-temporal facts, AGPL-3.0. |
| Laya | Replacing `decide()` | Rejected as an engine. It is a text classifier with no rule model, no citations, no evidence. Kept as a candidate `Classifier` backend (0.3 item 4). |
| Jev (TypeSafe AI) | Replacing `decide()` | Rejected as an engine; hosted service. Candidate `Classifier` backend only where sending content to a third party is acceptable. |
| Open Policy Agent, Cedar | Replacing `decide()` | Not planned. Both are permit/deny kernels without escalation tiers, citations or evidence. Cedar remains a candidate expression language inside a rule. |
| PageIndex | Document structure | Rejected. Page indices are assigned by an LLM, so provenance is unverifiable; hard-pins `litellm`. |
| mem0 | Run state and memory | Rejected. Telemetry and network clients in its core install; LLM round-trip on the write path. |
| headroom | Context compression | Rejected. `litellm` in core; lossy compression before the model sees content conflicts with attestable evidence. Content-hash handle idea adopted (0.3 item 2). |
| caveman / cavemem | Run state | Rejected as a dependency; engine is BSL-1.1. Lifecycle-boundary checkpoint idea adopted (0.3 item 2). |
| Daytona | Sandboxes | Rejected. AGPL-3.0, and public development stopped in June 2026. Constraint vocabulary borrowed (0.5 item 5). |
| Scrapling | Web data | Rejected. Anti-bot evasion tooling does not belong in a compliance product's dependency tree. |
| OpenSpec, spec-kit, Fabric | Policy language and packs | Ideas only (0.4 items 3 to 5). |
| TrendRadar, hyperframes, OpenMontage, AI Engineering Hub | — | Out of scope (news aggregation, video, tutorials). |

---

## Go-to-Market

### Positioning

**For developers:** point your agent's base URL at Kognita and get cited decisions and tamper-evident audit logs. No framework swap.

**For compliance teams:** every decision cites the rule it came from, including when the input was free text.

**For operators:** budget agent spend per run and per agent; one misconfigured agent cannot drain the API budget.

### Adoption Sequence

1. **0.3:** launch with the AI gateway and MCP proxy demo; five minutes end to end
2. **0.4:** starter packs and adapters lower the cost of trying it
3. **0.5:** fleets make multi-agent deployments safe
4. **0.6:** signatures and external review satisfy regulated buyers
5. **1.0:** production-ready, benchmarked, documented

### Why Kognita Wins

1. **No framework lock-in:** explicit proxies and small adapters
2. **Fail-closed by design:** governance is the execution path, not a filter bolted on after
3. **Every decision is cited:** including decisions on free text, with the classifier label named
4. **Tamper-evident:** the audit trail cannot be edited retroactively
5. **Minimal core:** four dependencies, no network; everything heavy is an extra

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Framework fatigue | Explicit proxies need no code changes; adapters stay under 200 lines |
| Classifier errors become policy errors | Classifier output only narrows permission; low confidence escalates; labels are evidenced and replayable |
| Gateway adds latency | Benchmark every release; gateway overhead target excludes classifier inference, which is reported separately |
| AI gateway incumbents add policy features | Move fast on 0.3; the citation and evidence model is the part that is hard to retrofit |
| Heavy optional dependencies leak into core | Import-linter contracts and the no-extras install test stay mandatory in CI |

---

## Decision Log

**September 2026**
- **AI gateway replaces the in-process governed model wrapper.** Explicit proxy (base URL), not transparent TLS interception. OpenAI-compatible format first.
- **Classifier-derived envelopes** added to 0.3 so policy applies to free text. The classifier builds the envelope and never decides; its output is evidence; low confidence escalates; it may only narrow permission.
- **Scope bounded** to AI and agent traffic; general enterprise integration is out of scope.
- **Docling-backed structured ingestion** added to 0.4 to make citations passage-level and classification per section.
- **Graph extra** to be finished or extracted in 0.4; extraction is the default.
- **Policy language design** informed by OpenSpec (delta changes, strict validation), spec-kit (clarification markers) and Fabric (pack distribution).
- **hermes-agent** chosen as the first adapter.
- **Timeline corrected.** The first version of this document dated 0.3 at Q4 2025, which had already passed; releases now run Q4 2026 through Q4 2027.
