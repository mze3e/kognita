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
- **Not yet examinable**: a regulator cannot select one AI-assisted client interaction and reconstruct it end to end (see [Supervisory Examinability](#supervisory-examinability))

The wedge to adoption: explicit proxies. `kognita serve` fronts an MCP server or a model provider, and every call is authorized and evidenced with zero changes to agent code.

### Target state

**Governable → Explainable → Controllable → Replayable → Resilient.**

The direction of travel in supervision is from AI governance *documentation* toward supervisory *examinability*: not "do you have a policy" but "show me this interaction, and prove it". Every release below is measured against these five properties.

**Governance by architecture, not by instruction.** Never rely on the LLM to enforce a control that can be enforced outside it. Not "please don't access restricted information" but an entitlement check; not "don't trade more than $100,000" but a limit in the action path; not "ask the RM before sending" but a communication API that requires approval; not "only use current data" but expired sources rejected in code. Where a model's judgment is needed, it supplies an attribute and a deterministic policy decides.

| Property | Meaning | Where it is delivered |
|---|---|---|
| Governable | Every AI use case is registered, and every action is authorized before it happens | Core today; use-case register (0.4) |
| Explainable | Why this client, why this product, which controls ran, which rule decided | Citations today; origination evidence (0.4) |
| Controllable | Budgets, approvals, delegation limits, a human decision point, automatic halts when harm spreads, bounded agent authority | Run and approvals (0.3); risk-based review and circuit breaker (0.4); agent authority, autonomy levels and blast radius (0.5) |
| Replayable | The bank can reproduce the evidence later, exactly | Pinned evidence and reconstruction (0.3) |
| Resilient | Governance survives outages, key compromise and supplier failure | Gateway failure mode (0.3); resilience track (0.7) |

### Scope boundary

Kognita governs **AI and agent traffic**: agent-to-tool calls, agent-to-model calls, and agent-to-agent messages. It is **not** a general enterprise integration platform. Protocol mediation, data transformation, and connector catalogs (the MuleSoft, Kong, Apigee space) are out of scope. In that market the differentiator is connector count; in ours it is fail-closed decisions with citations and a tamper-evident chain.

---

## Supervisory Examinability

The first regulated use case is relationship-manager (RM) facing AI in wealth and private banking. The requirements below apply to any regulated deployment.

### The record chain

For every RM AI use case, Kognita must record:

**use case → affected clients → investor impact → data used → model → agent authority → recommendation or action → human decision point → evidence retained**

### The reconstruction test

A regulator selects one AI-assisted client interaction. Kognita must answer each question from evidence alone, and verify the chain while doing so.

| Question | Status at v0.2 | Delivered by |
|---|---|---|
| Why this client? | Gap: the subject is recorded, not why it was selected | Origination evidence (0.4) |
| Why this insight or product? | Partial: eligibility checks say why it was *permitted*, not why it was *recommended* | Origination evidence (0.4) |
| What information did the AI use? | Partial: retrieved item IDs are logged, content is neither hashed nor immutable | Pinned evidence (0.3); structured ingestion (0.4) |
| What model or agent produced it? | Gap: agent name only; no model name or version | Pinned evidence (0.3) |
| What was the agent authorized to do? | Covered: roles, scopes, cited checks | Manifests and grants (0.5) extend it |
| What suitability or policy controls ran? | Covered: regime, citation and policy ID per check; policy content not pinned | Pinned evidence (0.3) |
| What did the RM see and change? | Partial: a proposal model with before-state exists; RM edits are not captured | RM review capture (0.4) |
| Who made the final decision? | Partial: recorded only when policy forced an approval | RM review capture (0.4) |
| What was communicated to the client? | Gap: client communication is not a governed event | Governed client communication (0.4) |
| Can the bank reproduce that evidence later? | Partial: chain integrity and point-in-time policy replay exist; policies, sources and model I/O are not pinned | Pinned evidence and reconstruction (0.3) |

The acceptance test for 1.0 is this table with every row covered, demonstrated by `kognita evidence reconstruct` on a real interaction.

### The five charges

Five AI governance questions a supervisor such as the FCA could ask today. Each is mapped to what answers it.

| Charge | Kognita's answer | Delivered by |
|---|---|---|
| **Unsupervised AI agents:** who is accountable when no one oversees what the AI does? | Every action is authorized before it runs, naming the agent and the principal it acts for; every use case names an accountable owner, cited on each decision; a final decision is recorded on every path | Core today; Run (0.3); accountable owner and RM review capture (0.4) |
| **No retrievable audit trail:** can you show what the AI said and why? | "Why" is cited and tamper-evident today; "what it said" is retained by hash and reconstructable | Pinned evidence, retention store, reconstruction (0.3) |
| **3% sampling treated as oversight:** are you reviewing enough to catch what matters? | 100% of governed actions are checked *before* they happen; every response is classified and every high-risk one goes to a human, replacing random sampling with risk-based review and reported coverage | Classifier-derived envelopes (0.3); risk-based review (0.4) |
| **Poor guidance scaling unchecked:** when AI gets it wrong, how fast does harm spread? | A circuit breaker halts a use case or model version automatically when flags cross a threshold, and an affected-client query lists everyone who received its output | Pinned evidence (0.3); circuit breaker and affected-client query (0.4) |
| **Models that cannot explain themselves:** can you trace where the AI came from and how it works? | Provenance and approval: which model and version produced each output, its model card, and that it was approved for this use case. Explaining a model's internal reasoning is out of scope, and Kognita does not claim it | Pinned evidence (0.3); model card in the use-case register (0.4); provider register (0.7) |

Kognita checks *permission*, not *advice quality*. Charge 3 is answered by routing every high-risk interaction to a human reviewer, not by Kognita judging the advice itself.

### Bank-grade control framework

A consolidated control framework for agentic AI in private banking, drawing on MAS and ABS material, NIST and OWASP concepts, and production patterns from large banks, defines 40 control domains and 12 non-negotiable controls. The full mapping of all 40 domains is in [control-framework.md](control-framework.md).

Its overarching rule is Kognita's founding premise:

> **Never rely on the LLM to enforce a control that can be enforced outside the LLM.** Govern the chain from human intent → agent authority → data → reasoning → tools → action → outcome, not the model.

| # | Non-negotiable control | Status at v0.2 | Delivered by |
|---|---|---|---|
| 1 | Unique agent identity | Partial: agent registry with owner, version, tier and kill switch exists; a call with no agent name skips it | Tier 0 (0.3); agent identity (0.5) |
| 2 | Explicit delegated authority | Gap | Delegated authority (0.5) |
| 3 | Purpose-bound access | Partial: purpose recorded, vocabulary fails open | Tier 0 (0.3); use-case register (0.4) |
| 4 | Least-privilege data and tool entitlements | Partial: zones and ceilings filter data before scoring | Tool allow-lists and field minimization (0.5) |
| 5 | Trusted client and portfolio truth layer | Boundary: Kognita enforces the fact contract, the bank owns the data | Fact provenance and freshness (0.6) |
| 6 | Evidence and provenance for material claims | Partial: decision citations, retrieval evidence | Pinned evidence (0.3); claim provenance (0.6) |
| 7 | Fact, inference and recommendation separation | Gap | Claim types (0.6) |
| 8 | Deterministic financial calculations | Boundary: Kognita requires figures to cite a registered calculation service | Calculation provenance (0.6) |
| 9 | Runtime policy enforcement outside the LLM | **Covered** | Core |
| 10 | Sandboxed, bounded execution | Partial | Blast-radius limits and sandbox constraints (0.5) |
| 11 | Human approval for high-impact decisions | Partial: HUMAN_APPROVAL outcome and two-signature approvals | Suspend/resume (0.3); autonomy levels (0.5) |
| 12 | Complete action audit trail and kill switch | Partial: hash-chained evidence, per-agent kill switch | Reconstruction (0.3); granular revocation (0.4) |

**Boundary** marks controls whose capability belongs to another system: the truth layer, calculation engines, evaluation harnesses, sandboxes and challenger models. Kognita's role in each is to refuse an output unless the authoritative system was used, and to record that it was. Building those systems inside Kognita would make it the thing being governed.

### Wealth management AI control framework

A second consolidated framework, covering RM-facing, client-facing and agentic AI, defines 70 controls in 10 domains, drawing on MAS, PDPC, SFC, FCA, the Bank of England, APRA and ASIC, CSBS, BIS/FSI and ESMA. Not every control is mandated everywhere today; it is used as the target baseline. The full mapping is in [wealth-ai-control-framework.md](wealth-ai-control-framework.md).

It sets one overarching test, which Kognita adopts as the measure for every release:

> **Can the bank explain, control, stop and reconstruct every material AI action that affects a client?**

| # | Non-negotiable control | Status after the previous audits | Delivered by |
|---|---|---|---|
| 1 | AI use-case inventory and materiality classification | Partial: single risk tier planned | Multi-axis classification and risk appetite (0.4) |
| 2 | Named business owner | Planned | Use-case register (0.4) |
| 3 | Unique agent identity | Partial: agent names are self-asserted, not authenticated | Tier 0 (0.3); agent credentials (0.5) |
| 4 | Explicit delegated authority | Planned | Delegated authority (0.5) |
| 5 | Least-privilege data and tool access | Planned | 0.5 |
| 6 | Separation of recommendation and execution authority | Partial | Per-verb action permissions (0.5) |
| 7 | Runtime policy enforcement | **Covered** | Core |
| 8 | Human approval for consequential client actions | Planned | 0.3; 0.4; 0.5 |
| 9 | "Why this client?" and "why this recommendation?" | Planned | Origination, including "do not contact" (0.4) |
| 10 | Approved-source grounding and research expiry | Partial | Approved sources and supersession (0.6) |
| 11 | Complete provenance and audit trail | Planned | 0.3; 0.4; 0.6 |
| 12 | Pause, override, revoke and kill | Partial | Intervention controls (0.4) |
| 13 | Agent-behaviour monitoring and automated containment | Partial | Containment sequence (0.5) |
| 14 | Third-party dependency and exit controls | Partial | Dependency map and concentration risk (0.7) |
| 15 | Continuous testing, revalidation and incident learning | Partial | Near misses (0.4); adversarial suite (0.5); revalidation (0.6) |

**Two naming collisions.** The framework's C0 to C5 client-impact scale is unrelated to Kognita's C1 to C3 data classification, and its L0 to L5 communication scale is unrelated to Kognita's L0 to L6 autonomy levels. This roadmap calls them the **client-impact class** and the **communication level**, and uses their names rather than codes.

---

## Release Timeline

```
Q4 2026     Q1 2027     Q2 2027     Q3 2027     Q4 2027     Q1 2028
│           │           │           │           │           │
├─ 0.3 ─────┼─ 0.4 ─────┼─ 0.5 ─────┼─ 0.6 ─────┼─ 0.7 ─────┼─► 1.0
  Gateways,   Ingestion,  Agents,     Claims      Trust &     Production
  the Run &   Policy &    Authority               Resilience  ready and
  Replay      Client      & Fleets                            examinable
              Lifecycle
```

---

## 0.3 "Gateways, the Run and Replay" (Critical Launch Release)

**Timeline:** Q4 2026
**Goal:** Make Kognita the default harness for agentic governance. Every call an agent makes, to a tool or to a model, passes through a Kognita gateway first, and every decision can be reproduced exactly later.

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

### 7. Pinned Evidence for Replay

Today the chain proves that records were not altered, but three inputs to a decision can change underneath it. Each gets pinned:

- **Policy content hash on every decision.** A check records `policy_id` but not the content of the rule that ran. Policy rows can be edited in place, so replay can silently diverge. Every check will carry a hash of the policy row as evaluated, and replay fails loudly on a mismatch. In-place edits to an effective policy become an error; changes must be new effective-dated rows.
- **Content hash on every retrieved item.** `RETRIEVAL` evidence records returned item IDs. It will also record a hash of each item's content and its embedding model, so a later edit or re-index is detectable.
- **Model identity and I/O hashes on every model call.** `MODEL_CALL` evidence records the destination only. It will record provider, model name and version as reported by the provider, the prompt template version, and hashes of the prompt as sent and the response as received. The AI gateway (item 3) sees all of these.
- **Response hash on every tool call.** `TOOL_CALL` and `EGRESS` evidence record the tool and the response size, not what came back. They will record a hash of the tool response, with the content in the retention store, so "what did the system return" is reproducible.

**Content retention store.** The evidence chain deliberately holds hashes, not content, so erasure rights can be honored. Reproduction needs the content too. A separate content-addressed store, keyed by the same hashes, holds prompts, responses and source snapshots under a retention policy set per use case. Erasure removes content from the store; the chain keeps the hash and records the erasure as an event, so the record shows that content existed and was lawfully removed.

### 8. Reconstruction Report

- `kognita evidence reconstruct <interaction_id>` produces a regulator-readable report answering the ten questions in [Supervisory Examinability](#supervisory-examinability)
- Verifies the chain and every pinned hash while building the report; any mismatch is reported as a finding, not skipped
- Output as JSON (machine-verifiable) and as a readable document
- In 0.3 it covers what 0.3 records: decisions, controls, data used, model, authority. Questions answered by 0.4 items are marked "not recorded" until then, never omitted

### 9. Gateway Failure Mode

A gateway that governs every call is also a single point of failure. 0.3 makes the behavior explicit and configurable per use case:
- **Fail closed** (default): if the gateway or evidence store is unavailable, calls are refused
- **Degraded**: only calls to local models, with no client data, may proceed, and are evidenced once the store recovers
Silent pass-through when governance is unavailable is not an option.

### Tier 0 Defect Closure

From `docs/gap-analysis-bmos.md`:
- [ ] HUMAN_APPROVAL withholds nothing (item 2)
- [ ] Approval loop unclosed (item 2)
- [ ] Entitlements fail open
- [ ] SqliteVecIndex silent failure
- [ ] Classifiers never invoked (item 4 makes them load-bearing)
- [ ] `engages` missing from protocol
- [ ] No foreign keys on evidence references
- [ ] Purpose check passes everything when no purpose list is configured; must fail closed (superseded by the use-case register in 0.4)
- [ ] Anonymous agent path: when a request carries no agent name, the agent registry and kill-switch checks are skipped and the call is treated as a human. Through the gateways, every call must carry either a registered agent identity or an approved system trigger; neither is a DENY
- [ ] Self-asserted agent identity: the agent name is a string the caller supplies, so any caller can claim to be any registered agent. Until agents authenticate with their own credentials (0.5), the gateways bind each agent name to the authenticated client configuration that may use it, and reject a mismatch

### Definition of Done

- [ ] All tests pass, including new coverage for Run, suspend/resume, AI gateway, classifier-derived envelopes, MCP proxy
- [ ] Replay test: a decision made from a classifier-derived envelope replays identically without calling the classifier
- [ ] Injection test: text crafted to relabel itself cannot widen permission
- [ ] Tamper tests: editing a policy row, a retrieved item, or a stored prompt after the fact is detected by replay and by `reconstruct`
- [ ] Erasure test: erasing retained content leaves the chain verifiable and records the erasure
- [ ] Outage test: with the evidence store down, the gateway refuses calls in fail-closed mode
- [ ] Conformance kit passes
- [ ] Gateway overhead benchmarked (target: under 50 ms per call, excluding classifier inference)
- [ ] Flagship demo runs end-to-end in under 3 minutes from scaffold
- [ ] Import contracts pass: the core still has no optional dependencies
- [ ] All Tier 0 defects closed

---

## 0.4 "Ingestion, Policy Language and the Client Lifecycle"

**Timeline:** Q1 2027
**Goal:** Make citations real down to the passage, let non-engineers author and review policy, record a client interaction from origination to communication, and meet developers in the frameworks they already use.

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

**Private-banking pack.** A pack built on real RM workflows: meeting transcript to suitability report, onboarding and KYC with human approval, pre-call preparation, portfolio review, and client communication. It encodes the rules that make private banking different: client domicile, RM location and booking centre; cross-border solicitation and research-distribution restrictions; product eligibility and suitability; mandate and concentration limits. The question it answers is not only "is product X suitable?" but "may this RM discuss product X with this client, from this location, today?"

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
- **OpenTelemetry export,** so decisions, denials, latency, token use, tool failures, injection attempts and entitlement blocks flow into the bank's existing monitoring rather than a separate console
- **Effectiveness rates from evidence:** RM acceptance, edit, rejection and wrong-client rates per use case. Business KPIs such as conversations per RM or research-to-conversation conversion belong to the bank's analytics; Kognita supplies the underlying events

### Client Interaction Lifecycle

Items 8 to 15 close the ends of the record chain (why an interaction started, and what happened after the AI produced something) and answer the [five charges](#the-five-charges).

### 8. Use-Case Register

- Each AI use case is a registered, versioned entry: purpose, affected client segments, investor impact assessment, permitted data classes, approved models and versions, agent authority, required human decision points, retention period
- **Accountable owner:** every use case names a responsible individual, mapped to the firm's senior-manager accountability regime where one applies. Every decision under that use case cites the owner, so "who is accountable" is answered per decision, not per policy document. A use case with no current owner is a DENY
- **Owner roles:** business owner (the accountable owner), technology owner, risk owner for residual risk, and model owner. Accountability cannot be delegated to an agent: an agent can never be named in an owner role
- **Accountability matrix:** beyond the four owner roles, each use case states who is accountable for data, the model, the agent, the business outcome, client communication, suitability and advice, and compliance. The institution stays accountable when a third-party model, a vendor or a sub-agent performs part of the work. Functions may be delegated; accountability may not
- **Materiality classification on five axes:** business criticality (low to critical); **client-impact class** (internal productivity, RM assistance, client influence, client communication, advice, execution); autonomy; decision consequence (informational, operational, client communication, suitability, financial action); and data sensitivity (public to highly sensitive)
- **Risk tier 1 to 4,** derived from the classification: research summarization, client intelligence and meeting preparation, KYC and suitability and recommendations, financial execution. The tier sets minimum required controls: approval gates, review sampling rate, maximum autonomy level (0.5), monitoring, explanation depth, and who must validate it. A use case configured below its tier's minimum fails validation
- **AI risk appetite as top-level policy:** allowed and prohibited use-case categories, maximum permitted autonomy, permitted client impact, external communication and transaction authority. A use case outside the appetite cannot be registered. Changing the appetite is a governed change at senior-management level
- **Periodic review:** every use case carries a review date set by its tier; past it, the use case is a DENY until re-approved. Revalidation is also triggered by a change of autonomy, jurisdiction, client population or intended use
- **Linked to the bank, not kept apart:** each entry links to its business process, client segments, products, data sources, vendors, owner, risk and controls. A senior-management report lists every material AI system with its classification, owner, status and open findings
- **Training attestation:** a higher-tier use case may require that the invoking user has a current training attestation, checked as a policy attribute. The training itself is the bank's
- **Independent validation:** activating a tier 3 or 4 use case, or a material change to one, requires sign-off from functions independent of the builders, such as model risk, operational risk, compliance, information security and legal. Enforced as separation of duties: a builder cannot sign off their own use case
- **Model card per approved model:** provider, model name and version, intended use, known limitations, evaluation results the firm relied on to approve it, approval date and approver. Pinned model versions on each call (0.3) link back to the card
- Every decision must reference a registered use case; **an unregistered or retired use case is a DENY**. This replaces the free-string purpose check
- Registry changes are `POLICY_CHANGE` events and go through the same delta, validate and apply lifecycle as policy
- `kognita usecase list | show | validate` gives compliance a single inventory of AI use

### 9. Interaction Record

- One `interaction_id` spans a whole client journey: trigger, retrieval, model calls, recommendation, RM review, final decision, client communication
- Runs (0.3) and correlation IDs attach to an interaction; `reconstruct` operates on interactions

### 10. Origination Evidence

Answers "why this client?" and "why this product?", which today have no record.
- The step that selects a client or a product emits an `ORIGINATION` event before anything is shown to the RM: the trigger (event, schedule, RM request), the selection criteria, the candidate set size, and the scores or rules that ranked this client or product first
- Recommendation rationale is recorded as structured fields with citations to the sources used, not as free model text
- Suitability and eligibility checks remain separate cited checks: origination says why it was *proposed*, suitability says why it was *permitted*
- **Reasons, not scores.** "Relevance score 89%" is not an acceptable rationale. Origination must record the client facts and evidence behind it, such as exposure, a recent discussion, and the house-view change that triggered it
- **"Do not contact" is a first-class outcome.** Origination can conclude that no communication is recommended, with reasons: low relevance, poor timing, vulnerability, excessive frequency, a recent rejection, or a relationship circumstance. Contact-frequency limits per client are enforced by policy. Relationship intelligence must not optimise only for outreach

### 11. RM Review Capture

Builds on the existing proposal model (ADR 0007), which already stores before-state and rationale but is not yet on the roadmap.
- **What the RM saw:** a hash of the exact recommendation as rendered to the RM, with its content in the retention store
- **What the RM changed:** a structured diff between the AI recommendation and what the RM approved
- **Who decided:** a `FINAL_DECISION` event on every path, including plain ALLOW paths where no approval was forced, naming the RM, the outcome and time
- Two-signature approval (ADR 0006) applies where the use case requires it
- **Approval quality, not checkbox approval.** An approval request must carry a complete packet: proposed action, reason, evidence, risks, alternatives, policy checks run, and agent confidence. A packet missing any of these cannot be approved. Outcomes are approve, modify, reject or escalate
- **Automation-bias detection:** per approver, track time to decision, approval rate, edit rate and rejection rate. Approvals faster than a threshold, or an approver who approves nearly everything, are flagged to the accountable owner and count toward risk-based review (item 13). A 99.9 percent approval rate is evidence of rubber-stamping, not oversight
- **Defined human boundaries:** each use case states where a human must intervene, who that human is, what they must review, what evidence they see, and what happens when they reject. Human review is designed before deployment, not added afterwards
- **Content provenance label:** every piece of content carries a label (human, AI, AI then human-edited, or human with AI assistance) that survives edits and is recorded on any resulting communication. AI-generated content stays identifiable internally even after an RM rewrites it

### 12. Governed Client Communication

- Sending anything to a client is its own governed action: authorized against the use case, suitability and communication policy, then evidenced as a `CLIENT_COMMUNICATION` event
- The event records channel, recipient reference, a hash of the content as sent, and a link back to the recommendation and final decision it came from
- A communication that does not trace back to a final decision is a DENY
- **Content policy before sending:** required disclosures, prohibited claims, investment-risk language, research attribution, approved tone, and channel restrictions, checked as cited rules. Cross-border communication rules apply per recipient. Higher-risk communications require review by use-case tier
- **Communication level:** every communication is classified as general information, house view, contextualised insight, investment discussion, product recommendation, or transaction. Each level carries its own required controls, so moving from insight to recommendation triggers suitability checks and RM approval automatically
- **Approved-channel register:** a channel may carry client communication only if registered as meeting the bank's archiving and supervision requirements. Personal WhatsApp, Telegram or personal email are not registrable; a bank-approved messaging channel can be
- **Client transparency:** AI-disclosure rules per jurisdiction and use case decide when a client must be told that AI generated content, assisted analysis, powers a chatbot, or materially contributed to a decision. The content provenance label (item 11) drives them
- **Public lane:** public content such as LinkedIn posts, newsletters and event material uses approved research, approved claims, brand style and disclaimers only, and never client context. The separation is enforced in code (0.5 item 6), not by procedure

### 13. Risk-Based Review

Replaces random sampling as oversight. Kognita does not judge advice quality; it makes sure the right interactions reach a human who does.
- Every model response is classified (0.3 item 4) and scored against risk criteria defined per use case: product complexity, client vulnerability indicators, deviation from the client's profile, low classifier confidence
- High-risk interactions go to a **review queue** before or after delivery, as the use case requires; a review outcome is an evidenced event linked to the interaction
- Low-risk interactions are still sampled, at a rate set per use case, so reviewers keep seeing normal traffic
- **Coverage reporting:** per use case and risk tier, the share of interactions screened, queued and reviewed, and the time to review. The answer to "are you reviewing enough" becomes a number with evidence behind it
- Reviewer findings feed back: a confirmed problem raises a flag that counts toward the circuit breaker (item 14)

### 14. Circuit Breaker and Affected-Client Query

Stops poor guidance from scaling, and finds everyone it reached.
- **Circuit breaker:** when flags for a use case, model version, prompt version or policy version cross a threshold in a time window, Kognita inserts a prohibiting policy itself, escalates to the accountable owner, and records the trip as an incident. The mechanism already exists: a prohibiting policy takes effect on the next decision for every client. This item makes it automatic
- Resetting a tripped breaker is a governed action requiring the accountable owner's approval
- **Affected-client query:** `kognita evidence affected --model <version> | --policy <id> | --usecase <id> --from --to` lists every client who received output under that version in that window, with links to each interaction for remediation
- **Granular revocation.** A per-agent kill switch exists today. This extends it so any single dimension can be revoked without stopping the platform: `kognita revoke --agent | --model | --tool | --source | --client | --usecase | --action`. Each revocation is a governed, evidenced action, effective on the next decision, and its reversal needs the accountable owner's approval
- **Intervention controls, not just approve or reject.** For higher-autonomy workflows a human can **observe** what an agent is doing in a live run, **pause** it, **override** its decision with their own, **restrict** its permissions mid-run, **revoke** its credentials, and **recover** to a safe state using registered compensating actions (0.5 item 13). Every intervention is evidenced against the interaction

### 15. Outcomes and Near Misses

Do not wait for client harm.
- **Outcome events:** client complaints, corrections, wrong-client targeting, unsuitable suggestions, communication errors, policy breaches and operational incidents are recorded and linked back to the originating interaction, so a complaint traces to the exact AI output, model version and approver
- **Near-miss register:** blocked actions, verifier blocks, RM rejections, unsafe drafts and unusual agent behaviour, aggregated by use case, agent and model version
- Both feed risk-based review (item 13) and count toward the circuit breaker (item 14)
- Depends on pinned evidence (0.3) and the interaction record (item 9)

### Definition of Done

- [ ] Docling-backed ingestion with per-section classification and passage-level citations
- [ ] Use-case register enforced: unregistered use cases are denied, and every decision cites an accountable owner
- [ ] Every approved model has a model card linked from each call that used it
- [ ] Risk-based review queue with coverage reporting per use case and risk tier
- [ ] Circuit breaker trips automatically in a test scenario, and the affected-client query lists exactly the clients who received output in the window
- [ ] Revocation by each dimension takes effect on the next decision without affecting other use cases
- [ ] Risk tiers enforce minimum controls; a tier 3 use case cannot be activated by its own builder
- [ ] An incomplete approval packet cannot be approved; rubber-stamp approvals are flagged
- [ ] Private-banking starter pack with scenarios, including cross-border and booking-centre denials
- [ ] OpenTelemetry export
- [ ] A use case outside the risk appetite cannot be registered; a use case past its review date is denied
- [ ] "Do not contact" outcomes recorded with reasons; a bare score is rejected as rationale
- [ ] Every communication carries a communication level and content provenance label; unregistered channels are denied
- [ ] Pause, override, restrict and recover work on a live run and are evidenced
- [ ] A complaint links back to the exact output, model version and approver
- [ ] One interaction reconstructs end to end: all ten examinability questions answered from evidence, with no "not recorded" rows
- [ ] Graph extra either tested and unpinned, or extracted
- [ ] Policy language: load, diff, validate, explain, test
- [ ] Five starter packs with scenarios
- [ ] hermes-agent adapter plus at least three others
- [ ] Flight recorder with export

---

## 0.5 "Agents, Authority and Fleets"

**Timeline:** Q2 2027
**Goal:** Every agent has its own identity and acts only under explicit, specific, revocable authority from an accountable human. Then many such agents can run together safely.

The principle: not "Ahmed can access this, so Ahmed's agent can", but "Ahmed can access this **and** this agent is authorized to access it **for this task**."

### 1. Agent Identity

Extends the agent registry that exists today (name, version, accountable owner, risk class, materiality tier, kill switch).
- Each agent gains: purpose, linked use cases, permitted systems, tool allow-list (item 5), risk tier, autonomy level (item 3), blast-radius limits (item 4), and a review or expiry date. **An agent past its review date is a DENY**
- **Deployment version covers behavior, not just code:** a hash over the system prompt, instructions, tool set, permissions, data sources, orchestration logic and memory behavior. A prompt can change behavior like code does, so any change to these produces a new version that needs re-approval before it can act. This is change management enforced, not documented
- Identities follow a readable convention such as `AGENT-RM-PRECALL-01`
- **Agents authenticate with their own credentials.** Today the agent name is a self-asserted string. Each agent gets its own workload credential, verified by the gateways on every call; no "AI service account", and no credentials shared between agents. The registry also records environment, deployment date, model used, and the principal on whose behalf it operates
- The registry entry is the agent's manifest: `kognita agent deploy --manifest …`, signed once 0.7 lands

### 2. Delegated Authority

An agent needs explicit authority to act for someone.
- An authorization object binds **human + agent + purpose + client + allowed action + allowed data + time period + limits**. Example: "RM John authorizes the Portfolio Review Agent to retrieve portfolio information for client 123, to prepare tomorrow's meeting, until 18:00"
- Authority is specific, temporary by default, purpose-bound and revocable; revocation takes effect on the next decision
- **Dual check:** a request is allowed only if the human holds the entitlement *and* the agent is authorized for it under a live delegation. Agent entitlements are always narrower than the human's
- Every decision and action envelope references the delegation it relied on
- The object also carries **jurisdiction** (which booking centre or country) and an **approval threshold** (which actions require human approval)
- **Authority ends when the task does,** not only when the clock runs out: "access client X to prepare tomorrow's meeting" is revoked once the meeting preparation completes

### 3. Autonomy Levels

Autonomy is classified, not implied.

| Level | Agent authority |
|---|---|
| L0 | Retrieve |
| L1 | Explain |
| L2 | Recommend |
| L3 | Draft |
| L4 | Act after approval |
| L5 | Bounded autonomous action |
| L6 | Broad autonomy |

- Every agent carries a maximum level; every tool declares the level it requires. A call above the agent's level is a DENY; a call at L4 always routes to human approval
- The use-case risk tier caps the level. Early private-banking deployments are expected at L1 to L4
- Raising an agent's level is a governed change requiring the accountable owner and independent validation. Autonomy never increases silently: moving from "generate draft" to "queue draft" to "send automatically" is treated as a material change
- **Action permissions are per verb.** Read, analyse, recommend, draft, queue, send and execute are separate permissions. **The ability to recommend never implies the ability to execute:** an agent may conclude "client X should be contacted" without being permitted to contact client X

### 4. Blast-Radius Limits

Answers "if this agent misbehaves, how much damage can it do?"
- Per agent: maximum distinct clients, specific clients or portfolios, maximum monetary value, maximum daily actions and transactions, allowed products and systems, execution authority (none, propose, execute), and time to live
- Example: Portfolio Agent, client ABC, portfolio 1234, may propose a rebalance, maximum value $0, execution authority none, expires in 4 hours
- **Rate limits per minute and per hour,** not only per day, including maximum clients contacted, allowed channels and allowed jurisdictions. An RM agent sending 2,500 client messages in four minutes is a DENY, whether the cause is malfunction or compromise
- Limits are enforced by `decide()` and are independent of Run budgets: a Run budget bounds one task, a blast-radius limit bounds the agent

### 5. Tool Allow-Lists and Data Minimization

- Each agent has its own tool allow-list. A pre-call agent may read CRM, portfolio, house view and approved research, and is denied sending email, placing orders, changing KYC or moving money. Tools are capabilities, governed separately from prompts
- **Field-level minimization:** an agent receives only the fields its purpose needs, not whole records. A pre-call agent sees the risk profile but not passport details
- Access windows: authority may be tied to an event, such as "meeting within 24 hours"

### 6. External Content Isolation

Anything from outside the bank is treated as potentially hostile: websites, PDFs, emails, attachments, third-party APIs.
- **Taint tracking:** content from an external source is marked when it enters a run. A tainted run cannot invoke tools above a set autonomy level or with write authority. Instructions inside external content never acquire authority because a model read them
- **Separation of duties between agents:** an agent permitted to read the internet cannot also hold internal write permissions. External research agents hand over structured evidence, not free text, to internal agents
- Injection attempts detected by the classifier are evidenced and alerted
- **Prompt injection is treated as an access-control problem,** not only an LLM problem. A malicious document can never cause an agent to access another client, reveal confidential data, invoke a prohibited tool, change permissions or send an external message, because none of those depend on the model obeying
- **Two lanes, separated in code:** client-relationship intelligence may use permissioned client data; public content (LinkedIn, newsletters, marketing, events) may not. Client data carries a taint label just as external content does, and a run that has touched client data can never reach a public-content tool

### 7. Instruction Authenticity

"Communication is not authorization." A request arriving by WhatsApp, email, voice or video, even from a known number, does not prove who sent it.
- Envelopes carry an authentication assurance level for the human behind a request
- High-risk actions require step-up authentication evidence in an authenticated workflow, with transaction context and independent authorization. A channel message alone cannot satisfy them
- Applies equally to instructions relayed by an agent: an agent cannot raise the assurance level of an instruction it received

### 8. Behavioral Anomaly Detection

- A baseline per agent: distinct clients per day, documents retrieved, actions taken, tools used, data classes touched
- Deviations block, isolate and alert: a pre-call agent querying 10,000 clients, a KYC agent downloading thousands of documents, a service agent creating 700 tickets, a portfolio agent requesting transaction tools
- Also watched: privilege probing and repeated denied actions, unusual tool sequences, unexpected network access, and sudden behaviour drift
- **Machine-speed containment.** Agentic failures and attacks can outpace human response, so containment is automatic when a threshold is crossed: quarantine the agent, revoke its credential, stop its tools, then alert a human. The human decides on release, not on containment
- Blocks are recorded as incidents and count toward the circuit breaker (0.4 item 14)

### 9. Delegation Between Agents
- `Run.delegate_to(agent_name, attenuated_scope=…)`; the child's budget, scope and autonomy are carved from the parent's and can only narrow
- Every delegation is evidenced with the authority transferred: who delegated, why, permitted data and tools, client scope, permitted actions and expiry. A worker never inherits its orchestrator's full privileges
- **Authority lineage:** `kognita evidence authority <action_id>` answers "show me the chain of authority that permitted this action", from the human through every orchestrator and worker to the action

### 10. Capability Grants
- `grant_to(grantee, capability, duration, subject_scope)`; issue, use, revoke and expiry are all evidenced
- Revocation takes effect on the next decision

### 11. Fleet Controls
- Per-agent quotas, run isolation, and authorization of agent-to-agent calls through the same gateway path as agent-to-tool calls
- Agent-to-agent message bodies are free text, so classifier-derived envelopes (0.3 item 4) and taint tracking apply
- **External agents are untrusted principals.** An agent from outside the bank calling the bank's agents or MCP endpoints must authenticate, is held to the lowest trust level, and its messages are tainted as external content

### 12. Sandbox Constraints as Decision Output
An ALLOW for code execution carries constraints the executing sandbox must honor: credentials, network destinations (block-all or allow-list), file system scope, permitted APIs and commands, data export, persistence, an auto-stop interval, and CPU, memory and disk limits. The pattern is agent → sandbox → policy enforcement point → approved systems, never agent → enterprise network. Kognita emits and evidences the constraints; it does not ship a sandbox.

### 13. Reversibility

- Every tool declares whether its action is reversible and, if so, registers a compensating action: delete a draft, cancel a queued communication, roll back a CRM update, reverse a permission change, withdraw a recommendation
- Reversibility feeds the decision: an irreversible action requires stronger pre-execution controls, such as a higher autonomy requirement, human approval, or a lower blast-radius limit
- The **recover** intervention (0.4 item 14) runs the registered compensations for a run, in reverse order, as evidenced actions

### 14. Adversarial Suite and Threat Model

- **Adversarial conformance suite:** Kognita ships attack scenarios that test its own controls, alongside the existing conformance kit: prompt injection, wrong-client data, stale information, malicious documents, poisoned retrieval, excessive tool permissions, privilege escalation, looping agents and policy bypass. A pack passes only if every attack is denied, escalated or contained
- **Published threat model** covering both "our agent fails" and "an external agent attacks us": prompt injection, credential theft, malicious tools, poisoned data, autonomous reconnaissance and automated exploitation, each mapped to the control that answers it

### Definition of Done
- [ ] Every agent has a full identity; agents past review date and changed deployment versions are denied until re-approved
- [ ] Dual check enforced: a human's entitlement alone never authorizes their agent
- [ ] Autonomy levels enforced; L4 always routes to approval
- [ ] Blast-radius limits enforced per agent
- [ ] Per-agent tool allow-lists and field-level minimization
- [ ] Injection test: a tainted run cannot call a write tool
- [ ] Impersonation test: a high-risk instruction from a channel message without step-up evidence is denied
- [ ] Anomaly test: a volume spike blocks the agent and raises an incident
- [ ] Delegation, grants, fleet controls, sandbox constraints
- [ ] Agents authenticate with their own credentials; a spoofed agent name is denied
- [ ] Recommend and execute are separate permissions; an agent with only recommend cannot act
- [ ] A burst above the per-minute rate limit is denied
- [ ] A run that has touched client data cannot reach a public-content tool
- [ ] Containment quarantines, revokes and stops an agent without a human in the loop, then alerts
- [ ] Authority lineage reconstructs the full chain for a delegated action
- [ ] Recover runs compensating actions in reverse order
- [ ] Adversarial suite passes; threat model published

---

## 0.6 "Claims"

**Timeline:** Q3 2027
**Goal:** Every material statement an agent produces is traceable, typed, current, and checked before a human relies on it. Agents reason over trusted data; they do not decide what is true.

### 1. Claim Provenance Envelope

Every material claim carries a record, so an RM can click "why am I seeing this?" and see the evidence:

```
claim_id, client_id, claim, claim_type, source_system, source_record,
source_timestamp, retrieval_timestamp, calculation_method, agent_id,
model_version, prompt_version, confidence, validator_status, policy_status
```

Claims are hashed into the evidence chain with their content in the retention store. `reconstruct` lists the claims an RM saw.

### 2. Fact Contract and Freshness

Kognita does not own client data; the bank's systems of record do. Kognita enforces a contract on facts that reach an agent:
- Every authoritative fact carries **value, source, owner, as-of date, classification, entitlement and freshness threshold**. Example: risk profile Moderate Growth, from the suitability system, updated 18 August 2026, confidential
- **Freshness thresholds per data type,** set by policy: portfolio positions near real time or latest end of day, house view the current publication, market prices from a permitted feed, KYC within the current review cycle, suitability the current approved profile
- Stale data is refreshed, suppressed, or shown with a warning, as the use case decides. It is never silently used. "Only use current data" is enforced in code, not asked of the model
- **Research provenance and lifecycle:** every research source records author, publication date, passage and version, and carries created, effective, last-reviewed, superseded and expiry dates. When research expires, recommendations that depend on it stop automatically

### 3. Approved-Source Grounding

- Each use case has a register of approved sources: CIO research, house views, product documentation, permitted market data, policy sources
- A regulated investment claim must cite an approved source. **The model's own knowledge is never a source** for a regulated claim; a claim grounded only in model knowledge is blocked from RMs and clients

### 4. House-View Change Propagation

- When a source is superseded, for example a house view moving from neutral to overweight, every open recommendation that depends on it is invalidated automatically
- `kognita evidence affected --source <id>` lists the affected clients, previous recommendations, previous communications and outstanding opportunities, so a stale view is never distributed and past communications can be reviewed

### 5. Claim Types: Fact, Inference, Recommendation

- Every claim is typed: **verified fact**, **inference**, **conversation hypothesis**, or **possible bank capability**
- Unlabelled claims cannot be shown to an RM or sent to a client
- An inference cannot be relabelled as a fact without a cited source and verification, so an LLM inference cannot quietly become perceived fact

### 6. Calculation Provenance

Kognita is not a calculation engine and does not perform financial math.
- Authoritative figures (performance, P&L, NAV, concentration, allocation, FX, leverage, VaR, maturity values, interest, exposure, tax, suitability scores) must cite a registered calculation service in `calculation_method`
- A figure presented as authoritative without one is blocked. The model may explain a calculation; it may not invent one

### 7. Verification Gate

Before material output reaches an RM, a registered verifier tries to prove it wrong.
- The verifier is a registered agent or service with its own identity. Kognita does not ship one; it requires one per use case at tier 2 and above
- It checks for wrong client, duplicate identity, stale data, conflicting sources, incorrect calculations, questionable sources, unsupported inferences, wrong currency, wrong company match, prohibited content and broken entitlements
- Each claim is classified **verified, inferred, unverified, conflicting, stale or blocked**, recorded in `validator_status`. Policy decides what each status may reach

### 8. Memory Governance

- **Working memory** is temporary context and expires with the run
- **Relationship memory** is durable client information, governed like any other bank record
- Writing to relationship memory is a governed action. An inference such as "client may be considering a sale" can be proposed, but becomes a record only after RM verification, through the proposal model (ADR 0007). Hallucinations cannot become institutional facts automatically
- Memory is typed like claims (verified fact, inferred preference, AI hypothesis), scoped to one client, and carries a retention period, a correction path and an expiry. "The client probably prefers X" cannot accumulate as if it were a fact

### 9. Feedback-Loop Controls

- Interactions are never used for learning by default. Exporting interactions, RM edits or outcomes to any training, fine-tuning or evaluation pipeline is a governed action: purpose, anonymisation, validation, approval and exclusions, all evidenced
- Clients and records can be excluded, and the exclusion holds for every future export. Otherwise bad behaviour can become self-reinforcing

### 10. Sensitive Attribute Controls

- The classifier labels special-category inferences: religion, health, political views, ethnicity, sexuality, personal vulnerabilities
- Policy blocks them from relationship workflows unless the use case explicitly permits, even when they are technically inferable from external sources. Vulnerability indicators may be permitted where the use case exists to protect the client
- **Vulnerable-client safeguards:** an AI may flag a potential vulnerability indicator, but it may not make the determination or change how a client is treated. A flag goes to RM review with the signal explained; unsupported diagnosis is blocked; any change of treatment goes through a governed process with human escalation

### 11. Evaluation Evidence Gate

Kognita is not an evaluation harness. It records evaluation results and enforces them.
- Evaluation results are attached to agent and model versions: accuracy, groundedness, completeness, relevance, hallucination rate, freshness, policy compliance, entitlement, action correctness and escalation
- Activating or changing a tier 2 or higher agent requires results above thresholds set per use case, such as a benchmark against a golden set of historical meetings defined by experienced RMs
- Kognita measures the dimensions only it can see from evidence: entitlement blocks, escalation correctness, action correctness, stale-data use, permission and client-boundary compliance, delegation behaviour and retries
- **Three levels of evaluation:** the model; the agent (task planning, tool selection, permission compliance, delegation, retries, escalation); and the end-to-end system (model, retrieval, tools, data, agent, policies, UI and human interaction). A good model can still produce an unsafe system, so the gate accepts system-level results
- **Bias and fairness reporting:** recommendation, contact and prioritisation rates computed from origination evidence across segments the bank defines, such as nationality, age group, language, portfolio size, channel and RM team. Disparities above a threshold are flagged to the risk owner. This matters most for client prioritisation, prospect scoring, product recommendations and vulnerability detection
- **The continuous loop:** evaluate, deploy, observe, test, challenge, intervene, learn, re-authorise. Validation does not end at deployment

### Definition of Done
- [ ] Claim provenance envelope on every material claim, reconstructable
- [ ] Stale data refreshed, suppressed or flagged per policy; never silently used
- [ ] Unlabelled claims blocked from RM and client
- [ ] Authoritative figures without a registered calculation service blocked
- [ ] Verification gate required at tier 2 and above, with statuses enforced
- [ ] AI inferences cannot reach relationship memory without RM verification
- [ ] Special-category inferences blocked unless permitted
- [ ] Activation blocked without evaluation results above threshold
- [ ] A regulated claim citing only model knowledge is blocked
- [ ] Superseding a house view invalidates dependent recommendations and lists affected clients
- [ ] Exporting interactions for learning without approval is denied; exclusions hold
- [ ] A vulnerability flag cannot change treatment without RM review
- [ ] Fairness report produced across bank-defined segments

---

## 0.7 "Trust and Resilience"

**Timeline:** Q4 2027
**Goal:** Make evidence tamper-proof, verifiable by outside parties, and survivable. Governance is itself critical infrastructure once every AI call depends on it, so it falls under the same operational-resilience expectations as any other ICT system.

### Trust

- **Ed25519 signing:** each evidence event signed; `kognita evidence verify --public-key` walks the chain and verifies every signature
- **Postgres backend:** concurrency for many writers, replication, row-level security so agents cannot read each other's runs
- **External review API:** a reviewer requests a period, annotates decisions, and submits a signed review into the chain
- **TypeScript evidence verifier:** `@kognita/verify`, so auditors can verify without installing Python
- **Schema versioning:** evidence events carry a schema version; migrations rename, recreate, copy intersecting columns and drop inside one transaction, and old events stay readable
- **External anchoring:** periodically publish the chain head hash to a store the bank does not control, so even a party with full database access cannot rewrite history undetected

### Resilience

- **Evidence backup and restore,** tested: restore to a point in time and verify the chain and retention store end to end; restore drills are part of CI
- **Key management:** signing key rotation, revocation and escrow; verification works across rotations
- **Model-provider register:** every model provider the gateway can reach is recorded as an ICT third party, with the use cases that depend on it, data residency, data retention, whether data may be used for training, sub-processors, encryption, incident notification terms, availability commitments, model change notice, audit rights and exit plan. Routing to an unregistered provider is a DENY. Zero-retention agreements are recorded but do not replace the rest; the contracts themselves remain the bank's
- **Incident playbooks** are the bank's. Kognita supplies the detect, contain and investigate steps for each agent incident type (hallucination, data leakage, prompt injection, unauthorized tool execution, incorrect transaction, model outage, third-party compromise, deepfake instruction, sensitive-data exposure) through revocation, the circuit breaker, the affected-client query and incident evidence
- **Provider failover under policy:** switching to a fallback model is itself a governed decision, allowed only to models approved for that use case, and evidenced
- **Provider register additions:** incident history and regulatory cooperation
- **Silent model-change detection:** the gateway compares the provider-reported model version on every call with the approved version. A change the provider did not announce is treated as a revalidation trigger, and calls fall back or stop as the use case decides
- **AI dependency map:** business service → agent → model → cloud → vector store → tool or API → data source, built from the registers and checked against what evidence shows was actually called. Divergence between declared and observed dependencies is a finding
- **Concentration risk:** from the dependency map, the share of critical use cases resting on each model provider, cloud, agent framework, vector platform and orchestration layer. "What happens if every critical workload depends on one provider?" becomes a report, not a guess
- **Failure simulations,** run as drills: model unavailable, vendor outage, API outage, retrieval failure, identity service failure, corrupted context, compromised model, tool failure. Each drill verifies the configured failure mode and that evidence stays complete
- **Retention by record class:** retention periods are set per record class (client communications, advice, recommendations, approvals, transactions, compliance records), so an AI record never expires before the regulatory record it supports
- **Business continuity is the bank's.** Kognita's own failure mode is defined in 0.3; keeping RM work running without AI (falling back to CRM and research access) is the bank's design, which the dependency map informs
- **Incident evidence:** chain breaks, gateway outages, degraded-mode periods and provider failures are recorded as incidents with timelines, exportable for incident reporting

---

## 1.0 "Production Ready"

**Timeline:** Q1 2028

- **Benchmarks:** decision latency p50 and p99, evidence write throughput, gateway overhead, resume time; published with hardware specs
- **Docs rewrite:** lead with the problem ("prove an AI request was allowed before any data moved"), scenarios by industry, glossary, honest comparison with content guardrails and AI gateways
- **Ten single-file examples**, each runnable in under five minutes and tested in CI, including the AI gateway, the MCP proxy, a policy-only YAML deployment, an approval workflow, and an evidence audit
- **Examinability acceptance:** every row of the reconstruction test in [Supervisory Examinability](#supervisory-examinability) answered from evidence for a real RM interaction, including after a backup restore and a signing key rotation
- **Control framework acceptance:** all twelve non-negotiable controls demonstrated, and every one of the 40 domains in [control-framework.md](control-framework.md) either covered or explicitly marked boundary or out of scope
- **Wealth AI framework acceptance:** all fifteen non-negotiable controls demonstrated, every one of the 70 controls in [wealth-ai-control-framework.md](wealth-ai-control-framework.md) covered or explicitly marked boundary or out of scope, and the overarching test passed on a real interaction: the bank can explain, control, stop and reconstruct it
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
3. **0.5:** agent identity and delegated authority make multi-agent deployments safe
4. **0.6:** typed, verified, current claims make agent output trustworthy to RMs
5. **0.7:** signatures, external review and tested resilience satisfy regulated buyers
6. **1.0:** production-ready, benchmarked, documented, examinable

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
| Reproducibility conflicts with erasure rights | Chain holds hashes only; content lives in a separate retention store with per-use-case retention; erasure is itself an evidenced event |
| The gateway becomes a single point of failure | Explicit fail-closed or degraded mode per use case (0.3); tested backup, restore and failover (0.7) |
| Scope creep into building truth layers, calculation engines, evaluators or sandboxes | Those are marked boundary: Kognita requires that the authoritative system was used and records it, and does not become that system |
| Model output cannot be regenerated identically | Reproduction means retrieving what was recorded, not re-running the model: prompts and responses are retained by hash, never regenerated |

---

## Decision Log

**4 October 2026: Wealth management AI control framework**
- Mapped the roadmap against a 70-control, 10-domain framework for RM-facing and agentic AI and its 15 non-negotiable controls. Full mapping in [wealth-ai-control-framework.md](wealth-ai-control-framework.md). Measured after the 40-domain audit, 7 non-negotiables were already planned or covered and 8 were partial; none were missing outright.
- Adopted its overarching test: can the bank explain, control, stop and reconstruct every material AI action that affects a client?
- Found that agent names are self-asserted and unauthenticated. Added to Tier 0 (gateways bind names to authenticated clients) and to 0.5 (agents get their own credentials).
- 0.4: accountability matrix, five-axis materiality classification with the tier derived from it, AI risk appetite, periodic review, links to business processes, training attestation; reasons instead of scores and a "do not contact" outcome; defined human boundaries, content provenance labels, edit and rejection rates; communication levels, approved-channel register, client AI-disclosure rules, a public lane; intervention controls; new item 15, outcomes and near misses.
- 0.5: agent credentials; jurisdiction, approval threshold and task-completion expiry on delegated authority; per-verb action permissions; per-minute rate limits; prompt injection as access control and a client-data lane; automatic containment; authority lineage; external agents as untrusted principals; new items 13, reversibility, and 14, adversarial suite and threat model.
- 0.6: research lifecycle, approved-source grounding, house-view change propagation, typed and expiring memory, feedback-loop controls, vulnerable-client safeguards, three-level evaluation, fairness reporting. Items renumbered 1 to 11.
- 0.7: silent model-change detection, AI dependency map, concentration risk, failure drills, retention by record class.
- Named two scales to avoid collisions with Kognita's own codes: the client-impact class and the communication level.
- No release dates changed.

**4 October 2026: Bank-grade control framework**
- Mapped the roadmap against a 40-domain control framework for agentic AI in private banking and its 12 non-negotiable controls. Full mapping in [control-framework.md](control-framework.md). Only one non-negotiable, runtime policy outside the LLM, was fully covered at v0.2.
- Adopted the framework's rule, "governance by architecture, not by instruction", as a stated principle.
- Found that an agent registry with per-agent kill switch already exists in code, and that requests without an agent name bypass it. Added the bypass to Tier 0.
- 0.3: tool response hashes and prompt template versions added to pinned evidence.
- 0.4: owner roles, risk tiers and independent validation in the use-case register; approval packets and automation-bias detection; communication content policy; granular revocation; OpenTelemetry export; a private-banking starter pack.
- 0.5 renamed **Agents, Authority and Fleets**: full agent identity with behavioral versioning, delegated authority with a dual check, autonomy levels L0 to L6, blast-radius limits, per-agent tool allow-lists and field minimization, external content isolation, instruction authenticity, anomaly detection.
- New 0.6 **Claims**: claim provenance envelope, fact contract and freshness, claim types, calculation provenance, verification gate, memory governance, sensitive attributes, evaluation evidence gate.
- Trust and Resilience moves to 0.7, with an expanded provider register and incident support. 1.0 moves from Q4 2027 to Q1 2028 to absorb the added release.
- Recorded five boundary domains where Kognita enforces and evidences but does not build: truth layer, calculations, evaluation, sandbox, challenger.

**3 October 2026: The five charges**
- Mapped the roadmap against five AI governance questions a supervisor such as the FCA could ask. Charges 1 and 2 were largely covered; charges 3, 4 and 5 had gaps.
- Added to the 0.4 use-case register: an accountable owner cited on every decision, and a model card per approved model.
- Added 0.4 item 13, risk-based review, replacing random sampling with screening of every response, a review queue for high-risk interactions, and coverage reporting.
- Added 0.4 item 14, an automatic circuit breaker and an affected-client query.
- Recorded explicitly that Kognita checks permission, not advice quality, and that it provides model provenance, not interpretability.

**28 September 2026: Supervisory examinability**
- Target state set as **Governable → Explainable → Controllable → Replayable → Resilient**, reflecting supervisory focus moving from AI governance documentation to examinability, alongside operational-resilience priorities.
- Audit against the RM record chain and the ten reconstruction questions found the roadmap strong on authority and controls, weak at both ends of the interaction, and weak on reproducibility.
- Added to 0.3: pinned evidence (policy content hashes, retrieved content hashes, model identity and I/O hashes), a content retention store, `evidence reconstruct`, and an explicit gateway failure mode.
- Added to 0.4: use-case register, interaction record, origination evidence, RM review capture, governed client communication.
- 0.6 widened to Trust and Resilience: external anchoring, tested backup and restore, key management, model-provider register, governed failover, incident evidence.
- 1.0 acceptance now includes a full reconstruction of a real RM interaction.
- Purpose check found to pass everything when unconfigured; added to Tier 0.

**September 2026**
- **AI gateway replaces the in-process governed model wrapper.** Explicit proxy (base URL), not transparent TLS interception. OpenAI-compatible format first.
- **Classifier-derived envelopes** added to 0.3 so policy applies to free text. The classifier builds the envelope and never decides; its output is evidence; low confidence escalates; it may only narrow permission.
- **Scope bounded** to AI and agent traffic; general enterprise integration is out of scope.
- **Docling-backed structured ingestion** added to 0.4 to make citations passage-level and classification per section.
- **Graph extra** to be finished or extracted in 0.4; extraction is the default.
- **Policy language design** informed by OpenSpec (delta changes, strict validation), spec-kit (clarification markers) and Fabric (pack distribution).
- **hermes-agent** chosen as the first adapter.
- **Timeline corrected.** The first version of this document dated 0.3 at Q4 2025, which had already passed; releases now run Q4 2026 through Q4 2027.
