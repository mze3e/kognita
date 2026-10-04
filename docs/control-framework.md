# Bank-Grade Agentic AI Control Framework: Coverage

**Last revised:** 4 October 2026. Companion to [ROADMAP.md](ROADMAP.md).

This maps a consolidated control framework for agentic AI in wealth management and private banking against Kognita. The framework draws on supervisory material (MAS, ABS), security references (NIST, OWASP), and production patterns described for large banks. The bank names and incidents below are the framework author's illustrations; they have not been independently verified here.

## The design rule

> **Never rely on the LLM to enforce a control that can be enforced outside the LLM.**
>
> Do not govern the model. Govern the complete chain from human intent → agent authority → data → reasoning → tools → action → outcome.

This is Kognita's founding premise. Every decision is made by deterministic code before any data is retrieved or any tool runs, never by asking a model to behave. Where a domain below needs model judgment, such as classifying free text, the model only *supplies an attribute*; a deterministic policy makes the decision, and the model's output is recorded as evidence.

## Status legend

| Status | Meaning |
|---|---|
| **Covered** | Shipped in v0.2 |
| **Planned** | On the roadmap before this audit |
| **Added** | A gap found by this audit, now on the roadmap |
| **Partial** | Some of the domain shipped or planned; the rest added |
| **Boundary** | Kognita enforces and evidences the control; the capability itself belongs to another system |
| **Out of scope** | Not a software control Kognita can provide |

## The twelve non-negotiables

| # | Control | Status | Kognita today | Delivered by |
|---|---|---|---|---|
| 1 | Unique agent identity | Partial | Agent registry with version, accountable owner, risk class, materiality tier; unregistered agents denied. **Gap:** a call with no agent name skips the check | Tier 0 fix (0.3); full agent identity (0.5) |
| 2 | Explicit delegated authority | Added | Approvals bind to an envelope hash; agent-to-agent grants planned | Delegated authority object (0.5) |
| 3 | Purpose-bound access | Partial | Purpose in every envelope; closed vocabulary optional and fails open when unset | Tier 0 fix (0.3); use-case register (0.4) |
| 4 | Least-privilege data and tool entitlements | Partial | Zones and classification ceilings filter data before scoring; explicit tool registry | Per-agent tool allow-lists and field minimization (0.5) |
| 5 | Trusted client and portfolio truth layer | Boundary | Retrieval returns entitled, cited items | Fact provenance and freshness contract (0.6) |
| 6 | Evidence and provenance for material claims | Partial | Citations on decisions; retrieval evidence | Pinned evidence (0.3); claim provenance envelope (0.6) |
| 7 | Fact, inference and recommendation separation | Added | None | Claim types (0.6) |
| 8 | Deterministic financial calculations | Boundary | None | Calculation provenance enforcement (0.6) |
| 9 | Runtime policy enforcement outside the LLM | Covered | `decide()` is deterministic, pure, replayable, fail-closed | Core |
| 10 | Sandboxed, bounded execution | Partial | Run budgets planned | Blast-radius limits (0.5); sandbox constraints (0.5) |
| 11 | Human approval for high-impact decisions | Partial | HUMAN_APPROVAL outcome; two-signature approvals | Suspend/resume (0.3); autonomy levels (0.5) |
| 12 | Complete action audit trail and kill switch | Partial | Hash-chained evidence; per-agent kill switch | Reconstruction (0.3); granular revocation (0.4) |

## All forty domains

| # | Domain | Status | Kognita's answer | Delivered by |
|---|---|---|---|---|
| 1 | Human identity and accountability | Partial | Principal on every envelope; accountable owner on each agent. Added: business, technology, risk and model owner roles; no anonymous invocation | Tier 0 (0.3); use-case register (0.4) |
| 2 | Agent identity | Partial | Agent registry exists (version, owner, risk class, tier, kill switch). Added: purpose, permitted systems and tools, deployment version covering prompts and tools, review and expiry date | Agent identity (0.5) |
| 3 | Delegated authority | Added | Authorization object: human + agent + purpose + client + action + data + time + limits; specific, temporary, revocable | Delegated authority (0.5) |
| 4 | Purpose limitation | Planned | Unregistered use case is a DENY; permitted data classes per use case | Use-case register (0.4) |
| 5 | Least privilege and entitlements | Partial | Attribute-based policy over envelope attributes today. Added: human ∧ agent intersection, per-agent tool allow-list, field-level minimization, time windows | Agent identity and authority (0.5) |
| 6 | Data governance, client truth layer | Boundary | Kognita does not own client data. It requires facts to arrive with value, source, owner, as-of date, classification and freshness threshold, and enforces that contract | Fact contract (0.6) |
| 7 | Source and provenance controls | Partial | Retrieval citations, pinned hashes, structured origination rationale. Added: per-claim provenance envelope including calculation method and prompt version | Pinned evidence (0.3); claims (0.6) |
| 8 | Facts vs inferences vs recommendations | Added | Every claim carries a type; unlabelled claims cannot reach an RM or a client | Claim types (0.6) |
| 9 | Data freshness | Added | Freshness thresholds per data type; stale data is refreshed, suppressed or flagged, decided by policy | Freshness (0.6) |
| 10 | Deterministic financial calculations | Boundary | Kognita is not a calculation engine. Numeric claims must cite a registered calculation service; a model-generated figure presented as authoritative is blocked | Calculation provenance (0.6) |
| 11 | Model risk controls | Planned | Model and version pinned on every call; model card per approved model; provider register; governed failover | 0.3, 0.4, 0.7 |
| 12 | Agent evaluation and quality | Boundary | Kognita is not an evaluation harness. It records evaluation results against model and agent versions, gates activation on them, and measures what only it can see: entitlement blocks, escalation correctness, action correctness | Evaluation evidence gate (0.6) |
| 13 | Challenger or verification agent | Added | A registered verifier must clear material outputs before display; its findings classify each claim as verified, inferred, unverified, conflicting, stale or blocked | Verification gate (0.6) |
| 14 | External content and prompt injection | Partial | Classifier output can only narrow permission (0.3). Added: taint tracking, so a run that has read external content cannot invoke high-authority tools | External content isolation (0.5) |
| 15 | Tool and capability permissions | Partial | Explicit tool registry; tools run only after an ALLOW. Added: per-agent allow-lists | Agent identity (0.5) |
| 16 | Execution sandbox | Boundary | Kognita does not ship a sandbox. An ALLOW for execution carries constraints the sandbox must honor, and they are evidenced | Sandbox constraints (0.5) |
| 17 | Action and transaction controls | Covered | Decide, then record, then execute; limits, eligibility, cross-border and mandate rules as rule types; propose-then-apply model | Core; private-banking starter pack (0.4) |
| 18 | Agent blast radius | Partial | Run budgets (0.3) and fleet quotas planned. Added: maximum distinct clients, maximum value, daily action caps, execution authority, time to live, per agent | Blast-radius limits (0.5) |
| 19 | Human approval gates and autonomy levels | Added | Autonomy level L0 to L6 per agent and use case; each tool declares the level it needs; tier caps the level | Autonomy levels (0.5) |
| 20 | Instruction authenticity | Added | "Communication is not authorization": instructions arriving by email, chat, voice or video cannot authorize high-risk actions without step-up authentication evidence | Instruction authenticity (0.5) |
| 21 | Human-in-the-loop quality | Partial | RM review capture planned. Added: required approval packet, four outcomes, rubber-stamp detection | RM review capture (0.4) |
| 22 | Memory governance | Added | Writing to durable memory is a governed action; inferences cannot be persisted as facts without RM verification; working memory expires | Memory governance (0.6) |
| 23 | Sensitive attribute controls | Added | Classifier labels special-category inferences; policy blocks them from relationship workflows unless the use case permits | Sensitive attributes (0.6) |
| 24 | Cross-border and booking centre | Covered | Actor location and data zones are first-class; fixture packs cite multiple regulators per decision | Core; private-banking starter pack (0.4) |
| 25 | Product eligibility and suitability | Planned | Origination records why proposed; suitability records why permitted; client communication must trace to a passed check | Origination and communication (0.4) |
| 26 | Agent action envelope | Partial | Envelope, cited checks, approvals, interaction record, reconstruction. Added: delegated authority reference and tool response hash | Pinned evidence (0.3); 0.4; 0.5 |
| 27 | Continuous monitoring | Partial | Flight recorder alerts planned. Added: OpenTelemetry export into the bank's existing monitoring | Flight recorder (0.4) |
| 28 | Behavioural anomaly detection | Added | Per-agent baselines; deviations such as a sudden spike in distinct clients or a new tool category block, isolate and alert | Anomaly detection (0.5) |
| 29 | Kill switch and revocation | Partial | Per-agent kill switch exists. Added: revoke by agent, model, tool, data source, client, workflow or action without stopping the platform | Granular revocation (0.4) |
| 30 | Incident response | Partial | Detection and containment through circuit breaker and incident evidence. Playbooks are the bank's | 0.4; 0.7 |
| 31 | Change management | Partial | Policy changes go through delta, validate and apply. Added: agent deployment version covers prompts, instructions, tools and orchestration; any change requires re-approval | Agent identity (0.5) |
| 32 | Third-party and vendor risk | Partial | Provider register planned. Added fields: retention, training use, residency, sub-processors, incident notification, audit rights, exit plan. Contracts are the bank's | Provider register (0.7) |
| 33 | Model and vendor portability | Covered | Policy, entitlements, evidence and agent state live outside the model; the gateway is provider-neutral | Architecture |
| 34 | Audit logging | Planned | Every item in the list is captured across 0.3 and 0.4 | 0.3; 0.4 |
| 35 | Explainability | Planned | Operational explainability: "why this client, why this recommendation", from origination evidence | Origination evidence (0.4) |
| 36 | Client communication controls | Partial | Governed communication planned. Added: disclosures, prohibited claims, risk language, research attribution, channel rules | Governed client communication (0.4) |
| 37 | Record retention | Planned | Retention store with per-use-case retention; erasure evidenced | 0.3; 0.4 |
| 38 | Workforce controls | Out of scope | Training is the bank's. Kognita can require a training attestation as a policy attribute before a user may invoke an agent | Policy attribute (any release) |
| 39 | Risk tiering | Partial | Agents carry a materiality tier today. Added: tier per use case drives required approvals, review rate, autonomy cap and monitoring | Use-case register (0.4) |
| 40 | Independent validation | Added | Activating a use case or agent at high tiers requires sign-off from distinct functions, enforced as separation of duties | Use-case register (0.4) |

## What Kognita deliberately does not become

Five domains are marked **Boundary**: the truth layer, calculation engines, evaluation harnesses, sandboxes and challenger models. Kognita's role in each is the same. It refuses to let an output through unless the authoritative system was used, and it records that it was. Building those systems inside Kognita would contradict the design rule above, because it would make Kognita the thing being governed.
