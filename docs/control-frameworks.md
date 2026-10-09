# AI Control Frameworks: Wealth Management and Agentic AI Coverage

**Last revised:** 8 October 2026. Companion to [ROADMAP.md](ROADMAP.md).

**Attribution:** This document maps two control frameworks and one supervisory guideline against Kognita:
- A **40-domain bank-grade agentic AI control framework** for wealth management and private banking, drawing on supervisory material (MAS, ABS), security references (NIST, OWASP), and production patterns described for large banks.
- A **70-control wealth management framework**, consolidating material from MAS, PDPC, SFC, FCA, the Bank of England, APRA and ASIC, CSBS, BIS/FSI and ESMA.
- Integration of the **EVOLVE framework** for continuous learning and institutional memory, authored by Ahmed Muzammil under Creative Commons Attribution Share-Alike 4.0 (CC BY SA 4.0).
- The **MAS Guidelines on Artificial Intelligence Risk Management**, issued 7 October 2026, mapped paragraph by paragraph in [section VII](#vii-mas-guidelines-on-ai-risk-management-october-2026).

The frameworks overlap heavily. Statuses below are measured against the current roadmap.

## The Overarching Test

> **Can the bank explain, control, stop and reconstruct every material AI action that affects a client?**

Kognita adopts this as the single test every release is measured against. *Explain* is citations and origination evidence; *control* is the decision point, authority and limits; *stop* is revocation, intervention and containment; *reconstruct* is pinned evidence and the reconstruction report.

## The Design Rule

> **Never rely on the LLM to enforce a control that can be enforced outside the LLM.**
>
> Do not govern the model. Govern the complete chain from human intent → agent authority → data → reasoning → tools → action → outcome.

This is Kognita's founding premise and the core of the EVOLVE framework's governance approach.

## Status Legend

| Status | Meaning |
|---|---|
| **Covered** | Shipped (v0.3.0 or earlier) |
| **Planned** | On the roadmap |
| **Added** | A gap found by audit, now on the roadmap |
| **Gap** | A gap found by audit, not yet on the roadmap |
| **Partial** | Some shipped or planned; the rest added |
| **Boundary** | Kognita enforces and evidences; capability belongs to another system or contracts |
| **Out of scope** | Not a software control Kognita can provide |

## Naming Collisions to Avoid

The frameworks use labels that clash with Kognita's own:
- The 70-control framework's **C0 to C5** client-impact scale is unrelated to Kognita's **C1 to C3** data classification. We call the framework's scale the **client-impact class** (internal productivity, RM assistance, client influence, client communication, advice, execution).
- The 70-control framework's **L0 to L5** communication scale is unrelated to Kognita's **L0 to L6** agent autonomy levels. We call the framework's scale the **communication level** (general information, house view, contextualised insight, investment discussion, product recommendation, transaction).
- The EVOLVE framework's five maturity **stages** (Experimental, Enabled, Embedded, Empowered, Evolved) are adoption milestones, not enforceable controls. Kognita enforces what each stage requires, not the stage itself.

---

## I. The Twelve Non-Negotiables (Bank-Grade Framework)

| # | Control | Status | Kognita Today | Delivered By |
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

---

## II. The Fifteen Non-Negotiables (Wealth Management Framework)

| # | Control | Status | Delivered By |
|---|---|---|---|
| 1 | AI use-case inventory and materiality classification | Partial | Use-case register (0.4) |
| 2 | Named business owner | Partial | Use-case register (0.4); Knowledge Lead role added (0.4) |
| 3 | Unique agent identity | Partial | Tier 0 (0.3); agent identity (0.5) |
| 4 | Explicit delegated authority | Partial | Delegated authority (0.5) |
| 5 | Least-privilege data and tool access | Planned | Tool allow-lists and field minimization (0.5) |
| 6 | Separation of recommendation and execution authority | Partial | Autonomy and action permissions (0.5) |
| 7 | Runtime policy enforcement | **Covered** | Core |
| 8 | Human approval for consequential client actions | Partial | Suspend/resume (0.3); RM review (0.4); autonomy levels (0.5) |
| 9 | "Why this client?" and "why this recommendation?" | Partial | Origination evidence (0.4); bare scores rejected |
| 10 | Approved-source grounding and research expiry | Partial | Approved-source grounding (0.6); supersession (0.6) |
| 11 | Complete provenance and audit trail | Planned | Pinned evidence (0.3); interaction record (0.4); claims (0.6) |
| 12 | Pause, override, revoke and kill | Partial | Intervention controls (0.4); granular revocation (0.4) |
| 13 | Agent-behaviour monitoring and automated containment | Partial | Anomaly detection and containment (0.5) |
| 14 | Third-party dependency and exit controls | Partial | Provider register (0.7); dependency map (0.7) |
| 15 | Continuous testing, revalidation and incident learning | **Partial** | Evaluation gate (0.5, 0.6); **evidence-gated autonomy promotion and automatic step-down (0.5)** |

---

## III. The Ten Domains (Wealth Management Framework)

| Domain | Core Question | Where Kognita Answers It |
|---|---|---|
| Governance | Who owns the AI and remains accountable? | Use-case register, owner roles including Knowledge Lead (0.4), risk appetite (0.4) |
| Risk classification | How consequential is this use case? | Multi-axis classification (0.4); outcome metrics and primary metric (0.4) |
| Identity and authority | Who is the agent and what may it do? | Agent identity, delegated authority, autonomy with evidence gates (0.5) |
| Data and privacy | What may it access and retain? | Entitlement filtering (core); minimization, lanes (0.5); governed memory including institutional memory (0.6) |
| Model and agent assurance | Does it behave correctly and reliably? | Evaluation gate (0.5, 0.6); drift tracking per agent/model/prompt (0.5); adversarial suite (0.5) |
| Client and conduct | Could it affect advice, suitability or communications? | Origination, communication levels, suitability (0.4); claims (0.6) |
| Human control | Where can humans understand, intervene and override? | RM review with outcome metrics (0.4); intervention controls (0.4); Knowledge Lead oversight (0.4) |
| Runtime and cyber | Can abnormal behaviour be detected and contained? | Anomaly detection, containment, taint tracking (0.5); automatic step-down on drift (0.5) |
| Resilience and third parties | What happens when providers or systems fail? | Gateway failure mode (0.3); resilience track (0.7) |
| Evidence and auditability | Can the bank reconstruct exactly what happened? | Hash chain (core); pinned evidence, reconstruction (0.3); institutional memory versioning (0.6) |

---

## IV. The Forty Domains (Bank-Grade Framework)

*(See the comprehensive 40-domain table in [control-framework.md](control-framework.md), covering:)*

1. Human identity and accountability
2. Agent identity
3. Delegated authority
4. Purpose limitation
5. Least privilege and entitlements
6. Data governance, client truth layer
7. Source and provenance controls
8. Facts vs inferences vs recommendations
9. Data freshness
10. Deterministic financial calculations
... *(30 more; full table on the separate document)*

---

## V. The Seventy Controls (Wealth Management Framework)

*(Comprehensive 70-control table covering all domains, with status and delivery timelines. Excerpted below; full table on the separate document.)*

**Notable updates for EVOLVE integration:**

| # | Control | Status | Kognita's Answer | Delivered By |
|---|---|---|---|---|
| 2 | Materiality and risk classification | Added | Multi-axis classification; outcome metrics per use case with baseline and target | Use-case register (0.4) |
| 15 | Continuous testing, revalidation and incident learning | **Partial** | Evaluation gate planned; **evidence gates for autonomy promotion and automatic step-down on drift added** | 0.4; 0.5; 0.6 |
| 32 | Continuous testing | Partial | Periodic review planned; re-authorisation enforced by review dates; **quality trends tracked per agent, model, prompt** | 0.4; 0.5 |
| 58 | Automation-bias controls | Partial | Approval rate and time; **edit, rejection and escalation rates; decision quality tracked per use case** | RM review (0.4); outcome metrics (0.4) |
| 61 | Autonomy escalation approval | **Planned** | **Raising autonomy requires evidence thresholds on outcome metrics; independent validation still required on top** | 0.5 |

*(Full 70-control table preserved; updates above highlight EVOLVE-informed changes.)*

---

## VI. EVOLVE Integration

The EVOLVE framework, authored by Ahmed Muzammil (CC BY SA 4.0), contributes three critical capabilities to the roadmap:

### 1. Outcome Metrics and the Knowledge Lead (0.4)

Each use case registers:
- **One primary metric** (cycle time, decision quality, or reliability) with a baseline and target, computed from Kognita's own records
- **Knowledge Lead:** a named owner of the quality standard, separate from the business owner, who approves consequential outputs, receives escalations, and approves changes to the standard
- A tier 2 or higher use case with no named Knowledge Lead is denied

### 2. Evidence-Gated Autonomy with Automatic Step-Down (0.5)

- **Promotion gates:** Raising an agent's autonomy level requires evidence thresholds on outcome metrics (quality ≥ 95%, exception rate < 5%, full traceability). Owner approval and independent validation remain required on top.
- **Automatic step-down:** When metrics fall below a lower threshold over a rolling window, the agent's autonomy drops one level on the next decision, with alerts to the owner and Knowledge Lead.
- **Quality drift:** Metrics are tracked as trends per agent, model version and prompt version, so gradual degradation is caught before a hard breach.

### 3. Institutional Memory (0.6)

- A pattern in RM corrections or outcomes becomes a *proposed change* to the shared standard through the proposal model (ADR 0007), carrying evidence: which interactions, which corrections, which metrics moved.
- The **Knowledge Lead** approves or rejects it. Approved changes produce a new deployment version, so the standard is versioned, effective-dated, and re-approved like any behaviour change.
- **Standards are effective-dated:** Every output records which standard version produced it, and an expired or superseded standard cannot be used.
- Client data never crosses into institutional memory; feedback-loop controls apply to any learning that leaves Kognita.

### 4. Governed Business Definitions (0.6)

- The bank registers governed terms (AUM, concentration, etc.) with an authoritative definition and version.
- Claims using governed terms must cite the registered definition and version, or are blocked (same pattern as calculation provenance).
- `kognita evidence affected --definition <id>` lists claims, recommendations and communications produced under a superseded version.

---

## VII. MAS Guidelines on AI Risk Management (October 2026)

MAS issued the final Guidelines on Artificial Intelligence Risk Management on 7 October 2026, after consultation P017-2025. They apply to all financial institutions and all forms of AI, including generative AI and AI agents, in proportion to size and risk profile (paragraphs 1.6, 2.1). They take effect on 7 October 2027. Sections 3 and 4 apply from that date; Sections 5 and 6 by 7 October 2028 (1.8).

Two caveats govern every row below:
- **Kognita sees only the AI it mediates.** MAS expects identification and inventory of all AI use, including AI embedded in third-party services and shadow AI (4.2 to 4.4). Kognita can hold that inventory, but it cannot by itself find AI that never passes through it.
- **Kognita's patterns are not MAS-mandated controls.** The guidelines are principles-based. Kognita supplies enforcement and evidence that a bank meets an expectation; how the bank meets it remains the bank's choice.

### Timing

| MAS deadline | Sections | Kognita releases that land before it |
|---|---|---|
| 7 October 2027 | 3 (AI oversight), 4 (identification, inventory, risk materiality) | 0.4 (Q1 2027): use-case register, materiality classification, risk appetite, owner roles, senior-management report |
| 7 October 2028 | 5 (life cycle controls), 6 (capability and capacity) | 0.5 (Q2 2027), 0.6 (Q3 2027), 0.7 (Q4 2027), 1.0 (Q1 2028) |

One timing conflict was found and resolved: the single-provider concentration measure that MAS cites as a risk-appetite measure (3.4(b), footnote 18) depended on the dependency map in 0.7, which lands after the Section 3 deadline. A single-provider dependency count now ships in the 0.4 register (gap 11).

### Section 2: Basic AI Governance (2.3 to 2.5)

AI whose poor performance or unavailability is unlikely to have a material adverse impact may run under basic policies (2.3). MAS's examples include drafting emails, internal summaries and internal search chatbots (2.4).

| Para | Expectation | Status | Kognita | Release |
|---|---|---|---|---|
| 2.5(a) | Senior manager accountable for AI oversight | Planned | Accountable owner per use case; agents cannot hold owner roles | 0.4 |
| 2.5(b) | Permitted and prohibited uses; no client data into public AI tools | Covered | Egress guard decides allow, redact or deny per classification and destination | Core |
| 2.5(b) | Mandatory human review before use | Covered | `HUMAN_APPROVAL` outcome; durable suspend and resume | Core, 0.3 |
| 2.5(c) | Approved list of AI tools and an approval process for new ones | Partial | Agent and tool registry today; use-case register with governed registry changes | Core, 0.4 |
| 2.5(d) | Staff education on AI policies | Boundary | Training attestation checked as a policy attribute; the training is the bank's | 0.4 |
| 2.5(e)(f) | Compliance checks; periodic and trigger-based review of materiality | Planned | Review dates per tier; revalidation on change of autonomy, jurisdiction or intended use | 0.4 |

### Section 3: AI Oversight

| Para | Expectation | Status | Kognita | Release |
|---|---|---|---|---|
| 3.2 to 3.4(a)(c)(d)(e) | Board approves the AI governance approach, sets roles, understands AI, reviews regularly | Out of scope | Governance decisions belong to the board | — |
| 3.4(b) | AI risk in the risk appetite framework, qualitative and quantitative | Added | Risk appetite as top-level policy: prohibited categories, maximum autonomy, client impact; quantitative measures computed from evidence (gap 12) | 0.4 |
| 3.5(a)(c) | Implement frameworks consistent with appetite; controls across the life cycle | Planned | A use case outside the appetite cannot be registered; tier sets minimum controls | 0.4 |
| 3.5(d) | Clear roles, including control functions for identification, inventory and materiality | Added | Five owner roles, an accountability matrix, and a designated control function (gap 2) | 0.4 |
| 3.5(e) | Escalation of AI incidents and breaches of risk thresholds | Planned | Circuit breaker, intervention controls, outcomes and near misses (0.4); step-down alerts (0.5) | 0.4, 0.5 |
| 3.5(f) | Timely board updates on material AI risk | Planned | Senior-management report of every material AI system, its classification, owner, status and open findings | 0.4 |
| 3.5(g) | Competent personnel and adequate resources | Out of scope | — | — |
| 3.6 | Local senior management visibility and escalation for Singapore | Partial | Jurisdiction and booking centre are envelope attributes; reports filtered by jurisdiction are not yet specified | 0.4 |

### Section 4: Identification, Inventory and Risk Materiality

| Para | Expectation | Status | Kognita | Release |
|---|---|---|---|---|
| 4.2 | Consistent identification of AI use, including AI in material third-party services | Added | Unmediated AI entries in the register; gateways flag calls to unregistered AI endpoints (gap 1). Finding AI outside the gateways remains the bank's | 0.4 |
| 4.3, 4.9, 4.13 | A designated control function is final arbiter of what is AI, owns inventory policy, and approves materiality | Added | Designated control function decides what is AI and approves each materiality rating (gap 2) | 0.4 |
| 4.4 | Residual risk from unidentified AI within appetite; mitigants such as network monitoring and DLP | Partial | Gateways deny unregistered agents and egress guard redacts (0.3); flags on unregistered AI endpoints (gap 1, 0.4); network monitoring and DLP are the bank's | 0.3, 0.4 |
| 4.5 | Accurate inventory with update frequency for new, changed and decommissioned AI | Planned | Use-case register; registry changes are evidenced policy changes; `kognita usecase list` | 0.4 |
| 4.6 | Links to other inventories (data assets, vendors, outsourcing registers) | Added | Each entry links to data sources and vendors using the identifiers in the bank's data inventory and outsourcing register (gap 5) | 0.4 |
| 4.7 | Attributes: purpose, approved scope (jurisdiction), model type, data, dependencies, lifecycle status, materiality, review status, roles, documentation | Planned | Use-case register fields, model card per approved model, review dates | 0.4 |
| 4.7, fn 25 | For agents: identifiers, tools and systems accessible, components, guardrails | Planned | Agent registry: identity, tool allow-list, permitted systems, autonomy level, blast-radius limits, behaviour-version hash | 0.5 |
| 4.8 | Inventory design reviewed for newer AI technologies | Boundary | The register schema is versioned; the review is the bank's | — |
| 4.10, 4.12(a) | Materiality methodology; impact on the FI and customers, data sensitivity | Planned | Six axes: business criticality, client-impact class, decision consequence, data sensitivity, autonomy, complexity | 0.4 |
| 4.12(b) | Complexity: technology, novelty, explainability, visibility into third-party AI | Added | Complexity added as a sixth axis, mapped to MAS's impact, complexity and reliance (gap 4) | 0.4 |
| 4.12(c) | Reliance, including autonomy and degree of human oversight | Planned | Autonomy axis (0.4); autonomy levels L0 to L6 (0.5) | 0.4, 0.5 |
| 4.11 | Inherent and residual materiality; residual within appetite before deployment | Added | Inherent, credited controls and residual recorded; activation denied while residual is outside appetite (gap 3) | 0.4 |

### Section 5: AI Life Cycle Controls

| Para | Expectation | Status | Kognita | Release |
|---|---|---|---|---|
| 5.2, fn 28 | Pilots and phased rollouts: time and user limits, success criteria, close monitoring | Added | Pilot status with end date, named users and success criteria (gap 6) | 0.4 |
| 5.3 | Contingency plans for high-risk AI; kill-switch activation protocols tested regularly | Partial | Per-agent kill switch (Core); gateway failure mode per use case (0.3); kill-switch drills (gap 7, 0.4); failure drills for providers and tools (0.7) | Core, 0.3, 0.4, 0.7 |
| 5.4(a)(b)(c) | Data fit for purpose, representative, high quality | Boundary | Fact contract and freshness checks on facts used at run time (0.6); training and test data quality is the bank's | 0.6 |
| 5.4(d) | Data classification guides use | Covered | Classification ceilings and zones filter data before retrieval; classifier envelopes at the decision boundary | Core, 0.3 |
| 5.4(e) | Data security, including destruction of outputs no longer required | Covered | Content-addressed retention store with retention per use case; erasure deletes bytes and keeps the chain verifiable | 0.3 |
| 5.4(f) | Data privacy, consent for sensitive personal data | Partial | Egress redaction; purpose-bound access; consent records are the bank's | Core |
| 5.4(g) | Auditability and lineage of data | Partial | Retrieval records content hash and embedding model per item; training-data lineage is the bank's | 0.3 |
| 5.5, 5.6 | Transparency and explainability proportionate to materiality; inform customers; redress | Planned | Cited decisions (Core); reconstruction report (0.3); reasons instead of scores, client AI-disclosure rules (0.4) | Core, 0.3, 0.4 |
| 5.7, 5.8 | Define fair outcomes; fairness assessments on protected attributes and proxies | Partial | Fairness reporting across bank-defined segments, including proxy attributes, as a monitored metric (gap 8). The definition of fair is the bank's | 0.6 |
| 5.9(a)(b) | Roles for human oversight; authority and ability to intervene | Planned | Owner roles (0.4); intervention controls: observe, pause, override, restrict, recover (0.4) | 0.4 |
| 5.9(c) | Design for oversight; escalate where reliability conditions are met | Covered | `ESCALATE` below a classifier confidence threshold; `HUMAN_APPROVAL` holds | 0.3 |
| 5.9(d) | Logs of oversight; review of interventions, incidents and near misses; automation bias | Planned | Approval rate and time, edit and rejection rates, outcomes and near misses | 0.4 |
| 5.10 | Test third-party AI in the FI's context; compensating tests; contractual visibility of changes | Boundary | Model card records the evaluation relied on (0.4); testing and contracts are the bank's | 0.4 |
| 5.11 | Limit, suspend or replace a third-party AI service when residual risk exceeds appetite | Added | Revocation by provider at the gateway (gap 10, 0.4); provider failover under policy (0.7) | 0.4, 0.7 |
| 5.11(b) | Supply chain assessment of models, datasets, dependencies | Boundary | Dependency map (0.7); the assessment is the bank's | 0.7 |
| 5.11(c) | Concentration risk | Planned | Share of critical use cases per provider, cloud and framework | 0.7 |
| 5.11(d) | Notification and assessment of third-party changes | Planned | Pinned provider model version per call (0.3); silent model-change detection (0.7) | 0.3, 0.7 |
| 5.11(e) | Contingency for third-party failure or discontinued support | Planned | Gateway failure mode (0.3); failover and failure drills (0.7) | 0.3, 0.7 |
| 5.12, 5.13 | Selection of systems, models and features | Out of scope | — | — |
| 5.14 | Evaluation thresholds agreed by business owners, developers and reviewers | Boundary | Evaluation evidence gate requires recorded results before release; the harness is the bank's | 0.6 |
| 5.15 | Guardrails tested for agent failure modes | Planned | Conformance kit (Core); adversarial suite (0.5) | Core, 0.5 |
| 5.16(a) | Secure deployment: input validation, API authentication, encryption, DLP | Partial | Egress guard and gateway decide before forwarding (0.3); agent workload credentials (0.5); infrastructure hardening is the bank's | 0.3, 0.5 |
| 5.16(b) | Role-based access, MFA, separation of duties | Partial | Two-signature approval for separation of duties (Core); human authentication is the bank's identity provider | Core |
| 5.16(c) | Govern plugins and APIs; restrict data exposure | Covered | MCP proxy authorises every tool call; tool arguments evidenced as hashes | 0.3 |
| 5.17 | Documentation sufficient for an independent party to replicate | Partial | Run-time reproducibility: pinned policy, retrieval, model and tool hashes and the reconstruction report (0.3). Development documentation is the bank's | 0.3 |
| 5.18 to 5.21 | Independent pre-deployment review; formal independent validation for high materiality | Planned | Tier 3 and 4 activation needs independent sign-off, enforced as separation of duties | 0.4 |
| 5.22 | Technology and cybersecurity review; red teaming; anomaly detection | Partial | Anomaly detection and containment (0.5); adversarial suite (0.5); penetration testing is the bank's | 0.5 |
| 5.23(a) | Monitoring metrics, tiered thresholds with early warning, drift; monitoring of actions and tools used | Planned | Outcome metrics per use case (0.4); evidence-gated autonomy with two thresholds and drift tracking (0.5); every action and tool call evidenced (Core) | 0.4, 0.5 |
| 5.23(b) | Incident management; kill switches for high-risk AI; user feedback | Partial | Kill switch (Core); circuit breaker, outcomes and near misses (0.4); reviewer edits and rejections are feedback (0.4) | Core, 0.4 |
| 5.23(c) | Accountable person for monitoring and incidents | Planned | Accountable owner per use case | 0.4 |
| 5.23(d) | Records of monitoring; access logged; prompts, responses, model versions | Covered | Hash-chained evidence; `MODEL_CALL` records provider, model version, prompt template version, prompt and response hashes, with content in the retention store | 0.3 |
| 5.24 | Re-validation, independent for high materiality; triggered by alerts and changes | Planned | Review dates per tier; revalidation on material change (0.4); drift findings count toward review (0.5) | 0.4, 0.5 |
| 5.25(a)(b) | Significance of changes; re-approval; version control and rollback | Planned | Policy rows effective-dated, edits refused (0.3); behaviour-version hash over prompt, tools, permissions and data needs re-approval (0.5) | 0.3, 0.5 |
| 5.25(c) | Enhanced controls on automatic updates | Planned | Learned changes to standards need Knowledge Lead approval and become new versions (0.6) | 0.6 |
| 5.26 | Retirement and decommissioning | Added | A retired use case is a DENY; retirement stops its agents, applies retention and notifies dependants (gap 9) | 0.4 |

### Section 6: AI Capability and Capacity

| Para | Expectation | Status | Kognita |
|---|---|---|---|
| 6.1, 6.2 | Competence, training and resources | Out of scope | Training attestation can gate use (0.4); the programme is the bank's |
| 6.3 | Technology infrastructure adequate for AI | Out of scope | Gateway overhead is benchmarked under 50 ms (0.3) |

### Gaps Found by This Audit

All twelve were adopted into the roadmap on 8 October 2026: gaps 1 to 7 and 9 to 12 in 0.4 (use-case register, item 8; revocation, item 14) and gap 8 in 0.6 (item 11).

1. **Identification of unmediated AI (4.2, 4.4).** Register entries for AI that does not pass through Kognita, such as vendor-embedded features and approved copilots, recording the assurance gap and the compensating control. The gateways flag traffic to AI endpoints that no registered use case names, as a discovery signal for the control function.
2. **Designated control function (4.3, 4.9, 4.13).** A control-function role, distinct from the five owner roles, that decides whether a use is AI and approves each materiality rating. The tier derived from classification stays a proposal until approved.
3. **Inherent and residual materiality (4.11).** Record inherent materiality, the controls credited, and residual materiality per use case. Activation is a DENY while residual materiality is outside the risk appetite.
4. **Complexity dimension (4.12(b)).** Add complexity to the classification: technology type, novelty, explainability of outputs, and visibility into third-party AI. Map the classification to MAS's three minimum dimensions: impact, complexity and reliance.
5. **Shared inventory identifiers (4.6).** Data-asset and vendor identifiers that match the bank's data inventory and outsourcing register, so entries cross-reference.
6. **Pilot scope (5.2, fn 28).** A pilot status with an end date, a named user list and success criteria. Calls outside the user list or after the end date are denied.
7. **Kill-switch drills (5.3).** High-materiality use cases record kill-switch activation drills as evidence, with an overdue drill flagged like a past review date.
8. **Proxy attributes in fairness (5.8, fn 36).** Fairness reporting covers attributes that substitute for or correlate with protected attributes, and fairness can be a monitored metric with thresholds.
9. **Decommissioning (5.26).** Retiring a use case also revokes its agents' credentials, applies the retention policy to its records, and notifies dependent use cases.
10. **Per-provider suspend (5.11).** Suspend or limit a provider at the gateway across all use cases, separate from failover, so the bank can act when residual risk exceeds appetite.
11. **Concentration measure before October 2027 (3.4(b), fn 18).** Bring the count of material use cases depending on a single provider forward from the 0.7 dependency map into the 0.4 register, since it belongs to the Section 3 risk appetite.
12. **Quantitative risk-appetite measures (3.4(b), fn 18).** Thresholds on the number or impact of AI incidents, single-provider dependencies, and use cases breaching performance thresholds, computed from the evidence log and reported against the appetite.

---

## VIII. What Kognita Deliberately Does Not Become

Five domains are marked **Boundary**: the truth layer, calculation engines, evaluation harnesses, sandboxes and challenger models. Kognita's role in each is the same. It refuses to let an output through unless the authoritative system was used, and it records that it was. Building those systems inside Kognita would contradict the design rule above, because it would make Kognita the thing being governed.

---

## IX. Attribution and Licensing

- **40-domain bank-grade framework:** consolidated from supervisory material (MAS, ABS), security references (NIST, OWASP), and production patterns.
- **MAS Guidelines on Artificial Intelligence Risk Management,** Monetary Authority of Singapore, 7 October 2026. Paragraph numbers in section VII refer to that document; the mapping is Kognita's reading, not MAS guidance.
- **70-control wealth management framework:** consolidated from MAS, PDPC, SFC, FCA, the Bank of England, APRA and ASIC, CSBS, BIS/FSI and ESMA.
- **EVOLVE framework integration:** The continuous learning loop, institutional memory, Knowledge Lead role, evidence-gated autonomy and drift tracking are based on the EVOLVE AI-native enterprise framework, authored by **Ahmed Muzammil**, licensed under **Creative Commons Attribution Share-Alike 4.0 (CC BY SA 4.0)**.

This document and the referenced frameworks are published under the same license as Kognita (MIT). The EVOLVE contributions are used and adapted under CC BY SA 4.0.
