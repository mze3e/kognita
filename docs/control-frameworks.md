# AI Control Frameworks: Wealth Management and Agentic AI Coverage

**Last revised:** 7 October 2026. Companion to [ROADMAP.md](ROADMAP.md).

**Attribution:** This document maps two control frameworks against Kognita:
- A **40-domain bank-grade agentic AI control framework** for wealth management and private banking, drawing on supervisory material (MAS, ABS), security references (NIST, OWASP), and production patterns described for large banks.
- A **70-control wealth management framework**, consolidating material from MAS, PDPC, SFC, FCA, the Bank of England, APRA and ASIC, CSBS, BIS/FSI and ESMA.
- Integration of the **EVOLVE framework** for institutional memory, authored by Ahmed Muzammil under Creative Commons Attribution Share-Alike 4.0 (CC BY SA 4.0).

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
| **Covered** | Shipped in v0.2 |
| **Planned** | On the roadmap |
| **Added** | A gap found by audit, now on the roadmap |
| **Partial** | Some shipped or planned; the rest added |
| **Boundary** | Kognita enforces and evidences; capability belongs to another system or contracts |
| **Out of scope** | Not a software control Kognita can provide |

## Naming Collisions to Avoid

The frameworks use labels that clash with Kognita's own:
- The 70-control framework's **C0 to C5** client-impact scale is unrelated to Kognita's **C1 to C3** data classification. We call the framework's scale the **client-impact class** (internal productivity, RM assistance, client influence, client communication, advice, execution).
- The 70-control framework's **L0 to L5** scale is a communication scale. We call it the **communication level** (general information, house view, contextualised insight, investment discussion, product recommendation, transaction).
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
| 11 | Human approval for high-impact decisions | Partial | HUMAN_APPROVAL outcome; two-signature approvals | Suspend/resume (0.3) |
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
| 6 | Separation of recommendation and execution authority | Partial | Delegated authority and tool allow-lists (0.5) |
| 7 | Runtime policy enforcement | **Covered** | Core |
| 8 | Human approval for consequential client actions | Partial | Suspend/resume (0.3); RM review (0.4) |
| 9 | "Why this client?" and "why this recommendation?" | Partial | Origination evidence (0.4); bare scores rejected |
| 10 | Approved-source grounding and research expiry | Partial | Approved-source grounding (0.6); supersession (0.6) |
| 11 | Complete provenance and audit trail | Planned | Pinned evidence (0.3); interaction record (0.4); claims (0.6) |
| 12 | Pause, override, revoke and kill | Partial | Intervention controls (0.4); granular revocation (0.4) |
| 13 | Agent-behaviour monitoring and automated containment | Partial | Anomaly detection and containment (0.5) |
| 14 | Third-party dependency and exit controls | Partial | Provider register (0.7); dependency map (0.7) |
| 15 | Continuous testing, revalidation and incident learning | **Partial** | Evaluation gate (0.6); adversarial suite (0.5) |

---

## III. The Ten Domains (Wealth Management Framework)

| Domain | Core Question | Where Kognita Answers It |
|---|---|---|
| Governance | Who owns the AI and remains accountable? | Use-case register, owner roles including Knowledge Lead (0.4), risk appetite (0.4) |
| Risk classification | How consequential is this use case? | Multi-axis classification (0.4); outcome metrics and primary metric (0.4) |
| Identity and authority | Who is the agent and what may it do? | Agent identity, delegated authority (0.5) |
| Data and privacy | What may it access and retain? | Entitlement filtering (core); minimization, lanes (0.5); governed memory including institutional memory (0.6) |
| Model and agent assurance | Does it behave correctly and reliably? | Evaluation gate (0.6); adversarial suite (0.5) |
| Client and conduct | Could it affect advice, suitability or communications? | Origination, communication levels, suitability (0.4); claims (0.6) |
| Human control | Where can humans understand, intervene and override? | RM review with outcome metrics (0.4); intervention controls (0.4); Knowledge Lead oversight (0.4) |
| Runtime and cyber | Can abnormal behaviour be detected and contained? | Anomaly detection, containment, taint tracking (0.5) |
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
| 15 | Continuous testing, revalidation and incident learning | **Partial** | Evaluation gate planned; near misses; adversarial suite | 0.4; 0.5; 0.6 |
| 32 | Continuous testing | Partial | Periodic review planned; re-authorisation enforced by review dates | 0.4; 0.5 |
| 58 | Automation-bias controls | Partial | Approval rate and time; **edit, rejection and escalation rates** | RM review (0.4) |

*(Full 70-control table preserved; updates above highlight EVOLVE-informed changes.)*

---

## VI. EVOLVE Integration

The EVOLVE framework, authored by Ahmed Muzammil (CC BY SA 4.0), contributes three critical capabilities to the roadmap:

### 1. Outcome Metrics and the Knowledge Lead (0.4)

Each use case registers:
- **One primary metric** (cycle time, decision quality, or reliability) with a baseline and target, computed from Kognita's own records
- **Knowledge Lead:** a named owner of the quality standard, separate from the business owner, who approves consequential outputs, receives escalations, and approves changes to the standard
- A tier 2 or higher use case with no named Knowledge Lead is denied

### 2. Institutional Memory (0.6)

- A pattern in RM corrections or outcomes becomes a *proposed change* to the shared standard through the proposal model (ADR 0007), carrying evidence: which interactions, which corrections, which metrics moved.
- The **Knowledge Lead** approves or rejects it. Approved changes produce a new deployment version, so the standard is versioned, effective-dated, and re-approved like any behaviour change.
- **Standards are effective-dated:** Every output records which standard version produced it, and an expired or superseded standard cannot be used.
- Client data never crosses into institutional memory; feedback-loop controls apply to any learning that leaves Kognita.

### 3. Governed Business Definitions (0.6)

- The bank registers governed terms (AUM, concentration, etc.) with an authoritative definition and version.
- Claims using governed terms must cite the registered definition and version, or are blocked (same pattern as calculation provenance).
- `kognita evidence affected --definition <id>` lists claims, recommendations and communications produced under a superseded version.

---

## VII. What Kognita Deliberately Does Not Become

Five domains are marked **Boundary**: the truth layer, calculation engines, evaluation harnesses, sandboxes and challenger models. Kognita's role in each is the same. It refuses to let an output through unless the authoritative system was used, and it records that it was. Building those systems inside Kognita would contradict the design rule above, because it would make Kognita the thing being governed.

---

## VIII. Attribution and Licensing

- **40-domain bank-grade framework:** consolidated from supervisory material (MAS, ABS), security references (NIST, OWASP), and production patterns.
- **70-control wealth management framework:** consolidated from MAS, PDPC, SFC, FCA, the Bank of England, APRA and ASIC, CSBS, BIS/FSI and ESMA.
- **EVOLVE framework integration:** Outcome metrics, institutional memory and the Knowledge Lead role are based on the EVOLVE AI-native enterprise framework, authored by **Ahmed Muzammil**, licensed under **Creative Commons Attribution Share-Alike 4.0 (CC BY SA 4.0)**.

This document and the referenced frameworks are published under the same license as Kognita (MIT). The EVOLVE contributions are used and adapted under CC BY SA 4.0.
