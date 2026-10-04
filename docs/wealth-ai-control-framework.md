# Wealth Management AI and Agentic AI Control Framework: Coverage

**Last revised:** 4 October 2026. Companion to [ROADMAP.md](ROADMAP.md) and [control-framework.md](control-framework.md).

This maps a 70-control framework for RM-facing, client-facing and agentic AI in wealth management against Kognita. The framework consolidates material from MAS, PDPC, SFC, FCA, the Bank of England, APRA and ASIC, CSBS, BIS/FSI and ESMA. As its author notes, not every control is mandated by every regulator today: some flow from binding privacy, conduct and outsourcing rules, others are emerging supervisory expectations or prudent practice. It is used here as the target baseline.

It overlaps heavily with the 40-domain framework in [control-framework.md](control-framework.md). Statuses below are measured against the roadmap *after* that audit, so most of its additions show as **Planned** here.

## The overarching test

> **Can the bank explain, control, stop and reconstruct every material AI action that affects a client?**

Kognita adopts this as the single test every release is measured against. *Explain* is citations and origination evidence; *control* is the decision point, authority and limits; *stop* is revocation, intervention and containment; *reconstruct* is pinned evidence and the reconstruction report.

## Status legend

| Status | Meaning |
|---|---|
| **Covered** | Shipped in v0.2 |
| **Planned** | On the roadmap before this audit |
| **Added** | A gap found by this audit, now on the roadmap |
| **Partial** | Some planned; the rest added |
| **Boundary** | Kognita enforces and evidences the control; the capability belongs to another system or to contracts |
| **Out of scope** | Not a software control Kognita can provide |

## Naming collisions to avoid

The framework uses two labels that clash with Kognita's own:
- Its **C0 to C5** client-impact scale is unrelated to Kognita's **C1 to C3** data classification. The roadmap calls the framework's scale the **client-impact class** and uses its names (internal productivity, RM assistance, client influence, client communication, advice, execution), never C-codes.
- Its **L0 to L5** communication scale is unrelated to Kognita's **L0 to L6** agent autonomy levels. The roadmap calls the framework's scale the **communication level** and uses its names (general information, house view, contextualised insight, investment discussion, product recommendation, transaction).

## The fifteen non-negotiables

| # | Control | Status | Delivered by |
|---|---|---|---|
| 1 | AI use-case inventory and materiality classification | Partial: register planned with a single risk tier. Added: multi-axis classification and risk appetite | Use-case register (0.4) |
| 2 | Named business owner | Planned; agents already carry an accountable owner | Use-case register (0.4) |
| 3 | Unique agent identity | Partial: registry exists, but agent names are self-asserted. Added: agents authenticate with their own credentials | Tier 0 (0.3); agent identity (0.5) |
| 4 | Explicit delegated authority | Planned. Added: jurisdiction and approval threshold fields, expiry on task completion | Delegated authority (0.5) |
| 5 | Least-privilege data and tool access | Planned | Tool allow-lists and field minimization (0.5) |
| 6 | Separation of recommendation and execution authority | Partial: autonomy levels planned. Added: each action verb is a separate permission | Autonomy and action permissions (0.5) |
| 7 | Runtime policy enforcement | **Covered** | Core |
| 8 | Human approval for consequential client actions | Planned | Suspend/resume (0.3); RM review (0.4); autonomy levels (0.5) |
| 9 | "Why this client?" and "why this recommendation?" | Planned. Added: a "do not contact" outcome, and bare scores are not a valid rationale | Origination evidence (0.4) |
| 10 | Approved-source grounding and research expiry | Partial: freshness planned. Added: approved-source register and automatic invalidation on supersession | Claims (0.6) |
| 11 | Complete provenance and audit trail | Planned | Pinned evidence (0.3); interaction record (0.4); claims (0.6) |
| 12 | Pause, override, revoke and kill | Partial: per-agent kill switch exists, granular revocation planned. Added: observe, pause, override, restrict, recover | Intervention controls (0.4) |
| 13 | Agent-behaviour monitoring and automated containment | Partial: anomaly detection planned. Added: automatic containment sequence | Anomaly detection and containment (0.5) |
| 14 | Third-party dependency and exit controls | Partial: provider register planned. Added: dependency map, concentration risk, silent model-change detection | Resilience (0.7) |
| 15 | Continuous testing, revalidation and incident learning | Partial: evaluation gate planned. Added: revalidation triggers, adversarial suite, near-miss register | 0.4; 0.5; 0.6 |

## The ten domains

| Domain | Core question | Where Kognita answers it |
|---|---|---|
| Governance | Who owns the AI and remains accountable? | Use-case register, owner roles, risk appetite (0.4) |
| Risk classification | How consequential is this use case? | Multi-axis classification (0.4) |
| Identity and authority | Who is the agent and what may it do? | Agent identity, delegated authority, autonomy (0.5) |
| Data and privacy | What may it access and retain? | Entitlement filtering (core); minimization, lanes (0.5); memory (0.6) |
| Model and agent assurance | Does it behave correctly and reliably? | Evaluation gate, adversarial suite (0.5, 0.6), boundary |
| Client and conduct | Could it affect advice, suitability or communications? | Origination, communication levels, suitability (0.4); claims (0.6) |
| Human control | Where can humans understand, intervene and override? | RM review, intervention controls (0.4) |
| Runtime and cyber | Can abnormal behaviour be detected and contained? | Anomaly detection, containment, taint tracking (0.5) |
| Resilience and third parties | What happens when providers or systems fail? | Gateway failure mode (0.3); resilience track (0.7) |
| Evidence and auditability | Can the bank reconstruct exactly what happened? | Hash chain (core); pinned evidence, reconstruction (0.3) |

## All seventy controls

| # | Control | Status | Kognita's answer | Delivered by |
|---|---|---|---|---|
| 1 | Governance, ownership, accountability | Partial | Inventory, owner roles and periodic review planned. Added: AI risk appetite as top-level policy (prohibited use cases, maximum autonomy, transaction authority); accountability matrix extended to data, communication, suitability and compliance; senior-management inventory report | Use-case register (0.4) |
| 2 | Materiality and risk classification | Added | Classification on five axes (criticality, client-impact class, autonomy, decision consequence, data sensitivity); required controls derived from the result | Use-case register (0.4) |
| 3 | Agent identity | Partial | Registry exists. Added: agents authenticate with their own credentials; no shared credentials; environment and deployment date recorded | Tier 0 (0.3); agent identity (0.5) |
| 4 | Delegated authority | Partial | Authority object planned. Added: jurisdiction and approval-threshold fields | Delegated authority (0.5) |
| 5 | Sub-agent delegation | Partial | Attenuating delegation planned. Added: authority lineage query, "show me the chain of authority that permitted this action" | Delegation between agents (0.5) |
| 6 | Least privilege | Planned | Per-agent tool allow-lists and field minimization | 0.5 |
| 7 | Separation of reasoning and action | Added | Read, analyse, recommend, draft, queue, send and execute are separate permissions; recommending never implies executing | Autonomy and action permissions (0.5) |
| 8 | Human judgment boundaries | Partial | Required decision points per use case planned. Added: who reviews, what evidence they see, and the defined path on rejection | Use-case register; RM review (0.4) |
| 9 | Human intervention capability | Added | Observe, pause, override, restrict, revoke, recover, all governed and evidenced | Intervention controls (0.4) |
| 10 | Kill switch | Partial | Per-agent kill switch exists; granular revocation planned | 0.4 |
| 11 | Reversibility | Added | Actions declare whether they are reversible and register a compensating action; irreversible actions require stronger pre-execution controls | Reversibility (0.5) |
| 12 | Pre-execution policy check | Covered | Decide, record, then execute; denials return nothing | Core |
| 13 | Data-access control | Partial | Entitlement filtering, redaction and tokenisation exist. Field-level permissions and agent segregation planned | Core; 0.5 |
| 14 | Data minimisation | Planned | Field minimization; scope expiry | 0.5 |
| 15 | Sensitive-data controls | Covered / Planned | Redaction before external models; evidence holds hashes, never content, so nothing leaks into logs; provider training-use recorded; memory governed | Core; 0.5; 0.6; 0.7 |
| 16 | Agent memory | Partial | Working vs relationship memory planned. Added: retention, correction and expiry per client | Memory governance (0.6) |
| 17 | Public vs private data separation | Added | Client-data taint: a run that has touched client data can never reach a public-content tool, enforced in code | External content isolation and lanes (0.5) |
| 18 | Research provenance | Partial | Passage-level citations planned. Added: author, publication date and version on every source | Ingestion (0.4); fact contract (0.6) |
| 19 | Evidence grounding | Added | Approved-source register per use case; a regulated investment claim must cite an approved source; model knowledge alone is never a source | Approved-source grounding (0.6) |
| 20 | Research freshness and expiry | Partial | Freshness thresholds planned. Added: created, effective, reviewed, superseded and expiry dates; dependent recommendations stop automatically | Fact contract; supersession (0.6) |
| 21 | House-view change detection | Added | When a source is superseded, list affected clients, past recommendations, communications and open opportunities | Supersession propagation (0.6) |
| 22 | "Why this client?" | Planned | Origination evidence. Added: a bare relevance score is not an acceptable rationale | Origination (0.4) |
| 23 | "Why this recommendation?" | Planned | Origination and claim provenance; required detail rises with consequence | 0.4; 0.6 |
| 24 | Suitability boundary | Added | Six communication levels from general information to transaction, each with its own controls | Governed communication (0.4) |
| 25 | Communication controls | Planned | Channel, disclosures, jurisdiction, policy, source validity, suitability, RM approval | Governed communication (0.4) |
| 26 | "Do not contact" | Added | Origination supports a "no contact recommended" outcome with reasons; contact-frequency limits; vulnerability blocks | Origination (0.4) |
| 27 | Vulnerable-client safeguards | Partial | Sensitive-attribute controls planned. Added: flag for RM review, explain the signal, no diagnosis, no change of treatment without a governed process | Sensitive attributes (0.6) |
| 28 | Bias and fairness testing | Added | Fairness reporting from origination and recommendation evidence across segments the bank defines | Evaluation gate (0.6) |
| 29 | Model evaluation | Boundary | Results recorded against model versions and gate activation | Evaluation gate (0.6) |
| 30 | Agent evaluation | Boundary / Planned | Kognita measures permission compliance, client-boundary compliance, delegation, retries and escalation directly from evidence | Evaluation gate (0.6) |
| 31 | End-to-end evaluation | Added | The gate accepts system-level results covering model, retrieval, tools, data, agent, policies, UI and human interaction | Evaluation gate (0.6) |
| 32 | Continuous testing | Partial | Periodic review planned; re-authorisation enforced by review dates | 0.4; 0.5 |
| 33 | Revalidation triggers | Partial | Deployment version covers model, prompt, tools, permissions and sources. Added: autonomy, jurisdiction, client population and intended use | Agent identity (0.5); use-case register (0.4) |
| 34 | Adversarial testing | Added | Kognita ships an adversarial conformance suite that attacks its own controls: injection, wrong client, stale data, poisoned retrieval, privilege escalation, looping agents, policy bypass | Adversarial suite (0.5) |
| 35 | Prompt-injection defence | Planned | Treated as access control: classifier output only narrows permission; tainted runs cannot invoke write tools | 0.3; 0.5 |
| 36 | Behaviour monitoring | Partial | Anomaly detection planned. Added: privilege probing, unusual tool sequences, unexpected network access | Anomaly detection (0.5) |
| 37 | Machine-speed containment | Added | Automatic sequence: quarantine, revoke credential, stop tools, alert a human | Containment (0.5) |
| 38 | Blast-radius controls | Planned | Per-agent limits; network and credential segmentation is the bank's | 0.5; boundary |
| 39 | Cyber threat modelling | Added | Published threat model covering "our agent fails" and "an external agent attacks us"; external agents treated as untrusted principals | Threat model (0.5) |
| 40 | Incident response | Partial | Detect and contain planned; playbooks are the bank's | 0.4; 0.7 |
| 41 | Near-miss reporting | Added | Near-miss register: blocked actions, RM rejections, unsafe drafts, verifier blocks, unusual behaviour, linked to interactions and fed into review | Outcomes and near misses (0.4) |
| 42 | Operational-resilience testing | Partial | Backup and restore drills planned. Added: failure simulations for model, vendor, API, retrieval, identity and tool outages | Resilience (0.7) |
| 43 | Degraded mode | Boundary | Kognita's governance failure mode is planned. Keeping RM work running without AI is the bank's business-continuity design | 0.3; boundary |
| 44 | Third-party AI risk | Partial | Provider register planned. Added: incident history and regulatory cooperation | Provider register (0.7) |
| 45 | Dependency mapping | Added | Business service, agent, model, cloud, vector store, tool and data source, built from the registers and checked against what evidence shows was actually called | Dependency map (0.7) |
| 46 | Vendor change notification | Partial | Contracts are the bank's. Added: the gateway detects a change in provider-reported model version and treats it as a revalidation trigger | Provider register (0.7) |
| 47 | Vendor incident notification | Boundary | Terms recorded in the provider register; contracts are the bank's | 0.7 |
| 48 | Provider exit strategy | Planned | Exit plan in the register; provider-neutral architecture | 0.7; core |
| 49 | Concentration risk | Added | Share of critical use cases on each provider, cloud, framework and vector platform, computed from the dependency map | Dependency map (0.7) |
| 50 | Immutable audit trail | Planned | Hash chain today; signing and external anchoring planned | Core; 0.7 |
| 51 | Provenance envelope | Planned | Interaction record and reconstruction report | 0.3; 0.4 |
| 52 | Replayability | Planned | Pinned evidence and reconstruction | 0.3 |
| 53 | Record retention | Partial | Retention per use case planned. Added: retention per record class (communications, advice, recommendations, approvals, transactions) so AI records never expire before the regulatory record they support | Retention store (0.3); 0.7 |
| 54 | RM edit capture | Planned | Structured diff of RM changes | RM review (0.4) |
| 55 | Feedback-loop controls | Added | Exporting interactions to any learning pipeline is a governed action: purpose, anonymisation, approval and exclusions, all evidenced | Feedback-loop controls (0.6) |
| 56 | AI-generated content identification | Added | Every piece of content carries a provenance label (human, AI, AI then human-edited, human with AI assistance) that survives edits | RM review (0.4) |
| 57 | AI literacy and training | Out of scope | Training is the bank's; Kognita can require a training attestation before a user may invoke a higher-tier use case | Use-case register (0.4) |
| 58 | Automation-bias controls | Partial | Approval rate and time planned. Added: edit and rejection rates | RM review (0.4) |
| 59 | Outcome monitoring | Added | Complaints, corrections, wrong-client targeting, unsuitable suggestions and communication errors recorded as outcome events linked to the interaction, feeding the circuit breaker | Outcomes and near misses (0.4) |
| 60 | Business effectiveness | Boundary | Kognita exposes acceptance, edit, rejection and wrong-client rates from evidence; business KPIs belong to the bank's analytics | Flight recorder (0.4) |
| 61 | Autonomy escalation approval | Planned | Raising autonomy is a governed change | 0.5 |
| 62 | Action limits | Planned | Blast-radius limits, channels, jurisdictions | 0.5 |
| 63 | Rate limiting | Added | Limits per minute and per hour, not only per day: 2,500 messages in four minutes is a DENY | Blast-radius limits (0.5) |
| 64 | Scope expiry | Partial | Time-bound authority planned. Added: authority also ends when the task completes | Delegated authority (0.5) |
| 65 | Jurisdiction controls | Covered | Location, booking centre and zones are first-class | Core; private-banking pack (0.4) |
| 66 | Approved-channel controls | Added | Channel register: a channel may be used only if registered as meeting archiving and supervision requirements | Governed communication (0.4) |
| 67 | Public communication controls | Added | Public-content agents use approved research and claims only, with zero client context, enforced by lanes | 0.4; 0.5 |
| 68 | Client transparency | Added | AI-disclosure rules per jurisdiction and use case, driven by the content provenance label | Governed communication (0.4) |
| 69 | Inventory linked to business processes | Partial | Register planned. Added: links to business process, products, vendors and controls | Use-case register (0.4) |
| 70 | Independent assurance | Planned | Independent validation and separation of duties; external review API for second line and internal audit | 0.4; 0.7 |
