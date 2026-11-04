# PWM Ontology Roadmap: 0.1 to 1.0

Revised 5 November 2026. The first version of this plan was written on 2 November, before 0.2 to 0.5 were built. This version follows what was actually built.

**PWM version N follows Kognita roadmap milestone N** ([docs/ROADMAP.md](../../ROADMAP.md)). Versions 0.1 to 0.5 exist as drafts. The business items from the first plan that no milestone covered are scheduled in 0.8 and 0.9, and the release candidates come after.

## Where 1.0 should land

**PWM 1.0 is a stable, citable, FIBO-aligned vocabulary that a wealth manager can adopt as the data contract for governed AI, and that a supervisor can use to reconstruct a client interaction.**

Four tests define it:

1. **Coverage.** Every control in Kognita's [control-frameworks.md](../../control-frameworks.md) and every paragraph of the MAS Guidelines on AI Risk Management maps to PWM terms and rules, or is marked boundary or out of scope with a reason.
2. **Reconstruction.** Every question in Kognita's reconstruction test ([Supervisory Examinability](../../ROADMAP.md#supervisory-examinability)) can be answered by a published query over PWM data.
3. **Adoption.** At least one institution extension and Kognita itself use PWM. Their data passes the shapes and their queries run unchanged.
4. **Stability.** IRIs are frozen. Changes follow a published deprecation policy, and every citation is verified against its source.

## Timeline

| Version | Follows Kognita | Theme | Status |
|---|---|---|---|
| 0.1 | Before the roadmap | First public draft | Done, 2 Nov 2026 |
| 0.2 | Releases 0.1 and 0.2 | Verified FIBO grounding and sustainability preferences | Done, merged 3 Nov 2026 (#162) |
| 0.3 | 0.3 Gateways, the Run and Replay | Runs, budgets, approvals, gateway calls, pinned evidence, reconstruction | Done, merged 3 Nov 2026 (#163) |
| 0.4 | 0.4 Ingestion, Policy Language and the Client Lifecycle | Ingestion, policy artefacts, review, circuit breaker, outcomes | Draft, in review (#164) |
| 0.5 | 0.5 Agents, Authority and Fleets | Agent identity, scoped authority, autonomy, blast radius, fleets | Draft, in review (#165) |
| 0.6 | 0.6 Claims | Claim envelope, facts, verification, memory, evaluation, definitions | Planned |
| 0.7 | 0.7 Trust and Resilience | Signed evidence, provider register, dependencies, drills, incidents | Planned |
| 0.8 | None: business scope | Verified sources, costs and charges, product governance, operations | Planned |
| 0.9 | None: adoption | Supervisory readiness and binding to Kognita | Planned |
| 1.0-rc | None | External review, freeze, conformance kit | Planned |
| 1.0 | 1.0 Production Ready | Stable | Planned, before 7 Oct 2028 |

Kognita's own published targets are Q3 2027 for 0.6, Q4 2027 for 0.7 and Q1 2028 for 1.0. PWM follows when each milestone is built, not by date. Two dates anchor the end: MAS AIRG Sections 3 and 4 take effect on 7 October 2027, and Sections 5 and 6 on 7 October 2028.

---

## 0.1 to 0.5: built

0.1 set the shape: 10 modules, every term cited, OWL 2 DL, SHACL rules and tests. 0.2 grounded the main classes in FIBO against a pinned commit and added EU sustainability preferences. 0.3 to 0.5 each add a module for the Kognita milestone of the same number, with a gap analysis per roadmap item in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

What the first plan listed for 0.2 and did not get built is carried forward below, in 0.8.

---

## 0.6 "Claims" (follows Kognita 0.6)

**Goal:** model what an agent says, what it remembers and how it was tested, at the depth Kognita 0.6 enforces.

PWM already has a basic claim record from 0.1. The first step is a gap analysis of Kognita 0.6 items 1 to 13 against it. Then:

- **Claim provenance envelope.** Source record and timestamp, retrieval time, calculation method, prompt version, confidence and policy status.
- **Fact contract and freshness,** approved-source grounding, and house-view change propagation, so that invalidated recommendations can be queried.
- **Claim types and calculation provenance.** Fact, inference and recommendation; figures from registered calculation services.
- **Verification gate,** including verifier independence (a different model, version or provider), checked by a shape.
- **Memory governance.** Working, relationship and institutional memory; approval of what enters institutional memory, expiry and erasure.
- **Feedback loops and sensitive attributes.**
- **Evaluation evidence.** Runs, datasets, metrics, fairness rates and the evaluation gate, linked to model cards and promotion gates.
- **Governed business definitions.** Owner, version and effective date for terms such as AUM, net new money and client segment, with claims that cite them.
- **Reasoning traces.** Record that a trace was kept and where, without its content.

**Exit:** the Kognita 0.6 claim envelope and memory records round-trip through PWM; verifier independence is checked by a shape; every new term has a citation, an example and a negative test.

## 0.7 "Trust and Resilience" (follows Kognita 0.7)

**Goal:** model the evidence and operating records that make governance itself auditable and survivable.

- **Signed evidence.** Signature, key identity and rotation, revocation and escrow records; evidence schema version; external anchoring of the chain head; signed external review submissions.
- **Model-provider register.** Each provider as an ICT third party: dependent use cases, data residency and retention, training use, sub-processors, incident notification terms, incident history and regulatory cooperation.
- **Failover and model change.** Provider failover as a governed decision allowed only to approved models; silent model-change detection as a revalidation trigger.
- **Dependency map and concentration.** Business service to agent, model, cloud, vector store, tool and data source, with declared against observed dependencies; concentration by provider and platform layer.
- **Failure drills and retention by record class.** The eight failure kinds as drill records; retention periods set per record class so an AI record never outlives the regulatory record it supports.
- **Incident evidence and reports.** Chain breaks, outages, degraded periods and provider failures with timelines, and the structured incident report: affected clients, controls that ran, failed or missing, and the runtime change behind it.

**Exit:** a restore drill, a key rotation and an incident report can each be described and checked in PWM data; the provider register and dependency map are covered by shapes.

## 0.8 Business scope and verified sources

**Goal:** finish the business model the first plan promised, and make every citation one a reviewer can trust. This release is not tied to a Kognita milestone.

- **Citation verification.** At 0.5, 24 of the 47 references are cited at instrument level. Re-check each against the official text and record the paragraph. Priority: MiFID II, the Delegated Regulation, PRIIPs, FCA COBS, the EU AI Act, FATF, the MAS Fair Dealing Guidelines and the HK Professional Investor rules. Verify the ISO and W3C standard references as well.
- **Costs and charges.** Ex-ante and ex-post cost disclosure, and the inducement disclosure linked to `ThirdPartyPayment` (MiFID II Art. 24(4) and (9); PRIIPs).
- **Product governance.** Negative target market, distribution strategy, target market review and the product approval process, building on the existing `TargetMarket`.
- **Client circumstances.** Vulnerability indicators as structured fields building on `VulnerabilityIndicator`, complaint outcomes, and periodic suitability review.
- **Operations and lifecycle.** Orders, executions and allocations with best-execution evidence, corporate actions and income, CRS and FATCA reporting, lending beyond Lombard, and performance figures as provenance from registered calculation services, never computed in the ontology. Reuse FIBO's securities and transaction terms.
- **Examples.** At 0.5, 77 of 198 classes never appear in the positive example. Give every module example data, and every written shape at least one negative test.

**Exit:** zero instrument-level references; each new area has terms, shapes, an example and FIBO alignment.

## 0.9 Supervisory readiness and binding to Kognita

**Goal:** produce from PWM data what MAS expects under Sections 3 and 4, and make PWM the data contract Kognita actually uses.

- **AIRG paragraph map.** Every paragraph of Sections 3 to 6 mapped to PWM terms and shapes, or marked out of scope, in a table published with the spec.
- **Inventory profile.** A SHACL profile that checks a full AI inventory against the 4.7 attributes and footnote 25, and a query that produces the senior-management report: material AI by classification, owner, status and open findings, filterable by jurisdiction.
- **Reconstruction queries.** A published SPARQL query set for each question in the reconstruction test: who decided, under which rule and delegation, with which data and model version, and what the client received.
- **Mapping table.** A machine-readable mapping from every Kognita vocabulary value and every private-banking policy pack to its PWM IRI. A Kognita test fails when a value has no mapping.
- **JSON-LD context and round trip.** A context so that Kognita's evidence and register exports read as PWM RDF; Kognita exports a register, an interaction and its evidence chain, and the shapes and queries run in Kognita's CI.
- **Jurisdiction profiles.** Data packages, not ontology changes, for SG, HK, CH, the EU and the UK: investor categories, disclosure triggers and cross-border defaults, each with citations.
- **Code-list profiles.** A profile file that rebinds code lists and switches off generated shapes, with a worked example.

**Exit:** every MAS Section 3 to 6 paragraph mapped; the inventory profile and report query run on Kognita's exported register; Kognita CI produces PWM data that conforms.

## 1.0-rc "Release candidates"

- **Practitioner review.** Two or more reviewers from wealth management operations, compliance and data architecture, each with written findings and dispositions.
- **Ontology review.** A review by an ontology engineer familiar with FIBO, covering alignment choices, `rdfs:subClassOf` versus SKOS mappings, and reasoning behaviour.
- **FIBO proposals.** Submit the practitioner-defined (NEW) terms that would belong in FIBO to the EDM Council, starting with mandate types, suitability assessment, investor classification and coverage assignment. Record each outcome.
- **DPV alignment.** Confirm the mappings to DPV for consent, purpose, personal data and AI lifecycle stages with the DPV community.
- **Second implementation.** At least one institution extension, which can be private, built only from the public spec. Every gap it hits is logged as an issue.
- **Freeze.** Freeze IRIs and labels. From then only definitions, citations and examples change.
- **Migration notes** from every earlier version, with `owl:deprecated` and `dcterms:isReplacedBy` on anything renamed.
- **Conformance kit.** Shapes, test data and queries packaged so that an implementer can show their data conforms, plus a glossary and a short adoption guide.

Release candidates are named `1.0.0-rc.1`, `1.0.0-rc.2` and so on. **Exit:** four weeks with no new major findings.

## 1.0 "Stable" (before 7 October 2028)

- The four tests at the top of this document are met and the evidence is published with the release.
- Versioned IRIs are permanent; `https://ahmedmuzammil.com/ontology/pwm/1.0` resolves.
- A governance note says who can propose changes, how they are reviewed and how deprecations work.

---

## Versioning policy

- **Before 1.0**, a minor version can rename or remove terms. Each removal is listed in the change log with its replacement, and the old IRI stays declared with `owl:deprecated true` for one version.
- **From 1.0**, PWM follows semantic versioning. Minor versions only add terms, code-list values and shapes. Major versions may remove terms, after at least one minor version in which they are deprecated. Patch versions change definitions, citations and examples only.
- Each version publishes its own version IRI and files, and older versions stay online.
- Generated code-list shapes follow the code lists. Adding a code value is a minor change; removing one is a major change.

## Standing workstreams

These run through every release:

- **Kognita coupling.** Each PWM release follows the Kognita milestone of the same number. When Kognita adds a vocabulary value or policy pack, the PWM mapping and shapes follow in the next release.
- **Review by breaking things.** Before a release, break the example one fact at a time and check that a rule catches it. The 0.5 review found three rules that let bad data through this way.
- **Regulatory watch.** Track MAS consultations, including the planned 2027 guidance on agentic AI, the EU AI Act implementing acts, and FCA and SFC updates. Each change becomes a citation update or a new term.
- **FIBO tracking.** Re-run the integrity check against each FIBO quarterly release, and adopt Provisional terms as they reach Release.
- **Institution extension.** Keep a private extension for a real firm alongside each release, to test that the generic core needs no firm-specific edits. Confidential content never enters this repository.

## Risks

| Risk | Mitigation |
|---|---|
| The ontology drifts from Kognita and becomes a second description of the same system | Follow Kognita milestone by milestone now; the 0.9 mapping table and CI round trip make drift a failing test |
| Over-modelling: terms nobody populates | Every new term needs example data and a consumer, either a shape, a query or a Kognita mapping. 77 classes have no example data at 0.5 |
| Rules that look complete but accept bad data | Mutation checks before each release; every written rule exercised by a negative fixture |
| Regulatory citations go stale | Register-level verification status, re-checked each release; the regulatory watch workstream |
| Jurisdiction rules leak into the core | Thresholds and rule tables live in jurisdiction profiles as data, never as ontology axioms |
| Single-author blind spots | External review and a second implementation before the freeze |
| The namespace host changes | IRIs sit on a domain the author controls; a redirect plan is documented before 1.0 |

## Decision log

- **2026-11-02.** Dates anchored to the MAS AIRG effective dates rather than to the Kognita release calendar, which is running ahead of its published timeline. 1.0 targeted before Sections 5 and 6 apply on 7 October 2028.
- **2026-11-02.** Jurisdiction thresholds and rule tables kept as data profiles, not ontology terms, so the core stays generic.
- **2026-11-05.** Renumbered to follow the Kognita milestones as built: PWM 0.3 to 0.5 map to Kognita 0.3 to 0.5, and 0.6 and 0.7 will map to Kognita 0.6 and 0.7. The business items the first plan put in 0.2 to 0.6 and no Kognita milestone covers moved to 0.8 and 0.9.
