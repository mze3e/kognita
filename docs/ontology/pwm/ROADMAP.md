# PWM Ontology Roadmap: 0.1 to 1.0

Revised 19 November 2026. The first version of this plan was written on 2 November, before 0.2 to 0.5 were built. The 5 November revision followed what was built to 0.5. This revision follows the 0.5 review, which found PWM three Kognita milestones behind, and the 0.8 release that closed that gap.

**Where a matching Kognita milestone exists, PWM version N follows Kognita roadmap milestone N** ([docs/ROADMAP.md](../../ROADMAP.md)). Versions 0.1 to 0.5, 0.8 and 0.10 exist as drafts. PWM 0.8 was built as one release that covers Kognita milestones 0.6 to 0.8, and PWM 0.10 as one release that covers the planned 0.9 and 0.10 scope, so there are no separate PWM 0.6, 0.7 or 0.9 releases. The business items from the first plan that no milestone covered are scheduled in 0.9 and 0.10, numbered so they can no longer be mistaken for Kognita milestones, and the release candidates come after.

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
| 0.4 | 0.4 Ingestion, Policy Language and the Client Lifecycle | Ingestion, policy artifacts, review, circuit breaker, outcomes | Done, merged 5 Nov 2026 (#164) |
| 0.5 | 0.5 Agents, Authority and Fleets | Agent identity, scoped authority, autonomy, blast radius, fleets | Done, merged 5 Nov 2026 (#165) |
| 0.6 | 0.6 Claims | Folded into 0.8 (the verification, evaluation, reasoning, memory, learning and sensitive-attribute records); see below | Done in 0.8, with open items |
| 0.7 | 0.7 Trust and Resilience | Folded into 0.8 | Done in 0.8 |
| 0.8 | 0.8 Supervisory Alignment and Runtime Assurance | Trust and resilience, runtime assurance, the 0.6 gaps, Kognita vocabulary parity, US English | Draft, 18 Nov 2026 |
| 0.9 | None: business scope | Costs and charges, product governance, client circumstances, operations; verified sources | Built in 0.10, citation verification open |
| 0.10 | None: adoption | Supervisory readiness and binding to Kognita | Draft, 19 Nov 2026, with open items |
| 1.0-rc | None | External review, freeze, conformance kit | Planned |
| 1.0 | 1.0 Production Ready | Stable | Planned, before 7 Oct 2028 |

Kognita's own published targets are Q3 2027 for 0.6, Q4 2027 for 0.7 and Q1 2028 for 1.0. PWM follows when each milestone is built, not by date. Two dates anchor the end: MAS AIRG Sections 3 and 4 take effect on 7 October 2027, and Sections 5 and 6 on 7 October 2028.

---

## 0.1 to 0.5: built

0.1 set the shape: 10 modules, every term cited, OWL 2 DL (ROBOT DL profile validation passes; the legacy xsd:date and xsd:duration ranges sit outside the OWL 2 datatype map, see [README.md](README.md)), SHACL rules and tests. 0.2 grounded the main classes in FIBO against a pinned commit and added EU sustainability preferences. 0.3 to 0.5 each add a module for the Kognita milestone of the same number, with a gap analysis per roadmap item in [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

PWM 0.5 ([pwm.ttl](pwm.ttl)) uses one FastContext live-status vocabulary, `AuthorityLiveStatus`. Blast-radius limits and autonomy gates are set by the firm's risk committee, data governance committee or CRO.

What the first plan listed for 0.2 and did not get built is carried forward below, in 0.8.

---

## 0.6 "Claims" (follows Kognita 0.6): built in 0.8, with open items

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

**Built in 0.8 (18 November 2026):** the claim envelope fields and the verified-claim shape (item 1), verifier independence checked by a shape (item 7), memory governance records and shapes (item 8), learning exports and exclusions (item 9), sensitive attribute inferences with their shapes (item 10), evaluation results, metric values and fairness reports (item 11) and reasoning traces (item 13). Items 2 to 6 and 12 were already covered by the 0.1 to 0.3 terms (`Fact`, `ApprovedSource`, `HouseView`, `ClaimTypeScheme`, `GovernedDefinition`). **Still open:** house-view change propagation as a queryable relation (item 4), the evaluation gate thresholds per use case (item 11), and the claim envelope round trip through Kognita's own export, which waits for the 0.10 JSON-LD work.

**Exit (met except the round trip):** verifier independence is checked by a shape; every new term has a citation, an example and a negative test.

## 0.7 "Trust and Resilience" (follows Kognita 0.7): built in 0.8

**Goal:** model the evidence and operating records that make governance itself auditable and survivable.

- **Signed evidence.** Signature, key identity and rotation, revocation and escrow records; evidence schema version; external anchoring of the chain head; signed external review submissions.
- **Model-provider register.** Each provider as an ICT third party: dependent use cases, data residency and retention, training use, sub-processors, incident notification terms, incident history and regulatory cooperation.
- **Failover and model change.** Provider failover as a governed decision allowed only to approved models; silent model-change detection as a revalidation trigger.
- **Dependency map and concentration.** Business service to agent, model, cloud, vector store, tool and data source, with declared against observed dependencies; concentration by provider and platform layer.
- **Failure drills and retention by record class.** The eight failure kinds as drill records; retention periods set per record class so an AI record never outlives the regulatory record it supports.
- **Incident evidence and reports.** Chain breaks, outages, degraded periods and provider failures with timelines, and the structured incident report: affected clients, controls that ran, failed or missing, and the runtime change behind it.

**Exit (met in 0.8):** a restore drill, a key rotation and an incident report can each be described and checked in PWM data; the provider register and dependency map are covered by shapes. All 31 Kognita evidence event types are PWM codes.

## 0.8 Supervisory alignment and runtime assurance (follows Kognita 0.8): built

**Built on 18 November 2026:** the assurance policy, agent manifest, external-agent assertions, deployment attestations, third-party update approvals, approved components and advisories, technology transfer classification, production outcomes and supervisory conformance reports, with shapes, example data and negative tests. The Kognita 0.8 AI cyber drill scenarios are covered by the existing `AdversarialTestCase` and the new `FailureDrill`; per-scenario codes for the eight hostile-agent drills are not yet modeled.

## 0.9 Business scope and verified sources: built in 0.10, with open items

**Built on 19 November 2026 (module 13):** costs and charges, inducement disclosure, product approval, target market review and negative target market, complaint outcomes, periodic suitability review, structured vulnerability drivers, orders, executions, allocations, best-execution records, corporate actions, income, CRS and FATCA reporting records, lending facilities (Lombard as one kind) and performance figures as provenance. Each has terms, shapes, example data and negative tests. **Still open:** citation verification (24 of 47 references remain instrument-level; the new regulatory citations are instrument-level too), FIBO alignment of the operations terms (needs verified FIBO IRIs and was not attempted), and a ready-made distribution-strategy model beyond a reference.

**Goal:** finish the business model the first plan promised, and make every citation one a reviewer can trust. This release is not tied to a Kognita milestone.

- **Citation verification.** At 0.8, 24 of the 47 references are still cited at instrument level, and SPEC marks each such citation with a dagger. Re-check each against the official text and record the paragraph. Priority: MiFID II, the Delegated Regulation, PRIIPs, FCA COBS, the EU AI Act, FATF, the MAS Fair Dealing Guidelines and the HK Professional Investor rules. Verify the ISO and W3C standard references as well.
- **Costs and charges.** Ex-ante and ex-post cost disclosure, and the inducement disclosure linked to `ThirdPartyPayment` (MiFID II Art. 24(4) and (9); PRIIPs).
- **Product governance.** Negative target market, distribution strategy, target market review and the product approval process, building on the existing `TargetMarket`.
- **Client circumstances.** Vulnerability indicators as structured fields building on `VulnerabilityIndicator`, complaint outcomes, and periodic suitability review.
- **Operations and lifecycle.** Orders, executions and allocations with best-execution evidence, corporate actions and income, CRS and FATCA reporting, lending beyond Lombard, and performance figures as provenance from registered calculation services, never computed in the ontology. Reuse FIBO's securities and transaction terms.
- **Examples.** Done in 0.8: every class and property appears in the positive example (three properties are on a reasoned allow-list) and every written shape has a negative test, enforced by tests/test_example_coverage.py. For new 0.9 terms the same tests apply.

**Exit:** zero instrument-level references; each new area has terms, shapes, an example and FIBO alignment.

## 0.10 Supervisory readiness and binding to Kognita: built, with open items

**Built on 19 November 2026:** `AIRG_MAP.md` (every paragraph row of the MAS table in control-frameworks.md mapped to PWM terms or to boundary), `profiles/ai-inventory-profile.ttl`, `queries/` (the ten reconstruction questions, an inventory report and a superseded-source query), `mapping/kognita-pwm-map.json` (114 vocabulary values and 25 policies), `context/pwm-evidence.jsonld` with `tools/roundtrip.py` and a signed Kognita export read back as PWM RDF, five jurisdiction profiles, and a worked code-list profile, each with tests. **Still open:** the round trip covers evidence exports, not register exports; the sample export is committed, not regenerated in CI; the AIRG map is only as complete as the table in control-frameworks.md and its citations are instrument-level; the jurisdiction profiles carry investor categories only, with no disclosure triggers or cross-border rules, which stay with the firm; and the exit condition that Kognita's CI produces PWM data from live code is not yet wired.

**Goal:** produce from PWM data what MAS expects under Sections 3 and 4, and make PWM the data contract Kognita actually uses.

- **AIRG paragraph map.** Every paragraph of Sections 2 to 6 mapped to PWM terms and shapes, or marked out of scope, in a table published with the spec.
- **Inventory profile.** A SHACL profile that checks a full AI inventory against the 4.7 attributes and footnote 25, and a query that produces the senior-management report: material AI by classification, owner, status and open findings, filterable by jurisdiction.
- **Reconstruction queries.** A published SPARQL query set for each question in the reconstruction test: who decided, under which rule and delegation, with which data and model version, and what the client received.
- **Mapping table.** The enum check in tests/test_kognita_vocabulary.py already fails when a Kognita vocabulary value has no PWM code (done in 0.8). Still to do: a machine-readable table from each value and each private-banking policy pack to its PWM IRI, and running the check in Kognita's own CI.
- **JSON-LD context and round trip.** A context so that Kognita's evidence and register exports read as PWM RDF; Kognita exports a register, an interaction and its evidence chain, and the shapes and queries run in Kognita's CI.
- **Jurisdiction profiles.** Data packages, not ontology changes, for SG, HK, CH, the EU and the UK: investor categories, disclosure triggers and cross-border defaults, each with citations.
- **Code-list profiles.** A profile file that rebinds code lists and switches off generated shapes, with a worked example.

**Exit:** every MAS Section 2 to 6 paragraph mapped; the inventory profile and report query run on Kognita's exported register; the reconstruction queries, mapping table, JSON-LD context and round trip, jurisdiction profiles, and code-list profiles are complete and checked; Kognita CI produces PWM data that conforms.

## 1.0-rc "Release candidates"

- **Practitioner review.** Two or more reviewers from wealth management operations, compliance and data architecture, each with written findings and dispositions.
- **Ontology review.** A review by an ontology engineer familiar with FIBO, covering alignment choices, `rdfs:subClassOf` versus SKOS mappings, and reasoning behavior.
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
- **From 1.0**, PWM follows semantic versioning. Minor versions only add validation-compatible terms, code-list values and shapes. Any change that rejects data conforming to the prior version requires a major version or a separately versioned profile. Major versions may remove terms, after at least one minor version in which they are deprecated. Patch versions change definitions, citations and examples only.
- Each version publishes its own version IRI and files, and older versions stay online.
- Generated code-list shapes follow the code lists. Adding a code value is a minor change; removing one is a major change.

## Standing workstreams

These run through every release:

- **Kognita coupling.** Each PWM release with a matching Kognita milestone follows that milestone; 0.9, 0.10 and 1.0-rc have no matching milestone. When Kognita adds a vocabulary value or policy pack, the PWM mapping and shapes follow in the next release.
- **Review by breaking things.** Before a release, break the example one fact at a time and check that a rule catches it. The 0.5 review found three rules that let bad data through this way.
- **Regulatory watch.** Track MAS consultations, including the planned 2027 guidance on agentic AI, the EU AI Act implementing acts, and FCA and SFC updates. Each change becomes a citation update or a new term.
- **FIBO tracking.** Re-run the integrity check against each FIBO quarterly release, and adopt Provisional terms as they reach Release.
- **Institution extension.** Keep a private extension for a real firm alongside each release, to test that the generic core needs no firm-specific edits. Confidential content never enters this repository.

## Risks

| Risk | Mitigation |
|---|---|
| The ontology drifts from Kognita and becomes a second description of the same system | Follow Kognita milestone by milestone now; the enum check in tests/test_kognita_vocabulary.py already makes vocabulary drift a failing test, and the 0.10 round trip extends it to records |
| Over-modeling: terms nobody populates | Every new term needs example data and a consumer, either a shape, a query or a Kognita mapping. none at 0.8 |
| Rules that look complete but accept bad data | Mutation checks before each release; every written rule exercised by a negative fixture |
| Regulatory citations go stale | Register-level verification status, re-checked each release; the regulatory watch workstream |
| Jurisdiction rules leak into the core | Thresholds and rule tables live in jurisdiction profiles as data, never as ontology axioms |
| Single-author blind spots | External review and a second implementation before the freeze |
| The namespace host changes | IRIs sit on a domain the author controls; a redirect plan is documented before 1.0 |

## Decision log

- **2026-11-02.** Dates anchored to the MAS AIRG effective dates rather than to the Kognita release calendar, which is running ahead of its published timeline. 1.0 targeted before Sections 5 and 6 apply on 7 October 2028.
- **2026-11-02.** Jurisdiction thresholds and rule tables kept as data profiles, not ontology terms, so the core stays generic.
- **2026-11-05.** Renumbered to follow the Kognita milestones as built: PWM 0.3 to 0.5 map to Kognita 0.3 to 0.5, and 0.6 and 0.7 will map to Kognita 0.6 and 0.7. The business items the first plan put in 0.2 to 0.6 and no Kognita milestone covers moved to 0.8 and 0.9.
- **2026-11-18.** Review of PWM 0.5 found it three Kognita milestones behind and the 0.8 label used twice. Built PWM 0.8 as one release covering Kognita 0.6 to 0.8, renumbered the unrelated business-scope and supervisory-readiness releases to 0.9 and 0.10, adopted US English for all PWM-defined names, and made vocabulary parity, example coverage and axioms failing tests. Renamed terms keep deprecated aliases for one version.
- **2026-11-19.** Built the planned 0.9 and 0.10 scope as PWM 0.10, in one release, because the 0.10 artifacts depend on the 0.9 terms. Migrated the last non-OWL-2 datatypes. Removed the 0.8 deprecated aliases as promised. Exit conditions that need official texts, ROBOT or reviewers are listed as open rather than claimed.
