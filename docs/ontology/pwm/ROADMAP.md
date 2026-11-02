# PWM Ontology Roadmap: 0.1 to 1.0

PWM 0.1 is a first public draft. It has the right shape: 10 modules, every term cited, OWL 2 DL, SHACL rules and tests. It has not yet been used on real data, reviewed outside its author, or tied to Kognita in code. The road to 1.0 is about earning stability: covering the gaps, proving the model on a real institution, binding it to Kognita, and freezing the IRIs that others will import.

## Where 1.0 should land

**PWM 1.0 is a stable, citable, FIBO-aligned vocabulary that a wealth manager can adopt as the data contract for governed AI, and that a supervisor can use to reconstruct a client interaction.**

Four tests define it:

1. **Coverage.** Every control in Kognita's [control-frameworks.md](../../control-frameworks.md) and every paragraph of the MAS Guidelines on AI Risk Management maps to PWM terms and rules, or is marked boundary or out of scope with a reason.
2. **Reconstruction.** Every question in Kognita's reconstruction test ([Supervisory Examinability](../../ROADMAP.md#supervisory-examinability)) can be answered by a published query over PWM data.
3. **Adoption.** At least one institution extension and Kognita itself use PWM. Their data passes the shapes and their queries run unchanged.
4. **Stability.** IRIs are frozen. Changes follow a published deprecation policy, and every citation is verified against its source.

## Timeline

Dates are anchored to the MAS AIRG: Sections 3 and 4 take effect on 7 October 2027, and Sections 5 and 6 on 7 October 2028. Kognita capability numbers are cross-referenced by item, not by date, because the Kognita release line is running ahead of its published timeline.

```
Q1 2027      Q2 2027      Q3 2027      Q4 2027      Q1 2028      Q2 2028      Q3 2028
│            │            │            │            │            │            │
├─ 0.2 ──────┼─ 0.3 ──────┼─ 0.4 ──────┼─ 0.5 ──────┼─ 0.6 ──────┼─ 0.7 ── 0.9 ┼─► 1.0
  Verified &   Bound to     Claims,      Supervisory  Operations   External     Stable
  suitability  Kognita      memory &     readiness    & lifecycle  review &     (before
  complete                  evaluation   (MAS 3-4)                 freeze       7 Oct 2028)
```

| Version | Target | Theme | Gate it must pass |
|---|---|---|---|
| 0.1 | Nov 2026 | First public draft | Done: OWL 2 DL, HermiT consistent, all shapes tested |
| 0.2 | Q1 2027 | Verified and suitability-complete | No instrument-level citations left; sustainability preferences modelled |
| 0.3 | Q2 2027 | Bound to Kognita | Kognita emits and validates PWM data in its CI |
| 0.4 | Q3 2027 | Claims, memory and evaluation | Kognita 0.6 claim envelope round-trips through PWM |
| 0.5 | Q4 2027 | Supervisory readiness | MAS AIRG Sections 3 and 4 fully mapped before 7 Oct 2027 |
| 0.6 | Q1 2028 | Operations and lifecycle | Trade lifecycle, tax reporting and AIRG Sections 5 and 6 modelled |
| 0.7 | Q2 2028 | External review | Independent review closed; FIBO proposals submitted |
| 0.9 | Q3 2028 | Release candidate | IRI freeze; conformance kit; no open major findings |
| 1.0 | by Oct 2028 | Stable | The four tests above |

---

## 0.2 "Verified and suitability-complete" (Q1 2027)

**Goal:** remove the known gaps in 0.1, and make every citation one a reviewer can trust.

- **Citation verification.** Re-check every reference marked instrument-level against the official text and record the paragraph. Priority: MAS FAA-N16, SFA 04-N12, Notice 626, the SFC Code of Conduct, FCA COBS and FinSA. Also fix the broken link in `control-frameworks.md` and cite the right file.
- **Sustainability preferences.** Model the client's sustainability preferences and the product's sustainability characteristics, with the suitability check between them (MiFID II Delegated Regulation 2017/565 as amended by 2021/1253; FCA and FINMA equivalents where they exist).
- **Costs and charges.** Add the ex-ante and ex-post cost disclosure, and the inducement disclosure linked to `ThirdPartyPayment` (MiFID II Art. 24(4) and (9); PRIIPs).
- **Product governance.** Fill in the target market (negative target market, distribution strategy, review) and the product approval process.
- **Client circumstances.** Model vulnerability indicators as structured fields; complaints and their outcomes; and periodic suitability review.
- **Investor categories.** Add Hong Kong, Swiss and UAE details where they are thin, and record thresholds as data in jurisdiction profiles, never in the ontology.
- **One example per module.** Each module gets synthetic example data, and each written shape gets at least one negative test.

**Exit:** zero instrument-level references; sustainability and cost terms covered by shapes and tests; DL and reasoner checks pass.

## 0.3 "Bound to Kognita" (Q2 2027)

**Goal:** make PWM the data contract Kognita actually uses, not a parallel description of it.

- **Mapping table.** Publish a machine-readable mapping from every Kognita vocabulary value (outcomes, data classes, autonomy levels, communication levels, claim types, validator statuses, event types, owner roles, risk tiers) and every private-banking policy pack to its PWM IRI. A Kognita test fails when a vocabulary value has no PWM mapping.
- **JSON-LD context.** Publish a context so that Kognita's evidence and register exports read as PWM RDF without hand conversion.
- **Agents and authority (Kognita 0.5).** Cover blast-radius limits, capability grants, agent-to-agent delegation, granular revocation, kill-switch drills, sandbox constraints and reversibility as terms and shapes.
- **Round trip in CI.** Kognita exports a use-case register, an interaction and its evidence chain as PWM. The shapes validate them, and the reconstruction queries run against them in Kognita's CI.
- **Code-list profiles.** Treat firm code lists as a first-class pattern: a profile file that rebinds code lists and switches off generated shapes, with a worked example.

**Exit:** Kognita CI produces PWM data that conforms; every Kognita vocabulary value is mapped.

## 0.4 "Claims, memory and evaluation" (Q3 2027)

**Goal:** model what an agent says, what it remembers and how it was tested, at the depth Kognita 0.6 enforces.

- **Claim provenance envelope.** Every field of Kognita's claim record: source record and timestamp, retrieval time, calculation method, prompt version, confidence and policy status.
- **Verification.** Model verifier independence (a different model, version or provider) and house-view change propagation, so that invalidated recommendations can be queried.
- **Memory governance.** Model working, relationship and institutional memory: approval of what enters institutional memory, expiry, and erasure.
- **Evaluation evidence.** Model evaluation runs, datasets, metrics, fairness rates and the evaluation gate, linked to model cards and to promotion gates.
- **Governed definitions.** Model the business glossary: owner, version and effective date for terms such as AUM, net new money and client segment, with claims that cite them.
- **Reasoning traces.** Record that a trace was kept and where, without its content.

**Exit:** the Kognita 0.6 claim envelope and memory records round-trip through PWM; verifier independence is checked by a shape.

## 0.5 "Supervisory readiness" (Q4 2027)

**Goal:** be able to produce, from PWM data, what MAS expects under Sections 3 and 4 from 7 October 2027.

- **AIRG paragraph map.** Every paragraph of Sections 3 and 4 is mapped to PWM terms and shapes, or marked out of scope, in a table published with the spec.
- **Inventory profile.** A SHACL profile that checks a full AI inventory against the 4.7 attributes and footnote 25, plus a query that produces the senior-management report: material AI by classification, owner, status and open findings, filterable by jurisdiction.
- **Reconstruction queries.** A published SPARQL query set for each question in the reconstruction test: who decided, under which rule and delegation, with which data and model version, and what the client received.
- **Jurisdiction profiles.** Data packages, not ontology changes, for SG, HK, CH, the EU and the UK: investor categories, disclosure triggers and cross-border defaults, each with citations.
- **Third-party AI.** Model provider concentration and the single-provider dependency measure.

**Exit:** every Section 3 and 4 paragraph mapped; the inventory profile and report query run on Kognita's exported register.

## 0.6 "Operations and lifecycle" (Q1 2028)

**Goal:** close the business scope that 0.1 left out, and cover AIRG Sections 5 and 6 ahead of October 2028.

- **Orders and execution.** Orders, executions and allocations, best-execution evidence and trade confirmations, reusing FIBO's securities and transaction terms.
- **Corporate actions and income** affecting holdings.
- **Tax reporting.** CRS and FATCA reporting records, built on the existing tax residency self-certification.
- **Lending.** Credit facilities beyond Lombard, including guarantees and margin calls.
- **Performance provenance.** Return series and attribution as figures from registered calculation services, never computed in the ontology.
- **AIRG Sections 5 and 6.** Model monitoring thresholds, drift findings, change significance and re-approval, dynamic updates, retirement steps, and incident and remediation records.

**Exit:** each new area has terms, shapes, an example and FIBO alignment; Sections 5 and 6 are mapped.

## 0.7 "External review" (Q2 2028)

**Goal:** have people other than the author find what is wrong.

- **Practitioner review.** Two or more reviewers from wealth management operations, compliance and data architecture, each with written findings and dispositions.
- **Ontology review.** A review by an ontology engineer familiar with FIBO, covering alignment choices, `rdfs:subClassOf` versus SKOS mappings, and reasoning behaviour.
- **FIBO proposals.** Submit the practitioner-defined (NEW) terms that would belong in FIBO to the EDM Council, starting with mandate types, suitability assessment, investor classification and coverage assignment. Record each outcome.
- **DPV alignment.** Confirm the mappings to DPV for consent, purpose, personal data and AI lifecycle stages with the DPV community.
- **Second implementation.** At least one institution extension, which can be private, built only from the public spec. Every gap it hits is logged as an issue.

**Exit:** every review finding is closed or deferred with a reason; the institution extension validates without changes to PWM.

## 0.9 "Release candidate" (Q3 2028)

- **Freeze.** Freeze IRIs and labels. From here only definitions, citations and examples change.
- **Migration notes** from every earlier version, with `owl:deprecated` and `dcterms:isReplacedBy` on anything renamed.
- **Conformance kit.** Shapes, test data and queries packaged so that an implementer can show their data conforms.
- **Docs.** The spec rewritten for implementers, a glossary, and a "PWM in a week" adoption guide.

**Exit:** four weeks with no new major findings.

## 1.0 "Stable" (by 7 October 2028)

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

- **Kognita coupling.** When Kognita adds a vocabulary value or policy pack, the PWM mapping and shapes follow in the next PWM release.
- **Regulatory watch.** Track MAS consultations, including the planned 2027 guidance on agentic AI, the EU AI Act implementing acts, and FCA and SFC updates. Each change becomes a citation update or a new term.
- **FIBO tracking.** Re-run the integrity check against each FIBO quarterly release, and adopt Provisional terms as they reach Release.
- **Institution extension.** Keep a private extension for a real firm alongside each release, to test that the generic core needs no firm-specific edits. Confidential content never enters this repository.

## Risks

| Risk | Mitigation |
|---|---|
| The ontology drifts from Kognita and becomes a second description of the same system | The 0.3 mapping table and CI round trip make drift a failing test |
| Over-modelling: terms nobody populates | Every new term needs example data and a consumer, either a shape, a query or a Kognita mapping |
| Regulatory citations go stale | Register-level verification status, re-checked each release; the regulatory watch workstream |
| Jurisdiction rules leak into the core | Thresholds and rule tables live in jurisdiction profiles as data, never as ontology axioms |
| Single-author blind spots | 0.7 external review and a second implementation before the freeze |
| The namespace host changes | IRIs sit on a domain the author controls; a redirect plan is documented before 1.0 |

## Decision log

- **2026-11-02.** Dates anchored to the MAS AIRG effective dates rather than to the Kognita release calendar, which is running ahead of its published timeline. 1.0 targeted before Sections 5 and 6 apply on 7 October 2028.
- **2026-11-02.** Jurisdiction thresholds and rule tables kept as data profiles, not ontology terms, so the core stays generic.
