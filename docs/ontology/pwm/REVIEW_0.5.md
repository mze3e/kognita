# PWM 0.5 review and proposed changelog entries

Reviewed 18 November 2026 against `main` at `4ffd948` (Kognita software 0.8.0). Reviewed from the repository sources in `docs/ontology/pwm`. The public site (ahmedmuzammil.com) could not be reached from the review environment, so the repo copy is the version reviewed; check that the site matches it.

## What I ran

| Check | Result |
|---|---|
| `python3 tools/build.py` | Rebuilds `pwm.ttl` and `SPEC.md` with no diff, so the committed build is reproducible. |
| `python3 tools/validate.py` | Positive example conforms; evidence chain unbroken; negative fixture raises exactly the 193 expected violations; all 130 written shapes exercised. |
| `tests/test_review_regressions.py` | 113 passed (8 min 24 s). |
| `tools/check_shapes_terms.py` on shapes, positive and negative examples | Every `pwm:` term used is defined (0 missing). |

## Findings, in priority order

### 1. PWM is three Kognita milestones behind, and the numbering rule has broken

PWM says "version N follows Kognita milestone N". Kognita software is at 0.8.0, and 0.6 (claims), 0.7 (trust and resilience) and 0.8 (assurance) have all shipped. PWM stops at 0.5. The roadmap also reuses "0.8" for a business-scope release that is unrelated to Kognita 0.8 (supervisory alignment and runtime assurance), so the label now means two different things.

Concrete drift, checked against `src/kognita/vocabulary.py`:

- `EvidenceEventTypeScheme` has 17 codes. Kognita's `EventType` has 31. Fifteen Kognita codes have no PWM concept: `ANCHOR_PUBLISHED`, `ASSURANCE`, `CLAIM_VERIFIED`, `EVALUATION_RECORDED`, `EXTERNAL_REVIEW_SUBMITTED`, `INCIDENT_UPDATED`, `INCIDENT_CLOSED`, `INTERACTION`, `KEY_REVOKED`, `KEY_ROTATED`, `LEARNING_EXPORT`, `MEMORY_WRITE`, `REVALIDATION_TRIGGERED`, `REVIEW`, `SOURCE_SUPERSEDED`. PWM has `MODEL_REFUSAL`, which Kognita does not.
- `EvidenceRecord` carries `recordHash` and `previousRecordHash` only. It has no signature, key identifier or schema version (Kognita 0.7 T1/R2), and nothing for anchors, recovery bundles or key rotation.
- Nothing models the 0.7 provider register (R3/R6), dependency map and concentration report (R8/R9), failover (R5), drills (R10), or retention by record class (R11). PWM has `ConcentrationLimit` (mandate concentration) only.
- Nothing models 0.8: KYA manifests, deployment attestations, third-party update evaluation, production outcome and revalidation, approved component digests, or the supervisory theme reports.
- The pinned source is software 0.7.0 at `9303769`. The README says so, but `main` is now 0.8.0.

Recommendation: renumber so PWM names stop colliding with Kognita's. Keep "PWM N follows Kognita N" for 0.6 to 0.8, and rename the business-scope and supervisory-readiness releases (for example 0.9 and 0.10, or "B1" and "B2"). Move the Kognita 0.8 coverage in as a new PWM 0.8 module. Re-pin the source to 0.8.0.

### 2. Code-list values cannot be mapped to Kognita mechanically

- Only 87 of 235 concepts carry `skos:notation`. For example `StaleActionScheme` has labels but no notation, so `REFRESH`, `SUPPRESS`, `WARN` cannot be matched by value.
- Where notation exists, case differs: PWM `fail_closed` and `degraded` against Kognita `FAIL_CLOSED` and `DEGRADED`; `CheckResult` is lower case in Kognita.
- There is no PWM scheme for Kognita's `ActorType`, `ApprovalStatus` (including `MARKED`, the first signature of a two-signature approval), `EgressDecision` (`REDACT`) or `CheckResult`. `approvalStatus` is bound to `AssetStatusScheme`, which looks wrong for approval states.

Recommendation: give every concept a `skos:notation` that equals the Kognita enum value exactly; add the missing schemes; add a test in Kognita CI that loads `pwm.ttl`, compares every enum with its scheme, and fails on any difference. That is the roadmap's 0.9 "mapping table" goal and can start now, cheaply.

### 3. Example coverage is thin where it matters most

66 of 198 classes (counting subclass instances; 77 by exact type) have no instance in the positive example. They include core business classes: `Trust`, `Trustee`, `Settlor`, `BeneficialOwner`, `ControllingPerson`, `PoliticallyExposedPerson`, `SourceOfWealthAssessment`, `SuitabilityReport`, `DiscretionaryMandate`, `Fee`, `ThirdPartyPayment`, `Position`, `LombardFacility`, `TargetMarket`, `Complaint`, `VulnerabilityIndicator`, `Consent`, and `ModelOwner`. 37 of 408 properties are also never used. A class nobody populates is untested: the shapes cannot catch a modeling error in it. Of the 66 unpopulated classes, 21 are in the parties module.

Recommendation: add a build gate that fails when a non-meta class or property has no example, with an explicit allow-list for the exceptions. Then fill the parties, suitability and mandates modules first, which also tests the FIBO alignments.

### 4. Reasoning axioms are almost absent; every constraint lives in SHACL

The ontology has only 2 `owl:Restriction`s, no `owl:FunctionalProperty`, no inverse pairs, no disjointness and no equivalent classes. SHACL covers recorded structure well, but an OWL consumer (or a firm extending PWM) gets almost no inference or early inconsistency detection.

Recommendation, kept to what is safe in OWL 2 DL: declare single-valued properties functional (`contentHash`, `recordHash`, `previousRecordHash`, `classification` on a record, `decisionResult`); declare disjointness between Person, Organization and AIAgent (checking the parent classes first), and between sibling mandate types; add inverses where the example queries traverse both directions (`partOfInteraction`, `inRun`). Run ROBOT with HermiT before and after.

### 5. 142 of 408 properties have no `rdfs:domain`; 10 have no range

This looks deliberate (shared properties such as `classification` and `approvedBy`), and it keeps reuse flexible, but it is undocumented and gives tooling nothing for form generation or query validation. The 10 with no range (`assesses`, `supersedes`, `connectedTo`, `valuationOf`, `decisionOn`, `dependsOn`, `delegatedBy`, `citesSource`, `evidenceFor`, `findingRecord`) are the weakest spots.

Recommendation: state the policy in the design principles ("shared properties omit a domain; SHACL supplies it per class"), give the 10 ranges, and emit `schema:domainIncludes` or an annotation listing intended classes.

### 6. Citation verification is half done

Of 47 references, 24 are still "instrument-level" (cited from the instrument, not checked paragraph by paragraph). The roadmap already schedules this, but the README, SPEC and example still describe the 0.5 terms with regulatory grounds as if fully verified. Priority list is in `ROADMAP.md` (MiFID II, Delegated Regulation, PRIIPs, COBS, AI Act, FATF, MAS Fair Dealing, HK PI rules).

Recommendation: render the verification status next to each citation in SPEC, and make the build print the count so the number is visible on every release.

### 7. Documentation is cumulative prose rather than a current description

`README.md` repeats history inside current-state sections: the "briefly carried a second set of fields" story appears in README, SPEC change log, GAP_ANALYSIS and the root CHANGELOG, and the 0.3, 0.4 and 0.5 notes are appended after the status section. A new reader cannot tell what is true now from what changed.

Recommendation: keep README to "what 0.5 is"; move history to one `CHANGELOG.md` in the PWM folder; keep GAP_ANALYSIS as per-release appendices; link rather than repeat.

### 8. Smaller items

- Mixed British and American spelling in PWM-defined terms: `AuthorizedSignatory` and `InternationalOrganizationPEP` use British spelling, while the reused OMG and FIBO terms (`Authorization`, `AuthorizedIndividual`) use American. Pick one policy for PWM-defined names and note that reused external names keep their source spelling.
- Legacy `xsd:date` and `xsd:duration` ranges (`nextReviewDate`, `retentionUntil`, `freshnessThreshold`) are still open. The cheapest time to migrate is before the 1.0 IRI freeze; add it to the 0.6 scope.
- Hash checks: SHACL cannot recompute hashes (stated plainly, which is good). Kognita already has an independent verifier (`@kognita/verify`, `kognita evidence verify`). A small conformance script that exports an example chain and runs it through that verifier would turn "the chain is unbroken" from a structural claim into a cryptographic one.
- The 66 generated code-list shapes are exercised by regression tests but only 2 by the negative fixture, so the fixture count (193) understates what is tested. Say so in the README, or add one negative per scheme to the fixture.
- `GAP_ANALYSIS.md` is titled "0.5" and holds 0.4 and 0.5 sections; split per release as the file grows.

## Proposed changelog entries

### Root `CHANGELOG.md`

The root changelog currently mentions PWM only inside the 0.8.0 "Added" list (0.4 and two 0.5 review bullets). It omits PWM 0.1, 0.2, 0.3, the 0.5 release itself, and the roadmap (#166), and it mixes ontology versions into software release 0.8.0, although PWM 0.1 to 0.5 merged between the 0.7.0 and 0.8.0 releases (2 to 5 November). Suggested: add a separate `### Ontology (PWM)` heading under the release that shipped them, and put new items under `[Unreleased]`.

```markdown
## [Unreleased]

### Ontology (PWM)

- PWM review of 18 November 2026 (`docs/ontology/pwm/REVIEW_0.5.md`): PWM trails Kognita software by three milestones (0.6 claims, 0.7 trust and resilience, 0.8 assurance); 15 of 31 Kognita evidence event types have no PWM concept; 148 of 235 code-list concepts lack a `skos:notation`; 66 of 198 classes have no example data. Proposes renumbering to separate the Kognita-aligned releases from the business-scope releases, a Kognita CI check that every vocabulary enum matches its PWM scheme, and a build gate on example coverage.

### Documentation

- Add `docs/ontology/pwm/REVIEW_0.5.md`.
```

Entries for the PWM work that is already merged, to place under the release that shipped it (`[0.8.0]`, "Ontology (PWM)"), since only 0.4 and the 0.5 review fixes are there now:

```markdown
### Ontology (PWM)

- PWM 0.1 (2 Nov 2026, #160): first public draft. Ten modules, a reference register where every term cites its grounds, OWL 2 DL with SKOS code lists, SHACL shapes, a synthetic example and negative tests. FIBO first, then OMG Commons, W3C PROV-O and DPV.
- PWM 0.2 (3 Nov 2026, #162): FIBO grounding re-verified against a pinned commit (`33a92cd`) with Release or Provisional maturity recorded; EU sustainability preferences with two rules; versioned shapes IRI.
- PWM 0.3 (3 Nov 2026, #163): runs and budgets, durable approvals, gateway model and MCP calls, classifier evidence, pinned policy checks, retention and erasure, reconstruction coverage, failure modes, authenticated client identity. Retention duration is whole days (`xsd:nonNegativeInteger`).
- PWM 0.4 (3 Nov 2026, #164): ingestion, declarative policy and tests, policy packs, recorder replay, RM edits, review queues, circuit breaker and affected-client results, outcomes and near misses. Adds `AuthorityLiveStatus` (FastContext).
- PWM 0.5 (4 Nov 2026, #165): agent identity, scoped and expiring delegated authority, autonomy transitions, blast-radius limits, content trust, instruction authenticity, anomaly records, narrowing delegation chains, capability grants, fleets, sandbox output, reversibility, adversarial tests, enforcement attestations and stopping in flight. 193 domain classes, 408 properties, 130 written rules and 66 generated code-list rules.
- PWM roadmap from 0.1 to 1.0, following the Kognita milestones (#166, revised 5 Nov 2026).
```

### `docs/ontology/pwm/SPEC.md` change log and a new PWM `CHANGELOG.md`

The SPEC table is the only complete PWM history. Suggested: promote it to `docs/ontology/pwm/CHANGELOG.md` (generated into SPEC), with one row per version in the same style as the table above and a "Removed or renamed" column, as the versioning policy already requires.

## Suggested order of work

1. Fix the numbering and re-pin to Kognita 0.8.0 (documentation only).
2. Add notations and missing schemes, plus the Kognita CI enum check (small, high value).
3. Model the 0.6 to 0.8 evidence additions: signatures and keys, the 15 event types, provider register, dependency map, assurance records.
4. Add the example-coverage gate and fill the parties, suitability and mandates examples.
5. Add functional, disjoint and inverse axioms, with the ROBOT run in CI.
6. Consolidate the docs into README plus CHANGELOG.
