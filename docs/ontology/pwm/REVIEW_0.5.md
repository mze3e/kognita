# Review of PWM 0.5 and what 0.8 did about it

Reviewed 18 November 2026 against `main` at `4ffd948` (Kognita software 0.8.0), from the repository sources in `docs/ontology/pwm`. The public site (ahmedmuzammil.com) could not be reached from the review environment, so the repository copy is the version reviewed; check that the published site matches it.

**What was run on 0.5:** the build reproduced `pwm.ttl` and `SPEC.md` with no diff; `tools/validate.py` passed (positive example conforms, evidence chain unbroken, negative fixture raised exactly the 193 expected violations); all 113 regression tests passed. The ontology was sound. The findings were about coverage, drift and documentation.

**Corrections to the first write-up.** Two statements in the first version of this review were wrong and are corrected here. `approvalStatus` is bound to `AssetStatusScheme` correctly: it records the status of a governed information asset, not of an approval request. The real gap was that no code list existed for Kognita's `ApprovalStatus`. And the spelling finding named `Organisation`, which is not a PWM term; the PWM-defined British spellings were `AuthorisedSignatory` and `InternationalOrganisationPEP`, among others found by a full scan.

## Findings and resolutions

| # | Finding | Status in 0.8 |
|---|---|---|
| 1 | PWM trailed Kognita software by three milestones, and the label 0.8 meant two things | **Done.** Module 12 adds milestones 0.7 and 0.8 and the 0.6 records that were missing. Roadmap renumbered: business scope is 0.9, supervisory readiness 0.10. Source re-pinned to 0.8.0 |
| 1a | 15 of 31 Kognita evidence event types had no PWM concept | **Done.** All 31 present; `MODEL_REFUSAL` is the one PWM-only value |
| 1b | No signature, key, schema version, anchor, provider register, dependency map, failover, drill, retention class, incident report or assurance records | **Done.** See [GAP_ANALYSIS.md](GAP_ANALYSIS.md#pwm-08-gap-analysis) |
| 2 | Code-list values could not be matched to Kognita mechanically: 87 of 235 concepts had a notation, case differed, four enums had no list | **Done.** All concepts have a notation equal to the Kognita value; four lists added; `tests/test_kognita_vocabulary.py` fails on any difference. On its first run it found a real mismatch (`POSSIBLE_PROVIDER_CAPABILITY` against Kognita's `POSSIBLE_BANK_CAPABILITY`), now fixed. Running the same check inside Kognita's own CI is a 0.10 item |
| 3 | 66 of 198 classes and 37 properties had no example data | **Done.** Every class and property appears in the positive example, except three properties on a reasoned allow-list (`chainStart`, `pendingApproval`, `systemTrigger`). `tests/test_example_coverage.py` is the gate |
| 4 | Almost no OWL axioms | **Partly done.** Added disjoint mandate types, disjoint agent, model, use case, system and provider, AI agent disjoint from person, and 14 functional properties, checked on the example by an rdflib test. Inverse properties were not added on purpose (no consumer, and an unpopulated term is a cost). **HermiT was not run in this review** (ROBOT could not be run in the review environment); run it with `ROBOT_JAR` before publishing |
| 5 | 142 properties without a domain and 10 without a range | **Done for ranges and policy.** The ten ranges are declared. The no-domain pattern is now a stated design principle (shared properties omit a domain so using them never infers a class) |
| 6 | 24 of 47 references still at instrument level | **Not done.** Verifying a citation means reading the official text, which cannot be done from the review environment. The status is now visible: SPEC marks every instrument-level citation with a dagger and states the count. The 0.9 release remains responsible for the work |
| 7 | Documentation was cumulative prose | **Done.** README describes the current state; history is in the generated `CHANGELOG.md`; GAP_ANALYSIS has one section per release |
| 8a | Mixed British and American spelling | **Done.** US English throughout PWM and the prose docs. Nine PWM terms renamed with deprecated aliases for one version. Kognita followed in a second step (see below) |
| 8b | Legacy `xsd:date` and `xsd:duration` ranges | **Not done.** Migration touches the review-expiry rule and its fixtures; scheduled before the 1.0 freeze (README, open items) |
| 8c | Hash checks are structural only | **Not done.** A real cryptographic check needs a chain written by Kognita and exported as RDF, which is the 0.10 JSON-LD round trip |
| 8d | Generated code-list shapes mostly untested by the fixture | **Done differently.** The README states that regression tests cover all generated shapes; the count is derived in the build |

## US English: what changed

PWM: term names, labels and definitions, and the prose of the README, roadmap, gap analysis and specification. Nine PWM IRIs changed spelling (`AuthorisedSignatory`, `BehaviourAnomaly`, `BehaviourBaseline`, `BookingCentre`, `bookingCentreCode`, `CommContextualisedInsight`, `InternationalOrganisationPEP`, `UnusualAgentBehaviour`, `VerbAnalyse`); each old IRI stays declared as a deprecated alias (`src/91-deprecated.ttl`).

Kognita (a second step, after the review's first publication): identifiers, values and policy-pack data now use US spelling too, so the PWM notations match with no exceptions. The renames include the action verb `analyze`, the transfer type `license`, the drill scenario `unauthorized-network`, the deployment key `memory_behavior`, the policy-pack attribute `booking_center` and the pack files `booking-center*.yaml`, the check codes `C1-LABELED` and `C1-UNLABELED`, the export field `anonymization`, and helper names such as `_authorized_human`. See the Kognita change log for the breaking-change notes and the compatibility that was kept (the `--anonymisation` flag and legacy `licence` and `memory_behaviour` values still load).
