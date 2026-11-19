# Handover: what was not run or not done in the Claude session

Branch: `claude/dreamy-bohr-5shfkg` (PWM 0.10 and Kognita 0.9.0). Written 19 November 2026.
Everything below needs a machine, a tool or a source that the cloud session did not have. Each item says what to do and what a pass looks like. Run PWM commands from `docs/ontology/pwm`, Kognita commands from the repository root.

## Needs a normal machine

1. **OWL 2 DL profile and HermiT (ROBOT).** The session could not run a ROBOT jar. Last recorded run was for 0.5.
   ```sh
   robot validate-profile --profile DL --input pwm.ttl
   robot reason --reasoner HermiT --equivalent-classes-allowed none --input pwm.ttl --output /tmp/reasoned.ttl
   ROBOT_JAR=/path/to/robot.jar python3 -m pytest tests/test_axioms.py   # HermiT on ontology plus example
   ```
   Pass: no profile violations, no unsatisfiable classes, the example stays consistent. Watch the new axioms in `src/92-axioms.ttl` (disjoint classes, functional properties) and `LombardFacility` as a subclass of `LendingFacility`. If a profile violation names an annotation `xsd:date` value, that is allowed.
2. **FIBO, OMG and DPV integrity check (`tools/check.py`).** Not run in this session.
   ```sh
   git clone https://github.com/edmcouncil/fibo /tmp/fibo && git -C /tmp/fibo checkout 33a92cd
   python3 tools/index_fibo.py /tmp/fibo /tmp/fibo-index.json
   python3 tools/check.py pwm.ttl /tmp/fibo-index.json path/to/dpv/2.3
   ```
   Pass: no problems. No new external IRIs were added in 0.8 to 0.10, so failures would point at a label, definition, module or citation gap in the new modules.
3. **Gateway timing tests.** `tests/test_gateway_overhead.py` (two tests) fails in the cloud sandbox, before and after every change. Run on normal hardware.
4. **Skipped Kognita tests.** 43 tests skip in the sandbox: Postgres 16 integration (`kognita db`, concurrent writers), the TypeScript verifier build and its agreement tests, and optional extras. Run `uv sync --extra dev --extra vec --extra signing --extra postgres`, `npm ci --prefix verify-ts && npm run build --prefix verify-ts`, then `uv run pytest` with Postgres available, as in `.github/workflows/ci.yml`.
5. **Full Kognita suite after the 0.9.0 bump.** The last full run (2089 passed, the two timing failures) was before the version bump and the PWM 0.10 commits. Only version strings, docs and PWM files changed since, but re-run `uv run pytest` once.
6. **Full PWM suite.** `python3 -m pytest tests` (about 10 minutes). It was running when this was written; the earlier full run for PWM 0.8 passed (135 passed, 1 skipped).

## Needs sources or people

7. **Citation verification.** 24 of 47 references are cited at instrument level (marked † in `SPEC.md`). Verify against the official text and record the paragraph. Priority: MiFID II and the Delegated Regulation, PRIIPs, FCA COBS and FG21/1, the EU AI Act, FATF, MAS Fair Dealing and AIRG, HK professional investor rules, OECD CRS. The 0.8 and 0.10 additions (module 13, `src/93-airg-citations.ttl`) cite MiFID II articles from memory at instrument level; check each locator.
8. **FIBO alignment of the operations terms** (module 13: orders, executions, allocations, corporate actions, income, lending). Index FIBO with step 2, find matching Release terms, add them to `src/90-external.ttl` through `tools/gen_external.py`, and align with `rdfs:subClassOf` or `skos:closeMatch`.
9. **Review of the jurisdiction profiles.** `profiles/jurisdictions/` carries investor categories only, derived from the PWM code lists. Disclosure triggers and cross-border rules were deliberately left out; a compliance reviewer should say what belongs there.
10. **Practitioner, ontology-engineer and second-implementation reviews** for 1.0-rc (see `ROADMAP.md`).

## Needs wiring

11. **Round trip in CI.** `examples/kognita-export.json` is committed. Add a CI step that writes a fresh export with `tools/roundtrip.py make` (needs Kognita installed plus `rdflib` and `pyshacl`) and then runs `python3 tools/roundtrip.py check <file>`. Extend the context to register exports.
12. **Publication.** The site (ahmedmuzammil.com) was unreachable from the cloud session. Nothing was deployed. To publish: run `python3 tools/build.py --site PATH`, make `https://ahmedmuzammil.com/ontology/pwm/0.10` and `.../0.10/pwm-shapes.ttl` resolve, and check that the published pages match the repository.
13. **Releases.** Kognita 0.9.0 was not tagged and not published to PyPI or npm. No pull request was opened.

## Breaking changes to communicate

Kognita 0.9.0 renamed identifiers and values to US English (see the `[0.9.0]` and Changed entries in `/CHANGELOG.md`). Existing deployments must re-register delegations and tool lists that use the verb `analyse`, refresh attestations whose digests cover `memory_behaviour`, and update policy overlays that name `booking_centre`. PWM 0.10 removed the 0.8 deprecated aliases and changed `nextReviewDate`, `retentionUntil` and `freshnessThreshold` (replaced by `freshnessThresholdSeconds`).

## Known limits to keep in mind

- `AIRG_MAP.md` is only as complete as the MAS table in `docs/control-frameworks.md`; its citations are instrument-level.
- `tools/gen_module12.py` and `tools/gen_module13.py` generate parts of the ontology, shapes, examples and negative fixtures. Edit the generators, then run the sequence in the README, not the generated blocks.
- The shape-count assertions in `tests/test_review_regressions.py` are literals; update them when shapes or fixtures change (the build prints the numbers).
