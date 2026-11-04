# Private Wealth Management Ontology (PWM)

**Version 0.4, draft for comment (3 November 2026).**
Web page: https://ahmedmuzammil.com/ontology/pwm · Full specification: [SPEC.md](SPEC.md)

PWM is a generic ontology for private banking and wealth management, and for the AI that works on it. It covers parties and relationships, due diligence, investor profiling and suitability, mandates, accounts and holdings, products and research, the client journey from origination to communication, and the governance of AI: use cases, risk materiality, accountability, agents, delegated authority, claims and evidence.

It is the data model that Kognita's controls map to. Each Kognita concept it carries (outcomes, data classes C0 to C3, autonomy levels L0 to L6, communication levels, claim types, validator statuses, owner roles, risk tiers, delegated authority, the evidence chain) is cited to the roadmap item or vocabulary it comes from.

## How it is built

- **FIBO first.** Business terms specialise or map to the Financial Industry Business Ontology, then OMG Commons, W3C PROV-O and DPV. A term is defined here only when none fits, and it is marked NEW.
- **Declared, not imported.** External terms are declared in `src/90-external.ttl` with label, source and FIBO maturity (Release or Provisional), so `pwm.ttl` loads on its own.
- **Every term cites its grounds.** Each class, property and code list carries citations with locators to the reference register (`src/01-references.ttl`): standards (ONT, ISO), regulation (REG), Kognita (KOG), the author's frameworks and talks (AM), or a practitioner definition (NEW). Each reference records whether it was verified against the source while drafting or cited at instrument level.
- **OWL, SKOS and SHACL.** Light OWL axioms, code lists as SKOS concept schemes, and data rules as SHACL shapes in a separate file.

## Files

| Path | What it is |
|---|---|
| `pwm.ttl` | The ontology, built from `src/`. Do not edit; edit `src/` and rebuild |
| `pwm-shapes.ttl` | SHACL data rules, each citing the paragraph or item it enforces, plus one generated code-list rule per coded property |
| `SPEC.md` | The specification, generated from the Turtle |
| `examples/pwm-example.ttl` | Synthetic data for a fictional firm: one client journey end to end |
| `examples/pwm-example-invalid.ttl` | Negative fixture records and their supporting records |
| `src/*.ttl` | Sources: header, reference register, one file per module group, code-list bindings, external declarations |
| `tools/` | Build, integrity checks and validation |

IRIs: ontology `https://ahmedmuzammil.com/ontology/pwm`, version `https://ahmedmuzammil.com/ontology/pwm/0.4` (resolves to the Turtle file), namespace `https://ahmedmuzammil.com/ontology/pwm#` (each term resolves to its entry on the web page).

Shapes use `https://ahmedmuzammil.com/ontology/pwm/0.4/pwm-shapes.ttl` as their version IRI. The unversioned `https://ahmedmuzammil.com/ontology/pwm/shapes` remains the latest pointer. Both retain the original host without www.

The roadmap source is pinned to software release **0.7.0**, commit `9303769`, not roadmap milestone 0.7. See the source verification table in SPEC for every FIBO IRI and its maturity. The sitemap and speaking page expose no talk-specific Filing Cabinet page, so its general speaking link is retained with a note.

Sustainability preferences support combined taxonomy and SFDR minimum proportions (fractions from 0 to 1), client-determined PAI elements, or exclusive no preference. EU suitability records marked `jurisdictionCode "EU"` require one preference record. These shapes check recorded structure; product matching and the full legal assessment remain with the firm.

## Using it

Import the version IRI:

```turtle
<https://example.org/firm/ontology> a owl:Ontology ;
    owl:imports <https://ahmedmuzammil.com/ontology/pwm/0.4> .
```

The version IRI resolves to `pwm.ttl` once the web page is published. Until then, or to work offline, map it to a local copy. In Protégé, put a `catalog-v001.xml` next to your ontology:

```xml
<catalog prefer="public" xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
  <uri name="https://ahmedmuzammil.com/ontology/pwm/0.4" uri="pwm.ttl"/>
</catalog>
```

ROBOT reads the same catalog with `--catalog catalog-v001.xml`. With rdflib or pyshacl, load `pwm.ttl` directly.

Keep an institution's detail in its own extension: specialise PWM classes, add the firm's code lists (risk scale, booking centres, product types), and add firm thresholds in a separate shapes graph. Keep confidential content out of public repositories.

**Replacing a code list.** `pwm:valuesFrom` in `pwm.ttl` names PWM's default code list for each coded property, and the generated shape `pwmsh:CodeList_<property>` enforces it. A binding added in an extension does not remove PWM's: both remain in the merged graph. To use the firm's own list, for example its risk scale in place of the illustrative one:

1. Add `pwm:valuesFrom firm:RiskScale` bindings for `pwm:riskProfileLevel`, `pwm:productRiskLevel` and `pwm:targetRiskProfile`.
2. In the firm's shapes graph, set `sh:deactivated true` on `pwmsh:CodeList_riskProfileLevel`, `pwmsh:CodeList_productRiskLevel` and `pwmsh:CodeList_targetRiskProfile`.
3. Add shapes enforcing the firm's allowed codes on all three properties. See the separate ontology and shapes examples and graph-loading code in [SPEC.md](SPEC.md#extending-pwm-for-an-institution).

Treat the firm's binding as the one in force for this validation configuration; ontology triples do not override each other. `check.py` checks the base ontology only.

Deactivation alone does not support arbitrary list replacement. Handwritten constraints and code-dependent rules still require built-in codes, including lifecycle status, communication level, classification, claim type, validator status and autonomy level. Replacing those lists requires adapting every affected handwritten constraint and rule in the firm's shapes graph too. The risk-scale example has no such additional code restrictions.

## Checks

Run these commands from `docs/ontology/pwm`.

```sh
pip install rdflib pyshacl pytest

# Rebuild pwm.ttl and SPEC.md from src/
python3 tools/build.py

# Integrity: labels, definitions, modules, citations, code-list bindings, and every
# FIBO, OMG Commons and DPV IRI checked against its source
# (FIBO at the commit the register cites; DPV 2.3 from https://github.com/w3c/dpv)
git clone https://github.com/edmcouncil/fibo /tmp/fibo && git -C /tmp/fibo checkout 33a92cd
python3 tools/index_fibo.py /tmp/fibo /tmp/fibo-index.json
python3 tools/check.py pwm.ttl /tmp/fibo-index.json path/to/dpv/2.3   # without the DPV folder, DPV IRIs are reported as unchecked

# Every pwm: term used in the shapes and examples is defined (exits 1 if not)
python3 tools/check_shapes_terms.py pwm.ttl pwm-shapes.ttl
python3 tools/check_shapes_terms.py pwm.ttl examples/pwm-example.ttl
python3 tools/check_shapes_terms.py pwm.ttl examples/pwm-example-invalid.ttl

# SHACL: the example conforms and its evidence chain is unbroken; the negative
# fixture raises exactly the expected violations (-v prints them)
python3 tools/validate.py

# PWM regression tests (separate from root pytest collection)
python3 -m pytest tests/test_review_regressions.py

# OWL 2 DL profile and reasoning (ROBOT, https://robot.obolibrary.org)
robot validate-profile --profile DL --input pwm.ttl
robot reason --reasoner HermiT --equivalent-classes-allowed none --input pwm.ttl --output /tmp/reasoned.ttl
```

To reason over data, merge it with `pwm.ttl` as RDF first (for example with rdflib), so that properties keep the types `pwm.ttl` declares.

Version 0.4 contains 168 domain classes, 295 properties (173 object, 122 datatype), 46 code lists with 229 codes, 84 written rules and 59 generated rules. There are 144 node shapes, including one untargeted helper, and 143 targeted shapes. The positive example contains 152 typed records. The negative fixture contains 127 typed records: 101 violating records and 26 supporting records, plus 4 untyped violating subjects, raising 112 expected violations across 105 focus nodes. It covers all 84 written shapes and two generated code-list shapes. Regression tests exercise all 59 generated code-list shapes.

Pre-existing pwm:nextReviewDate and pwm:retentionUntil use xsd:date; pwm:freshnessThreshold uses xsd:duration. These datatypes fall outside the OWL 2 datatype map. Their ranges are preserved for compatibility with 0.2. ROBOT DL profile validation passes with xsd:date and xsd:duration as declared datatypes; this does not establish conformance to the OWL 2 datatype map. Open item for a later version: migrate these legacy ranges and their data to OWL 2 datatype-map datatypes. HermiT consistency checks pass with no unsatisfiable classes. The positive example conforms, is consistent with the ontology, and has one unbroken evidence chain. The negative fixture is intentionally inconsistent under HermiT because it names an AI agent in an accountability role.

See [the 0.4 gap analysis](GAP_ANALYSIS.md) for reused terms, additions, grounds and runtime boundaries.

The inherited 0.3 model includes run budgets and consumption, durable approval checkpoints and resolution, gateway model and MCP records, classifier evidence and typed-attribute precedence, pinned policy checks and retrieved content, retention and evidenced erasure, ten-question reconstruction coverage, failure modes, and authenticated client identity bindings. Existing model, hash, provider, decision and event vocabulary is reused. New specialised records and fields cite specific 0.3 items and are marked NEW; Run, RunResumption, GovernedCall and its subclasses ModelCall, ToolCall and MCPProxiedCall, and ErasureEvent specialise prov:Activity. RunBudget, RunCheckpoint, ApprovalRequest, ApprovalResolution, BudgetExceededDenial (through PolicyDecision), AuthenticatedClientConfiguration, ClassifierOutput, EnvelopeAttribute, PolicyCheck, RetrievedItem, RetentionEntry, RetentionPolicy, ReconstructionReport, QuestionCoverage and ReconstructionFinding specialise prov:Entity. SystemTrigger specialises prov:Agent. MAS AIRG grounds use the existing 7 October 2026 register entry, including 5.9 human oversight, 5.17 reproducibility and 5.23(d) record keeping. Retention and policy checks also reuse the existing Filing Cabinet slide grounds.

## Context

Retention duration is measured in whole days as xsd:nonNegativeInteger. RetentionPolicy uses SlowContext/C1; AuthenticatedClientConfiguration and SystemTrigger use SlowContext/C2. The 0.4 versioned policy artefacts and breaker configuration use SlowContext; recorded evidence and review state use MediumContext. DelegatedAuthority and KillSwitch definitions are cached policy/config in SlowContext. AuthorityLiveStatus is the single shared live-status vocabulary for both controls and will be reused in 0.5 without a parallel class or code list. Live revocation and kill-switch state must be read from AuthorityLiveStatus (FastContext) at decision time, never from cache. Use the latest asOf observation at or before the decision time; conflicting observations fail closed. Runtime must obtain fresh observations; SHACL validates the recorded state only (KOG roadmap 0.4 item 14; 0.3 item 2; MAS AIRG 5.9). Durable evidence, snapshots and checkpoints use MediumContext, including inherited 0.3 records.

PWM version N aligns with Kognita roadmap milestone N. Version 0.4 adds 18 classes, 70 properties and two classification schemes for structured ingestion, declarative policy artefacts and tests, policy packs, recorder replay, RM edits, review queues and coverage, circuit-breaker incidents, affected-client results and outcomes. New terms cite individual 0.4 items and are marked NEW. Existing register, materiality, origination, decision, communication and near-miss terms are reused. MAS AIRG grounds retain the registered 7 October 2026 instrument. SPEC describes the exact PROV parents and the runtime boundary. Removed roadmap item 2 adds no model.

The shapes validate recorded structure and relationships. They do not fetch retained payloads, execute gateway controls, or recompute content hashes. The example uses synthetic content references; production reconstruction must retrieve the retained content and compare hashes. A report records that verification result and any mismatch finding. Per-use-case degraded mode permits only local models without client data and requires evidence after recovery; this policy meaning is recorded in the code definition, while enforcement belongs to the gateway.

The build validates the positive example and calls `chain_problems` before reporting an unbroken evidence chain. A broken chain fails the build.

## Status and licence

Draft. Terms may change before 1.0; changes are recorded in the change log in SPEC.md. Regulatory citations show where a term comes from, not how a rule applies to a firm; they are not legal advice.

The ontology, shapes, examples and specification in this folder are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The scripts in `tools/` are MIT, like the rest of Kognita. FIBO is MIT-licensed by the EDM Council.

The 0.3 review fixes use `xsd:nonNegativeInteger` in whole days for `retentionDuration`, with matching shapes and example data. `RetentionPolicy` now uses `SlowContext` and `C1`; `AuthenticatedClientConfiguration` and the persistent configured identity `SystemTrigger` use `SlowContext` and retain `C2`. Inherited execution records, durable evidence, snapshots and checkpoints use `MediumContext`. `PolicyDecision` and `EvidenceRecord` also use `MediumContext`; `DelegatedAuthority` and `KillSwitch` use `SlowContext` for governance configuration.

The proxy portfolio read uses the pre-drafting PolicyDecisionModelAllow. Held calls require an approved request whose checkpoint belongs to the call's run, including when the decision names multiple runs.

Shape sources live in `src/shapes/pwm-shapes.ttl`; the build copies them and regenerates the code-list rules. Review expiry uses typed timestamps and dates, treating absent timezones as UTC and allowing the entire review date. Every policy decision requires exactly one recorded timestamp.
