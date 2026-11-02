# Private Wealth Management Ontology (PWM)

**Version 0.1, draft for comment (2 November 2026).**
Web page: https://ahmedmuzammil.com/ontology/pwm · Full specification: [SPEC.md](SPEC.md) · Road to 1.0: [ROADMAP.md](ROADMAP.md)

PWM is a generic ontology for private banking and wealth management, and for the AI that works on it. It covers parties and relationships, due diligence, investor profiling and suitability, mandates, accounts and holdings, products and research, the client journey from origination to communication, and the governance of AI: use cases, risk materiality, accountability, agents, delegated authority, claims and evidence.

It is the data model that Kognita's controls map to. Each Kognita concept it carries (outcomes, data classes C0 to C3, autonomy levels L0 to L6, communication levels, claim types, validator statuses, owner roles, risk tiers, delegated authority, the evidence chain) is cited to the roadmap item or vocabulary it comes from.

## How it is built

- **FIBO first.** Business terms specialise or map to the Financial Industry Business Ontology, then OMG Commons, W3C PROV-O and DPV. A term is defined here only when none fits, and it is marked NEW.
- **Declared, not imported.** External terms are declared in `src/90-external.ttl` with label, source and FIBO maturity (Release or Provisional), so `pwm.ttl` loads on its own.
- **Every term cites its grounds.** Each class, property and code list carries citations with locators to the reference register (`src/01-references.ttl`): standards (ONT, ISO), regulation (REG), Kognita (KOG), the author's frameworks and talks (AM), or a practitioner definition (NEW). Each reference records whether it was verified against the source while drafting or cited at instrument level.
- **OWL 2 DL, SKOS and SHACL.** Light OWL axioms, code lists as SKOS concept schemes, and data rules as SHACL shapes in a separate file.

## Files

| Path | What it is |
|---|---|
| `pwm.ttl` | The ontology, built from `src/`. Do not edit; edit `src/` and rebuild |
| `pwm-shapes.ttl` | SHACL data rules, each citing the paragraph or item it enforces, plus one generated code-list rule per coded property |
| `SPEC.md` | The specification, generated from the Turtle |
| `ROADMAP.md` | The plan from 0.1 to 1.0 |
| `examples/pwm-example.ttl` | Synthetic data for a fictional firm: one client journey end to end |
| `examples/pwm-example-invalid.ttl` | Negative fixture: each record breaks one rule |
| `src/*.ttl` | Sources: header, reference register, one file per module group, code-list bindings, external declarations |
| `tools/` | Build, integrity checks and validation |

IRIs: ontology `https://ahmedmuzammil.com/ontology/pwm`, version `https://ahmedmuzammil.com/ontology/pwm/0.1` (resolves to the Turtle file), namespace `https://ahmedmuzammil.com/ontology/pwm#` (each term resolves to its entry on the web page).

## Using it

Import the version IRI:

```turtle
<https://example.org/firm/ontology> a owl:Ontology ;
    owl:imports <https://ahmedmuzammil.com/ontology/pwm/0.1> .
```

The version IRI resolves to `pwm.ttl` once the web page is published. Until then, or to work offline, map it to a local copy. In Protégé, put a `catalog-v001.xml` next to your ontology:

```xml
<catalog prefer="public" xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">
  <uri name="https://ahmedmuzammil.com/ontology/pwm/0.1" uri="pwm.ttl"/>
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

At 0.1 all of these pass. The ontology is in the OWL 2 DL profile and consistent under HermiT, with no unsatisfiable classes. The example conforms, is consistent with the ontology, and has an unbroken evidence chain. The negative fixture raises the 36 expected violations: it covers all 28 written shapes and one of the 48 generated code-list shapes. Under HermiT it is inconsistent, because it names an AI agent in an accountability role.

## Status and licence

Draft. Terms may change before 1.0; changes are recorded in the change log in SPEC.md. Regulatory citations show where a term comes from, not how a rule applies to a firm; they are not legal advice.

The ontology, shapes, examples and specification in this folder are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The scripts in `tools/` are MIT, like the rest of Kognita. FIBO is MIT-licensed by the EDM Council.
