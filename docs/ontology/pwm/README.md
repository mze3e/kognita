# Private Wealth Management Ontology (PWM)

**Version 0.1, draft for comment (2 November 2026).**
Web page: https://ahmedmuzammil.com/ontology/pwm · Full specification: [SPEC.md](SPEC.md)

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
| `pwm-shapes.ttl` | SHACL data rules, each citing the paragraph or item it enforces |
| `SPEC.md` | The specification, generated from the Turtle |
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

Keep an institution's detail in its own extension: specialise PWM classes, add the firm's code lists (risk scale, booking centres, product types), bind coded properties to them with `pwm:valuesFrom`, and add firm thresholds in a separate shapes graph. A generic rule can be switched off there with `sh:deactivated true`. Keep confidential content out of public repositories.

## Checks

```sh
pip install rdflib pyshacl

# Rebuild pwm.ttl and SPEC.md from src/
python3 tools/build.py

# Integrity: labels, definitions, modules, citations, code-list bindings, and every
# FIBO, OMG Commons and DPV IRI checked against its source
git clone https://github.com/edmcouncil/fibo /tmp/fibo
python3 tools/index_fibo.py /tmp/fibo /tmp/fibo-index.json
python3 tools/check.py pwm.ttl /tmp/fibo-index.json [path-to-dpv-2.3]

# SHACL: the example conforms; the negative fixture raises exactly the expected violations
python3 tools/validate.py

# OWL 2 DL profile and reasoning (ROBOT, https://robot.obolibrary.org)
robot validate-profile --profile DL --input pwm.ttl
robot reason --reasoner HermiT --equivalent-classes-allowed none --input pwm.ttl --output /tmp/reasoned.ttl
```

At 0.1 all of these pass: OWL 2 DL, consistent with no unsatisfiable classes under HermiT, the example conforms, and the negative fixture raises the 33 expected violations across all 28 targeted shapes.

## Status and licence

Draft. Terms may change before 1.0; changes are recorded in the change log in SPEC.md. Regulatory citations show where a term comes from, not how a rule applies to a firm; they are not legal advice.

The ontology, shapes, examples and specification in this folder are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The scripts in `tools/` are MIT, like the rest of Kognita. FIBO is MIT-licensed by the EDM Council.
