"""Integrity checks for the PWM ontology.

Usage: check.py pwm.ttl fibo-index.json [dpv-dir]

Without dpv-dir, DPV IRIs are not checked and the script says so.

Fails (exit 1) if any term lacks a label, definition, module or citation, if a
citation points at an unregistered reference, if any pwm: IRI is used but not
defined, or if any FIBO, OMG Commons or DPV IRI cannot be found in its source.
"""
import json
import pathlib
import re
import sys

from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import OWL, RDF, RDFS, SKOS

ttl, fibo_path = sys.argv[1], sys.argv[2]
dpv_dir = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
g = Graph()
g.parse(ttl, format="turtle")
fibo = json.loads(pathlib.Path(fibo_path).read_text(encoding="utf-8"))

PWM = "https://ahmedmuzammil.com/ontology/pwm#"
P = lambda x: URIRef(PWM + x)
problems = []

refs = set(g.subjects(RDF.type, P("Reference")))
meta = {P(x) for x in ["Module", "Reference", "Citation", "SourceType", "ContextSpeed"]}
terms = [s for t in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, SKOS.ConceptScheme)
         for s in g.subjects(RDF.type, t) if isinstance(s, URIRef) and str(s).startswith(PWM) and s not in meta]
for s in terms:
    if not (g.value(s, RDFS.label) or g.value(s, SKOS.prefLabel)):
        problems.append(f"no label: {s}")
    if not g.value(s, SKOS.definition) and (s, RDF.type, SKOS.ConceptScheme) not in g:
        problems.append(f"no definition: {s}")
    if not g.value(s, P("inModule")):
        problems.append(f"no module: {s}")
    cits = list(g.objects(s, P("citation")))
    if not cits:
        problems.append(f"no citation: {s}")
    for c in cits:
        r = g.value(c, P("cites"))
        if r not in refs:
            problems.append(f"citation to unregistered reference {r} on {s}")
        if not g.value(c, P("locator")):
            problems.append(f"citation without locator on {s}")

# Every coded property names the one code list it takes values from.
for prop in g.subjects(RDFS.range, SKOS.Concept):
    if not str(prop).startswith(PWM):
        continue
    schemes = list(g.objects(prop, P("valuesFrom")))
    if len(schemes) != 1:
        problems.append(f"coded property needs exactly one pwm:valuesFrom: {prop}")
    for sch in schemes:
        if (sch, RDF.type, SKOS.ConceptScheme) not in g:
            problems.append(f"pwm:valuesFrom is not a concept scheme: {prop} -> {sch}")

concepts = [s for s in g.subjects(RDF.type, SKOS.Concept)]
for c in concepts:
    if not g.value(c, SKOS.prefLabel) or not g.value(c, SKOS.definition) or not g.value(c, SKOS.inScheme):
        problems.append(f"concept missing prefLabel/definition/inScheme: {c}")

# Every pwm: IRI used must be defined (have an rdf:type).
for node in set(g.subjects()) | set(o for o in g.objects() if isinstance(o, URIRef)) | set(g.predicates()):
    if isinstance(node, URIRef) and str(node).startswith(PWM):
        if (node, RDF.type, None) not in g:
            problems.append(f"used but not defined: {node}")

# External IRIs must resolve in their sources.
commons_ok = {
    "PartiesAndSituations/Party", "PartiesAndSituations/PartyRole", "PartiesAndSituations/Situation",
    "PartiesAndSituations/SoftwareAgent", "Organizations/LegalEntity", "Organizations/ServiceProvider",
    "Organizations/isDomiciledIn", "Organizations/isProvidedBy", "RegulatoryAgencies/Jurisdiction",
    "RegulatoryAgencies/RegulatoryAgency", "RegulatoryAgencies/hasJurisdiction",
    "BusinessAuthorizations/Authorization", "RolesAndCompositions/isPlayedBy", "Documents/Document",
}
dpv_terms = {}
dpv_unchecked = set()
if dpv_dir:
    for f, ns in ((dpv_dir / "dpv" / "dpv.ttl", "https://w3id.org/dpv#"), (dpv_dir / "ai" / "ai.ttl", "https://w3id.org/dpv/ai#")):
        text = f.read_text(encoding="utf-8")
        prefix = "dpv:" if ns.endswith("dpv#") else "ai:"
        dpv_terms[ns] = set(re.findall(r"^" + re.escape(prefix) + r"([A-Za-z0-9]+) a ", text, flags=re.M))
for node in set(o for o in g.objects() if isinstance(o, URIRef)) | set(g.subjects()) | set(g.predicates()):
    iri = str(node)
    if iri.startswith("https://spec.edmcouncil.org/fibo/ontology/") and not iri.endswith("/"):
        if iri not in fibo:
            problems.append(f"FIBO IRI not found in FIBO source: {iri}")
    elif iri.startswith("https://www.omg.org/spec/Commons/") and not iri.endswith("/"):
        if iri.split("Commons/", 1)[1] not in commons_ok:
            problems.append(f"Commons IRI not on the verified list: {iri}")
    elif iri.startswith("https://w3id.org/dpv#") or iri.startswith("https://w3id.org/dpv/ai#"):
        if not dpv_dir:
            dpv_unchecked.add(iri)
            continue
        ns, local = iri.split("#")[0] + "#", iri.split("#")[1]
        if local not in dpv_terms.get(ns, set()):
            problems.append(f"DPV IRI not found in DPV 2.3: {iri}")

stats = {
    "classes": sum(1 for s in g.subjects(RDF.type, OWL.Class) if str(s).startswith(PWM) and s not in meta),
    "object properties": sum(1 for s in g.subjects(RDF.type, OWL.ObjectProperty) if str(s).startswith(PWM)),
    "datatype properties": sum(1 for s in g.subjects(RDF.type, OWL.DatatypeProperty) if str(s).startswith(PWM)),
    "concept schemes": sum(1 for s in g.subjects(RDF.type, SKOS.ConceptScheme)),
    "concepts": len(concepts),
    "references": len(refs),
    "citations": sum(1 for _ in g.subject_objects(P("citation"))),
    "external terms declared": sum(1 for s in g.subjects(RDFS.isDefinedBy, None) if not str(s).startswith(PWM)),
    "triples": len(g),
}
for k, v in stats.items():
    print(f"{k:24} {v}")
if problems:
    print(f"\n{len(problems)} problem(s):")
    for p in sorted(set(problems)):
        print("  -", p)
    sys.exit(1)
if dpv_unchecked:
    print(f"\nWARNING: {len(dpv_unchecked)} DPV IRIs were not checked; pass the DPV 2.3 folder as the third argument")
print("\nall integrity checks passed" + (" except the DPV check, which was skipped" if dpv_unchecked else ""))
