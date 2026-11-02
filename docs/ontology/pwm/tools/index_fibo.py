"""Index FIBO classes and properties with label, definition, module and maturity."""
import json, sys, pathlib
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, OWL, SKOS

root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
AV = Namespace("https://www.omg.org/spec/Commons/AnnotationVocabulary/")
AV_FIBO = Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/Utilities/AnnotationVocabulary/")
index = {}
errors = []
for f in sorted(root.rglob("*.rdf")):
    rel = str(f.relative_to(root))
    if rel.startswith(("etc/", "EXMP/")) or "/All" in rel or rel.startswith("All") or "Metadata" in rel or rel.startswith("About"):
        continue
    g = Graph()
    try:
        g.parse(f, format="xml")
    except Exception as e:
        errors.append((rel, str(e)[:120]))
        continue
    onts = list(g.subjects(RDF.type, OWL.Ontology))
    maturity = None
    for o in onts:
        for p in (AV.hasMaturityLevel, AV_FIBO.hasMaturityLevel):
            for m in g.objects(o, p):
                maturity = str(m).rsplit("/", 1)[-1]
    for kind, t in (("Class", OWL.Class), ("ObjectProperty", OWL.ObjectProperty), ("DatatypeProperty", OWL.DatatypeProperty)):
        for s in g.subjects(RDF.type, t):
            if not isinstance(s, URIRef):
                continue
            iri = str(s)
            if not iri.startswith("https://spec.edmcouncil.org/fibo/ontology/"):
                continue
            # only record terms defined in this file (isDefinedBy points at this ontology)
            defined_by = [str(x) for x in g.objects(s, RDFS.isDefinedBy)]
            label = next((str(x) for x in g.objects(s, RDFS.label)), None)
            definition = next((str(x) for x in g.objects(s, SKOS.definition)), None)
            supers = [str(x) for x in g.objects(s, RDFS.subClassOf) if isinstance(x, URIRef)]
            if not defined_by and label is None:
                continue
            entry = index.setdefault(iri, {})
            entry.update({k: v for k, v in {
                "kind": kind, "label": label, "definition": definition,
                "file": rel, "maturity": maturity, "defined_by": defined_by[:1],
                "super": supers,
            }.items() if v})
out.write_text(json.dumps(index, indent=1, sort_keys=True))
print("terms:", len(index), "parse errors:", len(errors))
for e in errors[:10]:
    print("ERR", e)
