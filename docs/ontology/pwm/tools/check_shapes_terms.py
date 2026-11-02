"""List pwm IRIs used in a data/shapes file that pwm.ttl does not define."""
import sys
from rdflib import Graph, URIRef
from rdflib.namespace import RDF

PWM = "https://ahmedmuzammil.com/ontology/pwm#"
ont = Graph().parse(sys.argv[1])
defined = {s for s in ont.subjects(RDF.type, None) if isinstance(s, URIRef) and str(s).startswith(PWM)}
other = Graph().parse(sys.argv[2])
used = set()
for t in other:
    for x in t:
        if isinstance(x, URIRef) and str(x).startswith(PWM):
            used.add(x)
missing = sorted(str(u)[len(PWM):] for u in used - defined)
print("used:", len(used), "missing:", len(missing))
for m in missing:
    print("  ", m)
