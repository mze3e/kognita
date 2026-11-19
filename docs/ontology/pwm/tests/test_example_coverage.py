"""Every PWM class and property appears in the positive example, or is on an explicit allow-list.

A class nobody populates is untested: the shapes cannot catch a modeling error in it.
Classes count as populated when an instance of the class or of any subclass appears.
"""
from pathlib import Path

from rdflib import Graph, OWL, RDF, RDFS, URIRef

BASE = Path(__file__).resolve().parents[1]
PWM = "https://ahmedmuzammil.com/ontology/pwm#"
ONT = Graph().parse(BASE / "pwm.ttl")
EX = Graph().parse(BASE / "examples/pwm-example.ttl")

# Meta-model classes describe the ontology itself and have no instances in business data.
META_CLASSES = {"Module", "Reference", "Citation", "SourceType", "ContextSpeed"}
# Properties that need data the synthetic firm does not have, with the reason. Keep this short;
# an entry here is a debt, not a decision.
UNUSED_PROPERTIES = {
    "chainStart": "the example excerpt starts mid-chain and points to an external predecessor; the negative fixtures use it",
    "pendingApproval": "needs a suspended run with an unresolved approval request; the example run has completed",
    "systemTrigger": "needs a governed call made by a system trigger instead of an agent",
}


def local(term):
    return str(term)[len(PWM):]


def subclasses(c):
    out = {c}
    for s in ONT.subjects(RDFS.subClassOf, c):
        out |= subclasses(s)
    return out


def test_every_class_has_example_data():
    used = set(EX.objects(None, RDF.type))
    missing = []
    for c in ONT.subjects(RDF.type, OWL.Class):
        if not str(c).startswith(PWM) or local(c) in META_CLASSES or ONT.value(c, OWL.deprecated):
            continue
        if not (subclasses(c) & used):
            missing.append(local(c))
    assert sorted(missing) == []


def test_every_property_is_used_in_the_example():
    used = set(EX.predicates())
    missing = []
    for t in (OWL.ObjectProperty, OWL.DatatypeProperty):
        for p in ONT.subjects(RDF.type, t):
            if str(p).startswith(PWM) and not ONT.value(p, OWL.deprecated) and p not in used:
                missing.append(local(p))
    assert sorted(set(missing) - set(UNUSED_PROPERTIES)) == []
