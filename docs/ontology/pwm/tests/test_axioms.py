"""The positive example satisfies the reasoning axioms, and ROBOT agrees when it is installed.

HermiT consistency of the merged ontology and example runs only when a ROBOT jar is
named in the ROBOT_JAR environment variable; the rdflib check below always runs.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from rdflib import Graph, RDF, OWL
from rdflib.collection import Collection
from rdflib.namespace import RDFS

BASE = Path(__file__).resolve().parents[1]
ONT = Graph().parse(BASE / "pwm.ttl")
EX = Graph().parse(BASE / "examples/pwm-example.ttl")


def closure(types):
    seen = set(types)
    stack = list(types)
    while stack:
        t = stack.pop()
        for parent in ONT.objects(t, RDFS.subClassOf):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def test_functional_properties_have_one_value():
    problems = []
    for p in ONT.subjects(RDF.type, OWL.FunctionalProperty):
        for s in set(EX.subjects(p, None)):
            values = set(EX.objects(s, p))
            if len(values) > 1:
                problems.append((str(s), str(p), len(values)))
    assert problems == []


def test_disjoint_classes_do_not_overlap_in_the_example():
    groups = []
    for node in ONT.subjects(RDF.type, OWL.AllDisjointClasses):
        groups.append(list(Collection(ONT, ONT.value(node, OWL.members))))
    for a, b in ONT.subject_objects(OWL.disjointWith):
        groups.append([a, b])
    assert groups
    problems = []
    for s in set(EX.subjects(RDF.type, None)):
        types = closure(set(EX.objects(s, RDF.type)))
        for group in groups:
            hit = [g for g in group if g in types]
            if len(hit) > 1:
                problems.append((str(s), [str(h) for h in hit]))
    assert problems == []


@pytest.mark.skipif(not (os.environ.get("ROBOT_JAR") and shutil.which("java")), reason="ROBOT_JAR not set")
def test_hermit_consistent_with_example(tmp_path):
    merged = tmp_path / "merged.ttl"
    g = Graph()
    g += ONT
    g += EX
    g.serialize(merged, format="turtle")
    out = subprocess.run(["java", "-jar", os.environ["ROBOT_JAR"], "reason", "--reasoner", "HermiT",
                          "--equivalent-classes-allowed", "none", "--input", str(merged),
                          "--output", str(tmp_path / "out.ttl")], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
