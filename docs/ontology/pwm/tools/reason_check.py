"""OWL 2 DL profile and HermiT consistency checks, run through ROBOT.

Usage:
    ROBOT_JAR=/path/to/robot.jar python3 tools/reason_check.py [pwm.ttl]

Requires Java and a ROBOT jar (https://robot.obolibrary.org). Exits 0 only when
the ontology is in the OWL 2 DL profile, HermiT classifies it, no pair of named
classes is inferred equivalent, and no named class is unsatisfiable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rdflib import Graph, OWL, RDF, RDFS, URIRef

BASE = Path(__file__).resolve().parents[1]


def _run(robot: list[str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(robot + list(args), capture_output=True, text=True)


def main() -> int:
    jar = os.environ.get("ROBOT_JAR")
    if not jar or not Path(jar).is_file():
        print("ROBOT_JAR is not set or is not a file", file=sys.stderr)
        return 2
    if not shutil.which("java"):
        print("java is not on PATH", file=sys.stderr)
        return 2
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "pwm.ttl"
    robot = ["java", "-jar", jar]
    problems = []

    profile = Path(tempfile.mkdtemp()) / "dl-profile.txt"
    result = _run(robot, "validate-profile", "--profile", "DL", "--input", str(source), "--output", str(profile))
    report = profile.read_text() if profile.exists() else ""
    if result.returncode != 0:
        problems.append("OWL 2 DL profile check failed:\n" + result.stdout + result.stderr + report)
    elif "not in profile" in report.lower() or "violation" in report.lower():
        problems.append("OWL 2 DL profile violations:\n" + report)
    else:
        print("OWL 2 DL profile: in profile")

    reasoned = Path(tempfile.mkdtemp()) / "reasoned.ttl"
    result = _run(
        robot,
        "reason",
        "--reasoner", "HermiT",
        "--equivalent-classes-allowed", "none",
        "--input", str(source),
        "--output", str(reasoned),
    )
    if result.returncode != 0:
        problems.append("HermiT reasoning failed:\n" + result.stdout + result.stderr)
    else:
        print("HermiT: consistent, no inferred equivalent named classes")
        graph = Graph().parse(reasoned)
        unsat = set()
        for s in graph.subjects(OWL.equivalentClass, OWL.Nothing):
            if isinstance(s, URIRef):
                unsat.add(s)
        for s in graph.subjects(RDFS.subClassOf, OWL.Nothing):
            if isinstance(s, URIRef):
                unsat.add(s)
        for s in graph.objects(OWL.Nothing, OWL.equivalentClass):
            if isinstance(s, URIRef):
                unsat.add(s)
        # owl:Nothing itself is not a problem
        unsat.discard(OWL.Nothing)
        print(f"unsatisfiable named classes: {len(unsat)}")
        if unsat:
            problems.append("unsatisfiable named classes:\n" + "\n".join(f"  {i}" for i in sorted(unsat, key=str)))
        named = sum(1 for s in graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef))
        print(f"named classes in the reasoned graph: {named}")

    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("all reasoner checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
