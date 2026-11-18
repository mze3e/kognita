"""Every Kognita vocabulary enum matches its PWM code list, value for value.

Reads the enums from the Kognita source with ast, so it needs neither Kognita's
dependencies nor an installed package. A new or renamed Kognita value fails here
until pwm:notation in the matching scheme follows it.
"""
import ast
from pathlib import Path

import pytest
from rdflib import Graph, Namespace, RDF
from rdflib.namespace import SKOS

BASE = Path(__file__).resolve().parents[1]
SRC = BASE.parents[2] / "src" / "kognita"
PWM = Namespace("https://ahmedmuzammil.com/ontology/pwm#")
G = Graph().parse(BASE / "pwm.ttl")


def enum_values(path, cls):
    tree = ast.parse((SRC / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {s.value.value for s in node.body
                    if isinstance(s, ast.Assign) and isinstance(s.value, ast.Constant)
                    and isinstance(s.value.value, str)}
    raise AssertionError(f"{cls} not found in {path}")


def tuple_values(path, name):
    tree = ast.parse((SRC / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not found in {path}")


def dict_keys(path, name):
    tree = ast.parse((SRC / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return {k.value for k in node.value.keys}
    raise AssertionError(f"{name} not found in {path}")


def notations(scheme):
    return {str(n) for c in G.subjects(SKOS.inScheme, PWM[scheme]) for n in G.objects(c, SKOS.notation)}


# (scheme, Kognita values, PWM-only values allowed, reason)
CASES = {
    "Outcome": ("DecisionOutcomeScheme", lambda: enum_values("vocabulary.py", "Outcome"), set()),
    "Classification": ("DataClassificationScheme", lambda: enum_values("vocabulary.py", "Classification"), None),
    "EventType": ("EvidenceEventTypeScheme", lambda: enum_values("vocabulary.py", "EventType"), {"MODEL_REFUSAL"}),
    "ActorType": ("ActorTypeScheme", lambda: enum_values("vocabulary.py", "ActorType"), set()),
    "ApprovalStatus": ("ApprovalStatusScheme", lambda: enum_values("vocabulary.py", "ApprovalStatus"), set()),
    "EgressDecision": ("EgressDecisionScheme", lambda: enum_values("vocabulary.py", "EgressDecision"), set()),
    "CheckResult": ("CheckResultScheme", lambda: enum_values("vocabulary.py", "CheckResult"), set()),
    "FailureMode": ("GatewayFailureModeScheme", lambda: enum_values("vocabulary.py", "FailureMode"), set()),
    "StaleAction": ("StaleActionScheme", lambda: enum_values("vocabulary.py", "StaleAction"), set()),
    "ClaimType": ("ClaimTypeScheme", lambda: enum_values("claims.py", "ClaimType"), set()),
    "ValidatorStatus": ("ValidatorStatusScheme", lambda: enum_values("claims.py", "ValidatorStatus"), set()),
    "MemoryScope": ("MemoryScopeScheme", lambda: enum_values("memory.py", "MemoryScope"), set()),
    "DrillKind": ("DrillKindScheme", lambda: enum_values("breaker.py", "DrillKind"), set()),
    "RecordClasses": ("RecordClassScheme", lambda: tuple_values("retention.py", "_RECORD_CLASSES"), set()),
    "ActionVerbs": ("ActionVerbScheme", lambda: tuple_values("authority.py", "_VERBS"), set()),
    "SensitiveKinds": ("SensitiveKindScheme", lambda: dict_keys("sensitive.py", "_PATTERNS"), set()),
    "ConformanceProfiles": ("ConformanceProfileScheme", lambda: dict_keys("assurance/reports.py", "PROFILES"), set()),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_enum_matches_scheme(name):
    scheme, values, allowed_extra = CASES[name]
    kognita = values()
    pwm = notations(scheme)
    assert kognita - pwm == set(), f"{name}: Kognita values with no PWM concept: {sorted(kognita - pwm)}"
    if allowed_extra is not None:
        assert pwm - kognita == allowed_extra, f"{name}: PWM concepts with no Kognita value: {sorted(pwm - kognita - allowed_extra)}"


def test_every_concept_has_a_notation():
    concepts = set(G.subjects(SKOS.inScheme, None))
    missing = sorted(str(c).split("#")[1] for c in concepts if not list(G.objects(c, SKOS.notation)))
    assert missing == []
