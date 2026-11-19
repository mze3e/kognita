"""Reads Kognita's vocabulary enums from its source with ast, so no Kognita dependencies are needed.

Shared by tests/test_kognita_vocabulary.py and tools/gen_mapping.py.
"""
import ast
from pathlib import Path

from rdflib import Namespace
from rdflib.namespace import SKOS

BASE = Path(__file__).resolve().parents[1]
SRC = BASE.parents[2] / "src" / "kognita"
PWM = Namespace("https://ahmedmuzammil.com/ontology/pwm#")


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


def notations(scheme, graph):
    return {str(n) for c in graph.subjects(SKOS.inScheme, PWM[scheme]) for n in graph.objects(c, SKOS.notation)}


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
    "DrillScenarios": ("DrillScenarioScheme", lambda: tuple_values("assurance/drills.py", "SCENARIOS"), set()),
    "SensitiveKinds": ("SensitiveKindScheme", lambda: dict_keys("sensitive.py", "_PATTERNS"), set()),
    "ConformanceProfiles": ("ConformanceProfileScheme", lambda: dict_keys("assurance/reports.py", "PROFILES"), set()),
}


