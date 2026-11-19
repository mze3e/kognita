"""Every Kognita vocabulary enum matches its PWM code list, value for value.

The enums are read from the Kognita source (tools/kognita_enums.py), so this needs neither
Kognita's dependencies nor an installed package. A new or renamed Kognita value fails here
until pwm:notation in the matching scheme follows it.
"""
import sys
from pathlib import Path

import pytest
from rdflib import Graph
from rdflib.namespace import SKOS

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "tools"))
from kognita_enums import CASES, notations  # noqa: E402

G = Graph().parse(BASE / "pwm.ttl")


@pytest.mark.parametrize("name", sorted(CASES))
def test_enum_matches_scheme(name):
    scheme, values, allowed_extra = CASES[name]
    kognita = values()
    pwm = notations(scheme, G)
    assert kognita - pwm == set(), f"{name}: Kognita values with no PWM concept: {sorted(kognita - pwm)}"
    if allowed_extra is not None:
        assert pwm - kognita == allowed_extra, f"{name}: PWM concepts with no Kognita value: {sorted(pwm - kognita - allowed_extra)}"


def test_every_concept_has_a_notation():
    concepts = set(G.subjects(SKOS.inScheme, None))
    missing = sorted(str(c).split("#")[1] for c in concepts if not list(G.objects(c, SKOS.notation)))
    assert missing == []
