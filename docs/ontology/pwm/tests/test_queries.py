"""Each reconstruction question has a published SPARQL query that answers it from the example data."""
from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import XSD

BASE = Path(__file__).resolve().parents[1]
PWM = "https://ahmedmuzammil.com/ontology/pwm#"
EX = "https://example.org/pwm-example#"
G = Graph().parse(BASE / "pwm.ttl") + Graph().parse(BASE / "examples/pwm-example.ttl")
RECONSTRUCTION = sorted((BASE / "queries").glob("q[0-9][0-9]-*.rq"))


def run(path, graph=G, **bindings):
    return list(graph.query(path.read_text(), initBindings=bindings))


def test_there_is_one_query_per_reconstruction_question():
    assert len(RECONSTRUCTION) == 10
    questions = Graph().parse(BASE / "pwm.ttl")
    from rdflib.namespace import SKOS
    assert len(set(questions.subjects(SKOS.inScheme, URIRef(PWM + "ReconstructionQuestionScheme")))) == 10


@pytest.mark.parametrize("path", RECONSTRUCTION, ids=lambda p: p.stem)
def test_reconstruction_query_answers_the_example_interaction(path):
    rows = run(path, interaction=URIRef(EX + "Interaction"))
    assert rows, f"{path.name} returned nothing for the example interaction"


@pytest.mark.parametrize("path", RECONSTRUCTION, ids=lambda p: p.stem)
def test_reconstruction_query_returns_nothing_for_another_interaction(path):
    assert run(path, interaction=URIRef(EX + "NoSuchInteraction")) == []


def test_inventory_report_lists_the_use_case_by_tier_and_owner():
    rows = run(BASE / "queries/inventory-report.rq")
    assert any(str(r.useCase).endswith("UseCaseMeetingPrep") for r in rows)
    sg = run(BASE / "queries/inventory-report.rq", jurisdiction=Literal("SG"))
    xx = run(BASE / "queries/inventory-report.rq", jurisdiction=Literal("XX"))
    assert sg and not xx


def test_superseded_source_query_finds_dependent_claims():
    g = Graph()
    g += G
    g.add((URIRef(EX + "FactRiskProfileV2"), URIRef(PWM + "supersedes"), URIRef(EX + "FactRiskProfile")))
    rows = run(BASE / "queries/affected-by-superseded-source.rq", graph=g)
    assert any(str(r.affected).endswith("ClaimRiskProfile") for r in rows)
