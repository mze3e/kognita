"""Profiles: the AI inventory profile and the worked code-list profile."""
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

BASE = Path(__file__).resolve().parents[1]
PWM = "https://ahmedmuzammil.com/ontology/pwm#"
EX = "https://example.org/pwm-example#"
ONT = Graph().parse(BASE / "pwm.ttl")


def example():
    return Graph().parse(BASE / "examples/pwm-example.ttl")


def run(profile, data, extra_ont=None):
    ont = ONT if extra_ont is None else ONT + extra_ont
    return validate(data, shacl_graph=Graph().parse(BASE / profile), ont_graph=ont, inference="rdfs")


def test_example_register_meets_the_inventory_profile():
    conforms, _, text = run("profiles/ai-inventory-profile.ttl", example())
    assert conforms, text


def test_each_attribute_is_required():
    use_case = URIRef(EX + "UseCaseMeetingPrep")
    agent = URIRef(EX + "Agent")
    pairs = [(use_case, n) for n in ("hasPurpose", "approvedJurisdiction", "modelType", "usesDataClass", "dependsOn",
                                     "lifecycleStatus", "hasMaterialityAssessment", "riskTier", "nextReviewDate", "hasDocumentation")]
    pairs += [(agent, n) for n in ("agentIdentifier", "permittedTool", "hasComponent", "hasBlastRadiusLimit", "registeredForUseCase")]
    g = example()
    for subject, name in pairs:
        g.remove((subject, URIRef(PWM + name), None))
    conforms, _, text = run("profiles/ai-inventory-profile.ttl", g)
    assert not conforms
    for phrase in ("purpose", "approved scope", "model type", "data used", "dependencies", "lifecycle status",
                   "materiality assessment", "risk tier", "review status", "documentation", "identifier",
                   "tools and systems", "components", "guardrails", "the use case it is registered for"):
        assert phrase in text, f"removing the attribute for {phrase!r} was not caught"


def test_use_case_needs_both_owner_roles():
    g = example()
    for role in list(g.objects(URIRef(EX + "UseCaseMeetingPrep"), URIRef(PWM + "hasAccountabilityRole"))):
        if (role, RDF.type, URIRef(PWM + "TechnologyOwner")) in g:
            g.remove((URIRef(EX + "UseCaseMeetingPrep"), URIRef(PWM + "hasAccountabilityRole"), role))
    assert not run("profiles/ai-inventory-profile.ttl", g)[0]


FIRM = "https://example.org/firm/ontology#"


def _risk_data(level):
    """The example, with every risk-scale value replaced by the given code."""
    g = example()
    for name in ("riskProfileLevel", "productRiskLevel", "targetRiskProfile"):
        predicate = URIRef(PWM + name)
        for subject, old in list(g.subject_objects(predicate)):
            g.remove((subject, predicate, old))
            g.add((subject, predicate, URIRef(level)))
    return g


def _shapes(with_profile):
    shapes = Graph().parse(BASE / "pwm-shapes.ttl")
    if with_profile:
        shapes.parse(BASE / "profiles/code-list-profile-example.shapes.ttl")
    return shapes


def test_firm_code_is_refused_without_the_profile_and_accepted_with_it():
    ont = ONT + Graph().parse(BASE / "profiles/code-list-profile-example.ttl")
    data = _risk_data(FIRM + "Balanced")
    assert not validate(data, shacl_graph=_shapes(False), ont_graph=ont, inference="rdfs")[0]
    conforms, _, text = validate(data, shacl_graph=_shapes(True), ont_graph=ont, inference="rdfs")
    assert conforms, text


def test_profile_enforces_the_firm_codes_and_refuses_the_pwm_default():
    ont = ONT + Graph().parse(BASE / "profiles/code-list-profile-example.ttl")
    assert not validate(_risk_data(FIRM + "Aggressive"), shacl_graph=_shapes(True), ont_graph=ont, inference="rdfs")[0]
    assert not validate(_risk_data(PWM + "RP3"), shacl_graph=_shapes(True), ont_graph=ont, inference="rdfs")[0]


def test_profile_rebinds_the_three_properties():
    from rdflib import Graph as G
    profile = G().parse(BASE / "profiles/code-list-profile-example.ttl")
    for name in ("riskProfileLevel", "productRiskLevel", "targetRiskProfile"):
        assert (URIRef(PWM + name), URIRef(PWM + "valuesFrom"), URIRef(FIRM + "RiskScale")) in profile
