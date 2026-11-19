"""Jurisdiction profiles are data over existing PWM terms, and cover every jurisdictional investor category."""
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import SKOS

BASE = Path(__file__).resolve().parents[1]
PWM = "https://ahmedmuzammil.com/ontology/pwm#"
ONT = Graph().parse(BASE / "pwm.ttl")
FILES = sorted((BASE / "profiles/jurisdictions").glob("*.ttl"))
# The roadmap names SG, HK, CH, the EU and the UK. The US accredited investor has no profile yet.
NO_PROFILE_YET = {"AccreditedInvestorUS"}
GENERIC = {PWM + "RetailCategory", PWM + "NonRetailCategory"}


def test_five_profiles():
    assert [f.stem for f in FILES] == ["ch", "eu", "hk", "sg", "uk"]


def test_profiles_use_only_existing_pwm_terms_and_generic_categories():
    scheme = URIRef(PWM + "InvestorCategoryScheme")
    members = set(ONT.subjects(SKOS.inScheme, scheme))
    for f in FILES:
        g = Graph().parse(f)
        for specific, generic in g.subject_objects(SKOS.broader):
            assert specific in members, f"{f.name}: {specific} is not an investor category"
            assert str(generic) in GENERIC


def test_every_jurisdictional_category_is_covered_exactly_once():
    scheme = URIRef(PWM + "InvestorCategoryScheme")
    jurisdictional = {c for c in ONT.subjects(SKOS.inScheme, scheme)} - {URIRef(p) for p in GENERIC}
    seen = []
    for f in FILES:
        seen += [s for s in Graph().parse(f).subjects(SKOS.broader, None)]
    covered = set(seen)
    assert len(seen) == len(covered), "a category appears in two profiles"
    missing = {str(c)[len(PWM):] for c in jurisdictional - covered}
    assert missing == NO_PROFILE_YET
