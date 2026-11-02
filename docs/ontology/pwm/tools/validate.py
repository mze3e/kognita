"""Validate the PWM example data against the shapes.

The valid example must conform. The negative fixture, merged with the valid
example, must produce exactly the violations listed in EXPECTED: nothing
missing, nothing extra.

It also checks that the evidence records in the example form one unbroken
hash chain, and that a broken chain is detected.

Usage: python3 tools/validate.py [ontology_dir] [-v]
  ontology_dir  folder holding pwm.ttl and pwm-shapes.ttl (default: current directory)
  -v            print every violation message raised by the negative fixture
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF

SH = Namespace("http://www.w3.org/ns/shacl#")
PWM = "https://ahmedmuzammil.com/ontology/pwm#"
PWMSH = "https://ahmedmuzammil.com/ontology/pwm/shapes#"
EX = "https://example.org/pwm-example#"

# (focus node, node shape, path) for each expected violation. Path is None
# for node-level constraints (sh:or, sh:sparql).
EXPECTED = Counter([
    ("BadUseCaseUnapproved", "LiveUseCaseShape", None),
    ("BadUseCaseNoKnowledgeLead", "AIUseCaseShape", None),
    ("BadUseCaseSelfApproved", "ControlFunctionSeparationShape", None),
    ("BadAgentOverreach", "AgentToolLevelShape", None),
    ("BadAgentNoReview", "AIAgentShape", "nextReviewDate"),
    ("BadAgentOwner", "AccountabilityRoleShape", "isPlayedBy"),
    ("BadAgentOwner", "AccountabilityRoleShape", "isPlayedBy"),
    ("BadDelegationOpenEnded", "DelegatedAuthorityShape", None),
    ("BadCommunicationNoDisclosure", "AIDisclosureShape", None),
    ("BadCommunicationNoDecision", "ClientCommunicationShape", "derivedFromDecision"),
    ("BadCommunicationChannel", "ClientCommunicationShape", "viaChannel"),
    ("BadClaimNoSource", "ClaimShape", None),
    ("BadClaimUntyped", "ClaimShape", "claimType"),
    ("BadValuation", "ValuationShape", "calculatedBy"),
    ("BadSuitability", "SuitabilityAssessmentShape", "usesRiskProfile"),
    ("BadSuitability", "SuitabilityAssessmentShape", "rationale"),
    ("BadCodes", "CurrencyCodeShape", "currency"),
    ("BadCodes", "ISINShape", "isin"),
    ("BadCodes", "CFIShape", "cfiCode"),
    ("BadCodes", "LEIShape", "legalEntityIdentifier"),
    ("BadCodes", "CountryCodeShape", "domicileCountry"),
    ("BadAsset", "GovernedInformationAssetShape", "retentionUntil"),
    ("BadAsset", "GovernedInformationAssetShape", None),
    ("BadPolicyDecision", "PolicyDecisionShape", "decidedByRule"),
    ("BadMaterialityNoReliance", "RiskMaterialityAssessmentShape", "reliance"),
    ("BadToolNoVerb", "ToolShape", "actionVerb"),
    ("BadFactNoFreshness", "FactShape", "freshnessThreshold"),
    ("BadRiskProfileNoLevel", "InvestorRiskProfileShape", "riskProfileLevel"),
    ("BadClassificationTwoJurisdictions", "InvestorClassificationShape", "jurisdictionCode"),
    ("BadSectionUnclassified", "DocumentSectionShape", "classification"),
    ("BadRuleNoVersion", "PolicyRuleShape", "versionId"),
    ("BadCrossBorderCode", "CrossBorderRuleShape", "bookingCentreCode"),
    ("BadEvidenceNoHash", "EvidenceRecordShape", "recordHash"),
    ("BadEvidenceUnchained", "EvidenceRecordShape", None),
    ("BadClaimUnverified", "ClaimShape", None),
    ("BadRiskProfileWrongCode", "CodeList_riskProfileLevel", "riskProfileLevel"),
])


def local(term):
    if term is None:
        return None
    s = str(term)
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s


def run(base: Path, data_files):
    ont = Graph().parse(base / "pwm.ttl")
    shapes = Graph().parse(base / "pwm-shapes.ttl")
    data = Graph()
    for f in data_files:
        data.parse(base / f)
    conforms, report, text = validate(
        data, shacl_graph=shapes, ont_graph=ont, inference="rdfs",
        abort_on_first=False, allow_warnings=False, meta_shacl=True)
    if not isinstance(report, Graph):
        sys.exit(f"validation could not run: {text}")
    parent = {ps: ns for ns, ps in shapes.subject_objects(SH.property)}
    nested = set(report.objects(None, SH.detail))
    found = Counter()
    messages = []
    for r in report.subjects(RDF.type, SH.ValidationResult):
        if r in nested:
            continue
        focus = report.value(r, SH.focusNode)
        source = report.value(r, SH.sourceShape)
        node_shape = parent.get(source, source)
        path = report.value(r, SH.resultPath)
        found[(local(focus), local(node_shape), local(path))] += 1
        messages.append(f"  {local(focus)} [{local(node_shape)}] {report.value(r, SH.resultMessage)}")
    return conforms, found, messages


def chain_problems(g):
    """Links between evidence records: each record names the hash of the one
    before it. Only one record may point outside the data (where the excerpt
    starts) unless it is marked chain start; no two records may share a
    predecessor; and links must go forward in time."""
    P = lambda x: URIRef(PWM + x)
    prov_time = URIRef("http://www.w3.org/ns/prov#generatedAtTime")
    records = list(g.subjects(RDF.type, P("EvidenceRecord")))
    by_hash = {str(g.value(r, P("recordHash"))): r for r in records}
    problems, outside, seen = [], [], {}
    for r in records:
        prev = g.value(r, P("previousRecordHash"))
        if prev is None:
            continue
        prev = str(prev)
        if prev in seen:
            problems.append(f"fork: {local(r)} and {local(seen[prev])} share a predecessor")
        seen[prev] = r
        before = by_hash.get(prev)
        if before is None:
            outside.append(r)
        elif g.value(before, prov_time).toPython() > g.value(r, prov_time).toPython():
            problems.append(f"link goes back in time: {local(r)}")
    if len(outside) > 1:
        problems.append("more than one record points outside the data: " + ", ".join(sorted(local(r) for r in outside)))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ontology_dir", nargs="?", default=".")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    base = Path(args.ontology_dir)
    ok = True

    example = Graph().parse(base / "examples/pwm-example.ttl")
    problems = chain_problems(example)
    print(f"example evidence chain unbroken: {not problems}")
    if problems:
        ok = False
        print("\n".join("  " + x for x in problems))
    broken = Graph()
    broken += example
    e2 = URIRef(EX + "Evidence2")
    broken.set((e2, URIRef(PWM + "previousRecordHash"), Literal("sha256:" + "0" * 64)))
    if not chain_problems(broken):
        ok = False
        print("a broken evidence chain was not detected")

    conforms, found, messages = run(base, ["examples/pwm-example.ttl"])
    print(f"valid example conforms: {conforms}")
    if not conforms:
        ok = False
        print("\n".join(messages))

    conforms, found, messages = run(base, ["examples/pwm-example.ttl", "examples/pwm-example-invalid.ttl"])
    missing = EXPECTED - found
    extra = found - EXPECTED
    print(f"negative fixture: {sum(found.values())} violations, {sum(EXPECTED.values())} expected")
    if missing:
        ok = False
        print("MISSING (expected but not raised):")
        for k, n in sorted(missing.items(), key=str):
            print("  ", k, "x", n)
    if extra:
        ok = False
        print("EXTRA (raised but not expected):")
        for k, n in sorted(extra.items(), key=str):
            print("  ", k, "x", n)
    if args.verbose:
        print("\n".join(sorted(messages)))

    shapes = Graph().parse(base / "pwm-shapes.ttl")
    targeted = {local(s) for s in shapes.subjects(RDF.type, SH.NodeShape)
                if any((s, p, None) in shapes for p in (SH.targetClass, SH.targetSubjectsOf))}
    exercised = {k[1] for k in EXPECTED}
    written = {s for s in targeted if not s.startswith("CodeList_")}
    print(f"written shapes: {len(written)}; exercised by the negative fixture: {len(written & exercised)}")
    print(f"generated code-list shapes: {len(targeted - written)}; exercised: {len((targeted - written) & exercised)}")
    print("\nall validation tests passed" if ok else "\nVALIDATION TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
