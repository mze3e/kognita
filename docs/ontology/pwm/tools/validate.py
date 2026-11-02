"""Validate the PWM example data against the shapes.

The valid example must conform. The negative fixture, merged with the valid
example, must produce exactly the violations listed in EXPECTED: nothing
missing, nothing extra.

Usage: python3 tools/validate.py [ontology_dir]   (default: current directory)
"""
import sys
from collections import Counter
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Namespace, URIRef
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


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    ok = True

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
    if "-v" in sys.argv:
        print("\n".join(sorted(messages)))

    shapes = Graph().parse(base / "pwm-shapes.ttl")
    targeted = {local(s) for s in shapes.subjects(RDF.type, SH.NodeShape)
                if any((s, p, None) in shapes for p in (SH.targetClass, SH.targetSubjectsOf))}
    exercised = {k[1] for k in EXPECTED}
    print(f"shapes with targets: {len(targeted)}; exercised by the negative fixture: {len(targeted & exercised)}")
    print("\nall validation tests passed" if ok else "\nVALIDATION TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
