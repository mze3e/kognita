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

from owlrl import DeductiveClosure, RDFS_Semantics
from pyshacl import validate
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF

SH = Namespace("http://www.w3.org/ns/shacl#")
PWM = "https://ahmedmuzammil.com/ontology/pwm#"
PWMSH = "https://ahmedmuzammil.com/ontology/pwm/shapes#"
EX = "https://example.org/pwm-example#"

# (focus node, node shape, path) for each expected violation. Path is None
# for node-level constraints (sh:or, sh:sparql).
# 0.3 fixtures, followed by inherited 0.2 fixtures.
EXPECTED = Counter([
    ('BadPinnedDecision', 'PinnedDecisionShape', 'hasCheck'),
    ('BadSuspendedApproval', 'SuspendedApprovalShape', None),
    ('BadRun', 'RunShape', 'hasBudget'),
    ('BadBudget', 'RunBudgetShape', 'classificationCeiling'),
    ('BadBudgetDenial', 'BudgetExceededDenialShape', 'exceededBudget'),
    ('BadCheckpoint', 'CheckpointShape', 'contentHash'),
    ('BadApprovalRequest', 'ApprovalRequestShape', 'atCheckpoint'),
    ('BadResolution', 'ApprovalResolutionShape', 'wasAttributedTo'),
    ('BadResumption', 'RunResumptionShape', 'resolvedApproval'),
    ('BadModelCall', 'ModelCallShape', 'responseHash'),
    ('BadToolCall', 'ToolCallShape', 'responseHash'),
    ('BadMCP', 'MCPProxiedCallShape', 'mcpBackend'),
    ('BadClassifier', 'ClassifierOutputShape', 'classifierVersion'),
    ('BadAttribute', 'EnvelopeAttributeShape', None),
    ('BadLowConfidence', 'ClassifierThresholdShape', None),
    ('BadCheck', 'PolicyCheckShape', 'policyContentHash'),
    ('BadRetrievedItem', 'RetrievedItemShape', 'embeddingModel'),
    ('BadRetentionEntry', 'RetentionEntryShape', 'contentHash'),
    ('BadRetentionPolicy', 'RetentionPolicyShape', 'retentionDuration'),
    ('BadErasure', 'ErasureShape', None),
    ('BadReport', 'ReconstructionReportShape', None),
    ('BadCoverage', 'QuestionCoverageShape', 'coverageStatus'),
    ('BadFinding', 'ReconstructionFindingShape', None),
    ('BadGatewayConfig', 'GatewayConfigurationShape', 'gatewayFailureMode'),
    ('BadPendingApprovalCall', 'GovernedCallShape', None),
    ('BadNoRequestApprovalCall', 'GovernedCallShape', None),
    ('BadCrossRunApprovalCall', 'GovernedCallShape', None),
    ('BadMultiRunApprovalCall', 'GovernedCallShape', None),
    ('BadDeniedApprovalCall', 'GovernedCallShape', None),
    ('BadCallIdentity', 'GovernedCallShape', None),
    ('BadMultipleCallAgents', 'GovernedCallShape', 'callAgent'),
    ('BadMultipleSystemTriggers', 'GovernedCallShape', 'systemTrigger'),

    ("BadEUSuitability", "EUSustainabilityAssessmentShape", None),
    ("BadTaxonomyPreference", "SustainabilityPreferenceShape", None),
    ("BadSFDRProportion", "SustainabilityPreferenceShape", "minimumSFDRSustainableProportion"),
    ("BadPAIPreference", "SustainabilityPreferenceShape", None),
    ("BadNoPreference", "SustainabilityPreferenceShape", None),
    ("BadPreferenceCode", "CodeList_sustainabilityPreferenceKind", "sustainabilityPreferenceKind"),
    ("BadUseCaseUnapproved", "LiveUseCaseShape", None),
    ("BadUseCaseNoKnowledgeLead", "AIUseCaseShape", None),
    ("BadUseCaseSelfApproved", "ControlFunctionSeparationShape", None),
    ("BadAgentOverreach", "AgentToolLevelShape", None),
    ("BadAgentNoReview", "AIAgentShape", "nextReviewDate"),
    ("BadAgentOwner", "AccountabilityRoleShape", "isPlayedBy"),
    ("BadAgentOwner", "AccountabilityRoleShape", "isPlayedBy"),
    ("BadDelegationOpenEnded", "DelegatedAuthorityShape", "validUntil"),
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

EXPECTED.update([('Bad04IngestedDocumentShape', 'IngestedDocumentShape', 'parserVersion'),
 ('Bad04DocumentElementShape', 'DocumentElementShape', 'sectionPath'),
 ('Bad04DeclarativePolicyShape', 'DeclarativePolicyShape', 'policySerialization'),
 ('Bad04PolicyTestCaseShape', 'PolicyTestCaseShape', 'expectedOutcome'),
 ('Bad04PolicyPackShape', 'PolicyPackShape', 'versionId'),
 ('Bad04RMReviewEditShape', 'RMReviewEditShape', 'afterHash'),
 ('Bad04ReviewQueueEntryShape', 'ReviewQueueEntryShape', 'queueReason'),
 ('Bad04ReviewCoverageReportShape', 'ReviewCoverageReportShape', 'reviewedCount'),
 ('Bad04CircuitBreakerTripShape', 'CircuitBreakerTripShape', 'flagThreshold'),
 ('Bad04AffectedClientQueryShape', 'AffectedClientQueryShape', 'queryResult'),
 ('Bad04InteractionOutcomeShape', 'InteractionOutcomeShape', 'outcomeKind'),
 ('Bad04BreakerScopeShape', 'BreakerScopeShape', None),
 ('Bad04TwoSignatureShape', 'TwoSignatureShape', None),
 ('Bad04PastReviewDecisionShape', 'PastReviewDecisionShape', None),
 ('Bad04RiskAppetiteDecisionShape', 'RiskAppetiteDecisionShape', None),
 ('Bad04CoverageCountsShape', 'CoverageCountsShape', None)])
EXPECTED.update([('Bad04CircuitBreakerConfigurationShape', 'CircuitBreakerConfigurationShape', 'flagThreshold'),
 ('Bad04FlightRecorderSessionShape', 'FlightRecorderSessionShape', 'contentHash'),
 ('Bad04ReplayShape', 'ReplayShape', 'replayReport'),
 ('Bad04PolicyChangeShape', 'PolicyChangeShape', 'changeStage'),
 ('Bad04NearMissClassificationShape', 'NearMissClassificationShape', 'nearMissKind'),
 ('Bad04ReviewSamplingRateShape', 'ReviewSamplingRateShape', 'lowRiskSamplingRate'),
 ('Bad04AffectedClientResultShape', 'AffectedClientResultShape', 'resultCount'),
 ('Bad04AffectedClientCountShape', 'AffectedClientCountShape', None),
 ('Bad04ModelCallCardShape', 'ModelCallCardShape', None)])
EXPECTED.update([("BadDiffLocator", "RMReviewEditShape", "diffLocator"),
                 ("BadMissingModel", "ModelCallShape", "usesModel"),
                 ("BadAffectedRecipient", "AffectedClientCountShape", None)])
EXPECTED.update([("OutsideAppetite", "LiveRiskAppetiteShape", None)])

# 0.5: explicit negative fixtures for every new written constraint.
EXPECTED.update([('Bad05ActionStopShape', 'ActionStopShape', 'stopOutcome'),
 ('Bad05AdversarialTestShape', 'AdversarialTestShape', 'threatModel'),
 ('Bad05AgentDecisionShape', 'AgentDecisionShape', None),
 ('Bad05AgentFleetShape', 'AgentFleetShape', 'fleetIsolation'),
 ('Bad05AgentIdentityShape', 'AgentIdentityShape', 'agentOwner'),
 ('Bad05AuthenticatedInstructionShape', 'AuthenticatedInstructionShape', 'instructionPrincipal'),
 ('Bad05AuthorityRequestScopeShape', 'AuthorityRequestScopeShape', None),
 ('Bad05AuthorityRevocationShape', 'AuthorityRevocationShape', 'revokedAt'),
 ('Bad05AuthorityScopeShape', 'AuthorityScopeShape', 'scopeField'),
 ('Bad05AutonomyCallShape', 'AuthorityRequestScopeShape', None),
 ('Bad05AutonomyCallShape', 'AutonomyCallShape', None),
 ('Bad05AutonomyCallShape', 'ToolAllowListShape', None),
 ('Bad05AutonomyEvidenceShape', 'AutonomyEvidenceShape', None),
 ('Bad05AutonomyTransitionShape', 'AutonomyEvidenceShape', None),
 ('Bad05AutonomyTransitionShape', 'AutonomyTransitionShape', 'sampleCount'),
 ('Bad05BehaviourAnomalyShape', 'BehaviourAnomalyShape', 'anomalyIncident'),
 ('Bad05BehaviourBaselineShape', 'BehaviourBaselineShape', 'baselineMetric'),
 ('Bad05BlastRadiusDecisionShape', 'BlastRadiusDecisionShape', None),
 ('Bad05BlastRadiusLimitShape', 'BlastRadiusLimitShape', 'limitUnit'),
 ('Bad05BlastRadiusObservationShape', 'BlastRadiusObservationShape', 'projectedConsumption'),
 ('Bad05BoundedToolShape', 'BoundedToolShape', None),
 ('Bad05CapabilityGrantShape', 'CapabilityGrantShape', 'validUntil'),
 ('Bad05ContentIsolationShape', 'ContentIsolationShape', None),
 ('Bad05DelegationScopeShape', 'DelegationScopeShape', None),
 ('Bad05DeploymentEnforcementShape', 'DeploymentEnforcementShape', None),
 ('Bad05EnforcementAttestationShape', 'EnforcementAttestationShape', 'egressRestricted'),
 ('Bad05ExternalContentShape', 'ExternalContentShape', 'trustLabel'),
 ('Bad05ExternalContentShape', 'ExternalContentShape', 'trustLabel'),
 ('Bad05FieldMinimizationShape', 'AuthorityRequestScopeShape', None),
 ('Bad05FieldMinimizationShape', 'FieldMinimizationShape', None),
 ('Bad05FleetHaltShape', 'FleetHaltShape', 'haltScope'),
 ('Bad05HumanEntitlementShape', 'HumanEntitlementShape', 'validUntil'),
 ('Bad05InstructionAssuranceShape', 'InstructionAssuranceShape', None),
 ('Bad05IrreversibleApprovalShape', 'FleetHaltDecisionShape', None),
 ('Bad05IrreversibleApprovalShape', 'IrreversibleApprovalShape', None),
 ('Bad05LiveAuthorityDecisionShape', 'LiveAuthorityDecisionShape', None),
 ('Bad05LiveGrantShape', 'LiveGrantShape', None),
 ('Bad05PromotionGateShape', 'PromotionGateShape', 'minimumSample'),
 ('Bad05ReversibilityShape', 'ReversibilityShape', None),
 ('Bad05SandboxConstraintsShape', 'SandboxConstraintsShape', 'cpuSeconds'),
 ('Bad05ScopeNarrowingLimitsShape', 'ScopeNarrowingLimitsShape', None),
 ('Bad05SubDelegationShape', 'SubDelegationShape', None),
 ('Bad05ThreatModelShape', 'ThreatModelShape', 'threatModelReference'),
 ('Bad05ToolAllowListShape', 'AuthorityRequestScopeShape', None),
 ('Bad05ToolAllowListShape', 'ToolAllowListShape', None),
 ('Bad05ToolDurationShape', 'ToolDurationShape', None),
 ('Bad05WorkloadIdentityShape', 'WorkloadIdentityShape', 'credentialIssuer'),
 ('BadMultipleCallAgents', 'AgentDecisionShape', None),
 ('BadMultipleCallAgents', 'LiveAuthorityDecisionShape', None),
 ('SharedAgentA', 'ExclusiveWorkloadShape', None),
 ('SharedAgentB', 'ExclusiveWorkloadShape', None),
 ('DuplicateCredentialA', 'WorkloadIdentityShape', None),
 ('DuplicateCredentialB', 'WorkloadIdentityShape', None),
 ('Bad05IrreversibleMaxOnly', 'IrreversibleApprovalShape', None),
 ('Bad05ScopeRank', 'AuthorityRequestScopeShape', None),
 ('Bad05MissingClassification', 'AuthorityRequestScopeShape', None),
 ('Bad05MissingJurisdiction', 'AuthorityRequestScopeShape', None),
 ('Bad05MultipleClassifications', 'AuthorityRequestScopeShape', None),
 ('Bad05MultipleJurisdictions', 'AuthorityRequestScopeShape', None),
 ('Bad05UncoveredVerb', 'AuthorityRequestScopeShape', None),
 ('Bad05WrongApprovalDecision', 'IrreversibleApprovalShape', None),
 ('Bad05LiveEntitlementCoverage', 'LiveAuthorityDecisionShape', None),
])

EXPECTED.update([('BadLiveStatus', 'AuthorityLiveStatusShape', None),
 ('BadLimitApproval', 'RiskControlApprovalShape', 'riskControlApproval'),
 ('BadGateApproval', 'RiskControlApprovalShape', 'riskControlApproval'),
 ('BadApprover', 'RiskControlApprovalRecordShape', 'approverRole'),
 ('BadApprovalTime', 'RiskControlApprovalRecordShape', 'approvedAt')])

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


def chain_problems(g, ontology=None):
    """Links between evidence records: each record names the hash of the one
    before it. One root is allowed: a genesis record or an external predecessor
    at the start of an excerpt. All records must form one acyclic chain, with
    no shared predecessors and links that go forward in time."""
    P = lambda x: URIRef(PWM + x)
    prov_time = URIRef("http://www.w3.org/ns/prov#generatedAtTime")
    inferred = Graph()
    if ontology is None:
        ontology = Graph().parse(Path(__file__).resolve().parents[1] / "pwm.ttl")
    inferred += ontology
    inferred += g
    DeductiveClosure(RDFS_Semantics).expand(inferred)
    records = list(inferred.subjects(RDF.type, P("EvidenceRecord")))
    problems, outside, seen = [], [], {}
    by_hash = {}
    for r in records:
        for record_hash in inferred.objects(r, P("recordHash")):
            key = str(record_hash)
            if key in by_hash:
                problems.append(f"duplicate recordHash: {key}")
            else:
                by_hash[key] = r
    roots, successors = [], {}
    for r in records:
        prev = inferred.value(r, P("previousRecordHash"))
        if prev is None:
            roots.append(r)
            continue
        prev = str(prev)
        if prev in seen:
            problems.append(f"fork: {local(r)} and {local(seen[prev])} share a predecessor")
        seen[prev] = r
        before = by_hash.get(prev)
        if before is None:
            outside.append(r)
            roots.append(r)
        else:
            successors[before] = r
            earlier = inferred.value(before, prov_time)
            later = inferred.value(r, prov_time)
            if earlier is not None and later is not None:
                try:
                    if earlier.toPython() > later.toPython():
                        problems.append(f"link goes back in time: {local(r)}")
                except TypeError:
                    problems.append(f"indeterminate time ordering: {local(r)}")
    if records:
        if len(roots) != 1:
            problems.append("evidence records must have exactly one chain root")
        visited = set()
        current = roots[0] if roots else None
        while current is not None and current not in visited:
            visited.add(current)
            current = successors.get(current)
        if visited != set(records) or current is not None:
            problems.append("evidence records do not form one connected acyclic chain")
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
    ontology = Graph().parse(base / "pwm.ttl")
    problems = chain_problems(example, ontology)
    print(f"example evidence chain unbroken: {not problems}")
    if problems:
        ok = False
        print("\n".join("  " + x for x in problems))
    broken = Graph()
    broken += example
    e2 = URIRef(EX + "Evidence2")
    broken.set((e2, URIRef(PWM + "previousRecordHash"), Literal("sha256:" + "0" * 64)))
    if not chain_problems(broken, ontology):
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
