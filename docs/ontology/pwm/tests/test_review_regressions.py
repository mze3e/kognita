import importlib.util
from pathlib import Path

import pytest

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

BASE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load('validate')
builder = load('build')
P = lambda name: URIRef(validator.PWM + name)
E = lambda name: URIRef(validator.EX + name)


def example():
    return Graph().parse(BASE / 'examples/pwm-example.ttl')


def test_inferred_fork():
    g = example()
    for predicate, value in list(g.predicate_objects(E('Evidence3'))):
        if predicate != RDF.type:
            g.add((E('UntypedFork'), predicate, value))
    assert any('fork:' in p for p in validator.chain_problems(g))


def test_second_root():
    g = example()
    g.add((E('ExtraRoot'), RDF.type, P('EvidenceRecord')))
    g.add((E('ExtraRoot'), P('chainStart'), Literal(True)))
    g.add((E('ExtraRoot'), P('recordHash'), Literal('sha256:' + 'f' * 64)))
    assert any('root' in p for p in validator.chain_problems(g))


def test_mixed_timezones():
    g = example()
    time = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    g.set((E('Evidence2'), time, Literal('2026-11-02T12:00:00', datatype=XSD.dateTime)))
    assert any('indeterminate' in p for p in validator.chain_problems(g))


def test_escaped_labels(tmp_path):
    g = Graph().parse(BASE / 'pwm.ttl')
    from rdflib.namespace import RDFS, SKOS
    g.set((P('riskProfileLevel'), RDFS.label, Literal('a "quote" and \\slash', lang='en')))
    g.set((P('IllustrativeRiskScale'), SKOS.prefLabel, Literal('a "quote" and \\slash', lang='en')))
    shapes = tmp_path / 'shapes.ttl'
    shapes.write_text((BASE / 'pwm-shapes.ttl').read_text())
    builder.write_code_list_shapes(g, shapes)
    Graph().parse(shapes)


def test_firm_shapes():
    from pyshacl import validate
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    shapes.parse(data=builder.EXTENSION_SHAPES_TTL, format='turtle')
    ontology = Graph().parse(BASE / 'pwm.ttl')
    ontology.parse(data=builder.EXTENSION_TTL, format='turtle')
    data = Graph()
    data.add((E('FirmRisk'), P('validFrom'), Literal('2026-11-02T00:00:00Z', datatype=XSD.dateTime)))
    data.add((E('FirmRisk'), URIRef('http://www.w3.org/ns/prov#wasAttributedTo'), E('Author')))
    for prop in ('riskProfileLevel', 'productRiskLevel', 'targetRiskProfile'):
        data.add((E('FirmRisk'), P(prop), URIRef('https://example.org/firm/ontology#R1')))
    assert validate(data, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]
    data.add((E('FirmRisk'), P('productRiskLevel'), P('RP1')))
    assert not validate(data, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]


def inferred_chain():
    from rdflib.namespace import RDFS
    g = example()
    ontology = Graph().parse(BASE / 'pwm.ttl')
    for predicate in (P('recordHash'), P('previousRecordHash'),
                      URIRef('http://www.w3.org/ns/prov#generatedAtTime')):
        subproperty = E(str(predicate).rsplit('#', 1)[-1])
        ontology.add((subproperty, RDFS.subPropertyOf, predicate))
        for subject, value in list(g.subject_objects(predicate)):
            g.remove((subject, predicate, value))
            g.add((subject, subproperty, value))
    return g, ontology


def test_inferred_chain_values():
    g, ontology = inferred_chain()
    assert validator.chain_problems(g, ontology) == []
    g.set((E('Evidence2'), E('generatedAtTime'),
           Literal('2000-01-01T00:00:00Z', datatype=XSD.dateTime)))
    assert any('back in time' in p for p in validator.chain_problems(g, ontology))


def test_duplicate_inferred_hash():
    g, ontology = inferred_chain()
    g.set((E('Evidence2'), E('recordHash'), g.value(E('Evidence1'), E('recordHash'))))
    assert any('duplicate recordHash:' in p for p in validator.chain_problems(g, ontology))


def test_version_and_counts(tmp_path):
    from rdflib.namespace import OWL, SKOS
    g = Graph().parse(BASE / 'pwm.ttl')
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    site = 'https://ahmedmuzammil.com/ontology/pwm'
    assert g.value(URIRef(site), OWL.versionIRI) == URIRef(site + '/0.5')
    assert shapes.value(URIRef(site + '/shapes'), OWL.versionIRI) == URIRef(site + '/0.5/pwm-shapes.ttl')
    assert str(g.value(URIRef(site), OWL.versionInfo)) == '0.5 (draft)'
    assert g.value(URIRef(site), OWL.priorVersion) == URIRef(site + '/0.4')
    assert shapes.value(URIRef(site + '/shapes'), OWL.priorVersion) == URIRef(site + '/0.4/pwm-shapes.ttl')
    assert len(set(g.subjects(RDF.type, SKOS.Concept))) == 235
    assert len(set(shapes.subjects(RDF.type, validator.SH.NodeShape))) == 197
    assert len(set(example().subjects(RDF.type, None))) == 191
    import shutil
    shutil.copytree(BASE / 'src', tmp_path / 'src')
    shutil.copy(BASE / 'pwm-shapes.ttl', tmp_path / 'pwm-shapes.ttl')
    shutil.copytree(BASE / 'examples', tmp_path / 'examples')
    shutil.copy(BASE / 'GAP_ANALYSIS.md', tmp_path / 'GAP_ANALYSIS.md')
    counts = builder.build(tmp_path, None)['counts']
    assert (counts['classes'], counts['objectProperties'], counts['datatypeProperties'],
            counts['codeLists'], counts['rules'], counts['codeListRules']) == (193, 220, 188, 48, 130, 66)
    assert sum(validator.EXPECTED.values()) == 193
    assert len(set(Graph().parse(BASE / 'examples/pwm-example-invalid.ttl').subjects(RDF.type, None))) == 259
    assert f"yielding {sum(validator.EXPECTED.values())} expected violations" in (tmp_path / "SPEC.md").read_text()
    first_spec = (tmp_path / 'SPEC.md').read_text()
    status_row = next(row for row in first_spec.splitlines() if '<a id="statusOf"></a>' in row)
    assert '[delegated authority](#DelegatedAuthority), [kill switch](#KillSwitch)' in status_row
    governance = tmp_path / 'src/06-governance.ttl'
    governance.write_text(governance.read_text().replace(
        'owl:unionOf ( pwm:DelegatedAuthority pwm:KillSwitch )',
        'owl:unionOf ( pwm:KillSwitch pwm:DelegatedAuthority )'))
    builder.build(tmp_path, None)
    assert (tmp_path / 'SPEC.md').read_text() == first_spec
    negative_count = len(set(Graph().parse(BASE / 'examples/pwm-example-invalid.ttl').subjects(RDF.type, None)))
    assert f'negative fixture contains {negative_count} typed records' in (tmp_path / 'SPEC.md').read_text()


def test_every_generated_code_list_rule():
    """Each generated rule accepts all its codes and rejects a foreign IRI.

    Use only generated shapes here, so unrelated required fields do not mask
    a missing or incorrectly generated membership constraint.
    """
    from pyshacl import validate
    from rdflib.namespace import SKOS
    from rdflib.collection import Collection
    ontology = Graph().parse(BASE / 'pwm.ttl')
    all_shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    shapes = Graph()
    good, bad = Graph(), Graph()
    expected = set()
    for prop, scheme in ontology.subject_objects(P('valuesFrom')):
        shape = URIRef(validator.PWMSH + 'CodeList_' + builder.local(prop))
        ps = all_shapes.value(shape, validator.SH.property)
        assert ps is not None
        codes = set(ontology.subjects(SKOS.inScheme, scheme))
        assert set(Collection(all_shapes, all_shapes.value(ps, validator.SH['in']))) == codes
        # Include the RDF list closure as well as shape and property triples.
        pending, seen = [shape], set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            for triple in all_shapes.triples((node, None, None)):
                shapes.add(triple)
                from rdflib import BNode
                if isinstance(triple[2], BNode):
                    pending.append(triple[2])
        for code in codes:
            good.add((E(builder.local(prop)), prop, code))
        focus = E('Invalid_' + builder.local(prop))
        bad.add((focus, prop, E('UnknownCode')))
        expected.add((focus, prop))
    assert validate(good, shacl_graph=shapes)[0]
    conforms, report, _ = validate(bad, shacl_graph=shapes)
    assert not conforms
    actual = {(report.value(r, validator.SH.focusNode), report.value(r, validator.SH.resultPath))
              for r in report.subjects(RDF.type, validator.SH.ValidationResult)}
    assert actual == expected
    assert len(actual) == 66


def test_sustainability_preference_edges():
    from pyshacl import validate
    ontology = Graph().parse(BASE / 'pwm.ttl')
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')

    def check(kinds, tax=None, sfdr=None, pai=None):
        g = Graph()
        node = E('Preference')
        g.add((node, RDF.type, P('SustainabilityPreference')))
        for kind in kinds:
            g.add((node, P('sustainabilityPreferenceKind'), P(kind)))
        for prop, value in [('minimumTaxonomyAlignedProportion', tax),
                            ('minimumSFDRSustainableProportion', sfdr),
                            ('paiConsiderationElements', pai)]:
            if value is not None:
                g.add((node, P(prop), value))
        return validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]

    decimal = lambda value: Literal(value, datatype=XSD.decimal)
    assert check(['NoSustainabilityPreference'])
    assert check(['TaxonomyAlignedMinimum'], tax=decimal('0'))
    assert check(['SFDRSustainableMinimum'], sfdr=decimal('1'))
    assert check(['TaxonomyAlignedMinimum', 'SFDRSustainableMinimum', 'PAIConsideration'],
                 tax=decimal('.2'), sfdr=decimal('.4'), pai=Literal('Emissions intensity'))
    assert not check([])
    assert not check(['TaxonomyAlignedMinimum'])
    assert not check(['SFDRSustainableMinimum'])
    assert not check(['PAIConsideration'], pai=Literal(''))
    assert not check(['TaxonomyAlignedMinimum'], tax=decimal('-.1'))
    assert not check(['SFDRSustainableMinimum'], sfdr=decimal('1.1'))
    assert not check(['TaxonomyAlignedMinimum'], tax=Literal('0.2'))
    assert not check(['NoSustainabilityPreference'], tax=decimal('.2'))
    assert not check(['NoSustainabilityPreference', 'PAIConsideration'], pai=Literal('Emissions'))


def test_eu_sustainability_requirement():
    from pyshacl import validate
    ontology = Graph().parse(BASE / 'pwm.ttl')
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    g = example()
    # Existing SG assessment is valid without imposing the EU rule globally.
    assert validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]
    g.remove((E('EUSuitability'), P('usesSustainabilityPreference'), None))
    conforms, report, _ = validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')
    assert not conforms
    assert set(report.objects(None, validator.SH.focusNode)) == {E('EUSuitability')}
    # An explicit no-preference record satisfies collection of the choice.
    g.add((E("NoSustainabilityChoice"), RDF.type, P("SustainabilityPreference")))
    g.add((E("NoSustainabilityChoice"), P("sustainabilityPreferenceKind"),
           P("NoSustainabilityPreference")))
    g.add((E('EUSuitability'), P('usesSustainabilityPreference'), E('NoSustainabilityChoice')))
    assert validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]
    # Two versions at once must not satisfy the one-version requirement.
    g.add((E('EUSuitability'), P('usesSustainabilityPreference'), E('SustainabilityChoice')))
    assert not validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]


def test_eu_example_is_self_contained():
    from pyshacl import validate
    g = example()
    eu = Graph()
    pending, seen = [E('EUSuitability')], set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        assert str(node).startswith(validator.EX + 'EU') or node == E('SustainabilityChoice')
        assert (node, RDF.type, None) in g
        for triple in g.triples((node, None, None)):
            eu.add(triple)
            if str(triple[2]).startswith(validator.EX):
                pending.append(triple[2])
    assert validate(eu, shacl_graph=Graph().parse(BASE / 'pwm-shapes.ttl'),
                    ont_graph=Graph().parse(BASE / 'pwm.ttl'), inference='rdfs')[0]
    assert not list(g.triples((E('NoSustainabilityChoice'), None, None)))


def test_gateway_rule_negative_coverage():
    """The merged fixture must raise precisely the declared failures for all rules."""
    conforms, found, _ = validator.run(BASE, ['examples/pwm-example.ttl',
                                            'examples/pwm-example-invalid.ttl'])
    assert not conforms
    assert found == validator.EXPECTED
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    written = {validator.local(s) for s in shapes.subjects(RDF.type, validator.SH.NodeShape)
               if not validator.local(s).startswith('CodeList_') and
               any((s, p, None) in shapes for p in (validator.SH.targetClass,
                                                  validator.SH.targetSubjectsOf))}
    assert written == {name for _, name, _ in validator.EXPECTED
                       if not name.startswith('CodeList_')}


def test_gateway_linked_classifier_and_resolution():
    """Call-only classifier evidence and a wrong approval checkpoint cannot bypass rules."""
    from pyshacl import validate
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    ontology = Graph().parse(BASE / 'pwm.ttl')
    g = example()
    # Move the classifier attribute off the decision: the linked gateway call still carries it.
    g.remove((E('PolicyDecisionModelAllow'), P('hasAttribute'), None))
    g.set((E('PolicyDecisionModelAllow'), P('decisionOutcome'), P('OutcomeAllow')))
    g.set((E('PolicyDecisionModelAllow'), P('escalationThreshold'), Literal('.99', datatype=XSD.decimal)))
    # Retain a valid but unrelated resolution request to test checkpoint identity.
    g.add((E('UnrelatedRequest'), RDF.type, P('ApprovalRequest')))
    g.add((E('UnrelatedRequest'), P('requestsDecision'), E('PolicyDecisionRelease')))
    g.add((E('UnrelatedRequest'), P('atCheckpoint'), E('OtherCheckpoint')))
    g.add((E('OtherCheckpoint'), RDF.type, P('RunCheckpoint')))
    g.add((E('OtherCheckpoint'), P('inRun'), E('Run')))
    g.add((E('OtherCheckpoint'), P('contentHash'), Literal('sha256:' + 'c' * 64)))
    g.set((E('Resolution'), P('resolvesApproval'), E('UnrelatedRequest')))
    conforms, report, _ = validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')
    assert not conforms
    failures = {(validator.local(report.value(r, validator.SH.focusNode)),
                 validator.local(report.value(r, validator.SH.sourceShape)))
                for r in report.subjects(RDF.type, validator.SH.ValidationResult)}
    assert ('GatewayCall', 'ClassifierThresholdShape') in failures
    assert ('Resumption', 'RunResumptionShape') in failures


def test_classifier_identity_and_typed_precedence():
    from pyshacl import validate
    g = example()
    g.add((E('TypedClassification'), RDF.type, P('EnvelopeAttribute')))
    g.add((E('TypedClassification'), P('attributeSource'), P('AttributeTyped')))
    g.add((E('TypedClassification'), P('attributeName'), Literal('classification')))
    g.add((E('TypedClassification'), P('attributeValue'), Literal('C3')))
    g.add((E('GatewayCall'), P('hasAttribute'), E('TypedClassification')))
    g.add((E('DerivedPrincipal'), RDF.type, P('EnvelopeAttribute')))
    for prop, value in [('attributeSource', P('AttributeClassifierDerived')),
                        ('attributeName', Literal('principal')),
                        ('attributeValue', Literal('forged-admin')),
                        ('derivedFromClassifier', E('Classifier'))]:
        g.add((E('DerivedPrincipal'), P(prop), value))
    conforms, report, _ = validate(g, shacl_graph=Graph().parse(BASE / 'pwm-shapes.ttl'),
                                 ont_graph=Graph().parse(BASE / 'pwm.ttl'), inference='rdfs')
    assert not conforms
    failures = {(validator.local(report.value(r, validator.SH.focusNode)),
                 validator.local(report.value(r, validator.SH.sourceShape)))
                for r in report.subjects(RDF.type, validator.SH.ValidationResult)}
    assert ('GatewayCall', 'ClassifierThresholdShape') in failures
    assert ('DerivedPrincipal', 'EnvelopeAttributeShape') in failures


def test_ingestion_edit_and_scope_required_fields():
    """Removing each audit-critical value is observable, including the reused classification."""
    from pyshacl import validate
    g = example()
    cases = [('Passage', 'pageNumber'), ('Passage', 'classification'),
             ('Passage', 'contentHash'), ('Edit', 'beforeHash'),
             ('Edit', 'afterHash'), ('Edit', 'editDiffHash'), ('Edit', 'diffLocator'),
             ('Trip', 'scopeModel'), ('Trip', 'flagThreshold'),
             ('PolicyScenario', 'expectedOutcome')]
    for i, (source, missing) in enumerate(cases):
        clone = E('Missing04_' + str(i))
        for predicate, value in list(g.predicate_objects(E(source))):
            if predicate != P(missing):
                g.add((clone, predicate, value))
    conforms, report, _ = validate(g, shacl_graph=Graph().parse(BASE / 'pwm-shapes.ttl'),
                                 ont_graph=Graph().parse(BASE / 'pwm.ttl'), inference='rdfs')
    assert not conforms
    focuses = set(report.objects(None, validator.SH.focusNode))
    assert focuses == {E('Missing04_' + str(i)) for i in range(len(cases))}
    paths = {(report.value(r, validator.SH.focusNode), report.value(r, validator.SH.resultPath))
             for r in report.subjects(RDF.type, validator.SH.ValidationResult)}
    for i, (_, missing) in enumerate(cases):
        assert (E('Missing04_' + str(i)), None if missing == 'scopeModel' else P(missing)) in paths


def test_historical_review_boundary_and_distinct_signatures():
    """Same-day review is valid; the following day requires DENY; repeated signer is one human."""
    from pyshacl import validate
    g = example()
    for predicate, value in list(g.predicate_objects(E('PolicyDecisionModelAllow'))):
        g.add((E('HistoricalReviewDecision'), predicate, value))
    due = g.value(E('UseCaseMeetingPrep'), P('nextReviewDate'))
    at = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    g.set((E('HistoricalReviewDecision'), at, Literal(str(due) + 'T23:59:59Z', datatype=XSD.dateTime)))
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    ontology = Graph().parse(BASE / 'pwm.ttl')
    assert validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]
    from datetime import date, timedelta
    later = date.fromisoformat(str(due)) + timedelta(days=1)
    g.set((E('HistoricalReviewDecision'), at, Literal(str(later) + 'T00:00:00Z', datatype=XSD.dateTime)))
    g.remove((E('Communication'), P('communicationSigner'), E('PersonModelRisk')))
    g.add((E('Communication'), P('communicationSigner'), E('PersonRM')))
    conforms, report, _ = validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')
    assert not conforms
    failures = {(validator.local(report.value(r, validator.SH.focusNode)),
                 validator.local(report.value(r, validator.SH.sourceShape)))
                for r in report.subjects(RDF.type, validator.SH.ValidationResult)}
    assert ('HistoricalReviewDecision', 'PastReviewDecisionShape') in failures
    assert ('Communication', 'TwoSignatureShape') in failures
    g.set((E('HistoricalReviewDecision'), P('decisionOutcome'), P('OutcomeDeny')))
    g.add((E('Communication'), P('communicationSigner'), E('PersonModelRisk')))
    assert validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]


def test_04_reconstruction_and_zero_affected_clients():
    from pyshacl import validate
    g = example()
    assert len(list(g.subjects(P('coverageStatus'), P('QuestionAnswered')))) == 10
    assert not list(g.subjects(P('coverageStatus'), P('QuestionNotRecorded')))
    g.remove((E('Affected'), P('affectedCommunication'), None))
    g.remove((E('Affected'), P('withClient'), None))
    g.set((E('Affected'), P('resultCount'), Literal(0, datatype=XSD.nonNegativeInteger)))
    assert validate(g, shacl_graph=Graph().parse(BASE / 'pwm-shapes.ttl'),
                    ont_graph=Graph().parse(BASE / 'pwm.ttl'), inference='rdfs')[0]


def validate_05(data, names):
    """Exercise selected policy boundaries against full ontology inference."""
    from pyshacl import validate
    from rdflib import BNode
    all_shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    shapes = Graph()
    pending = [URIRef(validator.PWMSH + name) for name in names]
    # Include prefix declarations and nested property/RDF list nodes.
    pending.append(URIRef('https://ahmedmuzammil.com/ontology/pwm/shapes'))
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        for triple in all_shapes.triples((node, None, None)):
            shapes.add(triple)
            if isinstance(triple[2], BNode):
                pending.append(triple[2])
    return validate(data, shacl_graph=shapes, ont_graph=Graph().parse(BASE / 'pwm.ttl'), inference='rdfs')[0]


def test_05_promotion_thresholds_and_step_down():
    shapes = ['AutonomyEvidenceShape', 'AutonomyTransitionShape']
    for prop, value, allowed in [
        ('observedQuality', Literal('.95', datatype=XSD.decimal), True),
        ('observedQuality', Literal('.949', datatype=XSD.decimal), False),
        ('observedExceptionRate', Literal('.05', datatype=XSD.decimal), False),
        ('sampleCount', Literal(99, datatype=XSD.nonNegativeInteger), False),
        ('fullyReconstructable', Literal(False), False),
        ('windowEnd', Literal('2026-11-01T00:59:59Z', datatype=XSD.dateTime), False),
        ('windowEnd', Literal('2026-11-01T01:00:00Z', datatype=XSD.dateTime), True),
    ]:
        g = example()
        g.set((E('Promotion'), P(prop), value))
        assert validate_05(g, shapes) == allowed, prop
    g = example()
    g.set((E('StepDown'), P('toAutonomyLevel'), P('L1Explain')))
    assert not validate_05(g, shapes)
    g.set((E('StepDown'), P('toAutonomyLevel'), P('L2Recommend')))
    g.remove((E('StepDown'), P('notifiedRole'), E('KnowledgeLead')))
    assert not validate_05(g, shapes)


def test_05_authority_revocation_and_expiry_boundaries():
    shapes = ['LiveAuthorityDecisionShape', 'LiveGrantShape']
    g = example()
    g.set((E('ProxyCall'), P('governedByDecision'), E('PolicyDecisionModelAllow')))
    assert validate_05(g, shapes)
    decision_time = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    at = g.value(E('PolicyDecisionModelAllow'), decision_time)
    g.set((E('GrantRevocation'), P('revokedAt'), at))
    assert not validate_05(g, shapes)
    # All example calls stop relying on the revoked optional grant.
    g.remove((None, P('reliesOnGrant'), None))
    assert validate_05(g, shapes)
    g.set((E('Delegation'), P('validUntil'), at))
    assert not validate_05(g, shapes)
    g.set((E('PolicyDecisionModelAllow'), P('decisionOutcome'), P('OutcomeDeny')))
    assert validate_05(g, shapes)


def test_05_scope_narrowing_and_depth():
    shapes = ['SubDelegationShape', 'ScopeNarrowingLimitsShape']
    for prop, value in [('delegationDepth', Literal(3, datatype=XSD.nonNegativeInteger)),
                        ('maxDelegationDepth', Literal(3, datatype=XSD.nonNegativeInteger)),
                        ('validUntil', Literal('2026-11-04T00:00:00Z', datatype=XSD.dateTime))]:
        g = example()
        g.set((E('ChildDelegation'), P(prop), value))
        assert not validate_05(g, shapes)
    g = example()
    g.add((E('ReadScope'), P('scopeField'), Literal('passportNumber')))
    assert not validate_05(g, shapes)
    g = example()
    g.remove((E('ReadScope'), P('scopeLimit'), None))
    assert not validate_05(g, shapes)
    g = example()
    g.set((E('Delegation'), P('parentDelegation'), E('ChildDelegation')))
    assert not validate_05(g, shapes)


def test_05_blast_radius_and_content_lanes():
    g = example()
    g.set((E('Observation'), P('projectedConsumption'), Literal('5', datatype=XSD.decimal)))
    assert validate_05(g, ['BlastRadiusDecisionShape'])
    g.set((E('Observation'), P('projectedConsumption'), Literal('5.01', datatype=XSD.decimal)))
    assert not validate_05(g, ['BlastRadiusDecisionShape'])
    g.set((E('PolicyDecisionModelAllow'), P('decisionOutcome'), P('OutcomeHumanApproval')))
    assert validate_05(g, ['BlastRadiusDecisionShape'])
    g = example()
    g.set((E('ProxyCall'), P('governedByDecision'), E('PolicyDecisionModelAllow')))
    g.add((E('Run'), P('inputContent'), E('ClientSnapshot')))
    g.set((E('ToolPortfolioRead'), P('publicContentTool'), Literal(True)))
    assert not validate_05(g, ['ContentIsolationShape'])
    g.set((E('PolicyDecisionModelAllow'), P('decisionOutcome'), P('OutcomeDeny')))
    assert validate_05(g, ['ContentIsolationShape'])


def test_05_owl_datatypes_context_and_grounding():
    from rdflib.namespace import OWL, RDFS
    g = Graph().parse(BASE / 'pwm.ttl')
    terms = set(g.subjects(P('inModule'), P('ModuleAgents')))
    allowed = {XSD.string, XSD.anyURI, XSD.boolean, XSD.dateTime, XSD.decimal,
               XSD.nonNegativeInteger, XSD.positiveInteger}
    for term in terms:
        if (term, RDF.type, OWL.DatatypeProperty) in g:
            assert set(g.objects(term, RDFS.range)) <= allowed
        if (term, RDF.type, OWL.Class) in g:
            if term == P('AuthorityLiveStatus'):
                assert g.value(term, P('contextSpeed')) == P('FastContext')
            else:
                assert g.value(term, P('contextSpeed')) in (P('SlowContext'), P('MediumContext'))
        citations = list(g.objects(term, P('citation')))
        assert any(g.value(c, P('cites')) == P('ref-KOG-ROADMAP') and
                   '0.5' in str(g.value(c, P('locator'))) for c in citations)
    assert g.value(P('DelegatedAuthority'), P('contextSpeed')) == P('SlowContext')
    assert g.value(P('WorkloadIdentity'), P('contextSpeed')) == P('SlowContext')
    assert g.value(P('KillSwitch'), P('contextSpeed')) == P('SlowContext')


def test_05_no_term_redefinition_or_em_dash():
    import re
    existing = ''.join(p.read_text() for p in sorted((BASE / 'src').glob('*.ttl'))
                       if p.name != '11-agents-authority-fleets.ttl')
    new = (BASE / 'src/11-agents-authority-fleets.ttl').read_text()
    pattern = r'^pwm:(\w+) a (?:owl:Class|owl:ObjectProperty|owl:DatatypeProperty|skos:ConceptScheme)'
    assert not set(re.findall(pattern, existing, re.M)) & set(re.findall(pattern, new, re.M))
    for path in BASE.rglob('*'):
        if path.suffix in ('.ttl', '.md', '.py'):
            assert chr(0x2014) not in path.read_text(), path


def test_05_approval_repairs_and_combined_taint():
    # Human approval is necessary at high autonomy and can satisfy the rule.
    g = example()
    g.set((E('Agent'), P('effectiveAutonomyLevel'), P('L5BoundedAutonomy')))
    g.set((E('Agent'), P('maxAutonomyLevel'), P('L5BoundedAutonomy')))
    assert not validate_05(g, ['IrreversibleApprovalShape'])
    g.add((E('PolicyDecisionModelAllow'), P('humanApproval'), E('Resolution')))
    # Approval of a different held decision in the same run is insufficient.
    assert not validate_05(g, ['IrreversibleApprovalShape'])
    prepare_held_approval(g, 'PolicyDecisionModelAllow')
    g.set((E('Resolution'), URIRef('http://www.w3.org/ns/prov#generatedAtTime'), Literal('2026-11-03T01:58:01Z', datatype=XSD.dateTime)))
    assert validate_05(g, ['IrreversibleApprovalShape'])
    # Both external and client-data taint may apply to the same content.
    g = example()
    g.add((E('ExternalResearch'), P('trustLabel'), P('TrustClientData')))
    assert validate_05(g, ['ExternalContentShape'])
    g.set((E('ToolPortfolioRead'), P('publicContentTool'), Literal(True)))
    assert not validate_05(g, ['ContentIsolationShape'])


@pytest.mark.parametrize('case', ['denied', 'pending', 'no_request', 'cross_run', 'multi_run_checkpoint', 'no_run'])
def test_held_approval_cannot_authorize_governed_calls(case):
    """All call types require an approved held decision in their own run."""
    from pyshacl import validate
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    ontology = Graph().parse(BASE / 'pwm.ttl')
    for kind in ('ModelCall', 'ToolCall', 'MCPProxiedCall'):
        g = example()
        g.set((E('ProxyCall'), P('governedByDecision'), E('PolicyDecisionModelAllow')))
        call = E('GatewayCall')
        g.set((call, RDF.type, P(kind)))
        g.set((call, P('governedByDecision'), E('PolicyDecisionRelease')))
        g.add((call, P('invokedTool'), E('ToolPortfolioRead')))
        g.add((call, P('mcpBackend'), Literal('portfolio-local')))
        g.add((call, P('aboutClient'), E('Client')))
        g.add((call, P('classification'), P('C2')))
        g.add((call, P('jurisdictionCode'), Literal('SG')))
        assert validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]
        if case == 'denied':
            g.set((E('Resolution'), P('approvalResolution'), P('ApprovalDenied')))
        elif case == 'pending':
            g.remove((E('Resolution'), None, None))
        elif case == 'no_request':
            g.remove((E('Resolution'), None, None))
            g.remove((E('Approval'), None, None))
        elif case == 'cross_run':
            other = E('OtherRun')
            for predicate, value in list(g.predicate_objects(E('Run'))):
                g.add((other, predicate, value))
            g.set((E('PolicyDecisionRelease'), P('inRun'), other))
        elif case == 'multi_run_checkpoint':
            other = E('OtherRun')
            for predicate, value in list(g.predicate_objects(E('Run'))):
                g.add((other, predicate, value))
            g.add((E('PolicyDecisionRelease'), P('inRun'), other))
            g.set((E('Checkpoint'), P('inRun'), other))
        else:
            g.remove((E('PolicyDecisionRelease'), P('inRun'), None))
        conforms, report, _ = validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')
        assert not conforms
        failures = {(report.value(r, validator.SH.focusNode),
                     report.value(r, validator.SH.sourceShape))
                    for r in report.subjects(RDF.type, validator.SH.ValidationResult)
                    if report.value(r, validator.SH.sourceShape) == URIRef(validator.PWMSH + 'GovernedCallShape')}
        assert failures == {(call, URIRef(validator.PWMSH + 'GovernedCallShape'))}


def test_proxy_read_uses_pre_drafting_decision():
    g = example()
    assert g.value(E('ProxyCall'), P('governedByDecision')) == E('PolicyDecisionModelAllow')
    assert not validator.chain_problems(g)


def test_05_duplicate_credential_references_on_distinct_identities():
    g = example()
    for name in ('CredentialA', 'CredentialB'):
        for predicate, value in list(g.predicate_objects(E('Workload'))):
            g.add((E(name), predicate, value))
        g.set((E(name), P('workloadSubject'), Literal(name)))
        g.set((E(name), P('credentialReference'), Literal('urn:credential:duplicate', datatype=XSD.anyURI)))
    assert not validate_05(g, ['WorkloadIdentityShape'])
    g.set((E('CredentialB'), P('credentialReference'), Literal('urn:credential:unique', datatype=XSD.anyURI)))
    assert validate_05(g, ['WorkloadIdentityShape'])


def test_05_irreversible_approval_with_maximum_autonomy_only():
    g = example()
    g += Graph().parse(BASE / 'examples/pwm-example-invalid.ttl')
    assert not list(g.objects(E('MaxOnlyAgent'), P('effectiveAutonomyLevel')))
    # Isolate the max-only call from the other negative irreversible case.
    g.remove((E('Bad05IrreversibleApprovalShape'), P('invokedTool'), None))
    g.remove((E('Bad05WrongApprovalDecision'), P('invokedTool'), None))
    g.remove((E('BadLateApprovalCall'), P('invokedTool'), None))
    assert not validate_05(g, ['IrreversibleApprovalShape'])
    g.set((E('MaxOnlyDecision'), P('humanApproval'), E('Resolution')))
    assert not validate_05(g, ['IrreversibleApprovalShape'])
    prepare_held_approval(g, 'MaxOnlyDecision')
    at = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    g.set((E('Resolution'), at, Literal('2026-11-03T01:58:01Z', datatype=XSD.dateTime)))
    g.set((E('Bad05IrreversibleMaxOnly'), at, Literal('2026-11-03T02:00:00Z', datatype=XSD.dateTime)))
    assert validate_05(g, ['IrreversibleApprovalShape'])


def test_review_date_without_timestamp_is_not_evaluated():
    from pyshacl import validate
    g = Graph()
    g.add((E('UntimedDecision'), RDF.type, P('PolicyDecision')))
    g.add((E('UntimedDecision'), P('underUseCase'), E('ExpiredUseCase')))
    g.add((E('UntimedDecision'), P('decisionOutcome'), P('OutcomeAllow')))
    g.add((E('ExpiredUseCase'), P('nextReviewDate'), Literal('2000-01-01', datatype=XSD.date)))
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    for shape in set(shapes.subjects(RDF.type, validator.SH.NodeShape)):
        if shape != URIRef(validator.PWMSH + 'PastReviewDecisionShape'):
            shapes.add((shape, validator.SH.deactivated, Literal(True)))
    assert validate(g, shacl_graph=shapes)[0]
    g.add((E('UntimedDecision'), URIRef('http://www.w3.org/ns/prov#generatedAtTime'),
           Literal('2026-11-03T00:00:00Z', datatype=XSD.dateTime)))
    assert not validate(g, shacl_graph=shapes)[0]


@pytest.mark.parametrize('role_class', ['RiskCommittee', 'DataGovernanceCommittee', 'ChiefRiskOfficer'])
def test_05_approving_authority_roles(role_class):
    g = example()
    g.set((E('FirmRiskCommittee'), RDF.type, P(role_class)))
    names = ['RiskControlApprovalShape', 'RiskControlApprovalRecordShape']
    assert validate_05(g, names)
    g.set((E('CommitteeApproval'), P('approverRole'), E('Agent')))
    assert not validate_05(g, names)


@pytest.mark.parametrize('timestamp,expired', [
    ('2026-11-03T23:30:00-02:00', True),
    ('2026-11-04T00:30:00+02:00', False),
    ('2026-11-03T23:59:59Z', False),
    ('2026-11-04T00:00:00Z', True),
])
def test_review_expiry_compares_typed_instants(timestamp, expired):
    from pyshacl import validate
    g = Graph()
    g.add((E('TimedDecision'), RDF.type, P('PolicyDecision')))
    g.add((E('TimedDecision'), P('underUseCase'), E('ReviewUseCase')))
    g.add((E('TimedDecision'), P('decisionOutcome'), P('OutcomeAllow')))
    g.add((E('TimedDecision'), URIRef('http://www.w3.org/ns/prov#generatedAtTime'),
           Literal(timestamp, datatype=XSD.dateTime)))
    g.add((E('ReviewUseCase'), P('nextReviewDate'), Literal('2026-11-03', datatype=XSD.date)))
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    for shape in set(shapes.subjects(RDF.type, validator.SH.NodeShape)):
        if shape != URIRef(validator.PWMSH + 'PastReviewDecisionShape'):
            shapes.add((shape, validator.SH.deactivated, Literal(True)))
    assert validate(g, shacl_graph=shapes)[0] is not expired


def test_model_authorization_precedes_dispatch():
    g = example()
    at = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    assert g.value(E('PolicyDecisionModelAllow'), at).toPython() < g.value(E('Evidence4'), at).toPython()


def test_review_rank_narrowing_and_entitlement_direction():
    g = example()
    assert g.value(E('ReadScope'), P('classification')) == P('C1')
    assert validate_05(g, ['DelegationScopeShape', 'LiveAuthorityDecisionShape', 'SubDelegationShape'])
    g.set((E('ReadScope'), P('classification'), P('C3')))
    assert not validate_05(g, ['SubDelegationShape'])
    g = example()
    g.remove((E('PersonRM'), P('holdsEntitlement'), E('ReadEntitlement')))
    g.set((E('Scope'), P('classification'), P('C1')))
    # A C1 entitlement cannot authorize a C2 delegation.
    for pred, value in list(g.predicate_objects(E('Scope'))):
        g.add((E('DelegationScopeC2'), pred, value))
    g.set((E('DelegationScopeC2'), P('classification'), P('C2')))
    g.set((E('Delegation'), P('authorityScope'), E('DelegationScopeC2')))
    assert not validate_05(g, ['DelegationScopeShape', 'LiveAuthorityDecisionShape'])


def test_review_step_up_binds_exact_decision_and_run():
    g = example()
    g.set((E('PolicyDecisionRelease'), P('decisionOutcome'), P('OutcomeAllow')))
    assert not validate_05(g, ['InstructionAssuranceShape'])
    prepare_held_approval(g, 'PolicyDecisionModelAllow')
    assert not validate_05(g, ['InstructionAssuranceShape'])
    g = example()
    g.set((E('PolicyDecisionRelease'), P('decisionOutcome'), P('OutcomeAllow')))
    g.set((E('Checkpoint'), P('inRun'), E('OtherRun')))
    assert not validate_05(g, ['InstructionAssuranceShape'])


def test_review_blast_radius_complete_agent_limit_coverage():
    g = example()
    assert validate_05(g, ['BlastRadiusDecisionShape'])
    g.remove((E('PolicyDecisionModelAllow'), P('limitObservation'), None))
    assert not validate_05(g, ['BlastRadiusDecisionShape'])
    g = example()
    g.set((E('Observation'), P('observedLimit'), E('UnrelatedLimit')))
    assert not validate_05(g, ['BlastRadiusDecisionShape'])


def test_review_irreversible_approval_strictly_precedes_call():
    g = example()
    g += Graph().parse(BASE / 'examples/pwm-example-invalid.ttl')
    # Only target the late approval call, isolating this timing regression.
    for call in list(g.subjects(P('invokedTool'), None)):
        if call != E('BadLateApprovalCall'):
            g.remove((call, P('invokedTool'), None))
    assert not validate_05(g, ['IrreversibleApprovalShape'])
    at = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    g.set((E('LateResolution'), at, Literal('2026-11-03T01:59:00Z', datatype=XSD.dateTime)))
    assert validate_05(g, ['IrreversibleApprovalShape'])


def test_positive_decisions_precede_every_governed_call():
    g = example()
    at = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    for call, decision in g.subject_objects(P('governedByDecision')):
        assert g.value(decision, at).toPython() < g.value(call, at).toPython()
def test_build_rejects_broken_evidence_chain(tmp_path):
    import shutil
    shutil.copytree(BASE / 'src', tmp_path / 'src')
    shutil.copytree(BASE / 'examples', tmp_path / 'examples')
    shutil.copy(BASE / 'GAP_ANALYSIS.md', tmp_path / 'GAP_ANALYSIS.md')
    g = example()
    g.set((E('Evidence2'), P('previousRecordHash'), Literal('sha256:' + '0' * 64)))
    g.serialize(tmp_path / 'examples/pwm-example.ttl', format='turtle')
    with pytest.raises(ValueError, match='Broken example evidence chain'):
        builder.build(tmp_path, None)


@pytest.mark.parametrize('kind', ['delegation', 'switch'])
@pytest.mark.parametrize('case,allowed', [
    ('active', False), ('future', True), ('reset', True),
    ('tied', False), ('unrelated', True), ('absent', True),
])
def test_live_status_at_decision_time(kind, case, allowed):
    from pyshacl import validate
    g = Graph()
    decision, subject = E('Decision'), E('Authority')
    time = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    g.add((decision, RDF.type, P('PolicyDecision')))
    g.add((decision, P('decisionOutcome'), P('OutcomeAllow')))
    g.add((decision, time, Literal('2026-11-03T10:00:00Z', datatype=XSD.dateTime)))
    if kind == 'delegation':
        g.add((subject, RDF.type, P('DelegatedAuthority')))
        g.add((decision, P('reliesOnDelegation'), subject))
        flag, shape = 'revoked', 'LiveDelegationDecisionShape'
        # Missing decision-time authority observations fail closed too.
        if case in ('future', 'absent', 'unrelated'):
            allowed = False
    else:
        g.add((subject, RDF.type, P('KillSwitch')))
        g.add((subject, P('scopeUseCase'), E('UseCase')))
        g.add((decision, P('underUseCase'), E('UseCase')))
        flag, shape = 'engaged', 'LiveKillSwitchDecisionShape'
    def status(name, value, at, target=subject):
        g.add((E(name), RDF.type, P('AuthorityLiveStatus')))
        g.add((E(name), P('statusOf'), target))
        g.add((E(name), P(flag), Literal(value)))
        g.add((E(name), P('asOf'), Literal(at, datatype=XSD.dateTime)))
    if case != 'absent':
        status('On', True, '2026-11-03T10:00:01Z' if case == 'future' else '2026-11-03T10:00:00Z',
               E('Other') if case == 'unrelated' else subject)
    if case == 'reset':
        g.set((E('On'), P('asOf'), Literal('2026-11-03T09:00:00Z', datatype=XSD.dateTime)))
        status('Off', False, '2026-11-03T10:00:00Z')
    if case == 'tied':
        status('Off', False, '2026-11-03T10:00:00Z')
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    for node in set(shapes.subjects(RDF.type, validator.SH.NodeShape)):
        if node != URIRef(validator.PWMSH + shape):
            shapes.add((node, validator.SH.deactivated, Literal(True)))
    assert validate(g, shacl_graph=shapes)[0] == allowed


@pytest.mark.parametrize('dimension', ['scopeUseCase', 'scopePolicy', 'scopeModel', 'scopePromptVersion', 'global'])
def test_engaged_switch_scope(dimension):
    from pyshacl import validate
    g = Graph()
    decision, switch, call = E('Decision'), E('Switch'), E('Call')
    g.add((decision, RDF.type, P('PolicyDecision')))
    g.add((decision, P('decisionOutcome'), P('OutcomeAllow')))
    g.add((decision, URIRef('http://www.w3.org/ns/prov#generatedAtTime'), Literal('2026-11-03T10:00:00Z', datatype=XSD.dateTime)))
    g.add((switch, RDF.type, P('KillSwitch')))
    g.add((E('Status'), RDF.type, P('AuthorityLiveStatus')))
    for prop, value in [('statusOf', switch), ('engaged', Literal(True)), ('asOf', Literal('2026-11-03T10:00:00Z', datatype=XSD.dateTime))]:
        g.add((E('Status'), P(prop), value))
    if dimension != 'global':
        prop = {'scopeUseCase': 'underUseCase', 'scopePolicy': 'decidedByRule', 'scopeModel': 'usesModel', 'scopePromptVersion': 'promptTemplateVersion'}[dimension]
        scope = Literal('v1') if dimension == 'scopePromptVersion' else E('Scope')
        g.add((switch, P(dimension), scope))
        target = call if dimension in ('scopeModel', 'scopePromptVersion') else decision
        g.add((call, P('governedByDecision'), decision))
        g.add((target, P(prop), scope))
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    for node in set(shapes.subjects(RDF.type, validator.SH.NodeShape)):
        if node != URIRef(validator.PWMSH + 'LiveKillSwitchDecisionShape'):
            shapes.add((node, validator.SH.deactivated, Literal(True)))
    assert not validate(g, shacl_graph=shapes)[0]
    if dimension != 'global':
        g.remove((switch, P(dimension), None))
        g.add((switch, P(dimension), Literal('other') if dimension == 'scopePromptVersion' else E('Other')))
        assert validate(g, shacl_graph=shapes)[0]


def prepare_held_approval(g, decision):
    """Resolve a separate held decision under the same policy, use case and run."""
    held = E('HeldFor' + decision)
    for predicate, value in list(g.predicate_objects(E(decision))):
        g.add((held, predicate, value))
    g.set((held, P('decisionOutcome'), P('OutcomeHumanApproval')))
    g.remove((held, P('humanApproval'), None))
    g.set((E('Approval'), P('requestsDecision'), held))


@pytest.mark.parametrize('timestamp,allowed', [
    ('2026-11-04T00:30:00+02:00', True),
    ('2026-11-03T23:30:00-02:00', False),
    ('2026-11-03T23:59:59Z', True),
    ('2026-11-04T00:00:00Z', False),
    ('2026-11-03T23:59:59', True),
])
def test_agent_review_uses_typed_utc_instants(timestamp, allowed):
    g = example()
    g.set((E('Agent'), P('nextReviewDate'), Literal('2026-11-03', datatype=XSD.date)))
    g.set((E('PolicyDecisionModelAllow'), URIRef('http://www.w3.org/ns/prov#generatedAtTime'),
           Literal(timestamp, datatype=XSD.dateTime)))
    assert validate_05(g, ['AgentDecisionShape']) == allowed


@pytest.mark.parametrize('entitlement,delegation,allowed', [('C1', 'C2', False), ('C3', 'C2', True), ('C2', 'C2', True)])
def test_entitlement_classification_rank(entitlement, delegation, allowed):
    g = example()
    g.remove((E('PersonRM'), P('holdsEntitlement'), E('ReadEntitlement')))
    for pred, value in list(g.predicate_objects(E('Scope'))):
        g.add((E('IndependentScope'), pred, value))
    g.set((E('IndependentScope'), P('classification'), P(delegation)))
    g.set((E('Scope'), P('classification'), P(entitlement)))
    g.set((E('Delegation'), P('authorityScope'), E('IndependentScope')))
    for shape in ['DelegationScopeShape', 'LiveAuthorityDecisionShape']:
        assert validate_05(g, [shape]) == allowed


def test_completed_approval_and_suspended_request_are_consistent():
    g = example()
    allowed = E('StepUpAllowed')
    for predicate, value in list(g.predicate_objects(E('PolicyDecisionRelease'))):
        g.add((allowed, predicate, value))
    g.set((allowed, P('decisionOutcome'), P('OutcomeAllow')))
    at = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    g.set((allowed, at, Literal('2026-11-03T02:07:00Z', datatype=XSD.dateTime)))
    assert validate_05(g, ['InstructionAssuranceShape', 'SuspendedApprovalShape'])
    g.set((E('Resolution'), P('approvalResolution'), P('ApprovalPending')))
    assert not validate_05(g, ['InstructionAssuranceShape'])
    assert validate_05(g, ['SuspendedApprovalShape'])


def test_prov_constraints_declare_their_prefix():
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    sh = validator.SH
    for constraint, query in shapes.subject_objects(sh.select):
        if 'prov:' in str(query):
            tables = list(shapes.objects(constraint, sh.prefixes))
            assert any(str(shapes.value(d, sh.prefix)) == 'prov'
                       for table in tables for d in shapes.objects(table, sh.declare))


def test_irreversible_approval_coexists_with_suspended_checkpoint():
    g = example()
    g.set((E('Agent'), P('effectiveAutonomyLevel'), P('L5BoundedAutonomy')))
    g.set((E('Agent'), P('maxAutonomyLevel'), P('L5BoundedAutonomy')))
    g.add((E('PolicyDecisionModelAllow'), P('humanApproval'), E('Resolution')))
    prepare_held_approval(g, 'PolicyDecisionModelAllow')
    at = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    g.set((E('Resolution'), at, Literal('2026-11-03T01:58:01Z', datatype=XSD.dateTime)))
    assert validate_05(g, ['IrreversibleApprovalShape', 'SuspendedApprovalShape'])
    g.remove((E('Resolution'), P('approvalResolution'), None))
    assert not validate_05(g, ['IrreversibleApprovalShape'])
    assert validate_05(g, ['SuspendedApprovalShape'])


@pytest.mark.parametrize('level,conforms', [('C3', True), ('C2', True), ('C1', False), ('C0', False)])
def test_05_delegation_ceiling_covers_scope_and_children(level, conforms):
    g = example()
    # The scope is C2 and the child delegation permits C2, so a lower parent ceiling fails on both counts.
    g.set((E('Delegation'), P('permitsDataClass'), P(level)))
    assert validate_05(g, ['DelegationCeilingShape']) == conforms


@pytest.mark.parametrize('level,conforms', [('C1', True), ('C2', True), ('C3', False), ('C0', False)])
def test_05_child_delegation_cannot_exceed_parent_or_undercut_scope(level, conforms):
    g = example()
    g.set((E('ChildDelegation'), P('permitsDataClass'), P(level)))
    assert validate_05(g, ['DelegationCeilingShape']) == conforms


@pytest.mark.parametrize('effective,conforms', [('L0Retrieve', True), ('L1Explain', False), ('L3Draft', False)])
def test_05_effective_autonomy_cannot_exceed_maximum(effective, conforms):
    g = example()
    g.set((E('Worker'), P('effectiveAutonomyLevel'), P(effective)))
    assert validate_05(g, ['AgentAutonomyShape']) == conforms


def test_05_one_live_status_format():
    ontology = (BASE / 'pwm.ttl').read_text(encoding='utf-8')
    for removed in ('liveStatusSubject', 'liveStatusValue', 'pwm:observedAt a '):
        assert removed not in ontology
    assert 'pwm:statusOf a owl:ObjectProperty' in ontology


def test_05_spec_names_union_ranges():
    spec = (BASE / 'SPEC.md').read_text(encoding='utf-8')
    row = next(line for line in spec.splitlines() if line.startswith('| <a id="statusOf">'))
    assert '[delegated authority](#DelegatedAuthority), [kill switch](#KillSwitch)' in row


def test_review_agent_requires_registered_decision_use_case():
    g = example()
    assert validate_05(g, ['AgentDecisionShape'])
    g.set((E('Agent'), P('registeredForUseCase'), E('UnregisteredUseCase')))
    assert not validate_05(g, ['AgentDecisionShape'])
    g.add((E('Agent'), P('registeredForUseCase'), E('UseCaseMeetingPrep')))
    assert validate_05(g, ['AgentDecisionShape'])


@pytest.mark.parametrize('observation', ['absent', 'exceeded', 'within', 'duplicate'])
@pytest.mark.parametrize('agent_limits', [True, False])
def test_review_delegation_scope_cap(observation, agent_limits):
    g = example()
    # This separate delegation cap is tighter than the agent's own limit.
    g.add((E('Scope'), P('scopeLimit'), E('TightScopeCap')))
    g.add((E('TightScopeCap'), P('limitMaximum'), Literal('2', datatype=XSD.decimal)))
    if not agent_limits:
        g.remove((E('Agent'), P('hasBlastRadiusLimit'), None))
    if observation != 'absent':
        for decision in set(g.subjects(P('decisionOutcome'), P('OutcomeAllow'))):
            obs = URIRef(str(decision) + 'ScopeObservation')
            g.add((decision, P('limitObservation'), obs))
            g.add((obs, P('observedLimit'), E('TightScopeCap')))
            g.add((obs, P('projectedConsumption'), Literal('1' if observation == 'within' else '4', datatype=XSD.decimal)))
            if observation == 'duplicate':
                other = URIRef(str(obs) + 'Duplicate')
                g.add((decision, P('limitObservation'), other))
                g.add((other, P('observedLimit'), E('TightScopeCap')))
    assert validate_05(g, ['BlastRadiusDecisionShape']) == (observation == 'within')


def test_review_isolated_negative_fixtures():
    from pyshacl import validate
    g = example()
    g += Graph().parse(BASE / 'examples/pwm-review-invalid.ttl')
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    names = {'BlastRadiusDecisionShape', 'AgentDecisionShape'}
    for shape in set(shapes.subjects(RDF.type, validator.SH.NodeShape)):
        if str(shape).split('#')[-1] not in names:
            shapes.add((shape, validator.SH.deactivated, Literal(True)))
    conforms, report, _ = validate(g, shacl_graph=shapes,
                                   ont_graph=Graph().parse(BASE / 'pwm.ttl'), inference='rdfs')
    assert not conforms
    failures = {(report.value(r, validator.SH.focusNode), report.value(r, validator.SH.sourceShape))
                for r in report.subjects(RDF.type, validator.SH.ValidationResult)}
    assert failures == {
        (E('ReviewMissingDecision'), URIRef(validator.PWMSH + 'BlastRadiusDecisionShape')),
        (E('ReviewExceededDecision'), URIRef(validator.PWMSH + 'BlastRadiusDecisionShape')),
        (E('ReviewUseCaseCall'), URIRef(validator.PWMSH + 'AgentDecisionShape')),
    }

@pytest.mark.parametrize('timestamp,explicit', [
    ('2026-11-03T10:00:00', False),
    ('2026-11-03T10:00:00Z', True),
    ('2026-11-03T10:00:00+00:00', True),
])
@pytest.mark.parametrize('property_name', ['asOf', 'generatedAtTime'])
def test_live_decision_requires_explicit_timezones(timestamp, explicit, property_name):
    from pyshacl import validate
    g = example()
    time = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    g.set((E('LiveAuthorityAllow'), time,
           Literal('2026-11-03T10:05:00Z', datatype=XSD.dateTime)))
    g.set((E('SwitchLive'), P('engaged'), Literal(True)))
    g.set((E('SwitchLive'), P('asOf'),
           Literal('2026-11-03T10:00:00Z', datatype=XSD.dateTime)))
    focus, path = ((E('SwitchLive'), P('asOf')) if property_name == 'asOf'
                   else (E('LiveAuthorityAllow'), time))
    g.set((focus, path, Literal(timestamp, datatype=XSD.dateTime)))
    conforms, report, _ = validate(g, shacl_graph=Graph().parse(BASE / 'pwm-shapes.ttl'),
                                 ont_graph=Graph().parse(BASE / 'pwm.ttl'), inference='rdfs')
    # An engaged switch still blocks ALLOW when the timezone pattern passes.
    assert not conforms
    pattern_failures = {(report.value(r, validator.SH.focusNode),
                         report.value(r, validator.SH.resultPath))
                        for r in report.subjects(validator.SH.sourceConstraintComponent,
                                                 validator.SH.PatternConstraintComponent)}
    assert ((focus, path) in pattern_failures) == (not explicit)


def test_flight_recorder_session_with_two_runs():
    from pyshacl import validate
    g = Graph()
    g.add((E('Recorder'), RDF.type, P('FlightRecorderSession')))
    g.add((E('Recorder'), P('contentHash'), Literal('sha256:' + 'a' * 64)))
    for name in ('FirstRun', 'SecondRun'):
        g.add((E(name), RDF.type, P('Run')))
        g.add((E('Recorder'), P('recordedRun'), E(name)))
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    for shape in set(shapes.subjects(RDF.type, validator.SH.NodeShape)):
        if shape != URIRef(validator.PWMSH + 'FlightRecorderSessionShape'):
            shapes.add((shape, validator.SH.deactivated, Literal(True)))
    assert validate(g, shacl_graph=shapes)[0]


@pytest.mark.parametrize('level,allowed', [('L0Retrieve', False), ('L3Draft', True), ('L5BoundedAutonomy', True)])
@pytest.mark.parametrize('outcome', ['OutcomeAllow', 'OutcomeDeny'])
def test_call_respects_delegated_scope_autonomy(level, allowed, outcome):
    g = example()
    g.set((E('ProxyCall'), P('invokedTool'), E('ToolDraftNote')))
    g.set((E('Scope'), P('scopeAutonomyLevel'), P(level)))
    g.set((E('PolicyDecisionModelAllow'), P('decisionOutcome'), P(outcome)))
    assert validate_05(g, ['AutonomyCallShape']) == (allowed or outcome == 'OutcomeDeny')
