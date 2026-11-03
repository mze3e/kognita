import importlib.util
from pathlib import Path

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
    assert g.value(URIRef(site), OWL.versionIRI) == URIRef(site + '/0.4')
    assert shapes.value(URIRef(site + '/shapes'), OWL.versionIRI) == URIRef(site + '/0.4/pwm-shapes.ttl')
    assert str(g.value(URIRef(site), OWL.versionInfo)) == '0.4 (draft)'
    assert g.value(URIRef(site), OWL.priorVersion) == URIRef(site + '/0.3')
    assert shapes.value(URIRef(site + '/shapes'), OWL.priorVersion) == URIRef(site + '/0.3/pwm-shapes.ttl')
    assert len(set(g.subjects(RDF.type, SKOS.Concept))) == 229
    assert len(set(shapes.subjects(RDF.type, validator.SH.NodeShape))) == 141
    assert len(set(example().subjects(RDF.type, None))) == 146
    import shutil
    shutil.copytree(BASE / 'src', tmp_path / 'src')
    shutil.copy(BASE / 'pwm-shapes.ttl', tmp_path / 'pwm-shapes.ttl')
    shutil.copytree(BASE / 'examples', tmp_path / 'examples')
    shutil.copy(BASE / 'GAP_ANALYSIS.md', tmp_path / 'GAP_ANALYSIS.md')
    counts = builder.build(tmp_path, None)['counts']
    assert (counts['classes'], counts['objectProperties'], counts['datatypeProperties'],
            counts['codeLists'], counts['rules'], counts['codeListRules']) == (167, 171, 118, 46, 81, 59)
    assert sum(validator.EXPECTED.values()) == 96
    assert len(set(Graph().parse(BASE / 'examples/pwm-example-invalid.ttl').subjects(RDF.type, None))) == 94


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
    assert len(actual) == 59


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
             ('Edit', 'afterHash'), ('Edit', 'editDiffHash'),
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
    due = g.value(E('UseCaseMeetingPrep'), P('nextReviewDate'))
    at = URIRef('http://www.w3.org/ns/prov#generatedAtTime')
    g.set((E('PolicyDecisionModelAllow'), at, Literal(str(due) + 'T23:59:59Z', datatype=XSD.dateTime)))
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    ontology = Graph().parse(BASE / 'pwm.ttl')
    assert validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]
    from datetime import date, timedelta
    later = date.fromisoformat(str(due)) + timedelta(days=1)
    g.set((E('PolicyDecisionModelAllow'), at, Literal(str(later) + 'T00:00:00Z', datatype=XSD.dateTime)))
    g.remove((E('Communication'), P('communicationSigner'), E('PersonModelRisk')))
    g.add((E('Communication'), P('communicationSigner'), E('PersonRM')))
    conforms, report, _ = validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')
    assert not conforms
    failures = {(validator.local(report.value(r, validator.SH.focusNode)),
                 validator.local(report.value(r, validator.SH.sourceShape)))
                for r in report.subjects(RDF.type, validator.SH.ValidationResult)}
    assert ('PolicyDecisionModelAllow', 'PastReviewDecisionShape') in failures
    assert ('Communication', 'TwoSignatureShape') in failures
    g.set((E('PolicyDecisionModelAllow'), P('decisionOutcome'), P('OutcomeDeny')))
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


def test_denied_approval_cannot_authorize_governed_calls():
    """All call types reject denial while an approved held decision still passes."""
    from pyshacl import validate
    shapes = Graph().parse(BASE / 'pwm-shapes.ttl')
    ontology = Graph().parse(BASE / 'pwm.ttl')
    for kind in ('ModelCall', 'ToolCall', 'MCPProxiedCall'):
        g = example()
        call = E('GatewayCall')
        g.set((call, RDF.type, P(kind)))
        g.set((call, P('governedByDecision'), E('PolicyDecisionRelease')))
        g.add((call, P('invokedTool'), E('ToolPortfolioRead')))
        g.add((call, P('mcpBackend'), Literal('portfolio-local')))
        assert validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]
        g.set((E('Resolution'), P('approvalResolution'), P('ApprovalDenied')))
        conforms, report, _ = validate(g, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')
        assert not conforms
        failures = {(report.value(r, validator.SH.focusNode),
                     report.value(r, validator.SH.sourceShape))
                    for r in report.subjects(RDF.type, validator.SH.ValidationResult)}
        assert failures == {(call, URIRef(validator.PWMSH + 'GovernedCallShape'))}
