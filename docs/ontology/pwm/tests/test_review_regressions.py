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
    assert g.value(URIRef(site), OWL.versionIRI) == URIRef(site + '/0.2')
    assert shapes.value(URIRef(site + '/shapes'), OWL.versionIRI) == URIRef(site + '/0.2/pwm-shapes.ttl')
    assert str(g.value(URIRef(site), OWL.versionInfo)) == '0.2 (draft)'
    import shutil
    shutil.copytree(BASE / 'src', tmp_path / 'src')
    shutil.copy(BASE / 'pwm-shapes.ttl', tmp_path / 'pwm-shapes.ttl')
    counts = builder.build(tmp_path, None)['counts']
    assert (counts['classes'], counts['objectProperties'], counts['datatypeProperties'],
            counts['codeLists'], counts['rules'], counts['codeListRules']) == (127, 105, 48, 38, 30, 49)
    assert sum(validator.EXPECTED.values()) == 42
    assert len(set(Graph().parse(BASE / 'examples/pwm-example-invalid.ttl').subjects(RDF.type, None))) == 40


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
    assert len(actual) == 49


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
