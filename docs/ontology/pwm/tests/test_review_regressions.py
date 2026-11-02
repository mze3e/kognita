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
    data.add((E('FirmRisk'), P('productRiskLevel'), P('RiskLevel1')))
    assert not validate(data, shacl_graph=shapes, ont_graph=ontology, inference='rdfs')[0]
