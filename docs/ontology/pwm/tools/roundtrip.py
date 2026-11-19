"""Round trip between a Kognita evidence export and PWM RDF.

    python3 tools/roundtrip.py context                 write context/pwm-evidence.jsonld from pwm.ttl
    .venv/bin/python tools/roundtrip.py make           (Kognita installed) write examples/kognita-export.json
    python3 tools/roundtrip.py check [export.json]     convert to RDF, validate, compare hashes

The context maps the keys of `kognita evidence export` to PWM terms. Payloads are not mapped:
the RDF carries hashes and references, never content. Hashes are kept exactly as exported.
"""
import json, pathlib, sys
from rdflib import Graph, Namespace, URIRef, RDF
from rdflib.namespace import SKOS

BASE = pathlib.Path(__file__).resolve().parents[1]
PWM_NS = "https://ahmedmuzammil.com/ontology/pwm#"
CONTEXT_PATH = BASE / "context/pwm-evidence.jsonld"
EXPORT_PATH = BASE / "examples/kognita-export.json"
PUBKEY_PATH = BASE / "examples/kognita-export.pub"
GENESIS = "0" * 64

def build_context(ont: Graph) -> dict:
    ctx = {"pwm": PWM_NS, "prov": "http://www.w3.org/ns/prov#", "xsd": "http://www.w3.org/2001/XMLSchema#",
           "event_hash": "pwm:recordHash", "prev_hash": "pwm:previousRecordHash", "chain_start": {"@id": "pwm:chainStart", "@type": "xsd:boolean"},
           "sequence": {"@id": "pwm:evidenceSequence", "@type": "xsd:nonNegativeInteger"},
           "signature": "pwm:signature", "schema_version": "pwm:schemaVersion", "payload_hash": "pwm:contentHash",
           "recorded_at": {"@id": "prov:generatedAtTime", "@type": "xsd:dateTime"},
           "key_id": {"@id": "pwm:signedBy", "@type": "@id"},
           "event_type": {"@id": "pwm:eventType", "@type": "@vocab"},
           "actor_type": {"@id": "pwm:actorType", "@type": "@vocab"},
           "classification": {"@id": "pwm:classification", "@type": "@vocab"}}
    for scheme in ("EvidenceEventTypeScheme", "ActorTypeScheme", "DataClassificationScheme"):
        for concept in sorted(ont.subjects(SKOS.inScheme, URIRef(PWM_NS + scheme))):
            notation = str(ont.value(concept, SKOS.notation))
            ctx[notation] = "pwm:" + str(concept)[len(PWM_NS):]
    return {"@context": ctx}

def to_jsonld(export: dict, context: dict) -> dict:
    graph = []
    keys = {}
    for event in export["events"]:
        node = {"@id": f"urn:kognita:event:{event['sequence']}", "@type": "pwm:EvidenceRecord"}
        for key in ("event_hash", "sequence", "signature", "schema_version", "payload_hash", "recorded_at", "event_type", "actor_type", "classification"):
            if event.get(key) not in (None, ""):
                node[key] = event[key]
        if event["prev_hash"] == GENESIS:
            node["chain_start"] = True
        else:
            node["prev_hash"] = event["prev_hash"]
        if event.get("key_id"):
            node["key_id"] = f"urn:kognita:key:{event['key_id']}"
            keys[event["key_id"]] = {"@id": f"urn:kognita:key:{event['key_id']}", "@type": "pwm:SigningKey",
                                     "pwm:keyId": event["key_id"], "pwm:publicKeyReference": f"urn:kognita:key:{event['key_id']}"}
        graph.append(node)
    return {"@context": context["@context"], "@graph": graph + list(keys.values())}

def export_to_graph(export: dict) -> Graph:
    context = json.loads(CONTEXT_PATH.read_text())
    g = Graph()
    g.parse(data=json.dumps(to_jsonld(export, context)), format="json-ld")
    return g

def check(export: dict) -> list:
    sys.path.insert(0, str(BASE / "tools"))
    import validate
    from pyshacl import validate as shacl
    g = export_to_graph(export)
    ont = Graph().parse(BASE / "pwm.ttl")
    problems = list(validate.chain_problems(g, ont))
    shapes = Graph().parse(BASE / "pwm-shapes.ttl")
    conforms, _, text = shacl(g, shacl_graph=shapes, ont_graph=ont, inference="rdfs")
    if not conforms:
        problems.append("SHACL: " + text[:400])
    P = lambda n: URIRef(PWM_NS + n)
    rdf_hashes = {str(o) for o in g.objects(None, P("recordHash"))}
    json_hashes = {e["event_hash"] for e in export["events"]}
    if rdf_hashes != json_hashes:
        problems.append("record hashes differ between the export and the RDF")
    if len(set(g.subjects(RDF.type, P("EvidenceRecord")))) != export["event_count"]:
        problems.append("record count differs")
    return problems

def make():
    from sqlmodel import Session
    from kognita.db import create_all, make_engine
    from kognita.evidence import EvidenceWriter, export_chain, verify_export
    from kognita.signing import EvidenceSigner, _generate
    from kognita.vocabulary import ActorType, EventType
    private, public = _generate()
    engine = make_engine(":memory:")
    create_all(engine)
    with Session(engine) as session:
        writer = EvidenceWriter(engine, signing=EvidenceSigner(private))
        for kind, actor in [(EventType.POLICY_DECISION, ActorType.SYSTEM), (EventType.MODEL_CALL, ActorType.AGENT),
                            (EventType.TOOL_CALL, ActorType.AGENT), (EventType.APPROVAL, ActorType.HUMAN),
                            (EventType.CLIENT_COMMUNICATION, ActorType.AGENT)]:
            writer.emit(session, correlation_id="roundtrip-example", event_type=kind, actor_type=actor, actor_id="synthetic")
        session.commit()
        export = export_chain(session)
    assert verify_export(export, public_key=public) == len(export["events"])
    for event in export["events"]:
        event["payload"] = {}  # the sample carries no content
    EXPORT_PATH.write_text(json.dumps(export, indent=1, sort_keys=True) + "\n")
    PUBKEY_PATH.write_text(public.hex() + "\n")
    print("wrote", EXPORT_PATH, "events:", len(export["events"]))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "context":
        CONTEXT_PATH.write_text(json.dumps(build_context(Graph().parse(BASE / "pwm.ttl")), indent=1, sort_keys=True) + "\n")
        print("wrote", CONTEXT_PATH)
    elif cmd == "make":
        make()
    else:
        path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else EXPORT_PATH
        issues = check(json.loads(path.read_text()))
        print("round trip ok" if not issues else "\n".join(issues))
        sys.exit(1 if issues else 0)
