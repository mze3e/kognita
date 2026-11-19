"""A Kognita evidence export reads as PWM RDF through the JSON-LD context.

The sample export was written and signed by Kognita (tools/roundtrip.py make). When Kognita
is importable, the test also verifies hashes and signatures with Kognita's own verifier.
"""
import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "tools"))
import roundtrip  # noqa: E402

EXPORT = json.loads((BASE / "examples/kognita-export.json").read_text())


def test_context_is_current():
    from rdflib import Graph
    expected = roundtrip.build_context(Graph().parse(BASE / "pwm.ttl"))
    assert json.loads((BASE / "context/pwm-evidence.jsonld").read_text()) == expected


def test_export_reads_as_conforming_pwm_data():
    assert roundtrip.check(EXPORT) == []


def test_tampered_export_is_caught_by_the_chain_check():
    broken = json.loads(json.dumps(EXPORT))
    broken["events"][2]["prev_hash"] = "1" * 64
    assert any("chain" in p or "fork" in p or "predecessor" in p or "outside" in p for p in roundtrip.check(broken))


def test_every_kognita_event_type_in_the_export_has_a_pwm_concept():
    from rdflib import Graph
    g = roundtrip.export_to_graph(EXPORT)
    from rdflib import URIRef
    types = set(g.objects(None, URIRef(roundtrip.PWM_NS + "eventType")))
    assert len(types) == len({e["event_type"] for e in EXPORT["events"]})


def test_kognita_verifies_the_same_export():
    pytest.importorskip("kognita.evidence")
    from kognita.evidence import verify_export
    public = bytes.fromhex((BASE / "examples/kognita-export.pub").read_text().strip())
    assert verify_export(EXPORT, public_key=public) == len(EXPORT["events"])
