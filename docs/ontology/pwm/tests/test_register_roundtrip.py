"""A Kognita register export reads as PWM RDF through the JSON-LD context.

The sample export is written by tools/register_roundtrip.py make. When Kognita
is importable, the test also rebuilds the export from a fresh store and checks
that the committed file stays current.
"""
import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "tools"))
import register_roundtrip as rt  # noqa: E402

EXPORT = json.loads((BASE / "examples/kognita-register.json").read_text())


def test_context_is_current():
    from rdflib import Graph
    expected = rt.build_context(Graph().parse(BASE / "pwm.ttl"))
    assert json.loads((BASE / "context/pwm-register.jsonld").read_text()) == expected


def test_committed_export_round_trips():
    assert rt.check(EXPORT) == []


def test_tampered_export_does_not_match_the_committed_rdf():
    from rdflib import Graph
    broken = json.loads(json.dumps(EXPORT))
    broken["use_cases"][0]["purpose"] = "a different purpose"
    ont = Graph().parse(BASE / "pwm.ttl")
    imported = rt.canonical(rt.from_graph(rt.export_to_graph(EXPORT, ont), ont))
    assert imported != rt.canonical(broken)
    assert imported == rt.canonical(EXPORT)


def test_export_covers_use_cases_agents_and_sources():
    assert EXPORT["use_cases"] and EXPORT["agents"] and EXPORT["sources"]
    assert {u["usecase_id"] for u in EXPORT["use_cases"]} == {"research-summary"}
    assert {a["name"] for a in EXPORT["agents"]} == {"research-assistant"}
    assert {s["source_id"] for s in EXPORT["sources"]} == {"cio-note"}


def test_kognita_rebuilds_the_same_export(tmp_path, monkeypatch):
    pytest.importorskip("kognita.usecases")
    monkeypatch.setattr(rt, "EXPORT_PATH", tmp_path / "kognita-register.json")
    rebuilt = rt.canonical(rt.make())
    assert rebuilt == rt.canonical(EXPORT)
    assert json.loads((BASE / "examples/kognita-register.json").read_text()) == EXPORT


def test_distinct_owner_identities_round_trip():
    from rdflib import Graph
    sample = json.loads(json.dumps(EXPORT))
    sample["use_cases"][0]["business_owner"] = "Jane Smith"
    sample["use_cases"][0]["technology_owner"] = "Jane-Smith"
    assert rt.check(sample) == []
    ont = Graph().parse(BASE / "pwm.ttl")
    imported = rt.canonical(rt.from_graph(rt.export_to_graph(sample, ont), ont))
    assert imported["use_cases"][0]["business_owner"] == "Jane Smith"
    assert imported["use_cases"][0]["technology_owner"] == "Jane-Smith"


def test_source_version_is_preserved():
    sample = json.loads(json.dumps(EXPORT))
    sample["sources"][0]["version"] = "1.0.0"
    assert rt.check(sample) == []


def test_every_numeric_blast_limit_round_trips():
    sample = json.loads(json.dumps(EXPORT))
    sample["agents"][0]["blast_radius"] = {"per_minute": 5, "per_hour": 100}
    assert rt.check(sample) == []
    assert rt.canonical(sample)["agents"][0]["blast_radius"] == {"per_minute": 5, "per_hour": 100}
