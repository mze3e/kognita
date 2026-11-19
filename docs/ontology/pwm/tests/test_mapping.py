"""The Kognita-to-PWM mapping table and the AIRG paragraph map are complete and current."""
import json
import subprocess
import sys
from pathlib import Path

from rdflib import Graph, URIRef

BASE = Path(__file__).resolve().parents[1]
MAP = json.loads((BASE / "mapping/kognita-pwm-map.json").read_text())
ONT = Graph().parse(BASE / "pwm.ttl")


def run(script, *args):
    return subprocess.run([sys.executable, str(BASE / "tools" / script), *args], capture_output=True, text=True)


def test_mapping_table_is_current():
    assert run("gen_mapping.py", "--check").returncode == 0, "run python3 tools/gen_mapping.py"


def test_every_vocabulary_value_maps_to_an_existing_concept():
    assert MAP["vocabulary"]
    for entry in MAP["vocabulary"]:
        assert (URIRef(entry["concept"]), None, None) in ONT, entry


def test_every_private_banking_policy_maps_to_pwm_terms():
    policies = [p for p in MAP["policy_packs"] if p["pack"] == "private-banking"]
    assert len(policies) >= 15
    assert [p["policy"] for p in policies if not p["pwm_terms"]] == []


def test_no_policy_is_left_unmapped_without_a_reason():
    assert [p for p in MAP["policy_packs"] if p["disposition"] == "unmapped"] == []


def test_airg_map_is_current_and_complete():
    assert run("gen_airg_map.py", "--check").returncode == 0, "run python3 tools/gen_airg_map.py"
    text = (BASE / "AIRG_MAP.md").read_text()
    assert "no PWM term cites this paragraph yet |" not in text
