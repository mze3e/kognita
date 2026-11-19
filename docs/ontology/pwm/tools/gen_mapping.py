"""Write mapping/kognita-pwm-map.json: every Kognita vocabulary value and every policy-pack policy mapped to PWM IRIs.

    python3 tools/gen_mapping.py [--check]
"""
import json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rdflib import Graph, URIRef
from rdflib.namespace import SKOS
from kognita_enums import CASES, BASE, SRC, PWM as NS

PWMS = str(NS)
OUT = BASE / "mapping/kognita-pwm-map.json"
# Packs whose policies are generic access-control templates. PWM models business records and their
# governance, not the templates, so they have no PWM terms.
TEMPLATE_PACKS = {"role-based-access", "geo-fencing", "data-classification", "time-gated-access", "two-signature-approval-chains"}

def build():
    ont = Graph().parse(BASE / "pwm.ttl")
    vocab = []
    for name, (scheme, values, _extra) in sorted(CASES.items()):
        by_notation = {str(ont.value(c, SKOS.notation)): c for c in ont.subjects(SKOS.inScheme, URIRef(PWMS + scheme))}
        for value in sorted(values()):
            vocab.append({"kognita": name, "value": value, "scheme": PWMS + scheme, "concept": str(by_notation[value])})
    cites, locator, citation = URIRef(PWMS + "cites"), URIRef(PWMS + "locator"), URIRef(PWMS + "citation")
    by_policy = {}
    for term, c in ont.subject_objects(citation):
        if ont.value(c, cites) == URIRef(PWMS + "ref-KOG-PB-PACK"):
            for policy in re.findall(r"[a-z]+(?:-[a-z]+)*", str(ont.value(c, locator))):
                by_policy.setdefault(policy, set()).add(str(term))
    packs = []
    for pack_dir in sorted((SRC / "packs/data").iterdir()):
        if not (pack_dir / "pack.toml").exists():
            continue
        for policy_file in sorted((pack_dir / "policy").glob("*.yaml")):
            if policy_file.name.endswith(".scenarios.yaml"):
                continue
            policy = policy_file.stem
            terms = sorted(by_policy.get(policy, []))
            packs.append({"pack": pack_dir.name, "policy": policy, "pwm_terms": terms,
                          "disposition": "mapped" if terms else ("template: PWM models records, not templates" if pack_dir.name in TEMPLATE_PACKS else "unmapped")})
    return {"generated_by": "tools/gen_mapping.py", "pwm_namespace": PWMS, "vocabulary": vocab, "policy_packs": packs}

if __name__ == "__main__":
    text = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        sys.exit(0 if OUT.exists() and OUT.read_text() == text else 1)
    OUT.write_text(text)
    data = json.loads(text)
    print(len(data["vocabulary"]), "vocabulary values;", len(data["policy_packs"]), "policies;",
          sum(1 for p in data["policy_packs"] if p["disposition"] == "unmapped"), "unmapped")
