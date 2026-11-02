"""Build the Private Wealth Management Ontology artefacts.

    python3 tools/build.py [--site PATH]

1. Concatenates src/*.ttl into pwm.ttl.
2. Reads pwm.ttl and pwm-shapes.ttl.
3. Writes SPEC.md, the human-readable specification.
4. With --site, writes the TypeScript data module the website page renders.

Run from the ontology folder (docs/ontology/pwm).
"""
import argparse
import json
import re
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

PWM = "https://ahmedmuzammil.com/ontology/pwm#"
SH = Namespace("http://www.w3.org/ns/shacl#")
P = lambda x: URIRef(PWM + x)
SITE = "https://ahmedmuzammil.com/ontology/pwm"

SOURCES = [
    ("https://spec.edmcouncil.org/fibo/ontology/", "FIBO"),
    ("https://www.omg.org/spec/Commons/", "OMG Commons"),
    ("http://www.w3.org/ns/prov#", "PROV-O"),
    ("https://w3id.org/dpv/ai#", "DPV-AI"),
    ("https://w3id.org/dpv#", "DPV"),
    ("http://purl.org/dc/terms/", "DCMI"),
    ("http://www.w3.org/2001/XMLSchema#", "XSD"),
    ("http://www.w3.org/2004/02/skos/core#", "SKOS"),
]
MARK_ORDER = ["ONT", "ISO", "REG", "KOG", "AM", "NEW"]

STATUS = (
    "Draft for comment. Version 0.1 is the first public draft. Terms, code lists and rules may change "
    "before 1.0, and every change will be recorded in the change log. It is not legal advice: the "
    "regulatory citations show where a term comes from, not how a rule applies to a given firm."
)
SCOPE_IN = [
    "Parties, roles and the client relationship, including trusts, foundations and other legal arrangements",
    "Due diligence, risk rating, source of wealth and funds, screening and tax residency",
    "Investor classification, risk profile, knowledge and experience, suitability and appropriateness",
    "Mandates, investment restrictions and fees",
    "Accounts, portfolios, holdings and valuations",
    "The product shelf, product risk, target market, cross-border rules, offering documents and the house view",
    "The client journey: origination, recommendation, final decision and client communication",
    "AI governance: the use-case inventory, risk materiality, accountability, agents, tools, delegated authority and autonomy",
    "Information governance, facts and claims, policy decisions and hash-chained evidence",
]
SCOPE_OUT = [
    "Orders, execution and the trade lifecycle; corporate actions",
    "Performance and attribution methods (only the provenance of figures is modelled)",
    "Tax reporting detail (CRS and FATCA reporting records), lending beyond the Lombard facility",
    "Sustainability preferences in suitability (planned for 0.2)",
    "Any institution's thresholds, scales, product shelf or internal names: these belong in an extension",
]
PRINCIPLES = [
    ("Reuse before inventing",
     "Business terms specialise or map to FIBO first (Release before Provisional), then OMG Commons, W3C PROV-O and DPV. A term is defined here only when none fits, and it is marked NEW."),
    ("Declare, don't import",
     "External terms are declared with their label, source and FIBO maturity instead of importing whole ontologies, so the file loads on its own. Specialisation uses rdfs:subClassOf; looser alignment uses SKOS exact, close and related matches."),
    ("Every term cites its grounds",
     "Each class, property and code list carries at least one citation with a locator: an article, paragraph, roadmap item or slide. The reference register records how each source was checked."),
    ("Logic in OWL, codes in SKOS, rules in SHACL",
     "The ontology stays within OWL 2 DL with light axioms that an extension can build on. Code lists are SKOS concept schemes. Data rules are SHACL shapes kept in a separate file."),
    ("Context speed and classification",
     "Each class states how fast its facts change and the data classification it carries by default. Fast facts, such as positions and prices, are looked up live and are never cached or embedded for AI."),
    ("Accountability stays with people",
     "Accountability roles are played by named people. A shape rejects any record that names an AI agent in one."),
    ("Generic core, firm extension",
     "No institution's data, thresholds or names appear here. A firm adds them in its own ontology that imports PWM."),
]
CHANGELOG = [("0.1", "2026-11-02", "First public draft: 10 modules, reference register, SHACL shapes, example data and negative tests.")]

EXTENSION_TTL = """@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix pwm:  <https://ahmedmuzammil.com/ontology/pwm#> .
@prefix pwmsh: <https://ahmedmuzammil.com/ontology/pwm/shapes#> .
@prefix firm: <https://example.org/firm/ontology#> .

<https://example.org/firm/ontology> a owl:Ontology ;
    owl:imports <https://ahmedmuzammil.com/ontology/pwm/0.1> .

# The firm's own risk scale, in place of the illustrative one
firm:RiskScale a skos:ConceptScheme ; skos:prefLabel "Firm risk scale"@en .
firm:R1 a skos:Concept ; skos:inScheme firm:RiskScale ; skos:notation "R1" ;
    skos:prefLabel "capital preservation"@en .

# Additional bindings do not remove the defaults; see the separate shapes below
pwm:riskProfileLevel pwm:valuesFrom firm:RiskScale .
pwm:productRiskLevel pwm:valuesFrom firm:RiskScale .
pwm:targetRiskProfile pwm:valuesFrom firm:RiskScale .

# A firm product type specialising a PWM class
firm:StructuredNote a owl:Class ;
    rdfs:subClassOf pwm:Product ;
    rdfs:label "structured note"@en ."""


EXTENSION_SHAPES_TTL = """@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix pwm: <https://ahmedmuzammil.com/ontology/pwm#> .
@prefix pwmsh: <https://ahmedmuzammil.com/ontology/pwm/shapes#> .
@prefix firm: <https://example.org/firm/ontology#> .
pwmsh:CodeList_riskProfileLevel sh:deactivated true .
pwmsh:CodeList_productRiskLevel sh:deactivated true .
pwmsh:CodeList_targetRiskProfile sh:deactivated true .
firm:RiskCodes a sh:NodeShape ;
    sh:targetSubjectsOf pwm:riskProfileLevel, pwm:productRiskLevel, pwm:targetRiskProfile ;
    sh:property [ sh:path pwm:riskProfileLevel ; sh:in ( firm:R1 ) ],
                [ sh:path pwm:productRiskLevel ; sh:in ( firm:R1 ) ],
                [ sh:path pwm:targetRiskProfile ; sh:in ( firm:R1 ) ] ."""


def lit(g, s, p):
    vals = list(g.objects(s, p))
    for v in vals:
        if isinstance(v, Literal) and v.language == "en":
            return str(v)
    return str(vals[0]) if vals else None


def lits(g, s, p):
    return sorted(str(v) for v in g.objects(s, p))


def local(iri):
    s = str(iri)
    return re.split(r"[#/]", s)[-1]


def humanise(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).lower()


def build(base: Path, site_path):
    src = sorted((base / "src").glob("*.ttl"))
    text = "".join(f.read_text(encoding="utf-8") for f in src)
    (base / "pwm.ttl").write_text(text, encoding="utf-8")
    order = {}
    for m in re.finditer(r"^pwm(?:ref)?:([A-Za-z0-9_-]+)\s+a\s", text, flags=re.M):
        order.setdefault(m.group(1), m.start())

    g = Graph().parse(base / "pwm.ttl")
    write_code_list_shapes(g, base / "pwm-shapes.ttl")
    sg = Graph().parse(base / "pwm-shapes.ttl")
    ont = URIRef(SITE)

    def is_pwm(x):
        return isinstance(x, URIRef) and str(x).startswith(PWM)

    def pid(x):
        return str(x)[len(PWM):]

    def label(x):
        return lit(g, x, RDFS.label) or lit(g, x, SKOS.prefLabel) or humanise(local(x))

    def link(x):
        if is_pwm(x):
            return {"id": pid(x), "label": label(x)}
        iri = str(x)
        source = next((n for ns, n in SOURCES if iri.startswith(ns)), "")
        out = {"iri": iri, "label": label(x) if (x, RDFS.label, None) in g else (
            "SKOS concept" if x == SKOS.Concept else local(x) if source == "XSD" else humanise(local(x))), "source": source}
        mat = lit(g, x, P("fiboMaturity"))
        if mat:
            out["maturity"] = mat
        return out

    def citations(x):
        out = []
        for c in g.objects(x, P("citation")):
            r = g.value(c, P("cites"))
            out.append({"ref": pid(r)[4:], "locator": lit(g, c, P("locator")) or ""})
        refmark = lambda c: MARK_ORDER.index(refs[c["ref"]]["mark"])
        return sorted(out, key=lambda c: (refmark(c), c["ref"], c["locator"]))

    # References
    refs = {}
    for r in g.subjects(RDF.type, P("Reference")):
        st = g.value(r, P("sourceType"))
        refs[pid(r)[4:]] = {
            "id": pid(r)[4:],
            "short": lit(g, r, DCTERMS.alternative),
            "title": lit(g, r, DCTERMS.title),
            "publisher": lit(g, r, DCTERMS.publisher) or lit(g, r, DCTERMS.creator),
            "date": lit(g, r, DCTERMS.date),
            "url": (lits(g, r, RDFS.seeAlso) or [None])[0],
            "mark": lit(g, st, SKOS.notation),
            "jurisdiction": lit(g, r, P("jurisdiction")),
            "status": lit(g, r, P("verificationStatus")),
            "note": lit(g, r, SKOS.note),
        }
    ref_list = sorted(refs.values(), key=lambda r: (MARK_ORDER.index(r["mark"]), order.get("ref-" + r["id"], 0)))

    marks = []
    for st in g.subjects(RDF.type, P("SourceType")):
        marks.append({"mark": lit(g, st, SKOS.notation), "label": label(st), "definition": lit(g, st, SKOS.definition)})
    marks.sort(key=lambda m: MARK_ORDER.index(m["mark"]))
    speeds = [{"id": pid(s), "label": label(s), "definition": lit(g, s, SKOS.definition)}
              for s in sorted(g.subjects(RDF.type, P("ContextSpeed")), key=lambda s: order.get(pid(s), 0))]

    def restriction(b):
        prop = g.value(b, OWL.onProperty)
        for kind, word in ((OWL.someValuesFrom, "some"), (OWL.allValuesFrom, "only"), (OWL.hasValue, "value")):
            v = g.value(b, kind)
            if v is not None:
                comp = g.value(v, OWL.complementOf) if isinstance(v, BNode) else None
                filler = {**link(comp), "label": "anything except " + label(comp)} if comp is not None else link(v)
                return {"property": link(prop), "quantifier": word, "filler": filler}
        return None

    domain_of, used_by = {}, {}
    for p_, d in g.subject_objects(RDFS.domain):
        if is_pwm(p_) and is_pwm(d):
            domain_of.setdefault(pid(d), []).append(pid(p_))
    for p_, sch in g.subject_objects(P("valuesFrom")):
        used_by.setdefault(pid(sch), []).append(pid(p_))

    def term(x, kind):
        t = {"id": pid(x), "kind": kind, "label": label(x),
             "altLabels": lits(g, x, SKOS.altLabel),
             "definition": lit(g, x, SKOS.definition) or "",
             "citations": citations(x)}
        for key, p_ in (("scopeNote", SKOS.scopeNote), ("example", SKOS.example), ("editorialNote", SKOS.editorialNote)):
            v = lit(g, x, p_)
            if v:
                t[key] = v
        supers, restr = [], []
        for o in g.objects(x, RDFS.subClassOf if kind == "class" else RDFS.subPropertyOf):
            if isinstance(o, BNode):
                r = restriction(o)
                if r:
                    restr.append(r)
            else:
                supers.append(link(o))
        t["supers"] = sorted(supers, key=lambda l: l["label"])
        if restr:
            t["restrictions"] = restr
        matches = []
        for rel, p_ in (("exact match", SKOS.exactMatch), ("close match", SKOS.closeMatch), ("related match", SKOS.relatedMatch)):
            for o in g.objects(x, p_):
                matches.append({"relation": rel, "target": link(o)})
        t["matches"] = matches
        if kind != "class":
            t["domain"] = [link(o) for o in g.objects(x, RDFS.domain)]
            t["range"] = [link(o) for o in g.objects(x, RDFS.range)]
            vf = g.value(x, P("valuesFrom"))
            if vf is not None:
                t["valuesFrom"] = link(vf)
        else:
            sp = g.value(x, P("contextSpeed"))
            if sp is not None:
                t["contextSpeed"] = label(sp)
            dc = g.value(x, P("defaultClassification"))
            if dc is not None:
                t["classification"] = pid(dc)
            if pid(x) in domain_of:
                t["properties"] = sorted(domain_of[pid(x)], key=lambda i: order.get(i, 0))
        return t

    def scheme(x):
        t = term(x, "code list")
        t["definition"] = lit(g, x, SKOS.definition) or t.pop("scopeNote", "")
        t["usedBy"] = sorted(used_by.get(pid(x), []))
        cs = []
        for c in sorted(g.subjects(SKOS.inScheme, x), key=lambda c: order.get(pid(c), 0)):
            cd = {"id": pid(c), "label": label(c), "definition": lit(g, c, SKOS.definition) or ""}
            for key, p_ in (("notation", SKOS.notation), ("scopeNote", SKOS.scopeNote)):
                v = lit(g, c, p_)
                if v:
                    cd[key] = v
            alt = lits(g, c, SKOS.altLabel)
            if alt:
                cd["altLabels"] = alt
            b = g.value(c, SKOS.broader)
            if b is not None:
                cd["broader"] = pid(b)
            m = [{"relation": "close match", "target": link(o)} for o in g.objects(c, SKOS.closeMatch)]
            if m:
                cd["matches"] = m
            cit = citations(c)
            if cit:
                cd["citations"] = cit
            cs.append(cd)
        # Parents before their children.
        tops = [c for c in cs if "broader" not in c]
        ordered = []
        for c in tops:
            ordered.append(c)
            ordered += [k for k in cs if k.get("broader") == c["id"]]
        t["concepts"] = ordered + [c for c in cs if c not in ordered]
        return t

    modules = []
    for m in sorted(g.subjects(RDF.type, P("Module")), key=lambda m: int(g.value(m, P("moduleOrder")))):
        members = [s for s in g.subjects(P("inModule"), m) if is_pwm(s)]
        bykind = lambda t: sorted((s for s in members if (s, RDF.type, t) in g), key=lambda s: order.get(pid(s), 0))
        modules.append({
            "id": pid(m), "label": label(m), "definition": lit(g, m, SKOS.definition),
            "classes": [term(s, "class") for s in bykind(OWL.Class)],
            "objectProperties": [term(s, "object property") for s in bykind(OWL.ObjectProperty)],
            "datatypeProperties": [term(s, "datatype property") for s in bykind(OWL.DatatypeProperty)],
            "codeLists": [scheme(s) for s in bykind(SKOS.ConceptScheme)],
        })

    meta_terms = []
    for cls in ("Module", "Reference", "Citation", "SourceType", "ContextSpeed"):
        meta_terms.append({"id": cls, "kind": "class", "label": label(P(cls)), "definition": lit(g, P(cls), SKOS.definition)})
    for a in sorted((s for s in g.subjects(RDF.type, OWL.AnnotationProperty) if is_pwm(s)), key=lambda s: order.get(pid(s), 0)):
        meta_terms.append({"id": pid(a), "kind": "annotation property", "label": label(a), "definition": lit(g, a, SKOS.definition)})

    # SHACL rules
    def describe(ps):
        path = sg.value(ps, SH.path)
        bits = []
        mn, mx = sg.value(ps, SH.minCount), sg.value(ps, SH.maxCount)
        if mn is not None and mx is not None and int(mn) == int(mx) == 1:
            bits.append("exactly one")
        else:
            if mn is not None:
                bits.append(f"at least {mn}")
            if mx is not None:
                bits.append(f"at most {mx}")
        for p_, word in ((SH.datatype, "of type"), (SH["class"], "an instance of"), (SH.pattern, "matching")):
            v = sg.value(ps, p_)
            if v is not None:
                bits.append(f"{word} {local(v) if p_ != SH.pattern else v}")
        inn = sg.value(ps, SH["in"])
        if inn is not None:
            bits.append("one of " + ", ".join(label(v) for v in Collection(sg, inn)))
        return f"{label(path) if path is not None else 'value'}: " + ", ".join(bits or ["checked"])

    rules = []
    for s in sg.subjects(RDF.type, SH.NodeShape):
        targets = [link(t) for t in sg.objects(s, SH.targetClass)]
        targets += [{**link(t), "subjectsOf": True} for t in sg.objects(s, SH.targetSubjectsOf)]
        if not targets:
            continue
        checks = []
        for ps in sg.objects(s, SH.property):
            msg = sg.value(ps, SH.message)
            checks.append(str(msg) if msg else describe(ps))
        node_msg = sg.value(s, SH.message)
        if node_msg:
            checks.append(str(node_msg))
        for c in sg.objects(s, SH.sparql):
            checks.append(str(sg.value(c, SH.message)))
        rules.append({
            "id": local(s), "label": str(sg.value(s, RDFS.label)), "targets": targets,
            "sources": sorted(local(r)[4:] for r in sg.objects(s, DCTERMS.source)),
            "checks": checks, "comment": lit(sg, s, RDFS.comment),
        })
    generated = [r for r in rules if r["id"].startswith("CodeList_")]
    rules = [r for r in rules if not r["id"].startswith("CodeList_")]
    if generated:
        rules.append({
            "id": "CodeListShapes", "label": "Code lists", "targets": [], "sources": ["W3C-SKOS"],
            "checks": [f"Each of the {len(generated)} coded properties takes its values only from the code list it is bound to with "
                       "pwm:valuesFrom. The shapes, named pwmsh:CodeList_ followed by the property name, are generated from the bindings. "
                       "List replacement requires firm shapes and review of handwritten constraints and code-dependent rules; see README.md."],
            "comment": None,
        })
    shape_order = {m.group(1): m.start() for m in re.finditer(r"^pwmsh:(\w+) a sh:NodeShape", (base / "pwm-shapes.ttl").read_text(encoding="utf-8"), re.M)}
    rules.sort(key=lambda r: shape_order.get(r["id"], 10**9))

    allterms = [t for m in modules for k in ("classes", "objectProperties", "datatypeProperties", "codeLists") for t in m[k]]
    counts = {
        "classes": sum(len(m["classes"]) for m in modules),
        "objectProperties": sum(len(m["objectProperties"]) for m in modules),
        "datatypeProperties": sum(len(m["datatypeProperties"]) for m in modules),
        "codeLists": sum(len(m["codeLists"]) for m in modules),
        "concepts": sum(len(t["concepts"]) for m in modules for t in m["codeLists"]),
        "references": len(refs),
        "citations": sum(1 for _ in g.subject_objects(P("citation"))),
        "rules": len(rules) - (1 if generated else 0),
        "codeListRules": len(generated),
        "externalTerms": sum(1 for s in g.subjects(RDFS.isDefinedBy, None) if not is_pwm(s) and (s, RDF.type, None) in g),
    }
    fibo_terms = {t["iri"]: t for x in allterms for t in [*x.get("supers", []), *(m["target"] for m in x.get("matches", []))]
                  if t.get("source") == "FIBO"}
    meta = {
        "title": lit(g, ont, DCTERMS.title), "short": lit(g, ont, DCTERMS.alternative),
        "version": str(g.value(ont, OWL.versionInfo)), "ontologyIri": SITE, "versionIri": str(g.value(ont, OWL.versionIRI)),
        "namespace": PWM, "shapesIri": f"{SITE}/shapes", "created": lit(g, ont, DCTERMS.created), "modified": lit(g, ont, DCTERMS.modified),
        "creator": lit(g, ont, DCTERMS.creator), "license": str(g.value(ont, DCTERMS.license)), "description": lit(g, ont, DCTERMS.description),
        "status": STATUS, "fiboAligned": len(fibo_terms),
    }
    data = {"meta": meta, "counts": counts, "principles": [{"title": a, "body": b} for a, b in PRINCIPLES],
            "scopeIn": SCOPE_IN, "scopeOut": SCOPE_OUT, "marks": marks, "speeds": speeds, "modules": modules,
            "metaTerms": meta_terms, "rules": rules, "references": ref_list, "extensionExample": EXTENSION_TTL,
            "changelog": [{"version": v, "date": d, "note": n} for v, d, n in CHANGELOG]}

    (base / "SPEC.md").write_text(spec_md(data, refs), encoding="utf-8")
    if site_path:
        Path(site_path).write_text(site_ts(data), encoding="utf-8")
    return data


BEGIN = "# BEGIN GENERATED CODE-LIST SHAPES"
END = "# END GENERATED CODE-LIST SHAPES"


def write_code_list_shapes(g, shapes_path):
    """One shape per coded property: values must come from its bound code list.

    The concept IRIs are written into the shape, so the check works without the
    ontology's concept data in the data graph.
    """
    lines = []
    for prop in sorted(g.subjects(P("valuesFrom"), None)):
        sch = g.value(prop, P("valuesFrom"))
        concepts = sorted(g.subjects(SKOS.inScheme, sch))
        name = str(prop)[len(PWM):]
        sname = str(sch)[len(PWM):]
        members = " ".join(f"pwm:{str(c)[len(PWM):]}" for c in concepts)
        lines += [
            f"pwmsh:CodeList_{name} a sh:NodeShape ;",
            f"    rdfs:label {Literal('Code list: ' + lit(g, prop, RDFS.label), lang='en').n3()} ;",
            f"    sh:targetSubjectsOf pwm:{name} ;",
            f"    dcterms:source pwmref:W3C-SKOS ;",
            f"    sh:property [ sh:path pwm:{name} ; sh:in ( {members} ) ;",
            f"        sh:message {Literal(f'Use a code from {lit(g, sch, SKOS.prefLabel)} (pwm:{sname}). See README.md for replacement requirements.').n3()} ] .",
            "",
        ]
    text = shapes_path.read_text(encoding="utf-8")
    head, rest = text.split(BEGIN, 1)
    tail = rest.split(END, 1)[1]
    block = BEGIN + " (tools/build.py writes this block from pwm:valuesFrom; do not edit)\n\n" + "\n".join(lines) + END
    shapes_path.write_text(head + block + tail, encoding="utf-8")


# --- Markdown -----------------------------------------------------------------------------

def md_link(l):
    if "id" in l:
        return f"[{l['label']}](#{l['id']})"
    extra = l.get("source", "")
    if l.get("maturity"):
        extra += f", {l['maturity']}"
    return f"[{l['label']}]({l['iri']}) ({extra})" if extra else f"[{l['label']}]({l['iri']})"


def md_cites(cits, refs):
    return "; ".join(f"**{refs[c['ref']]['mark']}** [{refs[c['ref']]['short']}](#ref-{c['ref']}) {c['locator']}".strip() for c in cits)


def cell(s):
    return (s or "").replace("|", "\\|").replace("\n", " ")


def spec_md(d, refs):
    m, c = d["meta"], d["counts"]
    out = [f"# {m['title']} ({m['short']}) {m['version'].split()[0]}", "",
           "> Generated from the Turtle sources by `tools/build.py`. Do not edit by hand.", "",
           "| | |", "|---|---|",
           f"| Version | {m['version']}, {m['modified']} |",
           f"| Ontology IRI | {m['ontologyIri']} |",
           f"| Version IRI | {m['versionIri']} |",
           f"| Namespace | `pwm:` {m['namespace']} |",
           f"| Shapes | {m['shapesIri']} (`pwm-shapes.ttl`) |",
           f"| Licence | [CC BY-SA 4.0]({m['license']}) |",
           f"| Author | {m['creator']} |",
           f"| Web page | {m['ontologyIri']} |", "",
           "## Abstract", "", m["description"], "",
           "## Status", "", m["status"], "",
           f"Size: {c['classes']} classes, {c['objectProperties']} object properties, {c['datatypeProperties']} datatype properties, "
           f"{c['codeLists']} code lists with {c['concepts']} codes, {c['rules']} data rules plus {c['codeListRules']} generated code-list rules, {c['references']} references and "
           f"{c['citations']} citations. {m['fiboAligned']} FIBO terms are specialised or mapped.", "",
           "## Scope", "", "In scope:", ""] + [f"- {s}" for s in d["scopeIn"]] + ["", "Not in 0.1:", ""] + \
          [f"- {s}" for s in d["scopeOut"]] + ["", "## Design principles", ""] + \
          [f"{i}. **{p['title']}.** {p['body']}" for i, p in enumerate(d["principles"], 1)] + \
          ["", "## How references are marked", "",
           "Every term lists its citations. Each citation carries a mark for the kind of source, the source, and a locator inside it.", "",
           "| Mark | Source type | Meaning |", "|---|---|---|"] + \
          [f"| **{k['mark']}** | {k['label']} | {cell(k['definition'])} |" for k in d["marks"]] + \
          ["", "Each reference in the register has a verification status. **verified**: checked against the source file, official text or regulator page while drafting this version. **instrument-level**: cited from the published instrument; article and paragraph numbers are to be re-checked before formal use.", "",
           "FIBO terms show their maturity, Release or Provisional, as read from the FIBO file that defines them.", "",
           "Classes carry a context speed:", ""] + [f"- **{s['label']}**: {s['definition']}" for s in d["speeds"]] + \
          ["", "## Modules", "", "| # | Module | Covers | Classes | Properties | Code lists |", "|---|---|---|---|---|---|"]
    for i, mod in enumerate(d["modules"]):
        out.append(f"| {i} | [{mod['label']}](#{mod['id']}) | {cell(mod['definition'])} | {len(mod['classes'])} | "
                   f"{len(mod['objectProperties']) + len(mod['datatypeProperties'])} | {len(mod['codeLists'])} |")
    for i, mod in enumerate(d["modules"]):
        out += ["", f'<a id="{mod["id"]}"></a>', "", f"## Module {i}: {mod['label']}", "", mod["definition"]]
        if mod["classes"]:
            out += ["", "### Classes"]
        for t in mod["classes"]:
            out += ["", f'<a id="{t["id"]}"></a>', f"#### {t['label']} (`pwm:{t['id']}`)", "", t["definition"]]
            if t.get("scopeNote"):
                out += ["", f"*Note:* {t['scopeNote']}"]
            lines = []
            if t["supers"]:
                lines.append("**Specialises:** " + ", ".join(md_link(l) for l in t["supers"]))
            for r in t.get("restrictions", []):
                lines.append(f"**Restriction:** {md_link(r['property'])} {r['quantifier']} {md_link(r['filler'])}")
            for mt in t["matches"]:
                lines.append(f"**{mt['relation'].capitalize()}:** {md_link(mt['target'])}")
            if t["altLabels"]:
                lines.append("**Also called:** " + ", ".join(t["altLabels"]))
            ctx = [x for x in (t.get("contextSpeed") and f"{t['contextSpeed']} context", t.get("classification") and f"default classification {t['classification']}") if x]
            if ctx:
                lines.append("**Handling:** " + ", ".join(ctx))
            if t.get("properties"):
                lines.append("**Properties:** " + ", ".join(f"[`{p}`](#{p})" for p in t["properties"]))
            lines.append("**References:** " + md_cites(t["citations"], refs))
            out += [""] + [f"- {l}" for l in lines]
        props = mod["objectProperties"] + mod["datatypeProperties"]
        if props:
            out += ["", "### Properties", "", "| Property | Domain → range | Definition | References |", "|---|---|---|---|"]
            for t in props:
                dom = ", ".join(md_link(l) for l in t["domain"]) or "any"
                rng = md_link(t["valuesFrom"]) if t.get("valuesFrom") else (", ".join(md_link(l) for l in t["range"]) or "any")
                extra = (" Sub-property of " + ", ".join(md_link(l) for l in t["supers"]) + ".") if t["supers"] else ""
                extra += "".join(f" {mt['relation'].capitalize()}: {md_link(mt['target'])}." for mt in t["matches"])
                note = f" *{t['scopeNote']}*" if t.get("scopeNote") else ""
                out.append(f'| <a id="{t["id"]}"></a>`pwm:{t["id"]}` {t["label"]} | {cell(dom)} → {cell(rng)} | '
                           f"{cell(t['definition'] + note + extra)} | {cell(md_cites(t['citations'], refs))} |")
        if mod["codeLists"]:
            out += ["", "### Code lists"]
        for t in mod["codeLists"]:
            out += ["", f'<a id="{t["id"]}"></a>', f"#### {t['label']} (`pwm:{t['id']}`)", "", t["definition"]]
            if t["usedBy"]:
                out += ["", "Used by: " + ", ".join(f"[`{p}`](#{p})" for p in t["usedBy"])]
            out += ["", "References: " + md_cites(t["citations"], refs), "", "| Code | Concept | Definition |", "|---|---|---|"]
            for k in t["concepts"]:
                indent = "↳ " if "broader" in k else ""
                extra = "".join(f" {mt['relation'].capitalize()}: {md_link(mt['target'])}." for mt in k.get("matches", []))
                alt = f" Also: {', '.join(k['altLabels'])}." if k.get("altLabels") else ""
                out.append(f'| {k.get("notation", "")} | <a id="{k["id"]}"></a>{indent}{k["label"]} (`pwm:{k["id"]}`) | {cell(k["definition"] + alt + extra)} |')
    out += ["", '<a id="rules"></a>', "", "## Data rules (SHACL)", "",
            "`pwm-shapes.ttl` holds the rules below. Validate data together with `pwm.ttl` and RDFS inference:", "",
            "```sh", "pyshacl -s pwm-shapes.ttl -e pwm.ttl -i rdfs your-data.ttl", "```", "",
            "A firm can switch a rule off in its own shapes graph with `sh:deactivated true`, and add its own thresholds there."]
    for r in d["rules"]:
        tg = ", ".join(("subjects of " if t.get("subjectsOf") else "") + md_link(t) for t in r["targets"]) or "every coded property"
        out += ["", f'<a id="{r["id"]}"></a>', f"### {r['label']}", "", f"Applies to {tg}. Sources: " +
                ", ".join(f"[{refs[s]['short']}](#ref-{s})" for s in r["sources"]) + ".", ""] + [f"- {x}" for x in r["checks"]]
        if r.get("comment"):
            out += ["", f"*{r['comment']}*"]
    out += ["", "## Extending PWM for an institution", "",
            "Keep institution detail in a separate ontology that imports PWM by its version IRI, which resolves to the Turtle file. Specialise PWM classes, add the firm's code lists, and bind coded properties to them with `pwm:valuesFrom`. Keep confidential content out of public repositories.", "",
            "Save as `firm.ttl`:", "", "```turtle", d["extensionExample"], "```", "",
            "Save separately as `firm-shapes.ttl`:", "", "```turtle", EXTENSION_SHAPES_TTL, "```", "",
            "Load both shapes graphs together:", "", "```python",
            "from rdflib import Graph", "from pyshacl import validate",
            "shapes = Graph().parse('pwm-shapes.ttl').parse('firm-shapes.ttl')",
            "ontology = Graph().parse('pwm.ttl').parse('firm.ttl')",
            "conforms, report, text = validate(Graph().parse('your-data.ttl'),",
            "    shacl_graph=shapes, ont_graph=ontology, inference='rdfs')",
            "print(text)", "```", "",
            "Other code lists may require changes to handwritten constraints and code-dependent rules too; see README.md.", "",
            '<a id="references"></a>', "", "## Reference register", "",
            "| Ref | Mark | Title | Publisher | Date | Jurisdiction | Status | Note |", "|---|---|---|---|---|---|---|---|"]
    for r in d["references"]:
        title = f"[{cell(r['title'])}]({r['url']})" if r["url"] else cell(r["title"])
        out.append(f'| <a id="ref-{r["id"]}"></a>{r["short"]} | {r["mark"]} | {title} | {cell(r["publisher"])} | {r["date"] or ""} | '
                   f'{r["jurisdiction"] or ""} | {r["status"]} | {cell(r["note"])} |')
    out += ["", "## Annotation vocabulary", "", "| Term | Kind | Definition |", "|---|---|---|"]
    out += [f'| <a id="{t["id"]}"></a>`pwm:{t["id"]}` | {t["kind"]} | {cell(t["definition"])} |' for t in d["metaTerms"]]
    out += ["", "## Change log", "", "| Version | Date | Change |", "|---|---|---|"]
    out += [f"| {x['version']} | {x['date']} | {x['note']} |" for x in d["changelog"]]
    return "\n".join(out) + "\n"


def site_ts(d):
    body = json.dumps(d, ensure_ascii=False, indent=1)
    return (
        "// Generated by docs/ontology/pwm/tools/build.py in github.com/mze3e/kognita.\n"
        "// Do not edit by hand: change the Turtle sources there and rebuild.\n\n"
        "export type PwmLink = { id?: string; iri?: string; label: string; source?: string; maturity?: string; subjectsOf?: boolean };\n"
        "export type PwmCitation = { ref: string; locator: string };\n"
        "export type PwmMatch = { relation: string; target: PwmLink };\n"
        "export type PwmConcept = { id: string; label: string; definition: string; notation?: string; scopeNote?: string; altLabels?: string[]; broader?: string; matches?: PwmMatch[]; citations?: PwmCitation[] };\n"
        "export type PwmTerm = { id: string; kind: string; label: string; altLabels: string[]; definition: string; scopeNote?: string; example?: string; editorialNote?: string; citations: PwmCitation[]; supers: PwmLink[]; restrictions?: { property: PwmLink; quantifier: string; filler: PwmLink }[]; matches: PwmMatch[]; domain?: PwmLink[]; range?: PwmLink[]; valuesFrom?: PwmLink; contextSpeed?: string; classification?: string; properties?: string[]; usedBy?: string[]; concepts?: PwmConcept[] };\n"
        "export type PwmModule = { id: string; label: string; definition: string; classes: PwmTerm[]; objectProperties: PwmTerm[]; datatypeProperties: PwmTerm[]; codeLists: PwmTerm[] };\n"
        "export type PwmReference = { id: string; short: string; title: string; publisher: string | null; date: string | null; url: string | null; mark: string; jurisdiction: string | null; status: string; note: string | null };\n"
        "export type PwmRule = { id: string; label: string; targets: PwmLink[]; sources: string[]; checks: string[]; comment: string | null };\n"
        "export type PwmOntology = {\n"
        "  meta: { title: string; short: string; version: string; ontologyIri: string; versionIri: string; namespace: string; shapesIri: string; created: string; modified: string; creator: string; license: string; description: string; status: string; fiboAligned: number };\n"
        "  counts: { classes: number; objectProperties: number; datatypeProperties: number; codeLists: number; concepts: number; references: number; citations: number; rules: number; codeListRules: number; externalTerms: number };\n"
        "  principles: { title: string; body: string }[];\n"
        "  scopeIn: string[];\n  scopeOut: string[];\n"
        "  marks: { mark: string; label: string; definition: string }[];\n"
        "  speeds: { id: string; label: string; definition: string }[];\n"
        "  modules: PwmModule[];\n"
        "  metaTerms: { id: string; kind: string; label: string; definition: string }[];\n"
        "  rules: PwmRule[];\n  references: PwmReference[];\n  extensionExample: string;\n"
        "  changelog: { version: string; date: string; note: string }[];\n};\n\n"
        f"export const pwmOntology: PwmOntology = {body};\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="write the website data module to this path")
    ap.add_argument("--base", default=".")
    a = ap.parse_args()
    data = build(Path(a.base), a.site)
    print("built pwm.ttl, SPEC.md" + (f", {a.site}" if a.site else ""), json.dumps(data["counts"]))
