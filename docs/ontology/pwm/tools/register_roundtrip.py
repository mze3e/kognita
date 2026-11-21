"""Round trip between a Kognita register export and PWM RDF.

    python3 tools/register_roundtrip.py context              write context/pwm-register.jsonld from pwm.ttl
    .venv/bin/python tools/register_roundtrip.py make        (Kognita installed) write examples/kognita-register.json
    python3 tools/register_roundtrip.py check [export.json]  convert to RDF and compare the re-imported register

The context maps the keys of a Kognita register export (use cases, agents and
approved sources) to PWM terms. Object-valued properties are expanded into
typed nodes. The sample export is deterministic: the same store always writes
the same file, so CI can regenerate it and fail on a diff.

This round trip does not claim that a live Kognita register satisfies the AI
inventory profile. Kognita does not yet record inventory fields such as
documentation and approved components, so those properties are omitted.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from datetime import datetime, timezone

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD
from rdflib.namespace import SKOS

BASE = pathlib.Path(__file__).resolve().parents[1]
PWM_NS = "https://ahmedmuzammil.com/ontology/pwm#"
CMNS_ROLE = "https://www.omg.org/spec/Commons/RolesAndCompositions/"
CMNS_PARTY = "https://www.omg.org/spec/Commons/PartiesAndSituations/"
CONTEXT_PATH = BASE / "context/pwm-register.jsonld"
EXPORT_PATH = BASE / "examples/kognita-register.json"
P = Namespace(PWM_NS)
NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

DATA_CLASS = {
    "public": "C0",
    "internal": "C1",
    "confidential": "C2",
    "highly sensitive": "C3",
}
ROLE_TYPES = {
    "business_owner": "BusinessOwner",
    "technology_owner": "TechnologyOwner",
}


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.strip().casefold()).strip("-")
    return text or "unnamed"


def _iso(value) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value)
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def _notation_iri(ont: Graph, scheme: str, notation: str) -> str:
    for concept in ont.subjects(SKOS.inScheme, URIRef(PWM_NS + scheme)):
        if str(ont.value(concept, SKOS.notation)) == str(notation):
            return str(concept)
    raise ValueError(f"no PWM concept in {scheme} with notation {notation!r}")


def build_context(_ont: Graph | None = None) -> dict:
    ctx = {
        "pwm": PWM_NS,
        "rdfs": str(RDFS),
        "xsd": str(XSD),
        "prov": "http://www.w3.org/ns/prov#",
        "cmns-rlcmp": CMNS_ROLE,
        "cmns-pts": CMNS_PARTY,
        "label": "rdfs:label",
        "purpose_node": {"@id": "pwm:hasPurpose", "@type": "@id"},
        "jurisdiction": "pwm:approvedJurisdiction",
        "status": {"@id": "pwm:lifecycleStatus", "@type": "@id"},
        "risk_tier": {"@id": "pwm:riskTier", "@type": "@id"},
        "review_date": {"@id": "pwm:nextReviewDate", "@type": "xsd:dateTime"},
        "data_class": {"@id": "pwm:usesDataClass", "@type": "@id"},
        "model_type": "pwm:modelType",
        "depends_on": {"@id": "pwm:dependsOn", "@type": "@id"},
        "role": {"@id": "pwm:hasAccountabilityRole", "@type": "@id"},
        "played_by": {"@id": "cmns-rlcmp:isPlayedBy", "@type": "@id"},
        "assessment": {"@id": "pwm:hasMaterialityAssessment", "@type": "@id"},
        "assesses": {"@id": "pwm:assesses", "@type": "@id"},
        "agent_name": "pwm:agentIdentifier",
        "registered_for": {"@id": "pwm:registeredForUseCase", "@type": "@id"},
        "tool": {"@id": "pwm:permittedTool", "@type": "@id"},
        "blast_limit": {"@id": "pwm:hasBlastRadiusLimit", "@type": "@id"},
        "limit_dimension": "pwm:limitDimension",
        "limit_maximum": {"@id": "pwm:limitMaximum", "@type": "xsd:decimal"},
        "approved_for": {"@id": "pwm:approvedForUseCase", "@type": "@id"},
        "effective": {"@id": "prov:generatedAtTime", "@type": "xsd:dateTime"},
    }
    return {"@context": ctx}


def _usecase_iri(usecase_id: str) -> str:
    return f"urn:kognita:usecase:{usecase_id}"


def _agent_iri(name: str) -> str:
    return f"urn:kognita:agent:{_slug(name)}"


def _source_iri(source_id: str, version: str) -> str:
    return f"urn:kognita:source:{_slug(source_id)}:{_slug(version)}"


def _person_iri(name: str) -> str:
    return f"urn:kognita:person:{_slug(name)}"


def to_jsonld(export: dict, ont: Graph) -> dict:
    graph = []
    people = {}

    def person(name: str) -> str:
        iri = _person_iri(name)
        if iri not in people:
            people[iri] = {
                "@id": iri,
                "@type": "cmns-pts:Party",
                "label": name,
            }
        return iri

    for entry in export.get("use_cases", []):
        usecase_id = entry["usecase_id"]
        iri = _usecase_iri(usecase_id)
        purpose_iri = iri + ":purpose"
        node = {
            "@id": iri,
            "@type": "pwm:AIUseCase",
            "label": usecase_id,
            "purpose_node": purpose_iri,
        }
        graph.append({"@id": purpose_iri, "@type": "pwm:Purpose", "label": entry["purpose"]})
        if entry.get("jurisdiction"):
            node["jurisdiction"] = entry["jurisdiction"]
        if entry.get("status"):
            node["status"] = _notation_iri(ont, "UseCaseStatusScheme", entry["status"])
        if entry.get("risk_tier") is not None:
            node["risk_tier"] = _notation_iri(ont, "RiskTierScheme", str(entry["risk_tier"]))
        if entry.get("review_date"):
            node["review_date"] = entry["review_date"]
        if entry.get("data_sensitivity"):
            node["data_class"] = _notation_iri(ont, "DataClassificationScheme", DATA_CLASS[entry["data_sensitivity"]])
        if entry.get("model_type"):
            node["model_type"] = entry["model_type"]
        depends = []
        for dep in entry.get("depends_on") or []:
            dep_iri = f"urn:kognita:dependency:{_slug(str(dep))}"
            graph.append({"@id": dep_iri, "@type": "pwm:ModelProvider", "label": str(dep)})
            depends.append(dep_iri)
        if depends:
            node["depends_on"] = depends
        roles = []
        for field, typ in ROLE_TYPES.items():
            name = entry.get(field)
            if not name:
                continue
            role_iri = f"{iri}:role:{field.replace('_', '-')}"
            graph.append({
                "@id": role_iri,
                "@type": f"pwm:{typ}",
                "played_by": person(name),
                "label": field.replace("_", " "),
            })
            roles.append(role_iri)
        if roles:
            node["role"] = roles
        if entry.get("residual_materiality") or entry.get("inherent_materiality"):
            assessment_iri = iri + ":materiality"
            graph.append({
                "@id": assessment_iri,
                "@type": "pwm:RiskMaterialityAssessment",
                "assesses": iri,
                "label": entry.get("residual_materiality") or entry.get("inherent_materiality"),
            })
            node["assessment"] = assessment_iri
        graph.append(node)

    for entry in export.get("agents", []):
        name = entry["name"]
        iri = _agent_iri(name)
        node = {
            "@id": iri,
            "@type": "pwm:AIAgent",
            "agent_name": name,
            "label": name,
        }
        registered = [_usecase_iri(u) for u in entry.get("registered_use_cases") or []]
        if registered:
            node["registered_for"] = registered
        tools = []
        for tool in entry.get("tools") or []:
            tool_iri = f"{iri}:tool:{_slug(tool)}"
            graph.append({"@id": tool_iri, "@type": "pwm:Tool", "label": tool})
            tools.append(tool_iri)
        if tools:
            node["tool"] = tools
        radius = entry.get("blast_radius") or {}
        numeric = [
            (key, radius[key])
            for key in ("per_minute", "per_hour", "max_clients", "max_daily_actions", "max_daily_transactions", "max_value")
            if radius.get(key) is not None
        ]
        if numeric:
            dim, maximum = numeric[0]
            limit_iri = iri + ":limit"
            graph.append({
                "@id": limit_iri,
                "@type": "pwm:BlastRadiusLimit",
                "limit_dimension": dim,
                "limit_maximum": maximum,
            })
            node["blast_limit"] = limit_iri
        graph.append(node)

    for entry in export.get("sources", []):
        iri = _source_iri(entry["source_id"], entry["version"])
        node = {
            "@id": iri,
            "@type": "pwm:ApprovedSource",
            "label": entry["source_id"],
        }
        if entry.get("effective"):
            node["effective"] = entry["effective"]
        approved = [_usecase_iri(u) for u in entry.get("approved_for") or []]
        if approved:
            node["approved_for"] = approved
        graph.append(node)

    graph.extend(people.values())
    return {"@context": build_context(ont)["@context"], "@graph": graph}


def export_to_graph(export: dict, ont: Graph | None = None) -> Graph:
    ont = ont if ont is not None else Graph().parse(BASE / "pwm.ttl")
    g = Graph()
    g.parse(data=json.dumps(to_jsonld(export, ont)), format="json-ld")
    return g


def _literal(graph: Graph, subject, predicate) -> str | None:
    value = graph.value(subject, predicate)
    return None if value is None else str(value)


def _notation(ont: Graph, iri) -> str | None:
    if iri is None:
        return None
    value = ont.value(iri, SKOS.notation)
    return None if value is None else str(value)


def from_graph(graph: Graph, ont: Graph) -> dict:
    sensitivity = {v: k for k, v in DATA_CLASS.items()}
    use_cases = []
    for subject in sorted(graph.subjects(RDF.type, P.AIUseCase), key=str):
        purpose = graph.value(subject, P.hasPurpose)
        status = _notation(ont, graph.value(subject, P.lifecycleStatus))
        tier = _notation(ont, graph.value(subject, P.riskTier))
        data_class = _notation(ont, graph.value(subject, P.usesDataClass))
        owners = {}
        for role in graph.objects(subject, P.hasAccountabilityRole):
            typ = {str(t)[len(PWM_NS):] for t in graph.objects(role, RDF.type) if str(t).startswith(PWM_NS)}
            player = graph.value(role, URIRef(CMNS_ROLE + "isPlayedBy"))
            name = _literal(graph, player, RDFS.label) if player else None
            if "BusinessOwner" in typ:
                owners["business_owner"] = name
            if "TechnologyOwner" in typ:
                owners["technology_owner"] = name
        depends = sorted(_literal(graph, d, RDFS.label) or str(d) for d in graph.objects(subject, P.dependsOn))
        entry = {
            "usecase_id": _literal(graph, subject, RDFS.label),
            "purpose": _literal(graph, purpose, RDFS.label) if purpose else None,
            "jurisdiction": _literal(graph, subject, P.approvedJurisdiction),
            "status": status,
            "risk_tier": int(tier) if tier is not None else None,
            "review_date": _iso(_literal(graph, subject, P.nextReviewDate)),
            "data_sensitivity": sensitivity.get(data_class) if data_class else None,
            "model_type": _literal(graph, subject, P.modelType),
            "depends_on": depends,
            **owners,
        }
        use_cases.append(entry)

    agents = []
    for subject in sorted(graph.subjects(RDF.type, P.AIAgent), key=str):
        tools = sorted(filter(None, (_literal(graph, t, RDFS.label) for t in graph.objects(subject, P.permittedTool))))
        registered = []
        for usecase in graph.objects(subject, P.registeredForUseCase):
            registered.append(_literal(graph, usecase, RDFS.label) or str(usecase).rsplit(":", 1)[-1])
        radius = {}
        limit = graph.value(subject, P.hasBlastRadiusLimit)
        if limit is not None:
            dim = _literal(graph, limit, P.limitDimension)
            maximum = graph.value(limit, P.limitMaximum)
            if dim and maximum is not None:
                number = float(maximum) if "." in str(maximum) else int(str(maximum).split(".")[0])
                try:
                    if float(maximum) == int(float(maximum)):
                        number = int(float(maximum))
                except ValueError:
                    pass
                radius[dim] = number
        agents.append({
            "name": _literal(graph, subject, P.agentIdentifier),
            "registered_use_cases": sorted(registered),
            "tools": tools,
            "blast_radius": radius,
        })

    sources = []
    for subject in sorted(graph.subjects(RDF.type, P.ApprovedSource), key=str):
        approved = []
        for usecase in graph.objects(subject, P.approvedForUseCase):
            approved.append(_literal(graph, usecase, RDFS.label) or str(usecase).rsplit(":", 1)[-1])
        sources.append({
            "source_id": _literal(graph, subject, RDFS.label),
            "version": str(subject).rsplit(":", 1)[-1],
            "effective": _iso(_literal(graph, subject, URIRef("http://www.w3.org/ns/prov#generatedAtTime"))),
            "approved_for": sorted(approved),
        })

    return {"use_cases": use_cases, "agents": agents, "sources": sources}


def canonical(export: dict) -> dict:
    def clean_usecase(entry: dict) -> dict:
        return {
            "usecase_id": entry["usecase_id"],
            "purpose": entry["purpose"],
            "jurisdiction": entry.get("jurisdiction") or None,
            "status": entry.get("status") or None,
            "risk_tier": entry.get("risk_tier"),
            "review_date": _iso(entry.get("review_date")),
            "data_sensitivity": entry.get("data_sensitivity") or None,
            "model_type": entry.get("model_type") or None,
            "depends_on": sorted(entry.get("depends_on") or []),
            "business_owner": entry.get("business_owner") or None,
            "technology_owner": entry.get("technology_owner") or None,
        }

    def clean_agent(entry: dict) -> dict:
        radius = entry.get("blast_radius") or {}
        numeric = {
            key: radius[key]
            for key in ("per_minute", "per_hour", "max_clients", "max_daily_actions", "max_daily_transactions", "max_value")
            if radius.get(key) is not None
        }
        # Re-import keeps the first numeric limit only, matching to_jsonld.
        first = {}
        for key in ("per_minute", "per_hour", "max_clients", "max_daily_actions", "max_daily_transactions", "max_value"):
            if key in numeric:
                first[key] = numeric[key]
                break
        return {
            "name": entry["name"],
            "registered_use_cases": sorted(entry.get("registered_use_cases") or []),
            "tools": sorted(entry.get("tools") or []),
            "blast_radius": first,
        }

    def clean_source(entry: dict) -> dict:
        return {
            "source_id": entry["source_id"],
            "version": entry["version"],
            "effective": _iso(entry.get("effective")),
            "approved_for": sorted(entry.get("approved_for") or []),
        }

    return {
        "use_cases": [clean_usecase(e) for e in sorted(export.get("use_cases") or [], key=lambda e: e["usecase_id"])],
        "agents": [clean_agent(e) for e in sorted(export.get("agents") or [], key=lambda e: e["name"])],
        "sources": [clean_source(e) for e in sorted(export.get("sources") or [], key=lambda e: (e["source_id"], e["version"]))],
    }


def check(export: dict) -> list[str]:
    ont = Graph().parse(BASE / "pwm.ttl")
    graph = export_to_graph(export, ont)
    expected = canonical(export)
    got = canonical(from_graph(graph, ont))
    problems = []
    if got != expected:
        problems.append("re-imported register does not match the export")
        problems.append("expected: " + json.dumps(expected, sort_keys=True))
        problems.append("got: " + json.dumps(got, sort_keys=True))
    if not expected["use_cases"] or not expected["agents"] or not expected["sources"]:
        problems.append("export is missing a use case, agent or approved source")
    return problems


def dump_register(session, as_of: datetime | None = None) -> dict:
    from kognita.registry import all_agents
    from kognita.sources import _load as load_sources
    from kognita.usecases import list_usecases

    when = as_of or NOW
    use_cases = []
    approved = {}
    for entry in list_usecases(session, as_of=when):
        model_type = entry.approved_models[0] if entry.approved_models else ""
        depends = list(entry.depends_on or ()) or list(entry.model_providers or ()) or list(entry.approved_models or ())
        use_cases.append({
            "usecase_id": entry.usecase_id,
            "purpose": entry.purpose,
            "jurisdiction": entry.jurisdiction,
            "status": entry.status or None,
            "risk_tier": entry.proposed_tier,
            "review_date": _iso(entry.review_date),
            "data_sensitivity": entry.data_sensitivity,
            "model_type": model_type,
            "depends_on": depends,
            "business_owner": entry.business_owner,
            "technology_owner": entry.technology_owner,
            "residual_materiality": entry.residual_materiality,
            "inherent_materiality": entry.inherent_materiality,
        })
        for source_id in entry.approved_sources:
            approved.setdefault(source_id, []).append(entry.usecase_id)

    agents = []
    for agent in all_agents(session):
        tools = sorted((agent.tool_allowlist or {}).keys())
        agents.append({
            "name": agent.name,
            "registered_use_cases": list(agent.use_cases or ()),
            "tools": tools,
            "blast_radius": dict(agent.blast_radius or {}),
        })

    sources = []
    for source in load_sources(session).entries:
        sources.append({
            "source_id": source.source_id,
            "kind": source.kind,
            "version": source.version,
            "owner": source.owner,
            "effective": _iso(source.effective),
            "approved_for": sorted(approved.get(source.source_id, [])),
        })
    return {"use_cases": use_cases, "agents": agents, "sources": sources}


def make() -> dict:
    from sqlmodel import Session

    from kognita.db import create_all, make_engine
    from kognita.evidence import EvidenceWriter
    from kognita.registry import register as register_agent
    from kognita.signing import EvidenceSigner, _generate
    from kognita.sources import Source, register_source
    from kognita.usecases import (
        ACCOUNTABILITY,
        Appetite,
        UseCaseSpec,
        activate,
        approve_rating,
        model_card_id,
        register,
        register_model_card,
        set_appetite,
    )

    private, _public = _generate()
    engine = make_engine(":memory:")
    create_all(engine)
    card_id = model_card_id("house", "summary", "1")
    complexity = {
        "technology": "established",
        "novelty": "familiar",
        "explainability": "explainable",
        "third_party_visibility": "not_applicable",
    }
    appetite = Appetite(
        allowed_categories=("research summarization",),
        prohibited_categories=(),
        maximum_autonomy=2,
        permitted_client_impact=("internal productivity",),
        external_communication=False,
        transaction_authority=False,
        maximum_residual_materiality="low",
    )
    spec = UseCaseSpec(
        usecase_id="research-summary",
        purpose="Summarize approved research for an RM",
        category="research summarization",
        business_owner="Jane Smith",
        technology_owner="Omar Aziz",
        risk_owner="Priya Shah",
        model_owner="Luis Ortega",
        control_function="Alex Chen",
        accountability={seat: "Jane Smith" for seat in ACCOUNTABILITY},
        builders=("Ann Builder",),
        business_criticality="low",
        client_impact="internal productivity",
        autonomy="none",
        decision_consequence="informational",
        data_sensitivity="internal",
        complexity=complexity,
        inherent_materiality="medium",
        residual_materiality="low",
        approval_gate="none",
        review_sampling_rate=0.1,
        maximum_autonomy=1,
        monitoring=True,
        explanation_depth="citation",
        validators=("Jane Smith",),
        approved_models=(card_id,),
        approved_sources=("cio-note",),
        jurisdiction="SG",
        review_date=NOW,
        model_providers=("house",),
    )
    with Session(engine) as session:
        writer = EvidenceWriter(engine, signing=EvidenceSigner(private))
        register_model_card(
            session,
            provider="house",
            model="summary",
            version="1",
            intended_use=spec.purpose,
            limitations="Does not read tables in exhibits",
            evaluation_results="Reviewed against the research fixture, May 2026",
            approver="Luis Ortega",
            approved_at=NOW,
            evidence=writer,
            actor_id="Luis Ortega",
            at=NOW,
        )
        set_appetite(session, appetite, evidence=writer, actor_id="CRO", senior_management=True, at=NOW)
        register(session, spec, evidence=writer, actor_id="Jane Smith", at=NOW)
        approve_rating(session, spec.usecase_id, spec.control_function, evidence=writer, at=NOW)
        activate(session, spec.usecase_id, evidence=writer, actor_id=spec.control_function, at=NOW, status="live")
        register_source(
            session,
            Source(
                source_id="cio-note",
                kind="CIO research",
                version="1",
                owner="Jane Smith",
                effective=NOW,
            ),
            evidence=writer,
            interaction_id="register-roundtrip",
        )
        register_agent(
            session,
            name="research-assistant",
            owner_exec="Omar Aziz",
            version="1.0.0",
            evidence=writer,
            actor_id="Omar Aziz",
            use_cases=["research-summary"],
            tool_allowlist={"research-retriever": {"verb": "read"}},
            blast_radius={"per_minute": 5},
            autonomy_level=0,
        )
        session.commit()
        export = dump_register(session, as_of=NOW)
    EXPORT_PATH.write_text(json.dumps(canonical(export), indent=1, sort_keys=True) + "\n")
    return export


def _write_json(path: pathlib.Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "context":
        _write_json(CONTEXT_PATH, build_context(Graph().parse(BASE / "pwm.ttl")))
    elif cmd == "make":
        export = make()
        print("wrote", EXPORT_PATH, "use cases:", len(export["use_cases"]), "agents:", len(export["agents"]), "sources:", len(export["sources"]))
    else:
        path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else EXPORT_PATH
        issues = check(json.loads(path.read_text()))
        print("register round trip ok" if not issues else "\n".join(issues))
        sys.exit(1 if issues else 0)
