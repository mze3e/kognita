"""Generate declarations for external terms used by PWM (src/90-external.ttl).

Labels and maturity for FIBO terms come from the FIBO index built from the
FIBO source files, so nothing here is typed by hand.
"""
import json
import pathlib
import sys

from rdflib import Graph, URIRef, BNode
from rdflib.namespace import OWL, RDF, RDFS, SKOS

src = pathlib.Path(sys.argv[1])
fibo_index = json.loads(pathlib.Path(sys.argv[2]).read_text())
out = src / "90-external.ttl"

g = Graph()
# Section files share the prefix block in 00-header.ttl, so parse them as one document.
g.parse(data="\n".join(f.read_text() for f in sorted(src.glob("[0-8]*.ttl"))), format="turtle")

PWM = "https://ahmedmuzammil.com/ontology/pwm#"
FIBO = "https://spec.edmcouncil.org/fibo/ontology/"
CMNS = "https://www.omg.org/spec/Commons/"
PROV = "http://www.w3.org/ns/prov#"

# Terms PWM reuses without change, and the module that uses them.
REUSED = {
    "ModuleProducts": [
        FIBO + "FBC/FinancialInstruments/FinancialInstruments/FinancialInstrument",
        FIBO + "FBC/FinancialInstruments/FinancialInstruments/Security",
        FIBO + "FBC/FinancialInstruments/FinancialInstruments/EquityInstrument",
        FIBO + "SEC/Equities/EquityInstruments/Share",
        FIBO + "FBC/FinancialInstruments/FinancialInstruments/DebtInstrument",
        FIBO + "SEC/Debt/Bonds/Bond",
        FIBO + "FBC/FinancialInstruments/FinancialInstruments/DerivativeInstrument",
        FIBO + "FBC/FinancialInstruments/FinancialInstruments/Option",
        FIBO + "DER/DerivativesContracts/StructuredInstruments/StructuredProduct",
        FIBO + "DER/DerivativesContracts/StructuredInstruments/CapitalProtectedProduct",
        FIBO + "DER/DerivativesContracts/StructuredInstruments/LeveragedProduct",
        FIBO + "SEC/Securities/Pools/CollectiveInvestmentVehicle",
        FIBO + "SEC/Funds/Funds/MutualFund",
        FIBO + "SEC/Funds/Funds/ExchangeTradedFund",
        FIBO + "SEC/Funds/Funds/HedgeFund",
        FIBO + "SEC/Funds/Funds/PrivateEquityFund",
        FIBO + "SEC/Funds/Funds/UnitTrust",
        FIBO + "SEC/Funds/Funds/FundUnit",
        FIBO + "FND/OwnershipAndControl/Ownership/DigitalAsset",
        FIBO + "SEC/Securities/SecuritiesIssuance/Prospectus",
        FIBO + "FBC/FinancialInstruments/FinancialInstruments/Issuer",
    ],
    "ModuleParties": [
        FIBO + "FND/AgentsAndPeople/People/Person",
        FIBO + "FND/Parties/Parties/TaxIdentifier",
        CMNS + "RegulatoryAgencies/Jurisdiction",
        CMNS + "RegulatoryAgencies/RegulatoryAgency",
    ],
}

# Non-FIBO external terms: label, kind (verified definitions noted in the register).
OTHER = {
    CMNS + "PartiesAndSituations/Party": ("party", "Class"),
    CMNS + "PartiesAndSituations/PartyRole": ("party role", "Class"),
    CMNS + "PartiesAndSituations/Situation": ("situation", "Class"),
    CMNS + "PartiesAndSituations/SoftwareAgent": ("software agent", "Class"),
    CMNS + "Organizations/LegalEntity": ("legal entity", "Class"),
    CMNS + "Organizations/ServiceProvider": ("service provider", "Class"),
    CMNS + "RegulatoryAgencies/Jurisdiction": ("jurisdiction", "Class"),
    CMNS + "RegulatoryAgencies/RegulatoryAgency": ("regulatory agency", "Class"),
    CMNS + "BusinessAuthorizations/Authorization": ("authorization", "Class"),
    CMNS + "Documents/Document": ("document", "Class"),
    CMNS + "RolesAndCompositions/isPlayedBy": ("is played by", "ObjectProperty"),
    PROV + "Entity": ("Entity", "Class"),
    PROV + "Activity": ("Activity", "Class"),
    PROV + "SoftwareAgent": ("SoftwareAgent", "Class"),
    PROV + "wasGeneratedBy": ("wasGeneratedBy", "ObjectProperty"),
    PROV + "wasDerivedFrom": ("wasDerivedFrom", "ObjectProperty"),
    PROV + "wasAttributedTo": ("wasAttributedTo", "ObjectProperty"),
    "http://purl.org/dc/terms/references": ("References", "ObjectProperty"),
}

# Non-FIBO targets of SKOS mappings (labels as published by each source).
MAPPED = {
    "http://purl.org/dc/terms/valid": "Date Valid",
    "http://purl.org/dc/terms/replaces": "Replaces",
    CMNS + "Organizations/isDomiciledIn": "is domiciled in",
    CMNS + "Organizations/isProvidedBy": "is provided by",
    CMNS + "RegulatoryAgencies/hasJurisdiction": "has jurisdiction",
    "https://w3id.org/dpv#Consent": "Consent",
    "https://w3id.org/dpv#Purpose": "Purpose",
    "https://w3id.org/dpv#PersonalData": "Personal Data",
    "https://w3id.org/dpv#SpecialCategoryPersonalData": "Special Category Personal Data",
    "https://w3id.org/dpv/ai#AISystem": "AI System",
    "https://w3id.org/dpv/ai#Model": "Model",
    "https://w3id.org/dpv/ai#AIAgent": "AI Agent",
    "https://w3id.org/dpv/ai#DeploymentStage": "Deployment Stage",
    "https://w3id.org/dpv/ai#OperationStage": "Operation Stage",
    "https://w3id.org/dpv/ai#DecommissionStage": "Decommission Stage",
}

def ontology_of(iri: str) -> str:
    if iri.startswith(PROV):
        return "http://www.w3.org/ns/prov-o#"
    if "#" in iri:
        return iri.split("#", 1)[0]
    return iri.rsplit("/", 1)[0] + "/"

# External IRIs that appear in logical axioms.
logical_preds = {RDFS.subClassOf, RDFS.domain, RDFS.range, RDFS.subPropertyOf, OWL.someValuesFrom, OWL.onProperty}
used = set()
for s, p, o in g:
    if p in logical_preds and isinstance(o, URIRef) and not str(o).startswith(PWM):
        if str(o).startswith(("http://www.w3.org/2001/XMLSchema#", str(SKOS), str(OWL), str(RDFS))):
            continue
        used.add(str(o))
reused_module = {iri: m for m, iris in REUSED.items() for iri in iris}
used |= set(reused_module)

missing = []
lines = ["", "# " + "=" * 77,
         "#  External terms (declared, not imported). Generated by tools/gen_external.py",
         "#  from the FIBO source files; do not edit by hand.",
         "# " + "=" * 77, ""]
for iri in sorted(used):
    if iri.startswith(FIBO):
        entry = fibo_index.get(iri)
        if not entry:
            missing.append(iri)
            continue
        label, kind, maturity = entry.get("label") or iri.rsplit("/", 1)[-1], entry["kind"], entry.get("maturity")
    elif iri in OTHER:
        (label, kind), maturity = OTHER[iri], None
    else:
        missing.append(iri)
        continue
    owl_kind = {"Class": "owl:Class", "ObjectProperty": "owl:ObjectProperty", "DatatypeProperty": "owl:DatatypeProperty"}[kind]
    parts = [f"<{iri}> a {owl_kind}", f'    rdfs:label "{label}"@en', f"    rdfs:isDefinedBy <{ontology_of(iri)}>"]
    if maturity:
        parts.append(f'    pwm:fiboMaturity "{maturity}"')
    if iri in reused_module:
        parts.append(f"    pwm:reusedIn pwm:{reused_module[iri]}")
    lines.append(" ;\n".join(parts) + " .\n")

# Targets of SKOS mappings that no logical axiom uses: label (and FIBO maturity)
# only, with no OWL type, because the mappings are annotations.
mapping_preds = {SKOS.exactMatch, SKOS.closeMatch, SKOS.relatedMatch}
mapped = sorted({str(o) for s, p, o in g if p in mapping_preds and isinstance(o, URIRef)
                 and not str(o).startswith(PWM)} - used)
if mapped:
    lines += ["# Targets of SKOS mappings (annotation only).", ""]
for iri in mapped:
    if iri.startswith(FIBO):
        entry = fibo_index.get(iri)
        if not entry:
            missing.append(iri)
            continue
        label, maturity = entry.get("label") or iri.rsplit("/", 1)[-1], entry.get("maturity")
    elif iri in MAPPED:
        label, maturity = MAPPED[iri], None
    else:
        missing.append(iri)
        continue
    parts = [f'<{iri}> rdfs:label "{label}"@en', f"    rdfs:isDefinedBy <{ontology_of(iri)}>"]
    if maturity:
        parts.append(f'    pwm:fiboMaturity "{maturity}"')
    lines.append(" ;\n".join(parts) + " .\n")

if missing:
    print("MISSING external terms (not in FIBO index, OTHER or MAPPED):", *missing, sep="\n  ")
    sys.exit(1)

header = ['pwm:fiboMaturity a owl:AnnotationProperty ;',
          '    rdfs:label "FIBO maturity"@en ;',
          '    skos:definition "Maturity level of a FIBO term, Release or Provisional, read from the FIBO source file that defines it."@en ;',
          '    rdfs:isDefinedBy <https://ahmedmuzammil.com/ontology/pwm> .', '']
out.write_text("\n".join(lines[:5] + header + lines[5:]) + "\n")
print(f"wrote {out.name}: {len(used)} declared terms, {len(mapped)} mapping targets")
