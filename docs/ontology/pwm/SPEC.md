# Private Wealth Management Ontology (PWM) 0.5

> Generated from the Turtle sources by `tools/build.py`. Do not edit by hand.

| | |
|---|---|
| Version | 0.5 (draft), 2026-11-03 |
| Ontology IRI | https://ahmedmuzammil.com/ontology/pwm |
| Version IRI | https://ahmedmuzammil.com/ontology/pwm/0.5 |
| Namespace | `pwm:` https://ahmedmuzammil.com/ontology/pwm# |
| Shapes | https://ahmedmuzammil.com/ontology/pwm/shapes (`pwm-shapes.ttl`), latest pointer |
| Shapes version IRI | https://ahmedmuzammil.com/ontology/pwm/0.5/pwm-shapes.ttl |
| Licence | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| Author | Ahmed Muzammil |
| Web page | https://ahmedmuzammil.com/ontology/pwm |

## Abstract

A generic ontology for private banking and wealth management. It models parties and relationships, due diligence, investor profiling and suitability, mandates, accounts and holdings, products and research, client interactions, and the governance of AI that acts on them: use cases, agents, delegated authority, claims and evidence. Business concepts reuse or specialise the Financial Industry Business Ontology (FIBO) and the OMG Commons library. Governance concepts follow the MAS Guidelines on Artificial Intelligence Risk Management (7 October 2026) and the Kognita roadmap. Every term cites the references it is grounded in. Institution-specific detail belongs in an extension that imports this ontology.

## Status

Draft for comment. Version 0.5 aligns with roadmap milestone 0.5: agents, authority and fleets. Terms, code lists and rules may change before 1.0, and every change will be recorded in the change log. It is not legal advice: the regulatory citations show where a term comes from, not how a rule applies to a given firm.

Size: 187 classes, 216 object properties, 183 datatype properties, 48 code lists with 235 codes, 123 data rules plus 66 generated code-list rules, 47 references and 1294 citations. 39 FIBO terms are specialised or mapped.

## Scope

In scope:

- Parties, roles and the client relationship, including trusts, foundations and other legal arrangements
- Due diligence, risk rating, source of wealth and funds, screening and tax residency
- Investor classification, risk profile, knowledge and experience, sustainability preferences, suitability and appropriateness
- Mandates, investment restrictions and fees
- Accounts, portfolios, holdings and valuations
- The product shelf, product risk, target market, cross-border rules, offering documents and the house view
- The client journey: origination, recommendation, final decision and client communication
- AI governance: the use-case inventory, risk materiality, accountability, agents, tools, delegated authority and autonomy
- Information governance, facts and claims, policy decisions and hash-chained evidence

Not in 0.5:

- Orders, execution and the trade lifecycle; corporate actions
- Performance and attribution methods (only the provenance of figures is modelled)
- Tax reporting detail (CRS and FATCA reporting records), lending beyond the Lombard facility
- Any institution's thresholds, scales, product shelf or internal names: these belong in an extension

## Design principles

1. **Reuse before inventing.** Business terms specialise or map to FIBO first (Release before Provisional), then OMG Commons, W3C PROV-O and DPV. A term is defined here only when none fits, and it is marked NEW.
2. **Declare, don't import.** External terms are declared with their label, source and FIBO maturity instead of importing whole ontologies, so the file loads on its own. Specialisation uses rdfs:subClassOf; looser alignment uses SKOS exact, close and related matches.
3. **Every term cites its grounds.** Each class, property and code list carries at least one citation with a locator: an article, paragraph, roadmap item or slide. The reference register records how each source was checked.
4. **Logic in OWL, codes in SKOS, rules in SHACL.** The ontology uses light OWL axioms that an extension can build on. Pre-existing pwm:nextReviewDate and pwm:retentionUntil use xsd:date; pwm:freshnessThreshold uses xsd:duration. These datatypes fall outside the OWL 2 datatype map. Their ranges are preserved for compatibility with 0.2. ROBOT DL profile validation passes with xsd:date and xsd:duration as declared datatypes; this does not establish conformance to the OWL 2 datatype map. Open item for a later version: migrate these legacy ranges and their data to OWL 2 datatype-map datatypes. Code lists are SKOS concept schemes. Data rules are SHACL shapes kept in a separate file.
5. **Context speed and classification.** Each class states how fast its facts change and the data classification it carries by default. Fast facts, such as positions and prices, are looked up live and are never cached or embedded for AI.
6. **Accountability stays with people.** Accountability roles are played by named people. A shape rejects any record that names an AI agent in one.
7. **Generic core, firm extension.** No institution's data, thresholds or names appear here. A firm adds them in its own ontology that imports PWM.

## How references are marked

Every term lists its citations. Each citation carries a mark for the kind of source, the source, and a locator inside it.

| Mark | Source type | Meaning |
|---|---|---|
| **ONT** | standard ontology | Published ontology reused or specialised: FIBO, OMG Commons, W3C PROV-O, DPV. |
| **ISO** | technical standard | ISO or ISO/IEC standard, used for identifiers, code formats and terminology. |
| **REG** | regulation or supervisory guidance | Law, regulation, supervisory guideline, notice or international standard-setter recommendation. |
| **KOG** | Kognita | Kognita roadmap, regulations library (control-frameworks.md), controlled vocabularies or policy packs. |
| **AM** | author framework | EVOLVE framework, essays and talks by Ahmed Muzammil. |
| **NEW** | practitioner definition | Defined in this ontology because no published standard term was found at the versions checked. |

Each reference in the register has a verification status. **verified**: checked against the source file, official text or regulator page while drafting this version. **instrument-level**: cited from the published instrument; article and paragraph numbers are to be re-checked before formal use.

FIBO terms show their maturity, Release or Provisional, as read from the FIBO file that defines them.

Classes carry a context speed:

- **slow**: Changes over weeks or months, such as policies, definitions and the house view. Curated, versioned and owned; safe to cache with an effective date.
- **medium**: Changes over days or weeks, such as client profile, KYC status and mandate. Synchronised from the system of record and cached with an explicit as-of time.
- **fast**: Changes in seconds or minutes, such as positions, prices, cash and orders. Looked up live from the system of record; never cached or embedded for AI.

## Modules

| # | Module | Covers | Classes | Properties | Code lists |
|---|---|---|---|---|---|
| 0 | [Core patterns](#ModuleCore) | Assessments, validity periods and supersession: the patterns every other module reuses. | 1 | 7 | 0 |
| 1 | [Parties, roles and relationships](#ModuleParties) | Clients, related parties, legal arrangements, providers and the relationship that binds them. | 30 | 16 | 2 |
| 2 | [Due diligence, financial crime and tax](#ModuleCompliance) | Customer due diligence, risk rating, source of wealth, screening and tax residency. | 8 | 4 | 2 |
| 3 | [Investor profile and suitability](#ModuleSuitability) | Regulatory investor classification, risk profile, knowledge and experience, sustainability preferences, and suitability and appropriateness. | 10 | 15 | 5 |
| 4 | [Mandates, restrictions and fees](#ModuleMandates) | The service model agreed with the client, its investment restrictions and its charges. | 11 | 7 | 1 |
| 5 | [Accounts, portfolios and holdings](#ModuleAccounts) | Where assets are held, what is held, and how it is valued. | 11 | 9 | 1 |
| 6 | [Products, distribution and research](#ModuleProducts) | The product shelf, product risk and target market, cross-border distribution rules, offering documents and the house view. | 9 | 11 | 2 |
| 7 | [Interactions, advice and communication](#ModuleInteractions) | Why a client was approached, what was recommended, who decided, and what reached the client. | 9 | 16 | 6 |
| 8 | [AI use cases, authority and accountability](#ModuleGovernance) | The AI inventory, risk materiality, agents, delegated authority, autonomy and the people accountable for them. | 24 | 40 | 11 |
| 9 | [Information governance, claims and evidence](#ModuleEvidence) | Governed documents and definitions, facts and claims, policy decisions and the evidence that reconstructs them. | 37 | 98 | 14 |
| 10 | [Ingestion, policy and lifecycle controls](#ModuleLifecycle) | Structured ingestion and the missing policy, review and monitoring records for milestone 0.4. | 17 | 66 | 2 |
| 11 | [Agents, authority and fleets](#ModuleAgents) | Workload identities, scoped human authority, autonomy evidence, containment and fleet controls. | 20 | 110 | 2 |

<a id="ModuleCore"></a>

## Module 0: Core patterns

Assessments, validity periods and supersession: the patterns every other module reuses.

### Classes

<a id="Assessment"></a>
#### assessment (`pwm:Assessment`)

Dated, attributable record of an evaluation of a party, product, portfolio or AI use case, valid for a period and replaceable by a later assessment.

*Note:* Due diligence, investor classification, risk profile, suitability, product risk rating and AI risk materiality are all assessments. Who assessed it is recorded with prov:wasAttributedTo; when, with prov:generatedAtTime.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context
- **Properties:** [`assesses`](#assesses)
- **References:** **ONT** [PROV-O](#ref-W3C-PROV-O) prov:Entity, prov:wasAttributedTo, prov:generatedAtTime; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Shared pattern for dated evaluations

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="assesses"></a>`pwm:assesses` assesses | [assessment](#Assessment) → any | Thing an assessment evaluates. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Core pattern |
| <a id="supersedes"></a>`pwm:supersedes` supersedes | any → any | Earlier version that this record, rule or document replaces. The earlier version stays readable for reconstruction but may no longer be used. Close match: [Replaces](http://purl.org/dc/terms/replaces) (DCMI). | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 4: house-view change propagation; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: what it supersedes, as a minimum label |
| <a id="approvedBy"></a>`pwm:approvedBy` approved by | any → [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons) | Role whose holder approved this record. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.13: control function approves materiality; 5.21: approval body |
| <a id="validFrom"></a>`pwm:validFrom` valid from | any → [dateTime](http://www.w3.org/2001/XMLSchema#dateTime) (XSD) | Start of the period in which a record, rule or document is in force. Close match: [Date Valid](http://purl.org/dc/terms/valid) (DCMI). | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.3: policy rows effective-dated; 0.6 item 8: standards are effective-dated; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: effective date as a minimum label |
| <a id="validUntil"></a>`pwm:validUntil` valid until | any → [dateTime](http://www.w3.org/2001/XMLSchema#dateTime) (XSD) | End of the period in which a record, rule or document is in force. Absent while it remains in force. Close match: [Date Valid](http://purl.org/dc/terms/valid) (DCMI). | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 2: created, effective, superseded and expiry dates |
| <a id="nextReviewDate"></a>`pwm:nextReviewDate` next review date | any → [date](http://www.w3.org/2001/XMLSchema#date) (XSD) | Date by which a record must be reviewed. Past this date the record is treated as expired. | **REG** [FATF Recommendations](#ref-FATF) R.10: ongoing due diligence; **REG** [MAS AIRG](#ref-MAS-AIRG) 4.11, 4.13: assessments regularly reviewed; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: past its review date, a use case is a DENY |
| <a id="rationale"></a>`pwm:rationale` rationale | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Reasons for an assessment or decision, stated as facts and evidence rather than as a bare score. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 10: reasons, not scores |

<a id="ModuleParties"></a>

## Module 1: Parties, roles and relationships

Clients, related parties, legal arrangements, providers and the relationship that binds them.

### Classes

<a id="Client"></a>
#### client (`pwm:Client`)

Role of a party that has, or is establishing, a business relationship with a wealth manager to receive investment, banking, advisory or fiduciary services.

- **Specialises:** [client](https://spec.edmcouncil.org/fibo/ontology/FND/ProductsAndServices/ProductsAndServices/Client) (FIBO, Release)
- **Handling:** medium context, default classification C2
- **Properties:** [`inGroup`](#inGroup)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.10: customer subject to due diligence

<a id="IndividualClient"></a>
#### individual client (`pwm:IndividualClient`)

Client role played by a natural person.

- **Specialises:** [client](#Client)
- **Restriction:** [is played by](https://www.omg.org/spec/Commons/RolesAndCompositions/isPlayedBy) (OMG Commons) some [person](https://spec.edmcouncil.org/fibo/ontology/FND/AgentsAndPeople/People/Person) (FIBO, Release)
- **Handling:** default classification C2
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.10

<a id="EntityClient"></a>
#### entity client (`pwm:EntityClient`)

Client role played by a legal person or a legal arrangement, such as a company, foundation or trust.

- **Specialises:** [client](#Client)
- **Handling:** default classification C2
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.10, R.24, R.25

<a id="Prospect"></a>
#### prospect (`pwm:Prospect`)

Role of a party a wealth manager is considering or approaching before any business relationship exists.

- **Specialises:** [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons)
- **Handling:** default classification C2
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 11: prospect scoring in fairness reporting; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Pre-relationship role

<a id="ClientRelationship"></a>
#### client relationship (`pwm:ClientRelationship`)

Business relationship between a wealth manager and one or more clients, booked in a booking centre and covered by named relationship managers for a period.

- **Specialises:** [situation](https://www.omg.org/spec/Commons/PartiesAndSituations/Situation) (OMG Commons)
- **Handling:** medium context, default classification C2
- **Properties:** [`hasClient`](#hasClient), [`hasProvider`](#hasProvider)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.10: business relationship; **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) booking-centre, relationship-manager

<a id="RelationshipGroup"></a>
#### relationship group (`pwm:RelationshipGroup`)

Set of related clients, such as a family, its holding structures and its trusts, managed and reported together.

- **Also called:** family group, household
- **Handling:** medium context, default classification C2
- **References:** **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Household and family-group view used in private banking

<a id="RelationshipManager"></a>
#### relationship manager (`pwm:RelationshipManager`)

Person responsible for a client relationship on behalf of the wealth manager.

- **Specialises:** [relationship manager](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/RelationshipManager) (FIBO, Release)
- **Exact match:** [relationship manager](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/RelationshipManager) (FIBO, Release)
- **References:** **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) relationship-manager (acting principal equals the named RM); rm-location

<a id="CoverageAssignment"></a>
#### coverage assignment (`pwm:CoverageAssignment`)

Assignment of a relationship manager or a specialist to a client relationship, in a stated coverage role, for a period.

- **Specialises:** [situation](https://www.omg.org/spec/Commons/PartiesAndSituations/Situation) (OMG Commons)
- **Handling:** medium context
- **Properties:** [`assignedCoverage`](#assignedCoverage), [`assignedRelationshipManager`](#assignedRelationshipManager), [`coversRelationship`](#coversRelationship), [`coverageRole`](#coverageRole)
- **References:** **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) relationship-manager; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Primary and secondary coverage

<a id="InvestmentAdviser"></a>
#### investment adviser (`pwm:InvestmentAdviser`)

Role of a person who gives investment advice to clients under the wealth manager's licence or registration.

- **Specialises:** [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons)
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 4(1)(4): investment advice; **REG** [MAS FAA-N16](#ref-MAS-FAA-N16) Recommendations on investment products

<a id="ExternalAssetManager"></a>
#### external asset manager (`pwm:ExternalAssetManager`)

Independent firm that manages a client's assets booked with the wealth manager, acting under a limited power of attorney granted by the client.

- **Specialises:** [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons)
- **Also called:** independent asset manager
- **References:** **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Intermediated private banking model

<a id="AuthorisedSignatory"></a>
#### authorised signatory (`pwm:AuthorisedSignatory`)

Person authorised to give instructions on an account or for an entity client, within stated signing rules.

- **Specialises:** [authorized individual](https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/Executives/AuthorizedIndividual) (FIBO, Release)
- **References:** **ONT** [FIBO](#ref-FIBO) BE/OwnershipAndControl/Executives/AuthorizedIndividual, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Account mandate and signing authority

<a id="PowerOfAttorneyHolder"></a>
#### power of attorney holder (`pwm:PowerOfAttorneyHolder`)

Party authorised by a client, under a power of attorney, to act for the client within the scope that instrument grants.

- **Specialises:** [authorized individual](https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/Executives/AuthorizedIndividual) (FIBO, Release)
- **References:** **ONT** [FIBO](#ref-FIBO) BE/OwnershipAndControl/Executives/AuthorizedIndividual, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) General and limited powers of attorney

<a id="BeneficialOwner"></a>
#### beneficial owner (`pwm:BeneficialOwner`)

Natural person who ultimately owns or controls a client, or on whose behalf a transaction is conducted, including those exercising ultimate effective control over a legal person or arrangement.

- **Specialises:** [beneficial owner](https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/CorporateOwnership/BeneficialOwner) (FIBO, Release)
- **Handling:** default classification C2
- **References:** **REG** [FATF Recommendations](#ref-FATF) Glossary: beneficial owner; R.24; R.25; **REG** [MAS Notice 626](#ref-MAS-626) Customer due diligence on beneficial owners

<a id="ControllingPerson"></a>
#### controlling person (`pwm:ControllingPerson`)

Natural person who exercises control over an entity account holder for tax-reporting purposes; for a trust, the settlor, trustees, protector, beneficiaries and anyone with ultimate effective control.

- **Specialises:** [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons)
- **Close match:** [beneficial owner](#BeneficialOwner)
- **Handling:** default classification C2
- **References:** **REG** [OECD CRS](#ref-OECD-CRS) Section VIII(D)(6)

<a id="PoliticallyExposedPerson"></a>
#### politically exposed person (`pwm:PoliticallyExposedPerson`)

Role of a person who is or has been entrusted with a prominent public function, or is a family member or close associate of such a person, attracting enhanced due diligence.

- **Specialises:** [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons)
- **Also called:** PEP
- **Handling:** default classification C3
- **Properties:** [`pepCategory`](#pepCategory)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.12; Glossary: politically exposed persons; **REG** [MAS Notice 626](#ref-MAS-626) Politically exposed persons

<a id="ConnectedParty"></a>
#### connected party (`pwm:ConnectedParty`)

Party linked to a client by family, business or control, recorded for due diligence and conflict checks.

- **Specialises:** [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons)
- **Handling:** default classification C2
- **Properties:** [`connectedTo`](#connectedTo)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.12: family members and close associates

<a id="LegalArrangement"></a>
#### legal arrangement (`pwm:LegalArrangement`)

Express trust or similar arrangement through which assets are held and controlled, treated as a party to the relationship.

- **Specialises:** [party](https://www.omg.org/spec/Commons/PartiesAndSituations/Party) (OMG Commons)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.25; Glossary: legal arrangements

<a id="Trust"></a>
#### trust (`pwm:Trust`)

Arrangement in which a settlor places assets under the control of a trustee for the benefit of beneficiaries.

- **Specialises:** [legal arrangement](#LegalArrangement), [trust](https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/Trust) (FIBO, Release)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.25

<a id="Settlor"></a>
#### settlor (`pwm:Settlor`)

Party that establishes a trust and places assets into it.

- **Specialises:** [trustor](https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/Trustor) (FIBO, Release)
- **Exact match:** [trustor](https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/Trustor) (FIBO, Release)
- **Also called:** trustor
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.25; **REG** [OECD CRS](#ref-OECD-CRS) Section VIII(D)(6)

<a id="Trustee"></a>
#### trustee (`pwm:Trustee`)

Party that holds and manages trust assets for the beneficiaries.

- **Specialises:** [trustee](https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/Trustee) (FIBO, Release)
- **Exact match:** [trustee](https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/Trustee) (FIBO, Release)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.25

<a id="TrustBeneficiary"></a>
#### trust beneficiary (`pwm:TrustBeneficiary`)

Party, or class of parties, for whose benefit a trust holds assets.

- **Specialises:** [trust beneficiary](https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/TrustBeneficiary) (FIBO, Release)
- **Exact match:** [trust beneficiary](https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/TrustBeneficiary) (FIBO, Release)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.25

<a id="Protector"></a>
#### protector (`pwm:Protector`)

Party appointed under a trust instrument to oversee the trustee, often with powers of consent or removal.

- **Specialises:** [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.25; **REG** [OECD CRS](#ref-OECD-CRS) Section VIII(D)(6)

<a id="Foundation"></a>
#### private foundation (`pwm:Foundation`)

Legal person, without shareholders, established to hold and administer assets for stated purposes or beneficiaries.

- **Specialises:** [legal entity](https://www.omg.org/spec/Commons/Organizations/LegalEntity) (OMG Commons)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.24: legal persons; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) No FIBO class for foundations at the version checked

<a id="PrivateInvestmentCompany"></a>
#### private investment company (`pwm:PrivateInvestmentCompany`)

Company formed to hold the investments of an individual or family, usually with no operating business.

- **Specialises:** [legal entity](https://www.omg.org/spec/Commons/Organizations/LegalEntity) (OMG Commons)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.24; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Personal holding structure

<a id="FamilyOffice"></a>
#### family office (`pwm:FamilyOffice`)

Organisation that administers and manages the affairs and wealth of one or more families.

- **Specialises:** [family office](https://spec.edmcouncil.org/fibo/ontology/BE/FunctionalEntities/FunctionalEntities/FamilyOffice) (FIBO, Release)
- **Exact match:** [family office](https://spec.edmcouncil.org/fibo/ontology/BE/FunctionalEntities/FunctionalEntities/FamilyOffice) (FIBO, Release)
- **References:** **ONT** [FIBO](#ref-FIBO) BE/FunctionalEntities/FunctionalEntities/FamilyOffice, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Client or intermediary

<a id="WealthManager"></a>
#### wealth manager (`pwm:WealthManager`)

Financial service provider that delivers wealth management services: advice, portfolio management, custody, lending and planning.

- **Specialises:** [financial service provider](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/FinancialProductsAndServices/FinancialServiceProvider) (FIBO, Release)
- **References:** **ISO** [ISO 22222](#ref-ISO-22222) Scope of personal financial planning

<a id="PrivateBank"></a>
#### private bank (`pwm:PrivateBank`)

Bank that delivers wealth management services to high-net-worth clients.

- **Specialises:** [bank](https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/Bank) (FIBO, Release), [wealth manager](#WealthManager)
- **References:** **REG** [Banking Act (SG)](#ref-SG-BANKING-ACT) s47: customer information may be disclosed only as the Act provides

<a id="WealthManagementService"></a>
#### wealth management service (`pwm:WealthManagementService`)

Service combining investment advice or portfolio management with planning, custody, credit and estate services for a client.

- **Specialises:** [wealth management service](https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/WealthManagementService) (FIBO, Release)
- **Exact match:** [wealth management service](https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/WealthManagementService) (FIBO, Release)
- **References:** **ISO** [ISO 22222](#ref-ISO-22222) Personal financial planning process

<a id="BookingCentre"></a>
#### booking centre (`pwm:BookingCentre`)

Role of a legal entity or branch of a wealth manager in which client accounts, contracts and assets are booked, and whose jurisdiction's rules apply to them.

- **Specialises:** [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons)
- **Handling:** slow context
- **References:** **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) booking-centre: recognised booking centres; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Booking model

<a id="Custodian"></a>
#### custodian (`pwm:Custodian`)

Financial institution that holds clients' securities for safekeeping.

- **Specialises:** [custodian](https://spec.edmcouncil.org/fibo/ontology/SEC/Equities/EquityInstruments/Custodian) (FIBO, Release)
- **Exact match:** [custodian](https://spec.edmcouncil.org/fibo/ontology/SEC/Equities/EquityInstruments/Custodian) (FIBO, Release)
- **References:** **ONT** [FIBO](#ref-FIBO) SEC/Equities/EquityInstruments/Custodian, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Custody chain

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="hasClient"></a>`pwm:hasClient` has client | [client relationship](#ClientRelationship) → [client](#Client) | Client in a client relationship. | **REG** [FATF Recommendations](#ref-FATF) R.10 |
| <a id="hasProvider"></a>`pwm:hasProvider` has provider | [client relationship](#ClientRelationship) → [wealth manager](#WealthManager) | Wealth manager in a client relationship. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Relationship pattern |
| <a id="isBookedAt"></a>`pwm:isBookedAt` is booked at | any → [booking centre](#BookingCentre) | Booking centre in which a relationship, account or contract is booked. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) booking-centre |
| <a id="inGroup"></a>`pwm:inGroup` in group | [client](#Client) → [relationship group](#RelationshipGroup) | Relationship group a client belongs to. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Household view |
| <a id="assignedCoverage"></a>`pwm:assignedCoverage` assigned coverage | [coverage assignment](#CoverageAssignment) → [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons) | Party role named in a coverage assignment: a relationship manager, or a product, credit, investment or planning specialist. The coverage role says which. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Coverage model |
| <a id="assignedRelationshipManager"></a>`pwm:assignedRelationshipManager` assigned relationship manager | [coverage assignment](#CoverageAssignment) → [relationship manager](#RelationshipManager) | Relationship manager named in a coverage assignment. Specialists are named with the parent property, assigned coverage. Sub-property of [assigned coverage](#assignedCoverage). | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) relationship-manager |
| <a id="coversRelationship"></a>`pwm:coversRelationship` covers relationship | [coverage assignment](#CoverageAssignment) → [client relationship](#ClientRelationship) | Client relationship that a coverage assignment covers. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) relationship-manager |
| <a id="coverageRole"></a>`pwm:coverageRole` coverage role | [coverage assignment](#CoverageAssignment) → [coverage roles](#CoverageRoleScheme) | Role of the assignee in a coverage assignment: primary, secondary or specialist. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Coverage model |
| <a id="hasBeneficialOwner"></a>`pwm:hasBeneficialOwner` has beneficial owner | any → [beneficial owner](#BeneficialOwner) | Beneficial owner of a client, entity or account. Close match: [has beneficial owner](https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/CorporateOwnership/hasBeneficialOwner) (FIBO, Release). | **REG** [FATF Recommendations](#ref-FATF) R.10, R.24, R.25 |
| <a id="pepCategory"></a>`pwm:pepCategory` PEP category | [politically exposed person](#PoliticallyExposedPerson) → [politically exposed person categories](#PEPCategoryScheme) | Kind of politically exposed person. | **REG** [FATF Recommendations](#ref-FATF) R.12 |
| <a id="connectedTo"></a>`pwm:connectedTo` connected to | [connected party](#ConnectedParty) → any | Client or party that a connected party is linked to. | **REG** [FATF Recommendations](#ref-FATF) R.12 |
| <a id="residenceCountry"></a>`pwm:residenceCountry` residence country | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Country where a natural person resides, as an ISO 3166-1 alpha-2 code. | **ISO** [ISO 3166](#ref-ISO-3166) Alpha-2 codes; **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) client-domicile |
| <a id="domicileCountry"></a>`pwm:domicileCountry` domicile country | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Country of a party's domicile, as an ISO 3166-1 alpha-2 code. Drives cross-border rules. Close match: [is domiciled in](https://www.omg.org/spec/Commons/Organizations/isDomiciledIn) (OMG Commons). | **ISO** [ISO 3166](#ref-ISO-3166) Alpha-2 codes; **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) client-domicile: recognised domiciles |
| <a id="nationalityCountry"></a>`pwm:nationalityCountry` nationality country | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Country of a natural person's nationality, as an ISO 3166-1 alpha-2 code. | **ISO** [ISO 3166](#ref-ISO-3166) Alpha-2 codes |
| <a id="jurisdictionCode"></a>`pwm:jurisdictionCode` jurisdiction code | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Jurisdiction of a booking centre, rule or use case, as an ISO 3166-1 alpha-2 code. Close match: [has jurisdiction](https://www.omg.org/spec/Commons/RegulatoryAgencies/hasJurisdiction) (OMG Commons). | **ISO** [ISO 3166](#ref-ISO-3166) Alpha-2 codes; **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) booking-centre, rm-location |
| <a id="legalEntityIdentifier"></a>`pwm:legalEntityIdentifier` legal entity identifier | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Twenty-character ISO 17442 identifier of a legal entity. Close match: [legal entity identifier](https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LEIEntities/LegalEntityIdentifier) (FIBO, Release). | **ISO** [ISO 17442](#ref-ISO-17442) LEI code structure |

### Code lists

<a id="CoverageRoleScheme"></a>
#### coverage roles (`pwm:CoverageRoleScheme`)



Used by: [`coverageRole`](#coverageRole)

References: **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Coverage model

| Code | Concept | Definition |
|---|---|---|
|  | <a id="CoveragePrimary"></a>primary (`pwm:CoveragePrimary`) | Lead relationship manager, accountable for the relationship. |
|  | <a id="CoverageSecondary"></a>secondary (`pwm:CoverageSecondary`) | Backup relationship manager. |
|  | <a id="CoverageSpecialist"></a>specialist (`pwm:CoverageSpecialist`) | Product, credit, investment or planning specialist supporting the relationship. |

<a id="PEPCategoryScheme"></a>
#### politically exposed person categories (`pwm:PEPCategoryScheme`)



Used by: [`pepCategory`](#pepCategory)

References: **REG** [FATF Recommendations](#ref-FATF) R.12; Glossary

| Code | Concept | Definition |
|---|---|---|
|  | <a id="ForeignPEP"></a>foreign PEP (`pwm:ForeignPEP`) | Entrusted with a prominent public function by a foreign country. |
|  | <a id="DomesticPEP"></a>domestic PEP (`pwm:DomesticPEP`) | Entrusted domestically with a prominent public function. |
|  | <a id="InternationalOrganisationPEP"></a>international organisation PEP (`pwm:InternationalOrganisationPEP`) | Entrusted with a prominent function by an international organisation. |
|  | <a id="PEPFamilyMember"></a>family member of a PEP (`pwm:PEPFamilyMember`) | Related to a PEP by blood or marriage. |
|  | <a id="PEPCloseAssociate"></a>close associate of a PEP (`pwm:PEPCloseAssociate`) | Closely connected to a PEP, socially or professionally. |

<a id="ModuleCompliance"></a>

## Module 2: Due diligence, financial crime and tax

Customer due diligence, risk rating, source of wealth, screening and tax residency.

### Classes

<a id="CustomerDueDiligence"></a>
#### customer due diligence (`pwm:CustomerDueDiligence`)

Assessment that identifies and verifies a client and its beneficial owners, and establishes the purpose and intended nature of the relationship.

- **Specialises:** [assessment](#Assessment)
- **Also called:** CDD, KYC
- **Handling:** medium context, default classification C2
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.10; **REG** [MAS Notice 626](#ref-MAS-626) Customer due diligence; **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) kyc-onboarding: prepared by the agent, approved by a person

<a id="EnhancedDueDiligence"></a>
#### enhanced due diligence (`pwm:EnhancedDueDiligence`)

Customer due diligence with additional measures, applied where the client presents higher risk, such as a politically exposed person.

- **Specialises:** [customer due diligence](#CustomerDueDiligence)
- **Also called:** EDD
- **Handling:** default classification C3
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.10, R.12; **REG** [MAS Notice 626](#ref-MAS-626) Enhanced customer due diligence

<a id="PeriodicReview"></a>
#### periodic review (`pwm:PeriodicReview`)

Scheduled re-assessment of a client's due diligence, at a frequency set by the client's risk rating, or earlier on a trigger event.

- **Specialises:** [assessment](#Assessment)
- **Handling:** default classification C2
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.10: ongoing due diligence

<a id="CustomerRiskRating"></a>
#### customer risk rating (`pwm:CustomerRiskRating`)

Assessment of the money-laundering and terrorist-financing risk a client presents, which sets due diligence depth and review frequency.

- **Specialises:** [assessment](#Assessment)
- **Handling:** default classification C3
- **Properties:** [`amlRiskLevel`](#amlRiskLevel)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.1: risk-based approach; R.10; **REG** [MAS Notice 626](#ref-MAS-626) Risk assessment

<a id="SourceOfWealthAssessment"></a>
#### source of wealth assessment (`pwm:SourceOfWealthAssessment`)

Assessment of how a client accumulated their total wealth, with the evidence relied on.

- **Specialises:** [assessment](#Assessment)
- **Handling:** default classification C3
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.12: establish the source of wealth; **REG** [MAS Notice 626](#ref-MAS-626) Source of wealth

<a id="SourceOfFundsAssessment"></a>
#### source of funds assessment (`pwm:SourceOfFundsAssessment`)

Assessment of where specific funds or assets entering the relationship come from.

- **Specialises:** [assessment](#Assessment)
- **Handling:** default classification C3
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.12: establish the source of funds; **REG** [MAS Notice 626](#ref-MAS-626) Source of funds

<a id="ScreeningResult"></a>
#### screening result (`pwm:ScreeningResult`)

Outcome of screening a party against sanctions, politically exposed person and adverse-media sources, with its disposition.

- **Specialises:** [assessment](#Assessment)
- **Handling:** default classification C3
- **Properties:** [`screeningDisposition`](#screeningDisposition)
- **References:** **REG** [FATF Recommendations](#ref-FATF) R.6, R.7: targeted financial sanctions; R.12

<a id="TaxResidencySelfCertification"></a>
#### tax residency self-certification (`pwm:TaxResidencySelfCertification`)

Client's declaration of tax residence countries and taxpayer identification numbers, collected at account opening and on change of circumstances.

- **Specialises:** [assessment](#Assessment)
- **Handling:** default classification C2
- **Properties:** [`taxResidenceCountry`](#taxResidenceCountry), [`hasTaxIdentifier`](#hasTaxIdentifier)
- **References:** **REG** [OECD CRS](#ref-OECD-CRS) Sections IV and VI: self-certification for new accounts; Section VIII: definitions

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="amlRiskLevel"></a>`pwm:amlRiskLevel` AML risk level | [customer risk rating](#CustomerRiskRating) → [AML risk levels](#AMLRiskLevelScheme) | Level of a customer risk rating. | **REG** [FATF Recommendations](#ref-FATF) R.1 |
| <a id="screeningDisposition"></a>`pwm:screeningDisposition` screening disposition | [screening result](#ScreeningResult) → [screening dispositions](#ScreeningDispositionScheme) | How a screening alert was resolved. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Alert disposition |
| <a id="hasTaxIdentifier"></a>`pwm:hasTaxIdentifier` has tax identifier | [tax residency self-certification](#TaxResidencySelfCertification) → [tax identifier](https://spec.edmcouncil.org/fibo/ontology/FND/Parties/Parties/TaxIdentifier) (FIBO, Release) | Taxpayer identification number declared in a self-certification. | **REG** [OECD CRS](#ref-OECD-CRS) Section VIII: TIN |
| <a id="taxResidenceCountry"></a>`pwm:taxResidenceCountry` tax residence country | [tax residency self-certification](#TaxResidencySelfCertification) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Country of tax residence declared in a self-certification, as an ISO 3166-1 alpha-2 code. | **ISO** [ISO 3166](#ref-ISO-3166) Alpha-2 codes; **REG** [OECD CRS](#ref-OECD-CRS) Section VIII |

### Code lists

<a id="AMLRiskLevelScheme"></a>
#### AML risk levels (`pwm:AMLRiskLevelScheme`)



Used by: [`amlRiskLevel`](#amlRiskLevel)

References: **REG** [FATF Recommendations](#ref-FATF) R.1, R.10

| Code | Concept | Definition |
|---|---|---|
| L | <a id="AMLLow"></a>low (`pwm:AMLLow`) | Simplified measures may apply where permitted. |
| M | <a id="AMLMedium"></a>medium (`pwm:AMLMedium`) | Standard due diligence. |
| H | <a id="AMLHigh"></a>high (`pwm:AMLHigh`) | Enhanced due diligence and senior approval. |
| X | <a id="AMLOutsideAppetite"></a>outside risk appetite (`pwm:AMLOutsideAppetite`) | Relationship declined or exited. |

<a id="ScreeningDispositionScheme"></a>
#### screening dispositions (`pwm:ScreeningDispositionScheme`)



Used by: [`screeningDisposition`](#screeningDisposition)

References: **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Alert disposition

| Code | Concept | Definition |
|---|---|---|
|  | <a id="ScreeningPending"></a>pending review (`pwm:ScreeningPending`) | Alert raised, not yet reviewed. |
|  | <a id="ScreeningFalsePositive"></a>false positive (`pwm:ScreeningFalsePositive`) | Reviewed; not the listed party. |
|  | <a id="ScreeningTrueMatch"></a>true match (`pwm:ScreeningTrueMatch`) | Reviewed; confirmed as the listed party. |
|  | <a id="ScreeningEscalated"></a>escalated (`pwm:ScreeningEscalated`) | Referred to compliance for decision. |

<a id="ModuleSuitability"></a>

## Module 3: Investor profile and suitability

Regulatory investor classification, risk profile, knowledge and experience, sustainability preferences, and suitability and appropriateness.

### Classes

<a id="InvestorClassification"></a>
#### investor classification (`pwm:InvestorClassification`)

Regulatory categorisation of a client under one jurisdiction's rules, such as retail, professional or accredited, with any opt-in or opt-out recorded.

- **Specialises:** [assessment](#Assessment)
- **Related match:** [qualified investor restriction](https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesRestrictions/QualifiedInvestorRestriction) (FIBO, Release)
- **Handling:** medium context, default classification C2
- **Properties:** [`investorCategory`](#investorCategory), [`optStatus`](#optStatus)
- **References:** **REG** [FinSA](#ref-CH-FINSA) Art. 4 (client segmentation), Art. 5 (opting out and opting in); **REG** [MiFID II](#ref-EU-MIFID2) Annex II (professional clients); Art. 30 (eligible counterparties); **REG** [HK Professional Investor Rules](#ref-HK-PI-RULES) Professional investor tests; **REG** [SFC Code of Conduct](#ref-HK-SFC-COC) para 15; **REG** [Securities and Futures Act (SG)](#ref-SG-SFA) s4A; opt-in regime for accredited investors; **REG** [FCA COBS](#ref-UK-FCA-COBS) COBS 3; **REG** [SEC Regulation D](#ref-US-REG-D) Rule 501(a)

<a id="InvestorRiskProfile"></a>
#### investor risk profile (`pwm:InvestorRiskProfile`)

Versioned assessment of a client's risk tolerance, capacity for loss and investment horizon, expressed on the wealth manager's risk scale.

*Note:* Each change produces a new version that supersedes the last. A suitability assessment cites the version in force at the time.

- **Specialises:** [assessment](#Assessment)
- **Handling:** medium context, default classification C2
- **Properties:** [`riskProfileLevel`](#riskProfileLevel)
- **References:** **ISO** [ISO 22222](#ref-ISO-22222) Gathering client information; **REG** [FinSA](#ref-CH-FINSA) Art. 10 to 14; **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(2): investment objectives including risk tolerance; **REG** [MiFID II Delegated Regulation](#ref-EU-MIFID2-DR) Art. 54; **REG** [SFC Code of Conduct](#ref-HK-SFC-COC) para 5.2; **REG** [MAS FAA-N16](#ref-MAS-FAA-N16) Know-your-customer information

<a id="KnowledgeAndExperienceAssessment"></a>
#### knowledge and experience assessment (`pwm:KnowledgeAndExperienceAssessment`)

Assessment of whether a client understands the risks of a type of product or service, from education, work and past investments.

- **Specialises:** [assessment](#Assessment)
- **Handling:** default classification C2
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(2), 25(3); **REG** [MiFID II Delegated Regulation](#ref-EU-MIFID2-DR) Art. 55; **REG** [MAS FAA-N16](#ref-MAS-FAA-N16) Assessment of customer knowledge and experience for specified investment products; **REG** [MAS SFA 04-N12](#ref-MAS-SFA04-N12) Assessment of customer knowledge and experience for specified investment products

<a id="FinancialSituationAssessment"></a>
#### financial situation assessment (`pwm:FinancialSituationAssessment`)

Assessment of a client's income, assets, liabilities and liquidity needs, including the ability to bear losses.

- **Specialises:** [assessment](#Assessment)
- **Handling:** default classification C2
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(2): financial situation including ability to bear losses; **REG** [MiFID II Delegated Regulation](#ref-EU-MIFID2-DR) Art. 54

<a id="InvestmentObjective"></a>
#### investment objective (`pwm:InvestmentObjective`)

Financial objective a client states for their investments, used to judge whether an investment is appropriate for them.

- **Specialises:** [investment objective](https://spec.edmcouncil.org/fibo/ontology/FND/GoalsAndObjectives/Objectives/InvestmentObjective) (FIBO, Release)
- **Exact match:** [investment objective](https://spec.edmcouncil.org/fibo/ontology/FND/GoalsAndObjectives/Objectives/InvestmentObjective) (FIBO, Release)
- **Handling:** default classification C2
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(2)

<a id="SuitabilityAssessment"></a>
#### suitability assessment (`pwm:SuitabilityAssessment`)

Determination, before a recommendation or a discretionary decision, that a product or transaction fits the client's objectives, including any sustainability preferences, financial situation, and knowledge and experience, within the mandate.

*Note:* Origination records why a client or product was proposed. Suitability records why it was permitted. Keep them separate.

- **Specialises:** [assessment](#Assessment)
- **Handling:** medium context, default classification C2
- **Properties:** [`assessesRecommendation`](#assessesRecommendation), [`usesRiskProfile`](#usesRiskProfile), [`usesMandate`](#usesMandate), [`usesProductRiskRating`](#usesProductRiskRating), [`usesSustainabilityPreference`](#usesSustainabilityPreference)
- **References:** **REG** [FinSA](#ref-CH-FINSA) Art. 10 to 14; **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(2); **REG** [MiFID II Delegated Regulation](#ref-EU-MIFID2-DR) Art. 54; **REG** [SFC Code of Conduct](#ref-HK-SFC-COC) para 5.2; **REG** [MAS FAA-N16](#ref-MAS-FAA-N16) Know-your-customer and suitability of recommendations; **REG** [SEC Regulation Best Interest](#ref-US-REG-BI) Care obligation; **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) suitability: the pack requires the flag and does not compute suitability; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 10: origination says why proposed; suitability says why permitted

<a id="AppropriatenessAssessment"></a>
#### appropriateness assessment (`pwm:AppropriatenessAssessment`)

Check, for services without advice, that a client has the knowledge and experience to understand the risks of the product.

- **Specialises:** [assessment](#Assessment)
- **Handling:** default classification C2
- **References:** **REG** [FinSA](#ref-CH-FINSA) Art. 10 to 14; **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(3); **REG** [MiFID II Delegated Regulation](#ref-EU-MIFID2-DR) Art. 56; **REG** [FCA COBS](#ref-UK-FCA-COBS) COBS 10A

<a id="SuitabilityReport"></a>
#### suitability report (`pwm:SuitabilityReport`)

Statement given to the client explaining how advice meets their preferences, objectives and other characteristics.

- **Specialises:** [document](https://www.omg.org/spec/Commons/Documents/Document) (OMG Commons)
- **Handling:** default classification C2
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(6); **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) suitability-report: a draft from a meeting transcript needs a person before use

<a id="ClientAcknowledgement"></a>
#### client acknowledgement (`pwm:ClientAcknowledgement`)

Record that a client received, read or accepted a disclosure, risk warning or document, and when.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** default classification C2
- **References:** **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Evidence of disclosure

<a id="SustainabilityPreference"></a>
#### sustainability preference (`pwm:SustainabilityPreference`)

Versioned record of a client's choice of one or more sustainability categories, minimum proportions and PAI elements, or an explicit absence of preference.

*Note:* Proportions are decimals from 0 to 1. Categories may be combined; no preference is exclusive. This records preferences and does not certify product sustainability or implement the full Article 54 assessment.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`sustainabilityPreferenceKind`](#sustainabilityPreferenceKind), [`minimumTaxonomyAlignedProportion`](#minimumTaxonomyAlignedProportion), [`minimumSFDRSustainableProportion`](#minimumSFDRSustainableProportion), [`paiConsiderationElements`](#paiConsiderationElements)
- **References:** **REG** [MiFID sustainability preferences](#ref-EU-SUSTAINABILITY) 2021/1253 Art. 1(1), 1(6); 2017/565 Art. 2(7), 54(2)(a), (5), (10)

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="investorCategory"></a>`pwm:investorCategory` investor category | [investor classification](#InvestorClassification) → [investor categories](#InvestorCategoryScheme) | Category assigned by an investor classification. | **REG** [FinSA](#ref-CH-FINSA) Art. 4; **REG** [Securities and Futures Act (SG)](#ref-SG-SFA) s4A |
| <a id="optStatus"></a>`pwm:optStatus` opt status | [investor classification](#InvestorClassification) → [opt statuses](#OptStatusScheme) | Whether the classification is the default, or results from the client opting in or out. | **REG** [FinSA](#ref-CH-FINSA) Art. 5; **REG** [Securities and Futures Act (SG)](#ref-SG-SFA) Opt-in regime |
| <a id="riskProfileLevel"></a>`pwm:riskProfileLevel` risk profile level | [investor risk profile](#InvestorRiskProfile) → [illustrative risk scale](#IllustrativeRiskScale) | Point on the wealth manager's risk scale assigned by a risk profile. | **REG** [MiFID II Delegated Regulation](#ref-EU-MIFID2-DR) Art. 54 |
| <a id="capacityForLoss"></a>`pwm:capacityForLoss` capacity for loss | any → [rating levels](#RatingLevelScheme) | Client's ability to bear losses without a material change to their standard of living or objectives. | **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(2): ability to bear losses |
| <a id="suitabilityOutcome"></a>`pwm:suitabilityOutcome` suitability outcome | any → [suitability outcomes](#SuitabilityOutcomeScheme) | Result of a suitability or appropriateness assessment. | **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(2), 25(3) |
| <a id="assessesRecommendation"></a>`pwm:assessesRecommendation` assesses recommendation | [suitability assessment](#SuitabilityAssessment) → [recommendation](#Recommendation) | Recommendation that a suitability assessment evaluates. Sub-property of [assesses](#assesses). | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 10 |
| <a id="usesRiskProfile"></a>`pwm:usesRiskProfile` uses risk profile | [suitability assessment](#SuitabilityAssessment) → [investor risk profile](#InvestorRiskProfile) | Version of the client's risk profile in force when the assessment was made. | **REG** [MiFID II Delegated Regulation](#ref-EU-MIFID2-DR) Art. 54 |
| <a id="usesMandate"></a>`pwm:usesMandate` uses mandate | [suitability assessment](#SuitabilityAssessment) → [mandate](#Mandate) | Mandate whose terms and restrictions the assessment was checked against. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) mandate |
| <a id="usesProductRiskRating"></a>`pwm:usesProductRiskRating` uses product risk rating | [suitability assessment](#SuitabilityAssessment) → [product risk rating](#ProductRiskRating) | Product risk rating in force when the assessment was made. | **REG** [MiFID II](#ref-EU-MIFID2) Art. 24(2) |
| <a id="usesSustainabilityPreference"></a>`pwm:usesSustainabilityPreference` uses sustainability preference | [suitability assessment](#SuitabilityAssessment) → [sustainability preference](#SustainabilityPreference) | Version of the client sustainability preference considered by a suitability assessment. | **REG** [MiFID sustainability preferences](#ref-EU-SUSTAINABILITY) 2021/1253 Art. 1(1), 1(6); 2017/565 Art. 2(7), 54(2)(a), (5), (10) |
| <a id="sustainabilityPreferenceKind"></a>`pwm:sustainabilityPreferenceKind` sustainability preference kind | [sustainability preference](#SustainabilityPreference) → [sustainability preference categories](#SustainabilityPreferenceScheme) | One or more client-selected sustainability categories, or explicit no preference. | **REG** [MiFID sustainability preferences](#ref-EU-SUSTAINABILITY) 2021/1253 Art. 1(1), 1(6); 2017/565 Art. 2(7), 54(2)(a), (5), (10) |
| <a id="investmentHorizonYears"></a>`pwm:investmentHorizonYears` investment horizon (years) | any → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Length of time, in years, the client intends to hold the investments. | **REG** [MiFID II Delegated Regulation](#ref-EU-MIFID2-DR) Art. 54 |
| <a id="minimumTaxonomyAlignedProportion"></a>`pwm:minimumTaxonomyAlignedProportion` minimum taxonomy aligned proportion | [sustainability preference](#SustainabilityPreference) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Client minimum proportion invested in environmentally sustainable investments under Regulation (EU) 2020/852 Article 2(1), as a fraction from 0 to 1. | **REG** [MiFID sustainability preferences](#ref-EU-SUSTAINABILITY) 2021/1253 Art. 1(1), 1(6); 2017/565 Art. 2(7), 54(2)(a), (5), (10) |
| <a id="minimumSFDRSustainableProportion"></a>`pwm:minimumSFDRSustainableProportion` minimum SFDR sustainable proportion | [sustainability preference](#SustainabilityPreference) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Client minimum proportion invested in sustainable investments under Regulation (EU) 2019/2088 Article 2(17), as a fraction from 0 to 1. | **REG** [MiFID sustainability preferences](#ref-EU-SUSTAINABILITY) 2021/1253 Art. 1(1), 1(6); 2017/565 Art. 2(7), 54(2)(a), (5), (10) |
| <a id="paiConsiderationElements"></a>`pwm:paiConsiderationElements` PAI consideration elements | [sustainability preference](#SustainabilityPreference) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Client-determined qualitative or quantitative elements demonstrating consideration of principal adverse impacts on sustainability factors. | **REG** [MiFID sustainability preferences](#ref-EU-SUSTAINABILITY) 2021/1253 Art. 1(1), 1(6); 2017/565 Art. 2(7), 54(2)(a), (5), (10) |

### Code lists

<a id="InvestorCategoryScheme"></a>
#### investor categories (`pwm:InvestorCategoryScheme`)

Two generic top concepts, retail and non-retail, with jurisdiction-specific categories beneath them. Thresholds stay in each jurisdiction's rules, not in the ontology.

Used by: [`appliesToInvestorCategory`](#appliesToInvestorCategory), [`investorCategory`](#investorCategory)

References: **REG** [FinSA](#ref-CH-FINSA) Art. 4; **REG** [MiFID II](#ref-EU-MIFID2) Annex II; Art. 30; **REG** [HK Professional Investor Rules](#ref-HK-PI-RULES) Professional investor; **REG** [Securities and Futures Act (SG)](#ref-SG-SFA) s4A; **REG** [FCA COBS](#ref-UK-FCA-COBS) COBS 3; **REG** [SEC Regulation D](#ref-US-REG-D) Rule 501(a)

| Code | Concept | Definition |
|---|---|---|
|  | <a id="RetailCategory"></a>retail (`pwm:RetailCategory`) | Client given the full protections of the jurisdiction's conduct rules. |
|  | <a id="RetailInvestorSG"></a>↳ retail investor (SG) (`pwm:RetailInvestorSG`) | Not an accredited, expert or institutional investor under the Securities and Futures Act. |
|  | <a id="PrivateClientCH"></a>↳ private client (CH) (`pwm:PrivateClientCH`) | Private client under FinSA Art. 4. |
|  | <a id="RetailClientEU"></a>↳ retail client (EU) (`pwm:RetailClientEU`) | Client that is not a professional client under MiFID II. |
|  | <a id="RetailClientUK"></a>↳ retail client (UK) (`pwm:RetailClientUK`) | Retail client under COBS 3. |
|  | <a id="NonRetailCategory"></a>non-retail (`pwm:NonRetailCategory`) | Client treated as able to bear more risk, with some protections relaxed. |
|  | <a id="AccreditedInvestorSG"></a>↳ accredited investor (SG) (`pwm:AccreditedInvestorSG`) | Accredited investor under s4A of the Securities and Futures Act, subject to the opt-in regime. |
|  | <a id="ExpertInvestorSG"></a>↳ expert investor (SG) (`pwm:ExpertInvestorSG`) | Expert investor under s4A of the Securities and Futures Act. |
|  | <a id="InstitutionalInvestorSG"></a>↳ institutional investor (SG) (`pwm:InstitutionalInvestorSG`) | Institutional investor under s4A of the Securities and Futures Act. |
|  | <a id="ProfessionalInvestorHK"></a>↳ professional investor (HK) (`pwm:ProfessionalInvestorHK`) | Professional investor under the Securities and Futures Ordinance and the Professional Investor Rules. |
|  | <a id="ProfessionalClientCH"></a>↳ professional client (CH) (`pwm:ProfessionalClientCH`) | Professional client under FinSA Art. 4. |
|  | <a id="InstitutionalClientCH"></a>↳ institutional client (CH) (`pwm:InstitutionalClientCH`) | Institutional client under FinSA Art. 4. |
|  | <a id="ProfessionalClientEU"></a>↳ professional client (EU) (`pwm:ProfessionalClientEU`) | Professional client under MiFID II Annex II, per se or on request. |
|  | <a id="EligibleCounterpartyEU"></a>↳ eligible counterparty (EU) (`pwm:EligibleCounterpartyEU`) | Eligible counterparty under MiFID II Art. 30. |
|  | <a id="ProfessionalClientUK"></a>↳ professional client (UK) (`pwm:ProfessionalClientUK`) | Professional client under COBS 3. |
|  | <a id="AccreditedInvestorUS"></a>↳ accredited investor (US) (`pwm:AccreditedInvestorUS`) | Accredited investor under Regulation D, Rule 501(a). |

<a id="OptStatusScheme"></a>
#### opt statuses (`pwm:OptStatusScheme`)

Directions differ by jurisdiction. In Singapore an eligible investor opts in to accredited-investor status and gives up some retail protections. Under FinSA a high-net-worth retail client opts out to be treated as a professional client, and some professional clients opt in to be treated as retail. Record the resulting category on the classification.

Used by: [`optStatus`](#optStatus)

References: **REG** [FinSA](#ref-CH-FINSA) Art. 5; **REG** [Securities and Futures Act (SG)](#ref-SG-SFA) Opt-in regime

| Code | Concept | Definition |
|---|---|---|
|  | <a id="OptDefault"></a>default (`pwm:OptDefault`) | Category applies by default, with no election. |
|  | <a id="OptedIn"></a>opted in (`pwm:OptedIn`) | Client made an opt-in election under the jurisdiction's rules. Which category that moves the client into, and whether protection rises or falls, depends on the jurisdiction, so record the classification it produced. |
|  | <a id="OptedOut"></a>opted out (`pwm:OptedOut`) | Client made an opt-out election under the jurisdiction's rules. Which category that moves the client into, and whether protection rises or falls, depends on the jurisdiction, so record the classification it produced. |

<a id="SuitabilityOutcomeScheme"></a>
#### suitability outcomes (`pwm:SuitabilityOutcomeScheme`)



Used by: [`suitabilityOutcome`](#suitabilityOutcome)

References: **REG** [FinSA](#ref-CH-FINSA) Art. 10 to 14; **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(2), 25(3)

| Code | Concept | Definition |
|---|---|---|
|  | <a id="Suitable"></a>suitable (`pwm:Suitable`) | Fits the profile and mandate. |
|  | <a id="SuitableWithConditions"></a>suitable with conditions (`pwm:SuitableWithConditions`) | Fits if stated conditions are met, such as a size limit. |
|  | <a id="NotSuitable"></a>not suitable (`pwm:NotSuitable`) | Does not fit; may not be recommended. |
|  | <a id="InsufficientInformation"></a>insufficient information (`pwm:InsufficientInformation`) | Cannot be assessed; no recommendation may be made. |

<a id="IllustrativeRiskScale"></a>
#### illustrative risk scale (`pwm:IllustrativeRiskScale`)

Placeholder five-point scale. An institution replaces it with its own scale in its extension.

Used by: [`productRiskLevel`](#productRiskLevel), [`riskProfileLevel`](#riskProfileLevel), [`targetRiskProfile`](#targetRiskProfile)

References: **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Illustrative only

| Code | Concept | Definition |
|---|---|---|
| RP1 | <a id="RP1"></a>conservative (`pwm:RP1`) | Capital preservation first. |
| RP2 | <a id="RP2"></a>moderately conservative (`pwm:RP2`) | Income with limited volatility. |
| RP3 | <a id="RP3"></a>balanced (`pwm:RP3`) | Growth and income in balance. |
| RP4 | <a id="RP4"></a>growth (`pwm:RP4`) | Growth with higher volatility. |
| RP5 | <a id="RP5"></a>aggressive (`pwm:RP5`) | Maximum growth; accepts large drawdowns. |

<a id="SustainabilityPreferenceScheme"></a>
#### sustainability preference categories (`pwm:SustainabilityPreferenceScheme`)

Three categories under Article 2(7) of 2017/565, plus explicit no preference for recording.

Used by: [`sustainabilityPreferenceKind`](#sustainabilityPreferenceKind)

References: **REG** [MiFID sustainability preferences](#ref-EU-SUSTAINABILITY) 2021/1253 Art. 1(1), 1(6); 2017/565 Art. 2(7), 54(2)(a), (5), (10)

| Code | Concept | Definition |
|---|---|---|
|  | <a id="TaxonomyAlignedMinimum"></a>taxonomy aligned minimum (`pwm:TaxonomyAlignedMinimum`) | Article 2(7)(a): minimum proportion of environmentally sustainable investments. |
|  | <a id="SFDRSustainableMinimum"></a>SFDR sustainable minimum (`pwm:SFDRSustainableMinimum`) | Article 2(7)(b): minimum proportion of sustainable investments. |
|  | <a id="PAIConsideration"></a>PAI consideration (`pwm:PAIConsideration`) | Article 2(7)(c): principal adverse impacts considered using client-determined elements. |
|  | <a id="NoSustainabilityPreference"></a>no sustainability preference (`pwm:NoSustainabilityPreference`) | Explicit record that the client has no sustainability preference; not a fourth Article 2(7) category. |

<a id="ModuleMandates"></a>

## Module 4: Mandates, restrictions and fees

The service model agreed with the client, its investment restrictions and its charges.

### Classes

<a id="Mandate"></a>
#### mandate (`pwm:Mandate`)

Written agreement under which a wealth manager serves a client on a stated service model: advisory, discretionary, execution-only or custody-only. Investment restrictions apply where the service model calls for them.

- **Specialises:** [written contract](https://spec.edmcouncil.org/fibo/ontology/FND/Agreements/Contracts/WrittenContract) (FIBO, Release)
- **Handling:** medium context, default classification C2
- **Properties:** [`hasRestriction`](#hasRestriction)
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 4(1)(4), 4(1)(8), 25(4); **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) mandate: the discussion must sit inside the recorded mandate

<a id="AdvisoryMandate"></a>
#### advisory mandate (`pwm:AdvisoryMandate`)

Mandate under which the wealth manager makes personal recommendations and the client takes each investment decision.

- **Specialises:** [mandate](#Mandate)
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 4(1)(4): investment advice; **REG** [MAS FAA-N16](#ref-MAS-FAA-N16) Recommendations

<a id="DiscretionaryMandate"></a>
#### discretionary mandate (`pwm:DiscretionaryMandate`)

Mandate under which the wealth manager takes investment decisions for the client within agreed guidelines.

- **Specialises:** [mandate](#Mandate)
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 4(1)(8): portfolio management

<a id="ExecutionOnlyArrangement"></a>
#### execution-only arrangement (`pwm:ExecutionOnlyArrangement`)

Arrangement under which the wealth manager executes client instructions without advice.

- **Specialises:** [mandate](#Mandate)
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(4)

<a id="CustodyOnlyArrangement"></a>
#### custody-only arrangement (`pwm:CustodyOnlyArrangement`)

Arrangement under which the wealth manager holds assets and services them, without advice or dealing.

- **Specialises:** [mandate](#Mandate)
- **References:** **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Custody-only service model

<a id="InvestmentRestriction"></a>
#### investment restriction (`pwm:InvestmentRestriction`)

Provision of a mandate that limits what may be held or done: asset-class bands, concentration limits, excluded issuers or sectors, prohibited instruments, leverage limits.

- **Specialises:** [contractual commitment](https://spec.edmcouncil.org/fibo/ontology/FND/Agreements/Contracts/ContractualCommitment) (FIBO, Release)
- **Handling:** medium context
- **Properties:** [`limitValue`](#limitValue), [`limitBasis`](#limitBasis)
- **References:** **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) mandate; concentration

<a id="ConcentrationLimit"></a>
#### concentration limit (`pwm:ConcentrationLimit`)

Investment restriction capping exposure to one issuer, sector, country, currency or product type as a share of the portfolio.

- **Specialises:** [investment restriction](#InvestmentRestriction)
- **References:** **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) concentration: inside the recorded concentration limit

<a id="ModelPortfolio"></a>
#### model portfolio (`pwm:ModelPortfolio`)

Target allocation, for a risk profile or strategy, from which client portfolios are built and against which drift is measured.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C1
- **References:** **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Model-based portfolio management

<a id="Benchmark"></a>
#### benchmark (`pwm:Benchmark`)

Index or composite against which a portfolio's performance is evaluated.

- **Close match:** [portfolio benchmark](https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/CollectiveInvestmentVehicles/PortfolioBenchmark) (FIBO, Provisional)
- **Handling:** slow context
- **References:** **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) FIBO's PortfolioBenchmark is Provisional, so it is matched, not specialised

<a id="Fee"></a>
#### fee (`pwm:Fee`)

Charge for services under a mandate or account, of a stated type.

- **Specialises:** [fee](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/Fee) (FIBO, Release)
- **Properties:** [`feeType`](#feeType)
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 24(4): information on costs and charges

<a id="ThirdPartyPayment"></a>
#### third-party payment (`pwm:ThirdPartyPayment`)

Payment or non-monetary benefit the wealth manager receives from a third party, such as a product provider, in connection with a client service. It is not a charge to the client.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Also called:** inducement, retrocession
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 24(9)

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="governedByMandate"></a>`pwm:governedByMandate` governed by mandate | any → [mandate](#Mandate) | Mandate under which an account or portfolio is managed or advised. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) mandate |
| <a id="hasRestriction"></a>`pwm:hasRestriction` has restriction | [mandate](#Mandate) → [investment restriction](#InvestmentRestriction) | Investment restriction that forms part of a mandate. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) mandate; concentration |
| <a id="feeType"></a>`pwm:feeType` fee type | [fee](#Fee) → [fee types](#FeeTypeScheme) | Kind of fee. | **REG** [MiFID II](#ref-EU-MIFID2) Art. 24(4) |
| <a id="targetRiskProfile"></a>`pwm:targetRiskProfile` target risk profile | any → [illustrative risk scale](#IllustrativeRiskScale) | Risk profile level a model portfolio or product is designed for. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Model portfolio design |
| <a id="hasBenchmark"></a>`pwm:hasBenchmark` has benchmark | any → [benchmark](#Benchmark) | Benchmark a mandate or portfolio is measured against. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Performance measurement |
| <a id="limitValue"></a>`pwm:limitValue` limit value | [investment restriction](#InvestmentRestriction) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Numeric limit of a restriction, read together with its limit basis. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) concentration |
| <a id="limitBasis"></a>`pwm:limitBasis` limit basis | [investment restriction](#InvestmentRestriction) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | What the limit value is measured against, such as percent of portfolio market value. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Restriction definition |

### Code lists

<a id="FeeTypeScheme"></a>
#### fee types (`pwm:FeeTypeScheme`)



Used by: [`feeType`](#feeType)

References: **REG** [MiFID II](#ref-EU-MIFID2) Art. 24(4)

| Code | Concept | Definition |
|---|---|---|
|  | <a id="FeeManagement"></a>management fee (`pwm:FeeManagement`) | For discretionary portfolio management. |
|  | <a id="FeeAdvisory"></a>advisory fee (`pwm:FeeAdvisory`) | For advice under an advisory mandate. |
|  | <a id="FeeCustody"></a>custody fee (`pwm:FeeCustody`) | For safekeeping and servicing of assets. |
|  | <a id="FeeTransaction"></a>transaction fee (`pwm:FeeTransaction`) | Per transaction, such as brokerage. |
|  | <a id="FeePerformance"></a>performance fee (`pwm:FeePerformance`) | Contingent on returns above a hurdle or high-water mark. |

<a id="ModuleAccounts"></a>

## Module 5: Accounts, portfolios and holdings

Where assets are held, what is held, and how it is valued.

### Classes

<a id="Account"></a>
#### account (`pwm:Account`)

Record of a business arrangement between a client and a wealth manager, in which balances and transactions are kept.

- **Specialises:** [account](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/Account) (FIBO, Release)
- **Handling:** medium context, default classification C2
- **Properties:** [`accountHeldBy`](#accountHeldBy), [`hasPortfolio`](#hasPortfolio)
- **References:** **REG** [Banking Act (SG)](#ref-SG-BANKING-ACT) s47: customer information includes particulars of an account

<a id="InvestmentAccount"></a>
#### investment account (`pwm:InvestmentAccount`)

Account that records securities and cash held for investment.

- **Specialises:** [account](#Account), [investment account](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/InvestmentAccount) (FIBO, Release)
- **Handling:** default classification C2
- **References:** **ONT** [FIBO](#ref-FIBO) FBC/ProductsAndServices/ClientsAndAccounts/InvestmentAccount, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Specialises FIBO InvestmentAccount within the PWM account hierarchy

<a id="CustodyAccount"></a>
#### custody account (`pwm:CustodyAccount`)

Investment account in which the wealth manager holds a client's securities in safekeeping.

- **Specialises:** [investment account](#InvestmentAccount)
- **Also called:** securities account
- **Handling:** default classification C2
- **References:** **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) No FIBO class for custody accounts at the version checked

<a id="CashAccount"></a>
#### cash account (`pwm:CashAccount`)

Current or deposit account in a single currency, linked to a client's investment activity.

- **Specialises:** [account](#Account), [customer account](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/CustomerAccount) (FIBO, Release), [deposit account](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/DepositAccount) (FIBO, Release)
- **Handling:** fast context, default classification C2
- **References:** **ONT** [FIBO](#ref-FIBO) FBC/ProductsAndServices/ClientsAndAccounts/DepositAccount, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Cash account

<a id="JointAccount"></a>
#### joint account (`pwm:JointAccount`)

Account held by two or more holders, operated under a stated signing rule.

- **Specialises:** [account](#Account)
- **Handling:** default classification C2
- **References:** **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Joint holding and signing rules

<a id="AccountHolder"></a>
#### account holder (`pwm:AccountHolder`)

Role of a party that owns an account.

- **Specialises:** [account holder](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/AccountHolder) (FIBO, Release)
- **Exact match:** [account holder](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/AccountHolder) (FIBO, Release)
- **Handling:** default classification C2
- **References:** **REG** [OECD CRS](#ref-OECD-CRS) Section VIII: account holder

<a id="Portfolio"></a>
#### portfolio (`pwm:Portfolio`)

Collection of holdings managed or advised as a unit under one mandate.

- **Specialises:** [portfolio](https://spec.edmcouncil.org/fibo/ontology/FND/OwnershipAndControl/Ownership/Portfolio) (FIBO, Release)
- **Handling:** medium context, default classification C2
- **Properties:** [`hasHolding`](#hasHolding)
- **References:** **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) review_portfolio tool under contact-frequency

<a id="Holding"></a>
#### holding (`pwm:Holding`)

Quantity of a financial instrument or other asset held in a portfolio or account at a point in time.

- **Specialises:** [holding](https://spec.edmcouncil.org/fibo/ontology/FND/OwnershipAndControl/Ownership/Holding) (FIBO, Release)
- **Handling:** fast context, default classification C2
- **References:** **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 7: fast-moving facts are looked up live

<a id="Position"></a>
#### position (`pwm:Position`)

Financial exposure from owning, borrowing, shorting or contracting, including derivatives and credit.

- **Specialises:** [position](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/FinancialProductsAndServices/Position) (FIBO, Release)
- **Handling:** fast context, default classification C2
- **References:** **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 7

<a id="Valuation"></a>
#### valuation (`pwm:Valuation`)

Monetary value of a portfolio, account or holding at a stated time, produced by a registered calculation service.

*Note:* A figure presented as authoritative must cite its calculation service. A model may explain a valuation; it may not produce one.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** fast context, default classification C2
- **Properties:** [`valuationOf`](#valuationOf)
- **References:** **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Non-negotiable 8: deterministic financial calculations; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 6: calculation provenance

<a id="LombardFacility"></a>
#### Lombard facility (`pwm:LombardFacility`)

Credit extended to a client against a pledged portfolio of financial assets, with an advance rate per asset.

- **Specialises:** [collateralized loan](https://spec.edmcouncil.org/fibo/ontology/LOAN/LoansGeneral/Loans/CollateralizedLoan) (FIBO, Release)
- **Also called:** securities-backed loan
- **Handling:** fast context, default classification C2
- **Properties:** [`securedBy`](#securedBy)
- **References:** **ONT** [FIBO](#ref-FIBO) LOAN/LoansGeneral/Loans/CollateralizedLoan, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Portfolio-backed lending

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="accountHeldBy"></a>`pwm:accountHeldBy` account held by | [account](#Account) → [account holder](#AccountHolder) | Account holder of an account. | **REG** [OECD CRS](#ref-OECD-CRS) Section VIII |
| <a id="hasPortfolio"></a>`pwm:hasPortfolio` has portfolio | [account](#Account) → [portfolio](#Portfolio) | Portfolio recorded in an account. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Account and portfolio |
| <a id="hasHolding"></a>`pwm:hasHolding` has holding | [portfolio](#Portfolio) → [holding](#Holding) | Holding that forms part of a portfolio. Close match: [has member](https://www.omg.org/spec/Commons/Collections/hasMember) (OMG Commons). | **ONT** [OMG Commons](#ref-OMG-COMMONS) Collections/hasMember, verified in https://www.omg.org/spec/Commons/20250801/Collections.rdf; portfolio membership is narrower; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Portfolio composition |
| <a id="inInstrument"></a>`pwm:inInstrument` in instrument | any → [financial instrument](https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/FinancialInstrument) (FIBO, Release) | Financial instrument a holding or position is in. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Holding to instrument |
| <a id="valuationOf"></a>`pwm:valuationOf` valuation of | [valuation](#Valuation) → any | Portfolio, account or holding a valuation measures. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 6 |
| <a id="signingRule"></a>`pwm:signingRule` signing rule | any → [signing rules](#SigningRuleScheme) | Rule for who must sign instructions on a joint account or for an entity. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Signing authority |
| <a id="securedBy"></a>`pwm:securedBy` secured by | [Lombard facility](#LombardFacility) → [portfolio](#Portfolio) | Portfolio pledged as collateral for a credit facility. Close match: [is collateralized by](https://spec.edmcouncil.org/fibo/ontology/FBC/DebtAndEquities/Debt/isCollateralizedBy) (FIBO, Release). | **ONT** [FIBO](#ref-FIBO) FBC/DebtAndEquities/Debt/isCollateralizedBy, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Pledge |
| <a id="amount"></a>`pwm:amount` amount | any → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Numeric amount, read together with its currency. Close match: [has amount](https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/hasAmount) (FIBO, Release). | **ONT** [FIBO](#ref-FIBO) FND/Accounting/CurrencyAmount/hasAmount, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Amount and currency pattern |
| <a id="currency"></a>`pwm:currency` currency | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Currency of an amount, as an ISO 4217 alphabetic code. Close match: [has currency](https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/hasCurrency) (FIBO, Release). | **ISO** [ISO 4217](#ref-ISO-4217) Alphabetic codes |

### Code lists

<a id="SigningRuleScheme"></a>
#### signing rules (`pwm:SigningRuleScheme`)



Used by: [`signingRule`](#signingRule)

References: **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Signing authority

| Code | Concept | Definition |
|---|---|---|
|  | <a id="SignAnyOne"></a>any one to sign (`pwm:SignAnyOne`) | Any single authorised signatory may instruct. |
|  | <a id="SignTwoJointly"></a>two to sign (`pwm:SignTwoJointly`) | Any two authorised signatories together. |
|  | <a id="SignAllJointly"></a>all to sign (`pwm:SignAllJointly`) | Every holder or signatory must sign. |

<a id="ModuleProducts"></a>

## Module 6: Products, distribution and research

The product shelf, product risk and target market, cross-border distribution rules, offering documents and the house view.

### Classes

<a id="Product"></a>
#### product (`pwm:Product`)

Financial product on the wealth manager's shelf, such as a fund, structured note, bond, deposit or credit facility, linked to the instruments that underlie it.

- **Specialises:** [financial product](https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/FinancialProductsAndServices/FinancialProduct) (FIBO, Release)
- **Handling:** slow context, default classification C1
- **Properties:** [`hasUnderlyingInstrument`](#hasUnderlyingInstrument), [`productComplexity`](#productComplexity)
- **References:** **ONT** [FIBO](#ref-FIBO) FBC/ProductsAndServices/FinancialProductsAndServices/FinancialProduct, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Product shelf

<a id="ProductRiskRating"></a>
#### product risk rating (`pwm:ProductRiskRating`)

Wealth manager's assessment of a product's risk on its own scale, used to match products to client risk profiles.

- **Specialises:** [assessment](#Assessment)
- **Related match:** [risk level](https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/CollectiveInvestmentVehicles/RiskLevel) (FIBO, Provisional)
- **Handling:** slow context, default classification C1
- **Properties:** [`productRiskLevel`](#productRiskLevel)
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 16(3), 24(2): product governance

<a id="TargetMarket"></a>
#### target market (`pwm:TargetMarket`)

Description of the clients a product is designed for, and those it is not, by category, knowledge, risk and objectives.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C1
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 16(3), 24(2)

<a id="ApprovedProductList"></a>
#### approved product list (`pwm:ApprovedProductList`)

Versioned list of products approved for distribution from a booking centre.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Also called:** product shelf
- **Handling:** slow context, default classification C1
- **References:** **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) product-eligibility; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Shelf governance

<a id="CrossBorderRule"></a>
#### cross-border rule (`pwm:CrossBorderRule`)

Deterministic rule stating whether an activity, such as solicitation, advice or research distribution, is permitted for a client domicile, booking centre, investor category and product.

*Note:* A rule table evaluated outside the model. It is not AI under MAS AIRG para 1.3 unless a model takes part in the decision. Activity, client domicile and booking centre are required; an absent investor category or product means all. When rules overlap, the most specific rule (most scope values set) decides; on a tie, not permitted wins; with no matching rule the activity is not permitted (fail closed).

- **Specialises:** [policy rule](#PolicyRule)
- **Related match:** [investors' domicile restriction](https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesRestrictions/InvestorsDomicileRestriction) (FIBO, Release)
- **Related match:** [qualified investor restriction](https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesRestrictions/QualifiedInvestorRestriction) (FIBO, Release)
- **Handling:** slow context, default classification C1
- **Properties:** [`appliesToActivity`](#appliesToActivity), [`clientDomicileCode`](#clientDomicileCode), [`bookingCentreCode`](#bookingCentreCode), [`permitted`](#permitted)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 1.3, fn 6: rule-based tools are not AI; **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) client-domicile, booking-centre, rm-location, cross-border-solicitation, research-distribution

<a id="ProductTermSheet"></a>
#### product term sheet (`pwm:ProductTermSheet`)

Offering document summarising a product's terms, payoff, risks and costs.

- **Specialises:** [offering document](https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesIssuance/OfferingDocument) (FIBO, Release)
- **Handling:** slow context, default classification C0
- **References:** **ONT** [FIBO](#ref-FIBO) SEC/Securities/SecuritiesIssuance/OfferingDocument, Release; master source verified 2026-11-03; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Product documentation

<a id="KeyInformationDocument"></a>
#### key information document (`pwm:KeyInformationDocument`)

Short standardised disclosure of a packaged product's nature, risks, costs and potential returns.

- **Specialises:** [offering document](https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesIssuance/OfferingDocument) (FIBO, Release)
- **Close match:** [key investor information document](https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/CollectiveInvestmentVehicles/KeyInvestorInformationDocument) (FIBO, Provisional)
- **Also called:** KID
- **Handling:** slow context, default classification C0
- **References:** **REG** [PRIIPs Regulation](#ref-EU-PRIIPS) Key information document

<a id="ResearchPublication"></a>
#### research publication (`pwm:ResearchPublication`)

Authored, dated and versioned investment research, with effective, last-reviewed, superseded and expiry dates and any distribution restrictions.

- **Specialises:** [document](https://www.omg.org/spec/Commons/Documents/Document) (OMG Commons)
- **Handling:** slow context, default classification C1
- **References:** **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) research-distribution; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 2: research provenance and lifecycle

<a id="HouseView"></a>
#### house view (`pwm:HouseView`)

Wealth manager's current official view on markets and asset allocation. When it changes, recommendations that depend on it are invalidated.

- **Specialises:** [research publication](#ResearchPublication)
- **Also called:** CIO view
- **Handling:** slow context, default classification C1
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 4: house-view change propagation; **AM** [EVOLVE](#ref-EVOLVE) Define truth once: one source for policies

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="productRiskLevel"></a>`pwm:productRiskLevel` product risk level | [product risk rating](#ProductRiskRating) → [illustrative risk scale](#IllustrativeRiskScale) | Level a product risk rating assigns, on a scale that can be compared with client risk profile levels. *Use the same scale as the risk profile, or a published mapping between the two, so a suitability check can compare them.* | **REG** [MiFID II](#ref-EU-MIFID2) Art. 16(3), 24(2): product governance; **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) suitability |
| <a id="hasUnderlyingInstrument"></a>`pwm:hasUnderlyingInstrument` has underlying instrument | [product](#Product) → [financial instrument](https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/FinancialInstrument) (FIBO, Release) | Financial instrument a product consists of or refers to. | **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) Product to instrument |
| <a id="productComplexity"></a>`pwm:productComplexity` product complexity | [product](#Product) → [product complexity classes](#ProductComplexityScheme) | Regulatory complexity class of a product. | **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(4); **REG** [MAS SFA 04-N12](#ref-MAS-SFA04-N12) Specified and excluded investment products |
| <a id="appliesToActivity"></a>`pwm:appliesToActivity` applies to activity | [cross-border rule](#CrossBorderRule) → [cross-border activities](#CrossBorderActivityScheme) | Activity a cross-border rule decides on. Required: a rule without an activity is incomplete, not a rule for every activity. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) cross-border-solicitation, research-distribution |
| <a id="appliesToProduct"></a>`pwm:appliesToProduct` applies to product | any → [product](#Product) | Product a rule covers. Absent when the rule covers all products. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) product-eligibility |
| <a id="appliesToInvestorCategory"></a>`pwm:appliesToInvestorCategory` applies to investor category | any → [investor categories](#InvestorCategoryScheme) | Investor category a rule covers. Absent when the rule covers all investor categories. | **REG** [MiFID II](#ref-EU-MIFID2) Art. 16(3): target market |
| <a id="isin"></a>`pwm:isin` ISIN | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Twelve-character ISO 6166 international securities identification number. | **ISO** [ISO 6166](#ref-ISO-6166) ISIN structure |
| <a id="cfiCode"></a>`pwm:cfiCode` CFI code | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Six-letter ISO 10962 classification of financial instruments code. Close match: [financial instrument classification code](https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesClassification/FinancialInstrumentClassificationCode) (FIBO, Release). | **ISO** [ISO 10962](#ref-ISO-10962) CFI code |
| <a id="clientDomicileCode"></a>`pwm:clientDomicileCode` client domicile code | [cross-border rule](#CrossBorderRule) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Client domicile a rule covers, as an ISO 3166-1 alpha-2 code. Required: a rule without a domicile is incomplete, not a rule for every domicile. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) client-domicile |
| <a id="bookingCentreCode"></a>`pwm:bookingCentreCode` booking centre code | [cross-border rule](#CrossBorderRule) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Booking centre jurisdiction a rule covers, as an ISO 3166-1 alpha-2 code. Required: a rule without a booking centre is incomplete, not a rule for every booking centre. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) booking-centre |
| <a id="permitted"></a>`pwm:permitted` permitted | [cross-border rule](#CrossBorderRule) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether the activity is permitted under the rule. | **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) cross-border-solicitation: the pack enforces the flag the bank recorded |

### Code lists

<a id="ProductComplexityScheme"></a>
#### product complexity classes (`pwm:ProductComplexityScheme`)



Used by: [`productComplexity`](#productComplexity)

References: **REG** [MiFID II](#ref-EU-MIFID2) Art. 25(4); **REG** [MAS SFA 04-N12](#ref-MAS-SFA04-N12) Specified investment products

| Code | Concept | Definition |
|---|---|---|
|  | <a id="ComplexProduct"></a>complex (`pwm:ComplexProduct`) | Requires an assessment of knowledge and experience before sale. |
|  | <a id="SpecifiedInvestmentProductSG"></a>↳ specified investment product (SG) (`pwm:SpecifiedInvestmentProductSG`) | Product that MAS requires a customer knowledge or account assessment for before sale to retail customers. |
|  | <a id="NonComplexProduct"></a>non-complex (`pwm:NonComplexProduct`) | May be sold without an appropriateness check where the rules allow. |
|  | <a id="ExcludedInvestmentProductSG"></a>↳ excluded investment product (SG) (`pwm:ExcludedInvestmentProductSG`) | Product outside the specified-investment-product regime. |

<a id="CrossBorderActivityScheme"></a>
#### cross-border activities (`pwm:CrossBorderActivityScheme`)



Used by: [`appliesToActivity`](#appliesToActivity)

References: **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) cross-border-solicitation, research-distribution

| Code | Concept | Definition |
|---|---|---|
|  | <a id="ActivitySolicitation"></a>solicitation (`pwm:ActivitySolicitation`) | Approaching a client or prospect unprompted. |
|  | <a id="ActivityReverseEnquiry"></a>reverse enquiry (`pwm:ActivityReverseEnquiry`) | Responding to a request the client initiated. |
|  | <a id="ActivityAdvice"></a>advice (`pwm:ActivityAdvice`) | Giving a personal recommendation. |
|  | <a id="ActivityExecution"></a>execution (`pwm:ActivityExecution`) | Executing a client instruction. |
|  | <a id="ActivityMarketing"></a>marketing (`pwm:ActivityMarketing`) | Promoting a product or service. |
|  | <a id="ActivityResearchDistribution"></a>research distribution (`pwm:ActivityResearchDistribution`) | Sending research to a client. |

<a id="ModuleInteractions"></a>

## Module 7: Interactions, advice and communication

Why a client was approached, what was recommended, who decided, and what reached the client.

### Classes

<a id="ClientInteraction"></a>
#### client interaction (`pwm:ClientInteraction`)

One client journey, from its trigger through retrieval, recommendation, review and decision to any communication, recorded under a single identifier.

- **Specialises:** [Activity](http://www.w3.org/ns/prov#Activity) (PROV-O)
- **Handling:** medium context, default classification C2
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 9: interaction record

<a id="Origination"></a>
#### origination (`pwm:Origination`)

Step that selects a client or product for attention, recorded with its trigger, selection criteria, candidate set and reasons, before anything is shown to a relationship manager.

*Note:* An origination may conclude that no contact is recommended, with a reason.

- **Specialises:** [Activity](http://www.w3.org/ns/prov#Activity) (PROV-O)
- **Handling:** default classification C2
- **Properties:** [`originationTrigger`](#originationTrigger), [`doNotContactReason`](#doNotContactReason)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 10: origination evidence; do not contact is a first-class outcome

<a id="Recommendation"></a>
#### recommendation (`pwm:Recommendation`)

Personal recommendation to a client about a product, transaction or allocation, with structured rationale and cited sources.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`originatedBy`](#originatedBy), [`recommendsProduct`](#recommendsProduct)
- **References:** **REG** [MiFID II](#ref-EU-MIFID2) Art. 4(1)(4): investment advice; **REG** [MAS FAA-N16](#ref-MAS-FAA-N16) Recommendations on investment products; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 10: rationale as structured fields with citations

<a id="FinalDecision"></a>
#### final decision (`pwm:FinalDecision`)

Named person's decision on an AI-prepared recommendation or action, approving, modifying, rejecting or escalating it, with what they saw and what they changed.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** default classification C2
- **Properties:** [`decisionOn`](#decisionOn)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11: RM review capture

<a id="ClientCommunication"></a>
#### client communication (`pwm:ClientCommunication`)

Message sent to a client through a channel, classified by communication level, carrying its content provenance and any AI disclosure, and traceable to the final decision it came from.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** default classification C2
- **Properties:** [`derivedFromDecision`](#derivedFromDecision), [`communicationLevel`](#communicationLevel), [`viaChannel`](#viaChannel), [`aiDisclosureGiven`](#aiDisclosureGiven)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.6: inform customers of AI use; channels for redress; **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) client-communication (two signatures), client-communication-content, client-ai-disclosure; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 12: governed client communication

<a id="CommunicationChannel"></a>
#### communication channel (`pwm:CommunicationChannel`)

Means of reaching a client, registered as approved only if it meets archiving and supervision requirements.

- **Handling:** slow context
- **Properties:** [`isApprovedChannel`](#isApprovedChannel)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 12: approved-channel register

<a id="Complaint"></a>
#### complaint (`pwm:Complaint`)

Expression of dissatisfaction by a client, linked to the interaction, output, model version and approver it concerns.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** default classification C2
- **References:** **REG** [MAS Fair Dealing Guidelines](#ref-MAS-FAIR-DEALING) Fair dealing outcomes; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 15: outcomes and near misses

<a id="VulnerabilityIndicator"></a>
#### vulnerability indicator (`pwm:VulnerabilityIndicator`)

Flag that a client may be vulnerable, raised for human review. AI may raise the flag; only a person may decide, and only a governed process may change treatment.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** default classification C3
- **References:** **REG** [FCA FG21/1](#ref-UK-FCA-FG21-1) Fair treatment of vulnerable customers; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 10: vulnerable-client safeguards

<a id="Consent"></a>
#### consent (`pwm:Consent`)

Record of a client's consent to a stated processing purpose, with its scope, date and withdrawal.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Close match:** [Consent](https://w3id.org/dpv#Consent) (DPV)
- **Handling:** default classification C2
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.4(f): consent for sensitive personal data used with AI

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="withClient"></a>`pwm:withClient` with client | any → [client](#Client) | Client an interaction, recommendation or communication concerns. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 9 |
| <a id="partOfInteraction"></a>`pwm:partOfInteraction` part of interaction | any → [client interaction](#ClientInteraction) | Client interaction an origination, recommendation, decision, communication or evidence record belongs to. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 9: one interaction_id spans a whole client journey |
| <a id="originationTrigger"></a>`pwm:originationTrigger` origination trigger | [origination](#Origination) → [origination triggers](#OriginationTriggerScheme) | What started an origination. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) origination.TRIGGERS |
| <a id="doNotContactReason"></a>`pwm:doNotContactReason` do-not-contact reason | [origination](#Origination) → [do-not-contact reasons](#DoNotContactReasonScheme) | Why an origination concluded that no contact is recommended. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) origination.DO_NOT_CONTACT_REASONS |
| <a id="originatedBy"></a>`pwm:originatedBy` originated by | [recommendation](#Recommendation) → [origination](#Origination) | Origination that proposed a recommendation. Sub-property of [wasGeneratedBy](http://www.w3.org/ns/prov#wasGeneratedBy) (PROV-O). | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 10 |
| <a id="recommendsProduct"></a>`pwm:recommendsProduct` recommends product | [recommendation](#Recommendation) → [product](#Product) | Product a recommendation proposes. | **REG** [MiFID II](#ref-EU-MIFID2) Art. 4(1)(4) |
| <a id="decisionOn"></a>`pwm:decisionOn` decision on | [final decision](#FinalDecision) → any | Recommendation or proposed action a final decision rules on. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11 |
| <a id="decisionResult"></a>`pwm:decisionResult` decision result | any → [final decision results](#DecisionResultScheme) | Approve, modify, reject or escalate. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11: outcomes are approve, modify, reject or escalate |
| <a id="derivedFromDecision"></a>`pwm:derivedFromDecision` derived from decision | [client communication](#ClientCommunication) → [final decision](#FinalDecision) | Final decision a client communication traces back to. A communication without one is refused. Sub-property of [wasDerivedFrom](http://www.w3.org/ns/prov#wasDerivedFrom) (PROV-O). | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 12: a communication that does not trace back to a final decision is a DENY |
| <a id="communicationLevel"></a>`pwm:communicationLevel` communication level | [client communication](#ClientCommunication) → [communication levels](#CommunicationLevelScheme) | How consequential a communication is, which sets the controls it must pass. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.COMMUNICATION_LEVELS |
| <a id="viaChannel"></a>`pwm:viaChannel` via channel | [client communication](#ClientCommunication) → [communication channel](#CommunicationChannel) | Channel a communication was sent through. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 12 |
| <a id="contentProvenance"></a>`pwm:contentProvenance` content provenance | any → [content provenance labels](#ContentProvenanceScheme) | Who produced content: a person, AI, AI then edited by a person, or a person with AI assistance. The label survives edits. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11: content provenance label |
| <a id="disclosureTrigger"></a>`pwm:disclosureTrigger` disclosure trigger | any → [AI disclosure triggers](#DisclosureTriggerScheme) | Reason recorded for an AI disclosure: what the AI did that the disclosure describes. Whether disclosure is required is decided by the jurisdiction's disclosure rule and the content provenance. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.DISCLOSURE_TRIGGERS |
| <a id="isApprovedChannel"></a>`pwm:isApprovedChannel` is approved channel | [communication channel](#CommunicationChannel) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether a channel is registered as meeting archiving and supervision requirements. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 12: approved-channel register |
| <a id="aiDisclosureGiven"></a>`pwm:aiDisclosureGiven` AI disclosure given | [client communication](#ClientCommunication) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether the communication told the client that AI was involved. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.6; **KOG** [Kognita private-banking pack](#ref-KOG-PB-PACK) client-ai-disclosure |
| <a id="contentHash"></a>`pwm:contentHash` content hash | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Cryptographic hash of content as rendered, sent or stored, so the content can be proved later without being copied into the record. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 items 11, 12: hash of the content as sent; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7: pinned evidence; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: version ID or content hash; keep the hash when content is deleted |

### Code lists

<a id="CommunicationLevelScheme"></a>
#### communication levels (`pwm:CommunicationLevelScheme`)

Ordered least to most consequential. A higher level keeps every control of the levels below it.

Used by: [`communicationLevel`](#communicationLevel)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 12; **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.COMMUNICATION_LEVELS

| Code | Concept | Definition |
|---|---|---|
| 1 | <a id="CommGeneralInformation"></a>general information (`pwm:CommGeneralInformation`) | Information not specific to the client. |
| 2 | <a id="CommHouseView"></a>house view (`pwm:CommHouseView`) | The institution's published view, unadapted. |
| 3 | <a id="CommContextualisedInsight"></a>contextualised insight (`pwm:CommContextualisedInsight`) | Information related to the client's circumstances, without a recommendation. |
| 4 | <a id="CommInvestmentDiscussion"></a>investment discussion (`pwm:CommInvestmentDiscussion`) | Discussion of options for the client's portfolio. |
| 5 | <a id="CommProductRecommendation"></a>product recommendation (`pwm:CommProductRecommendation`) | Personal recommendation of a product; triggers suitability checks and approval. |
| 6 | <a id="CommTransaction"></a>transaction (`pwm:CommTransaction`) | Communication that confirms or effects a transaction. |

<a id="ContentProvenanceScheme"></a>
#### content provenance labels (`pwm:ContentProvenanceScheme`)



Used by: [`contentProvenance`](#contentProvenance)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11

| Code | Concept | Definition |
|---|---|---|
|  | <a id="ProvenanceHuman"></a>human (`pwm:ProvenanceHuman`) | Written by a person without AI. |
|  | <a id="ProvenanceAI"></a>AI (`pwm:ProvenanceAI`) | Generated by AI. |
|  | <a id="ProvenanceAIHumanEdited"></a>AI then human-edited (`pwm:ProvenanceAIHumanEdited`) | Generated by AI, then edited by a person; stays identifiable as AI-originated. |
|  | <a id="ProvenanceHumanAIAssisted"></a>human with AI assistance (`pwm:ProvenanceHumanAIAssisted`) | Written by a person with AI help. |

<a id="DisclosureTriggerScheme"></a>
#### AI disclosure triggers (`pwm:DisclosureTriggerScheme`)



Used by: [`disclosureTrigger`](#disclosureTrigger)

References: **REG** [MAS AIRG](#ref-MAS-AIRG) 5.6; **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.DISCLOSURE_TRIGGERS

| Code | Concept | Definition |
|---|---|---|
|  | <a id="TriggerAIGeneratedContent"></a>AI generated content (`pwm:TriggerAIGeneratedContent`) | The content was generated by AI. |
|  | <a id="TriggerAssistedAnalysis"></a>assisted analysis (`pwm:TriggerAssistedAnalysis`) | AI assisted the analysis behind the content. |
|  | <a id="TriggerPowersChatbot"></a>powers a chatbot (`pwm:TriggerPowersChatbot`) | The client is talking to an AI system. |
|  | <a id="TriggerMaterialContribution"></a>materially contributed to a decision (`pwm:TriggerMaterialContribution`) | AI materially contributed to a decision affecting the client. |

<a id="OriginationTriggerScheme"></a>
#### origination triggers (`pwm:OriginationTriggerScheme`)



Used by: [`originationTrigger`](#originationTrigger)

References: **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) origination.TRIGGERS

| Code | Concept | Definition |
|---|---|---|
|  | <a id="OriginationEvent"></a>event (`pwm:OriginationEvent`) | A client or market event, such as a maturity or a house-view change. |
|  | <a id="OriginationSchedule"></a>schedule (`pwm:OriginationSchedule`) | A scheduled review. |
|  | <a id="OriginationRMRequest"></a>RM request (`pwm:OriginationRMRequest`) | A relationship manager asked for it. |

<a id="DoNotContactReasonScheme"></a>
#### do-not-contact reasons (`pwm:DoNotContactReasonScheme`)



Used by: [`doNotContactReason`](#doNotContactReason)

References: **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) origination.DO_NOT_CONTACT_REASONS

| Code | Concept | Definition |
|---|---|---|
|  | <a id="DNCLowRelevance"></a>low relevance (`pwm:DNCLowRelevance`) | Not relevant enough to the client. |
|  | <a id="DNCPoorTiming"></a>poor timing (`pwm:DNCPoorTiming`) | Wrong moment for the client. |
|  | <a id="DNCVulnerability"></a>vulnerability (`pwm:DNCVulnerability`) | A vulnerability indicator is open. |
|  | <a id="DNCExcessiveFrequency"></a>excessive frequency (`pwm:DNCExcessiveFrequency`) | Contact-frequency limit reached. |
|  | <a id="DNCRecentRejection"></a>a recent rejection (`pwm:DNCRecentRejection`) | The client recently declined a similar proposal. |
|  | <a id="DNCRelationshipCircumstance"></a>a relationship circumstance (`pwm:DNCRelationshipCircumstance`) | A circumstance in the relationship makes contact inappropriate. |

<a id="DecisionResultScheme"></a>
#### final decision results (`pwm:DecisionResultScheme`)



Used by: [`decisionResult`](#decisionResult)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11

| Code | Concept | Definition |
|---|---|---|
|  | <a id="DecisionApprove"></a>approve (`pwm:DecisionApprove`) | Accepted as prepared. |
|  | <a id="DecisionModify"></a>modify (`pwm:DecisionModify`) | Accepted with changes, recorded as a structured diff. |
|  | <a id="DecisionReject"></a>reject (`pwm:DecisionReject`) | Not accepted. |
|  | <a id="DecisionEscalate"></a>escalate (`pwm:DecisionEscalate`) | Referred to a more senior or specialist decision-maker. |

<a id="ModuleGovernance"></a>

## Module 8: AI use cases, authority and accountability

The AI inventory, risk materiality, agents, delegated authority, autonomy and the people accountable for them.

### Classes

<a id="AIUseCase"></a>
#### AI use case (`pwm:AIUseCase`)

Specific real-world context that an AI model or system is applied to, registered as a versioned inventory entry with its purpose, approved scope including jurisdiction, data, dependencies, lifecycle status, risk materiality, review status and accountable people.

*Note:* The use case, not the model, is the unit of risk assessment. Every policy decision cites the use case it was taken under; an unregistered or retired use case is refused.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C1
- **Properties:** [`approvedJurisdiction`](#approvedJurisdiction), [`modelType`](#modelType), [`lifecycleStatus`](#lifecycleStatus), [`isMediated`](#isMediated), [`riskTier`](#riskTier), [`hasMaterialityAssessment`](#hasMaterialityAssessment), [`retentionPolicy`](#retentionPolicy), [`enforcementAttestation`](#enforcementAttestation), [`deploymentMode`](#deploymentMode)
- **References:** **ISO** [ISO/IEC 42001](#ref-ISO-IEC-42001) AI management system; **REG** [MAS AIRG](#ref-MAS-AIRG) 1.4(c) use case; 4.5 to 4.9 inventory; 4.7 attributes; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: use-case register; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 19: an unregistered use case is refused

<a id="AISystem"></a>
#### AI system (`pwm:AISystem`)

Machine-based system of one or more models and other components that derives outputs, such as predictions, content, recommendations or decisions, from learned premises.

*Note:* Tools whose outputs are fully determined by manually specified rules, and robotic process automation without learning, are not AI for this purpose.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Close match:** [AI System](https://w3id.org/dpv/ai#AISystem) (DPV-AI)
- **Handling:** slow context
- **References:** **ISO** [ISO/IEC 22989](#ref-ISO-IEC-22989) AI system; **REG** [EU AI Act](#ref-EU-AI-ACT) Art. 3(1); **REG** [MAS AIRG](#ref-MAS-AIRG) 1.3 and fn 6 (definition and exclusions); 1.4(b) system

<a id="AIModel"></a>
#### AI model (`pwm:AIModel`)

Method that converts assumptions and input data into outputs such as estimates, decisions or recommendations, pinned to a provider and version.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Close match:** [Model](https://w3id.org/dpv/ai#Model) (DPV-AI)
- **Handling:** slow context
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 1.4(a) model; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.3: pinned provider model version per call

<a id="AIAgent"></a>
#### AI agent (`pwm:AIAgent`)

Software agent that plans and takes actions with tools on behalf of a person, holding its own credentials, a maximum autonomy level, a tool allow-list, guardrails and a deployment version that covers prompt, tools, permissions and data.

*Note:* An agent can never hold an accountability role. Its entitlements are always narrower than those of the person it acts for.

- **Specialises:** [SoftwareAgent](http://www.w3.org/ns/prov#SoftwareAgent) (PROV-O), [software agent](https://www.omg.org/spec/Commons/PartiesAndSituations/SoftwareAgent) (OMG Commons)
- **Close match:** [AI Agent](https://w3id.org/dpv/ai#AIAgent) (DPV-AI)
- **Handling:** slow context
- **Properties:** [`registeredForUseCase`](#registeredForUseCase), [`agentIdentifier`](#agentIdentifier), [`deploymentVersion`](#deploymentVersion), [`maxAutonomyLevel`](#maxAutonomyLevel), [`permittedTool`](#permittedTool), [`deploymentDate`](#deploymentDate), [`agentOwner`](#agentOwner), [`operatesFor`](#operatesFor), [`agentStatus`](#agentStatus), [`approvedDeploymentVersion`](#approvedDeploymentVersion), [`effectiveAutonomyLevel`](#effectiveAutonomyLevel), [`hasBlastRadiusLimit`](#hasBlastRadiusLimit), [`permittedField`](#permittedField), [`deploymentModel`](#deploymentModel)
- **References:** **REG** [IMDA agentic AI framework](#ref-IMDA-MGF-AGENTIC) Agentic AI governance; **REG** [MAS AIRG](#ref-MAS-AIRG) 1.11 (agent risks); fn 25 (agent identifiers, tools, components, guardrails); **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 1: agent identity

<a id="ThirdPartyAI"></a>
#### third-party AI (`pwm:ThirdPartyAI`)

AI system, model, or service using AI, that an external party develops, owns, operates or provides and the institution procures, accesses or relies on, including AI embedded in software services.

- **Specialises:** [AI system](#AISystem)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) fn 10 (definition); 4.2 and fn 21 (embedded AI); 5.10, 5.11 (management)

<a id="ModelProvider"></a>
#### model provider (`pwm:ModelProvider`)

Organisation that supplies AI models or AI services, tracked for concentration, change notification and exit.

- **Specialises:** [service provider](https://www.omg.org/spec/Commons/Organizations/ServiceProvider) (OMG Commons)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.11(c) concentration; 5.11(d) change management; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.7: model-provider register

<a id="ModelCard"></a>
#### model card (`pwm:ModelCard`)

Record of an approved model: provider, name and version, intended use, known limitations, the evaluation relied on, approval date and approver.

- **Specialises:** [document](https://www.omg.org/spec/Commons/Documents/Document) (OMG Commons)
- **Handling:** slow context
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.10: documentation of how third-party AI was tested; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: model card per approved model

<a id="Tool"></a>
#### tool (`pwm:Tool`)

Capability an agent can invoke, such as an MCP tool or API, declaring the action verb it performs, the autonomy level it requires and a maximum duration.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context
- **Properties:** [`requiredAutonomyLevel`](#requiredAutonomyLevel), [`publicContentTool`](#publicContentTool), [`reversibilityClass`](#reversibilityClass), [`compensatingAction`](#compensatingAction), [`maximumDurationSeconds`](#maximumDurationSeconds), [`supportsCancellation`](#supportsCancellation), [`decisionSeparatedSteps`](#decisionSeparatedSteps)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.16(c): govern plugins and APIs; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 3 (per-verb permissions), item 5 (allow-lists), item 16 (maximum duration)

<a id="RiskMaterialityAssessment"></a>
#### risk materiality assessment (`pwm:RiskMaterialityAssessment`)

Assessment of an AI use case on impact, complexity and reliance, including the autonomy granted, recorded before controls (inherent) and after them (residual), and approved by the designated control function.

*Note:* Residual materiality must be within risk appetite before deployment. Kognita's six axes (business criticality, client-impact class, decision consequence, data sensitivity, autonomy, complexity) roll up into the three MAS dimensions.

- **Specialises:** [assessment](#Assessment)
- **Handling:** slow context
- **Properties:** [`impact`](#impact), [`complexity`](#complexity), [`reliance`](#reliance), [`inherentMateriality`](#inherentMateriality), [`residualMateriality`](#residualMateriality), [`businessCriticality`](#businessCriticality), [`clientImpactClass`](#clientImpactClass), [`decisionConsequence`](#decisionConsequence), [`autonomyAxis`](#autonomyAxis)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 4.10 to 4.13; **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Gaps 3 and 4; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: six axes; inherent and residual materiality

<a id="AIRiskAppetite"></a>
#### AI risk appetite (`pwm:AIRiskAppetite`)

Board-approved statement of the AI uses the institution will and will not accept, with quantitative measures such as AI incident counts, material use cases depending on one provider, and use cases breaching performance thresholds.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 3.4(b) and fn 18; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: AI risk appetite as top-level policy; quantitative measures

<a id="AccountabilityRole"></a>
#### accountability role (`pwm:AccountabilityRole`)

Role of a named person accountable for an AI use case. Accountability can be supported by functions but cannot be delegated to an agent.

- **Specialises:** [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons)
- **Restriction:** [is played by](https://www.omg.org/spec/Commons/RolesAndCompositions/isPlayedBy) (OMG Commons) only [anything except AI agent](#AIAgent)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 3.5(d): clear roles and responsibilities; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: an agent can never be named in an owner role; **AM** [Governing agentic AI (essay)](#ref-AM-GOVERNING-AGENTIC-AI) Section 05: a name, not a forum

<a id="BusinessOwner"></a>
#### business owner (`pwm:BusinessOwner`)

Accountability role answerable for an AI use case's business outcome and residual risk acceptance. Every decision under the use case cites this person.

- **Specialises:** [accountability role](#AccountabilityRole)
- **Also called:** accountable owner
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(c): accountable person for monitoring and incidents; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: a use case with no current owner is a DENY

<a id="TechnologyOwner"></a>
#### technology owner (`pwm:TechnologyOwner`)

Accountability role for the technology that delivers an AI use case.

- **Specialises:** [accountability role](#AccountabilityRole)
- **References:** **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases._OWNER_ROLES

<a id="RiskOwner"></a>
#### risk owner (`pwm:RiskOwner`)

Accountability role for an AI use case's residual risk; receives fairness and drift findings.

- **Specialises:** [accountability role](#AccountabilityRole)
- **References:** **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases._OWNER_ROLES

<a id="ModelOwner"></a>
#### model owner (`pwm:ModelOwner`)

Accountability role for the models an AI use case relies on.

- **Specialises:** [accountability role](#AccountabilityRole)
- **References:** **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases._OWNER_ROLES

<a id="KnowledgeLead"></a>
#### Knowledge Lead (`pwm:KnowledgeLead`)

Named owner of an AI workflow's quality standard: defines the standard and primary metric, approves consequential outputs, receives escalations of ambiguity and exceptions, and approves changes to the standard, including those learned from operations.

- **Specialises:** [accountability role](#AccountabilityRole)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: a tier 2+ use case with no current Knowledge Lead is a DENY; **AM** [Governing agentic AI (essay)](#ref-AM-GOVERNING-AGENTIC-AI) Section 05: named owner; **AM** [EVOLVE](#ref-EVOLVE) Stage 4: every AI-powered workflow needs a named Knowledge Lead

<a id="DesignatedControlFunction"></a>
#### designated control function (`pwm:DesignatedControlFunction`)

Control function, independent of a use case's business and technology owners, that is final arbiter of whether a use constitutes AI, owns inventory policy, and approves risk materiality ratings.

- **Specialises:** [accountability role](#AccountabilityRole)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 4.3, 4.9, 4.13; **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Gap 2

<a id="IndependentValidator"></a>
#### independent validator (`pwm:IndependentValidator`)

Person or function, independent of development and deployment, that validates a high-materiality AI use case before deployment and periodically afterwards.

- **Specialises:** [accountability role](#AccountabilityRole)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.18 to 5.21 (pre-deployment review); 5.24 (re-validation); **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: independent validation as separation of duties

<a id="DelegatedAuthority"></a>
#### delegated authority (`pwm:DelegatedAuthority`)

Authorisation that binds a person, an agent, a purpose, a client, permitted actions, permitted data, a period and limits. Specific, temporary, revocable, and ended by task completion as well as by time.

*Note:* A request is allowed only if the person holds the entitlement and the agent holds a live delegation for it.

- **Specialises:** [authorization](https://www.omg.org/spec/Commons/BusinessAuthorizations/Authorization) (OMG Commons)
- **Handling:** slow context, default classification C2
- **Properties:** [`delegatedBy`](#delegatedBy), [`delegatedTo`](#delegatedTo), [`permitsDataClass`](#permitsDataClass), [`endsOnTaskCompletion`](#endsOnTaskCompletion), [`approvalThreshold`](#approvalThreshold), [`parentDelegation`](#parentDelegation), [`delegationDepth`](#delegationDepth), [`maxDelegationDepth`](#maxDelegationDepth)
- **References:** **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Non-negotiable 2: explicit delegated authority; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 2: delegated authority

<a id="AutonomyPromotionGate"></a>
#### autonomy promotion gate (`pwm:AutonomyPromotionGate`)

Evidence thresholds an agent must meet over a minimum sample and window before its autonomy level may rise, and below which it steps down automatically.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context
- **Properties:** [`qualityThreshold`](#qualityThreshold), [`exceptionRateThreshold`](#exceptionRateThreshold), [`requiresFullTraceability`](#requiresFullTraceability), [`minimumSample`](#minimumSample), [`minimumWindowSeconds`](#minimumWindowSeconds), [`stepDownThreshold`](#stepDownThreshold)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 3: evidence gates for promotion; automatic step-down; **AM** [Governing agentic AI (essay)](#ref-AM-GOVERNING-AGENTIC-AI) Section 04: autonomy is a licence, not a launch setting; **AM** [EVOLVE](#ref-EVOLVE) Stage 4: allow more autonomy only when quality metrics improve

<a id="OutcomeMetric"></a>
#### outcome metric (`pwm:OutcomeMetric`)

Primary metric of an AI use case, cycle time, decision quality or reliability, with a baseline and target computed from the institution's own records.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Properties:** [`primaryMetric`](#primaryMetric), [`baselineValue`](#baselineValue), [`targetValue`](#targetValue)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 7; item 8; **AM** [EVOLVE](#ref-EVOLVE) Measure what matters

<a id="Purpose"></a>
#### purpose (`pwm:Purpose`)

Declared reason for processing data or invoking AI. Access is bound to it; a request without a registered purpose is refused.

- **Close match:** [Purpose](https://w3id.org/dpv#Purpose) (DPV)
- **Handling:** slow context
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 4.7: purpose and description; **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Non-negotiable 3: purpose-bound access; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 11: who, why, what, where before retrieval

<a id="KillSwitch"></a>
#### kill switch (`pwm:KillSwitch`)

Mechanism that deactivates an AI system expeditiously when it exceeds risk tolerances, effective on the next decision, with a tested activation protocol and fallback.

- **Handling:** slow context
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.3 and fn 29; 5.23(b); **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Non-negotiable 12

<a id="KillSwitchDrill"></a>
#### kill-switch drill (`pwm:KillSwitchDrill`)

Test activation of a kill switch, recording who activated it, how long until the next decision was denied, and how the fallback was reached.

- **Specialises:** [Activity](http://www.w3.org/ns/prov#Activity) (PROV-O)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.3: activation protocols tested regularly; **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Gap 7

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="hasPurpose"></a>`pwm:hasPurpose` has purpose | any → [purpose](#Purpose) | Registered purpose of a use case, delegation or request. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.7 |
| <a id="usesDataClass"></a>`pwm:usesDataClass` uses data class | any → [data classifications](#DataClassificationScheme) | Data classification a use case is permitted to process. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.7: data used; 5.4(d): data classification |
| <a id="dependsOn"></a>`pwm:dependsOn` depends on | any → any | AI system, model, provider or other use case that a use case relies on. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.7: dependencies; fn 26 |
| <a id="lifecycleStatus"></a>`pwm:lifecycleStatus` lifecycle status | [AI use case](#AIUseCase) → [use-case lifecycle statuses](#UseCaseStatusScheme) | Lifecycle status of a use case: pilot, live, suspended or retired. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.7: lifecycle status; **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.STATUSES |
| <a id="riskTier"></a>`pwm:riskTier` risk tier | [AI use case](#AIUseCase) → [use-case risk tiers](#RiskTierScheme) | Risk tier derived from a use case's classification, which sets its minimum controls and maximum autonomy. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: risk tier 1 to 4 |
| <a id="hasMaterialityAssessment"></a>`pwm:hasMaterialityAssessment` has materiality assessment | [AI use case](#AIUseCase) → [risk materiality assessment](#RiskMaterialityAssessment) | Risk materiality assessment of a use case. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.10 |
| <a id="impact"></a>`pwm:impact` impact | [risk materiality assessment](#RiskMaterialityAssessment) → [rating levels](#RatingLevelScheme) | Impact rating: consequence of failure or poor performance for the institution, its customers and stakeholders, including the sensitivity of the data processed. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.12(a) |
| <a id="complexity"></a>`pwm:complexity` complexity | [risk materiality assessment](#RiskMaterialityAssessment) → [rating levels](#RatingLevelScheme) | Complexity rating: from the technology, novelty, data and explainability, and for third-party AI the visibility into it. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.12(b) |
| <a id="reliance"></a>`pwm:reliance` reliance | [risk materiality assessment](#RiskMaterialityAssessment) → [rating levels](#RatingLevelScheme) | Reliance rating: how far the decision or output rests on AI, including the autonomy granted and the degree of human oversight. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.12(c) |
| <a id="inherentMateriality"></a>`pwm:inherentMateriality` inherent materiality | [risk materiality assessment](#RiskMaterialityAssessment) → [rating levels](#RatingLevelScheme) | Materiality before controls are applied. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.11 |
| <a id="residualMateriality"></a>`pwm:residualMateriality` residual materiality | [risk materiality assessment](#RiskMaterialityAssessment) → [rating levels](#RatingLevelScheme) | Materiality after controls; must be within risk appetite before deployment. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.11 |
| <a id="businessCriticality"></a>`pwm:businessCriticality` business criticality | [risk materiality assessment](#RiskMaterialityAssessment) → [rating levels](#RatingLevelScheme) | Kognita classification axis: how critical the use case is to the business. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.BUSINESS_CRITICALITY |
| <a id="clientImpactClass"></a>`pwm:clientImpactClass` client-impact class | [risk materiality assessment](#RiskMaterialityAssessment) → [client-impact classes](#ClientImpactClassScheme) | Kognita classification axis: how directly the use case reaches clients. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.CLIENT_IMPACT |
| <a id="decisionConsequence"></a>`pwm:decisionConsequence` decision consequence | [risk materiality assessment](#RiskMaterialityAssessment) → [decision consequences](#DecisionConsequenceScheme) | Kognita classification axis: the kind of decision the use case informs or takes. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.DECISION_CONSEQUENCE |
| <a id="autonomyAxis"></a>`pwm:autonomyAxis` autonomy axis | [risk materiality assessment](#RiskMaterialityAssessment) → [use-case autonomy axis](#AutonomyAxisScheme) | Kognita classification axis: autonomy granted to the use case. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.AUTONOMY_AXIS |
| <a id="hasAccountabilityRole"></a>`pwm:hasAccountabilityRole` has accountability role | any → [accountability role](#AccountabilityRole) | Links a use case or agent to an accountability role held for it. The person holding the role is given on the role, with is played by. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: owner roles; accountability matrix |
| <a id="registeredForUseCase"></a>`pwm:registeredForUseCase` registered for use case | [AI agent](#AIAgent) → [AI use case](#AIUseCase) | Use case an agent is registered to act under. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 1: linked use cases |
| <a id="maxAutonomyLevel"></a>`pwm:maxAutonomyLevel` maximum autonomy level | [AI agent](#AIAgent) → [agent autonomy levels](#AutonomyLevelScheme) | Highest autonomy level an agent may operate at, capped by its use case's risk tier. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 3 |
| <a id="requiredAutonomyLevel"></a>`pwm:requiredAutonomyLevel` required autonomy level | [tool](#Tool) → [agent autonomy levels](#AutonomyLevelScheme) | Autonomy level a tool requires. A call above the agent's level is refused; a call at L4 routes to human approval. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 3 |
| <a id="actionVerb"></a>`pwm:actionVerb` action verb | any → [action verbs](#ActionVerbScheme) | Action a tool performs, or a delegation permits. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 3: action permissions are per verb |
| <a id="permittedTool"></a>`pwm:permittedTool` permitted tool | [AI agent](#AIAgent) → [tool](#Tool) | Tool on an agent's allow-list. | **REG** [MAS AIRG](#ref-MAS-AIRG) fn 25: tools and systems the agent can access; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 5 |
| <a id="boundaryTier"></a>`pwm:boundaryTier` boundary tier | any → [boundary tiers](#BoundaryTierScheme) | Whether an action is always permitted, needs asking first, or is never permitted for an agent. | **AM** [Governing agentic AI (essay)](#ref-AM-GOVERNING-AGENTIC-AI) Section 03: three-tier boundaries; **AM** [EVOLVE](#ref-EVOLVE) Stage 4: the three-tier boundary system |
| <a id="delegatedBy"></a>`pwm:delegatedBy` delegated by | [delegated authority](#DelegatedAuthority) → any | Person who grants a delegated authority. They must hold the entitlement themselves. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 2: dual check |
| <a id="delegatedTo"></a>`pwm:delegatedTo` delegated to | [delegated authority](#DelegatedAuthority) → [AI agent](#AIAgent) | Agent that receives a delegated authority. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 2 |
| <a id="aboutClient"></a>`pwm:aboutClient` about client | any → [client](#Client) | Client a delegation, claim or fact is about. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 2; 0.6 item 1 |
| <a id="permitsDataClass"></a>`pwm:permitsDataClass` permits data class | [delegated authority](#DelegatedAuthority) → [data classifications](#DataClassificationScheme) | Highest data classification a delegation permits the agent to access. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 2: allowed data |
| <a id="primaryMetric"></a>`pwm:primaryMetric` primary metric | [outcome metric](#OutcomeMetric) → [primary outcome metrics](#PrimaryMetricScheme) | Kind of an outcome metric. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.PRIMARY_METRICS |
| <a id="providedBy"></a>`pwm:providedBy` provided by | any → [model provider](#ModelProvider) | Model provider that supplies a model or AI service. Close match: [is provided by](https://www.omg.org/spec/Commons/Organizations/isProvidedBy) (OMG Commons). | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.11; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7: pinned evidence |
| <a id="approvedJurisdiction"></a>`pwm:approvedJurisdiction` approved jurisdiction | [AI use case](#AIUseCase) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Jurisdiction in which a use case is approved for use, as an ISO 3166-1 alpha-2 code. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.7: approved scope of use (e.g., jurisdiction) |
| <a id="modelType"></a>`pwm:modelType` model type | [AI use case](#AIUseCase) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Kind of model a use case relies on, such as a classifier, ranking model or large language model. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.7 |
| <a id="isMediated"></a>`pwm:isMediated` is mediated | [AI use case](#AIUseCase) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether a use case runs through the institution's AI control plane. Unmediated entries record the assurance gap and compensating control. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.2, 4.4: AI in third-party services and shadow AI; **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Gap 1 |
| <a id="agentIdentifier"></a>`pwm:agentIdentifier` agent identifier | [AI agent](#AIAgent) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Readable, unique identifier of an agent, such as AGENT-RM-PRECALL-01. | **REG** [MAS AIRG](#ref-MAS-AIRG) fn 25: agent-specific identifiers; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 1 |
| <a id="deploymentVersion"></a>`pwm:deploymentVersion` deployment version | [AI agent](#AIAgent) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Hash over an agent's system prompt, instructions, tools, permissions, data sources, orchestration and memory behaviour. Any change needs re-approval before the agent can act. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.25: change management; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 1: deployment version covers behaviour |
| <a id="endsOnTaskCompletion"></a>`pwm:endsOnTaskCompletion` ends on task completion | [delegated authority](#DelegatedAuthority) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether a delegation is revoked when its task completes, not only when its period ends. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 2: authority ends when the task does |
| <a id="approvalThreshold"></a>`pwm:approvalThreshold` approval threshold | [delegated authority](#DelegatedAuthority) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Statement of which actions under a delegation require human approval. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 2 |
| <a id="qualityThreshold"></a>`pwm:qualityThreshold` quality threshold | [autonomy promotion gate](#AutonomyPromotionGate) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Minimum share of outputs meeting the agreed standard, as a fraction from 0 to 1. | **AM** [EVOLVE](#ref-EVOLVE) Quality at least 95 percent |
| <a id="exceptionRateThreshold"></a>`pwm:exceptionRateThreshold` exception rate threshold | [autonomy promotion gate](#AutonomyPromotionGate) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Maximum share of runs that leave the always-do path, as a fraction from 0 to 1. | **AM** [EVOLVE](#ref-EVOLVE) Exception rate below 5 percent |
| <a id="requiresFullTraceability"></a>`pwm:requiresFullTraceability` requires full traceability | [autonomy promotion gate](#AutonomyPromotionGate) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether every action in the evaluation window must be reconstructable for the gate to pass. | **AM** [EVOLVE](#ref-EVOLVE) Evidence fully traceable |
| <a id="baselineValue"></a>`pwm:baselineValue` baseline value | [outcome metric](#OutcomeMetric) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Measured value of an outcome metric before the use case. | **AM** [EVOLVE](#ref-EVOLVE) Outcome baselines |
| <a id="targetValue"></a>`pwm:targetValue` target value | [outcome metric](#OutcomeMetric) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Agreed target for an outcome metric. | **AM** [EVOLVE](#ref-EVOLVE) Targets agreed with the process owner |

### Code lists

<a id="AutonomyLevelScheme"></a>
#### agent autonomy levels (`pwm:AutonomyLevelScheme`)

Early private-banking deployments are expected at L1 to L4. Raising a level is a governed change needing the accountable owner and independent validation. The four-step gate in the Filing Cabinet talk (suggest, draft, act with approval, act alone) corresponds to L2, L3, L4 and L5 or above.

Used by: [`effectiveAutonomyLevel`](#effectiveAutonomyLevel), [`fromAutonomyLevel`](#fromAutonomyLevel), [`maxAutonomyLevel`](#maxAutonomyLevel), [`requiredAutonomyLevel`](#requiredAutonomyLevel), [`scopeAutonomyLevel`](#scopeAutonomyLevel), [`toAutonomyLevel`](#toAutonomyLevel)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 3: autonomy levels; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 12: reading is not the same as acting

| Code | Concept | Definition |
|---|---|---|
| L0 | <a id="L0Retrieve"></a>retrieve (`pwm:L0Retrieve`) | Fetch authorised information. |
| L1 | <a id="L1Explain"></a>explain (`pwm:L1Explain`) | Summarise or explain retrieved information. |
| L2 | <a id="L2Recommend"></a>recommend (`pwm:L2Recommend`) | Propose an option. Recommending never implies permission to execute. |
| L3 | <a id="L3Draft"></a>draft (`pwm:L3Draft`) | Prepare an artefact a person reviews before use. |
| L4 | <a id="L4ActAfterApproval"></a>act after approval (`pwm:L4ActAfterApproval`) | Take an action once a named person, not the requester, has approved that exact action. |
| L5 | <a id="L5BoundedAutonomy"></a>bounded autonomous action (`pwm:L5BoundedAutonomy`) | Act without per-action approval inside explicit limits. |
| L6 | <a id="L6BroadAutonomy"></a>broad autonomy (`pwm:L6BroadAutonomy`) | Act across tasks with broad discretion. |

<a id="BoundaryTierScheme"></a>
#### boundary tiers (`pwm:BoundaryTierScheme`)

Set per action for each agent, independently of the autonomy-level scale. As a rule of thumb, internal work at L0 to L3 sits in always do, and L4 in ask first.

Used by: [`boundaryTier`](#boundaryTier)

References: **AM** [Governing agentic AI (essay)](#ref-AM-GOVERNING-AGENTIC-AI) Section 03; **AM** [EVOLVE](#ref-EVOLVE) Stage 4: the three-tier boundary system

| Code | Concept | Definition |
|---|---|---|
|  | <a id="AlwaysDo"></a>always do (`pwm:AlwaysDo`) | Safe, reversible, inside the mandate and already authorised, such as retrieving authorised data or drafting an internal briefing. |
|  | <a id="AskFirst"></a>ask first (`pwm:AskFirst`) | Consequential or client-facing; the agent prepares and a person decides. |
|  | <a id="NeverDo"></a>never do (`pwm:NeverDo`) | Outside this agent's scope, whatever it is asked, enforced in its permissions rather than by asking. Which actions sit here is set per agent, for example sending client communications for an agent that only drafts. |

<a id="ActionVerbScheme"></a>
#### action verbs (`pwm:ActionVerbScheme`)



Used by: [`actionVerb`](#actionVerb)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 3: read, analyse, recommend, draft, queue, send and execute

| Code | Concept | Definition |
|---|---|---|
|  | <a id="VerbRead"></a>read (`pwm:VerbRead`) | Read information. |
|  | <a id="VerbAnalyse"></a>analyse (`pwm:VerbAnalyse`) | Derive insight from information. |
|  | <a id="VerbRecommend"></a>recommend (`pwm:VerbRecommend`) | Propose an option. |
|  | <a id="VerbDraft"></a>draft (`pwm:VerbDraft`) | Prepare an artefact. |
|  | <a id="VerbQueue"></a>queue (`pwm:VerbQueue`) | Place an artefact for release by a person. |
|  | <a id="VerbSend"></a>send (`pwm:VerbSend`) | Release an artefact outside the institution. |
|  | <a id="VerbExecute"></a>execute (`pwm:VerbExecute`) | Change a system of record or move value. |

<a id="UseCaseStatusScheme"></a>
#### use-case lifecycle statuses (`pwm:UseCaseStatusScheme`)



Used by: [`agentStatus`](#agentStatus), [`lifecycleStatus`](#lifecycleStatus)

References: **REG** [MAS AIRG](#ref-MAS-AIRG) 4.7; 5.2 fn 28 (pilots); 5.26 (decommissioning); **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.STATUSES

| Code | Concept | Definition |
|---|---|---|
|  | <a id="StatusPilot"></a>pilot (`pwm:StatusPilot`) | Limited by end date, named users and success criteria. Close match: [Deployment Stage](https://w3id.org/dpv/ai#DeploymentStage) (DPV-AI). |
|  | <a id="StatusLive"></a>live (`pwm:StatusLive`) | In production use. Close match: [Operation Stage](https://w3id.org/dpv/ai#OperationStage) (DPV-AI). |
|  | <a id="StatusSuspended"></a>suspended (`pwm:StatusSuspended`) | Stopped pending review; decisions are refused. |
|  | <a id="StatusRetired"></a>retired (`pwm:StatusRetired`) | Decommissioned; its agents are stopped and retention applies. Close match: [Decommission Stage](https://w3id.org/dpv/ai#DecommissionStage) (DPV-AI). |

<a id="RiskTierScheme"></a>
#### use-case risk tiers (`pwm:RiskTierScheme`)



Used by: [`riskTier`](#riskTier)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: risk tier 1 to 4

| Code | Concept | Definition |
|---|---|---|
| 1 | <a id="Tier1"></a>tier 1 (`pwm:Tier1`) | Research summarisation. |
| 2 | <a id="Tier2"></a>tier 2 (`pwm:Tier2`) | Client intelligence and meeting preparation. |
| 3 | <a id="Tier3"></a>tier 3 (`pwm:Tier3`) | KYC, suitability and recommendations. |
| 4 | <a id="Tier4"></a>tier 4 (`pwm:Tier4`) | Financial execution. |

<a id="RatingLevelScheme"></a>
#### rating levels (`pwm:RatingLevelScheme`)

Used for impact, complexity, reliance, inherent and residual materiality, business criticality and capacity for loss.

Used by: [`businessCriticality`](#businessCriticality), [`capacityForLoss`](#capacityForLoss), [`complexity`](#complexity), [`impact`](#impact), [`inherentMateriality`](#inherentMateriality), [`reliance`](#reliance), [`residualMateriality`](#residualMateriality)

References: **REG** [MAS AIRG](#ref-MAS-AIRG) 4.12; **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.MATERIALITY, BUSINESS_CRITICALITY

| Code | Concept | Definition |
|---|---|---|
| 1 | <a id="RatingLow"></a>low (`pwm:RatingLow`) | Low. |
| 2 | <a id="RatingMedium"></a>medium (`pwm:RatingMedium`) | Medium. |
| 3 | <a id="RatingHigh"></a>high (`pwm:RatingHigh`) | High; material. |
| 4 | <a id="RatingCritical"></a>critical (`pwm:RatingCritical`) | Critical; material. |

<a id="ClientImpactClassScheme"></a>
#### client-impact classes (`pwm:ClientImpactClassScheme`)



Used by: [`clientImpactClass`](#clientImpactClass)

References: **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Naming collisions: client-impact class; **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.CLIENT_IMPACT

| Code | Concept | Definition |
|---|---|---|
| 1 | <a id="ImpactInternalProductivity"></a>internal productivity (`pwm:ImpactInternalProductivity`) | Staff productivity with no client reach. |
| 2 | <a id="ImpactRMAssistance"></a>RM assistance (`pwm:ImpactRMAssistance`) | Helps a relationship manager prepare. |
| 3 | <a id="ImpactClientInfluence"></a>client influence (`pwm:ImpactClientInfluence`) | Shapes what a client is told or offered. |
| 4 | <a id="ImpactClientCommunication"></a>client communication (`pwm:ImpactClientCommunication`) | Produces content that reaches a client. |
| 5 | <a id="ImpactAdvice"></a>advice (`pwm:ImpactAdvice`) | Contributes to a personal recommendation. |
| 6 | <a id="ImpactExecution"></a>execution (`pwm:ImpactExecution`) | Contributes to executing a transaction. |

<a id="DecisionConsequenceScheme"></a>
#### decision consequences (`pwm:DecisionConsequenceScheme`)



Used by: [`decisionConsequence`](#decisionConsequence)

References: **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.DECISION_CONSEQUENCE

| Code | Concept | Definition |
|---|---|---|
|  | <a id="ConsequenceInformational"></a>informational (`pwm:ConsequenceInformational`) | Informs only. |
|  | <a id="ConsequenceOperational"></a>operational (`pwm:ConsequenceOperational`) | Changes an internal operation. |
|  | <a id="ConsequenceClientCommunication"></a>client communication (`pwm:ConsequenceClientCommunication`) | Determines what reaches a client. |
|  | <a id="ConsequenceSuitability"></a>suitability (`pwm:ConsequenceSuitability`) | Bears on whether advice is suitable. |
|  | <a id="ConsequenceFinancialAction"></a>financial action (`pwm:ConsequenceFinancialAction`) | Moves value or changes a position. |

<a id="AutonomyAxisScheme"></a>
#### use-case autonomy axis (`pwm:AutonomyAxisScheme`)



Used by: [`autonomyAxis`](#autonomyAxis)

References: **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.AUTONOMY_AXIS

| Code | Concept | Definition |
|---|---|---|
|  | <a id="AutonomyNone"></a>none (`pwm:AutonomyNone`) | No autonomous action. |
|  | <a id="AutonomyAssisted"></a>assisted (`pwm:AutonomyAssisted`) | Assists a person who acts. |
|  | <a id="AutonomyConditional"></a>conditional (`pwm:AutonomyConditional`) | Acts under conditions or approval. |
|  | <a id="AutonomyHigh"></a>high (`pwm:AutonomyHigh`) | Acts with little human involvement. |

<a id="PrimaryMetricScheme"></a>
#### primary outcome metrics (`pwm:PrimaryMetricScheme`)



Used by: [`primaryMetric`](#primaryMetric)

References: **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) usecases.PRIMARY_METRICS; **AM** [EVOLVE](#ref-EVOLVE) Measure what matters

| Code | Concept | Definition |
|---|---|---|
|  | <a id="MetricCycleTime"></a>cycle time (`pwm:MetricCycleTime`) | Time to complete the workflow; breaches above target. |
|  | <a id="MetricDecisionQuality"></a>decision quality (`pwm:MetricDecisionQuality`) | Share of outputs meeting the standard; breaches below target. |
|  | <a id="MetricReliability"></a>reliability (`pwm:MetricReliability`) | Share of runs completing without failure; breaches below target. |
|  | <a id="MetricKnowledgeRetention"></a>knowledge retention (`pwm:MetricKnowledgeRetention`) | Capability that remains when people leave. An EVOLVE measure; not a Kognita primary metric. |

<a id="GovernanceRiskScheme"></a>
#### agentic governance risks (`pwm:GovernanceRiskScheme`)

Five risks that sit on top of model risk when AI moves from assistance to action.

References: **AM** [Governing agentic AI (essay)](#ref-AM-GOVERNING-AGENTIC-AI) Section 02: five risks

| Code | Concept | Definition |
|---|---|---|
|  | <a id="RiskAccess"></a>access (`pwm:RiskAccess`) | An agent inherits whatever its tools can reach. |
|  | <a id="RiskEvidence"></a>evidence (`pwm:RiskEvidence`) | The chain of retrievals, judgements and actions is not kept. |
|  | <a id="RiskOperational"></a>operational (`pwm:RiskOperational`) | Agents loop, stall, call the wrong tool or keep moving when they should stop. |
|  | <a id="RiskChange"></a>change (`pwm:RiskChange`) | Behaviour changes with a prompt, tool, source or silent model update. |
|  | <a id="RiskOwnership"></a>ownership (`pwm:RiskOwnership`) | No named person is accountable for what the agent did. |

<a id="ModuleEvidence"></a>

## Module 9: Information governance, claims and evidence

Governed documents and definitions, facts and claims, policy decisions and the evidence that reconstructs them.

### Classes

<a id="GovernedInformationAsset"></a>
#### governed information asset (`pwm:GovernedInformationAsset`)

Document, record or dataset carrying the minimum governance labels that make it safe to give to AI: owner, approval status, effective date, what it supersedes, classification per section, version identifier or content hash, and retention date.

*Note:* When content is deleted, keep its hash and record the deletion.

- **Specialises:** [document](https://www.omg.org/spec/Commons/Documents/Document) (OMG Commons)
- **Handling:** slow context
- **Properties:** [`hasSection`](#hasSection)
- **References:** **ONT** [DCMI Terms](#ref-DCMI-TERMS) dcterms:valid, dcterms:replaces; **ISO** [ISO 15489](#ref-ISO-15489) Records: authenticity, reliability, integrity, usability; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: minimum labels

<a id="DocumentSection"></a>
#### document section (`pwm:DocumentSection`)

Section of a governed information asset with its own classification, so entitlements apply per section rather than per document.

- **Specialises:** [document](https://www.omg.org/spec/Commons/Documents/Document) (OMG Commons)
- **Handling:** slow context
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1: per-section classification at ingestion; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: classification per section, not per document

<a id="ApprovedSource"></a>
#### approved source (`pwm:ApprovedSource`)

Source registered as approved for a use case's regulated claims, such as CIO research, product documentation, permitted market data or policy. A model's own knowledge is never an approved source.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context
- **Properties:** [`approvedForUseCase`](#approvedForUseCase)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 3: approved-source grounding

<a id="GovernedDefinition"></a>
#### governed definition (`pwm:GovernedDefinition`)

Authoritative, versioned definition of a business term, such as assets under management, net new money or concentration, with an owner. Figures using the term must cite it.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C1
- **Properties:** [`definesTerm`](#definesTerm)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 12: governed business definitions; **AM** [EVOLVE](#ref-EVOLVE) Stage 5: define truth once (one definition of AUM)

<a id="CalculationService"></a>
#### calculation service (`pwm:CalculationService`)

Registered service that produces authoritative figures, such as performance, valuation, concentration, exposure or suitability scores. A model may explain a calculation; it may not invent one.

- **Specialises:** [SoftwareAgent](http://www.w3.org/ns/prov#SoftwareAgent) (PROV-O)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 6: calculation provenance

<a id="Fact"></a>
#### fact (`pwm:Fact`)

Authoritative value from a system of record, carrying its source, owner, as-of time, classification, entitlement and freshness threshold. Stale facts are refreshed, suppressed or flagged, never silently used.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`freshnessThreshold`](#freshnessThreshold)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 2: fact contract and freshness

<a id="Claim"></a>
#### claim (`pwm:Claim`)

Material statement an agent produces for a person or client, typed as verified fact, inference, conversation hypothesis or possible provider capability, with its sources, calculation method, model and prompt versions and validator status.

*Note:* Unlabelled claims cannot be shown. An inference cannot be relabelled as fact without a cited source and verification.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`validatorStatus`](#validatorStatus)
- **References:** **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Non-negotiables 6 and 7; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 items 1, 5, 7: claim provenance, claim types, verification gate

<a id="PolicyRule"></a>
#### policy rule (`pwm:PolicyRule`)

Effective-dated, citable rule evaluated outside the model before data moves or an action runs. Rules are versioned, never edited in place.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C1
- **Properties:** [`ruleCitation`](#ruleCitation)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.25(b): change control and version control; **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Design rule: never rely on the LLM to enforce a control that can be enforced outside it

<a id="PolicyDecision"></a>
#### policy decision (`pwm:PolicyDecision`)

Deterministic outcome of evaluating every applicable rule for a request, allow, deny, escalate or hold for human approval, citing the deciding rule and its version, the use case and any delegation relied on. One failing rule denies.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **Properties:** [`decisionOutcome`](#decisionOutcome), [`decidedByRule`](#decidedByRule), [`hasCheck`](#hasCheck), [`limitObservation`](#limitObservation), [`instructionEvidence`](#instructionEvidence), [`requiresStepUp`](#requiresStepUp), [`sandboxConstraints`](#sandboxConstraints), [`humanApproval`](#humanApproval)
- **References:** **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Non-negotiable 9: runtime policy enforcement outside the LLM; **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.Outcome, OUTCOME_PRECEDENCE; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 14: the AI answer receipt, for approvals and denials

<a id="EvidenceRecord"></a>
#### evidence record (`pwm:EvidenceRecord`)

Append-only, hash-chained record of an event, such as a policy decision, model call, tool call, retrieval, approval or communication, holding hashes and references rather than content, from which an interaction can be reconstructed.

*Note:* Record a model refusal separately from a policy denial. Otherwise the evidence shows only that the model usually behaves, not that the controls work.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **Properties:** [`eventType`](#eventType), [`evidenceFor`](#evidenceFor), [`recordHash`](#recordHash), [`previousRecordHash`](#previousRecordHash), [`chainStart`](#chainStart)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): logging of prompts, responses, model versions; 5.17 reproducibility; **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Non-negotiable 12: complete action audit trail; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 15: append-only, tamper-evident; log refusals and blocks separately

<a id="HumanReview"></a>
#### human review (`pwm:HumanReview`)

Review of AI output or a proposed action by a competent person with authority to intervene, given a complete packet of action, reason, evidence, risks, alternatives and checks run.

- **Specialises:** [Activity](http://www.w3.org/ns/prov#Activity) (PROV-O)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(a) to (d): human oversight; automation bias and decision fatigue; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11 (approval packet); item 13 (risk-based review)

<a id="AIIncident"></a>
#### AI incident (`pwm:AIIncident`)

Incident involving a failure, malfunction or poor performance of an AI system or model, escalated, tracked and linked to the clients affected.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** default classification C2
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) fn 20 (definition); 3.5(e) escalation; 5.23(b) incident management; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14 (affected-client query); 0.7 (incident evidence and report)

<a id="NearMiss"></a>
#### near miss (`pwm:NearMiss`)

Event where harm was avoided, such as a blocked action, a verifier block, a reviewer rejection, an unsafe draft or unusual agent behaviour, aggregated to find weak controls before harm occurs.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** default classification C1
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(d): review incidents and near misses; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 15: near-miss register

<a id="MemoryRecord"></a>
#### memory record (`pwm:MemoryRecord`)

Item an agent retains beyond a single step, scoped as working, relationship or institutional memory and typed like a claim, with retention, a correction path and expiry.

*Note:* An inference becomes a relationship record only after verification by a person. Client data never enters institutional memory.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`memoryScope`](#memoryScope)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 8: memory governance; **AM** [EVOLVE](#ref-EVOLVE) Closed-loop learning; human expertise becomes institutional capability

<a id="Run"></a>
#### run (`pwm:Run`)

Governed execution grouping calls across agent steps, budget snapshots, pending approvals and lifecycle transitions.

- **Specialises:** [Activity](http://www.w3.org/ns/prov#Activity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`hasBudget`](#hasBudget), [`pendingApproval`](#pendingApproval)
- **References:** **ONT** [PROV-O](#ref-W3C-PROV-O) Section 3.1: Activity; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="RunBudget"></a>
#### run budget (`pwm:RunBudget`)

Limits and consumption snapshot for one run. Cost is USD; wall clock is seconds. Evidence records successive snapshots rather than overwriting history.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C2
- **Properties:** [`maxCalls`](#maxCalls), [`maxTokens`](#maxTokens), [`maxCostUSD`](#maxCostUSD), [`wallClockSeconds`](#wallClockSeconds), [`consumedCalls`](#consumedCalls), [`consumedTokens`](#consumedTokens), [`consumedCostUSD`](#consumedCostUSD), [`elapsedSeconds`](#elapsedSeconds), [`classificationCeiling`](#classificationCeiling)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="BudgetExceededDenial"></a>
#### budget exceeded denial (`pwm:BudgetExceededDenial`)

Denial caused by an exceeded run budget, naming the budget and breached dimension.

- **Specialises:** [policy decision](#PolicyDecision)
- **Handling:** medium context, default classification C2
- **Properties:** [`exceededBudget`](#exceededBudget), [`budgetDimension`](#budgetDimension)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="ApprovalRequest"></a>
#### approval request (`pwm:ApprovalRequest`)

Request to a human to approve an exact held policy decision at a run checkpoint.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`requestsDecision`](#requestsDecision), [`atCheckpoint`](#atCheckpoint)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(a) to (d): human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="RunCheckpoint"></a>
#### run checkpoint (`pwm:RunCheckpoint`)

Durable snapshot at a synchronous policy evaluation boundary; contentHash is the continuation handle in the retention store.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(a) to (d): human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="ApprovalResolution"></a>
#### approval resolution (`pwm:ApprovalResolution`)

Human resolution of a specific approval request, approved or denied, with attribution and time.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`resolvesApproval`](#resolvesApproval)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(a) to (d): human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="RunResumption"></a>
#### run resumption (`pwm:RunResumption`)

Evidenced continuation of a checkpointed run after a human resolution; denied resolutions never authorize the held action.

- **Specialises:** [Activity](http://www.w3.org/ns/prov#Activity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`resumedFrom`](#resumedFrom), [`resolvedApproval`](#resolvedApproval)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(a) to (d): human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="GovernedCall"></a>
#### governed call (`pwm:GovernedCall`)

Gateway call carrying a run, policy decision and identity bound to authenticated client configuration.

- **Specialises:** [Activity](http://www.w3.org/ns/prov#Activity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`governedByDecision`](#governedByDecision), [`clientConfiguration`](#clientConfiguration), [`callAgent`](#callAgent), [`systemTrigger`](#systemTrigger)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="ModelCall"></a>
#### model call (`pwm:ModelCall`)

Gateway model invocation recording provider-reported identity, prompt as sent and response as received, template version and token usage.

- **Specialises:** [governed call](#GovernedCall)
- **Handling:** medium context, default classification C2
- **Properties:** [`usesModel`](#usesModel), [`modelName`](#modelName), [`promptTemplateVersion`](#promptTemplateVersion), [`promptHash`](#promptHash), [`inputTokens`](#inputTokens), [`outputTokens`](#outputTokens)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="ToolCall"></a>
#### tool call (`pwm:ToolCall`)

Governed tool invocation with response hash and retained response reference.

- **Specialises:** [governed call](#GovernedCall)
- **Handling:** medium context, default classification C2
- **Properties:** [`invokedTool`](#invokedTool), [`requestedField`](#requestedField), [`callDurationSeconds`](#callDurationSeconds)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="MCPProxiedCall"></a>
#### MCP proxied call (`pwm:MCPProxiedCall`)

Tool invocation mediated by the MCP proxy, recording backend and invoked tool.

- **Specialises:** [tool call](#ToolCall)
- **Handling:** medium context, default classification C2
- **Properties:** [`mcpBackend`](#mcpBackend)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="EnvelopeAttribute"></a>
#### envelope attribute (`pwm:EnvelopeAttribute`)

Recorded request or response attribute, with name, value and typed or classifier-derived source. Typed values are authoritative; classification fills gaps and may only narrow permission. Authentication supplies identity.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`derivedFromClassifier`](#derivedFromClassifier), [`attributeName`](#attributeName), [`attributeValue`](#attributeValue)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="ClassifierOutput"></a>
#### classifier output (`pwm:ClassifierOutput`)

Calibrated classifier evidence with model identifier, version, label, confidence and input hash. Replay uses the recorded label without invoking the classifier.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`classifierModelId`](#classifierModelId), [`classifierVersion`](#classifierVersion), [`classifierLabel`](#classifierLabel), [`confidence`](#confidence), [`inputHash`](#inputHash)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="PolicyCheck"></a>
#### policy check (`pwm:PolicyCheck`)

Individual rule evaluation, pinning the evaluated policy row content hash; a policy decision may contain several checks.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`checkedRule`](#checkedRule), [`policyContentHash`](#policyContentHash)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 14: the rule and rule version that decided it; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="RetrievedItem"></a>
#### retrieved item (`pwm:RetrievedItem`)

Retrieved source snapshot with content hash and embedding model identity.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`embeddingModel`](#embeddingModel)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="RetentionEntry"></a>
#### retention entry (`pwm:RetentionEntry`)

Content-addressed retention store entry, separate from chain evidence. Erasure removes content but preserves its hash and erasure evidence.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`storeLocator`](#storeLocator), [`erased`](#erased)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="RetentionPolicy"></a>
#### retention policy (`pwm:RetentionPolicy`)

Per-use-case policy for how long source snapshots, prompts and responses are retained, subject to lawful erasure.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C1
- **Properties:** [`retentionDuration`](#retentionDuration)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: retention date; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="ErasureEvent"></a>
#### erasure event (`pwm:ErasureEvent`)

Erasure of retained content, recording its hash on the surviving evidence chain.

- **Specialises:** [Activity](http://www.w3.org/ns/prov#Activity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`erasedEntry`](#erasedEntry)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="ReconstructionReport"></a>
#### reconstruction report (`pwm:ReconstructionReport`)

Report for an interaction verifying chain and pinned hashes, listing findings and explicitly covering all ten supervisory questions, including not recorded answers.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`hasFinding`](#hasFinding), [`questionCoverage`](#questionCoverage), [`chainVerified`](#chainVerified), [`pinnedHashesVerified`](#pinnedHashesVerified)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="ReconstructionFinding"></a>
#### reconstruction finding (`pwm:ReconstructionFinding`)

Finding from reconstruction, including hash mismatch, affected record, expected hash and observed hash.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`findingRecord`](#findingRecord), [`expectedHash`](#expectedHash), [`observedHash`](#observedHash), [`findingKind`](#findingKind)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="QuestionCoverage"></a>
#### question coverage (`pwm:QuestionCoverage`)

Answer status for exactly one supervisory reconstruction question; absent evidence is explicitly not recorded.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`answerText`](#answerText)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): record keeping; 5.17: reproducibility; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="AuthenticatedClientConfiguration"></a>
#### authenticated client configuration (`pwm:AuthenticatedClientConfiguration`)

Authenticated gateway client configuration binding permitted agent names and listed system triggers.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C2
- **Properties:** [`boundAgent`](#boundAgent), [`listedTrigger`](#listedTrigger)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 Tier 0; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

<a id="SystemTrigger"></a>
#### system trigger (`pwm:SystemTrigger`)

Listed authenticated system identity admitted for calls without an agent; still subject to purpose, use-case, policy and budget checks.

- **Specialises:** [Agent](http://www.w3.org/ns/prov#Agent) (PROV-O)
- **Handling:** slow context, default classification C2
- **References:** **ONT** [PROV-O](#ref-W3C-PROV-O) Section 3.1: Agent; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 Tier 0; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="informationOwner"></a>`pwm:informationOwner` information owner | any → [party role](https://www.omg.org/spec/Commons/PartiesAndSituations/PartyRole) (OMG Commons) | Role accountable for a governed information asset or definition. | **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: owner |
| <a id="approvalStatus"></a>`pwm:approvalStatus` approval status | any → [information asset statuses](#AssetStatusScheme) | Approval status of a governed information asset: draft, current, superseded or withdrawn. | **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slides 8 and 9: approval status |
| <a id="hasSection"></a>`pwm:hasSection` has section | [governed information asset](#GovernedInformationAsset) → [document section](#DocumentSection) | Section of a governed information asset. | **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8 |
| <a id="classification"></a>`pwm:classification` classification | any → [data classifications](#DataClassificationScheme) | Data classification of an asset, section, fact or record. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.4(d); **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.Classification |
| <a id="approvedForUseCase"></a>`pwm:approvedForUseCase` approved for use case | [approved source](#ApprovedSource) → [AI use case](#AIUseCase) | Use case for which a source is approved. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 3: each use case has a register of approved sources |
| <a id="claimType"></a>`pwm:claimType` claim type | any → [claim types](#ClaimTypeScheme) | Epistemic type of a claim or memory record. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) claims: VERIFIED_FACT, INFERENCE, CONVERSATION_HYPOTHESIS, POSSIBLE_BANK_CAPABILITY |
| <a id="validatorStatus"></a>`pwm:validatorStatus` validator status | [claim](#Claim) → [validator statuses](#ValidatorStatusScheme) | Verdict of an independent verifier on a claim. Policy decides what each status may reach. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 7: verification gate |
| <a id="citesSource"></a>`pwm:citesSource` cites source | any → any | Approved source, governed asset or fact a claim relies on. Sub-property of [wasDerivedFrom](http://www.w3.org/ns/prov#wasDerivedFrom) (PROV-O). | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 items 1, 3 |
| <a id="usesDefinition"></a>`pwm:usesDefinition` uses definition | any → [governed definition](#GovernedDefinition) | Governed definition, at a version, that a claim or figure uses. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 12 |
| <a id="calculatedBy"></a>`pwm:calculatedBy` calculated by | any → [calculation service](#CalculationService) | Calculation service that produced a figure. Sub-property of [wasAttributedTo](http://www.w3.org/ns/prov#wasAttributedTo) (PROV-O). | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 6: calculation_method |
| <a id="staleAction"></a>`pwm:staleAction` stale action | any → [stale-data actions](#StaleActionScheme) | What happens when a fact is older than its threshold. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.StaleAction |
| <a id="decisionOutcome"></a>`pwm:decisionOutcome` decision outcome | [policy decision](#PolicyDecision) → [policy decision outcomes](#DecisionOutcomeScheme) | Outcome of a policy decision. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.Outcome |
| <a id="decidedByRule"></a>`pwm:decidedByRule` decided by rule | [policy decision](#PolicyDecision) → [policy rule](#PolicyRule) | Rule, at its version, that determined a policy decision. | **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 14: the rule and rule version that decided it |
| <a id="underUseCase"></a>`pwm:underUseCase` under use case | any → [AI use case](#AIUseCase) | Registered use case a decision, claim or event was produced under. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: every decision must reference a registered use case |
| <a id="reliesOnDelegation"></a>`pwm:reliesOnDelegation` relies on delegation | any → [delegated authority](#DelegatedAuthority) | Delegated authority a decision or action relied on. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.5 item 2: every decision references the delegation it relied on |
| <a id="eventType"></a>`pwm:eventType` event type | [evidence record](#EvidenceRecord) → [evidence event types](#EvidenceEventTypeScheme) | Kind of event an evidence record captures. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.EventType |
| <a id="evidenceFor"></a>`pwm:evidenceFor` evidence for | [evidence record](#EvidenceRecord) → any | Decision, action, communication or other record that an evidence record documents. The record points to it by identifier and hash and does not copy its content. Sub-property of [References](http://purl.org/dc/terms/references) (DCMI). | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d); **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Hash-chained evidence; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7: pinned evidence |
| <a id="memoryScope"></a>`pwm:memoryScope` memory scope | [memory record](#MemoryRecord) → [memory scopes](#MemoryScopeScheme) | Scope of a memory record. | **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) memory.MemoryScope |
| <a id="hasBudget"></a>`pwm:hasBudget` has budget | [run](#Run) → [run budget](#RunBudget) | Budget snapshot applied to a run. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="inRun"></a>`pwm:inRun` in run | any → [run](#Run) | Run grouping a call, checkpoint, decision or resumption. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="exceededBudget"></a>`pwm:exceededBudget` exceeded budget | [budget exceeded denial](#BudgetExceededDenial) → [run budget](#RunBudget) | Budget whose limit caused this denial. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="pendingApproval"></a>`pwm:pendingApproval` pending approval | [run](#Run) → [approval request](#ApprovalRequest) | Unresolved human approval currently holding the run. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="requestsDecision"></a>`pwm:requestsDecision` requests decision | [approval request](#ApprovalRequest) → [policy decision](#PolicyDecision) | Exact held policy decision requiring approval. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="atCheckpoint"></a>`pwm:atCheckpoint` at checkpoint | [approval request](#ApprovalRequest) → [run checkpoint](#RunCheckpoint) | Checkpoint at which the action was withheld. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="resolvesApproval"></a>`pwm:resolvesApproval` resolves approval | [approval resolution](#ApprovalResolution) → [approval request](#ApprovalRequest) | Approval request resolved by a human. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="resumedFrom"></a>`pwm:resumedFrom` resumed from | [run resumption](#RunResumption) → [run checkpoint](#RunCheckpoint) | Durable checkpoint rehydrated for resumption. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="resolvedApproval"></a>`pwm:resolvedApproval` resolved approval | [run resumption](#RunResumption) → [approval resolution](#ApprovalResolution) | Resolved approval cited by resumption of the same run and checkpoint. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="governedByDecision"></a>`pwm:governedByDecision` governed by decision | [governed call](#GovernedCall) → [policy decision](#PolicyDecision) | Policy authorization evaluated before gateway dispatch. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="usesModel"></a>`pwm:usesModel` uses model | [model call](#ModelCall) → [AI model](#AIModel) | Model invoked by the gateway. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="invokedTool"></a>`pwm:invokedTool` invoked tool | [tool call](#ToolCall) → [tool](#Tool) | Tool invoked through the gateway. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="hasAttribute"></a>`pwm:hasAttribute` has attribute | any → [envelope attribute](#EnvelopeAttribute) | Recorded attribute used in the request or response envelope. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="derivedFromClassifier"></a>`pwm:derivedFromClassifier` derived from classifier | [envelope attribute](#EnvelopeAttribute) → [classifier output](#ClassifierOutput) | Classifier evidence supplying this gap in typed attributes. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="hasCheck"></a>`pwm:hasCheck` has check | [policy decision](#PolicyDecision) → [policy check](#PolicyCheck) | Pinned evaluation of one policy rule contributing to this decision. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="checkedRule"></a>`pwm:checkedRule` checked rule | [policy check](#PolicyCheck) → [policy rule](#PolicyRule) | Policy row evaluated by this check. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="retrievedItem"></a>`pwm:retrievedItem` retrieved item | any → [retrieved item](#RetrievedItem) | Pinned item used by a governed call. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="retainedIn"></a>`pwm:retainedIn` retained in | any → [retention entry](#RetentionEntry) | Store entry holding content keyed by its hash. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="retentionPolicy"></a>`pwm:retentionPolicy` retention policy | [AI use case](#AIUseCase) → [retention policy](#RetentionPolicy) | Retention policy applied to this use case. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="erasedEntry"></a>`pwm:erasedEntry` erased entry | [erasure event](#ErasureEvent) → [retention entry](#RetentionEntry) | Store entry whose content was erased; hash remains. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="hasFinding"></a>`pwm:hasFinding` has finding | [reconstruction report](#ReconstructionReport) → [reconstruction finding](#ReconstructionFinding) | Verification finding included in the report. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="findingRecord"></a>`pwm:findingRecord` finding record | [reconstruction finding](#ReconstructionFinding) → any | Record affected by the finding. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="questionCoverage"></a>`pwm:questionCoverage` question coverage | [reconstruction report](#ReconstructionReport) → [question coverage](#QuestionCoverage) | Explicit coverage record for a supervisory question. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="clientConfiguration"></a>`pwm:clientConfiguration` client configuration | [governed call](#GovernedCall) → [authenticated client configuration](#AuthenticatedClientConfiguration) | Authenticated client configuration authorizing the asserted caller identity. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 Tier 0; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="callAgent"></a>`pwm:callAgent` call agent | [governed call](#GovernedCall) → [AI agent](#AIAgent) | Registered agent identity admitted under the authenticated configuration. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 Tier 0; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="systemTrigger"></a>`pwm:systemTrigger` system trigger | [governed call](#GovernedCall) → [system trigger](#SystemTrigger) | Listed system trigger admitted on a call without an agent. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 Tier 0; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="boundAgent"></a>`pwm:boundAgent` bound agent | [authenticated client configuration](#AuthenticatedClientConfiguration) → [AI agent](#AIAgent) | Agent bound to this authenticated client configuration. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 Tier 0; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="listedTrigger"></a>`pwm:listedTrigger` listed trigger | [authenticated client configuration](#AuthenticatedClientConfiguration) → [system trigger](#SystemTrigger) | System trigger admitted by this authenticated client configuration. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 Tier 0; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="runState"></a>`pwm:runState` run state | any → [run states](#RunStateScheme) | Recorded runState code. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="approvalResolution"></a>`pwm:approvalResolution` approval resolution | any → [approval resolutions](#ApprovalResolutionScheme) | Recorded approvalResolution code. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="attributeSource"></a>`pwm:attributeSource` attribute source | any → [attribute sources](#AttributeSourceScheme) | Recorded attributeSource code. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="coverageStatus"></a>`pwm:coverageStatus` coverage status | any → [question coverage statuses](#QuestionCoverageStatusScheme) | Recorded coverageStatus code. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="gatewayFailureMode"></a>`pwm:gatewayFailureMode` gateway failure mode | any → [gateway failure modes](#GatewayFailureModeScheme) | Recorded gatewayFailureMode code. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 9; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="reconstructionQuestion"></a>`pwm:reconstructionQuestion` reconstruction question | any → [supervisory reconstruction questions](#ReconstructionQuestionScheme) | Recorded reconstructionQuestion code. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="classificationCeiling"></a>`pwm:classificationCeiling` classification ceiling | [run budget](#RunBudget) → [data classifications](#DataClassificationScheme) | Highest data sensitivity the run may touch. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="versionId"></a>`pwm:versionId` version identifier | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Version identifier of an asset, definition, rule, model or deployment. | **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: version ID or content hash |
| <a id="retentionUntil"></a>`pwm:retentionUntil` retention until | any → [date](http://www.w3.org/2001/XMLSchema#date) (XSD) | Date until which a record must be kept. An AI record never expires before the regulatory record it supports. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.7: retention by record class; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: retention date |
| <a id="definesTerm"></a>`pwm:definesTerm` defines term | [governed definition](#GovernedDefinition) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Business term a governed definition defines. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 12 |
| <a id="asOf"></a>`pwm:asOf` as of | any → [dateTime](http://www.w3.org/2001/XMLSchema#dateTime) (XSD) | Time at which a fact or valuation was true in its system of record. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 2: as-of date |
| <a id="freshnessThreshold"></a>`pwm:freshnessThreshold` freshness threshold | [fact](#Fact) → [duration](http://www.w3.org/2001/XMLSchema#duration) (XSD) | Maximum age at which a fact may still be used, as an xsd:duration. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 2: freshness thresholds per data type |
| <a id="ruleCitation"></a>`pwm:ruleCitation` rule citation | [policy rule](#PolicyRule) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Text citing the policy, regulation or standard a rule implements. | **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Every decision citable to policy |
| <a id="recordHash"></a>`pwm:recordHash` record hash | [evidence record](#EvidenceRecord) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Hash of an evidence record, chained to the previous record so deletion or alteration is detectable. | **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Hash-chained evidence |
| <a id="previousRecordHash"></a>`pwm:previousRecordHash` previous record hash | [evidence record](#EvidenceRecord) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Hash of the record that precedes this one in the chain. Required on every record except the first in a chain. | **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Hash-chained evidence |
| <a id="chainStart"></a>`pwm:chainStart` chain start | [evidence record](#EvidenceRecord) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | True on the first record of an evidence chain, the only record without a previous record hash. | **KOG** [Kognita regulations library](#ref-KOG-CONTROLS) Hash-chained evidence |
| <a id="modelVersion"></a>`pwm:modelVersion` model version | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Provider model version as served for a model call, claim or evaluation. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(d): applicable model versions; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7: pinned evidence |
| <a id="maxCalls"></a>`pwm:maxCalls` max calls | [run budget](#RunBudget) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Maximum number of governed calls. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="maxTokens"></a>`pwm:maxTokens` max tokens | [run budget](#RunBudget) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Maximum aggregate model tokens. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="maxCostUSD"></a>`pwm:maxCostUSD` max cost usd | [run budget](#RunBudget) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Maximum spend in US dollars. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="wallClockSeconds"></a>`pwm:wallClockSeconds` wall clock seconds | [run budget](#RunBudget) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Maximum elapsed wall clock seconds. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="consumedCalls"></a>`pwm:consumedCalls` consumed calls | [run budget](#RunBudget) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Calls consumed at this snapshot. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="consumedTokens"></a>`pwm:consumedTokens` consumed tokens | [run budget](#RunBudget) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Tokens consumed at this snapshot. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="consumedCostUSD"></a>`pwm:consumedCostUSD` consumed cost usd | [run budget](#RunBudget) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | US dollars consumed at this snapshot. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="elapsedSeconds"></a>`pwm:elapsedSeconds` elapsed seconds | [run budget](#RunBudget) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Elapsed wall clock seconds at this snapshot. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="budgetDimension"></a>`pwm:budgetDimension` budget dimension | [budget exceeded denial](#BudgetExceededDenial) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Exceeded dimension, such as max_calls or classification_ceiling. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="modelName"></a>`pwm:modelName` model name | [model call](#ModelCall) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Model name as reported for this call. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="promptTemplateVersion"></a>`pwm:promptTemplateVersion` prompt template version | [model call](#ModelCall) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Version of the prompt template used. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="promptHash"></a>`pwm:promptHash` prompt hash | [model call](#ModelCall) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Hash of prompt as sent after redaction. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="responseHash"></a>`pwm:responseHash` response hash | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Hash of model or tool response as received. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="inputTokens"></a>`pwm:inputTokens` input tokens | [model call](#ModelCall) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Provider-reported input token usage. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="outputTokens"></a>`pwm:outputTokens` output tokens | [model call](#ModelCall) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Provider-reported output token usage. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="mcpBackend"></a>`pwm:mcpBackend` mcp backend | [MCP proxied call](#MCPProxiedCall) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Backend MCP server identifier. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="attributeName"></a>`pwm:attributeName` attribute name | [envelope attribute](#EnvelopeAttribute) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Envelope field name; identity fields must come from authentication. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="attributeValue"></a>`pwm:attributeValue` attribute value | [envelope attribute](#EnvelopeAttribute) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Recorded effective attribute value. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="classifierModelId"></a>`pwm:classifierModelId` classifier model id | [classifier output](#ClassifierOutput) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Classifier model identifier. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="classifierVersion"></a>`pwm:classifierVersion` classifier version | [classifier output](#ClassifierOutput) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Classifier model version used for inference. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="classifierLabel"></a>`pwm:classifierLabel` classifier label | [classifier output](#ClassifierOutput) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Label reused during deterministic policy replay. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="confidence"></a>`pwm:confidence` confidence | [classifier output](#ClassifierOutput) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Calibrated confidence in the closed interval zero to one. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="inputHash"></a>`pwm:inputHash` input hash | [classifier output](#ClassifierOutput) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Hash of classified prompt, response or tool input. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="escalationThreshold"></a>`pwm:escalationThreshold` escalation threshold | any → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Policy confidence threshold below which the classifier cannot yield ALLOW. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="policyContentHash"></a>`pwm:policyContentHash` policy content hash | [policy check](#PolicyCheck) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Hash of the exact policy row evaluated by this check. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="embeddingModel"></a>`pwm:embeddingModel` embedding model | [retrieved item](#RetrievedItem) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Embedding model identifier and version used for retrieval. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="retentionDuration"></a>`pwm:retentionDuration` retention duration in days | [retention policy](#RetentionPolicy) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Retention duration for content under this policy, measured in whole days. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="storeLocator"></a>`pwm:storeLocator` store locator | [retention entry](#RetentionEntry) → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | Reference to content in the separate retention store, never embedded chain content. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="erased"></a>`pwm:erased` erased | [retention entry](#RetentionEntry) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether the retained payload was erased. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="expectedHash"></a>`pwm:expectedHash` expected hash | [reconstruction finding](#ReconstructionFinding) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Expected pinned hash for a mismatch finding. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="observedHash"></a>`pwm:observedHash` observed hash | [reconstruction finding](#ReconstructionFinding) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Observed hash of the mismatching content. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="findingKind"></a>`pwm:findingKind` finding kind | [reconstruction finding](#ReconstructionFinding) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Finding type, including hash_mismatch or chain_break. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="answerText"></a>`pwm:answerText` answer text | [question coverage](#QuestionCoverage) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Evidence-based answer, or reason the question was not recorded. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="chainVerified"></a>`pwm:chainVerified` chain verified | [reconstruction report](#ReconstructionReport) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether the evidence chain passed verification. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |
| <a id="pinnedHashesVerified"></a>`pwm:pinnedHashesVerified` pinned hashes verified | [reconstruction report](#ReconstructionReport) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether all available pinned hashes matched; mismatches require findings. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified |

### Code lists

<a id="DataClassificationScheme"></a>
#### data classifications (`pwm:DataClassificationScheme`)

Ordered by sensitivity. Compare by rank, never by label: entitlement filtering and egress control depend on the order.

Used by: [`classification`](#classification), [`classificationCeiling`](#classificationCeiling), [`permitsDataClass`](#permitsDataClass), [`usesDataClass`](#usesDataClass)

References: **REG** [MAS AIRG](#ref-MAS-AIRG) 5.4(d); **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.Classification

| Code | Concept | Definition |
|---|---|---|
| C0 | <a id="C0"></a>public (`pwm:C0`) | Approved for public release. |
| C1 | <a id="C1"></a>internal (`pwm:C1`) | For staff; no client identity. |
| C2 | <a id="C2"></a>confidential (`pwm:C2`) | Identifies a client; within banking secrecy where it applies. Also: client-identifying. Close match: [Personal Data](https://w3id.org/dpv#PersonalData) (DPV). |
| C3 | <a id="C3"></a>restricted (`pwm:C3`) | Need-to-know only, such as screening hits, investigations, special-category data and material non-public information. Also: highly sensitive, need-to-know. Close match: [Special Category Personal Data](https://w3id.org/dpv#SpecialCategoryPersonalData) (DPV). |

<a id="DecisionOutcomeScheme"></a>
#### policy decision outcomes (`pwm:DecisionOutcomeScheme`)

Resolution is fail-closed: the most severe outcome present decides, in the order deny, escalate, human approval, allow.

Used by: [`decisionOutcome`](#decisionOutcome), [`expectedOutcome`](#expectedOutcome)

References: **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.Outcome, OUTCOME_PRECEDENCE

| Code | Concept | Definition |
|---|---|---|
| DENY | <a id="OutcomeDeny"></a>deny (`pwm:OutcomeDeny`) | Refused; nothing moves. |
| ESCALATE | <a id="OutcomeEscalate"></a>escalate (`pwm:OutcomeEscalate`) | Uncertain; referred to a person. |
| HUMAN_APPROVAL | <a id="OutcomeHumanApproval"></a>human approval (`pwm:OutcomeHumanApproval`) | Held until a named person approves this exact action. |
| ALLOW | <a id="OutcomeAllow"></a>allow (`pwm:OutcomeAllow`) | Every rule passed. |
| OBSERVE | <a id="OutcomeObserve"></a>observe (`pwm:OutcomeObserve`) | Recorded without gating, in monitor-only deployments. |

<a id="ClaimTypeScheme"></a>
#### claim types (`pwm:ClaimTypeScheme`)



Used by: [`claimType`](#claimType)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 5; **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) claims.ClaimType

| Code | Concept | Definition |
|---|---|---|
|  | <a id="VerifiedFact"></a>verified fact (`pwm:VerifiedFact`) | Backed by a cited source and verification. |
|  | <a id="Inference"></a>inference (`pwm:Inference`) | Reasoned from facts; not itself verified. |
|  | <a id="ConversationHypothesis"></a>conversation hypothesis (`pwm:ConversationHypothesis`) | Something worth raising with the client, to be tested in conversation. |
|  | <a id="PossibleProviderCapability"></a>possible provider capability (`pwm:PossibleProviderCapability`) | Something the institution might be able to offer, not yet confirmed. Also: possible bank capability. |

<a id="ValidatorStatusScheme"></a>
#### validator statuses (`pwm:ValidatorStatusScheme`)



Used by: [`validatorStatus`](#validatorStatus)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.6 item 7; **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) claims.ValidatorStatus

| Code | Concept | Definition |
|---|---|---|
|  | <a id="ValidatorVerified"></a>verified (`pwm:ValidatorVerified`) | Verifier confirmed it. |
|  | <a id="ValidatorInferred"></a>inferred (`pwm:ValidatorInferred`) | Supported only by inference. |
|  | <a id="ValidatorUnverified"></a>unverified (`pwm:ValidatorUnverified`) | Not yet checked. |
|  | <a id="ValidatorConflicting"></a>conflicting (`pwm:ValidatorConflicting`) | Sources disagree. |
|  | <a id="ValidatorStale"></a>stale (`pwm:ValidatorStale`) | Relies on data past its freshness threshold. |
|  | <a id="ValidatorBlocked"></a>blocked (`pwm:ValidatorBlocked`) | May not be released. |

<a id="AssetStatusScheme"></a>
#### information asset statuses (`pwm:AssetStatusScheme`)



Used by: [`approvalStatus`](#approvalStatus)

References: **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 9: superseded, current, draft

| Code | Concept | Definition |
|---|---|---|
|  | <a id="AssetDraft"></a>draft (`pwm:AssetDraft`) | Not approved; never given to AI as authoritative. |
|  | <a id="AssetCurrent"></a>current (`pwm:AssetCurrent`) | Approved and in force. |
|  | <a id="AssetSuperseded"></a>superseded (`pwm:AssetSuperseded`) | Replaced by a later version; kept for reconstruction only. |
|  | <a id="AssetWithdrawn"></a>withdrawn (`pwm:AssetWithdrawn`) | Withdrawn without replacement. |

<a id="StaleActionScheme"></a>
#### stale-data actions (`pwm:StaleActionScheme`)



Used by: [`staleAction`](#staleAction)

References: **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.StaleAction

| Code | Concept | Definition |
|---|---|---|
|  | <a id="StaleRefresh"></a>refresh (`pwm:StaleRefresh`) | Fetch a current value before use. |
|  | <a id="StaleSuppress"></a>suppress (`pwm:StaleSuppress`) | Do not use the value. |
|  | <a id="StaleWarn"></a>warn (`pwm:StaleWarn`) | Use with a visible staleness warning. |

<a id="MemoryScopeScheme"></a>
#### memory scopes (`pwm:MemoryScopeScheme`)



Used by: [`memoryScope`](#memoryScope)

References: **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) memory.MemoryScope

| Code | Concept | Definition |
|---|---|---|
|  | <a id="MemoryWorking"></a>working (`pwm:MemoryWorking`) | Temporary context; expires with the run. |
|  | <a id="MemoryRelationship"></a>relationship (`pwm:MemoryRelationship`) | Durable client information, governed like any bank record. |
|  | <a id="MemoryInstitutional"></a>institutional (`pwm:MemoryInstitutional`) | What the organisation learns across clients; no client data. |

<a id="EvidenceEventTypeScheme"></a>
#### evidence event types (`pwm:EvidenceEventTypeScheme`)

Notations match Kognita EventType values. Model refusal is added from the Filing Cabinet talk so refusals and policy denials are logged apart.

Used by: [`eventType`](#eventType)

References: **KOG** [Kognita vocabularies](#ref-KOG-VOCAB) vocabulary.EventType; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 15

| Code | Concept | Definition |
|---|---|---|
| POLICY_DECISION | <a id="EventPolicyDecision"></a>policy decision (`pwm:EventPolicyDecision`) | A decision was taken. |
| MODEL_CALL | <a id="EventModelCall"></a>model call (`pwm:EventModelCall`) | A model was called; provider, version and prompt and response hashes recorded. |
| TOOL_CALL | <a id="EventToolCall"></a>tool call (`pwm:EventToolCall`) | A tool was invoked; arguments recorded as hashes. |
| RETRIEVAL | <a id="EventRetrieval"></a>retrieval (`pwm:EventRetrieval`) | Content was retrieved, with item hashes. |
| APPROVAL | <a id="EventApproval"></a>approval (`pwm:EventApproval`) | A person signed an approval. |
| EGRESS | <a id="EventEgress"></a>egress (`pwm:EventEgress`) | Data left for a provider, allowed, redacted or denied. |
| POLICY_CHANGE | <a id="EventPolicyChange"></a>policy change (`pwm:EventPolicyChange`) | A rule or register entry changed. |
| ORIGINATION | <a id="EventOrigination"></a>origination (`pwm:EventOrigination`) | A client or product was selected. |
| FINAL_DECISION | <a id="EventFinalDecision"></a>final decision (`pwm:EventFinalDecision`) | A person decided. |
| CLIENT_COMMUNICATION | <a id="EventClientCommunication"></a>client communication (`pwm:EventClientCommunication`) | Something was sent to a client. |
| INTERVENTION | <a id="EventIntervention"></a>intervention (`pwm:EventIntervention`) | A person paused, overrode, restricted, revoked or recovered a live run. |
| OUTCOME | <a id="EventOutcome"></a>outcome (`pwm:EventOutcome`) | A complaint, correction or other outcome was linked to the interaction. |
| NEAR_MISS | <a id="EventNearMiss"></a>near miss (`pwm:EventNearMiss`) | A near miss was recorded. |
| CLAIM | <a id="EventClaim"></a>claim (`pwm:EventClaim`) | A claim was recorded. |
| INCIDENT_OPENED | <a id="EventIncidentOpened"></a>incident opened (`pwm:EventIncidentOpened`) | An AI incident was opened. |
| ERASURE | <a id="EventErasure"></a>erasure (`pwm:EventErasure`) | Content was erased; the chain stays verifiable. |
| MODEL_REFUSAL | <a id="EventModelRefusal"></a>model refusal (`pwm:EventModelRefusal`) | The model declined, as distinct from a policy denial. |

<a id="RunStateScheme"></a>
#### run states (`pwm:RunStateScheme`)

Codes for runState.

Used by: [`runState`](#runState)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

| Code | Concept | Definition |
|---|---|---|
| ready | <a id="RunReady"></a>ready (`pwm:RunReady`) | Run created and ready to start governed calls. |
| running | <a id="RunRunning"></a>running (`pwm:RunRunning`) | Run is executing governed calls within its budgets. |
| suspended | <a id="RunSuspended"></a>suspended (`pwm:RunSuspended`) | Run is held at a durable checkpoint awaiting human approval. |
| resumed | <a id="RunResumed"></a>resumed (`pwm:RunResumed`) | Run continuation was rehydrated after a recorded human resolution. |
| completed | <a id="RunCompleted"></a>completed (`pwm:RunCompleted`) | Run finished its authorized work. |
| denied | <a id="RunDenied"></a>denied (`pwm:RunDenied`) | Run stopped because a governing decision denied continuation. |
| failed | <a id="RunFailed"></a>failed (`pwm:RunFailed`) | Run terminated with a recorded execution failure. |

<a id="ApprovalResolutionScheme"></a>
#### approval resolutions (`pwm:ApprovalResolutionScheme`)

Codes for approvalResolution.

Used by: [`approvalResolution`](#approvalResolution)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

| Code | Concept | Definition |
|---|---|---|
| approved | <a id="ApprovalApproved"></a>approved (`pwm:ApprovalApproved`) | A named human approved the exact held action. |
| denied | <a id="ApprovalDenied"></a>denied (`pwm:ApprovalDenied`) | A named human denied the held action; continuation records the denial and does not authorize that action. |

<a id="AttributeSourceScheme"></a>
#### attribute sources (`pwm:AttributeSourceScheme`)

Codes for attributeSource.

Used by: [`attributeSource`](#attributeSource)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

| Code | Concept | Definition |
|---|---|---|
| typed | <a id="AttributeTyped"></a>typed (`pwm:AttributeTyped`) | Caller-supplied typed attribute is authoritative; a classifier cannot replace it. |
| classifier_derived | <a id="AttributeClassifierDerived"></a>classifier-derived (`pwm:AttributeClassifierDerived`) | Classifier fills a gap in typed attributes and can only narrow permission. |

<a id="QuestionCoverageStatusScheme"></a>
#### question coverage statuses (`pwm:QuestionCoverageStatusScheme`)

Codes for coverageStatus.

Used by: [`coverageStatus`](#coverageStatus)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

| Code | Concept | Definition |
|---|---|---|
| answered | <a id="QuestionAnswered"></a>answered (`pwm:QuestionAnswered`) | Answered from recorded evidence; citesSource identifies the evidence. |
| not_recorded | <a id="QuestionNotRecorded"></a>not recorded (`pwm:QuestionNotRecorded`) | Evidence needed to answer was not recorded at this milestone; the question remains explicit. |

<a id="GatewayFailureModeScheme"></a>
#### gateway failure modes (`pwm:GatewayFailureModeScheme`)

Codes for gatewayFailureMode.

Used by: [`gatewayFailureMode`](#gatewayFailureMode)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 9; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

| Code | Concept | Definition |
|---|---|---|
| fail_closed | <a id="FailClosed"></a>fail closed (`pwm:FailClosed`) | Default: refuse calls while the gateway or evidence store is unavailable. |
| degraded | <a id="Degraded"></a>degraded (`pwm:Degraded`) | Only local-model calls without client data may proceed during an outage; evidence is written when the store recovers. |

<a id="ReconstructionQuestionScheme"></a>
#### supervisory reconstruction questions (`pwm:ReconstructionQuestionScheme`)

Codes for reconstructionQuestion.

Used by: [`reconstructionQuestion`](#reconstructionQuestion)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.3 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.3 gateway and replay records: NEW specialised record or property; no exact standard term identified

| Code | Concept | Definition |
|---|---|---|
| QuestionWhyClient | <a id="QuestionWhyClient"></a>Why this client? (`pwm:QuestionWhyClient`) | Supervisory reconstruction question: Why this client? |
| QuestionWhyProduct | <a id="QuestionWhyProduct"></a>Why this insight or product? (`pwm:QuestionWhyProduct`) | Supervisory reconstruction question: Why this insight or product? |
| QuestionInformation | <a id="QuestionInformation"></a>What information did the AI use? (`pwm:QuestionInformation`) | Supervisory reconstruction question: What information did the AI use? |
| QuestionModelAgent | <a id="QuestionModelAgent"></a>What model or agent produced it? (`pwm:QuestionModelAgent`) | Supervisory reconstruction question: What model or agent produced it? |
| QuestionAuthority | <a id="QuestionAuthority"></a>What was the agent authorized to do? (`pwm:QuestionAuthority`) | Supervisory reconstruction question: What was the agent authorized to do? |
| QuestionControls | <a id="QuestionControls"></a>What suitability or policy controls ran? (`pwm:QuestionControls`) | Supervisory reconstruction question: What suitability or policy controls ran? |
| QuestionRMChanges | <a id="QuestionRMChanges"></a>What did the RM see and change? (`pwm:QuestionRMChanges`) | Supervisory reconstruction question: What did the RM see and change? |
| QuestionFinalDecision | <a id="QuestionFinalDecision"></a>Who made the final decision? (`pwm:QuestionFinalDecision`) | Supervisory reconstruction question: Who made the final decision? |
| QuestionCommunication | <a id="QuestionCommunication"></a>What was communicated to the client? (`pwm:QuestionCommunication`) | Supervisory reconstruction question: What was communicated to the client? |
| QuestionReproduction | <a id="QuestionReproduction"></a>Can the bank reproduce that evidence later? (`pwm:QuestionReproduction`) | Supervisory reconstruction question: Can the bank reproduce that evidence later? |

<a id="ModuleLifecycle"></a>

## Module 10: Ingestion, policy and lifecycle controls

Structured ingestion and the missing policy, review and monitoring records for milestone 0.4.

### Classes

<a id="IngestedDocument"></a>
#### ingested document (`pwm:IngestedDocument`)

Immutable source-file ingestion record identifying the parser and its exact version. Reuses contentHash and versionId.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="DocumentElement"></a>
#### document element (`pwm:DocumentElement`)

Parser-derived passage or structural element of an ingested document, with page, section path, character span, classification and content hash. Parent links preserve the parsed tree.

- **Specialises:** [document section](#DocumentSection)
- **Handling:** medium context, default classification C1
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="DeclarativePolicy"></a>
#### declarative policy (`pwm:DeclarativePolicy`)

Versioned declarative artefact serialising existing PolicyRule records; not a replacement rule engine.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C1
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="PolicyChange"></a>
#### policy change (`pwm:PolicyChange`)

Archived proposed, reviewed, strictly validated and applied delta against a policy version, containing added, modified or removed rule identifiers.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C1
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="PolicyTestCase"></a>
#### policy test case (`pwm:PolicyTestCase`)

Named scenario under a policy rule, with given attributes, requested action and expected decision outcome.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C1
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="PolicyPack"></a>
#### policy pack (`pwm:PolicyPack`)

Named versioned bundle of policy artefacts, scenarios and fixture references; a private overlay may shadow a built-in pack.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C1
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="FlightRecorderSession"></a>
#### flight recorder session (`pwm:FlightRecorderSession`)

Evidence export manifest for recorded runs and interactions, with a content hash and retained export locator. Reuses Run and ReconstructionReport rather than duplicating their evidence.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="Replay"></a>
#### replay (`pwm:Replay`)

Verification of a recorder session against retained evidence, producing an existing ReconstructionReport; does not mean re-executing client actions.

- **Specialises:** [Activity](http://www.w3.org/ns/prov#Activity) (PROV-O)
- **Handling:** medium context, default classification C1
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="RMReviewEdit"></a>
#### rmreview edit (`pwm:RMReviewEdit`)

Structured edit captured during HumanReview, preserving hashes of what the RM saw, what was approved and the retained diff.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="ReviewQueue"></a>
#### review queue (`pwm:ReviewQueue`)

Use-case review queue receiving high-risk and sampled low-risk interactions for human review.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="ReviewQueueEntry"></a>
#### review queue entry (`pwm:ReviewQueueEntry`)

Risk-selected interaction awaiting or completing human review, with reason, queue time and delivery boundary.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="ReviewCoverageReport"></a>
#### review coverage report (`pwm:ReviewCoverageReport`)

Evidence-derived counts of eligible, screened, queued and reviewed interactions over a window, with review latency; grouped by use case and its risk tier.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="CircuitBreaker"></a>
#### circuit breaker (`pwm:CircuitBreaker`)

Automatic halt configuration specifying flag threshold, time window and scope; trips create a prohibiting policy and incident, and reset requires accountable-owner approval.

- **Specialises:** [kill switch](#KillSwitch)
- **Handling:** slow context, default classification C1
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="CircuitBreakerTrip"></a>
#### circuit breaker trip (`pwm:CircuitBreakerTrip`)

Recorded threshold crossing with exact scope, threshold, observed flags, window and inserted prohibiting rule.

- **Specialises:** [AI incident](#AIIncident)
- **Handling:** medium context, default classification C1
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="AffectedClientQuery"></a>
#### affected client query (`pwm:AffectedClientQuery`)

Version-scoped and time-bounded query for clients receiving output, with an explicit result even when no clients match.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="AffectedClientResult"></a>
#### affected client result (`pwm:AffectedClientResult`)

Affected-client query result with matching communication and interaction links and result count; zero results is distinct from no result.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

<a id="InteractionOutcome"></a>
#### interaction outcome (`pwm:InteractionOutcome`)

Classified outcome linked to the originating interaction, exact output, model and approver; includes complaints, corrections and operational failures.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C1
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(d): incidents and near misses; 5.23(b); **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="ingestedFrom"></a>`pwm:ingestedFrom` ingested from | any → [ingested document](#IngestedDocument) | Source-file ingestion record for a parser-derived element. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="parentElement"></a>`pwm:parentElement` parent element | any → [document element](#DocumentElement) | Parser-derived parent element in the document tree. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="containsRule"></a>`pwm:containsRule` contains rule | any → [policy rule](#PolicyRule) | Existing rule serialised by a declarative policy. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="changesPolicy"></a>`pwm:changesPolicy` changes policy | any → [declarative policy](#DeclarativePolicy) | Policy version against which a delta is validated. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="testsRule"></a>`pwm:testsRule` tests rule | any → [policy rule](#PolicyRule) | Named rule exercised by a scenario. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="expectedOutcome"></a>`pwm:expectedOutcome` expected outcome | any → [policy decision outcomes](#DecisionOutcomeScheme) | Expected deterministic decision outcome. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="hasPolicyTest"></a>`pwm:hasPolicyTest` has policy test | any → [policy test case](#PolicyTestCase) | Scenario validating a policy artefact. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="packPolicy"></a>`pwm:packPolicy` pack policy | any → [declarative policy](#DeclarativePolicy) | Policy artefact in a versioned pack. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="shadowsPack"></a>`pwm:shadowsPack` shadows pack | any → [policy pack](#PolicyPack) | Built-in pack shadowed by a private overlay with the same name. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="recordedRun"></a>`pwm:recordedRun` recorded run | any → [run](#Run) | Run included in a recorder export manifest. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="replaysSession"></a>`pwm:replaysSession` replays session | any → [flight recorder session](#FlightRecorderSession) | Recorder manifest verified by a replay. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="replayReport"></a>`pwm:replayReport` replay report | any → [reconstruction report](#ReconstructionReport) | Existing reconstruction report produced by a replay. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="hasOutcomeMetric"></a>`pwm:hasOutcomeMetric` has outcome metric | any → [outcome metric](#OutcomeMetric) | Versioned primary metric definition with baseline and target for a use case. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="hasModelCard"></a>`pwm:hasModelCard` has model card | any → [model card](#ModelCard) | Approved model card for a model or pinned model call. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.5 to 4.13: inventory and materiality; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="creditedControl"></a>`pwm:creditedControl` credited control | any → [policy rule](#PolicyRule) | Control credited in reducing inherent to residual materiality. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.5 to 4.13: inventory and materiality; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="reviewEdit"></a>`pwm:reviewEdit` review edit | any → [rmreview edit](#RMReviewEdit) | Structured edit produced in a human review or referenced by a final decision. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="communicationSigner"></a>`pwm:communicationSigner` communication signer | any → [person](https://spec.edmcouncil.org/fibo/ontology/FND/AgentsAndPeople/People/Person) (FIBO, Release) | Named human signing the exact communication content hash. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="inReviewQueue"></a>`pwm:inReviewQueue` in review queue | any → [review queue](#ReviewQueue) | Queue receiving a risk-selected interaction. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="scopeUseCase"></a>`pwm:scopeUseCase` scope use case | any → [AI use case](#AIUseCase) | Use-case dimension of a breaker or affected-client query scope. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="scopeModel"></a>`pwm:scopeModel` scope model | any → [AI model](#AIModel) | Pinned model entity identifying the exact modelVersion in scope. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="scopePolicy"></a>`pwm:scopePolicy` scope policy | any → [policy rule](#PolicyRule) | Versioned policy rule in scope. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="trippedBreaker"></a>`pwm:trippedBreaker` tripped breaker | any → [circuit breaker](#CircuitBreaker) | Breaker whose threshold crossing generated an incident. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="prohibitingRule"></a>`pwm:prohibitingRule` prohibiting rule | any → [policy rule](#PolicyRule) | Inserted rule denying subsequent actions in the tripped scope. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="queryResult"></a>`pwm:queryResult` query result | any → [affected client result](#AffectedClientResult) | Explicit affected-client query result. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="affectedCommunication"></a>`pwm:affectedCommunication` affected communication | any → [client communication](#ClientCommunication) | Communication that delivered affected output to a client. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="outcomeKind"></a>`pwm:outcomeKind` outcome kind | any → [interaction outcome kind scheme](#InteractionOutcomeKindScheme) | Classification of an interaction outcome. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(d): incidents and near misses; 5.23(b); **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="nearMissKind"></a>`pwm:nearMissKind` near miss kind | any → [near miss kind scheme](#NearMissKindScheme) | Classification of a harm-avoiding event. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(d): incidents and near misses; 5.23(b); **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="affectedOutput"></a>`pwm:affectedOutput` affected output | any → [recommendation](#Recommendation) | Exact retained output implicated in an outcome or near miss. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(d): incidents and near misses; 5.23(b); **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="outcomeApprover"></a>`pwm:outcomeApprover` outcome approver | any → [person](https://spec.edmcouncil.org/fibo/ontology/FND/AgentsAndPeople/People/Person) (FIBO, Release) | Human approver associated with the implicated output. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(d): incidents and near misses; 5.23(b); **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="implicatedModel"></a>`pwm:implicatedModel` implicated model | any → [AI model](#AIModel) | Pinned model implicated in an outcome or near miss, without asserting that the outcome is itself a model call. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) NEW specialised link |
| <a id="evaluatedAgainstAppetite"></a>`pwm:evaluatedAgainstAppetite` evaluated against appetite | any → [AI risk appetite](#AIRiskAppetite) | Versioned institutional appetite against which registration and residual materiality were evaluated. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.11: residual materiality; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8: risk appetite; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) NEW specialised evaluation link |
| <a id="parserName"></a>`pwm:parserName` parser name | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Parser implementation used for ingestion. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="parserVersion"></a>`pwm:parserVersion` parser version | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Exact ingestion parser version. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="pageNumber"></a>`pwm:pageNumber` page number | any → [positiveInteger](http://www.w3.org/2001/XMLSchema#positiveInteger) (XSD) | One-based source page number. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="sectionPath"></a>`pwm:sectionPath` section path | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Ordered source heading path; use root for a document without headings. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="characterStart"></a>`pwm:characterStart` character start | any → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Inclusive character offset within the source element text. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="characterEnd"></a>`pwm:characterEnd` character end | any → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Exclusive character offset within the source element text. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="boundingBox"></a>`pwm:boundingBox` bounding box | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Parser bounding box with coordinate system and units. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="policySerialization"></a>`pwm:policySerialization` policy serialization | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Retained declarative rule text, normally YAML. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="deltaText"></a>`pwm:deltaText` delta text | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Added, modified and removed rule delta; identifiers must resolve against the base policy. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="changeStage"></a>`pwm:changeStage` change stage | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Policy delta stage: propose, review, validated or applied. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="givenAttributes"></a>`pwm:givenAttributes` given attributes | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Serialised scenario input attributes. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="whenAction"></a>`pwm:whenAction` when action | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Requested action in the scenario. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="fixtureLocator"></a>`pwm:fixtureLocator` fixture locator | any → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | Location of a pack fixture bundle. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="appetitePermits"></a>`pwm:appetitePermits` appetite permits | any → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Recorded risk-appetite evaluation permits registration and activation, including residual materiality. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.5 to 4.13: inventory and materiality; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="riskAppetiteStatement"></a>`pwm:riskAppetiteStatement` risk appetite statement | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Versioned institutional AI appetite, including allowed categories, impact, authority and quantitative measures. | **REG** [MAS AIRG](#ref-MAS-AIRG) 4.5 to 4.13: inventory and materiality; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="beforeHash"></a>`pwm:beforeHash` before hash | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Hash of the exact recommendation rendered before RM edits. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: version ID or content hash; keep the hash when content is deleted; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="afterHash"></a>`pwm:afterHash` after hash | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Hash of the content approved after RM edits. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: version ID or content hash; keep the hash when content is deleted; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="editDiffHash"></a>`pwm:editDiffHash` edit diff hash | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Content hash of the retained structured edit diff. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 11; **AM** [Filing Cabinet talk](#ref-AM-FILING-CABINET) Slide 8: version ID or content hash; keep the hash when content is deleted; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="requiresTwoSignatures"></a>`pwm:requiresTwoSignatures` requires two signatures | any → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether this communication requires two distinct human signatures. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="lowRiskSamplingRate"></a>`pwm:lowRiskSamplingRate` low risk sampling rate | any → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Low-risk sampling fraction between zero and one, set per use case. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="queueReason"></a>`pwm:queueReason` queue reason | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Risk criterion or low-risk sampling reason for selection. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="queuedAt"></a>`pwm:queuedAt` queued at | any → [dateTime](http://www.w3.org/2001/XMLSchema#dateTime) (XSD) | Time the interaction entered review. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="reviewedAt"></a>`pwm:reviewedAt` reviewed at | any → [dateTime](http://www.w3.org/2001/XMLSchema#dateTime) (XSD) | Time human review completed. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="reviewBeforeDelivery"></a>`pwm:reviewBeforeDelivery` review before delivery | any → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether review must finish before client delivery. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="eligibleCount"></a>`pwm:eligibleCount` eligible count | any → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Number of interactions eligible in the reporting window. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="screenedCount"></a>`pwm:screenedCount` screened count | any → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Number of eligible interactions screened. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="queuedCount"></a>`pwm:queuedCount` queued count | any → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Number of screened interactions selected for review. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="reviewedCount"></a>`pwm:reviewedCount` reviewed count | any → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Number of queued interactions reviewed. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="meanReviewSeconds"></a>`pwm:meanReviewSeconds` mean review seconds | any → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Mean queue-to-review latency in seconds for completed reviews. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9: human oversight; 5.23: monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="scopePromptVersion"></a>`pwm:scopePromptVersion` scope prompt version | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Prompt version in scope. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="flagThreshold"></a>`pwm:flagThreshold` flag threshold | any → [positiveInteger](http://www.w3.org/2001/XMLSchema#positiveInteger) (XSD) | Flag count at which a breaker trips within its window. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="observedFlags"></a>`pwm:observedFlags` observed flags | any → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Flag count observed at the trip. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="windowStart"></a>`pwm:windowStart` window start | any → [dateTime](http://www.w3.org/2001/XMLSchema#dateTime) (XSD) | Inclusive start of a monitoring or query window. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="windowEnd"></a>`pwm:windowEnd` window end | any → [dateTime](http://www.w3.org/2001/XMLSchema#dateTime) (XSD) | Exclusive end of a monitoring or query window. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |
| <a id="resultCount"></a>`pwm:resultCount` result count | any → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Number of distinct affected clients returned by a query. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.23(b): incidents and monitoring; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified |

### Code lists

<a id="InteractionOutcomeKindScheme"></a>
#### interaction outcome kind scheme (`pwm:InteractionOutcomeKindScheme`)

Classifications recorded under roadmap item 15.

Used by: [`outcomeKind`](#outcomeKind)

References: **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(d): incidents and near misses; 5.23(b); **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

| Code | Concept | Definition |
|---|---|---|
|  | <a id="ClientComplaint"></a>client complaint (`pwm:ClientComplaint`) | Client complaint recorded against the originating interaction. |
|  | <a id="Correction"></a>correction (`pwm:Correction`) | Correction recorded against the originating interaction. |
|  | <a id="WrongClientTargeting"></a>wrong client targeting (`pwm:WrongClientTargeting`) | Wrong client targeting recorded against the originating interaction. |
|  | <a id="UnsuitableSuggestion"></a>unsuitable suggestion (`pwm:UnsuitableSuggestion`) | Unsuitable suggestion recorded against the originating interaction. |
|  | <a id="CommunicationError"></a>communication error (`pwm:CommunicationError`) | Communication error recorded against the originating interaction. |
|  | <a id="PolicyBreach"></a>policy breach (`pwm:PolicyBreach`) | Policy breach recorded against the originating interaction. |
|  | <a id="OperationalIncident"></a>operational incident (`pwm:OperationalIncident`) | Operational incident recorded against the originating interaction. |

<a id="NearMissKindScheme"></a>
#### near miss kind scheme (`pwm:NearMissKindScheme`)

Classifications recorded under roadmap item 15.

Used by: [`nearMissKind`](#nearMissKind)

References: **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9(d): incidents and near misses; 5.23(b); **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) 0.4 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.4: NEW specialised record or field; no exact standard term identified

| Code | Concept | Definition |
|---|---|---|
|  | <a id="BlockedAction"></a>blocked action (`pwm:BlockedAction`) | Blocked action recorded against the originating interaction. |
|  | <a id="VerifierBlock"></a>verifier block (`pwm:VerifierBlock`) | Verifier block recorded against the originating interaction. |
|  | <a id="RMRejection"></a>rmrejection (`pwm:RMRejection`) | Rmrejection recorded against the originating interaction. |
|  | <a id="UnsafeDraft"></a>unsafe draft (`pwm:UnsafeDraft`) | Unsafe draft recorded against the originating interaction. |
|  | <a id="UnusualAgentBehaviour"></a>unusual agent behaviour (`pwm:UnusualAgentBehaviour`) | Unusual agent behaviour recorded against the originating interaction. |

<a id="ModuleAgents"></a>

## Module 11: Agents, authority and fleets

Workload identities, scoped human authority, autonomy evidence, containment and fleet controls.

### Classes

<a id="WorkloadIdentity"></a>
#### workload identity (`pwm:WorkloadIdentity`)

Registry of an individual agent workload credential reference, issuer, subject and environment. Stores no private key or token. Gateways verify possession on every call.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C2
- **Properties:** [`credentialReference`](#credentialReference), [`credentialIssuer`](#credentialIssuer), [`workloadSubject`](#workloadSubject)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 1.11 agent risks; fn 25 agent identifiers and components; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="AuthorityScope"></a>
#### authority scope (`pwm:AuthorityScope`)

Explicit finite sets of clients, purposes, verbs, tools, fields, systems, products and jurisdictions. Omission grants nothing for dimensions checked by the written narrowing rules. Numeric caps and autonomy may only narrow.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C2
- **Properties:** [`scopeTool`](#scopeTool), [`scopeField`](#scopeField), [`scopeSystem`](#scopeSystem), [`scopeProduct`](#scopeProduct), [`accessEvent`](#accessEvent), [`scopeLimit`](#scopeLimit), [`scopeAutonomyLevel`](#scopeAutonomyLevel)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="HumanEntitlement"></a>
#### human entitlement (`pwm:HumanEntitlement`)

Time-bounded entitlement held by a named person, with explicit subject scope. An agent delegation cannot enlarge this scope.

- **Specialises:** [authorization](https://www.omg.org/spec/Commons/BusinessAuthorizations/Authorization) (OMG Commons)
- **Handling:** slow context, default classification C2
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="AuthorityRevocation"></a>
#### authority revocation (`pwm:AuthorityRevocation`)

Record that authority or a credential was revoked at a stated time, including task completion. Revocation is effective on the next decision.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`revokedAuthority`](#revokedAuthority), [`revokedAt`](#revokedAt)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="AutonomyTransition"></a>
#### autonomy transition (`pwm:AutonomyTransition`)

Evidenced promotion or automatic one-level step-down for an agent, with before and after levels, evaluation window, metric sample, approval and notification records. Restoration requires promotion again.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`transitionAgent`](#transitionAgent), [`fromAutonomyLevel`](#fromAutonomyLevel), [`toAutonomyLevel`](#toAutonomyLevel), [`promotionGate`](#promotionGate), [`sampleCount`](#sampleCount), [`observedQuality`](#observedQuality), [`observedExceptionRate`](#observedExceptionRate), [`fullyReconstructable`](#fullyReconstructable), [`transitionApproved`](#transitionApproved), [`ownerApproval`](#ownerApproval), [`independentValidation`](#independentValidation), [`qualityDrift`](#qualityDrift)
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 5.9 human oversight; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **AM** [Governing agentic AI (essay)](#ref-AM-GOVERNING-AGENTIC-AI) Section 04: autonomy is a licence, not a launch setting; **AM** [EVOLVE](#ref-EVOLVE) Stage 4: quality metrics before more autonomy; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="BlastRadiusLimit"></a>
#### blast radius limit (`pwm:BlastRadiusLimit`)

Agent-wide damage limits independent of RunBudget. Each limit names a dimension, unit, aggregation basis, time window and maximum. Decisions carry projected consumption including the proposed action.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C2
- **Properties:** [`limitDimension`](#limitDimension), [`limitWindowBasis`](#limitWindowBasis), [`limitUnit`](#limitUnit), [`limitMaximum`](#limitMaximum)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="BlastRadiusObservation"></a>
#### blast radius observation (`pwm:BlastRadiusObservation`)

Recorded projected cumulative consumption against one agent limit, including the requested action. Window and dimension are inherited from that exact limit.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`observedLimit`](#observedLimit), [`projectedConsumption`](#projectedConsumption)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="ExternalContent"></a>
#### external content (`pwm:ExternalContent`)

Content from outside the institution, including agent messages, PDFs and API responses. Its trust label never conveys instruction authority.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **References:** **REG** [MAS AIRG](#ref-MAS-AIRG) 1.11: agent risks; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 6; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="AuthenticatedInstruction"></a>
#### authenticated instruction (`pwm:AuthenticatedInstruction`)

Instruction with authenticated human, assurance level, source and transaction context. A signature reference is evidence to verify, never a secret or a substitute for step-up authorization.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`instructionSource`](#instructionSource), [`instructionPrincipal`](#instructionPrincipal), [`assuranceLevel`](#assuranceLevel), [`signatureReference`](#signatureReference), [`transactionContext`](#transactionContext)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="BehaviourBaseline"></a>
#### behaviour baseline (`pwm:BehaviourBaseline`)

Versioned per-agent expected behaviour and containment threshold for a named metric over a window.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C2
- **Properties:** [`baselineAgent`](#baselineAgent), [`baselineMetric`](#baselineMetric), [`baselineExpected`](#baselineExpected), [`anomalyThreshold`](#anomalyThreshold)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="BehaviourAnomaly"></a>
#### behaviour anomaly (`pwm:BehaviourAnomaly`)

Observed deviation from an agent baseline, with automatic containment and incident links. Human release follows containment.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`anomalyBaseline`](#anomalyBaseline), [`anomalyValue`](#anomalyValue), [`containmentApplied`](#containmentApplied), [`anomalyIncident`](#anomalyIncident)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="CapabilityGrant"></a>
#### capability grant (`pwm:CapabilityGrant`)

Explicit expiring and revocable capability assigned to an agent within a subject scope. Issue, use, expiry and revocation are evidenced.

- **Specialises:** [authorization](https://www.omg.org/spec/Commons/BusinessAuthorizations/Authorization) (OMG Commons)
- **Handling:** slow context, default classification C2
- **Properties:** [`grantee`](#grantee), [`grantedCapability`](#grantedCapability)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 10; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="AgentFleet"></a>
#### agent fleet (`pwm:AgentFleet`)

Registered group of agents under common quotas, isolation and halt controls. External members are untrusted principals.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C2
- **Properties:** [`fleetMember`](#fleetMember), [`fleetIsolation`](#fleetIsolation), [`fleetQuota`](#fleetQuota)
- **References:** **REG** [IMDA agentic AI framework](#ref-IMDA-MGF-AGENTIC) Agentic AI governance (instrument-level); multi-agent controls; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 11; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="FleetHalt"></a>
#### fleet halt (`pwm:FleetHalt`)

Recorded fleet-wide stop instruction with explicit fleet scope and time. Runtime dispatch and cancellation are performed by the gateway.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`haltScope`](#haltScope)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 11; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="SandboxConstraints"></a>
#### sandbox constraints (`pwm:SandboxConstraints`)

Decision output specifying the permitted credential references, network destinations, filesystem scope, APIs, commands, export, persistence, auto-stop interval and resource caps. Kognita emits constraints, not a sandbox.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C2
- **Properties:** [`networkDestination`](#networkDestination), [`filesystemScope`](#filesystemScope), [`permittedCommand`](#permittedCommand), [`permittedAPI`](#permittedAPI), [`sandboxCredential`](#sandboxCredential), [`permitsExport`](#permitsExport), [`permitsPersistence`](#permitsPersistence), [`autoStopSeconds`](#autoStopSeconds), [`cpuSeconds`](#cpuSeconds), [`memoryBytes`](#memoryBytes), [`diskBytes`](#diskBytes)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="CompensatingAction"></a>
#### compensating action (`pwm:CompensatingAction`)

Registered recovery action for a reversible tool; recovery records reference the original call and an explicit reverse execution order.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C2
- **Properties:** [`compensatesCall`](#compensatesCall), [`recoveryOrder`](#recoveryOrder)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="AdversarialTestCase"></a>
#### adversarial test case (`pwm:AdversarialTestCase`)

Attack scenario specialising the existing policy test case, linked to a versioned threat model and the control tested. Passing requires denial, escalation or containment.

- **Specialises:** [policy test case](#PolicyTestCase)
- **Handling:** slow context, default classification C2
- **Properties:** [`threatModel`](#threatModel), [`attackKind`](#attackKind)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="ThreatModel"></a>
#### threat model (`pwm:ThreatModel`)

Versioned threat model covering own-agent failures and hostile external agents, mapping attack scenarios to controls.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** slow context, default classification C2
- **Properties:** [`threatModelReference`](#threatModelReference)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="EnforcementAttestation"></a>
#### enforcement attestation (`pwm:EnforcementAttestation`)

Attributed time-stamped evidence about gateway mediation, credential custody, egress routing and evidence-store isolation for a use case. The attestation is a claim to verify operationally.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`attestedUseCase`](#attestedUseCase), [`gatewayOnlyCredentials`](#gatewayOnlyCredentials), [`egressRestricted`](#egressRestricted), [`evidenceStoreIsolated`](#evidenceStoreIsolated)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 15; **AM** [Governing agentic AI (essay)](#ref-AM-GOVERNING-AGENTIC-AI) Section 03: boundaries enforced in permissions; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

<a id="ActionStop"></a>
#### action stop (`pwm:ActionStop`)

Recorded stop or cancellation request for an in-flight tool call with outcome cancelled, completed, timed out or unsupported; unsupported cancellation requires compensation or decision-separated steps.

- **Specialises:** [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O)
- **Handling:** medium context, default classification C2
- **Properties:** [`stoppedCall`](#stoppedCall), [`stopOutcome`](#stopOutcome), [`stopReason`](#stopReason)
- **References:** **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 16; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

### Properties

| Property | Domain → range | Definition | References |
|---|---|---|---|
| <a id="workloadIdentity"></a>`pwm:workloadIdentity` workload identity | any → [workload identity](#WorkloadIdentity) | Workload identity assigned exclusively to this agent. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="agentOwner"></a>`pwm:agentOwner` agent owner | [AI agent](#AIAgent) → [person](https://spec.edmcouncil.org/fibo/ontology/FND/AgentsAndPeople/People/Person) (FIBO, Release) | Named human accountable for this agent deployment. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="operatesFor"></a>`pwm:operatesFor` operates for | [AI agent](#AIAgent) → [person](https://spec.edmcouncil.org/fibo/ontology/FND/AgentsAndPeople/People/Person) (FIBO, Release) | Human principal on whose behalf an agent operates. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="agentStatus"></a>`pwm:agentStatus` agent status | [AI agent](#AIAgent) → [use-case lifecycle statuses](#UseCaseStatusScheme) | Agent lifecycle state, reusing the use-case status codes without inferring an agent is a use case. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="authorityScope"></a>`pwm:authorityScope` authority scope | any → [authority scope](#AuthorityScope) | Explicit scope of an entitlement, delegation or capability grant. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="holdsEntitlement"></a>`pwm:holdsEntitlement` holds entitlement | any → [human entitlement](#HumanEntitlement) | Entitlement held by a human grantor. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="scopeTool"></a>`pwm:scopeTool` scope tool | [authority scope](#AuthorityScope) → [tool](#Tool) | Tool permitted within an authority scope. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.16(c): govern plugins and APIs; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="scopeProduct"></a>`pwm:scopeProduct` scope product | [authority scope](#AuthorityScope) → [product](#Product) | Product permitted within a scope. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="revokedAuthority"></a>`pwm:revokedAuthority` revoked authority | [authority revocation](#AuthorityRevocation) → [thing](http://www.w3.org/2002/07/owl#Thing) | Delegation, capability, entitlement or credential invalidated by this record. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="parentDelegation"></a>`pwm:parentDelegation` parent delegation | [delegated authority](#DelegatedAuthority) → [delegated authority](#DelegatedAuthority) | Immediate parent authorization; the root human remains delegatedBy and parent delegatedTo identifies the sub-delegating agent. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 9; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="transitionAgent"></a>`pwm:transitionAgent` transition agent | [autonomy transition](#AutonomyTransition) → [AI agent](#AIAgent) | Agent whose effective autonomy changes. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="fromAutonomyLevel"></a>`pwm:fromAutonomyLevel` from autonomy level | [autonomy transition](#AutonomyTransition) → [agent autonomy levels](#AutonomyLevelScheme) | Effective autonomy before a transition. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="toAutonomyLevel"></a>`pwm:toAutonomyLevel` to autonomy level | [autonomy transition](#AutonomyTransition) → [agent autonomy levels](#AutonomyLevelScheme) | Effective autonomy after a transition. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="effectiveAutonomyLevel"></a>`pwm:effectiveAutonomyLevel` effective autonomy level | [AI agent](#AIAgent) → [agent autonomy levels](#AutonomyLevelScheme) | Current evidenced level, no higher than the registered maximum. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="promotionGate"></a>`pwm:promotionGate` promotion gate | [autonomy transition](#AutonomyTransition) → [autonomy promotion gate](#AutonomyPromotionGate) | Evidence gate evaluated for this transition. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="ownerApproval"></a>`pwm:ownerApproval` owner approval | [autonomy transition](#AutonomyTransition) → [person](https://spec.edmcouncil.org/fibo/ontology/FND/AgentsAndPeople/People/Person) (FIBO, Release) | Human owner approving this transition. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="independentValidation"></a>`pwm:independentValidation` independent validation | [autonomy transition](#AutonomyTransition) → [independent validator](#IndependentValidator) | Independent validation role approving this transition. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="notifiedRole"></a>`pwm:notifiedRole` notified role | any → [accountability role](#AccountabilityRole) | Accountable role alerted about a step-down, drift or containment. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="hasBlastRadiusLimit"></a>`pwm:hasBlastRadiusLimit` has blast radius limit | [AI agent](#AIAgent) → [blast radius limit](#BlastRadiusLimit) | Agent-wide cap, separate from a run budget. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="limitObservation"></a>`pwm:limitObservation` limit observation | [policy decision](#PolicyDecision) → [blast radius observation](#BlastRadiusObservation) | Projected consumption evaluated by a policy decision. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="observedLimit"></a>`pwm:observedLimit` observed limit | [blast radius observation](#BlastRadiusObservation) → [blast radius limit](#BlastRadiusLimit) | Exact limit evaluated. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="trustLabel"></a>`pwm:trustLabel` trust label | any → [Content trust and taint labels](#ContentTrustScheme) | Trust or taint label on content, distinct from confidentiality classification. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 6; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="inputContent"></a>`pwm:inputContent` input content | any → [Entity](http://www.w3.org/ns/prov#Entity) (PROV-O) | Content consumed by a call or run, including agent-to-agent messages. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 6; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="instructionPrincipal"></a>`pwm:instructionPrincipal` instruction principal | [authenticated instruction](#AuthenticatedInstruction) → [person](https://spec.edmcouncil.org/fibo/ontology/FND/AgentsAndPeople/People/Person) (FIBO, Release) | Authenticated human behind the instruction. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="instructionEvidence"></a>`pwm:instructionEvidence` instruction evidence | [policy decision](#PolicyDecision) → [authenticated instruction](#AuthenticatedInstruction) | Authenticated instruction relied on by a decision. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="baselineAgent"></a>`pwm:baselineAgent` baseline agent | [behaviour baseline](#BehaviourBaseline) → [AI agent](#AIAgent) | Agent to which a behavioural baseline applies. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="anomalyBaseline"></a>`pwm:anomalyBaseline` anomaly baseline | [behaviour anomaly](#BehaviourAnomaly) → [behaviour baseline](#BehaviourBaseline) | Baseline against which deviation was detected. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="anomalyIncident"></a>`pwm:anomalyIncident` anomaly incident | [behaviour anomaly](#BehaviourAnomaly) → [AI incident](#AIIncident) | Incident opened by automatic anomaly containment. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="grantee"></a>`pwm:grantee` grantee | [capability grant](#CapabilityGrant) → [AI agent](#AIAgent) | Agent receiving a capability grant. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 10; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="grantedCapability"></a>`pwm:grantedCapability` granted capability | [capability grant](#CapabilityGrant) → [tool](#Tool) | Tool granted within the subject scope. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 10; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="reliesOnGrant"></a>`pwm:reliesOnGrant` relies on grant | any → [capability grant](#CapabilityGrant) | Capability grant a call or decision relied on. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 10; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="fleetMember"></a>`pwm:fleetMember` fleet member | [agent fleet](#AgentFleet) → [AI agent](#AIAgent) | Agent registered in a fleet. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 11; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="haltScope"></a>`pwm:haltScope` halt scope | [fleet halt](#FleetHalt) → [agent fleet](#AgentFleet) | Explicit fleet stopped by this halt. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 11; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="sandboxConstraints"></a>`pwm:sandboxConstraints` sandbox constraints | [policy decision](#PolicyDecision) → [sandbox constraints](#SandboxConstraints) | Sandbox constraints emitted by an ALLOW decision. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="reversibilityClass"></a>`pwm:reversibilityClass` reversibility class | [tool](#Tool) → [Tool reversibility classes](#ReversibilityScheme) | Whether tool effects are reversible, compensatable or irreversible. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="compensatingAction"></a>`pwm:compensatingAction` compensating action | [tool](#Tool) → [compensating action](#CompensatingAction) | Registered action restoring or compensating tool effects. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="compensatesCall"></a>`pwm:compensatesCall` compensates call | [compensating action](#CompensatingAction) → [tool call](#ToolCall) | Original call whose effects are compensated. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="humanApproval"></a>`pwm:humanApproval` human approval | [policy decision](#PolicyDecision) → [approval resolution](#ApprovalResolution) | Exact human approval resolution relied on by an action decision. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="threatModel"></a>`pwm:threatModel` threat model | [adversarial test case](#AdversarialTestCase) → [threat model](#ThreatModel) | Versioned threat model grounding an adversarial scenario. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="enforcementAttestation"></a>`pwm:enforcementAttestation` enforcement attestation | [AI use case](#AIUseCase) → [enforcement attestation](#EnforcementAttestation) | Evidence of non-bypassable enforcement for a use case. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="attestedUseCase"></a>`pwm:attestedUseCase` attested use case | [enforcement attestation](#EnforcementAttestation) → [AI use case](#AIUseCase) | Use case covered by the enforcement attestation. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="stoppedCall"></a>`pwm:stoppedCall` stopped call | [action stop](#ActionStop) → [tool call](#ToolCall) | In-flight call addressed by a stop request. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 16; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="deploymentModel"></a>`pwm:deploymentModel` deployment model | [AI agent](#AIAgent) → [AI model](#AIModel) | Model approved as part of this agent deployment, distinct from the provider-reported model of a call. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1: model used at deployment; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW deployment field |
| <a id="scopeLimit"></a>`pwm:scopeLimit` scope limit | [authority scope](#AuthorityScope) → [blast radius limit](#BlastRadiusLimit) | Quantitative cap attached to authority scope; child scopes must retain every parent cap with the same dimension, basis and unit and an equal or lower maximum. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 9: budget and scope only narrow; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW scope cap binding |
| <a id="scopeAutonomyLevel"></a>`pwm:scopeAutonomyLevel` scope autonomy level | [authority scope](#AuthorityScope) → [agent autonomy levels](#AutonomyLevelScheme) | Highest autonomy delegated within an authority scope. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 9: autonomy only narrows; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW scope autonomy binding |
| <a id="credentialReference"></a>`pwm:credentialReference` credential reference | [workload identity](#WorkloadIdentity) → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | Non-secret opaque credential handle held by the gateway. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="credentialIssuer"></a>`pwm:credentialIssuer` credential issuer | [workload identity](#WorkloadIdentity) → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | Issuer of the workload identity. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="workloadSubject"></a>`pwm:workloadSubject` workload subject | [workload identity](#WorkloadIdentity) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Authenticated subject bound to exactly one workload. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="deploymentEnvironment"></a>`pwm:deploymentEnvironment` deployment environment | any → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Environment in which a workload or agent is deployed. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="deploymentDate"></a>`pwm:deploymentDate` deployment date | [AI agent](#AIAgent) → [dateTime](http://www.w3.org/2001/XMLSchema#dateTime) (XSD) | Time of the owner-approved deployment. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="approvedDeploymentVersion"></a>`pwm:approvedDeploymentVersion` approved deployment version | [AI agent](#AIAgent) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Exact deployment hash approved by the owner. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 1; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="scopeField"></a>`pwm:scopeField` scope field | [authority scope](#AuthorityScope) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Field name permitted within a scope; whole-record access is never implicit. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="scopeSystem"></a>`pwm:scopeSystem` scope system | [authority scope](#AuthorityScope) → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | System identifier permitted within a scope. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="revokedAt"></a>`pwm:revokedAt` revoked at | [authority revocation](#AuthorityRevocation) → [dateTime](http://www.w3.org/2001/XMLSchema#dateTime) (XSD) | Effective revocation time, including completion of the authorized task. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 2; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="delegationDepth"></a>`pwm:delegationDepth` delegation depth | [delegated authority](#DelegatedAuthority) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Depth from root human authorization, zero at the root. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 9; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="maxDelegationDepth"></a>`pwm:maxDelegationDepth` max delegation depth | [delegated authority](#DelegatedAuthority) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Maximum total depth allowed by the root, preserved or reduced by children. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 9; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="minimumSample"></a>`pwm:minimumSample` minimum sample | [autonomy promotion gate](#AutonomyPromotionGate) → [positiveInteger](http://www.w3.org/2001/XMLSchema#positiveInteger) (XSD) | Minimum observations for a promotion gate. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="minimumWindowSeconds"></a>`pwm:minimumWindowSeconds` minimum window seconds | [autonomy promotion gate](#AutonomyPromotionGate) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Minimum evidence window in seconds. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="sampleCount"></a>`pwm:sampleCount` sample count | [autonomy transition](#AutonomyTransition) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Number of evaluated metric observations. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="observedQuality"></a>`pwm:observedQuality` observed quality | [autonomy transition](#AutonomyTransition) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Fraction meeting the quality standard in the evaluation window. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="observedExceptionRate"></a>`pwm:observedExceptionRate` observed exception rate | [autonomy transition](#AutonomyTransition) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Exception fraction in the evaluation window. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="fullyReconstructable"></a>`pwm:fullyReconstructable` fully reconstructable | [autonomy transition](#AutonomyTransition) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether every action in the evidence window is reconstructable. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="stepDownThreshold"></a>`pwm:stepDownThreshold` step down threshold | [autonomy promotion gate](#AutonomyPromotionGate) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Lower quality threshold triggering automatic step-down. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="transitionApproved"></a>`pwm:transitionApproved` transition approved | [autonomy transition](#AutonomyTransition) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether the proposed autonomy change was approved. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="qualityDrift"></a>`pwm:qualityDrift` quality drift | [autonomy transition](#AutonomyTransition) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether sustained declining quality contributed to this transition. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 3; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="limitDimension"></a>`pwm:limitDimension` limit dimension | [blast radius limit](#BlastRadiusLimit) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Measured dimension: records, clients, value, actions, transactions or clients-contacted. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="limitWindowBasis"></a>`pwm:limitWindowBasis` limit basis | [blast radius limit](#BlastRadiusLimit) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Aggregation basis: action, run, minute, hour or UTC-day. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="limitUnit"></a>`pwm:limitUnit` limit unit | [blast radius limit](#BlastRadiusLimit) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Unit for the dimension, including currency for monetary limits. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="limitMaximum"></a>`pwm:limitMaximum` limit maximum | [blast radius limit](#BlastRadiusLimit) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Maximum permitted consumption for the stated dimension and window. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="projectedConsumption"></a>`pwm:projectedConsumption` projected consumption | [blast radius observation](#BlastRadiusObservation) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Projected consumption including the proposed action in the limit window. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 4; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="permittedField"></a>`pwm:permittedField` permitted field | [AI agent](#AIAgent) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Field name this agent may receive for its purpose. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="requestedField"></a>`pwm:requestedField` requested field | [tool call](#ToolCall) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Field requested by a tool call. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="accessEvent"></a>`pwm:accessEvent` access event | [authority scope](#AuthorityScope) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Event boundary constraining a delegation, for example a meeting within 24 hours. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 5; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="publicContentTool"></a>`pwm:publicContentTool` public content tool | [tool](#Tool) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether a tool publishes or prepares public content that must never receive client data. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 6; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="instructionSource"></a>`pwm:instructionSource` instruction source | [authenticated instruction](#AuthenticatedInstruction) → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | Source URI of an authenticated instruction. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="assuranceLevel"></a>`pwm:assuranceLevel` assurance level | [authenticated instruction](#AuthenticatedInstruction) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Authentication assurance: channel-only, authenticated or step-up. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="signatureReference"></a>`pwm:signatureReference` signature reference | [authenticated instruction](#AuthenticatedInstruction) → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | Opaque reference to a verified signed instruction source; signature checking is external. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="transactionContext"></a>`pwm:transactionContext` transaction context | [authenticated instruction](#AuthenticatedInstruction) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Hash or reference identifying the exact transaction authenticated. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="requiresStepUp"></a>`pwm:requiresStepUp` requires step up | [policy decision](#PolicyDecision) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether policy requires step-up and independent authorization for this decision. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 7; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="baselineMetric"></a>`pwm:baselineMetric` baseline metric | [behaviour baseline](#BehaviourBaseline) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Named metric such as distinct clients per day or denied calls per minute. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="baselineExpected"></a>`pwm:baselineExpected` baseline expected | [behaviour baseline](#BehaviourBaseline) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Expected metric value. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="anomalyThreshold"></a>`pwm:anomalyThreshold` anomaly threshold | [behaviour baseline](#BehaviourBaseline) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Containment threshold for the baseline metric. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="anomalyValue"></a>`pwm:anomalyValue` anomaly value | [behaviour anomaly](#BehaviourAnomaly) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Observed value crossing the baseline threshold. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="containmentApplied"></a>`pwm:containmentApplied` containment applied | [behaviour anomaly](#BehaviourAnomaly) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether quarantine, credential revocation and tool stop were applied automatically before human release. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 8; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="fleetIsolation"></a>`pwm:fleetIsolation` fleet isolation | [agent fleet](#AgentFleet) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Declared isolation boundary for fleet runs. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 11; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="fleetQuota"></a>`pwm:fleetQuota` fleet quota | [agent fleet](#AgentFleet) → [positiveInteger](http://www.w3.org/2001/XMLSchema#positiveInteger) (XSD) | Maximum concurrent fleet runs. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 11; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="networkDestination"></a>`pwm:networkDestination` network destination | [sandbox constraints](#SandboxConstraints) → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | Allowed network destination; an empty set means block all. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="filesystemScope"></a>`pwm:filesystemScope` filesystem scope | [sandbox constraints](#SandboxConstraints) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Permitted filesystem paths and access modes. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="permittedCommand"></a>`pwm:permittedCommand` permitted command | [sandbox constraints](#SandboxConstraints) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Command allowed by the sandbox. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="permittedAPI"></a>`pwm:permittedAPI` permitted a p i | [sandbox constraints](#SandboxConstraints) → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | API allowed by the sandbox. | **REG** [MAS AIRG](#ref-MAS-AIRG) 5.16(c): govern plugins and APIs; **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="sandboxCredential"></a>`pwm:sandboxCredential` sandbox credential | [sandbox constraints](#SandboxConstraints) → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | Opaque gateway credential handle available within the sandbox. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="permitsExport"></a>`pwm:permitsExport` permits export | [sandbox constraints](#SandboxConstraints) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether sandbox data export is permitted. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="permitsPersistence"></a>`pwm:permitsPersistence` permits persistence | [sandbox constraints](#SandboxConstraints) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether sandbox state may persist after execution. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="autoStopSeconds"></a>`pwm:autoStopSeconds` auto stop seconds | [sandbox constraints](#SandboxConstraints) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Maximum sandbox lifetime in seconds. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="cpuSeconds"></a>`pwm:cpuSeconds` cpu seconds | [sandbox constraints](#SandboxConstraints) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | CPU time ceiling in seconds. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="memoryBytes"></a>`pwm:memoryBytes` memory bytes | [sandbox constraints](#SandboxConstraints) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Memory ceiling in bytes. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="diskBytes"></a>`pwm:diskBytes` disk bytes | [sandbox constraints](#SandboxConstraints) → [nonNegativeInteger](http://www.w3.org/2001/XMLSchema#nonNegativeInteger) (XSD) | Disk ceiling in bytes. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 12; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="recoveryOrder"></a>`pwm:recoveryOrder` recovery order | [compensating action](#CompensatingAction) → [positiveInteger](http://www.w3.org/2001/XMLSchema#positiveInteger) (XSD) | Execution order for compensation, in reverse original-call order. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="threatModelReference"></a>`pwm:threatModelReference` threat model reference | [threat model](#ThreatModel) → [anyURI](http://www.w3.org/2001/XMLSchema#anyURI) (XSD) | Resolvable reference to the published threat model. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="attackKind"></a>`pwm:attackKind` attack kind | [adversarial test case](#AdversarialTestCase) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Attack scenario, such as injection, poisoned retrieval, theft or bypass. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 14; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="deploymentMode"></a>`pwm:deploymentMode` deployment mode | [AI use case](#AIUseCase) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Deployment mode: gateway or library. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="gatewayOnlyCredentials"></a>`pwm:gatewayOnlyCredentials` gateway only credentials | [enforcement attestation](#EnforcementAttestation) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Attested provider and tool credentials are held only by gateways. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="egressRestricted"></a>`pwm:egressRestricted` egress restricted | [enforcement attestation](#EnforcementAttestation) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Attested routing prevents direct access to providers or tools. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="evidenceStoreIsolated"></a>`pwm:evidenceStoreIsolated` evidence store isolated | [enforcement attestation](#EnforcementAttestation) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Attested agent has no write access to the evidence store. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 15; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="maximumDurationSeconds"></a>`pwm:maximumDurationSeconds` maximum duration seconds | [tool](#Tool) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Maximum duration of each tool call in seconds. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 16; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="supportsCancellation"></a>`pwm:supportsCancellation` supports cancellation | [tool](#Tool) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether the proxy can cancel this tool in flight. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 16; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="decisionSeparatedSteps"></a>`pwm:decisionSeparatedSteps` decision separated steps | [tool](#Tool) → [boolean](http://www.w3.org/2001/XMLSchema#boolean) (XSD) | Whether an uncancellable action is split into steps with a decision between them. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 16; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="stopOutcome"></a>`pwm:stopOutcome` stop outcome | [action stop](#ActionStop) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Cancellation result: cancelled, completed, timed-out or unsupported. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 16; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="stopReason"></a>`pwm:stopReason` stop reason | [action stop](#ActionStop) → [string](http://www.w3.org/2001/XMLSchema#string) (XSD) | Trigger: pause, revocation, kill switch or duration timeout. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 16; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified |
| <a id="callDurationSeconds"></a>`pwm:callDurationSeconds` call duration seconds | [tool call](#ToolCall) → [decimal](http://www.w3.org/2001/XMLSchema#decimal) (XSD) | Observed tool-call duration in seconds; exceeding the registered maximum must be recorded as a timeout. | **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 16: bounded tool calls; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW call timing evidence |

### Code lists

<a id="ContentTrustScheme"></a>
#### Content trust and taint labels (`pwm:ContentTrustScheme`)

Content trust and taint labels.

Used by: [`trustLabel`](#trustLabel)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 6; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

| Code | Concept | Definition |
|---|---|---|
|  | <a id="TrustInternal"></a>internal content (`pwm:TrustInternal`) | Approved internal content, subject to ordinary entitlements. |
|  | <a id="TrustExternal"></a>external content (`pwm:TrustExternal`) | Potentially hostile external content; never authority to act. |
|  | <a id="TrustClientData"></a>client data (`pwm:TrustClientData`) | Client data taint; cannot reach public-content tools. |

<a id="ReversibilityScheme"></a>
#### Tool reversibility classes (`pwm:ReversibilityScheme`)

Tool reversibility classes.

Used by: [`reversibilityClass`](#reversibilityClass)

References: **KOG** [Kognita roadmap](#ref-KOG-ROADMAP) roadmap milestone 0.5 item 13; **NEW** [PWM practitioner definition](#ref-PWM-PRACTICE) 0.5: NEW specialised record or property; no exact standard term identified

| Code | Concept | Definition |
|---|---|---|
|  | <a id="Reversible"></a>reversible (`pwm:Reversible`) | Effects can be undone by a registered compensating action. |
|  | <a id="Compensatable"></a>compensatable (`pwm:Compensatable`) | Effects can be mitigated by a registered compensation, without claiming exact reversal. |
|  | <a id="Irreversible"></a>irreversible (`pwm:Irreversible`) | No registered reversal; stronger pre-execution controls required. |

<a id="rules"></a>

## Data rules (SHACL)

`pwm-shapes.ttl` holds the rules below. Validate data together with `pwm.ttl` and RDFS inference:

```sh
pyshacl -s pwm-shapes.ttl -e pwm.ttl -i rdfs your-data.ttl
```

A firm can switch a rule off in its own shapes graph with `sh:deactivated true`, and add its own thresholds there.

<a id="AccountabilityRoleShape"></a>
### Accountability role

Applies to [accountability role](#AccountabilityRole). Sources: [Kognita roadmap](#ref-KOG-ROADMAP), [MAS AIRG](#ref-MAS-AIRG).

- An accountability role must be held by a named person (Kognita 0.4 item 8; MAS AIRG 4.7 key roles).
- An AI agent can never be named in an accountability role (Kognita 0.4 item 8).

<a id="AIUseCaseShape"></a>
### AI use case

Applies to [AI use case](#AIUseCase). Sources: [Kognita roadmap](#ref-KOG-ROADMAP), [MAS AIRG](#ref-MAS-AIRG).

- A use case records its purpose (MAS AIRG 4.7).
- A use case records its approved jurisdiction as an ISO 3166-1 alpha-2 code (MAS AIRG 4.7 approved scope of use).
- A use case has one lifecycle status: pilot, live, suspended or retired (MAS AIRG 4.7; Kognita 0.4 item 8).
- A use case has one risk tier, 1 to 4 (Kognita 0.4 item 8).
- A use case states whether the control plane mediates it. Embedded vendor AI is registered too, marked unmediated (Kognita 0.4 item 8; MAS AIRG 4.2).
- A use case carries a review date set by its tier; past it, the use case is refused (Kognita 0.4 item 8).
- A use case with no business owner is refused (Kognita 0.4 item 8; MAS AIRG 4.7).
- A use case at tier 2 or above with no named Knowledge Lead is refused (Kognita 0.4 item 8).

<a id="LiveUseCaseShape"></a>
### Live use case

Applies to [AI use case](#AIUseCase). Sources: [Kognita roadmap](#ref-KOG-ROADMAP), [MAS AIRG](#ref-MAS-AIRG).

- A live use case needs a materiality assessment with inherent and residual ratings, approved by the designated control function (MAS AIRG 4.11, 4.13; Kognita 0.4 item 8).

<a id="ControlFunctionSeparationShape"></a>
### Control function separation

Applies to [AI use case](#AIUseCase). Sources: [Kognita roadmap](#ref-KOG-ROADMAP), [MAS AIRG](#ref-MAS-AIRG).

- The person who approves a use case's materiality rating for the control function cannot also be its business or technology owner (Kognita 0.4 item 8; MAS AIRG 4.13 independent oversight).

<a id="RiskMaterialityAssessmentShape"></a>
### Risk materiality assessment

Applies to [risk materiality assessment](#RiskMaterialityAssessment). Sources: [MAS AIRG](#ref-MAS-AIRG).

- Rate impact, low to critical (MAS AIRG 4.12(a)).
- Rate complexity, low to critical (MAS AIRG 4.12(b)).
- Rate reliance, including the autonomy granted, low to critical (MAS AIRG 4.12(c)).
- Record materiality before controls (MAS AIRG 4.11).
- Record materiality after controls (MAS AIRG 4.11).

<a id="AIAgentShape"></a>
### AI agent

Applies to [AI agent](#AIAgent). Sources: [Kognita roadmap](#ref-KOG-ROADMAP), [MAS AIRG](#ref-MAS-AIRG).

- An agent has one identifier (MAS AIRG footnote 25; Kognita 0.5 item 1).
- An agent carries one maximum autonomy level, L0 to L6 (Kognita 0.5 item 3).
- An agent is linked to at least one registered use case (Kognita 0.5 item 1).
- An agent's deployment version covers prompt, tools, permissions and data sources, not only code (Kognita 0.5 item 1; MAS AIRG 5.25(b)).
- An agent past its review date is refused, so it needs one (Kognita 0.5 item 1).

<a id="AgentToolLevelShape"></a>
### Agent tool level

Applies to [AI agent](#AIAgent). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- An agent is permitted a tool that needs a higher autonomy level than the agent's maximum. A call above the agent's level is refused (Kognita 0.5 item 3).

<a id="ToolShape"></a>
### Tool

Applies to [tool](#Tool). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Every tool declares the autonomy level it requires (Kognita 0.5 item 3).
- A tool declares the verbs it performs, because action permissions are per verb (Kognita 0.5 item 3).

<a id="DelegatedAuthorityShape"></a>
### Delegated authority

Applies to [delegated authority](#DelegatedAuthority). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- A delegation is granted by one named person (Kognita 0.5 item 2).
- A delegation is granted to one registered agent (Kognita 0.5 item 2).
- A delegation is bound to a purpose (Kognita 0.5 item 2).
- A delegation names the actions it allows (Kognita 0.5 item 2).
- A delegation must expire even when it also ends on task completion (Kognita 0.5 item 2).

<a id="ClientCommunicationShape"></a>
### Client communication

Applies to [client communication](#ClientCommunication). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- A communication that does not trace back to a final decision is refused (Kognita 0.4 item 12).
- Every communication is classified by communication level (Kognita 0.4 item 12).
- Every communication carries its content provenance label (Kognita 0.4 item 11).
- Client communication goes only through a registered, approved channel (Kognita 0.4 item 12).

<a id="AIDisclosureShape"></a>
### AI disclosure

Applies to [client communication](#ClientCommunication). Sources: [Kognita private-banking pack](#ref-KOG-PB-PACK), [MAS AIRG](#ref-MAS-AIRG).

- When AI generated the content, the client communication must say so (Kognita private-banking pack, client-ai-disclosure; MAS AIRG 5.6).

*Kognita's private-banking pack applies this rule in SG, HK, CH and AE. Where no disclosure rule applies, a firm can switch the shape off in its own shapes graph with sh:deactivated true.*

<a id="ClaimShape"></a>
### Claim

Applies to [claim](#Claim). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Unlabelled claims cannot be shown to an RM or sent to a client (Kognita 0.6 item 5).
- validator status: at most 1
- A claim typed as verified fact must cite a source and carry the validator status verified. The model's own knowledge is never a source, and an inference cannot become a fact without verification (Kognita 0.6 items 3, 5 and 7).

<a id="FactShape"></a>
### Fact

Applies to [fact](#Fact). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- A fact carries its as-of time (Kognita 0.6 item 2).
- A fact carries a freshness threshold (Kognita 0.6 item 2).
- A fact carries its classification (Kognita 0.6 item 2).
- A fact names its source system (Kognita 0.6 item 2).

<a id="ValuationShape"></a>
### Valuation

Applies to [valuation](#Valuation). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- amount: exactly one, of type decimal
- currency: exactly one
- as of: at least 1, of type dateTime
- A figure presented as authoritative must cite a registered calculation service (Kognita 0.6 item 6).

<a id="SuitabilityAssessmentShape"></a>
### Suitability assessment

Applies to [suitability assessment](#SuitabilityAssessment). Sources: [MiFID II](#ref-EU-MIFID2), [MiFID II Delegated Regulation](#ref-EU-MIFID2-DR), [Kognita private-banking pack](#ref-KOG-PB-PACK).

- Name the recommendation being assessed.
- Cite the one risk profile in force when the assessment was made (MiFID II Art. 25(2); Delegated Regulation (EU) 2017/565 Art. 54).
- Check the recommendation against the recorded mandate (Kognita private-banking pack, mandate).
- suitability outcome: exactly one, one of suitable, suitable with conditions, not suitable, insufficient information
- Record how the recommendation meets the client's objectives and characteristics, not a bare score (MiFID II Art. 25(6)).
- Record who made the assessment.

<a id="InvestorRiskProfileShape"></a>
### Investor risk profile

Applies to [investor risk profile](#InvestorRiskProfile). Sources: [MiFID II Delegated Regulation](#ref-EU-MIFID2-DR).

- risk profile level: exactly one
- valid from: at least 1, of type dateTime
- wasAttributedTo: at least 1

<a id="InvestorClassificationShape"></a>
### Investor classification

Applies to [investor classification](#InvestorClassification). Sources: [FinSA](#ref-CH-FINSA), [MiFID II](#ref-EU-MIFID2), [Securities and Futures Act (SG)](#ref-SG-SFA).

- investor category: exactly one
- A classification is made under one jurisdiction's rules.
- opt status: at most 1

<a id="GovernedInformationAssetShape"></a>
### Governed information asset

Applies to [governed information asset](#GovernedInformationAsset). Sources: [Filing Cabinet talk](#ref-AM-FILING-CABINET).

- Minimum label: owner (Filing Cabinet, slide 8).
- Minimum label: approval status (Filing Cabinet, slide 8).
- Minimum label: effective date (Filing Cabinet, slide 8).
- Minimum label: retention date (Filing Cabinet, slide 8).
- Minimum label: version identifier or content hash (Filing Cabinet, slide 8).

<a id="DocumentSectionShape"></a>
### Document section

Applies to [document section](#DocumentSection). Sources: [Filing Cabinet talk](#ref-AM-FILING-CABINET).

- Classify per section, not per document (Filing Cabinet, slide 8).

<a id="PolicyRuleShape"></a>
### Policy rule

Applies to [policy rule](#PolicyRule). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Every rule carries a citation to what it implements (Kognita 0.4 item 3).
- Rules are effective-dated (Kognita 0.3 item 7).
- A change to a rule is a new version, never an edit in place (Kognita 0.3 item 7).

<a id="CrossBorderRuleShape"></a>
### Cross-border rule

Applies to [cross-border rule](#CrossBorderRule). Sources: [Kognita private-banking pack](#ref-KOG-PB-PACK).

- applies to activity: at least 1
- permitted: exactly one, of type boolean
- Client domicile is an ISO 3166-1 alpha-2 code.
- Booking centre is an ISO 3166-1 alpha-2 code.

<a id="PolicyDecisionShape"></a>
### Policy decision

Applies to [policy decision](#PolicyDecision). Sources: [Filing Cabinet talk](#ref-AM-FILING-CABINET), [Kognita roadmap](#ref-KOG-ROADMAP).

- decision outcome: exactly one, one of deny, escalate, human approval, allow, observe
- Every decision names the rule and rule version that decided it (Filing Cabinet, slide 14).
- Every decision references a registered use case (Kognita 0.4 item 8).

<a id="EvidenceRecordShape"></a>
### Evidence record

Applies to [evidence record](#EvidenceRecord). Sources: [Kognita regulations library](#ref-KOG-CONTROLS), [MAS AIRG](#ref-MAS-AIRG).

- event type: exactly one
- Evidence records are hash-chained (Kognita control frameworks).
- An evidence record states when the event happened (MAS AIRG 5.23(d)).
- previous record hash: at most 1
- Every evidence record names the hash of the record before it, except the first record of a chain, which is marked chain start (Kognita control frameworks: hash-chained evidence).

<a id="CurrencyCodeShape"></a>
### Currency code

Applies to subjects of [currency](#currency). Sources: [ISO 4217](#ref-ISO-4217).

- Currency is an ISO 4217 alphabetic code, such as SGD.

<a id="CountryCodeShape"></a>
### Country code

Applies to subjects of [residence country](#residenceCountry), subjects of [domicile country](#domicileCountry), subjects of [nationality country](#nationalityCountry), subjects of [jurisdiction code](#jurisdictionCode), subjects of [tax residence country](#taxResidenceCountry). Sources: [ISO 3166](#ref-ISO-3166).

- Residence is an ISO 3166-1 alpha-2 code, such as SG.
- Domicile is an ISO 3166-1 alpha-2 code, such as SG.
- Nationality is an ISO 3166-1 alpha-2 code, such as SG.
- Jurisdiction is an ISO 3166-1 alpha-2 code, such as SG.
- Tax residence is an ISO 3166-1 alpha-2 code, such as SG.

<a id="LEIShape"></a>
### Legal Entity Identifier

Applies to subjects of [legal entity identifier](#legalEntityIdentifier). Sources: [ISO 17442](#ref-ISO-17442).

- An LEI is 18 letters or digits followed by two check digits (ISO 17442).

<a id="ISINShape"></a>
### ISIN

Applies to subjects of [ISIN](#isin). Sources: [ISO 6166](#ref-ISO-6166).

- An ISIN is a two-letter prefix, nine letters or digits and a check digit (ISO 6166).

<a id="CFIShape"></a>
### CFI code

Applies to subjects of [CFI code](#cfiCode). Sources: [ISO 10962](#ref-ISO-10962).

- A CFI code is six letters (ISO 10962).

<a id="EUSustainabilityAssessmentShape"></a>
### EU sustainability assessment

Applies to [suitability assessment](#SuitabilityAssessment). Sources: [MiFID sustainability preferences](#ref-EU-SUSTAINABILITY).

- EU suitability cites one recorded sustainability preference, including explicit no preference (2017/565 Art. 54(2)(a), (5), as amended by 2021/1253 Art. 1(6)).

<a id="SustainabilityPreferenceShape"></a>
### Sustainability preference

Applies to [sustainability preference](#SustainabilityPreference). Sources: [MiFID sustainability preferences](#ref-EU-SUSTAINABILITY).

- sustainability preference kind: at least 1
- minimum taxonomy aligned proportion: at most 1, of type decimal
- minimum SFDR sustainable proportion: at most 1, of type decimal
- PAI consideration elements: of type string
- Record the minimum proportion for each selected category and client-determined PAI elements; no preference is exclusive and has no category details (2017/565 Art. 2(7)(a)-(c)).

<a id="RunShape"></a>
### RunShape

Applies to [run](#Run). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record hasBudget (roadmap milestone 0.3).
- Record runState (roadmap milestone 0.3).
- A resumed run must have a resumption referencing a resolved approval (0.3 item 2).

*Roadmap milestone 0.3 item 1.*

<a id="RunBudgetShape"></a>
### RunBudgetShape

Applies to [run budget](#RunBudget). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record maxCalls (roadmap milestone 0.3).
- Record maxTokens (roadmap milestone 0.3).
- Record maxCostUSD (roadmap milestone 0.3).
- Record wallClockSeconds (roadmap milestone 0.3).
- Record consumedCalls (roadmap milestone 0.3).
- Record consumedTokens (roadmap milestone 0.3).
- Record consumedCostUSD (roadmap milestone 0.3).
- Record elapsedSeconds (roadmap milestone 0.3).
- Record classificationCeiling (roadmap milestone 0.3).

*Roadmap milestone 0.3 item 1.*

<a id="BudgetExceededDenialShape"></a>
### BudgetExceededDenialShape

Applies to [budget exceeded denial](#BudgetExceededDenial). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record exceededBudget (roadmap milestone 0.3).
- Record budgetDimension (roadmap milestone 0.3).
- decision outcome: checked

*Roadmap milestone 0.3 item 1.*

<a id="CheckpointShape"></a>
### CheckpointShape

Applies to [run checkpoint](#RunCheckpoint). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record inRun (roadmap milestone 0.3).
- Record contentHash (roadmap milestone 0.3).

*Roadmap milestone 0.3 item 2.*

<a id="ApprovalRequestShape"></a>
### ApprovalRequestShape

Applies to [approval request](#ApprovalRequest). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record requestsDecision (roadmap milestone 0.3).
- Record atCheckpoint (roadmap milestone 0.3).

*Roadmap milestone 0.3 item 2.*

<a id="ApprovalResolutionShape"></a>
### ApprovalResolutionShape

Applies to [approval resolution](#ApprovalResolution). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record resolvesApproval (roadmap milestone 0.3).
- Record approvalResolution (roadmap milestone 0.3).
- wasAttributedTo: exactly one, an instance of Person
- generated at time: exactly one, of type dateTime

*Roadmap milestone 0.3 item 2.*

<a id="RunResumptionShape"></a>
### RunResumptionShape

Applies to [run resumption](#RunResumption). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record inRun (roadmap milestone 0.3).
- Record resumedFrom (roadmap milestone 0.3).
- Record resolvedApproval (roadmap milestone 0.3).
- Resumption must cite a resolved approval for its run checkpoint (0.3 item 2).

*Roadmap milestone 0.3 item 2.*

<a id="ModelCallShape"></a>
### ModelCallShape

Applies to [model call](#ModelCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record modelName (roadmap milestone 0.3).
- Record modelVersion (roadmap milestone 0.3).
- Record promptTemplateVersion (roadmap milestone 0.3).
- Record promptHash (roadmap milestone 0.3).
- Record responseHash (roadmap milestone 0.3).
- Record providedBy (roadmap milestone 0.3).
- Record inputTokens (roadmap milestone 0.3).
- Record outputTokens (roadmap milestone 0.3).

*Roadmap milestone 0.3 item 7.*

<a id="ToolCallShape"></a>
### ToolCallShape

Applies to [tool call](#ToolCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record responseHash (roadmap milestone 0.3).
- Record invokedTool (roadmap milestone 0.3).

*Roadmap milestone 0.3 item 7.*

<a id="MCPProxiedCallShape"></a>
### MCPProxiedCallShape

Applies to [MCP proxied call](#MCPProxiedCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record mcpBackend (roadmap milestone 0.3).

*Roadmap milestone 0.3 item 5.*

<a id="ClassifierOutputShape"></a>
### ClassifierOutputShape

Applies to [classifier output](#ClassifierOutput). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record classifierModelId (roadmap milestone 0.3).
- Record classifierVersion (roadmap milestone 0.3).
- Record classifierLabel (roadmap milestone 0.3).
- Record inputHash (roadmap milestone 0.3).
- Record confidence (roadmap milestone 0.3).

*Roadmap milestone 0.3 item 4.*

<a id="EnvelopeAttributeShape"></a>
### EnvelopeAttributeShape

Applies to [envelope attribute](#EnvelopeAttribute). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record attributeName (roadmap milestone 0.3).
- Record attributeValue (roadmap milestone 0.3).
- Record attributeSource (roadmap milestone 0.3).
- Classifier-derived attribute needs calibrated confidence and classifier version (0.3 item 4).
- Identity attributes come from authentication, never classifier text (0.3 item 4 and Tier 0).

*Roadmap milestone 0.3 item 4.*

<a id="ClassifierThresholdShape"></a>
### ClassifierThresholdShape

Applies to subjects of [has attribute](#hasAttribute), subjects of [escalation threshold](#escalationThreshold). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- escalation threshold: at most 1, of type decimal
- Below-threshold or missing-threshold classifier evidence cannot yield ALLOW (0.3 item 4).
- Classifier fills gaps only; a typed attribute remains authoritative (0.3 item 4).

*Roadmap milestone 0.3 item 4.*

<a id="PolicyCheckShape"></a>
### PolicyCheckShape

Applies to [policy check](#PolicyCheck). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record checkedRule (roadmap milestone 0.3).
- Record policyContentHash (roadmap milestone 0.3).

*Roadmap milestone 0.3 item 7.*

<a id="RetrievedItemShape"></a>
### RetrievedItemShape

Applies to [retrieved item](#RetrievedItem). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record contentHash (roadmap milestone 0.3).
- Record embeddingModel (roadmap milestone 0.3).

*Roadmap milestone 0.3 item 7.*

<a id="RetentionEntryShape"></a>
### RetentionEntryShape

Applies to [retention entry](#RetentionEntry). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record contentHash (roadmap milestone 0.3).
- Record underUseCase (roadmap milestone 0.3).
- store locator: at most 1, of type anyURI

*Roadmap milestone 0.3 item 7.*

<a id="RetentionPolicyShape"></a>
### RetentionPolicyShape

Applies to [retention policy](#RetentionPolicy). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record retentionDuration (roadmap milestone 0.3).

*Roadmap milestone 0.3 item 7.*

<a id="ErasureShape"></a>
### ErasureShape

Applies to [erasure event](#ErasureEvent). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record erasedEntry (roadmap milestone 0.3).
- Record contentHash (roadmap milestone 0.3).
- Erasure removes stored content reference and keeps the same hash on the entry, event and surviving chain evidence (0.3 item 7).

*Roadmap milestone 0.3 item 7.*

<a id="ReconstructionReportShape"></a>
### ReconstructionReportShape

Applies to [reconstruction report](#ReconstructionReport). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record partOfInteraction (roadmap milestone 0.3).
- Record chainVerified (roadmap milestone 0.3).
- Record pinnedHashesVerified (roadmap milestone 0.3).
- Cover all ten questions explicitly, marking absent evidence not recorded (0.3 item 8).
- Report hash mismatches as findings (0.3 item 8).

*Roadmap milestone 0.3 item 8.*

<a id="QuestionCoverageShape"></a>
### QuestionCoverageShape

Applies to [question coverage](#QuestionCoverage). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record reconstructionQuestion (roadmap milestone 0.3).
- Record coverageStatus (roadmap milestone 0.3).
- Answered question must cite recorded evidence (0.3 item 8).

*Roadmap milestone 0.3 item 8.*

<a id="ReconstructionFindingShape"></a>
### ReconstructionFindingShape

Applies to [reconstruction finding](#ReconstructionFinding). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record findingKind (roadmap milestone 0.3).
- Record findingRecord (roadmap milestone 0.3).
- Hash mismatch finding records different expected and observed hashes (0.3 item 8).

*Roadmap milestone 0.3 item 8.*

<a id="GatewayConfigurationShape"></a>
### GatewayConfigurationShape

Applies to [AI use case](#AIUseCase). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record gatewayFailureMode (roadmap milestone 0.3).
- retention policy: exactly one, an instance of RetentionPolicy

*Roadmap milestone 0.3 item 9.*

<a id="GovernedCallShape"></a>
### GovernedCallShape

Applies to [governed call](#GovernedCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- call agent: at most 1
- system trigger: at most 1
- Record inRun (roadmap milestone 0.3).
- Record governedByDecision (roadmap milestone 0.3).
- Record clientConfiguration (roadmap milestone 0.3).
- Call identity must be an agent bound to the authenticated client or a listed system trigger (0.3 Tier 0).

*Roadmap milestone 0.3 item 3.*

<a id="PinnedDecisionShape"></a>
### Pinned decision checks

Applies to [policy decision](#PolicyDecision). Sources: [Kognita roadmap](#ref-KOG-ROADMAP), [MAS AIRG](#ref-MAS-AIRG).

- Every policy decision carries pinned component checks (0.3 item 7).

*Roadmap milestone 0.3 item 7; MAS AIRG 5.23(d) record keeping.*

<a id="SuspendedApprovalShape"></a>
### Approval checkpoint consistency

Applies to [approval request](#ApprovalRequest). Sources: [Kognita roadmap](#ref-KOG-ROADMAP), [MAS AIRG](#ref-MAS-AIRG).

- An approval request holds a HUMAN_APPROVAL decision at a suspended checkpoint of the same run (0.3 item 2).

*Roadmap milestone 0.3 item 2; MAS AIRG 5.9 human oversight.*

<a id="IngestedDocumentShape"></a>
### IngestedDocumentShape

Applies to [ingested document](#IngestedDocument). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record parserName (Kognita 0.4 item 1).
- Record parserVersion (Kognita 0.4 item 1).
- Record contentHash (Kognita 0.4 item 1).

<a id="DocumentElementShape"></a>
### DocumentElementShape

Applies to [document element](#DocumentElement). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record ingestedFrom (Kognita 0.4 item 1).
- Record pageNumber (Kognita 0.4 item 1).
- Record sectionPath (Kognita 0.4 item 1).
- Record characterStart (Kognita 0.4 item 1).
- Record characterEnd (Kognita 0.4 item 1).
- Record classification (Kognita 0.4 item 1).
- Record contentHash (Kognita 0.4 item 1).

<a id="DeclarativePolicyShape"></a>
### DeclarativePolicyShape

Applies to [declarative policy](#DeclarativePolicy). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record versionId (Kognita 0.4 item 3).
- Record containsRule (Kognita 0.4 item 3).
- Record policySerialization (Kognita 0.4 item 3).
- Record hasPolicyTest (Kognita 0.4 item 3).

<a id="PolicyTestCaseShape"></a>
### PolicyTestCaseShape

Applies to [policy test case](#PolicyTestCase). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record testsRule (Kognita 0.4 item 4).
- Record givenAttributes (Kognita 0.4 item 4).
- Record whenAction (Kognita 0.4 item 4).
- Record expectedOutcome (Kognita 0.4 item 4).

<a id="PolicyPackShape"></a>
### PolicyPackShape

Applies to [policy pack](#PolicyPack). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record versionId (Kognita 0.4 item 5).
- Record packPolicy (Kognita 0.4 item 5).
- Record fixtureLocator (Kognita 0.4 item 5).

<a id="RMReviewEditShape"></a>
### RMReviewEditShape

Applies to [rmreview edit](#RMReviewEdit). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record beforeHash (Kognita 0.4 item 11).
- Record afterHash (Kognita 0.4 item 11).
- Record editDiffHash (Kognita 0.4 item 11).

<a id="ReviewQueueEntryShape"></a>
### ReviewQueueEntryShape

Applies to [review queue entry](#ReviewQueueEntry). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record inReviewQueue (Kognita 0.4 item 13).
- Record partOfInteraction (Kognita 0.4 item 13).
- Record queueReason (Kognita 0.4 item 13).
- Record queuedAt (Kognita 0.4 item 13).
- Record reviewBeforeDelivery (Kognita 0.4 item 13).

<a id="ReviewCoverageReportShape"></a>
### ReviewCoverageReportShape

Applies to [review coverage report](#ReviewCoverageReport). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record eligibleCount (Kognita 0.4 item 13).
- Record screenedCount (Kognita 0.4 item 13).
- Record queuedCount (Kognita 0.4 item 13).
- Record reviewedCount (Kognita 0.4 item 13).
- Record underUseCase (Kognita 0.4 item 13).
- Record windowStart (Kognita 0.4 item 13).
- Record windowEnd (Kognita 0.4 item 13).

<a id="CircuitBreakerTripShape"></a>
### CircuitBreakerTripShape

Applies to [circuit breaker trip](#CircuitBreakerTrip). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record trippedBreaker (Kognita 0.4 item 14).
- Record flagThreshold (Kognita 0.4 item 14).
- Record observedFlags (Kognita 0.4 item 14).
- Record prohibitingRule (Kognita 0.4 item 14).
- Record windowStart (Kognita 0.4 item 14).
- Record windowEnd (Kognita 0.4 item 14).

<a id="AffectedClientQueryShape"></a>
### AffectedClientQueryShape

Applies to [affected client query](#AffectedClientQuery). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record windowStart (Kognita 0.4 item 14).
- Record windowEnd (Kognita 0.4 item 14).
- Record queryResult (Kognita 0.4 item 14).

<a id="InteractionOutcomeShape"></a>
### InteractionOutcomeShape

Applies to [interaction outcome](#InteractionOutcome). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record outcomeKind (Kognita 0.4 item 15).
- Record partOfInteraction (Kognita 0.4 item 15).
- Record affectedOutput (Kognita 0.4 item 15).
- Record usesModel (Kognita 0.4 item 15).
- Record outcomeApprover (Kognita 0.4 item 15).

<a id="BreakerScopeShape"></a>
### BreakerScopeShape

Applies to [circuit breaker trip](#CircuitBreakerTrip), [circuit breaker](#CircuitBreaker), [affected client query](#AffectedClientQuery). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- BreakerScopeShape: denied or incomplete record (Kognita 0.4 item 14).

<a id="TwoSignatureShape"></a>
### TwoSignatureShape

Applies to subjects of [requires two signatures](#requiresTwoSignatures). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- TwoSignatureShape: denied or incomplete record (Kognita 0.4 item 12).

<a id="PastReviewDecisionShape"></a>
### PastReviewDecisionShape

Applies to [policy decision](#PolicyDecision). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- PastReviewDecisionShape: denied or incomplete record (Kognita 0.4 item 8).

<a id="RiskAppetiteDecisionShape"></a>
### RiskAppetiteDecisionShape

Applies to [policy decision](#PolicyDecision). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- RiskAppetiteDecisionShape: denied or incomplete record (Kognita 0.4 item 8).

<a id="CoverageCountsShape"></a>
### CoverageCountsShape

Applies to [review coverage report](#ReviewCoverageReport). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- CoverageCountsShape: denied or incomplete record (Kognita 0.4 item 13).

<a id="CircuitBreakerConfigurationShape"></a>
### CircuitBreakerConfigurationShape

Applies to [circuit breaker](#CircuitBreaker). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record flagThreshold (Kognita 0.4 item 14).
- Record windowStart (Kognita 0.4 item 14).
- Record windowEnd (Kognita 0.4 item 14).

<a id="FlightRecorderSessionShape"></a>
### FlightRecorderSessionShape

Applies to [flight recorder session](#FlightRecorderSession). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record recordedRun (Kognita 0.4 item 7).
- Record contentHash (Kognita 0.4 item 7).

<a id="ReplayShape"></a>
### ReplayShape

Applies to [replay](#Replay). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record replaysSession (Kognita 0.4 item 7).
- Record replayReport (Kognita 0.4 item 7).

<a id="PolicyChangeShape"></a>
### PolicyChangeShape

Applies to [policy change](#PolicyChange). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record changesPolicy (Kognita 0.4 item 3).
- Record deltaText (Kognita 0.4 item 3).
- Record changeStage (Kognita 0.4 item 3).

<a id="NearMissClassificationShape"></a>
### NearMissClassificationShape

Applies to [near miss](#NearMiss). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record nearMissKind (Kognita 0.4 item 15).
- Record partOfInteraction (Kognita 0.4 item 15).

<a id="ReviewSamplingRateShape"></a>
### ReviewSamplingRateShape

Applies to subjects of [low risk sampling rate](#lowRiskSamplingRate). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record lowRiskSamplingRate (Kognita 0.4 item 13).

<a id="AffectedClientResultShape"></a>
### AffectedClientResultShape

Applies to [affected client result](#AffectedClientResult). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record resultCount (Kognita 0.4 item 14).

<a id="AffectedClientCountShape"></a>
### AffectedClientCountShape

Applies to [affected client result](#AffectedClientResult). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Affected-client count must equal distinct recipients of the returned communications (0.4 item 14).

<a id="ModelCallCardShape"></a>
### ModelCallCardShape

Applies to [model call](#ModelCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Pinned model calls must link to their approved model card and exact version (0.4 item 8).

<a id="LiveRiskAppetiteShape"></a>
### Live use case risk appetite

Applies to [AI use case](#AIUseCase). Sources: [Kognita roadmap](#ref-KOG-ROADMAP), [MAS AIRG](#ref-MAS-AIRG).

- An explicit failed risk-appetite evaluation prevents activation (Kognita 0.4 item 8; MAS AIRG 4.11).

<a id="AgentIdentityShape"></a>
### AgentIdentityShape

Applies to [AI agent](#AIAgent). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record workloadIdentity (Kognita 0.5 item 1).
- Record agentOwner (Kognita 0.5 item 1).
- Record operatesFor (Kognita 0.5 item 1).
- Record hasPurpose (Kognita 0.5 item 1).
- Record deploymentDate (Kognita 0.5 item 1).
- Record deploymentModel (Kognita 0.5 item 1).
- Record deploymentEnvironment (Kognita 0.5 item 1).
- Record agentStatus (Kognita 0.5 item 1).
- Record approvedDeploymentVersion (Kognita 0.5 item 1).

*Roadmap milestone 0.5 item 1. Checks recorded evidence; runtime enforcement remains external.*

<a id="WorkloadIdentityShape"></a>
### WorkloadIdentityShape

Applies to [workload identity](#WorkloadIdentity). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record credentialReference (Kognita 0.5 item 1).
- Record credentialIssuer (Kognita 0.5 item 1).
- Record workloadSubject (Kognita 0.5 item 1).
- Record deploymentEnvironment (Kognita 0.5 item 1).
- Workload subjects must not be shared (Kognita 0.5 item 1).

*Roadmap milestone 0.5 item 1. Checks recorded evidence; runtime enforcement remains external.*

<a id="AgentDecisionShape"></a>
### AgentDecisionShape

Applies to [governed call](#GovernedCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Agent calls require live approved deployment, current review and unrevoked own workload identity (Kognita 0.5 item 1).

*Roadmap milestone 0.5 item 1. Checks recorded evidence; runtime enforcement remains external.*

<a id="AuthorityScopeShape"></a>
### AuthorityScopeShape

Applies to [authority scope](#AuthorityScope). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record aboutClient (Kognita 0.5 item 2).
- Record hasPurpose (Kognita 0.5 item 2).
- Record actionVerb (Kognita 0.5 item 2).
- Record classification (Kognita 0.5 item 2).
- Record scopeTool (Kognita 0.5 item 2).
- Record scopeField (Kognita 0.5 item 2).
- Record jurisdictionCode (Kognita 0.5 item 2).
- scope autonomy level: exactly one

*Roadmap milestone 0.5 item 2. Checks recorded evidence; runtime enforcement remains external.*

<a id="HumanEntitlementShape"></a>
### HumanEntitlementShape

Applies to [human entitlement](#HumanEntitlement). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record authorityScope (Kognita 0.5 item 2).
- Record validFrom (Kognita 0.5 item 2).
- Record validUntil (Kognita 0.5 item 2).
- Record validFrom (Kognita 0.5 item 2).

*Roadmap milestone 0.5 item 2. Checks recorded evidence; runtime enforcement remains external.*

<a id="DelegationScopeShape"></a>
### DelegationScopeShape

Applies to [delegated authority](#DelegatedAuthority). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record authorityScope (Kognita 0.5 item 2).
- Record validFrom (Kognita 0.5 item 2).
- Record delegationDepth (Kognita 0.5 item 2).
- Record maxDelegationDepth (Kognita 0.5 item 2).
- Record validFrom (Kognita 0.5 item 2).
- Record delegationDepth (Kognita 0.5 item 2).
- Grantor must hold an entitlement covering the complete explicit delegation scope and time (Kognita 0.5 item 2).

*Roadmap milestone 0.5 item 2. Checks recorded evidence; runtime enforcement remains external.*

<a id="AuthorityRevocationShape"></a>
### AuthorityRevocationShape

Applies to [authority revocation](#AuthorityRevocation). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record revokedAuthority (Kognita 0.5 item 2).
- Record revokedAt (Kognita 0.5 item 2).
- Record rationale (Kognita 0.5 item 2).

*Roadmap milestone 0.5 item 2. Checks recorded evidence; runtime enforcement remains external.*

<a id="LiveAuthorityDecisionShape"></a>
### LiveAuthorityDecisionShape

Applies to [governed call](#GovernedCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Both call and decision must cite the same live delegation to this agent and a live human entitlement (Kognita 0.5 item 2).

*Roadmap milestone 0.5 item 2. Checks recorded evidence; runtime enforcement remains external.*

<a id="SubDelegationShape"></a>
### SubDelegationShape

Applies to subjects of [parent delegation](#parentDelegation). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- parent delegation: at most 1, an instance of DelegatedAuthority
- Sub-delegation must be acyclic, narrow every explicit scope set and period, and respect depth (Kognita 0.5 item 9).

*Roadmap milestone 0.5 item 9. Checks recorded evidence; runtime enforcement remains external.*

<a id="PromotionGateShape"></a>
### PromotionGateShape

Applies to [autonomy promotion gate](#AutonomyPromotionGate). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record minimumSample (Kognita 0.5 item 3).
- Record minimumWindowSeconds (Kognita 0.5 item 3).
- Record qualityThreshold (Kognita 0.5 item 3).
- Record exceptionRateThreshold (Kognita 0.5 item 3).
- Record requiresFullTraceability (Kognita 0.5 item 3).
- Record stepDownThreshold (Kognita 0.5 item 3).
- Record qualityThreshold (Kognita 0.5 item 3).
- Record exceptionRateThreshold (Kognita 0.5 item 3).
- Record stepDownThreshold (Kognita 0.5 item 3).
- Record minimumWindowSeconds (Kognita 0.5 item 3).

*Roadmap milestone 0.5 item 3. Checks recorded evidence; runtime enforcement remains external.*

<a id="AutonomyTransitionShape"></a>
### AutonomyTransitionShape

Applies to [autonomy transition](#AutonomyTransition). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record transitionAgent (Kognita 0.5 item 3).
- Record promotionGate (Kognita 0.5 item 3).
- Record fromAutonomyLevel (Kognita 0.5 item 3).
- Record toAutonomyLevel (Kognita 0.5 item 3).
- Record sampleCount (Kognita 0.5 item 3).
- Record observedQuality (Kognita 0.5 item 3).
- Record observedExceptionRate (Kognita 0.5 item 3).
- Record fullyReconstructable (Kognita 0.5 item 3).
- Record transitionApproved (Kognita 0.5 item 3).
- Record windowStart (Kognita 0.5 item 3).
- Record windowEnd (Kognita 0.5 item 3).
- Record windowStart (Kognita 0.5 item 3).
- Record observedQuality (Kognita 0.5 item 3).
- Record observedExceptionRate (Kognita 0.5 item 3).

*Roadmap milestone 0.5 item 3. Checks recorded evidence; runtime enforcement remains external.*

<a id="AutonomyEvidenceShape"></a>
### AutonomyEvidenceShape

Applies to [autonomy transition](#AutonomyTransition). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Promotion requires evidence thresholds, sample/window, owner and independent validation; step-down is one level and alerts owners (Kognita 0.5 item 3).

*Roadmap milestone 0.5 item 3. Checks recorded evidence; runtime enforcement remains external.*

<a id="BlastRadiusLimitShape"></a>
### BlastRadiusLimitShape

Applies to [blast radius limit](#BlastRadiusLimit). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record limitDimension (Kognita 0.5 item 4).
- Record limitBasis (Kognita 0.5 item 4).
- Record limitUnit (Kognita 0.5 item 4).
- Record limitMaximum (Kognita 0.5 item 4).
- Record limitMaximum (Kognita 0.5 item 4).
- Record limitDimension (Kognita 0.5 item 4).
- Record limitBasis (Kognita 0.5 item 4).

*Roadmap milestone 0.5 item 4. Checks recorded evidence; runtime enforcement remains external.*

<a id="BlastRadiusObservationShape"></a>
### BlastRadiusObservationShape

Applies to [blast radius observation](#BlastRadiusObservation). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record observedLimit (Kognita 0.5 item 4).
- Record projectedConsumption (Kognita 0.5 item 4).
- Record projectedConsumption (Kognita 0.5 item 4).

*Roadmap milestone 0.5 item 4. Checks recorded evidence; runtime enforcement remains external.*

<a id="BlastRadiusDecisionShape"></a>
### BlastRadiusDecisionShape

Applies to [policy decision](#PolicyDecision). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Projected consumption exceeding any agent blast-radius cap cannot be ALLOW (Kognita 0.5 item 4).

*Roadmap milestone 0.5 item 4. Checks recorded evidence; runtime enforcement remains external.*

<a id="ToolAllowListShape"></a>
### ToolAllowListShape

Applies to [tool call](#ToolCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- A tool outside the agent allow-list must be DENY (Kognita 0.5 item 5).

*Roadmap milestone 0.5 item 5. Checks recorded evidence; runtime enforcement remains external.*

<a id="FieldMinimizationShape"></a>
### FieldMinimizationShape

Applies to [tool call](#ToolCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Requested fields outside purpose-specific agent permissions must be DENY (Kognita 0.5 item 5).

*Roadmap milestone 0.5 item 5. Checks recorded evidence; runtime enforcement remains external.*

<a id="ExternalContentShape"></a>
### ExternalContentShape

Applies to [external content](#ExternalContent). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record trustLabel (Kognita 0.5 item 6).
- Record trustLabel (Kognita 0.5 item 6).
- Record contentHash (Kognita 0.5 item 6).

*Roadmap milestone 0.5 item 6. Checks recorded evidence; runtime enforcement remains external.*

<a id="ContentIsolationShape"></a>
### ContentIsolationShape

Applies to [tool call](#ToolCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Externally tainted runs cannot write; client-data taint cannot reach public-content tools (Kognita 0.5 item 6).

*Roadmap milestone 0.5 item 6. Checks recorded evidence; runtime enforcement remains external.*

<a id="AuthenticatedInstructionShape"></a>
### AuthenticatedInstructionShape

Applies to [authenticated instruction](#AuthenticatedInstruction). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record instructionSource (Kognita 0.5 item 7).
- Record instructionPrincipal (Kognita 0.5 item 7).
- Record assuranceLevel (Kognita 0.5 item 7).
- Record assuranceLevel (Kognita 0.5 item 7).
- Record transactionContext (Kognita 0.5 item 7).

*Roadmap milestone 0.5 item 7. Checks recorded evidence; runtime enforcement remains external.*

<a id="InstructionAssuranceShape"></a>
### InstructionAssuranceShape

Applies to subjects of [requires step up](#requiresStepUp). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- High-risk instructions require step-up workflow evidence and independent authorization, never channel identity alone (Kognita 0.5 item 7).

*Roadmap milestone 0.5 item 7. Checks recorded evidence; runtime enforcement remains external.*

<a id="BehaviourBaselineShape"></a>
### BehaviourBaselineShape

Applies to [behaviour baseline](#BehaviourBaseline). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record baselineAgent (Kognita 0.5 item 8).
- Record versionId (Kognita 0.5 item 8).
- Record baselineMetric (Kognita 0.5 item 8).
- Record baselineExpected (Kognita 0.5 item 8).
- Record anomalyThreshold (Kognita 0.5 item 8).
- Record windowStart (Kognita 0.5 item 8).
- Record windowEnd (Kognita 0.5 item 8).

*Roadmap milestone 0.5 item 8. Checks recorded evidence; runtime enforcement remains external.*

<a id="BehaviourAnomalyShape"></a>
### BehaviourAnomalyShape

Applies to [behaviour anomaly](#BehaviourAnomaly). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record anomalyBaseline (Kognita 0.5 item 8).
- Record anomalyValue (Kognita 0.5 item 8).
- Record containmentApplied (Kognita 0.5 item 8).
- Record anomalyIncident (Kognita 0.5 item 8).
- Record notifiedRole (Kognita 0.5 item 8).

*Roadmap milestone 0.5 item 8. Checks recorded evidence; runtime enforcement remains external.*

<a id="CapabilityGrantShape"></a>
### CapabilityGrantShape

Applies to [capability grant](#CapabilityGrant). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record grantee (Kognita 0.5 item 10).
- Record grantedCapability (Kognita 0.5 item 10).
- Record authorityScope (Kognita 0.5 item 10).
- Record validFrom (Kognita 0.5 item 10).
- Record validUntil (Kognita 0.5 item 10).
- Record validFrom (Kognita 0.5 item 10).

*Roadmap milestone 0.5 item 10. Checks recorded evidence; runtime enforcement remains external.*

<a id="LiveGrantShape"></a>
### LiveGrantShape

Applies to subjects of [relies on grant](#reliesOnGrant). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Used capability grants must match agent and tool and remain unexpired and unrevoked (Kognita 0.5 item 10).

*Roadmap milestone 0.5 item 10. Checks recorded evidence; runtime enforcement remains external.*

<a id="AgentFleetShape"></a>
### AgentFleetShape

Applies to [agent fleet](#AgentFleet). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record fleetMember (Kognita 0.5 item 11).
- Record fleetQuota (Kognita 0.5 item 11).
- Record fleetIsolation (Kognita 0.5 item 11).

*Roadmap milestone 0.5 item 11. Checks recorded evidence; runtime enforcement remains external.*

<a id="FleetHaltShape"></a>
### FleetHaltShape

Applies to [fleet halt](#FleetHalt). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record haltScope (Kognita 0.5 item 11).
- Record prov:generatedAtTime (Kognita 0.5 item 11).
- Record rationale (Kognita 0.5 item 11).

*Roadmap milestone 0.5 item 11. Checks recorded evidence; runtime enforcement remains external.*

<a id="SandboxConstraintsShape"></a>
### SandboxConstraintsShape

Applies to [sandbox constraints](#SandboxConstraints). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record filesystemScope (Kognita 0.5 item 12).
- Record permitsExport (Kognita 0.5 item 12).
- Record permitsPersistence (Kognita 0.5 item 12).
- Record autoStopSeconds (Kognita 0.5 item 12).
- Record cpuSeconds (Kognita 0.5 item 12).
- Record memoryBytes (Kognita 0.5 item 12).
- Record diskBytes (Kognita 0.5 item 12).
- Record autoStopSeconds (Kognita 0.5 item 12).
- Record cpuSeconds (Kognita 0.5 item 12).

*Roadmap milestone 0.5 item 12. Checks recorded evidence; runtime enforcement remains external.*

<a id="ReversibilityShape"></a>
### ReversibilityShape

Applies to [tool](#Tool). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record reversibilityClass (Kognita 0.5 item 13).
- Reversible and compensatable tools register a compensating action (Kognita 0.5 item 13).

*Roadmap milestone 0.5 item 13. Checks recorded evidence; runtime enforcement remains external.*

<a id="IrreversibleApprovalShape"></a>
### IrreversibleApprovalShape

Applies to [tool call](#ToolCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- High-autonomy irreversible action requires exact human approval for the same run (Kognita 0.5 item 13).

*Roadmap milestone 0.5 item 13. Checks recorded evidence; runtime enforcement remains external.*

<a id="AdversarialTestShape"></a>
### AdversarialTestShape

Applies to [adversarial test case](#AdversarialTestCase). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record threatModel (Kognita 0.5 item 14).
- Record attackKind (Kognita 0.5 item 14).

*Roadmap milestone 0.5 item 14. Checks recorded evidence; runtime enforcement remains external.*

<a id="ThreatModelShape"></a>
### ThreatModelShape

Applies to [threat model](#ThreatModel). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record versionId (Kognita 0.5 item 14).
- Record threatModelReference (Kognita 0.5 item 14).

*Roadmap milestone 0.5 item 14. Checks recorded evidence; runtime enforcement remains external.*

<a id="EnforcementAttestationShape"></a>
### EnforcementAttestationShape

Applies to [enforcement attestation](#EnforcementAttestation). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record attestedUseCase (Kognita 0.5 item 15).
- Record prov:generatedAtTime (Kognita 0.5 item 15).
- Record prov:wasAttributedTo (Kognita 0.5 item 15).
- Record gatewayOnlyCredentials (Kognita 0.5 item 15).
- Record egressRestricted (Kognita 0.5 item 15).
- Record evidenceStoreIsolated (Kognita 0.5 item 15).

*Roadmap milestone 0.5 item 15. Checks recorded evidence; runtime enforcement remains external.*

<a id="DeploymentEnforcementShape"></a>
### DeploymentEnforcementShape

Applies to [AI use case](#AIUseCase). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record deploymentMode (Kognita 0.5 item 15).
- Record deploymentMode (Kognita 0.5 item 15).
- Tier 2 or above requires gateway mode and attestation of isolated credentials, routing and evidence (Kognita 0.5 item 15).

*Roadmap milestone 0.5 item 15. Checks recorded evidence; runtime enforcement remains external.*

<a id="BoundedToolShape"></a>
### BoundedToolShape

Applies to [tool](#Tool). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record maximumDurationSeconds (Kognita 0.5 item 16).
- Record maximumDurationSeconds (Kognita 0.5 item 16).
- Record supportsCancellation (Kognita 0.5 item 16).
- Uncancellable tools need compensation or steps with intervening decisions (Kognita 0.5 item 16).

*Roadmap milestone 0.5 item 16. Checks recorded evidence; runtime enforcement remains external.*

<a id="ActionStopShape"></a>
### ActionStopShape

Applies to [action stop](#ActionStop). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record stoppedCall (Kognita 0.5 item 16).
- Record stopOutcome (Kognita 0.5 item 16).
- Record stopOutcome (Kognita 0.5 item 16).
- Record stopReason (Kognita 0.5 item 16).
- Record prov:generatedAtTime (Kognita 0.5 item 16).

*Roadmap milestone 0.5 item 16. Checks recorded evidence; runtime enforcement remains external.*

<a id="AuthorityRequestScopeShape"></a>
### AuthorityRequestScopeShape

Applies to [tool call](#ToolCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- A live delegation must authorize this tool, verb, purpose, client, fields, jurisdiction and data class (Kognita 0.5 item 2).

*Roadmap milestone 0.5 item 2. Checks recorded evidence; runtime enforcement remains external.*

<a id="ScopeNarrowingLimitsShape"></a>
### ScopeNarrowingLimitsShape

Applies to subjects of [parent delegation](#parentDelegation). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Sub-delegation retains all parent numeric caps and may not increase autonomy (Kognita 0.5 item 9).

*Roadmap milestone 0.5 item 9. Checks recorded evidence; runtime enforcement remains external.*

<a id="AutonomyCallShape"></a>
### AutonomyCallShape

Applies to [tool call](#ToolCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Calls cannot exceed effective or maximum autonomy; L4 routes to human approval (Kognita 0.5 item 3).

*Roadmap milestone 0.5 item 3. Checks recorded evidence; runtime enforcement remains external.*

<a id="FleetHaltDecisionShape"></a>
### FleetHaltDecisionShape

Applies to [governed call](#GovernedCall). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- A halted fleet member must be denied on the next decision (Kognita 0.5 item 11).

*Roadmap milestone 0.5 item 11. Checks recorded evidence; runtime enforcement remains external.*

<a id="ToolDurationShape"></a>
### ToolDurationShape

Applies to subjects of [call duration seconds](#callDurationSeconds). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Record callDurationSeconds (Kognita 0.5 item 16).
- A call beyond maximum duration needs an evidenced timeout (Kognita 0.5 item 16).

*Roadmap milestone 0.5 item 16. Checks recorded evidence; runtime enforcement remains external.*

<a id="ExclusiveWorkloadShape"></a>
### Exclusive workload identity

Applies to [AI agent](#AIAgent). Sources: [Kognita roadmap](#ref-KOG-ROADMAP).

- Each registered agent has its own workload identity (Kognita 0.5 item 1).

*Roadmap milestone 0.5 item 1: credentials are never shared between agents.*

<a id="CodeListShapes"></a>
### Code lists

Applies to every coded property. Sources: [SKOS](#ref-W3C-SKOS).

- Each of the 66 coded properties takes its values only from the code list it is bound to with pwm:valuesFrom. The shapes, named pwmsh:CodeList_ followed by the property name, are generated from the bindings. List replacement requires firm shapes and review of handwritten constraints and code-dependent rules; see https://github.com/mze3e/kognita/blob/main/docs/ontology/pwm/README.md.

## Extending PWM for an institution

Keep institution detail in a separate ontology that imports PWM by its version IRI, which resolves to the Turtle file. Specialise PWM classes, add the firm's code lists, and bind coded properties to them with `pwm:valuesFrom`. Keep confidential content out of public repositories.

Save as `firm.ttl`:

```turtle
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix pwm:  <https://ahmedmuzammil.com/ontology/pwm#> .
@prefix pwmsh: <https://ahmedmuzammil.com/ontology/pwm/shapes#> .
@prefix firm: <https://example.org/firm/ontology#> .

<https://example.org/firm/ontology> a owl:Ontology ;
    owl:imports <https://ahmedmuzammil.com/ontology/pwm/0.5> .

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
    rdfs:label "structured note"@en .
```

Save separately as `firm-shapes.ttl`:

```turtle
@prefix sh: <http://www.w3.org/ns/shacl#> .
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
                [ sh:path pwm:targetRiskProfile ; sh:in ( firm:R1 ) ] .
```

Load both shapes graphs together:

```python
from rdflib import Graph
from pyshacl import validate
shapes = Graph().parse('pwm-shapes.ttl').parse('firm-shapes.ttl')
ontology = Graph().parse('pwm.ttl').parse('firm.ttl')
conforms, report, text = validate(Graph().parse('your-data.ttl'),
    shacl_graph=shapes, ont_graph=ontology, inference='rdfs')
print(text)
```

Other code lists may require changes to handwritten constraints and code-dependent rules too; see https://github.com/mze3e/kognita/blob/main/docs/ontology/pwm/README.md.

<a id="references"></a>

## Reference register

| Ref | Mark | Title | Publisher | Date | Jurisdiction | Status | Note |
|---|---|---|---|---|---|---|---|
| <a id="ref-FIBO"></a>FIBO | ONT | [Financial Industry Business Ontology (FIBO)](https://github.com/edmcouncil/fibo) | EDM Council and Object Management Group | 2026-10-27 | Global | verified | Every FIBO IRI and maturity used here was rechecked on 3 November 2026 against raw .rdf files from master at GitHub commit 33a92cd34285d532c158da97bb42add089f135d3 (27 October 2026). Each term's maturity level, Release or Provisional, is taken from the file that defines it. |
| <a id="ref-OMG-COMMONS"></a>OMG Commons | ONT | [Commons Ontology Library](https://www.omg.org/spec/Commons/) | Object Management Group | 2025-08-01 | Global | verified | Collections/hasMember was verified against https://www.omg.org/spec/Commons/20250801/Collections.rdf on 3 November 2026. PartiesAndSituations and Organizations were read at version 20250801. Jurisdiction, RegulatoryAgency, Authorization, isPlayedBy and Document are referenced by FIBO production ontologies; their module files were not opened. |
| <a id="ref-W3C-PROV-O"></a>PROV-O | ONT | [PROV-O: The PROV Ontology](https://www.w3.org/TR/prov-o/) | W3C | 2013-04-30 | Global | verified | Section 3.1 Activity and Agent definitions verified against the W3C Recommendation on 3 November 2026 for Run and SystemTrigger. Other PROV-O terms retain their existing instrument-level grounding. |
| <a id="ref-W3C-SKOS"></a>SKOS | ONT | [SKOS Simple Knowledge Organization System Reference](https://www.w3.org/TR/skos-reference/) | W3C | 2009-08-18 | Global | instrument-level |  |
| <a id="ref-DCMI-TERMS"></a>DCMI Terms | ONT | [DCMI Metadata Terms](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/) | Dublin Core Metadata Initiative |  | Global | instrument-level |  |
| <a id="ref-W3C-DPV"></a>DPV | ONT | [Data Privacy Vocabulary (DPV) 2.3, with the AI extension](https://w3id.org/dpv) | W3C Data Privacy Vocabularies and Controls Community Group |  | Global | verified | Concept IRIs checked against the DPV 2.3 Turtle files (dpv and ai) in the w3c-cg/dpv repository. |
| <a id="ref-ISO-17442"></a>ISO 17442 | ISO | [ISO 17442-1 Financial services: Legal entity identifier (LEI)](https://www.gleif.org/en/about-lei/iso-17442-the-lei-code-structure) | ISO |  | Global | instrument-level |  |
| <a id="ref-ISO-6166"></a>ISO 6166 | ISO | ISO 6166 Securities: International securities identification number (ISIN) | ISO |  | Global | instrument-level |  |
| <a id="ref-ISO-10962"></a>ISO 10962 | ISO | ISO 10962 Securities: Classification of financial instruments (CFI) | ISO |  | Global | instrument-level |  |
| <a id="ref-ISO-4217"></a>ISO 4217 | ISO | ISO 4217 Codes for the representation of currencies | ISO |  | Global | instrument-level |  |
| <a id="ref-ISO-3166"></a>ISO 3166 | ISO | ISO 3166-1 Codes for the representation of names of countries | ISO |  | Global | instrument-level |  |
| <a id="ref-ISO-20022"></a>ISO 20022 | ISO | [ISO 20022 Universal financial industry message scheme](https://www.iso20022.org/) | ISO |  | Global | instrument-level |  |
| <a id="ref-ISO-22222"></a>ISO 22222 | ISO | [ISO 22222:2005 Personal financial planning: Requirements for personal financial planners](https://www.iso.org/standard/43033.html) | ISO | 2005 | Global | verified | Listed as current; confirmed at review in 2023. |
| <a id="ref-ISO-IEC-22989"></a>ISO/IEC 22989 | ISO | ISO/IEC 22989:2022 Artificial intelligence concepts and terminology | ISO/IEC |  | Global | instrument-level | Cited by the MAS AI Risk Management Guidelines (footnote 27) for the AI life cycle. |
| <a id="ref-ISO-IEC-42001"></a>ISO/IEC 42001 | ISO | ISO/IEC 42001:2023 Artificial intelligence management system | ISO/IEC |  | Global | instrument-level |  |
| <a id="ref-ISO-15489"></a>ISO 15489 | ISO | ISO 15489-1:2016 Records management: Concepts and principles | ISO |  | Global | instrument-level |  |
| <a id="ref-FATF"></a>FATF Recommendations | REG | [International Standards on Combating Money Laundering and the Financing of Terrorism and Proliferation (the FATF Recommendations)](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html) | Financial Action Task Force | 2012, as updated | Global | instrument-level | Recommendation 12 (politically exposed persons, including source of wealth and source of funds) was checked; other Recommendation numbers are cited from the published text. |
| <a id="ref-OECD-CRS"></a>OECD CRS | REG | [Standard for Automatic Exchange of Financial Account Information in Tax Matters: Common Reporting Standard](https://www.oecd.org/tax/automatic-exchange/common-reporting-standard/) | OECD |  | Global | verified | Controlling Persons is defined in Section VIII(D)(6). |
| <a id="ref-IMDA-MGF-AGENTIC"></a>IMDA agentic AI framework | REG | [Model AI Governance Framework for Agentic AI](https://www.imda.gov.sg/-/media/imda/files/about/emerging-tech-and-research/artificial-intelligence/mgf-for-agentic-ai.pdf) | Infocomm Media Development Authority (Singapore) |  | SG | instrument-level | Referenced by the MAS AI Risk Management Guidelines, footnote 11. |
| <a id="ref-MAS-AIRG"></a>MAS AIRG | REG | [Guidelines on Artificial Intelligence Risk Management](https://www.mas.gov.sg/-/media/mas-media-library/regulation/guidelines/bd/guidelines-on-artificial-intelligence-risk-management-for-financial-institutions/guidelines-on-artificial-intelligence-risk-management.pdf) | Monetary Authority of Singapore | 2026-10-07 | SG | verified | Read in full from the MAS PDF. Effective 7 October 2027: Sections 3 and 4 from that date; Sections 5 and 6 by 7 October 2028 (para 1.8). MAS intends to consult in 2027 on further guidance for agentic AI. |
| <a id="ref-MAS-FEAT"></a>MAS FEAT | REG | [Principles to Promote Fairness, Ethics, Accountability and Transparency (FEAT) in the Use of AI and Data Analytics in Singapore's Financial Sector](https://www.mas.gov.sg/publications/monographs-or-information-paper/2018/feat) | Monetary Authority of Singapore | 2018 | SG | verified |  |
| <a id="ref-MAS-FAA-N16"></a>MAS FAA-N16 | REG | [Notice FAA-N16 Recommendations on Investment Products](https://www.mas.gov.sg/regulation/notices/notice-faa-n16) | Monetary Authority of Singapore |  | SG | verified |  |
| <a id="ref-MAS-SFA04-N12"></a>MAS SFA 04-N12 | REG | [Notice SFA 04-N12 Sale of Investment Products](https://www.mas.gov.sg/regulation/notices/notice-sfa-04-n12) | Monetary Authority of Singapore |  | SG | verified |  |
| <a id="ref-MAS-626"></a>MAS Notice 626 | REG | [Notice 626 Prevention of Money Laundering and Countering the Financing of Terrorism: Banks](https://www.mas.gov.sg/regulation/notices/notice-626) | Monetary Authority of Singapore |  | SG | verified |  |
| <a id="ref-SG-SFA"></a>Securities and Futures Act (SG) | REG | [Securities and Futures Act 2001, section 4A (accredited, expert and institutional investors), with the opt-in regime for accredited investors](https://sso.agc.gov.sg/Act/SFA2001) | Singapore Parliament; Monetary Authority of Singapore |  | SG | verified |  |
| <a id="ref-SG-BANKING-ACT"></a>Banking Act (SG) | REG | [Banking Act 1970, section 47 (banking secrecy) and the Third Schedule](https://sso.agc.gov.sg/Act/BA1970) | Singapore Parliament |  | SG | instrument-level |  |
| <a id="ref-MAS-FAIR-DEALING"></a>MAS Fair Dealing Guidelines | REG | Guidelines on Fair Dealing: Board and Senior Management Responsibilities for Delivering Fair Dealing Outcomes to Customers | Monetary Authority of Singapore |  | SG | instrument-level | Cited by the MAS AI Risk Management Guidelines, footnote 15. |
| <a id="ref-EU-MIFID2"></a>MiFID II | REG | [Directive 2014/65/EU on markets in financial instruments (MiFID II)](https://eur-lex.europa.eu/eli/dir/2014/65/oj) | European Parliament and Council | 2014-05-15 | EU | instrument-level |  |
| <a id="ref-EU-MIFID2-DR"></a>MiFID II Delegated Regulation | REG | [Commission Delegated Regulation (EU) 2017/565 supplementing MiFID II (organisational requirements and operating conditions)](https://eur-lex.europa.eu/eli/reg_del/2017/565/oj) | European Commission |  | EU | instrument-level |  |
| <a id="ref-EU-PRIIPS"></a>PRIIPs Regulation | REG | [Regulation (EU) No 1286/2014 on key information documents for packaged retail and insurance-based investment products (PRIIPs)](https://eur-lex.europa.eu/eli/reg/2014/1286/oj) | European Parliament and Council |  | EU | instrument-level |  |
| <a id="ref-EU-AI-ACT"></a>EU AI Act | REG | [Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (AI Act)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) | European Parliament and Council |  | EU | instrument-level |  |
| <a id="ref-UK-FCA-COBS"></a>FCA COBS | REG | [FCA Handbook, Conduct of Business Sourcebook (COBS 3 client categorisation; COBS 9A suitability; COBS 10A appropriateness)](https://www.handbook.fca.org.uk/handbook/COBS/) | Financial Conduct Authority |  | GB | instrument-level |  |
| <a id="ref-UK-FCA-FG21-1"></a>FCA FG21/1 | REG | [FG21/1 Guidance for firms on the fair treatment of vulnerable customers](https://www.fca.org.uk/publication/finalised-guidance/fg21-1.pdf) | Financial Conduct Authority | 2021 | GB | instrument-level |  |
| <a id="ref-HK-SFC-COC"></a>SFC Code of Conduct | REG | [Code of Conduct for Persons Licensed by or Registered with the Securities and Futures Commission (para 5.2 suitability; para 15 professional investors)](https://www.sfc.hk/en/Rules-and-standards/Suitability-requirement) | Securities and Futures Commission |  | HK | verified | Paragraph 5.2 (suitability) checked. Paragraph 15 cited from the published Code. |
| <a id="ref-HK-PI-RULES"></a>HK Professional Investor Rules | REG | [Securities and Futures (Professional Investor) Rules (Cap. 571D)](https://www.elegislation.gov.hk/hk/cap571D) | Hong Kong SAR Government |  | HK | instrument-level |  |
| <a id="ref-CH-FINSA"></a>FinSA | REG | [Federal Act on Financial Services (FinSA), SR 950.1](https://www.fedlex.admin.ch/eli/cc/2019/758/en) | Swiss Confederation |  | CH | verified | Article 4 (client segmentation) and Article 5 (opting out and opting in) checked. Articles 10 to 14 (suitability and appropriateness) cited as a range. |
| <a id="ref-US-REG-D"></a>SEC Regulation D | REG | [SEC Regulation D, Rule 501(a) definition of accredited investor (17 CFR 230.501)](https://www.ecfr.gov/current/title-17/section-230.501) | US Securities and Exchange Commission |  | US | instrument-level |  |
| <a id="ref-US-REG-BI"></a>SEC Regulation Best Interest | REG | [SEC Regulation Best Interest (17 CFR 240.15l-1)](https://www.ecfr.gov/current/title-17/section-240.15l-1) | US Securities and Exchange Commission |  | US | instrument-level |  |
| <a id="ref-EU-SUSTAINABILITY"></a>MiFID sustainability preferences | REG | [Commission Delegated Regulation (EU) 2021/1253 amending Delegated Regulation (EU) 2017/565](https://eur-lex.europa.eu/eli/reg_del/2021/1253/oj) | European Commission | 2021-04-21 | EU | verified | Verified against EUR-Lex CELEX 32021R1253, EN/ALL text and relationship table on 3 November 2026. Article 1(1) adds Article 2(7)(a)-(c) to 2017/565. Article 1(6)(a)-(f) amends Article 54(2)(a), (5), (9), (10), (12), (13). These are articles of the amended 2017/565, not Articles 2(7) and 54 of 2021/1253. No preference is an explicit PWM recording code, not a fourth statutory category. |
| <a id="ref-KOG-ROADMAP"></a>Kognita roadmap | KOG | [Kognita Roadmap: From Governance Engine to Agent Harness (docs/ROADMAP.md)](https://github.com/mze3e/kognita/blob/9303769/docs/ROADMAP.md) | Kognita (open source, MIT) | 2026-09, at release 0.7.0 | Global | verified | Read at software release 0.7.0, commit 9303769, not roadmap milestone 0.7. Locators such as 0.4 item 8 refer to roadmap sections as recorded at that software release. Milestone 0.5 items 1 to 16 and its Definition of Done were also re-read in the local docs/ROADMAP.md on 3 November 2026 for ontology 0.5; the original pinned software-release reference is retained. |
| <a id="ref-KOG-CONTROLS"></a>Kognita regulations library | KOG | [AI Control Frameworks: Wealth Management and Agentic AI Coverage (docs/control-frameworks.md), the Kognita regulations library](https://github.com/mze3e/kognita/blob/main/docs/control-frameworks.md) | Kognita (open source, MIT) | 2026-10-12 | Global | verified | Consolidates the 40-domain bank-grade framework, the 70-control wealth management framework (MAS, PDPC, SFC, FCA, Bank of England, APRA, ASIC, CSBS, BIS/FSI, ESMA), a paragraph map of the MAS AI Risk Management Guidelines, and the trust-architecture principles. |
| <a id="ref-KOG-VOCAB"></a>Kognita vocabularies | KOG | [Kognita controlled vocabularies (src/kognita/vocabulary.py, usecases.py, claims.py, memory.py, origination.py)](https://github.com/mze3e/kognita/blob/main/src/kognita/vocabulary.py) | Kognita (open source, MIT) | release 0.7.0 | Global | verified |  |
| <a id="ref-KOG-PB-PACK"></a>Kognita private-banking pack | KOG | [Kognita private-banking policy pack 1.0.0 (src/kognita/packs/data/private-banking)](https://github.com/mze3e/kognita/tree/main/src/kognita/packs/data/private-banking) | Kognita (open source, MIT) |  | Global | verified |  |
| <a id="ref-EVOLVE"></a>EVOLVE | AM | [EVOLVE: Building the AI-Native Enterprise (organisational maturity model for AI)](https://ahmedmuzammil.com/frameworks/evolve) | Ahmed Muzammil |  | Global | verified |  |
| <a id="ref-AM-GOVERNING-AGENTIC-AI"></a>Governing agentic AI (essay) | AM | [Governing agentic AI inside a regulated bank](https://ahmedmuzammil.com/ideas/governing-agentic-ai) | Ahmed Muzammil | 2026-09-21 | Global | verified |  |
| <a id="ref-AM-FILING-CABINET"></a>Filing Cabinet talk | AM | [Your AI Is Only as Good as Your Filing Cabinet (AIIM ASEAN Forum, Singapore)](https://ahmedmuzammil.com/speaking) | Ahmed Muzammil | 2026-10-22 | Global | verified | Checked https://www.ahmedmuzammil.com/sitemap.xml and the speaking page on 3 November 2026. No talk-specific page exists in the sitemap or speaking-page links; the general speaking page is retained. |
| <a id="ref-PWM-PRACTICE"></a>PWM practitioner definition | NEW | Practitioner definition in this ontology | Ahmed Muzammil |  | Global | verified | Used where no term was found in FIBO, OMG Commons or DPV at the versions checked. Each use is a candidate for proposal to the relevant standard. |

## Annotation vocabulary

| Term | Kind | Definition |
|---|---|---|
| <a id="Module"></a>`pwm:Module` | class | Thematic grouping of PWM terms used to organise this specification. |
| <a id="Reference"></a>`pwm:Reference` | class | Published source that a PWM term is grounded in: a standard, an ontology, a regulation or supervisory guideline, or a Kognita or EVOLVE document. |
| <a id="Citation"></a>`pwm:Citation` | class | Pointer from a PWM term to a reference, with a locator inside that reference. |
| <a id="SourceType"></a>`pwm:SourceType` | class | Kind of reference, used to mark how a term is grounded. |
| <a id="ContextSpeed"></a>`pwm:ContextSpeed` | class | How quickly facts of a given kind change, which decides how an AI context layer may hold them. |
| <a id="inModule"></a>`pwm:inModule` | annotation property | Module of this specification that a term belongs to. |
| <a id="moduleOrder"></a>`pwm:moduleOrder` | annotation property | Position of a module in the specification. |
| <a id="citation"></a>`pwm:citation` | annotation property | Links a term to a citation node naming a reference and a locator. |
| <a id="cites"></a>`pwm:cites` | annotation property | Reference that a citation points at. |
| <a id="locator"></a>`pwm:locator` | annotation property | Article, paragraph, section or item inside the cited reference. |
| <a id="sourceType"></a>`pwm:sourceType` | annotation property | Kind of a reference. |
| <a id="jurisdiction"></a>`pwm:jurisdiction` | annotation property | Jurisdiction a reference applies in, as an ISO 3166-1 alpha-2 code, EU, or Global. |
| <a id="verificationStatus"></a>`pwm:verificationStatus` | annotation property | How the reference was checked while drafting this version. |
| <a id="valuesFrom"></a>`pwm:valuesFrom` | annotation property | Code list (SKOS concept scheme) that a coded property takes its values from. |
| <a id="contextSpeed"></a>`pwm:contextSpeed` | annotation property | How quickly facts of this kind change. Fast facts are looked up live at the moment of use and are never cached or embedded for AI. |
| <a id="defaultClassification"></a>`pwm:defaultClassification` | annotation property | Data classification that instances of this class carry unless a more restrictive one applies. |
| <a id="reusedIn"></a>`pwm:reusedIn` | annotation property | Marks an external term that PWM reuses without change, and the module that uses it. |
| <a id="fiboMaturity"></a>`pwm:fiboMaturity` | annotation property | Maturity level of a FIBO term, Release or Provisional, read from the FIBO source file that defines it. |

## Change log

| Version | Date | Change |
|---|---|---|
| 0.5 | 2026-11-03 | Aligned to roadmap milestone 0.5: workload identity, explicit scoped and expiring human authority, autonomy evidence and step-down, blast-radius caps, content trust, instruction assurance, anomaly records, narrowing delegation chains, capability grants, fleets, sandbox output, reversibility, adversarial tests, enforcement attestations and stopping in-flight calls. Existing governance terms are reused. Configuration/grants are SlowContext; execution evidence is MediumContext. |
| 0.4 | 2026-11-03 | Aligned to roadmap milestone 0.4: structured ingestion provenance, versioned declarative policy and scenarios, policy packs, recorder replay, RM edit hashes, review queues and coverage, scoped breaker incidents, affected-client results and outcome classifications. Existing lifecycle and register terms are reused; removed graph item 2 adds nothing. |
| 0.3 | 2026-11-03 | Aligned to roadmap milestone 0.3: run budgets, durable approval continuation, gateway model and MCP calls, classifier evidence, pinned checks and retention, reconstruction coverage, failure mode and authenticated client identity bindings. Review fixes: retentionDuration uses xsd:nonNegativeInteger in whole days; legacy date and duration ranges remain unchanged with an explicit datatype-map limitation and migration open item. RetentionPolicy uses SlowContext/C1; AuthenticatedClientConfiguration and SystemTrigger use SlowContext/C2. |
| 0.2 | 2026-11-03 | Reverified FIBO grounding and maturity; added EU sustainability preferences, code list, two written rules and fixtures; versioned shapes IRI; clarified software release 0.7.0 provenance and repaired the control framework link. |
| 0.1 | 2026-11-02 | First public draft: 10 modules, reference register, SHACL shapes, example data and negative tests. |

## Gap analysis for 0.4

Compared with the inherited PWM 0.3 vocabulary and shapes before adding terms. PWM version N aligns with roadmap milestone N. Locators below refer to `docs/ROADMAP.md`, milestone 0.4, items 1 to 15 and its Definition of Done. Existing terms retain their original grounds.

| Item | Already present in 0.3 | Missing coverage added for 0.4 |
|---|---|---|
| 1 | GovernedInformationAsset, DocumentSection, hasSection, classification, contentHash, RetrievedItem and retained content | IngestedDocument and DocumentElement; exact parser identity/version, source document, page, section path, character offsets, bounding box and parent link. Elements reuse classification and contentHash. Required locators, classification and hashes are written constraints. |
| 3 | Effective-dated PolicyRule, versionId, ruleCitation, PolicyCheck and pinned policy hash | DeclarativePolicy containing existing rules, PolicyChange with base artefact, delta and lifecycle stage. Serialization is data, not a new rule engine. |
| 4 | DecisionOutcomeScheme | PolicyTestCase with rule, given attributes, requested action and expectedOutcome; artefacts link tests. |
| 5 | Existing private-banking policy references and rules | Versioned PolicyPack links policies, tests through their artefacts, fixtures and optional shadowed built-in pack. |
| 7 | Run, EvidenceRecord, ReconstructionReport, ten-question coverage, OutcomeMetric with baseline/target | FlightRecorderSession export manifest and Replay verification activity reusing the report; hasOutcomeMetric binds a use case to its versioned primary measure. |
| 8 | AIUseCase; purpose, status, jurisdiction, mediation, dependencies, owners and review date; six-axis RiskMaterialityAssessment (data sensitivity can use classification), inherent/residual ratings, AIRiskAppetite, ModelCard, tier and control-function approval | hasModelCard links models and exact pinned calls. creditedControl records mitigation. riskAppetiteStatement and appetitePermits record versioned appetite and evaluation. Decision shapes require DENY after review expiry or explicit failed appetite evaluation; an explicitly failed evaluation also prevents live activation. Existing axes and ratings are reused. |
| 9 | ClientInteraction, partOfInteraction and Run | No new journey class or identifier property. Existing RDF identifiers and interaction links span the example. |
| 10 | Origination, trigger, rationale, cited facts, candidate-set fields and do-not-contact codes and constraints | No new origination vocabulary. Reconstruction now answers why-client and why-product from existing origination and recommendation records. |
| 11 | HumanReview, FinalDecision, decision results, contentHash and contentProvenance | RMReviewEdit, reviewEdit, beforeHash, afterHash and editDiffHash preserve rendered input, approved output and structured diff. Existing retention fields locate the diff. |
| 12 | ClientCommunication, channel, recipient, content hash, final-decision lineage, communication levels, approved-channel flag and disclosure; written approved-channel and missing-decision constraints | requiresTwoSignatures and communicationSigner with distinct-human constraint. Existing channel and lineage constraints are reused without duplicates. |
| 13 | HumanReview, classifier evidence, use-case tier | ReviewQueue, ReviewQueueEntry, low-risk sampling fraction, reason and timing, delivery boundary, ReviewCoverageReport with counts, reporting window and latency. Counts permit coverage fractions per use case and its risk tier; impossible count order is rejected. |
| 14 | KillSwitch, KillSwitchDrill, AIIncident, versioned models/rules and evidence | CircuitBreaker configuration and CircuitBreakerTrip incident with scoped use case, exact model, prompt or policy, threshold, window, flags and prohibiting rule. AffectedClientQuery and explicit AffectedClientResult retain recipients and communications with a checked distinct-client count, including zero results. |
| 15 | Complaint, AIIncident, NearMiss and outcome/near-miss evidence event codes | InteractionOutcome with seven outcome classes as codes, and five near-miss classification codes. Exact output, implicated model and approver links support trace-back without inferring that outcomes are model calls. |

Item 2, the graph extra, was removed under ADR 0008. PWM 0.4 models nothing for it. Item 6 concerns framework adapters and adds no domain terms.

Every new class, property and scheme has item-level KOG grounds and a NEW practitioner citation. MAS AIRG grounds reuse the registered 7 October 2026 instrument: 4.5 to 4.13 for inventory/materiality, 5.9 for review, and 5.23 for monitoring/incidents. Existing communication disclosure retains 5.6 and the existing AM Filing Cabinet grounds remain on classification, content hashes, governance and evidence. No new FIBO, OMG or DPV close matches are asserted. Existing verified external parents and ranges are reused.

The Definition of Done describes software behaviour as well as records. These artefacts model the evidence and enforce the written structural constraints; they do not implement Docling, policy compilation, CLI delta resolution, strict validation of serialized YAML, telemetry export, adapters, automatic halt dispatch, query execution, revocation, approval-packet execution or rubber-stamp detection. Appetite evaluation and tier derivation remain external policy computations. An explicit failed appetite evaluation is denied; absence of that evaluation is not treated as proof of permission. Affected-client result counts are checked against recorded communications, not an external delivery system. A production query must establish completeness for the stated scope and window. New code-list replacement requires reviewing these constraints along with the existing ones.


## Gap analysis for 0.5

Compared with inherited PWM 0.4 before adding terms. PWM version N follows roadmap milestone N. All sixteen items and the Definition of Done in `docs/ROADMAP.md`, 0.5 "Agents, Authority and Fleets", were reviewed. Additions live in `src/11-agents-authority-fleets.ttl`; the existing governance terms and their grounds remain in place.

| Item | Already present in 0.4 | Missing coverage added for 0.5 |
|---|---|---|
| 1 Agent identity | AIAgent, agentIdentifier, deploymentVersion, registeredForUseCase, nextReviewDate, hasPurpose, hasAccountabilityRole | WorkloadIdentity with issuer, subject, non-secret credential reference and environment; deployment date/model, named owner, operating principal, approved deployment hash and agentStatus using existing lifecycle codes. Calls bind their workload identity. Written checks reject shared credentials, stale review, unapproved version or inactive agent on non-DENY decisions. Agent status has its own property because lifecycleStatus has an AIUseCase domain. |
| 2 Delegated authority | DelegatedAuthority, delegatedBy/To, aboutClient, actionVerb, permitsDataClass, purpose, jurisdiction, approvalThreshold, validFrom/Until, endsOnTaskCompletion and reliesOnDelegation | AuthorityScope and HumanEntitlement record the dual check explicitly; AuthorityRevocation records revocation including task completion. Expiry is mandatory even with task completion. Calls and decisions reference the same live delegation and its live human entitlement. Explicit client, purpose, verb, tool, data and field sets support request checks. The root human stays delegatedBy throughout lineage. |
| 3 Autonomy | L0 to L6, maximum and required levels, per-verb codes, AutonomyPromotionGate, quality/exception thresholds and full traceability flag | Effective autonomy; AutonomyTransition with old/new levels, sample/window and observed quality/exception metrics, owner and independent validator approval, drift and notifications. Minimum sample/window and lower step-down threshold extend the existing gate. Approved promotions below gates fail; step-down must be one level with owner and Knowledge Lead alerts. Calls above effective/max levels fail and L4 routes to approval. Model version and deployment content hash identify the evaluated behaviour. |
| 4 Blast radius | RunBudget, wall-clock and cost/call/token consumption; mandate concentration limits | BlastRadiusLimit and projected BlastRadiusObservation, independent of run budgets. Dimensions include records, distinct clients, value, actions, transactions and clients contacted; bases include action, run, minute, hour and UTC-day. Units identify currency for value. Agent limits and scope caps reuse these records. Projected excess cannot be ALLOW. Explicit products, systems and jurisdictions use scope sets. limitWindowBasis is distinct from existing concentration limitBasis. |
| 5 Tool/data minimization | Tool, permittedTool, actionVerb, requiredAutonomyLevel | Purpose-specific permittedField and requestedField; scope field sets and event access window. Written rules require DENY for tools or fields outside the agent allow-list and reject requests outside delegated scope. |
| 6 External content | RetrievedItem, classifier evidence and confidentiality classification | ExternalContent, ContentTrustScheme and trustLabel on content, inputContent on calls/runs, publicContentTool. External taint blocks write tools; client-data taint blocks public-content tools. Confidentiality and trust are separate. External content cannot become instruction authority. |
| 7 Instruction authenticity | AuthenticatedClientConfiguration and classifier prohibition on asserting identity | AuthenticatedInstruction with source, authenticated human, assurance, transaction context and optional verified signature reference. High-risk ALLOW needs step-up and human authorization evidence; a channel identity alone is insufficient. Cryptographic verification remains external. |
| 8 Behaviour anomalies | AIIncident, NearMiss, UnusualAgentBehaviour, CircuitBreaker and incident evidence | Versioned BehaviourBaseline and BehaviourAnomaly with metric, threshold, window, observed value, containment assertion, incident and owner notification. Containment means quarantine, credential revocation and tool stop before human release; SHACL checks the record, not dispatch. |
| 9 Agent delegation | Human-to-agent authority, Run and evidence lineage | parentDelegation, depth and total depth cap; children retain the accountable root human. Written checks reject cycles, scope or period widening, depth excess, increased autonomy and omitted/increased parent numeric caps. Parent recipients identify the orchestrator in the authority chain. Budgets can be represented by scope caps on the same dimension/basis/unit. |
| 10 Capability grants | Delegation and validity/evidence patterns | CapabilityGrant with agent grantee, granted tool, subject scope, mandatory expiry and revocation via the common revocation record. Calls link their grant; live checks match tool, agent, time and revocation. Existing evidence records capture issue, use, expiry and revoke events. |
| 11 Fleets | Agents, gateway call identity, Run isolation and classifier evidence | AgentFleet with members, quota and isolation boundary; FleetHalt with mandatory scope and timestamp. Non-DENY decisions after fleet halt fail. Agent messages use inputContent and trust labels; external senders are recorded as untrusted. Actual inter-agent routing, quota admission and isolation remain gateway responsibilities. |
| 12 Sandbox output | PolicyDecision and outcome codes | SandboxConstraints linked from the decision, with credential handles, network destinations, filesystem/API/command scopes, export/persistence flags, auto-stop, CPU/memory/disk caps. An empty network allow-list means block all. The ontology does not ship a sandbox. |
| 13 Reversibility | Tool, HumanReview, ApprovalRequest/Resolution, recover intervention evidence | ReversibilityScheme and CompensatingAction, original-call reference and reverse recovery order. Reversible/compensatable tools register compensation. High-autonomy irreversible ALLOW requires approved human evidence in the same run. Runtime recovery order and exact transaction approval binding require gateway verification. |
| 14 Adversarial suite | PolicyTestCase, testsRule, givenAttributes, whenAction, expectedOutcome and PolicyPack | AdversarialTestCase specialises the existing test record with attack kind and versioned ThreatModel reference. Test-to-rule links map attacks to controls. SHACL validates these declarations; the software conformance suite must execute the attacks. |
| 15 Enforcement | Use-case register, isMediated, gateways and MCP calls | Deployment mode and attributed EnforcementAttestation with gateway credential custody, egress restriction and evidence-store isolation. Tier 2+ use cases require gateway mode and affirmative attestation. A false attestation remains an operational finding, not proof of enforcement. |
| 16 Stop in flight | KillSwitch, Run wall clock, tool/call and evidence records | Maximum tool-call duration, cancellation support, decision-separated steps, observed duration and ActionStop with cancellation/completion/timeout/unsupported result. Uncancellable tools require compensation or intervening decisions. Exceeding duration requires recorded timeout. Positive fixtures include completion, cancellation and timeout. |

All new classes, properties, schemes and codes carry item-level KOG and NEW practitioner citations. Existing verified PROV Entity, OMG Authorization and PolicyTestCase parents are reused; no new FIBO, OMG or DPV matches are invented. MAS AIRG retains the registered 7 October 2026 instrument: 1.11/fn 25 for agent risks and identity, 5.16(c) for plugins/APIs and 5.9 for autonomy oversight. IMDA is retained at instrument level. EVOLVE Stage 4 and the Governing agentic AI essay Sections 03 and 04 ground boundaries, quality gates and autonomy. New specialised definitions remain NEW even when guidance motivates the control.

Configuration, policies, grants, scopes, registry entries, baselines and limits are SlowContext. Recorded decisions, calls, transitions, revocations, attestations and evidence use MediumContext. FastContext remains for live financial state such as holdings and positions. Reclassification of inherited configuration and evidence corrects the context-speed rule without changing their PROV parents.

The Definition of Done includes executable software controls. These artefacts validate the recorded structure, comparisons and decision outcomes; they do not authenticate credentials, verify signatures, compile policy, derive tier caps, calculate complete consumption, infer taint over unrecorded inputs, detect trends, dispatch containment, enforce network isolation, cancel tools or execute compensation. Scope lists are finite and explicit, not wildcard entitlements. Data-class membership is checked exactly, so a child must list each authorized class; a confidentiality ceiling is not silently expanded. Grant and delegation effects are judged at the decision's recorded time. Completeness of runtime snapshots, metrics, authority lineage and enforcement attestations is the firm's responsibility.

Open questions: migrate inherited date/duration ranges in a later compatibility change; choose firm-specific limits, assurance levels, tier/autonomy ceilings and threat-model publication locations; verify runtime proofs of custody, isolation, transaction-specific approval and compensation ordering. No new property uses xsd:date or xsd:duration.


## Verification for 0.5

New 0.5 records specialise prov:Entity except HumanEntitlement and CapabilityGrant (OMG Commons Authorization), and AdversarialTestCase (PolicyTestCase, itself a prov:Entity). AuthorityRevocation, FleetHalt and ActionStop are records about interventions, not prov:Activity. AIAgent retains prov:SoftwareAgent and OMG SoftwareAgent. Existing Run, governed calls and Replay retain their activity parents. No new external matches are asserted. All new datatype-property ranges use OWL 2 datatype-map datatypes.

There are 190 node shapes: 123 written targeted shapes, 66 generated targeted shapes and one untargeted helper.

The positive example contains 178 typed records. The negative fixture contains 152 typed records. There are 126 typed violating records, 26 typed supporting records and 4 untyped violating subjects, yielding 146 expected violations across 130 focus nodes. tools/validate.py checks the exact expected violation multiset; the regression tests require negative coverage of every written shape.

The shapes version IRI is `https://ahmedmuzammil.com/ontology/pwm/0.5/pwm-shapes.ttl`; the unversioned shapes IRI remains the latest pointer. Publication is separate from this build.

All 123 written rules have negative fixtures. Two generated code-list rules are exercised there; all 66 generated rules are tested by regression tests. The positive example conforms and its evidence chain is unbroken. All ten reconstruction questions have recorded answers. Pre-existing pwm:nextReviewDate and pwm:retentionUntil use xsd:date; pwm:freshnessThreshold uses xsd:duration. These datatypes fall outside the OWL 2 datatype map. Their ranges are preserved for compatibility with 0.2. ROBOT DL profile validation passes with xsd:date and xsd:duration as declared datatypes; this does not establish conformance to the OWL 2 datatype map. Open item for a later version: migrate these legacy ranges and their data to OWL 2 datatype-map datatypes. ROBOT verifies HermiT consistency, also against the merged positive example. No DL profile conformance is claimed for example data using xsd:date and xsd:duration outside the OWL 2 DL datatype map. SHACL checks recorded structure; it does not retrieve payloads, recompute hashes or execute runtime controls. Review expiry is evaluated at a policy decision's recorded generatedAtTime, falling back to NOW when absent; historical decisions are judged at their decision time. A decision on an expired use case must be DENY.

New 0.4 PROV specialisations: IngestedDocument, DeclarativePolicy, PolicyChange, PolicyTestCase, PolicyPack, FlightRecorderSession, RMReviewEdit, ReviewQueue, ReviewQueueEntry, ReviewCoverageReport, AffectedClientQuery, AffectedClientResult and InteractionOutcome specialise prov:Entity. Replay specialises prov:Activity. DocumentElement specialises existing DocumentSection (OMG Commons Document); CircuitBreaker specialises KillSwitch; CircuitBreakerTrip specialises AIIncident (prov:Entity). No PROV activity parent is asserted for recorder manifests, edits or trips.

PWM version N aligns to roadmap milestone N. Version 0.5 models milestone 0.5, not the software release number. The inherited 0.3 PROV specialisations remain as follows. New specialised records and fields are marked NEW, with item-level KOG grounds. Run, RunResumption, GovernedCall and its subclasses ModelCall, ToolCall and MCPProxiedCall, and ErasureEvent specialise prov:Activity. RunBudget, RunCheckpoint, ApprovalRequest, ApprovalResolution, BudgetExceededDenial (through PolicyDecision), AuthenticatedClientConfiguration, ClassifierOutput, EnvelopeAttribute, PolicyCheck, RetrievedItem, RetentionEntry, RetentionPolicy, ReconstructionReport, QuestionCoverage and ReconstructionFinding specialise prov:Entity. SystemTrigger specialises prov:Agent. No new FIBO, OMG or DPV alignment is asserted for these specialised gateway fields. Existing modelVersion, providedBy, contentHash, decision outcomes, classifications, event types and evidence links are reused.

Kognita references mean software release **0.7.0**, commit `9303769`, not roadmap milestone 0.7. Item locators retain the roadmap section numbers at that commit.

EUR-Lex CELEX 32021R1253 (EN/ALL text and relationship table) confirms that 2021/1253 Art. 1(1) adds Art. 2(7)(a)-(c) to 2017/565; Art. 1(6) amends Art. 54(2)(a), (5), (9), (10), (12) and (13). The EU rule requires a preference record when jurisdictionCode is EU. Firms must mark applicability; this is a structural check, not product matching or the complete legal suitability test. No preference is an explicit recording code, not a fourth statutory category. New sustainability terms cite that regulation; they do not claim MAS, roadmap or talk grounds for EU law.

No verifiable close match was found for `inInstrument` or `hasUnderlyingInstrument` with their broad PWM semantics, so both remain NEW. FIBO `hasOwnedAsset` describes an ownership situation rather than a holding or position, and `hasUnderlier` describes a derivative rather than any shelf product. No FIBO `CashAccount` was found; PWM CashAccount specialises verified Release `DepositAccount`. `hasHolding` maps closely to verified OMG Commons Collections/hasMember (20250801); `securedBy` maps closely to FIBO isCollateralizedBy. No equivalence is asserted for these narrower properties.

All FIBO term IRIs below were checked against the actual master RDF/XML source, which resolved to commit `33a92cd34285d532c158da97bb42add089f135d3` on 3 November 2026. Maturity comes from the defining ontology's hasMaturityLevel. Source links are pinned to that verified master commit.

| FIBO IRI | Maturity | Verification source |
|---|---|---|
| https://spec.edmcouncil.org/fibo/ontology/BE/FunctionalEntities/FunctionalEntities/FamilyOffice | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/BE/FunctionalEntities/FunctionalEntities.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LEIEntities/LegalEntityIdentifier | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/BE/LegalEntities/LEIEntities.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/CorporateOwnership/BeneficialOwner | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/BE/OwnershipAndControl/CorporateOwnership.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/CorporateOwnership/hasBeneficialOwner | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/BE/OwnershipAndControl/CorporateOwnership.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/BE/OwnershipAndControl/Executives/AuthorizedIndividual | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/BE/OwnershipAndControl/Executives.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/Trust | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/BE/Trusts/Trusts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/TrustBeneficiary | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/BE/Trusts/Trusts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/Trustee | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/BE/Trusts/Trusts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/BE/Trusts/Trusts/Trustor | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/BE/Trusts/Trusts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/DER/DerivativesContracts/StructuredInstruments/CapitalProtectedProduct | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/DER/DerivativesContracts/StructuredInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/DER/DerivativesContracts/StructuredInstruments/LeveragedProduct | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/DER/DerivativesContracts/StructuredInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/DER/DerivativesContracts/StructuredInstruments/StructuredProduct | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/DER/DerivativesContracts/StructuredInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/DebtAndEquities/Debt/isCollateralizedBy | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/DebtAndEquities/Debt.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/DebtInstrument | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/FinancialInstruments/FinancialInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/DerivativeInstrument | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/FinancialInstruments/FinancialInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/EquityInstrument | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/FinancialInstruments/FinancialInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/FinancialInstrument | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/FinancialInstruments/FinancialInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/Issuer | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/FinancialInstruments/FinancialInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/Option | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/FinancialInstruments/FinancialInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/Security | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/FinancialInstruments/FinancialInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/Bank | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/FunctionalEntities/FinancialServicesEntities.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/FunctionalEntities/FinancialServicesEntities/WealthManagementService | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/FunctionalEntities/FinancialServicesEntities.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/Account | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/ProductsAndServices/ClientsAndAccounts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/AccountHolder | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/ProductsAndServices/ClientsAndAccounts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/CustomerAccount | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/ProductsAndServices/ClientsAndAccounts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/DepositAccount | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/ProductsAndServices/ClientsAndAccounts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/Fee | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/ProductsAndServices/ClientsAndAccounts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/InvestmentAccount | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/ProductsAndServices/ClientsAndAccounts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/ClientsAndAccounts/RelationshipManager | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/ProductsAndServices/ClientsAndAccounts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/FinancialProductsAndServices/FinancialProduct | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/ProductsAndServices/FinancialProductsAndServices.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/FinancialProductsAndServices/FinancialServiceProvider | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/ProductsAndServices/FinancialProductsAndServices.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FBC/ProductsAndServices/FinancialProductsAndServices/Position | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FBC/ProductsAndServices/FinancialProductsAndServices.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/hasAmount | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/Accounting/CurrencyAmount.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/Accounting/CurrencyAmount/hasCurrency | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/Accounting/CurrencyAmount.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/AgentsAndPeople/People/Person | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/AgentsAndPeople/People.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/Agreements/Contracts/ContractualCommitment | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/Agreements/Contracts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/Agreements/Contracts/WrittenContract | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/Agreements/Contracts.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/GoalsAndObjectives/Objectives/InvestmentObjective | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/GoalsAndObjectives/Objectives.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/OwnershipAndControl/Ownership/DigitalAsset | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/OwnershipAndControl/Ownership.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/OwnershipAndControl/Ownership/Holding | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/OwnershipAndControl/Ownership.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/OwnershipAndControl/Ownership/Portfolio | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/OwnershipAndControl/Ownership.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/Parties/Parties/TaxIdentifier | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/Parties/Parties.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/FND/ProductsAndServices/ProductsAndServices/Client | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/FND/ProductsAndServices/ProductsAndServices.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/LOAN/LoansGeneral/Loans/CollateralizedLoan | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/LOAN/LoansGeneral/Loans.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Debt/Bonds/Bond | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Debt/Bonds.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Equities/EquityInstruments/Custodian | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Equities/EquityInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Equities/EquityInstruments/Share | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Equities/EquityInstruments.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/CollectiveInvestmentVehicles/KeyInvestorInformationDocument | Provisional | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Funds/CollectiveInvestmentVehicles.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/CollectiveInvestmentVehicles/PortfolioBenchmark | Provisional | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Funds/CollectiveInvestmentVehicles.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/CollectiveInvestmentVehicles/RiskLevel | Provisional | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Funds/CollectiveInvestmentVehicles.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/Funds/ExchangeTradedFund | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Funds/Funds.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/Funds/FundUnit | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Funds/Funds.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/Funds/HedgeFund | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Funds/Funds.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/Funds/MutualFund | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Funds/Funds.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/Funds/PrivateEquityFund | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Funds/Funds.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Funds/Funds/UnitTrust | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Funds/Funds.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/Pools/CollectiveInvestmentVehicle | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Securities/Pools.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesClassification/FinancialInstrumentClassificationCode | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Securities/SecuritiesClassification.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesIssuance/OfferingDocument | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Securities/SecuritiesIssuance.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesIssuance/Prospectus | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Securities/SecuritiesIssuance.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesRestrictions/InvestorsDomicileRestriction | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Securities/SecuritiesRestrictions.rdf) |
| https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/SecuritiesRestrictions/QualifiedInvestorRestriction | Release | [RDF source](https://raw.githubusercontent.com/edmcouncil/fibo/33a92cd34285d532c158da97bb42add089f135d3/SEC/Securities/SecuritiesRestrictions.rdf) |
