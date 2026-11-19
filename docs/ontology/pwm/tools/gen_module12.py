"""Generate PWM module 12 (trust, resilience and assurance): ontology, shapes, example and negative fixtures."""
import re, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
LOC = {
 "INV":"MAS AIRG 4.7 and footnote 25: inventory attributes for AI and for agents",
 "C11G":"roadmap milestone 0.6 item 11 (evaluation evidence gate: thresholds per use case)", "A8":"roadmap milestone 0.8 item A8 (AI-enabled cyber and adversarial drills)",
 "C1":"roadmap milestone 0.6 item 1 (claim provenance envelope)", "C8":"roadmap milestone 0.6 item 8 (memory governance)", "C9":"roadmap milestone 0.6 item 9 (feedback-loop controls)", "C10":"roadmap milestone 0.6 item 10 (sensitive attribute controls)",
 "C7":"roadmap milestone 0.6 item 7 (verification gate)", "C11":"roadmap milestone 0.6 item 11 (evaluation evidence gate)", "C13":"roadmap milestone 0.6 item 13 (reasoning traces)",
 "T1":"roadmap milestone 0.7 item T1 (Ed25519 signing)", "T3":"roadmap milestone 0.7 item T3 (external review API)",
 "T5":"roadmap milestone 0.7 item T5 (schema versioning)", "T6":"roadmap milestone 0.7 item T6 (external anchoring)",
 "R1":"roadmap milestone 0.7 item R1 (backup and restore)", "R2":"roadmap milestone 0.7 item R2 (key rotation, revocation and escrow)",
 "R3":"roadmap milestone 0.7 item R3 (model-provider register)", "R5":"roadmap milestone 0.7 item R5 (provider failover)",
 "R7":"roadmap milestone 0.7 item R7 (silent model-change detection)", "R8":"roadmap milestone 0.7 item R8 (AI dependency map)",
 "R9":"roadmap milestone 0.7 item R9 (concentration risk)", "R10":"roadmap milestone 0.7 item R10 (failure drills)",
 "R11":"roadmap milestone 0.7 item R11 (retention by record class)", "R14":"roadmap milestone 0.7 item R14 (structured incident report)",
 "R13":"roadmap milestone 0.7 item R13 (incident evidence)",
 "A0":"roadmap milestone 0.8 (accountable-owner assurance policy)", "A1":"roadmap milestone 0.8 item A1 (KYA manifest)",
 "A2":"roadmap milestone 0.8 item A2 (external-agent trust)", "A3":"roadmap milestone 0.8 item A3 (outcome validation)",
 "A4":"roadmap milestone 0.8 item A4 (third-party update gate)", "A5":"roadmap milestone 0.8 items A5 to A7, A10 and A11 (attestations)",
 "A9":"roadmap milestone 0.8 item A9 (software supply chain)", "A10":"roadmap milestone 0.8 item A10 (technology transfer)",
 "A12":"roadmap milestone 0.8 item A12 (supervisory conformance profiles)", "V":"vocabulary.py (Kognita 0.9.0)",
 "L":"roadmap milestone 0.6 and vocabulary.py (Kognita 0.9.0)",
}
NEWP = "0.8: NEW specialized record or property; no exact standard term identified"
def cit(*keys, ref="KOG-ROADMAP", new=True):
    parts = [f'[ pwm:cites pwmref:{ref} ; pwm:locator "{LOC[k]}" ]' for k in keys]
    if new: parts.append(f'[ pwm:cites pwmref:PWM-PRACTICE ; pwm:locator "{NEWP}" ]')
    return ", ".join(parts)

# ---------------------------------------------------------------- classes
# name, label, definition, parent, speed, classification, loc keys
C = [
("SigningKey","signing key","Public identity of an evidence-signing key: key identifier, public key reference and escrow reference. Holds no private key material.","prov:Entity","Slow","C1",["T1","R2"]),
("KeyRotation","key rotation","Record that one signing key replaced another. Verification works across rotations.","prov:Entity","Medium","C1",["R2"]),
("KeyRevocation","key revocation","Record that a signing key is revoked from a stated evidence sequence. Signatures at or after that sequence fail closed.","prov:Entity","Medium","C1",["R2"]),
("EvidenceAnchor","evidence anchor","Checkpoint of the chain head (record hash and sequence) published to a store the firm does not control, so a rewrite of the anchored prefix is detectable.","prov:Entity","Medium","C1",["T6"]),
("RecoveryBundle","recovery bundle","Backup of the evidence chain, retained content, holds, retention policies, key registry and anchors, recoverable to a sequence or time and verified before publication.","prov:Entity","Medium","C1",["R1"]),
("ExternalReviewSubmission","external review submission","Signed review by an external reviewer of an immutable range of evidence sequences, with annotations. The reviewer signature is checked against the key registry.","prov:Entity","Medium","C2",["T3"]),
("ModelProviderRegistration","model provider registration","A model provider recorded as an ICT third party: use cases that depend on it, data residency and retention, training use, sub-processors, encryption, incident notification, availability, model-change notice, audit rights, exit plan, incident history and regulatory cooperation.","prov:Entity","Slow","C1",["R3"]),
("ProviderFailover","provider failover","Governed switch to a fallback model after a connection failure, server error or model change. Allowed only to a model approved for the use case, and evidenced before the retry.","prov:Entity","Medium","C1",["R5"]),
("ModelChangeDetection","model change detection","Comparison of the model version reported by the provider with the version approved at admission. A difference the provider did not announce triggers revalidation.","prov:Entity","Medium","C1",["R7"]),
("AIDependency","AI dependency","One edge of the AI dependency map: a use case depends on a provider, cloud, framework, vector platform, orchestration layer, tool or data source. Declared in the registers, observed in evidence, or both.","prov:Entity","Slow","C1",["R8"]),
("DependencyDivergence","dependency divergence","Finding that a dependency is undeclared but observed in evidence, or declared but never observed.","prov:Entity","Medium","C1",["R8"]),
("ConcentrationMeasure","concentration measure","Share of critical use cases that rest on one provider or platform layer member, computed from the dependency map.","prov:Entity","Medium","C1",["R9"]),
("FailureDrill","failure drill","Offline simulation of one of eight fault kinds against a use case, checking the configured failure mode and that evidence stays complete.","prov:Entity","Medium","C1",["R10"]),
("RecordClassRetention","record class retention","Retention period for a regulatory record class, so an AI record never expires before the regulatory record it supports. A class may be held indefinitely.","prov:Entity","Slow","C1",["R11"]),
("IncidentReport","incident report","Report built from evidence for an incident: affected clients, controls that ran, failed or were missing, and the runtime change behind it. A redacted report exposes fixed labels and aggregate counts only.","prov:Entity","Medium","C2",["R14","R13"]),
("AssurancePolicy","assurance policy","Accountable-owner requirements for a use case: which external assertions, attestations, component approvals, transfer classification and update approvals are required, and the outcome thresholds that trigger revalidation.","prov:Entity","Slow","C1",["A0"]),
("AgentManifest","agent manifest","Signed export of an agent's identity, live authority, permissions, dependencies and intervention controls (Know Your Agent). Differences from observed authority or dependencies are findings.","prov:Entity","Medium","C1",["A1"]),
("ExternalAgentAssertion","external agent assertion","Protocol-neutral identity assertion from an external agent: issuer, subject, principal, purpose, audience, exact scopes and a bounded validity window. Not transitive by default.","prov:Entity","Medium","C2",["A2"]),
("DeploymentAttestation","deployment attestation","Independent signed boundary evidence bound to one deployment and policy: privacy, environment, monitoring, instruction or transfer. Expires.","prov:Entity","Medium","C2",["A5"]),
("ThirdPartyUpdateApproval","third-party update approval","Approval of a candidate third-party AI update with regression evidence, rollback plan and a named approver. A high-materiality update cannot become approved without them.","prov:Entity","Medium","C1",["A4"]),
("ApprovedComponent","approved component","Approved software component of an agent deployment: name, version, SBOM digest, optional container image digest and provenance.","prov:Entity","Slow","C1",["A9"]),
("ComponentAdvisory","component advisory","Vulnerability or provenance advisory that affects an approved component and can quarantine the agent.","prov:Entity","Medium","C1",["A9"]),
("TechnologyTransferClassification","technology transfer classification","Classification of an AI technology transfer: type, origin of the technology, deployment jurisdiction and the policy reference that governs it.","prov:Entity","Slow","C1",["A10"]),
("ProductionOutcome","production outcome","Authoritative downstream facts for one interaction, bound to a unique outcome identifier and source: intended and actual result, client impact, adverse flag. Adverse outcomes beyond the configured threshold trigger revalidation and autonomy step-down.","pwm:InteractionOutcome","Medium","C2",["A3"]),
("SupervisoryConformanceReport","supervisory conformance report","Point-in-time report from evidence and registers against a versioned supervisory profile. A profile is a mapping, not a legal opinion.","prov:Entity","Medium","C1",["A12"]),
("Verifier","verifier","Registered agent or service, with its own identity, that checks another agent's claims. Its model, model version and provider are pinned so independence from the producer can be checked.","prov:Entity","Slow","C1",["C7"]),
("ClaimVerification","claim verification","One verification of a claim by a registered verifier, with the producer's and the verifier's model pinned. The verifier must differ from the producer in model, model version and provider unless the use case permits otherwise.","prov:Entity","Medium","C1",["C7"]),
("EvaluationResult","evaluation result","Recorded evaluation of an agent version on a model version at the model, agent or system level, with metric values and the deployment content hash that identifies the evaluated behavior.","prov:Entity","Medium","C1",["C11"]),
("EvaluationMetricValue","evaluation metric value","One named metric in an evaluation result, such as accuracy, groundedness, hallucination rate or escalation correctness.","prov:Entity","Medium","C1",["C11"]),
("FairnessReport","fairness report","Recommendation, contact and prioritization rates across firm-defined segments, with the disparity threshold, named proxy attributes and a flag to the risk owner.","prov:Entity","Medium","C2",["C11"]),
("EvaluationGate","evaluation gate","Threshold per use case and metric that an evaluation result must meet before a tier 2 or higher agent is activated or changed: a minimum, or for rates such as hallucination a maximum.","prov:Entity","Slow","C1",["C11G"]),
("ReasoningTrace","reasoning trace","Record that a provider returned a reasoning trace for a model call: its hash and where the content is retained. The model's own account, never a basis for a decision.","prov:Entity","Medium","C2",["C13"]),
("LearningExport","learning export","Governed export of interactions, RM edits or outcomes to a training, fine-tuning or evaluation pipeline: purpose, anonymization, validation, approval and requester recorded. Never the default.","prov:Entity","Medium","C2",["C9"]),
("LearningExclusion","learning exclusion","Exclusion of a client or record from every future learning export. The exclusion holds permanently.","prov:Entity","Medium","C2",["C9"]),
("SensitiveAttributeInference","sensitive attribute inference","Special-category inference (religion, health, political views, ethnicity, sexuality or personal vulnerability) labeled by the classifier. Blocked from relationship workflows unless the use case permits. A vulnerability flag goes to RM review; an AI may flag but never determine.","prov:Entity","Medium","C3",["C10"]),
]
# ---------------------------------------------------------------- properties
# name, kind(o/d/c), label, definition, domain|None, range, loc keys, scheme
P = [
("signature","d","signature","Ed25519 signature over the evidence record, base64 encoded.","pwm:EvidenceRecord","xsd:string",["T1"]),
("signedBy","o","signed by","Signing key whose signature covers this record. Verification checks the key against the key registry.",None,"pwm:SigningKey",["T1","R2"]),
("evidenceSequence","d","evidence sequence","Position of the record in its chain, starting at zero without gaps.","pwm:EvidenceRecord","xsd:nonNegativeInteger",["T1"]),
("schemaVersion","d","schema version","Version of the evidence event schema the record was written under. Older events stay readable and verify.","pwm:EvidenceRecord","xsd:string",["T5"]),
("keyId","d","key identifier","Identifier derived from the public key.","pwm:SigningKey","xsd:string",["T1"]),
("publicKeyReference","d","public key reference","Location or fingerprint of the public key.","pwm:SigningKey","xsd:string",["T1"]),
("escrowReference","d","escrow reference","Reference to the escrowed recovery material for a key, never the material itself.","pwm:SigningKey","xsd:string",["R2"]),
("rotatesFrom","o","rotates from","Key that is retired by a rotation.","pwm:KeyRotation","pwm:SigningKey",["R2"]),
("rotatesTo","o","rotates to","Key that takes over in a rotation.","pwm:KeyRotation","pwm:SigningKey",["R2"]),
("revokesKey","o","revokes key","Key that a revocation withdraws.","pwm:KeyRevocation","pwm:SigningKey",["R2"]),
("effectiveFromSequence","d","effective from sequence","First evidence sequence at which a revocation applies.","pwm:KeyRevocation","xsd:nonNegativeInteger",["R2"]),
("anchoredRecordHash","d","anchored record hash","Hash of the chain head that the anchor publishes.","pwm:EvidenceAnchor","xsd:string",["T6"]),
("anchoredSequence","d","anchored sequence","Sequence of the chain head that the anchor publishes.","pwm:EvidenceAnchor","xsd:nonNegativeInteger",["T6"]),
("anchorLocation","d","anchor location","Where the checkpoint is kept, in storage the firm does not control.","pwm:EvidenceAnchor","xsd:string",["T6"]),
("bundleHash","d","bundle hash","Hash of the recovery bundle.","pwm:RecoveryBundle","xsd:string",["R1"]),
("recoveryCutoffSequence","d","recovery cutoff sequence","Evidence sequence to which the bundle restores.","pwm:RecoveryBundle","xsd:nonNegativeInteger",["R1"]),
("restoreVerified","d","restore verified","Whether a restore verified the chain and retention store end to end.","pwm:RecoveryBundle","xsd:boolean",["R1"]),
("includesRetentionState","d","includes retention state","Whether the bundle holds the historical retention state needed to restore to the cutoff.","pwm:RecoveryBundle","xsd:boolean",["R1"]),
("reviewedFromSequence","d","reviewed from sequence","First sequence of the reviewed period.","pwm:ExternalReviewSubmission","xsd:nonNegativeInteger",["T3"]),
("reviewedToSequence","d","reviewed to sequence","Last sequence of the reviewed period.","pwm:ExternalReviewSubmission","xsd:nonNegativeInteger",["T3"]),
("registeredProvider","o","registered provider","Model provider that the registration describes.","pwm:ModelProviderRegistration","pwm:ModelProvider",["R3"]),
("dependentUseCase","o","dependent use case","Use case that depends on a provider.","pwm:ModelProviderRegistration","pwm:AIUseCase",["R3"]),
("dataResidency","d","data residency","Where the provider processes and stores data.","pwm:ModelProviderRegistration","xsd:string",["R3"]),
("dataRetentionTerms","d","data retention terms","How long the provider retains data.","pwm:ModelProviderRegistration","xsd:string",["R3"]),
("trainingUse","d","training use","Whether the provider may use the firm's data for training.","pwm:ModelProviderRegistration","xsd:boolean",["R3"]),
("subProcessor","d","sub-processor","Sub-processor the provider relies on.","pwm:ModelProviderRegistration","xsd:string",["R3"]),
("encryptionDescription","d","encryption description","Encryption in transit and at rest.","pwm:ModelProviderRegistration","xsd:string",["R3"]),
("incidentNotificationTerms","d","incident notification terms","How and how fast the provider notifies incidents.","pwm:ModelProviderRegistration","xsd:string",["R3"]),
("availabilityCommitment","d","availability commitment","Availability the provider commits to.","pwm:ModelProviderRegistration","xsd:string",["R3"]),
("modelChangeNotice","d","model change notice","Notice the provider gives before changing a served model.","pwm:ModelProviderRegistration","xsd:string",["R3","R7"]),
("auditRights","d","audit rights","Whether the firm holds audit rights over the provider.","pwm:ModelProviderRegistration","xsd:boolean",["R3"]),
("exitPlanReference","d","exit plan reference","Reference to the exit plan for the provider.","pwm:ModelProviderRegistration","xsd:string",["R3"]),
("regulatoryCooperation","d","regulatory cooperation","Whether and how the provider cooperates with the firm's regulators.","pwm:ModelProviderRegistration","xsd:string",["R3"]),
("providerIncident","o","provider incident","Incident in the provider's history.","pwm:ModelProviderRegistration","pwm:AIIncident",["R3"]),
("failoverFrom","o","failover from","Model card the call was first routed to.","pwm:ProviderFailover","pwm:ModelCard",["R5"]),
("failoverTo","o","failover to","Fallback model card the call was routed to.","pwm:ProviderFailover","pwm:ModelCard",["R5"]),
("failoverUseCase","o","failover use case","Use case whose approved fallbacks govern the failover.","pwm:ProviderFailover","pwm:AIUseCase",["R5"]),
("failoverTrigger","c","failover trigger","Why the failover happened.","pwm:ProviderFailover","skos:Concept",["R5","R7"],"FailoverTriggerScheme"),
("approvedFallbackModel","o","approved fallback model","Model card approved as a fallback for the use case.","pwm:AIUseCase","pwm:ModelCard",["R5"]),
("approvedModelVersion","d","approved model version","Model version approved at admission.","pwm:ModelChangeDetection","xsd:string",["R7"]),
("servedModelVersion","d","served model version","Model version the provider reported on the call.","pwm:ModelChangeDetection","xsd:string",["R7"]),
("triggersRevalidation","d","triggers revalidation","Whether the detection requires revalidation of the use case.","pwm:ModelChangeDetection","xsd:boolean",["R7"]),
("detectedOnCall","o","detected on call","Model call on which the change was detected.","pwm:ModelChangeDetection","pwm:ModelCall",["R7"]),
("dependencyLayer","c","dependency layer","Layer of the dependency map that the dependency sits in.","pwm:AIDependency","skos:Concept",["R8","R9"],"DependencyLayerScheme"),
("dependencyOf","o","dependency of","Use case that has the dependency.","pwm:AIDependency","pwm:AIUseCase",["R8"]),
("dependencyTarget","d","dependency target","Name of the provider, platform, tool or data source depended on.","pwm:AIDependency","xsd:string",["R8"]),
("dependencyDeclared","d","dependency declared","Whether the registers declare the dependency.","pwm:AIDependency","xsd:boolean",["R8"]),
("dependencyObserved","d","dependency observed","Whether evidence shows the dependency was actually used.","pwm:AIDependency","xsd:boolean",["R8"]),
("divergenceKind","c","divergence kind","Whether the dependency is undeclared or unused.","pwm:DependencyDivergence","skos:Concept",["R8"],"DependencyDivergenceScheme"),
("divergenceDependency","o","divergence dependency","Dependency that diverges.","pwm:DependencyDivergence","pwm:AIDependency",["R8"]),
("concentrationLayer","c","concentration layer","Layer that the concentration is measured in.","pwm:ConcentrationMeasure","skos:Concept",["R9"],"DependencyLayerScheme"),
("concentrationTarget","d","concentration target","Provider or platform that the measure covers.","pwm:ConcentrationMeasure","xsd:string",["R9"]),
("criticalUseCaseCount","d","critical use case count","Number of critical use cases resting on the target.","pwm:ConcentrationMeasure","xsd:nonNegativeInteger",["R9"]),
("criticalUseCaseTotal","d","critical use case total","Number of critical use cases in scope.","pwm:ConcentrationMeasure","xsd:nonNegativeInteger",["R9"]),
("criticalUseCaseShare","d","critical use case share","Fraction from 0 to 1 of critical use cases resting on the target.","pwm:ConcentrationMeasure","xsd:decimal",["R9"]),
("drillKind","c","drill kind","Fault simulated by the drill.","pwm:FailureDrill","skos:Concept",["R10"],"DrillKindScheme"),
("drillUseCase","o","drill use case","Use case exercised by the drill.","pwm:FailureDrill","pwm:AIUseCase",["R10"]),
("evidenceCompleteAfterDrill","d","evidence complete after drill","Whether evidence stayed complete and the chain verified after the simulated fault.","pwm:FailureDrill","xsd:boolean",["R10"]),
("recordClass","c","record class","Regulatory record class that the retention period covers.","pwm:RecordClassRetention","skos:Concept",["R11"],"RecordClassScheme"),
("classRetentionDays","d","class retention in days","Retention period for the record class, in whole days.","pwm:RecordClassRetention","xsd:nonNegativeInteger",["R11"]),
("retainIndefinitely","d","retain indefinitely","Whether the record class is held with no expiry.","pwm:RecordClassRetention","xsd:boolean",["R11"]),
("reportsIncident","o","reports incident","Incident that the report describes.","pwm:IncidentReport","pwm:AIIncident",["R14"]),
("incidentKind","c","incident kind","Kind of operational incident recorded as evidence.","pwm:AIIncident","skos:Concept",["R13"],"IncidentKindScheme"),
("reportAffectedClients","o","report affected clients","Affected-client result that the report uses.","pwm:IncidentReport","pwm:AffectedClientResult",["R14"]),
("controlRan","d","control ran","Control that ran for the incident's interactions.","pwm:IncidentReport","xsd:string",["R14"]),
("controlFailed","d","control failed","Control that failed or was missing for the incident's interactions.","pwm:IncidentReport","xsd:string",["R14"]),
("runtimeChange","d","runtime change","Deployment or configuration change behind the incident.","pwm:IncidentReport","xsd:string",["R14"]),
("redactedReport","d","redacted report","Whether the report exposes fixed labels and aggregate counts only.","pwm:IncidentReport","xsd:boolean",["R14"]),
("assuranceUseCase","o","assurance use case","Use case that the assurance policy covers.","pwm:AssurancePolicy","pwm:AIUseCase",["A0"]),
("assuranceAudience","d","assurance audience","Audience that external assertions must name.","pwm:AssurancePolicy","xsd:string",["A0"]),
("requiresExternalAssertion","d","requires external assertion","Whether external-agent assertions are required.","pwm:AssurancePolicy","xsd:boolean",["A0","A2"]),
("requiresDeploymentAttestation","c","requires deployment attestation","Attestation kind the policy requires for the deployment.","pwm:AssurancePolicy","skos:Concept",["A0","A5"],"AttestationKindScheme"),
("requiresComponentApproval","d","requires component approval","Whether approved components are required.","pwm:AssurancePolicy","xsd:boolean",["A0","A9"]),
("requiresTransferClassification","d","requires transfer classification","Whether a technology transfer classification is required.","pwm:AssurancePolicy","xsd:boolean",["A0","A10"]),
("requiresUpdateApproval","d","requires update approval","Whether third-party updates need approval.","pwm:AssurancePolicy","xsd:boolean",["A0","A4"]),
("outcomeStepDown","d","outcome step-down","Whether adverse outcomes beyond the threshold step autonomy down.","pwm:AssurancePolicy","xsd:boolean",["A0","A3"]),
("maxAdverseOutcomes","d","maximum adverse outcomes","Adverse outcomes allowed in the window before revalidation.","pwm:AssurancePolicy","xsd:positiveInteger",["A0","A3"]),
("outcomeWindowDays","d","outcome window days","Length of the adverse-outcome window in days, 1 to 3650.","pwm:AssurancePolicy","xsd:positiveInteger",["A0","A3"]),
("manifestSchemaVersion","d","manifest schema version","Version of the manifest schema.","pwm:AgentManifest","xsd:string",["A1"]),
("manifestOfAgent","o","manifest of agent","Agent that the manifest describes.","pwm:AgentManifest","pwm:AIAgent",["A1"]),
("manifestDigest","d","manifest digest","Digest of the manifest document that the signature covers.","pwm:AgentManifest","xsd:string",["A1"]),
("assertionIssuer","d","assertion issuer","Issuer of the assertion.","pwm:ExternalAgentAssertion","xsd:string",["A2"]),
("assertionSubject","d","assertion subject","Agent that the assertion identifies.","pwm:ExternalAgentAssertion","xsd:string",["A2"]),
("assertionPrincipal","o","assertion principal","Person on whose behalf the external agent acts.","pwm:ExternalAgentAssertion","fibo-fnd-aap-ppl:Person",["A2"]),
("assertionPurpose","d","assertion purpose","Purpose that the assertion is limited to.","pwm:ExternalAgentAssertion","xsd:string",["A2"]),
("assertionAudience","d","assertion audience","Audience that the assertion is addressed to.","pwm:ExternalAgentAssertion","xsd:string",["A2"]),
("assertionScope","d","assertion scope","Exact scope granted, as key=value. A wildcard is never valid.","pwm:ExternalAgentAssertion","xsd:string",["A2"]),
("assertionDepth","d","assertion depth","Delegation depth, 0 to 16. Zero means no parent.","pwm:ExternalAgentAssertion","xsd:nonNegativeInteger",["A2"]),
("parentAssertion","o","parent assertion","Assertion that this one narrows.","pwm:ExternalAgentAssertion","pwm:ExternalAgentAssertion",["A2"]),
("issuedAt","d","issued at","When a credential or assertion was issued.",None,"xsd:dateTime",["A2","A5"]),
("expiresAt","d","expires at","When a credential or assertion stops being valid.",None,"xsd:dateTime",["A2","A5"]),
("attestationKind","c","attestation kind","Which boundary the attestation covers.","pwm:DeploymentAttestation","skos:Concept",["A5"],"AttestationKindScheme"),
("attestationIssuer","d","attestation issuer","Independent issuer of the attestation.","pwm:DeploymentAttestation","xsd:string",["A5"]),
("attestationSubject","d","attestation subject","Subject that the attestation is about.","pwm:DeploymentAttestation","xsd:string",["A5"]),
("attestsDeployment","d","attests deployment","Deployment digest that the attestation is bound to.","pwm:DeploymentAttestation","xsd:string",["A5"]),
("attestsPolicy","d","attests policy","Policy digest that the attestation is bound to.","pwm:DeploymentAttestation","xsd:string",["A5"]),
("attestationReference","d","attestation reference","Reference to the independent evidence behind the attestation.","pwm:DeploymentAttestation","xsd:string",["A5"]),
("candidateDigest","d","candidate digest","Digest of the candidate third-party update.","pwm:ThirdPartyUpdateApproval","xsd:string",["A4"]),
("regressionReference","d","regression reference","Reference to the regression evidence for the candidate.","pwm:ThirdPartyUpdateApproval","xsd:string",["A4"]),
("rollbackReference","d","rollback reference","Reference to the rollback plan.","pwm:ThirdPartyUpdateApproval","xsd:string",["A4"]),
("providerChange","d","provider change","Description of the change the provider made.","pwm:ThirdPartyUpdateApproval","xsd:string",["A4"]),
("componentName","d","component name","Name of the software component.","pwm:ApprovedComponent","xsd:string",["A9"]),
("componentVersion","d","component version","Version of the software component.","pwm:ApprovedComponent","xsd:string",["A9"]),
("sbomDigest","d","SBOM digest","SHA-256 of the software bill of materials, 64 lowercase hexadecimal characters.","pwm:ApprovedComponent","xsd:string",["A9"]),
("imageDigest","d","image digest","Container image digest, sha256: and 64 lowercase hexadecimal characters.","pwm:ApprovedComponent","xsd:string",["A9"]),
("provenanceReference","d","provenance reference","Reference to the provenance attestation.","pwm:ApprovedComponent","xsd:string",["A9"]),
("advisoryReference","d","advisory reference","Reference to the advisory.","pwm:ComponentAdvisory","xsd:string",["A9"]),
("affectsComponent","o","affects component","Approved component that the advisory affects.","pwm:ComponentAdvisory","pwm:ApprovedComponent",["A9"]),
("quarantined","d","quarantined","Whether the advisory quarantined the agent.","pwm:ComponentAdvisory","xsd:boolean",["A9"]),
("transferType","c","transfer type","Type of AI technology transfer.","pwm:TechnologyTransferClassification","skos:Concept",["A10"],"TransferTypeScheme"),
("technologyOrigin","d","technology origin","Jurisdiction of origin of the technology.","pwm:TechnologyTransferClassification","xsd:string",["A10"]),
("deploymentJurisdiction","d","deployment jurisdiction","Jurisdiction where the technology is deployed.","pwm:TechnologyTransferClassification","xsd:string",["A10"]),
("transferPolicyReference","d","transfer policy reference","Policy that governs the transfer.","pwm:TechnologyTransferClassification","xsd:string",["A10"]),
("intendedResult","d","intended result","What the interaction was meant to achieve.","pwm:ProductionOutcome","xsd:string",["A3"]),
("actualResult","d","actual result","What happened downstream.","pwm:ProductionOutcome","xsd:string",["A3"]),
("adverse","d","adverse","Whether the outcome counts as adverse.","pwm:ProductionOutcome","xsd:boolean",["A3"]),
("outcomeSourceReference","d","outcome source reference","System of record that reports the outcome.","pwm:ProductionOutcome","xsd:string",["A3"]),
("outcomeId","d","outcome identifier","Unique outcome identifier; a duplicate source is refused.","pwm:ProductionOutcome","xsd:string",["A3"]),
("conformanceProfile","c","conformance profile","Supervisory profile the report is built against.","pwm:SupervisoryConformanceReport","skos:Concept",["A12"],"ConformanceProfileScheme"),
("reportAsOf","d","report as of","Point in time that the report describes.","pwm:SupervisoryConformanceReport","xsd:dateTime",["A12"]),
("coversControl","d","covers control","Roadmap control code the report covers, A1 to A12.","pwm:SupervisoryConformanceReport","xsd:string",["A12"]),
("referencesEvidence","o","references evidence","Evidence record that supports the report.","pwm:SupervisoryConformanceReport","pwm:EvidenceRecord",["A12"]),
("verifierAgent","o","verifier agent","Agent identity of the verifier.","pwm:Verifier","pwm:AIAgent",["C7"]),
("verifierModelCard","o","verifier model card","Model card of the model that the verifier runs on. Its model version and provider are the verifier's pinned identity.","pwm:Verifier","pwm:ModelCard",["C7"]),
("verifiedClaim","o","verified claim","Claim that was verified.","pwm:ClaimVerification","pwm:Claim",["C7"]),
("verifiedBy","o","verified by","Registered verifier that checked the claim.","pwm:ClaimVerification","pwm:Verifier",["C7"]),
("producerModelCard","o","producer model card","Model card of the model that produced the claim.","pwm:ClaimVerification","pwm:ModelCard",["C7"]),
("verificationVerdict","c","verification verdict","Validator status that the verifier assigned to the claim.","pwm:ClaimVerification","skos:Concept",["C7"],"ValidatorStatusScheme"),
("permitsSelfVerification","d","permits self verification","Whether the use case explicitly permits a verifier that shares the producer's model, version or provider. Only at tier 1.","pwm:AIUseCase","xsd:boolean",["C7"]),
("evaluatesAgent","o","evaluates agent","Agent that the result evaluates.","pwm:EvaluationResult","pwm:AIAgent",["C11"]),
("evaluatedVersion","d","evaluated version","Agent version that was evaluated.","pwm:EvaluationResult","xsd:string",["C11"]),
("evaluatedModelVersion","d","evaluated model version","Model version that was evaluated.","pwm:EvaluationResult","xsd:string",["C11"]),
("evaluationLevel","c","evaluation level","Whether the model, the agent or the end-to-end system was evaluated.","pwm:EvaluationResult","skos:Concept",["C11"],"EvaluationLevelScheme"),
("evaluationDeploymentHash","d","evaluation deployment hash","SHA-256 of the deployment content that was evaluated, 64 lowercase hexadecimal characters.","pwm:EvaluationResult","xsd:string",["C11"]),
("hasMetricValue","o","has metric value","Metric value recorded in the evaluation result.","pwm:EvaluationResult","pwm:EvaluationMetricValue",["C11"]),
("metricName","d","metric name","Name of the evaluation metric.","pwm:EvaluationMetricValue","xsd:string",["C11"]),
("metricScore","d","metric score","Metric value as a fraction from 0 to 1.","pwm:EvaluationMetricValue","xsd:decimal",["C11"]),
("fairnessThreshold","d","fairness threshold","Disparity above which the report flags a segment to the risk owner.","pwm:FairnessReport","xsd:decimal",["C11"]),
("proxyAttribute","d","proxy attribute","Attribute that stands in for, or correlates with, a protected attribute.","pwm:FairnessReport","xsd:string",["C11"]),
("fairnessFlagged","d","fairness flagged","Whether a disparity above the threshold was flagged to the risk owner.","pwm:FairnessReport","xsd:boolean",["C11"]),
("hasDocumentation","o","has documentation","Documentation held for an AI use case, such as a model card or design record (MAS AIRG 4.7).","pwm:AIUseCase","prov:Entity",["INV"]),
("hasComponent","o","has component","Approved software component that an agent deployment is built from (MAS AIRG fn 25).","pwm:AIAgent","pwm:ApprovedComponent",["INV"]),
("gateUseCase","o","gate use case","Use case that the gate applies to.","pwm:EvaluationGate","pwm:AIUseCase",["C11G"]),
("gateMetricName","d","gate metric name","Name of the metric that the gate checks.","pwm:EvaluationGate","xsd:string",["C11G"]),
("gateMinimumScore","d","gate minimum score","Lowest acceptable score, from 0 to 1.","pwm:EvaluationGate","xsd:decimal",["C11G"]),
("gateMaximumScore","d","gate maximum score","Highest acceptable score, from 0 to 1, for rates where lower is better.","pwm:EvaluationGate","xsd:decimal",["C11G"]),
("activatedOnEvaluation","o","activated on evaluation","Evaluation result that the agent's activation or last change relied on.","pwm:AIAgent","pwm:EvaluationResult",["C11G"]),
("drillScenario","c","drill scenario","Hostile-agent scenario exercised by the drill.","pwm:FailureDrill","skos:Concept",["A8"],"DrillScenarioScheme"),
("traceOfCall","o","trace of call","Model call whose reasoning trace this is.","pwm:ReasoningTrace","pwm:ModelCall",["C13"]),
("traceHash","d","trace hash","Hash of the reasoning trace in the evidence chain; the content sits in the retention store.","pwm:ReasoningTrace","xsd:string",["C13"]),
("requiresReasoningTrace","d","requires reasoning trace","Whether the use case requires a reasoning trace on material output.","pwm:AIUseCase","xsd:boolean",["C13"]),
("claimSourceSystem","d","claim source system","System of record that the claim's source belongs to.","pwm:Claim","xsd:string",["C1"]),
("claimSourceRecord","d","claim source record","Record in the source system that the claim relies on.","pwm:Claim","xsd:string",["C1"]),
("claimSourceTimestamp","d","claim source timestamp","As-of time of the source record.","pwm:Claim","xsd:dateTime",["C1"]),
("claimRetrievalTimestamp","d","claim retrieval timestamp","When the source record was retrieved.","pwm:Claim","xsd:dateTime",["C1"]),
("calculationMethod","d","calculation method","Registered calculation service or method that produced a figure in the claim.","pwm:Claim","xsd:string",["C1"]),
("claimPromptVersion","d","claim prompt version","Prompt version that produced the claim.","pwm:Claim","xsd:string",["C1"]),
("claimConfidence","d","claim confidence","Confidence from 0 to 1 that the producer assigned to the claim.","pwm:Claim","xsd:decimal",["C1"]),
("claimPolicyStatus","d","claim policy status","Whether policy allows the claim to reach an RM or client.","pwm:Claim","xsd:string",["C1"]),
("memorySubjectClient","o","memory subject client","Client that a relationship memory record is about. Institutional memory names no client.","pwm:MemoryRecord","pwm:Client",["C8"]),
("memoryExpiresAt","d","memory expires at","When the memory record expires.","pwm:MemoryRecord","xsd:dateTime",["C8"]),
("memoryCorrectionPath","d","memory correction path","How a person can correct or remove the memory record.","pwm:MemoryRecord","xsd:string",["C8"]),
("exportPurpose","d","export purpose","Purpose of the learning export.","pwm:LearningExport","xsd:string",["C9"]),
("anonymizationReference","d","anonymization reference","Reference to the anonymization applied before export.","pwm:LearningExport","xsd:string",["C9"]),
("validationReference","d","validation reference","Reference to the validation report that cleared the export.","pwm:LearningExport","xsd:string",["C9"]),
("exportRequester","o","export requester","Person who requested the export.","pwm:LearningExport","fibo-fnd-aap-ppl:Person",["C9"]),
("exportedRecordId","d","exported record identifier","Identifier of a record included in the export.","pwm:LearningExport","xsd:string",["C9"]),
("exclusionReason","d","exclusion reason","Why the client or record is excluded from learning exports.","pwm:LearningExclusion","xsd:string",["C9"]),
("excludesClient","o","excludes client","Client excluded from every future learning export.","pwm:LearningExclusion","pwm:Client",["C9"]),
("excludesRecordId","d","excludes record identifier","Record excluded from every future learning export.","pwm:LearningExclusion","xsd:string",["C9"]),
("sensitiveKind","c","sensitive kind","Special category that the inference concerns.","pwm:SensitiveAttributeInference","skos:Concept",["C10"],"SensitiveKindScheme"),
("permittedByUseCase","d","permitted by use case","Whether the use case explicitly permits the special-category inference.","pwm:SensitiveAttributeInference","xsd:boolean",["C10"]),
("blockedByPolicy","d","blocked by policy","Whether policy blocked the inference from the relationship workflow.","pwm:SensitiveAttributeInference","xsd:boolean",["C10"]),
("changesClientTreatment","d","changes client treatment","Whether the inference changes how the client is treated. Requires a person's determination.","pwm:SensitiveAttributeInference","xsd:boolean",["C10"]),
("reviewedByRM","o","reviewed by RM","Person who reviewed the flag and made any determination.","pwm:SensitiveAttributeInference","fibo-fnd-aap-ppl:Person",["C10"]),
# code-list properties on existing classes
("actorType","c","actor type","Whether a human, an agent or the system produced the event.","pwm:EvidenceRecord","skos:Concept",["V"],"ActorTypeScheme"),
("approvalRequestStatus","c","approval request status","State of an approval request, including the first signature of a two-signature approval.","pwm:ApprovalRequest","skos:Concept",["V"],"ApprovalStatusScheme"),
("egressDecision","c","egress decision","Whether data left for a provider unchanged, redacted or not at all.","pwm:EvidenceRecord","skos:Concept",["V"],"EgressDecisionScheme"),
("checkResult","c","check result","Verdict of one policy check.","pwm:PolicyCheck","skos:Concept",["V"],"CheckResultScheme"),
]
# ---------------------------------------------------------------- schemes: name, label, scope, loc, [(local, notation, label, definition)]
S = [
("ActorTypeScheme","actor types","vocabulary.ActorType",[("ActorHuman","HUMAN","human","A person."),("ActorAgent","AGENT","agent","An AI agent."),("ActorSystem","SYSTEM","system","Kognita or another system component.")]),
("ApprovalStatusScheme","approval request statuses","vocabulary.ApprovalStatus",[("ApprovalPending","PENDING","pending","Awaiting a decision."),("ApprovalMarked","MARKED","marked","First signature recorded in a two-signature approval."),("ApprovalStatusApproved","APPROVED","approved","Approved."),("ApprovalRejected","REJECTED","rejected","Rejected."),("ApprovalExpired","EXPIRED","expired","Expired without a decision.")]),
("EgressDecisionScheme","egress decisions","vocabulary.EgressDecision",[("EgressAllow","ALLOW","allow","Data left unchanged."),("EgressRedact","REDACT","redact","Data left after redaction."),("EgressDeny","DENY","deny","Data did not leave.")]),
("CheckResultScheme","check results","vocabulary.CheckResult",[("CheckPass","pass","pass","The check passed."),("CheckFail","fail","fail","The check failed."),("CheckEscalate","escalate","escalate","The check escalates to a person."),("CheckRequiresHuman","requires_human","requires human","A human decision is required.")]),
("FailoverTriggerScheme","failover triggers","roadmap milestone 0.7 items R5 and R7",[("TriggerConnectionFailure","connection_failure","connection failure","The provider connection failed."),("TriggerHttp5xx","http_5xx","HTTP 5xx","The provider returned a server error."),("TriggerModelChanged","model_changed","model changed","The served model differs from the approved model.")]),
("DependencyLayerScheme","dependency layers","roadmap milestone 0.7 items R8 and R9",[("LayerModelProvider","model_provider","model provider","Model provider."),("LayerCloud","cloud","cloud","Cloud platform."),("LayerFramework","framework","framework","Agent framework."),("LayerVectorPlatform","vector_platform","vector platform","Vector store or retrieval platform."),("LayerOrchestration","orchestration","orchestration","Orchestration layer."),("LayerTool","tool","tool","Tool or API."),("LayerDataSource","data_source","data source","Data source.")]),
("DependencyDivergenceScheme","dependency divergences","roadmap milestone 0.7 item R8",[("DivergenceUndeclared","R8-UNDECLARED","undeclared","Observed in evidence but not declared in the registers."),("DivergenceUnused","R8-UNUSED","unused","Declared in the registers but never observed.")]),
("DrillKindScheme","drill kinds","breaker.DrillKind",[("DrillModelUnavailable","model_unavailable","model unavailable","The model is unavailable."),("DrillVendorOutage","vendor_outage","vendor outage","The vendor is down."),("DrillApiOutage","api_outage","API outage","An API is down."),("DrillRetrievalFailure","retrieval_failure","retrieval failure","Retrieval fails."),("DrillIdentityServiceFailure","identity_service_failure","identity service failure","The identity service fails."),("DrillCorruptedContext","corrupted_context","corrupted context","Context is corrupted."),("DrillCompromisedModel","compromised_model","compromised model","The model is compromised."),("DrillToolFailure","tool_failure","tool failure","A tool fails.")]),
("RecordClassScheme","regulatory record classes","retention._RECORD_CLASSES",[("RecordClientCommunications","client communications","client communications","Client communications."),("RecordAdvice","advice","advice","Advice."),("RecordRecommendations","recommendations","recommendations","Recommendations."),("RecordApprovals","approvals","approvals","Approvals."),("RecordTransactions","transactions","transactions","Transactions."),("RecordComplianceRecords","compliance records","compliance records","Compliance records.")]),
("IncidentKindScheme","operational incident kinds","incidents.IncidentKind",[("IncidentChainBreak","chain_break","chain break","The evidence chain failed verification."),("IncidentGatewayOutage","gateway_outage","gateway outage","The gateway was unavailable."),("IncidentDegradedMode","degraded_mode","degraded mode","The gateway ran in degraded mode."),("IncidentProviderFailure","provider_failure","provider failure","A model provider failed.")]),
("AttestationKindScheme","attestation kinds","assurance.trust.Attestation",[("AttestationPrivacy","privacy","privacy","Privacy impact context."),("AttestationEnvironment","environment","environment","Execution environment."),("AttestationMonitoring","monitoring","monitoring","Monitoring coverage."),("AttestationInstruction","instruction","instruction","High-risk instruction authenticity."),("AttestationTransfer","transfer","transfer","Jurisdictional technology transfer.")]),
("TransferTypeScheme","technology transfer types","assurance.records.AssurancePolicy.transfer_type",[("TransferApi","api","API/SaaS","Access through an API or hosted service."),("TransferWeights","weights","model weights","Transfer of model weights."),("TransferLicense","license","license","License transfer. The Kognita value keeps its original spelling, license."),("TransferLocal","local","local deployment","Local deployment."),("TransferOwnership","ownership","ownership or control","Transfer of ownership or control.")]),
("ConformanceProfileScheme","supervisory conformance profiles","assurance.reports.PROFILES",[("ProfileMAS","MAS","MAS","Monetary Authority of Singapore."),("ProfileHKMAART","HKMA-ART","HKMA ART","Hong Kong Monetary Authority."),("ProfileFSB","FSB","FSB","Financial Stability Board sound practices."),("ProfileOCC","OCC","OCC","US Office of the Comptroller of the Currency examiner considerations."),("ProfileSEC","SEC","SEC","Selected US Securities and Exchange Commission themes."),("ProfileFCA","FCA","FCA","Selected UK Financial Conduct Authority themes.")]),
]
S.append(("SensitiveKindScheme","special categories","sensitive._PATTERNS",[("SensitiveReligion","religion","religion","Religion."),("SensitiveHealth","health","health","Health."),("SensitivePolitical","political_views","political views","Political views."),("SensitiveEthnicity","ethnicity","ethnicity","Ethnicity."),("SensitiveSexuality","sexuality","sexuality","Sexuality."),("SensitiveVulnerability","vulnerability","personal vulnerability","Personal vulnerability.")]))
S.append(("DrillScenarioScheme","hostile-agent drill scenarios","assurance.drills.SCENARIOS",[("ScenarioPromptInjection","prompt-injection-tool-escalation","prompt injection leading to tool escalation","Injected instructions try to escalate tool use."),("ScenarioDiscoveredCredentials","discovered-credentials","discovered third-party credentials","The agent finds credentials it was not given."),("ScenarioUnauthorizedNetwork","unauthorized-network","unauthorized external network access","The agent reaches for a network destination outside its allow-list."),("ScenarioMachineSpeedProbing","machine-speed-probing","machine-speed probing","High-rate API probing and multi-technique attack."),("ScenarioCompromisedProvider","compromised-provider","compromised provider","The model provider is compromised."),("ScenarioAgentPrivilegeEscalation","agent-privilege-escalation","agent-to-agent privilege escalation","One agent tries to gain another's privileges."),("ScenarioMaliciousRetrieval","malicious-retrieval","malicious retrieved content","Retrieved content carries hostile instructions."),("ScenarioContainmentRecovery","containment-recovery","emergency containment and recovery","Containment and recovery after a breach.")]))
S.append(("EvaluationLevelScheme","evaluation levels","evaluation.py and cli evaluation record --level",[("LevelModel","model","model","The model alone."),("LevelAgent","agent","agent","The agent: planning, tool selection, permission compliance, delegation, retries and escalation."),("LevelSystem","system","system","The end-to-end system: model, retrieval, tools, data, agent, policies, interface and human interaction.")]))
EVENTS = [("EventAnchorPublished","ANCHOR_PUBLISHED","anchor published","A chain-head checkpoint was published to an external sink."),
("EventAssurance","ASSURANCE","assurance","A runtime assurance record was evidenced: manifest, attestation, update approval, component approval, drill or production outcome."),
("EventClaimVerified","CLAIM_VERIFIED","claim verified","A claim passed the verification gate."),
("EventEvaluationRecorded","EVALUATION_RECORDED","evaluation recorded","An evaluation result was recorded."),
("EventExternalReviewSubmitted","EXTERNAL_REVIEW_SUBMITTED","external review submitted","An external reviewer submitted a signed review."),
("EventIncidentUpdated","INCIDENT_UPDATED","incident updated","An incident was updated."),
("EventIncidentClosed","INCIDENT_CLOSED","incident closed","An incident was closed."),
("EventInteraction","INTERACTION","interaction","A client interaction was recorded."),
("EventKeyRevoked","KEY_REVOKED","key revoked","A signing key was revoked."),
("EventKeyRotated","KEY_ROTATED","key rotated","A signing key was rotated."),
("EventLearningExport","LEARNING_EXPORT","learning export","An approved export of interaction data for learning or evaluation was made."),
("EventMemoryWrite","MEMORY_WRITE","memory write","A memory record was written."),
("EventRevalidationTriggered","REVALIDATION_TRIGGERED","revalidation triggered","A model change or other trigger required revalidation."),
("EventReview","REVIEW","review","A human review was recorded."),
("EventSourceSuperseded","SOURCE_SUPERSEDED","source superseded","An approved source was superseded, so facts that depend on it are no longer current.")]

def ttl_module():
    o=["# =============================================================================",
       "#  PWM 0.8: Trust, resilience and assurance (module 12)",
       "#",
       "#  Signed evidence and keys, anchors, recovery, the model-provider register,",
       "#  failover and model change, the dependency map, failure drills, retention by",
       "#  record class, incident reports, and runtime assurance records. Follows",
       "#  Kognita milestones 0.7 (Trust and Resilience) and 0.8 (Supervisory Alignment",
       "#  and Runtime Assurance), and fills the vocabulary gaps found in the 0.5 review.",
       "#  Generated from tools/gen_module12.py; edit that script, not this file.",
       "# =============================================================================","",
       'pwm:ModuleTrust a pwm:Module ; rdfs:label "Trust, resilience and assurance"@en ; pwm:moduleOrder 12 ;',
       '    skos:definition "Signed and anchored evidence, provider and dependency registers, failover and drills, retention by record class, incident reports and runtime assurance records."@en .',""]
    for n,l,d,par,sp,cl,locs in C:
        o.append(f'pwm:{n} a owl:Class ; rdfs:label "{l}"@en ;\n    skos:definition "{d}"@en ; rdfs:subClassOf {par} ;\n    pwm:inModule pwm:ModuleTrust ; pwm:contextSpeed pwm:{sp}Context ; pwm:defaultClassification pwm:{cl} ;\n    pwm:citation {cit(*locs)} .\n')
    for p in P:
        n,k,l,d,dom,rng,locs=p[:7]
        t = "owl:ObjectProperty" if k in "oc" else "owl:DatatypeProperty"
        s=f'pwm:{n} a {t} ; rdfs:label "{l}"@en ;\n    skos:definition "{d}"@en ;'
        if dom: s+=f'\n    rdfs:domain {dom} ;'
        s+=f'\n    rdfs:range {rng} ;\n    pwm:inModule pwm:ModuleTrust ;\n    pwm:citation {cit(*locs)} .\n'
        o.append(s)
    for sn,sl,scope,cs in S:
        loc = scope
        o.append(f'pwm:{sn} a skos:ConceptScheme ;\n    skos:prefLabel "{sl}"@en ;\n    pwm:inModule pwm:ModuleTrust ;\n    pwm:citation [ pwm:cites pwmref:KOG-VOCAB ; pwm:locator "{loc}" ] .')
        for cn,nt,lab,df in cs:
            o.append(f'pwm:{cn} a skos:Concept ; skos:notation "{nt}" ; skos:inScheme pwm:{sn} ; skos:topConceptOf pwm:{sn} ; skos:prefLabel "{lab}"@en ; skos:definition "{df}"@en .')
        o.append("")
    o.append("# Evidence event types added to match Kognita vocabulary.EventType (31 values).")
    for cn,nt,lab,df in EVENTS:
        o.append(f'pwm:{cn} a skos:Concept ; skos:notation "{nt}" ; skos:inScheme pwm:EvidenceEventTypeScheme ; skos:topConceptOf pwm:EvidenceEventTypeScheme ; skos:prefLabel "{lab}"@en ; skos:definition "{df}"@en .')
    return "\n".join(o)+"\n"

def bindings():
    return "\n".join(f"pwm:{p[0]} pwm:valuesFrom pwm:{p[7]} ." for p in P if p[1]=="c")+"\n"

# ---------------------------------------------------------------- shapes
def prop(path,mn=None,mx=None,dt=None,cls=None,extra="",label=None):
    parts=[f"sh:path pwm:{path}"]; msg=[f"pwm:{path}:"]; c=[]
    if mn is not None: parts.append(f"sh:minCount {mn}"); c.append(f"minimum count {mn}")
    if mx is not None: parts.append(f"sh:maxCount {mx}"); c.append(f"maximum count {mx}")
    if dt: parts.append(f"sh:datatype {dt}"); c.append(f"datatype {dt}")
    if cls: parts.append(f"sh:class {cls}"); c.append(f"class {cls}")
    if extra: parts.append(extra[0]); c.append(extra[1])
    return f'    sh:property [ {" ; ".join(parts)} ; sh:message "pwm:{path}: {", ".join(c)}." ]'
def node(name,target,locs,body):
    return (f'pwmsh:{name}Shape a sh:NodeShape ; rdfs:label "{name}Shape"@en ; sh:targetClass {target} ;\n'
            f'    dcterms:source {", ".join("pwmref:"+r for r in locs)} ; rdfs:comment "{locs and ""}Checks recorded evidence; runtime enforcement remains external."@en ;\n'
            + " ;\n".join(body) + " .\n")
def sparql(msg,select,comment=None):
    c=f'        rdfs:comment "{comment}"@en ;\n' if comment else ""
    return (f'    sh:sparql [\n        a sh:SPARQLConstraint ;\n        sh:message "{msg}" ;\n        sh:prefixes <https://ahmedmuzammil.com/ontology/pwm/shapes> ;\n{c}        sh:select """\n{select}\n        """ ;\n    ]')
S1=lambda n,t="xsd:string": prop(n,1,1,t)
SH=[]
def add(name,target,body,locs=("KOG-ROADMAP",)): SH.append(node(name,target,locs,body))
add("SigningKey","pwm:SigningKey",[S1("keyId"),prop("publicKeyReference",1,1,"xsd:string"),prop("escrowReference",0,1,"xsd:string")])
SH.append('pwmsh:SignedEvidenceShape a sh:NodeShape ; rdfs:label "SignedEvidenceShape"@en ; sh:targetClass pwm:EvidenceRecord ;\n    dcterms:source pwmref:KOG-ROADMAP ; rdfs:comment "Roadmap milestone 0.7 items T1 and R2. Checks recorded evidence; signature verification is external."@en ;\n'
 + prop("signature",0,1,"xsd:string")+" ;\n"+prop("signedBy",0,1,None,"pwm:SigningKey")+" ;\n"+prop("evidenceSequence",0,1,"xsd:nonNegativeInteger")+" ;\n"
 + sparql("A signed evidence record names its signing key and its sequence, so the signature can be checked against the key registry and any revocation cutoff (Kognita 0.7 items T1 and R2).",
  "            SELECT $this WHERE {\n                $this pwm:signature ?s .\n                FILTER NOT EXISTS { $this pwm:signedBy ?k . $this pwm:evidenceSequence ?n }\n            }")+" ;\n"
 + sparql("A record signed with a key at or after that key's revocation sequence is not valid (Kognita 0.7 item R2; revoked keys fail closed).",
  "            SELECT $this ?value WHERE {\n                $this pwm:signedBy ?value ; pwm:evidenceSequence ?seq .\n                ?rev pwm:revokesKey ?value ; pwm:effectiveFromSequence ?cut .\n                FILTER (?seq >= ?cut)\n            }")+" .\n")
add("KeyRotation","pwm:KeyRotation",[prop("rotatesFrom",1,1,None,"pwm:SigningKey"),prop("rotatesTo",1,1,None,"pwm:SigningKey"),prop("rotatesFrom",extra=("sh:disjoint pwm:rotatesTo","must differ from rotatesTo"))])
add("KeyRevocation","pwm:KeyRevocation",[prop("revokesKey",1,1,None,"pwm:SigningKey"),prop("effectiveFromSequence",1,1,"xsd:nonNegativeInteger")])
add("EvidenceAnchor","pwm:EvidenceAnchor",[S1("anchoredRecordHash"),prop("anchoredSequence",1,1,"xsd:nonNegativeInteger"),S1("anchorLocation"),prop("generatedAtTime",1,1,"xsd:dateTime").replace("pwm:generatedAtTime","prov:generatedAtTime").replace("sh:path prov:generatedAtTime","sh:path prov:generatedAtTime")])
add("RecoveryBundle","pwm:RecoveryBundle",[S1("bundleHash"),prop("recoveryCutoffSequence",1,1,"xsd:nonNegativeInteger"),prop("restoreVerified",1,1,"xsd:boolean"),prop("includesRetentionState",1,1,"xsd:boolean")])
add("ExternalReviewSubmission","pwm:ExternalReviewSubmission",[prop("reviewedFromSequence",1,1,"xsd:nonNegativeInteger",extra=("sh:lessThanOrEquals pwm:reviewedToSequence","not after reviewedToSequence")),prop("reviewedToSequence",1,1,"xsd:nonNegativeInteger"),prop("signedBy",1,1,None,"pwm:SigningKey")])
REG=["registeredProvider"]
add("ModelProviderRegistration","pwm:ModelProviderRegistration",
    [prop("registeredProvider",1,1,None,"pwm:ModelProvider"),prop("dependentUseCase",1,None,None,"pwm:AIUseCase")]
    +[S1(x) for x in ["dataResidency","dataRetentionTerms","encryptionDescription","incidentNotificationTerms","availabilityCommitment","modelChangeNotice","exitPlanReference","regulatoryCooperation"]]
    +[prop("trainingUse",1,1,"xsd:boolean"),prop("auditRights",1,1,"xsd:boolean"),prop("subProcessor",1,None,"xsd:string")])
SH.append('pwmsh:RegisteredProviderCallShape a sh:NodeShape ; rdfs:label "RegisteredProviderCallShape"@en ; sh:targetClass pwm:ModelCall ;\n    dcterms:source pwmref:KOG-ROADMAP ; rdfs:comment "Roadmap milestone 0.7 item R3. Routing to an unregistered provider is a denial (R3-UNREGISTERED); this checks the recorded call."@en ;\n'
 + sparql("A model call names a provider that has no model provider registration (Kognita 0.7 item R3, R3-UNREGISTERED).",
  "            SELECT $this ?value WHERE {\n                $this pwm:providedBy ?value .\n                FILTER NOT EXISTS { ?r a pwm:ModelProviderRegistration ; pwm:registeredProvider ?value }\n            }")+" .\n")
add("ProviderFailover","pwm:ProviderFailover",[prop("failoverFrom",1,1,None,"pwm:ModelCard"),prop("failoverTo",1,1,None,"pwm:ModelCard"),prop("failoverUseCase",1,1,None,"pwm:AIUseCase"),prop("failoverTrigger",1,1),
 sparql("A failover target must be a model card approved as a fallback for the use case (Kognita 0.7 item R5, R5-UNAPPROVED-FALLBACK).",
  "            SELECT $this ?value WHERE {\n                $this pwm:failoverTo ?value ; pwm:failoverUseCase ?uc .\n                FILTER NOT EXISTS { ?uc pwm:approvedFallbackModel ?value }\n            }")])
add("ModelChangeDetection","pwm:ModelChangeDetection",[S1("approvedModelVersion"),S1("servedModelVersion"),prop("triggersRevalidation",1,1,"xsd:boolean"),prop("detectedOnCall",1,1,None,"pwm:ModelCall"),
 sparql("A served model version that differs from the approved version must trigger revalidation (Kognita 0.7 item R7, R7-MODEL-CHANGED).",
  "            SELECT $this WHERE {\n                $this pwm:approvedModelVersion ?a ; pwm:servedModelVersion ?s ; pwm:triggersRevalidation ?t .\n                FILTER (?a != ?s && ?t = false)\n            }")])
add("AIDependency","pwm:AIDependency",[prop("dependencyLayer",1,1),prop("dependencyOf",1,1,None,"pwm:AIUseCase"),S1("dependencyTarget"),prop("dependencyDeclared",1,1,"xsd:boolean"),prop("dependencyObserved",1,1,"xsd:boolean")])
add("DependencyDivergence","pwm:DependencyDivergence",[prop("divergenceKind",1,1),prop("divergenceDependency",1,1,None,"pwm:AIDependency"),
 sparql("An undeclared divergence needs a dependency that is observed and not declared; an unused one needs a dependency that is declared and not observed (Kognita 0.7 item R8).",
  "            SELECT $this ?value WHERE {\n                $this pwm:divergenceDependency ?value .\n                { $this pwm:divergenceKind pwm:DivergenceUndeclared . ?value pwm:dependencyDeclared true }\n                UNION\n                { $this pwm:divergenceKind pwm:DivergenceUnused . ?value pwm:dependencyObserved true }\n            }")])
add("ConcentrationMeasure","pwm:ConcentrationMeasure",[prop("concentrationLayer",1,1),S1("concentrationTarget"),
 prop("criticalUseCaseCount",1,1,"xsd:nonNegativeInteger",extra=("sh:lessThanOrEquals pwm:criticalUseCaseTotal","not above criticalUseCaseTotal")),prop("criticalUseCaseTotal",1,1,"xsd:nonNegativeInteger"),
 "    sh:property [ sh:path pwm:criticalUseCaseShare ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minInclusive 0 ; sh:maxInclusive 1 ; sh:message \"pwm:criticalUseCaseShare: minimum count 1, maximum count 1, datatype xsd:decimal, between 0 and 1.\" ]"])
add("FailureDrill","pwm:FailureDrill",[prop("drillKind",1,1),prop("drillUseCase",1,1,None,"pwm:AIUseCase"),prop("evidenceCompleteAfterDrill",1,1,"xsd:boolean"),prop("gatewayFailureMode",1,1)])
SH.append('pwmsh:RecordClassRetentionShape a sh:NodeShape ; rdfs:label "RecordClassRetentionShape"@en ; sh:targetClass pwm:RecordClassRetention ;\n    dcterms:source pwmref:KOG-ROADMAP ; rdfs:comment "Roadmap milestone 0.7 item R11. Checks recorded evidence; runtime enforcement remains external."@en ;\n'
 + prop("recordClass",1,1)+" ;\n"
 + '    sh:xone (\n        [ sh:property [ sh:path pwm:classRetentionDays ; sh:minCount 1 ; sh:maxCount 1 ; sh:message "classRetentionDays: one value." ] ]\n        [ sh:property [ sh:path pwm:retainIndefinitely ; sh:hasValue true ; sh:message "retainIndefinitely: true." ] ]\n    ) ;\n    sh:message "A record class has a retention period in days or is held indefinitely, not both and not neither (Kognita 0.7 item R11)." .\n')
add("IncidentReport","pwm:IncidentReport",[prop("reportsIncident",1,1,None,"pwm:AIIncident"),prop("redactedReport",1,1,"xsd:boolean"),prop("controlRan",1,None,"xsd:string")])
add("AssurancePolicy","pwm:AssurancePolicy",[prop("assuranceUseCase",1,1,None,"pwm:AIUseCase"),S1("assuranceAudience"),prop("maxAdverseOutcomes",1,1,"xsd:positiveInteger"),
 "    sh:property [ sh:path pwm:outcomeWindowDays ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:positiveInteger ; sh:minInclusive 1 ; sh:maxInclusive 3650 ; sh:message \"pwm:outcomeWindowDays: minimum count 1, maximum count 1, datatype xsd:positiveInteger, between 1 and 3650.\" ]"])
add("AgentManifest","pwm:AgentManifest",[S1("manifestSchemaVersion"),prop("manifestOfAgent",1,1,None,"pwm:AIAgent"),S1("manifestDigest"),prop("signedBy",1,1,None,"pwm:SigningKey")])
add("ExternalAgentAssertion","pwm:ExternalAgentAssertion",[S1("assertionIssuer"),S1("assertionSubject"),prop("assertionPrincipal",1,1,None,"fibo-fnd-aap-ppl:Person"),S1("assertionPurpose"),S1("assertionAudience"),prop("assertionScope",1,None,"xsd:string"),
 "    sh:property [ sh:path pwm:assertionDepth ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:nonNegativeInteger ; sh:maxInclusive 16 ; sh:message \"pwm:assertionDepth: minimum count 1, maximum count 1, at most 16.\" ]",
 prop("issuedAt",1,1,"xsd:dateTime",extra=("sh:lessThan pwm:expiresAt","before expiresAt")),prop("expiresAt",1,1,"xsd:dateTime"),
 sparql("An assertion has a parent exactly when its delegation depth is above zero (Kognita 0.8 item A2).",
  "            SELECT $this WHERE {\n                { $this pwm:assertionDepth ?d . FILTER (?d > 0) FILTER NOT EXISTS { $this pwm:parentAssertion ?p } }\n                UNION\n                { $this pwm:parentAssertion ?p ; pwm:assertionDepth ?d . FILTER (?d = 0) }\n            }")])
add("DeploymentAttestation","pwm:DeploymentAttestation",[prop("attestationKind",1,1),S1("attestationIssuer"),S1("attestationSubject"),S1("attestsDeployment"),S1("attestsPolicy"),S1("attestationReference"),
 prop("issuedAt",1,1,"xsd:dateTime",extra=("sh:lessThan pwm:expiresAt","before expiresAt")),prop("expiresAt",1,1,"xsd:dateTime")])
add("ThirdPartyUpdateApproval","pwm:ThirdPartyUpdateApproval",[S1("candidateDigest"),S1("regressionReference"),S1("rollbackReference"),prop("approvedBy",1,1)])
add("ApprovedComponent","pwm:ApprovedComponent",[S1("componentName"),S1("componentVersion"),S1("provenanceReference"),
 "    sh:property [ sh:path pwm:sbomDigest ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string ; sh:pattern \"^[a-f0-9]{64}$\" ; sh:message \"pwm:sbomDigest: 64 lowercase hexadecimal characters.\" ]",
 "    sh:property [ sh:path pwm:imageDigest ; sh:maxCount 1 ; sh:datatype xsd:string ; sh:pattern \"^sha256:[a-f0-9]{64}$\" ; sh:message \"pwm:imageDigest: sha256: followed by 64 lowercase hexadecimal characters.\" ]"])
add("ComponentAdvisory","pwm:ComponentAdvisory",[S1("advisoryReference"),prop("affectsComponent",1,1,None,"pwm:ApprovedComponent"),prop("quarantined",1,1,"xsd:boolean")])
add("TechnologyTransferClassification","pwm:TechnologyTransferClassification",[prop("transferType",1,1),S1("technologyOrigin"),S1("deploymentJurisdiction"),S1("transferPolicyReference")])
add("ProductionOutcome","pwm:ProductionOutcome",[S1("outcomeId"),S1("outcomeSourceReference"),S1("intendedResult"),S1("actualResult"),prop("adverse",1,1,"xsd:boolean")])
add("SupervisoryConformanceReport","pwm:SupervisoryConformanceReport",[prop("conformanceProfile",1,1),prop("reportAsOf",1,1,"xsd:dateTime"),prop("coversControl",1,None,"xsd:string")])
add("Verifier","pwm:Verifier",[prop("verifierAgent",1,1,None,"pwm:AIAgent"),prop("verifierModelCard",1,1,None,"pwm:ModelCard")])
add("ClaimVerification","pwm:ClaimVerification",[prop("verifiedClaim",1,1,None,"pwm:Claim"),prop("verifiedBy",1,1,None,"pwm:Verifier"),prop("producerModelCard",1,1,None,"pwm:ModelCard"),prop("verificationVerdict",1,1),
 sparql("A verifier must differ from the producer in model, model version and provider unless the use case explicitly permits self verification (Kognita 0.6 item 7, C7).",
  "            SELECT $this WHERE {\n                $this pwm:verifiedBy/pwm:verifierModelCard ?vc ; pwm:producerModelCard ?pc .\n                FILTER ( ?vc = ?pc || EXISTS { ?vc pwm:providedBy ?p . ?pc pwm:providedBy ?p } || EXISTS { ?vc pwm:modelVersion ?v . ?pc pwm:modelVersion ?v } )\n                FILTER NOT EXISTS { $this pwm:underUseCase/pwm:permitsSelfVerification true }\n            }")])
add("EvaluationResult","pwm:EvaluationResult",[prop("evaluatesAgent",1,1,None,"pwm:AIAgent"),S1("evaluatedVersion"),S1("evaluatedModelVersion"),prop("evaluationLevel",1,1),prop("hasMetricValue",1,None,None,"pwm:EvaluationMetricValue"),
 "    sh:property [ sh:path pwm:evaluationDeploymentHash ; sh:maxCount 1 ; sh:datatype xsd:string ; sh:pattern \"^[a-f0-9]{64}$\" ; sh:message \"pwm:evaluationDeploymentHash: 64 lowercase hexadecimal characters.\" ]"])
add("EvaluationMetricValue","pwm:EvaluationMetricValue",[S1("metricName"),
 "    sh:property [ sh:path pwm:metricScore ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minInclusive 0 ; sh:maxInclusive 1 ; sh:message \"pwm:metricScore: minimum count 1, maximum count 1, between 0 and 1.\" ]"])
add("FairnessReport","pwm:FairnessReport",[prop("fairnessThreshold",1,1,"xsd:decimal"),prop("fairnessFlagged",1,1,"xsd:boolean"),prop("proxyAttribute",0,None,"xsd:string")])
add("EvaluationGate","pwm:EvaluationGate",[prop("gateUseCase",1,1,None,"pwm:AIUseCase"),S1("gateMetricName"),
 "    sh:property [ sh:path pwm:gateMinimumScore ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minInclusive 0 ; sh:maxInclusive 1 ; sh:message \"pwm:gateMinimumScore: between 0 and 1.\" ]",
 "    sh:property [ sh:path pwm:gateMaximumScore ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minInclusive 0 ; sh:maxInclusive 1 ; sh:message \"pwm:gateMaximumScore: between 0 and 1.\" ]",
 "    sh:xone ( [ sh:property [ sh:path pwm:gateMinimumScore ; sh:minCount 1 ] ] [ sh:property [ sh:path pwm:gateMaximumScore ; sh:minCount 1 ] ] ) ;\n    sh:message \"A gate states a minimum or a maximum score, not both and not neither (Kognita 0.6 item 11).\""])
add("ActivationGate","pwm:AIAgent",[
 sparql("An agent activated on an evaluation result that misses a minimum gate for its use case (Kognita 0.6 item 11).","            SELECT $this ?value WHERE {\n                $this pwm:activatedOnEvaluation ?value ; pwm:registeredForUseCase ?uc .\n                ?gate a pwm:EvaluationGate ; pwm:gateUseCase ?uc ; pwm:gateMetricName ?m ; pwm:gateMinimumScore ?min .\n                ?value pwm:hasMetricValue ?mv . ?mv pwm:metricName ?m ; pwm:metricScore ?s .\n                FILTER (?s < ?min)\n            }"),
 sparql("An agent activated on an evaluation result that exceeds a maximum gate for its use case (Kognita 0.6 item 11).","            SELECT $this ?value WHERE {\n                $this pwm:activatedOnEvaluation ?value ; pwm:registeredForUseCase ?uc .\n                ?gate a pwm:EvaluationGate ; pwm:gateUseCase ?uc ; pwm:gateMetricName ?m ; pwm:gateMaximumScore ?max .\n                ?value pwm:hasMetricValue ?mv . ?mv pwm:metricName ?m ; pwm:metricScore ?s .\n                FILTER (?s > ?max)\n            }")])
add("ReasoningTrace","pwm:ReasoningTrace",[prop("traceOfCall",1,1,None,"pwm:ModelCall"),S1("traceHash"),prop("retainedIn",1,1)])
SH.append('pwmsh:ClaimEnvelopeShape a sh:NodeShape ; rdfs:label "ClaimEnvelopeShape"@en ; sh:targetClass pwm:Claim ;\n    dcterms:source pwmref:KOG-ROADMAP ; rdfs:comment "Roadmap milestone 0.6 item 1. A verified claim carries its source and retrieval record."@en ;\n'
 + sparql("A verified claim names its source system and record and the source and retrieval times, so an RM can see why they are seeing it (Kognita 0.6 item 1).",
  "            SELECT $this WHERE {\n                $this pwm:validatorStatus pwm:ValidatorVerified .\n                FILTER NOT EXISTS { $this pwm:claimSourceSystem ?a . $this pwm:claimSourceRecord ?b . $this pwm:claimSourceTimestamp ?c . $this pwm:claimRetrievalTimestamp ?d }\n            }")+" .\n")
add("MemoryRecord","pwm:MemoryRecord",[prop("memoryScope",1,1),
 sparql("Relationship memory names its client, an expiry and a correction path (Kognita 0.6 item 8).",
  "            SELECT $this WHERE {\n                $this pwm:memoryScope pwm:MemoryRelationship .\n                FILTER NOT EXISTS { $this pwm:memorySubjectClient ?c . $this pwm:memoryExpiresAt ?e . $this pwm:memoryCorrectionPath ?p }\n            }"),
 sparql("Institutional memory carries no client; client data never crosses into it (Kognita 0.6 item 8).",
  "            SELECT $this ?value WHERE {\n                $this pwm:memoryScope pwm:MemoryInstitutional ; pwm:memorySubjectClient ?value .\n            }")])
add("LearningExport","pwm:LearningExport",[S1("exportPurpose"),S1("anonymizationReference"),S1("validationReference"),prop("approvedBy",1,1),prop("exportRequester",1,1,None,"fibo-fnd-aap-ppl:Person"),prop("exportedRecordId",1,None,"xsd:string")])
add("LearningExclusion","pwm:LearningExclusion",[S1("exclusionReason"),
 "    sh:or ( [ sh:path pwm:excludesClient ; sh:minCount 1 ] [ sh:path pwm:excludesRecordId ; sh:minCount 1 ] ) ;\n    sh:message \"An exclusion names a client or a record (Kognita 0.6 item 9).\""])
add("SensitiveAttributeInference","pwm:SensitiveAttributeInference",[prop("sensitiveKind",1,1),prop("permittedByUseCase",1,1,"xsd:boolean"),prop("blockedByPolicy",1,1,"xsd:boolean"),prop("changesClientTreatment",1,1,"xsd:boolean"),
 sparql("A special-category inference the use case does not permit must be blocked by policy (Kognita 0.6 item 10).",
  "            SELECT $this WHERE {\n                $this pwm:permittedByUseCase false ; pwm:blockedByPolicy false .\n            }"),
 sparql("An inference that changes how a client is treated needs a person's review and determination; an AI may flag, never determine (Kognita 0.6 item 10).",
  "            SELECT $this WHERE {\n                $this pwm:changesClientTreatment true .\n                FILTER NOT EXISTS { $this pwm:reviewedByRM ?p }\n            }")])
def shapes_text(): return "\n".join(SH)

# ---------------------------------------------------------------- example data
KEY_HEX="a"*64
POS = f'''
# --- Module 12: trust, resilience and assurance (synthetic) ------------------------------------
ex:SigningKey1 a pwm:SigningKey ; rdfs:label "Evidence signing key 1 (fictional)"@en ; pwm:keyId "key-0001" ; pwm:publicKeyReference "urn:synthetic:pubkey:0001" ; pwm:escrowReference "urn:synthetic:escrow:0001" .
ex:SigningKey2 a pwm:SigningKey ; rdfs:label "Evidence signing key 2 (fictional)"@en ; pwm:keyId "key-0002" ; pwm:publicKeyReference "urn:synthetic:pubkey:0002" .
ex:KeyRotation1 a pwm:KeyRotation ; pwm:rotatesFrom ex:SigningKey1 ; pwm:rotatesTo ex:SigningKey2 ; prov:generatedAtTime "2026-11-10T09:00:00+08:00"^^xsd:dateTime .
ex:KeyRevocation1 a pwm:KeyRevocation ; pwm:revokesKey ex:SigningKey1 ; pwm:effectiveFromSequence "100"^^xsd:nonNegativeInteger ; prov:generatedAtTime "2026-11-10T09:05:00+08:00"^^xsd:dateTime .
ex:Anchor1 a pwm:EvidenceAnchor ; pwm:anchoredRecordHash "sha256:{KEY_HEX}" ; pwm:anchoredSequence "8"^^xsd:nonNegativeInteger ; pwm:anchorLocation "urn:synthetic:independent-store/anchors-0001" ; prov:generatedAtTime "2026-11-10T09:10:00+08:00"^^xsd:dateTime .
ex:Recovery1 a pwm:RecoveryBundle ; pwm:bundleHash "sha256:{'b'*64}" ; pwm:recoveryCutoffSequence "8"^^xsd:nonNegativeInteger ; pwm:restoreVerified true ; pwm:includesRetentionState true .
ex:ExternalReview1 a pwm:ExternalReviewSubmission ; pwm:reviewedFromSequence "0"^^xsd:nonNegativeInteger ; pwm:reviewedToSequence "8"^^xsd:nonNegativeInteger ; pwm:signedBy ex:SigningKey2 ; prov:wasAttributedTo ex:PersonHeadOfAdvisory .
ex:UseCaseMeetingPrep pwm:approvedFallbackModel ex:FallbackModelCard .
ex:ProviderRegistration1 a pwm:ModelProviderRegistration ; pwm:registeredProvider ex:Provider ; pwm:dependentUseCase ex:UseCaseMeetingPrep ;
    pwm:dataResidency "Singapore" ; pwm:dataRetentionTerms "No retention after the response" ; pwm:trainingUse false ; pwm:subProcessor "Cloud host (fictional)" ;
    pwm:encryptionDescription "TLS 1.3 in transit; AES-256 at rest" ; pwm:incidentNotificationTerms "Within 24 hours" ; pwm:availabilityCommitment "99.9 percent monthly" ;
    pwm:modelChangeNotice "30 days written notice" ; pwm:auditRights true ; pwm:exitPlanReference "urn:synthetic:exit-plan:0001" ; pwm:regulatoryCooperation "Cooperates with the firm's regulators" .
ex:FallbackModelCard a pwm:ModelCard ; rdfs:label "Fallback model card (fictional)"@en .
ex:Failover1 a pwm:ProviderFailover ; pwm:failoverFrom ex:ModelCard ; pwm:failoverTo ex:FallbackModelCard ; pwm:failoverUseCase ex:UseCaseMeetingPrep ; pwm:failoverTrigger pwm:TriggerHttp5xx ; prov:generatedAtTime "2026-11-10T10:00:00+08:00"^^xsd:dateTime .
ex:ModelChange1 a pwm:ModelChangeDetection ; pwm:approvedModelVersion "model-2026-09" ; pwm:servedModelVersion "model-2026-10" ; pwm:triggersRevalidation true ; pwm:detectedOnCall ex:GatewayCall .
ex:Dependency1 a pwm:AIDependency ; pwm:dependencyLayer pwm:LayerModelProvider ; pwm:dependencyOf ex:UseCaseMeetingPrep ; pwm:dependencyTarget "Model provider (fictional)" ; pwm:dependencyDeclared true ; pwm:dependencyObserved true .
ex:Dependency2 a pwm:AIDependency ; pwm:dependencyLayer pwm:LayerVectorPlatform ; pwm:dependencyOf ex:UseCaseMeetingPrep ; pwm:dependencyTarget "Vector platform (fictional)" ; pwm:dependencyDeclared false ; pwm:dependencyObserved true .
ex:Divergence1 a pwm:DependencyDivergence ; pwm:divergenceKind pwm:DivergenceUndeclared ; pwm:divergenceDependency ex:Dependency2 .
ex:Concentration1 a pwm:ConcentrationMeasure ; pwm:concentrationLayer pwm:LayerModelProvider ; pwm:concentrationTarget "Model provider (fictional)" ; pwm:criticalUseCaseCount "1"^^xsd:nonNegativeInteger ; pwm:criticalUseCaseTotal "1"^^xsd:nonNegativeInteger ; pwm:criticalUseCaseShare "1.0"^^xsd:decimal .
ex:Drill1 a pwm:FailureDrill ; pwm:drillKind pwm:DrillModelUnavailable ; pwm:drillUseCase ex:UseCaseMeetingPrep ; pwm:evidenceCompleteAfterDrill true ; pwm:gatewayFailureMode pwm:FailClosed .
ex:RecordRetention1 a pwm:RecordClassRetention ; pwm:recordClass pwm:RecordAdvice ; pwm:classRetentionDays "2555"^^xsd:nonNegativeInteger .
ex:RecordRetention2 a pwm:RecordClassRetention ; pwm:recordClass pwm:RecordComplianceRecords ; pwm:retainIndefinitely true .
ex:IncidentReport1 a pwm:IncidentReport ; pwm:reportsIncident ex:Incident ; pwm:redactedReport false ; pwm:controlRan "policy decision before model call" ; pwm:controlFailed "none missing" ; pwm:runtimeChange "none" .
ex:AssurancePolicy1 a pwm:AssurancePolicy ; pwm:assuranceUseCase ex:UseCaseMeetingPrep ; pwm:assuranceAudience "example-private-bank" ; pwm:requiresExternalAssertion false ; pwm:requiresDeploymentAttestation pwm:AttestationMonitoring ; pwm:requiresComponentApproval true ; pwm:requiresTransferClassification true ; pwm:requiresUpdateApproval true ; pwm:outcomeStepDown true ; pwm:maxAdverseOutcomes "1"^^xsd:positiveInteger ; pwm:outcomeWindowDays "30"^^xsd:positiveInteger .
ex:Manifest1 a pwm:AgentManifest ; pwm:manifestSchemaVersion "kognita-kya-1" ; pwm:manifestOfAgent ex:Agent ; pwm:manifestDigest "sha256:{'c'*64}" ; pwm:signedBy ex:SigningKey2 .
ex:Assertion1 a pwm:ExternalAgentAssertion ; pwm:assertionIssuer "https://issuer.example.org" ; pwm:assertionSubject "external-agent-1" ; pwm:assertionPrincipal ex:PersonRM ; pwm:assertionPurpose "meeting preparation" ; pwm:assertionAudience "example-private-bank" ; pwm:assertionScope "read=portfolio" ; pwm:assertionDepth "0"^^xsd:nonNegativeInteger ; pwm:issuedAt "2026-11-10T08:00:00+08:00"^^xsd:dateTime ; pwm:expiresAt "2026-11-10T20:00:00+08:00"^^xsd:dateTime .
ex:Attestation1 a pwm:DeploymentAttestation ; pwm:attestationKind pwm:AttestationMonitoring ; pwm:attestationIssuer "independent-assessor (fictional)" ; pwm:attestationSubject "meeting preparation agent" ; pwm:attestsDeployment "sha256:{'d'*64}" ; pwm:attestsPolicy "sha256:{'e'*64}" ; pwm:attestationReference "urn:synthetic:attestation:0001" ; pwm:issuedAt "2026-11-10T08:00:00+08:00"^^xsd:dateTime ; pwm:expiresAt "2027-02-10T08:00:00+08:00"^^xsd:dateTime .
ex:UpdateApproval1 a pwm:ThirdPartyUpdateApproval ; pwm:candidateDigest "sha256:{'f'*64}" ; pwm:regressionReference "urn:synthetic:regression:0001" ; pwm:rollbackReference "urn:synthetic:rollback:0001" ; pwm:providerChange "Provider updated the served model" ; pwm:approvedBy ex:PersonHeadOfAdvisory .
ex:Component1 a pwm:ApprovedComponent ; pwm:componentName "agent-runtime" ; pwm:componentVersion "1.4.2" ; pwm:sbomDigest "{'1'*64}" ; pwm:imageDigest "sha256:{'2'*64}" ; pwm:provenanceReference "urn:synthetic:provenance:0001" .
ex:Advisory1 a pwm:ComponentAdvisory ; pwm:advisoryReference "urn:synthetic:advisory:0001" ; pwm:affectsComponent ex:Component1 ; pwm:quarantined false .
ex:Transfer1 a pwm:TechnologyTransferClassification ; pwm:transferType pwm:TransferApi ; pwm:technologyOrigin "SG" ; pwm:deploymentJurisdiction "SG" ; pwm:transferPolicyReference "urn:synthetic:transfer-policy:0001" .
ex:ProductionOutcome1 a pwm:ProductionOutcome ; pwm:outcomeKind pwm:Correction ; pwm:partOfInteraction ex:Interaction ; pwm:affectedOutput ex:Recommendation ; pwm:implicatedModel ex:Model ; pwm:outcomeApprover ex:PersonRM ; pwm:outcomeId "outcome-0001" ; pwm:outcomeSourceReference "urn:synthetic:crm:outcome-0001" ; pwm:intendedResult "Meeting brief delivered" ; pwm:actualResult "Meeting brief delivered" ; pwm:adverse false .
ex:VerifierProvider a pwm:ModelProvider ; rdfs:label "Verifier model provider (fictional)"@en .
ex:VerifierModelCard a pwm:ModelCard ; pwm:providedBy ex:VerifierProvider ; pwm:modelVersion "verifier-2026-10" ; rdfs:label "Verifier model card (fictional)"@en .
ex:Verifier1 a pwm:Verifier ; pwm:verifierAgent ex:Agent ; pwm:verifierModelCard ex:VerifierModelCard .
ex:ClaimVerification1 a pwm:ClaimVerification ; pwm:verifiedClaim ex:ClaimRiskProfile ; pwm:verifiedBy ex:Verifier1 ; pwm:producerModelCard ex:ModelCard ; pwm:verificationVerdict pwm:ValidatorVerified ; pwm:underUseCase ex:UseCaseMeetingPrep .
ex:Evaluation1 a pwm:EvaluationResult ; pwm:evaluatesAgent ex:Agent ; pwm:evaluatedVersion "agent-1.4" ; pwm:evaluatedModelVersion "model-2026-09" ; pwm:evaluationLevel pwm:LevelAgent ; pwm:evaluationDeploymentHash "{'3'*64}" ; pwm:hasMetricValue ex:Metric1, ex:Metric2 .
ex:Metric1 a pwm:EvaluationMetricValue ; pwm:metricName "groundedness" ; pwm:metricScore "0.96"^^xsd:decimal .
ex:Metric2 a pwm:EvaluationMetricValue ; pwm:metricName "hallucination rate" ; pwm:metricScore "0.02"^^xsd:decimal .
ex:UseCaseMeetingPrep pwm:hasDocumentation ex:ModelCard .
ex:Agent pwm:hasComponent ex:Component1 .
ex:Worker pwm:hasComponent ex:Component1 ; pwm:hasBlastRadiusLimit ex:Limit .
ex:Gate1 a pwm:EvaluationGate ; pwm:gateUseCase ex:UseCaseMeetingPrep ; pwm:gateMetricName "groundedness" ; pwm:gateMinimumScore "0.90"^^xsd:decimal .
ex:Gate2 a pwm:EvaluationGate ; pwm:gateUseCase ex:UseCaseMeetingPrep ; pwm:gateMetricName "hallucination rate" ; pwm:gateMaximumScore "0.05"^^xsd:decimal .
ex:Agent pwm:activatedOnEvaluation ex:Evaluation1 .
ex:Drill2 a pwm:FailureDrill ; pwm:drillKind pwm:DrillCompromisedModel ; pwm:drillScenario pwm:ScenarioCompromisedProvider ; pwm:drillUseCase ex:UseCaseMeetingPrep ; pwm:evidenceCompleteAfterDrill true ; pwm:gatewayFailureMode pwm:FailClosed .
ex:Fairness1 a pwm:FairnessReport ; pwm:fairnessThreshold "0.10"^^xsd:decimal ; pwm:fairnessFlagged false ; pwm:proxyAttribute "postcode" .
ex:Trace1 a pwm:ReasoningTrace ; pwm:traceOfCall ex:GatewayCall ; pwm:traceHash "sha256:{'4'*64}" ; pwm:retainedIn ex:RetainedPrompt .
ex:ClaimRiskProfile pwm:claimSourceSystem "portfolio-system (fictional)" ; pwm:claimSourceRecord "risk-profile/0001" ; pwm:claimSourceTimestamp "2026-11-03T08:00:00+08:00"^^xsd:dateTime ; pwm:claimRetrievalTimestamp "2026-11-03T09:59:30+08:00"^^xsd:dateTime ; pwm:calculationMethod "none: recorded value" ; pwm:claimPromptVersion "meeting-v3" ; pwm:claimConfidence "0.97"^^xsd:decimal ; pwm:claimPolicyStatus "may reach RM" .
ex:Memory2 a pwm:MemoryRecord ; pwm:memoryScope pwm:MemoryRelationship ; pwm:memorySubjectClient ex:Client ; pwm:memoryExpiresAt "2027-11-03T00:00:00+08:00"^^xsd:dateTime ; pwm:memoryCorrectionPath "Ask the RM; the RM corrects through the review queue" .
ex:Memory3 a pwm:MemoryRecord ; pwm:memoryScope pwm:MemoryInstitutional .
ex:LearningExport1 a pwm:LearningExport ; pwm:exportPurpose "Quality evaluation" ; pwm:anonymizationReference "urn:synthetic:anonymization:0001" ; pwm:validationReference "urn:synthetic:validation:0001" ; pwm:approvedBy ex:PersonHeadOfAdvisory ; pwm:exportRequester ex:PersonRM ; pwm:exportedRecordId "interaction:journey-1" .
ex:LearningExclusion1 a pwm:LearningExclusion ; pwm:exclusionReason "Client withdrew consent to learning use" ; pwm:excludesClient ex:Client .
ex:LearningExclusion2 a pwm:LearningExclusion ; pwm:exclusionReason "Record excluded after a complaint" ; pwm:excludesRecordId "interaction:journey-0" .
ex:Sensitive1 a pwm:SensitiveAttributeInference ; pwm:sensitiveKind pwm:SensitiveVulnerability ; pwm:permittedByUseCase true ; pwm:blockedByPolicy false ; pwm:changesClientTreatment false ; pwm:reviewedByRM ex:PersonRM .
ex:Conformance1 a pwm:SupervisoryConformanceReport ; pwm:conformanceProfile pwm:ProfileMAS ; pwm:reportAsOf "2026-11-10T12:00:00+08:00"^^xsd:dateTime ; pwm:coversControl "A1", "A3" ; pwm:referencesEvidence ex:Evidence1 .
'''
# Bad fixtures: (name, shape, path, ttl)
BAD = [
("BadTrustKey","SigningKey","keyId",'a pwm:SigningKey ; pwm:publicKeyReference "urn:synthetic:pubkey:bad"'),
("BadTrustUnsigned","SignedEvidenceShape",None,'a pwm:EvidenceRecord ; pwm:eventType pwm:EventAssurance ; pwm:chainStart true ; pwm:recordHash "sha256:{h}" ; prov:generatedAtTime "2026-11-10T11:00:00+08:00"^^xsd:dateTime ; pwm:signature "c2lnbmF0dXJl"'),
("BadTrustRevoked","SignedEvidenceShape",None,'a pwm:EvidenceRecord ; pwm:eventType pwm:EventAssurance ; pwm:chainStart true ; pwm:recordHash "sha256:{h2}" ; prov:generatedAtTime "2026-11-10T11:01:00+08:00"^^xsd:dateTime ; pwm:signature "c2lnbmF0dXJl" ; pwm:signedBy ex:SigningKey1 ; pwm:evidenceSequence "150"^^xsd:nonNegativeInteger'),
("BadTrustRotation","KeyRotationShape","rotatesFrom",'a pwm:KeyRotation ; pwm:rotatesFrom ex:SigningKey1 ; pwm:rotatesTo ex:SigningKey1'),
("BadTrustRevocation","KeyRevocationShape","effectiveFromSequence",'a pwm:KeyRevocation ; pwm:revokesKey ex:SigningKey2'),
("BadTrustAnchor","EvidenceAnchorShape","anchorLocation",'a pwm:EvidenceAnchor ; pwm:anchoredRecordHash "sha256:{h}" ; pwm:anchoredSequence "8"^^xsd:nonNegativeInteger ; prov:generatedAtTime "2026-11-10T09:10:00+08:00"^^xsd:dateTime'),
("BadTrustRecovery","RecoveryBundleShape","restoreVerified",'a pwm:RecoveryBundle ; pwm:bundleHash "sha256:{h}" ; pwm:recoveryCutoffSequence "8"^^xsd:nonNegativeInteger ; pwm:includesRetentionState true'),
("BadTrustReview","ExternalReviewSubmissionShape","reviewedFromSequence",'a pwm:ExternalReviewSubmission ; pwm:reviewedFromSequence "9"^^xsd:nonNegativeInteger ; pwm:reviewedToSequence "3"^^xsd:nonNegativeInteger ; pwm:signedBy ex:SigningKey2'),
("BadTrustProviderReg","ModelProviderRegistrationShape","exitPlanReference",'a pwm:ModelProviderRegistration ; pwm:registeredProvider ex:ProviderOther ; pwm:dependentUseCase ex:UseCaseMeetingPrep ; pwm:dataResidency "Singapore" ; pwm:dataRetentionTerms "None" ; pwm:trainingUse false ; pwm:subProcessor "Cloud host" ; pwm:encryptionDescription "TLS" ; pwm:incidentNotificationTerms "24h" ; pwm:availabilityCommitment "99.9" ; pwm:modelChangeNotice "30 days" ; pwm:auditRights true ; pwm:regulatoryCooperation "Yes"'),
("BadTrustFailover","ProviderFailoverShape",None,'a pwm:ProviderFailover ; pwm:failoverFrom ex:ModelCard ; pwm:failoverTo ex:UnapprovedModelCard ; pwm:failoverUseCase ex:UseCaseMeetingPrep ; pwm:failoverTrigger pwm:TriggerConnectionFailure'),
("BadTrustModelChange","ModelChangeDetectionShape",None,'a pwm:ModelChangeDetection ; pwm:approvedModelVersion "model-2026-09" ; pwm:servedModelVersion "model-2026-10" ; pwm:triggersRevalidation false ; pwm:detectedOnCall ex:GatewayCall'),
("BadTrustDependency","AIDependencyShape","dependencyTarget",'a pwm:AIDependency ; pwm:dependencyLayer pwm:LayerCloud ; pwm:dependencyOf ex:UseCaseMeetingPrep ; pwm:dependencyDeclared true ; pwm:dependencyObserved true'),
("BadTrustDivergence","DependencyDivergenceShape",None,'a pwm:DependencyDivergence ; pwm:divergenceKind pwm:DivergenceUndeclared ; pwm:divergenceDependency ex:Dependency1'),
("BadTrustConcentration","ConcentrationMeasureShape","criticalUseCaseShare",'a pwm:ConcentrationMeasure ; pwm:concentrationLayer pwm:LayerCloud ; pwm:concentrationTarget "Cloud (fictional)" ; pwm:criticalUseCaseCount "1"^^xsd:nonNegativeInteger ; pwm:criticalUseCaseTotal "2"^^xsd:nonNegativeInteger ; pwm:criticalUseCaseShare "1.5"^^xsd:decimal'),
("BadTrustConcentrationCount","ConcentrationMeasureShape","criticalUseCaseCount",'a pwm:ConcentrationMeasure ; pwm:concentrationLayer pwm:LayerCloud ; pwm:concentrationTarget "Cloud (fictional)" ; pwm:criticalUseCaseCount "3"^^xsd:nonNegativeInteger ; pwm:criticalUseCaseTotal "2"^^xsd:nonNegativeInteger ; pwm:criticalUseCaseShare "0.5"^^xsd:decimal'),
("BadTrustDrill","FailureDrillShape","evidenceCompleteAfterDrill",'a pwm:FailureDrill ; pwm:drillKind pwm:DrillToolFailure ; pwm:drillUseCase ex:UseCaseMeetingPrep ; pwm:gatewayFailureMode pwm:FailClosed'),
("BadTrustRecordClass","RecordClassRetentionShape",None,'a pwm:RecordClassRetention ; pwm:recordClass pwm:RecordAdvice ; pwm:retainIndefinitely true ; pwm:classRetentionDays "10"^^xsd:nonNegativeInteger'),
("BadTrustIncidentReport","IncidentReportShape","reportsIncident",'a pwm:IncidentReport ; pwm:redactedReport true ; pwm:controlRan "policy decision"'),
("BadTrustAssurancePolicy","AssurancePolicyShape","outcomeWindowDays",'a pwm:AssurancePolicy ; pwm:assuranceUseCase ex:UseCaseMeetingPrep ; pwm:assuranceAudience "example-private-bank" ; pwm:maxAdverseOutcomes "1"^^xsd:positiveInteger ; pwm:outcomeWindowDays "4000"^^xsd:positiveInteger'),
("BadTrustManifest","AgentManifestShape","signedBy",'a pwm:AgentManifest ; pwm:manifestSchemaVersion "kognita-kya-1" ; pwm:manifestOfAgent ex:Agent ; pwm:manifestDigest "sha256:{h}"'),
("BadTrustAssertionExpiry","ExternalAgentAssertionShape","issuedAt",'a pwm:ExternalAgentAssertion ; pwm:assertionIssuer "https://issuer.example.org" ; pwm:assertionSubject "x" ; pwm:assertionPrincipal ex:PersonRM ; pwm:assertionPurpose "p" ; pwm:assertionAudience "a" ; pwm:assertionScope "read=portfolio" ; pwm:assertionDepth "0"^^xsd:nonNegativeInteger ; pwm:issuedAt "2026-11-10T20:00:00+08:00"^^xsd:dateTime ; pwm:expiresAt "2026-11-10T08:00:00+08:00"^^xsd:dateTime'),
("BadTrustAssertionDepth","ExternalAgentAssertionShape",None,'a pwm:ExternalAgentAssertion ; pwm:assertionIssuer "https://issuer.example.org" ; pwm:assertionSubject "x" ; pwm:assertionPrincipal ex:PersonRM ; pwm:assertionPurpose "p" ; pwm:assertionAudience "a" ; pwm:assertionScope "read=portfolio" ; pwm:assertionDepth "1"^^xsd:nonNegativeInteger ; pwm:issuedAt "2026-11-10T08:00:00+08:00"^^xsd:dateTime ; pwm:expiresAt "2026-11-10T20:00:00+08:00"^^xsd:dateTime'),
("BadTrustAttestation","DeploymentAttestationShape","issuedAt",'a pwm:DeploymentAttestation ; pwm:attestationKind pwm:AttestationPrivacy ; pwm:attestationIssuer "i" ; pwm:attestationSubject "s" ; pwm:attestsDeployment "sha256:{h}" ; pwm:attestsPolicy "sha256:{h}" ; pwm:attestationReference "urn:synthetic:a" ; pwm:issuedAt "2026-12-10T08:00:00+08:00"^^xsd:dateTime ; pwm:expiresAt "2026-11-10T08:00:00+08:00"^^xsd:dateTime'),
("BadTrustUpdate","ThirdPartyUpdateApprovalShape","rollbackReference",'a pwm:ThirdPartyUpdateApproval ; pwm:candidateDigest "sha256:{h}" ; pwm:regressionReference "urn:synthetic:r" ; pwm:approvedBy ex:PersonHeadOfAdvisory'),
("BadTrustComponent","ApprovedComponentShape","sbomDigest",'a pwm:ApprovedComponent ; pwm:componentName "x" ; pwm:componentVersion "1" ; pwm:provenanceReference "urn:synthetic:p" ; pwm:sbomDigest "NOTHEX"'),
("BadTrustComponentImage","ApprovedComponentShape","imageDigest",'a pwm:ApprovedComponent ; pwm:componentName "x" ; pwm:componentVersion "1" ; pwm:provenanceReference "urn:synthetic:p" ; pwm:sbomDigest "{h}" ; pwm:imageDigest "latest"'),
("BadTrustAdvisory","ComponentAdvisoryShape","quarantined",'a pwm:ComponentAdvisory ; pwm:advisoryReference "urn:synthetic:adv" ; pwm:affectsComponent ex:Component1'),
("BadTrustTransfer","TechnologyTransferClassificationShape","transferPolicyReference",'a pwm:TechnologyTransferClassification ; pwm:transferType pwm:TransferLocal ; pwm:technologyOrigin "SG" ; pwm:deploymentJurisdiction "SG"'),
("BadTrustOutcome","ProductionOutcomeShape","adverse",'a pwm:ProductionOutcome ; pwm:outcomeKind pwm:Correction ; pwm:partOfInteraction ex:Interaction ; pwm:affectedOutput ex:Recommendation ; pwm:implicatedModel ex:Model ; pwm:outcomeApprover ex:PersonRM ; pwm:outcomeId "o" ; pwm:outcomeSourceReference "urn:synthetic:crm:o" ; pwm:intendedResult "i" ; pwm:actualResult "a"'),
("BadTrustVerifier","VerifierShape","verifierModelCard",'a pwm:Verifier ; pwm:verifierAgent ex:Agent'),
("BadTrustSelfVerification","ClaimVerificationShape",None,'a pwm:ClaimVerification ; pwm:verifiedClaim ex:ClaimRiskProfile ; pwm:verifiedBy ex:SameProviderVerifier ; pwm:producerModelCard ex:ModelCard ; pwm:verificationVerdict pwm:ValidatorVerified ; pwm:underUseCase ex:UseCaseMeetingPrep'),
("BadTrustEvaluation","EvaluationResultShape","evaluationLevel",'a pwm:EvaluationResult ; pwm:evaluatesAgent ex:Agent ; pwm:evaluatedVersion "v" ; pwm:evaluatedModelVersion "m" ; pwm:hasMetricValue ex:Metric1'),
("BadTrustEvaluationHash","EvaluationResultShape","evaluationDeploymentHash",'a pwm:EvaluationResult ; pwm:evaluatesAgent ex:Agent ; pwm:evaluatedVersion "v" ; pwm:evaluatedModelVersion "m" ; pwm:evaluationLevel pwm:LevelModel ; pwm:hasMetricValue ex:Metric1 ; pwm:evaluationDeploymentHash "xyz"'),
("BadTrustMetric","EvaluationMetricValueShape","metricScore",'a pwm:EvaluationMetricValue ; pwm:metricName "accuracy" ; pwm:metricScore "1.7"^^xsd:decimal'),
("BadTrustFairness","FairnessReportShape","fairnessFlagged",'a pwm:FairnessReport ; pwm:fairnessThreshold "0.10"^^xsd:decimal'),
("BadTrustTrace","ReasoningTraceShape","traceHash",'a pwm:ReasoningTrace ; pwm:traceOfCall ex:GatewayCall ; pwm:retainedIn ex:RetainedPrompt'),
("BadTrustClaimEnvelope","ClaimEnvelopeShape",None,'a pwm:Claim ; pwm:aboutClient ex:Client ; pwm:claimType pwm:VerifiedFact ; pwm:citesSource ex:FactRiskProfile ; pwm:validatorStatus pwm:ValidatorVerified ; pwm:underUseCase ex:UseCaseMeetingPrep ; pwm:modelVersion "model-2026-09"'),
("BadTrustMemoryRelationship","MemoryRecordShape",None,'a pwm:MemoryRecord ; pwm:memoryScope pwm:MemoryRelationship ; pwm:memorySubjectClient ex:Client'),
("BadTrustMemoryInstitutional","MemoryRecordShape",None,'a pwm:MemoryRecord ; pwm:memoryScope pwm:MemoryInstitutional ; pwm:memorySubjectClient ex:Client'),
("BadTrustLearningExport","LearningExportShape","approvedBy",'a pwm:LearningExport ; pwm:exportPurpose "Training" ; pwm:anonymizationReference "urn:synthetic:a" ; pwm:validationReference "urn:synthetic:v" ; pwm:exportRequester ex:PersonRM ; pwm:exportedRecordId "interaction:x"'),
("BadTrustLearningExclusion","LearningExclusionShape",None,'a pwm:LearningExclusion ; pwm:exclusionReason "Withdrawn"'),
("BadTrustSensitiveBlocked","SensitiveAttributeInferenceShape",None,'a pwm:SensitiveAttributeInference ; pwm:sensitiveKind pwm:SensitiveHealth ; pwm:permittedByUseCase false ; pwm:blockedByPolicy false ; pwm:changesClientTreatment false'),
("BadTrustSensitiveDetermination","SensitiveAttributeInferenceShape",None,'a pwm:SensitiveAttributeInference ; pwm:sensitiveKind pwm:SensitiveVulnerability ; pwm:permittedByUseCase true ; pwm:blockedByPolicy false ; pwm:changesClientTreatment true'),
("BadTrustGate","EvaluationGateShape",None,'a pwm:EvaluationGate ; pwm:gateUseCase ex:UseCaseMeetingPrep ; pwm:gateMetricName "accuracy"'),
("BadTrustConformance","SupervisoryConformanceReportShape","coversControl",'a pwm:SupervisoryConformanceReport ; pwm:conformanceProfile pwm:ProfileFCA ; pwm:reportAsOf "2026-11-10T12:00:00+08:00"^^xsd:dateTime'),
]
SUPPORT = '''ex:ProviderOther a pwm:ModelProvider ; rdfs:label "Other provider (fictional)"@en .
ex:ProviderUnregistered a pwm:ModelProvider ; rdfs:label "Unregistered provider (fictional)"@en .
ex:SameProviderVerifier a pwm:Verifier ; pwm:verifierAgent ex:Agent ; pwm:verifierModelCard ex:SameProviderCard .
ex:SameProviderCard a pwm:ModelCard ; pwm:providedBy ex:Provider ; pwm:modelVersion "other-version" ; rdfs:label "Same provider card (fictional)"@en .
ex:BadEvalLow a pwm:EvaluationResult ; pwm:evaluatesAgent ex:Agent ; pwm:evaluatedVersion "agent-1.5" ; pwm:evaluatedModelVersion "model-2026-09" ; pwm:evaluationLevel pwm:LevelAgent ; pwm:hasMetricValue ex:BadMetricLow, ex:BadMetricRate .\nex:BadMetricLow a pwm:EvaluationMetricValue ; pwm:metricName "groundedness" ; pwm:metricScore "0.50"^^xsd:decimal .\nex:BadMetricRate a pwm:EvaluationMetricValue ; pwm:metricName "hallucination rate" ; pwm:metricScore "0.30"^^xsd:decimal .\nex:Agent pwm:activatedOnEvaluation ex:BadEvalLow .\nex:UnapprovedModelCard a pwm:ModelCard ; rdfs:label "Unapproved fallback model card (fictional)"@en .
ex:GatewayCall pwm:providedBy ex:ProviderUnregistered .
'''
# POSITIVE example also needs approvedFallbackModel on the use case and a fallback provider registration for ProviderOther in the negative fixture (not needed).
def example_blocks():
    h="9"*64; h2="8"*64
    bad=SUPPORT+"\n"+"\n".join(f"ex:{n} {t.format(h=h,h2=h2)} ." for n,s,p,t in BAD)+"\n"
    return POS, bad
def expected():
    out=[]
    for n,s,p,t in BAD:
        shape = s if s.endswith("Shape") else s+"Shape"
        out.append((n,shape,p))
    out.append(("Agent","ActivationGateShape",None)); out.append(("Agent","ActivationGateShape",None))
    out.append(("GatewayCall","RegisteredProviderCallShape",None))
    out.append(("GatewayCall","ModelCallShape","providedBy"))  # the extra provider also breaks the single-provider rule
    return out

if __name__=="__main__":
    BEGIN,END="# >>> module 12 BEGIN (generated by tools/gen_module12.py)","# <<< module 12 END"
    (ROOT/"src/12-trust-resilience-assurance.ttl").write_text(ttl_module())
    # bindings
    b=ROOT/"src/08-code-list-bindings.ttl"; t=b.read_text()
    t=re.sub(r"\n+"+re.escape(BEGIN)+r".*?"+re.escape(END)+r"\n","\n",t,flags=re.S)
    b.write_text(t.rstrip("\n")+"\n\n"+BEGIN+"\n"+bindings()+END+"\n")
    sf=ROOT/"src/shapes/pwm-shapes.ttl"; t=sf.read_text()
    t=re.sub(r"\n+"+re.escape(BEGIN)+r".*?"+re.escape(END)+r"\n","\n",t,flags=re.S)
    sf.write_text(t.rstrip("\n")+"\n\n"+BEGIN+"\n"+shapes_text()+END+"\n")
    pos,bad=example_blocks()
    for fn,blk in (("examples/pwm-example.ttl",pos),("examples/pwm-example-invalid.ttl",bad)):
        f=ROOT/fn; t=f.read_text()
        t=re.sub(r"\n+"+re.escape(BEGIN)+r".*?"+re.escape(END)+r"\n","\n",t,flags=re.S)
        f.write_text(t.rstrip("\n")+"\n\n"+BEGIN+"\n"+blk+END+"\n")
    vf=ROOT/"tools/validate.py"; t=vf.read_text()
    t=re.sub(r"\n+"+re.escape(BEGIN)+r".*?"+re.escape(END)+r"\n","\n",t,flags=re.S)
    anchor="\ndef local(term):"
    blk=BEGIN+"\nEXPECTED.update("+repr(expected())+")\n"+END+"\n"
    t=t.replace(anchor,"\n"+blk+anchor,1)
    vf.write_text(t)
    print("generated",len(C),"classes",len(P),"properties",len(S),"schemes",len(BAD),"bad fixtures")
