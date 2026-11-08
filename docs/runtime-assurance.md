# Runtime assurance (0.8)

Kognita checks evidence supplied by the institution before it admits a governed
agent action. It does not run an identity provider, sandbox, network monitor,
software scanner, instruction-verification channel, or legal assessment.

The controls are implemented in `kognita.assurance`. Configure them per agent,
for a linked use case. Existing 0.7 deployments keep their existing admission
rules until the accountable owner installs an assurance policy. A configured
requirement fails closed when its evidence is missing, expired, mismatched,
quarantined, or corrupt. It applies to `decide`, governed tools, the AI gateway,
the MCP proxy, and standing checks before releasing a held action.

## Trust and administration

The host must authenticate the administrator before calling an administration
API or allowing use of the local CLI. An `actor_id` string is not authentication;
this is the same administration boundary used by the existing authority and
policy APIs. `configure`, component/update approvals and outcome registration
require the agent's accountable human owner. Registered workloads cannot act as
owners or independent attestors. Advisory ingestion requires an authenticated
security operator. Protect database access and private key files separately
from agent credentials.

An owner pins external assertion issuers and independent runtime attestors by
Ed25519 public PEM. Request data cannot introduce trust roots. Installing signed
trust roots or generating signed records needs the existing `[signing]` extra;
ordinary unconfigured decisions and policies without trust roots need no new
dependency. To revoke a trust root, replace the policy without that key. Every
policy replacement invalidates previous policy-bound attestations and approvals.

All policies, approvals, signed attestations, outcomes and findings are
`ASSURANCE` events in the existing append-only evidence chain. Existing backup,
restore, anchoring, signing and chain verification apply. No database column
migration is needed. PostgreSQL 12 or newer is required (CI tests PostgreSQL 16). Existing
PostgreSQL event enums are extended idempotently
by `create_all(engine)` during initialization. Run initialization with schema
owner privileges before starting upgraded serving processes; runtime workloads
must not have DDL privileges. Callers own transactions; commit successful administrative
operations and roll back exceptions. Reports verify local chain integrity;
independent authenticity still requires the existing root-key/anchor verification
workflow.

## Configure a reviewed deployment

Start with an existing registered agent, linked use case, evaluation and approved
behaviour deployment. For a live agent, configure the policy and install all evidence for its exact
binding in one transaction, verify admission, then commit. Alternatively, use a
maintenance window: stop normal operation before committing the policy, obtain
and commit the bound evidence, verify admission, then resume operation. A policy
committed without its required evidence immediately denies admission. For a deployment change, submit the update packet and evaluation,
approve the changed behaviour, and refresh deployment-bound attestations before
running it. An updated deployment cannot reuse the previous attestations.

```python
from kognita.assurance import AssurancePolicy, configure

policy = configure(session, "briefing-agent", AssurancePolicy(
    use_case="client-intelligence",
    audience="bank-ingress",
    issuers={"partner-identity": partner_public_pem},
    attestors={"bank-runtime": runtime_public_pem},
    external_required=True,
    privacy_required=True,
    environment_required=True,
    monitoring_required=True,
    components_required=True,
    update_required=True,
    instruction_tools=["send_payment", "change_account"],
    jurisdictions=["SG"],
    data_classes=["client-identifying"],
    client_impact="advice",
), evidence=writer, actor_id="Accountable Owner")
```

`credential_expiry` can impose an additional credential expiry floor on every
admitted request. It does not extend an expired delegation or revoked workload
credential. Existing authority, use-case, autonomy, privacy and tool checks all
remain necessary; passing an assurance check cannot override another denial.

Assurance reads batch agents and advisories and verify the chain once per
snapshot. A matching head digest alone does not prove that an older payload
is intact, so a new snapshot rechecks the prefix instead of trusting a
cross-snapshot head-only cache. Monitoring also reconciles action history.
Plan retention and capacity for these integrity checks. Replay rebuilds assurance
state from the recorded decision sequence and time using the pinned identity;
later actions, including actions at the same timestamp, cannot change a past
monitoring verdict. Live admission continues to reconcile all current actions.

## Signed boundary attestations (A5, A6, A7, A10, A11)

A runtime signs an `Attestation` with issuer, subject agent, issue/expiry times,
policy digest, deployment digest, source reference and claims. Use
`binding(agent, policy)` to obtain the current binding. The bank's runtime
must verify the actual condition before signing; a reference is not a scanner
or a live sandbox measurement performed by Kognita.

```python
from datetime import timedelta
from kognita.assurance import Attestation, sign_record
from kognita.assurance.records import binding, context
from kognita.assurance.runtime import attest
from kognita.canonical import canonical_hash

packet = sign_record(Attestation(
    kind="privacy", issuer="bank-runtime", subject=agent.name,
    issued_at=now, expires_at=now + timedelta(days=30),
    **binding(agent, policy),
    claims={"context_digest": canonical_hash(context(session, agent, policy))},
    reference="privacy-office:review-123",
), runtime_signer)
attest(session, agent.name, packet, evidence=writer,
       actor_id="bank-runtime", now=now)
```

| Kind | Required claims | Admission/report behaviour |
|---|---|---|
| `privacy` | `context_digest` | Fingerprints purpose, linked use-case purpose, tools/connectors, data classes/sources, memory behaviour, model/provider declarations, jurisdiction, autonomy and client impact. A material change or policy replacement invalidates it. |
| `environment` | `enforced: true`, nonempty `network_policy`, `isolation`, `credential_scope`, `sandbox_identity`, `termination_path` references | Required before high-risk admission and activation. The signed record binds the deployment and policy. |
| `monitoring` | `from_sequence`, `through_sequence`, `action_hashes`, `unexplained_external_actions` | Reconciles exact local action hashes against independently supplied telemetry. Missing or unexplained actions create a finding. Actions outside the attested window also prevent admission when required. Refresh coverage after actions; there is no assumption that a future action was already monitored. |
| `transfer` | `transfer_digest`, `approved: true` | The digest comes from `transfer_digest(policy)`. It binds institution-supplied origin, destination, transfer type and legal-policy reference; Kognita does not classify the law. |
| `instruction` | `request_digest`, `verification_channel`, different `maker` and `checker` | Exact request binding from `request_digest(envelope)`, covering principal, purpose, tool, arguments and subjects. The request supplies the recorded attestation digest. Verification-channel operation remains external. |

A missing or stale record is a denial, including after a human approval hold.
Environment attestations require all five references. An assertion or attestation
with an unknown key, invalid signature, future issue time or reached expiry is
refused. A newly installed policy requires new approvals, even if its text change
only concerned trust keys.

## External assertions (A2)

`Assertion` carries issuer, subject agent, principal, purpose, exact scopes,
audience, issue/expiry times and delegation depth. Use `sign_record` and a pinned
issuer key. Scope dimensions include `tools` and referenced subject types such
as `client`, `model` or `portfolio`. Wildcards and empty scopes grant nothing.
A child must supply the complete parent chain; each parent is independently
verified under a pinned key, with narrowing scopes, expiry and depth. Trust is
not inherited from an orchestrator.

For library calls, put the packet in `Envelope.context['external_assertion']`
and the immediate-parent-first list in `external_parents`. For gateway/MCP:

- `x-kognita-agent-assertion`: JSON signed packet.
- `x-kognita-agent-parents`: JSON parent packet array, when delegated.
- `x-kognita-instruction-attestation`: stored instruction attestation digest.

Headers carry proofs only. They do not change the authenticated principal,
agent workload credential, delegated authority or approved tool manifest.
Malformed or oversized proof headers fail closed. Kognita identity headers are
not forwarded as provider credentials.

## Updates and software supply chain (A4, A9)

`approve_update` requires an exact candidate deployment digest, provider change
identifier, regression evidence reference, rollback target/plan and accountable
owner. Before `approve_deployment` installs it, a current agent/system evaluation
must match the agent version, model version and candidate deployment digest.
Populate the optional `EvaluationResult.deployment_hash` for this evaluation.
An earlier evaluation for a different behaviour is insufficient. The packet's
previous-deployment binding prevents reuse after another change.

`approve_components` records unique package names/versions, SHA-256 SBOM/lockfile
digests, optional `sha256:` image digests and provenance references. The component
set digest is attached to the exact deployment/policy binding. `advisory` accepts
an operator-supplied compromised package/version and returns every affected
agent/deployment/use case. Required provenance admission quarantines matches;
an agent cannot remove the advisory. Approve a remediated component set/version
and deployment rather than clearing the advisory. Package scanning and artifact
signing remain external.

## Outcomes and KYA exports (A1, A3)

`record_outcome` accepts a bank-supplied `ProductionOutcome` with a unique source
outcome ID, linked interaction, intended and actual outcome, client impact,
complaint/correction/reversal and human override/escalation. It never infers
success from a completed run. The interaction must belong to the configured use
case. Duplicate source IDs are refused. The configured adverse-outcome count in
the rolling window emits a finding and a revalidation trigger; optional
`outcome_step_down` reduces autonomy. `assurance_report` shows evaluation results
beside recorded production outcomes and findings.

Outcome registration acquires transaction-level write serialization before reading
owner, policy, identity or outcome history. PostgreSQL uses the same advisory lock
as evidence appends and requires the READ COMMITTED isolation level. Database
reads after waiting see committed state; outcome registration explicitly refreshes
retained identities, interactions and evidence rows before validation. Pending
caller changes are flushed after locking and before refreshing, even when
autoflush is disabled. SQLite reserves its single writer before these reads. The lock
lasts until the caller commits or rolls back, including across the nested savepoint
that makes the outcome, finding, revalidation and autonomy reduction atomic.
Keep these administrative transactions short. An incomplete required evidence
write refuses the operation and rolls back the entire safety transition.

`export_manifest` returns JSON plus an Ed25519-signed digest for stable identity,
behaviour version, owner/principal/use case, live authority and expiry, autonomy,
client impact, permitted tools/data/jurisdictions, dependencies and kill mechanism.
`verify_manifest` takes an independent public PEM. `compare_manifest` compares
identity, live authority and observed dependency evidence, producing findings for
divergence. No external registry is created.

## Drills and supervisory reports (A8, A12)

`simulate` requires an admitted baseline request and runs eight offline hostile
scenarios through the real decision/record path: injected tool escalation,
discovered credentials, unapproved network tools, repeated API probes,
compromised-provider containment, child-agent escalation, malicious retrieved
instructions, and emergency containment/recovery. It does not execute malicious
tools or network traffic. Containment uses a simulated kill-switch snapshot,
without stopping real in-flight work. Savepoints restore database state; retained
transcripts can be checked against the baseline chain. Simulated path events are
not sent to external telemetry sinks; only the final drill record is published.
These are governance-path drills, not an external sandbox penetration test.

`conformance_report` supports versioned `MAS`, `HKMA-ART`, `FSB`, `OCC`, `SEC`
and `FCA` theme profiles. Reports reconstruct evidenced identity at the requested
time and cite evidence sequences, implemented controls, boundary evidence and
gaps. A policy without operational evidence leaves a gap. Expired/mismatched
attestations and quarantined component sets are not counted. The profiles are
theme-level mappings from the sources in `docs/control-frameworks.md`, not a
paragraph-by-paragraph legal opinion, certification, or a claim that absent
evidence is compliance. Updating a profile never changes admission policy.

## CLI

All commands use `--db PATH`. JSON administrative inputs come from `--input`;
write commands require an authenticated local `--actor`. On a signed evidence
store, also supply `--evidence-signing-key PATH` for the active chain signing
key. It signs the evidence event and is independent of the manifest
`--signing-key`, which signs the exported document. Omitting an evidence signer
on a signed chain is a refusal; unsigned stores can continue without this flag.

```sh
kognita assurance configure --db bank.db --agent briefing-agent --actor owner --input policy.json
kognita assurance attest --db bank.db --agent briefing-agent --actor bank-runtime --input signed-attestation.json
kognita assurance components --db bank.db --agent briefing-agent --actor owner --input components.json
kognita assurance update --db bank.db --agent briefing-agent --actor owner --input update.json
kognita assurance outcome --db bank.db --agent briefing-agent --actor owner --input outcome.json
kognita assurance advisory --db bank.db --actor security-operator --input advisory.json
kognita assurance manifest --db bank.db --agent briefing-agent --actor owner --signing-key private.pem
kognita assurance coverage --db bank.db --agent briefing-agent
kognita assurance report --db bank.db --agent briefing-agent
kognita assurance conformance --db bank.db --agent briefing-agent --profile MAS --as-of 2026-11-07T12:00:00+00:00
```

For `compare`, supply the manifest export as `--input` and an independent
`--public-key`. For `drill`, the input contains `scenario` and `envelope` fields.
Secrets and private signing keys are not stored in assurance records.
