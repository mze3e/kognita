# Content isolation and instruction authenticity

Kognita's S5 boundary assumes the host authenticates people and agents, owns the
registered tool implementations and deployment manifest, and protects the
evidence database. Model output, channel messages, tool arguments and retrieved
text cannot issue grants, approve deployments or create authentication evidence.
Library mode shares the host's process; it cannot defend against arbitrary code
that can modify its own enforcement code or evidence store. Gateway deployments
must keep those capabilities outside the agent process.

| Threat | Enforced control | Rule |
| --- | --- | --- |
| A retrieved document instructs an agent to modify internal records | Sticky `external` taint denies draft, queue, send and execute, and denies tools requiring more than L1 | `S5-EXTERNAL-AUTHORITY` |
| Client information reaches a public publishing tool | Sticky `client_data` taint denies tools registered with `public_content=True` | `S5-PUBLIC-CONTENT` |
| An internet reader also receives internal write permissions | Identity registration and deployment approval reject conflicting duties | `S5-SEPARATION` |
| Recognizable injection instructions enter a request or tool result | The pattern classifier detects them; append-only injection alerts name security and, when available, the agent's owner. Tool requests containing detected instructions are denied | `S5-INJECTION` |
| A message claims to authorize a high-risk action | L4+ tools and queue/send/execute require authentication assurance of at least 2 and exact transaction-bound step-up evidence | `S5-STEP-UP` |
| An agent relays an instruction with increased assurance | A run's assurance ceiling can only decrease, including across gateway calls and approval continuations | `S5-ASSURANCE-RELAY` |
| Tool provenance is changed after deployment approval | Runtime tool metadata must match the approved manifest, including its content labels and lane | `S1-TOOL-DECLARATION` |
| An external-data tool is invoked without a run | Explicit external/client-data tools require a run so their results can retain provenance | `S5-RUN-REQUIRED` |

## Host integration

Use one `Run` for the whole task, including model calls, tool calls, document
retrieval and later approvals. Gateway and MCP calls share its `run_id`. The
trusted host must preserve task identity; a client able to start unrelated tasks
must not be allowed to relabel an existing task as new to discard its provenance.

Register tool provenance using `external=True` or `client_data=True`, and the
public lane using `public_content=True`. Include the same flags in each agent's
approved tool allow-list. Defaults represent an internal tool in the client lane;
they do not make its output public. C2/C3 tool results automatically mark client
data, even after purpose-specific field minimization. Client references and
classified request arguments also mark client data before tool execution.

Retrieval accepts `run=...` and conservatively marks returned documents external
by default. A trusted host can use `external=False` for a known internal corpus;
C2/C3 results still mark client data. The broker passes its run into retrieval,
including retrieval resumed after approval. Other document ingress paths use the
private `_ingest` adapter at their trusted content boundary. Unknown provenance
must be labelled external by the host. Labels are stored in additive nullable
SQLite columns, merged monotonically, and evidenced before results are released.

`Envelope.assurance_level` is an integer from 0 to 3; zero is the default. A header
with the same name carries a claim, not proof. For high-risk calls the trusted
host uses the private `_step_up` adapter after its authenticated workflow obtains
transaction context and authorization from a different human. The record binds
the complete envelope (including principal, agent, delegation, arguments and
assurance), has an expiry, and is checked again on approval release. Changing
arguments or asserting authentication inside message text cannot satisfy it.
Approval of the action and authentication of its requester are separate gates.

## Limits and residual risks

Pattern-based detection recognizes a limited set of injection forms. Taint,
authority, purpose, client scope, deployment and tool checks remain necessary
when a malicious instruction evades detection. Alert events are durable records;
the host's alert consumer is responsible for delivering them to their recipients.

A tool's declared provenance and permissions must describe its real behavior.
The library cannot inspect a tool body to prove that it obeys those declarations.
This slice does not provide sandbox execution, credential isolation within a
shared process, automatic fleet containment or cancellation of in-flight calls.

## Behavioral containment

The S6a monitor keeps an indexed daily summary per agent and reads at most eight
rows: today and seven previous active days. It tracks distinct client references,
documents released by the governed broker, attempted actions, denied actions,
tools, tool transitions and observed data classifications. Cold-start ceilings
are 100 distinct clients, 1,000 actions and 1,000 documents per UTC day. After
three historical days, volume is capped at three times the largest historical
day, with a floor of ten and the cold-start ceiling still applying. The current
day cannot raise its own baseline. Three requests for undeclared tools or ten
denials trigger containment. Once warm, unseen tools, transitions and data
classes also trigger containment. These conservative defaults require reviewed
redeployment and operational calibration before production use.

| Threat | Automatic response | Rule |
| --- | --- | --- |
| Client enumeration, action loops or bulk document extraction | Deny the crossing request and quarantine | `S6-VOLUME` |
| Tool privilege probing or repeated denied requests | Quarantine before further tool admission | `S6-PRIVILEGE-PROBING` |
| A changed tool workflow or data classification | Deny and quarantine | `S6-TOOL-SEQUENCE`, `S6-DATA-CLASS` |
| Reuse of a stale decision after quarantine | Fresh kill-switch lookup at recording denies release | `S6-QUARANTINE` |

Containment sets the kill switch, calls credential revocation for all the agent's
credentials, stops new tool admissions, records an incident and then appends an
alert addressed to the accountable owner. The triggering denial counts toward
the circuit breaker; repeated quarantine denials do not multiply that flag.
All updates and evidence share the caller's transaction. Hosts must commit a
denied request's transaction to retain containment. Alert delivery uses the
host's evidence consumer. Only the accountable human owner may release an anomaly
quarantine; old credentials remain revoked after release.

This monitor cannot observe direct network requests, direct provider calls or
unregistered document ingestion. It does not cancel synchronous transports that
were released before containment. In-flight cancellation and sandbox network
constraints require their respective enforcement adapters. Gateway isolation
and the existing authority controls remain necessary against an attacker who
stays below behavioral thresholds or compromises the host itself.

## Agent handoffs, fleet boundaries and execution constraints

The trusted host calls `Run.delegate_to(agent_name, attenuated_scope=..., session=...,
evidence=...)` after the parent's first governed call binds its identity. The
scope supplies `why`, `allowed_actions`, `allowed_data`, `tools`, `autonomy_level`,
`valid_until`, `max_calls`, `max_tokens`, `max_cost_usd` and `wall_clock_seconds`.
The child must fit inside the parent's live human delegation, approved tools,
autonomy ceiling and remaining budgets. All three numeric budgets must be bounded
on the parent. Reservations immediately reduce parent capacity; unused child
capacity is not automatically refunded. Transactions must be committed by the
host; a rollback removes both the reservation and its evidence.

The returned child run's private `_authority_scope` contains its delegation ID
for the host's envelope or gateway `delegation_id` header. Use that child run's
ID as `run_id` on MCP/model calls. Identity bindings include agent, human, purpose
and delegation. A delegated worker cannot move its authority into a fresh run,
reuse its parent's run, or change the bound identity. Child runs inherit content
labels, authentication ceilings, classification ceilings and tool restrictions.
A paused parent cannot delegate. Subsequent parent delegation revocation, human
entitlement revocation, expiry and task completion deny descendant decisions.
Each hop is revalidated through primary-key lookups with a maximum depth of 32;
oversized authority sets fail closed.

Expose worker agents as registered MCP capabilities, with the worker endpoint
configured as the backend. Requests then use the same `tools/call` admission,
workload credential, manifest tool permission, classifier, taint, quota and
evidence path as every other tool. Message text supplies data, never authority.
The host authenticates the worker endpoint and binds its response provenance;
Kognita does not discover or trust endpoints named in messages. External callers
need their own registered identity and credential and a run. They are limited
to L0 reads with assurance zero, and their messages mark the run `external`.
The MCP proxy does not forward caller workload secrets to worker backends.
Per-agent blast-radius quotas aggregate across runs; run budgets remain separate.

`kognita evidence authority <action_id> --db <path>` uses the decision request ID
and verifies the evidence chain before reconstructing the human grants and each
orchestrator/worker hop. It reads immutable issuance evidence, so revocation or
later registry changes do not rewrite historical authority. Missing issuance
records and excessive or cyclic lineage fail closed.

| Threat | Enforced control | Rule |
| --- | --- | --- |
| Workers multiply the orchestrator's budget or broaden its privileges | Reserve child budgets and attenuate data, clients, actions, tools, autonomy and expiry | `S6B-ATTENUATION`, `S6B-DELEGATION` |
| A worker reuses another task's run to launder provenance | Bind each run to its identity and reject foreign-run writes before ingestion or budget consumption | `S6B-RUN-ISOLATION` |
| An external agent sends instructions as apparent internal authority | Authenticate workload credentials, cap authority at L0 and taint messages external | `S2-CREDENTIAL`, `S3-DELEGATION`, `S5-TAINT` |
| Code execution receives ambient machine privileges | Emit explicit constraints with restrictive defaults and evidence them with the decision | `S6B-SANDBOX` |

For an `execute` tool, the approved tool permission may contain private `_sandbox`
configuration. The private constraints model rejects unknown keys and invalid
resource bounds. It emits `credentials` (credential references), `network`
(destinations), `fs_scope` (bounded absolute paths), `permitted_apis`,
`permitted_commands`, `data_export`, `persistence`, `auto_stop` (seconds), `cpu`
(cores), `mem` (MiB) and `disk` (MiB). Empty lists deny all access; defaults are
no export or persistence, a 30-second auto-stop, one CPU, 256 MiB memory and
64 MiB disk. Explicit allow-lists reject wildcard entries. An ALLOW carries
these constraints in decision attributes under private `_sandbox_constraints`,
bound into its hash and append-only evidence. A later denial removes that output.

The executing host must route execution through an independently enforced
sandbox, honor every constraint, resolve credential references outside the
agent, and enforce network destinations and filesystem paths after resolution.
Kognita emits policy output; it ships no sandbox and cannot prove that a host
obeys it. Filesystem symlinks, DNS changes, backend authentication and direct
process/network access remain the deployment's responsibility. Library mode
shares the host's trust boundary. No in-flight cancellation is added here.
