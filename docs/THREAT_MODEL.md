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
shares the host's trust boundary. In-flight controls are described below.


## S6c: Agent and gateway threats

Kognita evaluates explicit authority before releasing a governed action. Model
instructions, retrieved documents and tool arguments cannot confer authority.
This model covers a deployed agent malfunctioning or being compromised, and an
external actor attacking an agent, gateway or MCP endpoint.

| Threat | Our agent fails | External actor attacks | Controls and evidence |
| --- | --- | --- | --- |
| Prompt injection | Treats retrieved instructions as permission to send or execute | Places instructions in a document, message or tool result | Sticky external taint, tool allow-lists, independent per-verb authority, client scopes, public-content lane separation; S5 injection alerts and denied decision evidence |
| Credential theft | Shares a workload secret or exposes a provider key | Replays a stolen credential or claims another agent's name | Per-agent hashed revocable workload credentials, gateway verification on every call, deployment binding, provider-key findings; credential lifecycle and authority lineage evidence |
| Malicious tools | Invokes an unsafe or irreversible capability | Backend lies, stalls, or continues after containment | Approved tool declarations, required autonomy, irreversible-action human approval (S6C-IRREVERSIBLE), finite proxy deadlines, cancellation and reverse compensation; S6C-IN-FLIGHT outcomes and S6C-RECOVER actions |
| Poisoned data | Mixes clients or relies on stale or hostile retrieval | Poisons indexed content, attachments or source responses | Dual human/agent entitlements, purpose and subject scopes, field minimization, external/client-data taint, pack freshness and source rules; cited decisions, data lineage and alerts |
| Autonomous reconnaissance | Enumerates clients, probes tools or loops | Uses an external agent to enumerate endpoints and data | Lowest-trust external principals, narrow temporary delegations, burst and daily limits, anomaly quarantine and credential revocation; anomaly incidents, limits and refused calls |
| Automated exploitation | Attempts privilege escalation or chains write tools | Probes policy bypasses, execution endpoints or compromised tools | Fail-closed precedence, independent tool verbs, attenuating delegation, assurance and transaction-bound step-up, gateway deployment requirements and sandbox constraints; decision citations, authority reconstruction and intervention evidence |

### Trust boundaries

The agent, model output, external documents and backend tool output are outside
the enforcement boundary. The gateway, MCP proxy, policy store, registry and
evidence writer are trusted. Administrators deploy the proxy with its credentials
and evidence store outside the agent's process and filesystem access. Enterprise
routing must prevent direct agent access to providers or backend systems.
Library execution cannot contain arbitrary code that shares its process.

An authenticated human is accountable for delegation and governed changes.
Authentication does not itself grant a tool capability. Revoked, expired,
out-of-scope or changed deployments remain denied even after an earlier approval.
DENY outranks ESCALATE, which outranks HUMAN_APPROVAL, which outranks ALLOW.

### Stopping and recovery

Every registered tool has a positive finite `max_duration` in seconds (default
30). MCP waits no longer than that duration or the remaining run wall-clock
budget. An upstream timeout or exception is recorded before returning. Late
results are withheld. The default HTTP transport shuts down its request socket on cancellation or timeout, including while reading a response body. This aborts the connection and withholds late results; it does not prove that a remote side effect stopped.

Transports supporting confirmed cancellation expose the private adapter hook
`_cancel(rpc_id)`, returning `True` only when cancellation is confirmed. Kill
switches, credential and delegation revocation, task completion and run pauses
signal active proxy calls in the same process; indexed live-state checks also
observe committed changes made by other processes. The proxy commits admission
evidence before contacting the backend so SQLite does not block interventions. Evidence distinguishes completed,
cancelled, cancellation_requested and timeout outcomes. Completion winning a
cancellation race is recorded as completed. Cross-process control changes must
be committed; the local callback mechanism is not a distributed cancellation bus. Cancellation of a remote side effect must be authenticated and correlated to the specific request, never inferred from a closed socket. Default transport cancellation evidence confirms local connection abortion only; custom transport hooks must define their confirmation semantics.

Mutating MCP tools without cancellation support must declare `reversible=True`
and a compensation tool name in the approved manifest. Otherwise admission fails
with S6C-UNCANCELLABLE. Long operations can instead expose bounded cancellable
steps, each independently governed. Compensation must be idempotent and undo only
the associated action. For library tools, `compensation` is a callback accepting
the original envelope, evaluation and session. Its function name must match the
manifest's compensation identifier; private proxy adapters bind the backend tool
name instead.

Recovery stops new calls and executes registered compensations in reverse
registration order. Every attempt records completed, failed or unavailable;
previously completed compensation records are not rerun. Compensation context is
retained by hash, separately from evidence. A restarted library process supplies
its trusted tool registry through the private `_registry` recovery integration.
Missing executable code or erased retention yields unavailable, never a claim
of successful rollback. Recovery is a trusted human intervention; it does not
create broader agent authority. External side effects and the evidence database
are not a distributed transaction, so operators must reconcile interrupted or
uncertain calls with backend action identifiers before declaring recovery done.

### Conformance and remaining assumptions

`kognita.testing._adversarial` supplies the nine-scenario pack conformance runner.
Pack-specific envelopes exercise prompt injection, wrong-client data, stale
information, malicious documents, poisoned retrieval, excessive tool permissions,
privilege escalation, loops and policy bypass. All scenarios must be provided;
every result must deny or escalate, cite a rule, and have verifiable decision
evidence without a tool release. Human approval alone does not pass this suite.

Kognita does not validate the truth of every backend response, implement a
sandbox, authenticate human channels, or provide network isolation. Deployment
owners implement those boundaries. Freshness and approved-source semantics are
pack rules; taint and access control contain poisoned input but do not establish
that its factual content is correct. A compromised policy administrator, gateway
host or backend capable of ignoring cancellation remains a trust-boundary breach.

### Shipped-pack ingress attack coverage

`tests/test_shipped_pack_ingress_attacks.py::test_shipped_pack_nine_attacks_through_gateway_and_mcp`
runs all nine scenarios over HTTP through the actual Gateway and McpProxy for
every entry in `PACK_NAMES`: all five starter packs and private-banking. None
is excluded. Policies are compiled from the shipped files without rule overlays.
A registered COLLABORATION use case supplies the agent integration context;
pack policy restrictions remain active. Real workload credentials, entitlements
and delegations are issued. Each case requires the attack-specific failing
check as well as a cited DENY/ESCALATE, persisted decision, intact evidence chain,
and absence of tool release or egress. These checks prevent unrelated pack
denials from hiding a missing attack control.

| Scenario | Implemented control checked by that test |
| --- | --- |
| Prompt injection | S5-INJECTION on model prompts and MCP arguments |
| Wrong-client data | S3-DELEGATION on another client's MCP subject or separately scoped AI model endpoint |
| Stale information | S3-DELEGATION rejects an expired authority window before information access |
| Malicious documents | S5-INJECTION on instructions embedded in document text |
| Poisoned retrieval | S5-INJECTION on instructions embedded in retrieved text |
| Excessive tool permissions | S1 tool manifest admission for MCP; S3 model scope for AI endpoints |
| Privilege escalation | Bound principal mismatch refuses caller-supplied identity |
| Looping agents | Persisted run max_calls budget, including alongside other policy denials |
| Policy bypass | Bound purpose refuses an unregistered purpose despite a claimed authorization |

This suite tests containment of stale-authority access and instruction-bearing
retrieval. It does not establish factual freshness or detect all false content;
domain freshness rules and trusted source validation remain necessary.

`tests/test_mcp_http_cancellation.py::test_default_http_transport_closes_inflight_connection`
uses a real upstream HTTP server and the default transport. It checks kill-switch
and max_duration termination before headers and during body reads, observes EOF
at the upstream socket, verifies S6C-IN-FLIGHT evidence and withheld results,
and separately checks successful completion.
