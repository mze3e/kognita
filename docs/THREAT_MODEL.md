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
