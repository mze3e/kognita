# Incident reports and playbooks

The bank owns incident response, recovery and disclosure. Kognita supplies local
operational guidance through `kognita incident playbook '<type>'`. Each playbook
has detect, contain and investigate steps and cites R4. The fixed types are:

| Type | Detection focus | Containment focus | Investigation focus |
| --- | --- | --- | --- |
| hallucination | Claim verification and screening | Circuit breaker, withhold output | Claims, source versions, affected clients |
| data leakage | Access denials and releases | Revoke authority, stop use case | Released outputs and affected clients |
| prompt injection | Isolation and instruction checks | Revoke tool access, circuit breaker | Source provenance and authority lineage |
| unauthorized tool execution | Tool authorization failures | Revoke agent, stop tools | Calls, approvals, affected clients |
| incorrect transaction | Approvals and outcomes | Stop agent, bank reversal process | Authorization and transaction evidence |
| model outage | Provider and gateway timelines | Stop or governed failover | Timelines, versions, deployment changes |
| third-party compromise | Provider incidents and model changes | Revoke provider, stop dependent use cases | Affected clients and provider evidence |
| deepfake instruction | Authenticity and authority checks | Revoke authority, independent approval | Instruction provenance and approvals |
| sensitive-data exposure | Sensitive attributes and field checks | Revoke access, withhold output | Released fields and affected clients |

`kognita incident report <id> --db PATH` verifies the evidence chain before
building an R14 JSON report. It includes the incident timeline, affected-client
query results, recorded control checks and failures, evidence gaps, and deployment
changes including identity changes. Corrupted evidence refuses report generation.

The scope is all recorded activity from detection until closure, or the report
snapshot for an open incident. Temporal association does not establish causation.
Affected clients are recipients according to the existing affected-client query;
unlinked outputs remain explicit. No expected-control inventory exists, so the
report explicitly marks that gap instead of treating absent controls as passed.
Deployment comparisons use the latest approval before the report cutoff and its
preceding approval as baseline, and mark a missing baseline. Approvals before
detection are labeled explicitly. Neither these changes nor changes during the
window establish the cause of the incident.

Add `--redacted` for a report containing only fixed labels and aggregate counts.
It omits identifiers, timestamps, free-text summaries, check identifiers, source
endpoints, deployment bodies and all client content. It does not load retained
content. The report and playbook commands do not execute containment actions or
send disclosures.
