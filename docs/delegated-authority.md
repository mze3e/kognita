# Delegated authority

Agent calls require a live human capability grant and a live delegation. The
human grant is an entitlement recorded by the deployment's trusted
administration boundary. Envelope roles, administrator flags and context never
create one. Authenticate the issuer before calling these library APIs; do not
expose grant issuance directly to agents.

```python
from datetime import timedelta
from kognita.authority import Delegation, grant_to, delegate
from kognita.models import utcnow

at = utcnow()
scope = {"client": ["123"], "portfolio": ["portfolio-7"]}
grant_id = grant_to(
    "alice", "read", timedelta(hours=2), scope,
    session=session, evidence=evidence, actor_id="entitlement-admin", now=at,
)
delegation_id = delegate(
    session,
    Delegation(
        human="alice", agent="portfolio-agent", purpose="meeting-preparation",
        client="123", allowed_actions=["read"], allowed_data=scope,
        valid_from=at, valid_until=at + timedelta(hours=1),
        jurisdiction="SG", approval_threshold="read", task_ref="meeting-123",
    ),
    evidence=evidence, actor_id="alice", now=at,
)
```

The agent still needs a reviewed, approved identity and tool permissions.
Capabilities are exact verbs: `read`, `analyse`, `recommend`, `draft`, `queue`,
`send`, `execute`. An entitlement for one verb never grants another. Scope maps
subject types to explicit identifiers; no wildcard matching is performed.
Delegations bind one client and may reference additional data subjects. Every
allowed action, data subject and the entire delegation time window must fit a
live human entitlement when the delegation is issued. A delegation defaults to
one hour. Time windows include their start and exclude their end.

Set `Envelope.delegation_id` in library calls. Gateway and MCP requests carry
that same ID in the `delegation_id` header, alongside the agent's workload
credential. The ID selects authority; possession alone does not authenticate
the human or agent. The decision checks principal, agent, purpose, client/data,
jurisdiction, time and the tool's approved verb against it. Additional policy
checks continue to apply. Actions strictly above `approval_threshold` in the
verb order require human approval; an unrelated denial still wins.

`limits` maps numeric tool argument names to nonnegative maxima. Missing,
negative, nonnumeric or excessive arguments deny the request. Supply explicit
subjects on the envelope for all referenced data. Tools and domain packs must
resolve their actual data access from these governed references and arguments.

Use `revoke(session, delegation_id, evidence=..., actor_id="alice")` to revoke
a delegation; the same function revokes a grant using its ID and original
issuer. `complete_task(session, "meeting-123", evidence=..., actor_id="alice")`
ends all of Alice's delegations for that task. Fresh snapshots observe these
changes on the next decision, including approval release checks. Library users
must load a fresh snapshot for each live decision; retained snapshots are for
replay and hypothetical evaluation.

Callers commit or roll back authority state and evidence in the same
transaction. Issuance, use, revocation, task completion and observed expiry are
append-only evidence events. Decisions pin grant IDs and include the delegation
ID in the hashed envelope. Replay reconstructs authority from that evidence,
even after current authority has ended. Expiry is recorded when the authority
is next observed through an evidenced snapshot or recorded decision, rather
than by a background scheduler.

Authority reads use indexed grantee lookups and load at most 1,001 rows. An
authority set exceeding 1,000 active records fails closed instead of granting
permissions from a truncated set.
