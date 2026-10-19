# Agent identity and autonomy (S1)

Existing databases migrate when `create_all()` runs. New identity columns are
nullable and added with `ALTER TABLE ADD COLUMN`. Existing agents retain their
records, have effective autonomy L0, and cannot act until their identity,
permitted systems, tools and deployment are configured and approved.

Deploy a JSON or YAML manifest with:

```sh
kognita agent deploy --manifest agent.json --db kognita.db --actor owner@example.org
```

The local operator identity defaults to the manifest's accountable owner. This
command is an administrative operation on the governance store. It deploys at
L0; increasing an existing agent's autonomy requires `kognita.autonomy.promote()`.

```json
{
  "name": "AGENT-RESEARCH-01",
  "owner_exec": "owner@example.org",
  "purpose": "Explain approved research for meeting preparation",
  "use_cases": ["research-summary"],
  "permitted_systems": ["research"],
  "tool_allowlist": {
    "read_research": {
      "verb": "read",
      "required_level": 0,
      "permissions": ["read"],
      "system": "research"
    }
  },
  "autonomy_level": 0,
  "review_date": "2027-01-01T00:00:00+00:00",
  "deployment_date": "2026-10-19T00:00:00+00:00",
  "model": "research-model-v1",
  "environment": "production",
  "principal": "rm@example.org",
  "deployment": {
    "system_prompt": "Explain approved research with citations.",
    "instructions": "Use the research system.",
    "tools": [],
    "permissions": ["read_research:read"],
    "data_sources": ["approved-research"],
    "orchestration": "retrieve then explain",
    "memory_behaviour": "no persistent memory"
  }
}
```

The linked use case must also be registered and active under the existing
use-case controls. Tool permissions are exact names and verbs. `recommend`
never confers `execute`. Register tool declarations with `required_level` and
`verb`; the runner checks these against the agent's approved tool record.
Level requirements have floors: read 0, analyse 1, recommend 2, draft 3, and
queue/send/execute 4. Requirements of L5 or L6 represent autonomous actions.
An L4 requirement always withholds execution until human approval. The
use-case maximum and existing tier ceilings also apply (tiers 1/2/3/4:
L2/L3/L4/L4).

The deployment hash binds all seven behaviour components and the authority
configuration, including autonomy, review date and `Agent.model`. `registry.approve_deployment()`
requires the accountable owner and appends approval evidence. Changed behaviour
or authority and expired reviews deny with `S1-*` rule citations. The approved
manifest is kept under the private `_deployment` entry in `tool_allowlist`;
callers must preserve that entry when editing tool permissions. Behaviour
supplied in `Envelope.context["deployment"]` is compared with the approved hash;
it cannot grant authority. Library integrations supply their actual current
behaviour manifest here when runtime configuration differs from the registry.
The AI gateway checks system/developer messages (joined with newlines), provider
tool declarations and supplied instructions against the stored manifest on every
call. Administrative manifests describe the other runtime components.

`Agent.deployment_date` and `Agent.model` are nullable database columns for
SQLite 3.31 and Postgres; existing rows stay null. Both must be set at registration
or deployment approval: either missing (or a blank model) denies with
`S1-IDENTITY`. Changing the model denies with `S1-DEPLOYMENT` until the owner
re-approves the deployment.

Identity changes through `registry.register()` accept the identity fields with
`evidence` and `actor_id`. These changes invalidate approval and cannot raise
an agent's level. Deployment and autonomy changes include identity snapshots
in append-only evidence, so replay uses the authority that applied at the
original decision. Approval continuations recheck current authority, review
expiry, deployment hash and tool declarations before releasing work.

Promotion gates are private configuration on `UseCaseSpec._autonomy_gates`,
keyed by destination level as a string. Each gate supplies `minimum_sample`,
`window_days`, `quality` and `reliability`; `exception_rate` defaults to 0.05.
For example, `{"1": {"minimum_sample": 100, "window_days": 30,
"quality": 0.95, "reliability": 0.95}}`. Configuration is included in the
versioned use-case record and evidenced through use-case registration/revision.
Changing these gates is a material revision that clears activation.

`autonomy.promote(session, name, evidence=writer, actor_id=owner,
validator=independent_validator)` recomputes the 0.4 outcome metrics for the
agent's linked interactions in the observation window. It requires sufficient
samples, a deployment approved before the window, complete action evidence,
no unfinished runs, acceptable exception rates and consistent model/prompt
versions. The validator must be on the use-case validator list, differ from the
owner and not be a builder. Every linked use case must pass its gate and risk
cap. Failed evidence gates cannot be overridden by signatures.

`UseCaseSpec._autonomy_step_down` supplies the same sample/window/quality/
reliability fields with lower thresholds. Live gateway, tool and retrieval
paths evaluate three consecutive windows before deciding. Sustained threshold
breaches or strictly declining quality/reliability reduce autonomy by one level,
append the change, and append an `ALERT` addressed to the owner and Knowledge
Lead. Alert evidence is available to operational consumers; this slice does not
send external messages. A window permits at most one step-down, and restoration
requires promotion again. `autonomy.step_down()` runs the same metric gate;
callers cannot choose a target level. Pure `decide()` and historical replay do
not mutate the registry.

## Blast radius and field minimization (S4)

Manifests may include `blast_radius`, a JSON object matching
`kognita.models.BlastRadius`, and `field_allowlist`, a mapping from purpose to
explicit field paths. Both are nullable additive database columns. Changing
either through `registry.register()` requires owner evidence and invalidates
deployment approval. For example:

```json
{
  "blast_radius": {
    "max_clients": 1,
    "max_value": 0,
    "max_daily_actions": 100,
    "max_daily_transactions": 10,
    "products": ["approved-research"],
    "per_minute": 5,
    "per_hour": 30,
    "channels": ["crm"],
    "jurisdictions": ["SG"],
    "execution_authority": "none",
    "ttl": "2027-01-01T00:00:00+00:00"
  },
  "field_allowlist": {
    "CLIENT_PRECALL": ["risk_profile", "portfolio.rating"]
  }
}
```

`BlastRadius.max_daily_transactions` counts released calls whose approved tool
permission verb is `execute`, using atomic quota admission and the UTC calendar
day (`S4-DAILY-TRANSACTIONS`). Pending approvals do not spend transaction quota.
Other verbs do not spend this cap. `BlastRadius.products` checks the resolved
`product` subject and argument values, including nested values and private
manifest subject paths (`S4-PRODUCTS`). `None` means unrestricted; `[]` denies
any named product. These settings require the same owner approval as other limits.

Limits apply across runs. Minute and hour limits use rolling windows; daily
limits use the UTC calendar day. Client and monetary limits cover the agent's
recorded lifetime usage. Monetary calls must supply a finite, nonnegative
`arguments.value`; channels use `arguments.channel` and jurisdictions use
`actor_location`. `ttl` is an absolute expiry. Queue, send and execute require
execution authority `execute` as well as the existing tool, autonomy and
delegation checks. `none` and `propose` confer no execution permission.

Allowed decisions reserve quota, including calls awaiting human approval.
Approval release excludes its own reservation, rechecks current limits, and
moves the reservation to release time with evidence. Failed tool calls still
consume quota. Usage reads use an agent/time index and at most 10,001 rows;
reaching that bound denies with `S4-USAGE`. Limits are independent of Run
budgets and every failing gate cites its `S4-*` rule.

Governed tool results are projected before retention and release to the agent.
Nested dictionaries and lists require explicit leaf paths; allowing `portfolio`
does not expose its descendants. `$` permits a scalar result. Missing purposes
or declarations release no fields. Each projection appends `S4-FIELDS` evidence
without copying removed values into the evidence record. Tools absent from the
agent's allow-list are denied, including private manifest entries.

Delegations can bind access to an event through private numeric limits
`_event_at` (Unix timestamp) and `_access_window` (seconds, at most 86,400).
The default window is 24 hours. Access requires the event to be between now and
the end of that window, in addition to all existing delegation checks. Task
completion, expiry and revocation continue to terminate authority.
