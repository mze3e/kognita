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
configuration, including autonomy and review date. `registry.approve_deployment()`
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
