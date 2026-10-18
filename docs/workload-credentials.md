# Agent workload credentials

AI gateway and MCP proxy agent calls require the `agent_credential` header,
following the existing `agent_name` header convention. Header names are case
insensitive. A secret must match the named agent and the configured client must
admit that agent. Missing, invalid, mismatched and revoked secrets produce a
DENY with rule `S2-CREDENTIAL`. A configured system trigger alone cannot bypass workload authentication.

```python
from kognita.credentials import issue, verify, revoke

# Use the registry owner's authenticated administrative workflow.
secret = issue(session, "research-agent", evidence=evidence, actor_id=owner)
session.commit()
# Deliver secret to this agent's secret store once, without logging it.
# Send agent_name: research-agent and agent_credential: <secret> on calls.
revoke(session, "research-agent", secret, evidence=evidence, actor_id=owner)
session.commit()
```

Only SHA-256 digests are stored. Each issue creates an independent random
secret. Issue and revoke evidence commits in the same transaction as the
credential change. Verification reads the digest's primary-key index on every
request; revocation applies on the next request after commit. Workload secrets
are consumed at the gateways and never forwarded to providers or MCP backends.
Provider and backend authentication belongs in gateway configuration.

Use-case registration and revision accept `deployment_mode="gateway"` or
`"library"`. Tier 2 and above reject library mode. Existing records retain a
nullable mode until explicitly updated. A mode change requires material-change
review through the existing use-case lifecycle.

Agent manifest and environment declarations containing provider keys create
`provider-key finding` records available through `review_flags()`. Values are
redacted before declaration evidence is stored. This checks declared
configuration; network egress and evidence-store isolation remain deployment
controls managed outside the agent process.
