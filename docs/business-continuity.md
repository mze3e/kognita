# Business continuity boundary

Business continuity is the bank's. Kognita does not keep relationship manager work running when AI is unavailable.

## What Kognita supplies

- **Its own failure mode.** The gateway fails closed by default, with an optional degraded mode (0.3). Failure drills (`kognita drill simulate`) check that the configured mode and evidence hold for eight failure kinds.
- **Governed failover.** A fallback model is used only when it is approved for the use case, and the switch is evidenced.
- **The dependency map and concentration report.** `kognita deps map|check|concentration` show which critical use cases rest on which provider, cloud, framework, vector platform and orchestration layer.
- **Incident evidence.** Outages, degraded periods and provider failures are recorded as incidents with timelines.

## What the bank owns

- Keeping RM work running without AI, for example falling back to CRM and research access.
- Recovery objectives, continuity plans and their testing.
- Database replication and infrastructure availability.
- Provider contracts, exit plans and incident disclosure.

The dependency map and drills inform the bank's continuity design; they do not replace it.
