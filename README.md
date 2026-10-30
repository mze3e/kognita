# Kognita

[![PyPI version](https://img.shields.io/pypi/v/kognita.svg)](https://pypi.org/project/kognita/)
[![Python Versions](https://img.shields.io/pypi/pyversions/kognita.svg)](https://pypi.org/project/kognita/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/mze3e/kognita/actions/workflows/ci.yml/badge.svg)](https://github.com/mze3e/kognita/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/kognita)](https://pepy.tech/project/kognita)

![Kognita Logo](docs/kognita-logo.png)

**Prove an AI action was permitted, and evidence it.**

Content guardrails check what an AI *says*. Kognita decides whether the agent was *allowed to ask*, before any data is retrieved, and writes a tamper-evident record of the decision and the rule behind it.

Your agent pulls a client's portfolio, summarises it, and sends it to a model API. Was it allowed to? With Kognita, that question has an answer you can show a regulator: which rule permitted it, which rules would have stopped it, and proof that the record hasn't been edited since.

```python
from datetime import datetime, timezone
from sqlmodel import Session

from kognita import Envelope, Policy, decide, load_snapshot
from kognita.db import create_all, make_engine

engine = make_engine()
create_all(engine)
start = datetime(2026, 1, 1, tzinfo=timezone.utc)

with Session(engine) as session:
    session.add_all([
        Policy(regime="BOOKING_CENTRE", rule_type="ATTRIBUTE_ALLOWLIST",
               rule={"allow": {"actor_location": ["SG"]}},
               citation="Cross-border manual s3.2: SG-booked clients served from SG",
               effective_from=start),
        Policy(regime="INVESTOR_STATUS", rule_type="REQUIRES_FLAG",
               rule={"flags": ["accredited_investor"]},
               citation="Product governance policy s7: complex products",
               effective_from=start),
    ])
    session.commit()

    evaluation = decide(
        Envelope(principal="rm@bank.example", purpose="PRODUCT_DISCUSSION",
                 tool="discuss_product", actor_location="HK",
                 subject_type="client", subject_id="123"),
        load_snapshot(session),
        attributes={"actor_location": "HK", "accredited_investor": False},
        purposes=["PRODUCT_DISCUSSION"],
    )

print(evaluation.outcome.value)
for check in evaluation.basis():
    print(f"{check.regime:16} {check.citation}")
```

```console
DENY
BOOKING_CENTRE   Cross-border manual s3.2: SG-booked clients served from SG
INVESTOR_STATUS  Product governance policy s7: complex products
```

Two independent rules refused. Neither masked the other, each names its source, and nothing was retrieved. The policy names and citations above are illustrative; in a real deployment they come from your own policy set.

## Install

```bash
pip install kognita
```

The core depends on four packages (`pydantic`, `sqlmodel`, `numpy`, `python-dotenv`) and runs with no network and no API key. Deciding whether a request is permitted should not require the machinery that answers it. `import-linter` contracts and a no-extras install test keep it that way.

Optional extras add provider-backed embedders (`kognita[openai]`), a SQLite vector index (`kognita[vec]`), local embeddings (`kognita[local-embeddings]`), structured document ingestion (`kognita[docling]`), and OpenTelemetry export (`kognita[otel]`). `kognita[docling]` parses markdown, HTML and DOCX. PDF is refused until section headings can be read without torch. One extra per framework adapter: `kognita[hermes-agent]`, `kognita[openai-agents]`, `kognita[langgraph]` and `kognita[pydantic-ai]`. The core does not import those frameworks or OpenTelemetry.

## What it does today

**Authorise before discovery.** An agent's intent is described as an envelope (who, for what purpose, with which tool, about which subject, from where) and evaluated before anything is fetched. A denial returns no data, not filtered data. Retrieval filters on zones and classification before scoring. An item with no zones is not visible in any zone.

**Fail closed.** Outcomes resolve as `DENY > ESCALATE > HUMAN_APPROVAL > ALLOW`. One failing check among a hundred passes still denies, so a policy set cannot be widened by adding permissive rules. A policy whose rule type has no evaluator escalates rather than being skipped. A missing or empty purpose list fails the purpose check.

**Use-case register.** `run_governed`, `ask`, the MCP proxy and the AI gateway always decide against the register. An empty register denies, and the denial cites the register rule. A passing check cites that use case's accountable owner. `decide` still uses the free-text purpose list when the caller does not pass a register; that check cites the agent's accountable owner, or the principal when the call names no agent. No owner fails closed. An approved model is a model card, and a call that uses one links that card. An unmediated entry records its assurance gap and compensating control; the gateway flags a call to an AI endpoint no use case names, and the flag does not decide the call. Retiring a use case stops agents registered only to it and applies its retention period. Tier floors and review periods are constants in `kognita.usecases`:

| Tier | Approval gate | Sampling | Autonomy ceiling | Explanation | Review |
| --- | --- | --- | --- | --- | --- |
| 1 | none | 5% | L2 | citation | 365 days |
| 2 | human review | 10% | L3 | sources | 180 days |
| 3 | human approval | 50% | L4 | suitability | 90 days |
| 4 | two signatures | 100% | L4 | full | 90 days |

Monitoring is required at every tier. On the review date the use case is denied. Activating it again waits until the designated control function approves the materiality rating at or after that date. A change of purpose or category clears activation, as does a change of autonomy, jurisdiction, client population, or intended use.

**Every decision cites its rule.** Each check carries the regime and citation it came from. A check without a citation is an assertion, not a decision, and the conformance kit enforces it.

**Agents are registered, and can be stopped.** An agent that is not in the registry is denied. Registration includes `deployment_date` and `model`, alongside its version, accountable owner and materiality tier. Missing identity denies with `S1-IDENTITY`; a changed deployment needs owner approval (`S1-DEPLOYMENT`). Each agent has a kill switch that denies its next request:

```console
DENY  Kill switch engaged — accountable owner: Head of Wealth Advisory
DENY  Agent inventory — 'AGENT-UNKNOWN' is not registered
```

Through `kognita serve`, a call needs an admitted agent identity or an approved system trigger. An agent name is accepted only when the bound client configuration lists it.

**Workload credentials bind calls to agents.** Gateway and MCP calls carry `agent_name` and `agent_credential` headers. Missing, mismatched or revoked secrets deny with `S2-CREDENTIAL`. Only digests are stored. Rotate by issuing a replacement and revoking the old secret; there is no separate rotation API. The owner performs these operations and commits the transaction:

```python
from kognita.credentials import issue, verify, revoke

secret = issue(session, "research-agent", evidence=evidence, actor_id=owner)
assert verify(session, "research-agent", secret)
replacement = issue(session, "research-agent", evidence=evidence, actor_id=owner)
revoke(session, "research-agent", secret, evidence=evidence, actor_id=owner)
session.commit()
```

**Delegation checks both parties.** A live human entitlement and a live agent delegation must cover the action, subjects and time window (`S3-HUMAN-ENTITLEMENT`, `S3-DELEGATION`). The delegation cannot exceed the human grant. Calls select it through `Envelope.delegation_id` or the `delegation_id` header. After the grant and delegation in [the authority example](docs/delegated-authority.md), task completion ends that authority:

```python
from kognita.authority import complete_task

complete_task(session, "meeting-123", evidence=evidence, actor_id="alice")
```

**Autonomy is evidence-gated.** Levels L0-L6 constrain approved tool verbs. Deployment starts at L0; L4 actions require human approval. `kognita.autonomy.promote` requires measured outcomes, owner approval and an independent validator. `kognita.autonomy.step_down` checks sustained drift and lowers the level one step (`S1-DRIFT`). Deploy the [identity manifest](docs/AGENT_IDENTITY.md), including its deployment date and model:

```console
$ kognita agent deploy --manifest agent.json --db kognita.db --actor owner@example.org
```

**Authority can be reconstructed.** Evidence links an action to its human grants, delegations and identity at decision time, including authority that later ended. Use the decision request ID:

```console
$ kognita evidence authority <action_id> --db kognita.db
```

**Blast radius is bounded across runs.** `BlastRadius` caps clients, value, daily actions, minute/hour rates, channels, jurisdictions and expiry. `max_daily_transactions` counts released `execute` calls on the UTC day (`S4-DAILY-TRANSACTIONS`). `products` restricts resolved product subjects and arguments (`S4-PRODUCTS`). These limits need deployment approval. For example, the [manifest limits](docs/AGENT_IDENTITY.md) include:

```python
from kognita.models import BlastRadius

limits = BlastRadius(max_daily_transactions=10, products=["approved-research"])
```

**Taint stays with the task.** External content blocks writes and tools above L1 (`S5-EXTERNAL-AUTHORITY`). Client data blocks public publishing (`S5-PUBLIC-CONTENT`). Labels persist across reloads and child runs. Declare external tool results at registration, as in the [isolation tests](tests/test_isolation_s5.py):

```python
registry.register("tool", lambda *_: {"document": "harmless"}, external=True,
                  classification=Classification.C0)
```

**Anomalies contain agents.** Volume, privilege probing, new tool sequences and data-class changes can quarantine an agent, revoke its credentials and stop work (`S6-VOLUME`, `S6-PRIVILEGE-PROBING`). Release requires an accountable human. Fleet delegation narrows authority, reserves child budgets and inherits taint and restrictions. With the bounded scope from the [fleet tests](tests/test_fleet_s6b.py):

```python
child = run.delegate_to("child", scope, session=session, evidence=evidence, now=NOW)
```

**System triggers need approval.** Gateway and MCP clients admit only configured triggers (`S2-SYSTEM-TRIGGER`). A trigger remains a system actor and cannot bypass agent credentials or policy. The [trigger tests](tests/test_system_triggers.py) configure admission with:

```python
from kognita.gateway import ClientConfiguration

client = ClientConfiguration(principal="alice", purpose="COLLABORATION",
                             system_triggers=frozenset({"nightly-refresh"}))
```

**Propose before applying.** `propose_document` records a diff without changing its target. `apply_proposal_tool` requires the exact persisted, live human approval and applies it once (`PROPOSAL-BINDING`, `PROPOSAL-APPROVAL`). With the proposal and approval from the [public API tests](tests/test_proposal_public_api.py):

```python
from kognita import propose_document, apply_proposal_tool

propose_document(session, proposal, evidence=evidence, correlation_id="public")
apply_proposal_tool(session, proposal, target, approval, diff_hash=digest,
                    approver_id="owner", actor_type=ActorType.HUMAN,
                    evidence=evidence, correlation_id="public", now=NOW)
```

**Recovery compensates released work.** Reversible library tools register a compensation callback. MCP write tools declare `reversible: true` and a named `compensation` tool in the approved tool allowlist. `recover_run` stops the run and compensates latest actions first; failed compensation leaves it paused, and retries skip completed actions (`S6C-RECOVER`). With the callbacks and attached interaction from the [recovery tests](tests/test_recovery_retry.py):

```python
from kognita.intervention import recover_run

registry.register(name, lambda *_: "ok", reversible=True, compensation=undo(name))
recover_run(session, run.id, evidence=evidence, human="human",
            citation="S6C-RECOVER", now=NOW)
```

**Decisions are pure and replayable.** `decide()` writes nothing and takes the instant as a parameter. Policies are effective-dated rows, so *"what would this have decided in March?"* has an answer:

```python
decide(envelope, load_snapshot(session, as_of=march), as_of=march, ...)
```

**Evidence is tamper-evident.** Each event carries the previous event's hash, so altering any payload breaks every hash after it:

```console
$ kognita evidence verify --db store.db
BROKEN: evidence chain broken at sequence 2: payload does not match its hash

$ kognita evidence export --db store.db -o audit.json   # portable, self-verifying

$ kognita evidence reconstruct <interaction_id> --db store.db -o report
# writes report.json and report.txt
```

`kognita evidence affected --model <version> | --policy <id> | --usecase <id> --from --to` lists every client who received model output or released tool output in that window, with the interaction. `kognita revoke` stops one agent, model, provider, tool, source, client, use case, or action. `kognita reinstate` closes that revocation when an accountable owner cites the rule. A provider stays revoked until every accountable owner of every use case that lists it has cited a reinstatement. A registered circuit-breaker threshold inserts a prohibiting policy when the flags in its window reach it. Nothing trips until a threshold is registered. `kognita drill run` records a kill-switch drill for a high-materiality use case. A high-materiality pilot or live use case with no drill scheduled, or a drill past its due date, is a denial. A payload rewritten behind the store's back, with its hashes left as they were, is reported by `kognita evidence verify`.

Payloads hold hashes and references by default, because an append-only log full of personal data collides with erasure rights.

**Pinned evidence and replay.** Each policy check records the hash of the policy row as evaluated. `replay_decision()` fails if a pinned policy row, retrieved item, or retained prompt or response no longer matches. An in-place edit of an effective policy is refused; a change is a new row via `supersede_policy()`. Retrieval records a content hash and the embedding model. A model call records the provider, the model name and version reported by the provider, the prompt template version, and hashes of the prompt as sent and the response as received. Tool and egress events record a hash of the response. Prompts, responses, and source snapshots live in a retention store keyed by those hashes. Erasure deletes the bytes and appends an `ERASURE` event; the chain keeps the hash. `kognita evidence reconstruct` answers the ten reconstruction-test questions from that record. It checks the chain and every pinned hash. Origination, RM review, and client communication are answered when those events were recorded on the interaction. A journey with no client send still marks that question "not recorded".

**Humans approve what they actually reviewed.** Through `run_governed()`, a `HUMAN_APPROVAL` decision holds the tool until approval is granted. The hold checkpoints the run. `continue_run(run_id, approvals_resolved={approval_id: True})` resumes a single-signature hold; a denied approval does not execute. A two-signature hold is `mark()`, then `continue_run` confirms the second signature. Pass `confirmer` for that signature; when it is omitted, `approver_name` is the confirming party. A missing confirmer, or the same identity as the marker, does not confirm and does not resume the run. A true resolution before `mark()` does not grant the approval and does not resume the run. Approvals bind to a hash of the envelope, attributes and checks, so an approval for one request cannot be replayed for a different one. `mark` and `confirm` are exported from `kognita`.

**Egress is guarded, not merely refused.** The egress guard decides per classification and destination whether content may leave, must be redacted, or may not leave at all:

```python
from kognita import Classification
from kognita.egress import EgressGuard, PatternRedactor

guard = EgressGuard(redactor=PatternRedactor(extra_terms=["Jane Tan"]))
result = guard.send("Jane Tan (jane.tan@example.com) holds account SG-PB-004417",
                    call_the_model, classification=Classification.C2,
                    destination="api.openai.com", destination_is_local=False)

result.decision   # REDACT
# the provider saw:  [TERM_1] ([EMAIL_1]) holds account [ACCOUNT_1]
# the caller got the real values back; evidence records the redaction
# manifest hash, never the content
```

> `PatternRedactor` is a floor, not a guarantee. It catches the patterns it knows (emails, IBANs, card numbers, phone numbers and similar) plus terms you list, and it misses names in prose that you didn't list. Deployments handling real personal data should supply an NER-based `Redactor`. The tests cover the plumbing, that nothing unredacted escapes the guard, not detection recall.

**Governed tools and questions.** `run_governed()` is the only path to a registered tool: decide, record, and only then execute, with tool-call and egress evidence. `ask()` answers a question from entitled, cited sources only, and when it refuses, the basis for refusing is the answer.

**Run budgets.** A `Run` on `run_governed()` and `ask()` caps call count, wall clock, classification ceiling, and cost when the caller already knows it. Token spend is recorded when the call site supplies it. The AI gateway adds the provider's token totals and cost to that budget. Exceeding a budget is a `DENY` that cites that budget, and the consumption is written to the evidence chain.

**Classifier-derived envelopes.** `ask()` classifies the question, and `run_governed()` classifies free-text arguments, with the pattern classifier in core. A typed attribute wins. Identity, purpose, and subject are not taken from the text. The recorded model, version, label, calibrated confidence, and input hash are what `decide()` replays, so replay does not run the classifier. Below a `CLASSIFIER_CONFIDENCE` threshold the outcome is `ESCALATE`, never `ALLOW`. A citation that acted on a classifier label names both the policy rule and that label.

**The AI gateway.** `kognita serve` fronts an OpenAI-compatible provider. The forwarded inference paths are exactly `/v1/chat/completions`, `/v1/completions`, `/v1/responses` and `/v1/embeddings`. File, image and audio attachments are refused with `S5-UNSUPPORTED-CONTENT`. Agents point `base_url` at the gateway. It decides before any byte is forwarded, redacts through the egress guard, restores redacted spans, and classifies the response. `MODEL_CALL` and `EGRESS` evidence record hashes and references, not the prompt or the response. If the evidence store cannot record the call, the default `--failure-mode FAIL_CLOSED` refuses it and does not call the provider. `DEGRADED` may proceed only for a local model and content below classification C2 (C2 is client-identifying), and writes the decision and the model evidence once the store accepts writes. The gateway overhead benchmark drops its warm-up calls, then fails unless the 95th percentile of the remaining sample is under 50 ms and the mean is under 25 ms. Classifier inference and the upstream stand-in are excluded. A single slow call is printed and is not the gate.

```console
$ kognita serve --provider openai-compatible --upstream https://api.openai.com \
    --purpose COLLABORATION --purposes COLLABORATION --agent dossier-agent
```

`--provider` accepts `openai-compatible` only, and `--upstream` is the provider origin. `--purposes` is the allowlist; with none, every call is denied. `--purpose` is the purpose claimed when a request has none. The process listens on `127.0.0.1:8080`. Clients set `base_url` to `http://127.0.0.1:8080/v1`.

The gateway holds the provider credential; the agent never does. `--provider-key`, or `KOGNITA_PROVIDER_KEY` when the flag is omitted, is sent upstream as `Authorization: Bearer <key>` on every forwarded call. A caller that presents a provider credential of its own (`authorization`, `x-api-key`, `api-key`, or `x-goog-api-key`) is denied with `S2-PROVIDER-KEY` and raises a `provider-key finding` review flag, with the value never copied into that flag, a decision, a held continuation, or a log line. With no key configured, the gateway forwards no provider credential at all.

**Tier 2 and above require gateway ingress.** A tier 2+ use case is registered with `UseCaseSpec(deployment_mode="gateway")` and runs only through `kognita serve` or the MCP proxy. Calling `run_governed()` or `ask()` for it directly from library code is denied with `S2-DEPLOYMENT-MODE`, as the [library execution tests](tests/test_library_execution_gates.py) check:

```python
evaluation = run_governed(session, envelope, registry=registry, pack=pack,
                          evidence=evidence, purposes=["COLLABORATION"]).evaluation
assert evaluation.outcome is Outcome.DENY
assert "S2-DEPLOYMENT-MODE" in [c.citation for c in evaluation.failures()]
```

**The MCP proxy.** `kognita serve --mcp --root-config config.json` fronts the MCP servers named in that file. The file also names the policy pack, the evidence database, and the default actor context. Every call becomes an envelope and is authorised with `run_governed()`. The backend is contacted only when the call is released: an allow, or a human approval that already has a live grant. A denial, an escalation, and an ungranted human approval return the outcome and the citations. If the evidence store cannot record the call, the proxy refuses it and does not forward. It has no degraded mode.

**Flagship demo.** `kognita scaffold --template governed-agent` creates a small app and a SQLite policy and evidence store.

**Conformance.** The [conformance kit](#conformance) asserts that a domain pack's rules are decided fail-closed, cited, and evidenced.

## Continuous Learning and Drift

The outcome and recorder parts shipped in v0.4.1. Autonomy gates shipped in v0.5.0. The [roadmap](docs/ROADMAP.md) plans institutional memory for 0.6:

**Outcome metrics per use case (0.4).** Each use case names one primary metric (cycle time, decision quality, or reliability) with a baseline and a target. A Knowledge Lead owns the quality standard. The three metrics are computed from the interaction record. Cycle time is mean seconds from the interaction opening to the final decision. A named primary metric with no measurement is a breach. Autonomy gates that consume those metrics shipped in 0.5.

**Flight recorder (0.4).** `kognita recorder` lists recent runs from the evidence chain, shows one run's decisions and citations, and exports a run as self-verifying JSON. Alerts cover a budget breach, repeated denials from one actor, a chain break, and classifier confidence drift. RM acceptance, edit, rejection, and wrong-client rates are counted from evidence. A missing measurement is unmeasured. `kognita serve --otel` exports decisions, denials, latency, token use, and tool failures when `kognita[otel]` is installed and `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Injection attempts and entitlement blocks are unmeasured until an event records them. An export failure is reported and does not change the decision.

**Autonomy earns its expansion (0.5, shipped).** Raising an agent's autonomy level requires thresholds on those metrics. The evidence is necessary, not sufficient: the accountable owner still approves. If three consecutive observation windows breach the lower thresholds or show sustained decline, the agent's effective level steps down one step.

**Institutional memory (0.6, planned).** A pattern in RM corrections or outcomes becomes a proposed change. The Knowledge Lead approves it as a new dated version. Learning is versioned, never edited in place.

## Domain packs

The core is domain-blind. A pack supplies what it cannot know: what a request's *attributes* are, and how to load the *subjects* it refers to.

```python
class MyPack:
    name = "my-domain"
    def load_subjects(self, envelope, session): ...
    def resolve_attributes(self, envelope, subjects): ...
    def rules(self): return build_registry(MY_EVALUATORS)
```

Policies are data: effective-dated rows with a JSON payload, interpreted by the evaluator registered for their `rule_type`. The core ships seven primitives (allowlist, denylist, required flag, required human approval, two-signature approval, classifier confidence (`CLASSIFIER_CONFIDENCE`), prohibited); a pack registers whatever its regimes need beyond them.

### Conformance

The conformance kit is a set of assertions every domain pack must satisfy: whatever a pack's rules say, they are decided fail-closed, cited and evidenced.

```bash
pytest --pyargs kognita.testing.conformance      # the kit against its bundled pack
```

```python
from kognita.testing import ConformanceCase, Harness

class TestMyPack(ConformanceCase):
    @pytest.fixture(autouse=True)
    def _bind(self):
        self.harness = Harness(pack=MyPack(), purposes=PURPOSES, seed=seed)
        self.allow_envelope = Envelope(...)
        self.deny_envelope = Envelope(...)
```

The invariants are importable, so external packs run them in their own repositories.

## Known limitations

Kognita is alpha. These are known limitations in v0.5.2. The [changelog](CHANGELOG.md) includes the migration guide and full release details.

- **The gateway refuses attachments.** File, image and audio content parts cannot be inspected locally, so they are refused (S5-UNSUPPORTED-CONTENT).
- **Degraded-mode evidence is held in the gateway process.** Restarting before the evidence store recovers drops buffered evidence. `FAIL_CLOSED` is the default; MCP has no degraded mode.
- **Schema migration is limited.** `create_all` applies supported additive columns and SQLite compatibility updates, not arbitrary schema migrations. Import a domain pack's models before creating its tables.
- **Append-only evidence triggers are SQLite-only.** Other databases do not receive those trigger protections.
- **Policy citations default to empty at model level.** `decide()` escalates empty citations with CITATION-REQUIRED and preserves denials.
- **Anomaly buckets use UTC calendar days**, except the rolling 24-hour volume window.
- **Circuit-breaker window boundaries lack a regression test.**
- **Run budgets can overshoot by one call.** Post-call usage is accounted for before response release.

## Where it's heading

The direction is **supervisory examinability**: a regulator should be able to pick one AI-assisted client interaction and reconstruct it from evidence alone. Why this client, why this product, what data the AI used, which model produced it, what the agent was authorised to do, which controls ran, what the human saw and changed, who decided, what the client was told, and whether all of that can be reproduced later.

The test every release is measured against:

> **Can the bank explain, control, stop and reconstruct every material AI action that affects a client?**

**0.3 "Gateways, the Run and Replay" is shipped** in v0.3.0: the AI gateway, the MCP proxy, classifier-derived envelopes, run budgets, suspend and resume, pinned evidence, `kognita evidence reconstruct`, degraded gateway mode, and the Tier 0 defect closures.

**0.4 "Ingestion, Policy Language and the Client Lifecycle" is shipped** in v0.4.1: structured ingestion with passage-level citations, the policy language, the use-case register, the client interaction record, outcomes and the review queue, communication and intervention controls, the circuit breaker and revocation, starter packs, the flight recorder, and the hermes-agent, openai-agents, langgraph and pydantic-ai adapters.

**0.5 "Agents, Authority and Fleets" is released as an alpha** in v0.5.1: authenticated agent identity and workload credentials, delegated authority with dual checks, autonomy levels with evidence-gated promotion and step-down, blast-radius limits, taint isolation, anomaly containment, agent delegation and fleet controls, reversibility, and a threat model. The 0.5 definition-of-done items are complete; v0.5.1 strengthens these controls. See Known limitations for remaining gaps.

Planned, not yet built:

- **0.6: Claims and institutional memory.** Typed, sourced, current claims checked before an RM relies on them; approved-source grounding; supervised memory that turns lessons into shared standards; governed business definitions.
- **0.7: Trust and resilience.** Signed evidence, external verification, provider and dependency registers.

The full plan is in [docs/ROADMAP.md](docs/ROADMAP.md). How it maps to bank control frameworks for agentic AI is in [docs/control-frameworks.md](docs/control-frameworks.md), which consolidates a 40-domain bank-grade framework and a 70-control wealth management framework, plus the EVOLVE framework's learning loop and institutional memory.

The design rule behind all of it: **never rely on the LLM to enforce a control that can be enforced outside the LLM.**

## What Kognita does not hold

Kognita governs policies, guidelines and governance records: the rules, the registers and the evidence. It does not host client data, document stores, agent memory or a knowledge graph, and it does not ship connectors to file stores. Those stay in the bank's systems, and Kognita decides and evidences access to them.

> **Removed after 0.3.0.** The Graphiti and Kuzu graph engine (`kognita.graph`, the `[graph]` extra, and the `[anthropic]`, `[groq]` and `[gemini]` extras that only installed it) is no longer part of the package. Pin `kognita[graph]==0.3.0` if you still need it. Details: [docs/decisions/0008-remove-the-graph-engine.md](docs/decisions/0008-remove-the-graph-engine.md).

## Layout

```
kognita              the decision engine: decisions, evidence, approvals,
                     retrieval, egress, tools, the AI gateway and the MCP proxy.
                     Four dependencies, no network.
kognita.registry     agent identity, deployment approval and kill switches
kognita.credentials  workload secret issue, verification and revocation
kognita.authority    human grants, bounded delegation and task completion
kognita.autonomy     evidence-gated promotion and automatic step-down
kognita.models       BlastRadius and persisted governance records
kognita.tools        Run budgets, child delegation and reversible tools
kognita.anomaly      automatic baseline checks and containment
kognita.intervention human run controls, recovery and compensation
kognita.approvals    proposal review and approved application
kognita.testing      the conformance kit
kognita.adapters     provider-backed embedders                 [openai]
                     framework adapters                         [hermes-agent]
                                                                [openai-agents]
                                                                [langgraph]
                                                                [pydantic-ai]
```

`import kognita` never loads a graph database or a provider SDK, and `tests/test_packaging.py` asserts it.

> **Moved in 0.2.** `kognita.core.*` → `kognita.*`. Details: [docs/decisions/0003-the-top-level-namespace.md](docs/decisions/0003-the-top-level-namespace.md).

## Status

**Alpha.** v0.5.2 is on PyPI. See the [changelog](CHANGELOG.md) for what changed and [CONTRIBUTING.md](CONTRIBUTING.md) to get involved.

MIT licensed.

### Typed claims

`kognita.claims` provides `Claim`, `ClaimType`, `ValidatorStatus` and the
`record_claim`, `release_claim`, `relabel_claim`, `list_claims` functions.
Record claims with a session, an evidence writer and an `interaction_id`. Content
is retained separately from its hash in the evidence chain. Release checks load
the stored claim by ID and deny unlabelled claims for RM and client audiences.
Relabelling as a verified fact requires both source references and VERIFIED status.
Successful RM releases appear in `kognita evidence reconstruct`.

Inspect records with `kognita claims show <claim_id> --db <path>` or
`kognita claims list --client <id> --interaction <id> --db <path>`.

## Fact freshness

`Fact` records a value, source, owner, timestamp, classification, entitlement,
and freshness threshold. `admit_fact` returns `(fact_or_none, warning)` and
requires a session, evidence writer, retention store and correlation id.
`UseCase.stale_action` defaults to `StaleAction.SUPPRESS`. `REFRESH` calls a
supplied refresher and checks its result again; a missing or failed refresher
suppresses the fact. `WARN` admits stale facts with an explicit warning flag.
Expired, superseded or not-yet-effective research is always withheld.

Policy files may contain a top-level freshness mapping (optionally beside a
`rules:` list):

```yaml
freshness:
  positions: 5m
  house_view: 7d
  market_prices: 30s
  kyc: 365d
  suitability: 180d
```

Thresholds accept seconds as numbers, or `s`, `m`, `h`, `d` suffixes. Load them
with `read_policy_set`, then pass
`{rule.data_type: rule for rule in policies.freshness}` to `admit_fact` or
`is_stale`. Python callers can supply `FreshnessRule` with a `timedelta`
`max_age` or a callable `rule(fact, now)` that must return exactly `True` for a
fresh fact. A fact's own threshold can tighten policy. Unknown data types,
missing timestamps, naive dates and failed rules count as stale.

Callers supply the authoritative data type and, for research, its
`ResearchLifecycle`. This gate checks freshness and contract completeness;
existing access controls still authorize the fact's entitlement. Each admission
records a `RETRIEVAL` event with `action: FACT_ADMISSION`, a reason code, admission
and warning flags, and a content hash. Fact and research content live in the
retention store, including the original and refreshed fact when applicable.

### Approved-source grounding

Claim releases to RMs and clients require `usecase_id` and a current approved
source. Register `Source` snapshots with `register_source`, set the use case's
`approved_sources` to permitted source IDs, and cite the exact ID and version
in the claim's `source_system` and `source_record`. `release_claim` checks the
stored register automatically. Missing approval, model knowledge, future dates,
supersession and expiry fail closed with C3 reason codes. Source snapshots and
grounding decisions are retained by hash in the evidence chain.

A `CONTENT_POLICY` YAML rule can further restrict citations with
`sources: [research-id, policy-id]`, checked against the `source_system`
attribute. This policy list supplements the use-case approval and lifecycle
checks performed at claim release.

## Learning export controls

Interactions, RM edits and outcomes are excluded from external training,
fine-tuning and evaluation by default. `ExportRequest` requires a purpose,
anonymisation and validation evidence references, a requester and a distinct
approver identity. The trusted calling application supplies these identities.
`request_export` returns an approved record selection manifest or a denial with
reason codes. It does not send data to a pipeline. Commit the transaction for
both approved and denied requests to preserve their evidence.

Select stored records with `interaction:<id>`, `review:<id>` or
`outcome:<evidence sequence>`. `add_exclusion` and `list_exclusions` manage
permanent client or record exclusions. Client exclusions cover all three record
types; excluding an interaction also covers its reviews and outcomes. Missing
or damaged exclusion evidence denies export. Request-local exclusions can only
add restrictions. Request details are retained by hash; the chain records each
decision and exclusion.

The CLI commits approved and denied requests, prints a JSON decision, and returns
exit code 0 for approval or 2 for denial. Repeat `--record` to select multiple
records. Use `export exclude --record <id>` to persist a record exclusion.

```sh
kognita export exclude --db kognita.db --client client-1 --reason "Opt out" --actor operator
kognita export request --db kognita.db --purpose "Quality evaluation" --anonymisation "control-17" --validation "report-42" --requester analyst --approval reviewer --record interaction:journey-1
```

### Sensitive attributes and vulnerable clients

`detect_sensitive` labels religion, health, political views, ethnicity, sexuality
and vulnerability using built-in patterns. A bank classifier callable may extend
or replace detection (`classifier=`, `extend=False` for replacement). Detection
errors deny access. Use-case `permitted_sensitive` defaults to an empty set and
is stored under `sensitive:` in its record. CONTENT_POLICY YAML also accepts a
`sensitive:` list as an additional constraint.

`check_sensitive` blocks unpermitted categories and unsupported diagnoses.
`raise_vulnerability_flag` retains an explained signal and routes it to RM review.
Even a permitted vulnerability cannot enter ordinary relationship processing.
`change_treatment` requires a completed human RM review for the same interaction,
with an approval packet whose `proposed_action` is the canonical JSON text from
`canonical_json({"flag_id": flag.flag_id, "treatment": treatment})`. The authorization is evidenced;
the bank applies the approved treatment. A flag alone never authorizes a change.

### Reasoning traces

The AI gateway retains provider reasoning summaries and thinking blocks by hash
in the `MODEL_CALL` field `reasoning_trace`. Evidence reconstruction displays
these beside the output as the "model's own account". They are evidence only:
policy rules that reference a reasoning trace are rejected.

Set `UseCaseSpec(require_reasoning_trace=True, ...)` to require an account for
material use cases at tier 3 and above. Missing accounts enter risk-based review
with reason `REASONING_TRACE_MISSING` and the named risk owner. Without review
criteria, delivery is held for review. Trace content follows retention and erasure
rules, while its hash remains in the evidence chain.
