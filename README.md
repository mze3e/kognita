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

Optional extras add provider-backed embedders (`kognita[openai]`), a SQLite vector index (`kognita[vec]`), local embeddings (`kognita[local-embeddings]`) and a knowledge-graph engine (`kognita[graph]`, see [below](#optional-knowledge-graph)).

## What it does today

**Authorise before discovery.** An agent's intent is described as an envelope (who, for what purpose, with which tool, about which subject, from where) and evaluated before anything is fetched. A denial returns no data, not filtered data. Retrieval filters on zones and classification before scoring. An item with no zones is not visible in any zone.

**Fail closed.** Outcomes resolve as `DENY > ESCALATE > HUMAN_APPROVAL > ALLOW`. One failing check among a hundred passes still denies, so a policy set cannot be widened by adding permissive rules. A policy whose rule type has no evaluator escalates rather than being skipped. A missing or empty purpose list fails the purpose check.

**Every decision cites its rule.** Each check carries the regime and citation it came from. A check without a citation is an assertion, not a decision, and the conformance kit enforces it.

**Agents are registered, and can be stopped.** An agent that is not in the registry is denied. Each registered agent carries a version, an accountable owner and a materiality tier, and has a kill switch that denies its next request:

```console
DENY  Kill switch engaged — accountable owner: Head of Wealth Advisory
DENY  Agent inventory — 'AGENT-UNKNOWN' is not registered
```

Through `kognita serve`, a call with no agent name is denied. An agent name is accepted only when the bound client configuration lists it. An approved system trigger is admitted, and it is not treated as a human.

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

Payloads hold hashes and references by default, because an append-only log full of personal data collides with erasure rights.

**Pinned evidence and replay.** Each policy check records the hash of the policy row as evaluated. `replay_decision()` fails if a pinned policy row, retrieved item, or retained prompt or response no longer matches. An in-place edit of an effective policy is refused; a change is a new row via `supersede_policy()`. Retrieval records a content hash and the embedding model. A model call records the provider, the model name and version reported by the provider, the prompt template version, and hashes of the prompt as sent and the response as received. Tool and egress events record a hash of the response. Prompts, responses, and source snapshots live in a retention store keyed by those hashes. Erasure deletes the bytes and appends an `ERASURE` event; the chain keeps the hash. `kognita evidence reconstruct` answers the ten reconstruction-test questions from that record. It checks the chain and every pinned hash, and it marks questions answered by origination, RM review capture, and governed client communication "not recorded".

**Humans approve what they actually reviewed.** Through `run_governed()`, a `HUMAN_APPROVAL` decision holds the tool until approval is granted. The hold checkpoints the run. `continue_run(run_id, approvals_resolved={approval_id: True})` resumes it; a denied approval does not execute. Approvals bind to a hash of the envelope, attributes and checks, so an approval for one request cannot be replayed for a different one. Two-signature approval, where one person marks and a different person confirms, is built in.

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

**The AI gateway.** `kognita serve` fronts an OpenAI-compatible provider. Agents point `base_url` at the gateway. It decides before any byte is forwarded, redacts through the egress guard, restores redacted spans, and classifies the response. `MODEL_CALL` and `EGRESS` evidence record hashes and references, not the prompt or the response. If the evidence store cannot record the call, the default `--failure-mode FAIL_CLOSED` refuses it and does not call the provider. `DEGRADED` may proceed only for a local model and content below classification C2 (C2 is client-identifying), and writes the decision and the model evidence once the store accepts writes. The gateway overhead benchmark fails a call whose overhead, excluding classifier inference and the upstream stand-in, is not under 50 ms.

```console
$ kognita serve --provider openai-compatible --upstream https://api.openai.com \
    --purpose COLLABORATION --purposes COLLABORATION --agent dossier-agent
```

`--provider` accepts `openai-compatible` only, and `--upstream` is the provider origin. `--purposes` is the allowlist; with none, every call is denied. `--purpose` is the purpose claimed when a request has none. The process listens on `127.0.0.1:8080`. Clients set `base_url` to `http://127.0.0.1:8080/v1`.

**The MCP proxy.** `kognita serve --mcp --root-config config.json` fronts the MCP servers named in that file. The file also names the policy pack, the evidence database, and the default actor context. Every call becomes an envelope and is authorised with `run_governed()`. The backend is contacted only when the call is released: an allow, or a human approval that already has a live grant. A denial, an escalation, and an ungranted human approval return the outcome and the citations. If the evidence store cannot record the call, the proxy refuses it and does not forward. It has no degraded mode.

**Flagship demo.** `kognita scaffold --template governed-agent` creates a small app and a SQLite policy and evidence store.

**Conformance.** The [conformance kit](#conformance) asserts that a domain pack's rules are decided fail-closed, cited, and evidenced.

## Institutional memory

**Institutional memory (0.6).** A pattern in RM corrections or outcomes becomes a proposed change. The Knowledge Lead approves it as a new dated version. Learning is versioned, never edited in place. The [roadmap](docs/ROADMAP.md) places this after 0.3.

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

Kognita is alpha. These are known gaps in v0.3.0:

- **Reconstruction covers what 0.3 records.** `kognita evidence reconstruct` answers from pinned evidence and the retention store. Why this client, why this insight or product, what the RM saw and changed, who made the final decision, and what was communicated to the client are present and marked "not recorded" until the client-lifecycle records exist. The roadmap places those records in 0.4.
- **Degraded-mode evidence is held in the gateway process.** A `DEGRADED` call that proceeds while the evidence store is down keeps the decision and the model evidence in memory until the store accepts writes. Restarting the process before then drops them. `FAIL_CLOSED` is the default, and it refuses the call. The MCP proxy has no degraded mode.
- **`create_all` does not migrate an existing SQLite file.** It creates every registered table that does not yet exist. Tables already in the file are left as they are. A domain pack's models must be imported first, or its tables are skipped.
- **Agent credentials are not issued yet.** Through `kognita serve`, a call with no agent name is denied, and an agent name is accepted only when the bound client configuration lists it. `decide()` still skips the registry when the envelope has no agent name, and the in-process runner records that request as the human principal. Agents authenticating with their own credentials is 0.5.

## Where it's heading

The direction is **supervisory examinability**: a regulator should be able to pick one AI-assisted client interaction and reconstruct it from evidence alone. Why this client, why this product, what data the AI used, which model produced it, what the agent was authorised to do, which controls ran, what the human saw and changed, who decided, what the client was told, and whether all of that can be reproduced later.

The test every release is measured against:

> **Can the bank explain, control, stop and reconstruct every material AI action that affects a client?**

**0.3 "Gateways, the Run and Replay" is shipped** in v0.3.0: the AI gateway, the MCP proxy, classifier-derived envelopes, run budgets, suspend and resume, pinned evidence, `kognita evidence reconstruct`, degraded gateway mode, and the Tier 0 defect closures.

Planned, not yet built:

- **0.4 "Ingestion, Policy Language and the Client Lifecycle".** Make citations real down to the passage, let non-engineers author and review policy, record a client interaction from origination to communication, and meet developers in the frameworks they already use. The roadmap places this in Q1 2027.
- **0.5: Agents, authority and fleets.** Authenticated agent identity, delegated authority, blast-radius limits, containment.
- **0.6: Claims and institutional memory.** Typed, sourced, current claims checked before an RM relies on them; approved-source grounding; supervised memory that turns lessons into shared standards; governed business definitions.
- **0.7: Trust and resilience.** Signed evidence, external verification, provider and dependency registers.

The full plan is in [docs/ROADMAP.md](docs/ROADMAP.md). How it maps to bank control frameworks for agentic AI is in [docs/control-frameworks.md](docs/control-frameworks.md), which consolidates a 40-domain bank-grade framework and a 70-control wealth management framework, plus the EVOLVE framework's institutional memory.

The design rule behind all of it: **never rely on the LLM to enforce a control that can be enforced outside the LLM.**

## Optional: knowledge graph

`kognita[graph]` adds a Graphiti and Kuzu engine that turns documents into a bi-temporal knowledge graph:

```python
from kognita.graph import GraphEngine, GraphConfig

async with GraphEngine(config) as kg:
    await kg.ingest_text(document, source="policy-handbook")
    hits = await kg.search("cross-border disclosure")
```

Its future is under review. It currently pins `graphiti-core` and caps `openai` below version 2, and the roadmap favours moving it to a separate package with structured document ingestion as the primary retrieval path. See [docs/decisions/0001-kuzu-cotenancy.md](docs/decisions/0001-kuzu-cotenancy.md) for its design.

## Layout

```
kognita            the decision engine: decisions, evidence, approvals,
                   retrieval, egress, tools, the AI gateway and the MCP proxy.
                   Four dependencies, no network.
kognita.testing    the conformance kit
kognita.adapters   provider-backed embedders and clients     [openai] …
kognita.graph      Graphiti + Kuzu knowledge engine          [graph]
```

`import kognita` never loads a graph database or a provider SDK, and `tests/test_packaging.py` asserts it.

> **Moved in 0.2.** `kognita.Kognita` → `kognita.graph.GraphEngine`, `kognita.KognitaConfig` → `kognita.graph.GraphConfig`, `kognita.core.*` → `kognita.*`. Touching a retired name raises an `AttributeError` naming its new home. Details: [docs/decisions/0003-the-top-level-namespace.md](docs/decisions/0003-the-top-level-namespace.md).

## Status

**Alpha.** v0.3.0 is on PyPI. See the [changelog](CHANGELOG.md) for what changed and [CONTRIBUTING.md](CONTRIBUTING.md) to get involved.

MIT licensed.
