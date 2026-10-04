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

**Authorise before discovery.** An agent's intent is described as an envelope (who, for what purpose, with which tool, about which subject, from where) and evaluated before anything is fetched. A denial returns no data, not filtered data.

**Fail closed.** Outcomes resolve as `DENY > ESCALATE > HUMAN_APPROVAL > ALLOW`. One failing check among a hundred passes still denies, so a policy set cannot be widened by adding permissive rules. A policy whose rule type has no evaluator escalates rather than being skipped.

**Every decision cites its rule.** Each check carries the regime and citation it came from. A check without a citation is an assertion, not a decision, and the conformance kit enforces it.

**Agents are registered, and can be stopped.** An agent that is not in the registry is denied. Each registered agent carries a version, an accountable owner and a materiality tier, and has a kill switch that denies its next request:

```console
DENY  Kill switch engaged — accountable owner: Head of Wealth Advisory
DENY  Agent inventory — 'AGENT-UNKNOWN' is not registered
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

Payloads hold hashes and references by default, because an append-only log full of personal data collides with erasure rights.

**Humans approve what they actually reviewed.** Through `run_governed()`, a `HUMAN_APPROVAL` decision holds the tool until approval is granted. Approvals bind to a hash of the envelope, attributes and checks, so an approval for one request cannot be replayed for a different one. Two-signature approval, where one person marks and a different person confirms, is built in.

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

## Continuous Learning and Drift

Kognita is designed to capture not just permissions but outcomes, so agents and workflows improve over time:

**Outcome metrics per use case.** Each workflow registers one primary metric (cycle time, decision quality, reliability) with a baseline and target. Kognita computes these from the evidence it records: how long from trigger to final decision, RM edit and rejection rates, exception handling. A Knowledge Lead owns the quality standard and approves improvements to it.

**Autonomy earns its expansion.** Raising an agent's autonomy level requires evidence thresholds on those metrics. An agent that maintains quality ≥ 95%, exceptions < 5% and full traceability can move to the next level; if metrics degrade, it steps back automatically. Quality is tracked as trends per agent and model version, so gradual drift is caught before a hard breach.

**Institutional memory.** A pattern in RM corrections or outcomes becomes a *proposed change* to the shared standard. The Knowledge Lead approves it; it becomes a new dated version. Each output records which version produced it, so learning is versioned and reproducible, never edited in place.

These close the loop: every correction and outcome feeds back into the next decision, so one RM's lesson improves all subsequent RMs' guidance.

## Domain packs

The core is domain-blind. A pack supplies what it cannot know: what a request's *attributes* are, and how to load the *subjects* it refers to.

```python
class MyPack:
    name = "my-domain"
    def load_subjects(self, envelope, session): ...
    def resolve_attributes(self, envelope, subjects): ...
    def rules(self): return build_registry(MY_EVALUATORS)
```

Policies are data: effective-dated rows with a JSON payload, interpreted by the evaluator registered for their `rule_type`. The core ships six primitives (allowlist, denylist, required flag, required human approval, two-signature approval, prohibited); a pack registers whatever its regimes need beyond them.

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

Kognita is alpha. These are known gaps in v0.2, each scheduled on the [roadmap](docs/ROADMAP.md):

- **The purpose check fails open when no purpose list is configured.** Pass `purposes=` explicitly until it fails closed.
- **Agent names are self-asserted.** The registry denies unknown names, but nothing yet authenticates that a caller is the agent it claims to be. A request with no agent name skips the registry and is treated as a human.
- **Reconstruction covers what 0.3 records.** `kognita evidence reconstruct` answers from pinned evidence and the retention store. Why this client, why this insight or product, what the RM saw and changed, who made the final decision, and what was communicated to the client are present and marked "not recorded" until the client-lifecycle records exist.
- **Gateway degraded mode is not implemented.** If the evidence store is down the gateway refuses the call. A degraded path for local models is still open.

## Where it's heading

The direction is **supervisory examinability**: a regulator should be able to pick one AI-assisted client interaction and reconstruct it from evidence alone. Why this client, why this product, what data the AI used, which model produced it, what the agent was authorised to do, which controls ran, what the human saw and changed, who decided, what the client was told, and whether all of that can be reproduced later.

The test every release is measured against:

> **Can the bank explain, control, stop and reconstruct every material AI action that affects a client?**

Planned, not yet built:

- **0.3 still open:** degraded gateway mode, and the remaining Tier 0 defects. The gateway, the MCP proxy, run budgets, suspend and resume, pinned evidence, and `kognita evidence reconstruct` are already in the tree.
- **0.4: Ingestion, policy language and the client lifecycle.** Passage-level citations, a YAML policy language with diff and validation, a use-case register, risk-based review, a circuit breaker.
- **0.5: Agents, authority and fleets.** Authenticated agent identity, delegated authority, autonomy levels, evidence-gated promotion and automatic step-down on drift, blast-radius limits, containment.
- **0.6: Claims and institutional memory.** Typed, sourced, current claims checked before an RM relies on them; approved-source grounding; supervised memory that turns lessons into shared standards; governed business definitions.
- **0.7: Trust and resilience.** Signed evidence, external verification, provider and dependency registers.

The full plan is in [docs/ROADMAP.md](docs/ROADMAP.md). How it maps to bank control frameworks for agentic AI is in [docs/control-frameworks.md](docs/control-frameworks.md), which consolidates a 40-domain bank-grade framework and a 70-control wealth management framework, plus the EVOLVE framework's learning loop and institutional memory.

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
                   retrieval, egress, tools. Four dependencies, no network.
kognita.testing    the conformance kit
kognita.adapters   provider-backed embedders and clients     [openai] …
kognita.graph      Graphiti + Kuzu knowledge engine          [graph]
```

`import kognita` never loads a graph database or a provider SDK, and `tests/test_packaging.py` asserts it.

> **Moved in 0.2.** `kognita.Kognita` → `kognita.graph.GraphEngine`, `kognita.KognitaConfig` → `kognita.graph.GraphConfig`, `kognita.core.*` → `kognita.*`. Touching a retired name raises an `AttributeError` naming its new home. Details: [docs/decisions/0003-the-top-level-namespace.md](docs/decisions/0003-the-top-level-namespace.md).

## Status

**Alpha.** v0.2.0 is on PyPI. See the [changelog](CHANGELOG.md) for what changed and [CONTRIBUTING.md](CONTRIBUTING.md) to get involved.

MIT licensed.
