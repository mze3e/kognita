# Kognita

**Prove an AI answer was permitted — and evidence it.**

| You want to… | Reach for |
|---|---|
| Connect fragmented files and databases quickly | LlamaIndex |
| Build complex, customised LLM workflows | LangChain |
| Create a team of specialised agents | CrewAI |
| Map complex relationships across data | GraphRAG |
| **Prove an answer was permitted, and evidence it** | **Kognita** |

Content guardrails filter *what a model says*. Kognita decides *whether the
request was allowed — before any data is retrieved* — and writes the record.

```python
from kognita.core import Envelope, decide, load_snapshot

evaluation = decide(
    Envelope(principal="rm@bank.example", purpose="ELIGIBILITY_CHECK",
             tool="check_eligibility", actor_location="AE",
             subject_type="client", subject_id="1",
             subjects={"instrument": "1"}),
    load_snapshot(session),
    attributes=pack.resolve_attributes(...),
    rules=pack.rules(),
)

evaluation.outcome        # DENY
for check in evaluation.basis():
    print(check.regime, check.citation)
# HK_SFC        SFC Code of Conduct para 5.5
# DIFC_DFSA     DFSA COB 3; GEN 2
```

Two independent regimes refused; neither masked the other; each names the rule
it came from. Nothing was retrieved.

## Install

```bash
pip install kognita                 # the decision engine — 4 dependencies
pip install kognita[graph]          # + Graphiti/Kuzu knowledge graph
pip install kognita[openai]         # + a real embedder
pip install kognita[all]
```

The core installs on `pydantic`, `sqlmodel`, `numpy` and `python-dotenv`, and
runs with no network and no API key. Deciding whether a request is permitted
should not require the machinery that answers it — and that constraint is
enforced by `import-linter` contracts plus a test that installs with no extras
and asserts `decide()` still runs.

## What it does

**Authorise before discovery.** An envelope describes an intent and is evaluated
before anything is fetched. A denial returns no data, not filtered data.

**Fail closed.** `DENY > ESCALATE > HUMAN_APPROVAL > ALLOW`. One failing check
among a hundred passes still denies, so a policy set cannot be widened by adding
permissive rules.

**Every decision cites its rule.** A check without a citation is an assertion,
not a decision; the conformance kit enforces it.

**Decisions are pure and replayable.** `decide()` writes nothing and takes the
instant as a parameter, so *"what would this have decided in March?"* has an
answer:

```python
decide(envelope, snapshot, as_of=datetime(2026, 3, 1, tzinfo=timezone.utc))
```

**Evidence is tamper-evident.** Each event carries the previous event's hash.
Altering any payload breaks every hash after it:

```console
$ kognita evidence verify --db store.db
BROKEN: evidence chain broken at sequence 2: payload does not match its hash

$ kognita evidence export --db store.db -o audit.json   # portable, self-verifying
```

Payloads hold hashes and references by default — an append-only log full of
personal data collides with erasure rights — and `hashes_only` strips content
entirely while keeping the log provable.

**Egress is guarded, not merely refused.** A binary local-or-refuse rule confines
a governed system to whatever model runs on the box. The guard adds redaction:

```python
result = guard.send(text, call_the_model,
                    classification=Classification.C2,
                    destination="api.openai.com", destination_is_local=False)

result.decision          # REDACT
# the provider saw:  [TERM_1] ([EMAIL_1]) holds account [ACCOUNT_1]
# the caller got the real values back, and MODEL_CALL + EGRESS
# record the manifest hash — never the content.
```

> `PatternRedactor` is a floor, not a guarantee. Regexes miss names in prose and
> anything the patterns do not anticipate. Deployments handling real personal
> data should supply an NER-based `Redactor`; the tests cover the plumbing —
> that nothing unredacted escapes the guard — never detection recall.

## Domain packs

The core is domain-blind. A pack supplies the two things it cannot know: what a
request's *attributes* are, and how to load the *subjects* it refers to.

```python
class MyPack:
    name = "my-domain"
    def load_subjects(self, envelope, session): ...
    def resolve_attributes(self, envelope, subjects): ...
    def rules(self): return build_registry(MY_EVALUATORS)
```

Policies are data — effective-dated rows with a JSON payload interpreted by the
evaluator registered for their `rule_type`. The core ships five primitives
(allowlist, denylist, required flag, required human review, prohibited); a pack
registers whatever its regimes need beyond them. A policy whose `rule_type` has
no evaluator **escalates** rather than being skipped: it is a rule someone
believes is in force.

### Conformance

Kognita ships a conformance kit: a set of assertions that every domain pack must
satisfy. The kit proves that whatever a pack's regimes say, they are decided
fail-closed, cited, and evidenced.

Run the kit over the bundled fixture pack (proves the kit itself works):

```bash
pytest --pyargs kognita.testing.conformance
```

Or subclass it in your own pack's test suite:

```python
from kognita.testing import ConformanceCase, Harness

class TestMyPack(ConformanceCase):
    @pytest.fixture(autouse=True)
    def _bind(self):
        self.harness = Harness(pack=MyPack(), purposes=PURPOSES, seed=seed)
        self.allow_envelope = Envelope(...)
        self.deny_envelope = Envelope(...)
        self.human_envelope = Envelope(...)  # optional
```

The pattern follows `langchain-tests`: invariants are importable and reusable by
external packs running in their own repositories.

## The knowledge graph

`kognita[graph]` adds the Graphiti + Kuzu engine: documents become a bi-temporal,
auto-deduplicated knowledge graph.

```python
from kognita import Kognita, KognitaConfig

async with Kognita(config) as kg:
    await kg.ingest_text(document, source="policy-handbook")
    hits = await kg.search("cross-border disclosure")
```

Two graphs share one Kuzu database: Graphiti's LLM-extracted knowledge, and a
deterministic `SoR_*` mirror of a system of record, so one traversal crosses both
planes. All access goes through a single `KuzuSession` — two `kuzu.Database`
handles on one path do **not** share a consistent view and raise nothing when
they diverge. See [docs/decisions/0001-kuzu-cotenancy.md](docs/decisions/0001-kuzu-cotenancy.md).

## Layout

```
kognita.core       decisions, evidence, retrieval, egress, tools — 4 dependencies
kognita.graph      Graphiti + Kuzu knowledge engine          [graph]
kognita.adapters   provider adapters                         [openai] …
kognita.testing    the conformance kit
kognita.config     provider dataclasses — no dependencies at all
```

`from kognita import Kognita` still works; graph names are bound lazily and,
without the extra, raise a `ConfigError` naming what to install rather than an
import traceback.

## Status

Alpha. The decision engine, evidence plane, retrieval, egress guard, tool runner
and broker are implemented and tested. The deterministic graph mirror and
governed document ingestion are next.

MIT licensed.
