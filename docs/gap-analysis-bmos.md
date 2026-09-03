# Gap analysis: Kognita measured against BMOS

**Status:** Analysis, 2026-09-02 · **Subject:** Kognita 0.2.0 (`claude/company-brain-python-port-8awtnv`)
**Forcing consumer:** BMOS (Business Maximizer® OS) app inventory v2, 24 August 2026

## Why this document exists

The 0.2 plan named a risk it could not resolve on its own:

> **Abstracting against an absent consumer.** The wealth pack is the only real consumer and it now
> lives elsewhere, so the core can drift into plausible-but-unusable APIs.

BMOS is the first application described in enough detail to test the abstractions. It is a shipped
TypeScript/TanStack/Postgres product whose *doctrine* is governance-shaped — two-signature gates,
"agents propose, humans sign", a human-in-the-loop proposal queue, exportable artefacts, and a
guarantee that is resolved on screen from locked runs at day 90. Six of its eight stated doctrines
are the concerns Kognita exists to serve.

This document asks one question: **what would Kognita need in order to express what BMOS does?**

### Framing decisions

- **Scope.** Library gap analysis and roadmap. BMOS stays TypeScript; this drives Kognita's roadmap
  rather than moving BMOS.
- **Boundary.** Kognita would own only the governed slice — decisions, approvals, evidence, the
  agent tool contract. Orgs, metrics, gates, criteria, documents and check-ins stay in the
  application's own schema, read through a domain pack's `load_subjects` / `resolve_attributes`.
- **Tenancy.** One Kognita store per org. No tenant column in core.

### Two costs of that framing, stated once

**Kognita is Python; BMOS is TypeScript on Lovable Cloud.** With BMOS staying TS and no service
boundary, no BMOS code will call Kognita. This analysis sharpens the library's design against a
real, demanding shape — it does not retire the absent-consumer risk. Only an actual consumer does
that. Everything below should stand on its own merits; none of it should be justified as "BMOS needs
it", because BMOS will not be the one calling it.

**Store-per-org makes one BMOS surface ungovernable.** `facilitator_cohort()` deliberately reads
*across* installs, ranked by drift evidence — the renewal-risk surface. With a database per org
there is no Kognita query that spans them, so a facilitator console stays a native application read:
ungoverned, unevidenced, outside the hash chain. An acceptable trade for a cohort-ops view, but a
written limitation rather than a discovered one.

---

## The headline: the gap runs in both directions

| Concern | BMOS today | Kognita today | Direction |
|---|---|---|---|
| Policy as data, effective-dated, cited | rules hardcoded in TS + RLS | `Policy` rows, `decide()`, a citation per check | **Kognita ahead** |
| Tamper-evident audit | ordinary tables | hash-chained evidence, verify + export | **Kognita ahead** |
| Replay of a decision | none | `as_of` on `decide()` / `load_snapshot()` | **Kognita ahead** |
| LLM egress control | none — raw send to the AI gateway | `EgressGuard` ALLOW / REDACT / DENY | **Kognita ahead** |
| Agent registry + kill switch | none | `agents` table, kill switch, evidenced | **Kognita ahead** |
| Identity and authority | 5 roles, scopes, expiring grants | `principal: str` + `is_admin: bool` | **BMOS far ahead** |
| Two signatures / separation of duties | core doctrine, enforced in SQL | structurally impossible | **BMOS far ahead** |
| Propose → diff → apply atomically | `proposals` + `decide_proposal()` | absent; an approval carries no payload | **BMOS far ahead** |
| Versioned artefacts, temporal validity | `document_versions` with author + range | single `body`, no version, no status | **BMOS far ahead** |
| Gated workflow | 4 gates, 12 criteria, 5 statuses | only `ApprovalStatus`, hardcoded | **BMOS far ahead** |

Adopting Kognita would *add* policy, evidence, replay and egress control to BMOS — it has none of
them. But Kognita cannot express BMOS's central doctrine — *"one person marks a criterion met; only
an owner confirms; nothing self-passes"* — at all. That sentence is the whole roadmap.

### How much of this is unproven rather than merely unbuilt

The pack extension point the README advertises — *"a pack registers whatever its regimes need beyond
them"* — has never been used. `build_registry(*overlays)` is called at three sites
(`governance.py:168`, `testing/conformance.py:256`, `tests/fixtures/demo_pack.py:159`) and **all
three pass zero overlays**. The only `@rule` registrations anywhere in the repo are the five core
primitives in `rules.py`. The mechanism by which a domain expresses its own regimes has no user, not
even a test.

That is the absent-consumer risk showing up as a checkable fact rather than a worry.

---

## Tier 0 — defects that make current claims weaker than advertised

Each was confirmed by reading the code, not inferred.

**1. `HUMAN_APPROVAL` withholds nothing.** `Evaluation.allowed` includes it (`envelope.py:123`), so
`run_governed` executes the tool body at `tools.py:159`, returns the data to the caller, and emits an
`EGRESS` event. The docstring immediately above, at `envelope.py:120-122`, says the work "must not
leave the boundary until the approval is granted". Nothing enforces that; `ToolRun.approval_required`
is advisory.

**2. The approval loop is opened but never closed.** `find_granted` (`approvals.py:187`) is exported
and covered by repo tests but called nowhere in `src/`. The library records that an approval is
required and leaves the enforcement entirely to the integrator.

**3. Entitlement fails open.** `entitled_items` (`retrieval.py:73-79`) treats an empty `zones` list
as *visible in every zone*. In a library whose stated posture is fail-closed, the default belongs the
other way round.

**4. The entitlement filter is not in SQL.** The same function issues `select(KnowledgeItem)` for the
whole table and filters in Python. Security-wise it is still *before scoring*, which is the part that
matters; but the plan's performance claim — "entitlement filter in SQL before embedding/scoring" — is
not what the code does.

**5. `approval_required` is hardcoded `False` on the denial path** (`tools.py:157`), including for
`ESCALATE`.

**6. `expire_stale` emits no evidence** (`approvals.py:157-172`), so a bulk expiry is invisible in
the chain.

**7. `recorded_at` is outside the hashed header.** The header hashed into `event_hash` has eight keys
(`evidence.py:109-118`); `recorded_at` is set on the row at `evidence.py:130` but never hashed. Event
timestamps are not tamper-evident.

**8. `SqliteVecIndex.search` can never match a row.** It keys candidates by Python `id()`
(`vectors.py:114`) and looks them up by SQLite `rowid` (`vectors.py:125`). The failure is silent, not
loud: `search` returns `[]`, every item scores `semantic = 0.0`, and results still come back ranked
on lexical overlap alone. The `[vec]` backend degrades retrieval without erroring.

**9. Classifiers are offered, never invoked.** `index_item` takes an asserted `classification`
(`retrieval.py:174`) and so does `EgressGuard.send` (`egress.py:210`). Nothing in the library ever
calls `Classifier.classify`. Classification is caller-asserted everywhere.

**10. `engages` is absent from the `DomainPack` protocol** (`protocols.py:63-84`) but is fetched
reflectively at `tools.py:144`, `broker.py:125` and `testing/harness.py:82`. A pack without it passes
`isinstance(pack, DomainPack)` and then *every* policy is evaluated — a silent behaviour change.

**11. No foreign keys** on `approvals.decision_id` or `entity_edges.*_entity_id`, despite
`PRAGMA foreign_keys=ON` at `db.py:56`.

---

## Tier 1 — blocking gaps for any BMOS-class application

### 1. Tool arguments, and the hole where they currently have to go

`ToolFn = Callable[[Envelope, Evaluation, Session], Any]` (`tools.py:29`), invoked as
`spec.fn(envelope, evaluation, session)` (`tools.py:159`). **There is no argument channel.**

Measured against BMOS's fifteen governed tools, only `get_org_state` works with no arguments. The
other fourteen need them: `get_document` takes a document id, every `list_*` takes a pagination
cursor, and all five `propose_*` carry the proposed change itself.

The only place an argument can travel today is `Envelope.context` — documented "Never trusted for
authorisation decisions" (`envelope.py:35`) **and excluded from `to_dict()`, and therefore from the
envelope hash** (`envelope.py:38-49`). Arguments passed that way are neither validated, nor hashed,
nor evidenced. Concretely: a request could be approved carrying one set of arguments and executed
carrying another, and `envelope_hash` would not change — defeating the binding that makes an approval
mean anything. This has to be fixed before propose-then-apply is worth building.

### 2. Identity and authority

BMOS has five roles, per-member scope and time-expiring access grants, and states the invariant *"an
agent never has more permission than the person using it."* Kognita has `principal: str` and
`is_admin: bool` (`envelope.py:22,34`), and `is_admin` has exactly one effect anywhere in the
library: `ceiling_for()` returns C3 instead of C2 (`retrieval.py:82-88`).

Two distinct problems:

- **Expressiveness.** There is no way to say "only an owner may confirm". Roles, scopes, memberships
  and grants do not exist.
- **Trust.** `is_admin` *is* inside the envelope hash (`envelope.py:48`), so the claim cannot be
  swapped after an approval — that part is sound. But the library never verifies the claim against
  anything: any caller can construct an envelope asserting `is_admin=True`. The hash binds the claim;
  nothing binds the claim to reality. The docstring at `retrieval.py:85` concedes this ("a deployment
  with real roles supplies its own ceiling"), but the conformance kit asserts nothing about it, so a
  pack can quietly do the wrong thing and still pass.

### 3. Two-signature / separation-of-duties approvals

BMOS's central doctrine. Kognita cannot express it, for four compounding reasons:

- `Approval` has one identity column, `approver_name: str`, defaulting to `"Unassigned (duty desk)"`
  (`models.py:174`) — it doubles as assignee and as decider.
- The requester's identity is never copied onto the approval row, so no comparison is possible.
- `approver_name` is an arbitrary unverified string, in no shared namespace with `principal`.
- `_transition` raises if the status is not `PENDING` (`approvals.py:76-79`), so a *second* signature
  on the same row is structurally impossible.

### 4. Propose-then-apply

BMOS: a `proposals` table carrying a before-snapshot, a kind, a rationale and a proposer, applied
atomically by `decide_proposal(id, accept, note)`. Kognita: nothing. `Approval` binds to an
`envelope_hash` and **carries no payload of the change being approved** — it records that a human
approved *a request*, not *a diff*. Combined with Tier 0 item 1, Kognita's own doctrine "agents
propose, humans sign" is currently convention, not enforcement.

---

## Tier 2 — substantial, not blocking

**5. Versioned, temporally-valid, status-bearing artefacts.** `KnowledgeItem` (`models.py:226-248`)
has a single `body` and no version, author, `valid_from`/`valid_to`, or draft/published status.
`published_at` is stored and surfaced but never filtered, scored or sorted on — display metadata
only.

*The consequence worth naming:* `as_of` replay covers **policies only**. `decide()` is genuinely
replayable, so the README's claim is accurate as written — but replay stops at the decision boundary.
Re-running a March decision retrieves against today's corpus, so *"what would this have answered in
March?"* has no answer. For an evidence library that is a meaningful half-guarantee.

**6. Retrieval entitlement.** Two predicates only — `zones` membership and a classification ceiling
(`retrieval.py:73-79`). No filter on principal, org, subject, purpose, date, status or ownership.
BMOS entitles by role × org × scope × gate-unlock.

**7. Evidence queryability.** `actor_id`, `actor_type`, `classification` and `recorded_at` are all
unindexed (`models.py:190-223`), and there is no query API at all — no `by_actor`, `by_subject`,
`by_time_range`, not even a `list_decisions`. `export_chain` loading the whole chain is correct (a
slice cannot self-verify) but it is not a reporting layer. Evidence also has no link to a subject
except inside the opaque JSON payload. BMOS's guarantee resolution and facilitator console are both
"what happened for this org, in this week" queries.

**8. Gated workflow.** BMOS has four gates, twelve criteria, five criterion statuses and metrics that
unlock by gate. Kognita's only state machine is `ApprovalStatus`, hardcoded inside `_transition` with
no extension point.

*Judgment: do not build a workflow engine.* Gates are a pack concern, and putting them in core would
turn a domain-blind library into a business-operating-system framework with one customer — precisely
the failure mode the 0.2 plan was written to avoid. The genuinely reusable primitive underneath
BMOS's gates is the two-signature attestation of Tier 1.3. Build that, and let packs compose gates
from it.

---

## Tier 3 — out of scope under the stated boundary

Noted so the boundary is explicit, not so they are dismissed: MCP / HTTP / OAuth transport, UI,
notifications and scheduling, CSV export, WhatsApp ingest, and multi-tenancy (resolved as
store-per-org). BMOS keeps all of these.

---

## Recommended sequence

**A. This document.** Standalone value even if nothing else is built.

**B. A BMOS-shaped fixture pack — `tests/fixtures/bmos_pack.py`.** The highest-value item after the
document. A neutral pack shaped like BMOS: two roles differing in authority, a criterion one party
marks and another confirms, a proposing agent whose output must not write live data — with the
assertions it cannot yet satisfy marked `xfail(strict=True)`. This turns the analysis into a target
that fails until it is closed, and it exercises the pack contract against a second, differently
shaped domain for the first time.

**C. Tier 0 fixes.** Two are behaviour changes needing a new conformance invariant — `run_governed`
must not execute on `HUMAN_APPROVAL`, and empty `zones` must mean "no zone". Seven are mechanical.
Two need a decision first: moving `recorded_at` into the hashed header is correct but **invalidates
every existing chain** (acceptable at alpha, but stated rather than slipped in), and making
`Classifier` authoritative rather than decorative is a design choice, not a fix.

**D. One ADR per Tier 1 item**, in dependency order: tool arguments → identity and authority →
two-signature approvals → propose-then-apply.

### Definition of done

Every `xfail` in the BMOS-shaped fixture pack flips to passing — without `strict=False`, and without
the core learning a single BMOS-specific concept.

### Regression bar

60 tests, 10 conformance cases, 3 import-linter contracts, `ruff check --select F,E9`, and the
bare-install claim (scratch venv, no extras, 11 packages, `import kognita` → `decide()` runs) stay
green throughout. New behaviour claims go in the portable conformance kit, not in repo-local tests,
so external packs inherit them.
