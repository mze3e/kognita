# Kognita Roadmap: From Governance Engine to Agent Harness

**Vision:** Transform Kognita from a policy decision engine into a complete agent harness—a framework anyone can adopt to build governed AI agents with proof of permission and tamper-evident audit logs.

**Current Status:** v0.2.0 (core governance engine complete). Starting from a foundation of fail-closed decisions, cited policies, and tamper-evident evidence, we build outward to become the standard way organizations safely deploy agentic systems.

---

## Strategic Direction

Kognita's core strength is **gating**: authorization before execution, with every decision citable to policy and every decision chain tamper-evident. Current gaps prevent adoption as a harness:

- No **Run context** to group and budget tool calls across agent steps
- No **model wrapper** to govern LLM calls themselves—only tool egress
- No **MCP integration** for agent-agnostic governance (Claude, OpenAI, Cursor, Claude Desktop agents all speak MCP)
- No **adapters** for existing frameworks (Claude Agent SDK, LangGraph, Pydantic AI, LangChain)
- No **policy language** or CLI for declaring and versioning rules without Python code

The wedge to viral adoption: an MCP proxy. One command (`kognita serve`) fronts any existing MCP server—governance with zero code changes, no framework lock-in, works with every agent at once.

---

## Release Timeline

```
Q4 2025     Q1 2026     Q2 2026     Q3 2026     Q4 2026     2027
│           │           │           │           │           │
├─ 0.3 ─────┤           ├─ 0.4 ─────┤           ├─ 0.5 ─────┤─ 0.6 ────────► 1.0
  Critical    (readiness)  Adapters    (readiness)  Fleets    Trust  (polish)
  Launch                   & CLI                    & Attn
  "The Run &              "Flight                 "Govern
  MCP Proxy"              Recorder"               Multiple"
```

---

## 0.3 "The Run and the MCP Proxy" — Critical Launch Release

**Timeline:** Q4 2025  
**Goal:** Make Kognita the default harness for agentic governance. Ship the two features every prospect asks for.

### Core Features

#### 1. Run Context and Tool Budgets
- `Run` dataclass grouping tool calls across agent steps with:
  - `max_calls`: deny beyond this count (prevent infinite loops)
  - `max_tokens`: track LLM token spend (cost control)
  - `max_cost_usd`: hard stop on spend
  - `wall_clock_seconds`: timeout for the whole run
  - `classification_ceiling`: tools can only access data up to this sensitivity level
  - `approvals_pending`: list of open HUMAN_APPROVAL decisions awaiting action
- `run_governed(run=…)` overload accepting Run context
- Budget consumed tracked in evidence chain
- Remaining budget passed to tool as context

**Rationale:** Agents make multiple calls; budgets prevent runaway spend and recursive loops. Organizations require cost limits before deployment.

#### 2. Durable Suspend/Resume for HUMAN_APPROVAL
- HUMAN_APPROVAL decisions write a continuation token to evidence database
- Caller resumes with `continue_run(run_id, approvals_resolved={approval_id: True/False})`
- Resumed run picks up where it left off—context replayed from evidence
- Evidence shows decision requested, approved/denied, then resumed
- Prevents approval loop silence (Tier 0 defect closure)

**Rationale:** The approval flow is currently broken. Fixing it unblocks regulated use cases (e.g., cross-border disclosure must have explicit human sign-off).

#### 3. Governed Model Wrapper
- `GovernedModel` wrapping `Anthropic`, `OpenAI`, or `LiteLLM` clients
- Every call goes through:
  1. Classification of the prompt (is this PII-leaking code generation?)
  2. Entitlement check (does this user have permission to call Claude on this data?)
  3. Redaction (if destination is remote API, apply egress guard)
  4. LLM call
  5. Evidence log (prompt redacted, tokens consumed, latency)
- Token consumption feeds into Run budget
- Prompt/response classification inferred from call context

**Rationale:** Current gap: tool outputs are guarded, but LLM prompts are not. An agent that sends unredacted PII to Claude API violates governance. This is a Tier 1 blocker.

#### 4. MCP Proxy Server
- `kognita serve --port 3001` launches an MCP server
- Takes a `--root-config` JSON file specifying:
  - Backend MCP server URL(s)
  - Policy pack class (which domain rules apply?)
  - Database path for evidence
  - Default session/actor context
- Every MCP call:
  1. Translates resource path to Envelope (e.g., `file:///etc/passwd` → `Envelope(tool="read_file", subject_id="passwd", …)`)
  2. Authorizes via `run_governed()`
  3. Proxies to backend if allowed
  4. Returns result or DENY + citations
  5. Logs to evidence chain

**Demo scenario:**
```bash
# Backend: Gmail MCP server (reads user's email)
# Governance: policy pack that restricts read to own mailbox only

kognita serve \
  --root-config gmail-config.json \
  --backend-url stdio:../node_modules/.bin/gmail-mcp
```

Agent requests: `resource://get_message?id=alice@corp/important_bids`  
Governance sees: subject_id=alice, tool=get_message, actor=agent_name  
Policy decides: denied (actor lacks authority on cross-corp message)  
Response: `{outcome: DENY, basis: [{regime: "RBAC", citation: "Role policy 2.1"}]}`

**Rationale:** MCP is the lingua franca for agent integrations. An MCP proxy makes governance transparent to every agent (Claude, OpenAI, Cursor, Claude Desktop, LangGraph, etc.) without framework lock-in.

#### 5. Demo: Flagship Scenario
- CLI: `kognita scaffold --template governed-agent`
- Scaffold creates:
  - Flask app with one agent endpoint
  - SQLite database with governance policies and evidence
  - Three demo scenarios:
    1. ✓ Allowed query (agent asks for own data, allowed, evidence logged)
    2. ✗ Denied query (agent asks for other user's data, denied with citations)
    3. ✗ Tampering test (user manually modifies a row in the evidence DB)
- `kognita evidence verify --db evidence.db` shows tampering detected
- Visual walkthrough in the README under "Getting Started"

**Rationale:** Everyone learns by example. A working demo in 5 minutes sells better than architecture docs.

### Tier 0 Defect Closure

From `docs/gap-analysis-bmos.md`, fix:
- [ ] **HUMAN_APPROVAL withholds nothing** → Implement durable suspend/resume
- [ ] **Approval loop unclosed** → Resume logic + evidence proof
- [ ] **Entitlements fail open** → Enforce deny-by-default in VectorIndex
- [ ] **SqliteVecIndex silent failure** → Test with corrupt DB, assert explicit error
- [ ] **Classifiers never invoked** → Invoke in GovernedModel; test coverage
- [ ] **Engages missing from protocol** → Add to DomainPack; test in conformance kit
- [ ] **No foreign keys on Evidence.policy_id** → Add DB migration; test referential integrity

### Definition of Done

- [ ] All tests pass: `pytest tests/` (existing + new 15 test cases for Run, resume, GovernedModel, MCP proxy)
- [ ] Conformance kit passes: `pytest --pyargs kognita.testing.conformance`
- [ ] Flagship scenario works end-to-end in < 3 minutes from scaffold
- [ ] MCP proxy benchmarked: latency overhead < 50ms per call, throughput > 100 req/s
- [ ] CLI commands tested: `kognita scaffold`, `kognita evidence verify`, `kognita serve`
- [ ] README updated: "Getting Started" includes flagship scenario, screenshots
- [ ] GitHub issues closed: all Tier 0 defects
- [ ] Import contracts still pass: core still has no optional dependencies

### Effort Estimate
- Run context: 2–3 weeks (new models, plumbing, test harness)
- Suspend/resume: 2–3 weeks (evidence integration, resume replay)
- GovernedModel: 2–3 weeks (adapters for Anthropic/OpenAI, test)
- MCP proxy: 3–4 weeks (routing logic, benchmark, demo)
- Tier 0 fixes: 1 week each (7–8 weeks total, can parallelize)
- Flagship demo: 1 week
- **Total: ~12–16 weeks (Q4 2025 + early Q1 2026)**

---

## 0.4 "Adapters and Flight Recorder" — Ecosystem Enablement

**Timeline:** Q1–Q2 2026  
**Goal:** Make Kognita work in existing frameworks with <200 LOC adapters. Add operational visibility.

### Core Features

#### 1. Framework Adapters (<200 LOC each)
- **Claude Agent SDK** → `KognitaAgent` wrapper on `Agent`
  - Intercepts `run()` calls; wraps in Kognita Run
  - Surfaces approval_required flag to Flask app
- **OpenAI Agents SDK** → `KognitaModel` wrapping model calls
- **LangGraph** → `KognitaNode` for tool nodes
- **Pydantic AI** → `KognitaTool` decorator
- **LangChain** → `KognitaCallback` + Tool wrapper

**Rationale:** Most teams use one of these frameworks. Adapters show "works with what you already use, zero refactor."

#### 2. Flight Recorder: Observability Dashboard
- Local Flask dashboard on `http://localhost:3002`
- Live table: all runs in the last 24h, filtered by actor/tool/outcome
- Drill-down: click a run → see all tool calls, decisions, evidence chain
- Export: run_id → JSON export of decision trace (portable, self-verifying)
- Alerts: runs exceeding budget, repeated denials from same actor, tampering detected

**Rationale:** Ops teams live in dashboards. A flight recorder makes governance visible without learning Kognita.

#### 3. Policy Language: YAML + CLI
- Declarative policy format (YAML, similar to Rego but Python-friendly):
  ```yaml
  rule: "client_email_access"
  subjects:
    - type: "client"
  actions:
    - "read_email"
  conditions:
    - actor_location == subject_location  # Same region
    - subject.kyc_status == "approved"
  effect: "ALLOW"
  expires: "2026-12-31"
  ```
- Commands:
  - `kognita policy load policy.yaml` → compile to Python, type-check, test
  - `kognita policy diff old.yaml new.yaml` → show what changed
  - `kognita policy lint` → validate syntax, catch dead rules
  - `kognita policy explain --envelope "tool=read_email&subject=alice"` → which rules apply?

**Rationale:** Policy is the language of governance. Python code is not. Policy teams, legal teams, and compliance officers need to *read* and *approve* rules without learning Python.

#### 4. Policy Test Format
- Colocated with policy files: `policy.test.yaml`
  ```yaml
  - name: "client_can_read_own_email"
    envelope:
      principal: "alice@corp"
      tool: "read_email"
      subject_type: "client"
      subject_id: "alice"
    decision: "ALLOW"  # Assert outcome
  
  - name: "client_cannot_read_other_email"
    envelope:
      principal: "alice@corp"
      tool: "read_email"
      subject_type: "client"
      subject_id: "bob"
    decision: "DENY"
    citation_regex: "isolation.policy.*"  # Assert which rule denied
  ```
- `kognita policy test policy.test.yaml` → run all, fail fast on mismatch

**Rationale:** Governance must be testable. Tests prove policy does what it claims.

#### 5. Starter Policy Packs
- Templates for common scenarios:
  - RBAC (role-based): actor roles ↔ resource zones
  - Geo-fencing: actor_location == data_location
  - Data classification: tool output C1/C2/C3 → who can see it
  - Time-gated: approved only 9am–5pm GMT
  - Approval chains: requires two approvals from different teams
- Each starter includes policy file + test suite + fixture pack for testing

**Rationale:** New adopters shouldn't start from scratch. Starters accelerate time-to-value.

### Definition of Done

- [ ] 5 adapters: Claude SDK, OpenAI SDK, LangGraph, Pydantic AI, LangChain (<200 LOC each)
- [ ] Adapter tests: each framework can run a simple end-to-end flow through Kognita
- [ ] Flight recorder: dashboard shows ≥20 runs, drill-down works, export is valid JSON
- [ ] Policy language: parser, compiler, CLI, tests all working
- [ ] 5 starter packs: RBAC, geo-fencing, data classification, time-gated, approval chain (1 pack = policy file + tests + fixture)
- [ ] Examples: 2–3 worked examples per starter pack
- [ ] Docs: "Adapters" guide, policy language reference, starter pack tour
- [ ] GitHub issues for common use cases linked from starters

### Effort Estimate
- 5 adapters: 2–3 weeks (can parallelize)
- Flight recorder: 2–3 weeks
- Policy language: 3–4 weeks (parsing, CLI, type-checking)
- Starter packs: 2 weeks
- Docs: 1 week
- **Total: ~10–13 weeks (Q1–Q2 2026)**

---

## 0.5 "Fleets" — Multi-Agent Governance

**Timeline:** Q2–Q3 2026  
**Goal:** Enable teams to deploy multiple governed agents with shared policies and inter-agent communication.

### Core Features

#### 1. Delegation with Attenuation
- One agent delegates to another: `Run.delegate_to(agent_name, attenuated_scope=…)`
- Delegated scope narrows: if parent run has `max_calls=10`, child gets `max_calls=5`
- Budget shared between agents: parent consumed 3 calls, child can only use 2 of remaining 7
- Every delegation logged in evidence chain with authority transfer recorded
- Child cannot escalate privilege (attenuation is monotonic)

**Rationale:** Complex workflows need hand-offs (e.g., sales agent → compliance review agent → execution agent). Attenuation prevents child from exceeding parent's authority.

#### 2. Capability Grants
- Actor can grant temporary capability: `grant_to(grantee, capability, duration, subject_scope)`
- Grant stored in evidence as immutable record
- Grantee can use capability within scope and duration
- Grant revocation is immediate (future checks see revoked state)

**Example:**
```python
# Sales agent needs to read KYC data to close a deal, but normally can't
grant_to(grantee="sales_agent_1", 
         capability="read_kyc",
         duration=timedelta(hours=2),
         subject_scope={"customer_id": "acme_corp"})
```

**Rationale:** Not everything can be pre-baked into policy. Some decisions happen at runtime (e.g., "let this agent access this data for the next 2 hours"). Grants are policy mutations with audit trails.

#### 3. Agent Manifests
- Declarative YAML describing an agent's capabilities, dependencies, identity:
  ```yaml
  name: "compliance_reviewer"
  version: "1.0.0"
  identity:
    principal: "compliance_bot@corp"
    is_admin: false
    roles: ["compliance_reviewer"]
  capabilities:
    - tool: "read_email"
      subjects: ["escalation"]
    - tool: "approve_transaction"
      subjects: ["all"]
      requires_approval: true
  dependencies:
    - service: "email_mcp"
      version: ">=1.0"
    - service: "policy_store"
  budget:
    max_calls_per_run: 50
    max_tokens_per_run: 50000
    max_cost_per_run: 5.00
  ```
- Manifests are versioned and signed (Ed25519 in 0.6)
- Deployed via: `kognita fleet deploy --manifest compliance_reviewer.yaml`

**Rationale:** Manifests are the contract. They tell operators what the agent can do, what it depends on, and what it costs. Deployment is declarative and auditable.

#### 4. Fleet Controls
- Deployed agents register with fleet controller
- Controller enforces:
  - Quota per agent (e.g., compliance_reviewer can only use 1000 tokens/day)
  - Isolation: agents cannot read other agents' runs
  - Coordination: service-to-service calls must be authorized (agent A calling agent B's tool goes through governance)
- Dashboard: see all agents, their quotas, their recent runs

**Rationale:** Fleets are heterogeneous. Controls prevent one rogue agent from consuming quota or accessing sensitive runs of peer agents.

### Definition of Done

- [ ] Delegation implemented: Run.delegate_to(), attenuation logic, evidence integration
- [ ] Grant lifecycle: issue, use, revoke, expiry logic, evidence
- [ ] Manifest schema: YAML parser, validation, versioning
- [ ] Fleet controller: registration, quota enforcement, isolation, dashboard
- [ ] Inter-agent calls authorized: agent→agent tool calls go through governance
- [ ] Tests: 20+ test cases covering delegation chains, grant edge cases, quota exhaustion
- [ ] Examples: 2–3 scenarios (e.g., sales → compliance → execution handoff)

### Effort Estimate
- Delegation: 2–3 weeks
- Grants: 1–2 weeks
- Manifests: 2 weeks
- Fleet controller: 3–4 weeks
- Docs: 1 week
- **Total: ~10–13 weeks (Q2–Q3 2026)**

---

## 0.6 "Trust" — Cryptographic Proof and Compliance

**Timeline:** Q3–Q4 2026  
**Goal:** Make evidence tamper-proof via Ed25519 signatures and exportable to compliance auditors.

### Core Features

#### 1. Evidence Signing with Ed25519
- Generate signing keypair on first run: `kognita init --generate-key`
- Each evidence event includes: `{payload, signature, previous_hash, timestamp}`
- Signature proves: "this payload was recorded by kognita on this server at this time"
- Verification command: `kognita evidence verify --db evidence.db --public-key pub.pem`
  - Walks the chain, verifies every signature
  - Detects insertion, deletion, tampering
  - Reports first break in chain

**Rationale:** Hash-chaining is tamper-*evident* (detects tampering). Signing makes evidence tamper-*proof* (cryptographically binds to issuer). Required for regulated industries (finance, healthcare, legal).

#### 2. Postgres Backend (Migration from SQLite)
- Optional migration: `kognita migrate --from sqlite:evidence.db --to postgres://…`
- Postgres backend:
  - Better concurrency (multiple agents writing simultaneously)
  - Built-in replication (for HA deployments)
  - Native JSON query support (policy analysis queries)
  - Row-level security: agents cannot query other agents' runs

**Rationale:** SQLite is great for prototypes. Postgres scales to production use cases.

#### 3. External Review API
- Governors (compliance officers, auditors) can request a review of an agent's decisions:
  ```python
  kognita review request --agent compliance_bot --from 2026-01-01 --to 2026-02-01
  ```
- Request creates an immutable, signed record
- Reviewer fetches paginated runs, annotates with findings
- Submission is signed and stored in evidence chain
- Report includes: decisions reviewed, findings, reviewer identity, timestamp

**Rationale:** Audit trails are useless if nobody reads them. External reviews formalize the governance assurance process.

#### 4. TypeScript Evidence Verifier (External Tool)
- Standalone verifier: `npm install @kognita/verify`
- `verify_evidence.ts`: import + check evidence signatures without Python
- Used by: auditors, regulators, external parties who need to verify evidence integrity
- Output: JSON report with verification status, any breaks in chain

**Rationale:** Evidence should be portable and verifiable by anyone. TypeScript verifier lets external parties audit without installing Python Kognita.

#### 5. Schema Versioning
- Evidence schema versioned: `{version: "2.0", payload: …, signature: …}`
- Migrations maintain backward compatibility: v1 events readable by v2 code
- CLI: `kognita evidence export --db evidence.db --output-version 1.0` → export in old format for legacy systems

**Rationale:** Systems evolve. Evidence must survive schema changes without losing provenance.

### Definition of Done

- [ ] Ed25519 signing integrated: generate key, sign on write, verify on read
- [ ] Verify command: detects tampering, reports first break, exit code on failure
- [ ] Postgres migration: schema, row-level security, concurrency tests
- [ ] External review API: request, fetch, annotate, submit, evidence integration
- [ ] TypeScript verifier: can verify Kognita evidence without Python
- [ ] Schema versioning: migrate v1 to v2, backward-compatible reads
- [ ] Tests: 25+ test cases (signing, tampering, schema migration, verifier)
- [ ] Docs: "Security Architecture", "Audit Trails for Regulators", verifier guide

### Effort Estimate
- Ed25519 signing: 2 weeks
- Postgres migration: 2–3 weeks
- External review API: 2 weeks
- TypeScript verifier: 2 weeks
- Schema versioning: 1 week
- Docs: 1 week
- **Total: ~11–13 weeks (Q3–Q4 2026)**

---

## 1.0 "Ready for Production" — Finalization & Polish

**Timeline:** Q4 2026–Q1 2027  
**Goal:** Performance benchmarks, comprehensive docs rewrite, 10 single-file examples, and official v1.0.0 release.

### Core Work

#### 1. Performance Benchmarks
- Baseline every component:
  - Decision latency: p50, p99 (target: <10ms)
  - Evidence write throughput (target: >1000 events/sec)
  - MCP proxy overhead (target: <50ms)
  - Run resume time (target: <100ms)
- Public report: `docs/benchmarks/v1.0.md` with graphs, hardware specs

#### 2. Docs Rewrite
- Current README is architecture-first. Rewrite to be:
  - **Fear-first:** "You need to prove an AI request was allowed before any data moves"
  - **Scenario-first:** 3–4 concrete stories (regulated financial services, healthcare, retail)
  - **Glossary:** approve-then-apply, attenuation, citation, egress, etc. (for non-technical readers)
  - **Comparison:** Kognita vs. content guardrails, LangChain, CrewAI, GraphRAG (in what scenarios you need each)
- API docs auto-generated from docstrings (Sphinx + ReadTheDocs)
- Migration guide: 0.5 → 1.0 (schema changes, adapter updates)

#### 3. 10 Single-File Examples
1. **Flask agent** (150 lines): simple chatbot with tool access, governance, evidence in SQLite
2. **Claude SDK integration** (120 lines): agent using Kognita adapter
3. **OpenAI Agents SDK** (120 lines): same, with OpenAI
4. **MCP proxy** (80 lines): config file + invocation for gmail MCP
5. **Policy-only governance** (100 lines): YAML policies, no Python
6. **Multi-agent fleet** (200 lines): 3 agents, shared Run budget
7. **Approval workflow** (150 lines): agent requests approval, waits, resumes
8. **Egress guard** (120 lines): redacts PII before sending to OpenAI
9. **Evidence audit** (80 lines): read evidence chain, export to JSON, verify
10. **Custom domain pack** (180 lines): build a pack for domain-specific subjects/attributes

Each example:
- Runnable in <5 minutes (includes data setup)
- Commented for clarity
- Linked from README under "Examples"
- Tested in CI

#### 4. Marketing & Positioning
- Blog post series:
  1. "Why Governance Matters: The AI Risk You're Not Measuring"
  2. "How We Built Tamper-Evident Audit Logs for Agents"
  3. "Comparing Kognita to (Framework X)" (choose 3–4 frameworks)
- Talk proposals for: QCon, PyCon, AI Systems & Safety conferences
- Timing announcement: "Kognita 1.0: agent governance for regulated industries"
- HN/Reddit: "Show HN: Kognita — prove your AI agent was authorized before it ran"

#### 5. Graduation Checklist
- [ ] All tests pass (coverage >85%)
- [ ] Zero known bugs (Tier 0 closed in 0.3, ongoing triage)
- [ ] Benchmarks published (all p99 latencies <100ms)
- [ ] Docs complete (README rewrite, API docs, 10 examples, glossary)
- [ ] GitHub: 1000+ stars, 30+ contributors, active discussions
- [ ] Production deployment: ≥1 customer using in regulated domain
- [ ] Security audit: external review completed
- [ ] Semantic versioning: 1.0.0 released, changelog updated

### Effort Estimate
- Benchmarks: 1–2 weeks
- Docs rewrite: 2–3 weeks
- 10 examples: 2–3 weeks (can parallelize)
- Blog/talks: 1 week (spread across release cycle)
- Graduation checklist: ongoing throughout Q4 2026
- **Total: ~6–10 weeks (Q4 2026–Q1 2027)**

---

## Go-to-Market Strategy

### Positioning: "Governance is a feature, not a burden."

**For developers:** "Drop in a decorator, get tamper-evident audit logs and policy enforcement. No framework swap."

**For compliance teams:** "Every agent decision cites the rule it came from. No guessing, no re-auditing."

**For operators:** "Budget agent spend per user per day. One misconfigured agent can't drain your API budget."

**For startups:** "Governance is table stakes in regulated industries. Kognita makes it cheap to ship safely."

### Adoption Funnel

**Phase 1: Awareness (0.3 release)**
- Launch HN: MCP proxy demo (5 min end-to-end)
- Tweet storm: 3 threads on governance patterns (attenuation, grants, quotas)
- Blog: "Agents Gone Wrong" (3 case studies of what happens without governance)
- Target: 500+ GitHub stars, 5K+ visitors to site

**Phase 2: Trial (0.4 release)**
- Adapters: one per popular framework (Claude SDK, LangGraph, OpenAI SDK)
- Starter packs: "I have a policy, I just need code" (RBAC, geo-fence, approval-chain)
- Target: 10K+ PyPI downloads, 50+ GitHub issues (signal of traction)

**Phase 3: Adoption (0.5 release)**
- Fleet controls: "now run 10 agents, safe"
- Case study: 1 customer (ideally bank, healthcare, or legal tech) going prod
- Target: 20+ production deployments, 1K+ stars added

**Phase 4: Trust (0.6 release)**
- Signing + compliance audit report: "evidence is tamper-proof"
- Postgres: "scales to 100M events/month"
- Target: Adoption by regulated institution (FI or healthcare)

**Phase 5: Dominance (1.0 release)**
- "The de facto harness for governed agents"
- Target: 10K+ downloads/month, adoption across 3+ industries

### Why Kognita Wins

1. **No framework lock-in.** Works with Claude, OpenAI, LangGraph, Pydantic AI, etc. via MCP proxy and adapters.
2. **Fail-closed by design.** Not a content filter bolted on top. Governance is baked into execution.
3. **Every decision is cited.** No black-box denials. Users know which rule said no, and can appeal.
4. **Tamper-evident.** Audit trail can't be edited retroactively. Proof of legitimacy.
5. **Minimal setup.** Flagship demo works in 5 minutes. Policy is YAML. No boilerplate frameworks.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Framework fatigue: Kognita is yet another framework | MCP proxy removes lock-in; adapters are <200 LOC; policy is just YAML |
| Adoption inertia: teams won't retro-fit governance | Start with MCP proxy (no code changes needed); position as cost control + compliance, not just policy |
| Performance concerns: governance adds latency | Benchmark every release; keep decision latency <10ms |
| Regulatory churn: rules change mid-year | Policy is version-controlled; diffs are readable; approval chains govern policy changes |
| Competitor entry: LangChain/Anthropic build similar | Move fast (0.3 critical release Q4 2025); differentiate on MCP proxy + signatures |

---

## Measuring Success

- **GitHub:** 10K stars by 1.0
- **PyPI:** 50K+ downloads/month by 1.0
- **Adoption:** 100+ organizations, 5+ regulated deployments by 1.0
- **Community:** 50+ contributors, active discussions/issues
- **Benchmarks:** p99 decision latency <10ms, evidence throughput >1000 events/sec
- **Security:** External audit passed, zero CVEs

---

## Key Decisions & Trade-offs

### Why MCP proxy first?

Most vendors go "build a framework, convince people to adopt." We're going "MCP proxy, zero adoption friction, then framework adapters."

**Trade-off:** MCP proxy is a larger engineering effort (3–4 weeks) than a simple SDK. But payoff is adoption across every agent platform at once. BMOS didn't have this; Kognita does.

### Why YAML policies, not Python?

Policy is governance, not code. Governance teams don't hire Python engineers. Operators and compliance folks need to *read* and *approve* policy without learning a language.

**Trade-off:** Fewer expressive power than Python (no arbitrary computation in rules). Acceptable: policies are usually simple (role checks, data classifications, time gates). Complex logic goes in domain packs.

### Why Postgres in 0.6, not earlier?

SQLite is fine for small deployments. Moving to Postgres earlier dilutes 0.3–0.5 focus. Once you have product-market fit (0.5), then optimize for scale.

**Trade-off:** Early adopters will hit SQLite concurrency limits. Mitigate with clear docs: "SQLite is for prototypes; move to Postgres before production."

### Why external review API, not built-in dashboard?

Dashboard is operational (see what happened). Review is governance (formal proof that someone audited it). They're different processes.

**Trade-off:** Slightly more complex UX. Payoff: external parties (auditors, regulators) can verify evidence without full Kognita deployment.

---

## Next Steps (Post 0.3)

1. Create GitHub milestone for 0.3: assign all Tier 0 defects + Run context + MCP proxy
2. Break 0.3 into 4-week sprints:
   - Sprint 1: Run context infrastructure, evidence changes
   - Sprint 2: suspend/resume logic, tests
   - Sprint 3: GovernedModel adapters (Anthropic + OpenAI)
   - Sprint 4: MCP proxy + flagship demo
3. Kick off 0.4 planning: adapter contracts, policy language grammar
4. Publish timeline on GitHub wiki
5. Monthly public updates: "Kognita Status: [release], [highlights], [blockers]"
