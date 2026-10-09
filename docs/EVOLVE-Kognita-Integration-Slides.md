# EVOLVE + Kognita: Complete Governance Stack Slide Brief

**Purpose:** Extend the EVOLVE AI-native enterprise framework with Kognita's enforcement, evidence, and resilience layer to create a complete AI governance stack.

**Total new slides:** 21 (distributed across the five maturity stages plus three new sections)

**Attribution:** EVOLVE framework authored by Ahmed Muzammil (CC BY SA 4.0). Kognita governance patterns from Kognita v0.2 roadmap. Integration authored by Claude for Ahmed Muzammil.

---

## Part 1: Critical Enforcement Layer

### Slide 1: "Before Any Data Moves: The Decision Layer"

**Stage:** Experimental (insert after Slide 11)

**Hero Text:**
> Permission is binary. No data leaves until every rule agrees.

**Diagram:**
```
Human Intent
    ↓
Envelope (who, what, why, where, about whom)
    ↓
Policy Evaluation (EVERY rule checked)
    ↓
Decision Hierarchy:
  DENY (one "no" blocks all)
    ↓
  ESCALATE (uncertain? ask human)
    ↓
  HUMAN_APPROVAL (high-risk actions need sign-off)
    ↓
  ALLOW (all rules pass)
    ↓
Data/Tool Access (only after decision)
    ↓
Execution (with evidence)
```

**Key Points:**
- One failing rule among a hundred passing rules still denies the request
- Every decision cites the rule and the evidence
- The decision is deterministic: same inputs, same output, always reproducible
- Nothing changes between decision and execution

**Visual Notes:**
- Left side: a red circle with "DENY" for each failing policy
- Center: a gold circle with "ALLOW" only when all pass
- Right side: data only flows after permission
- Color: red for DENY, amber for ESCALATE, gold for ALLOW

**Delivered By:** Core (v0.2) + 0.3 Gateways

**Why This Matters for EVOLVE:**
EVOLVE shows *how* RMs make decisions and *what* gets measured. This slide shows *when* the AI system checks whether the decision is *allowed*. It's the guardrail that keeps the learning loop from becoming a permission bypass.

---

### Slide 2: "The Dual Check: Authority Without Trust"

**Stage:** Enabled (insert after Slide 14)

**Hero Text:**
> "RM can access this, AND agent is authorized to access it for this task."
> Not cascading. Not inherited. Not assumed.

**Diagram:**
```
Human Authority                Agent Authorization
(RM's entitlements)       AND   (Agent's delegation)
     |                              |
     v                              v
  Portfolio              +      Specific task
  Risk data             +      Time window
  Client records        +      Data scope
  Market research       +      Tool set
                              + Approval threshold
     |                              |
     └──────────────┬───────────────┘
                    v
          DUAL CHECK PASSES
                    |
                    v
          Agent may access
          (only what both allow)
```

**Key Points:**
- An agent's access is ALWAYS narrower than the human's
- Delegation is specific: which client, which data, which action, which time period
- Delegation expires: on time, on task completion, or on revocation
- Jurisdiction and approval threshold are part of the delegation
- A human's entitlement does not auto-grant their agent access

**Visual Notes:**
- Left column: human icon with a briefcase (access)
- Right column: robot icon with a clipboard (authorization)
- Center: two arrows converging on a green checkmark
- At bottom: "Revocable on next decision" in small text

**Delivered By:** Delegated authority (0.5)

**Why This Matters for EVOLVE:**
EVOLVE's "agents execute" stage (Empowered) needs a hard rule: agents don't inherit human access. This is how you prevent the agent from becoming a back-door to the wrong data.

---

### Slide 3: "Autonomy Levels: L0 to L6 with Evidence Gates"

**Stage:** Enabled (insert after Slide 15, paired with dual check slide)

**Hero Text:**
> Autonomy earns its expansion. One bad metric doesn't earn the next level.

**Diagram:**
```
AUTONOMY STAIRCASE (with evidence gates between levels)

L0: Retrieve              ┌─ Quality ≥ 95%
    (Read only)           │  Exception rate < 5%
         ↓                │  Full traceability
    [GATE 1]    ─────────┘
         ↓
L1: Explain              ┌─ Quality ≥ 95%
    (Summarize)          │  Exception rate < 5%
         ↓                │  Full traceability
    [GATE 2]    ─────────┘
         ↓
L2: Recommend           ┌─ Quality ≥ 95%
    (Propose)           │  Exception rate < 5%
         ↓               │  Full traceability
    [GATE 3]   ─────────┘
         ↓
L3: Draft               ┌─ Quality ≥ 95%
    (Prepare content)    │  Exception rate < 5%
         ↓               │  Full traceability
    [GATE 4]   ─────────┘
         ↓
L4: Act After Approval  ┌─ Quality ≥ 95%
    (Execute, then review) │  Exception rate < 5%
         ↓               │  Full traceability
    [GATE 5]   ─────────┘
         ↓
L5: Bounded Autonomy    ┌─ Quality ≥ 98%
    (Execute, pre-approved) │  Exception rate < 2%
         ↓               │  Zero drift observed
    [GATE 6]   ─────────┘
         ↓
L6: Broad Autonomy
    (Independent operation)

AUTOMATIC STEP-DOWN:
If metrics fall below threshold → Agent drops one level on next decision
Knowledge Lead + Owner alerted
Recovery: must pass promotion gate again
```

**Key Points:**
- Every tool declares which level it requires
- Agent's maximum level is capped by use-case tier
- Promotion requires: owner approval + independent validation + evidence thresholds met
- Step-down is automatic and immediate (no human delay)
- Quality is tracked as trend per agent, model version, prompt version (drift detection)

**Action Permissions (vertical slice):**
```
Read ≠ Analyse ≠ Recommend ≠ Draft ≠ Queue ≠ Send ≠ Execute
Each is a separate permission. Recommend never implies Execute.
```

**Visual Notes:**
- Staircase in ascending order, left to right
- Gate symbols between each level (traffic light style)
- Three colored metrics for each gate (green, amber, red thresholds)
- Dotted line from L6 down with arrow label "Automatic if metrics fall"
- Bottom action permission row shows seven distinct boxes with "≠" between them

**Delivered By:** Autonomy levels (0.5) + Evidence gates (0.5)

**Why This Matters for EVOLVE:**
EVOLVE's "Empowered" stage says agents make decisions. This slide shows the hard evidence standard that earned them that right, and the automatic consequence if they lose it. It's how you prevent "one bad decision goes unnoticed and becomes systemic."

---

## Part 2: Critical Evidence & Reconstruction

### Slide 4: "Tamper-Evident Evidence: The Hash Chain"

**Stage:** Embedded (insert after Slide 19)

**Hero Text:**
> One altered byte breaks every hash that comes after it.

**Diagram:**
```
Event 1: Decision made
├─ Content: [policy + attributes + outcome]
├─ Hash(Event 1) = abc123
└─ Prev_Hash = [none]
         ↓
Event 2: Data retrieved
├─ Content: [retrieval ID, item hash, timestamp]
├─ Hash(Event 2) = def456
└─ Prev_Hash = abc123  ← Event 1's hash
         ↓
Event 3: Model call
├─ Content: [model version, prompt hash, response hash]
├─ Hash(Event 3) = ghi789
└─ Prev_Hash = def456  ← Event 2's hash
         ↓
Event 4: RM review
├─ Content: [RM ID, decision, edits]
├─ Hash(Event 4) = jkl012
└─ Prev_Hash = ghi789  ← Event 3's hash
         ↓
                    ↓ CHAIN BREAKS HERE
         ↓
VERIFICATION:
$ kognita evidence verify --db store.db
BROKEN: evidence chain broken at sequence 2
        payload does not match its hash
        someone altered a record after the fact

EXPORT:
$ kognita evidence export --db store.db -o audit.json
→ Portable, self-verifying audit
→ Content hashes let external auditor verify nothing changed
```

**Key Points:**
- Each event carries the hash of the event before it
- Payloads hold hashes + references, not full content (for GDPR/erasure compliance)
- Content stays in separate retention store, keyed by hash
- Tampering is detectable, not preventable (data ownership is auditor's job)
- Export produces JSON that auditor can verify independently

**Visual Notes:**
- Vertical chain of boxes, each linked to the previous by a hash reference
- Break point clearly marked with red ✗
- Lock icon on the unbroken portion
- Broken lock icon on the broken portion
- Command-line examples in monospace font

**Delivered By:** Core (v0.2) Hash chain + 0.3 Pinned evidence

**Why This Matters for EVOLVE:**
EVOLVE's outcome metrics and learning loop depend on accurate records. This slide proves the records can't be secretly edited. It's compliance confidence: "We measured this, and here's proof we didn't change it."

---

### Slide 5: "The Ten Questions: Supervisory Reconstruction"

**Stage:** Evolved (insert before Slide 28)

**Hero Text:**
> A regulator picks one interaction. You answer every question. From evidence alone.

**The Ten Questions:**
```
1. WHY THIS CLIENT?
   ↳ Origination evidence: what triggered the recommendation?
   ↳ Selection criteria, candidate set size, ranking that put them first
   ↳ Timestamp, RM or system-triggered, rationale

2. WHY THIS PRODUCT?
   ↳ Product eligibility rules evaluated at decision time
   ↳ Suitability checks run before recommendation
   ↳ Which rules passed, which were checked

3. WHAT DATA DID THE AI USE?
   ↳ Retrieval evidence: every item fetched, with source, owner, as-of date
   ↳ Content hash of each item (data governance audit)
   ↳ Freshness check: was it within the threshold for this use case?

4. WHICH MODEL PRODUCED IT?
   ↳ Model name, version (from provider)
   ↳ Prompt template version (from registry)
   ↳ Prompt sent (content hash) + response received (content hash)
   ↳ Model card: evaluation results and approval date

5. WHAT WAS THE RM AUTHORIZED TO DO?
   ↳ Delegated authority: what scope, what time window, what approval threshold?
   ↳ Authority lineage: if delegated through other agents, the full chain
   ↳ Expiry: when does this authority end?

6. WHICH CONTROLS RAN?
   ↳ Policy citations: regime + rule ID + rule content hash
   ↳ Basis of decision: which checks passed, which failed, which escalated
   ↳ Classification: did the request come in as typed or was it classified?

7. WHAT DID THE RM SEE?
   ↳ RM review capture: content hash of the exact recommendation rendered
   ↳ What evidence was presented alongside it?
   ↳ When did the RM start reviewing? How long?

8. WHAT DID THE RM CHANGE?
   ↳ Structured diff: before (AI) vs after (RM approval)
   ↳ Which fields changed, which stayed the same
   ↳ Is this within the RM's authority, or did it trigger escalation?

9. WHO DECIDED?
   ↳ Final decision event: RM ID, timestamp, decision (approve/modify/reject/escalate)
   ↳ If two-signature approval: marker + confirmer, both recorded
   ↳ Approval packet that was presented (content hash)

10. CAN YOU REPRODUCE IT?
    ↳ Pinned evidence: policy content hash, data content hashes, model version
    ↳ Reconstruction report: run through the same policies, data, model with same inputs
    ↳ Do we get the same decision? Yes → System is stable. No → Something changed.
```

**Visual Notes:**
- Ten numbered callouts arranged as a circle
- At center: "EVIDENCE CHAIN"
- Each callout points to a different part of the chain
- Color-coded by category: orange (client), blue (data), green (model), purple (control), gold (human)
- Footer: "Reconstruction test: the single measure that matters to a regulator"

**Delivered By:** 0.3 Reconstruction + 0.4 Interaction record + 0.6 Claims + 0.7 Resilience

**Why This Matters for EVOLVE:**
These ten questions are what a regulator will ask after an AI-assisted client action goes wrong or comes under audit. EVOLVE's learning loop answers them by design, but only if Kognita records them. This slide shows the complete record that your maturity journey builds.

---

### Slide 6: "Pinned Evidence: Freezing Inputs to Decisions"

**Stage:** Embedded (insert after Slide 21)

**Hero Text:**
> Decisions depend on inputs. Every input gets locked in place.

**Four Inputs That Get Pinned:**

```
INPUT 1: POLICY
├─ Problem: Policy rows can be edited in place
├─ Solution: Hash of policy content on every check
├─ Result: Edited policy is detected at verification time
└─ Evidence: policy_id + policy_content_hash + effective_from/to

INPUT 2: DATA
├─ Problem: Retrieved items can be edited after the fact
├─ Solution: Hash of each item's content, plus embedding model version
├─ Result: Re-index or edit is immediately visible
└─ Evidence: item_id + content_hash + retrieval_timestamp + source

INPUT 3: MODEL & PROMPT
├─ Problem: Model versions change, prompts evolve, outputs can't be reproduced
├─ Solution: Pinned model version, prompt template version, prompt content hash, response hash
├─ Result: Exact reproduction, or "this model/prompt no longer exists"
└─ Evidence: model_name + model_version + prompt_template_version + prompt_hash + response_hash

INPUT 4: TOOL RESPONSE
├─ Problem: Tools can return different data on next call
├─ Solution: Hash of tool response (content in retention store)
├─ Result: "What did the tool say at that moment?"
└─ Evidence: tool_name + response_hash + timestamp + tool_version
```

**Reconstruction Flow:**
```
Question: "Can you reproduce decision X from 6 months ago?"
    ↓
Load original decision with pinned hashes
    ↓
Load current versions of policy, data, model
    ↓
Compare hashes
    ├─ All match? → Re-run with same inputs
    │  ├─ Same outcome? → ✓ System is stable
    │  └─ Different? → ⚠ Something changed (model drift, policy interpretation, data quality)
    └─ Hash mismatch? → "Policy/data/model has changed; reproduction is not 1:1"
```

**Visual Notes:**
- Four stacked cards, each a different color
- Each card shows: input type → problem → solution
- Arrows connecting to a central "Evidence Record" box
- At bottom: branching decision tree for reconstruction

**Delivered By:** 0.3 Pinned evidence + 0.4 Interaction record

**Why This Matters for EVOLVE:**
When your Knowledge Lead asks "did this metric drop because the standard changed or because the data changed?", pinned evidence gives you the answer. It's how you keep learning loop improvements from being confused with model/data drift.

---

## Part 3: Critical Agent Identity & Credentials

### Slide 7: "Agent Identity: Not Self-Asserted"

**Stage:** Empowered (insert after Slide 23)

**Hero Text:**
> "I'm AGENT-PORTFOLIO-REVIEW-01" is not identity. It's a claim.
> Only authenticated credentials prove who you are.

**The Agent Registry:**
```
Agent Record:
├─ Name: AGENT-RM-PRECALL-01
├─ Version: 2.3.1 (deployed 2026-10-04)
├─ Owner (accountable): Head of Wealth Advisory
├─ Risk class: Tier 2
├─ Autonomy level: L3 (Draft)
├─ Maximum autonomy cap: L4 (Act after approval)
│
├─ Purpose: Prepare client review pack before RM meeting
├─ Permitted use cases: [USE-CASE-001, USE-CASE-042]
├─ Permitted systems: CRM, Portfolio platform, House view database
├─ Tool allow-list: retrieve_portfolio, retrieve_cio_research, draft_summary
│
├─ Deployment version hash: sha256(system_prompt + instructions + tools + permissions)
├─ Review date: 2027-01-15 (auto-deny if past this date)
├─ Kill switch: ACTIVE
│
└─ Authentication:
    ├─ Credential type: Workload identity
    ├─ Credential ID: agent-001.workspace.kognita
    ├─ Issued: 2026-10-04 by: Security team
    ├─ Expires: 2027-04-04
    └─ Signed: RSA key + timestamp (external auditor can verify)
```

**On Every Agent Call:**
```
Agent claims: "I'm AGENT-RM-PRECALL-01"
    ↓
System asks: "Prove it. Show credential."
    ↓
Agent provides: Workload credential token (signed)
    ↓
System verifies: Signature valid? Token not expired? Credential matches name?
    ├─ YES → Look up agent in registry
    │  ├─ Registered? → Load agent's autonomy cap, tool allow-list, use-case scope
    │  └─ Not registered? → DENY
    └─ NO → DENY (spoofed agent)
    ↓
Deployment version check:
├─ System prompt changed since registration? → Require re-approval
├─ Tools changed? → Require re-approval
├─ Permissions changed? → Require re-approval
└─ All unchanged? → Proceed (but record which version executed)
```

**Key Points:**
- Agents don't share credentials (no "AI service account")
- Each agent is a separate principal with its own identity
- Workload identities are issued and rotated by security (not the agent team)
- Kill switch: one command denies the agent's *next* request
- Unregistered agents are DENY by default
- Review date: auto-deny past expiry (forces re-approval)

**Visual Notes:**
- Left side: agent with a name badge (labeled "claim")
- Middle: security checkpoint (badge reader)
- Right side: registry lookup (onym-proof check)
- Red path for DENY, green path for verified agent
- Lock icon on credentials

**Delivered By:** Agent identity (0.5) + Tier 0 fix (0.3)

**Why This Matters for EVOLVE:**
EVOLVE says "agents execute," but agents are also attack surfaces. This slide shows how you prevent a compromised agent or a spoofed agent name from becoming a bypass. It's credential-based, not name-based.

---

### Slide 8: "Three Credentials: Human, Agent, Approval"

**Stage:** Empowered (insert after Slide 24)

**Hero Text:**
> Credential hierarchy: None cascade. Each is independent.

**Diagram:**
```
HUMAN (RM) AUTHENTICATES
├─ Credential: Multi-factor auth (FIDO2 + backup)
├─ Authority: "RM can access portfolios, client records, research"
├─ Scope: Assigned clients, desk, jurisdiction
└─ Revocation: Disable RM account → All her delegations void

                        ↓

AGENT AUTHENTICATES  
├─ Credential: Workload identity (separate from human)
├─ Authority: "Agent can draft summaries, retrieve house view, propose recommendations"
├─ Scope: Assigned use cases, tools, data classes (narrower than RM's)
└─ Revocation: Revoke agent credential → Agent cannot call any tool (independent of RM's access)

Note: Having RM credential does NOT auto-grant agent credential.
      Agent delegation check is separate from RM entitlement check.
      Both must pass.

                        ↓

APPROVAL AUTHENTICATES (High-Risk Actions)
├─ Credential: Signature (two-signature approval pattern)
├─ Authority: "Owner_Bob approved sending this to client Jane_Smith"
├─ Scope: Specific envelope hash + content hash (one approval per request)
└─ Revocation: Approval expires in 24h or after use (one-time use per request)

Note: Approval binds to the envelope hash and content hash.
      Same request approved twice is still two separate approvals.
      Approval for request A cannot be replayed for request B (even if similar).
      This prevents "rubber-stamp" approvals of different decisions.
```

**Key Points:**
- **No cascading:** RM auth does not grant agent auth, and vice versa
- **No delegation:** agent cannot raise the authentication level of an instruction (email-to-human → agent → high-risk tool still requires step-up auth)
- **Separation of duties:** different credentials, different revocation paths
- **Approval lifecycle:** issued for one request, expires or is consumed, cannot be reused

**Visual Notes:**
- Three circles (Human, Agent, Approval) arranged in a triangle
- No arrows connecting them (showing they're independent)
- Each circle has its own color (blue human, orange agent, red approval)
- Dotted line around each: "Revoke this → does NOT affect the others"

**Delivered By:** Agent identity (0.5) + Delegated authority (0.5)

**Why This Matters for EVOLVE:**
EVOLVE's "humans approve what they actually reviewed" principle depends on approval being bound to specific content, not replayable. This slide shows the credential infrastructure that enforces it. No approval shortcuts.

---

## Part 4: Critical Runtime Safety & Intervention

### Slide 9: "Runtime Intervention: Observe, Pause, Override, Restrict, Recover"

**Stage:** Empowered (insert after Slide 26)

**Hero Text:**
> An agent is never truly autonomous. A human is always one step away.

**The Five Interventions (Live Run):**

```
OBSERVE: Knowledge Lead watches the run in real-time
├─ What is the agent doing?
├─ Which clients is it accessing?
├─ Which tools is it calling?
├─ What is the outcome so far?
└─ No change to the run; read-only visibility

PAUSE: Halt the run at the next policy boundary
├─ Triggered by: Knowledge Lead command, or anomaly detection
├─ Effect: Run checkpoints at next `decide()` call
├─ Time: Immediate (within milliseconds at policy boundary)
├─ RM can: inspect partial results, resume, or abort

OVERRIDE: Insert a human decision mid-run
├─ Use case: "Agent recommends product X, but RM says product Y instead"
├─ Effect: Override decision is recorded as separate approval
├─ Evidence: "Agent decided X (cite), RM override to Y (cite), RM ID, timestamp"
├─ Consequence: Outcome metrics updated (edit to agent recommendation)

RESTRICT: Tighten permissions mid-run without stopping
├─ Example: "Agent was querying 50 clients, restrict to just this client"
├─ Effect: Next tool call uses restricted scope
├─ Evidence: Restriction event logged with reason + timestamp
├─ Consequence: Future calls to agent use new restricted scope

RECOVER: Undo reversible actions via compensating actions
├─ Registered per tool: draft → delete, queue → cancel, permission → revoke, etc.
├─ Triggered by: Knowledge Lead or auto-triggered by circuit breaker
├─ Flow: Run the compensating actions in reverse order
├─ Evidence: Recovery event lists each action undone + timestamp + who authorized
```

**Workflow Example:**
```
Timeline of a live run:

T=0:00    Agent starts run
T=0:05    Agent retrieves portfolio (allowed)
T=0:10    Agent drafts recommendation
          Knowledge Lead observes: "Good draft"
T=0:15    Agent queues for RM review
T=0:20    Knowledge Lead pauses (sees anomaly: agent trying to query unrelated client)
T=0:21    Knowledge Lead reviews partial results, overrides: agent's data scope was 2 clients, RM restricts to 1
T=0:22    Agent retries with restricted scope
T=0:25    Run completes, RM reviews, approves with minor edits
          RM edit is recorded (automation-bias detection: did RM just rubber-stamp, or actually review?)
T=0:30    Recommendation sent to client
```

**Visual Notes:**
- Timeline at top, with events marked
- Five boxes below timeline: Observe, Pause, Override, Restrict, Recover
- Each box shows: trigger → action → evidence
- Lock/key icons showing: Observe is read-only, others are write operations
- Audit trail showing all interventions recorded with RM ID + timestamp

**Delivered By:** Intervention controls (0.4) + Suspend/resume (0.3)

**Why This Matters for EVOLVE:**
EVOLVE's "empowered" stage means agents make decisions. This slide shows the safety net: every autonomous decision can still be overridden in real-time. It's how you get board approval for L5/L6 autonomy despite the risk.

---

### Slide 10: "Anomaly Detection & Automatic Containment"

**Stage:** Empowered (insert after Slide 25)

**Hero Text:**
> Agents fail fast. Containment is automatic.

**Per-Agent Baseline:**
```
Normal behavior for AGENT-PORTFOLIO-REVIEW-01:
├─ Distinct clients per day: 20–50 (baseline: 35)
├─ Documents retrieved: 15–30 per run (baseline: 22)
├─ Data classes touched: Portfolio, Research, House view (always these three)
├─ Tools called: 5–8 per run (baseline: retrieve_portfolio, draft_summary, retrieve_research)
├─ Query complexity: Simple filters on portfolio ID + date range
└─ Runtime: 30–120 seconds per run (baseline: 60s)
```

**Deviation Detection (Anomaly Signals):**
```
Spike in distinct clients:
├─ Normal: 35 clients/day
├─ Spike: 5,000 clients in one run
├─ Trigger: ANOMALY (attempt to bulk-scan client base)
└─ Action: BLOCK this call, quarantine agent, revoke credential, alert Knowledge Lead

Unusual data classes:
├─ Normal: Portfolio + Research + House view
├─ Attempt: KYC records + Personal data + Credit lines
├─ Trigger: ANOMALY (reaching for data the agent is not designed to touch)
└─ Action: BLOCK, quarantine, revoke, alert

Tool category never seen:
├─ Normal: retrieve_* and draft_* tools
├─ Attempt: send_client_email (agent is not permitted to contact clients)
├─ Trigger: ANOMALY (attempting unauthorized action)
└─ Action: BLOCK, quarantine, revoke, alert

Privilege probing:
├─ Pattern: Agent tries tool X, denied. Retries with different parameters. Denied. Tries third variation.
├─ Trigger: ANOMALY (systematic probing for bypass)
└─ Action: BLOCK, quarantine, revoke, alert

Repeated denied actions:
├─ Pattern: Agent makes same denied request 10 times in 60 seconds
├─ Trigger: ANOMALY (retry loop, either malfunction or deliberate)
└─ Action: BLOCK, quarantine, revoke, alert
```

**Machine-Speed Containment (No Human in Loop):**
```
Anomaly detected
    ↓
Quarantine agent: New calls from this agent are DENY
    ↓
Revoke credential: Agent's workload token is invalidated (immediate)
    ↓
Stop tools: No tool calls from this agent are allowed
    ↓
Alert Knowledge Lead: Page/email with incident details
    ↓
Knowledge Lead decision: Release? Investigate? Modify? (human decides recovery)
    ↓
Recovery options:
├─ Release: Restore credential, re-baseline agent behavior
├─ Restrict: Reload agent with tighter data scope or tool allow-list
├─ Investigate: Download evidence, review what the agent tried to do
└─ Isolate: Keep agent quarantined pending security review
```

**Visual Notes:**
- Left side: baseline curve (normal behavior range in light blue)
- Right side: spike in red (anomaly)
- Center: decision tree showing trigger → action
- Bottom: "Automatic containment: 0ms human delay" emphasized
- Icons: agent, stopwatch, lock, alert bell

**Delivered By:** Anomaly detection (0.5) + Containment (0.5)

**Why This Matters for EVOLVE:**
EVOLVE's learning loop depends on agents having freedom to act. This slide shows the automatic circuit-breaker that catches misbehavior at machine speed, before it scales. It's safety without requiring human monitoring.

---

### Slide 11: "Circuit Breaker: Automatic Policy Insertion"

**Stage:** Evolved (insert after Slide 27)

**Hero Text:**
> Bad guidance doesn't scale. A circuit breaker stops it at threshold.

**How the Circuit Breaker Works:**

```
MONITORING PHASE:
Track denial events per:
├─ Agent (agent_name)
├─ Model version (model_version)
├─ Use-case (use_case_id)
└─ Policy rule (policy_id)

COUNT DENIALS:
├─ Over a 1-hour rolling window
├─ If count exceeds threshold (e.g., 50 denials/hour for one agent)
└─ Trigger: Circuit breaker trip

AUTOMATIC ACTION (No human approval needed):
├─ Insert a NEW prohibiting policy: "Agent X is denied, effective immediately"
├─ Policy cites: "Circuit breaker trip: 50+ denials in 1 hour, [timestamp]"
├─ Take effect: On the NEXT decision (no in-flight action completes)
├─ Notify: Escalate to accountable owner immediately
└─ Record: Circuit breaker event in evidence chain

OUTCOME:
Agent X makes next call
    ↓
Policy evaluation encounters the new prohibiting policy
    ↓
Decision: DENY (no data retrieved, no action taken)
    ↓
Citation: "Circuit breaker trip: [reason] at [time]"
```

**Recovery (Human Gate):**
```
Owner reviews:
├─ Why did the circuit break? (examine the 50+ denials)
├─ What was the agent trying to do?
├─ Is the agent misconfigured, or is it compromised?
└─ Decision: Reconfigure, quarantine, or reset circuit

Reset circuit breaker:
├─ Owner approves reset
├─ Old prohibiting policy is marked expired
├─ Agent can run again (or a new version of the agent can run)
├─ Threshold counter resets
└─ Record reset as governed action
```

**Real Example:**
```
2026-10-04 16:30:00 — Agent-Portfolio-Review starts seeing denial spike
                       (Portfolio platform API rate limit exceeded)

2026-10-04 16:35:42 — Denials hit 50/hour threshold
                       Circuit breaker TRIPS
                       New policy inserted: "AGENT-PORTFOLIO-REVIEW-01 DENY"

2026-10-04 16:35:43 — Agent makes next call
                       Policy evaluation encounters new prohibiting policy
                       Decision: DENY
                       Citation: "Circuit breaker trip: 50 denials/1h at 16:35:42"
                       Evidence: policy_id = CIRCUIT_BREAKER_20261004_001

2026-10-04 16:45:00 — Knowledge Lead investigates
                       Root cause: Portfolio platform API changed its rate limit
                       Fix: Adjust agent's retrieval batch size
                       Redeploy agent with new configuration

2026-10-04 16:50:00 — Owner approves circuit breaker reset
                       Old prohibiting policy marked expired
                       Agent-Portfolio-Review (v2.3.2) released
                       Threshold counter resets to 0
```

**Affected-Client Query:**
```
After circuit breaker trip, run:
  $ kognita evidence affected --agent AGENT-PORTFOLIO-REVIEW-01 \
                              --from 2026-10-04T16:20:00 \
                              --to 2026-10-04T16:35:42

Result:
  Clients who saw recommendations from this agent in that window: 123
  Links to each interaction (RM can assess if guidance was bad)
  If bad guidance was given: escalation to client contact
```

**Visual Notes:**
- Monitoring graph: horizontal line (threshold), spike exceeding it
- Alert: red triangle with "TRIP"
- Flow: trip → insert policy → deny → notify owner
- Recovery path: separate decision tree
- Real example timeline at bottom with timestamps

**Delivered By:** Circuit breaker (0.4) + Granular revocation (0.4)

**Why This Matters for EVOLVE:**
EVOLVE's learning loop measures quality, but quality can degrade fast (model version change, data quality drop, prompt drift). This slide shows how a circuit breaker prevents gradual degradation from becoming a scandal. It's automatic, not manual.

---

## Part 5: Third-Party Risk & Gateways

### Slide 12: "The Gateway Pattern: Explicit, Not Transparent"

**Stage:** Embedded (insert after Slide 22)

**Hero Text:**
> Agents never talk to external systems directly.
> Every call goes through a gateway. Every gateway checks permission first.

**Two Gateways: Models & Tools**

```
GATEWAY 1: AI GATEWAY (in front of model providers)
├─ Listens on: http://localhost:8000/v1 (OpenAI-compatible format)
├─ Upstream: https://api.openai.com (or Groq, Ollama, vLLM, etc.)
│
├─ On every agent call:
│  ├─ Parse request → build Envelope (who, what, why)
│  ├─ Evaluate policy: `decide(envelope, policies)`
│  ├─ If DENY → Return outcome + citations (no bytes reach provider)
│  ├─ If ALLOW → Redact sensitive data using egress guard
│  ├─ Forward to provider with redacted content
│  ├─ Redactor.restore() → Replace redacted spans in response
│  └─ Classify response before returning to agent
│
└─ Result: Every model call is authorized, redacted, and evidenced

GATEWAY 2: MCP GATEWAY (in front of tools/systems)
├─ Listens on: MCP protocol (stdio, SSE, etc.)
├─ Backends: CRM, Portfolio platform, Email, Calendar, etc.
│
├─ On every tool call:
│  ├─ Parse MCP request → build Envelope
│  ├─ Evaluate policy: `decide(envelope, policies)`
│  ├─ If DENY → Return error + citations
│  ├─ If ALLOW → Forward to backend
│  ├─ On tool return → Hash response, record evidence
│  └─ Return result to agent (with evidence reference)
│
└─ Result: Every tool call is authorized, evidenced, and reversible
```

**Key Difference: Explicit vs Transparent**
```
TRANSPARENT INTERCEPTION (old approach):
Agent code: requests go to real providers
Behind the scenes: TLS interception, certificate injection, proxy certs
Problem: Agents don't know they're being governed
Problem: Governance can be bypassed if cert is stolen
Problem: "Transparent" is hard to debug

EXPLICIT GATEWAY (Kognita approach):
Agent code: explicit configuration, agents point to localhost:PORT
Agent knows: "I'm talking to a gateway, not the real provider"
Governance is visible: Denials are obvious, not mysterious network timeouts
Debugging is clear: Gateway logs show every decision
```

**Gateway Failure Modes:**
```
FAIL-CLOSED (default):
├─ Gateway down? API unavailable? Evidence store unreachable?
├─ Action: Refuse all agent calls
├─ Policy: "If governance can't run, no action proceeds"
└─ Cost: RM can't use AI until gateway recovers

DEGRADED (optional):
├─ Gateway down? But agent has local model available?
├─ Action: Allow only local, non-client-data calls
├─ Policy: "Degraded mode: read-only, local model, no client context"
├─ Evidence: Recorded as "degraded mode" once store recovers
└─ Cost: Limited AI, but work doesn't stop
```

**Visual Notes:**
- Left: Agent box
- Center-left: Gateway boxes (AI Gateway + MCP Gateway)
- Center-right: Policy engine (with Envelope → Policy → DENY/ALLOW)
- Right: External systems (Model provider, CRM, etc.)
- Arrows: thick arrows through gateway, thin arrows from agent to gateway
- Red blocked arrow if policy denies
- "Explicit, not transparent" emphasized at bottom

**Delivered By:** AI Gateway (0.3) + MCP Proxy (0.3)

**Why This Matters for EVOLVE:**
EVOLVE's governance is meaningless if agents can bypass it by calling external systems directly. This slide shows why the gateway is the enforcement point, not the policy. Every call, every time.

---

### Slide 13: "Provider Register & Dependency Mapping"

**Stage:** Evolved (insert before Slide 29)

**Hero Text:**
> You don't own the truth. You own the relationship to whoever does.

**Provider Register (What We Know About Each Provider):**
```
Model: OpenAI GPT-4
├─ Owner: OpenAI
├─ Hosted: api.openai.com (US region)
├─ Versions: [4-0314, 4-0613, 4-1106-preview]
├─ Current version: 4-1106-preview (reported by API)
├─ Rate limits: 10K tokens/min, 2K requests/min
├─ Cost: $0.03/1K input tokens
├─ Incident history: [2024-08 outage, 2025-03 rate limit change]
├─ Contracts: [SLA terms, privacy agreement, training-use opt-out]
├─ Exit plan: Fallback to Groq (2-day migration for prompts)
└─ Review date: 2027-01-15

Data Provider: Bloomberg (portfolio data)
├─ Owner: Bloomberg LP
├─ Hosted: data.bloomberg.com (encrypted)
├─ Data freshness: EOD (end of day) + real-time add-on
├─ Retention: 7 years
├─ Incidents: [2023-11 latency spike, corrected]
├─ SLA: 99.5% uptime
├─ Cost: $10K/month per terminal
├─ Contract: Non-disclosure, pass-through charges
└─ Review date: 2027-06-30

Vector DB: Pinecone (embeddings)
├─ Owner: Pinecone
├─ Hosted: api.pinecone.io
├─ Dimension: 1536 (OpenAI embedding model)
├─ Incident history: [None reported]
├─ Cost: $0.10 per 1M stored vectors
├─ Contract: Data processed in US, no training on customer data (written)
└─ Review date: 2027-03-15
```

**Dependency Map (What Depends On What):**
```
Use-Case: Portfolio Advisory
├─ Service: RM-facing agent
├─ Agent: AGENT-PORTFOLIO-REVIEW-01
├─ Model: GPT-4 (OpenAI)
├─ Retrieval: Bloomberg (data) + Pinecone (vectors)
├─ Tools: CRM, Portfolio platform, Email
└─ Risk: If OpenAI is down, agent can't run
          If Bloomberg is down, retrieval returns stale data
          If Pinecone is down, semantic search fails

Concentration Risk:
├─ 40% of use cases depend on OpenAI
├─ 60% of data comes from Bloomberg
├─ Single points of failure: [OpenAI, Bloomberg, Pinecone]
└─ Mitigation: Groq as fallback, maintain manual data feed
```

**Silent Model-Change Detection:**
```
Agent calls OpenAI API
API responds: "model_id: gpt-4-1106-preview-v2" (new version!)
Kognita sees: This is different from the pinned version (4-1106-preview)
Action: Flag as model change
Decision: Treat as revalidation trigger
├─ New model version was not approved
├─ Requires independent validation before use
├─ Record: "Silent model change detected: [old version] → [new version]"
└─ Until approved: Use old pinned version (if available) or DENY
```

**Visual Notes:**
- Registry table on left: provider name, key fields, review date
- Dependency map on right: use-case → service → component → provider
- Color-coding: green (healthy), amber (review upcoming), red (single point of failure)
- Concentration risk pie chart showing % of use-cases per provider
- Alert icon on single points of failure

**Delivered By:** Provider register (0.7) + Dependency map (0.7)

**Why This Matters for EVOLVE:**
EVOLVE's "Evolved" stage says AI is now your operating system. This slide shows that you can't govern what you don't understand. Provider register is how you track dependencies, concentration risk, and provider health across your AI fleet.

---

## Part 6: Data Governance & Claims

### Slide 14: "The Fact Contract: What Data Must Carry"

**Stage:** Embedded (insert after Slide 20)

**Hero Text:**
> Data has a contract. Break it → claim is blocked.

**Every Fact Must Carry:**
```
REQUIRED FIELDS:
├─ Value: "Risk profile: Moderate Growth" (actual data)
├─ Source: "Suitability system" (system of record)
├─ Owner: "Compliance team" (who maintains it)
├─ As-of date: "2026-09-30" (when was it recorded)
├─ Classification: "Confidential" (C2 data classification)
├─ Entitlement: "RM-access" (who may see it)
└─ Freshness threshold: "Must be ≤ 30 days old" (tolerance)

FRESHNESS ENFORCEMENT (PER DATA TYPE):
├─ Portfolio holdings: Near real-time or latest EOD
├─ House view: Current publication (not superseded)
├─ Market prices: From permitted feed, ≤ 15 min old
├─ KYC/AML: Within current review cycle (≤ 12 months)
├─ Suitability: Current approved profile (not expired)
└─ Research: Author + publication date + version

STALE DATA HANDLING:
├─ Automatic refresh: If available, pull fresh data
├─ Suppress: If fresh data unavailable, hide the claim
├─ Flag: If time-critical, alert RM that data is stale
└─ Never silent: "Use stale data quietly" is not allowed
```

**Research Lifecycle (Special Case):**
```
Research item: "Tech sector expected to outperform, 2026"
├─ Author: CIO team
├─ Published: 2026-08-15
├─ Created: 2026-08-10
├─ Last reviewed: 2026-09-01
├─ Effective until: 2027-02-15 (expiry)
├─ Superseded by: [future house view change]
└─ Status: ACTIVE

When CIO supersedes this view (e.g., publishes "Tech sector underperform"):
├─ This research becomes: SUPERSEDED
├─ Result: Recommendations depending on old view are invalidated
├─ Query: `kognita evidence affected --source <id>`
│  Returns: All clients who received recommendations under the old view
│  Action: Escalate to RM for client contact
└─ Future use: New recommendations must cite new view
```

**Visual Notes:**
- Fact card showing all required fields
- Freshness timeline below (with thresholds marked)
- Stale data decision tree: refresh → suppress → flag
- Research lifecycle diagram showing creation → active → superseded → affected clients

**Delivered By:** Fact contract (0.6) + Freshness (0.6)

**Why This Matters for EVOLVE:**
EVOLVE's learning loop depends on quality data. This slide shows the contract that enforces data quality: facts can't sneak into recommendations if they're stale, unsourced, or out of scope. It's the boundary between "what the AI was told" and "what's actually true."

---

### Slide 15: "Claim Types: Fact, Inference, Recommendation"

**Stage:** Embedded (insert after Slide 21)

**Hero Text:**
> Type claims. Unlabelled claims can't reach an RM or client.

**Three Claim Types:**

```
FACT:
├─ Definition: Verified information from an authoritative source
├─ Examples: "Portfolio value: $500K" (Bloomberg), "Risk profile: Moderate" (suitability system)
├─ Requirements: Source cited, value + as-of date
├─ May reach: RM, client (if appropriate)
├─ Cannot be: Inferred without verification
└─ Validation: Verification gate checks for stale data, source validity

INFERENCE:
├─ Definition: AI conclusion drawn from facts
├─ Examples: "Client is likely risk-averse" (from behavior), "Market may correct within Q1" (analysis)
├─ Requirements: Evidence cited (facts it was derived from)
├─ May reach: RM only (with flag: "This is inference, not fact")
├─ Cannot be: Presented as fact to client without RM review
└─ Validation: Verification gate checks for unsupported claims, contradictions

RECOMMENDATION:
├─ Definition: Proposed action based on facts + inferences
├─ Examples: "Rebalance portfolio to 60/40 equities/bonds", "Contact client re: tax optimization"
├─ Requirements: Rationale cited, suitability check passed
├─ May reach: RM for review, then client if RM approves
├─ Cannot be: Sent to client without RM approval
└─ Validation: Verification gate checks for suitability evidence, execution authority
```

**Claim Type Violations (What Gets Blocked):**
```
Unlabelled claim: "Tech sector expected to outperform"
├─ No type specified → Cannot determine source authority
├─ Block from RM view? Probably not (RM can research)
├─ Block from client? YES (client can't verify source)
└─ Error: "Claim must be typed: FACT (cite source), INFERENCE (cite evidence), or RECOMMENDATION (cite rationale)"

Inference masquerading as fact: "Client is wealthy based on portfolio size"
├─ Claim type: INFERENCE (not a fact, it's a conclusion)
├─ But presented as: FACT in client communication
├─ Problem: No source authority, just AI conclusion
└─ Block: "Client communication requires facts or RM-verified inferences, not bare conclusions"

Recommendation without suitability: "Buy tech stock (high growth, high risk)"
├─ Claim type: RECOMMENDATION
├─ But missing: Suitability check (is high risk right for this client?)
├─ Problem: Can't verify this is suitable before sending
└─ Block: "Recommendations require suitability evidence and RM approval"
```

**Verification Gate (0.6) Classifies Each Claim:**
```
Claim: "Portfolio will grow 7% in 2027"
     ↓
Verifier checks:
├─ Type correct? RECOMMENDATION (growth forecast with action implied)
├─ Evidence strong? "Based on historical return 7.2%, inflation 2.5%, fees 0.5%"? VERIFIED
├─ Or weak? Just AI model opinion? INFERRED
├─ Stale? No, derived from current data
├─ Conflicting sources? No
├─ Suitable? Need suitability gate, not verifier
├─ Source approved? Research source is in approved-source register? YES
└─ Final: VERIFIED or INFERRED (policy decides what may proceed)
```

**Visual Notes:**
- Three colored boxes: FACT (blue), INFERENCE (amber), RECOMMENDATION (orange)
- Each box shows: definition, examples, requirements, destination (RM/client)
- Violation examples below each with red ✗
- Verification gate flow showing claim type → evidence check → classification

**Delivered By:** Claim types (0.6) + Verification gate (0.6)

**Why This Matters for EVOLVE:**
EVOLVE's Knowledge Lead needs to know what's been verified and what's inference. This slide shows how claim types enforce the distinction. An RM can't accidentally send an unsupported inference to a client as if it were fact.

---

### Slide 16: "Verification Gate: The Challenger Agent"

**Stage:** Evolved (insert after Slide 28)

**Hero Text:**
> Before an RM sees material output, a verifier tries to prove it wrong.

**The Verification Gate (0.6 item 7):**
```
Material output produced by agent
(recommendation, insight, forecast, etc.)
    ↓
Verification gate activated (for tier 2+ use cases)
    ↓
Registered verifier (agent or service)
├─ Checks for wrong client
├─ Checks for duplicate identity
├─ Checks for stale data
├─ Checks for conflicting sources
├─ Checks for incorrect calculations
├─ Checks for unsupported inferences
├─ Checks for unregistered definitions (governed terms)
├─ Checks for prohibited content
└─ Checks for broken entitlements
    ↓
Verifier classifies each claim:
├─ VERIFIED: Passes all checks
├─ INFERRED: Supported by evidence, but not a fact
├─ UNVERIFIED: No evidence found
├─ CONFLICTING: Multiple sources disagree
├─ STALE: Data used is past freshness threshold
└─ BLOCKED: Violates policy (wrong client, prohibited term, etc.)
    ↓
Policy decides what status may reach RM:
├─ VERIFIED & INFERRED → show to RM
├─ UNVERIFIED → flag for RM review
├─ CONFLICTING → escalate (RM must pick source)
├─ STALE → refresh or suppress
└─ BLOCKED → never show
    ↓
RM sees output with status labels
(e.g., "VERIFIED fact", "INFERRED recommendation", "UNVERIFIED market call")
```

**Verifier Implementation Options:**
```
Option 1: Another AI agent (Challenger)
├─ Specialized verifier trained to find flaws
├─ Can check calculations, source validity, logical consistency
├─ Automated, fast, can run in parallel with main agent

Option 2: External service (API)
├─ Third-party verification (e.g., compliance checker, calculation validator)
├─ Human review (for tier 3+ use cases)
├─ Database lookup (for governed definitions)

Option 3: Hybrid
├─ Fast automated checks (agent verifier)
├─ Escalate to human if confidence is low
```

**Example: Wealth Advisory Recommendation**
```
Agent produces: "Rebalance to 60% equities, 40% bonds (growth + stability balance)"
    ↓
Verifier checks:
├─ FACT: "Current allocation 80% equities, 20% bonds" → Verified from suitability system
├─ FACT: "Client risk profile Moderate Growth" → Verified from current KYC
├─ CALCULATION: "60/40 is optimal for Moderate Growth" → Check against policy models
│  ├─ Verified? Yes (matches risk model)
│  └─ Recent? Yes (risk model updated 2026-09-30, within threshold)
├─ SUITABILITY: "Is 60/40 suitable?" → Cross-check with suitability gate (not verifier's job)
├─ INFERENCE: "This is better than current 80/20" → Based on fact + calculation
│  ├─ Supported? Yes (lower volatility, maintains growth)
│  └─ Grade: INFERRED (true, but depends on market assumptions)
└─ FINAL: Claim is VERIFIED + INFERRED (mixed)
    ↓
Policy says: "Mixed claims shown to RM with labels"
    ↓
RM sees:
"Rebalance to 60/40 [based on verified data + inferred optimization]
Rationale: Lower volatility while maintaining growth trajectory [calculation verified]"
```

**Visual Notes:**
- Material output box at top
- Verification gate showing six checks (each with icon)
- Classification outcomes (VERIFIED, INFERRED, UNVERIFIED, etc.) with colors
- Policy decision tree: which statuses → RM view
- RM display showing labels on each claim type

**Delivered By:** Verification gate (0.6) + Challenged agent (0.6)

**Why This Matters for EVOLVE:**
EVOLVE's RM review depends on the RM knowing what's been fact-checked and what's inference. This slide shows an automated challenger that flags weak claims before the RM sees them. It's the first line of quality assurance.

---

## Part 7: Egress & Redaction

### Slide 17: "Egress Guard: Decide Per Classification & Destination"

**Stage:** Embedded (insert after Slide 22, before data governance section)

**Hero Text:**
> Data leaving the system is as governed as data entering it.

**The Egress Decision:**
```
Agent is about to call external model (OpenAI)
with client portfolio data
    ↓
Egress guard intercepts:
├─ Content classification: C2 (Confidential, client PII)
├─ Destination: api.openai.com (external, third-party)
├─ Context: Client portfolio summary for model API call
    ↓
Decision: ALLOW / REDACT / DENY
├─ ALLOW: Safe to send as-is (non-sensitive data, internal destination)
├─ REDACT: Send redacted version, restore original in response
└─ DENY: Cannot send (too sensitive, no valid use case)
    ↓
REDACTION EXAMPLE:
Before:  "Client Jane Smith holds 100 shares of ACME Corp, worth $5,000"
After:   "[CLIENT_1] holds [NUM_1] shares of [TICKER_1], worth [VALUE_1]"
    ↓
Provider sees:  Redacted version (never sees Jane Smith, ACME, etc.)
Caller gets:    Original values restored (no data loss on client side)
Evidence:       Redaction manifest hash recorded (content never logged)
```

**Redaction Patterns (Built-in):**
```
Pattern-based redaction catches:
├─ Email addresses: jane.smith@company.com → [EMAIL_1]
├─ Phone numbers: +1-555-0123 → [PHONE_1]
├─ IBANs: DE89370400440532013000 → [IBAN_1]
├─ Card numbers: 4532-1234-5678-9010 → [CARD_1]
├─ URLs: https://internal.company.com/path → [URL_1]
└─ Dates (sensitive): 1980-05-15 → [DATE_1]

Custom terms (provided per use case):
├─ Client names: "Jane Smith", "Ahmed Muzammil" → [TERM_1], [TERM_2]
├─ Internal codes: "PORTFOLIO-ABC-123" → [CODE_1]
├─ Nicknames: "The Big Ones" (top 10 clients) → [NICKNAME_1]
└─ Brand names: "Acme Corp", "Corp X" → [BRAND_1]
```

**Important Limitation:**
```
PatternRedactor is a floor, not a guarantee.
├─ It catches: Email, IBAN, card, phone, URL patterns (and custom terms)
├─ It misses: Names in prose, nicknames, coded references
├─ Deployment: Use NER (Named Entity Recognition) model for better recall
├─ Test: Adversarial suite includes injection tests for redaction bypass
```

**Evidence Recording:**
```
Once redaction is applied:
├─ Do NOT log the original content (violates GDPR log minimization)
├─ DO log the redaction manifest: which patterns matched where
└─ Example: {email_count: 3, phone_count: 2, custom_terms: ["Jane Smith", "Ahmed"]}

Recovery:
├─ Caller can restore originals (they sent them)
├─ Auditor can verify redaction was applied (manifest matches)
├─ Provider never saw the original
```

**Visual Notes:**
- Left side: content + classification
- Center: decision gate (ALLOW/REDACT/DENY)
- Right side: provider receives redacted version
- Below: before/after redaction example
- Patterns chart showing built-in + custom categories
- Warning note: "Floor, not guarantee"

**Delivered By:** Egress guard (Core v0.2) + Classification-driven (0.3)

**Why This Matters for EVOLVE:**
EVOLVE's RM approval gates control what content reaches clients. This slide shows that the *reverse* is also gated: what content leaves the system to external models must be redacted. It's the other half of data protection.

---

## Part 8: Regulatory Mapping

### Slide 18: "The Control Frameworks: MAS, FCA, APRA, SFC"

**Stage:** Evolved (insert after Slide 29)

**Hero Text:**
> Governance is not your invention. It's your regulator's requirement.

**Two Frameworks (40 + 70 Controls):**
```
FRAMEWORK 1: BANK-GRADE AGENTIC AI (40 domains)
├─ Scope: Private banking, wealth management, agentic AI
├─ Sources: MAS, ABS, NIST, OWASP, production patterns
├─ Non-negotiables: 12 critical controls (e.g., runtime policy enforcement)
├─ Domains: Governance, Agent identity, Delegated authority, Data governance, etc.
└─ Status: Only 1/12 non-negotiables fully covered in v0.2 (runtime policy)

FRAMEWORK 2: WEALTH MANAGEMENT AI (70 controls)
├─ Scope: RM-facing, client-facing, agentic AI
├─ Sources: MAS, PDPC, SFC, FCA, Bank of England, APRA, ASIC, CSBS, BIS/FSI, ESMA
├─ Non-negotiables: 15 critical controls (use-case inventory, delegation, evidence)
├─ Domains: 10 (Governance, Risk, Identity, Data, Assurance, Client, Control, Runtime, Resilience, Evidence)
└─ Status: 7/15 non-negotiables covered or planned in roadmap
```

**The Ten Domains (Wealth AI Framework):**
```
1. GOVERNANCE: Who owns the AI? Who remains accountable?
   ├─ Kognita: Use-case register, owner roles, AI risk appetite
   ├─ EVOLVE: Knowledge Lead as quality standard owner

2. RISK CLASSIFICATION: How consequential is this use case?
   ├─ Kognita: Multi-axis classification (criticality, autonomy, consequence, sensitivity)
   ├─ EVOLVE: Outcome metrics (cycle time, quality, reliability)

3. IDENTITY & AUTHORITY: Who is the agent? What may it do?
   ├─ Kognita: Agent identity (not self-asserted), delegated authority, autonomy L0-L6
   ├─ EVOLVE: Humans approve consequential outputs

4. DATA & PRIVACY: What may agents access and retain?
   ├─ Kognita: Entitlement filtering, field minimization, governed memory
   ├─ EVOLVE: Outcome-driven learning (inferences can become facts with RM approval)

5. MODEL & AGENT ASSURANCE: Does it behave reliably?
   ├─ Kognita: Evaluation gate, adversarial suite, evidence gates for autonomy
   ├─ EVOLVE: Evidence-gated autonomy, automatic step-down on drift

6. CLIENT & CONDUCT: Could AI affect advice, suitability, communications?
   ├─ Kognita: Origination (why this client), claims (fact/inference/recommendation)
   ├─ EVOLVE: RM review capture, structured approval packets

7. HUMAN CONTROL: Where can humans understand, intervene, override?
   ├─ Kognita: RM review, intervention controls (observe, pause, override, restrict, recover)
   ├─ EVOLVE: Knowledge Lead escalations

8. RUNTIME & CYBER: Can abnormal behavior be detected and contained?
   ├─ Kognita: Anomaly detection, automatic containment, circuit breaker
   ├─ EVOLVE: Outcome metrics trigger intervention

9. RESILIENCE & THIRD PARTIES: What happens when providers fail?
   ├─ Kognita: Provider register, gateway failure modes (fail-closed vs degraded)
   ├─ EVOLVE: Business continuity (RM work continues without AI)

10. EVIDENCE & AUDITABILITY: Can you reconstruct what happened?
    ├─ Kognita: Hash chain, pinned evidence, reconstruction report
    ├─ EVOLVE: Every correction and outcome feeds institutional learning
```

**Coverage Summary:**
```
FRAMEWORK 1 (Bank-Grade):  
├─ Covered (v0.2): 1/40 domains (runtime policy enforcement)
├─ Planned (0.3-0.7): 25/40 domains
├─ Added (gaps): 14/40 domains
└─ Status: 88% of gaps addressed in roadmap

FRAMEWORK 2 (Wealth AI):
├─ Covered (v0.2): 3/70 controls
├─ Planned (0.3-0.7): 45/70 controls
├─ Added (gaps): 22/70 controls
└─ Status: 96% of gaps addressed in roadmap
```

**Visual Notes:**
- Framework 1 and Framework 2 side by side (40 vs 70 controls)
- Ten domains as circular icons or hexagons
- Each domain color-coded: Kognita component (left), EVOLVE component (right)
- Coverage bar chart: Covered / Planned / Added
- Title: "Regulatory compliance is built into the architecture, not bolted on"

**Delivered By:** Control frameworks (0.4 onwards) + EVOLVE integration

**Why This Matters for EVOLVE:**
EVOLVE is a business transformation framework. This slide connects it to *regulatory* requirements. It answers the question: "Are we building the right thing according to MAS/FCA/SFC?" The answer is: EVOLVE + Kognita together address 96% of the 70 controls.

---

### Slide 19: "The Five Charges: What Supervisors Ask"

**Stage:** Evolved (insert after Slide 30)

**Hero Text:**
> Supervisors ask five hard questions about AI in your firm.
> Kognita answers each one.

**The Five Charges (FCA/MAS Supervisory Pattern):**

```
CHARGE 1: Use-Case Inventory & Independent Validation
├─ Question: "Tell me every AI system you have. Who approved it? Is the approval independent?"
├─ Supervisor wants: List of use-cases, owners, tier, approval sign-off from separate functions
├─ EVOLVE coverage: Use-case register with named owner + periodic review
├─ Kognita coverage: 0.4 use-case register with independent validation enforcement (separation of duties)
├─ Evidence: `kognita usecase list --show-approval-chain`
└─ Delivered by: 0.4 (Use-case register)

CHARGE 2: Agent Identity & Authentication
├─ Question: "How do you know the agent is who it claims to be? Is identity self-asserted or verified?"
├─ Supervisor wants: Credential-based identity, workload tokens, no shared accounts, authentication audit
├─ EVOLVE coverage: Agent registry with version, owner, risk class
├─ Kognita coverage: 0.3 Tier 0 fix (bind names to authenticated clients) + 0.5 (agents get workload credentials)
├─ Evidence: `kognita agent list --show-credentials | kognita evidence authority <action_id>`
└─ Delivered by: 0.3 + 0.5 (Agent identity)

CHARGE 3: Approval Authority & Human Review
├─ Question: "Who approves high-risk decisions? Is approval genuine or rubber-stamp?"
├─ Supervisor wants: Named approvers, approval packets, automation-bias detection, time-to-decision analysis
├─ EVOLVE coverage: RM review, KnowledgeLead, approval gates per use-case
├─ Kognita coverage: 0.4 RM review capture (structured diff, approval quality, automation-bias detection)
├─ Evidence: `kognita evidence affected --interaction <id> --show-approvals`
└─ Delivered by: 0.4 (RM review capture + approval quality)

CHARGE 4: Evidence & Auditability
├─ Question: "Show me one interaction. Prove it can't have been altered. Reproduce it."
├─ Supervisor wants: Tamper-evident chain, complete audit trail, reconstruction report, external verification
├─ EVOLVE coverage: Outcome metrics, learning loop produces documented changes
├─ Kognita coverage: 0.3 hash chain + pinned evidence + 0.4 interaction record + reconstruction report
├─ Evidence: `kognita evidence verify --db store.db | kognita evidence reconstruct <interaction_id>`
└─ Delivered by: 0.3 + 0.4 (Evidence chain + reconstruction)

CHARGE 5: Third-Party & Model Risk
├─ Question: "What happens if your model provider goes down? If a data source changes? If a tool is compromised?"
├─ Supervisor wants: Provider register, dependency map, incident response plan, verified fallbacks
├─ EVOLVE coverage: Business continuity (RM work continues without AI)
├─ Kognita coverage: 0.7 provider register, dependency map, concentration risk, silent model-change detection
├─ Evidence: `kognita dependencies list | kognita risks concentration`
└─ Delivered by: 0.7 (Resilience + provider register)
```

**Evidence Artifacts (What Supervisor Can Verify):**
```
Use-case inventory (interactive):
├─ List all AI use-cases with owner, tier, approval, review date
├─ Filter by risk tier, owner, status (active/retired)
├─ Drill into each: policies, evaluation results, incident history

Agent audit trail:
├─ Each agent's identity, version, credential issue/expiry dates
├─ Who issued the credential, when, and whether it's active
├─ Deployment version hash (proves prompt, tools, permissions are versioned)

Interaction reconstruction:
├─ Pick any client-affecting decision
├─ Show: Why this client → Why this product → What data → Which model → RM authority → Controls → Approval → Change → Final decision
├─ Verify each input: policy hash, data hash, model version, approval signature
├─ Reproduce: Run through same policies with pinned inputs, compare outcome

Evidence chain:
├─ Verification: `kognita evidence verify` shows no tampering
├─ Affected-client query: Who saw bad output? (if model change / policy breach / data quality drop detected)
└─ Export: Portable JSON audit (supervisor can verify hashes externally)
```

**Visual Notes:**
- Five charge boxes arranged vertically
- Each box: Charge (question) → EVOLVE answer → Kognita answer → Evidence artifact
- Color-coded by domain: orange (governance), blue (identity), purple (control), green (evidence), red (risk)
- At bottom: "Supervisor drill-down" showing interactive evidence artifacts

**Delivered By:** Governance components across all releases (0.3-0.7)

**Why This Matters for EVOLVE:**
EVOLVE shows how to run AI-driven RM processes. This slide shows that running them well means being able to answer five hard regulatory questions on demand. Together, EVOLVE + Kognita provide the answers (and the evidence).

---

## Part 9: Implementation Architecture & Release Mapping

### Slide 20: "The Architecture: Core, Packs, Gateways, Adapters"

**Stage:** Evolved (insert after Slide 31)

**Hero Text:**
> Governance is architecture, not a feature. Build it in from day one.

**Four Layers:**

```
LAYER 1: CORE (Four dependencies, no network)
├─ What it is: Decision engine
├─ Decides: DENY / ESCALATE / HUMAN_APPROVAL / ALLOW
├─ Does not: Own client data, run models, call tools (only decides if they may)
├─ Dependencies: pydantic, sqlmodel, numpy, python-dotenv
├─ Test: `import kognita` must never import an LLM, graph DB, or provider SDK
│         (test_packaging.py enforces this)
└─ Why: Governance should not require the machinery that answers it

LAYER 2: DOMAIN PACKS (Pluggable policy evaluators)
├─ What they are: Rules + attribute resolution for a domain
├─ Implemented as: Python class with methods:
│  ├─ load_subjects(envelope, session) → resolve who/what the request is about
│  ├─ resolve_attributes(envelope, subjects) → derive policy-relevant attributes
│  ├─ rules() → custom rule types beyond the built-in six
│  └─ engages(policy, context) → does this regime apply to this request?
├─ Example packs: private-banking (cross-border rules, booking-centre), healthcare (PHI controls)
├─ Test: Conformance kit enforces all packs fail-closed, cite rules, emit evidence
└─ Ownership: Domain experts write packs; Kognita core is domain-blind

LAYER 3: GATEWAYS (Proxies for external systems)
├─ AI Gateway: Explicit proxy in front of model providers
│  ├─ Wire format: OpenAI-compatible (covers OpenAI, Groq, Ollama, vLLM)
│  └─ Flow: Authorize → Redact → Forward → Classify response
├─ MCP Gateway: Proxy in front of MCP tools and systems
│  ├─ Wire format: MCP protocol (stdio, SSE)
│  └─ Flow: Authorize → Forward → Hash response → Record evidence
└─ Both: Every external call is authorized, evidenced, reversible

LAYER 4: ADAPTERS (Framework integration)
├─ Hermes Agent: MIT licensed, approval hook already exists
├─ Claude Agent SDK: Official Anthropic SDK
├─ LangGraph: Multi-agent orchestration
├─ Pydantic AI: Type-safe agent building
└─ Others: Can add pluggable adapters for other frameworks
```

**Flight Recorder (Observability):**
```
Dashboard:
├─ Recent runs filtered by actor, tool, outcome
├─ Drill-down into each decision: policy citations, evidence, approval
├─ Alerts: budget exceeded, repeated denials, chain break, classifier drift

Export:
├─ Run as self-verifying JSON (agent can share with auditor)
├─ OpenTelemetry export (decisions, denials, latency, token use, failures flow into bank's existing monitoring)

Effectiveness metrics:
├─ RM acceptance, edit, rejection and wrong-client rates per use case
├─ Business KPIs (conversations per RM, research-to-action conversion) are the bank's analytics
├─ Kognita supplies underlying events
```

**Deploy Path:**
```
Step 1: Install core + domain pack
  $ pip install kognita
  $ pip install kognita-private-banking-pack

Step 2: Configure environment
  Install script: pip install -e .
  Start script: pytest (validate)
  Network access: (inherit or restrict)

Step 3: Deploy gateways
  AI Gateway: kognita serve --provider openai-compatible --upstream https://api.openai.com
  MCP Gateway: kognita serve --mcp --root-config config.json

Step 4: Integrate adapter
  Add: from kognita.adapters.hermes import HermesAdapter
  Agent harness points to gateway (not provider directly)

Step 5: Run conformance
  pytest --pyargs kognita.testing.conformance --with-pack=private-banking-pack
```

**Visual Notes:**
- Four-layer stack vertically arranged
- Core at bottom (red), solid foundation
- Packs on top (orange), pluggable
- Gateways to the right (blue), explicit proxies
- Adapters above (green), framework integration
- Flight recorder hovering (gold), observability
- Arrows showing data flow: agent → adapter → gateway → policy (core) → external

**Delivered By:** Core (v0.2) + 0.3 Gateways + 0.4 Flight recorder + all adapters

**Why This Matters for EVOLVE:**
EVOLVE's maturity stages require technology that scales from one team to an enterprise. This slide shows the architecture that lets you plug in domain-specific rules (packs), scale governance to external systems (gateways), and integrate into any agent framework (adapters). It's the technical foundation for governance at scale.

---

### Slide 21: "From EVOLVE Stages to Kognita Releases"

**Stage:** Evolved (final slide, insert after Slide 32)

**Hero Text:**
> You climb the maturity ladder. Kognita builds the rungs as you climb.

**The Mapping (EVOLVE Stage ↔ Kognita Release):**

```
STAGE 1: EXPERIMENTAL
├─ EVOLVE: AI as isolated tools and pilots
├─ Kognita: v0.2 Core ships
│  ├─ Fail-closed decisions (DENY > ESCALATE > APPROVAL > ALLOW)
│  ├─ Agent registry + kill switch
│  ├─ Hash-chained evidence (tamper detection)
│  └─ Egress redaction (data leaving system is governed)
├─ Question answered: "Is this decision allowed?"
└─ Time to value: RMs can use AI safely (yes/no gates in place)

STAGE 2: ENABLED
├─ EVOLVE: AI as employee copilot, productivity multiplier
├─ Kognita releases: 0.3 + 0.4
│  └─ 0.3 Gateways:
│     ├─ Explicit proxy (AI gateway, MCP gateway)
│     ├─ Run budgets (cost control, loop prevention)
│     ├─ Suspend/resume (HUMAN_APPROVAL workflow)
│     ├─ Classifier-derived envelopes (free-text governance)
│     └─ Pinned evidence (policy + data + model hashes)
│  └─ 0.4 Ingestion & Lifecycle:
│     ├─ Use-case register (accountable owner, risk tier)
│     ├─ Interaction record (one ID for whole client journey)
│     ├─ Origination evidence (why this client)
│     ├─ RM review capture (what they saw, what they changed, approval quality)
│     ├─ Outcome metrics (cycle time, quality, reliability)
│     └─ Knowledge Lead role (owns the quality standard)
├─ Question answered: "Who approved this? Are we measuring what matters?"
└─ Time to value: RMs use shared standard + documented approval gates

STAGE 3: EMBEDDED
├─ EVOLVE: AI in workflows and decision support
├─ Kognita releases: 0.4 + 0.5 early
│  └─ 0.4 continued:
│     ├─ Risk-based review (queue high-risk interactions)
│     ├─ Circuit breaker (auto-deny if flags cross threshold)
│     └─ Outcomes & near-misses (monitor for bad guidance early)
│  └─ 0.5 early items:
│     ├─ Agent identity (not self-asserted)
│     ├─ Delegated authority (dual check: human + agent)
│     └─ Behavioral anomaly detection
├─ Question answered: "Is this agent misbehaving? Can we stop it fast?"
└─ Time to value: Closed-loop learning: RM edits feed back to agent standard

STAGE 4: EMPOWERED
├─ EVOLVE: AI agents make decisions (reversible actions)
├─ Kognita releases: 0.5
│  ├─ Autonomy levels L0-L6 (read → explain → recommend → draft → act → bounded → broad)
│  ├─ Evidence gates for promotion (quality ≥ 95%, exceptions < 5%, traceability 100%)
│  ├─ Automatic step-down on drift (metrics fall → autonomy drops one level)
│  ├─ Tool allow-lists per agent (read, analyse, recommend, draft, queue, send, execute are separate)
│  ├─ Blast-radius limits (max clients, max value, max daily actions)
│  ├─ External content isolation (tainted runs cannot call write tools)
│  ├─ Instruction authenticity (channel message alone is not authorization)
│  └─ Runtime intervention (observe, pause, override, restrict, recover)
├─ Question answered: "What level of autonomy has this agent earned? Can we ratchet it down if metrics degrade?"
└─ Time to value: Agents handle routine decisions; RMs focus on exceptions

STAGE 5: EVOLVED
├─ EVOLVE: AI becomes the operating system, continuous learning loop
├─ Kognita releases: 0.6 + 0.7
│  └─ 0.6 Claims:
│     ├─ Claim provenance envelope (every material claim carries evidence)
│     ├─ Fact contract (data freshness, source authority)
│     ├─ Claim types (fact / inference / recommendation, each typed)
│     ├─ Approved-source grounding (no model knowledge as source)
│     ├─ Verification gate (challenger agent checks claims before RM sees them)
│     ├─ Institutional memory (proposed changes to standard, Knowledge Lead approves)
│     └─ Feedback-loop controls (learning is governed, exclusions enforced)
│  └─ 0.7 Trust & Resilience:
│     ├─ Provider register (track all dependencies, versions, incidents)
│     ├─ Dependency map (which use-case depends on which provider)
│     ├─ Concentration risk (share per provider)
│     ├─ Silent model-change detection (provider reports new version → revalidation trigger)
│     └─ Signed evidence + external verification (audit-ready)
├─ Question answered: "Every correction becomes institutional learning. Can we trust our evidence under external audit?"
└─ Time to value: AI is now infrastructure; outcomes drive governance, not guesswork

GATES BETWEEN STAGES (Evidence-Driven):
├─ Stage 1 → 2: Outcome metrics established, baseline taken
├─ Stage 2 → 3: One use-case at tier 2 running with interaction record + RM review capture
├─ Stage 3 → 4: Agent behavior anomalies detected and contained in test; no field incidents in 30 days
├─ Stage 4 → 5: Quality ≥ 95% for 90 days; autonomy steps down if it falls; institutional memory in place
```

**Timeline Visual:**
```
Timeline (24 months):

Month 0:
├─ Stage 1 (Experimental)
└─ Kognita v0.2 ships (core)

Month 3:
├─ Stage 2 (Enabled)
└─ 0.3 gateways + 0.4 ingestion (gates, interaction record, use-case register, Knowledge Lead role)

Month 6:
├─ Stage 3 (Embedded)
└─ Circuit breaker + closed-loop learning (risk-based review, outcomes)

Month 12:
├─ Stage 4 (Empowered)
└─ 0.5 agents (autonomy, evidence gates, step-down, intervention, anomaly detection)

Month 18:
├─ Stage 5 (Evolved)
└─ 0.6 claims (verification gate, institutional memory) + 0.7 resilience (provider register)
```

**Visual Notes:**
- Horizontal timeline showing five stages left to right
- Vertical stacks below each stage showing which Kognita release delivers it
- Gate icons between stages showing evidence requirements
- Color progression: red (core) → orange (gates) → yellow (learning) → blue (autonomy) → green (resilience)
- Title: "Each stage earns the right to the next by proving it through evidence"

**Delivered By:** Full roadmap (0.2-0.7) + EVOLVE integration

**Why This Matters for EVOLVE:**
This is the most important slide. It shows that EVOLVE and Kognita are not separate initiatives — they are the same journey. EVOLVE defines the business outcomes; Kognita delivers the governance infrastructure to achieve each stage safely. By Stage 5 (Evolved), you have both: AI as your operating system + complete auditability under external examination.

---

## Summary: How to Use These Slides

### In Your Deck Structure

**Before "Stages"** (new Part 1):
- Existing EVOLVE slides 1-10 (Why it matters, leadership stakes, destination, maturity map, FEAT, human constants)

**Interleaved with Stages 1-5** (new Part 2):
- Insert enforcement/decision slides into each stage
- Insert evidence/reconstruction slides
- Insert agent identity slides
- Insert intervention/containment slides

**After Stages** (new Parts 3-5):
- Enforcement & risk (anomaly, circuit breaker, gateway)
- Data, claims & egress (fact contract, claim types, verification, egress)
- Regulation & implementation (control frameworks, five charges, architecture, release mapping)

### Visual Consistency

Each new slide should follow EVOLVE's design language:
- **Hero text:** Large, bold principle at the top
- **Diagram:** Center showing the mechanism (flow, stack, hierarchy)
- **Bullets:** 2-4 key outcomes or rules
- **Color:** Consistent palette (blue for governance, orange for identity, green for evidence, red for risk)
- **Footer:** Which Kognita release delivers it

### Delivery Flow

The 21 slides transform EVOLVE from a maturity model into a **complete governance stack**:

1. **EVOLVE (existing):** Shows the business journey and human governance
2. **Enforcement layer (slides 1-3):** Shows how decisions are made before data moves
3. **Evidence layer (slides 4-6):** Shows how decisions are recorded and proven
4. **Agent layer (slides 7-8):** Shows how agents are authenticated and authorized
5. **Intervention layer (slides 9-11):** Shows how bad behavior is caught and stopped
6. **Data & claims layer (slides 14-17):** Shows how facts are verified and claims are typed
7. **Infrastructure layer (slides 12-13, 20):** Shows gateways, providers, architecture
8. **Regulatory mapping (slides 18-19, 21):** Shows how this answers supervisor questions and maps to releases

### Messaging

**"EVOLVE + Kognita = Complete AI Governance"**

- **EVOLVE** answers: How do we make better decisions? How do we learn from outcomes?
- **Kognita** answers: How do we enforce permission? How do we prove nothing was changed? How do we stop misbehavior?
- **Together** they answer: Can the bank explain, control, stop, and reconstruct every material AI action that affects a client?

---

## Slide Word Counts & Timing

For a 60-minute deck:

- Existing EVOLVE slides (1-10): 30 minutes
- New enforcement + evidence (1-6): 12 minutes
- New agent + intervention (7-11): 10 minutes
- New data + infrastructure (12-17, 20): 8 minutes
- New regulatory (18-19, 21): 5 minutes

**Total: 65 slides, ~60 minutes**

(If shorter: drop slides 11, 13, 17; condense to 50 slides, 45 minutes)

---

## Notes for Ahmed

These slides are written to be adapted and reworded in your voice. They're detailed briefs, not scripts. Feel free to:
- Simplify technical terms for non-technical audiences
- Add case studies or examples from your own deployments
- Reorder for your presentation flow
- Combine slides if needed (e.g., 9+10 together if short on time)
- Adjust colors and visual style to match EVOLVE branding

The key is: EVOLVE is the "why and what"; Kognita is the "how and proof." Together they form the governance stack.

---

**End of Slide Brief**

Created: 2026-10-05
Attribution: Kognita roadmap + EVOLVE framework (Ahmed Muzammil, CC BY SA 4.0)
Total new slides: 21 (distributed across existing EVOLVE structure)
Purpose: Complete the AI governance stack for wealth management and private banking
