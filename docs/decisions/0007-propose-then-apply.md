# ADR 0007: Propose-Then-Apply

**Status:** Accepted (0.2.0)  
**Depends on:** ADR 0004 (tool arguments), ADR 0006 (two-signature), Tier 0 (HUMAN_APPROVAL data hold)  
**Closes:** Tier 1.4 blocking gap

## Context

BMOS has a `proposals` table and `decide_proposal(id, accept, note)`: an agent proposes a change, humans review the diff, and only when approved does the change apply atomically.

Kognita's current model:
- `Approval` binds to an `envelope_hash` and **carries no payload** of the change.
- It records that a human approved *a request*, not *a diff*.
- Combined with Tier 0 issue #1 (HUMAN_APPROVAL executed anyway), Kognita's own doctrine "agents propose, humans sign" is convention, not enforcement.

This pattern is mandatory for:
- Document changes (title, status, version)
- Policy updates (proposer drafts, owner applies)
- Workflow state transitions (proposer marks met, owner confirms)

An approval that carries no payload cannot bind a specific change. Without it, the three-step pattern collapses:

1. Agent proposes change → Evaluation.HUMAN_APPROVAL, data held (Tier 0 fix)
2. Human reviews the proposal → **what did they see?**
3. Human approves → **applies what?**

Step 2–3 have no answer if the approval carries no payload.

## Decision

Extend `Approval` to carry a `proposal` payload describing what the human reviewed.

### 1. Schema changes

Add a proposal table and link approvals to it:

```python
class Proposal(SQLModel, table=True):
    """A proposed change awaiting approval.
    
    Captures the before-state, the kind of change, rationale and proposer.
    Approvals reference this so the human's signature binds to the exact change.
    """
    
    __tablename__ = "proposals"
    
    id: int | None = Field(default=None, primary_key=True)
    # Kind of change: "document_edit", "policy_update", "criterion_mark", etc.
    kind: str = Field(index=True)
    # Before-state snapshot (JSON)
    before_snapshot: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    # Change description (what the proposer is asking for)
    rationale: str = ""
    # Subject being changed
    subject_type: str | None = None
    subject_id: str | None = None
    # Proposer's identity
    proposer_id: str = ""
    # Proposed change data
    change: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    
    created_at: datetime = Field(default_factory=utcnow, sa_column=_utc_column())

# Link Approval to Proposal
class Approval(SQLModel, table=True):
    __tablename__ = "approvals"
    
    id: int | None = Field(default=None, primary_key=True)
    decision_id: int = Field(foreign_key="governance_decisions.id")
    proposal_id: int | None = Field(index=True, foreign_key="proposals.id")  # NEW
    # ... other fields ...
```

### 2. Workflow — propose, review, apply

**Step 1: Propose**

Agent runs `propose_document` tool:

```python
# Tool execution (held by HUMAN_APPROVAL, data=None returned)
result = run_governed(
    envelope,  # arguments={"title": "New Title", "content": "..."}
    registry=...,
    pack=...,
)
# outcome = HUMAN_APPROVAL, data = None (data held)
# Decision recorded in evidence

# Proposal created and linked
proposal = Proposal(
    kind="document_edit",
    before_snapshot={"title": old_title, "content": old_content},
    change={"title": new_title, "content": new_content},
    proposer_id="agent_proposer",
    rationale="Updated Q4 metrics",
)
approval = open_approval(
    ...,
    proposal_id=proposal.id,
    requester_id="agent_proposer",
    approver_id="owner_bob",
)
```

**Step 2: Human reviews**

Owner fetches the proposal:

```python
proposal = session.get(Proposal, proposal_id)
# Before: {title: "Q3 Metrics", ...}
# Change: {title: "Q4 Metrics", ...}
```

**Step 3: Apply**

Owner approves the proposal:

```python
approval = session.get(Approval, approval_id)
grant(
    approval,
    approver_id="owner_bob",
    evidence=...,
)
# approval.proposal_id is set → tool execution can proceed atomically
# Proposal change is applied (no diff possible; the human already saw it)
```

### 3. Tool execution — atomic apply

When an approval carries a proposal, tool execution is:

1. **Deterministic:** The tool reads `approval.proposal.change` and applies it, not the original request.
2. **Audited:** Both the proposal and the approval are in the evidence chain.
3. **Atomic:** Either the proposal applies in full or not at all.

```python
def apply_proposal_tool(envelope, evaluation, session):
    approval = find_granted(session, evaluation.envelope_hash)
    if not approval or not approval.proposal_id:
        raise RuntimeError("Approval must carry a proposal")
    
    proposal = session.get(Proposal, approval.proposal_id)
    # Apply the change
    subject = load_subject(proposal.subject_type, proposal.subject_id)
    subject.update(proposal.change)
    session.add(subject)
    return {"proposal_id": proposal.id, "result": "applied"}
```

### 4. Two-signature proposals

A proposal can require two signatures (ADR 0006):

```python
approval = open_approval(
    ...,
    proposal_id=proposal.id,
    confirmation_required=True,
    requester_id="marker_alice",
    approver_id="owner_bob",
)
# Alice proposes, Bob must confirm *and* approve
# Only then does the proposal apply
```

## Consequences

**What this buys:**

- Proposals are atomic: the change is either applied in full or not at all.
- Binding: the approval's signature is on the exact change the human reviewed.
- Auditability: before-state, change, and approver are all recorded.
- BMOS pattern: agent proposes, human reviews diff, owner applies.
- No diff substitution: once approved, the exact proposal applies (no side-channel proposal swaps).

**What it costs:**

- New `Proposal` table with `before_snapshot` and `change` JSON fields.
- `Approval.proposal_id` foreign key.
- New tool pattern: tools that apply proposals must read from `approval.proposal`, not the original envelope.
- Breaking change: propose-and-apply tools have a different signature than stateless tools.

**Storage:** Storing before-snapshots increases storage cost for mutable subjects (documents, policies). For immutable events (criterion marks, attestations) the cost is minimal. Large proposals can be compressed (ZSTD or similar) if needed.

## Verification

- 60 unit tests pass.
- New test `test_proposal_must_bind_to_diff` passes (was xfail in BMOS pack).
- New test `test_data_held_until_second_signature` passes (HUMAN_APPROVAL holds data).
- 10 conformance cases pass.
- BMOS-shaped fixture pack:
  - `test_marking_requires_two_signatures` passes (two signatures + proposal).
  - `test_data_held_until_second_signature` passes.
  - Proposal is applied atomically, not diff-substituted.

## Related

- ADR 0006 (two-signature) works seamlessly with proposals: first signature proposes, second applies.
- Tier 2 item 5 (versioned artefacts) is complementary: proposals track before-state; artefacts track versions post-apply.
- BMOS fixture pack: all xfails in two-signature and propose-then-apply tests flip to passing.

## Design notes

**Why store before-snapshots instead of diffs?**

A before-snapshot + change allows the human to see exactly what was proposed. A diff-only approach loses the context. With snapshot + change, the human sees both the old and the new state, and the approval binds to both.

**Why not versioning?**

Versioning (Tier 2 item 5) tracks the history of a thing *after* it's applied. Proposals track *before* a change is proposed. They're orthogonal: a proposal creates a snapshot, a version is a snapshot post-apply. Both matter, both belong in the schema.
