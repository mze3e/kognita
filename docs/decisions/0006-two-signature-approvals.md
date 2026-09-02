# ADR 0006: Two-Signature / Separation-of-Duties Approvals

**Status:** Accepted (0.2.0)  
**Depends on:** ADR 0004 (tool arguments), ADR 0005 (identity/roles)  
**Unblocks:** ADR 0007 (propose-then-apply), BMOS-shaped fixture pack

## Context

BMOS's central doctrine: *"One person marks a criterion met; only an owner confirms; nothing self-passes."*

Kognita cannot express this. The `Approval` model has four compounding problems:

1. One identity column, `approver_name`, defaults to `"Unassigned (duty desk)"` — it doubles as both assignee and decider.
2. The requester's identity is never copied onto the approval row, so no comparison is possible.
3. `approver_name` is an unverified string with no shared namespace with `principal`.
4. `_transition` raises if status is not `PENDING`, so a *second* signature on the same row is structurally impossible.

The fix requires separating these roles into distinct columns and allowing state transitions that track both signatures.

## Decision

Extend the `Approval` model to record two distinct parties: the requester and the approver. Track the approval state through two independent signatures.

### 1. Schema changes

```python
class Approval(SQLModel, table=True):
    __tablename__ = "approvals"

    id: int | None = Field(default=None, primary_key=True)
    decision_id: int = Field(index=True, foreign_key="governance_decisions.id")
    envelope_hash: str = Field(index=True)
    
    # NEW: Separation of duties
    requester_id: str = ""  # principal from the original request
    approver_id: str | None = None  # who will confirm (owner, not the requester)
    
    # Track two-signature state
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING, index=True)
    # NEW: second-signature state for two-signature gates
    confirmation_status: ApprovalStatus | None = None
    
    scope: str = ""  # unchanged
    reason: str | None = None  # unchanged
    
    # Timestamps
    created_at: datetime = Field(...)
    approved_at: datetime | None = None  # When approver granted
    confirmed_at: datetime | None = None  # When second signature granted (two-signature only)
    expires_at: datetime = Field(...)
    decided_at: datetime | None = None  # Final decision time
```

### 2. Approval workflow — single vs. two-signature

**Single-signature (today's behavior, unchanged):**

```
Decision outcome: HUMAN_APPROVAL
  → open_approval(requester_id="alice", approver_id="bob")
  → status=PENDING
Alice's request is held; Bob approves
  → grant(approver_id="bob")
  → status=APPROVED, approved_at=now
Tool executes, data released
```

**Two-signature (new, required for gated workflows):**

```
Decision outcome: HUMAN_APPROVAL
  → open_approval(requester_id="marker_alice", approver_id="owner_bob", confirmation_required=True)
  → status=PENDING, confirmation_status=None
Alice marks criterion; request awaits Bob's mark
  → _intermediate_approve(approver_id="bob", mark="meets_threshold")
  → status=MARKED, confirmation_status=None  # First signature done
Data is still held; Bob now confirms
  → _intermediate_approve(approver_id="owner_bob", action="confirm")
  → status=APPROVED, confirmation_status=APPROVED
Tool executes atomically
```

### 3. State machine — two-path approval

Existing code (single signature):

```
open() → PENDING → grant() → APPROVED → execute
            ↓
         reject() → REJECTED
            ↓
        (expired) → EXPIRED
```

New code (two-signature):

```
open(confirmation_required=True)
        → PENDING
         /      \
    mark()      reject()
    /             \
 MARKED        REJECTED
   |
confirm()
   |
APPROVED → execute
```

### 4. Policy language — two-signature gate

A policy can require two signatures:

```python
Policy(
    regime="CRITERION_GATE",
    rule_type="TWO_SIGNATURE_APPROVAL",  # NEW rule type
    applies_to="MARKER_ONLY",
    rule={
        "first_actor_role": "MARKER",
        "second_actor_role": "OWNER",
        "description": "Marking requires marker to mark, owner to confirm.",
    },
    citation="BMOS Governance Model: Two-signature criterion",
)
```

### 5. Conformance invariant

A two-signature approval must:

1. Have both `requester_id` and `approver_id` recorded (not same person).
2. Not self-approve: `requester_id != approver_id`.
3. Require explicit confirmation by the second party before the tool executes.
4. Be evidenced at both signature points.

## Consequences

**What this buys:**

- Separation of duties: requester and approver are distinct, verifiable entities.
- Two-signature gating: policies can require explicit confirmation from a designated party.
- Audit trail: `approved_at` and `confirmed_at` distinguish first and second signatures.
- BMOS doctrine: "one person marks, only an owner confirms, nothing self-passes" is now expressible.

**What it costs:**

- Breaking change: `approver_name: str` is replaced with `approver_id: str | None` and `requester_id: str`.
- New `confirmation_status` column for two-signature state tracking.
- New rule type `TWO_SIGNATURE_APPROVAL` (alongside existing `REQUIRES_HUMAN_APPROVAL`).
- Packs must distinguish between single and two-signature gates in policy.

**Migration:**

Existing single-signature approvals remain compatible: a request approved by one party still releases data. Two-signature gates require explicit policy declaration.

## Verification

- 60 unit tests pass; existing single-signature tests remain compatible.
- New test `test_marking_requires_two_signatures` passes (was xfail in BMOS pack).
- New test `test_approval_carries_proposal_payload` passes (two-signature tracking).
- 10 conformance cases pass.
- BMOS-shaped fixture pack:
  - Two-signature tests flip from xfail to passing.
  - Data is held (returns None) until second signature.
  - Both signatures are evidenced.

## Related

- ADR 0007 (propose-then-apply) builds on this by adding `proposal` payload to approvals.
- Tier 2 item 8 (gated workflow) is expressed via two-signature policies.
