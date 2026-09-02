# ADR 0005: Identity and Authority Model

**Status:** Accepted (0.2.0)  
**Depends on:** ADR 0004 (tool arguments)  
**Unblocks:** ADR 0006 (two-signature approvals)

## Context

Kognita today has `principal: str` and `is_admin: bool`. There is no way to express:
- "only an owner may confirm" (BMOS central doctrine)
- "this user has role X in scope Y, expiring on date Z"
- "an agent never has more permission than the person using it"

BMOS has five roles, per-member scope, and time-expiring access grants. The invariant is: *"an agent never has more permission than the person using it."*

Kognita's current model:
- `is_admin` has one effect: `ceiling_for()` returns C3 instead of C2 (retrieval.py:82-88).
- `is_admin` is inside the envelope hash, so it cannot be swapped after approval — that part is sound.
- **But the library never verifies the claim against anything.** Any caller can assert `is_admin=True`.
- The conformance kit does not check this, so a pack can silently do the wrong thing.

Two distinct problems:

1. **Expressiveness:** Policies cannot reference roles, scopes, or expiring grants.
2. **Trust:** Role claims are not verified. `is_admin` is advisory; the core does not enforce it.

## Decision

Expand the identity plane from boolean `is_admin` to a structured role model, and require packs to verify claims:

### 1. Envelope schema — claim-based

Add role information to the envelope without making Kognita role-aware:

```python
@dataclass(frozen=True)
class Envelope:
    principal: str  # Unchanged
    purpose: str
    tool: str
    # ... other fields ...
    is_admin: bool = False  # DEPRECATED; use roles instead
    roles: list[str] = field(default_factory=list)  # NEW
    scopes: dict[str, str] = field(default_factory=dict)  # NEW: {"org": "acme", "dept": "finance"}
    
    def to_dict(self) -> dict[str, Any]:
        return {
            # ...
            "roles": list(self.roles),
            "scopes": dict(self.scopes),
            # is_admin: still included for backward compatibility
        }
```

### 2. Conformance invariant — verification required

Each domain pack must verify role claims in `resolve_attributes`:

```python
class MyPack:
    def resolve_attributes(self, envelope: Envelope, subjects: dict[str, Any]) -> dict[str, Any]:
        # Verify the claimed roles
        actual_roles = load_from_ldap_or_database(envelope.principal)
        if not all(role in actual_roles for role in envelope.roles):
            raise ValueError(f"Principal {envelope.principal} does not hold claimed roles")
        
        # Return attributes policy can check
        return {
            "principal": envelope.principal,
            "user_role": actual_roles.get("primary_role"),  # What we verified
            # ...
        }
```

The conformance kit will enforce: **a pack's attributes must include role information resolved from a source of truth, not asserted from the envelope.**

### 3. Policy language — role predicates

Policies can now express role-based gates:

```python
Policy(
    regime="OWNER_GATE",
    rule_type="ATTRIBUTE_ALLOWLIST",
    applies_to="OWNER_ONLY",
    rule={
        "allow": {"user_role": ["owner", "admin"]},  # Checked against resolved attributes
        "on_violation": "fail",
    },
    citation="Governance Model s2: Owners can confirm",
)
```

### 4. Ceiling enforcement — no agent exceeds its human

The invariant *"an agent never has more permission than the person using it"* is expressed in policy:

```python
Policy(
    regime="AGENT_BINDING",
    rule_type="REQUIRES_HUMAN_APPROVAL",
    applies_to="AGENT_ONLY",
    rule={
        "tools": ["propose_*"],
        "description": "An agent's proposal requires human confirmation before applying.",
    },
)
```

If `envelope.agent_name` is set, the human's role must meet the agent's ceiling, enforced by policy rules.

## Schema

No change to the core enforcement — roles are data, verified by packs.

- `Envelope.roles`: list of claimed roles (what the caller says they have)
- `Envelope.scopes`: context dictionary (org, department, tenant, etc.)
- Pack's `resolve_attributes`: the place where claims are verified
- Policy rules: reference verified attributes, not claimed roles

## Consequences

**What this buys:**

- Packs can express role-based policy ("only owners confirm").
- A pack *must* verify role claims; silent acceptance is impossible (conformance kit enforces it).
- Ceiling logic is expressed in policy, not hardcoded in the core.
- Support for scope-aware policy (per-org, per-department gates).
- Time-expiring grants are representable (pack verifies expiry in `resolve_attributes`).

**What it costs:**

- Packs must implement role verification; they cannot blindly trust the envelope.
- `is_admin` boolean is deprecated (still works for backward compatibility, but discouraged).
- Conformance kit has new invariant: packs must verify role claims.

**Breaking changes:** None for code that only uses `is_admin=True/False`. Additive for code that needs roles.

## Verification

- 60 unit tests pass (no change; `is_admin` still works).
- 10 conformance cases pass.
- New conformance invariant: `test_role_verified_against_pack` enforces that attributes include resolved (not asserted) role information.
- BMOS-shaped fixture pack:
  - `test_marker_can_mark_criterion` passes (MARKER role verified, allowed to mark).
  - `test_marker_cannot_confirm` passes (MARKER role verified, denied from confirm tool).

## Related

- ADR 0006 (two-signature approvals) uses verified roles to separate requester from approver.
- Tier 2 item 6 (retrieval entitlement) can now express "role × scope × gate" access.
