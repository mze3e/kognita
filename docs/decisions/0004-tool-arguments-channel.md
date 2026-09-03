# ADR 0004: Tool Arguments Channel

**Status:** Accepted (0.2.0)  
**Depends on:** Tier 0 fixes (`recorded_at` hash, HUMAN_APPROVAL data hold)  
**Unblocks:** ADR 0006, ADR 0007

## Context

Tool functions have no way to receive arguments. The signature is:

```python
ToolFn = Callable[[Envelope, Evaluation, Session], Any]
```

The only place tool arguments travel today is `Envelope.context`, documented as "Never trusted for authorisation decisions" and excluded from `to_dict()`. That means:

- Arguments are not hashed
- Arguments are not evidenced
- A request could be approved with one argument set and executed with another
- `envelope_hash` does not change, defeating the binding that makes approvals meaningful

Measured against BMOS's 15 governed tools:
- `get_org_state()` needs no arguments (1 tool)
- `get_document(doc_id)` needs a document id
- Every `list_*` needs a pagination cursor
- All five `propose_*` carry the proposed change itself (14 tools)

This is a mandatory gap. Kognita cannot express the "propose, then humans confirm, then apply" pattern without tool arguments being evidenced.

## Decision

Add `arguments: dict[str, Any]` as a first-class field on `Envelope`:

1. **Included in hashing.** `Envelope.to_dict()` includes `arguments` so it's part of `envelope_hash`. An approval granted for one argument set binds to that set and cannot be replayed with different arguments.

2. **Passed to tools.** `ToolFn` receives arguments via `envelope.arguments`. Tools that need no arguments receive an empty dict; tools that need them read from there.

3. **Evidenced.** Arguments are part of the hashed envelope, so `DECISION`, `TOOL_CALL`, and `EGRESS` events all carry them (as part of envelope_hash in the payload).

4. **Validated by packs.** A pack's `resolve_attributes` receives the envelope with arguments already present, so it can validate them against policy.

## Schema

```python
@dataclass(frozen=True)
class Envelope:
    principal: str
    purpose: str
    tool: str
    actor_location: str = ""
    agent_name: str | None = None
    subject_type: str | None = None
    subject_id: str | None = None
    subjects: dict[str, str] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)  # NEW
    is_admin: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal": self.principal,
            "purpose": self.purpose,
            "tool": self.tool,
            "actor_location": self.actor_location,
            "agent_name": self.agent_name,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "subjects": dict(self.subjects),
            "arguments": dict(self.arguments),  # NEW
            "is_admin": self.is_admin,
        }
```

## Consequences

**What this buys:**

- Arguments are tamper-evident; approvals bind to specific arguments.
- Propose-then-apply becomes possible (ADR 0007 depends on this).
- Tools that need parameters can express them clearly.
- Policy can validate arguments before tool execution.

**What it costs:**

- Breaking change for existing `Envelope` consumers. Any code constructing envelopes by keyword must add `arguments={}` or accept the default.
- Tools with `ToolFn` signature unchanged (they just read `envelope.arguments`), so this is not a tool-signature change.

**Migration:**

For existing code that does not pass arguments:

```python
# Before (still works)
envelope = Envelope(principal="user@example.com", tool="get_org_state", ...)

# After (with new field)
envelope = Envelope(
    principal="user@example.com",
    tool="get_org_state",
    arguments={},  # NEW, but defaults to empty
    ...
)
```

For code that needs arguments:

```python
envelope = Envelope(
    principal="user@example.com",
    tool="get_document",
    arguments={"document_id": "123"},  # NEW
    ...
)
```

## Verification

- 60 unit tests pass (no test changes needed; default arguments={} maintains compatibility).
- 10 conformance cases pass.
- BMOS-shaped fixture pack: new xfail `test_mark_criterion_arguments_hashed` flips to passing once this lands.
- No tool-signature changes required; tools read from envelope rather than function parameters.

## Related

- Tier 1.4 "Propose-then-apply" depends on this for argument binding.
- Tier 1.3 "Two-signature approvals" uses hashed arguments to ensure approvals bind correctly.
