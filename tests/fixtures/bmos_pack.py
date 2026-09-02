"""A BMOS-shaped fixture pack for testing governance doctrine.

BMOS (Business Maximizer® OS) doctrine is governance-shaped: two-signature gates,
"agents propose, humans sign", a human-in-the-loop proposal queue, exportable
artefacts, and falsifiable guarantees. This pack models the core: one person marks
a criterion met; only an owner confirms; nothing self-passes.

The gap analysis (`docs/gap-analysis-bmos.md`) identifies what Kognita needs to
express this doctrine. This pack's assertions — marked `xfail(strict=True)` —
fail until each gap is closed. The pack is deliberately minimal: two roles, one
criterion type, one gatable tool, and a proposing agent.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlmodel import Session

from kognita.envelope import Envelope, RuleContext
from kognita.models import Policy, utcnow
from kognita.rules import build_registry
from kognita.vocabulary import Classification

# ── Domain vocabulary ────────────────────────────────────────────────────────

PURPOSES = (
    "MARK_CRITERION",      # Agent marks a criterion met
    "CONFIRM_CRITERION",   # Owner confirms the mark
    "PROPOSE_CHANGE",      # Agent proposes a change
    "APPLY_PROPOSAL",      # Owner applies the proposal
)

TOOLS = (
    "mark_criterion",      # Agent marks; produces no side-effect until confirmed
    "confirm_criterion",   # Owner confirms; binds the mark
    "propose_document",    # Agent proposes a change to a document
    "apply_proposal",      # Owner applies; the actual change happens here
    "get_state",           # Read-only; always permitted
)

#: A workflow gate: owner must confirm before criteria bind.
GATES = ("CRITERION_GATE", "PROPOSAL_GATE")

#: BMOS roles: marker and owner.
ROLES = ("MARKER", "OWNER")


# ── Subjects ─────────────────────────────────────────────────────────────────

USERS: dict[str, dict[str, Any]] = {
    "marker_alice": {
        "id": "marker_alice",
        "name": "Alice (Marker)",
        "role": "MARKER",
    },
    "owner_bob": {
        "id": "owner_bob",
        "name": "Bob (Owner)",
        "role": "OWNER",
    },
}

CRITERIA: dict[str, dict[str, Any]] = {
    "crit_1": {
        "id": "crit_1",
        "description": "Criterion 1: Risk assessment complete",
        "gate": "CRITERION_GATE",
        "status": "PENDING",  # PENDING, MARKED, CONFIRMED
    },
}

DOCUMENTS: dict[str, dict[str, Any]] = {
    "doc_1": {
        "id": "doc_1",
        "title": "Policy Change Proposal",
        "current_version": 1,
        "status": "DRAFT",  # DRAFT, PROPOSED, ACCEPTED
    },
}


class BMOSPack:
    """A pack shaped like BMOS governance."""

    name = "bmos"

    def load_subjects(self, envelope: Envelope, session: Session | None = None) -> dict[str, Any]:
        """Resolve subjects by reference."""
        loaded: dict[str, Any] = {}
        for kind, ref in envelope.all_subjects().items():
            table = (
                USERS if kind == "user"
                else CRITERIA if kind == "criterion"
                else DOCUMENTS if kind == "document"
                else {}
            )
            row = table.get(str(ref))
            if row is None:
                raise LookupError(f"{kind} {ref!r} not found")
            loaded[kind] = row
        return loaded

    def resolve_attributes(
        self, envelope: Envelope, subjects: dict[str, Any]
    ) -> dict[str, Any]:
        """Derive policy-relevant attributes from envelope and subjects.

        Key attributes:
        - user_role: the role of the principal (MARKER or OWNER)
        - criterion_status: current status of the criterion
        - gate: which gate (CRITERION_GATE, PROPOSAL_GATE)
        - can_mark: is this user allowed to mark?
        - can_confirm: is this user allowed to confirm?
        """
        user = subjects.get("user", {})
        criterion = subjects.get("criterion", {})
        document = subjects.get("document", {})

        user_role = user.get("role")
        criterion_status = criterion.get("status")
        gate = criterion.get("gate") or document.get("gate")

        return {
            "user_role": user_role,
            "criterion_status": criterion_status,
            "gate": gate,
            "can_mark": user_role == "MARKER",
            "can_confirm": user_role == "OWNER",
            "tool": envelope.tool,
            "purpose": envelope.purpose,
        }

    def rules(self) -> dict[str, Any]:
        """No custom rules for BMOS pack; core primitives are enough."""
        return build_registry()

    def engages(self, policy: Policy, context: RuleContext) -> bool:
        """Whether a regime applies to this request.

        BMOS policies are role-gated: a MARKER_ONLY policy only engages for markers.
        """
        attrs = context.attributes
        subjects = context.subjects

        # Role-based engagement
        if policy.applies_to == "MARKER_ONLY":
            return attrs.get("user_role") == "MARKER"
        if policy.applies_to == "OWNER_ONLY":
            return attrs.get("user_role") == "OWNER"

        # Gate-based engagement (optional)
        if policy.regime and "CRITERION" in policy.regime:
            return attrs.get("gate") == "CRITERION_GATE"
        if policy.regime and "PROPOSAL" in policy.regime:
            return attrs.get("gate") == "PROPOSAL_GATE"

        return False


def seed_policies(session: Session, *, now: datetime | None = None) -> list[Policy]:
    """BMOS governance policies: role-gated, gate-enforced.

    1. Only MARKER role can mark criteria.
    2. Only OWNER role can confirm criteria.
    3. Once marked, a criterion must not auto-pass (someone must confirm).
    4. Once proposed, a change must not auto-apply (someone must apply).
    """
    at = now or utcnow()
    start = at - timedelta(days=365)

    rows = [
        # MARKER may mark, but only if the tool is mark_criterion
        Policy(
            regime="MARKER_GATE",
            rule_type="ATTRIBUTE_ALLOWLIST",
            applies_to="MARKER_ONLY",
            rule={
                "allow": {"tool": ["mark_criterion"]},
                "on_violation": "fail",
                "description": "Markers can only execute mark_criterion tool.",
            },
            citation="BMOS Governance Model s1: Role-based tool access",
            effective_from=start,
        ),
        # OWNER may confirm, but only if the tool is confirm_criterion
        Policy(
            regime="OWNER_GATE",
            rule_type="ATTRIBUTE_ALLOWLIST",
            applies_to="OWNER_ONLY",
            rule={
                "allow": {"tool": ["confirm_criterion", "apply_proposal", "get_state"]},
                "on_violation": "fail",
                "description": "Owners can execute owner tools.",
            },
            citation="BMOS Governance Model s2: Owner-role gate",
            effective_from=start,
        ),
        # Criterion confirmation gate: MARKER marks, OWNER confirms
        Policy(
            regime="CRITERION_GATE",
            rule_type="REQUIRES_HUMAN_APPROVAL",
            applies_to="MARKER_ONLY",
            rule={
                "tools": ["mark_criterion"],
                "description": "Marking a criterion requires owner confirmation.",
            },
            citation="BMOS Governance Model s3: Two-signature criterion",
            effective_from=start,
        ),
        # Proposal application gate: AGENT proposes, OWNER applies
        Policy(
            regime="PROPOSAL_GATE",
            rule_type="REQUIRES_HUMAN_APPROVAL",
            applies_to="MARKER_ONLY",
            rule={
                "tools": ["propose_document"],
                "description": "Proposing a change requires owner application.",
            },
            citation="BMOS Governance Model s4: Two-signature proposal",
            effective_from=start,
        ),
    ]
    for row in rows:
        session.add(row)
    session.flush()
    return rows


def envelope(
    tool: str,
    *,
    purpose: str = "MARK_CRITERION",
    user: str | None = None,
    criterion: str | None = None,
    document: str | None = None,
    principal: str = "bmos-user",
    agent: str | None = None,
) -> Envelope:
    """Build a BMOS envelope."""
    subjects: dict[str, str] = {}
    if criterion is not None:
        subjects["criterion"] = criterion
    if document is not None:
        subjects["document"] = document

    arguments: dict[str, Any] = {}
    if criterion is not None:
        arguments["criterion_id"] = criterion
    if document is not None:
        arguments["document_id"] = document

    return Envelope(
        principal=principal,
        purpose=purpose,
        tool=tool,
        actor_location="",  # Not used in BMOS pack
        agent_name=agent,
        subject_type="user" if user is not None else None,
        subject_id=user,
        subjects=subjects,
        arguments=arguments,
        is_admin=False,
    )
