"""A neutral domain pack, standing in for the real one.

The wealth-management pack that motivated this library lives in a separate
repository, which would leave the core here unproven. This pack is deliberately
*not* financial — a research-collaboration domain — so that nothing
domain-specific creeps back into the core by accident, while exercising exactly
the same mechanics:

======================================  ====================================
Wealth pack (other repo)                Here
======================================  ====================================
Client domiciled in the EU              Subject in a restricted region
Booking centre / RM location            Home site / requester site
Accredited-investor flag                Cleared-collaborator flag
Structured note, complex product        Restricted dataset
Two regimes both failing one request    Two regimes both failing one request
GDPR Art 22 human review                Ethics-board human review
======================================  ====================================

Every conformance shape in ``tests/test_conformance.py`` is expressed against
this pack, so the invariants are checked without the domain being present.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlmodel import Session, select

from kognita.core.envelope import Envelope, RuleContext
from kognita.core.models import Policy, utcnow
from kognita.core.rules import build_registry
from kognita.core.vocabulary import Classification

# ── Domain vocabulary ────────────────────────────────────────────────────────

PURPOSES = (
    "COLLABORATION",
    "ELIGIBILITY_CHECK",
    "DOSSIER_PREP",
    "PUBLICATION",
    "ADMIN_REVIEW",
)

TOOLS = (
    "get_subject_profile",
    "get_dataset_snapshot",
    "check_dataset_eligibility",
    "get_house_guidance",
    "draft_publication",
)

#: Regions whose subjects attract the ethics regime.
RESTRICTED_REGIONS = frozenset({"EU", "DE", "FR", "NL"})

SITES = ("SG", "HK", "AE")


# ── Subjects (a pack's own system-of-record rows, here plain dicts) ───────────

SUBJECTS: dict[str, dict[str, Any]] = {
    "1": {
        "id": "1",
        "name": "Rivera Cohort",
        "region": "DE",
        "home_site": "SG",
        "classification": Classification.C2,
        "cleared_collaborator": True,
    },
    "2": {
        "id": "2",
        "name": "Tan Longitudinal Study",
        "region": "SG",
        "home_site": "SG",
        "classification": Classification.C2,
        "cleared_collaborator": True,
    },
    "3": {
        "id": "3",
        "name": "Al-Rashid Register",
        "region": "AE",
        "home_site": "HK",
        "classification": Classification.C3,
        "cleared_collaborator": True,
    },
    "4": {
        "id": "4",
        "name": "Dubois Panel",
        "region": "FR",
        "home_site": "HK",
        "classification": Classification.C2,
        "cleared_collaborator": False,
    },
}

DATASETS: dict[str, dict[str, Any]] = {
    "1": {
        "id": "1",
        "name": "Genomic Linkage Set",
        "kind": "RESTRICTED",
        "origin_site": "HK",
        "sensitivity": 4,
        "restricted": True,
    },
    "2": {
        "id": "2",
        "name": "Open Climate Index",
        "kind": "OPEN",
        "origin_site": "LU",
        "sensitivity": 2,
        "restricted": False,
    },
    "3": {
        "id": "3",
        "name": "Regional Survey Extract",
        "kind": "SURVEY",
        "origin_site": "SG",
        "sensitivity": 3,
        "restricted": False,
    },
}


class DemoPack:
    """The pack object an application composes with the core."""

    name = "demo"

    def load_subjects(self, envelope: Envelope, session: Session | None = None) -> dict[str, Any]:
        """Resolve the rows an envelope refers to, by reference."""
        loaded: dict[str, Any] = {}
        for kind, ref in envelope.all_subjects().items():
            table = SUBJECTS if kind == "subject" else DATASETS if kind == "dataset" else {}
            row = table.get(str(ref))
            if row is None:
                raise LookupError(f"{kind} {ref!r} not found")
            loaded[kind] = row
        return loaded

    def resolve_attributes(
        self, envelope: Envelope, subjects: dict[str, Any]
    ) -> dict[str, Any]:
        """Derive what policy turns on: the site tuple plus subject properties."""
        subject = subjects.get("subject")
        dataset = subjects.get("dataset")
        return {
            "requester_site": envelope.actor_location,
            "home_site": (subject or {}).get("home_site", "SG"),
            "subject_region": (subject or {}).get("region", "SG"),
            "processing_site": "SG",
            "cleared_collaborator": bool((subject or {}).get("cleared_collaborator")),
            "dataset_kind": (dataset or {}).get("kind"),
            "dataset_restricted": bool((dataset or {}).get("restricted")),
            "origin_site": (dataset or {}).get("origin_site"),
        }

    def rules(self) -> dict[str, Any]:
        """The core primitives are enough for this domain — no custom evaluators."""
        return build_registry()

    def engages(self, policy: Policy, context: RuleContext) -> bool:
        """Whether a regime is actually in scope for this request.

        Skipping un-engaged regimes matters: evaluating a rule from a regime the
        request never touches produces a wrong denial, not merely noise.
        """
        attrs = context.attributes
        subjects = context.subjects

        if policy.applies_to and policy.applies_to != attrs.get("dataset_kind"):
            return False

        if policy.regime == "ETHICS_BOARD":
            return attrs.get("subject_region") in RESTRICTED_REGIONS
        if policy.regime == "HOME_SITE":
            return attrs.get("home_site") == "SG"
        if policy.regime == "ORIGIN_SITE":
            return "dataset" in subjects and (
                attrs.get("origin_site") == "HK" or attrs.get("requester_site") == "HK"
            )
        if policy.regime == "REQUESTER_SITE":
            return attrs.get("requester_site") == "AE"
        return False


def seed_policies(session: Session, *, now: datetime | None = None) -> list[Policy]:
    """Four regimes, mirroring the shape of the real pack's policy set."""
    at = now or utcnow()
    start = at - timedelta(days=365)

    rows = [
        # A cross-site disclosure rule that permits all three sites.
        Policy(
            regime="HOME_SITE",
            rule_type="ATTRIBUTE_ALLOWLIST",
            rule={
                "allow": {"requester_site": ["SG", "HK", "AE"]},
                "on_violation": "escalate",
                "description": "Home-site data may be disclosed to the three federated sites.",
            },
            citation="Data Sharing Charter s4, Schedule 1",
            effective_from=start,
        ),
        # Restricted datasets: origin site restricts which regions may receive them.
        Policy(
            regime="ORIGIN_SITE",
            rule_type="ATTRIBUTE_ALLOWLIST",
            applies_to="RESTRICTED",
            rule={
                "allow": {"subject_region": ["HK", "SG", "AE"]},
                "on_violation": "fail",
                "description": "Restricted datasets may not be released to restricted regions.",
            },
            citation="Origin Site Handling Code para 5.5",
            effective_from=start,
        ),
        # Requester site adds its own bar on restricted datasets.
        Policy(
            regime="REQUESTER_SITE",
            rule_type="ATTRIBUTE_ALLOWLIST",
            applies_to="RESTRICTED",
            rule={
                "allow": {"subject_region": ["AE", "SG", "HK"]},
                "on_violation": "fail",
                "description": "Local promotion rules bar restricted datasets for other regions.",
            },
            citation="Requester Site Conduct Rules COB 3",
            effective_from=start,
        ),
        # Ethics board: drafting is fine, releasing needs a human.
        Policy(
            regime="ETHICS_BOARD",
            rule_type="REQUIRES_HUMAN_APPROVAL",
            rule={
                "tools": ["draft_publication"],
                "description": "No solely automated release affecting a restricted-region subject.",
            },
            citation="Ethics Board Standing Order 22",
            effective_from=start,
        ),
        # Clearance requirement on restricted datasets.
        Policy(
            regime="HOME_SITE",
            rule_type="REQUIRES_FLAG",
            applies_to="RESTRICTED",
            rule={
                "flags": ["cleared_collaborator"],
                "on_violation": "fail",
                "description": "Restricted datasets require a cleared collaborator.",
            },
            citation="Data Sharing Charter s9, clearance register",
            effective_from=start,
        ),
    ]
    for row in rows:
        session.add(row)
    session.flush()
    return rows


def seed_superseded_policy(session: Session, *, now: datetime | None = None) -> Policy:
    """A rule that used to bar AE requesters and was withdrawn 30 days ago.

    Exists so ``as_of`` replay has something to prove: the same envelope must
    deny before the withdrawal date and allow after it.
    """
    at = now or utcnow()
    policy = Policy(
        regime="HOME_SITE",
        rule_type="ATTRIBUTE_DENYLIST",
        rule={
            "deny": {"requester_site": ["AE"]},
            "on_violation": "fail",
            "description": "Historic bar on AE requesters, withdrawn.",
        },
        citation="Data Sharing Charter s4 (superseded)",
        effective_from=at - timedelta(days=365),
        effective_to=at - timedelta(days=30),
    )
    session.add(policy)
    session.flush()
    return policy


# ── Knowledge items, for retrieval conformance ───────────────────────────────

KNOWLEDGE: list[dict[str, Any]] = [
    {
        "title": "Data Sharing Charter — cross-site disclosure",
        "body": (
            "Section 4 of the Charter prohibits disclosure of subject information "
            "outside the home site except under Schedule 1. Cross-border transmission "
            "to a requester at another federated site requires a documented purpose."
        ),
        "kind": "POLICY",
        "classification": "C1",
        "zones": ["SG", "HK", "AE"],
        "source_label": "Data Sharing Charter s4 + Schedule 1",
    },
    {
        "title": "Ethics Board Standing Order 22 — automated release",
        "body": (
            "A subject has the right not to be subject to a release decision based "
            "solely on automated processing. Eligibility scoring for restricted-region "
            "subjects must carry a meaningful human review step before publication."
        ),
        "kind": "POLICY",
        "classification": "C1",
        "zones": ["SG", "HK", "AE"],
        "source_label": "Ethics Board Standing Order 22",
    },
    {
        "title": "Origin Site Handling Code — restricted datasets",
        "body": (
            "Restricted datasets, including genomic linkage sets, may only be "
            "distributed with additional safeguards: eligibility assessment, handling "
            "disclosure, and recorded risk acknowledgment for uncleared collaborators."
        ),
        "kind": "POLICY",
        "classification": "C1",
        "zones": ["SG", "HK"],
        "source_label": "Origin Site Handling Code para 5.5",
    },
    {
        "title": "Requester Site Conduct Rules — classification and inducements",
        "body": (
            "The Conduct of Business module requires classification of collaborators "
            "before any dataset promotion. A requester at the AE site working with a "
            "restricted-region subject triggers both local rules and the home regime."
        ),
        "kind": "POLICY",
        "classification": "C1",
        "zones": ["AE"],
        "source_label": "Requester Site Conduct Rules COB 2.3-2.6",
    },
    {
        "title": "House guidance Q3 — cohort methodology",
        "body": (
            "The methods office maintains a neutral stance on cohort weighting with a "
            "bias to quality controls. Restricted datasets should be used for linkage "
            "only within a cleared-collaborator arrangement."
        ),
        "kind": "RESEARCH",
        "classification": "C1",
        "zones": ["SG", "HK", "AE"],
        "source_label": "Methods Weekly, internal",
    },
    {
        "title": "Al-Rashid register consolidation — restricted note",
        "body": (
            "Consolidation of the Al-Rashid register across sites is under review by "
            "the steering group. Holdings, linkage keys and the consolidation timetable "
            "are need-to-know and must not be surfaced outside the steering group."
        ),
        "kind": "SUBJECT_NOTE",
        "classification": "C3",
        "zones": ["AE", "HK"],
        "source_label": "Steering group minutes, restricted",
    },
]


def envelope(
    tool: str,
    *,
    purpose: str = "COLLABORATION",
    site: str = "SG",
    subject: str | None = None,
    dataset: str | None = None,
    agent: str | None = None,
    principal: str = "test-principal",
    is_admin: bool = False,
) -> Envelope:
    """Build an envelope for this pack — keeps the tests readable."""
    subjects: dict[str, str] = {}
    if dataset is not None:
        subjects["dataset"] = dataset
    return Envelope(
        principal=principal,
        purpose=purpose,
        tool=tool,
        actor_location=site,
        agent_name=agent,
        subject_type="subject" if subject is not None else None,
        subject_id=subject,
        subjects=subjects,
        is_admin=is_admin,
    )
