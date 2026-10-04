"""Small governed agent created by ``kognita scaffold --template governed-agent``.

Subjects, tools, purposes, citations, and the agent name are the ones the
library already uses. The assigned subject is ``2`` (Tan Longitudinal Study).
Asking for subject ``1`` (Rivera Cohort) is the other subject's data.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from typing import Any

from sqlmodel import Session

from kognita import (
    Classification,
    Envelope,
    HashingEmbedder,
    Policy,
    RuleContext,
    build_registry,
    create_all,
    index_item,
    make_engine,
    utcnow,
)
from kognita.registry import register

# Names already used by the demo pack and the gateway tests.
PURPOSES = (
    "COLLABORATION",
    "ELIGIBILITY_CHECK",
    "DOSSIER_PREP",
    "PUBLICATION",
    "ADMIN_REVIEW",
)
AGENT_NAME = "dossier-agent"
OWNER_EXEC = "Head of Research Ops"
PRINCIPAL = "alice"
PURPOSE = "COLLABORATION"
SITE = "SG"
ASSIGNED_SUBJECT = "2"
OTHER_SUBJECT = "1"
QUESTION = "cohort methodology house guidance"
MODEL = "gateway-model"
PROMPT = "Email ana@example.org about the notes"

RESTRICTED_REGIONS = frozenset({"EU", "DE", "FR", "NL"})

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
    """The demo pack, plus a subject-scope allowlist for this agent."""

    name = "demo"

    def load_subjects(
        self, envelope: Envelope, session: Session | None = None
    ) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for kind, ref in envelope.all_subjects().items():
            table = (
                SUBJECTS if kind == "subject" else DATASETS if kind == "dataset" else {}
            )
            row = table.get(str(ref))
            if row is None:
                raise LookupError(f"{kind} {ref!r} not found")
            loaded[kind] = row
        return loaded

    def resolve_attributes(
        self, envelope: Envelope, subjects: dict[str, Any]
    ) -> dict[str, Any]:
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
        return build_registry()

    def engages(self, policy: Policy, context: RuleContext) -> bool:
        allow = policy.rule.get("allow") or {}
        # The assigned-subject rule applies only when the call names a subject.
        # A model call's subject is the model, and this rule does not engage.
        if policy.regime == "HOME_SITE" and "subject_id" in allow:
            return context.envelope.subject_type == "subject"

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


def envelope(subject_id: str) -> Envelope:
    return Envelope(
        principal=PRINCIPAL,
        purpose=PURPOSE,
        tool="get_subject_profile",
        actor_location=SITE,
        agent_name=AGENT_NAME,
        subject_type="subject",
        subject_id=subject_id,
    )


def seed_store(db_path: str) -> None:
    """Create the SQLite policy and evidence store and load the demo rows."""
    engine = make_engine(db_path)
    create_all(engine)
    embedder = HashingEmbedder()
    start = utcnow() - timedelta(days=365)
    with Session(engine) as session:
        session.add_all(
            [
                Policy(
                    regime="HOME_SITE",
                    rule_type="ATTRIBUTE_ALLOWLIST",
                    rule={
                        "allow": {"requester_site": ["SG", "HK", "AE"]},
                        "on_violation": "escalate",
                        "description": (
                            "Home-site data may be disclosed to the three federated sites."
                        ),
                    },
                    citation="Data Sharing Charter s4, Schedule 1",
                    effective_from=start,
                ),
                Policy(
                    regime="HOME_SITE",
                    rule_type="ATTRIBUTE_ALLOWLIST",
                    rule={
                        "allow": {"subject_id": [ASSIGNED_SUBJECT]},
                        "on_violation": "fail",
                        "description": (
                            "dossier-agent may read only the subject it is assigned."
                        ),
                    },
                    citation="Data Sharing Charter s4, Schedule 1",
                    effective_from=start,
                ),
                Policy(
                    regime="ORIGIN_SITE",
                    rule_type="ATTRIBUTE_ALLOWLIST",
                    applies_to="RESTRICTED",
                    rule={
                        "allow": {"subject_region": ["HK", "SG", "AE"]},
                        "on_violation": "fail",
                        "description": (
                            "Restricted datasets may not be released to restricted regions."
                        ),
                    },
                    citation="Origin Site Handling Code para 5.5",
                    effective_from=start,
                ),
                Policy(
                    regime="REQUESTER_SITE",
                    rule_type="ATTRIBUTE_ALLOWLIST",
                    applies_to="RESTRICTED",
                    rule={
                        "allow": {"subject_region": ["AE", "SG", "HK"]},
                        "on_violation": "fail",
                        "description": (
                            "Local promotion rules bar restricted datasets for other regions."
                        ),
                    },
                    citation="Requester Site Conduct Rules COB 3",
                    effective_from=start,
                ),
                Policy(
                    regime="ETHICS_BOARD",
                    rule_type="REQUIRES_HUMAN_APPROVAL",
                    rule={
                        "tools": ["draft_publication"],
                        "description": (
                            "No solely automated release affecting a restricted-region subject."
                        ),
                    },
                    citation="Ethics Board Standing Order 22",
                    effective_from=start,
                ),
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
        )
        register(session, name=AGENT_NAME, owner_exec=OWNER_EXEC)
        index_item(
            session,
            title="House guidance Q3 — cohort methodology",
            body=(
                "The methods office maintains a neutral stance on cohort weighting with a "
                "bias to quality controls. Restricted datasets should be used for linkage "
                "only within a cleared-collaborator arrangement."
            ),
            embedder=embedder,
            kind="RESEARCH",
            classification=Classification.C1,
            zones=["SG", "HK", "AE"],
            source_label="Methods Weekly, internal",
        )
        index_item(
            session,
            title="Data Sharing Charter — cross-site disclosure",
            body=(
                "Section 4 of the Charter prohibits disclosure of subject information "
                "outside the home site except under Schedule 1. Cross-border transmission "
                "to a requester at another federated site requires a documented purpose."
            ),
            embedder=embedder,
            kind="POLICY",
            classification=Classification.C1,
            zones=["SG", "HK", "AE"],
            source_label="Data Sharing Charter s4 + Schedule 1",
        )
        session.commit()
    engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Governed agent demo application")
    sub = parser.add_subparsers(dest="command", required=True)
    seed = sub.add_parser("seed", help="create the policy and evidence store")
    seed.add_argument("--db", required=True)
    args = parser.parse_args(argv)
    if args.command == "seed":
        seed_store(args.db)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
