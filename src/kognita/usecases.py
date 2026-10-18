"""The use-case register.

Every decision that is given a register must name a use case in it, and the
check cites that use case's accountable owner. An unregistered, retired,
suspended, or not-yet-activated use case is a denial. This is the check that
replaces the free-text purpose list on that path.

The envelope's ``purpose`` is the use case id. That is already how a use case
is named on a decision; the register's own purpose field is the statement of
what the use case is for.

Out of this slice: model cards, unmediated entries, provider-dependency counts,
and retirement beyond the status flag. Setting a use case to retired denies
later decisions and is evidenced. It does not stop agents, apply retention, or
notify anyone.

Autonomy levels appear here only as the ceiling a tier's minimum controls
allow (L0 retrieve through L6 broad autonomy). Enforcing a level on a call is
not this register.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from sqlmodel import Session, select

from kognita.envelope import Check, Envelope
from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.models import (
    Agent,
    RiskAppetite,
    UseCase,
    usecase_content_hash,
    utcnow,
)
from kognita.vocabulary import ActorType, CheckResult, EventType

#: Who is accountable, beyond the five owner roles. A seat names a person.
#: An agent cannot sit in one.
ACCOUNTABILITY: tuple[str, ...] = (
    "data",
    "model",
    "agent",
    "business outcome",
    "client communication",
    "suitability and advice",
    "compliance",
)

#: The use-case categories the tier names. A category outside this set is refused.
CATEGORIES: tuple[str, ...] = (
    "research summarization",
    "client intelligence",
    "meeting preparation",
    "kyc",
    "suitability",
    "recommendations",
    "financial execution",
)

BUSINESS_CRITICALITY: tuple[str, ...] = ("low", "medium", "high", "critical")
CLIENT_IMPACT: tuple[str, ...] = (
    "internal productivity",
    "rm assistance",
    "client influence",
    "client communication",
    "advice",
    "execution",
)
AUTONOMY_AXIS: tuple[str, ...] = ("none", "assisted", "conditional", "high")
DECISION_CONSEQUENCE: tuple[str, ...] = (
    "informational",
    "operational",
    "client communication",
    "suitability",
    "financial action",
)
DATA_SENSITIVITY: tuple[str, ...] = (
    "public",
    "internal",
    "confidential",
    "highly sensitive",
)
MATERIALITY: tuple[str, ...] = ("low", "medium", "high", "critical")
APPROVAL_GATES: tuple[str, ...] = (
    "none",
    "human_review",
    "human_approval",
    "two_signature",
)
EXPLANATION_DEPTHS: tuple[str, ...] = ("citation", "sources", "suitability", "full")
STATUSES: tuple[str, ...] = ("pilot", "live", "suspended", "retired")

#: Complexity is recorded as the four factors the classification names.
COMPLEXITY_FACTORS: dict[str, tuple[str, ...]] = {
    "technology": ("established", "emerging", "experimental"),
    "novelty": ("familiar", "adapted", "new"),
    "explainability": ("explainable", "partial", "opaque"),
    "third_party_visibility": ("not_applicable", "full", "partial", "none"),
}

# Rank within each closed set. The derived tier is the highest rank that applies,
# and never lower than the category. Confidential data and a new application can
# lift a research use case; they cannot lower an execution use case.
_CATEGORY_TIER = {
    "research summarization": 1,
    "client intelligence": 2,
    "meeting preparation": 2,
    "kyc": 3,
    "suitability": 3,
    "recommendations": 3,
    "financial execution": 4,
}
_CRITICALITY_TIER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_CLIENT_IMPACT_TIER = {
    "internal productivity": 1,
    "rm assistance": 2,
    "client influence": 2,
    "client communication": 3,
    "advice": 3,
    "execution": 4,
}
_AUTONOMY_TIER = {"none": 1, "assisted": 2, "conditional": 3, "high": 4}
_CONSEQUENCE_TIER = {
    "informational": 1,
    "operational": 2,
    "client communication": 3,
    "suitability": 3,
    "financial action": 4,
}
# Public and internal stay at tier 1. Confidential lifts to tier 2. Highly
# sensitive lifts to tier 3. Execution still comes from the other axes.
_DATA_TIER = {"public": 1, "internal": 1, "confidential": 2, "highly sensitive": 3}
_COMPLEXITY_RANK = {
    "technology": {"established": 1, "emerging": 2, "experimental": 4},
    "novelty": {"familiar": 1, "adapted": 2, "new": 3},
    "explainability": {"explainable": 1, "partial": 3, "opaque": 4},
    "third_party_visibility": {"not_applicable": 1, "full": 1, "partial": 3, "none": 4},
}
_COMPLEXITY_LEVEL = {1: "low", 2: "moderate", 3: "high", 4: "novel"}
_MATERIALITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_APPROVAL_RANK = {"none": 0, "human_review": 1, "human_approval": 2, "two_signature": 3}
_EXPLANATION_RANK = {"citation": 1, "sources": 2, "suitability": 3, "full": 4}

#: How long a tier may run before its review date. The date is the deadline:
#: on that date the use case is denied until it is activated again.
REVIEW_DAYS = {1: 365, 2: 180, 3: 90, 4: 90}

#: Minimum controls a tier requires. A weaker gate, a lower sampling rate, a
#: higher autonomy ceiling, missing monitoring, a shallower explanation, or a
#: missing validator fails validation. The autonomy integers are the ceiling
#: only: L0 is 0 and L6 is 6.
_TIER_MINIMUMS: dict[int, dict[str, Any]] = {
    1: {
        "approval": "none",
        "sampling": 0.05,
        "max_autonomy": 2,
        "monitoring": True,
        "explanation": "citation",
    },
    2: {
        "approval": "human_review",
        "sampling": 0.10,
        "max_autonomy": 3,
        "monitoring": True,
        "explanation": "sources",
    },
    3: {
        "approval": "human_approval",
        "sampling": 0.50,
        "max_autonomy": 4,
        "monitoring": True,
        "explanation": "suitability",
    },
    4: {
        "approval": "two_signature",
        "sampling": 1.0,
        "max_autonomy": 4,
        "monitoring": True,
        "explanation": "full",
    },
}

_OWNER_ROLES: tuple[tuple[str, str], ...] = (
    ("business_owner", "business owner"),
    ("technology_owner", "technology owner"),
    ("risk_owner", "risk owner"),
    ("model_owner", "model owner"),
    ("knowledge_lead", "Knowledge Lead"),
)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _aware(value)
    parsed = datetime.fromisoformat(str(value))
    return _aware(parsed)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _aware(value).isoformat()


def _sorted_texts(values: Sequence[str]) -> list[str]:
    return sorted(str(item) for item in values)


def _cite(usecase_id: str, text: str, owner: str = "") -> str:
    citation = f"Use-case register — {text}"
    if owner:
        return f"{citation} · accountable owner: {owner}"
    return citation


@dataclass(frozen=True)
class UseCaseSpec:
    """The entry a registrant proposes. The tier is derived from it."""

    usecase_id: str
    purpose: str
    category: str
    business_owner: str
    technology_owner: str
    risk_owner: str
    model_owner: str
    control_function: str
    accountability: Mapping[str, str] = field(default_factory=dict)
    knowledge_lead: str = ""
    builders: tuple[str, ...] = ()
    business_criticality: str = "low"
    client_impact: str = "internal productivity"
    autonomy: str = "none"
    decision_consequence: str = "informational"
    data_sensitivity: str = "public"
    complexity: Mapping[str, str] = field(default_factory=dict)
    inherent_materiality: str = "low"
    credited_controls: tuple[str, ...] = ()
    residual_materiality: str = "low"
    approval_gate: str = "none"
    review_sampling_rate: float = 0.0
    maximum_autonomy: int = 0
    monitoring: bool = False
    explanation_depth: str = "citation"
    validators: tuple[str, ...] = ()
    human_decision_points: tuple[str, ...] = ()
    client_segments: tuple[str, ...] = ()
    investor_impact: str = ""
    permitted_data_classes: tuple[str, ...] = ()
    approved_models: tuple[str, ...] = ()
    agent_authority: str = ""
    retention_days: int | None = None
    business_process: str = ""
    products: tuple[str, ...] = ()
    data_sources: tuple[str, ...] = ()
    vendors: tuple[str, ...] = ()
    jurisdiction: str = ""
    client_population: str = ""
    intended_use: str = ""
    pilot_users: tuple[str, ...] = ()
    pilot_end: datetime | None = None
    success_criteria: str = ""
    review_date: datetime | None = None


@dataclass(frozen=True)
class Appetite:
    """The risk appetite a registration or activation is checked against."""

    allowed_categories: tuple[str, ...]
    prohibited_categories: tuple[str, ...]
    maximum_autonomy: int
    permitted_client_impact: tuple[str, ...]
    external_communication: bool
    transaction_authority: bool
    maximum_residual_materiality: str
    version: int = 1

    def to_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "allowed_categories": _sorted_texts(self.allowed_categories),
            "prohibited_categories": _sorted_texts(self.prohibited_categories),
            "maximum_autonomy": self.maximum_autonomy,
            "permitted_client_impact": _sorted_texts(self.permitted_client_impact),
            "external_communication": self.external_communication,
            "transaction_authority": self.transaction_authority,
            "maximum_residual_materiality": self.maximum_residual_materiality,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Appetite:
        return cls(
            allowed_categories=tuple(record.get("allowed_categories") or ()),
            prohibited_categories=tuple(record.get("prohibited_categories") or ()),
            maximum_autonomy=int(record.get("maximum_autonomy") or 0),
            permitted_client_impact=tuple(record.get("permitted_client_impact") or ()),
            external_communication=bool(record.get("external_communication")),
            transaction_authority=bool(record.get("transaction_authority")),
            maximum_residual_materiality=str(
                record.get("maximum_residual_materiality") or ""
            ),
            version=int(record.get("version") or 1),
        )


@dataclass(frozen=True)
class RegisteredUseCase:
    """One effective use-case version, as :func:`decide` reads it."""

    usecase_id: str
    version: int
    purpose: str
    category: str
    status: str
    business_owner: str
    technology_owner: str
    risk_owner: str
    model_owner: str
    knowledge_lead: str
    accountability: tuple[tuple[str, str], ...]
    control_function: str
    builders: tuple[str, ...]
    business_criticality: str
    client_impact: str
    autonomy: str
    decision_consequence: str
    data_sensitivity: str
    complexity: tuple[tuple[str, str], ...]
    complexity_level: str
    proposed_tier: int
    rating_approved: bool
    rating_approved_by: str
    validation_signoff_by: str
    inherent_materiality: str
    credited_controls: tuple[str, ...]
    residual_materiality: str
    approval_gate: str
    review_sampling_rate: float
    maximum_autonomy: int
    monitoring: bool
    explanation_depth: str
    validators: tuple[str, ...]
    human_decision_points: tuple[str, ...]
    pilot_users: tuple[str, ...]
    pilot_end: datetime | None
    success_criteria: str
    criteria_met: bool
    review_date: datetime | None
    jurisdiction: str
    client_population: str
    intended_use: str
    record: dict[str, Any]
    content_hash: str = ""

    @property
    def owner(self) -> str:
        return self.business_owner


@dataclass(frozen=True)
class UseCaseRegister:
    """The use cases and appetite in force at one instant. Pure input to decide."""

    entries: tuple[RegisteredUseCase, ...] = ()
    appetite: Appetite | None = None
    agents: frozenset[str] = frozenset()

    def get(self, usecase_id: str) -> RegisteredUseCase | None:
        matches = [entry for entry in self.entries if entry.usecase_id == usecase_id]
        if not matches:
            return None
        return max(matches, key=lambda entry: entry.version)


def derived_tier(spec: UseCaseSpec) -> int:
    """The tier the classification proposes. The highest applicable rank wins."""
    ranks = [
        _CATEGORY_TIER.get(spec.category, 0),
        _CRITICALITY_TIER.get(spec.business_criticality, 0),
        _CLIENT_IMPACT_TIER.get(spec.client_impact, 0),
        _AUTONOMY_TIER.get(spec.autonomy, 0),
        _CONSEQUENCE_TIER.get(spec.decision_consequence, 0),
        _DATA_TIER.get(spec.data_sensitivity, 0),
        _complexity_rank(spec.complexity),
    ]
    return max(ranks) if any(ranks) else 0


def _complexity_rank(factors: Mapping[str, str]) -> int:
    ranks = [
        _COMPLEXITY_RANK[name].get(str(factors.get(name) or ""), 0)
        for name in COMPLEXITY_FACTORS
    ]
    return max(ranks) if ranks else 0


def complexity_level(factors: Mapping[str, str]) -> str:
    return _COMPLEXITY_LEVEL.get(_complexity_rank(factors), "")


def _owner_pairs(spec: UseCaseSpec) -> tuple[tuple[str, str], ...]:
    named = (
        ("business owner", spec.business_owner),
        ("technology owner", spec.technology_owner),
        ("risk owner", spec.risk_owner),
        ("model owner", spec.model_owner),
        ("Knowledge Lead", spec.knowledge_lead),
    )
    seats = tuple(
        (f"accountability for {seat}", str(spec.accountability.get(seat) or ""))
        for seat in ACCOUNTABILITY
    )
    return named + seats


def _entry_owner_pairs(entry: RegisteredUseCase) -> tuple[tuple[str, str], ...]:
    named = (
        ("business owner", entry.business_owner),
        ("technology owner", entry.technology_owner),
        ("risk owner", entry.risk_owner),
        ("model owner", entry.model_owner),
        ("Knowledge Lead", entry.knowledge_lead),
    )
    return named + entry.accountability


def vocabulary_citations(spec: UseCaseSpec) -> tuple[str, ...]:
    """Closed-set failures. A value nobody defined is refused."""
    found: list[str] = []
    usecase_id = spec.usecase_id or "(unnamed)"

    def add(text: str) -> None:
        found.append(_cite(usecase_id, text))

    if not spec.usecase_id.strip():
        add("a use case id is required")
    if not spec.purpose.strip():
        add(f"'{usecase_id}' has no purpose")
    if spec.category not in CATEGORIES:
        add(f"category '{spec.category}' is not a recognised use-case category")
    if spec.business_criticality not in BUSINESS_CRITICALITY:
        add(f"business criticality '{spec.business_criticality}' is not recognised")
    if spec.client_impact not in CLIENT_IMPACT:
        add(f"client impact '{spec.client_impact}' is not recognised")
    if spec.autonomy not in AUTONOMY_AXIS:
        add(f"autonomy '{spec.autonomy}' is not recognised")
    if spec.decision_consequence not in DECISION_CONSEQUENCE:
        add(f"decision consequence '{spec.decision_consequence}' is not recognised")
    if spec.data_sensitivity not in DATA_SENSITIVITY:
        add(f"data sensitivity '{spec.data_sensitivity}' is not recognised")
    if spec.inherent_materiality not in MATERIALITY:
        add(f"inherent materiality '{spec.inherent_materiality}' is not recognised")
    if spec.residual_materiality not in MATERIALITY:
        add(f"residual materiality '{spec.residual_materiality}' is not recognised")
    if spec.approval_gate not in APPROVAL_GATES:
        add(f"approval gate '{spec.approval_gate}' is not recognised")
    if spec.explanation_depth not in EXPLANATION_DEPTHS:
        add(f"explanation depth '{spec.explanation_depth}' is not recognised")
    if isinstance(spec.maximum_autonomy, bool) or not isinstance(
        spec.maximum_autonomy, int
    ):
        add("maximum autonomy must be an autonomy level from 0 to 6")
    elif not 0 <= spec.maximum_autonomy <= 6:
        add(f"maximum autonomy L{spec.maximum_autonomy} is outside L0 to L6")
    if not 0 <= float(spec.review_sampling_rate) <= 1:
        add("review sampling rate must be between 0 and 1")
    for factor, allowed in COMPLEXITY_FACTORS.items():
        value = str(spec.complexity.get(factor) or "")
        if value not in allowed:
            add(f"complexity factor '{factor}' is '{value or 'missing'}'")
    unknown_seats = sorted(set(spec.accountability) - set(ACCOUNTABILITY))
    for seat in unknown_seats:
        add(f"accountability seat '{seat}' is not part of the matrix")
    for seat in ACCOUNTABILITY:
        if not str(spec.accountability.get(seat) or "").strip():
            add(f"accountability for {seat} is not named")
    return tuple(found)


def _agent_citations(
    pairs: Sequence[tuple[str, str]], agents: frozenset[str], usecase_id: str
) -> tuple[str, ...]:
    found: list[str] = []
    for role, name in pairs:
        if name and name in agents:
            found.append(
                _cite(
                    usecase_id,
                    f"'{usecase_id}' names agent '{name}' as {role}",
                    owner="",
                )
            )
    return tuple(found)


def _control_function_conflict(
    business: str, technology: str, control: str, usecase_id: str
) -> str | None:
    if not control.strip():
        return _cite(usecase_id, f"'{usecase_id}' has no designated control function")
    if control == business or control == technology:
        return _cite(
            usecase_id,
            "the control function cannot approve a use case it owns",
        )
    return None


def communicates_externally(
    category: str, client_impact: str, consequence: str
) -> bool:
    return (
        client_impact == "client communication" or consequence == "client communication"
    )


def has_transaction_authority(
    category: str, client_impact: str, consequence: str
) -> bool:
    return (
        category == "financial execution"
        or client_impact == "execution"
        or consequence == "financial action"
    )


def appetite_citations(
    entry: RegisteredUseCase, appetite: Appetite | None
) -> tuple[str, ...]:
    """Why the use case is outside the appetite. Residual materiality is separate."""
    if appetite is None:
        return (_cite(entry.usecase_id, "no risk appetite is in force", entry.owner),)
    found: list[str] = []
    if entry.category in set(appetite.prohibited_categories):
        found.append(
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' is outside the risk appetite: "
                f"category '{entry.category}' is prohibited",
                entry.owner,
            )
        )
    elif appetite.allowed_categories and entry.category not in set(
        appetite.allowed_categories
    ):
        found.append(
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' is outside the risk appetite: "
                f"category '{entry.category}' is not allowed",
                entry.owner,
            )
        )
    if entry.maximum_autonomy > appetite.maximum_autonomy:
        found.append(
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' is outside the risk appetite: "
                f"maximum autonomy L{entry.maximum_autonomy} exceeds "
                f"L{appetite.maximum_autonomy}",
                entry.owner,
            )
        )
    if entry.client_impact not in set(appetite.permitted_client_impact):
        found.append(
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' is outside the risk appetite: "
                f"client impact '{entry.client_impact}' is not permitted",
                entry.owner,
            )
        )
    if (
        communicates_externally(
            entry.category, entry.client_impact, entry.decision_consequence
        )
        and not appetite.external_communication
    ):
        found.append(
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' is outside the risk appetite: "
                "external communication is not permitted",
                entry.owner,
            )
        )
    if (
        has_transaction_authority(
            entry.category, entry.client_impact, entry.decision_consequence
        )
        and not appetite.transaction_authority
    ):
        found.append(
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' is outside the risk appetite: "
                "transaction authority is not permitted",
                entry.owner,
            )
        )
    return tuple(found)


def residual_citation(
    entry: RegisteredUseCase, appetite: Appetite | None
) -> str | None:
    if appetite is None:
        return None
    ceiling = _MATERIALITY_RANK.get(appetite.maximum_residual_materiality)
    actual = _MATERIALITY_RANK.get(entry.residual_materiality)
    if ceiling is None or actual is None or actual > ceiling:
        return _cite(
            entry.usecase_id,
            f"'{entry.usecase_id}' residual materiality is outside the risk appetite",
            entry.owner,
        )
    return None


def knowledge_lead_citation(entry: RegisteredUseCase) -> str | None:
    if entry.proposed_tier >= 2 and not entry.knowledge_lead.strip():
        return _cite(
            entry.usecase_id,
            f"tier {entry.proposed_tier} use case '{entry.usecase_id}' has no Knowledge Lead",
            entry.owner,
        )
    return None


def minimum_control_citations(entry: RegisteredUseCase) -> tuple[str, ...]:
    """Controls below the tier's minimum. Each shortfall cites the tier."""
    floor = _TIER_MINIMUMS.get(entry.proposed_tier)
    if floor is None:
        return (
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' has no tier to set minimum controls",
                entry.owner,
            ),
        )
    found: list[str] = []
    tier = entry.proposed_tier
    if _APPROVAL_RANK.get(entry.approval_gate, -1) < _APPROVAL_RANK[floor["approval"]]:
        found.append(
            f"approval gate '{entry.approval_gate}' is below '{floor['approval']}'"
        )
    if float(entry.review_sampling_rate) + 1e-9 < float(floor["sampling"]):
        found.append(
            f"review sampling rate {entry.review_sampling_rate} is below {floor['sampling']}"
        )
    if entry.maximum_autonomy > int(floor["max_autonomy"]):
        found.append(
            f"maximum autonomy L{entry.maximum_autonomy} is above the tier ceiling "
            f"L{floor['max_autonomy']}"
        )
    if floor["monitoring"] and not entry.monitoring:
        found.append("monitoring is required")
    if (
        _EXPLANATION_RANK.get(entry.explanation_depth, 0)
        < _EXPLANATION_RANK[floor["explanation"]]
    ):
        found.append(
            f"explanation depth '{entry.explanation_depth}' is below '{floor['explanation']}'"
        )
    required_validators = [entry.business_owner] if entry.business_owner else []
    if tier >= 2 and entry.knowledge_lead:
        required_validators.append(entry.knowledge_lead)
    missing = [
        name for name in required_validators if name not in set(entry.validators)
    ]
    if missing:
        found.append("validators do not include " + ", ".join(missing))
    if tier >= 3:
        independent = [
            name
            for name in entry.validators
            if name and name not in set(entry.builders)
        ]
        if not independent:
            found.append("validators are not independent of the builders")
        if not entry.human_decision_points:
            found.append("no human decision point is recorded")
        if not entry.builders:
            found.append(
                "builders are not named, so separation of duties cannot be checked"
            )
    if not found:
        return ()
    detail = "; ".join(found)
    return (
        _cite(
            entry.usecase_id,
            f"'{entry.usecase_id}' is configured below tier {tier} minimum controls: {detail}",
            entry.owner,
        ),
    )


def review_deadline(tier: int, at: datetime) -> datetime:
    """The review date a tier sets, measured from ``at``."""
    return _aware(at) + timedelta(days=REVIEW_DAYS[tier])


def build_record(
    spec: UseCaseSpec,
    *,
    version: int,
    status: str = "",
    rating_approved: bool = False,
    rating_approved_by: str = "",
    validation_signoff_by: str = "",
    criteria_met: bool = False,
    review_date: datetime | None = None,
) -> dict[str, Any]:
    """The stored entry. The tier is written here so a later read does not re-derive it."""
    tier = derived_tier(spec)
    level = complexity_level(spec.complexity)
    return {
        "usecase_id": spec.usecase_id,
        "version": version,
        "purpose": spec.purpose,
        "category": spec.category,
        "status": status,
        "business_owner": spec.business_owner.strip(),
        "technology_owner": spec.technology_owner.strip(),
        "risk_owner": spec.risk_owner.strip(),
        "model_owner": spec.model_owner.strip(),
        "knowledge_lead": spec.knowledge_lead.strip(),
        "accountability": {
            seat: str(spec.accountability.get(seat) or "").strip()
            for seat in ACCOUNTABILITY
        },
        "control_function": spec.control_function.strip(),
        "builders": _sorted_texts(spec.builders),
        "business_criticality": spec.business_criticality,
        "client_impact": spec.client_impact,
        "autonomy": spec.autonomy,
        "decision_consequence": spec.decision_consequence,
        "data_sensitivity": spec.data_sensitivity,
        "complexity": {
            factor: str(spec.complexity.get(factor) or "")
            for factor in COMPLEXITY_FACTORS
        },
        "complexity_level": level,
        "proposed_tier": tier,
        "rating_approved": rating_approved,
        "rating_approved_by": rating_approved_by,
        "validation_signoff_by": validation_signoff_by,
        "inherent_materiality": spec.inherent_materiality,
        "credited_controls": _sorted_texts(spec.credited_controls),
        "residual_materiality": spec.residual_materiality,
        "approval_gate": spec.approval_gate,
        "review_sampling_rate": float(spec.review_sampling_rate),
        "maximum_autonomy": int(spec.maximum_autonomy),
        "monitoring": bool(spec.monitoring),
        "explanation_depth": spec.explanation_depth,
        "validators": _sorted_texts(spec.validators),
        "human_decision_points": _sorted_texts(spec.human_decision_points),
        "client_segments": _sorted_texts(spec.client_segments),
        "investor_impact": spec.investor_impact,
        "permitted_data_classes": _sorted_texts(spec.permitted_data_classes),
        "approved_models": _sorted_texts(spec.approved_models),
        "agent_authority": spec.agent_authority,
        "retention_days": spec.retention_days,
        "business_process": spec.business_process,
        "products": _sorted_texts(spec.products),
        "data_sources": _sorted_texts(spec.data_sources),
        "vendors": _sorted_texts(spec.vendors),
        "jurisdiction": spec.jurisdiction,
        "client_population": spec.client_population,
        "intended_use": spec.intended_use,
        "pilot_users": _sorted_texts(spec.pilot_users),
        "pilot_end": _iso(spec.pilot_end),
        "success_criteria": spec.success_criteria,
        "criteria_met": criteria_met,
        "review_date": _iso(
            review_date if review_date is not None else spec.review_date
        ),
    }


def from_record(
    record: Mapping[str, Any], *, content_hash: str = ""
) -> RegisteredUseCase:
    """Read a stored entry back into the shape the checks use."""
    accountability = record.get("accountability") or {}
    complexity = record.get("complexity") or {}
    return RegisteredUseCase(
        usecase_id=str(record.get("usecase_id") or ""),
        version=int(record.get("version") or 0),
        purpose=str(record.get("purpose") or ""),
        category=str(record.get("category") or ""),
        status=str(record.get("status") or ""),
        business_owner=str(record.get("business_owner") or ""),
        technology_owner=str(record.get("technology_owner") or ""),
        risk_owner=str(record.get("risk_owner") or ""),
        model_owner=str(record.get("model_owner") or ""),
        knowledge_lead=str(record.get("knowledge_lead") or ""),
        accountability=tuple(
            (f"accountability for {seat}", str(accountability.get(seat) or ""))
            for seat in ACCOUNTABILITY
        ),
        control_function=str(record.get("control_function") or ""),
        builders=tuple(record.get("builders") or ()),
        business_criticality=str(record.get("business_criticality") or ""),
        client_impact=str(record.get("client_impact") or ""),
        autonomy=str(record.get("autonomy") or ""),
        decision_consequence=str(record.get("decision_consequence") or ""),
        data_sensitivity=str(record.get("data_sensitivity") or ""),
        complexity=tuple(
            (factor, str(complexity.get(factor) or "")) for factor in COMPLEXITY_FACTORS
        ),
        complexity_level=str(record.get("complexity_level") or ""),
        proposed_tier=int(record.get("proposed_tier") or 0),
        rating_approved=bool(record.get("rating_approved")),
        rating_approved_by=str(record.get("rating_approved_by") or ""),
        validation_signoff_by=str(record.get("validation_signoff_by") or ""),
        inherent_materiality=str(record.get("inherent_materiality") or ""),
        credited_controls=tuple(record.get("credited_controls") or ()),
        residual_materiality=str(record.get("residual_materiality") or ""),
        approval_gate=str(record.get("approval_gate") or ""),
        review_sampling_rate=float(record.get("review_sampling_rate") or 0),
        maximum_autonomy=int(record.get("maximum_autonomy") or 0),
        monitoring=bool(record.get("monitoring")),
        explanation_depth=str(record.get("explanation_depth") or ""),
        validators=tuple(record.get("validators") or ()),
        human_decision_points=tuple(record.get("human_decision_points") or ()),
        pilot_users=tuple(record.get("pilot_users") or ()),
        pilot_end=_parse_time(record.get("pilot_end")),
        success_criteria=str(record.get("success_criteria") or ""),
        criteria_met=bool(record.get("criteria_met")),
        review_date=_parse_time(record.get("review_date")),
        jurisdiction=str(record.get("jurisdiction") or ""),
        client_population=str(record.get("client_population") or ""),
        intended_use=str(record.get("intended_use") or ""),
        record=dict(record),
        content_hash=content_hash,
    )


def registration_citations(
    spec: UseCaseSpec,
    appetite: Appetite | None,
    agents: frozenset[str],
) -> tuple[str, ...]:
    """Refusals that stop a use case being registered at all."""
    found = list(vocabulary_citations(spec))
    if found:
        return tuple(found)
    entry = from_record(build_record(spec, version=1))
    if not entry.business_owner:
        found.append(
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' has no current accountable owner",
            )
        )
    found.extend(_agent_citations(_owner_pairs(spec), agents, entry.usecase_id))
    conflict = _control_function_conflict(
        entry.business_owner,
        entry.technology_owner,
        entry.control_function,
        entry.usecase_id,
    )
    if conflict is not None:
        found.append(conflict)
    found.extend(appetite_citations(entry, appetite))
    residual = residual_citation(entry, appetite)
    if residual is not None:
        found.append(residual)
    return tuple(found)


def configuration_citations(
    entry: RegisteredUseCase,
    appetite: Appetite | None,
    agents: frozenset[str],
    *,
    at: datetime,
) -> tuple[str, ...]:
    """What ``usecase validate`` reports. Lifecycle status is not itself a failure."""
    found: list[str] = []
    if not entry.business_owner:
        found.append(
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' has no current accountable owner",
            )
        )
    found.extend(_agent_citations(_entry_owner_pairs(entry), agents, entry.usecase_id))
    conflict = _control_function_conflict(
        entry.business_owner,
        entry.technology_owner,
        entry.control_function,
        entry.usecase_id,
    )
    if conflict is not None:
        found.append(conflict)
    lead = knowledge_lead_citation(entry)
    if lead is not None:
        found.append(lead)
    found.extend(minimum_control_citations(entry))
    found.extend(appetite_citations(entry, appetite))
    residual = residual_citation(entry, appetite)
    if residual is not None:
        found.append(residual)
    if entry.status == "pilot":
        found.extend(_pilot_shape_citations(entry))
    if entry.status in ("pilot", "live"):
        review = _review_citation(entry, at)
        if review is not None:
            found.append(review)
        if not entry.rating_approved:
            found.append(
                _cite(
                    entry.usecase_id,
                    f"'{entry.usecase_id}' materiality rating is not approved by the control function",
                    entry.owner,
                )
            )
    return tuple(found)


def _pilot_shape_citations(entry: RegisteredUseCase) -> tuple[str, ...]:
    found: list[str] = []
    if not entry.pilot_users:
        found.append(
            _cite(
                entry.usecase_id,
                f"pilot '{entry.usecase_id}' has no named user list",
                entry.owner,
            )
        )
    if entry.pilot_end is None:
        found.append(
            _cite(
                entry.usecase_id,
                f"pilot '{entry.usecase_id}' has no end date",
                entry.owner,
            )
        )
    if not entry.success_criteria.strip():
        found.append(
            _cite(
                entry.usecase_id,
                f"pilot '{entry.usecase_id}' has no success criteria",
                entry.owner,
            )
        )
    return tuple(found)


def _review_citation(entry: RegisteredUseCase, at: datetime) -> str | None:
    if entry.review_date is None or _aware(at) >= entry.review_date:
        return _cite(
            entry.usecase_id,
            f"'{entry.usecase_id}' is past its review date",
            entry.owner,
        )
    return None


def _check(name: str, citation: str, *, passed: bool = False) -> Check:
    return Check(
        check=name,
        regime="INTERNAL",
        result=CheckResult.PASS if passed else CheckResult.FAIL,
        citation=citation,
    )


def usecase_checks(
    envelope: Envelope,
    register: UseCaseRegister,
    at: datetime,
) -> list[Check]:
    """The use-case rules for one decision. Each failure cites its rule.

    A passing decision is one check, and that check names the accountable owner.
    """
    usecase_id = envelope.purpose
    entry = register.get(usecase_id)
    if entry is None:
        return [
            _check(
                "USE_CASE_REGISTERED",
                _cite(
                    usecase_id or "(none)",
                    f"'{usecase_id or '(none)'}' is not registered",
                ),
            )
        ]

    failures: list[Check] = []
    if entry.status == "retired":
        failures.append(
            _check(
                "USE_CASE_STATUS",
                _cite(
                    entry.usecase_id, f"'{entry.usecase_id}' is retired", entry.owner
                ),
            )
        )
    elif entry.status == "suspended":
        failures.append(
            _check(
                "USE_CASE_STATUS",
                _cite(
                    entry.usecase_id, f"'{entry.usecase_id}' is suspended", entry.owner
                ),
            )
        )
    elif entry.status not in ("pilot", "live"):
        failures.append(
            _check(
                "USE_CASE_STATUS",
                _cite(
                    entry.usecase_id,
                    f"'{entry.usecase_id}' is not activated",
                    entry.owner,
                ),
            )
        )

    if not entry.business_owner:
        failures.append(
            _check(
                "USE_CASE_OWNER",
                _cite(
                    entry.usecase_id,
                    f"'{entry.usecase_id}' has no current accountable owner",
                ),
            )
        )
    for citation in _agent_citations(
        _entry_owner_pairs(entry), register.agents, entry.usecase_id
    ):
        failures.append(
            _check("USE_CASE_AGENT_OWNER", _cite_owner_if_needed(citation, entry))
        )

    lead = knowledge_lead_citation(entry)
    if lead is not None:
        failures.append(_check("USE_CASE_KNOWLEDGE_LEAD", lead))
    for citation in minimum_control_citations(entry):
        failures.append(_check("USE_CASE_MINIMUM_CONTROLS", citation))

    if entry.status in ("pilot", "live"):
        review = _review_citation(entry, at)
        if review is not None:
            failures.append(_check("USE_CASE_REVIEW", review))

    if not entry.rating_approved:
        failures.append(
            _check(
                "USE_CASE_RATING",
                _cite(
                    entry.usecase_id,
                    f"'{entry.usecase_id}' materiality rating is not approved by the control function",
                    entry.owner,
                ),
            )
        )

    for citation in appetite_citations(entry, register.appetite):
        failures.append(_check("USE_CASE_APPETITE", citation))
    residual = residual_citation(entry, register.appetite)
    if residual is not None:
        failures.append(_check("USE_CASE_RESIDUAL", residual))

    if entry.status == "pilot":
        if not entry.pilot_users:
            failures.append(
                _check(
                    "USE_CASE_PILOT_USERS",
                    _cite(
                        entry.usecase_id,
                        f"pilot '{entry.usecase_id}' has no named user list",
                        entry.owner,
                    ),
                )
            )
        elif envelope.principal not in set(entry.pilot_users):
            failures.append(
                _check(
                    "USE_CASE_PILOT_USERS",
                    _cite(
                        entry.usecase_id,
                        f"pilot '{entry.usecase_id}' denies {envelope.principal or '(none)'}, "
                        "who is not on the user list",
                        entry.owner,
                    ),
                )
            )
        if entry.pilot_end is None:
            failures.append(
                _check(
                    "USE_CASE_PILOT_END",
                    _cite(
                        entry.usecase_id,
                        f"pilot '{entry.usecase_id}' has no end date",
                        entry.owner,
                    ),
                )
            )
        elif _aware(at) > entry.pilot_end:
            failures.append(
                _check(
                    "USE_CASE_PILOT_END",
                    _cite(
                        entry.usecase_id,
                        f"pilot '{entry.usecase_id}' is past its end date",
                        entry.owner,
                    ),
                )
            )

    if failures:
        return failures
    return [
        _check(
            "USE_CASE",
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' v{entry.version} · tier {entry.proposed_tier} · "
                f"accountable owner: {entry.business_owner}",
            ),
            passed=True,
        )
    ]


def _cite_owner_if_needed(citation: str, entry: RegisteredUseCase) -> str:
    if entry.owner and "accountable owner:" not in citation:
        return f"{citation} · accountable owner: {entry.owner}"
    return citation


def _raise(citations: Sequence[str]) -> None:
    if citations:
        raise KognitaError("\n".join(citations))


def _agents(session: Session) -> frozenset[str]:
    return frozenset(agent.name for agent in session.exec(select(Agent)).all())


def _current_usecase(session: Session, usecase_id: str, at: datetime) -> UseCase | None:
    rows = [
        row
        for row in session.exec(
            select(UseCase).where(UseCase.usecase_id == usecase_id)
        ).all()
        if row.is_effective(at)
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: row.version)


def _current_appetite(session: Session, at: datetime) -> RiskAppetite | None:
    rows = [
        row for row in session.exec(select(RiskAppetite)).all() if row.is_effective(at)
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: row.version)


def _view(row: UseCase) -> RegisteredUseCase:
    return from_record(
        row.record, content_hash=row.content_hash or usecase_content_hash(row)
    )


def load_register(
    session: Session, *, as_of: datetime | None = None
) -> UseCaseRegister:
    """The use cases and appetite effective at ``as_of``."""
    at = as_of or utcnow()
    by_id: dict[str, UseCase] = {}
    for row in session.exec(select(UseCase)).all():
        if not row.is_effective(at):
            continue
        current = by_id.get(row.usecase_id)
        if current is None or row.version > current.version:
            by_id[row.usecase_id] = row
    appetite_row = _current_appetite(session, at)
    appetite = (
        Appetite.from_record(appetite_row.record) if appetite_row is not None else None
    )
    entries = tuple(
        _view(row) for row in sorted(by_id.values(), key=lambda item: item.usecase_id)
    )
    return UseCaseRegister(entries=entries, appetite=appetite, agents=_agents(session))


def _emit(
    session: Session,
    evidence: EvidenceWriter,
    *,
    correlation_id: str,
    actor_id: str,
    action: str,
    at: datetime,
    body: dict[str, Any],
) -> None:
    evidence.emit(
        session,
        correlation_id=correlation_id,
        event_type=EventType.POLICY_CHANGE,
        actor_type=ActorType.HUMAN,
        actor_id=actor_id,
        payload={"action": action, "at": _aware(at).isoformat(), **body},
    )


def _supersede(
    session: Session,
    current: UseCase | None,
    *,
    usecase_id: str,
    version: int,
    record: dict[str, Any],
    at: datetime,
) -> UseCase:
    if current is not None:
        current.effective_to = at
        session.add(current)
    row = UseCase(
        usecase_id=usecase_id,
        version=version,
        record=record,
        effective_from=at,
    )
    session.add(row)
    session.flush()
    return row


def appetite_vocabulary(appetite: Appetite) -> tuple[str, ...]:
    found: list[str] = []
    known = set(CATEGORIES)
    for category in (*appetite.allowed_categories, *appetite.prohibited_categories):
        if category not in known:
            found.append(
                f"Risk appetite — category '{category}' is not a recognised use-case category"
            )
    overlap = sorted(
        set(appetite.allowed_categories) & set(appetite.prohibited_categories)
    )
    for category in overlap:
        found.append(
            f"Risk appetite — category '{category}' is both allowed and prohibited"
        )
    for impact in appetite.permitted_client_impact:
        if impact not in CLIENT_IMPACT:
            found.append(f"Risk appetite — client impact '{impact}' is not recognised")
    if appetite.maximum_residual_materiality not in MATERIALITY:
        found.append(
            "Risk appetite — maximum residual materiality "
            f"'{appetite.maximum_residual_materiality}' is not recognised"
        )
    if not 0 <= appetite.maximum_autonomy <= 6:
        found.append(
            f"Risk appetite — maximum autonomy L{appetite.maximum_autonomy} is outside L0 to L6"
        )
    return tuple(found)


def set_appetite(
    session: Session,
    appetite: Appetite,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    senior_management: bool,
    at: datetime | None = None,
) -> RiskAppetite:
    """Put a risk appetite in force. Refused unless senior management makes the change."""
    when = _aware(at or utcnow())
    if not senior_management:
        raise KognitaError(
            "Risk appetite — changing the appetite is a governed change at senior-management level"
        )
    _raise(appetite_vocabulary(appetite))
    current = _current_appetite(session, when)
    version = 1 if current is None else current.version + 1
    record = Appetite(
        allowed_categories=appetite.allowed_categories,
        prohibited_categories=appetite.prohibited_categories,
        maximum_autonomy=appetite.maximum_autonomy,
        permitted_client_impact=appetite.permitted_client_impact,
        external_communication=appetite.external_communication,
        transaction_authority=appetite.transaction_authority,
        maximum_residual_materiality=appetite.maximum_residual_materiality,
        version=version,
    ).to_record()
    if current is not None:
        current.effective_to = when
        session.add(current)
    row = RiskAppetite(version=version, record=record, effective_from=when)
    session.add(row)
    session.flush()
    _emit(
        session,
        evidence,
        correlation_id=f"appetite:{version}",
        actor_id=actor_id,
        action="SET_APPETITE",
        at=when,
        body={"version": version, "content_hash": row.content_hash},
    )
    return row


def register(
    session: Session,
    spec: UseCaseSpec,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    at: datetime | None = None,
) -> UseCase:
    """Register a use case. It is not activated. Outside the appetite, it is not stored."""
    when = _aware(at or utcnow())
    if _current_usecase(session, spec.usecase_id, when) is not None:
        raise KognitaError(
            _cite(spec.usecase_id, f"'{spec.usecase_id}' is already registered")
        )
    appetite_row = _current_appetite(session, when)
    appetite = (
        Appetite.from_record(appetite_row.record) if appetite_row is not None else None
    )
    _raise(registration_citations(spec, appetite, _agents(session)))
    record = build_record(spec, version=1)
    row = _supersede(
        session, None, usecase_id=spec.usecase_id, version=1, record=record, at=when
    )
    _emit(
        session,
        evidence,
        correlation_id=f"usecase:{spec.usecase_id}",
        actor_id=actor_id,
        action="REGISTER",
        at=when,
        body={
            "usecase_id": spec.usecase_id,
            "version": 1,
            "content_hash": row.content_hash,
        },
    )
    return row


def _require(session: Session, usecase_id: str, at: datetime) -> UseCase:
    row = _current_usecase(session, usecase_id, at)
    if row is None:
        raise KognitaError(_cite(usecase_id, f"'{usecase_id}' is not registered"))
    return row


def approve_rating(
    session: Session,
    usecase_id: str,
    approver: str,
    *,
    evidence: EvidenceWriter,
    at: datetime | None = None,
) -> UseCase:
    """The designated control function approves the derived tier.

    The approver cannot be the use case's business or technology owner, and
    cannot be anyone other than the designated control function.
    """
    when = _aware(at or utcnow())
    current = _require(session, usecase_id, when)
    entry = _view(current)
    if approver == entry.business_owner or approver == entry.technology_owner:
        raise KognitaError(
            _cite(
                usecase_id,
                "the control function cannot approve a use case it owns",
                entry.owner,
            )
        )
    if approver != entry.control_function:
        raise KognitaError(
            _cite(
                usecase_id,
                "the materiality rating is approved by the designated control function",
                entry.owner,
            )
        )
    record = dict(entry.record)
    record["version"] = current.version + 1
    record["rating_approved"] = True
    record["rating_approved_by"] = approver
    row = _supersede(
        session,
        current,
        usecase_id=usecase_id,
        version=current.version + 1,
        record=record,
        at=when,
    )
    _emit(
        session,
        evidence,
        correlation_id=f"usecase:{usecase_id}",
        actor_id=approver,
        action="APPROVE_RATING",
        at=when,
        body={
            "usecase_id": usecase_id,
            "version": row.version,
            "content_hash": row.content_hash,
            "tier": entry.proposed_tier,
        },
    )
    return row


def _activation_citations(
    entry: RegisteredUseCase,
    *,
    status: str,
    actor_id: str,
    signoff_by: str,
    criteria_met: bool,
    appetite: Appetite | None,
    agents: frozenset[str],
    at: datetime,
) -> tuple[str, ...]:
    found = list(configuration_citations(entry, appetite, agents, at=at))
    # configuration_citations treats a not-yet-activated row as allowed to lack
    # a review date and an approved rating. Activation is stricter.
    found = [
        item
        for item in found
        if "is past its review date" not in item
        and "materiality rating is not approved" not in item
    ]
    if entry.status == "retired":
        found.append(
            _cite(entry.usecase_id, f"'{entry.usecase_id}' is retired", entry.owner)
        )
    if status not in ("pilot", "live"):
        found.append(
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' cannot be activated as '{status}'",
                entry.owner,
            )
        )
    if not entry.rating_approved:
        found.append(
            _cite(
                entry.usecase_id,
                "a materiality rating the control function has not approved cannot be activated",
                entry.owner,
            )
        )
    if status == "pilot":
        if not entry.pilot_users:
            found.append(
                _cite(
                    entry.usecase_id,
                    f"pilot '{entry.usecase_id}' has no named user list",
                    entry.owner,
                )
            )
        if entry.pilot_end is None:
            found.append(
                _cite(
                    entry.usecase_id,
                    f"pilot '{entry.usecase_id}' has no end date",
                    entry.owner,
                )
            )
        if not entry.success_criteria.strip():
            found.append(
                _cite(
                    entry.usecase_id,
                    f"pilot '{entry.usecase_id}' has no success criteria",
                    entry.owner,
                )
            )
    if (
        entry.status == "pilot"
        and status == "live"
        and (
            actor_id != entry.control_function
            or not criteria_met
            or not entry.success_criteria.strip()
        )
    ):
        found.append(
            _cite(
                entry.usecase_id,
                f"moving pilot '{entry.usecase_id}' to live needs the control function's "
                "approval against the recorded success criteria",
                entry.owner,
            )
        )
    if entry.proposed_tier >= 3:
        if not entry.builders:
            found.append(
                _cite(
                    entry.usecase_id,
                    f"a tier {entry.proposed_tier} use case does not name its builders",
                    entry.owner,
                )
            )
        elif not signoff_by.strip():
            found.append(
                _cite(
                    entry.usecase_id,
                    f"a tier {entry.proposed_tier} use case requires sign-off from a function "
                    "independent of its builders",
                    entry.owner,
                )
            )
        elif signoff_by in set(entry.builders) or actor_id in set(entry.builders):
            found.append(
                _cite(
                    entry.usecase_id,
                    f"a tier {entry.proposed_tier} use case cannot be signed off by its builder",
                    entry.owner,
                )
            )
    return tuple(dict.fromkeys(found))


def activate(
    session: Session,
    usecase_id: str,
    *,
    status: str,
    evidence: EvidenceWriter,
    actor_id: str,
    signoff_by: str = "",
    criteria_met: bool = False,
    at: datetime | None = None,
) -> UseCase:
    """Activate a use case as ``pilot`` or ``live``.

    The rating must already be approved. Residual materiality and the rest of
    the appetite are checked again. A tier 3 or 4 use case needs sign-off from
    someone who is not a builder.
    """
    when = _aware(at or utcnow())
    current = _require(session, usecase_id, when)
    entry = _view(current)
    appetite_row = _current_appetite(session, when)
    appetite = (
        Appetite.from_record(appetite_row.record) if appetite_row is not None else None
    )
    _raise(
        _activation_citations(
            entry,
            status=status,
            actor_id=actor_id,
            signoff_by=signoff_by,
            criteria_met=criteria_met,
            appetite=appetite,
            agents=_agents(session),
            at=when,
        )
    )
    deadline = review_deadline(entry.proposed_tier, when)
    if (
        entry.review_date is not None
        and entry.status in ("pilot", "live")
        and entry.review_date < deadline
    ):
        # A caller cannot stretch the tier's review period. An earlier date stands.
        deadline = entry.review_date if entry.review_date > when else deadline
    record = dict(entry.record)
    record["version"] = current.version + 1
    record["status"] = status
    record["review_date"] = _iso(deadline)
    record["validation_signoff_by"] = signoff_by or entry.validation_signoff_by
    record["criteria_met"] = criteria_met or entry.criteria_met
    row = _supersede(
        session,
        current,
        usecase_id=usecase_id,
        version=current.version + 1,
        record=record,
        at=when,
    )
    _emit(
        session,
        evidence,
        correlation_id=f"usecase:{usecase_id}",
        actor_id=actor_id,
        action="ACTIVATE",
        at=when,
        body={
            "usecase_id": usecase_id,
            "version": row.version,
            "status": status,
            "content_hash": row.content_hash,
        },
    )
    return row


_MATERIAL_FIELDS = (
    "autonomy",
    "maximum_autonomy",
    "jurisdiction",
    "client_population",
    "intended_use",
    "purpose",
    "category",
)


def revise(
    session: Session,
    spec: UseCaseSpec,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    at: datetime | None = None,
) -> UseCase:
    """Write a new version. A change of autonomy, jurisdiction, client population,
    or intended use clears activation until the use case is approved again.

    Purpose and category are the same kind of change: they are what the use case
    is for. A live use case cannot be revised below its tier's minimum controls.
    """
    when = _aware(at or utcnow())
    current = _require(session, spec.usecase_id, when)
    entry = _view(current)
    appetite_row = _current_appetite(session, when)
    appetite = (
        Appetite.from_record(appetite_row.record) if appetite_row is not None else None
    )
    _raise(registration_citations(spec, appetite, _agents(session)))
    proposed = from_record(build_record(spec, version=current.version + 1))
    material = any(
        getattr(proposed, name) != getattr(entry, name) for name in _MATERIAL_FIELDS
    )
    if material:
        status = ""
        rating_approved = False
        rating_approved_by = ""
        signoff = ""
        criteria_met = False
        review_date = when
    else:
        status = entry.status
        rating_approved = entry.rating_approved
        rating_approved_by = entry.rating_approved_by
        signoff = entry.validation_signoff_by
        criteria_met = entry.criteria_met
        review_date = entry.review_date
        if status in ("pilot", "live"):
            candidate = from_record(
                build_record(
                    spec,
                    version=current.version + 1,
                    status=status,
                    rating_approved=rating_approved,
                    rating_approved_by=rating_approved_by,
                    validation_signoff_by=signoff,
                    criteria_met=criteria_met,
                    review_date=review_date,
                )
            )
            _raise(
                configuration_citations(candidate, appetite, _agents(session), at=when)
            )
    record = build_record(
        spec,
        version=current.version + 1,
        status=status,
        rating_approved=rating_approved,
        rating_approved_by=rating_approved_by,
        validation_signoff_by=signoff,
        criteria_met=criteria_met,
        review_date=review_date,
    )
    row = _supersede(
        session,
        current,
        usecase_id=spec.usecase_id,
        version=current.version + 1,
        record=record,
        at=when,
    )
    _emit(
        session,
        evidence,
        correlation_id=f"usecase:{spec.usecase_id}",
        actor_id=actor_id,
        action="REVISE",
        at=when,
        body={
            "usecase_id": spec.usecase_id,
            "version": row.version,
            "content_hash": row.content_hash,
            "material": material,
        },
    )
    return row


def _set_status(
    session: Session,
    usecase_id: str,
    status: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    action: str,
    at: datetime | None,
) -> UseCase:
    when = _aware(at or utcnow())
    current = _require(session, usecase_id, when)
    entry = _view(current)
    record = dict(entry.record)
    record["version"] = current.version + 1
    record["status"] = status
    row = _supersede(
        session,
        current,
        usecase_id=usecase_id,
        version=current.version + 1,
        record=record,
        at=when,
    )
    _emit(
        session,
        evidence,
        correlation_id=f"usecase:{usecase_id}",
        actor_id=actor_id,
        action=action,
        at=when,
        body={
            "usecase_id": usecase_id,
            "version": row.version,
            "status": status,
            "content_hash": row.content_hash,
        },
    )
    return row


def suspend(
    session: Session,
    usecase_id: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    at: datetime | None = None,
) -> UseCase:
    """Mark a use case suspended. The next decision is a denial."""
    return _set_status(
        session,
        usecase_id,
        "suspended",
        evidence=evidence,
        actor_id=actor_id,
        action="SUSPEND",
        at=at,
    )


def retire(
    session: Session,
    usecase_id: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    at: datetime | None = None,
) -> UseCase:
    """Mark a use case retired. The next decision is a denial.

    This is the status only. It does not engage an agent's kill switch, apply
    the retention period, or notify owners of dependent use cases.
    """
    return _set_status(
        session,
        usecase_id,
        "retired",
        evidence=evidence,
        actor_id=actor_id,
        action="RETIRE",
        at=at,
    )


def list_usecases(
    session: Session, *, as_of: datetime | None = None
) -> tuple[RegisteredUseCase, ...]:
    """Every use case effective at ``as_of``, in id order."""
    return load_register(session, as_of=as_of).entries


def show_usecase(
    session: Session, usecase_id: str, *, as_of: datetime | None = None
) -> RegisteredUseCase | None:
    return load_register(session, as_of=as_of).get(usecase_id)


def validate_usecase(
    session: Session,
    usecase_id: str | None = None,
    *,
    as_of: datetime | None = None,
) -> tuple[str, ...]:
    """Citations for one use case, or for every use case when ``usecase_id`` is omitted."""
    when = _aware(as_of or utcnow())
    register = load_register(session, as_of=when)
    if usecase_id is None:
        entries = register.entries
    else:
        entry = register.get(usecase_id)
        if entry is None:
            return (_cite(usecase_id, f"'{usecase_id}' is not registered"),)
        entries = (entry,)
    found: list[str] = []
    for entry in entries:
        found.extend(
            configuration_citations(entry, register.appetite, register.agents, at=when)
        )
    return tuple(found)


__all__ = [
    "ACCOUNTABILITY",
    "CATEGORIES",
    "Appetite",
    "RegisteredUseCase",
    "UseCaseRegister",
    "UseCaseSpec",
    "activate",
    "approve_rating",
    "build_record",
    "derived_tier",
    "from_record",
    "list_usecases",
    "load_register",
    "register",
    "retire",
    "revise",
    "set_appetite",
    "show_usecase",
    "suspend",
    "usecase_checks",
    "validate_usecase",
]
