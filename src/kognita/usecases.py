"""The use-case register.

Every execution path passes the register, including an empty one. The
envelope's purpose must name a use case in it, and the check cites that use
case's accountable owner. An unregistered, retired, suspended, or
not-yet-activated use case is a denial. This replaces the free-text purpose
list on ``run_governed``, ``ask``, the MCP proxy and the gateway. ``decide``
still uses the purpose list when the caller does not pass a register.

The envelope's ``purpose`` is the use case id. That is already how a use case
is named on a decision; the register's own purpose field is the statement of
what the use case is for.

An approved model is a model card (provider, name, version, intended use,
known limitations, the evaluation the firm relied on, approval date and
approver). A call that uses one links that card from its model evidence.

An unmediated entry is AI that never passes through Kognita. It records the
assurance gap and the compensating control. A decision that cites it is a
denial. The gateways flag a call whose AI endpoint no registered use case
names; the flag does not decide the call.

Quantitative appetite thresholds cover AI incidents (count and impact), how
many material use cases depend on one model provider, and how many material
use cases breach their primary-metric target. Material means residual
materiality high or critical, or tier 3 or 4. A live use case with a primary
metric and no measurement counts as a performance breach and is reported as
unmeasured; a pilot does not. An incident with no use case counts in every
jurisdiction filter. A breach is reported on the senior-management report
and, when that report is written with an evidence writer, escalated to the
accountable owner.

Retiring a use case engages the kill switch of every agent registered only to
it, writes its retention period and erases content already past that period,
and notifies the owners of use cases that depend on it. A later decision on
a dependent use case is a denial while the dependency stays retired.

Autonomy levels appear here only as the ceiling a tier's minimum controls
allow (L0 retrieve through L6 broad autonomy). Enforcing a level on a call is
not this register.

Tier floors and review periods are constants in this module. A tier may not
run with a weaker approval gate, a lower sampling rate, a higher autonomy
ceiling, or a shallower explanation than its floor, and it may not run past
its review period.

- Tier 1: approval gate ``none``, sampling 5%, autonomy ceiling L2,
  explanation ``citation``, review every 365 days.
- Tier 2: approval gate ``human_review``, sampling 10%, autonomy ceiling L3,
  explanation ``sources``, review every 180 days.
- Tier 3: approval gate ``human_approval``, sampling 50%, autonomy ceiling L4,
  explanation ``suitability``, review every 90 days.
- Tier 4: approval gate ``two_signature``, sampling 100%, autonomy ceiling L4,
  explanation ``full``, review every 90 days.

Monitoring is required at every tier. On the review date the use case is
denied. Activation after that date waits until the designated control
function approves the materiality rating again, at or after the review date.
A change of purpose or category clears activation, as does a change of
autonomy, jurisdiction, client population, or intended use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from sqlmodel import Session, select

from kognita.envelope import Check, Envelope
from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.models import (
    Agent,
    EvidenceEvent,
    ModelCard,
    RiskAppetite,
    UseCase,
    as_utc,
    in_force,
    model_card_content_hash,
    usecase_content_hash,
    utcnow,
)
from kognita.registry import get as get_agent
from kognita.registry import set_kill_switch
from kognita.retention import RetentionStore
from kognita.vocabulary import ActorType, CheckResult, EventType, StaleAction

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

#: The one primary metric a use case may name. Direction is fixed by the name:
#: cycle time breaches above its target; decision quality and reliability
#: breach below theirs. The outcome-metric report computes these from the
#: interaction record. A use case with no interaction still compares the
#: latest PRIMARY_METRIC event. A live use case with an interaction compares
#: the computed value, and a missing measurement is a breach. A pilot with
#: no measurement stays exempt from the appetite count.
PRIMARY_METRICS: tuple[str, ...] = (
    "cycle time",
    "decision quality",
    "reliability",
)

#: Residual materiality at or above this rank is material, as is tier 3 or 4.
#: Material use cases feed the single-provider count, the performance-breach
#: count, and the senior-management report.
MATERIAL_RESIDUAL: frozenset[str] = frozenset({"high", "critical"})
MATERIAL_TIER: int = 3

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

#: How long a tier may run before its review date. These periods are constants:
#: tier 1 is 365 days, tier 2 is 180, and tiers 3 and 4 are 90. The date is the
#: deadline. On that date the use case is denied. ``activate`` does not clear
#: that denial by itself. The designated control function approves the rating
#: again at or after the deadline, and that later approval is what activation
#: requires.
REVIEW_DAYS = {1: 365, 2: 180, 3: 90, 4: 90}

#: Minimum controls a tier requires. These floors are constants. A weaker gate,
#: a lower sampling rate, a higher autonomy ceiling, missing monitoring, a
#: shallower explanation, or a missing validator fails validation. The autonomy
#: integers are the ceiling only: L0 is 0 and L6 is 6. The table is tier 1
#: none / 0.05 / L2 / citation, tier 2 human_review / 0.10 / L3 / sources,
#: tier 3 human_approval / 0.50 / L4 / suitability, tier 4 two_signature / 1.0
#: / L4 / full. Monitoring is required at every tier.
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


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _names(values: Sequence[str]) -> list[str]:
    return sorted({str(item).strip() for item in values if str(item).strip()})


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
    stale_action: StaleAction = StaleAction.SUPPRESS
    deployment_mode: str | None = None
    maximum_autonomy: int = 0
    _autonomy_gates: Mapping[str, Any] = field(default_factory=dict)
    _autonomy_step_down: Mapping[str, Any] = field(default_factory=dict)
    monitoring: bool = False
    explanation_depth: str = "citation"
    validators: tuple[str, ...] = ()
    human_decision_points: tuple[str, ...] = ()
    client_segments: tuple[str, ...] = ()
    investor_impact: str = ""
    permitted_data_classes: tuple[str, ...] = ()
    approved_models: tuple[str, ...] = ()
    approved_sources: tuple[str, ...] = ()
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
    model_providers: tuple[str, ...] = ()
    ai_endpoints: tuple[str, ...] = ()
    unmediated: bool = False
    assurance_gap: str = ""
    compensating_control: str = ""
    agents: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    primary_metric: str = ""
    baseline: float | None = None
    target: float | None = None


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
    incident_count_threshold: int | None = None
    incident_impact_threshold: float | None = None
    single_provider_threshold: int | None = None
    performance_breach_threshold: int | None = None

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
            "incident_count_threshold": self.incident_count_threshold,
            "incident_impact_threshold": self.incident_impact_threshold,
            "single_provider_threshold": self.single_provider_threshold,
            "performance_breach_threshold": self.performance_breach_threshold,
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
            incident_count_threshold=_optional_int(
                record.get("incident_count_threshold")
            ),
            incident_impact_threshold=_optional_number(
                record.get("incident_impact_threshold")
            ),
            single_provider_threshold=_optional_int(
                record.get("single_provider_threshold")
            ),
            performance_breach_threshold=_optional_int(
                record.get("performance_breach_threshold")
            ),
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
    rating_approved_at: datetime | None
    validation_signoff_by: str
    inherent_materiality: str
    credited_controls: tuple[str, ...]
    residual_materiality: str
    approval_gate: str
    review_sampling_rate: float
    deployment_mode: str | None
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
    stale_action: StaleAction = StaleAction.SUPPRESS
    content_hash: str = ""
    approved_models: tuple[str, ...] = ()
    approved_sources: tuple[str, ...] = ()
    model_providers: tuple[str, ...] = ()
    ai_endpoints: tuple[str, ...] = ()
    unmediated: bool = False
    assurance_gap: str = ""
    compensating_control: str = ""
    agents: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    retention_days: int | None = None
    primary_metric: str = ""
    baseline: float | None = None
    target: float | None = None

    @property
    def owner(self) -> str:
        return self.business_owner


@dataclass(frozen=True)
class RegisteredModelCard:
    """The model card a call links, and the register lists."""

    card_id: str
    provider: str
    model: str
    version: str
    intended_use: str
    limitations: str
    evaluation_results: str
    approver: str
    approved_at: datetime | None
    registry_version: int
    record: dict[str, Any]
    content_hash: str = ""


@dataclass(frozen=True)
class UseCaseRegister:
    """The use cases and appetite in force at one instant. Pure input to decide."""

    entries: tuple[RegisteredUseCase, ...] = ()
    appetite: Appetite | None = None
    agents: frozenset[str] = frozenset()
    model_cards: tuple[RegisteredModelCard, ...] = ()

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

    if spec.stale_action not in tuple(StaleAction):
        add("FACT_STALE_ACTION_INVALID: stale_action must be REFRESH, SUPPRESS or WARN")
    if spec.deployment_mode not in (None, "gateway", "library"):
        add("S2-DEPLOYMENT-MODE: deployment_mode must be gateway or library")
    if spec.deployment_mode == "library" and derived_tier(spec) >= 2:
        add("S2-DEPLOYMENT-MODE: tier 2 and above require gateway mode")
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
    def _validate_gate(gate):
        if not isinstance(gate, Mapping):
            add("S1-AUTONOMY: metric gate must be a mapping")
            return
        for key in ("minimum_sample", "window_days"):
            if type(gate.get(key)) is not int or gate[key] <= 0:
                add(f"S1-AUTONOMY: {key} must be a positive integer")
        for key in ("quality", "reliability", "exception_rate"):
            value = gate.get(key, .05 if key == "exception_rate" else None)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                add(f"S1-AUTONOMY: {key} must be a threshold between 0 and 1")

    if not isinstance(spec._autonomy_gates, Mapping):
        add("S1-AUTONOMY: transition gates must be a mapping")
    else:
        for level, gate in spec._autonomy_gates.items():
            if level not in {str(value) for value in range(1, 7)}:
                add("S1-AUTONOMY: transition must name a destination level 1 to 6")
            _validate_gate(gate)
    if spec._autonomy_step_down:
        _validate_gate(spec._autonomy_step_down)
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
    metric = spec.primary_metric.strip()
    if metric or spec.baseline is not None or spec.target is not None:
        if metric not in PRIMARY_METRICS:
            add(
                f"primary metric '{metric or 'missing'}' is not cycle time, "
                "decision quality, or reliability"
            )
        if spec.baseline is None or spec.target is None:
            add(f"'{usecase_id}' primary metric needs a baseline and a target")
        elif _optional_number(spec.baseline) is None or _optional_number(spec.target) is None:
            add(f"'{usecase_id}' primary metric baseline and target must be numbers")
    if spec.retention_days is not None and (
        isinstance(spec.retention_days, bool)
        or not isinstance(spec.retention_days, int)
        or spec.retention_days < 0
    ):
        add(f"'{usecase_id}' retention period must be a non-negative number of days")
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
    rating_approved_at: datetime | None = None,
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
        "rating_approved_at": _iso(rating_approved_at),
        "validation_signoff_by": validation_signoff_by,
        "inherent_materiality": spec.inherent_materiality,
        "credited_controls": _sorted_texts(spec.credited_controls),
        "residual_materiality": spec.residual_materiality,
        "approval_gate": spec.approval_gate,
        "review_sampling_rate": float(spec.review_sampling_rate),
        "maximum_autonomy": int(spec.maximum_autonomy),
        "stale_action": StaleAction(spec.stale_action).value,
        "deployment_mode": spec.deployment_mode,
        "_autonomy_gates": dict(spec._autonomy_gates),
        "_autonomy_step_down": dict(spec._autonomy_step_down),
        "monitoring": bool(spec.monitoring),
        "explanation_depth": spec.explanation_depth,
        "validators": _sorted_texts(spec.validators),
        "human_decision_points": _sorted_texts(spec.human_decision_points),
        "client_segments": _sorted_texts(spec.client_segments),
        "investor_impact": spec.investor_impact,
        "permitted_data_classes": _sorted_texts(spec.permitted_data_classes),
        "approved_models": _names(spec.approved_models),
        "approved_sources": _names(spec.approved_sources),
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
        "model_providers": _names(spec.model_providers),
        "ai_endpoints": _names(spec.ai_endpoints),
        "unmediated": bool(spec.unmediated),
        "assurance_gap": spec.assurance_gap.strip(),
        "compensating_control": spec.compensating_control.strip(),
        "agents": _names(spec.agents),
        "depends_on": _names(spec.depends_on),
        "primary_metric": spec.primary_metric.strip(),
        "baseline": spec.baseline,
        "target": spec.target,
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
        rating_approved_at=_parse_time(record.get("rating_approved_at")),
        validation_signoff_by=str(record.get("validation_signoff_by") or ""),
        inherent_materiality=str(record.get("inherent_materiality") or ""),
        credited_controls=tuple(record.get("credited_controls") or ()),
        residual_materiality=str(record.get("residual_materiality") or ""),
        approval_gate=str(record.get("approval_gate") or ""),
        review_sampling_rate=float(record.get("review_sampling_rate") or 0),
        stale_action=StaleAction(record.get("stale_action", "SUPPRESS")),
        deployment_mode=record.get("deployment_mode"),
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
        approved_models=tuple(record.get("approved_models") or ()),
        approved_sources=tuple(record.get("approved_sources") or ()),
        model_providers=tuple(record.get("model_providers") or ()),
        ai_endpoints=tuple(record.get("ai_endpoints") or ()),
        unmediated=bool(record.get("unmediated")),
        assurance_gap=str(record.get("assurance_gap") or ""),
        compensating_control=str(record.get("compensating_control") or ""),
        agents=tuple(record.get("agents") or ()),
        depends_on=tuple(record.get("depends_on") or ()),
        retention_days=_optional_int(record.get("retention_days")),
        primary_metric=str(record.get("primary_metric") or ""),
        baseline=_optional_number(record.get("baseline")),
        target=_optional_number(record.get("target")),
    )


def registration_citations(
    spec: UseCaseSpec,
    register: UseCaseRegister,
) -> tuple[str, ...]:
    """Refusals that stop a use case being registered at all."""
    found = list(vocabulary_citations(spec))
    if found:
        return tuple(found)
    entry = from_record(build_record(spec, version=1))
    appetite = register.appetite
    agents = register.agents
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
    found.extend(_completion_citations(entry, register))
    return tuple(found)


def configuration_citations(
    entry: RegisteredUseCase,
    register: UseCaseRegister,
    *,
    at: datetime,
) -> tuple[str, ...]:
    """What ``usecase validate`` reports. Lifecycle status is not itself a failure."""
    appetite = register.appetite
    agents = register.agents
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
    found.extend(_completion_citations(entry, register))
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


def _rating_activation_citation(entry: RegisteredUseCase, at: datetime) -> str | None:
    """The rating rule ``activate`` enforces.

    Before the review date, an approval by the control function is enough.
    Once that date has been reached, an earlier approval is stale. Activation
    waits until the control function approves the rating again at or after
    the review date. A missing timestamp on a closed window fails closed.
    """
    review_date = entry.review_date
    missed = review_date is not None and _aware(at) >= review_date
    if missed:
        approved_at = entry.rating_approved_at
        if (
            entry.rating_approved
            and approved_at is not None
            and approved_at >= review_date
        ):
            return None
        return _cite(
            entry.usecase_id,
            "a missed review date needs the control function to approve "
            "the materiality rating again",
            entry.owner,
        )
    if entry.rating_approved:
        return None
    return _cite(
        entry.usecase_id,
        "a materiality rating the control function has not approved cannot be activated",
        entry.owner,
    )


def _check(name: str, citation: str, *, passed: bool = False) -> Check:
    return Check(
        check=name,
        regime="INTERNAL",
        result=CheckResult.PASS if passed else CheckResult.FAIL,
        citation=citation,
    )


def _library_checks(envelope: Envelope, register: UseCaseRegister) -> list[Check]:
    """Enforce the tier's ingress requirement at library execution time."""
    entry = register.get(envelope.purpose)
    if (entry is not None and entry.proposed_tier >= 2
            and not envelope.context.get("_governed_ingress")):
        return [_check("S2-DEPLOYMENT-MODE", "S2-DEPLOYMENT-MODE")]
    return []


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
                    f"'{usecase_id or '(none)'}' is not registered · "
                    "no accountable owner can be cited",
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

    for citation in _completion_citations(entry, register):
        failures.append(_check("USE_CASE_REGISTER", citation))
    if entry.unmediated:
        failures.append(
            _check(
                "USE_CASE_UNMEDIATED",
                _cite(
                    entry.usecase_id,
                    f"'{entry.usecase_id}' is unmediated and does not authorise a call",
                    entry.owner,
                ),
            )
        )
    for dependency in entry.depends_on:
        other = register.get(dependency)
        if other is None or other.status == "retired":
            state = "retired" if other is not None else "not registered"
            failures.append(
                _check(
                    "USE_CASE_DEPENDENCY",
                    _cite(
                        entry.usecase_id,
                        f"'{entry.usecase_id}' depends on '{dependency}', which is {state}",
                        entry.owner,
                    ),
                )
            )
    model_failure = _model_call_citation(entry, envelope, register)
    if model_failure is not None:
        failures.append(_check("USE_CASE_MODEL", model_failure))

    if failures:
        return failures
    card_note = ""
    if envelope.subject_type == "model":
        matched = _matching_cards(entry, envelope, register)
        if len(matched) == 1:
            card_note = f" · model card: {matched[0].card_id}"
    return [
        _check(
            "USE_CASE",
            _cite(
                entry.usecase_id,
                f"'{entry.usecase_id}' v{entry.version} · tier {entry.proposed_tier} · "
                f"accountable owner: {entry.business_owner}",
            )
            + card_note,
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
    rows = in_force(session, RiskAppetite, at)
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
    for row in in_force(session, UseCase, at):
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
    return UseCaseRegister(
        entries=entries,
        appetite=appetite,
        agents=_agents(session),
        model_cards=_model_cards(session, at),
    )


def _governing_register(
    session: Session, *, as_of: datetime | None = None
) -> UseCaseRegister:
    """The register that governs an execution path.

    An empty register is still the register. The purpose must be a registered
    use case, and an empty register denies rather than falling back to the
    deployment's purpose list.
    """
    return load_register(session, as_of=as_of)


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
        stale_action=StaleAction(record.get("stale_action", "SUPPRESS")),
        deployment_mode=record.get("deployment_mode"),
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
    counts = (
        ("incident count", appetite.incident_count_threshold),
        ("single-provider dependencies", appetite.single_provider_threshold),
        ("performance breaches", appetite.performance_breach_threshold),
    )
    for label, value in counts:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            found.append(
                f"Risk appetite — {label} threshold must be a non-negative integer"
            )
    impact = appetite.incident_impact_threshold
    if impact is not None and (
        isinstance(impact, bool) or not isinstance(impact, (int, float)) or impact < 0
    ):
        found.append(
            "Risk appetite — AI incident impact threshold must be a non-negative number"
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
        incident_count_threshold=appetite.incident_count_threshold,
        incident_impact_threshold=appetite.incident_impact_threshold,
        single_provider_threshold=appetite.single_provider_threshold,
        performance_breach_threshold=appetite.performance_breach_threshold,
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
    register = load_register(session, as_of=when)
    _raise(registration_citations(spec, register))
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
    record["rating_approved_at"] = _iso(when)
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
    register: UseCaseRegister,
    *,
    status: str,
    actor_id: str,
    signoff_by: str,
    criteria_met: bool,
    at: datetime,
) -> tuple[str, ...]:
    found = list(configuration_citations(entry, register, at=at))
    # A closed review window is reported on a row that is already pilot or
    # live. Activation is what opens the next window, so that finding is not
    # itself a refusal. The rating check below is the refusal: once the
    # review date has been reached, the control function must approve again.
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
    rating = _rating_activation_citation(entry, at)
    if rating is not None:
        found.append(rating)
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

    The rating must already be approved. Once the review date has been
    reached, that approval is stale: the designated control function has to
    approve the rating again at or after the review date before activation
    can proceed. Residual materiality and the rest of the appetite are
    checked again. A tier 3 or 4 use case needs sign-off from someone who is
    not a builder.
    """
    when = _aware(at or utcnow())
    current = _require(session, usecase_id, when)
    entry = _view(current)
    register = load_register(session, as_of=when)
    _raise(
        _activation_citations(
            entry,
            register,
            status=status,
            actor_id=actor_id,
            signoff_by=signoff_by,
            criteria_met=criteria_met,
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
    "deployment_mode",
    "stale_action",
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
    """Write a new version.

    A change of autonomy, maximum autonomy, jurisdiction, client population,
    intended use, purpose, or category clears activation. The rating approval
    goes with it, including the time it was given, until the control function
    approves again. A live use case cannot be revised below its tier's minimum
    controls.
    """
    when = _aware(at or utcnow())
    current = _require(session, spec.usecase_id, when)
    entry = _view(current)
    register = load_register(session, as_of=when)
    _raise(registration_citations(spec, register))
    proposed = from_record(build_record(spec, version=current.version + 1))
    material = any(
        getattr(proposed, name) != getattr(entry, name) for name in _MATERIAL_FIELDS
    )
    material = material or any(entry.record.get(key, {}) != proposed.record.get(key, {})
                               for key in ("_autonomy_gates", "_autonomy_step_down"))
    if material:
        status = ""
        rating_approved = False
        rating_approved_by = ""
        rating_approved_at = None
        signoff = ""
        criteria_met = False
        review_date = when
    else:
        status = entry.status
        rating_approved = entry.rating_approved
        rating_approved_by = entry.rating_approved_by
        rating_approved_at = entry.rating_approved_at
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
                    rating_approved_at=rating_approved_at,
                    validation_signoff_by=signoff,
                    criteria_met=criteria_met,
                    review_date=review_date,
                )
            )
            _raise(configuration_citations(candidate, register, at=when))
    record = build_record(
        spec,
        version=current.version + 1,
        status=status,
        rating_approved=rating_approved,
        rating_approved_by=rating_approved_by,
        rating_approved_at=rating_approved_at,
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
    """Retire a use case completely.

    The next decision is a denial. Every agent registered only to this use
    case has its kill switch engaged. The use case's retention period is
    written and content already past that period is erased. Owners of use
    cases that depend on it are notified; their next decision is a denial
    while this use case stays retired.
    """
    row = _set_status(
        session,
        usecase_id,
        "retired",
        evidence=evidence,
        actor_id=actor_id,
        action="RETIRE",
        at=at,
    )
    when = _aware(at or utcnow())
    _retire_agents(session, usecase_id, evidence=evidence, actor_id=actor_id, at=when)
    _apply_retention(
        session, usecase_id, evidence=evidence, actor_id=actor_id, at=when
    )
    _notify_dependents(
        session, usecase_id, evidence=evidence, actor_id=actor_id, at=when
    )
    return row


def is_material(entry: RegisteredUseCase) -> bool:
    """Residual materiality high or critical, or tier 3 or 4."""
    return (
        entry.residual_materiality in MATERIAL_RESIDUAL
        or entry.proposed_tier >= MATERIAL_TIER
    )


def _card_index(register: UseCaseRegister) -> dict[str, RegisteredModelCard]:
    return {card.card_id: card for card in register.model_cards}


def _completion_citations(
    entry: RegisteredUseCase, register: UseCaseRegister
) -> tuple[str, ...]:
    """Cards, providers, agents, dependencies, and the unmediated record."""
    found: list[str] = []
    owner = entry.owner
    usecase_id = entry.usecase_id
    if entry.unmediated:
        if not entry.assurance_gap.strip():
            found.append(
                _cite(
                    usecase_id,
                    f"unmediated '{usecase_id}' records no assurance gap",
                    owner,
                )
            )
        if not entry.compensating_control.strip():
            found.append(
                _cite(
                    usecase_id,
                    f"unmediated '{usecase_id}' records no compensating control",
                    owner,
                )
            )
    cards = _card_index(register)
    for card_id in entry.approved_models:
        if card_id not in cards:
            found.append(
                _cite(
                    usecase_id,
                    f"approved model '{card_id}' has no model card",
                    owner,
                )
            )
    if is_material(entry) and not entry.model_providers:
        found.append(
            _cite(
                usecase_id,
                f"material use case '{usecase_id}' records no model provider",
                owner,
            )
        )
    for name in entry.agents:
        if name not in register.agents:
            found.append(
                _cite(usecase_id, f"agent '{name}' is not registered", owner)
            )
    known = {item.usecase_id: item for item in register.entries}
    for dependency in entry.depends_on:
        if dependency == usecase_id:
            found.append(_cite(usecase_id, f"'{usecase_id}' depends on itself", owner))
            continue
        other = known.get(dependency)
        if other is None:
            found.append(
                _cite(
                    usecase_id,
                    f"'{usecase_id}' depends on '{dependency}', which is not registered",
                    owner,
                )
            )
        elif other.status == "retired":
            found.append(
                _cite(
                    usecase_id,
                    f"'{usecase_id}' depends on '{dependency}', which is retired",
                    owner,
                )
            )
    if entry.primary_metric or entry.baseline is not None or entry.target is not None:
        if entry.primary_metric not in PRIMARY_METRICS:
            found.append(
                _cite(
                    usecase_id,
                    f"primary metric '{entry.primary_metric or 'missing'}' is not "
                    "cycle time, decision quality, or reliability",
                    owner,
                )
            )
        elif entry.baseline is None or entry.target is None:
            found.append(
                _cite(
                    usecase_id,
                    f"'{usecase_id}' primary metric needs a baseline and a target",
                    owner,
                )
            )
    return tuple(found)


def _matching_cards(
    entry: RegisteredUseCase, envelope: Envelope, register: UseCaseRegister
) -> tuple[RegisteredModelCard, ...]:
    requested = (envelope.subject_id or "").strip()
    version = str(envelope.arguments.get("model_version") or "").strip()
    approved = set(entry.approved_models)
    found: list[RegisteredModelCard] = []
    for card in register.model_cards:
        if card.card_id not in approved:
            continue
        pinned = requested in {card.card_id, f"{card.model}@{card.version}"}
        named = requested == card.model and (not version or version == card.version)
        if pinned or named:
            found.append(card)
    return tuple(found)


def _model_call_citation(
    entry: RegisteredUseCase, envelope: Envelope, register: UseCaseRegister
) -> str | None:
    if envelope.subject_type != "model":
        return None
    requested = (envelope.subject_id or "").strip()
    if not requested:
        return _cite(entry.usecase_id, "a model call names no model", entry.owner)
    matches = _matching_cards(entry, envelope, register)
    if len(matches) == 1:
        return None
    if not matches:
        return _cite(
            entry.usecase_id,
            f"model '{requested}' has no model card approved for '{entry.usecase_id}'",
            entry.owner,
        )
    return _cite(
        entry.usecase_id,
        f"model '{requested}' is not pinned to one model card",
        entry.owner,
    )


def _served_model_citation(
    entry: RegisteredUseCase, *, model: str, version: str
) -> str:
    """Why a served version was withheld. The stored decision is not rewritten."""
    shown = f"{model}@{version}" if model and version else (model or version or "unknown")
    return _cite(
        entry.usecase_id,
        f"served model '{shown}' has no model card approved for '{entry.usecase_id}'",
        entry.owner,
    )


def model_card_id(provider: str, model: str, version: str) -> str:
    """``provider/model@version``."""
    return f"{provider}/{model}@{version}"


def _component(label: str, value: str) -> str:
    text = value.strip()
    if not text or "/" in text or "@" in text:
        raise KognitaError(
            f"Model card — {label} must be a single token without '/' or '@'"
        )
    return text


def _model_cards(session: Session, at: datetime) -> tuple[RegisteredModelCard, ...]:
    by_id: dict[str, ModelCard] = {}
    for row in in_force(session, ModelCard, at):
        current = by_id.get(row.card_id)
        if current is None or row.version > current.version:
            by_id[row.card_id] = row
    return tuple(
        _view_card(row) for row in sorted(by_id.values(), key=lambda item: item.card_id)
    )


def _view_card(row: ModelCard) -> RegisteredModelCard:
    record = dict(row.record or {})
    return RegisteredModelCard(
        card_id=row.card_id,
        provider=str(record.get("provider") or ""),
        model=str(record.get("model") or ""),
        version=str(record.get("version") or ""),
        intended_use=str(record.get("intended_use") or ""),
        limitations=str(record.get("limitations") or ""),
        evaluation_results=str(record.get("evaluation_results") or ""),
        approver=str(record.get("approver") or ""),
        approved_at=_parse_time(record.get("approved_at")),
        registry_version=row.version,
        record=record,
        content_hash=row.content_hash or model_card_content_hash(row),
    )


def register_model_card(
    session: Session,
    *,
    provider: str,
    model: str,
    version: str,
    intended_use: str,
    limitations: str,
    evaluation_results: str,
    approver: str,
    approved_at: datetime,
    evidence: EvidenceWriter,
    actor_id: str = "",
    at: datetime | None = None,
) -> ModelCard:
    """Record a model card. A later approval of the same model version is a new row."""
    when = _aware(at or utcnow())
    approved = _aware(approved_at)
    provider_name = _component("provider", provider)
    model_name = _component("model", model)
    model_version = _component("version", version)
    intended = intended_use.strip()
    limits = limitations.strip()
    evaluation = evaluation_results.strip()
    approver_name = approver.strip()
    if not intended or not limits or not evaluation or not approver_name:
        raise KognitaError(
            "Model card — intended use, known limitations, evaluation results "
            "and approver are required"
        )
    card_id = model_card_id(provider_name, model_name, model_version)
    current = None
    for row in session.exec(select(ModelCard).where(ModelCard.card_id == card_id)).all():
        if not row.is_effective(when):
            continue
        if current is None or row.version > current.version:
            current = row
    registry_version = 1 if current is None else current.version + 1
    record = {
        "card_id": card_id,
        "provider": provider_name,
        "model": model_name,
        "version": model_version,
        "intended_use": intended,
        "limitations": limits,
        "evaluation_results": evaluation,
        "approver": approver_name,
        "approved_at": _iso(approved),
    }
    if current is not None:
        current.effective_to = when
        session.add(current)
    row = ModelCard(
        card_id=card_id,
        version=registry_version,
        record=record,
        effective_from=when,
    )
    session.add(row)
    session.flush()
    _emit(
        session,
        evidence,
        correlation_id=f"model-card:{card_id}",
        actor_id=actor_id or approver_name,
        action="MODEL_CARD",
        at=when,
        body={
            "card_id": card_id,
            "version": registry_version,
            "content_hash": row.content_hash,
        },
    )
    return row


def _endpoint_key(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or "").lower().strip("[]")
    if not host:
        return raw.lower().rstrip("/")
    port = f":{parsed.port}" if parsed.port else ""
    scheme = parsed.scheme.lower()
    if scheme:
        return f"{scheme}://{host}{port}"
    return f"{host}{port}"


def endpoint_named(register: UseCaseRegister, destination: str) -> bool:
    """True when a non-retired use case names this AI endpoint."""
    dest_key = _endpoint_key(destination)
    dest_full = destination.strip().rstrip("/").lower()
    if not dest_key and not dest_full:
        return False
    for entry in register.entries:
        if entry.status == "retired":
            continue
        for named in entry.ai_endpoints:
            key = _endpoint_key(named)
            full = named.strip().rstrip("/").lower()
            if dest_key and key and dest_key == key:
                return True
            if full and (dest_full == full or dest_full.startswith(full + "/")):
                return True
    return False


def resolve_model_card(
    register: UseCaseRegister,
    *,
    provider: str,
    model: str,
    version: str,
    use_case: str,
) -> RegisteredModelCard | None:
    """The approved card for this provider, model and version, if there is one."""
    provider_name = provider.strip()
    model_name = model.strip()
    model_version = version.strip()
    if not model_name or not model_version:
        return None
    cards = register.model_cards
    entry = register.get(use_case) if use_case else None
    if entry is not None:
        approved = set(entry.approved_models)
        cards = tuple(card for card in cards if card.card_id in approved)
    for card in cards:
        if (
            card.provider == provider_name
            and card.model == model_name
            and card.version == model_version
        ):
            return card
    return None


def model_card_link(
    session: Session,
    *,
    provider: str | None,
    model: str | None,
    version: str | None,
    use_case: str,
    at: datetime | None = None,
    register: UseCaseRegister | None = None,
) -> tuple[str, str]:
    """Card id and content hash for a call, or two empty strings.

    ``register`` is the register already loaded for this decision. Omitting
    it reads the register at ``at``.
    """
    if register is None:
        register = load_register(session, as_of=at)
    card = resolve_model_card(
        register,
        provider=provider or "",
        model=model or "",
        version=version or "",
        use_case=use_case,
    )
    if card is None:
        return "", ""
    return card.card_id, card.content_hash


def model_call_block(
    session: Session,
    *,
    use_case: str,
    provider: str | None,
    model: str | None,
    version: str | None,
    at: datetime | None = None,
) -> str | None:
    """Why this model call cannot be sent, or None when no register entry applies."""
    if not use_case:
        return None
    register = load_register(session, as_of=at)
    entry = register.get(use_case)
    if entry is None:
        return None
    if entry.unmediated:
        return _cite(
            entry.usecase_id,
            f"'{entry.usecase_id}' is unmediated and does not authorise a call",
            entry.owner,
        )
    if not entry.approved_models:
        return _cite(
            entry.usecase_id,
            f"'{entry.usecase_id}' has no approved model",
            entry.owner,
        )
    card_id, _hash = model_card_link(
        session,
        provider=provider,
        model=model,
        version=version,
        use_case=use_case,
        at=at,
    )
    if not card_id:
        return _cite(
            entry.usecase_id,
            "the model call has no model card approved for "
            f"'{entry.usecase_id}'",
            entry.owner,
        )
    return None


def _retire_agents(
    session: Session,
    usecase_id: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    at: datetime,
) -> None:
    register = load_register(session, as_of=at)
    entry = register.get(usecase_id)
    if entry is None:
        return
    others = [
        other
        for other in register.entries
        if other.usecase_id != usecase_id and other.status != "retired"
    ]
    for name in entry.agents:
        also = [other.usecase_id for other in others if name in other.agents]
        if also:
            _emit(
                session,
                evidence,
                correlation_id=f"usecase:{usecase_id}",
                actor_id=actor_id,
                action="AGENT_KEPT",
                at=at,
                body={"usecase_id": usecase_id, "agent": name, "also_registered_to": also},
            )
            continue
        if get_agent(session, name) is None:
            _emit(
                session,
                evidence,
                correlation_id=f"usecase:{usecase_id}",
                actor_id=actor_id,
                action="AGENT_NOT_REGISTERED",
                at=at,
                body={"usecase_id": usecase_id, "agent": name},
            )
            continue
        set_kill_switch(
            session,
            name,
            True,
            evidence=evidence,
            actor_id=actor_id,
            correlation_id=f"usecase:{usecase_id}",
            reason=f"use case {usecase_id} retired",
            now=at,
        )
        from kognita.credentials import revoke
        revoke(session, name, None, evidence=evidence, actor_id=actor_id, _automatic=True)


def _apply_retention(
    session: Session,
    usecase_id: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    at: datetime,
) -> None:
    register = load_register(session, as_of=at)
    entry = register.get(usecase_id)
    retain_days = entry.retention_days if entry is not None else None
    RetentionStore().set_policy(session, usecase_id, retain_days=retain_days)
    erased = RetentionStore().enforce(
        session, evidence=evidence, now=at, actor_id=actor_id
    )
    _emit(
        session,
        evidence,
        correlation_id=f"usecase:{usecase_id}",
        actor_id=actor_id,
        action="RETENTION_APPLIED",
        at=at,
        body={
            "usecase_id": usecase_id,
            "retain_days": retain_days,
            "erased": erased,
        },
    )


def _notify_dependents(
    session: Session,
    usecase_id: str,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    at: datetime,
) -> None:
    register = load_register(session, as_of=at)
    for other in register.entries:
        if other.usecase_id == usecase_id or usecase_id not in other.depends_on:
            continue
        _emit(
            session,
            evidence,
            correlation_id=f"usecase:{usecase_id}",
            actor_id=actor_id,
            action="DEPENDENT_NOTIFIED",
            at=at,
            body={
                "usecase_id": usecase_id,
                "dependent": other.usecase_id,
                "accountable_owner": other.owner,
            },
        )


def _metric_breaches(metric: str, value: float, target: float) -> bool:
    if metric == "cycle time":
        return value > target
    return value < target


def _events_until(session: Session, at: datetime) -> list[EvidenceEvent]:
    rows = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    kept: list[EvidenceEvent] = []
    for row in rows:
        recorded = as_utc(row.recorded_at)
        if recorded is not None and recorded <= at:
            kept.append(row)
    return kept


def record_incident(
    session: Session,
    usecase_id: str = "",
    *,
    impact: float,
    summary: str,
    evidence: EvidenceWriter,
    actor_id: str,
    at: datetime | None = None,
) -> None:
    """Record one AI incident.

    A named use case must be registered. An empty use case is an incident that
    is not yet assigned; it still counts in every jurisdiction filter.
    """
    when = _aware(at or utcnow())
    number = _optional_number(impact)
    if number is None or number < 0:
        raise KognitaError("AI incident — impact must be a non-negative number")
    if not summary.strip():
        raise KognitaError("AI incident — a summary is required")
    named = usecase_id.strip()
    owner = ""
    if named:
        entry = load_register(session, as_of=when).get(named)
        if entry is None:
            raise KognitaError(_cite(named, f"'{named}' is not registered"))
        owner = entry.owner
    _emit(
        session,
        evidence,
        correlation_id=f"usecase:{named}" if named else "incident",
        actor_id=actor_id,
        action="INCIDENT",
        at=when,
        body={
            "usecase_id": named,
            "impact": number,
            "summary": summary.strip(),
            "accountable_owner": owner,
        },
    )


def record_primary_metric(
    session: Session,
    usecase_id: str,
    value: float,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    at: datetime | None = None,
) -> None:
    """Record the use case's primary metric. The target on the event is the one compared."""
    when = _aware(at or utcnow())
    number = _optional_number(value)
    if number is None:
        raise KognitaError("Primary metric — the value must be a number")
    entry = load_register(session, as_of=when).get(usecase_id)
    if entry is None:
        raise KognitaError(_cite(usecase_id, f"'{usecase_id}' is not registered"))
    if (
        entry.primary_metric not in PRIMARY_METRICS
        or entry.baseline is None
        or entry.target is None
    ):
        raise KognitaError(
            _cite(
                usecase_id,
                f"'{usecase_id}' has no primary metric with a baseline and a target",
                entry.owner,
            )
        )
    _emit(
        session,
        evidence,
        correlation_id=f"usecase:{usecase_id}",
        actor_id=actor_id,
        action="PRIMARY_METRIC",
        at=when,
        body={
            "usecase_id": usecase_id,
            "primary_metric": entry.primary_metric,
            "baseline": entry.baseline,
            "target": entry.target,
            "value": number,
            "version": entry.version,
            "content_hash": entry.content_hash,
        },
    )


@dataclass(frozen=True)
class AppetiteMeasure:
    """One quantitative appetite measure against its threshold."""

    measure: str
    value: float
    threshold: float | None
    breached: bool
    accountable_owners: tuple[str, ...]
    usecase_ids: tuple[str, ...]
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "value": self.value,
            "threshold": self.threshold,
            "breached": self.breached,
            "accountable_owners": list(self.accountable_owners),
            "usecase_ids": list(self.usecase_ids),
            "detail": self.detail,
        }


def _over(value: float, threshold: float | None) -> bool:
    return threshold is not None and value > float(threshold)


def quantitative_measures(
    session: Session,
    *,
    as_of: datetime | None = None,
    jurisdiction: str | None = None,
) -> tuple[AppetiteMeasure, ...]:
    """Incident, single-provider and performance measures from the register and evidence."""
    when = _aware(as_of or utcnow())
    register = load_register(session, as_of=when)
    appetite = register.appetite
    events = _events_until(session, when)

    def in_scope(usecase_id: str) -> bool:
        if not jurisdiction:
            return True
        entry = register.get(usecase_id)
        return entry is not None and entry.jurisdiction == jurisdiction

    def incident_in_report(event: EvidenceEvent) -> bool:
        if str((event.payload or {}).get("action") or "") != "INCIDENT":
            return False
        usecase_id = str((event.payload or {}).get("usecase_id") or "").strip()
        # An incident that names no use case counts in every filtered report.
        if not usecase_id:
            return True
        return in_scope(usecase_id)

    incidents = [event for event in events if incident_in_report(event)]
    impact = 0.0
    owners: list[str] = []
    identifiers: list[str] = []
    for event in incidents:
        payload = event.payload or {}
        number = _optional_number(payload.get("impact"))
        if number is not None:
            impact += number
        usecase_id = str(payload.get("usecase_id") or "")
        entry = register.get(usecase_id)
        owner = entry.owner if entry is not None else str(payload.get("accountable_owner") or "")
        if usecase_id:
            identifiers.append(usecase_id)
        if owner:
            owners.append(owner)
    owner_names = tuple(sorted(set(owners)))
    usecase_ids = tuple(sorted(set(identifiers)))
    count_threshold = appetite.incident_count_threshold if appetite else None
    impact_threshold = appetite.incident_impact_threshold if appetite else None

    material = [
        entry
        for entry in register.entries
        if is_material(entry)
        and entry.status in ("pilot", "live")
        and (not jurisdiction or entry.jurisdiction == jurisdiction)
    ]
    by_provider: dict[str, list[RegisteredUseCase]] = {}
    for entry in material:
        for provider in entry.model_providers:
            by_provider.setdefault(provider, []).append(entry)
    if by_provider:
        top = max(by_provider, key=lambda name: (len(by_provider[name]), name))
        provider_value = float(len(by_provider[top]))
        provider_detail = top
    else:
        top = ""
        provider_value = 0.0
        provider_detail = ""
    provider_threshold = appetite.single_provider_threshold if appetite else None
    provider_owners: list[str] = []
    provider_ids: list[str] = []
    groups = by_provider.items()
    if provider_threshold is None:
        groups = ((top, by_provider.get(top, [])),) if top else ()
    for provider, group in groups:
        if provider_threshold is not None and len(group) <= provider_threshold:
            continue
        for entry in group:
            provider_owners.append(entry.owner)
            provider_ids.append(entry.usecase_id)

    latest: dict[str, EvidenceEvent] = {}
    for event in events:
        if str((event.payload or {}).get("action") or "") != "PRIMARY_METRIC":
            continue
        usecase_id = str((event.payload or {}).get("usecase_id") or "")
        if in_scope(usecase_id):
            latest[usecase_id] = event
    # Imported here: the outcome report reads the register, and the register
    # report reads the outcome report.
    from kognita.outcomes import interaction_measurement

    breaching: list[RegisteredUseCase] = []
    saw_unmeasured = False
    for entry in material:
        has_interaction, computed, measured = interaction_measurement(
            session, entry, when
        )
        if has_interaction and _has_primary_metric(entry):
            # The interaction record replaces a self-reported value. A live
            # use case with no computed measurement is a breach. A pilot is
            # still exempt until it is live.
            if not measured:
                if entry.status == "live":
                    breaching.append(entry)
                    saw_unmeasured = True
                continue
            if (
                computed is not None
                and entry.target is not None
                and _metric_breaches(entry.primary_metric, computed, float(entry.target))
            ):
                breaching.append(entry)
            continue
        event = latest.get(entry.usecase_id)
        if event is None:
            # A live use case with a primary metric and no measurement is a
            # breach. A pilot is exempt until it is live.
            if entry.status == "live" and _has_primary_metric(entry):
                breaching.append(entry)
                saw_unmeasured = True
            continue
        payload = event.payload or {}
        metric = str(payload.get("primary_metric") or "")
        value = _optional_number(payload.get("value"))
        target = _optional_number(payload.get("target"))
        if value is None or target is None or metric not in PRIMARY_METRICS:
            continue
        if _metric_breaches(metric, value, target):
            breaching.append(entry)
    performance_threshold = appetite.performance_breach_threshold if appetite else None
    performance_detail = "unmeasured" if saw_unmeasured else ""

    return (
        AppetiteMeasure(
            "AI incidents",
            float(len(incidents)),
            None if count_threshold is None else float(count_threshold),
            _over(float(len(incidents)), None if count_threshold is None else float(count_threshold)),
            owner_names,
            usecase_ids,
        ),
        AppetiteMeasure(
            "AI incident impact",
            impact,
            impact_threshold,
            _over(impact, impact_threshold),
            owner_names,
            usecase_ids,
        ),
        AppetiteMeasure(
            "single-provider dependencies",
            provider_value,
            None if provider_threshold is None else float(provider_threshold),
            _over(
                provider_value,
                None if provider_threshold is None else float(provider_threshold),
            ),
            tuple(sorted(set(provider_owners))),
            tuple(sorted(set(provider_ids))),
            provider_detail,
        ),
        AppetiteMeasure(
            "performance breaches",
            float(len(breaching)),
            None if performance_threshold is None else float(performance_threshold),
            _over(
                float(len(breaching)),
                None if performance_threshold is None else float(performance_threshold),
            ),
            tuple(sorted({entry.owner for entry in breaching})),
            tuple(entry.usecase_id for entry in breaching),
            performance_detail,
        ),
    )


def _has_primary_metric(entry: RegisteredUseCase) -> bool:
    return (
        entry.primary_metric in PRIMARY_METRICS
        and entry.baseline is not None
        and entry.target is not None
    )


def senior_management_report(
    session: Session,
    *,
    as_of: datetime | None = None,
    jurisdiction: str | None = None,
    evidence: EvidenceWriter | None = None,
    actor_id: str = "",
) -> dict[str, Any]:
    """Material use cases and quantitative measures.

    Pass ``evidence`` to record the report and escalate each breach to the
    accountable owners it names. ``jurisdiction`` limits the report to use
    cases in that jurisdiction.
    """
    when = _aware(as_of or utcnow())
    if evidence is not None and not actor_id.strip():
        raise KognitaError(
            "Senior-management report — publishing it names who published it"
        )
    register = load_register(session, as_of=when)
    measures = quantitative_measures(session, as_of=when, jurisdiction=jurisdiction)
    systems: list[dict[str, Any]] = []
    for entry in register.entries:
        if not is_material(entry):
            continue
        if jurisdiction and entry.jurisdiction != jurisdiction:
            continue
        systems.append(
            {
                "usecase_id": entry.usecase_id,
                "tier": entry.proposed_tier,
                "category": entry.category,
                "residual_materiality": entry.residual_materiality,
                "inherent_materiality": entry.inherent_materiality,
                "owner": entry.owner,
                "status": entry.status or "not activated",
                "jurisdiction": entry.jurisdiction,
                "model_providers": list(entry.model_providers),
                "unmediated": entry.unmediated,
                "open_findings": list(
                    configuration_citations(entry, register, at=when)
                ),
            }
        )
    body: dict[str, Any] = {
        "at": when.isoformat(),
        "jurisdiction": jurisdiction or "",
        "material_use_cases": systems,
        "measures": [measure.to_dict() for measure in measures],
    }
    if evidence is None:
        return body
    _emit(
        session,
        evidence,
        correlation_id="senior-management-report",
        actor_id=actor_id,
        action="SENIOR_MANAGEMENT_REPORT",
        at=when,
        body={
            "jurisdiction": jurisdiction or "",
            "measures": [
                {
                    "measure": measure.measure,
                    "value": measure.value,
                    "threshold": measure.threshold,
                    "breached": measure.breached,
                }
                for measure in measures
            ],
        },
    )
    for measure in measures:
        if not measure.breached:
            continue
        named = measure.accountable_owners or ("",)
        for owner in named:
            _emit(
                session,
                evidence,
                correlation_id="senior-management-report",
                actor_id=actor_id,
                action="APPETITE_BREACH",
                at=when,
                body={
                    "measure": measure.measure,
                    "value": measure.value,
                    "threshold": measure.threshold,
                    "accountable_owner": owner,
                    "usecase_ids": list(measure.usecase_ids),
                },
            )
    return body


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
        found.extend(configuration_citations(entry, register, at=when))
    return tuple(found)


__all__ = [
    "ACCOUNTABILITY",
    "CATEGORIES",
    "PRIMARY_METRICS",
    "Appetite",
    "AppetiteMeasure",
    "RegisteredModelCard",
    "RegisteredUseCase",
    "UseCaseRegister",
    "UseCaseSpec",
    "activate",
    "approve_rating",
    "build_record",
    "derived_tier",
    "endpoint_named",
    "from_record",
    "is_material",
    "list_usecases",
    "load_register",
    "model_call_block",
    "model_card_id",
    "model_card_link",
    "quantitative_measures",
    "record_incident",
    "record_primary_metric",
    "register",
    "register_model_card",
    "retire",
    "revise",
    "senior_management_report",
    "set_appetite",
    "show_usecase",
    "suspend",
    "usecase_checks",
    "validate_usecase",
]
