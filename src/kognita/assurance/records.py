"""0.8 records. The administration boundary authenticates actor_id.

Callers own the transaction. Policies and attestations live in the existing
append-only evidence chain, so backup, signing and independent verification
apply without another mutable register or a database migration.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlmodel import select

from kognita.canonical import canonical_hash
from kognita.models import Agent, EvidenceEvent, as_utc, utcnow
from kognita.vocabulary import ActorType, EventType


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class AssurancePolicy(Record):
    use_case: str = Field(min_length=1)
    audience: str = Field(min_length=1)
    # Keys are supplied by an authenticated owner, never discovered in a token.
    issuers: dict[str, str] = Field(default_factory=dict)
    attestors: dict[str, str] = Field(default_factory=dict)
    external_required: bool = False
    privacy_required: bool = False
    environment_required: bool = False
    monitoring_required: bool = False
    components_required: bool = False
    transfer_required: bool = False
    update_required: bool = False
    instruction_tools: list[str] = Field(default_factory=list)
    jurisdictions: list[str] = Field(default_factory=list)
    data_classes: list[str] = Field(default_factory=list)
    client_impact: str = Field(default='internal', min_length=1)
    credential_expiry: datetime | None = None
    max_adverse_outcomes: int = Field(default=1, ge=1, strict=True)
    outcome_window_days: int = Field(default=30, ge=1, le=3650, strict=True)
    outcome_step_down: bool = False
    technology_origin: str = ''
    deployment_jurisdiction: str = ''
    transfer_type: Literal['api', 'weights', 'licence', 'local', 'ownership'] = 'api'
    transfer_policy: str = ''

    @model_validator(mode='after')
    def independent(self):
        if not self.use_case.strip() or not self.audience.strip():
            raise ValueError('A0: use case and audience required')
        if self.transfer_required and not all((self.technology_origin.strip(),
                self.deployment_jurisdiction.strip(), self.transfer_policy.strip())):
            raise ValueError('A10: transfer classification and policy reference required')
        return self


def agent_for(session, name):
    agent = session.exec(select(Agent).where(Agent.name == name)
                         .execution_options(populate_existing=True)).first()
    if agent is None:
        raise ValueError('A0: unknown agent')
    return agent


def owner(session, name, actor_id):
    agent = agent_for(session, name)
    if not actor_id or actor_id != agent.owner_exec or session.exec(
            select(Agent).where(Agent.name == actor_id)).first() is not None:
        raise ValueError('A0: authenticated accountable human owner required')
    return agent


def emit(session, evidence, name, action, body, *, actor_id, at=None, system=False):
    event = evidence.emit(session, correlation_id=f'assurance:{name}',
        event_type=EventType.ASSURANCE, actor_id=actor_id,
        actor_type=ActorType.SYSTEM if system else ActorType.HUMAN,
        payload={'action': action, 'agent': name, 'body': body,
                 'at': as_utc(at or utcnow()).isoformat()})
    if event is None:
        raise ValueError('A0: assurance evidence write incomplete')
    return event


def events(session, name=None, *, as_of=None):
    query = select(EvidenceEvent).where(EvidenceEvent.event_type == EventType.ASSURANCE)
    if name is not None:
        query = query.where(EvidenceEvent.correlation_id == f'assurance:{name}')
    if as_of is not None:
        query = query.where(EvidenceEvent.recorded_at <= as_utc(as_of))
    rows = session.exec(query.order_by(EvidenceEvent.sequence)).all()
    if rows:
        from kognita.evidence import _verify_chain
        _verify_chain(session)
    return rows


def latest(rows, action):
    return next((r.payload['body'] for r in reversed(rows)
                 if r.payload.get('action') == action), None)


def policy_from(rows):
    body = latest(rows, 'POLICY')
    return AssurancePolicy.model_validate(body) if body else None


def configure(session, name, policy: AssurancePolicy, *, evidence, actor_id):
    agent = owner(session, name, actor_id)
    policy = AssurancePolicy.model_validate(policy)
    if policy.use_case not in (agent.use_cases or []):
        raise ValueError('A0: policy must name a linked use case')
    # Agents cannot be independent attestors, including another fleet member.
    names = set(session.exec(select(Agent.name)).all())
    if names & (set(policy.attestors) | set(policy.issuers)):
        raise ValueError('A0: external authorities cannot be registered workloads')
    from kognita.signing import _crypto
    keys = [*policy.attestors.values(), *policy.issuers.values()]
    if keys:
        serialization, _, public = _crypto()
    for pem in [*policy.attestors.values(), *policy.issuers.values()]:
        key = serialization.load_pem_public_key(pem.encode())
        if not isinstance(key, public):
            raise ValueError('A0: Ed25519 public key required')
    events(session, name)
    emit(session, evidence, name, 'POLICY', policy.model_dump(mode='json'), actor_id=actor_id)
    return policy


def policy(session, name):
    p = policy_from(events(session, name))
    if p is None:
        raise ValueError('A0: no assurance policy')
    return p


def privacy_context(agent, p, use_case_record):
    deployment = (agent.tool_allowlist or {}).get('_deployment') or {}
    # Approval/status changes are not privacy changes. These named dimensions are.
    return {'purpose': agent.purpose, 'use_case_purpose': use_case_record.get('purpose'),
        'tools': agent.tool_allowlist, 'systems': agent.permitted_systems,
        'data_classes': p.data_classes, 'data_sources': use_case_record.get('data_sources'),
        'memory': deployment.get('memory_behaviour'), 'model': agent.model,
        'providers': use_case_record.get('model_providers'),
        'approved_models': use_case_record.get('approved_models'),
        'approved_sources': use_case_record.get('approved_sources'),
        'deployment_jurisdiction': p.deployment_jurisdiction,
        'jurisdictions': p.jurisdictions, 'autonomy': agent.autonomy_level,
        'client_impact': p.client_impact, 'principal': agent.principal}


def context(session, agent, p):
    from kognita.usecases import load_register
    entry = load_register(session).get(p.use_case)
    return privacy_context(agent, p, entry.record if entry else {})


def binding(agent, p):
    return {'deployment': agent.deployment_hash, 'policy': canonical_hash(p.model_dump(mode='json'))}
