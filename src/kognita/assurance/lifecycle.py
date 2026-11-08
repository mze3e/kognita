"""Owner approvals, software provenance, production feedback and quarantine."""
from collections import defaultdict
from datetime import timedelta
import re

from pydantic import Field
from sqlmodel import select

from kognita.assurance.records import (
    Record, agent_for, binding, emit, events, latest, owner, policy,
)
from kognita.canonical import canonical_hash
from kognita.models import Agent, Interaction, as_utc, utcnow
from kognita.vocabulary import ActorType, EventType


class Component(Record):
    """Approved package provenance and optional container digest."""
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sbom_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    image_digest: str | None = Field(default=None, pattern=r'^sha256:[a-f0-9]{64}$')
    provenance: str = Field(min_length=1)


def approve_components(session, name, components, *, evidence, actor_id):
    """Approve a unique component set for the current deployment and policy."""
    agent = owner(session, name, actor_id)
    p = policy(session, name)
    parsed = [Component.model_validate(c).model_dump(mode='json') for c in components]
    if (not agent.deployment_hash or not parsed or
            len({c['name'] for c in parsed}) != len(parsed)):
        raise ValueError('A9: approved deployment and unique nonempty component set required')
    parsed.sort(key=lambda c: c['name'])
    body = {'binding': binding(agent, p), 'components': parsed, 'digest': canonical_hash(parsed)}
    emit(session, evidence, name, 'COMPONENTS', body, actor_id=actor_id)
    return body


def advisory(session, name, version, reference, *, evidence, actor_id):
    """An authenticated operator quarantines every approved matching deployment.

    Any registered workload identity is refused. Deployment administration must
    authenticate operator identity before invoking this API, as for policy edits.
    Advisories are append-only and cannot be cleared by the affected workload.
    """
    if (not all(isinstance(v, str) and v.strip() for v in (name, version, reference, actor_id))
            or session.exec(select(Agent).where(Agent.name == actor_id)).first()):
        raise ValueError('A9: authenticated operator and advisory reference required')
    emit(session, evidence, '*', 'ADVISORY', {'name': name, 'version': version,
          'reference': reference}, actor_id=actor_id)
    affected = []
    grouped = defaultdict(list)
    for row in events(session):
        grouped[row.correlation_id].append(row)
    for agent in session.exec(select(Agent)).all():
        rows = grouped[f'assurance:{agent.name}']
        record = latest(rows, 'COMPONENTS')
        if record and any(c['name'] == name and c['version'] == version for c in record['components']):
            affected.append({'agent': agent.name, 'deployment': record['binding']['deployment'],
                             'use_cases': agent.use_cases or []})
            from kognita._actions import _stop
            _stop(agent=agent.name)
    return affected


def approve_update(session, name, *, candidate_digest, provider_change, regression_reference,
                   rollback_target, rollback_plan, evidence, actor_id):
    """Record owner approval of an exact candidate and its rollback packet."""
    agent = owner(session, name, actor_id)
    p = policy(session, name)
    values = (provider_change, regression_reference, rollback_target, rollback_plan)
    if (not re.fullmatch('[a-f0-9]{64}', candidate_digest) or
            not all(isinstance(v, str) and v.strip() for v in values)):
        raise ValueError('A4-UPDATE: exact candidate, change, regression and rollback required')
    body = {'candidate_digest': candidate_digest, 'provider_change': provider_change,
            'regression_reference': regression_reference, 'rollback_target': rollback_target,
            'rollback_plan': rollback_plan, 'approver': actor_id,
            'previous_deployment': agent.deployment_hash,
            'policy': canonical_hash(p.model_dump(mode='json'))}
    emit(session, evidence, name, 'UPDATE', body, actor_id=actor_id)
    return body


class ProductionOutcome(Record):
    """Authoritative downstream facts bound to a unique bank outcome."""
    interaction_id: str = Field(min_length=1)
    intended: str = Field(min_length=1)
    actual: str = Field(min_length=1)
    client_impact: str = Field(min_length=1)
    complaint: str = ''
    correction: str = ''
    reversal: str = ''
    human_override: str = ''
    escalation: str = ''
    adverse: bool
    source_reference: str = Field(min_length=1)
    outcome_id: str = Field(min_length=1)


def _flush_outcome_state(session):
    """Preserve pending interaction edits without overwriting committed conflicts."""
    from sqlalchemy import inspect
    for interaction in tuple(session.dirty):
        if not isinstance(interaction, Interaction):
            continue
        state = inspect(interaction)
        changes = {column.key: state.attrs[column.key].history
                   for column in state.mapper.columns
                   if state.attrs[column.key].history.has_changes()}
        if not changes:
            continue
        # Lock fresh scalar values without replacing the caller's pending edits.
        with session.no_autoflush:
            stored = session.execute(select(*Interaction.__table__.columns)
                .where(Interaction.id == state.identity[0]).with_for_update()).mappings().one_or_none()
        if stored is None or any(
                getattr(interaction, key) != stored[key] and
                (not history.deleted or history.deleted[0] != stored[key])
                for key, history in changes.items()):
            raise ValueError('A3: conflicting pending interaction edit')
    session.flush()


def record_outcome(session, name, outcome, *, evidence, actor_id, now=None):
    """Store bank-supplied facts; no inferred success, even for a completed run."""
    # Use the append lock before reading any governing state, not only at emit.
    # Callers own commit/rollback, so competing outcomes cannot pass these reads.
    if session.get_bind().dialect.name == 'postgresql':
        from kognita.evidence import _serialize_postgres_writes
        _serialize_postgres_writes(session)
    elif session.get_bind().dialect.name == 'sqlite':
        from sqlalchemy import text
        # Reserve the single SQLite writer before reading the policy/history.
        session.execute(text('UPDATE agents SET name = name WHERE name = :name'), {'name': name})
    else:
        raise ValueError('A3: outcome serialization requires SQLite or PostgreSQL')
    # Preserve pending caller changes before populate_existing, even without autoflush.
    _flush_outcome_state(session)
    at = as_utc(now or utcnow())
    agent = owner(session, name, actor_id)
    p = policy(session, name)
    result = ProductionOutcome.model_validate(outcome)
    interaction = session.get(Interaction, result.interaction_id, populate_existing=True)
    if interaction is None or interaction.use_case != p.use_case:
        raise ValueError('A3: outcome must link to an interaction of the governed use case')
    rows = events(session, name)
    if any(r.payload.get('action') == 'PRODUCTION_OUTCOME' and
           r.payload['body']['outcome']['outcome_id'] == result.outcome_id for r in rows):
        raise ValueError('A3: duplicate authoritative outcome')
    body = {'outcome': result.model_dump(mode='json'), 'binding': binding(agent, p),
            'version': agent.version, 'model': agent.model}
    recent = [r for r in rows if r.payload.get('action') == 'PRODUCTION_OUTCOME'
              and as_utc(at - timedelta(days=p.outcome_window_days)) <=
                  as_utc(type(at).fromisoformat(r.payload['at'])) <= at
              and r.payload['body']['version'] == agent.version
              and r.payload['body']['outcome']['adverse']]
    adverse_count = len(recent) + int(result.adverse)
    breached = result.adverse and adverse_count >= p.max_adverse_outcomes
    with session.begin_nested():
        emit(session, evidence, name, 'PRODUCTION_OUTCOME', body, actor_id=actor_id, at=at)
        if breached:
            emit(session, evidence, name, 'FINDING', {'control': 'A3', 'use_case': p.use_case,
                 'adverse_count': adverse_count, 'outcome_id': result.outcome_id},
                 actor_id='kognita', at=at, system=True)
            trigger = evidence.emit(session, correlation_id=f'assurance:{name}',
                event_type=EventType.REVALIDATION_TRIGGERED, actor_type=ActorType.SYSTEM,
                actor_id='kognita', payload={'agent': name, 'use_case': p.use_case,
                    'reason': 'A3-PRODUCTION-OUTCOME', 'version': agent.version})
            if trigger is None:
                raise ValueError('A3: revalidation evidence write incomplete')
            if p.outcome_step_down and agent.autonomy_level:
                from kognita.autonomy import _change
                _change(session, agent, max(0, agent.autonomy_level - 1), evidence,
                        'kognita', 'AUTONOMY_STEP_DOWN', {'reason': 'A3-PRODUCTION-OUTCOME'})
    return {'breached': breached, 'adverse_count': adverse_count}


def assurance_report(session, name, *, now=None):
    """Report current evaluation beside evidenced outcomes and findings."""
    from kognita.evaluation import _latest_applicable
    agent = agent_for(session, name)
    at = as_utc(now or utcnow())
    rows = events(session, name, as_of=at)
    evaluation = _latest_applicable(session, agent)
    if evaluation and evaluation.get('deployment_hash') not in (None, agent.deployment_hash):
        evaluation = None
    return {'agent': name, 'as_of': at.isoformat(),
            'pre_deployment_evaluation': evaluation,
            'production_outcomes': [r.payload['body'] for r in rows
                                    if r.payload.get('action') == 'PRODUCTION_OUTCOME'],
            'findings': [r.payload['body'] for r in rows if r.payload.get('action') == 'FINDING']}
