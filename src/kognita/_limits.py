"""Private blast-radius accounting and purpose-bound result projection."""
from datetime import timedelta
from math import isfinite

from sqlmodel import select

from kognita.models import Agent, BlastRadius, _BlastUsage, as_utc
from kognita.envelope import Check
from kognita.vocabulary import CheckResult, Outcome, EventType, ActorType


def _validate(agent):
    from kognita._isolation import _validate
    _validate(agent)
    if agent.blast_radius is not None:
        # JSON timestamps are decoded separately to keep numeric fields strict.
        _radius(agent.blast_radius)
    fields = agent.field_allowlist
    if fields is not None and (not isinstance(fields, dict) or any(
        not isinstance(k, str) or not isinstance(v, list) or any(
            not isinstance(x, str) or not x or any(not part or part == '*' for part in x.split('.'))
            for x in v) for k, v in fields.items())):
        raise ValueError('S4-FIELDS: expected purpose to explicit field paths')


def _radius(body):
    from datetime import datetime
    body = dict(body)
    if isinstance(body.get('ttl'), str):
        body['ttl'] = datetime.fromisoformat(body['ttl'])
    return BlastRadius.model_validate(body)


def _load_usage(session, agents, at):
    result = {}
    start = as_utc(at).replace(hour=0, minute=0, second=0, microsecond=0)
    start = min(start, as_utc(at) - timedelta(hours=1))
    for name, agent in agents.items():
        if agent.blast_radius is None:
            continue
        # Lifetime client/value limits need lifetime history; rate-only limits
        # need at most today plus the previous hour. Both reads have a hard cap.
        try:
            b = _radius(agent.blast_radius)
        except (ValueError, TypeError):
            result[name] = ()
            continue
        query = select(_BlastUsage).where(
            _BlastUsage.agent_name == name,
            _BlastUsage.at <= at,
        )
        if b.max_clients is None and b.max_value is None:
            query = query.where(_BlastUsage.at >= start)
        rows = session.exec(query.order_by(_BlastUsage.at.desc()).limit(10001)).all()
        result[name] = tuple((as_utc(r.at), r.clients, r.value, r.request_id) for r in rows)
    return result


def _checks(env, snapshot, at, *, _exclude=None):
    agent = snapshot.agents.get(env.agent_name)
    if agent is None:
        return []
    checks = []
    def gate(rule, ok):
        checks.append(Check(check=rule, citation=rule, regime='INTERNAL',
                            result=CheckResult.PASS if ok else CheckResult.FAIL))
    permission = (agent.tool_allowlist or {}).get(env.tool)
    gate('S4-TOOL-ALLOWLIST', isinstance(permission, dict) and not env.tool.startswith('_'))
    if agent.blast_radius is None:
        return checks
    try:
        b = _radius(agent.blast_radius)
    except (ValueError, TypeError):
        gate('S4-BLAST-RADIUS', False)
        return checks
    rows = snapshot._usage.get(agent.name)
    gate('S4-USAGE', rows is not None and len(rows) < 10001)
    rows = tuple(r for r in (rows or ()) if r[3] != _exclude)
    at = as_utc(at)
    day = at.replace(hour=0, minute=0, second=0, microsecond=0)
    for rule, limit, start in [('S4-PER-MINUTE', b.per_minute, at-timedelta(minutes=1)),
                               ('S4-PER-HOUR', b.per_hour, at-timedelta(hours=1)),
                               ('S4-DAILY-ACTIONS', b.max_daily_actions, day)]:
        if limit is not None:
            gate(rule, sum(start < t <= at if start != day else day <= t <= at for t, _, _, _ in rows) < limit)
    if b.max_clients is not None:
        clients = {c for _, cs, _, _ in rows for c in cs}
        current = env.all_subjects().get('client')
        if current:
            clients.add(current)
        gate('S4-MAX-CLIENTS', len(clients) <= b.max_clients)
    if b.max_value is not None:
        value = env.arguments.get('value')
        gate('S4-MAX-VALUE', type(value) in (int, float) and isfinite(value) and 0 <= value and sum(v for _, _, v, _ in rows) + value <= b.max_value)
    if b.channels is not None:
        gate('S4-CHANNELS', env.arguments.get('channel') in b.channels)
    if b.jurisdictions is not None:
        gate('S4-JURISDICTIONS', env.actor_location in b.jurisdictions)
    verb = permission.get('verb') if isinstance(permission, dict) else None
    gate('S4-EXECUTION-AUTHORITY', verb not in ('queue', 'send', 'execute') or b.execution_authority == 'execute')
    if b.ttl is not None:
        gate('S4-TTL', at < as_utc(b.ttl))
    return checks


def _minimize(session, env, data, evidence, *, _request_id=None):
    if not env.agent_name:
        return data
    agent = session.exec(select(Agent).where(Agent.name == env.agent_name).execution_options(populate_existing=True)).first()
    # Legacy deployments without a minimization declaration release no fields.
    try:
        if agent:
            _validate(agent)
        fields = (agent.field_allowlist or {}).get(env.purpose, []) if agent else []
    except (ValueError, TypeError):
        fields = []
    def project(value, paths):
        if isinstance(value, list):
            return [project(v, paths) for v in value]
        if not isinstance(value, dict):
            return value if "$" in paths and type(value) in (str, int, float, bool, type(None)) else None
        out = {}
        for key, child in value.items():
            nested = [p[len(key)+1:] for p in paths if p.startswith(key+'.')]
            if isinstance(child, (dict, list)):
                if nested:
                    out[key] = project(child, nested)
                elif key in paths and not child:
                    out[key] = child
            elif key in paths and type(child) in (str, int, float, bool, type(None)):
                out[key] = child
        return out
    if type(data).__module__ == "kognita.mcp" and type(data).__name__ == "_BackendPayload":
        result = type(data)(result=project(data.result, fields), error=project(data.error, fields))
    else:
        result = project(data, fields)
    evidence.emit(session, correlation_id=_request_id or env.agent_name + ':minimization',
        event_type=EventType.EGRESS, actor_type=ActorType.SYSTEM, actor_id='governance-pdp',
        payload={'action': 'FIELDS_MINIMIZED', 'citation': 'S4-FIELDS',
                 'agent': env.agent_name, 'purpose': env.purpose, 'fields': fields})
    return result


def _reserve(session, evaluation, at):
    if evaluation.envelope.agent_name and evaluation.allowed:
        session.add(_BlastUsage(request_id=evaluation.request_id,
            agent_name=evaluation.envelope.agent_name, at=at,
            clients=evaluation.attributes.get("_s4_clients", []),
            value=evaluation.attributes.get("_s4_value", 0)))
        session.flush()


def _released(session, evaluation, evidence, at):
    row = session.get(_BlastUsage, evaluation.request_id)
    if row is not None:
        row.at = at
        evidence.emit(session, correlation_id=evaluation.request_id,
            event_type=EventType.POLICY_CHANGE, actor_type=ActorType.SYSTEM,
            actor_id="governance-pdp", payload={"action": "BLAST_RADIUS_RELEASED",
                "citation": "S4-USAGE", "agent": row.agent_name,
                "at": at.isoformat(), "request_id": row.request_id})
        session.add(row)
        session.flush()


def _recheck(session, evaluation, at):
    """A stale supplied snapshot cannot spend an already consumed quota."""
    from dataclasses import replace
    from types import SimpleNamespace
    from kognita.governance import resolve_outcome
    from kognita.envelope import envelope_hash
    env = evaluation.envelope
    if not env.agent_name or not evaluation.allowed:
        return evaluation
    # Take a database write lock before reading counters; hold through commit.
    from sqlalchemy import update as _update
    session.execute(_update(Agent).where(Agent.name == env.agent_name)
                    .values(name=Agent.name).execution_options(synchronize_session=False))
    agent = session.exec(select(Agent).where(Agent.name == env.agent_name)
                         .execution_options(populate_existing=True)).first()
    if agent is None or agent.blast_radius is None:
        return evaluation
    agents = {agent.name: agent}
    snap = SimpleNamespace(agents=agents, _usage=_load_usage(session, agents, at))
    fresh = _checks(env, snap, at)
    if not any(c.result == CheckResult.FAIL for c in fresh):
        return evaluation
    checks = tuple(c for c in evaluation.checks if not c.check.startswith('S4-')) + tuple(fresh)
    attrs = dict(evaluation.attributes)
    attrs["_s4_usage"] = [[t.isoformat(), cs, v, req] for t, cs, v, req in snap._usage[agent.name]]
    return replace(evaluation, checks=checks, attributes=attrs, outcome=resolve_outcome(checks),
                   envelope_hash=envelope_hash(env, attrs, checks))
