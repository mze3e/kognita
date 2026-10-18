"""Host-controlled run bindings and reservations for bounded agent handoffs."""
from datetime import datetime
from math import isfinite
from uuid import uuid4

from kognita.models import _RunScope, _Authority, as_utc
from kognita.envelope import Check
from kognita.vocabulary import CheckResult
from kognita.authority import Delegation, _live, _covers, _emit, _scope_valid


def _restore(session, run):
    row = session.get(_RunScope, run.id, populate_existing=True)
    run._authority_scope = dict(row.body) if row else {}


def _run_checks(session, run, envelope, evidence):
    authority = session.get(_Authority, envelope.delegation_id, populate_existing=True) if envelope.delegation_id else None
    required_run = authority.body.get('_run') if authority else None
    if (required_run and (run is None or run.id != required_run)
            or run is None and _external_agent(session, envelope.agent_name)):
        return [Check(check='S6B-RUN-ISOLATION', citation='S6B-RUN-ISOLATION',
                      regime='INTERNAL', result=CheckResult.FAIL)]
    if run is None:
        return []
    from kognita.tools import _save_run
    _restore(session, run)
    identity = dict(agent=envelope.agent_name, human=envelope.principal,
                    purpose=envelope.purpose, delegation_id=envelope.delegation_id)
    if not run._authority_scope:
        _save_run(session, run)
        session.add(_RunScope(id=run.id, body=identity))
        session.flush()
        run._authority_scope = identity
        from kognita._isolation import _emit as emit
        emit(session, evidence, run.id, 'RUN_BOUND', citation='S6B-RUN-ISOLATION',
             run_id=run.id, **identity)
    ok = all(run._authority_scope.get(k) == v for k, v in identity.items())
    return [Check(check='S6B-RUN-ISOLATION', citation='S6B-RUN-ISOLATION',
                  regime='INTERNAL', result=CheckResult.PASS if ok else CheckResult.FAIL)]


def _delegate(session, run, name, scope, evidence, at):
    from kognita.tools import Run, _refresh_run, _save_run
    from kognita.authority import _load, _lineage_valid
    _refresh_run(session, run)
    binding = run._authority_scope
    parent = session.get(_Authority, binding.get('delegation_id', ''), populate_existing=True)
    from kognita.registry import get
    agent = get(session, name)
    owner = get(session, binding.get('agent', ''))
    def fail():
        raise ValueError('S6B-ATTENUATION: explicit live parent and narrower child authority required')
    if (not isinstance(scope, dict) or parent is None or parent.ended or not _live(parent.body, at)
            or parent.grantee != binding.get('agent') or agent is None or owner is None
            or run.control_state != 'live' or not scope.get('why') or not isinstance(scope['why'], str)):
        fail()
    if set(scope) - {'why', 'allowed_actions', 'allowed_data', 'tools', 'autonomy_level',
                     'valid_until', 'max_calls', 'max_tokens', 'max_cost_usd', 'wall_clock_seconds'}:
        fail()
    if parent.body.get('_run', run.id) != run.id:
        fail()
    rows = _load(session, parent.grantee, at, evidence)
    if not _lineage_valid(parent.id, rows, at):
        fail()
    d = Delegation.model_validate(parent.body)
    if not all(any(kind == 'grant' and grantee == d.human
                   and _live(body, at) and body.get('capability') == action
                   and _covers(body.get('subject_scope', {}), d.allowed_data)
                   for _, kind, grantee, body in rows)
               for action in d.allowed_actions):
        fail()
    actions = scope.get('allowed_actions', [])
    data = scope.get('allowed_data', {})
    tools = scope.get('tools', [])
    level = scope.get('autonomy_level', 0)
    try:
        until = as_utc(datetime.fromisoformat(scope['valid_until'])) if isinstance(scope['valid_until'], str) else as_utc(scope['valid_until'])
    except (KeyError, TypeError, ValueError, AttributeError):
        fail()
    parent_tools = parent.body.get('_tools', [k for k in (owner.tool_allowlist or {}) if not k.startswith('_')])
    if (not isinstance(actions, list) or not actions or not all(isinstance(a, str) for a in actions)
            or not set(actions) <= set(d.allowed_actions)
            or not isinstance(data, dict) or not data or not _scope_valid(data) or not _covers(d.allowed_data, data)
            or data.get('client') != [d.client]
            or not isinstance(tools, list) or not tools or not all(isinstance(t, str) for t in tools)
            or not set(tools) <= set(parent_tools)
            or not set(tools) <= set(agent.tool_allowlist or {})
            or type(level) is not int or not 0 <= level <= min(
                0 if owner.environment == 'external' else owner.autonomy_level or 0,
                0 if agent.environment == 'external' else agent.autonomy_level or 0,
                parent.body.get('_level', 6))
            or not as_utc(at) < until <= as_utc(d.valid_until)):
        fail()
    for tool in tools:
        a = (agent.tool_allowlist or {})[tool]
        p = (owner.tool_allowlist or {}).get(tool, {})
        if (not isinstance(a, dict) or not isinstance(p, dict)
                or a.get('verb') not in actions or a.get('verb') != p.get('verb')
                or a.get('system') != p.get('system') or a.get('required_level', 0) > level):
            fail()
        if tool in run.tool_denylist:
            fail()
    budgets = {}
    for cap, used in [('max_calls', 'calls_used'), ('max_tokens', 'tokens_used'), ('max_cost_usd', 'cost_usd_used')]:
        value = scope.get(cap)
        maximum = getattr(run, cap)
        if (type(value) not in (int, float) or not isfinite(value) or value < 0
                or cap != 'max_cost_usd' and type(value) is not int
                or maximum is None or value > maximum - getattr(run, used)):
            fail()
        budgets[cap] = value
    seconds = scope.get('wall_clock_seconds')
    remaining = run.wall_clock_seconds
    if remaining is not None and run.started_at:
        remaining -= (as_utc(at) - as_utc(run.started_at)).total_seconds()
    if type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0 or remaining is None or seconds > remaining or seconds > (until - as_utc(at)).total_seconds():
        fail()
    child = Run(**budgets, wall_clock_seconds=seconds, started_at=at,
                classification_ceiling=run.classification_ceiling,
                external=run.external or agent.environment == 'external',
                client_data=run.client_data, assurance_level=run.assurance_level,
                tool_denylist=list(run.tool_denylist))
    body = d.model_copy(update=dict(agent=name, allowed_actions=actions, allowed_data=data,
        valid_from=at, valid_until=until)).model_dump(mode='json')
    body.update(_parent=parent.id, _tools=tools, _level=level, _why=scope['why'],
                _parent_run=run.id, _run=child.id)
    row = _Authority(id=str(uuid4()), grantee=name, kind='delegation', body=body)
    _emit(session, evidence, parent.grantee, 'AUTHORITY_TRANSFERRED', row,
          _context={'citation': 'S6B-DELEGATION', 'from_agent': parent.grantee,
                    'budget': {**budgets, 'wall_clock_seconds': seconds},
                    'external': child.external, 'client_data': child.client_data,
                    'assurance_level': child.assurance_level,
                    'tool_denylist': child.tool_denylist,
                    'classification_ceiling': child.classification_ceiling})
    session.add(row)
    # Reserve capacity, rather than giving every sibling the same remaining budget.
    for cap, value in budgets.items():
        setattr(run, cap, getattr(run, cap) - value)
    _save_run(session, run)
    _save_run(session, child)
    child._authority_scope = dict(agent=name, human=d.human, purpose=d.purpose, delegation_id=row.id)
    session.add(_RunScope(id=child.id, body=child._authority_scope))
    session.flush()
    return child


def _external_agent(session, name):
    from kognita.registry import get
    agent = get(session, name) if name else None
    return agent is not None and agent.environment == 'external'
