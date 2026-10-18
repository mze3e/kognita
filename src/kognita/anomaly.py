"""Bounded, server-observed daily baselines and automatic containment.

Cold-start ceilings apply until three historical days exist. Baselines use the
previous seven active days, never the current day's traffic. All state and
containment evidence share the decision transaction. No background release.
"""
from dataclasses import replace as _replace

from sqlmodel import select as _select
from kognita.models import Agent as _Agent, _AnomalyDay, _AnomalyWindow, as_utc as _utc
from kognita.envelope import Check as _Check, envelope_hash as _hash
from kognita.vocabulary import ActorType as _Actor, EventType as _Event, CheckResult as _Result, Outcome as _Outcome


def _observe(session, evaluation, evidence, at, *, _documents=None, _classification=None):
    env = evaluation.envelope
    if not env.agent_name:
        return evaluation
    # Take a database write lock before reading counters; hold through commit.
    from sqlalchemy import update as _update
    session.execute(_update(_Agent).where(_Agent.name == env.agent_name)
                    .values(name=_Agent.name).execution_options(synchronize_session=False))
    agent = session.exec(_select(_Agent).where(_Agent.name == env.agent_name)
                         .execution_options(populate_existing=True)).first()
    if agent is None:
        return evaluation
    day = _utc(at).date().isoformat()
    rows = session.exec(_select(_AnomalyDay).where(
        _AnomalyDay.agent_name == agent.name, _AnomalyDay.day <= day)
        .order_by(_AnomalyDay.day.desc()).limit(8)
        .execution_options(populate_existing=True)).all()
    row = next((r for r in rows if r.day == day), None)
    old = [r.body for r in rows if r.day < day][:7]
    body = dict(row.body) if row else {}
    # Repeated quarantined calls do not create incidents or grow the baseline.
    rule = 'S6-QUARANTINE' if agent.kill_switch else ''
    if not rule:
        permission = (agent.tool_allowlist or {}).get(env.tool)
        probing = not isinstance(permission, dict)
        denied = evaluation.outcome == _Outcome.DENY
        for key, increment in [('actions', int(_documents is None)), ('documents', len(_documents or [])),
                               ('denials', int(denied and _documents is None)), ('probing', int(probing and _documents is None))]:
            body[key] = body.get(key, 0) + increment
        for key, value in [('clients', env.all_subjects().get('client')), ('tools', env.tool),
                           ('data_classes', evaluation.attributes.get('classification'))]:
            values = set(body.get(key, []))
            if isinstance(value, str) and value:
                values.add(value)
            body[key] = sorted(values)[:1001]
        if _classification is not None and evaluation.allowed:
            body['data_classes'] = sorted(set(body['data_classes']) | {_classification.value})
        if _documents is not None:
            body['data_classes'] = sorted(set(body['data_classes']) | {str(h.classification.value) for h in _documents})
        pair = [body.get('last_tool'), env.tool]
        sequences = list(body.get('sequences', []))
        if _documents is None and pair[0] and pair not in sequences:
            sequences.append(pair)
        body['sequences'] = sequences[:1001]
        if _documents is None:
            body['last_tool'] = env.tool
        window = session.get(_AnomalyWindow, agent.name, populate_existing=True)
        cutoff = _utc(at).timestamp() - 86400
        recent = [r for r in (window.body if window else []) if r[0] > cutoff]
        client = env.all_subjects().get('client')
        if _documents is None or _documents or client and client not in {r[3] for r in recent}:
            recent.append([_utc(at).timestamp(), int(_documents is None), len(_documents or []), client])
        if window is None:
            window = _AnomalyWindow(agent_name=agent.name)
        window.body = recent[-2001:]
        session.add(window)
        if len(recent) > 2000:
            rule = 'S6-VOLUME'
        for key, ceiling in [('actions', 1000), ('documents', 1000), ('clients', 100)]:
            size = len(body[key]) if key == 'clients' else body[key]
            baseline = max((len(b.get(key, [])) if key == 'clients' else b.get(key, 0) for b in old), default=0)
            limit = min(ceiling, max(10, baseline * 3)) if len(old) >= 3 else ceiling
            rolling = (len({r[3] for r in recent if r[3]}) if key == 'clients'
                       else sum(r[1 if key == 'actions' else 2] for r in recent))
            if max(size, rolling) > limit:
                rule = 'S6-VOLUME'
        if body['denials'] >= 10 or body['probing'] >= 3:
            rule = 'S6-PRIVILEGE-PROBING'
        if len(old) >= 3:
            known_tools = {t for b in old for t in b.get('tools', [])}
            known_pairs = [p for b in old for p in b.get('sequences', [])]
            if _documents is None and (env.tool not in known_tools or (pair[0] and known_pairs and pair not in known_pairs)):
                rule = 'S6-TOOL-SEQUENCE'
            classes = {c for b in old for c in b.get('data_classes', [])}
            if classes and any(c not in classes for c in body['data_classes']):
                rule = 'S6-DATA-CLASS'
        if row is None:
            row = _AnomalyDay(agent_name=agent.name, day=day)
        if rule:
            body['quarantined'] = True
        row.body = body
        session.add(row)
        session.flush()
        evidence.emit(session, correlation_id='anomaly:' + evaluation.request_id,
            event_type=_Event.POLICY_CHANGE, actor_type=_Actor.SYSTEM,
            actor_id='anomaly', payload={'action': 'ANOMALY_OBSERVED',
                'citation': 'S6-BASELINE', 'agent': agent.name, 'request_id': evaluation.request_id, 'day': day, 'baseline': body})
        if rule:
            _contain(session, agent, evaluation, evidence, at, rule)
    if not rule:
        return evaluation
    check = _Check(check=rule, citation=rule, regime='INTERNAL', result=_Result.FAIL)
    checks = evaluation.checks + (check,)
    return _replace(evaluation, checks=checks, outcome=_Outcome.DENY,
                    envelope_hash=_hash(env, evaluation.attributes, checks))


def _contain(session, agent, evaluation, evidence, at, rule):
    from kognita.credentials import revoke
    from kognita.usecases import load_register
    agent.kill_switch = True
    session.add(agent)
    session.flush()
    def emit(action, **extra):
        evidence.emit(session, correlation_id=evaluation.request_id,
            event_type=_Event.POLICY_CHANGE, actor_type=_Actor.SYSTEM,
            actor_id='anomaly', payload={'action': action, 'citation': rule,
                'agent': agent.name, 'usecase_id': evaluation.envelope.purpose, **extra})
    emit('ANOMALY_QUARANTINED')
    revoke(session, agent.name, None, evidence=evidence, actor_id='anomaly', _automatic=True)
    from kognita._actions import _stop
    _stop(agent=agent.name)
    emit('ANOMALY_TOOLS_STOPPED', scope='new admissions and in-flight calls')
    purpose = evaluation.envelope.purpose
    entry = load_register(session, as_of=at).get(purpose)
    emit('INCIDENT', usecase_id=purpose if entry else '', impact=1,
         summary=f'{rule}: {agent.name} automatically quarantined',
         accountable_owner=agent.owner_exec, at=_utc(at).isoformat())
    emit('ANOMALY_ALERT', accountable_owner=agent.owner_exec,
         _recipients=[agent.owner_exec, entry.knowledge_lead if entry else 'Knowledge Lead'])


__all__ = []


def _retrieved(session, evaluation, evidence, hits):
    from kognita.governance import _evidence_envelope
    checked = _observe(session, evaluation, evidence, evidence.clock(), _documents=hits)
    if checked.outcome == _Outcome.DENY:
        evidence.emit(session, correlation_id=evaluation.request_id,
            event_type=_Event.POLICY_DECISION, actor_type=_Actor.SYSTEM,
            actor_id='anomaly', payload={'outcome': 'DENY', 'withheld': True,
                'envelope': _evidence_envelope(evaluation.envelope),
                'envelope_hash': checked.envelope_hash,
                'checks': [c.to_dict() for c in checked.checks]})
        from kognita.breaker import evaluate_breakers
        evaluate_breakers(session, evidence=evidence, now=evidence.clock())
        return [], checked
    return hits, checked
