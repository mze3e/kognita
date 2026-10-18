"""Explicit human entitlements and temporary, attenuated agent authority.

Grant capabilities are exact action verbs. Subject scopes map
subject types to explicit identifiers; empty scopes confer no subject access.
Callers supply an authenticated issuer and own the transaction. The issuer of
human entitlements must be authenticated by the deployment's administration
boundary; envelope roles and context never create entitlements.
"""
from datetime import datetime as _datetime, timedelta as _timedelta
from uuid import uuid4 as _uuid4
from math import isfinite as _isfinite

from pydantic import BaseModel as _BaseModel, Field as _Field
from sqlmodel import select as _select

from kognita.models import _Authority, as_utc as _utc, utcnow as _now
from kognita.envelope import Check as _Check
from kognita.vocabulary import ActorType as _Actor, EventType as _Event, CheckResult as _Result


class Delegation(_BaseModel):
    human: str
    agent: str
    purpose: str
    client: str
    allowed_actions: list[str]
    allowed_data: dict[str, list[str]]
    valid_from: _datetime = _Field(default_factory=_now)
    valid_until: _datetime = _Field(default_factory=lambda: _now() + _timedelta(hours=1))
    limits: dict[str, float] = _Field(default_factory=dict)
    jurisdiction: str
    approval_threshold: str = "read"
    task_ref: str


_VERBS = ("read", "analyse", "recommend", "draft", "queue", "send", "execute")


def _emit(session, evidence, actor_id, action, row, *, _context=None):
    evidence.emit(session, correlation_id=f"authority:{row.id}",
        event_type=_Event.POLICY_CHANGE,
        actor_type=(_Actor.SYSTEM if action == "AUTHORITY_EXPIRED" else
                    _Actor.AGENT if action == "AUTHORITY_TRANSFERRED" else _Actor.HUMAN),
        actor_id=actor_id, payload={"action": action, "citation": "S3-AUTHORITY",
                                   "id": row.id, "kind": row.kind, "grantee": row.grantee, "body": row.body, **(_context or {})})


def grant_to(grantee, capability, duration, subject_scope, *, session, evidence,
             actor_id, now=None):
    """Issue an exact capability; the deployment authenticates the issuer."""
    at = _utc(now or _now())
    if (not actor_id.strip() or not grantee.strip() or capability not in _VERBS
            or not isinstance(duration, _timedelta) or duration <= _timedelta(0)
            or not _scope_valid(subject_scope)):
        raise ValueError("S3-GRANT: invalid issuer, capability, duration or scope")
    row = _Authority(id=str(_uuid4()), grantee=grantee, kind="grant", body={
        "issuer": actor_id, "capability": capability, "subject_scope": subject_scope,
        "valid_from": at.isoformat(), "valid_until": (at + duration).isoformat()})
    _emit(session, evidence, actor_id, "GRANT_ISSUED", row)
    session.add(row)
    session.flush()
    return row.id


def _scope_valid(scope):
    return isinstance(scope, dict) and all(isinstance(k, str) and k and
        isinstance(v, list) and v and all(isinstance(x, str) and x and x != "*" for x in v)
        for k, v in scope.items())


def _live(body, at):
    return _utc(_datetime.fromisoformat(body["valid_from"])) <= _utc(at) < _utc(_datetime.fromisoformat(body["valid_until"]))


def _covers(scope, requested):
    return all(k in scope and set(v) <= set(scope[k]) for k, v in requested.items())


def delegate(session, delegation: Delegation, *, evidence, actor_id, now=None):
    """Attenuate a live human entitlement, never widen it."""
    d = Delegation.model_validate(delegation)
    at = now or _now()
    if (actor_id != d.human or not all(x.strip() for x in
        (d.human, d.agent, d.purpose, d.client, d.jurisdiction, d.task_ref))
        or not d.allowed_actions or any(a not in _VERBS for a in d.allowed_actions)
        or d.approval_threshold not in _VERBS or not _scope_valid(d.allowed_data)
        or d.allowed_data.get("client") != [d.client]
        or _utc(d.valid_until) <= _utc(d.valid_from) or _utc(d.valid_until) <= _utc(at)
        or any(k.startswith("_") and k not in ("_event_at", "_access_window") for k in d.limits)
        or ("_access_window" in d.limits and "_event_at" not in d.limits)
        or d.limits.get("_access_window", 86400) > 86400
        or any(not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0 or not _isfinite(v) for v in d.limits.values())):
        raise ValueError("S3-DELEGATION: invalid bounded authority")
    grants = session.exec(_select(_Authority).where(_Authority.grantee == d.human,
        _Authority.kind == "grant", _Authority.ended == False).execution_options(populate_existing=True)).all()
    for action in d.allowed_actions:
        if not any(_live(g.body, at) and g.body["capability"] == action
            and _covers(g.body["subject_scope"], d.allowed_data)
            and _datetime.fromisoformat(g.body["valid_from"]) <= _utc(d.valid_from)
            and _datetime.fromisoformat(g.body["valid_until"]) >= _utc(d.valid_until) for g in grants):
            raise ValueError("S3-ATTENUATION: exceeds human entitlement")
    row = _Authority(id=str(_uuid4()), grantee=d.agent, kind="delegation", body=d.model_dump(mode="json"))
    _emit(session, evidence, actor_id, "DELEGATION_ISSUED", row)
    session.add(row)
    session.flush()
    return row.id


def revoke(session, delegation_id, *, evidence, actor_id):
    row = session.get(_Authority, delegation_id, populate_existing=True)
    if row is None or actor_id != row.body.get("human", row.body.get("issuer")):
        raise ValueError("S3-REVOKE: issuer required")
    if not row.ended:
        _emit(session, evidence, actor_id, "AUTHORITY_REVOKED", row)
        from kognita._actions import _stop
        _stop(delegation=delegation_id)
        row.ended = True
        session.add(row)
        session.flush()


def complete_task(session, task_ref, *, evidence, actor_id):
    rows = session.exec(_select(_Authority).where(_Authority.kind == "delegation",
        _Authority.ended == False).execution_options(populate_existing=True)).all()
    for row in rows:
        if row.body["task_ref"] == task_ref and row.body["human"] == actor_id:
            _emit(session, evidence, actor_id, "TASK_COMPLETED", row)
            from kognita._actions import _stop
            _stop(delegation=row.id)
            row.ended = True
            session.add(row)
    session.flush()


def _load(session, agent_name, at, evidence):
    query = _select(_Authority).where(_Authority.ended == False)
    if agent_name is not None:
        humans = _select(_Authority.body["human"].as_string()).where(
            _Authority.grantee == agent_name, _Authority.kind == "delegation", _Authority.ended == False)
        query = query.where((_Authority.grantee == agent_name) | _Authority.grantee.in_(humans))
    rows = session.exec(query.limit(1001).execution_options(populate_existing=True)).all()
    if len(rows) > 1000:
        return ()  # refuse oversized authority sets rather than truncate permissions
    result = []
    for row in rows:
        if _utc(_datetime.fromisoformat(row.body["valid_until"])) <= _utc(at):
            if evidence is not None:
                _emit(session, evidence, "system", "AUTHORITY_EXPIRED", row)
                row.ended = True
                session.add(row)
        else:
            result.append((row.id, row.kind, row.grantee, dict(row.body)))
    # Follow primary keys only; bounded depth also rejects cycles and missing links.
    pending = list(result)
    seen = {r[0] for r in result}
    for _ in range(32):
        parents = {r[3].get('_parent') for r in pending} - seen - {None}
        if not parents:
            break
        pending = []
        for identity in sorted(parents):
            if len(result) + len(pending) >= 1000:
                return ()
            row = session.get(_Authority, identity, populate_existing=True)
            if row is not None and not row.ended and _live(row.body, at):
                pending.append((row.id, row.kind, row.grantee, dict(row.body)))
                seen.add(row.id)
        result.extend(pending)
    return tuple(result)


#: How deep into nested tool arguments a dimension's value is still resolved.
_MAX_ARGUMENT_DEPTH = 8


def _dimension_keys(dimension):
    keys = {dimension, dimension + "_id"}
    if dimension == "client":
        keys.add("clientid")
    return keys


def _argument_values(value, dimension, *, _depth=0):
    """Every value naming ``dimension`` anywhere inside ``value``.

    Walks dicts and lists up to :data:`_MAX_ARGUMENT_DEPTH` deep, matching a
    key equal to ``dimension`` or ``dimension + "_id"`` (``clientid`` too,
    for ``"client"``), case-insensitively, scalar or list; and a sibling
    ``{"subject_type": dimension, "subject_id": ...}`` pair, the same shape
    :class:`~kognita.envelope.Envelope` itself uses for its primary subject.
    Everything is coerced to a string so a numeric id still matches a
    delegation's string client.

    Returns ``None`` once the structure goes deeper than the bound: a value
    that deep is unresolved, not absent, so the caller must fail closed
    rather than silently treat it as a non-match.
    """
    if _depth > _MAX_ARGUMENT_DEPTH:
        return None
    values = set()
    if isinstance(value, dict):
        lowered = {k.lower(): v for k, v in value.items() if isinstance(k, str)}
        if str(lowered.get("subject_type", "")).lower() == dimension and "subject_id" in lowered:
            subject_id = lowered["subject_id"]
            if subject_id is not None:
                values.add(str(subject_id))
        keys = _dimension_keys(dimension)
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            if key.lower() in keys:
                # Under a key that names the dimension only scalar ids are
                # resolvable; an object or nested structure there is
                # unresolved and fails closed.
                items = item if isinstance(item, (list, tuple)) else [item]
                for element in items:
                    if isinstance(element, (dict, list, tuple)):
                        return None
                    if element is not None:
                        values.add(str(element))
            elif isinstance(item, (dict, list, tuple)):
                nested = _argument_values(item, dimension, _depth=_depth + 1)
                if nested is None:
                    return None
                values.update(nested)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                nested = _argument_values(item, dimension, _depth=_depth + 1)
                if nested is None:
                    return None
                values.update(nested)
    return values


def _checks(envelope, snapshot, at):
    if not envelope.agent_name:
        return []
    def check(rule, ok, human=False):
        return _Check(check=rule, regime="INTERNAL", citation=rule,
            result=(_Result.REQUIRES_HUMAN if ok and human else _Result.PASS if ok else _Result.FAIL))
    rows = snapshot._authority
    selected = next((r for r in rows if r[0] == envelope.delegation_id and r[1] == "delegation"), None)
    if selected is None:
        return [check("S3-DELEGATION", False)]
    try:
        d = Delegation.model_validate(selected[3])
    except ValueError:
        return [check("S3-DELEGATION", False)]
    if d.approval_threshold not in _VERBS:
        return [check("S3-APPROVAL-THRESHOLD", False)]
    agent = snapshot.agents.get(envelope.agent_name)
    permission = (agent.tool_allowlist or {}).get(envelope.tool, {}) if agent else {}
    verb = permission.get("verb") if isinstance(permission, dict) else None
    subjects = {k: [v] for k, v in envelope.all_subjects().items()}
    # A tool argument naming a delegation's own dimension (a client it is
    # scoped to, or any other `allowed_data` dimension) overrides the
    # delegation's implied value only in appearance: it must be covered by
    # `allowed_data`, however deep it is nested, or the call is for someone
    # else's data and is denied rather than silently rescoped. `client` is
    # always one such dimension (`delegate` requires it), so this alone
    # enforces the exact-match the delegation's own client needs.
    valid = (_lineage_valid(selected[0], rows, at) and _live(selected[3], at) and d.agent == envelope.agent_name
        and d.human == envelope.principal and d.purpose == envelope.purpose
        and d.jurisdiction == envelope.actor_location and verb in d.allowed_actions
        and _covers(d.allowed_data, subjects)
        and subjects.get("client", [d.client]) == [d.client]
        and all((named := _argument_values(envelope.arguments, dim)) is not None and named <= set(allowed)
                for dim, allowed in d.allowed_data.items()))
    if '_parent' in selected[3]:
        valid = valid and envelope.tool in selected[3].get('_tools', [])
        valid = valid and permission.get('required_level', 7) <= selected[3].get('_level', -1)
    if agent and agent.environment == 'external':
        valid = valid and verb == 'read' and permission.get('required_level') == 0 and envelope.assurance_level == 0
    event = d.limits.get("_event_at")
    window = d.limits.get("_access_window", 86400)
    if event is not None:
        valid = valid and 0 <= event - _utc(at).timestamp() <= window <= 86400
    # Numeric arguments are bound into the envelope hash; missing values deny.
    valid = valid and all(type(envelope.arguments.get(k)) in (int, float)
        and 0 <= envelope.arguments[k] <= v for k, v in d.limits.items() if k not in ("_event_at", "_access_window"))
    grants = [r for r in rows if r[1] == "grant" and r[2] == envelope.principal
        and _live(r[3], at) and r[3]["capability"] == verb
        and _covers(r[3]["subject_scope"], d.allowed_data)]
    return [check("S3-DELEGATION", valid), check("S3-HUMAN-ENTITLEMENT", bool(grants)),
        check("S3-APPROVAL-THRESHOLD", verb in _VERBS,
              human=verb in _VERBS and _VERBS.index(verb) > _VERBS.index(d.approval_threshold))]


__all__ = ["Delegation", "delegate", "revoke", "complete_task", "grant_to"]


def _record_use(session, evaluation, evidence):
    _load(session, evaluation.envelope.agent_name or "", evaluation.as_of, evidence)
    if evaluation.outcome.value not in ("ALLOW", "HUMAN_APPROVAL"):
        return
    env = evaluation.envelope
    if not env.agent_name:
        return
    # The exact grants relied on are pinned by the decision, not inferred later.
    for identity in evaluation.attributes.get("_authority_grants", []):
        row = session.get(_Authority, identity, populate_existing=True)
        if row is not None:
            _emit(session, evidence, env.principal, "GRANT_USED", row,
                  _context={"delegation_id": env.delegation_id, "request_id": evaluation.request_id})


def _grant_ids(envelope, snapshot, at):
    selected = next((r for r in snapshot._authority if r[0] == envelope.delegation_id
                     and r[1] == "delegation"), None)
    if selected is None:
        return []
    agent = snapshot.agents.get(envelope.agent_name)
    permission = (agent.tool_allowlist or {}).get(envelope.tool, {}) if agent else {}
    verb = permission.get("verb") if isinstance(permission, dict) else None
    return sorted(r[0] for r in snapshot._authority if r[1] == "grant"
        and r[2] == envelope.principal and _live(r[3], at)
        and r[3]["capability"] == verb
        and _covers(r[3]["subject_scope"], selected[3]["allowed_data"]))[:1]


def _lineage_valid(identity, rows, at):
    mapping = {r[0]: r for r in rows}
    seen = set()
    while identity:
        if identity in seen or len(seen) >= 32:
            return False
        seen.add(identity)
        row = mapping.get(identity)
        if row is None or row[1] != 'delegation' or not _live(row[3], at):
            return False
        body = row[3]
        parent = body.get('_parent')
        if parent:
            ancestor = mapping.get(parent)
            if ancestor is None:
                return False
            a = ancestor[3]
            if (any(body.get(k) != a.get(k) for k in ('human', 'purpose', 'client', 'jurisdiction', 'task_ref'))
                or not set(body['allowed_actions']) <= set(a['allowed_actions'])
                or not _covers(a['allowed_data'], body['allowed_data'])
                or _utc(_datetime.fromisoformat(body['valid_until'])) > _utc(_datetime.fromisoformat(a['valid_until']))
                or body.get('_level', 7) > a.get('_level', 6)
                or ('_tools' in a and not set(body.get('_tools', [])) <= set(a['_tools']))):
                return False
        identity = parent
    return bool(seen)


def _lineage(session, action_id):
    """Reconstruct historical authority from immutable decision/issuance evidence."""
    from kognita.models import EvidenceEvent
    events = session.exec(_select(EvidenceEvent).where(
        EvidenceEvent.correlation_id == str(action_id)).order_by(EvidenceEvent.sequence).limit(1001)).all()
    decision = next((e for e in events if 'envelope' in e.payload), None)
    if decision is None:
        raise ValueError('S6B-LINEAGE: no evidenced action for request id')
    identity = decision.payload['envelope'].get('delegation_id')
    chain, seen = [], set()
    while identity:
        if identity in seen or len(seen) >= 32:
            raise ValueError('S6B-LINEAGE: cyclic or excessive authority depth')
        seen.add(identity)
        issued = session.exec(_select(EvidenceEvent).where(
            EvidenceEvent.correlation_id == 'authority:' + identity,
            EvidenceEvent.sequence < decision.sequence).order_by(EvidenceEvent.sequence).limit(1001)).all()
        event = next((e for e in issued if e.payload.get('action') in ('DELEGATION_ISSUED', 'AUTHORITY_TRANSFERRED')), None)
        if event is None:
            raise ValueError('S6B-LINEAGE: missing authority issuance evidence')
        chain.append({'id': identity, 'actor': event.actor_id, 'body': event.payload['body'],
                      'event_hash': event.event_hash})
        identity = event.payload['body'].get('_parent')
    grants = decision.payload.get('attributes', {}).get('_authority_grants', [])
    entitlement = []
    for identity in grants:
        events = session.exec(_select(EvidenceEvent).where(
            EvidenceEvent.correlation_id == 'authority:' + identity,
            EvidenceEvent.sequence < decision.sequence).order_by(EvidenceEvent.sequence).limit(1001)).all()
        event = next((e for e in events if e.payload.get('action') == 'GRANT_ISSUED'), None)
        if event is None:
            raise ValueError('S6B-LINEAGE: missing entitlement evidence')
        entitlement.append({'id': identity, 'body': event.payload['body'], 'event_hash': event.event_hash})
    return {'action_id': str(action_id), 'outcome': decision.payload['outcome'],
            'human': decision.payload['envelope']['principal'], 'grants': entitlement,
            'chain': list(reversed(chain)), 'action': decision.payload['envelope'],
            'event_hash': decision.event_hash}
