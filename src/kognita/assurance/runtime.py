"""Admission checks shared by decisions, tool calls, gateway and MCP."""
from __future__ import annotations

from datetime import datetime
from sqlmodel import select

from kognita.assurance.records import (
    agent_for, binding, context, emit, events, latest, policy, policy_from,
)
from kognita.assurance.trust import Attestation, verify_assertion, verify_record
from kognita.canonical import canonical_hash
from kognita.envelope import Check
from kognita.models import EvidenceEvent, as_utc, utcnow
from kognita.vocabulary import CheckResult, EventType


def request_digest(envelope):
    """Exact instruction, arguments, principal and subjects; claims grant nothing."""
    return canonical_hash(envelope.to_dict())


def attest(session, name, packet, *, evidence, actor_id, now=None):
    """Verify and record an independent deployment-bound boundary assertion."""
    at = as_utc(now or utcnow())
    agent = agent_for(session, name)
    p = policy(session, name)
    a = verify_record(packet, Attestation, p.attestors, at)
    if actor_id != a.issuer or a.subject != name or actor_id == name:
        raise ValueError('A7: independent authenticated attestor required')
    if {'deployment': a.deployment, 'policy': a.policy} != binding(agent, p):
        raise ValueError('A7: stale deployment/policy binding')
    claims = a.claims
    if a.kind == 'privacy':
        if claims.get('context_digest') != canonical_hash(context(session, agent, p)):
            raise ValueError('A6: privacy context mismatch')
    elif a.kind == 'environment':
        required = ('network_policy', 'isolation', 'credential_scope', 'sandbox_identity', 'termination_path')
        if any(not isinstance(claims.get(k), str) or not claims[k].strip() for k in required):
            raise ValueError('A7: all five enforcement references required')
        if claims.get('enforced') is not True:
            raise ValueError('A7: enforcement not attested')
    elif a.kind == 'monitoring':
        _coverage(session, name, claims)
    elif a.kind == 'instruction':
        if (not isinstance(claims.get('request_digest'), str) or
                len(claims['request_digest']) != 64 or not claims.get('verification_channel')
                or not claims.get('maker') or not claims.get('checker')
                or len({claims['maker'], claims['checker'], name}) != 3):
            raise ValueError('A11: exact request and independent maker/checker required')
    elif a.kind == 'transfer':
        if claims.get('transfer_digest') != transfer_digest(p) or claims.get('approved') is not True:
            raise ValueError('A10: approval must match supplied transfer classification')
    # Retain the signed packet so later reports can independently recheck it.
    emit(session, evidence, name, 'ATTESTATION', packet, actor_id=actor_id, at=at)
    if a.kind == 'monitoring':
        coverage = _coverage(session, name, claims)
        if not coverage['complete']:
            emit(session, evidence, name, 'FINDING', {'control': 'A5', **coverage},
                 actor_id='kognita', at=at, system=True)
    return canonical_hash(packet)


def transfer_digest(p):
    """Fingerprint the institution-supplied technology-transfer classification."""
    return canonical_hash({k: getattr(p, k) for k in (
        'technology_origin', 'deployment_jurisdiction', 'transfer_type', 'transfer_policy')})


def _action_rows(session, name, *, as_of=None, through_sequence=None):
    # Independent telemetry references exact local hashes, never counts alone.
    """Read actions requiring independent telemetry, excluding unsent calls."""
    query = select(EvidenceEvent).where(EvidenceEvent.actor_id == name,
        EvidenceEvent.event_type.in_([EventType.TOOL_CALL, EventType.MODEL_CALL, EventType.EGRESS]))
    if as_of is not None:
        query = query.where(EvidenceEvent.recorded_at <= as_utc(as_of))
    if through_sequence is not None:
        query = query.where(EvidenceEvent.sequence <= through_sequence)
    rows = session.exec(query.order_by(EvidenceEvent.sequence)).all()
    return [r for r in rows if not (r.event_type == EventType.MODEL_CALL and r.payload.get('sent') is False)]


def _coverage(session, name, claims):
    """Reconcile exact action hashes within an explicitly bounded window."""
    start, end = claims.get('from_sequence'), claims.get('through_sequence')
    hashes = claims.get('action_hashes')
    external = claims.get('unexplained_external_actions')
    if (type(start) is not int or type(end) is not int or start < 1 or end < start - 1
            or not isinstance(hashes, list) or any(not isinstance(x, str) for x in hashes)
            or not isinstance(external, list) or any(not isinstance(x, str) for x in external)):
        raise ValueError('A5: explicit coverage window and actions required')
    head = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence.desc())).first()
    if end > (head.sequence if head else 0):
        raise ValueError('A5: coverage cannot attest future actions')
    local = {r.event_hash for r in _action_rows(session, name) if start <= r.sequence <= end}
    supplied = set(hashes)
    return {'from_sequence': start, 'through_sequence': end,
            'observed_actions': sorted(local), 'attested_actions': sorted(supplied),
            'missing': sorted(local - supplied), 'unexplained': sorted(supplied - local) + external,
            'complete': local == supplied and not external}


def coverage_report(session, name, *, now=None):
    """Report independently attested coverage and unexplained action gaps."""
    at = as_utc(now or utcnow())
    p = policy(session, name)
    agent = agent_for(session, name)
    rows = events(session, name)
    a = _attestation(rows, agent, p, 'monitoring', at)
    if a is None:
        return {'complete': False, 'findings': ['A5: missing or expired monitoring attestation']}
    result = _coverage(session, name, a.claims)
    unaccounted = [r.event_hash for r in _action_rows(session, name)
                   if r.sequence > a.claims['through_sequence'] or r.sequence < a.claims['from_sequence']]
    result['outside_window'] = unaccounted
    result['complete'] = result['complete'] and not unaccounted
    result['findings'] = [] if result['complete'] else ['A5: unexplained monitoring coverage gap']
    return result


def _attestation(rows, agent, p, kind, at, *, digest=None):
    """Select an exact request proof or the current superseding boundary record."""
    for row in reversed(rows):
        if row.payload.get('action') != 'ATTESTATION':
            continue
        packet = row.payload['body']
        body = packet.get('document', {}).get('body', {})
        if body.get('kind') != kind:
            continue
        if digest is not None and canonical_hash(packet) != digest:
            continue
        # Superseding a record with an invalid one cannot revive an older record.
        try:
            a = verify_record(packet, Attestation, p.attestors, at)
            if a.subject != agent.name or {'deployment': a.deployment, 'policy': a.policy} != binding(agent, p):
                return None
            return a
        except ValueError:
            return None
    return None


def load_state(session, names, at, *, through_sequence=None):
    """Batch configured agents and advisories with one integrity check per snapshot.

    A matching head cannot prove an old payload is intact. Verify the prefix
    on each new snapshot rather than trusting a cross-snapshot head-only cache.
    """
    from collections import defaultdict
    from collections.abc import Mapping
    identities = names if isinstance(names, Mapping) else None
    names = tuple(names)
    grouped = defaultdict(list)
    every = events(session, as_of=at, _names=(*names, '*'), _through_sequence=through_sequence)
    for row in every:
        grouped[row.correlation_id].append(row)
    advisories = tuple(row.payload['body'] for row in grouped['assurance:*']
                       if row.payload.get('action') == 'ADVISORY')
    state = {}
    for name in names:
        rows = grouped[f'assurance:{name}']
        p = policy_from(rows)
        if p is None:
            continue
        agent = identities[name] if identities is not None else agent_for(session, name)
        historical_at = at if through_sequence is not None else None
        state[name] = {'policy': p, 'rows': tuple(rows),
                       'privacy': canonical_hash(context(session, agent, p, as_of=historical_at)) if p.privacy_required else None,
                       'actions': tuple((r.sequence, r.event_hash) for r in _action_rows(session, name, as_of=historical_at,
                                      through_sequence=through_sequence))
                                  if p.monitoring_required else (),
                       'advisories': advisories}
    return state


def admission_checks(envelope, agent, state, at, *, activation=False):
    """Evaluate configured assurance requirements without performing external work."""
    if agent is None or agent.name not in state:
        if envelope.context.get('external_assertion') is not None:
            return [Check(check='A2-EXTERNAL-TRUST', regime='INTERNAL',
                          citation='A2: no configured external trust roots', result=CheckResult.FAIL)]
        return []
    s = state[agent.name]
    p, rows = s['policy'], s['rows']
    checks = []

    def gate(code, ok):
        """Append a cited pass or fail for one configured assurance requirement."""
        checks.append(Check(check=code, regime='INTERNAL',
                            citation=code, result=CheckResult.PASS if ok else CheckResult.FAIL))

    gate('A0-USECASE', p.use_case == envelope.purpose)
    if p.credential_expiry is not None:
        gate('A1-CREDENTIAL-EXPIRY', as_utc(at) < as_utc(p.credential_expiry))
    if not activation and (p.external_required or envelope.context.get('external_assertion') is not None):
        try:
            verify_assertion(envelope.context.get('external_assertion'), p, envelope, at,
                             parent=envelope.context.get('external_parents'))
            valid = True
        except (TypeError, ValueError):
            valid = False
        gate('A2-EXTERNAL-TRUST', valid)
    if p.privacy_required:
        a = _attestation(rows, agent, p, 'privacy', at)
        gate('A6-PRIVACY', a is not None and a.claims.get('context_digest') == s['privacy'])
    if p.environment_required:
        a = _attestation(rows, agent, p, 'environment', at)
        gate('A7-ENVIRONMENT', a is not None and a.claims.get('enforced') is True)
    if p.monitoring_required and not activation:
        a = _attestation(rows, agent, p, 'monitoring', at)
        ok = False
        if a:
            c = a.claims
            expected = {h for seq, h in s['actions']}
            ok = (all(c['from_sequence'] <= seq <= c['through_sequence'] for seq, _ in s['actions'])
                  and expected == set(c['action_hashes']) and not c['unexplained_external_actions'])
        gate('A5-MONITORING', ok)
    if p.components_required:
        components = latest(rows, 'COMPONENTS')
        ok = bool(components) and components.get('binding') == binding(agent, p)
        if ok:
            ok = not any(any(c['name'] == adv['name'] and c['version'] == adv['version']
                             for c in components['components']) for adv in s['advisories'])
        gate('A9-COMPONENTS', ok)
    if p.transfer_required:
        a = _attestation(rows, agent, p, 'transfer', at)
        gate('A10-TRANSFER', a is not None and a.claims.get('transfer_digest') == transfer_digest(p)
             and a.claims.get('approved') is True)
    if not activation and envelope.tool in p.instruction_tools:
        a = _attestation(rows, agent, p, 'instruction', at,
                         digest=envelope.context.get('instruction_attestation', ''))
        gate('A11-INSTRUCTION', a is not None and a.claims.get('request_digest') == request_digest(envelope))
    return checks


def check_activation(session, agent, *, now=None):
    """Refuse activation when required bound evidence is not current."""
    from kognita.envelope import Envelope
    at = now or utcnow()
    state = load_state(session, [agent.name], at)
    if agent.name not in state:
        return
    p = state[agent.name]['policy']
    checks = admission_checks(Envelope(principal=agent.principal or '', purpose=p.use_case, tool=''),
                              agent, state, at, activation=True)
    failed = [c.check for c in checks if c.result == CheckResult.FAIL]
    if failed:
        raise ValueError(', '.join(failed))


def check_update(session, agent, candidate, deployment, digest):
    """Require candidate-bound evaluation and a current owner update packet."""
    rows = events(session, agent.name)
    p = policy_from(rows)
    if p is None or not p.update_required or agent.deployment_hash == digest:
        return
    packet = latest(rows, 'UPDATE')
    from kognita.evaluation import _latest_applicable, _misses, _gated, _thresholds_for
    result = _latest_applicable(session, candidate)
    valid_eval = result is not None and result.get("deployment_hash") == digest
    entries = _gated(session, candidate)
    if result is not None and entries:
        valid_eval = valid_eval and not _misses(result['metrics'], _thresholds_for(entries))
    if (not packet or packet.get('candidate_digest') != digest
            or packet.get('previous_deployment') != agent.deployment_hash
            or packet.get('policy') != canonical_hash(p.model_dump(mode='json'))
            or not valid_eval):
        raise ValueError('A4-UPDATE: current evaluation, regression evidence, rollback and approval required')


def request_context(headers):
    """Untrusted headers carry proofs, never permissions or attestor keys."""
    import json
    result = {}
    for header, key in (('x-kognita-agent-assertion', 'external_assertion'),
                        ('x-kognita-agent-parents', 'external_parents')):
        if header in headers:
            value = headers[header]
            try:
                if len(value) > 65536:
                    raise ValueError('oversize')
                result[key] = json.loads(value)
                if result[key] is None:
                    result[key] = {}  # explicit null is an invalid supplied proof
            except (ValueError, TypeError, RecursionError):
                result[key] = {}
    if 'x-kognita-instruction-attestation' in headers:
        result['instruction_attestation'] = headers['x-kognita-instruction-attestation']
    return result
