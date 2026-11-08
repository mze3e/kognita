"""Signed KYA exports and evidence-backed, point-in-time profile mappings."""
from datetime import datetime

from sqlmodel import select

from kognita.assurance.records import (
    Record, agent_for, binding, context, emit, events, latest, policy_from,
)
from kognita.canonical import canonical_hash
from kognita.models import Agent, EvidenceEvent, _Authority, as_utc, utcnow
from kognita.vocabulary import EventType


class MaterialManifest(Record):
    """Signed export of identity, live authority, dependencies and intervention."""
    schema_version: str = 'kognita-kya-1'
    generated_at: datetime
    agent: dict
    use_case: str
    authority: list[dict]
    permissions: dict
    dependencies: list[dict]
    intervention: dict
    credential_expiry: str | None
    authority_expiry: list[str]


def export_manifest(session, name, signer, *, evidence, actor_id, now=None):
    """Export and evidence a signed KYA manifest for an approved agent."""
    from kognita.assurance.records import owner
    from kognita.assurance.trust import sign_record
    from kognita.dependencies import _map
    agent = owner(session, name, actor_id)
    rows = events(session, name)
    p = policy_from(rows)
    if p is None or not agent.deployment_hash:
        raise ValueError('A1: approved deployment and assurance policy required')
    at = as_utc(now or utcnow())
    auth = []
    expiries = []
    for row in session.exec(select(_Authority).where(_Authority.grantee == name, _Authority.ended == False)):
        body = row.body
        end = body.get('valid_until')
        start = body.get('valid_from')
        if end and start and as_utc(datetime.fromisoformat(start)) <= at < as_utc(datetime.fromisoformat(end)):
            auth.append({'id': row.id, 'kind': row.kind, 'body': body})
            expiries.append(end)
    dependencies = _map(session)
    manifest = MaterialManifest(generated_at=at, agent={k: getattr(agent, k) for k in (
        'name', 'version', 'purpose', 'owner_exec', 'principal', 'deployment_hash',
        'autonomy_level', 'materiality_tier', 'risk_class', 'model', 'review_date')},
        use_case=p.use_case, authority=sorted(auth, key=lambda r: r['id']),
        permissions={'tools': {k: v for k, v in (agent.tool_allowlist or {}).items() if k != '_deployment'},
                     'systems': agent.permitted_systems, 'data_classes': p.data_classes,
                     'jurisdictions': p.jurisdictions, 'client_impact': p.client_impact},
        dependencies=[r for r in dependencies['paths'] if r['agent'] == name],
        intervention={'kill_switch': agent.kill_switch, 'mechanism': f'agent:{name}:kill-switch'},
        credential_expiry=p.credential_expiry.isoformat() if p.credential_expiry else None,
        authority_expiry=sorted(expiries))
    packet = sign_record(manifest, signer)
    result = {'manifest': packet, 'digest': canonical_hash(packet['document'])}
    emit(session, evidence, name, 'MANIFEST', result, actor_id=actor_id, at=at)
    return result


def verify_manifest(export, public_key):
    """Check the export digest and signature against an independent public key."""
    from kognita.signing import _verify
    packet = export['manifest']
    document = packet['document']
    if (export['digest'] != canonical_hash(document) or document['domain'] != 'kognita-assurance-v1'
            or document['type'] != 'MaterialManifest'):
        raise ValueError('A1: manifest digest/domain mismatch')
    _verify(document, packet['signature'], document['key_id'], public_key, 0)
    return MaterialManifest.model_validate(document['body'])


def compare_manifest(session, name, export, public_key):
    """Report divergence from normalized live identity and observed evidence."""
    from kognita.dependencies import _map
    manifest = verify_manifest(export, public_key)
    agent = agent_for(session, name)
    findings = []
    live_identity = agent.model_dump(mode='json')
    if any(canonical_hash(value) != canonical_hash(live_identity[key])
           for key, value in manifest.agent.items()):
        findings.append({'control': 'A1', 'reason': 'identity or behavioural version changed'})
    # Include live authority divergence as well as observed undeclared calls.
    at = utcnow()
    live = {r.id for r in session.exec(select(_Authority).where(_Authority.grantee == name,
            _Authority.ended == False)) if as_utc(datetime.fromisoformat(r.body['valid_from'])) <=
            as_utc(at) < as_utc(datetime.fromisoformat(r.body['valid_until']))}
    if live != {r['id'] for r in manifest.authority}:
        findings.append({'control': 'A1', 'reason': 'live authority diverges from manifest'})
    mapping = _map(session)
    findings += [dict(control='A1', **f) for f in mapping['findings']
                 if f.get('agent') == name and f['rule_id'] == 'R8-UNDECLARED']
    return {'agent': name, 'manifest_digest': export['digest'], 'findings': findings}


# Theme mappings, deliberately not paragraph-level legal/compliance opinions.
# Profile updates are report metadata and never alter admission policy.
PROFILES = {
    'MAS': {'version': '2026-11-06.1', 'themes': {'AI inventory and accountability': ['A1'],
        'lifecycle validation and change control': ['A3', 'A4'],
        'security and third-party resilience': ['A7', 'A8', 'A9']}},
    'HKMA-ART': {'version': '2026-11-06.1', 'themes': {'agent identity and KYA': ['A1', 'A2'],
        'responsible deployment and runtime oversight': ['A3', 'A5', 'A7']}},
    'FSB': {'version': '2026-11-06.1', 'themes': {'Sound Practice 9: outcome validation': ['A3'],
        'Sound Practice 11: adversarial exercises': ['A8'],
        'third-party change and dependency controls': ['A4', 'A9']}},
    'OCC': {'version': '2026-11-06.1', 'themes': {'AI risk inventory and oversight': ['A1', 'A3'],
        'third-party and operational risk': ['A4', 'A7', 'A9']}},
    'SEC': {'version': '2026-11-06.1', 'themes': {'selected examination theme: instruction authenticity': ['A11'],
        'selected examination theme: technology risk controls': ['A5', 'A9']}},
    'FCA': {'version': '2026-11-06.1', 'themes': {'selected theme: customer outcomes': ['A3'],
        'selected theme: third-party regression and rollback': ['A4']}},
}

_ACTIONS = {'A1': {'MANIFEST'}, 'A2': set(), 'A3': {'PRODUCTION_OUTCOME'},
            'A4': {'UPDATE'}, 'A5': {'ATTESTATION'}, 'A6': {'ATTESTATION'},
            'A7': {'ATTESTATION'}, 'A8': {'DRILL'}, 'A9': {'COMPONENTS'},
            'A10': {'ATTESTATION'}, 'A11': {'ATTESTATION'}}
_KINDS = {'A5': 'monitoring', 'A6': 'privacy', 'A7': 'environment',
          'A10': 'transfer', 'A11': 'instruction'}


def conformance_report(session, name, profile, *, as_of=None):
    """Map verified point-in-time evidence to versioned supervisory themes."""
    if profile not in PROFILES:
        raise ValueError('A12: unknown profile')
    at = as_utc(as_of or utcnow())
    rows = events(session, name, as_of=at)
    all_rows = session.exec(select(EvidenceEvent).where(EvidenceEvent.recorded_at <= at)
                            .order_by(EvidenceEvent.sequence)).all()
    # Verify even when this profile has no assurance evidence.
    from kognita.evidence import _verify_chain
    _verify_chain(session)
    p = policy_from(rows)
    identity = next((r.payload['_identity'] for r in reversed(all_rows)
                     if isinstance(r.payload.get('_identity'), dict)
                     and r.payload['_identity'].get('name') == name), None)
    if identity is None:
        raise ValueError('A12: no evidenced identity at requested time')
    agent = Agent.model_validate(identity)
    mapped = []
    for theme, controls in PROFILES[profile]['themes'].items():
        for control in controls:
            refs = []
            candidates = list(rows)
            if control in _KINDS:
                candidates = [r for r in rows if r.payload.get('action') == 'ATTESTATION' and
                              r.payload['body'].get('document', {}).get('body', {}).get('kind') == _KINDS[control]][-1:]
            elif control == 'A4':
                candidates = [r for r in rows if r.payload.get('action') == 'UPDATE' and
                              r.payload['body'].get('candidate_digest') == agent.deployment_hash][-1:]
            elif control in ('A1', 'A9'):
                candidates = [r for r in rows if r.payload.get('action') in _ACTIONS[control]][-1:]
            for r in candidates:
                if r.payload.get('action') not in _ACTIONS[control]:
                    continue
                body = r.payload['body']
                if control == 'A4' and body.get('candidate_digest') != agent.deployment_hash:
                    continue
                if control == 'A1' and body.get('manifest', {}).get('document', {}).get('body', {}).get('agent', {}).get('deployment_hash') != agent.deployment_hash:
                    continue
                if control in _KINDS:
                    from kognita.assurance.trust import Attestation, verify_record
                    try:
                        a = verify_record(body, Attestation, p.attestors if p else {}, at)
                        if a.kind != _KINDS[control] or {'deployment': a.deployment, 'policy': a.policy} != binding(agent, p):
                            continue
                        if control == 'A6':
                            from kognita.assurance.records import privacy_context
                            from kognita.usecases import load_register
                            entry = load_register(session, as_of=at).get(p.use_case)
                            if a.claims.get('context_digest') != canonical_hash(privacy_context(agent, p, entry.record if entry else {})):
                                continue
                        if control == 'A5':
                            local = {r.event_hash for r in all_rows if r.actor_id == name and
                                     r.event_type in (EventType.TOOL_CALL, EventType.MODEL_CALL, EventType.EGRESS)}
                            if local != set(a.claims.get('action_hashes', [])) or a.claims.get('unexplained_external_actions'):
                                continue
                        if control == 'A10':
                            from kognita.assurance.runtime import transfer_digest
                            if a.claims.get('transfer_digest') != transfer_digest(p) or a.claims.get('approved') is not True:
                                continue
                    except ValueError:
                        continue
                if control == 'A9' and (p is None or body.get('binding') != binding(agent, p)):
                    continue
                if control == 'A9':
                    advisories = [r.payload['body'] for r in all_rows if r.event_type == EventType.ASSURANCE
                                  and r.payload.get('action') == 'ADVISORY']
                    if any(any(c['name'] == adv['name'] and c['version'] == adv['version']
                               for c in body['components']) for adv in advisories):
                        continue
                refs.append(r.sequence)
            if control == 'A2':
                refs = [r.sequence for r in all_rows if r.event_type == EventType.POLICY_DECISION
                        and r.actor_id == name and any(c.get('check') == 'A2-EXTERNAL-TRUST'
                        and c.get('result') == 'pass' for c in r.payload.get('checks', []))]
            if control == 'A8':
                from kognita.assurance.drills import SCENARIOS
                completed = {r.payload['body'].get('scenario'): r for r in rows
                             if r.payload.get('action') == 'DRILL'}
                refs = [completed[k].sequence for k in SCENARIOS] if all(
                    k in completed and completed[k].payload['body'].get('passed') is True
                    for k in SCENARIOS) else []
            mapped.append({'theme': theme, 'control': control, 'implemented_control':
                f'kognita.assurance ({control})', 'status': 'evidenced' if refs else 'gap',
                'evidence_sequences': refs, 'boundary_evidence': _KINDS.get(control),
                'mapping_reference': f'docs/control-frameworks.md (IX); docs/ROADMAP.md ({control})'})
    return {'profile': profile, 'profile_version': PROFILES[profile]['version'], 'agent': name,
            'as_of': at.isoformat(), 'through_sequence': max((r.sequence for r in all_rows), default=0),
            'expectations': mapped, 'interpretation': 'Theme-level evidence mapping; not a legal opinion or certification.'}
