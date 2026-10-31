"""R8 current declarations compared with locally verified call evidence."""
from sqlmodel import select as _select

from kognita.evidence import _verify_chain, ChainBreak as _ChainBreak
from kognita.models import Agent as _Agent, EvidenceEvent as _Event, Interaction as _Interaction
from kognita.usecases import load_register as _load_register


def _map(session):
    """Return deterministic edges and findings; never infer calls from permissions."""
    try:
        _verify_chain(session)
    except _ChainBreak as exc:
        raise ValueError('R8-UNDECLARED: dependency evidence verification failed') from exc
    register = _load_register(session)
    cards = {card.card_id: card for card in register.model_cards}
    declared, observed = set(), set()
    paths = []
    agents = session.exec(_select(_Agent).order_by(_Agent.name)).all()
    for agent in agents:
        declared.update((name, agent.name, 'agent', agent.name) for name in (agent.use_cases or []))
    for entry in register.entries:
        declared.update((entry.usecase_id, name, 'agent', name) for name in entry.agents)
        for agent in agents:
            if entry.usecase_id not in (agent.use_cases or []) and agent.name not in entry.agents:
                continue
            declared.add((entry.usecase_id, agent.name, 'agent', agent.name))
            if not entry.approved_models:
                paths.append({'business_service': entry.record.get('business_process') or entry.usecase_id,
                    'use_case': entry.usecase_id, 'agent': agent.name, 'model': None,
                    'provider': None, 'cloud': entry.cloud or None, 'framework': entry.framework or None,
                    'orchestration': entry.orchestration or None, 'vector_platform': entry.vector_platform or None,
                    'tools': sorted(agent.tool_allowlist or {}),
                    'data_sources': sorted(set(entry.approved_sources) | set(entry.record.get('data_sources') or []))})
            for provider in entry.model_providers:
                declared.add((entry.usecase_id, agent.name, 'provider', provider))
            for card_id in entry.approved_models:
                card = cards.get(card_id)
                declared.add((entry.usecase_id, agent.name, 'model', card_id))
                if card:
                    declared.add((entry.usecase_id, agent.name, 'provider', card.provider))
                paths.append({'business_service': entry.record.get('business_process') or entry.usecase_id,
                    'use_case': entry.usecase_id, 'agent': agent.name, 'model': card_id,
                    'provider': card.provider if card else None,
                    'cloud': entry.cloud or (card.record.get('cloud') if card else None),
                    'framework': entry.framework or None, 'orchestration': entry.orchestration or None,
                    'vector_platform': entry.record.get('vector_platform'),
                    'tools': sorted(agent.tool_allowlist or {}),
                    'data_sources': sorted(set(entry.approved_sources) | set(entry.record.get('data_sources') or []))})
            for kind, targets in (
                ('tool', agent.tool_allowlist or {}),
                ('source', set(entry.approved_sources) | set(entry.record.get('data_sources') or [])),
                ('system', agent.permitted_systems or []),
            ):
                declared.update((entry.usecase_id, agent.name, kind, target) for target in targets)
    events = session.exec(_select(_Event).order_by(_Event.sequence)).all()
    contexts = {}
    for event in events:
        envelope = event.payload.get('envelope')
        if isinstance(envelope, dict):
            context = (envelope.get('purpose') or '', envelope.get('agent_name') or event.actor_id)
            contexts.setdefault(event.correlation_id, set()).add(context)
    interactions = {row.id: row.use_case for row in session.exec(_select(_Interaction))}
    unattributed = []
    for event in events:
        kind = event.event_type.value
        payload = event.payload
        if kind not in ('MODEL_CALL', 'TOOL_CALL', 'RETRIEVAL'):
            continue
        if kind == 'MODEL_CALL' and payload.get('sent') is False:
            continue
        choices = contexts.get(event.correlation_id, set())
        if len(choices) == 1:
            use_case, agent = next(iter(choices))
            if event.actor_id and event.actor_id != agent:
                use_case, agent = '', event.actor_id
        else:
            use_case, agent = interactions.get(event.correlation_id, ''), event.actor_id
        if len(choices) > 1:
            use_case = ''
        if not use_case or not agent:
            unattributed.append({'rule_id': 'R8-UNDECLARED', 'sequence': event.sequence,
                'reason': 'call has missing or ambiguous use case or agent attribution'})
        observed.add((use_case, agent, 'agent', agent))
        if payload.get('destination'):
            observed.add((use_case, agent, 'system', payload['destination']))
        if kind == 'MODEL_CALL':
            card_id = payload.get('model_card')
            matches = [card.card_id for card in cards.values()
                if card.model == payload.get('model') and card.provider == payload.get('provider')
                and card.version == payload.get('model_version')]
            card = cards.get(card_id)
            if card and any(payload.get(key) and payload[key] != getattr(card, key)
                            for key in ('provider', 'model')):
                card_id = None
            if card and payload.get('model_version') and payload['model_version'] != card.version:
                card_id = None
            target = card_id or (matches[0] if len(matches) == 1 else
                '/'.join(str(payload.get(key) or '<unknown>') for key in ('provider', 'model', 'model_version')))
            observed.add((use_case, agent, 'provider', payload.get('provider') or '<unknown>'))
            observed.add((use_case, agent, 'model', target or '<unknown>'))
        elif kind == 'TOOL_CALL':
            observed.add((use_case, agent, 'tool', payload.get('tool') or '<unknown>'))
        else:
            for item in payload.get('items', []):
                observed.add((use_case, agent, 'source', item.get('source_label') or '<unknown>'))
    def _edge(value):
        return dict(zip(('use_case', 'agent', 'kind', 'target'), value))
    findings = unattributed + [dict(rule_id=rule, **_edge(value))
        for rule, values in (('R8-UNDECLARED', observed - declared), ('R8-UNUSED', declared - observed))
        for value in sorted(values)]
    return {'use_cases': [{'use_case': entry.usecase_id,
                'business_service': entry.record.get('business_process') or entry.usecase_id,
                'business_criticality': entry.business_criticality,
                'provider': sorted(set(entry.model_providers) | {cards[name].provider
                    for name in entry.approved_models if name in cards}),
                'cloud': entry.cloud, 'framework': entry.framework,
                'vector_platform': entry.vector_platform, 'orchestration': entry.orchestration,
                'unresolved_models': sorted(set(entry.approved_models) - set(cards)),
                'agents': sorted(set(entry.agents) | {agent.name for agent in agents
                    if entry.usecase_id in (agent.use_cases or [])}),
                'models': sorted(entry.approved_models),
                'data_sources': sorted(set(entry.approved_sources) | set(entry.record.get('data_sources') or []))}
            for entry in register.entries],
        'scope': 'Current registers versus all retained verified call evidence. Unused means unobserved in this evidence, not safe to remove. Infrastructure metadata absent from registers is unknown; permissions do not prove calls.',
        'paths': paths, 'declared': [_edge(value) for value in sorted(declared)],
        'observed': [_edge(value) for value in sorted(observed)], 'findings': findings}


def _concentration(session):
    """Count distinct critical use cases per declared dependency, including unknowns."""
    mapped = _map(session)
    entries = [entry for entry in mapped['use_cases']
               if entry['business_criticality'] == 'critical']
    total = len(entries)
    dimensions = {}
    for name in ('provider', 'cloud', 'framework', 'vector_platform', 'orchestration'):
        groups, unknown = {}, []
        for entry in entries:
            values = entry[name] if name == 'provider' else [entry[name]]
            if any(not isinstance(value, str) for value in values):
                raise ValueError(f'R9: invalid {name} declaration')
            values = {value.strip() for value in values if value.strip()}
            if not values or (name == 'provider' and entry['unresolved_models']):
                unknown.append(entry['use_case'])
            for value in values:
                groups.setdefault(value, set()).add(entry['use_case'])
        dimensions[name] = {
            'dependencies': [{'dependency': value, 'use_cases': sorted(ids),
                              'count': len(ids), 'share': len(ids) / total}
                             for value, ids in sorted(groups.items())],
            'unknown_use_cases': sorted(unknown),
            'unknown_count': len(unknown),
            'unknown_share': len(unknown) / total if total else None,
        }
    return {'scope': 'Current registered use cases with business_criticality=critical, including all statuses. '
            'Shares count distinct use cases per declared dependency, not traffic or permissions. '
            'Dependencies may overlap; missing declarations remain unknown. No critical use cases means shares are undefined.',
            'total_critical_use_cases': total, 'dimensions': dimensions,
            'findings': mapped['findings']}
