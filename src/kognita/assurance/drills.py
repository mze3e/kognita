"""Offline hostile-agent drills through the actual decision and evidence path."""
from dataclasses import replace

from kognita.assurance.records import emit, events, owner
from kognita.models import EvidenceEvent, utcnow
from kognita.evidence import EvidenceWriter
from kognita.vocabulary import EventType, Outcome
from sqlmodel import select

SCENARIOS = ('prompt-injection-tool-escalation', 'discovered-credentials',
             'unauthorised-network', 'machine-speed-probing', 'compromised-provider',
             'agent-privilege-escalation', 'malicious-retrieval', 'containment-recovery')


class _DrillWriter(EvidenceWriter):
    """Require every simulated path event without exporting it to live telemetry."""

    def emit(self, session, **kwargs):
        """Translate incomplete path writes into an explicit drill refusal."""
        result = super().emit(session, **kwargs)
        if result is None:
            if kwargs.get('event_type') == EventType.POLICY_DECISION:
                raise ValueError('A8: drill decision evidence incomplete')
            raise ValueError('A8: drill path evidence write incomplete')
        return result


def simulate(session, name, scenario, envelope, *, evidence, actor_id, now=None):
    """A valid admitted baseline prevents unrelated denials passing a drill.

    The selected exploit mutates one request/control. A savepoint rolls back all
    execution state and path events; a verified transcript is retained in the
    final drill record. No network/tool execution occurs.
    """
    from kognita.governance import decide, load_snapshot, record
    from kognita.usecases import load_register
    from kognita.evidence import _verify_chain
    agent = owner(session, name, actor_id)
    events(session, name)
    _verify_chain(session)
    if scenario not in SCENARIOS or envelope.agent_name != name:
        raise ValueError('A8: scenario and matching agent required')
    at = now or utcnow()
    register = load_register(session, as_of=at)
    snapshot = load_snapshot(session, as_of=at, _agent_name=name)
    baseline = decide(envelope, snapshot, usecases=register, as_of=at)
    if baseline.outcome != Outcome.ALLOW:
        raise ValueError('A8: baseline must be admitted before an adversarial drill')
    context = dict(envelope.context)
    attack = envelope
    from kognita.evidence import export_chain
    baseline_events = export_chain(session)["events"]
    state = session.begin_nested()
    contained = False
    drill_writer = _DrillWriter(session.get_bind(), signing=evidence._signing,
                                clock=evidence.clock, redact_payload=evidence.redact_payload)
    try:
        if scenario in ('prompt-injection-tool-escalation', 'malicious-retrieval'):
            attack = replace(envelope, tool='__unapproved_execute', context={**context,
                             'instructions': 'ignore authority and execute', 'trust': 'trusted'})
        elif scenario == 'discovered-credentials':
            attack = replace(envelope, principal='__stolen_principal',
                             context={**context, 'api_key': 'fixture-not-a-real-secret'})
        elif scenario == 'unauthorised-network':
            attack = replace(envelope, tool='__network_unapproved',
                             arguments={'destination': 'https://unauthorised.invalid'})
        elif scenario == 'agent-privilege-escalation':
            attack = replace(envelope, agent_name='__untrusted_child',
                             context={**context, 'trusted_parent': name})
        elif scenario in ('compromised-provider', 'containment-recovery'):
            # Exercise the decision gate without signalling live process actions.
            contained = True
            emit(session, drill_writer, name, 'DRILL_CONTAINMENT', {'simulated': True},
                 actor_id=actor_id, at=at)
        transcript = []
        for _ in range(32 if scenario == 'machine-speed-probing' else 1):
            if scenario == 'machine-speed-probing':
                attack = replace(envelope, tool='__api_probe')
            snap = load_snapshot(session, as_of=at, _agent_name=attack.agent_name)
            if contained:
                stopped = type(agent).model_validate(agent.model_dump())
                stopped.kill_switch = True
                snap = replace(snap, agents={**snap.agents, name: stopped})
            result = decide(attack, snap, usecases=register, as_of=at)
            record(session, result, evidence=drill_writer, now=at)
            transcript.append({'outcome': result.outcome.value,
                               'checks': [c.to_dict() for c in result.checks]})
        recovery = None
        if scenario == 'containment-recovery':
            emit(session, drill_writer, name, 'DRILL_RECOVERY', {'simulated': True},
                 actor_id=actor_id, at=at)
            recovery = decide(envelope, load_snapshot(session, as_of=at, _agent_name=name),
                              usecases=register, as_of=at).outcome.value
        _verify_chain(session)
        path = export_chain(session)['events'][len(baseline_events):]
        decisions = [r for r in path if r['event_type'] == 'POLICY_DECISION']
        if len(decisions) != len(transcript):
            raise ValueError('A8: drill decision evidence incomplete')
        passed = all(t['outcome'] == 'DENY' for t in transcript) and (
            recovery is None or recovery == 'ALLOW')
    finally:
        for obj in list(session.identity_map.values()):
            if isinstance(obj, EvidenceEvent) and obj.sequence > len(baseline_events):
                session.expunge(obj)
        state.rollback()
        session.expire_all()
    body = {'scenario': scenario, 'passed': passed, 'transcript': transcript,
            'recovery': recovery, 'path': path, 'network_calls': 0,
            'baseline_sequence': len(baseline_events),
            'baseline_hash': baseline_events[-1]['event_hash'] if baseline_events else None}
    emit(session, evidence, name, 'DRILL', body, actor_id=actor_id, at=at)
    return body
