"""Private execution and compensation controls; never execute on worker DB sessions."""
from __future__ import annotations

from threading import Event, Lock, Thread
from time import monotonic
from weakref import WeakKeyDictionary

from kognita.envelope import Check
from kognita.vocabulary import ActorType, CheckResult, EventType

_lock = Lock()
_active = {}
_compensations = WeakKeyDictionary()


def _matches(spec, permission):
    """Runtime capability metadata must match approved deployment behavior."""
    if not isinstance(permission, dict):
        return False
    compensation = getattr(spec, '_compensation_name', getattr(spec.compensation, '__name__', None))
    return (permission.get('verb') == spec.verb
        and permission.get('required_level') == spec.required_level
        and permission.get('reversible', False) == spec.reversible
        and permission.get('max_duration', 30.0) == spec.max_duration
        and (not spec.reversible or permission.get('compensation') == compensation)
        and all(bool(permission.get(key, False)) == getattr(spec, key)
                for key in ('external', 'client_data', 'public_content')))


def _checks(spec):
    if spec.verb in ('draft', 'queue', 'send', 'execute') and not spec.reversible:
        return [Check(check='S6C-IRREVERSIBLE', regime='INTERNAL',
                      result=CheckResult.REQUIRES_HUMAN, citation='S6C-IRREVERSIBLE')]
    return []


def _stop(*, agent=None, run=None, delegation=None):
    with _lock:
        for key, item in tuple(_active.items()):
            if ((agent and key[0] == agent) or (run and key[1] == run)
                    or (delegation and delegation in key[2])):
                item.set()


def _bounded(fn, *, seconds, agent, run, delegation, cancel=None, _authority=(), _check=None):
    """Return an outcome; cancellation is confirmed only by the transport."""
    if seconds <= 0:
        return 'timeout', None
    done, stop = Event(), Event()
    result = []
    finished = []
    key = (agent, run, frozenset([delegation, *_authority]), object())
    def work():
        try:
            result.append((True, fn()))
        except BaseException as exc:
            result.append((False, exc))
        finally:
            finished.append(monotonic())
            done.set()
    with _lock:
        _active[key] = stop
    try:
        deadline = monotonic() + seconds
        Thread(target=work, daemon=True).start()
        while not done.wait(min(0.01, max(0, deadline - monotonic()))):
            if _check is not None:
                try:
                    if not _check():
                        stop.set()
                except Exception:
                    stop.set()
            timeout = monotonic() >= deadline
            if stop.is_set() or timeout:
                confirmed = False
                if cancel:
                    cancellation_done = Event()
                    confirmation = []
                    def request_cancel():
                        try:
                            confirmation.append(cancel() is True)
                        except Exception:
                            confirmation.append(False)
                        finally:
                            cancellation_done.set()
                    Thread(target=request_cancel, daemon=True).start()
                    cancellation_done.wait(min(0.01, max(0, deadline - monotonic())))
                    confirmed = bool(confirmation and confirmation[0])
                if done.is_set():
                    if timeout:
                        return 'timeout', None
                    if confirmed and result and not result[0][0]:
                        return ('timeout' if timeout else 'cancelled'), None
                    break
                return ('timeout' if timeout else 'cancelled' if confirmed else 'cancellation_requested'), None
        if finished[0] > deadline:
            return 'timeout', None
        ok, value = result[0]
        if not ok:
            return 'failed', value
        return 'completed', value
    finally:
        with _lock:
            _active.pop(key, None)


def _register_compensation(session, run_id, spec, envelope, evaluation, evidence):
    from kognita.models import CompensatingAction
    from kognita.retention import RetentionStore
    digest = RetentionStore().retain_value(session, {**evaluation.to_dict(), 'context': {k: v for k, v in envelope.context.items() if k != '_agent_credential'}}, kind='compensation',
                use_case=envelope.purpose, correlation_id=evaluation.request_id)
    row = CompensatingAction(run_id=run_id, name=spec.name, citation='S6C-RECOVER:' + digest)
    session.add(row)
    session.flush()
    _compensations.setdefault(session.get_bind(), {})[(row.id, row.citation)] = (spec.compensation, envelope, evaluation)
    evidence.emit(session, correlation_id=evaluation.request_id, event_type=EventType.INTERVENTION,
                  actor_type=ActorType.SYSTEM, actor_id='governance-pdp',
                  payload={'action': 'register_compensating_action', 'run_id': run_id,
                           'compensating_action_id': row.id, 'name': spec.name, 'citation': 'S6C-RECOVER'})


def _recover(session, steps, evidence, correlation_id, registry=None):
    from sqlmodel import select
    from kognita.models import EvidenceEvent
    # Indexed correlation bounds the search to this run's interaction.
    events = session.exec(select(EvidenceEvent).where(EvidenceEvent.correlation_id == correlation_id,
                         EvidenceEvent.event_type == EventType.INTERVENTION)).all()
    completed = {e.payload.get('compensating_action_id') for e in events
                 if e.payload.get('action') == 'compensation' and e.payload.get('outcome') == 'completed'}
    for step in steps:
        if step['id'] in completed:
            step['outcome'] = 'already_completed'
            continue
        callback = _compensations.get(session.get_bind(), {}).get((step['id'], step['citation']))
        if callback is None and registry is not None and step['citation'].startswith('S6C-RECOVER:'):
            import json
            from kognita.retention import RetentionStore
            from kognita.envelope import Envelope
            from kognita.tools import _evaluation_from_continuation
            retained = RetentionStore().read(session, step['citation'].split(':', 1)[1])
            if retained and step['name'] in registry:
                spec = registry.get(step['name'])
                if spec.reversible and callable(spec.compensation):
                    payload = json.loads(retained)
                    envelope = Envelope(**payload['envelope'], context=payload.get('context', {}))
                    callback = (spec.compensation, envelope, _evaluation_from_continuation(payload, envelope))
        outcome = 'unavailable'
        if callback:
            fn, envelope, evaluation = callback
            try:
                fn(envelope, evaluation, session)
                outcome = 'completed'
            except Exception:
                outcome = 'failed'
        step['outcome'] = outcome
        evidence.emit(session, correlation_id=correlation_id, event_type=EventType.INTERVENTION,
                      actor_type=ActorType.SYSTEM, actor_id='governance-pdp',
                      payload={'action': 'compensation', 'compensating_action_id': step['id'],
                               'name': step['name'], 'outcome': outcome, 'citation': 'S6C-RECOVER'})
