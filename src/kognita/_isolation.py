"""Content provenance and authenticated transaction gates (trusted host boundary)."""
from datetime import datetime

from sqlmodel import select

from kognita.canonical import canonical_hash
from kognita.envelope import Check
from kognita.models import EvidenceEvent, RunRecord, as_utc
from kognita.vocabulary import ActorType, CheckResult, Classification, EventType


def _validate(agent):
    permissions = [p for k, p in (agent.tool_allowlist or {}).items()
                   if not k.startswith('_') and isinstance(p, dict)]
    if any(type(p.get(key, False)) is not bool for p in permissions
           for key in ('external', 'client_data', 'public_content')):
        raise ValueError('S5-TOOL-LABELS: content flags must be booleans')
    from math import isfinite
    for p in permissions:
        reversible = p.get('reversible', False)
        duration = p.get('max_duration', 30.0)
        if (type(reversible) is not bool or (not reversible and p.get('compensation') is not None) or (reversible and not (
                isinstance(p.get('compensation'), str) and p['compensation'].strip()))):
            raise ValueError('S6C-REVERSIBILITY: compensation identifier required')
        if (isinstance(duration, bool) or not isinstance(duration, (float, int))
                or not isfinite(duration) or duration <= 0):
            raise ValueError('S6C-DURATION: positive finite maximum required')
    internet = any(p.get('external') is True or p.get('system') in ('internet', 'web', 'external') for p in permissions) or any(
        s.lower() in ('internet', 'web', 'external') for s in (agent.permitted_systems or []))
    writes = any(p.get('verb') in ('draft', 'queue', 'send', 'execute') for p in permissions)
    if internet and writes:
        raise ValueError('S5-SEPARATION: internet access and internal write authority conflict')


def _emit(session, evidence, correlation_id, action, **payload):
    return evidence.emit(session, correlation_id=correlation_id,
        event_type=EventType.POLICY_CHANGE, actor_type=ActorType.SYSTEM,
        actor_id='governance-pdp', payload={'action': action, **payload})


def _ingest(session, run, evidence, *, external=False, client_data=False,
            text='', correlation_id='', _agent_name=None):
    from kognita.classify import PatternClassifier
    from kognita.tools import _refresh_run, _save_run

    classifier = PatternClassifier()
    injection = classifier._injection(text)
    # Caller-supplied classification can never erase detected client data.
    if run is not None:
        client_data = client_data or classifier.classify(text) in (Classification.C2, Classification.C3)
    external = external or injection
    if run is not None:
        stored = session.get(RunRecord, run.id)
        old = {label: bool(getattr(stored, label)) if stored else False
               for label in ('external', 'client_data')}
        _refresh_run(session, run)
        run.external |= external
        run.client_data |= client_data
        added = [label for label in old if getattr(run, label) and not old[label]]
        # Evidence links the run by foreign key; persist it before emitting.
        _save_run(session, run)
        if added:
            _emit(session, evidence, correlation_id or run.id, 'CONTENT_TAINTED',
                  citation='S5-TAINT', run_id=run.id, labels=added)
    if injection:
        from kognita.registry import get
        agent = get(session, _agent_name) if _agent_name else None
        recipients = ['security'] + ([agent.owner_exec] if agent else [])
        _emit(session, evidence, correlation_id or (run.id if run else 'injection'),
              'INJECTION_DETECTED', citation='S5-INJECTION', injection=True,
              alert=True, recipients=recipients, run_id=run.id if run else None)


def _step_up(session, envelope, *, evidence, actor_id, authenticated_workflow,
             transaction_context, independent_authorization, expires_at):
    """Called only by the authenticated host, never from model/channel arguments."""
    if (authenticated_workflow is not True or not transaction_context or
        not isinstance(independent_authorization, str) or not independent_authorization.strip() or independent_authorization in (envelope.principal, envelope.agent_name)
        or actor_id != envelope.principal or type(envelope.assurance_level) is not int
        or envelope.assurance_level < 2 or as_utc(expires_at) <= as_utc(evidence.clock())):
        raise ValueError('S5-STEP-UP: authenticated workflow and independent authorization required')
    _emit(session, evidence, 'step-up:' + canonical_hash(envelope.to_dict()), 'STEP_UP',
          citation='S5-STEP-UP', principal=actor_id, authenticated_workflow=True,
          transaction_context=transaction_context,
          independent_authorization=independent_authorization,
          expires_at=expires_at.isoformat())


def _tool_checks(session, envelope, spec, run, evidence, *, at=None):
    checks = []
    def deny(rule):
        checks.append(Check(check=rule, citation=rule, regime='INTERNAL', result=CheckResult.FAIL))
    if (spec.external or spec.client_data) and run is None:
        deny('S5-RUN-REQUIRED')
    checks.extend(_assurance_checks(session, envelope, run))
    if (run and run.external or spec.external) and (spec.verb in ('draft', 'queue', 'send', 'execute') or spec.required_level > 1):
        deny('S5-EXTERNAL-AUTHORITY')
    if (run and run.client_data or spec.client_data or 'client' in envelope.all_subjects()) and spec.public_content:
        deny('S5-PUBLIC-CONTENT')
    from kognita.classify import PatternClassifier
    if PatternClassifier()._injection(str(envelope.arguments)):
        deny('S5-INJECTION')
    if spec.required_level >= 4 or spec.verb in ('queue', 'send', 'execute'):
        row = session.exec(select(EvidenceEvent).where(
            EvidenceEvent.correlation_id == 'step-up:' + canonical_hash(envelope.to_dict())
        ).order_by(EvidenceEvent.sequence.desc()).limit(1)).first()
        if not _step_up_valid(row, envelope, at or evidence.clock()):
            deny('S5-STEP-UP')
    return checks


def _assurance(headers):
    # A header is only a claim. High-risk calls still need exact step-up evidence.
    value = headers.get('assurance_level', '0')
    return int(value) if value in ('0', '1', '2', '3') else 0


def _assurance_checks(session, envelope, run):
    from kognita.tools import _refresh_run, _save_run
    if run is None:
        return []
    _refresh_run(session, run)
    raised = run.assurance_level is not None and envelope.assurance_level > run.assurance_level
    run.assurance_level = min(run.assurance_level, envelope.assurance_level) if run.assurance_level is not None else envelope.assurance_level
    _save_run(session, run)
    return [Check(check='S5-ASSURANCE-RELAY', citation='S5-ASSURANCE-RELAY',
                  regime='INTERNAL', result=CheckResult.FAIL)] if raised else []


def _step_up_valid(row, envelope, at):
    if row is None or envelope.assurance_level < 2:
        return False
    body = row.payload
    try:
        authorizer = body.get('independent_authorization')
        return (row.event_type == EventType.POLICY_CHANGE and body.get('action') == 'STEP_UP'
            and body.get('citation') == 'S5-STEP-UP'
            and body.get('authenticated_workflow') is True
            and body.get('principal') == envelope.principal
            and bool(body.get('transaction_context'))
            and isinstance(authorizer, str) and bool(authorizer.strip())
            and authorizer not in (envelope.principal, envelope.agent_name)
            and as_utc(row.recorded_at) <= as_utc(at)
            and as_utc(datetime.fromisoformat(body['expires_at'])) > as_utc(at))
    except (KeyError, ValueError, TypeError):
        return False
