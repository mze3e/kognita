"""Explicit autonomy gates. Metric evidence is necessary but never sufficient."""
from __future__ import annotations

from datetime import timedelta

from sqlmodel import Session

from kognita.envelope import Check
from kognita.models import as_utc, utcnow
from kognita.vocabulary import CheckResult, ActorType, EventType

_VERBS = {"read": 0, "analyse": 1, "recommend": 2, "draft": 3,
          "queue": 4, "send": 4, "execute": 4}


def _checks(envelope, agent, at, usecases=None):
    from kognita.registry import _deployment_hash

    checks = []
    permissions = agent.tool_allowlist if isinstance(agent.tool_allowlist, dict) else {}

    def gate(rule, ok, human=False):
        checks.append(Check(check=rule, regime="INTERNAL", citation=rule,
                            result=(CheckResult.REQUIRES_HUMAN if human and ok else
                                    CheckResult.PASS if ok else CheckResult.FAIL)))

    gate("S1-REVIEW", agent.review_date is not None and as_utc(agent.review_date) >= as_utc(at))
    gate("S1-IDENTITY", isinstance(agent.purpose, str) and bool(agent.purpose.strip())
         and isinstance(agent.use_cases, list) and envelope.purpose in agent.use_cases
         and isinstance(agent.permitted_systems, list) and bool(agent.permitted_systems)
         and bool(permissions) and type(agent.autonomy_level) is int
         and isinstance(agent.environment, str) and bool(agent.environment.strip())
         and isinstance(agent.principal, str) and bool(agent.principal.strip())
         and envelope.principal == agent.principal
         and agent.deployment_date is not None
         and isinstance(agent.model, str) and bool(agent.model.strip()))
    # Context cannot grant permissions. It supplies the behaviour being checked,
    # whose digest must match an owner-approved manifest.
    try:
        digest = _deployment_hash(agent, envelope.context.get("deployment", permissions.get("_deployment")))
    except (ValueError, TypeError):
        digest = None
    gate("S1-DEPLOYMENT", bool(digest) and digest == agent.deployment_hash)
    permission = permissions.get(envelope.tool)
    if not isinstance(permission, dict):
        permission = {}
    verb = permission.get("verb")
    level = permission.get("required_level")
    valid = (verb in _VERBS and type(level) is int and 0 <= level <= 6
             and level >= _VERBS.get(verb, 7))
    gate("S1-TOOL", valid and isinstance(agent.permitted_systems, list) and permission.get("system") in agent.permitted_systems)
    gate("S1-VERB", valid and isinstance(permission.get("permissions"), list) and verb in permission["permissions"])
    maximum = agent.autonomy_level if type(agent.autonomy_level) is int else 0
    gate("S1-LEVEL", valid and level <= maximum and 0 <= maximum <= 6)
    if usecases is not None:
        entry = usecases.get(envelope.purpose)
        gate("S1-RISK-CAP", entry is not None and valid and
             level <= min(entry.maximum_autonomy, {1: 2, 2: 3, 3: 4, 4: 4}.get(entry.proposed_tier, 0)))
    if valid and level == 4:
        gate("S1-L4-APPROVAL", True, human=True)
    if verb in ('draft', 'queue', 'send', 'execute') and permission.get('reversible') is not True:
        gate("S6C-IRREVERSIBLE", True, human=True)
    return checks


def _change(session, agent, level, evidence, actor_id, action, payload):
    from kognita.registry import _deployment_hash
    old = agent.autonomy_level or 0
    candidate = type(agent).model_validate(agent.model_dump())
    candidate.autonomy_level = level
    current = (agent.tool_allowlist or {}).get("_deployment")
    if _deployment_hash(agent, current) == agent.deployment_hash:
        candidate.deployment_hash = _deployment_hash(candidate, current)
    # A reduction cannot approve unrelated, unapproved behaviour changes.
    evidence.emit(session, correlation_id=f"autonomy:{agent.name}",
                  event_type=EventType.POLICY_CHANGE, actor_type=(ActorType.SYSTEM if action == "AUTONOMY_STEP_DOWN" else ActorType.HUMAN),
                  actor_id=actor_id, payload={"action": action, "agent": agent.name,
                  "from": old, "to": level, "citation": "S1-AUTONOMY",
                  "deployment_hash": candidate.deployment_hash,
                  "_identity": candidate.model_dump(mode="json"), **payload})
    agent.autonomy_level = level
    agent.deployment_hash = candidate.deployment_hash
    session.add(agent)
    session.flush()
    return agent


def promote(session: Session, name: str, *, evidence, actor_id: str, validator: str,
            now=None):
    """Promote one level only after every linked use case passes its gates.

    Gates are governed use-case record data under ``_autonomy_gates``, keyed by
    destination level, with minimum_sample, window_days, quality, reliability.
    """
    from kognita.registry import get
    from kognita.usecases import load_register
    from kognita.evidence import verify_chain
    from kognita.registry import _deployment_hash

    verify_chain(session)
    agent = get(session, name)
    if agent is None:
        raise LookupError(name)
    if (_deployment_hash(agent, (agent.tool_allowlist or {}).get("_deployment")) != agent.deployment_hash
            or agent.review_date is None or as_utc(agent.review_date) < as_utc(now or utcnow())):
        raise ValueError("S1-PROMOTION: current reviewed deployment approval required")
    level = (agent.autonomy_level or 0) + 1
    register = load_register(session, as_of=now)
    if actor_id != agent.owner_exec or not validator.strip() or validator.strip() == actor_id.strip():
        raise ValueError("S1-PROMOTION: owner and independent validator required")
    if level > 6 or not agent.use_cases:
        raise ValueError("S1-PROMOTION: no eligible transition")
    results = []
    for usecase in agent.use_cases:
        entry = register.get(usecase)
        gate = entry.record.get("_autonomy_gates", {}).get(str(level)) if entry else None
        if (not gate or entry.status not in ("pilot", "live") or validator not in entry.validators or validator in entry.record.get("builders", []) or
                level > min(entry.maximum_autonomy, {1: 2, 2: 3, 3: 4, 4: 4}.get(entry.proposed_tier, 0))):
            raise ValueError("S1-PROMOTION: missing validated transition gates or risk cap")
        results.append(_metrics(session, agent, usecase, gate, now))
    return _change(session, agent, level, evidence, actor_id, "AUTONOMY_PROMOTED",
                   {"validator": validator, "metrics": results})


def _rows(session, model, field, values, *conditions):
    """Stay below older SQLite's bind-parameter limit for large windows."""
    from sqlmodel import select

    keys = list(values)
    result = []
    for offset in range(0, len(keys), 400):
        result.extend(session.exec(select(model).where(
            field.in_(keys[offset:offset + 400]), *conditions)).all())
    return result


def _metrics(session, agent, usecase, gate, now, *, enforce=True):
    # Reuse the outcome module's definitions, scoped to this agent and window.
    from kognita import outcomes
    from kognita.models import EvidenceEvent, Interaction, InteractionLink, RmReview, RunRecord
    from sqlmodel import select

    at = as_utc(now or utcnow())
    days, minimum = gate.get("window_days"), gate.get("minimum_sample")
    if type(days) is not int or days <= 0 or type(minimum) is not int or minimum <= 0:
        raise ValueError("S1-PROMOTION: positive sample and window required")
    start = at - timedelta(days=days)
    approvals = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.correlation_id.in_([f"deployment:{agent.name}", f"autonomy:{agent.name}"]),
        EvidenceEvent.recorded_at <= start).order_by(EvidenceEvent.sequence.desc()).limit(3)).all()
    approved = next((row for row in approvals if row.payload.get("deployment_hash")), None)
    if approved is None or approved.payload.get("deployment_hash") != agent.deployment_hash:
        raise ValueError("S1-PROMOTION: approved deployment lacks the minimum observation window")
    events = session.exec(select(EvidenceEvent).where(EvidenceEvent.recorded_at >= start,
                                                    EvidenceEvent.recorded_at <= at)).all()
    correlations = {event.correlation_id for event in events
                    if isinstance(event.payload, dict) and
                    event.payload.get("envelope", {}).get("agent_name") == agent.name
                    and event.payload.get("attributes", {}).get("deployment_hash",
                        event.payload.get("deployment_hash")) == agent.deployment_hash}
    links = _rows(session, InteractionLink, InteractionLink.correlation_id, correlations)
    ids = {link.interaction_id for link in links} | correlations
    interactions = _rows(session, Interaction, Interaction.id, ids,
        Interaction.use_case == usecase, Interaction.opened_at >= start, Interaction.opened_at <= at)
    if not interactions:
        raise ValueError("S1-PROMOTION: insufficient observation window")
    reviews = _rows(session, RmReview, RmReview.interaction_id, ids)
    run_ids = {link.run_id for link in links if link.run_id}
    runs = {row.id: row for row in _rows(session, RunRecord, RunRecord.id, run_ids)}
    versions = {(row.payload.get("model_version"), row.payload.get("prompt_template_version"))
                for row in events if row.correlation_id in correlations
                and row.event_type == EventType.MODEL_CALL}
    if len(versions) > 1:
        raise ValueError("S1-PROMOTION: mixed model or prompt versions")
    values = {"_versions": [list(pair) for pair in versions]}
    for metric, key in (("decision quality", "quality"), ("reliability", "reliability")):
        threshold = gate.get(key)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("S1-PROMOTION: metric thresholds required")
        value = outcomes._one_metric(metric, interactions, events, links, reviews, runs, at)
        if (not value.measured or value.sample < minimum or
                (metric == "decision quality" and value.sample != len(interactions)) or
                (enforce and value.value < threshold)):
            raise ValueError("S1-PROMOTION: evidence below threshold")
        if enforce and key == "reliability":
            if value.components.get("unfinished") or value.components.get("exceptions", 0) / value.sample > gate.get("exception_rate", .05):
                raise ValueError("S1-PROMOTION: incomplete reconstruction or excessive exceptions")
            actions = {event.correlation_id for event in events
                       if event.event_type in (EventType.TOOL_CALL, EventType.MODEL_CALL)}
            allowed = {event.correlation_id for event in events
                       if event.correlation_id in correlations and
                       event.event_type == EventType.POLICY_DECISION and
                       event.payload.get("outcome") == "ALLOW"}
            if allowed - actions:
                raise ValueError("S1-PROMOTION: actions are not fully reconstructable")
        values[key] = value.to_dict()
    return values


def _step_down(session, name, *, evidence, now=None):
    """Evaluate governed lower thresholds; at most one drop per rolling window."""
    from kognita.registry import get
    from kognita.usecases import load_register
    from kognita.models import EvidenceEvent
    from sqlmodel import select

    agent = get(session, name)
    if agent is None or not agent.autonomy_level:
        return agent
    at = as_utc(now or utcnow())
    register = load_register(session, as_of=at)
    for usecase in agent.use_cases or []:
        entry = register.get(usecase)
        gate = entry.record.get("_autonomy_step_down") if entry else None
        if not gate:
            continue
        days = gate.get("window_days")
        if type(days) is not int or days <= 0:
            continue
        start = at - timedelta(days=days)
        recent = session.exec(select(EvidenceEvent).where(
            EvidenceEvent.correlation_id == f"autonomy:{name}",
            EvidenceEvent.recorded_at >= start).order_by(EvidenceEvent.sequence.desc()).limit(2)).all()
        if any(row.payload.get("action") == "AUTONOMY_STEP_DOWN" for row in recent):
            continue
        try:
            windows = [_metrics(session, agent, usecase, gate,
                       at - timedelta(days=days * offset), enforce=False)
                       for offset in (2, 1, 0)]
        except ValueError:
            continue
        if any(window["_versions"] != windows[0]["_versions"] for window in windows[1:]):
            continue
        bad = False
        for key in ("quality", "reliability"):
            values = [window[key]["value"] for window in windows]
            bad = bad or (all(value < gate[key] for value in values) or
                          all(a > b for a, b in zip(values, values[1:])))
        if bad:
            lead = entry.record.get("knowledge_lead") or "Knowledge Lead"
            changed = _change(session, agent, agent.autonomy_level - 1, evidence,
                "autonomy-monitor", "AUTONOMY_STEP_DOWN",
                {"metrics": windows, "alerts": [agent.owner_exec, lead],
                 "usecase_id": usecase})
            evidence.emit(session, correlation_id=f"autonomy:{name}",
                event_type=EventType.POLICY_CHANGE, actor_type=ActorType.SYSTEM,
                actor_id="autonomy-monitor", payload={"action": "ALERT",
                "agent": name, "recipients": [agent.owner_exec, lead],
                "citation": "S1-DRIFT"})
            return changed
    return agent


step_down = _step_down

__all__ = ["promote", "step_down"]
