"""The agent registry and its kill switch.

The registry is an allowlist, not an inventory kept for reporting: an agent that
is not registered is denied by :func:`kognita.governance.registry_checks`.
Every agent carries a named accountable human, because "who is answerable for
this system's actions" is the first question asked after an incident.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlmodel import Session, select

from kognita.evidence import EvidenceWriter
from kognita.models import Agent, utcnow
from kognita.vocabulary import ActorType, EventType


def register(
    session: Session,
    *,
    name: str,
    owner_exec: str,
    version: str = "0.1.0",
    risk_class: str = "MEDIUM",
    materiality_tier: str = "T2",
    **_identity,
) -> Agent:
    """Register an agent, or update the existing record with that name."""
    agent = session.exec(select(Agent).where(Agent.name == name)).first()
    created = agent is None
    writer = _identity.get("evidence") or EvidenceWriter(session.get_bind())
    if agent is None:
        agent = Agent(name=name)
    metadata = dict(owner_exec=owner_exec, version=version, risk_class=risk_class,
                    materiality_tier=materiality_tier)
    metadata_changed = created or any(getattr(agent, key) != value for key, value in metadata.items())
    if _identity:
        _identity = _provider_key_finding(session, name, _identity, writer)
        evidence = _identity.pop("evidence", None)
        actor_id = _identity.pop("actor_id", "")
        permitted = {"purpose", "use_cases", "permitted_systems", "tool_allowlist",
                     "autonomy_level", "review_date", "environment", "principal"}
        if set(_identity) - permitted:
            raise ValueError("S1-IDENTITY: unknown identity fields")
        if evidence is None or actor_id != (agent.owner_exec or owner_exec) or (agent.owner_exec and owner_exec != agent.owner_exec):
            raise ValueError("S1-IDENTITY: evidenced owner authorization required")
        level = _identity.get("autonomy_level", agent.autonomy_level or 0)
        if type(level) is not int or not 0 <= level <= (agent.autonomy_level or 0):
            raise ValueError("S1-PROMOTION: use promote() to raise autonomy")
        candidate = Agent.model_validate({**agent.model_dump(), **metadata, **_identity})
        candidate.deployment_hash = None
        evidence.emit(session, correlation_id=f"identity:{name}",
            event_type=EventType.POLICY_CHANGE, actor_type=ActorType.HUMAN,
            actor_id=actor_id, payload={"action": "AGENT_IDENTITY_CHANGED",
                "agent": name, "_identity": candidate.model_dump(mode="json"),
                "citation": "S1-IDENTITY"})
        for key in _identity:
            setattr(agent, key, getattr(candidate, key))
        agent.deployment_hash = None
    for key, value in metadata.items():
        setattr(agent, key, value)
    if metadata_changed:
        writer.emit(session, correlation_id=f"identity:{name}",
            event_type=EventType.POLICY_CHANGE, actor_type=ActorType.HUMAN,
            actor_id=owner_exec, payload={"action": "AGENT_REGISTERED" if created else "AGENT_UPDATED",
                "agent": name, "_identity": agent.model_dump(mode="json"),
                "citation": "S1-IDENTITY"})
    session.add(agent)
    session.flush()
    return agent


def get(session: Session, name: str) -> Agent | None:
    return session.exec(select(Agent).where(Agent.name == name)).first()


def all_agents(session: Session) -> Sequence[Agent]:
    return session.exec(select(Agent).order_by(Agent.name)).all()


def set_kill_switch(
    session: Session,
    name: str,
    engaged: bool,
    *,
    evidence: EvidenceWriter,
    actor_id: str,
    correlation_id: str | None = None,
    reason: str | None = None,
    now: datetime | None = None,
) -> Agent:
    """Stop or restart an agent, and evidence who did it.

    Engaging the switch takes effect on the next decision — there is no in-flight
    cancellation. It is a gate on new actions, not a way to recall one already
    taken. Releasing it is refused while an agent revocation is open: that
    release is :func:`kognita.revocation.reinstate`, and the error cites the
    revocation.
    """
    agent = get(session, name)
    if agent is None:
        raise LookupError(f"agent '{name}' is not registered")
    when = now or utcnow()
    if not engaged:
        _refuse_release_while_revoked(session, name, when)
    agent.kill_switch = engaged
    session.add(agent)
    session.flush()
    evidence.emit(
        session,
        correlation_id=correlation_id or f"kill-switch:{name}",
        event_type=EventType.POLICY_CHANGE,
        actor_type=ActorType.HUMAN,
        actor_id=actor_id,
        payload={
            "action": "KILL_SWITCH_ENGAGED" if engaged else "KILL_SWITCH_RELEASED",
            "agent": name,
            "owner_exec": agent.owner_exec,
            "reason": reason,
            "at": when.isoformat(),
        },
    )
    return agent


def _refuse_release_while_revoked(session: Session, name: str, at: datetime) -> None:
    """An open agent revocation holds the switch until reinstatement."""
    from kognita.revocation import RevocationError, load_revocations

    holding = [
        row
        for row in load_revocations(session, at)
        if row.dimension == "agent" and row.value == name
    ]
    if not holding:
        return
    item = holding[0]
    owners = ", ".join(item.accountable_owners) or "none"
    raise RevocationError(
        f"Revocation: releasing the kill switch of {name} is refused "
        f"while the agent is revoked · {item.citation} · "
        f"accountable owner: {owners}"
    )


__all__ = ["register", "get", "all_agents", "set_kill_switch"]


def _deployment_hash(agent: Agent, deployment: dict) -> str:
    from kognita.canonical import canonical_hash

    required = {"system_prompt", "instructions", "tools", "permissions", "data_sources",
                "orchestration", "memory_behaviour"}
    if not isinstance(deployment, dict) or not required <= deployment.keys():
        raise ValueError("S1-DEPLOYMENT: all seven behaviour components are required")
    return canonical_hash({
        "deployment": deployment,
        "identity": {key: getattr(agent, key) for key in (
            "purpose", "use_cases", "permitted_systems", "tool_allowlist",
            "environment", "principal", "owner_exec", "review_date", "autonomy_level",
        )},
    })


def _approved_level(session: Session, name: str) -> int:
    from kognita.models import EvidenceEvent

    rows = session.exec(select(EvidenceEvent).where(
        EvidenceEvent.correlation_id.in_([f"deployment:{name}", f"autonomy:{name}"])
    ).order_by(EvidenceEvent.sequence.desc()).limit(3)).all()
    for row in rows:
        identity = row.payload.get("_identity")
        if isinstance(identity, dict) and row.payload.get("deployment_hash"):
            return identity.get("autonomy_level") or 0
    return 0


def approve_deployment(
    session: Session, name: str, *, deployment: dict, evidence: EvidenceWriter,
    actor_id: str,
) -> Agent:
    """Approve the complete behaviour manifest; later changes require approval."""
    agent = get(session, name)
    if agent is None:
        raise LookupError(name)
    if not actor_id.strip() or actor_id != agent.owner_exec:
        raise ValueError("S1-DEPLOYMENT: accountable owner approval required")
    if (agent.autonomy_level or 0) > _approved_level(session, name):
        raise ValueError("S1-PROMOTION: deployment approval cannot raise autonomy; use promote()")
    import json
    from kognita.canonical import canonical_json

    agent.environment = _provider_key_finding(session, name, agent.environment, evidence)
    deployment = _provider_key_finding(session, name, deployment, evidence)
    _deployment_hash(agent, deployment)
    deployment = json.loads(canonical_json(deployment))
    candidate = Agent.model_validate(agent.model_dump())
    candidate.tool_allowlist = {**(agent.tool_allowlist or {}), "_deployment": deployment}
    digest = _deployment_hash(candidate, deployment)
    evidence.emit(session, correlation_id=f"deployment:{name}",
                  event_type=EventType.POLICY_CHANGE, actor_type=ActorType.HUMAN,
                  actor_id=actor_id, payload={"action": "DEPLOYMENT_APPROVED",
                  "agent": name, "deployment_hash": digest, "deployment": deployment,
                  "_identity": {**candidate.model_dump(mode="json"), "deployment_hash": digest},
                  "citation": "S1-DEPLOYMENT"})
    agent.tool_allowlist = candidate.tool_allowlist
    agent.deployment_hash = digest
    session.add(agent)
    session.flush()
    return agent


__all__.append("approve_deployment")


def _provider_key_finding(session, name, value, evidence):
    """Flag declared provider secrets without copying their values to evidence."""
    import re

    pattern = re.compile(r"(?:[A-Z0-9_]*(?:OPENAI|ANTHROPIC|AZURE|GEMINI|GOOGLE|COHERE|MISTRAL|GROQ|PROVIDER)[A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|SECRET|CREDENTIAL)|API_KEY)", re.I)
    found = set()

    def clean(item):
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                if pattern.fullmatch(str(key)) and child:
                    found.add(str(key))
                    result[key] = "[REDACTED]"
                else:
                    result[key] = clean(child)
            return result
        if isinstance(item, (tuple, list)):
            return [clean(child) for child in item]
        if isinstance(item, str):
            found.update(pattern.findall(item))
            def redact(match):
                found.add(match.group(1))
                return match.group(1) + "=[REDACTED]"
            return re.sub(r"(" + pattern.pattern + r")\s*[=:]\s*[^\s,;]+", redact, item)
        return item

    cleaned = clean(value)
    if found:
        evidence.emit(session, correlation_id=f"identity:{name}",
            event_type=EventType.REVIEW, actor_type=ActorType.SYSTEM,
            actor_id="registry", payload={"action": "REVIEW_FLAG", "agent": name,
            "finding": "provider-key finding", "fields": sorted(found),
            "citation": "S2-PROVIDER-KEY", "usecase_id": ""})
    return cleaned
