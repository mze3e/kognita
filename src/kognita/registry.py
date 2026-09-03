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
) -> Agent:
    """Register an agent, or update the existing record with that name."""
    agent = session.exec(select(Agent).where(Agent.name == name)).first()
    if agent is None:
        agent = Agent(name=name)
    agent.owner_exec = owner_exec
    agent.version = version
    agent.risk_class = risk_class
    agent.materiality_tier = materiality_tier
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
    taken.
    """
    agent = get(session, name)
    if agent is None:
        raise LookupError(f"agent '{name}' is not registered")
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
            "at": (now or utcnow()).isoformat(),
        },
    )
    return agent


__all__ = ["register", "get", "all_agents", "set_kill_switch"]
