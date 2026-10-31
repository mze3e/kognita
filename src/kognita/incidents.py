"""R13 incident timelines stored in the existing evidence chain.

Gateway outage detection is buffered in memory during store failure. Recovery
persists the detection time; process termination can lose that pending record.
"""
from enum import Enum as _Enum
from uuid import uuid4 as _uuid4

from sqlmodel import select as _select

from kognita.models import EvidenceEvent as _EvidenceEvent
from kognita.vocabulary import EventType as _EventType
from kognita.models import utcnow as _utcnow


class IncidentKind(str, _Enum):
    _CHAIN_BREAK = "chain_break"
    _GATEWAY_OUTAGE = "gateway_outage"
    _DEGRADED = "degraded_mode"
    _PROVIDER_FAILURE = "provider_failure"


def _timeline(session, incident_id):
    return [event for event in session.exec(
        _select(_EvidenceEvent).order_by(_EvidenceEvent.sequence)).all()
        if event.event_type in (_EventType.INCIDENT_OPENED,
            _EventType.INCIDENT_UPDATED, _EventType.INCIDENT_CLOSED)
        and event.payload.get("incident_id") == incident_id]


def _list(session):
    return [dict(event.payload, sequence=event.sequence)
        for event in session.exec(_select(_EvidenceEvent)
            .where(_EvidenceEvent.event_type == _EventType.INCIDENT_OPENED)
            .order_by(_EvidenceEvent.sequence)).all()]


def _open(session, evidence, kind, *, summary, source, at=None):
    try:
        kind = IncidentKind(kind)
    except ValueError as exc:
        raise ValueError("R13: unknown incident kind") from exc
    for incident in _list(session):
        if incident["source"] == source and incident["kind"] == kind.value:
            if _timeline(session, incident["incident_id"])[-1].event_type != _EventType.INCIDENT_CLOSED:
                return incident["incident_id"]
    if not summary.strip():
        raise ValueError("R13: incident summary is required")
    from kognita.usecases import record_incident
    incident_id = str(_uuid4())
    with session.begin_nested():
        record_incident(session, impact=1, summary=summary,
                        evidence=evidence, actor_id="system", at=at)
        evidence.emit(session, correlation_id=incident_id,
            event_type=_EventType.INCIDENT_OPENED,
            payload={"incident_id": incident_id, "kind": kind.value,
                "summary": summary, "source": source,
                "at": (at or _utcnow()).isoformat(), "rule_id": "R13"})
    return incident_id


def _change(session, evidence, incident_id, *, summary, close=False, at=None):
    timeline = _timeline(session, incident_id)
    if not timeline or timeline[-1].event_type == _EventType.INCIDENT_CLOSED or not summary.strip():
        raise ValueError("R13: incident must be open and summary must be present")
    evidence.emit(session, correlation_id=incident_id,
        event_type=_EventType.INCIDENT_CLOSED if close else _EventType.INCIDENT_UPDATED,
        payload={"incident_id": incident_id, "summary": summary,
            "at": (at or _utcnow()).isoformat(), "rule_id": "R13"})


def _show(session, incident_id):
    timeline = _timeline(session, incident_id)
    if not timeline:
        raise ValueError("R13: unknown incident")
    return {"incident_id": incident_id, "timeline": [dict(event.payload,
        event_type=event.event_type.value, sequence=event.sequence)
        for event in timeline]}


def _export(session, incident_id):
    from kognita.evidence import ChainBreak, export_chain, _verify_chain
    result = dict(_show(session, incident_id), evidence=export_chain(session,
        correlation_id=incident_id))
    try:
        _verify_chain(session)
    except ChainBreak as exc:
        result["verification_error"] = f"R13: {exc}"
    else:
        result["verification_error"] = None
    return result
