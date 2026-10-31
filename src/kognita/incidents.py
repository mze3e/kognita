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


# Fixed operational guidance. These labels are values, not new API names.
_PLAYBOOKS = {
    "hallucination": ("Review claim verification failures and response screening.", "Trip the circuit breaker for the affected use case; withhold output.", "Reconstruct claims, source versions and affected clients."),
    "data leakage": ("Review access denials and released output evidence.", "Revoke affected agent authority and stop the use case.", "Query affected clients and reconstruct tool and model output releases."),
    "prompt injection": ("Review external-content isolation and instruction checks.", "Revoke affected tool access and trip the circuit breaker.", "Reconstruct source provenance and the authority lineage."),
    "unauthorized tool execution": ("Review tool authorization and permission failures.", "Revoke the agent and stop tool execution through the circuit breaker.", "Reconstruct tool calls, approvals and affected clients."),
    "incorrect transaction": ("Review transaction approvals and recorded outcomes.", "Stop the affected agent and invoke the bank's reversal process.", "Reconstruct authorization, transaction evidence and affected clients."),
    "model outage": ("Review provider-failure and gateway-outage incident timelines.", "Stop calls or use only governed approved failover.", "Compare incident timelines, model versions and deployment changes."),
    "third-party compromise": ("Review provider incidents and model-change triggers.", "Revoke the provider and stop dependent use cases.", "Query affected clients and reconstruct provider and model evidence."),
    "deepfake instruction": ("Review instruction authenticity and authority checks.", "Revoke the suspect authority and require independent human approval.", "Reconstruct instruction provenance, approvals and authority lineage."),
    "sensitive-data exposure": ("Review sensitive-attribute and field-minimization checks.", "Revoke affected access and withhold further output.", "Query affected clients and reconstruct released fields and access evidence."),
}


def _playbook(kind):
    if kind not in _PLAYBOOKS:
        raise ValueError("R4: unknown incident type")
    detect, contain, investigate = _PLAYBOOKS[kind]
    return {"type": kind, "rule_id": "R4", "detect": detect,
        "contain": contain, "investigate": investigate,
        "boundary": "The bank owns incident response and disclosure."}


def _report(session, incident_id, *, redacted=False):
    from datetime import datetime, timedelta
    from kognita.evidence import _verify_chain, ChainBreak
    from kognita.models import Interaction, as_utc
    from kognita.breaker import affected_clients

    timeline = _timeline(session, incident_id)
    if not timeline or timeline[0].event_type != _EventType.INCIDENT_OPENED:
        raise ValueError("R14: unknown incident")
    try:
        _verify_chain(session)
    except ChainBreak as exc:
        raise ValueError("R14: report refused because evidence verification failed") from exc
    start = datetime.fromisoformat(timeline[0].payload["at"])
    closed = timeline[-1].event_type == _EventType.INCIDENT_CLOSED
    end = datetime.fromisoformat(timeline[-1].payload["at"]) if closed else _utcnow()
    start, end = as_utc(start), as_utc(end)
    if end < start:
        raise ValueError("R14: incident end precedes detection")
    if end == start:
        end = start + timedelta(microseconds=1)
    events = session.exec(_select(_EvidenceEvent).order_by(_EvidenceEvent.sequence)).all()
    window = [event for event in events if start <= as_utc(event.recorded_at) < end]
    # Temporal association is deliberately explicit: it is not proof of causation.
    controls = []
    for event in window:
        for check in event.payload.get("checks", []):
            if isinstance(check, dict):
                controls.append({"sequence": event.sequence, "check": check.get("check"),
                    "result": check.get("result")})
    clients, unlinked = {}, {}
    usecases = {row.use_case for row in session.exec(_select(Interaction)).all() if row.use_case}
    models = {event.payload.get("model_version") for event in window
        if isinstance(event.payload.get("model_version"), str) and event.payload["model_version"]}
    for selector, targets in (("usecase", usecases), ("model", models)):
        for target in sorted(targets):
            affected = affected_clients(session, start=start, end=end, **{selector: target})
            for row in affected["clients"]:
                clients[(row["client_id"], row["interaction_id"])] = row
            for row in affected["unlinked"]:
                unlinked[row["correlation_id"]] = row
    deployments = {}
    for event in events:
        if event.payload.get("action") == "DEPLOYMENT_APPROVED" and as_utc(event.recorded_at) < end:
            deployments.setdefault(event.payload.get("agent"), []).append(event)
    diffs = []
    for agent, rows in deployments.items():
        latest = rows[-1]
        before = rows[-2].payload.get("deployment", {}) if len(rows) > 1 else {}
        after = latest.payload.get("deployment", {})
        diffs.append({"agent": agent, "sequence": latest.sequence,
            "recorded_at": latest.recorded_at.isoformat(),
            "before_detection": as_utc(latest.recorded_at) < start,
            "baseline_available": len(rows) > 1,
            "changes": {key: {"before": before.get(key), "after": after.get(key)}
                for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)},
            "identity_before": rows[-2].payload.get("_identity") if len(rows) > 1 else None,
            "identity_after": latest.payload.get("_identity")})
    missing = ["Expected control inventory is unavailable; absent checks cannot be classified as passed."]
    if not controls:
        missing.append("No control checks recorded in the incident window.")
    if not diffs:
        missing.append("No deployment approval recorded before the report cutoff.")
    failed = [row for row in controls if row["result"] == "fail"]
    if redacted:
        return {"rule_id": "R14", "redacted": True, "closed": closed,
            "timeline_events": len(timeline), "affected_clients": len({key[0] for key in clients}),
            "unlinked_outputs": len(unlinked), "controls_ran": len(controls),
            "controls_failed": len(failed), "controls_missing": missing,
            "deployment_changes": len(diffs)}
    return {"rule_id": "R14", "incident_id": incident_id,
        "scope": "All recorded activity in the incident time window and latest deployment approvals before the cutoff; temporal association does not establish causation.",
        "from": start.isoformat(), "to": end.isoformat(), "closed": closed,
        "what_happened": _show(session, incident_id),
        "affected_clients": list(clients.values()), "unlinked_outputs": list(unlinked.values()),
        "controls_ran": controls, "controls_failed": failed, "controls_missing": missing,
        "deployment_diff": diffs}
