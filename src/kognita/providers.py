"""Local ICT third-party register and fail-closed gateway admission (R3, R6).

Contract details use JSON payload keys, keeping the public schema limited to
approved names and the existing record storage pattern.
"""
from __future__ import annotations

from sqlalchemy import Column as _Column, JSON as _JSON, String as _String, Table as _Table
from sqlalchemy.orm import registry as _registry
from sqlmodel import SQLModel as _SQLModel


_registry = _registry(metadata=_SQLModel.metadata)


@_registry.mapped
class ProviderRecord:
    """Provider identity and declared ICT third-party contract details.

    Contract keys stay in private JSON storage until public field names are
    approved. Incident history and cooperation have approved public columns.
    """

    __table__ = _Table(
        "ProviderRecord", _SQLModel.metadata,
        _Column("provider", _String, primary_key=True),
        _Column("_record", _JSON, nullable=False),
        _Column("incident_history", _JSON, nullable=False, default=list),
        _Column("regulatory_cooperation", _String, nullable=False, default=""),
    )


_DETAILS = (
    "use_cases", "data_residency", "data_retention", "training_use",
    "sub_processors", "encryption", "incident_notification_terms",
    "availability_commitments", "model_change_notice", "audit_rights", "exit_plan",
)


def _validate(details):
    if not isinstance(details, dict):
        raise ValueError("R3-UNREGISTERED: provider record must be an object")
    for name in _DETAILS:
        value = details.get(name)
        if name in {"use_cases", "sub_processors"}:
            valid = isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)
        elif name == "training_use":
            valid = isinstance(value, bool)
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            raise ValueError(f"R3-UNREGISTERED: missing or invalid {name}")
    if not isinstance(details.get("incident_history"), list):
        raise ValueError("R3-UNREGISTERED: incident_history must be a list")
    if not isinstance(details.get("regulatory_cooperation"), str) or not details["regulatory_cooperation"].strip():
        raise ValueError("R3-UNREGISTERED: regulatory_cooperation is required")


def _register(session, provider, details):
    _validate(details)
    if not isinstance(provider, str) or not provider.strip() or provider != provider.strip():
        raise ValueError("R3-UNREGISTERED: provider identity is required")
    if session.get(ProviderRecord, provider) is not None:
        raise ValueError("R3-UNREGISTERED: provider is already registered")
    row = ProviderRecord(provider=provider, _record={key: details[key] for key in _DETAILS},
                         incident_history=details["incident_history"],
                         regulatory_cooperation=details["regulatory_cooperation"])
    session.add(row)
    session.flush()
    return row


def _view(row):
    return {"provider": row.provider, **row._record,
            "incident_history": row.incident_history,
            "regulatory_cooperation": row.regulatory_cooperation}


def _registered(session, provider):
    row = session.get(ProviderRecord, provider)
    if row is None:
        return False
    try:
        _validate(_view(row))
    except (ValueError, TypeError):
        return False
    return True
