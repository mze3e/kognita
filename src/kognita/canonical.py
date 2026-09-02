"""Canonical serialization and hashing.

Every hash Kognita takes — envelope hashes, evidence payload hashes, the
evidence chain — must be reproducible across processes and runs. That requires a
single serialization with a fixed key order: ``json.dumps`` defaults to insertion
order, so two dicts with the same contents but different construction order would
otherwise hash differently.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID


def _default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "model_dump"):  # pydantic / SQLModel
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize ``value`` deterministically: sorted keys, no incidental spacing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default)


def canonical_hash(value: Any) -> str:
    """SHA-256 of the canonical serialization of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_text(text: str) -> str:
    """SHA-256 of a string, for content addressing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
