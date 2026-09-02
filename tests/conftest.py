"""Shared fixtures. The core is exercised with no network and no API key."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures import demo_pack  # noqa: E402

from kognita.db import create_all, make_engine  # noqa: E402
from kognita.evidence import EvidenceWriter  # noqa: E402
from kognita.registry import register  # noqa: E402
from sqlmodel import Session  # noqa: E402

#: Every test decides against this instant, so effective-dating is deterministic.
NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    engine = make_engine()
    create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


@pytest.fixture
def evidence(engine):
    return EvidenceWriter(engine, clock=lambda: NOW)


@pytest.fixture
def pack():
    return demo_pack.DemoPack()


@pytest.fixture
def seeded(session):
    """Policies plus two registered agents."""
    demo_pack.seed_policies(session, now=NOW)
    register(session, name="eligibility-assistant", owner_exec="Head of Data Governance")
    register(session, name="dossier-agent", owner_exec="Head of Research Ops")
    session.commit()
    return session
