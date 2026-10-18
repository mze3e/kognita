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
from fixtures.agents import register  # noqa: E402
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


@pytest.fixture(autouse=True)
def _legacy_workload_credentials(request, monkeypatch):
    """Older policy tests use real issued credentials for their fixture agents.

    S2 tests supply headers themselves so missing-secret cases stay explicit.
    """
    if request.module.__name__ == "test_credentials":
        return
    from kognita.gateway import Gateway
    from kognita.mcp import McpProxy

    def wrap(original):
        def authenticated(self, method, path, headers, body, **kwargs):
            headers = dict(headers)
            name = headers.get("agent_name") or self.client.agent_name
            from fixtures.agents import _credentials
            key = str(self.engine.url) if self.engine.url.database else self.engine
            secrets = _credentials.get(key, {})
            if name in secrets and not any(k.lower() == "agent_credential" for k in headers):
                headers["agent_credential"] = secrets[name]
            return original(self, method, path, headers, body, **kwargs)
        return authenticated

    monkeypatch.setattr(Gateway, "handle", wrap(Gateway.handle))
    monkeypatch.setattr(McpProxy, "handle", wrap(McpProxy.handle))
