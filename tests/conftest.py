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



@pytest.fixture(autouse=True)
def _legacy_authority_prerequisites(request, monkeypatch):
    """Give pre-S3 control-isolation tests explicit snapshot authority.

    Authority lifecycle and integration tests deliberately receive no fixture
    authority. This fixture never changes production checks or outcomes.
    """
    if request.module.__name__ not in {
        'test_adapter_extras',
        'test_adapters',
        'test_autonomy_s1',
        'test_breaker',
        'test_classifier_envelope',
        'test_communication',
        'test_conformance',
        'test_core',
        'test_credentials',
        'test_effective_window',
        'test_engages',
        'test_entitlements',
        'test_evidence_references',
        'test_gateway',
        'test_gateway_overhead',
        'test_governance_conformance',
        'test_governed_agent_demo',
        'test_ingest',
        'test_interaction',
        'test_mcp',
        'test_outcomes',
        'test_packaging',
        'test_packs',
        'test_pinned_evidence',
        'test_policy',
        'test_purpose',
        'test_reconstruct',
        'test_recorder',
        'test_register_enforced',
        'test_review_queue',
        'test_run',
        'test_usecases',
        'test_vectors',
    }:
        return
    import kognita.tools as tools
    original_run = tools.run_governed

    def run_with_credential(session, envelope, *args, **kwargs):
        from dataclasses import replace
        from fixtures.agents import _credentials
        if envelope.agent_name and "_agent_credential" not in envelope.context:
            engine = session.get_bind()
            key = str(engine.url) if engine.url.database else engine
            secret = _credentials.get(key, {}).get(envelope.agent_name)
            if secret:
                envelope = replace(envelope, context={**envelope.context, "_agent_credential": secret})
        return original_run(session, envelope, *args, **kwargs)

    for module in list(sys.modules.values()):
        if module and module.__name__.startswith(("kognita", "test_")) and getattr(module, "run_governed", None) is original_run:
            monkeypatch.setattr(module, "run_governed", run_with_credential)
    from fixtures.agents import _authority_for
    import kognita.governance as governance
    original = governance.decide
    def authorized(envelope, snapshot, **kwargs):
        envelope, snapshot = _authority_for(envelope, snapshot)
        return original(envelope, snapshot, **kwargs)
    # Some tests and core entry points import decide directly.
    for module in list(sys.modules.values()):
        if module and getattr(module, "decide", None) is original:
            monkeypatch.setattr(module, "decide", authorized)
    standing = governance._standing_checks
    def standing_authorized(envelope, snapshot, **kwargs):
        envelope, snapshot = _authority_for(envelope, snapshot)
        return standing(envelope, snapshot, **kwargs)
    monkeypatch.setattr(governance, "_standing_checks", standing_authorized)
    import kognita.tools as tools
    if hasattr(tools, "_standing_checks"):
        monkeypatch.setattr(tools, "_standing_checks", standing_authorized)
