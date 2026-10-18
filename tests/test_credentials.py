"""Workload authentication uses actual headers, with no legacy fixture injection."""
import json
from dataclasses import replace

import pytest
from sqlmodel import Session, select

from kognita.credentials import issue, verify, revoke
from kognita.models import EvidenceEvent, _WorkloadCredential
from kognita.registry import approve_deployment
from kognita.review import review_flags
from kognita.usecases import vocabulary_citations, register as register_usecase
from kognita.vocabulary import Outcome
from fixtures.agents import register, _DEPLOYMENT
from test_gateway import _gateway, _Upstream, _body, _live, PATH
from test_mcp import _proxy, _Upstream as _Backend, _call, _register
from test_usecases import _spec, _tier2


def test_secrets_hash_binding_revocation_and_evidence(session, evidence):
    register(session, name="a", owner_exec="owner")
    register(session, name="b", owner_exec="owner")
    secret = issue(session, "a", evidence=evidence, actor_id="owner")
    second = issue(session, "a", evidence=evidence, actor_id="owner")
    assert second != secret
    assert verify(session, "a", secret)
    assert not verify(session, "b", secret)
    assert not verify(session, "a", None)
    assert not verify(session, "a", "invalid")
    rows = session.exec(select(_WorkloadCredential)).all()
    events = session.exec(select(EvidenceEvent)).all()
    assert secret not in repr(rows) + repr(events)
    with pytest.raises(ValueError, match="S2-CREDENTIAL"):
        revoke(session, "b", secret, evidence=evidence, actor_id="owner")
    revoke(session, "a", secret, evidence=evidence, actor_id="owner")
    assert not verify(session, "a", secret)
    assert verify(session, "a", second)
    assert any(e.payload.get("action") == "CREDENTIAL_REVOKED" for e in session.exec(select(EvidenceEvent)))


@pytest.mark.parametrize("mode", ["gateway", "mcp"])
@pytest.mark.parametrize("bad", ["missing", "invalid", "wrong-agent", "revoked"])
def test_gateways_deny_bad_credentials(session, engine, evidence, mode, bad):
    _live(session)
    _register(session)
    register(session, name="other", owner_exec="Head of Research Ops")
    name = "other" if bad == "wrong-agent" else "dossier-agent"
    secret = issue(session, name, evidence=evidence, actor_id="Head of Research Ops")
    if bad == "revoked":
        revoke(session, name, secret, evidence=evidence, actor_id="Head of Research Ops")
    headers = {"agent_name": "dossier-agent"}
    if bad != "missing":
        headers["AgEnT_CrEdEnTiAl"] = "invalid" if bad == "invalid" else secret
    if mode == "gateway":
        backend = _Upstream(b'{}')
        endpoint = _gateway(engine, evidence, backend)
        body = _body("hello")
    else:
        backend = _Backend()
        endpoint = _proxy(engine, evidence, backend)
        body = _call()
    response = endpoint.handle("POST", PATH, headers, body, session=session)
    assert response.evaluation.outcome == Outcome.DENY
    assert any(c.check == "AGENT_CREDENTIAL" and "S2-CREDENTIAL" in c.citation for c in response.evaluation.checks)
    assert not backend.calls


@pytest.mark.parametrize("mode", ["gateway", "mcp"])
def test_valid_then_revoked_next_call(session, engine, evidence, mode):
    _live(session)
    _register(session)
    secret = issue(session, "dossier-agent", evidence=evidence, actor_id="Head of Research Ops")
    if mode == "gateway":
        backend = _Upstream(b'{"model":"gateway-model","choices":[]}')
        endpoint, body = _gateway(engine, evidence, backend), _body("hello")
    else:
        backend = _Backend()
        endpoint, body = _proxy(engine, evidence, backend), _call()
    headers = {"agent_name": "dossier-agent", "agent_credential": secret}
    response = endpoint.handle("POST", PATH, headers, body, session=session)
    assert response.evaluation.outcome == Outcome.ALLOW
    assert len(backend.calls) == 1
    assert "agent_credential" not in backend.calls[0]["headers"]
    revoke(session, "dossier-agent", secret, evidence=evidence, actor_id="Head of Research Ops")
    response = endpoint.handle("POST", PATH, headers, body, session=session)
    assert response.evaluation.outcome == Outcome.DENY
    assert len(backend.calls) == 1


def test_deployment_mode_validation(session, evidence):
    spec = _tier2(deployment_mode="library")
    assert any("S2-DEPLOYMENT-MODE" in c for c in vocabulary_citations(spec))
    with pytest.raises(Exception, match="S2-DEPLOYMENT-MODE"):
        register_usecase(session, spec, evidence=evidence, actor_id="owner")
    assert not any("S2-DEPLOYMENT-MODE" in c for c in vocabulary_citations(replace(spec, deployment_mode="gateway")))
    assert any("S2-DEPLOYMENT-MODE" in c for c in vocabulary_citations(replace(spec, deployment_mode="bogus")))


def test_provider_keys_flagged_without_secret_in_evidence(session, evidence):
    register(session, name="a", owner_exec="owner")
    approve_deployment(session, "a", deployment={**_DEPLOYMENT,
        "environment": {"OPENAI_API_KEY": "secret-value"}}, evidence=evidence, actor_id="owner")
    flags = review_flags(session)
    assert any(f.get("finding") == "provider-key finding" for f in flags)
    events = session.exec(select(EvidenceEvent)).all()
    assert "secret-value" not in repr(events)


def test_indexed_lookup_and_revocation_across_sessions(engine):
    from kognita.evidence import EvidenceWriter
    from sqlalchemy import text
    writer = EvidenceWriter(engine)
    with Session(engine) as first:
        register(first, name="a", owner_exec="owner")
        secret = issue(first, "a", evidence=writer, actor_id="owner")
        first.commit()
    with Session(engine) as reader:
        assert verify(reader, "a", secret)
        with Session(engine) as admin:
            revoke(admin, "a", secret, evidence=writer, actor_id="owner")
            admin.commit()
        assert not verify(reader, "a", secret)
        plan = reader.execute(text('EXPLAIN QUERY PLAN SELECT * FROM _workload_credentials WHERE digest = :digest'), {"digest": "x"}).all()
        assert any("INDEX" in str(row) for row in plan)


def test_additive_usecase_migration(tmp_path):
    from kognita.db import make_engine, create_all
    from sqlalchemy import text
    engine = make_engine(tmp_path / "legacy.db")
    create_all(engine)
    # Recreate just this empty table in the shape of a pre-S2 store.
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE use_cases")
        connection.exec_driver_sql('''CREATE TABLE use_cases (
            id INTEGER PRIMARY KEY, usecase_id TEXT, version INTEGER,
            record JSON, content_hash TEXT, effective_from DATETIME,
            effective_to DATETIME)''')
        connection.exec_driver_sql("INSERT INTO use_cases (id, usecase_id, version, record) VALUES (1, 'legacy', 1, '{}')")
    create_all(engine)
    create_all(engine)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT deployment_mode FROM use_cases WHERE id=1")).scalar() is None
        columns = connection.exec_driver_sql("PRAGMA table_info(use_cases)").all()
        assert next(c for c in columns if c[1] == "deployment_mode")[3] == 0


def test_library_tier_one_allowed_and_mode_round_trips(session, evidence):
    from kognita.usecases import load_register
    from test_usecases import _open
    _open(session, evidence)
    row = register_usecase(session, _spec(deployment_mode="library"), evidence=evidence, actor_id="owner")
    assert row.deployment_mode == "library"
    assert load_register(session).get(row.usecase_id).deployment_mode == "library"


def test_environment_provider_key_finding(session, evidence):
    from kognita.registry import register as raw_register
    raw_register(session, name="a", owner_exec="owner", evidence=evidence,
                 actor_id="owner", environment="OPENAI_API_KEY=do-not-store")
    assert any(f.get("agent") == "a" for f in review_flags(session))
    assert "do-not-store" not in repr(session.exec(select(EvidenceEvent)).all())
