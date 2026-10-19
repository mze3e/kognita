"""The gateway holds the provider key; an agent that holds one is a finding.

ROADMAP item 15: "agent workload credentials are verified by the gateways,
and the provider and tool credentials behind them are held by the gateways,
never by the agent. An agent that holds a provider key directly is a
finding." Deliberately outside the pre-S3 fixture allowlist in conftest.py:
a real grant, delegation and issued credential, not fabricated authority, so
these tests exercise governance.decide() and Gateway._handle() unmodified.
"""
from __future__ import annotations

from datetime import timedelta

from sqlmodel import select

from fixtures import demo_pack as dp
from conftest import NOW
from fixtures.agents import register as _register_agent
from kognita.authority import Delegation, delegate, grant_to
from kognita.cli import build_parser, cmd_serve
from kognita.credentials import issue
from kognita.gateway import Gateway
from kognita.models import Approval, EvidenceEvent, RunRecord
from kognita.review import review_flags
from kognita.tools import Run, continue_run
from kognita.vocabulary import Outcome

from test_approval_continuation import _hold
from test_gateway import MODEL, PATH, _Upstream, _body, _gateway, _live
from test_gateway_run_controls import _setup as _run_controls_setup


def _authorize(session, evidence, *, client: str = "123"):
    scope = {"client": [client], "model": [MODEL]}
    grant_to(
        "alice", "read", timedelta(hours=2), scope,
        session=session, evidence=evidence, actor_id="admin", now=NOW,
    )
    delegation_id = delegate(
        session,
        Delegation(
            human="alice", agent="dossier-agent", purpose="COLLABORATION",
            client=client, allowed_actions=["read"], allowed_data=scope,
            valid_from=NOW, valid_until=NOW + timedelta(hours=1),
            jurisdiction="SG", task_ref="provider-key",
        ),
        evidence=evidence, actor_id="alice", now=NOW,
    )
    credential = issue(session, "dossier-agent", evidence=evidence, actor_id="Head of Research Ops")
    return delegation_id, credential


def _setup(session, evidence, **gateway_kwargs):
    _register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    _live(session)
    delegation_id, credential = _authorize(session, evidence)
    backend = _Upstream(b'{"model":"gateway-model","choices":[]}')
    gateway = _gateway(session.get_bind(), evidence, backend, **gateway_kwargs)
    headers = {
        "agent_name": "dossier-agent",
        "agent_credential": credential,
        "delegation_id": delegation_id,
    }
    return gateway, backend, headers


def test_caller_authorization_header_is_denied_and_flagged(session, evidence):
    gateway, backend, headers = _setup(session, evidence)
    secret = "sk-caller-should-not-hold-this"
    headers = {**headers, "authorization": f"Bearer {secret}"}

    response = gateway.handle("POST", PATH, headers, _body("hello"), session=session)

    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    assert any(c.citation == "S2-PROVIDER-KEY" for c in response.evaluation.checks)
    assert backend.calls == []

    flags = review_flags(session)
    matched = [f for f in flags if f.get("finding") == "provider-key finding"]
    assert matched
    assert matched[0]["fields"] == ["authorization"]
    events = session.exec(select(EvidenceEvent)).all()
    assert secret not in repr(events)


def test_caller_x_api_key_header_is_also_denied(session, evidence):
    gateway, backend, headers = _setup(session, evidence)
    secret = "sk-also-should-not-be-here"
    headers = {**headers, "x-api-key": secret}

    response = gateway.handle("POST", PATH, headers, _body("hello"), session=session)

    assert response.evaluation.outcome is Outcome.DENY
    assert any(c.citation == "S2-PROVIDER-KEY" for c in response.evaluation.checks)
    assert backend.calls == []
    events = session.exec(select(EvidenceEvent)).all()
    assert secret not in repr(events)


def test_host_key_is_forwarded_never_the_caller_key(session, evidence):
    gateway, backend, headers = _setup(session, evidence, provider_key="host-only-secret")

    response = gateway.handle("POST", PATH, headers, _body("hello"), session=session)

    assert response.status == 200
    assert len(backend.calls) == 1
    assert backend.calls[0]["headers"]["authorization"] == "Bearer host-only-secret"


def test_no_configured_key_forwards_no_provider_credential(session, evidence):
    gateway, backend, headers = _setup(session, evidence)

    response = gateway.handle("POST", PATH, headers, _body("hello"), session=session)

    assert response.status == 200
    assert len(backend.calls) == 1
    assert "authorization" not in backend.calls[0]["headers"]


def test_resumed_gateway_call_uses_the_host_key(session, evidence):
    gateway, backend, headers = _run_controls_setup(session, evidence)
    gateway.provider_key = "host-only-secret"
    _hold(session, "model_call")
    run = Run()
    held = gateway.handle(
        "POST", PATH, headers, _body("exact held question"), session=session, run=run
    )
    assert held.evaluation.outcome is Outcome.HUMAN_APPROVAL
    assert backend.calls == []
    approval = session.exec(
        select(Approval).where(Approval.decision_id == held.evaluation.decision_id)
    ).one()
    row = session.get(RunRecord, run.id)
    assert row.continuation_hash
    session.commit()

    result = continue_run(
        session, run.id, {approval.id: True}, evidence=evidence,
        approver_name="reviewer", now=NOW,
    )

    assert result.status == 200
    assert len(backend.calls) == 1
    assert backend.calls[0]["headers"]["authorization"] == "Bearer host-only-secret"


def test_cli_wires_provider_key_flag_and_env_var(tmp_path, monkeypatch):
    """``kognita serve --provider-key``, or ``KOGNITA_PROVIDER_KEY`` when it
    is omitted, reaches the ``Gateway`` the way the command actually builds
    one. The flag wins over the environment; neither set leaves no key."""
    held: dict[str, Gateway] = {}

    def serve(self, host="127.0.0.1", port=8080):
        held["gateway"] = self

    monkeypatch.setattr(Gateway, "serve", serve)

    def argv(extra: list[str]) -> list[str]:
        return [
            "serve", "--provider", "openai-compatible",
            "--upstream", "https://api.openai.com",
            "--db", str(tmp_path / "gw.db"), "--principal", "alice",
            "--purpose", dp.PURPOSES[0], "--actor-location", "SG",
            "--agent", "dossier-agent",
        ] + extra

    assert cmd_serve(build_parser().parse_args(argv(["--provider-key", "flag-secret"]))) == 0
    assert held["gateway"].provider_key == "flag-secret"

    held.clear()
    monkeypatch.setenv("KOGNITA_PROVIDER_KEY", "env-secret")
    assert cmd_serve(build_parser().parse_args(argv([]))) == 0
    assert held["gateway"].provider_key == "env-secret"

    held.clear()
    monkeypatch.delenv("KOGNITA_PROVIDER_KEY", raising=False)
    assert cmd_serve(build_parser().parse_args(argv([]))) == 0
    assert held["gateway"].provider_key is None
