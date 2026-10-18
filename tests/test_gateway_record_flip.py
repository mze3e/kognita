"""Regression: record() can flip ALLOW to DENY after decide() already
returned ALLOW (quota recheck, S6-VOLUME anomaly quarantine). The gateway
must not consume budget or forward the request to the upstream once that
happens.

This module is deliberately not on the legacy-authority allowlist in
conftest.py, so governance.decide is exercised unmodified: the ALLOW comes
from a real grant + delegation + credential, not from fixture authority
injected by an autouse fixture.
"""
from __future__ import annotations

from datetime import timedelta

from sqlmodel import select

from conftest import NOW
from kognita.authority import Delegation, delegate, grant_to
from kognita.credentials import issue
from kognita.models import GovernanceDecision, _AnomalyDay
from kognita.vocabulary import Outcome

from test_gateway import MODEL, PATH, _Upstream, _body, _gateway, _live
from test_mcp import _register


def test_anomaly_flip_after_allow_does_not_reach_backend(session, evidence, engine):
    _live(session)
    _register(session)
    scope = {"client": ["123"], "model": [MODEL]}
    grant_to(
        "alice", "read", timedelta(hours=2), scope,
        session=session, evidence=evidence, actor_id="admin", now=NOW,
    )
    delegation_id = delegate(
        session,
        Delegation(
            human="alice", agent="dossier-agent", purpose="COLLABORATION",
            client="123", allowed_actions=["read"], allowed_data=scope,
            valid_from=NOW, valid_until=NOW + timedelta(hours=1),
            jurisdiction="SG", task_ref="meeting",
        ),
        evidence=evidence, actor_id="alice", now=NOW,
    )
    # Already at the cold-start ceiling for today: the next observed action
    # pushes 'actions' over 1000 and anomaly._observe quarantines the agent.
    session.add(_AnomalyDay(agent_name="dossier-agent", day=NOW.date().isoformat(), body={"actions": 1000}))
    credential = issue(session, "dossier-agent", evidence=evidence, actor_id="Head of Research Ops")

    backend = _Upstream(b'{"model":"gateway-model","choices":[]}')
    response = _gateway(engine, evidence, backend).handle(
        "POST", PATH,
        {
            "agent_name": "dossier-agent",
            "agent_credential": credential,
            "delegation_id": delegation_id,
        },
        _body("hello"),
        session=session,
    )

    assert response.evaluation.outcome is Outcome.DENY
    assert any(c.citation == "S6-VOLUME" for c in response.evaluation.checks)
    # The decision governance.decide() made was ALLOW; only record()'s
    # anomaly observation flipped it. A fail-open gateway would still forward
    # to the upstream and answer 200 despite the stored outcome being DENY.
    assert response.status == 403
    assert backend.calls == []

    decision = session.exec(
        select(GovernanceDecision).where(
            GovernanceDecision.request_id == response.evaluation.request_id
        )
    ).first()
    assert decision is not None
    assert decision.outcome is Outcome.DENY
