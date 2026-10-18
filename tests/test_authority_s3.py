"""S3 exercises real grants, delegation, snapshots and evidence without shims."""
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlmodel import Session, select

from kognita.authority import Delegation, delegate, grant_to, revoke, complete_task
from kognita.envelope import Envelope
from kognita.governance import decide, load_snapshot, record, _standing_checks
from kognita.models import EvidenceEvent, _Authority
from kognita.evidence import verify_chain
from kognita.vocabulary import Outcome
from fixtures.agents import register
from conftest import NOW


def _setup(session, evidence, *, verb="read", threshold="read"):
    agent = register(session, name="worker", owner_exec="owner")
    from copy import deepcopy
    permissions = deepcopy(agent.tool_allowlist)
    permissions["read_note"]["verb"] = verb
    permissions["read_note"]["permissions"] = [verb]
    permissions["read_note"]["required_level"] = 0 if verb == "read" else 3
    agent.tool_allowlist = permissions
    session.add(agent)
    session.flush()
    from kognita.registry import approve_deployment
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist["_deployment"],
                       evidence=evidence, actor_id="owner")
    if verb != "read":
        from kognita.autonomy import _change
        _change(session, agent, 3, evidence, "owner", "AUTONOMY_PROMOTED", {})
    gid = grant_to("alice", verb, timedelta(hours=2), {"client": ["123"]},
                   session=session, evidence=evidence, actor_id="admin", now=NOW)
    d = Delegation(human="alice", agent="worker", purpose="READ", client="123",
        allowed_actions=[verb], allowed_data={"client": ["123"]},
        valid_from=NOW, valid_until=NOW + timedelta(hours=1), jurisdiction="SG",
        approval_threshold=threshold, task_ref="meeting")
    did = delegate(session, d, evidence=evidence, actor_id="alice", now=NOW)
    env = Envelope(principal="alice", agent_name="worker", purpose="READ", tool="read_note",
        actor_location="SG", subject_type="client", subject_id="123", delegation_id=did)
    return gid, did, d, env


def _decide(session, env, at=NOW):
    return decide(env, load_snapshot(session, as_of=at, _agent_name="worker"),
                  purposes=["READ"], as_of=at)


def test_live_dual_check_and_pinned_evidence(session, evidence):
    gid, did, _, env = _setup(session, evidence)
    evaluation = _decide(session, env)
    assert evaluation.outcome is Outcome.ALLOW
    assert evaluation.attributes["_authority_grants"] == [gid]
    assert all(c.citation for c in evaluation.checks)
    record(session, evaluation, evidence=evidence, now=NOW)
    events = session.exec(select(EvidenceEvent)).all()
    assert any(e.payload.get("action") == "GRANT_USED" for e in events)
    decision = next(e for e in events if e.payload.get("envelope"))
    assert decision.payload["envelope"]["delegation_id"] == did
    assert verify_chain(session) > 0


@pytest.mark.parametrize("changes", [
    {"delegation_id": None}, {"delegation_id": "forged"}, {"principal": "mallory"},
    {"purpose": "OTHER"}, {"subject_id": "456"}, {"actor_location": "HK"},
    {"subjects": {"portfolio": "secret"}}, {"agent_name": "other"},
])
def test_scopes_and_human_entitlement_alone_never_authorize(session, evidence, changes):
    _, _, _, env = _setup(session, evidence)
    assert _decide(session, replace(env, **changes)).outcome is Outcome.DENY


@pytest.mark.parametrize("grant", [False, True])
def test_revocation_is_fresh_next_decision_and_blocks_release(session, evidence, grant):
    gid, did, _, env = _setup(session, evidence)
    assert _decide(session, env).outcome is Outcome.ALLOW
    revoke(session, gid if grant else did, evidence=evidence,
           actor_id="admin" if grant else "alice")
    snapshot = load_snapshot(session, as_of=NOW, _agent_name="worker")
    assert decide(env, snapshot, purposes=["READ"], as_of=NOW).outcome is Outcome.DENY
    assert _standing_checks(env, snapshot, attributes={"deployment_hash": snapshot.agents["worker"].deployment_hash}, at=NOW)


def test_revocation_other_session_refreshes_cached_rows(session, engine, evidence):
    gid, did, _, env = _setup(session, evidence)
    session.commit()
    assert _decide(session, env).outcome is Outcome.ALLOW
    with Session(engine) as other:
        revoke(other, did, evidence=evidence, actor_id="alice")
        other.commit()
    assert _decide(session, env).outcome is Outcome.DENY


def test_task_completion_is_specific_and_immediate(session, evidence):
    _, did, d, env = _setup(session, evidence)
    other = delegate(session, d.model_copy(update={"task_ref": "other"}),
                     evidence=evidence, actor_id="alice", now=NOW)
    complete_task(session, "meeting", evidence=evidence, actor_id="alice")
    assert _decide(session, env).outcome is Outcome.DENY
    assert _decide(session, replace(env, delegation_id=other)).outcome is Outcome.ALLOW
    assert session.get(_Authority, did).ended


@pytest.mark.parametrize("delta", [timedelta(microseconds=-1), timedelta(hours=1)])
def test_time_window_is_half_open(session, evidence, delta):
    _, _, _, env = _setup(session, evidence)
    assert _decide(session, env, NOW + delta).outcome is Outcome.DENY


def test_expiry_evidenced_once_on_recorded_decision(session, evidence):
    _, _, _, env = _setup(session, evidence)
    at = NOW + timedelta(hours=3)
    for _ in range(2):
        record(session, _decide(session, env, at), evidence=evidence, now=at)
    expiries = [e for e in session.exec(select(EvidenceEvent)).all()
                if e.payload.get("action") == "AUTHORITY_EXPIRED"]
    assert len(expiries) == 2  # the grant and the delegation, once each


@pytest.mark.parametrize("update", [
    {"allowed_actions": ["execute"]}, {"allowed_data": {"client": ["123", "456"]}},
    {"valid_until": NOW + timedelta(hours=3)}, {"valid_from": NOW - timedelta(days=1)},
    {"allowed_actions": []}, {"allowed_actions": ["*"]}, {"client": "456"},
    {"limits": {"amount": float("nan")}}, {"task_ref": ""},
])
def test_wider_or_invalid_delegation_rejected(session, evidence, update):
    _, _, d, _ = _setup(session, evidence)
    before = len(session.exec(select(EvidenceEvent)).all())
    with pytest.raises(ValueError):
        delegate(session, d.model_copy(update=update), evidence=evidence, actor_id="alice", now=NOW)
    assert len(session.exec(select(EvidenceEvent)).all()) == before


def test_approval_threshold_and_deny_precedence(session, evidence):
    _, _, _, env = _setup(session, evidence, verb="draft")
    assert _decide(session, env).outcome is Outcome.HUMAN_APPROVAL, _decide(session, env).failures()
    assert _decide(session, replace(env, subject_id="wrong")).outcome is Outcome.DENY


def test_limits_fail_closed_on_missing_or_excess_value(session, evidence):
    _, _, d, env = _setup(session, evidence)
    did = delegate(session, d.model_copy(update={"limits": {"amount": 10}}),
                   evidence=evidence, actor_id="alice", now=NOW)
    env = replace(env, delegation_id=did)
    for value in [None, True, -1, 11, float("nan"), float("inf")]:
        assert _decide(session, replace(env, arguments={"amount": value})).outcome is Outcome.DENY
    assert _decide(session, replace(env, arguments={"amount": 10})).outcome is Outcome.ALLOW


def test_issuer_and_empty_scope_validation(session, evidence):
    _, did, d, _ = _setup(session, evidence)
    with pytest.raises(ValueError):
        revoke(session, did, evidence=evidence, actor_id="mallory")
    with pytest.raises(ValueError):
        delegate(session, d, evidence=evidence, actor_id="mallory", now=NOW)
    for duration, scope in [(timedelta(0), {}), (timedelta(hours=1), {"client": ["*"]})]:
        with pytest.raises(ValueError):
            grant_to("alice", "read", duration, scope, session=session, evidence=evidence,
                     actor_id="admin", now=NOW)


def test_replay_uses_evidenced_authority_after_revocation(session, evidence):
    from kognita.replay import replay_decision
    _, did, _, env = _setup(session, evidence)
    original = record(session, _decide(session, env), evidence=evidence, now=NOW)
    revoke(session, did, evidence=evidence, actor_id="alice")
    assert _decide(session, env).outcome is Outcome.DENY
    replayed = replay_decision(session, original.request_id, env, purposes=["READ"])
    assert replayed.outcome is Outcome.ALLOW
    assert replayed.envelope_hash == original.envelope_hash


def test_lifecycle_rolls_back_with_evidence(session, engine, evidence):
    _setup(session, evidence)
    session.commit()
    count = len(session.exec(select(EvidenceEvent)).all())
    gid = grant_to("bob", "read", timedelta(hours=1), {"client": ["x"]},
                   session=session, evidence=evidence, actor_id="admin", now=NOW)
    session.rollback()
    assert session.get(_Authority, gid) is None
    assert len(session.exec(select(EvidenceEvent)).all()) == count
    assert verify_chain(session) == count


def test_delegation_cannot_bind_one_client_and_permit_another(session, evidence):
    _, _, d, _ = _setup(session, evidence)
    grant_to("alice", "read", timedelta(hours=2), {"client": ["123", "456"]},
             session=session, evidence=evidence, actor_id="admin", now=NOW)
    with pytest.raises(ValueError, match="S3-DELEGATION"):
        delegate(session, d.model_copy(update={"allowed_data": {"client": ["123", "456"]}}),
                 evidence=evidence, actor_id="alice", now=NOW)


@pytest.mark.parametrize("mode", ["gateway", "mcp"])
def test_real_gateways_bind_delegation_and_recheck_revocation(session, engine, evidence, mode):
    from test_gateway import _gateway, _Upstream, _body, _live, PATH, MODEL
    from test_mcp import _proxy, _Upstream as _Backend, _call, _register
    from kognita.credentials import issue
    _live(session)
    _register(session)
    scope = {"client": ["123"], "model": [MODEL]}
    grant_to("alice", "read", timedelta(hours=2), scope, session=session,
             evidence=evidence, actor_id="admin", now=NOW)
    did = delegate(session, Delegation(human="alice", agent="dossier-agent", purpose="COLLABORATION",
        client="123", allowed_actions=["read"], allowed_data=scope, valid_from=NOW,
        valid_until=NOW + timedelta(hours=1), jurisdiction="SG", task_ref="meeting"),
        evidence=evidence, actor_id="alice", now=NOW)
    credential = issue(session, "dossier-agent", evidence=evidence, actor_id="Head of Research Ops")
    if mode == "gateway":
        backend = _Upstream(b'{"model":"gateway-model","choices":[]}')
        endpoint, body = _gateway(engine, evidence, backend), _body("hello")
    else:
        backend = _Backend()
        endpoint, body = _proxy(engine, evidence, backend), _call()
    headers = {"agent_name": "dossier-agent", "agent_credential": credential}
    assert endpoint.handle("POST", PATH, headers, body, session=session).evaluation.outcome is Outcome.DENY
    assert not backend.calls
    headers["delegation_id"] = did
    allowed = endpoint.handle("POST", PATH, headers, body, session=session)
    assert allowed.evaluation.outcome is Outcome.ALLOW, allowed.evaluation.failures()
    assert allowed.evaluation.envelope.delegation_id == did
    assert len(backend.calls) == 1
    actions = [e for e in session.exec(select(EvidenceEvent)).all()
               if e.correlation_id == allowed.evaluation.request_id
               and e.event_type.value in ("TOOL_CALL", "MODEL_CALL")]
    assert actions and all(e.payload["delegation_id"] == did for e in actions)
    revoke(session, did, evidence=evidence, actor_id="alice")
    assert endpoint.handle("POST", PATH, headers, body, session=session).evaluation.outcome is Outcome.DENY
    assert len(backend.calls) == 1


def test_authority_lookup_uses_composite_index(session, evidence):
    from sqlalchemy import text
    _setup(session, evidence)
    plan = session.execute(text("EXPLAIN QUERY PLAN SELECT * FROM authority WHERE grantee='worker' AND kind='delegation' AND ended=0")).all()
    assert any("ix_authority_grantee_kind_ended" in str(row) for row in plan)


def test_evidence_failure_prevents_authority_change(session, evidence, monkeypatch):
    _, did, _, env = _setup(session, evidence)
    before = len(session.exec(select(_Authority)).all())
    def unavailable(*args, **kwargs):
        raise ConnectionError("evidence unavailable")
    monkeypatch.setattr(evidence, "emit", unavailable)
    with pytest.raises(ConnectionError):
        grant_to("bob", "read", timedelta(hours=1), {"client": ["x"]},
                 session=session, evidence=evidence, actor_id="admin", now=NOW)
    with pytest.raises(ConnectionError):
        revoke(session, did, evidence=evidence, actor_id="alice")
    assert len(session.exec(select(_Authority)).all()) == before
    assert not session.get(_Authority, did).ended
    assert _decide(session, env).outcome is Outcome.ALLOW


def test_oversized_authority_set_denies_instead_of_truncating(session, evidence):
    _, _, _, env = _setup(session, evidence)
    # Bulk fixture rows, never issued public grants: exercise the bounded reader.
    rows = [_Authority(id=f"bulk-{i}", grantee="alice", kind="grant", body={
        "capability": "read", "subject_scope": {"client": ["other"]},
        "valid_from": NOW.isoformat(), "valid_until": (NOW + timedelta(hours=2)).isoformat()
    }) for i in range(1000)]
    session.add_all(rows)
    session.flush()
    snapshot = load_snapshot(session, as_of=NOW, _agent_name="worker")
    assert snapshot._authority == ()
    assert decide(env, snapshot, purposes=["READ"], as_of=NOW).outcome is Outcome.DENY


def test_snapshot_and_continuation_keep_delegation_binding(session, evidence):
    from kognita.tools import _envelope_from_continuation
    _, _, _, env = _setup(session, evidence)
    restored = _envelope_from_continuation({"envelope": env.to_dict()})
    assert restored == env
    original = _decide(session, env)
    changed = _decide(session, replace(env, delegation_id="different"))
    assert original.envelope_hash != changed.envelope_hash
