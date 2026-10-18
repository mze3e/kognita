"""Blast radius is shared across tasks and projection is purpose-specific."""
from fixtures.agents import _usecases
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlmodel import select

from conftest import NOW
from test_authority_s3 import _setup, _decide
from kognita.evidence import verify_chain
from kognita.governance import record, load_snapshot, decide, _standing_checks
from kognita.models import Agent, BlastRadius, EvidenceEvent
from kognita.registry import approve_deployment, register
from kognita.vocabulary import Outcome
from kognita._limits import _minimize


def _configure(session, evidence, **limits):
    _, _, _, env = _setup(session, evidence)
    agent = session.exec(select(Agent).where(Agent.name == 'worker')).one()
    agent.blast_radius = BlastRadius(**limits).model_dump(mode='json', exclude_none=True)
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id='owner')
    return agent, env


@pytest.mark.parametrize('limit', ['per_minute', 'per_hour', 'max_daily_actions'])
def test_rate_is_shared_by_decisions_and_blocks_release(session, evidence, limit):
    agent, env = _configure(session, evidence, **{limit: 1})
    first = _decide(session, env)
    assert first.outcome == Outcome.ALLOW
    record(session, first, evidence=evidence, now=NOW)
    second = _decide(session, env)
    assert second.outcome == Outcome.DENY
    snap = load_snapshot(session, as_of=NOW, _agent_name=agent.name)
    assert _standing_checks(env, snap, attributes=first.attributes, at=NOW)
    assert verify_chain(session)


def test_minute_boundary_and_no_run_needed(session, evidence):
    _, env = _configure(session, evidence, per_minute=1)
    record(session, _decide(session, env), evidence=evidence, now=NOW)
    assert _decide(session, env, NOW + timedelta(seconds=59)).outcome == Outcome.DENY
    assert _decide(session, env, NOW + timedelta(minutes=1)).outcome == Outcome.ALLOW


@pytest.mark.parametrize('limits,changes,rule', [
    ({'max_clients': 0}, {}, 'S4-MAX-CLIENTS'),
    ({'max_value': 0.0}, {'arguments': {'value': 1}}, 'S4-MAX-VALUE'),
    ({'max_value': 0.0}, {}, 'S4-MAX-VALUE'),
    ({'channels': []}, {}, 'S4-CHANNELS'),
    ({'jurisdictions': ['HK']}, {}, 'S4-JURISDICTIONS'),
    ({'ttl': NOW}, {}, 'S4-TTL'),
])
def test_independent_limits_deny(session, evidence, limits, changes, rule):
    _, env = _configure(session, evidence, **limits)
    result = _decide(session, replace(env, **changes))
    assert result.outcome == Outcome.DENY
    assert any(c.check == rule and c.result.value == 'fail' for c in result.checks)


def test_configuration_is_owner_governed_and_invalidates_deployment(session, evidence):
    agent, env = _configure(session, evidence)
    with pytest.raises(ValueError):
        register(session, name='worker', owner_exec='owner', blast_radius={'per_minute': -1},
                 evidence=evidence, actor_id='owner')
    register(session, name='worker', owner_exec='owner', blast_radius={'per_minute': 2},
             evidence=evidence, actor_id='owner')
    assert agent.deployment_hash is None
    assert _decide(session, env).outcome == Outcome.DENY


def test_result_projection_recurses_without_leaking_whole_objects(session, evidence):
    agent, env = _configure(session, evidence)
    agent.field_allowlist = {'READ': ['risk', 'profile.rating']}
    session.add(agent)
    data = {'risk': 'low', 'passport': 'secret', 'profile': {'rating': 3, 'passport': 'secret'}}
    expected = {'risk': 'low', 'profile': {'rating': 3}}
    assert _minimize(session, env, data, evidence) == expected
    assert _minimize(session, env, [data], evidence) == [expected]
    assert _minimize(session, replace(env, purpose='OTHER'), data, evidence) == {}
    assert any(e.payload.get('citation') == 'S4-FIELDS' for e in session.exec(select(EvidenceEvent)).all())
    assert verify_chain(session)


def test_missing_usage_snapshot_denies(session, evidence):
    from kognita.governance import PolicySnapshot
    agent, env = _configure(session, evidence, per_minute=2)
    result = decide(env, PolicySnapshot(agents={agent.name: agent}), purposes=['READ'], as_of=NOW, usecases=_usecases(['READ']))
    assert any(c.check == 'S4-USAGE' and c.result.value == 'fail' for c in result.checks)


def test_event_window_uses_delegated_authority(session, evidence):
    from kognita.authority import delegate
    _, _, delegation, env = _setup(session, evidence)
    limits = {'_event_at': (NOW + timedelta(hours=25)).timestamp(), '_access_window': 86400}
    did = delegate(session, delegation.model_copy(update={'limits': limits}),
                   evidence=evidence, actor_id='alice', now=NOW)
    assert _decide(session, replace(env, delegation_id=did)).outcome == Outcome.DENY
    limits['_event_at'] = (NOW + timedelta(hours=23)).timestamp()
    did = delegate(session, delegation.model_copy(update={'limits': limits}),
                   evidence=evidence, actor_id='alice', now=NOW)
    assert _decide(session, replace(env, delegation_id=did)).outcome == Outcome.ALLOW


def test_stale_snapshot_cannot_spend_twice(session, evidence):
    _, env = _configure(session, evidence, per_minute=1)
    snap = load_snapshot(session, as_of=NOW, _agent_name='worker')
    a = decide(env, snap, purposes=['READ'], as_of=NOW, usecases=_usecases(['READ']))
    b = decide(env, snap, purposes=['READ'], as_of=NOW, usecases=_usecases(['READ']))
    assert record(session, a, evidence=evidence, now=NOW).outcome == Outcome.ALLOW
    assert record(session, b, evidence=evidence, now=NOW).outcome == Outcome.DENY


def test_replay_uses_pinned_usage_before_admission(session, evidence):
    from kognita.replay import replay_decision
    _, env = _configure(session, evidence, per_minute=1)
    result = record(session, _decide(session, env), evidence=evidence, now=NOW)
    assert _decide(session, env).outcome == Outcome.DENY
    replayed = replay_decision(session, result.request_id, envelope=env, purposes=['READ'], usecases=_usecases(['READ']))
    assert replayed.outcome == Outcome.ALLOW
    assert replayed.envelope_hash == result.envelope_hash


def test_client_and_value_limits_survive_day_boundary(session, evidence):
    from kognita.models import _BlastUsage
    _, env = _configure(session, evidence, max_clients=1, max_value=10.0)
    session.add(_BlastUsage(request_id='yesterday', agent_name='worker',
        at=NOW-timedelta(days=1), clients=['other'], value=9.0))
    session.flush()
    result = _decide(session, replace(env, arguments={'value': 2}))
    failed = {c.check for c in result.checks if c.result.value == 'fail'}
    assert {'S4-MAX-CLIENTS', 'S4-MAX-VALUE'} <= failed


@pytest.mark.parametrize('bad', [{'per_minute': -1}, {'per_minute': True},
    {'max_value': float('nan')}, {'execution_authority': 'anything'}, {'extra': 1}])
def test_invalid_limits_fail_closed(session, evidence, bad):
    agent, env = _configure(session, evidence)
    agent.blast_radius = bad
    session.add(agent)
    session.flush()
    assert _decide(session, env).outcome == Outcome.DENY


def test_nested_container_does_not_authorize_all_descendants(session, evidence):
    agent, env = _configure(session, evidence)
    agent.field_allowlist = {'READ': ['profile']}
    session.add(agent)
    assert _minimize(session, env, {'profile': {'passport': 'secret'}}, evidence) == {}


@pytest.mark.parametrize('tool', ['send_email', 'place_order', '_deployment'])
def test_agent_cannot_invoke_unlisted_tools(session, evidence, tool):
    _, env = _configure(session, evidence)
    result = _decide(session, replace(env, tool=tool))
    assert result.outcome == Outcome.DENY
    assert any(c.check == 'S4-TOOL-ALLOWLIST' and c.result.value == 'fail'
               and c.citation == c.check for c in result.checks)


def test_usage_cap_fails_closed(session, evidence):
    agent, env = _configure(session, evidence, per_minute=20000)
    snap = load_snapshot(session, as_of=NOW, _agent_name=agent.name)
    rows = ((NOW, ['123'], 0, 'prior'),) * 10001
    result = decide(env, replace(snap, _usage={agent.name: rows}), purposes=['READ'], as_of=NOW, usecases=_usecases(['READ']))
    assert result.outcome == Outcome.DENY
    assert any(c.check == 'S4-USAGE' and c.result.value == 'fail' for c in result.checks)


def test_release_excludes_own_reservation_but_counts_other_calls(session, evidence):
    agent, env = _configure(session, evidence, per_minute=1)
    first = record(session, _decide(session, env), evidence=evidence, now=NOW)
    snap = load_snapshot(session, as_of=NOW, _agent_name=agent.name)
    checks = _standing_checks(env, snap, attributes=first.attributes, at=NOW,
                              _request_id=first.request_id)
    assert not any(c.check.startswith('S4-') and c.result.value == 'fail' for c in checks)
    assert any(c.check == 'S4-PER-MINUTE' and c.result.value == 'fail'
               for c in _standing_checks(env, snap, attributes=first.attributes, at=NOW))


@pytest.mark.parametrize('delta,allowed', [(-1, False), (0, True), (24, True), (25, False)])
def test_event_access_boundaries(session, evidence, delta, allowed):
    from kognita.authority import delegate
    _, _, delegation, env = _setup(session, evidence)
    did = delegate(session, delegation.model_copy(update={'limits': {
        '_event_at': (NOW + timedelta(hours=delta)).timestamp(), '_access_window': 86400}}),
        evidence=evidence, actor_id='alice', now=NOW)
    assert (_decide(session, replace(env, delegation_id=did)).outcome == Outcome.ALLOW) == allowed


def test_unknown_objects_and_malformed_projection_never_escape(session, evidence):
    agent, env = _configure(session, evidence)
    agent.field_allowlist = {'READ': ['$', 'risk']}
    session.add(agent)
    assert _minimize(session, env, object(), evidence) is None
    assert _minimize(session, env, {'risk': object()}, evidence) == {}
    agent.field_allowlist = {'READ': 'risk'}
    session.add(agent)
    assert _minimize(session, env, {'risk': 'secret'}, evidence) == {}


def test_governed_tool_projects_before_release_and_blocks_next_call(session, evidence):
    from types import SimpleNamespace
    from test_usecases import ensure_live_purposes
    from kognita.tools import ToolRegistry, run_governed
    ensure_live_purposes(session, evidence, ['READ'])
    pack = SimpleNamespace(load_subjects=lambda env, session: {},
                           resolve_attributes=lambda env, subjects: {}, rules=lambda: {})
    agent, env = _configure(session, evidence, per_minute=1)
    agent.field_allowlist = {'READ': ['risk_profile']}
    approve_deployment(session, agent.name, deployment=agent.tool_allowlist['_deployment'],
                       evidence=evidence, actor_id='owner')
    registry = ToolRegistry()
    calls = []

    def read_note(envelope, evaluation, session):
        calls.append(True)
        return {'risk_profile': 'low', 'passport': 'secret'}

    registry.register('read_note', read_note)
    first = run_governed(session, env, registry=registry, evidence=evidence,
                         pack=pack, purposes=['READ'], as_of=NOW, now=NOW)
    assert first.evaluation.outcome == Outcome.ALLOW, first.evaluation.failures()
    assert first.data == {'risk_profile': 'low'}
    second = run_governed(session, env, registry=registry, evidence=evidence,
                          pack=pack, purposes=['READ'], as_of=NOW, now=NOW)
    assert second.evaluation.outcome == Outcome.DENY
    assert second.data is None
    assert len(calls) == 1
    assert verify_chain(session)


@pytest.mark.parametrize('authority,verb,allowed', [
    ('none', 'read', True), ('none', 'recommend', True),
    ('propose', 'draft', True), ('none', 'send', False),
    ('propose', 'execute', False), ('execute', 'execute', True),
])
def test_execution_authority_never_implies_other_permissions(session, evidence, authority, verb, allowed):
    from copy import deepcopy
    from kognita._limits import _checks
    agent, env = _configure(session, evidence, execution_authority=authority)
    permissions = deepcopy(agent.tool_allowlist)
    permissions[env.tool]['verb'] = verb
    agent.tool_allowlist = permissions
    snapshot = load_snapshot(session, as_of=NOW, _agent_name=agent.name)
    check = next(c for c in _checks(env, snapshot, NOW) if c.check == 'S4-EXECUTION-AUTHORITY')
    assert (check.result.value == 'pass') == allowed
    assert check.citation == check.check


def test_usage_lookup_uses_composite_index(session, evidence):
    from sqlalchemy import text
    _configure(session, evidence, per_minute=1)
    plan = session.execute(text("EXPLAIN QUERY PLAN SELECT * FROM _blast_usage "
        "WHERE agent_name='worker' AND at >= '2026-06-01' AND at <= '2026-06-02' "
        "ORDER BY at DESC LIMIT 10001")).all()
    assert any('ix_blast_usage_agent_time' in str(row) for row in plan)
