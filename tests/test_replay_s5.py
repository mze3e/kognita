"""Replay real 0.5 authority, outside the legacy authority fixture allowlist."""
from dataclasses import replace

from fixtures.agents import _usecases

import pytest
from sqlmodel import select

from conftest import NOW
from kognita.exceptions import ReplayMismatch
from kognita.evidence import ChainBreak
from kognita.governance import decide, load_snapshot, record
from kognita.models import EvidenceEvent
from kognita.replay import replay_decision
from kognita.vocabulary import Outcome
from test_fleet_s6b import _family, _child, _env


def _record(session, evidence, env):
    evaluation = decide(env, load_snapshot(session, as_of=NOW), purposes=['READ'], as_of=NOW, usecases=_usecases(['READ']))
    assert evaluation.outcome is Outcome.ALLOW, evaluation.checks
    return record(session, evaluation, evidence=evidence, now=NOW)


def test_child_delegated_allow_replays_with_same_checks(session, evidence):
    _, _, env, parent = _family(session, evidence)
    child = _child(session, evidence, parent)
    env = _env(env, child)
    original = _record(session, evidence, env)
    for supplied in (env, None):
        replayed = replay_decision(session, original.request_id, supplied, purposes=['READ'], usecases=_usecases(['READ']))
        assert replayed.outcome is original.outcome
        assert replayed.checks == original.checks
        assert replayed.envelope_hash == original.envelope_hash


def test_assurance_two_replays_identically(session, evidence):
    _, _, env, _ = _family(session, evidence)
    env = replace(env, assurance_level=2)
    original = _record(session, evidence, env)
    restored = replay_decision(session, original.request_id)
    # Evidence omits workload secrets; authenticate the restored library envelope.
    restored = replace(restored, envelope=replace(restored.envelope, context={
        **restored.envelope.context, '_agent_credential': env.context['_agent_credential']}))
    assert restored.envelope == env
    replayed = replay_decision(session, original.request_id, restored.envelope, purposes=['READ'], usecases=_usecases(['READ']))
    assert replayed.outcome is original.outcome
    assert replayed.checks == original.checks
    assert replayed.envelope_hash == original.envelope_hash


@pytest.mark.parametrize('supplied', [True])
@pytest.mark.parametrize('missing', ['GRANT_ISSUED', 'DELEGATION_ISSUED', 'AUTHORITY_TRANSFERRED'])
def test_missing_authority_evidence_fails_closed(session, evidence, monkeypatch, supplied, missing):
    # Simulate issuance whose authority row survived but evidence was not recorded.
    emit = evidence.emit
    def omit_issuance(*args, **kwargs):
        if kwargs.get('payload', {}).get('action') != missing:
            return emit(*args, **kwargs)
    monkeypatch.setattr(evidence, 'emit', omit_issuance)
    _, _, env, parent = _family(session, evidence)
    child = _child(session, evidence, parent)
    env = _env(env, child)
    original = _record(session, evidence, env)
    with pytest.raises(ReplayMismatch):
        replay_decision(session, original.request_id, env if supplied else None, purposes=['READ'], usecases=_usecases(['READ']))


def test_tampered_child_authority_evidence_fails_closed(session, evidence):
    _, _, env, parent = _family(session, evidence)
    child = _child(session, evidence, parent)
    original = _record(session, evidence, _env(env, child))
    event = next(e for e in session.exec(select(EvidenceEvent)).all()
                 if e.payload.get('action') == 'AUTHORITY_TRANSFERRED')
    # In-memory corruption avoids disabling the database's append-only triggers.
    event.payload['body']['agent'] = 'intruder'
    with pytest.raises(ChainBreak):
        replay_decision(session, original.request_id)


def test_child_replay_pins_identity_and_kill_switch(session, evidence):
    from kognita.registry import approve_deployment, get, set_kill_switch

    _, _, env, parent = _family(session, evidence)
    child = _child(session, evidence, parent)
    env = _env(env, child)
    original = _record(session, evidence, env)
    agent = get(session, 'child')
    deployment = dict(agent.tool_allowlist['_deployment'])
    deployment['instructions'] = 'changed after decision'
    approve_deployment(session, 'child', deployment=deployment,
                       evidence=evidence, actor_id='owner')
    set_kill_switch(session, 'child', True, evidence=evidence, actor_id='owner', now=NOW)
    replayed = replay_decision(session, original.request_id, env, purposes=['READ'], usecases=_usecases(['READ']))
    assert replayed.outcome is original.outcome
    assert replayed.checks == original.checks
    assert replayed.envelope_hash == original.envelope_hash
    assert get(session, 'child').kill_switch is True
