"""Regressions for the 0.8 release review, without relaxed control thresholds."""
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlmodel import select

from test_assurance import setup, packet
from fixtures.agents import register
from kognita.assurance import AssurancePolicy, Attestation, configure, sign_record
from kognita.assurance.records import binding, events
from kognita.assurance.lifecycle import ProductionOutcome, approve_update, advisory, record_outcome
from kognita.assurance.runtime import _attestation, attest, load_state, request_digest
from kognita.assurance.reports import compare_manifest, conformance_report, export_manifest
from kognita.evidence import ChainBreak, verify_chain
from kognita.models import EvidenceEvent, Interaction, utcnow
from kognita.vocabulary import EventType


def test_good_outcomes_do_not_repeat_breaches_or_step_down(session, setup):
    """Only a new adverse result can trigger a threshold response."""
    a, w, _, _, _, at, config = setup
    from kognita.autonomy import _change
    _change(session, a, 3, w, 'owner', 'AUTONOMY_PROMOTED', {})
    config(max_adverse_outcomes=2, outcome_step_down=True)
    session.add(Interaction(id='review', client_id='client', use_case='p'))
    session.flush()
    for index, adverse in enumerate((True, True, False, False, False)):
        result = ProductionOutcome(interaction_id='review', intended='correct', actual='supplied result',
            client_impact='supplied impact', adverse=adverse,
            source_reference=f'bank:{index}', outcome_id=str(index))
        response = record_outcome(session, a.name, result, evidence=w, actor_id='owner', now=at)
        assert response == {'breached': index == 1, 'adverse_count': min(index + 1, 2)}
    assert a.autonomy_level == 2
    rows = session.exec(select(EvidenceEvent)).all()
    assert sum(r.event_type == EventType.REVALIDATION_TRIGGERED for r in rows) == 1
    assert sum(r.payload.get('action') == 'AUTONOMY_STEP_DOWN' for r in rows) == 1
    assert verify_chain(session) == len(rows)


def test_missing_revalidation_write_rolls_back_entire_outcome(session, setup, monkeypatch):
    """An incomplete required trigger cannot leave a finding or outcome behind."""
    a, w, _, _, _, at, config = setup
    config(outcome_step_down=True)
    session.add(Interaction(id='review', client_id='client', use_case='p'))
    session.flush()
    before = verify_chain(session)
    original = w.emit
    def drop_trigger(session, **kwargs):
        """Provide drop trigger for the assurance regression."""
        if kwargs['event_type'] != EventType.REVALIDATION_TRIGGERED:
            return original(session, **kwargs)
    monkeypatch.setattr(w, 'emit', drop_trigger)
    result = ProductionOutcome(interaction_id='review', intended='good', actual='bad',
        client_impact='loss', adverse=True, source_reference='bank:1', outcome_id='1')
    with pytest.raises(ValueError, match='A3: revalidation evidence write incomplete'):
        record_outcome(session, a.name, result, evidence=w, actor_id='owner', now=at)
    assert verify_chain(session) == before
    assert not any(r.payload.get('action') == 'PRODUCTION_OUTCOME' for r in events(session, a.name))


def test_exact_instruction_survives_unrelated_newer_expired_approval(session, setup):
    """Digest-bound records do not supersede records for other requests."""
    a, w, signer, _, envelope, at, config = setup
    p = config(instruction_tools=[envelope.tool])
    valid = packet(setup, p, 'instruction', {'request_digest': request_digest(envelope),
        'verification_channel': 'independent', 'maker': 'maker', 'checker': 'checker'})
    digest = attest(session, a.name, valid, evidence=w, actor_id='runtime', now=at)
    other = sign_record(Attestation(kind='instruction', issuer='runtime', subject=a.name,
        issued_at=at - timedelta(seconds=1), expires_at=at + timedelta(seconds=1),
        **binding(a, p), claims={'request_digest': request_digest(replace(envelope, arguments={'new': 1})),
        'verification_channel': 'independent', 'maker': 'maker', 'checker': 'checker'}, reference='other'), signer)
    attest(session, a.name, other, evidence=w, actor_id='runtime', now=at)
    rows = events(session, a.name)
    later = at + timedelta(seconds=2)
    assert _attestation(rows, a, p, 'instruction', later, digest=digest) is not None
    from kognita.canonical import canonical_hash
    assert _attestation(rows, a, p, 'instruction', later, digest=canonical_hash(other)) is None
    assert _attestation(rows, a, p, 'instruction', later) is None


def test_manifest_review_date_is_not_false_divergence(session, setup):
    """JSON datetime serialization must compare like for like."""
    a, w, signer, public, _, at, config = setup
    config()
    assert a.review_date is not None
    exported = export_manifest(session, a.name, signer, evidence=w, actor_id='owner', now=at)
    assert compare_manifest(session, a.name, exported, public)['findings'] == []
    a.review_date += timedelta(days=1)
    session.add(a)
    session.flush()
    assert any('identity' in f.get('reason', '') for f in
               compare_manifest(session, a.name, exported, public)['findings'])


def test_a4_report_keeps_live_deployment_packet_when_next_update_pending(session, setup):
    """A future candidate must not hide the currently deployed approval."""
    a, w, _, _, _, _, config = setup
    config()
    def approve(digest):
        """Provide approve for the assurance regression."""
        return approve_update(session, a.name, candidate_digest=digest, provider_change='change',
            regression_reference='suite', rollback_target='previous', rollback_plan='restore',
            evidence=w, actor_id='owner')
    approve(a.deployment_hash)
    live = events(session, a.name)[-1].sequence
    approve('b' * 64)
    report = conformance_report(session, a.name, 'MAS')
    row = next(r for r in report['expectations'] if r['control'] == 'A4')
    assert row['status'] == 'evidenced' and row['evidence_sequences'] == [live]


def test_snapshot_and_advisory_verify_once_for_multiple_agents(session, setup, monkeypatch):
    """Batching removes agent-multiplied verification without trusting old heads."""
    a, w, _, _, _, _, config = setup
    config()
    b = register(session, name='second-worker', owner_exec='owner')
    configure(session, b.name, AssurancePolicy(use_case='p', audience='bank'), evidence=w, actor_id='owner')
    import kognita.evidence as evidence_module
    original = evidence_module._verify_chain
    calls = []
    def counted(*args, **kwargs):
        """Provide counted for the assurance regression."""
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(evidence_module, '_verify_chain', counted)
    assert len(load_state(session, [a.name, b.name], utcnow())) == 2
    assert len(calls) == 1
    calls.clear()
    assert advisory(session, 'unmatched', '1', 'security:1', evidence=w, actor_id='security') == []
    assert len(calls) == 1


def test_same_head_tamper_is_rejected_after_previous_snapshot(session, setup):
    """A cached head alone is insufficient to detect an old-row rewrite."""
    a, _, _, _, _, _, config = setup
    config()
    load_state(session, [a.name], utcnow())
    row = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).first()
    from evidence_guard import tamper_for_test
    tamper_for_test(session, row.sequence, {'tampered': True})
    with pytest.raises(ChainBreak, match='payload does not match'):
        load_state(session, [a.name], utcnow())


@pytest.mark.parametrize('later_seconds', [0, 60])
def test_monitoring_replay_excludes_later_actions_and_live_still_denies(session, evidence, later_seconds):
    """A later action, even at the same clock tick, cannot change a past decision."""
    from test_authority_s3 import _setup
    from fixtures.agents import _usecases
    from conftest import NOW
    from kognita.governance import decide, load_snapshot, record
    from kognita.replay import replay_decision
    from kognita.signing import EvidenceSigner, _generate
    from kognita.assurance.records import agent_for
    from kognita.vocabulary import Outcome
    _, _, _, envelope = _setup(session, evidence)
    agent = agent_for(session, 'worker')
    private, public = _generate()
    p = configure(session, agent.name, AssurancePolicy(use_case='READ', audience='bank',
        monitoring_required=True, attestors={'runtime': public.decode()}), evidence=evidence, actor_id='owner')
    action = evidence.emit(session, correlation_id='earlier', event_type=EventType.TOOL_CALL,
                           actor_id=agent.name, payload={'tool': envelope.tool})
    proof = sign_record(Attestation(kind='monitoring', issuer='runtime', subject=agent.name,
        issued_at=NOW - timedelta(seconds=1), expires_at=NOW + timedelta(days=1), **binding(agent, p),
        claims={'from_sequence': 1, 'through_sequence': action.sequence,
                'action_hashes': [action.event_hash], 'unexplained_external_actions': []},
        reference='runtime:coverage'), EvidenceSigner(private))
    attest(session, agent.name, proof, evidence=evidence, actor_id='runtime', now=NOW)
    usecases = _usecases(['READ'])
    original = decide(envelope, load_snapshot(session, as_of=NOW), usecases=usecases, as_of=NOW)
    assert original.outcome == Outcome.ALLOW
    original = record(session, original, evidence=evidence, now=NOW)
    evidence.clock = lambda: NOW + timedelta(seconds=later_seconds)
    evidence.emit(session, correlation_id='later', event_type=EventType.TOOL_CALL,
                  actor_id=agent.name, payload={'tool': envelope.tool})
    replayed = replay_decision(session, original.request_id, envelope, usecases=usecases)
    assert replayed.outcome == original.outcome and replayed.checks == original.checks
    assert replayed.envelope_hash == original.envelope_hash
    current = decide(envelope, load_snapshot(session, as_of=NOW), usecases=usecases, as_of=NOW)
    assert current.outcome == Outcome.DENY
    assert any(c.check == 'A5-MONITORING' for c in current.failures())


def test_mapped_tools_subject_cannot_hide_invoked_tool(session, setup):
    """External trust must cover both the invoked tool and mapped tool subjects."""
    from test_assurance import assertion
    from kognita.assurance.trust import verify_assertion
    _, _, _, _, envelope, at, config = setup
    p = config()
    mapped = replace(envelope, subjects={'tools': 'other'},
                     context={'_effective_subjects': {'tools': ['other']}})
    proof = assertion(setup, scopes={'tools': ['other']})
    with pytest.raises(ValueError, match='authority expansion'):
        verify_assertion(proof, p, mapped, at)
    proof = assertion(setup, scopes={'tools': [envelope.tool, 'other']})
    assert verify_assertion(proof, p, mapped, at).subject == envelope.agent_name


def test_a4_report_rejects_approval_under_superseded_policy(session, setup):
    """Deployment identity alone cannot revive a previous policy's approval."""
    a, w, _, _, _, _, config = setup
    p = config()
    def approve(digest):
        """Record an accountable-owner candidate approval for this regression."""
        return approve_update(session, a.name, candidate_digest=digest, provider_change='change',
            regression_reference='suite', rollback_target='previous', rollback_plan='restore',
            evidence=w, actor_id='owner')
    approve(a.deployment_hash)
    configure(session, a.name, p.model_copy(update={'client_impact': 'changed'}), evidence=w, actor_id='owner')
    approve('c' * 64)
    report = conformance_report(session, a.name, 'MAS')
    row = next(r for r in report['expectations'] if r['control'] == 'A4')
    assert row['status'] == 'gap' and not row['evidence_sequences']
    approve(a.deployment_hash)
    report = conformance_report(session, a.name, 'MAS')
    assert next(r for r in report['expectations'] if r['control'] == 'A4')['status'] == 'evidenced'


def test_signed_assurance_cli_separates_manifest_and_evidence_keys(tmp_path, capsys):
    """Signed write commands preserve chain signing and independent manifest signing."""
    import json
    from sqlmodel import Session
    from kognita.cli import main
    from kognita.db import make_engine, create_all
    from kognita.evidence import EvidenceWriter
    from kognita.signing import EvidenceSigner, _generate
    from kognita.assurance.reports import verify_manifest
    from kognita.models import Agent
    db = tmp_path / 'signed.db'
    private, public = _generate()
    manifest_private, manifest_public = _generate()
    key, manifest_key = tmp_path / 'chain.pem', tmp_path / 'manifest.pem'
    key.write_bytes(private); manifest_key.write_bytes(manifest_private)
    engine = make_engine(str(db)); create_all(engine)
    with Session(engine) as session:
        register(session, name='signed-worker', owner_exec='owner')
        # Roll back the fixture setup before creating the signed chain.
        # Reuse its approved identity without weakening signing checks.
        identity = session.exec(select(Agent)).first().model_dump(mode='json')
        session.rollback()
    with Session(engine) as session:
        session.add(Agent.model_validate(identity)); session.flush()
        writer = EvidenceWriter(engine, signing=EvidenceSigner(private))
        writer.emit(session, correlation_id='identity:signed-worker', event_type=EventType.POLICY_CHANGE,
                    payload={'action':'DEPLOYMENT_APPROVED','agent':'signed-worker','_identity':identity})
        session.commit()
    policy_file = tmp_path / 'policy.json'
    policy_file.write_text(json.dumps({'use_case':'p','audience':'bank'}))
    args = ['assurance','configure','--db',str(db),'--agent','signed-worker','--actor','owner',
            '--input',str(policy_file)]
    assert main(args) == 1
    capsys.readouterr()
    assert main([*args, '--evidence-signing-key',str(key)]) == 0
    capsys.readouterr()
    assert main(['assurance','manifest','--db',str(db),'--agent','signed-worker','--actor','owner',
        '--evidence-signing-key',str(key),'--signing-key',str(manifest_key)]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert verify_manifest(exported, manifest_public).agent['name'] == 'signed-worker'
    with Session(engine) as session:
        assert verify_chain(session, public_key=public) == 3
    engine.dispose()
