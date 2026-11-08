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
