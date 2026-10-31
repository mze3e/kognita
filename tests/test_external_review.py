"""T3 public external review API and fail-closed signature verification."""
import copy

import pytest
from sqlmodel import Session

from kognita.db import create_all, make_engine
from kognita.evidence import ChainBreak, EvidenceWriter, export_chain, verify_export
from kognita.external_review import (
    annotate, request_review_period, submit_review,
)
from kognita.signing import EvidenceSigner, _generate, _key_id, _public_pem
from kognita.vocabulary import EventType


@pytest.fixture
def review():
    engine = make_engine()
    create_all(engine)
    private, root = _generate()
    reviewer = EvidenceSigner(private)
    writer_key = EvidenceSigner(_generate()[0])
    with Session(engine) as session:
        writer = EvidenceWriter(engine, signing=reviewer)
        writer.emit(session, correlation_id='decision', event_type=EventType.POLICY_DECISION)
        writer.emit(session, correlation_id='rotation', event_type=EventType.KEY_ROTATED,
                    payload={'key_id': _key_id(writer_key._load().public_key()),
                             '_public_key': _public_pem(writer_key).decode(), '_escrow': 'bank vault'})
        session.commit()
        period = request_review_period(session, start=1, end=1,
                                        key_id=_key_id(reviewer._load().public_key()), public_key=root)
        yield session, reviewer, EvidenceWriter(engine, signing=writer_key), root, period


def test_signed_review_into_chain(review):
    session, reviewer, writer, root, period = review
    annotated = annotate(period, sequence=1, text='Decision checked')
    assert annotated != period
    event = submit_review(session, annotated, signing=reviewer, evidence=writer, public_key=root)
    session.commit()
    assert event.event_type == EventType.EXTERNAL_REVIEW_SUBMITTED
    assert event.key_id != event.payload['review']['key_id']
    assert verify_export(export_chain(session), public_key=root) == 3
    with pytest.raises(ValueError, match='T3'):
        submit_review(session, annotated, signing=reviewer, evidence=writer, public_key=root)


def test_invalid_period_annotation_and_key_refused(review):
    session, reviewer, writer, root, period = review
    for start, end in ((0, 1), (2, 1), (1, 3), (True, 1)):
        with pytest.raises(ValueError, match='T3'):
            request_review_period(session, start=start, end=end,
                                   key_id=_key_id(reviewer._load().public_key()), public_key=root)
    for sequence, text in ((2, 'outside'), (1, ''), (True, 'invalid')):
        with pytest.raises(ValueError, match='T3'):
            annotate(period, sequence=sequence, text=text)
    with pytest.raises(ValueError, match='T3'):
        submit_review(session, period, signing=EvidenceSigner(_generate()[0]),
                       evidence=writer, public_key=root)
    with pytest.raises(ValueError, match='T3'):
        request_review_period(session, start=1, end=1, key_id='unknown', public_key=root)


def test_reviewer_signature_tamper_refused(review):
    session, reviewer, writer, root, period = review
    submit_review(session, period, signing=reviewer, evidence=writer, public_key=root)
    artifact = copy.deepcopy(export_chain(session))
    artifact['events'][-1]['payload']['signature'] = '00' * 64
    with pytest.raises(ChainBreak, match='T3'):
        verify_export(artifact, public_key=root)


def test_revoked_reviewer_refused(review):
    from datetime import datetime, timedelta, timezone
    session, reviewer, writer, root, period = review
    writer.emit(session, correlation_id='revoke', event_type=EventType.KEY_REVOKED,
                payload={'key_id': _key_id(reviewer._load().public_key()),
                         'recorded_at': (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat()})
    with pytest.raises(ValueError, match='T3.*R2'):
        submit_review(session, period, signing=reviewer, evidence=writer, public_key=root)


def test_missing_trust_root_and_changed_prefix_refused(review):
    from kognita.external_review import _decode, _period
    session, reviewer, writer, root, period = review
    with pytest.raises(ValueError, match='T3'):
        submit_review(session, period, signing=reviewer, evidence=writer, public_key=None)
    body = _decode(period)
    body['head'] = '0' * 64
    with pytest.raises(ValueError, match='T3'):
        submit_review(session, _period(body), signing=reviewer, evidence=writer, public_key=root)


def test_review_before_revocation_keeps_verifying(review):
    from datetime import datetime, timedelta, timezone
    session, reviewer, writer, root, period = review
    submit_review(session, period, signing=reviewer, evidence=writer, public_key=root)
    writer.emit(session, correlation_id='revoke', event_type=EventType.KEY_REVOKED,
                payload={'key_id': _key_id(reviewer._load().public_key()),
                         'recorded_at': (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat()})
    session.commit()
    assert verify_export(export_chain(session), public_key=root) == 4
