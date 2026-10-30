from dataclasses import replace
from types import SimpleNamespace

import pytest
from sqlmodel import select

from kognita import (VulnerabilityFlag, detect_sensitive, check_sensitive,
                     raise_vulnerability_flag, change_treatment)
from kognita.evidence import verify_chain
from kognita.exceptions import KognitaError
from kognita.models import Interaction, RmReview, EvidenceEvent, UseCase
from kognita.retention import RetentionStore
from kognita.review import define_human_boundary, review, PACKET_FIELDS
from kognita.vocabulary import CheckResult


def _usecase(*categories):
    return SimpleNamespace(usecase_id='protect', permitted_sensitive=set(categories))


@pytest.mark.parametrize('category,text', [
    ('religion', 'client is Muslim'), ('health', 'client has diabetes'),
    ('political_views', 'client votes Republican'), ('ethnicity', 'Asian ancestry'),
    ('sexuality', 'client is gay'), ('vulnerability', 'client appears confused'),
])
def test_categories_and_default_denial(category, text):
    assert category in detect_sensitive(text)
    assert check_sensitive(text).result == CheckResult.FAIL
    if category not in ('health', 'vulnerability'):
        assert check_sensitive(text, _usecase(category)).result == CheckResult.PASS


def test_patterns_normalization_and_classifier_failures():
    assert 'religion' in detect_sensitive('Ｍｕｓ\u200blim')
    assert detect_sensitive('ordinary meeting') == set()
    assert check_sensitive('ordinary meeting').result == CheckResult.PASS
    assert detect_sensitive('custom', classifier=lambda _: ['health']) == {'health'}
    assert detect_sensitive('Muslim', classifier=lambda _: [], extend=False) == set()
    assert detect_sensitive('Muslim', classifier=lambda _: ['health']) == {'religion', 'health'}
    for classifier in (lambda _: ['unknown'], lambda _: 'health', lambda _: 1):
        assert check_sensitive('custom', classifier=classifier).result == CheckResult.FAIL
    assert check_sensitive('Muslim', _usecase('*')).result == CheckResult.FAIL
    assert check_sensitive('client has dementia', _usecase('health', 'vulnerability'),
                           classifier=lambda _: [], extend=False).check == 'SENSITIVE_UNSUPPORTED_DIAGNOSIS'
    assert check_sensitive('health information', _usecase('health')).result == CheckResult.PASS
    assert check_sensitive('client seems confused', _usecase('vulnerability')).check == 'VULNERABILITY_RM_REVIEW_REQUIRED'


def _flag(session, evidence):
    session.add(Interaction(id='journey', client_id='client', use_case='protect',
                            approval_gate='human_review'))
    session.flush()
    return raise_vulnerability_flag(session, 'journey', 'Client seems confused during the meeting',
                                    usecase=_usecase('vulnerability'), evidence=evidence, actor_id='agent')


def _approve(session, evidence, flag, treatment, outcome='approve'):
    define_human_boundary(session, 'protect', intervention='before treatment', who='rm',
                          must_review='signal and treatment', evidence_seen='signal',
                          on_reject='no change', minimum_review_seconds=0, evidence=evidence)
    packet = {key: 'explained' for key in PACKET_FIELDS}
    from kognita.canonical import canonical_json
    packet['proposed_action'] = canonical_json({'flag_id': flag.flag_id, 'treatment': treatment})
    packet['agent_confidence'] = 0.8
    packet['policy_checks'] = [{'citation': 'human review'}]
    review(session, 'journey', rm='rm', outcome=outcome, packet=packet,
           rendered='Offer extra explanation', provenance='AI', evidence=evidence)
    return session.exec(select(RmReview)).all()[-1].id


def test_flag_requires_permission_explanation_and_scope(session, evidence):
    flag = _flag(session, evidence)
    assert isinstance(flag, VulnerabilityFlag) and flag.review_required
    assert 'confused' in RetentionStore().read(session, flag.signal_hash)
    event = session.exec(select(EvidenceEvent)).all()[-1]
    assert event.payload['route'] == 'RM'
    assert 'confused' not in str(event.payload)
    for signal, usecase, code in [
        ('confused', _usecase(), 'SENSITIVE_NOT_PERMITTED'),
        ('client is vulnerable', _usecase('vulnerability'), 'VULNERABILITY_DETERMINATION_BLOCKED'),
        ('client has dementia', _usecase('health', 'vulnerability'), 'SENSITIVE_UNSUPPORTED_DIAGNOSIS'),
        ('normal meeting', _usecase('vulnerability'), 'VULNERABILITY_SIGNAL_REQUIRED'),
    ]:
        with pytest.raises(KognitaError, match=code):
            raise_vulnerability_flag(session, 'journey', signal, usecase=usecase,
                                     evidence=evidence, actor_id='agent')
    with pytest.raises(KognitaError, match='SCOPE'):
        raise_vulnerability_flag(session, 'missing', 'confused', usecase=_usecase('vulnerability'),
                                 evidence=evidence, actor_id='agent')
    assert verify_chain(session) == 1


def test_treatment_requires_exact_recorded_human_decision(session, evidence):
    flag = _flag(session, evidence)
    kwargs = dict(usecase=_usecase('vulnerability'), evidence=evidence, actor_id='agent')
    assert change_treatment(session, flag, 'extra time', **kwargs).result == CheckResult.FAIL
    review_id = _approve(session, evidence, flag, 'extra time')
    assert change_treatment(session, flag, 'extra time', review_id=review_id, **kwargs).result == CheckResult.PASS
    assert change_treatment(session, flag, 'restrict access', review_id=review_id, **kwargs).result == CheckResult.FAIL
    assert change_treatment(session, replace(flag, flag_id='forged'), 'extra time',
                            review_id=review_id, **kwargs).result == CheckResult.FAIL
    assert change_treatment(session, replace(flag, client_id='another'), 'extra time',
                            review_id=review_id, **kwargs).result == CheckResult.FAIL
    kwargs['usecase'] = _usecase()
    assert change_treatment(session, flag, 'extra time', review_id=review_id, **kwargs).result == CheckResult.FAIL
    verify_chain(session)


def test_rejected_review_cannot_authorize(session, evidence):
    flag = _flag(session, evidence)
    review_id = _approve(session, evidence, flag, 'extra time', outcome='reject')
    assert change_treatment(session, flag, 'extra time', review_id=review_id,
                            usecase=_usecase('vulnerability'), evidence=evidence,
                            actor_id='agent').result == CheckResult.FAIL


def test_usecase_record_and_decision_hook():
    from kognita.sensitive import _checks
    from kognita.envelope import Envelope
    from kognita.usecases import from_record
    row = UseCase(usecase_id='protect', permitted_sensitive={'health'})
    assert row.permitted_sensitive == {'health'}
    assert from_record({'sensitive': ['health']}).permitted_sensitive == {'health'}
    envelope = Envelope(principal='rm', purpose='protect', tool='meeting',
                        arguments={'nested': {'religion': 'Muslim'}})
    assert _checks(envelope, None, {})[0].result == CheckResult.FAIL
    envelope = replace(envelope, arguments={'text': 'Ｍｕｓ\u200blim'})
    assert _checks(envelope, None, {})[0].result == CheckResult.FAIL
    register = SimpleNamespace(get=lambda _: _usecase('religion'))
    assert _checks(envelope, register, {})[0].result == CheckResult.PASS
    assert _checks(envelope, register, {'content': 'client seems confused'})[0].result == CheckResult.FAIL


def test_policy_yaml_allowlist():
    from kognita.policy import parse_yaml, parse_rule
    from kognita.rules import content_policy
    from kognita.envelope import Envelope, RuleContext
    from kognita.models import Policy
    from datetime import datetime, timezone
    text = '''id: special
regime: INTERNAL
rule_type: CONTENT_POLICY
citation: Sensitive controls
effective_from: 2026-01-01
rule:
  sensitive: [religion]
'''
    parsed = parse_rule(parse_yaml(text), source="test")
    assert parsed
    policy = Policy(regime='INTERNAL', rule_type='CONTENT_POLICY', citation='Sensitive controls',
                    rule={'sensitive': ['religion']})
    context = RuleContext(envelope=Envelope(principal='rm', purpose='p', tool='t'),
                          attributes={'content': 'Muslim'}, subjects={}, as_of=datetime.now(timezone.utc))
    assert content_policy(policy, context)[0].result == CheckResult.PASS
    policy.rule = {'sensitive': []}
    assert content_policy(policy, context)[0].result == CheckResult.FAIL


def test_review_must_be_final_and_same_interaction(session, evidence):
    flag = _flag(session, evidence)
    review_id = _approve(session, evidence, flag, 'extra time')
    row = session.get(RmReview, review_id)
    row.status = 'marked'
    session.flush()
    assert change_treatment(session, flag, 'extra time', review_id=review_id,
                            usecase=_usecase('vulnerability'), evidence=evidence,
                            actor_id='agent').result == CheckResult.FAIL
    row.status = 'final'
    session.add(Interaction(id='other', client_id='client', use_case='protect'))
    session.flush()
    row.interaction_id = 'other'
    session.flush()
    assert change_treatment(session, flag, 'extra time', review_id=review_id,
                            usecase=_usecase('vulnerability'), evidence=evidence,
                            actor_id='agent').result == CheckResult.FAIL


def test_reconstruction_lists_flags(session, evidence):
    from kognita.reconstruct import reconstruct
    flag = _flag(session, evidence)
    assert reconstruct(session, 'journey')['sensitive'][0]['flag_id'] == flag.flag_id


def test_two_signatures_and_retained_packet_integrity(session, evidence):
    from kognita.review import confirm_review
    from kognita.models import RetainedContent
    from kognita.canonical import canonical_json
    flag = _flag(session, evidence)
    interaction = session.get(Interaction, 'journey')
    interaction.approval_gate = 'two_signature'
    session.flush()
    review_id = _approve(session, evidence, flag, 'extra time')
    kwargs = dict(review_id=review_id, usecase=_usecase('vulnerability'),
                  evidence=evidence, actor_id='agent')
    assert change_treatment(session, flag, 'extra time', **kwargs).result == CheckResult.FAIL
    confirm_review(session, 'journey', confirmer='second-rm', evidence=evidence)
    assert change_treatment(session, flag, 'extra time', **kwargs).result == CheckResult.PASS
    row = session.get(RmReview, review_id)
    retained = session.get(RetainedContent, row.packet_hash)
    retained.body = canonical_json({'proposed_action': canonical_json(
        {'flag_id': flag.flag_id, 'treatment': 'restrict access'})})
    session.flush()
    assert change_treatment(session, flag, 'restrict access', **kwargs).result == CheckResult.FAIL


def test_communication_has_sensitive_gate_without_policy_opt_in(session):
    from kognita.communication import _policy_checks
    from kognita.models import utcnow
    for text, expected in [('ordinary meeting', CheckResult.PASS),
                           ('client is Muslim', CheckResult.FAIL),
                           ('client has dementia', CheckResult.FAIL)]:
        checks = _policy_checks(session, {'content': text, 'use_case': 'protect'},
                                utcnow(), tool='send_client_communication')
        assert checks[0].result == expected
