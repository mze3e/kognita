"""0.8 adversarial admission, approvals, signed artifacts and reports."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import select

from kognita.assurance import AssurancePolicy, Assertion, Attestation, configure, sign_record
from kognita.assurance.records import binding, context, events, agent_for
from kognita.assurance.runtime import (attest, admission_checks, load_state, coverage_report,
                                      request_digest, transfer_digest)
from kognita.assurance.lifecycle import (Component, ProductionOutcome, approve_components,
    approve_update, advisory, record_outcome)
from kognita.assurance.reports import export_manifest, verify_manifest, compare_manifest, conformance_report, PROFILES
from kognita.assurance.trust import verify_assertion
from kognita.canonical import canonical_hash
from kognita.envelope import Envelope
from kognita.evidence import EvidenceWriter, verify_chain
from kognita.models import EvidenceEvent, Interaction, utcnow
from kognita.signing import EvidenceSigner, _generate
from kognita.vocabulary import CheckResult, EventType
from fixtures.agents import register


@pytest.fixture
def setup(session):
    at = utcnow()
    writer = EvidenceWriter(session.get_bind())
    agent = register(session, name='assurance-agent', owner_exec='owner')
    private, public = _generate()
    signer = EvidenceSigner(private)
    env = Envelope(principal='alice', purpose='p', tool='read_note', agent_name=agent.name)
    def config(**kw):
        p = AssurancePolicy(use_case='p', audience='bank', attestors={'runtime': public.decode()},
                            issuers={'partner': public.decode()}, **kw)
        configure(session, agent.name, p, evidence=writer, actor_id='owner')
        return p
    return agent, writer, signer, public, env, at, config


def packet(setup, p, kind, claims):
    agent, writer, signer, public, env, at, _ = setup
    return sign_record(Attestation(kind=kind, issuer='runtime', subject=agent.name,
        issued_at=at - timedelta(seconds=1), expires_at=at + timedelta(hours=1),
        **binding(agent, p), claims=claims, reference='independent:123'), signer)


def gates(session, setup):
    a, _, _, _, e, at, _ = setup
    return {c.check: c.result for c in admission_checks(e, a, load_state(session, [a.name], utcnow()), utcnow())}


def test_owner_only_policy_and_independent_attestors(session, setup):
    a, w, s, public, e, at, config = setup
    p = config()
    with pytest.raises(ValueError, match='owner'):
        configure(session, a.name, p, evidence=w, actor_id=a.name)
    with pytest.raises(ValueError, match='workloads'):
        configure(session, a.name, p.model_copy(update={'attestors': {a.name: public.decode()}}), evidence=w, actor_id='owner')
    verify_chain(session)


def assertion(setup, **kw):
    a, _, s, _, e, at, _ = setup
    body = dict(issuer='partner', subject=a.name, principal=e.principal, purpose=e.purpose,
        audience='bank', scopes={'tools': ['read_note'], 'client': ['one']},
        issued_at=at-timedelta(minutes=1), expires_at=at+timedelta(minutes=10))
    return sign_record(Assertion(**{**body, **kw}), s)


@pytest.mark.parametrize('change', [dict(issuer='unknown'), dict(audience='other'),
    dict(subject='other'), dict(principal='other'), dict(purpose='other'),
    dict(scopes={'tools': ['execute']}), dict(expires_at=datetime(2026,1,1,tzinfo=timezone.utc),
    issued_at=datetime(2025,1,1,tzinfo=timezone.utc))])
def test_external_assertions_fail_closed(session, setup, change):
    a,w,s,public,e,at,config=setup
    p=config(external_required=True)
    with pytest.raises(ValueError):
        verify_assertion(assertion(setup, **change), p, e, at)
    assert gates(session, setup)['A2-EXTERNAL-TRUST'] == CheckResult.FAIL


def test_external_signature_and_nontransitive_attenuation(session, setup):
    a,w,s,public,e,at,config=setup
    p=config(external_required=True)
    root=assertion(setup, subject='orchestrator')
    child=assertion(setup, delegation_depth=1, parent_digest=canonical_hash(root),
        issued_at=at, expires_at=at+timedelta(minutes=1))
    assert verify_assertion(child,p,e,at,parent=[root]).subject==a.name
    with pytest.raises(ValueError): verify_assertion(child,p,e,at)
    broad=assertion(setup, delegation_depth=1, parent_digest=canonical_hash(root),
        scopes={'tools':['read_note','execute']}, issued_at=at, expires_at=at+timedelta(minutes=1))
    with pytest.raises(ValueError,match='expansion'): verify_assertion(broad,p,e,at,parent=[root])
    child['document']['body']['scopes']['tools'].append('execute')
    with pytest.raises(ValueError): verify_assertion(child,p,e,at,parent=[root])


def test_privacy_invalidates_on_material_change(session, setup):
    a,w,s,public,e,at,config=setup
    p=config(privacy_required=True)
    assert gates(session,setup)['A6-PRIVACY']==CheckResult.FAIL
    doc=packet(setup,p,'privacy',{'context_digest':canonical_hash(context(session,a,p))})
    attest(session,a.name,doc,evidence=w,actor_id='runtime',now=at)
    assert gates(session,setup)['A6-PRIVACY']==CheckResult.PASS
    a.purpose='new purpose'
    session.add(a);session.flush()
    assert gates(session,setup)['A6-PRIVACY']==CheckResult.FAIL


@pytest.mark.parametrize('field,value', [('model','new-provider'),('autonomy_level',1),
    ('permitted_systems',['other']),('tool_allowlist',{'new-tool':{}})])
def test_privacy_change_dimensions(session,setup,field,value):
    a,w,s,public,e,at,config=setup
    p=config(privacy_required=True)
    attest(session,a.name,packet(setup,p,'privacy',{'context_digest':canonical_hash(context(session,a,p))}),
        evidence=w,actor_id='runtime',now=at)
    setattr(a,field,value);session.add(a);session.flush()
    assert gates(session,setup)['A6-PRIVACY']==CheckResult.FAIL


def test_environment_requires_independent_complete_fresh_binding(session,setup):
    a,w,s,public,e,at,config=setup
    p=config(environment_required=True)
    with pytest.raises(ValueError,match='five'):
        attest(session,a.name,packet(setup,p,'environment',{'enforced':True}),evidence=w,actor_id='runtime',now=at)
    claims={k:'runtime:ref' for k in ('network_policy','isolation','credential_scope','sandbox_identity','termination_path')}
    doc=packet(setup,p,'environment',{**claims,'enforced':True})
    with pytest.raises(ValueError,match='independent'):
        attest(session,a.name,doc,evidence=w,actor_id=a.name,now=at)
    attest(session,a.name,doc,evidence=w,actor_id='runtime',now=at)
    assert gates(session,setup)['A7-ENVIRONMENT']==CheckResult.PASS
    a.deployment_hash='a'*64;session.add(a);session.flush()
    assert gates(session,setup)['A7-ENVIRONMENT']==CheckResult.FAIL


def test_coverage_detects_both_directions_and_new_actions(session,setup):
    a,w,s,public,e,at,config=setup
    p=config(monitoring_required=True)
    event=w.emit(session,correlation_id='action',event_type=EventType.TOOL_CALL,
                 actor_id=a.name,payload={'tool':e.tool})
    # emit returns an event (hash is the matching independent telemetry identity).
    rows=session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
    last=rows[-1]
    c={'from_sequence':1,'through_sequence':last.sequence,'action_hashes':[], 'unexplained_external_actions':['network:unlogged']}
    attest(session,a.name,packet(setup,p,'monitoring',c),evidence=w,actor_id='runtime',now=at)
    report=coverage_report(session,a.name,now=at)
    assert last.event_hash in report['missing'] and not report['complete']
    c['action_hashes']=[last.event_hash];c['unexplained_external_actions']=[]
    attest(session,a.name,packet(setup,p,'monitoring',c),evidence=w,actor_id='runtime',now=at)
    assert gates(session,setup)['A5-MONITORING']==CheckResult.PASS
    w.emit(session,correlation_id='action',event_type=EventType.MODEL_CALL,actor_id=a.name,payload={'model':'new'})
    assert gates(session,setup)['A5-MONITORING']==CheckResult.FAIL


def test_exact_instruction_binding_and_transfer(session,setup):
    a,w,s,public,e,at,config=setup
    p=config(instruction_tools=[e.tool],transfer_required=True,technology_origin='CN',
             deployment_jurisdiction='SG',transfer_policy='legal:policy')
    doc=packet(setup,p,'instruction',{'request_digest':request_digest(e),
        'verification_channel':'second channel','maker':'maker','checker':'checker'})
    digest=attest(session,a.name,doc,evidence=w,actor_id='runtime',now=at)
    e=replace(e,context={'instruction_attestation':digest})
    state=load_state(session,[a.name],utcnow())
    assert admission_checks(e,a,state,at)[-1].result==CheckResult.PASS
    changed=replace(e,arguments={'amount':100000})
    assert admission_checks(changed,a,state,at)[-1].result==CheckResult.FAIL
    assert gates(session,setup)['A10-TRANSFER']==CheckResult.FAIL
    attest(session,a.name,packet(setup,p,'transfer',{'transfer_digest':transfer_digest(p),'approved':True}),
           evidence=w,actor_id='runtime',now=at)
    assert gates(session,setup)['A10-TRANSFER']==CheckResult.PASS


def test_components_advisory_quarantines_all_affected(session,setup):
    a,w,s,public,e,at,config=setup
    p=config(components_required=True)
    components=[dict(name='proxy',version='1',sbom_digest='f'*64,provenance='build:approved')]
    result=approve_components(session,a.name,components,evidence=w,actor_id='owner')
    assert result['digest']==canonical_hash(result['components'])
    assert gates(session,setup)['A9-COMPONENTS']==CheckResult.PASS
    affected=advisory(session,'proxy','1','advisory:123',evidence=w,actor_id='security')
    assert affected[0]['agent']==a.name
    assert gates(session,setup)['A9-COMPONENTS']==CheckResult.FAIL
    verify_chain(session)


def test_manifest_signature_and_observed_divergence(session,setup):
    a,w,s,public,e,at,config=setup
    config()
    result=export_manifest(session,a.name,s,evidence=w,actor_id='owner',now=at)
    assert verify_manifest(result,public).agent['name']==a.name
    w.emit(session,correlation_id='unknown',event_type=EventType.TOOL_CALL,
        actor_id=a.name,payload={'tool':'rogue'})
    assert compare_manifest(session,a.name,result,public)['findings']
    result['manifest']['document']['body']['agent']['version']='tampered'
    with pytest.raises(ValueError):verify_manifest(result,public)


def test_outcome_threshold_and_duplicate_source(session,setup):
    a,w,s,public,e,at,config=setup
    config(max_adverse_outcomes=1)
    session.add(Interaction(id='journey',client_id='client',use_case='p'));session.flush()
    result=ProductionOutcome(interaction_id='journey',intended='correct advice',actual='wrong advice',
        client_impact='correction needed',adverse=True,source_reference='bank:outcome',outcome_id='one')
    assert record_outcome(session,a.name,result,evidence=w,actor_id='owner',now=at)['breached']
    assert any(r.event_type==EventType.REVALIDATION_TRIGGERED for r in session.exec(select(EvidenceEvent)))
    with pytest.raises(ValueError,match='duplicate'):
        record_outcome(session,a.name,result,evidence=w,actor_id='owner',now=at)
    verify_chain(session)


@pytest.mark.parametrize('profile', PROFILES)
def test_profiles_are_honest_and_point_in_time(session,setup,profile):
    a,w,s,public,e,at,config=setup
    config()
    before=utcnow()
    report=conformance_report(session,a.name,profile,as_of=before)
    assert report['expectations'] and all(r['status']=='gap' for r in report['expectations'])
    export_manifest(session,a.name,s,evidence=w,actor_id='owner')
    assert conformance_report(session,a.name,profile,as_of=before)==report


def test_tampered_evidence_refuses_assurance(session,setup):
    from evidence_guard import tamper_for_test
    a,w,s,public,e,at,config=setup
    config();session.commit()
    first=events(session,a.name)[0]
    tamper_for_test(session,first.sequence,{'action':'fake'})
    with pytest.raises(Exception): load_state(session,[a.name],utcnow())


def test_update_requires_current_evaluation_and_exact_owner_packet(session,setup):
    from kognita.registry import approve_deployment, _deployment_hash
    from kognita.evaluation import EvaluationResult, record_evaluation, METRICS
    a,w,s,public,e,at,config=setup
    p=config(update_required=True)
    deployment={**a.tool_allowlist['_deployment'],'instructions':'changed behaviour'}
    candidate=type(a).model_validate(a.model_dump())
    candidate.tool_allowlist={**a.tool_allowlist,'_deployment':deployment}
    digest=_deployment_hash(candidate,deployment)
    old=a.deployment_hash
    with pytest.raises(ValueError,match='A4-UPDATE'):
        approve_deployment(session,a.name,deployment=deployment,evidence=w,actor_id='owner')
    assert a.deployment_hash==old
    approve_update(session,a.name,candidate_digest=digest,provider_change='provider:2',regression_reference='suite:passed',
        rollback_target=old,rollback_plan='restore old image',evidence=w,actor_id='owner')
    with pytest.raises(ValueError,match='A4-UPDATE'):
        approve_deployment(session,a.name,deployment=deployment,evidence=w,actor_id='owner')
    record_evaluation(session,EvaluationResult(agent_id=a.name,version=a.version,model_version=a.model,
        level='system',deployment_hash=digest,metrics={k:0.01 if k=='hallucination_rate' else 0.99 for k in METRICS}),evidence=w)
    approve_deployment(session,a.name,deployment=deployment,evidence=w,actor_id='owner')
    assert a.deployment_hash==digest
    # An old approved packet cannot authorize a different version.
    with pytest.raises(ValueError,match='A4-UPDATE'):
        approve_deployment(session,a.name,deployment={**deployment,'instructions':'third'},evidence=w,actor_id='owner')


def test_shared_decision_and_release_recheck(session,evidence):
    from test_authority_s3 import _setup, _decide
    from kognita.governance import load_snapshot, _standing_checks
    from conftest import NOW
    from kognita.vocabulary import Outcome
    _,_,_,env=_setup(session,evidence)
    assert _decide(session,env).outcome==Outcome.ALLOW
    configure(session,'worker',AssurancePolicy(use_case='READ',audience='bank',privacy_required=True),
        evidence=evidence,actor_id='owner')
    verdict=_decide(session,env)
    assert verdict.outcome==Outcome.DENY
    assert any(c.check=='A6-PRIVACY' and c.result==CheckResult.FAIL for c in verdict.checks)
    snap=load_snapshot(session,as_of=NOW,_agent_name='worker')
    held=_standing_checks(env,snap,at=NOW,attributes={'deployment_hash':snap.agents['worker'].deployment_hash})
    assert any(c.check=='A6-PRIVACY' for c in held)


@pytest.mark.parametrize('scenario', __import__('kognita.assurance.drills',fromlist=['SCENARIOS']).SCENARIOS)
def test_adversarial_drills_have_verified_paths_and_restore_state(session,evidence,scenario,monkeypatch):
    import socket
    from test_authority_s3 import _setup
    from kognita.assurance.drills import simulate
    from kognita.evidence import export_chain, verify_export
    from conftest import NOW
    def no_network(*args,**kw): raise AssertionError('drill contacted network')
    monkeypatch.setattr(socket,'create_connection',no_network)
    _,_,_,env=_setup(session,evidence)
    configure(session,'worker',AssurancePolicy(use_case='READ',audience='bank'),evidence=evidence,actor_id='owner')
    session.commit()
    baseline=export_chain(session)
    before=agent_for(session,'worker').model_dump()
    result=simulate(session,'worker',scenario,env,evidence=evidence,actor_id='owner',now=NOW)
    assert result['passed'] and result['network_calls']==0
    assert before==agent_for(session,'worker').model_dump()
    assert verify_chain(session)==len(baseline['events'])+1
    combined=baseline['events']+result['path']
    assert verify_export(dict(baseline,events=combined,event_count=len(combined),
        head_hash=combined[-1]['event_hash']))==len(combined)


def test_activation_gate_and_policy_changes_invalidate_attestations(session,setup):
    from kognita.evaluation import check_activation
    a,w,s,public,e,at,config=setup
    p=config(privacy_required=True)
    with pytest.raises(ValueError,match='A6-PRIVACY'):check_activation(session,a)
    attest(session,a.name,packet(setup,p,'privacy',{'context_digest':canonical_hash(context(session,a,p))}),
        evidence=w,actor_id='runtime',now=at)
    check_activation(session,a)
    config(privacy_required=True,jurisdictions=['new'])
    with pytest.raises(ValueError,match='A6-PRIVACY'):check_activation(session,a)


def test_core_only_policy_does_not_require_signing_extra(session,setup,monkeypatch):
    import kognita.signing
    a,w,s,public,e,at,config=setup
    def unavailable():raise AssertionError('optional signing loaded')
    monkeypatch.setattr(kognita.signing,'_crypto',unavailable)
    configure(session,a.name,AssurancePolicy(use_case='p',audience='bank'),evidence=w,actor_id='owner')


def test_cli_config_and_profile_report(tmp_path,capsys):
    from kognita.cli import main
    from kognita.db import make_engine,create_all
    from sqlmodel import Session
    import json
    db=tmp_path/'assurance.db'
    engine=make_engine(str(db));create_all(engine)
    with Session(engine) as session:
        register(session,name='worker',owner_exec='owner');session.commit()
    file=tmp_path/'policy.json';file.write_text(json.dumps({'use_case':'p','audience':'bank'}))
    assert main(['assurance','configure','--db',str(db),'--agent','worker','--actor','owner','--input',str(file)])==0
    capsys.readouterr()
    assert main(['assurance','conformance','--db',str(db),'--agent','worker','--profile','MAS'])==0
    assert json.loads(capsys.readouterr().out)['profile']=='MAS'
    engine.dispose()


@pytest.mark.parametrize('path',['gateway','mcp'])
def test_ingress_denies_before_transport_without_required_attestation(session,evidence,engine,path):
    from test_identity_radius_completion import _setup, _invoke
    from kognita.vocabulary import Outcome
    agent,headers=_setup(session,evidence,engine,path)
    baseline,calls=_invoke(engine,evidence,session,path,headers)
    assert baseline.status==200
    configure(session,agent.name,AssurancePolicy(use_case='COLLABORATION',audience='bank',
        environment_required=True),evidence=evidence,actor_id=agent.owner_exec)
    response,calls=_invoke(engine,evidence,session,path,headers)
    assert response.evaluation.outcome==Outcome.DENY and not calls
    assert any(c.check=='A7-ENVIRONMENT' for c in response.evaluation.failures())


def test_malformed_proof_headers_never_confer_trust():
    from kognita.assurance.runtime import request_context
    assert request_context({'x-kognita-agent-assertion':'{bad'})=={'external_assertion':{}}
    assert request_context({'x-kognita-agent-assertion':'null'})=={'external_assertion':{}}
    assert request_context({'x-kognita-agent-assertion':'x'*65537})=={'external_assertion':{}}
    assert request_context({})=={}


def test_outcome_step_down_is_evidenced_and_transactional(session,setup,monkeypatch):
    from kognita.autonomy import _change
    a,w,s,public,e,at,config=setup
    _change(session,a,1,w,'owner','AUTONOMY_PROMOTED',{})
    config(outcome_step_down=True)
    session.add(Interaction(id='stepdown',client_id='client',use_case='p'));session.flush()
    result=ProductionOutcome(interaction_id='stepdown',intended='good',actual='bad',client_impact='loss',
        adverse=True,source_reference='bank:actual',outcome_id='stepdown')
    record_outcome(session,a.name,result,evidence=w,actor_id='owner',now=at)
    assert a.autonomy_level==0
    assert any(r.payload.get('action')=='AUTONOMY_STEP_DOWN' for r in session.exec(select(EvidenceEvent)))
    verify_chain(session)


def test_expiry_and_removed_trust_key_deny_existing_records(session,setup):
    a,w,s,public,e,at,config=setup
    p=config(environment_required=True)
    claims={k:'runtime:ref' for k in ('network_policy','isolation','credential_scope','sandbox_identity','termination_path')}
    attest(session,a.name,packet(setup,p,'environment',{**claims,'enforced':True}),evidence=w,actor_id='runtime',now=at)
    state=load_state(session,[a.name],utcnow())
    late=at+timedelta(hours=2)
    assert any(c.check=='A7-ENVIRONMENT' and c.result==CheckResult.FAIL for c in admission_checks(e,a,state,late))
    configure(session,a.name,p.model_copy(update={'attestors':{}}),evidence=w,actor_id='owner')
    assert gates(session,setup)['A7-ENVIRONMENT']==CheckResult.FAIL


def test_advisory_keeps_quarantined_components_out_of_profiles(session,setup):
    a,w,s,public,e,at,config=setup
    config(components_required=True)
    approve_components(session,a.name,[dict(name='proxy',version='1',sbom_digest='f'*64,provenance='build')],
        evidence=w,actor_id='owner')
    before=conformance_report(session,a.name,'MAS')
    assert any(r['control']=='A9' and r['status']=='evidenced' for r in before['expectations'])
    advisory(session,'proxy','1','security:report',evidence=w,actor_id='security')
    after=conformance_report(session,a.name,'MAS')
    assert all(r['status']=='gap' for r in after['expectations'] if r['control']=='A9')


def test_drill_refuses_missing_decision_evidence(session,evidence,monkeypatch):
    from test_authority_s3 import _setup
    from kognita.assurance.drills import simulate
    from kognita.evidence import EvidenceWriter
    from conftest import NOW
    _,_,_,env=_setup(session,evidence)
    original=EvidenceWriter.emit
    def drop(self,session,**kw):
        if kw['event_type']!=EventType.POLICY_DECISION:return original(self,session,**kw)
    monkeypatch.setattr(EvidenceWriter,'emit',drop)
    with pytest.raises((ValueError,AttributeError)):
        simulate(session,'worker','machine-speed-probing',env,evidence=evidence,actor_id='owner',now=NOW)


def test_assurance_survives_signed_backup_restore(tmp_path):
    from kognita.db import make_engine, create_all
    from kognita.backup import _create, _restore
    from sqlmodel import Session
    from kognita.signing import _generate
    private,public=_generate()
    engine=make_engine(str(tmp_path/'source.db'));create_all(engine)
    # Use the actual signed writer for all identity and assurance events.
    writer=EvidenceWriter(engine,signing=EvidenceSigner(private))
    from kognita.registry import register as register_agent
    with Session(engine) as session:
        agent=register_agent(session,name='signed-worker',owner_exec='owner',evidence=writer,actor_id='owner',
            use_cases=['p'])
        configure(session,agent.name,AssurancePolicy(use_case='p',audience='bank'),evidence=writer,actor_id='owner')
        session.commit()
        count=verify_chain(session,public_key=public)
        bundle=_create(session)
    destination=tmp_path/'restored.db'
    _restore(bundle,str(destination),_public_key=public)
    restored=make_engine(str(destination))
    with Session(restored) as session:
        assert verify_chain(session,public_key=public)==count
        assert events(session,'signed-worker')[-1].payload['action']=='POLICY'
    engine.dispose();restored.dispose()


@pytest.mark.parametrize('path',['gateway','mcp'])
def test_ingress_accepts_valid_signed_external_proof_and_rejects_tampering(session,evidence,engine,path):
    import json
    from test_identity_radius_completion import _setup, _invoke
    from test_gateway import MODEL
    from conftest import NOW
    from kognita.vocabulary import Outcome
    agent,headers=_setup(session,evidence,engine,path)
    private,public=_generate();signer=EvidenceSigner(private)
    configure(session,agent.name,AssurancePolicy(use_case='COLLABORATION',audience='bank',
        external_required=True,issuers={'partner':public.decode()}),evidence=evidence,actor_id=agent.owner_exec)
    denied,calls=_invoke(engine,evidence,session,path,headers)
    assert denied.evaluation.outcome==Outcome.DENY and not calls
    doc=sign_record(Assertion(issuer='partner',subject=agent.name,principal='alice',purpose='COLLABORATION',
        audience='bank',scopes={'tools':['model_call' if path=='gateway' else 'read_note'],
        'client':['123'],'model':[MODEL]},issued_at=NOW-timedelta(minutes=1),expires_at=NOW+timedelta(hours=1)),signer)
    headers['x-kognita-agent-assertion']=json.dumps(doc)
    response,calls=_invoke(engine,evidence,session,path,headers)
    assert response.status==200 and calls, response.evaluation.failures()
    assert not any('x-kognita' in str(c) for c in calls)
    doc['document']['body']['principal']='other'
    headers['x-kognita-agent-assertion']=json.dumps(doc)
    response,calls=_invoke(engine,evidence,session,path,headers)
    assert response.evaluation.outcome==Outcome.DENY and not calls


def test_external_assertion_covers_argument_derived_subjects(session,setup):
    a,w,s,public,e,at,config=setup
    p=config()
    e=replace(e,context={'_effective_subjects':{'client':['two']}})
    with pytest.raises(ValueError,match='expansion'):
        verify_assertion(assertion(setup),p,e,at)


def test_usecase_activation_refuses_missing_privacy_review(session,evidence):
    from test_authority_s3 import _setup
    from kognita.usecases import load_register, activate
    from conftest import NOW
    _setup(session,evidence)
    configure(session,'worker',AssurancePolicy(use_case='READ',audience='bank',privacy_required=True),
        evidence=evidence,actor_id='owner')
    entry=load_register(session,as_of=NOW).get('READ')
    with pytest.raises(ValueError,match='A6-PRIVACY'):
        activate(session,'READ',status='live',evidence=evidence,actor_id=entry.control_function,at=NOW)


def test_profile_does_not_revive_superseded_attestation(session,setup):
    a,w,s,public,e,at,config=setup
    p=config(environment_required=True)
    claims={k:'runtime:ref' for k in ('network_policy','isolation','credential_scope','sandbox_identity','termination_path')}
    long=packet(setup,p,'environment',{**claims,'enforced':True})
    attest(session,a.name,long,evidence=w,actor_id='runtime',now=at)
    short=sign_record(Attestation(kind='environment',issuer='runtime',subject=a.name,
        issued_at=at-timedelta(seconds=1),expires_at=at+timedelta(seconds=1),**binding(a,p),
        claims={**claims,'enforced':True},reference='new short approval'),s)
    attest(session,a.name,short,evidence=w,actor_id='runtime',now=at)
    report=conformance_report(session,a.name,'MAS',as_of=at+timedelta(seconds=2))
    assert all(r['status']=='gap' for r in report['expectations'] if r['control']=='A7')
