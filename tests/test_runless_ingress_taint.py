"""Runless ingress uses real authority and no legacy ingress wrappers."""
from dataclasses import replace
import pytest

from conftest import NOW
from kognita.broker import ask
from kognita.vocabulary import Classification, Outcome
from kognita.retrieval import index_item
from kognita.embedding import HashingEmbedder
from test_authority_s3 import _setup
from test_mcp import _Pack


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Supply credentials explicitly; leave production handlers intact."""


@pytest.mark.parametrize('classification', [Classification.C1, Classification.C2])
def test_runless_broker_denies_before_external_or_sensitive_release(session, evidence, classification):
    _, _, _, env = _setup(session, evidence)
    embedder = HashingEmbedder()
    body = 'meeting notes' if classification is Classification.C1 else 'meeting notes alice@example.com'
    index_item(session, title='meeting notes', body=body,
               embedder=embedder, kind='note', classification=classification,
               zones=['SG'], source_label='external-source')
    answer = ask(session, 'meeting notes', replace(env, subject_type=None, subject_id=None),
                 pack=_Pack(), embedder=embedder, evidence=evidence, purposes=['READ'],
                 run=None, now=NOW, as_of=NOW)
    assert answer.outcome is Outcome.DENY
    assert not answer.results
    assert 'S5-RUN-REQUIRED' in {c.citation for c in answer.evaluation.failures()}


def test_runless_public_model_denies_sensitive_prompt(session, evidence):
    import json
    from fixtures.agents import register
    from kognita.registry import approve_deployment
    from test_classification_floor import _authorize
    from test_gateway import _live, _gateway, _Upstream, _body, PATH, MODEL

    agent = register(session, name='dossier-agent', owner_exec='Head of Research Ops')
    permissions = {k: dict(v) if isinstance(v, dict) else v for k, v in agent.tool_allowlist.items()}
    permissions['model_call']['public_content'] = True
    agent.tool_allowlist = permissions
    approve_deployment(session, agent.name, deployment=permissions['_deployment'],
                       evidence=evidence, actor_id=agent.owner_exec)
    _live(session)
    did, credential = _authorize(session, evidence, subject_scope={'model': [MODEL]})
    session.commit()
    upstream = _Upstream(json.dumps({'model': MODEL, 'choices': [{'message': {'content': 'noted'}}]}).encode())
    response = _gateway(session.get_bind(), evidence, upstream).handle('POST', PATH,
        dict(agent_name=agent.name, agent_credential=credential, delegation_id=did),
        _body('Email alice@example.com about the notes'), session=session)
    assert response.evaluation.outcome is Outcome.DENY
    assert upstream.calls == []
    assert 'S5-RUN-REQUIRED' in {c.citation for c in response.evaluation.failures()}
