"""Explicitly approved L0 identities for tests of pre-existing policy controls.

S1 identity tests use the production register directly and cover legacy denial.
These fixtures give the older policy tests their declared read capabilities.
"""
from datetime import datetime, timezone
from functools import lru_cache as _lru_cache

from kognita.evidence import EvidenceWriter
from kognita.registry import register as _register, approve_deployment

_credentials = {}

_DEPLOYMENT = dict(system_prompt="", instructions="", tools=[], permissions=[],
                   data_sources=[], orchestration="", memory_behaviour="")


def register(session, **kwargs):
    principal = kwargs.pop("_principal", "alice")
    agent = _register(session, **kwargs)
    agent.principal = principal
    return configure(agent, session)


def configure(agent, session=None):
    agent.field_allowlist = {purpose: ["$", "marked", "content.text", "content.type",
        "tools.name", "tools.description", "tools.inputSchema", "protocolVersion",
        "serverInfo.name", "serverInfo.version", "capabilities.tools"] for purpose in PURPOSES}
    agent.environment = "test"
    agent.principal = agent.principal or "alice"
    agent.purpose = "policy regression fixture"
    agent.use_cases = PURPOSES
    agent.permitted_systems = ["fixture"]
    agent.tool_allowlist = {name: {"verb": "read", "required_level": 0,
        "permissions": ["read"], "system": "fixture"} for name in TOOLS}
    agent.review_date = datetime(2099, 1, 1, tzinfo=timezone.utc)
    if session is None:
        from kognita.registry import _deployment_hash
        agent.tool_allowlist["_deployment"] = _DEPLOYMENT
        agent.deployment_hash = _deployment_hash(agent, _DEPLOYMENT)
        return agent
    approve_deployment(session, agent.name, deployment=_DEPLOYMENT,
        evidence=EvidenceWriter(session.get_bind()), actor_id=agent.owner_exec)
    from kognita.credentials import issue
    engine = session.get_bind()
    if not hasattr(engine, "_test_workload_credentials"):
        engine._test_workload_credentials = {}
    engine._test_workload_credentials[agent.name] = issue(session, agent.name,
        evidence=EvidenceWriter(engine), actor_id=agent.owner_exec)
    _credentials[str(engine.url) if engine.url.database else engine] = engine._test_workload_credentials
    return agent

TOOLS = ['crm/read_note',
 'discuss_product',
 'draft_publication',
 'get_subject_profile',
 'initialize',
 'mark_criterion',
 'model_call',
 'notes/read_note',
 'propose_criterion',
 'read_email',
 'read_note',
 'read_notes',
 'read_record',
 'send',
 'send_note',
 'tools/list',
 'resources/read']
PURPOSES = ['A Singapore briefing',
 'A live use case with no primary metric',
 'ACCESS',
 'CLIENT_DISCUSSION',
 'CLIENT_PRECALL',
 'COLLABORATION',
 'CONFIRM_CRITERION',
 'Check suitability before a recommendation',
 'DOSSIER_PREP',
 'Draft a note',
 'Draft internal meeting notes',
 'ELIGIBILITY_CHECK',
 'MARK_CRITERION',
 'PROPOSE_CHANGE',
 'PUBLICATION',
 'Pilot the summary with a named user list',
 'Prepare a briefing',
 'Prepare a briefing from the summary',
 'Prepare a client briefing',
 'Prepare a short internal briefing',
 'READ',
 'REVIEW',
 'RM_PRECALL',
 'Recommend a product the RM has approved',
 'Summarise approved research for an RM',
 'briefing',
 'notes',
 'p',
 'pilot-summary',
 'research-summary']


def _authority_for(envelope, snapshot):
    """Explicit authority prerequisite for policy regression tests predating S3.

    These tests isolate other controls. S3 tests use the production lifecycle
    and do not call this helper. No authority is added for unregistered agents.
    """
    from dataclasses import replace
    if not envelope.agent_name or envelope.agent_name not in snapshot.agents:
        return envelope, snapshot
    identity = f"fixture:{envelope.principal}:{envelope.agent_name}:{envelope.purpose}"
    scope = {k: [v] for k, v in envelope.all_subjects().items()}
    client = scope.get("client", ["fixture"])[0]
    scope.setdefault("client", [client])
    start, end = "2000-01-01T00:00:00+00:00", "2099-01-01T00:00:00+00:00"
    verbs = ["read", "analyse", "recommend", "draft", "queue", "send", "execute"]
    body = dict(human=envelope.principal, agent=envelope.agent_name,
        purpose=envelope.purpose, client=client, allowed_actions=verbs,
        allowed_data=scope, valid_from=start, valid_until=end, limits={},
        jurisdiction=envelope.actor_location or "fixture", approval_threshold="execute", task_ref=identity)
    # Location is tested by the old policy; grant precisely the request's value.
    body["jurisdiction"] = envelope.actor_location
    rows = [(identity, "delegation", envelope.agent_name, body)]
    rows += [(identity + verb, "grant", envelope.principal, dict(capability=verb,
        subject_scope=scope, valid_from=start, valid_until=end)) for verb in verbs]
    return replace(envelope, delegation_id=identity), replace(snapshot, _authority=tuple(rows))


def _usecases(purposes):
    return _cached_usecases(tuple(purposes))


@_lru_cache(maxsize=128)
def _cached_usecases(purposes):
    """Explicit live register for older pure policy tests, independent of authority."""
    from kognita.db import make_engine, create_all
    from kognita.usecases import load_register
    from sqlmodel import Session
    from test_usecases import ensure_live_purposes, NOW
    engine = make_engine()
    create_all(engine)
    with Session(engine) as session:
        ensure_live_purposes(session, EvidenceWriter(engine, clock=lambda: NOW), purposes)
        register = load_register(session, as_of=NOW)
    engine.dispose()
    return register
