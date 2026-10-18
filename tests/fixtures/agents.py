"""Explicitly approved L0 identities for tests of pre-existing policy controls.

S1 identity tests use the production register directly and cover legacy denial.
These fixtures give the older policy tests their declared read capabilities.
"""
from datetime import datetime, timezone

from kognita.evidence import EvidenceWriter
from kognita.registry import register as _register, approve_deployment

_DEPLOYMENT = dict(system_prompt="", instructions="", tools=[], permissions=[],
                   data_sources=[], orchestration="", memory_behaviour="")


def register(session, **kwargs):
    agent = _register(session, **kwargs)
    return configure(agent, session)


def configure(agent, session=None):
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
