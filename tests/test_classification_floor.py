"""A caller-supplied ``classification`` header is a floor, never a ceiling.

Deliberately outside the pre-S3 fixture allowlist in ``conftest.py``: no
fabricated authority, and no auto-attached workload credential — this module
builds a real grant, a real delegation and a real issued credential, the way
``kognita.authority`` and ``kognita.credentials`` are actually exercised in
production. That is what lets these tests catch a classifier that a header
can skip: fabricated authority or an injected credential would reach
``decide`` regardless of what the gateway and the MCP proxy did with the
classification header, which is exactly what would hide this bug.
"""
from __future__ import annotations

from datetime import timedelta

import json

from conftest import NOW
from fixtures.agents import register as _register_agent
from kognita.authority import Delegation, delegate, grant_to
from kognita.credentials import issue
from kognita.tools import Run
from kognita.vocabulary import Outcome

from test_gateway import BENIGN, MODEL, PATH, SECRET, _body, _gateway, _live
from test_gateway import _Upstream as _GatewayUpstream
from test_mcp import _Upstream as _McpUpstream
from test_mcp import _call, _proxy
from test_usecases import ensure_live_purposes


def _authorize(session, evidence, *, client="123", subject_scope=None):
    """A real grant and delegation for ``dossier-agent``, plus its credential.

    ``subject_scope`` additionally restricts the grant/delegation to a named
    subject (e.g. the gateway's served model); omitted, only ``client`` is
    scoped, which covers calls that carry no subject at all.
    """
    scope = {"client": [client]}
    if subject_scope:
        scope.update(subject_scope)
    grant_to(
        "alice", "read", timedelta(hours=2), scope,
        session=session, evidence=evidence, actor_id="admin", now=NOW,
    )
    delegation_id = delegate(
        session,
        Delegation(
            human="alice", agent="dossier-agent", purpose="COLLABORATION",
            client=client, allowed_actions=["read"], allowed_data=scope,
            valid_from=NOW, valid_until=NOW + timedelta(hours=1),
            jurisdiction="SG", task_ref="classification-floor",
        ),
        evidence=evidence, actor_id="alice", now=NOW,
    )
    credential = issue(session, "dossier-agent", evidence=evidence, actor_id="Head of Research Ops")
    return delegation_id, credential


def test_gateway_header_floor_does_not_suppress_the_classifier(session, evidence):
    """``classification: C0`` cannot keep an emailed prompt below C2.

    The pattern classifier reads an email address as C2. A caller header
    claiming C0 must not skip that: the effective label stays C2 and the
    egress guard redacts the address before it reaches the upstream, exactly
    as it would with no header at all.
    """
    _register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    _live(session)
    did, cred = _authorize(session, evidence, subject_scope={"model": [MODEL]})
    session.commit()

    upstream = _GatewayUpstream(
        json.dumps({"model": MODEL, "choices": [{"message": {"content": "noted"}}]}).encode()
    )
    # Sensitive prompts require a run so this test reaches the egress guard.
    gateway = _gateway(session.get_bind(), evidence, upstream, run=Run())
    response = gateway.handle(
        "POST",
        PATH,
        {
            "agent_name": "dossier-agent",
            "agent_credential": cred,
            "delegation_id": did,
            "classification": "C0",
        },
        _body(f"Email {SECRET} about the notes"),
        session=session,
    )

    assert response.evaluation is not None
    assert response.evaluation.outcome is Outcome.ALLOW
    assert response.evaluation.attributes["classification"] == "C2"
    assert response.evaluation.attributes["classifier"]["label"] == "C2"
    assert response.status == 200
    assert len(upstream.calls) == 1
    assert SECRET.encode() not in upstream.calls[0]["body"]
    assert b"[EMAIL_1]" in upstream.calls[0]["body"]


def test_gateway_header_raises_a_benign_prompt(session, evidence):
    """``classification: C3`` on a benign prompt raises the effective label.

    No indicator in the prompt would classify it above the floor on its
    own. The header may still raise it — the ceiling then refuses to send
    C3 content to a remote provider, the same way an unheaded C3 model
    response would be refused.
    """
    _register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    _live(session)
    did, cred = _authorize(session, evidence, subject_scope={"model": [MODEL]})
    session.commit()

    upstream = _GatewayUpstream(
        json.dumps({"model": MODEL, "choices": [{"message": {"content": "noted"}}]}).encode()
    )
    # Sensitive prompts require a run so this test reaches the egress guard.
    gateway = _gateway(session.get_bind(), evidence, upstream, run=Run())
    response = gateway.handle(
        "POST",
        PATH,
        {
            "agent_name": "dossier-agent",
            "agent_credential": cred,
            "delegation_id": did,
            "classification": "C3",
        },
        _body(BENIGN),
        session=session,
    )

    assert response.evaluation is not None
    assert response.evaluation.attributes["classification"] == "C3"
    assert upstream.calls == []
    assert response.evaluation.outcome is Outcome.DENY
    assert any(
        check.check == "EGRESS" and "C3" in check.citation
        for check in response.evaluation.checks
    )


def test_mcp_header_floor_does_not_suppress_the_classifier(session, evidence):
    """The MCP proxy applies the same floor to ``tools/call`` arguments.

    A header claiming C0 on arguments the pattern classifier reads as C2
    (an email address) must not keep the recorded decision at C0.
    """
    _register_agent(session, name="dossier-agent", owner_exec="Head of Research Ops")
    ensure_live_purposes(session, evidence, ("COLLABORATION",), at=NOW)
    did, cred = _authorize(session, evidence)
    session.commit()

    upstream = _McpUpstream()
    proxy = _proxy(session.get_bind(), evidence, upstream)
    text = f"Email {SECRET} about the notes"
    response = proxy.handle(
        "POST",
        "/mcp",
        {
            "agent_name": "dossier-agent",
            "agent_credential": cred,
            "delegation_id": did,
            "classification": "C0",
        },
        _call(arguments={"query": text}),
        session=session,
    )

    assert response.evaluation is not None
    # Classified client data requires a run even when the header claims C0.
    assert response.evaluation.outcome is Outcome.DENY
    assert response.evaluation.attributes["classification"] == "C2"
    assert response.evaluation.attributes["classifier"]["label"] == "C2"
    assert "S5-RUN-REQUIRED" in {c.citation for c in response.evaluation.failures()}
    assert response.status == 403
    assert not upstream.calls


def test_typed_floor_holds_for_a_classifier_that_ignores_hints():
    from kognita.envelope import Envelope
    from kognita.governance import classifier_derived_envelope

    class _IgnoresHint:
        def classify(self, text, hint=None):
            return "C0"

    _, attrs = classifier_derived_envelope(
        "plain text", Envelope(principal="alice", purpose="COLLABORATION", tool="read_note"),
        attributes={"classification": "C3"}, classifier=_IgnoresHint())
    assert attrs["classification"] == "C3"
    assert attrs["classifier"]["label"] == "C3"


def test_mcp_header_cannot_lower_the_pack_label():
    from kognita.envelope import Envelope
    from kognita.mcp import _AttributePack

    class _Pack:
        def load_subjects(self, envelope, session):
            return {}

        def resolve_attributes(self, envelope, subjects):
            return {"classification": "C3"}

    envelope = Envelope(principal="alice", purpose="COLLABORATION", tool="read_note")
    assert _AttributePack(_Pack(), "C0").resolve_attributes(envelope, {})["classification"] == "C3"
    assert _AttributePack(_Pack(), None).resolve_attributes(envelope, {})["classification"] == "C3"
