"""Real adapter authority paths, without legacy credential injection."""
import asyncio
from pathlib import Path

import pytest

from kognita.adapters.binding import Binding, GovernedRefusal
from kognita.adapters.hermes import HermesAdapter
from kognita.adapters.langgraph import LangGraphAdapter
from kognita.adapters.openai_agents import OpenAIAgentsAdapter
from kognita.adapters.pydantic_ai import PydanticAIAdapter
from kognita.tools import ToolRegistry
from kognita.vocabulary import Outcome

from test_gateway import MODEL, _Upstream, _gateway, _live
from test_mcp_client_argument_scope import _delegated
from test_mcp import _Pack, _proxy, _call, _Upstream as _McpUpstream


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Override the global credential shim; never patch a production method."""


@pytest.mark.parametrize("adapter_cls", [
    HermesAdapter, OpenAIAgentsAdapter, LangGraphAdapter, PydanticAIAdapter,
])
@pytest.mark.parametrize("credential_present", [True, False])
@pytest.mark.parametrize("entry", ["core", "sdk"])
def test_adapter_model_and_tool_authority(session, evidence, engine, adapter_cls,
                                          credential_present, monkeypatch, entry):
    if entry == "sdk":
        pytest.importorskip({
            HermesAdapter: "hermes_cli.plugins",
            OpenAIAgentsAdapter: "agents",
            LangGraphAdapter: "langgraph",
            PydanticAIAdapter: "pydantic_ai",
        }[adapter_cls])
    from conftest import NOW
    monkeypatch.setattr("kognita.tools.utcnow", lambda: NOW)
    monkeypatch.setattr("kognita.governance.utcnow", lambda: NOW)
    _live(session)
    did, secret = _delegated(session, evidence)
    # Include model authority as well as the client-scoped tool authority.
    from datetime import timedelta
    from conftest import NOW
    from kognita.authority import Delegation, delegate, grant_to
    scope = {"client": ["123"], "model": [MODEL]}
    grant_to("alice", "read", timedelta(hours=2), scope, session=session,
             evidence=evidence, actor_id="admin", now=NOW)
    did = delegate(session, Delegation(
        human="alice", agent="dossier-agent", purpose="COLLABORATION",
        client="123", allowed_actions=["read"], allowed_data=scope,
        valid_from=NOW, valid_until=NOW + timedelta(hours=1),
        jurisdiction="SG", task_ref="adapter",
    ), evidence=evidence, actor_id="alice", now=NOW)
    backend = _Upstream(b'{"model":"gateway-model","choices":[{"message":{"content":"ok"}}]}')
    binding = Binding(
        session=session, evidence=evidence, pack=_Pack(),
        gateway=_gateway(engine, evidence, backend), registry=ToolRegistry(),
        principal="alice", purpose="COLLABORATION", agent_name="dossier-agent",
        purposes=("COLLABORATION",), actor_location="SG",
        subject_type="client", subject_id="123",
        agent_credential=secret if credential_present else None, delegation_id=did,
    )
    assert secret not in repr(binding)
    adapter = adapter_cls(binding)
    mcp_backend = _McpUpstream()
    proxy = _proxy(engine, evidence, mcp_backend)

    def read_note():
        response = proxy.handle("POST", "/mcp", {
            "agent_name": binding.agent_name,
            "agent_credential": binding.agent_credential or "",
            "delegation_id": binding.delegation_id,
        }, _call(arguments={"client_id": "123"}), session=session)
        assert response.evaluation.outcome is Outcome.ALLOW
        return "notes"

    if adapter_cls is HermesAdapter:
        model_call = lambda: adapter.client().chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hello"}])
        tool_call = binding.wrap(read_note)
    else:
        model_call = lambda: adapter.complete(
            model=MODEL, messages=[{"role": "user", "content": "hello"}])
        tool_call = adapter.tool(read_note)
    if entry == "sdk":
        if adapter_cls is OpenAIAgentsAdapter:
            from agents.tool_context import ToolContext
            sdk_tool = adapter.function_tool(read_note)
            ctx = ToolContext(context=None, tool_name=sdk_tool.name,
                              tool_call_id="call-1", tool_arguments="{}")
            tool_call = lambda: asyncio.run(sdk_tool.on_invoke_tool(ctx, "{}"))
            model_call = lambda: asyncio.run(adapter.sdk_model(MODEL).get_response(
                None, "hello", None, [], None, [], None,
                previous_response_id=None, conversation_id=None, prompt=None))
        elif adapter_cls is LangGraphAdapter:
            from langchain_core.messages import HumanMessage
            tool_call = lambda: adapter.structured_tool(read_note).invoke({})
            model_call = lambda: adapter.chat_model(MODEL).invoke(
                [HumanMessage(content="hello")])
        elif adapter_cls is PydanticAIAdapter:
            from pydantic_ai.messages import ModelRequest
            from pydantic_ai.models import ModelRequestParameters
            tool_call = lambda: adapter.agent_tool(read_note).function()
            model_call = lambda: asyncio.run(adapter.sdk_model(MODEL).request(
                [ModelRequest.user_text_prompt("hello")], None, ModelRequestParameters()))
    if credential_present:
        model_call()
        assert tool_call() == "notes"
        if adapter_cls is HermesAdapter:
            assert adapter.pre_tool_call("read_note", {}) is None
        assert len(backend.calls) == len(mcp_backend.calls) == 1
        assert secret not in repr(backend.calls)
        from sqlmodel import select
        from kognita.models import EvidenceEvent
        assert secret not in repr(session.exec(select(EvidenceEvent)).all())
    else:
        for call in (model_call, tool_call):
            with pytest.raises(GovernedRefusal) as exc:
                call()
            assert exc.value.outcome is Outcome.DENY
            assert any("S2-CREDENTIAL" in c for c in exc.value.citations)
        assert backend.calls == mcp_backend.calls == []


def test_binding_stays_within_adapter_line_limit():
    import kognita.adapters.binding as module
    assert len(Path(module.__file__).read_text().splitlines()) <= 200
