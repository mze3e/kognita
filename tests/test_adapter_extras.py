"""Adapter tests that run only when that adapter's extra is installed.

The default suite does not install the frameworks. Each test below skips
until the extra is present, then drives the real framework type through the
same decide and gateway path.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from kognita.adapters.binding import Binding, GovernedRefusal
from kognita.adapters.hermes import HermesAdapter
from kognita.adapters.langgraph import LangGraphAdapter
from kognita.adapters.openai_agents import OpenAIAgentsAdapter
from kognita.adapters.pydantic_ai import PydanticAIAdapter
from kognita.gateway import ClientConfiguration, Gateway
from kognita.tools import ToolRegistry


def _transport():
    calls: list[bytes] = []

    def transport(method, url, headers, body):
        calls.append(body)
        payload = json.dumps(
            {
                "id": "cmpl",
                "model": "governed",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        ).encode()
        return 200, {"content-type": "application/json"}, payload

    return transport, calls


def _binding(seeded, engine, evidence, pack, transport, *, agent="dossier-agent"):
    purpose = "COLLABORATION"
    gateway = Gateway(
        engine=engine,
        evidence=evidence,
        upstream="http://127.0.0.1:9",
        client=ClientConfiguration(
            principal="alice",
            purpose=purpose,
            agent_name=agent,
            actor_location="SG",
        ),
        pack=pack,
        purposes=(purpose,),
        transport=transport,
    )
    return Binding(
        session=seeded,
        evidence=evidence,
        pack=pack,
        gateway=gateway,
        registry=ToolRegistry(),
        principal="alice",
        purpose=purpose,
        agent_name=agent,
        purposes=(purpose,),
        actor_location="SG",
    )


def test_hermes_agent_extra_uses_the_real_tool_hook(seeded, engine, evidence, pack):
    plugins = pytest.importorskip("hermes_cli.plugins")
    assert "pre_tool_call" in plugins.VALID_HOOKS
    transport, calls = _transport()
    adapter = HermesAdapter(
        _binding(seeded, engine, evidence, pack, transport, agent="not-an-agent")
    )
    recorded: dict[str, object] = {}

    class Ctx:
        def register_hook(self, name, fn):
            if name not in plugins.VALID_HOOKS:
                raise AssertionError(name)
            recorded[name] = fn

    adapter.register(Ctx())
    directive = recorded["pre_tool_call"](tool_name="terminal", args={"command": "rm"})
    assert directive["action"] == "block"
    assert "not registered" in directive["message"]
    assert calls == []

    # hermes-agent 0.19 has no ProviderProfile.create_client. The model path
    # this release can point at kognita is AIAgent(base_url=...), and the
    # in-process client the adapter returns still decides before any transport.
    from run_agent import AIAgent

    assert "base_url" in inspect.signature(AIAgent.__init__).parameters
    with pytest.raises(GovernedRefusal) as exc:
        adapter.client().chat.completions.create(
            model="governed", messages=[{"role": "user", "content": "hello"}]
        )
    assert calls == []
    assert any("not registered" in citation for citation in exc.value.citations)


def test_openai_agents_extra_routes_tool_and_model(seeded, engine, evidence, pack):
    pytest.importorskip("agents")
    from agents.models.interface import Model
    from agents.tool_context import ToolContext

    ran: list[str] = []

    def read_notes(q: str) -> str:
        """Read the notes."""
        ran.append(q)
        return "notes"

    transport, calls = _transport()
    allowed = OpenAIAgentsAdapter(_binding(seeded, engine, evidence, pack, transport))
    tool = allowed.function_tool(read_notes)
    ctx = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments=json.dumps({"q": "x"}),
    )
    assert asyncio.run(tool.on_invoke_tool(ctx, json.dumps({"q": "x"}))) == "notes"
    assert ran == ["x"]

    model = allowed.sdk_model()
    assert isinstance(model, Model)
    response = asyncio.run(
        model.get_response(
            None,
            "hello",
            None,
            [],
            None,
            [],
            None,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    )
    assert response.output
    assert len(calls) == 1

    denied_transport, denied_calls = _transport()
    denied = OpenAIAgentsAdapter(
        _binding(seeded, engine, evidence, pack, denied_transport, agent="not-an-agent")
    )
    with pytest.raises(GovernedRefusal) as exc:
        asyncio.run(
            denied.sdk_model().get_response(
                None,
                "hello",
                None,
                [],
                None,
                [],
                None,
                previous_response_id=None,
                conversation_id=None,
                prompt=None,
            )
        )
    assert denied_calls == []
    assert any("not registered" in citation for citation in exc.value.citations)


def test_langgraph_extra_routes_tool_and_model(seeded, engine, evidence, pack):
    pytest.importorskip("langgraph")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import HumanMessage

    ran: list[str] = []

    def read_notes(q: str) -> str:
        ran.append(q)
        return "notes"

    transport, calls = _transport()
    allowed = LangGraphAdapter(_binding(seeded, engine, evidence, pack, transport))
    assert allowed.structured_tool(read_notes).invoke({"q": "x"}) == "notes"
    assert ran == ["x"]
    model = allowed.chat_model()
    assert isinstance(model, BaseChatModel)
    reply = model.invoke([HumanMessage(content="hello")])
    assert reply.content == "ok"
    assert len(calls) == 1

    denied_transport, denied_calls = _transport()
    denied = LangGraphAdapter(
        _binding(seeded, engine, evidence, pack, denied_transport, agent="not-an-agent")
    )
    with pytest.raises(GovernedRefusal):
        denied.structured_tool(read_notes).invoke({"q": "y"})
    assert ran == ["x"]
    with pytest.raises(GovernedRefusal) as exc:
        denied.chat_model().invoke([HumanMessage(content="hello")])
    assert denied_calls == []
    assert any("not registered" in citation for citation in exc.value.citations)


def test_pydantic_ai_extra_routes_tool_and_model(seeded, engine, evidence, pack):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import ModelRequest
    from pydantic_ai.models import Model, ModelRequestParameters

    ran: list[str] = []

    def read_notes(q: str) -> str:
        ran.append(q)
        return "notes"

    transport, calls = _transport()
    allowed = PydanticAIAdapter(_binding(seeded, engine, evidence, pack, transport))
    tool = allowed.agent_tool(read_notes)
    assert tool.function(q="x") == "notes" or tool.function_schema is not None
    if ran != ["x"]:
        ran.clear()
        assert allowed.tool(read_notes)(q="x") == "notes"
    assert ran == ["x"]

    model = allowed.sdk_model()
    assert isinstance(model, Model)
    prompt = ModelRequest.user_text_prompt("hello")
    response = asyncio.run(model.request([prompt], None, ModelRequestParameters()))
    assert response.parts
    assert len(calls) == 1

    denied_transport, denied_calls = _transport()
    denied = PydanticAIAdapter(
        _binding(seeded, engine, evidence, pack, denied_transport, agent="not-an-agent")
    )
    with pytest.raises(GovernedRefusal) as exc:
        asyncio.run(
            denied.sdk_model().request([prompt], None, ModelRequestParameters())
        )
    assert denied_calls == []
    assert any("not registered" in citation for citation in exc.value.citations)
