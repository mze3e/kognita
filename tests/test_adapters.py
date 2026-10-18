"""Framework adapters, offline.

The frameworks are not installed here. These tests call the adapter methods
that speak the hook and the gateway, with a fake provider transport. A denial
cites the rule that decided it and does not call the tool or the provider.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from kognita.adapters.binding import Binding, GovernedRefusal
from kognita.adapters.hermes import HermesAdapter
from kognita.adapters.langgraph import LangGraphAdapter, _content, _role
from kognita.adapters.openai_agents import OpenAIAgentsAdapter, _messages
from kognita.adapters.pydantic_ai import PydanticAIAdapter
from kognita.adapters.pydantic_ai import _messages as _pydantic_messages
from kognita.exceptions import ConfigError
from kognita.gateway import ClientConfiguration, Gateway
from kognita.tools import ToolRegistry
from kognita.vocabulary import Outcome

_ADAPTERS = (OpenAIAgentsAdapter, LangGraphAdapter, PydanticAIAdapter)


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


def _binding(
    seeded,
    engine,
    evidence,
    pack,
    transport,
    *,
    agent="dossier-agent",
    site="SG",
    subject=None,
):
    purpose = "COLLABORATION"
    client = ClientConfiguration(
        principal="alice",
        purpose=purpose,
        agent_name=agent,
        actor_location=site,
    )
    gateway = Gateway(
        engine=engine,
        evidence=evidence,
        upstream="http://127.0.0.1:9",
        client=client,
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
        actor_location=site,
        subject_type="subject" if subject is not None else None,
        subject_id=subject,
    )


def _notes():
    ran: list[str] = []

    def read_notes(q: str) -> str:
        ran.append(q)
        return "notes"

    return read_notes, ran


@pytest.mark.parametrize("adapter_cls", _ADAPTERS, ids=lambda cls: cls.__name__)
def test_an_unregistered_agent_denies_the_tool_and_cites_the_registry(
    seeded, engine, evidence, pack, adapter_cls
):
    transport, calls = _transport()
    binding = _binding(seeded, engine, evidence, pack, transport, agent="not-an-agent")
    read_notes, ran = _notes()
    with pytest.raises(GovernedRefusal) as exc:
        adapter_cls(binding).tool(read_notes)(q="x")
    assert ran == []
    assert calls == []
    assert exc.value.outcome is Outcome.DENY
    assert any("not registered" in citation for citation in exc.value.citations)


@pytest.mark.parametrize("adapter_cls", _ADAPTERS, ids=lambda cls: cls.__name__)
def test_an_allowed_tool_runs_once_and_the_decision_cites_a_rule(
    seeded, engine, evidence, pack, adapter_cls
):
    transport, _calls = _transport()
    binding = _binding(seeded, engine, evidence, pack, transport)
    read_notes, ran = _notes()
    assert adapter_cls(binding).tool(read_notes)(q="x") == "notes"
    assert ran == ["x"]
    direct, direct_ran = _notes()
    result = binding.call_tool("read_notes", {"q": "y"}, direct)
    assert result.released
    assert direct_ran == ["y"]
    assert result.citations


@pytest.mark.parametrize("adapter_cls", _ADAPTERS, ids=lambda cls: cls.__name__)
def test_an_escalation_does_not_run_the_tool(
    seeded, engine, evidence, pack, adapter_cls
):
    transport, _calls = _transport()
    binding = _binding(seeded, engine, evidence, pack, transport, site="XX")
    read_notes, ran = _notes()
    with pytest.raises(GovernedRefusal) as exc:
        adapter_cls(binding).tool(read_notes)(q="x")
    assert ran == []
    assert exc.value.outcome is Outcome.ESCALATE
    assert any("Data Sharing Charter" in citation for citation in exc.value.citations)


@pytest.mark.parametrize("adapter_cls", _ADAPTERS, ids=lambda cls: cls.__name__)
def test_human_approval_does_not_run_the_tool(
    seeded, engine, evidence, pack, adapter_cls
):
    transport, _calls = _transport()
    binding = _binding(seeded, engine, evidence, pack, transport, subject="1")

    def draft_publication(q: str) -> str:
        raise AssertionError("draft_publication ran")

    with pytest.raises(GovernedRefusal) as exc:
        adapter_cls(binding).tool(draft_publication)(q="x")
    assert exc.value.outcome is Outcome.HUMAN_APPROVAL
    assert any(
        "Ethics Board Standing Order 22" in citation for citation in exc.value.citations
    )


@pytest.mark.parametrize("adapter_cls", _ADAPTERS, ids=lambda cls: cls.__name__)
def test_a_denied_model_call_does_not_reach_the_provider(
    seeded, engine, evidence, pack, adapter_cls
):
    transport, calls = _transport()
    binding = _binding(seeded, engine, evidence, pack, transport, agent="not-an-agent")
    with pytest.raises(GovernedRefusal) as exc:
        adapter_cls(binding).complete(
            model="governed", messages=[{"role": "user", "content": "hello"}]
        )
    assert calls == []
    assert any("not registered" in citation for citation in exc.value.citations)


@pytest.mark.parametrize("adapter_cls", _ADAPTERS, ids=lambda cls: cls.__name__)
def test_an_allowed_model_call_returns_the_provider_text(
    seeded, engine, evidence, pack, adapter_cls
):
    transport, calls = _transport()
    binding = _binding(seeded, engine, evidence, pack, transport)
    result = adapter_cls(binding).complete(
        model="governed", messages=[{"role": "user", "content": "hello"}]
    )
    assert result.text == "ok"
    assert len(calls) == 1
    assert result.citations


def test_hermes_blocks_a_denied_tool_with_the_citation(seeded, engine, evidence, pack):
    transport, calls = _transport()
    binding = _binding(seeded, engine, evidence, pack, transport, agent="not-an-agent")
    ran: list[str] = []

    def terminal(command: str) -> str:
        ran.append(command)
        return "ran"

    adapter = HermesAdapter(binding)
    hooks: dict[str, object] = {}

    class Ctx:
        def register_hook(self, name, fn):
            hooks[name] = fn

    adapter.register(Ctx())
    directive = hooks["pre_tool_call"]("terminal", {"command": "rm -rf /"})
    if directive is None:
        terminal(command="rm -rf /")
    assert ran == []
    assert calls == []
    assert directive["action"] == "block"
    assert "not registered" in directive["message"]


def test_hermes_allows_a_tool_and_asks_for_approval_when_a_human_is_required(
    seeded, engine, evidence, pack
):
    transport, _calls = _transport()
    allowed = HermesAdapter(_binding(seeded, engine, evidence, pack, transport))
    assert allowed.pre_tool_call("get_subject_profile", {"q": "x"}) is None

    held = HermesAdapter(
        _binding(seeded, engine, evidence, pack, transport, subject="1")
    )
    directive = held.pre_tool_call("draft_publication", {"q": "x"})
    assert directive["action"] == "approve"
    assert "Ethics Board Standing Order 22" in directive["message"]


def test_hermes_model_client_fails_closed_and_cites_the_rule(
    seeded, engine, evidence, pack
):
    transport, calls = _transport()
    denied = HermesAdapter(
        _binding(seeded, engine, evidence, pack, transport, agent="not-an-agent")
    )
    with pytest.raises(GovernedRefusal) as exc:
        denied.client().chat.completions.create(
            model="governed", messages=[{"role": "user", "content": "hello"}]
        )
    assert calls == []
    assert any("not registered" in citation for citation in exc.value.citations)

    transport, calls = _transport()
    allowed = HermesAdapter(_binding(seeded, engine, evidence, pack, transport))
    body = allowed.create_client().chat.completions.create(
        model="governed", messages=[{"role": "user", "content": "hello"}]
    )
    assert body["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 1

    with pytest.raises(GovernedRefusal) as exc:
        allowed.client().chat.completions.create(
            model="governed",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    assert "streamed response" in exc.value.citations[0]
    assert len(calls) == 1


def test_hermes_blocks_arguments_that_are_not_a_mapping(seeded, engine, evidence, pack):
    transport, _calls = _transport()
    adapter = HermesAdapter(_binding(seeded, engine, evidence, pack, transport))
    directive = adapter.pre_tool_call("terminal", args="rm -rf /")  # type: ignore[arg-type]
    assert directive["action"] == "block"


def test_message_shapes_are_chat_completions_without_the_frameworks():
    assert _messages("be brief", "hello") == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]

    class HumanMessage:
        content = "hello"

    class AIMessage:
        def __init__(self) -> None:
            self.content = "earlier"

    assert _role(HumanMessage()) == "user"
    assert _role(AIMessage()) == "assistant"
    assert _content(HumanMessage()) == "hello"

    class UserPromptPart:
        content = "hello"

    class ModelRequest:
        parts = [UserPromptPart()]

    assert _pydantic_messages([ModelRequest()]) == [
        {"role": "user", "content": "hello"}
    ]


@pytest.mark.parametrize(
    ("adapter_cls", "method", "extra", "module"),
    [
        (OpenAIAgentsAdapter, "function_tool", "openai-agents", "agents"),
        (LangGraphAdapter, "structured_tool", "langgraph", "langchain_core"),
        (LangGraphAdapter, "chat_model", "langgraph", "langchain_core"),
        (PydanticAIAdapter, "agent_tool", "pydantic-ai", "pydantic_ai"),
        (PydanticAIAdapter, "sdk_model", "pydantic-ai", "pydantic_ai"),
        (OpenAIAgentsAdapter, "sdk_model", "openai-agents", "agents"),
    ],
)
def test_sdk_entry_names_the_extra_when_the_framework_is_absent(
    seeded, engine, evidence, pack, adapter_cls, method, extra, module
):
    if importlib.util.find_spec(module) is not None:
        pytest.skip(f"{extra} is installed")
    transport, _calls = _transport()
    adapter = adapter_cls(_binding(seeded, engine, evidence, pack, transport))
    with pytest.raises(ConfigError, match=extra):
        getattr(adapter, method)(lambda q: q)
