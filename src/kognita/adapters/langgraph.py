"""LangGraph adapter.

A graph's model node and its tools are the two places a turn can leave the
process. ``chat_model`` is a ``BaseChatModel`` whose ``_generate`` is
:meth:`kognita.gateway.Gateway.handle`. ``tool`` wraps a callable so the tool
node is :func:`kognita.tools.run_governed`. A refusal raises
:class:`~kognita.adapters.binding.GovernedRefusal` with the citations, and
neither the provider nor the tool body runs. ``_stream`` stays the base
implementation, which refuses to stream.

Installs with the ``langgraph`` extra. Importing this module does not import
LangGraph or LangChain.
"""

from __future__ import annotations

from typing import Any

from kognita.adapters.binding import Binding, GovernedRefusal, GovernedResult
from kognita.exceptions import ConfigError
from kognita.vocabulary import Outcome

_EXTRA = "langgraph"


def _role(message: Any) -> str:
    name = type(message).__name__
    if name == "SystemMessage":
        return "system"
    if name in {"AIMessage", "AIMessageChunk"}:
        return "assistant"
    return "user"


def _content(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


class LangGraphAdapter:
    """Governed tools and a chat model for a LangGraph agent."""

    def __init__(self, binding: Binding) -> None:
        self.binding = binding

    def tool(self, fn: Any) -> Any:
        """Wrap ``fn``. The original runs only after an ``ALLOW``."""
        return self.binding.wrap(fn)

    def complete(self, *, model: str, messages: list[dict[str, Any]]) -> Any:
        """One chat completion through the gateway. Refusals raise."""
        return self.binding.require_completion(model=model, messages=messages)

    def structured_tool(self, fn: Any) -> Any:
        """A LangChain ``StructuredTool`` whose body is :meth:`tool`."""
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as exc:
            raise ConfigError(
                f"install kognita[{_EXTRA}] to wrap a LangGraph tool"
            ) from exc
        description = getattr(fn, "__doc__", None) or getattr(fn, "__name__", "tool")
        return StructuredTool.from_function(self.tool(fn), description=str(description))

    def chat_model(self, model: str = "governed") -> Any:
        """A ``BaseChatModel`` that completes through the gateway."""
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult
        except ImportError as exc:
            raise ConfigError(
                f"install kognita[{_EXTRA}] to use the LangGraph model"
            ) from exc
        binding = self.binding
        name = model

        class KognitaChatModel(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "kognita"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
                rows = [
                    {"role": _role(message), "content": _content(message)}
                    for message in messages
                ]
                result = binding.require_completion(
                    model=str(kwargs.get("model") or name), messages=rows
                )
                reply = AIMessage(content=result.text)
                return ChatResult(generations=[ChatGeneration(message=reply)])

            def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
                raise GovernedRefusal(
                    GovernedResult(
                        released=False,
                        outcome=Outcome.DENY,
                        citations=(
                            "a streamed response cannot be restored or classified",
                        ),
                    )
                )
                yield  # pragma: no cover

        return KognitaChatModel()
