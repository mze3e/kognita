"""OpenAI Agents SDK adapter.

``tool`` wraps a function so the SDK's tool call is :func:`kognita.tools.run_governed`.
``sdk_model`` is an ``agents.models.interface.Model`` whose ``get_response`` is
:meth:`kognita.gateway.Gateway.handle`. A refusal raises
:class:`~kognita.adapters.binding.GovernedRefusal` with the decision's citations,
and the tool body and the provider are not called. Streaming is refused: the
gateway cannot restore or classify a stream.

Installs with the ``openai-agents`` extra. Importing this module does not
import the SDK.
"""

from __future__ import annotations

from typing import Any

from kognita.adapters.binding import Binding, GovernedRefusal, GovernedResult
from kognita.exceptions import ConfigError
from kognita.vocabulary import Outcome

_EXTRA = "openai-agents"


def _messages(system: str | None, data: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if system:
        rows.append({"role": "system", "content": system})
    if isinstance(data, str):
        rows.append({"role": "user", "content": data})
        return rows
    if not isinstance(data, list):
        rows.append({"role": "user", "content": str(data)})
        return rows
    for item in data:
        if isinstance(item, str):
            rows.append({"role": "user", "content": item})
            continue
        if isinstance(item, dict):
            content = item.get("content")
            text = content if isinstance(content, str) else ""
            rows.append({"role": str(item.get("role") or "user"), "content": text})
            continue
        content = getattr(item, "content", None)
        rows.append(
            {
                "role": str(getattr(item, "role", "user")),
                "content": content if isinstance(content, str) else str(item),
            }
        )
    return rows


def _response(text: str) -> Any:
    from agents.items import ModelResponse
    from agents.usage import Usage
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    message = ResponseOutputMessage(
        id="kognita",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
    )
    return ModelResponse(output=[message], usage=Usage(), response_id=None)


class OpenAIAgentsAdapter:
    """Governed tools and a governed model for the OpenAI Agents SDK."""

    def __init__(self, binding: Binding) -> None:
        self.binding = binding

    def tool(self, fn: Any) -> Any:
        """Wrap ``fn``. The original runs only after an ``ALLOW``."""
        return self.binding.wrap(fn)

    def complete(self, *, model: str, messages: list[dict[str, Any]]) -> Any:
        """One chat completion through the gateway. Refusals raise."""
        return self.binding.require_completion(model=model, messages=messages)

    def function_tool(self, fn: Any) -> Any:
        """An SDK ``FunctionTool`` whose body is :meth:`tool`."""
        try:
            from agents import function_tool
        except ImportError as exc:
            raise ConfigError(
                f"install kognita[{_EXTRA}] to wrap an Agents SDK tool"
            ) from exc
        return function_tool(self.tool(fn), name_override=getattr(fn, "__name__", None))

    def sdk_model(self, model: str = "governed") -> Any:
        """An Agents SDK ``Model`` that completes through the gateway."""
        try:
            from agents.models.interface import Model
        except ImportError as exc:
            raise ConfigError(
                f"install kognita[{_EXTRA}] to use the Agents SDK model"
            ) from exc
        binding = self.binding
        name = model

        class KognitaAgentsModel(Model):
            async def get_response(
                self,
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                **_kwargs,
            ):  # noqa: A002
                result = binding.require_completion(
                    model=name, messages=_messages(system_instructions, input)
                )
                return _response(result.text)

            def stream_response(self, *args, **kwargs):  # noqa: ARG002
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

        return KognitaAgentsModel()
