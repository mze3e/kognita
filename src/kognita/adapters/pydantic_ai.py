"""Pydantic AI adapter.

``tool`` wraps a function so a Pydantic AI tool call is
:func:`kognita.tools.run_governed`. ``sdk_model`` is a ``pydantic_ai.models.Model``
whose ``request`` is :meth:`kognita.gateway.Gateway.handle`. A refusal raises
:class:`~kognita.adapters.binding.GovernedRefusal` with the citations. The base
``request_stream`` already refuses to stream, so a stream never reaches the
provider.

Installs with the ``pydantic-ai`` extra. Importing this module does not import
Pydantic AI.
"""

from __future__ import annotations

from typing import Any

from kognita.adapters.binding import Binding
from kognita.exceptions import ConfigError

_EXTRA = "pydantic-ai"


def _messages(messages: list[Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for message in messages:
        parts = getattr(message, "parts", None)
        if parts is None:
            content = getattr(message, "content", "")
            rows.append(
                {
                    "role": "user",
                    "content": content if isinstance(content, str) else str(message),
                }
            )
            continue
        texts: list[str] = []
        for part in parts:
            content = getattr(part, "content", None)
            if isinstance(content, str) and content:
                texts.append(content)
        role = "assistant" if type(message).__name__ == "ModelResponse" else "user"
        rows.append({"role": role, "content": "\n".join(texts)})
    return rows


class PydanticAIAdapter:
    """Governed tools and a model for Pydantic AI."""

    def __init__(self, binding: Binding) -> None:
        self.binding = binding

    def tool(self, fn: Any) -> Any:
        """Wrap ``fn``. The original runs only after an ``ALLOW``."""
        return self.binding.wrap(fn)

    def complete(self, *, model: str, messages: list[dict[str, Any]]) -> Any:
        """One chat completion through the gateway. Refusals raise."""
        return self.binding.require_completion(model=model, messages=messages)

    def agent_tool(self, fn: Any) -> Any:
        """A Pydantic AI ``Tool`` whose body is :meth:`tool`."""
        try:
            from pydantic_ai import Tool
        except ImportError as exc:
            raise ConfigError(
                f"install kognita[{_EXTRA}] to wrap a Pydantic AI tool"
            ) from exc
        return Tool(self.tool(fn))

    def sdk_model(self, model: str = "governed") -> Any:
        """A Pydantic AI ``Model`` that completes through the gateway."""
        try:
            from pydantic_ai.messages import ModelResponse, TextPart
            from pydantic_ai.models import Model
        except ImportError as exc:
            raise ConfigError(
                f"install kognita[{_EXTRA}] to use the Pydantic AI model"
            ) from exc
        binding = self.binding
        name = model

        class KognitaModel(Model):
            def __init__(self) -> None:
                super().__init__()

            @property
            def model_name(self) -> str:
                return name

            @property
            def system(self) -> str:
                return "kognita"

            async def request(self, messages, model_settings, model_request_parameters):  # noqa: ARG002
                result = binding.require_completion(
                    model=name, messages=_messages(messages)
                )
                return ModelResponse(parts=[TextPart(content=result.text)])

        return KognitaModel()
