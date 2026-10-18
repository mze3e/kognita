"""Hermes Agent adapter.

Hermes already has an approvals mode. ``pre_tool_call`` is the hook that can
stop a tool: a ``block`` directive is an error returned to the model, and an
``approve`` directive escalates to Hermes's human gate, which fails closed on
denial or timeout. :meth:`HermesAdapter.pre_tool_call` sends the call through
:func:`kognita.tools.run_governed`. The tool Hermes would run is not the body
of that call. On ``ALLOW`` the hook returns no directive and Hermes executes.
On anything else it does not.

Hermes's ``pre_llm_call`` and ``pre_api_request`` hooks cannot veto, so a model
call is not governed by returning a value from them. hermes-agent 0.19 points a
model at an OpenAI-compatible base URL (``AIAgent(base_url=...)``); a newer
provider plugin would return :meth:`HermesAdapter.client` from ``create_client``.
Either way :meth:`kognita.gateway.Gateway.handle` decides, and the provider is
contacted only on ``ALLOW``.

Installs with the ``hermes-agent`` extra. This module does not import Hermes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kognita.adapters.binding import Binding, GovernedRefusal, GovernedResult
from kognita.vocabulary import Outcome


def _message(result: GovernedResult) -> str:
    cited = "; ".join(result.citations)
    return (
        f"{result.outcome.value}: {cited}"
        if cited
        else f"{result.outcome.value}: refused"
    )


class HermesAdapter:
    """Plugin callbacks and the governed chat client for Hermes Agent."""

    def __init__(self, binding: Binding) -> None:
        self.binding = binding

    def register(self, ctx: Any) -> None:
        """Subscribe the tool hook. ``ctx`` is Hermes's plugin context."""
        ctx.register_hook("pre_tool_call", self.pre_tool_call)

    def pre_tool_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None = None,
        **_kwargs: Any,
    ) -> dict[str, str] | None:
        """Return Hermes's directive, or ``None`` when the tool may run."""
        if args is not None and not isinstance(args, Mapping):
            return {
                "action": "block",
                "message": "DENY: tool arguments are not a mapping",
            }
        arguments = dict(args or {})

        def released(**_kw: Any) -> dict[str, bool]:
            return {"released": True}

        result = self.binding.call_tool(str(tool_name), arguments, released)
        if result.outcome is Outcome.HUMAN_APPROVAL:
            return {
                "action": "approve",
                "message": _message(result),
                "rule_key": "kognita",
            }
        if not result.released:
            return {"action": "block", "message": _message(result)}
        return None

    def create_client(self, **_kwargs: Any) -> Any:
        """The client a newer Hermes ``ProviderProfile.create_client`` returns.

        hermes-agent 0.19 has no such method. Use :meth:`client` directly, or
        point ``AIAgent(base_url=...)`` at ``kognita serve``.
        """
        return self.client()

    def client(self) -> Any:
        """An object with ``chat.completions.create``, backed by the gateway."""
        binding = self.binding

        class Completions:
            def create(
                self,
                *,
                model: str,
                messages: list[Mapping[str, Any]],
                stream: bool = False,
                **_kwargs: Any,
            ) -> dict[str, Any]:
                if stream:
                    raise GovernedRefusal(
                        GovernedResult(
                            released=False,
                            outcome=Outcome.DENY,
                            citations=(
                                "a streamed response cannot be restored or classified",
                            ),
                        )
                    )
                result = binding.require_completion(model=model, messages=messages)
                return result.data

        class Chat:
            def __init__(self) -> None:
                self.completions = Completions()

        class Client:
            def __init__(self) -> None:
                self.chat = Chat()

        return Client()
