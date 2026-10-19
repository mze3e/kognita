"""The decide and gateway path that framework adapters call.

Tool calls go through :func:`kognita.tools.run_governed`: admission or a
granted approval releases the body. Model calls go through
:meth:`kognita.gateway.Gateway.handle`, which decides before forwarding.
Denials, escalations, pending approvals and missing evaluations withhold
the result, with citations from the checks that decided it.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kognita.adapters._binding import _complete
from kognita.credentials import verify as _verify
from kognita.envelope import Evaluation
from kognita.exceptions import KognitaError
from kognita.gateway import Gateway
from kognita.tools import Run, ToolRegistry, run_governed
from kognita.vocabulary import Outcome


@dataclass(frozen=True)
class GovernedResult:
    """One tool or model call after the decision point."""

    released: bool
    outcome: Outcome
    citations: tuple[str, ...]
    data: Any = None
    evaluation: Evaluation | None = None

    @property
    def text(self) -> str:
        """Assistant text from an OpenAI-shaped completion, or an empty string."""
        data = self.data
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            return ""
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        return content if isinstance(content, str) else ""


class GovernedRefusal(KognitaError):
    """The framework must not continue. The decision's citations are the reason."""

    def __init__(self, result: GovernedResult) -> None:
        self.result = result
        self.outcome = result.outcome
        self.citations = result.citations
        cited = (
            "; ".join(result.citations)
            if result.citations
            else "refused before a rule was cited"
        )
        super().__init__(f"{result.outcome.value}: {cited}")


def citations_of(evaluation: Evaluation | None) -> tuple[str, ...]:
    if evaluation is None:
        return ()
    return tuple(check.citation for check in evaluation.basis() if check.citation)


def bind_arguments(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Map a framework call onto the dict ``run_governed`` hashes into the envelope."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
    except TypeError as exc:
        raise GovernedRefusal(
            GovernedResult(
                released=False,
                outcome=Outcome.DENY,
                citations=(f"tool arguments were refused: {exc}",),
            )
        ) from exc
    bound.apply_defaults()
    return dict(bound.arguments)


def _refused(message: str) -> GovernedResult:
    return GovernedResult(released=False, outcome=Outcome.DENY, citations=(message,))


@dataclass
class Binding:
    """Session, pack and gateway a deployment hands to every adapter."""

    session: Any
    evidence: Any
    pack: Any
    gateway: Gateway
    registry: ToolRegistry
    principal: str
    purpose: str
    agent_name: str | None
    purposes: Sequence[str]
    actor_location: str = ""
    subject_type: str | None = None
    subject_id: str | None = None
    run: Run | None = None
    agent_credential: str | None = field(default=None, repr=False)
    delegation_id: str | None = None

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        fn: Callable[..., Any],
    ) -> GovernedResult:
        """Authorise ``name``, then call ``fn`` only when released."""
        from kognita.envelope import Envelope

        ran: list[Any] = []
        def body(envelope: Envelope, evaluation: Evaluation, session: Any) -> Any:
            ran.append(args := envelope.arguments)
            return fn(**args) if args else fn()

        if not _verify(self.session, self.agent_name or "", self.agent_credential):
            return _refused("S2-CREDENTIAL: agent not registered or credential invalid")
        if name in self.registry:
            self.registry.get(name).fn = body
        else:
            self.registry.register(name, body)
        envelope = Envelope(
            principal=self.principal,
            purpose=self.purpose,
            tool=name,
            actor_location=self.actor_location,
            agent_name=self.agent_name,
            subject_type=self.subject_type,
            subject_id=self.subject_id,
            arguments=dict(arguments),
            delegation_id=self.delegation_id,
            context={"_agent_credential": self.agent_credential},
        )
        try:
            result = run_governed(
                self.session,
                envelope,
                registry=self.registry,
                evidence=self.evidence,
                pack=self.pack,
                purposes=self.purposes,
                run=self.run,
            )
        except Exception as exc:
            return _refused(f"the tool was refused before it ran: {exc}")
        released = bool(ran) and not result.approval_required and result.outcome in (
            Outcome.ALLOW, Outcome.HUMAN_APPROVAL)
        return GovernedResult(
            released=released,
            outcome=result.outcome,
            citations=citations_of(result.evaluation),
            data=result.data if released else None,
            evaluation=result.evaluation,
        )

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return ``fn``'s signature, with the body behind :meth:`call_tool`."""
        name = getattr(fn, "__name__", "tool")

        @functools.wraps(fn)
        def governed(*args: Any, **kwargs: Any) -> Any:
            arguments = bind_arguments(fn, args, kwargs)
            result = self.call_tool(str(name), arguments, fn)
            if not result.released:
                raise GovernedRefusal(result)
            return result.data

        return governed

    def complete(
        self, *, model: str, messages: Sequence[Mapping[str, Any]]
    ) -> GovernedResult:
        """One chat completion through the gateway. The provider is not called on a refusal."""
        return _complete(self, model, messages)

    def require_completion(
        self, *, model: str, messages: Sequence[Mapping[str, Any]]
    ) -> GovernedResult:
        result = self.complete(model=model, messages=messages)
        if not result.released:
            raise GovernedRefusal(result)
        return result
