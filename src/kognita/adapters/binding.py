"""The decide and gateway path that framework adapters call.

Tool calls go through :func:`kognita.tools.run_governed`, which calls
:func:`kognita.governance.decide` and does not run the tool body unless the
outcome is ``ALLOW``. Model calls go through :meth:`kognita.gateway.Gateway.handle`,
which decides before any byte is forwarded. Anything else — a denial, an
escalation, a human approval, a missing evaluation — is not released, and the
result carries the citations from the checks that decided it.
"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        fn: Callable[..., Any],
    ) -> GovernedResult:
        """Authorise ``name``, then call ``fn`` only when the outcome is ``ALLOW``."""
        from kognita.envelope import Envelope

        def body(envelope: Envelope, evaluation: Evaluation, session: Any) -> Any:
            args = envelope.arguments
            return fn(**args) if args else fn()

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
        released = result.outcome is Outcome.ALLOW and not result.approval_required
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
        try:
            body = json.dumps(
                {"model": model, "messages": [dict(message) for message in messages]}
            ).encode()
        except (TypeError, ValueError) as exc:
            return _refused(f"the model call was refused before it was sent: {exc}")
        try:
            response = self.gateway.handle(
                "POST",
                "/v1/chat/completions",
                {},
                body,
                session=self.session,
                run=self.run,
            )
        except Exception as exc:
            return _refused(f"the model call was refused before it was sent: {exc}")
        evaluation = response.evaluation
        released = (
            response.status == 200
            and evaluation is not None
            and evaluation.outcome is Outcome.ALLOW
        )
        data = None
        if released:
            try:
                parsed = json.loads(response.body.decode())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return _refused(f"the model call was refused before it was sent: {exc}")
            data = parsed if isinstance(parsed, dict) else None
            released = data is not None
        outcome = evaluation.outcome if evaluation is not None else Outcome.DENY
        return GovernedResult(
            released=released,
            outcome=outcome,
            citations=citations_of(evaluation),
            data=data,
            evaluation=evaluation,
        )

    def require_completion(
        self, *, model: str, messages: Sequence[Mapping[str, Any]]
    ) -> GovernedResult:
        result = self.complete(model=model, messages=messages)
        if not result.released:
            raise GovernedRefusal(result)
        return result
