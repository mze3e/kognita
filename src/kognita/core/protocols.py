"""The contracts a domain pack and a storage backend implement.

These are :class:`typing.Protocol` definitions, so implementations need not
import or subclass anything from Kognita — a pack is an ordinary module that
happens to have the right shape.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from kognita.core.envelope import Check, Envelope, RuleContext
from kognita.core.vocabulary import Classification


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a normalised vector."""

    @property
    def dimension(self) -> int: ...

    @property
    def model(self) -> str: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Redactor(Protocol):
    """Removes sensitive spans from text and can put them back.

    ``restore`` exists so a redacted round-trip through an external model returns
    something useful: the model sees ``[PERSON_1]``, the caller sees the name.
    """

    def redact(self, text: str) -> tuple[str, dict[str, str]]: ...

    def restore(self, text: str, token_map: dict[str, str]) -> str: ...


@runtime_checkable
class Classifier(Protocol):
    """Derives a sensitivity classification for a piece of content.

    Classification drives entitlement filtering and the egress guard, so deriving
    it is preferable to asserting it — an unclassified document is one nobody has
    decided the handling rules for.
    """

    def classify(self, text: str, *, hint: Classification | None = None) -> Classification: ...


@runtime_checkable
class RuleEvaluator(Protocol):
    """Turns one policy row into zero or more checks."""

    def __call__(self, policy: Any, context: RuleContext) -> list[Check]: ...


@runtime_checkable
class DomainPack(Protocol):
    """A business domain: its subjects, its rules, its tools.

    The core never imports a pack; an application composes them. A pack supplies
    the two things the core cannot know — what the request's *attributes* are,
    and how to load the *subjects* it refers to.
    """

    name: str

    def resolve_attributes(self, envelope: Envelope, subjects: dict[str, Any]) -> dict[str, Any]:
        """Derive the attributes policy turns on, from the envelope and subjects."""
        ...

    def load_subjects(self, envelope: Envelope, session: Any) -> dict[str, Any]:
        """Load the rows an envelope refers to. Called before any policy runs."""
        ...

    def rules(self) -> dict[str, RuleEvaluator]:
        """Rule evaluators by ``rule_type``, merged over the core primitives."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """A graph backend the deterministic mirror can be written to."""

    def cypher(self, statement: str, *, allow_writes: bool = False) -> list[dict]: ...

    def execute_many(self, statements: list[str]) -> None: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Nearest-neighbour search over an already entitlement-filtered candidate set."""

    def search(
        self,
        query_vector: list[float],
        candidates: list[Any],
        *,
        top_k: int,
    ) -> list[tuple[Any, float]]: ...


@runtime_checkable
class ToolFn(Protocol):
    """A governed capability: runs only after the decision point allows it."""

    def __call__(self, envelope: Envelope, evaluation: Any, session: Any) -> Any: ...


class Clock(Protocol):
    """Supplies 'now'. Injected so decisions and expiry are testable."""

    def __call__(self) -> datetime: ...


__all__ = [
    "Embedder",
    "Redactor",
    "Classifier",
    "RuleEvaluator",
    "DomainPack",
    "GraphStore",
    "VectorIndex",
    "ToolFn",
    "Clock",
]
