"""Adapters to external providers.

Everything here reaches the network and therefore lives behind an extra. The
core never imports this package; an application chooses an adapter and hands it
in, which is what keeps ``kognita.core`` installable on four dependencies.

Adapters are bound lazily (PEP 562), raising ConfigError if dependencies are
missing.
"""
from __future__ import annotations

import sys
from typing import Any

from kognita.exceptions import ConfigError


def __getattr__(name: str) -> Any:
    """Lazy binding for adapters."""
    if name == "OpenAICompatibleEmbedder":
        try:
            from kognita.adapters.embeddings import OpenAICompatibleEmbedder

            return OpenAICompatibleEmbedder
        except ImportError as e:
            raise ConfigError(
                "OpenAICompatibleEmbedder requires urllib (stdlib); "
                "it was not importable. Check your Python installation."
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return ["OpenAICompatibleEmbedder"]
