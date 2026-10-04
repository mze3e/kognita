"""Kognita exception hierarchy."""
from __future__ import annotations


class KognitaError(Exception):
    """Base class for all Kognita-specific errors."""


class ConfigError(KognitaError):
    """Invalid or incomplete configuration."""


class PolicyEditError(KognitaError):
    """An effective policy was edited in place.

    A policy that is or was in force is not rewritten. Closing its window is
    the one change the row accepts; anything else is a new effective-dated row.
    """


class ReplayMismatch(KognitaError):
    """A recorded decision does not match an input it pinned."""


class RetentionError(KognitaError):
    """Retained content could not be stored or erased."""


class ProviderError(KognitaError):
    """An LLM or embedder provider returned an error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def extract_api_error(exc: Exception) -> str:
    """Format a human-readable error string from a provider API exception.

    Handles OpenAI / Groq (openai.APIStatusError), Anthropic (anthropic.APIStatusError),
    and Google Gemini (google.api_core.exceptions.GoogleAPICallError), plus any
    generic exception that carries a status_code attribute.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "code", None)

    body = getattr(exc, "body", None)
    if body:
        detail = str(body)
    else:
        detail = getattr(exc, "message", None) or str(exc)

    if status_code:
        return f"HTTP {status_code}: {detail}"
    return f"{type(exc).__name__}: {detail}"
