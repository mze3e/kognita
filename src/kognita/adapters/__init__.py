"""Adapters to external providers.

Everything here reaches the network, which is why it is a package of its own
rather than part of the decision engine: ``kognita`` decides whether a request
is permitted, and must stay installable without the machinery that answers it.

:class:`~kognita.adapters.embeddings.OpenAICompatibleEmbedder` speaks the
``/v1/embeddings`` contract over :mod:`urllib`, so it costs no dependency and is
imported eagerly.

Framework adapters are not imported here. Each one is its own module and its
own extra — ``hermes-agent``, ``openai-agents``, ``langgraph``, ``pydantic-ai``
— and importing it does not import the framework. The decision engine never
imports them.
"""

from kognita.adapters.embeddings import OpenAICompatibleEmbedder

__all__ = ["OpenAICompatibleEmbedder"]
