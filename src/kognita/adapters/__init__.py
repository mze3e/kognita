"""Adapters to external providers.

Everything here reaches the network, which is why it is a package of its own
rather than part of the decision engine: ``kognita`` decides whether a request
is permitted, and must stay installable without the machinery that answers it.

:class:`~kognita.adapters.embeddings.OpenAICompatibleEmbedder` speaks the
``/v1/embeddings`` contract over :mod:`urllib`, so it costs no dependency and is
imported eagerly. Adapters that need a provider SDK will be bound lazily here
and raise :class:`~kognita.exceptions.ConfigError` naming the extra that
supplies them.
"""
from kognita.adapters.embeddings import OpenAICompatibleEmbedder

__all__ = ["OpenAICompatibleEmbedder"]
