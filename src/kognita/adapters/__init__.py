"""Adapters to external providers.

Everything here reaches the network and therefore lives behind an extra. The
core never imports this package; an application chooses an adapter and hands it
in, which is what keeps ``kognita.core`` installable on four dependencies.
"""
from kognita.adapters.embeddings import OpenAICompatibleEmbedder

__all__ = ["OpenAICompatibleEmbedder"]
