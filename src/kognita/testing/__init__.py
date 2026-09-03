"""The conformance kit — invariants a domain pack must not break.

A domain pack lives in its own repository and supplies its own rules, tools and
subjects. What it must not do is quietly weaken the guarantees the core exists to
provide: authorise before discovery, fail closed, cite every decision, evidence
everything.

Run the kit against a pack::

    # conftest.py in the pack's repository
    from kognita.testing import ConformanceCase, Harness

    class TestMyPack(ConformanceCase):
        harness = Harness(pack=MyPack(), purposes=(...), seed=...)
        allow_envelope = Envelope(...)
        deny_envelope = Envelope(...)

or as a suite over the bundled fixture pack::

    pytest --pyargs kognita.testing.conformance
"""
from kognita.testing.conformance import ConformanceCase
from kognita.testing.harness import Harness, PackUnderTest

__all__ = ["ConformanceCase", "Harness", "PackUnderTest"]
