"""Classification of one section.

A section is classified on its own text. A parent section's classification
is a floor for everything under it: a nested section may be more sensitive
and cannot be less. The floor of the pattern classifier still applies, so
an explicit public marking does not make a section publishable on its own.

Every result names the rule that decided it.
"""

from __future__ import annotations

from typing import Any

from kognita.classify import FixedClassifier, PatternClassifier
from kognita.exceptions import KognitaError
from kognita.vocabulary import Classification, classification_rank


def _name(classifier: Any) -> str:
    if isinstance(classifier, PatternClassifier):
        return f"{classifier.model} {classifier.version}"
    if isinstance(classifier, FixedClassifier):
        return "FixedClassifier 1"
    return type(classifier).__name__


def _levels(value: object) -> list[Classification]:
    if not isinstance(value, list):
        return []
    return [Classification(item) for item in value]


def _own_rule(classifier: Any, text: str, label: Classification) -> str:
    """The rule for a reading that was not raised by a parent section."""
    name = _name(classifier)
    if isinstance(classifier, PatternClassifier):
        explained = classifier.explain(text)
        markings = _levels(explained.get("markings"))
        indicators = _levels(explained.get("indicators"))
        if any(mark == label for mark in markings):
            return f"{name}: marking {label.value}"
        if any(found == label for found in indicators):
            return f"{name}: identifier {label.value}"
        if markings or indicators:
            seen = ", ".join(item.value for item in (*markings, *indicators))
            return (
                f"{name}: floor {label.value}; reading {seen} is below the floor "
                "and does not lower the section"
            )
        return (
            f"{name}: floor {label.value}; no handling marking and no identifier; "
            "absence of indicators is not a lower classification"
        )
    if isinstance(classifier, FixedClassifier):
        return f"{name}: fixed handling rule {label.value}"
    return (
        f"{name}: uncalibrated classifier returned {label.value}; "
        "confidence is not a permit"
    )


def classify_section(
    text: str,
    *,
    parent: Classification | None = None,
    parent_section: str = "",
    classifier: Any = None,
) -> tuple[Classification, str]:
    """Classify ``text`` and cite the rule.

    ``parent`` is the classification of the heading this section sits under.
    It is a floor. ``classifier`` defaults to the pattern classifier already
    in the core; its own floor still applies when ``parent`` is absent.
    """
    chosen = classifier if classifier is not None else PatternClassifier()
    own = Classification(chosen.classify(text))
    if parent is None:
        label = own
    else:
        label = Classification(chosen.classify(text, hint=parent))
    if parent is not None and classification_rank(label) > classification_rank(own):
        where = parent_section or "parent section"
        return label, (
            f"{_name(chosen)}: inherited {label.value} from {where}; "
            "a section subtree cannot be classified below its parent"
        )
    rule = _own_rule(chosen, text, label)
    if not rule.strip():
        raise KognitaError(
            "section classification produced no rule; ingestion indexes nothing"
        )
    return label, rule


__all__ = ["classify_section"]
