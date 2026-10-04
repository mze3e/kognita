"""Deriving a classification for content.

Entitlement filtering and the egress guard both key off ``classification``, so
where that value comes from matters. Asserting it by hand means an unclassified
document is one nobody has decided the handling rules for — and in practice
defaults to whatever is most convenient.

:class:`PatternClassifier` derives it from indicators in the text: contact
details and account identifiers suggest confidential, explicit restriction
markings suggest restricted. Like the redactor it is a floor, not a guarantee,
and it fails *upward* — an ambiguous document is treated as more sensitive
rather than less.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from kognita.canonical import hash_text
from kognita.vocabulary import Classification, classification_rank

#: Explicit handling markings, checked first — an author's own label wins.
DEFAULT_MARKINGS: tuple[tuple[str, Classification], ...] = (
    (r"\brestricted\b|\bneed[- ]to[- ]know\b|\bstrictly confidential\b", Classification.C3),
    (r"\bconfidential\b|\bclient[- ]identifying\b|\bpersonal data\b", Classification.C2),
    (r"\binternal(?: use)?(?: only)?\b", Classification.C1),
    (r"\bpublic\b|\bunclassified\b|\bfor publication\b", Classification.C0),
)

#: Indicators that content carries identifying data even if unmarked.
DEFAULT_INDICATORS: tuple[tuple[str, Classification], ...] = (
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", Classification.C2),
    (r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}\b", Classification.C2),
    (r"\b[A-Z]{2}-[A-Z]{2}-\d{4,}\b", Classification.C2),
    (r"\+\d[\d\s().-]{7,}\d", Classification.C2),
    (r"\b(?:\d[ -]*?){13,16}\b", Classification.C2),
)

#: Calibrated confidence for the pattern classifier. These are not probabilities
#: fitted to a corpus. They are a fixed scale so a policy threshold can tell a
#: clear label from an absence of evidence. Absence of indicators is not
#: evidence that the text is publishable, so the floor sits below a threshold
#: a deployment would treat as sure.
MARKING_CONFIDENCE = 0.99
INDICATOR_CONFIDENCE = 0.94
FLOOR_CONFIDENCE = 0.50

#: Model identifier and calibration version recorded with every label.
#: ``version`` changes when the patterns or this scale change, so a recorded
#: decision names the classifier that produced it.
PATTERN_MODEL = "pattern"
PATTERN_VERSION = "1"


@dataclass
class PatternClassifier:
    """Rule-based classification with a conservative floor.

    ``floor`` is the least sensitive result this classifier will ever return.
    It defaults to C1 because "we found no indicators" is not evidence that
    content is publishable.
    """

    markings: Sequence[tuple[str, Classification]] = DEFAULT_MARKINGS
    indicators: Sequence[tuple[str, Classification]] = DEFAULT_INDICATORS
    floor: Classification = Classification.C1
    #: Recorded on classifier evidence. Not a product name — the rule and
    #: pattern classifier the core already ships.
    model: str = PATTERN_MODEL
    version: str = PATTERN_VERSION
    _compiled_markings: list[tuple[re.Pattern[str], Classification]] = field(
        init=False, repr=False, default_factory=list
    )
    _compiled_indicators: list[tuple[re.Pattern[str], Classification]] = field(
        init=False, repr=False, default_factory=list
    )

    def __post_init__(self) -> None:
        self._compiled_markings = [
            (re.compile(p, re.IGNORECASE), c) for p, c in self.markings
        ]
        self._compiled_indicators = [
            (re.compile(p), c) for p, c in self.indicators
        ]

    def classify(
        self, text: str, *, hint: Classification | None = None
    ) -> Classification:
        """Return the most sensitive classification the evidence supports.

        A ``hint`` (an ingest-time assertion, or a source's known handling) acts
        as another lower bound, never an upper one: a caller may raise the
        classification of a document but not talk it down.
        """
        candidates: list[Classification] = [self.floor]
        if hint is not None:
            candidates.append(Classification(hint))

        for pattern, level in self._compiled_markings:
            if pattern.search(text):
                candidates.append(level)
                break

        for pattern, level in self._compiled_indicators:
            if pattern.search(text):
                candidates.append(level)

        return max(candidates, key=classification_rank)

    def calibrated_confidence(
        self, text: str, *, label: Classification | None = None
    ) -> float:
        """How sure this reading is, on the fixed scale above.

        An explicit marking that supports the label is the surest reading.
        An identifier pattern is next. A label that rests only on the floor —
        including text that calls itself public and is held at the floor — is
        not sure, and a confidence policy treats it as uncertain.

        Pass ``label`` when the caller has already classified, so recording
        confidence does not classify a second time.
        """
        resolved = self.classify(text) if label is None else Classification(label)
        marking: Classification | None = None
        for pattern, level in self._compiled_markings:
            if pattern.search(text):
                marking = Classification(level)
                break
        if marking is not None and marking == resolved:
            return MARKING_CONFIDENCE
        if any(
            pattern.search(text) and Classification(level) == resolved
            for pattern, level in self._compiled_indicators
        ):
            return INDICATOR_CONFIDENCE
        return FLOOR_CONFIDENCE

    def explain(self, text: str) -> dict[str, object]:
        """Which patterns fired — for a review screen, and for evidence."""
        marks = [
            Classification(level).value
            for pattern, level in self._compiled_markings
            if pattern.search(text)
        ]
        indicators = [
            Classification(level).value
            for pattern, level in self._compiled_indicators
            if pattern.search(text)
        ]
        return {
            "classification": Classification(self.classify(text)).value,
            "markings": marks,
            "indicators": indicators,
            "floor": Classification(self.floor).value,
            "model": self.model,
            "version": self.version,
            "confidence": self.calibrated_confidence(text),
            "input_hash": hash_text(text),
        }


@dataclass
class FixedClassifier:
    """Always returns one classification. For corpora with a known handling rule."""

    level: Classification = Classification.C1

    def classify(
        self, text: str, *, hint: Classification | None = None
    ) -> Classification:
        if hint is None:
            return self.level
        return max([self.level, Classification(hint)], key=classification_rank)


def classifier_record(classifier: Any, text: str) -> dict[str, Any]:
    """The evidence a decision keeps so it can be replayed without reclassifying.

    ``model``, ``version``, ``label``, ``confidence`` and ``input_hash`` are
    the record. The text itself is not: the hash is enough to show which input
    produced the label, and the label is what ``decide`` will read.
    """
    label = Classification(classifier.classify(text))
    if isinstance(classifier, PatternClassifier):
        model = classifier.model
        version = classifier.version
        confidence = classifier.calibrated_confidence(text, label=label)
    elif isinstance(classifier, FixedClassifier):
        # A fixed handling rule is the deployment's choice, not a guess.
        model = "FixedClassifier"
        version = "1"
        confidence = 1.0
    else:
        # An uncalibrated backend has no basis for ALLOW. A confidence policy
        # sees 0 and escalates.
        model = type(classifier).__name__
        version = str(getattr(classifier, "version", "") or "")
        confidence = 0.0
    return {
        "model": model,
        "version": version,
        "label": label.value,
        "confidence": confidence,
        "input_hash": hash_text(text),
    }


def most_sensitive(values: Iterable[Classification | str]) -> Classification:
    """The most restricted classification in ``values``, defaulting to C0."""
    levels = [Classification(v) for v in values]
    return max(levels, key=classification_rank) if levels else Classification.C0


__all__ = [
    "PatternClassifier",
    "FixedClassifier",
    "classifier_record",
    "most_sensitive",
    "MARKING_CONFIDENCE",
    "INDICATOR_CONFIDENCE",
    "FLOOR_CONFIDENCE",
    "PATTERN_MODEL",
    "PATTERN_VERSION",
]
