"""Policy files: a serialization of the rows :func:`kognita.governance.decide` already evaluates.

A policy file is YAML for the columns on :class:`kognita.models.Policy`. A change
is a document of ``## ADDED``, ``## MODIFIED`` and ``## REMOVED`` rules. Scenario
files sit next to the policy files and name the rule they exercise.

The YAML this module reads is the subset those documents use: block mappings and
sequences, flow mappings and sequences, and scalars. It is not a general YAML
implementation. A document outside that subset is an error.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlmodel import Session, select

from kognita.canonical import canonical_hash, canonical_json
from kognita.envelope import Check, Envelope
from kognita.facts import FreshnessRule
from kognita.exceptions import KognitaError
from kognita.governance import PolicySnapshot, decide
from kognita.models import Policy, as_utc, utcnow
from kognita.rules import CORE_RULES
from kognita.vocabulary import (
    COMMUNICATION_LEVELS,
    DISCLOSURE_TRIGGERS,
    CheckResult,
    Outcome,
)

#: A rule may carry this marker until the question in it is answered.
CLARIFICATION_RE = re.compile(r"\[NEEDS CLARIFICATION:\s*([^\]]*)\]")
_CLARIFICATION_OPEN = "[NEEDS CLARIFICATION:"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEADING_RE = re.compile(r"^##[ \t]+(ADDED|MODIFIED|REMOVED)\b(.*)$")
_ON_VIOLATION = {item.value for item in CheckResult}
_OUTCOMES = {item.value for item in Outcome}

#: Purpose a scenario uses when it does not name one. The scenario's purpose is
#: on the allowlist, so the closed purpose check does not decide the case.
_SCENARIO_PURPOSE = "POLICY_TEST"

_RULE_FIELDS = {
    "id",
    "regime",
    "rule_type",
    "applies_to",
    "citation",
    "effective_from",
    "effective_to",
    "rule",
}
_GIVEN_FIELDS = {
    "principal",
    "purpose",
    "tool",
    "actor_location",
    "agent_name",
    "subject_type",
    "subject_id",
    "subjects",
    "arguments",
    "roles",
    "scopes",
    "is_admin",
    "context",
    "attributes",
}
_WHEN_FIELDS = {"tool", "arguments"}
_THEN_FIELDS = {"decision", "cites"}
_ENVELOPE_FIELDS = _GIVEN_FIELDS - {"attributes"}
_EXPLAIN_FIELDS = _ENVELOPE_FIELDS | {"attributes", "as_of", "purposes"}

_SCHEMAS: dict[str, dict[str, Any]] = {
    "ATTRIBUTE_ALLOWLIST": {
        "required_maps": ("allow",),
        "optional_lists": (),
        "optional": ("on_violation", "description"),
    },
    "ATTRIBUTE_DENYLIST": {
        "required_maps": ("deny",),
        "optional_lists": (),
        "optional": ("on_violation", "description"),
    },
    "REQUIRES_FLAG": {
        "required_maps": (),
        "required_lists": ("flags",),
        "optional": ("on_violation", "description"),
    },
    "REQUIRES_HUMAN_APPROVAL": {
        "required_maps": (),
        "optional_lists": ("tools", "purposes"),
        "optional": ("description",),
    },
    "TWO_SIGNATURE_APPROVAL": {
        "required_maps": (),
        "optional_lists": ("tools", "purposes"),
        "optional": ("description",),
    },
    "CLASSIFIER_CONFIDENCE": {
        "required_maps": (),
        "optional_lists": (),
        "optional": ("min_confidence", "description"),
    },
    "CONTACT_FREQUENCY": {
        "required_maps": (),
        "optional_lists": ("tools",),
        "optional": ("max_contacts", "window_days", "description"),
    },
    "PRINCIPAL_EQUALITY": {
        "required_maps": (),
        "optional_lists": ("tools", "purposes"),
        "optional": ("equals", "on_violation", "description"),
    },
    "PROHIBITED": {
        "required_maps": (),
        "optional_lists": (),
        "optional": ("on_violation", "description"),
    },
    "CONTENT_POLICY": {
        "required_maps": (),
        "optional_lists": (
            "sources",
            "levels",
            "lanes",
            "required_disclosures",
            "prohibited_claims",
            "sensitive",
            "approved_tones",
            "permitted_jurisdictions",
            "permitted_channels",
        ),
        "optional": (
            "description",
            "require_risk_language",
            "require_research_attribution",
            "validator_audiences",
        ),
    },
    "AI_DISCLOSURE": {
        "required_maps": (),
        "required_lists": ("triggers",),
        "optional_lists": ("jurisdictions", "use_cases"),
        "optional": ("text", "description"),
    },
    "CIRCUIT_BREAKER": {
        "required_maps": (),
        "optional_lists": ("usecases",),
        "optional": (
            "dimension",
            "value",
            "on_violation",
            "description",
            "accountable_owner",
        ),
    },
}

_BREAKER_DIMENSIONS = frozenset({"usecase", "model", "prompt", "policy"})


class PolicyLanguageError(KognitaError):
    """A policy file, change, or scenario could not be accepted."""

    def __init__(self, findings: list[Finding]):
        self.findings = list(findings)
        super().__init__("; ".join(format_finding(item) for item in self.findings))


@dataclass(frozen=True)
class Finding:
    """One problem in a policy set or a change."""

    code: str
    message: str
    rule_id: str | None = None
    #: True when only ``validate --strict`` rejects the document.
    strict: bool = False


@dataclass(frozen=True)
class RuleDocument:
    """One policy row, identified by the stable ``id`` the files use."""

    id: str
    regime: str
    rule_type: str
    citation: str
    effective_from: datetime
    rule: dict[str, Any]
    applies_to: str | None = None
    effective_to: datetime | None = None
    source: str = ""


@dataclass(frozen=True)
class Scenario:
    """One ``given / when / then`` case for a named rule."""

    name: str
    given: dict[str, Any]
    when: dict[str, Any]
    then: dict[str, Any]


@dataclass(frozen=True)
class ScenarioDocument:
    """The scenarios colocated with one rule."""

    rule: str
    scenarios: tuple[Scenario, ...]
    source: str = ""


@dataclass(frozen=True)
class PolicySet:
    """The rules and the scenarios that sit beside them."""

    rules: tuple[RuleDocument, ...]
    scenarios: tuple[ScenarioDocument, ...] = ()
    source: str = ""
    freshness: tuple[FreshnessRule, ...] = ()

    def by_id(self) -> dict[str, RuleDocument]:
        return {rule.id: rule for rule in self.rules}


@dataclass(frozen=True)
class PolicyChange:
    """A delta: rules added, rules replaced, and rule ids removed."""

    added: tuple[RuleDocument, ...] = ()
    modified: tuple[RuleDocument, ...] = ()
    removed: tuple[str, ...] = ()
    source: str = ""
    text: str = ""


@dataclass(frozen=True)
class ScenarioResult:
    """The outcome of one scenario."""

    rule_id: str
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RuleExplanation:
    """Whether one rule engaged, and the checks it produced."""

    rule_id: str
    engaged: bool
    why: str
    applies_to: str | None
    checks: tuple[Check, ...]


@dataclass(frozen=True)
class Explanation:
    """The engagement report for one envelope."""

    outcome: Outcome
    rules: tuple[RuleExplanation, ...]
    other_checks: tuple[Check, ...]
    as_of: datetime


def format_finding(finding: Finding) -> str:
    """One finding, as a single line."""
    prefix = f"{finding.rule_id}: " if finding.rule_id else ""
    return f"{finding.code}: {prefix}{finding.message}"


def blocking_findings(findings: Iterable[Finding], *, strict: bool) -> list[Finding]:
    """Findings that fail validation. Strict mode keeps the strict ones too."""
    if strict:
        return list(findings)
    return [item for item in findings if not item.strict]


# ── YAML subset ──────────────────────────────────────────────────────────────


class _YamlError(ValueError):
    def __init__(self, message: str, *, line: int | None = None) -> None:
        self.line = line
        if line is None:
            super().__init__(message)
        else:
            super().__init__(f"line {line}: {message}")


@dataclass(frozen=True)
class _Line:
    number: int
    indent: int
    text: str


def _strip_comment(text: str) -> str:
    out: list[str] = []
    i = 0
    quote: str | None = None
    while i < len(text):
        ch = text[i]
        if quote is not None:
            out.append(ch)
            if quote == "'" and ch == "'" and i + 1 < len(text) and text[i + 1] == "'":
                out.append("'")
                i += 2
                continue
            if quote == '"' and ch == "\\":
                if i + 1 < len(text):
                    out.append(text[i + 1])
                    i += 2
                    continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "#":
            break
        out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _prepare(text: str) -> list[_Line]:
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalised.startswith("\ufeff"):
        normalised = normalised[1:]
    lines: list[_Line] = []
    for number, raw in enumerate(normalised.split("\n"), start=1):
        if "\t" in raw:
            raise _YamlError("tabs are not allowed in policy YAML", line=number)
        body = _strip_comment(raw)
        if not body.strip():
            continue
        if body.strip() == "---":
            continue
        indent = len(body) - len(body.lstrip(" "))
        lines.append(_Line(number, indent, body.strip()))
    return lines


def _parse_scalar(text: str, line: int) -> Any:
    value = text.strip()
    if value == "" or value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return _unquote(value, line)
    if re.fullmatch(r"-?(0|[1-9][0-9]*)", value):
        return int(value)
    if re.fullmatch(r"-?(0|[1-9][0-9]*)\.[0-9]+(?:[eE][+-]?[0-9]+)?", value):
        return float(value)
    if re.fullmatch(r"-?(0|[1-9][0-9]*)[eE][+-]?[0-9]+", value):
        return float(value)
    if value[0] in "{[|>":
        raise _YamlError(f"unsupported scalar {value!r}", line=line)
    return value


def _unquote(value: str, line: int) -> str:
    quote = value[0]
    body = value[1:-1]
    if quote == "'":
        return body.replace("''", "'")
    chars: list[str] = []
    i = 0
    while i < len(body):
        if body[i] == "\\":
            if i + 1 >= len(body):
                raise _YamlError("bad escape", line=line)
            esc = body[i + 1]
            chars.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"'}.get(esc, esc))
            i += 2
            continue
        chars.append(body[i])
        i += 1
    return "".join(chars)


def _end_of_quoted(text: str, start: int, line: int) -> int:
    quote = text[start]
    i = start + 1
    while i < len(text):
        if quote == "'" and text[i] == "'":
            if i + 1 < len(text) and text[i + 1] == "'":
                i += 2
                continue
            return i
        if quote == '"' and text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i
        i += 1
    raise _YamlError("unterminated string", line=line)


def _split_key(text: str, line: int) -> tuple[str, str | None]:
    if text.startswith(("'", '"')):
        end = _end_of_quoted(text, 0, line)
        key = _unquote(text[: end + 1], line)
        rest = text[end + 1 :].lstrip()
        if not rest.startswith(":"):
            raise _YamlError("expected ':' after a key", line=line)
        rest = rest[1:].strip()
        return key, rest or None
    colon = text.find(":")
    if colon <= 0:
        raise _YamlError(f"expected a key, found {text!r}", line=line)
    key = text[:colon].strip()
    if not key:
        raise _YamlError("empty key", line=line)
    rest = text[colon + 1 :].strip()
    return key, rest or None


def _looks_like_key(text: str) -> bool:
    if text.startswith(("'", '"')):
        return ":" in text
    colon = text.find(":")
    if colon <= 0:
        return False
    return colon == len(text) - 1 or text[colon + 1] in " \t"


def _flow_delta(text: str) -> int:
    delta = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "#":
            break
        if ch in "\"'":
            end = _end_of_quoted(text, i, 0)
            i = end + 1
            continue
        if ch in "{[":
            delta += 1
        elif ch in "}]":
            delta -= 1
        i += 1
    return delta


class _Flow:
    """A flow mapping or sequence on one or more lines."""

    def __init__(self, text: str, line: int) -> None:
        self.text = text
        self.line = line
        self.i = 0

    def eof(self) -> bool:
        return self.i >= len(self.text)

    def peek(self) -> str:
        if self.eof():
            return ""
        return self.text[self.i]

    def skip(self) -> None:
        while not self.eof():
            ch = self.peek()
            if ch in " \t\r\n":
                self.i += 1
                continue
            if ch == "#":
                while not self.eof() and self.peek() != "\n":
                    self.i += 1
                continue
            break

    def parse_document(self) -> Any:
        value = self.parse_value()
        self.skip()
        if not self.eof():
            raise _YamlError("trailing flow content", line=self.line)
        return value

    def parse_value(self) -> Any:
        self.skip()
        if self.eof():
            raise _YamlError("missing value", line=self.line)
        ch = self.peek()
        if ch == "{":
            return self.parse_map()
        if ch == "[":
            return self.parse_list()
        if ch in "\"'":
            return self.parse_quoted()
        return self.parse_plain()

    def parse_map(self) -> dict[str, Any]:
        self.i += 1
        result: dict[str, Any] = {}
        while True:
            self.skip()
            if self.peek() == "}":
                self.i += 1
                return result
            if self.peek() in "\"'":
                key = str(self.parse_quoted())
            else:
                key = self.parse_plain_key()
            self.skip()
            if self.peek() != ":":
                raise _YamlError("expected ':' in a flow mapping", line=self.line)
            self.i += 1
            value = self.parse_value()
            if key in result:
                raise _YamlError(f"duplicate key {key!r}", line=self.line)
            result[key] = value
            self.skip()
            if self.peek() == ",":
                self.i += 1
                continue
            if self.peek() == "}":
                self.i += 1
                return result
            raise _YamlError("expected ',' or '}' in a flow mapping", line=self.line)

    def parse_list(self) -> list[Any]:
        self.i += 1
        items: list[Any] = []
        while True:
            self.skip()
            if self.peek() == "]":
                self.i += 1
                return items
            items.append(self.parse_value())
            self.skip()
            if self.peek() == ",":
                self.i += 1
                self.skip()
                if self.peek() == "]":
                    self.i += 1
                    return items
                continue
            if self.peek() == "]":
                self.i += 1
                return items
            raise _YamlError("expected ',' or ']' in a flow sequence", line=self.line)

    def parse_quoted(self) -> str:
        start = self.i
        end = _end_of_quoted(self.text, start, self.line)
        value = _unquote(self.text[start : end + 1], self.line)
        self.i = end + 1
        return value

    def parse_plain(self) -> Any:
        start = self.i
        while not self.eof() and self.peek() not in ",{}[]\n#":
            self.i += 1
        text = self.text[start : self.i].strip()
        if not text:
            raise _YamlError("missing flow scalar", line=self.line)
        return _parse_scalar(text, self.line)

    def parse_plain_key(self) -> str:
        start = self.i
        while not self.eof() and self.peek() not in ",{}[]:\n#":
            self.i += 1
        text = self.text[start : self.i].strip()
        if not text:
            raise _YamlError("missing flow key", line=self.line)
        return text


class _Parser:
    def __init__(self, lines: list[_Line]) -> None:
        self.lines = lines
        self.index = 0

    def parse_document(self) -> Any:
        if not self.lines:
            return None
        if (
            len(self.lines) == 1
            and not _looks_like_key(self.lines[0].text)
            and not self.lines[0].text.startswith("-")
        ):
            return _parse_scalar(self.lines[0].text, self.lines[0].number)
        value = self.parse_block(-1)
        if self.index < len(self.lines):
            raise _YamlError("trailing content", line=self.lines[self.index].number)
        return value

    def parse_block(self, parent_indent: int) -> Any:
        if self.index >= len(self.lines):
            raise _YamlError("expected a value")
        line = self.lines[self.index]
        if line.indent <= parent_indent:
            raise _YamlError("expected an indented value", line=line.number)
        if line.text == "-" or line.text.startswith("- "):
            return self.parse_sequence(line.indent)
        return self.parse_mapping(line.indent)

    def parse_sequence(self, indent: int) -> list[Any]:
        items: list[Any] = []
        while self.index < len(self.lines):
            line = self.lines[self.index]
            if line.indent != indent or not (
                line.text == "-" or line.text.startswith("- ")
            ):
                break
            rest = line.text[1:].strip()
            self.index += 1
            if rest == "":
                if (
                    self.index < len(self.lines)
                    and self.lines[self.index].indent > indent
                ):
                    items.append(self.parse_block(indent))
                else:
                    items.append(None)
            elif rest[0] in "{[":
                items.append(self._finish_flow(rest, line))
            elif rest[0] in "|>":
                raise _YamlError(
                    "block scalars are outside the policy YAML subset", line=line.number
                )
            elif _looks_like_key(rest):
                key_indent = line.indent + line.text.find(rest)
                items.append(
                    self.parse_mapping(key_indent, first_text=rest, first_line=line)
                )
            else:
                items.append(_parse_scalar(rest, line.number))
        if not items:
            raise _YamlError(
                "expected a sequence",
                line=self.lines[min(self.index, len(self.lines) - 1)].number,
            )
        return items

    def parse_mapping(
        self,
        key_indent: int,
        *,
        first_text: str | None = None,
        first_line: _Line | None = None,
    ) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        pending = first_text
        pending_line = first_line
        while True:
            if pending is not None:
                text = pending
                line = pending_line
                pending = None
                number = line.number if line is not None else 0
            else:
                if self.index >= len(self.lines):
                    break
                line = self.lines[self.index]
                if (
                    line.indent != key_indent
                    or line.text == "-"
                    or line.text.startswith("- ")
                ):
                    break
                text = line.text
                number = line.number
                self.index += 1
            key, rest = _split_key(text, number)
            if key in mapping:
                raise _YamlError(f"duplicate key {key!r}", line=number)
            if rest is None:
                if (
                    self.index < len(self.lines)
                    and self.lines[self.index].indent > key_indent
                ):
                    mapping[key] = self.parse_block(key_indent)
                else:
                    mapping[key] = None
            elif rest[0] in "{[":
                assert line is not None
                mapping[key] = self._finish_flow(rest, line)
            elif rest[0] in "|>":
                raise _YamlError(
                    "block scalars are outside the policy YAML subset", line=number
                )
            else:
                mapping[key] = _parse_scalar(rest, number)
        return mapping

    def _finish_flow(self, rest: str, line: _Line) -> Any:
        text = rest
        balance = _flow_delta(rest)
        while balance > 0:
            if self.index >= len(self.lines):
                raise _YamlError("unbalanced flow collection", line=line.number)
            nxt = self.lines[self.index]
            text = f"{text}\n{nxt.text}"
            balance += _flow_delta(nxt.text)
            self.index += 1
        if balance < 0:
            raise _YamlError("unbalanced flow collection", line=line.number)
        return _Flow(text, line.number).parse_document()


def parse_yaml(text: str) -> Any:
    """Parse the policy YAML subset. Empty input is ``None``."""
    return _Parser(_prepare(text)).parse_document()


def _emit_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format(value, ".12g")
    text = str(value)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", text) and text not in {
        "true",
        "false",
        "null",
        "True",
        "False",
        "Null",
        "NULL",
    }:
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def emit_yaml(value: Any, *, indent: int = 0) -> str:
    """Emit ``value`` as policy YAML that :func:`parse_yaml` reads back."""
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines: list[str] = []
        for key, item in value.items():
            rendered = _emit_scalar(key)
            if isinstance(item, dict) and item:
                lines.append(f"{pad}{rendered}:")
                lines.append(emit_yaml(item, indent=indent + 2))
            elif isinstance(item, list) and item:
                lines.append(f"{pad}{rendered}:")
                lines.append(emit_yaml(item, indent=indent + 2))
            else:
                lines.append(f"{pad}{rendered}: {_emit_inline(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for item in value:
            if isinstance(item, dict) and item:
                rendered = emit_yaml(item, indent=indent + 2)
                first, *rest = rendered.splitlines()
                lines.append(f"{pad}- {first.strip()}")
                lines.extend(rest)
            else:
                lines.append(f"{pad}- {_emit_inline(item)}")
        return "\n".join(lines)
    return _emit_inline(value)


def _emit_inline(value: Any) -> str:
    if isinstance(value, dict):
        return "{}" if not value else emit_yaml(value)
    if isinstance(value, list):
        return "[]" if not value else emit_yaml(value)
    return _emit_scalar(value)


# ── Documents ────────────────────────────────────────────────────────────────


def _parse_error(exc: _YamlError, source: str) -> PolicyLanguageError:
    return PolicyLanguageError([Finding("parse", f"{source}: {exc}")])


def _check_id(value: Any) -> str | None:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        return None
    return value


def _parse_datetime(value: Any, field: str) -> tuple[datetime | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, f"{field} must be an ISO-8601 timestamp with a timezone"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None, f"{field} must be an ISO-8601 timestamp with a timezone"
    if parsed.tzinfo is None:
        return None, f"{field} must include a timezone"
    return parsed.astimezone(timezone.utc), None


def _string_list(value: Any, field: str) -> str | None:
    if not isinstance(value, list) or not value:
        return f"{field} must be a non-empty list of strings"
    if any(not isinstance(item, str) or not item for item in value):
        return f"{field} must be a non-empty list of strings"
    return None


def _optional_string_list(value: Any, field: str) -> str | None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return f"{field} must be a list of strings"
    return None


def _string_map(value: Any, field: str) -> str | None:
    if not isinstance(value, dict) or not value:
        return f"{field} must map each attribute to a non-empty list of strings"
    for key, item in value.items():
        if not isinstance(key, str):
            return f"{field} keys must be strings"
        message = _string_list(item, f"{field}.{key}")
        if message:
            return message
    return None


def _payload_messages(rule_type: str, rule: Any) -> list[str]:
    if rule_type not in CORE_RULES:
        return [f"unknown rule_type {rule_type!r}"]
    if not isinstance(rule, dict):
        return ["rule must be a mapping"]
    spec = _SCHEMAS.get(rule_type)
    if spec is None:
        return [f"unknown rule_type {rule_type!r}"]
    allowed = set(spec.get("required_maps", ()))
    allowed.update(spec.get("required_lists", ()))
    allowed.update(spec.get("optional_lists", ()))
    allowed.update(spec.get("optional", ()))
    messages: list[str] = []
    unknown = sorted(set(rule) - allowed)
    if unknown:
        messages.append(f"unknown rule field {unknown[0]!r}")
    for field in spec.get("required_maps", ()):
        message = _string_map(rule.get(field), field)
        if message:
            messages.append(message)
    for field in spec.get("required_lists", ()):
        message = _string_list(rule.get(field), field)
        if message:
            messages.append(message)
    for field in spec.get("optional_lists", ()):
        if field in rule:
            message = _optional_string_list(rule.get(field), field)
            if message:
                messages.append(message)
    if "on_violation" in rule and rule["on_violation"] not in _ON_VIOLATION:
        messages.append("on_violation must be fail, escalate, or requires_human")
    if "description" in rule and not isinstance(rule["description"], str):
        messages.append("description must be a string")
    if "min_confidence" in rule and (
        isinstance(rule["min_confidence"], bool)
        or not isinstance(rule["min_confidence"], (int, float))
    ):
        messages.append("min_confidence must be a number")
    if rule_type == "CONTACT_FREQUENCY":
        messages.extend(_frequency_messages(rule))
    if rule_type == "PRINCIPAL_EQUALITY":
        messages.extend(_principal_messages(rule))
    if rule_type == "CONTENT_POLICY":
        messages.extend(_content_policy_messages(rule))
    if rule_type == "AI_DISCLOSURE":
        messages.extend(_disclosure_messages(rule))
    if rule_type == "CIRCUIT_BREAKER":
        messages.extend(_breaker_messages(rule))
    return messages


def _breaker_messages(rule: dict[str, Any]) -> list[str]:
    """A breaker that does not name one dimension and value does not validate."""
    messages: list[str] = []
    if rule.get("dimension") not in _BREAKER_DIMENSIONS:
        messages.append("dimension must be usecase, model, prompt, or policy")
    value = rule.get("value")
    if not isinstance(value, str) or not value.strip():
        messages.append("value must be a non-empty string")
    owner = rule.get("accountable_owner")
    if owner is not None and (not isinstance(owner, str) or not owner.strip()):
        messages.append("accountable_owner must be a non-empty string")
    return messages


def _principal_messages(rule: dict[str, Any]) -> list[str]:
    """Principal equality names the one attribute the principal must match."""
    equals = rule.get("equals")
    if "equals" not in rule:
        return ["equals is required"]
    if not isinstance(equals, str) or not equals.strip():
        return ["equals must be a non-empty string"]
    return []


def _frequency_messages(rule: dict[str, Any]) -> list[str]:
    """A contact-frequency rule that cannot be counted does not validate."""
    messages: list[str] = []
    limit = rule.get("max_contacts")
    window = rule.get("window_days")
    if "max_contacts" not in rule:
        messages.append("max_contacts is required")
    elif isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        messages.append("max_contacts must be a non-negative integer")
    if "window_days" not in rule:
        messages.append("window_days is required")
    elif isinstance(window, bool) or not isinstance(window, int) or window < 1:
        messages.append("window_days must be a positive integer")
    return messages


def _content_policy_messages(rule: dict[str, Any]) -> list[str]:
    """A content policy that checks nothing, or names an unknown level, does not validate."""
    messages: list[str] = []
    substantive = (
        "sources",
        "required_disclosures",
        "sensitive",
        "prohibited_claims",
        "approved_tones",
        "permitted_jurisdictions",
        "permitted_channels",
        "require_risk_language",
        "require_research_attribution",
        "validator_audiences",
    )
    if not any(key in rule for key in substantive):
        messages.append("a content policy names at least one constraint")
    if "sensitive" in rule:
        from kognita.sensitive import _categories
        try:
            _categories(rule["sensitive"])
        except Exception:
            messages.append("sensitive must list known special categories")
    levels = rule.get("levels")
    if isinstance(levels, list):
        unknown = [item for item in levels if item not in COMMUNICATION_LEVELS]
        if unknown:
            messages.append(f"unknown communication level {unknown[0]!r}")
    lanes = rule.get("lanes")
    if isinstance(lanes, list):
        unknown = [item for item in lanes if item not in ("client", "public")]
        if unknown:
            messages.append(f"unknown lane {unknown[0]!r}")
    for field in ("require_risk_language", "require_research_attribution"):
        if field in rule and not isinstance(rule[field], bool):
            messages.append(f"{field} must be a boolean")
    for field in ("approved_tones", "permitted_jurisdictions", "permitted_channels"):
        value = rule.get(field)
        if isinstance(value, list) and not value:
            messages.append(f"{field} must name at least one value")
    messages.extend(_validator_audience_messages(rule))
    return messages


def _validator_audience_messages(rule: dict[str, Any]) -> list[str]:
    """Status audiences name a known status and only rm or client."""
    if "validator_audiences" not in rule:
        return []
    mapping = rule.get("validator_audiences")
    if not isinstance(mapping, dict) or not mapping:
        return ["validator_audiences must map each status to a list of audiences"]
    from kognita.claims import ValidatorStatus
    known = {item.value for item in ValidatorStatus}
    messages: list[str] = []
    for key, value in mapping.items():
        if key not in known:
            messages.append(f"unknown validator status {key!r}")
        if (
            not isinstance(value, list)
            or any(
                not isinstance(item, str) or item.strip().casefold() not in {"rm", "client"}
                for item in value
            )
        ):
            messages.append("validator_audiences values must list rm, client, or both")
    return messages


def _disclosure_messages(rule: dict[str, Any]) -> list[str]:
    """A disclosure rule needs the text the client must see, and known triggers."""
    messages: list[str] = []
    text = rule.get("text")
    if not isinstance(text, str) or not text.strip():
        messages.append("text must be a non-empty string")
    triggers = rule.get("triggers")
    if isinstance(triggers, list):
        unknown = [item for item in triggers if item not in DISCLOSURE_TRIGGERS]
        if unknown:
            messages.append(f"unknown disclosure trigger {unknown[0]!r}")
    return messages


def _findings_for(messages: Iterable[str], rule_id: str | None) -> list[Finding]:
    return [Finding("type", message, rule_id) for message in messages]


def parse_rule(data: Any, *, source: str) -> tuple[RuleDocument | None, list[Finding]]:
    """Type-check one rule mapping. A failure returns findings and no rule."""
    if not isinstance(data, dict):
        return None, [Finding("type", f"{source}: a rule must be a mapping")]
    unknown = sorted(set(data) - _RULE_FIELDS)
    rule_id = data.get("id")
    ident = _check_id(rule_id)
    findings: list[Finding] = []
    if ident is None:
        findings.append(
            Finding(
                "type",
                f"{source}: id must be a string of letters, digits, '.', '_' or '-'",
            )
        )
    if unknown:
        findings.append(Finding("type", f"unknown field {unknown[0]!r}", ident))
    regime = data.get("regime")
    if not isinstance(regime, str) or not regime.strip():
        findings.append(Finding("type", "regime must be a non-empty string", ident))
    rule_type = data.get("rule_type")
    if not isinstance(rule_type, str) or not rule_type.strip():
        findings.append(Finding("type", "rule_type must be a non-empty string", ident))
        payload_messages: list[str] = []
    else:
        payload_messages = _payload_messages(rule_type, data.get("rule"))
    if any("reasoningtrace" in re.sub(r"[^a-z0-9]", "", item.lower())
           for item in _walk_strings(data.get("rule"))):
        findings.append(Finding("reasoning_trace", "rules may not reference a reasoning trace", ident))
    findings.extend(_findings_for(payload_messages, ident))
    if "citation" not in data or not isinstance(data.get("citation"), str):
        findings.append(Finding("type", "citation must be a string", ident))
    effective_from, date_error = _parse_datetime(
        data.get("effective_from"), "effective_from"
    )
    if date_error:
        findings.append(Finding("type", date_error, ident))
    effective_to: datetime | None = None
    if "effective_to" in data and data.get("effective_to") is not None:
        effective_to, end_error = _parse_datetime(
            data.get("effective_to"), "effective_to"
        )
        if end_error:
            findings.append(Finding("type", end_error, ident))
    applies_to = data.get("applies_to")
    if applies_to is not None and not isinstance(applies_to, str):
        findings.append(Finding("type", "applies_to must be a string or null", ident))
    if (
        findings
        or ident is None
        or effective_from is None
        or not isinstance(regime, str)
    ):
        return None, findings
    if not isinstance(rule_type, str) or not isinstance(data.get("rule"), dict):
        return None, findings
    return (
        RuleDocument(
            id=ident,
            regime=regime,
            rule_type=rule_type,
            citation=str(data.get("citation")),
            effective_from=effective_from,
            rule=dict(data["rule"]),
            applies_to=applies_to if isinstance(applies_to, str) else None,
            effective_to=effective_to,
            source=source,
        ),
        [],
    )


def _scenario_from(
    data: Any, *, source: str, rule_id: str
) -> tuple[Scenario | None, list[Finding]]:
    if not isinstance(data, dict):
        return None, [
            Finding("type", f"{source}: a scenario must be a mapping", rule_id)
        ]
    unknown = sorted(set(data) - {"name", "given", "when", "then"})
    findings: list[Finding] = []
    if unknown:
        findings.append(
            Finding("type", f"unknown scenario field {unknown[0]!r}", rule_id)
        )
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        findings.append(
            Finding("type", "scenario name must be a non-empty string", rule_id)
        )
    given = data.get("given")
    when = data.get("when")
    then = data.get("then")
    if not isinstance(given, dict):
        findings.append(Finding("type", "scenario given must be a mapping", rule_id))
        given = {}
    if not isinstance(when, dict):
        findings.append(Finding("type", "scenario when must be a mapping", rule_id))
        when = {}
    if not isinstance(then, dict):
        findings.append(Finding("type", "scenario then must be a mapping", rule_id))
        then = {}
    findings.extend(_mapping_fields(given, _GIVEN_FIELDS, "given", rule_id))
    findings.extend(_mapping_fields(when, _WHEN_FIELDS, "when", rule_id))
    findings.extend(_mapping_fields(then, _THEN_FIELDS, "then", rule_id))
    if "tool" not in when and "tool" not in given:
        findings.append(Finding("type", "scenario when.tool is required", rule_id))
    if isinstance(given, dict) and "principal" not in given:
        findings.append(
            Finding("type", "scenario given.principal is required", rule_id)
        )
    decision = then.get("decision") if isinstance(then, dict) else None
    if decision not in _OUTCOMES:
        findings.append(
            Finding("type", "scenario then.decision must be an outcome", rule_id)
        )
    cites = then.get("cites") if isinstance(then, dict) else None
    if cites is not None and not isinstance(cites, str):
        findings.append(
            Finding("type", "scenario then.cites must be a string", rule_id)
        )
    if "arguments" in given and "arguments" in when:
        findings.append(
            Finding(
                "type", "scenario arguments belong in given or when, not both", rule_id
            )
        )
    findings.extend(_scenario_value_findings(given, when, rule_id))
    if findings or not isinstance(name, str):
        return None, findings
    return Scenario(name=name, given=dict(given), when=dict(when), then=dict(then)), []


def _scenario_value_findings(given: Any, when: Any, rule_id: str) -> list[Finding]:
    """Reject values that would be coerced into a different envelope."""
    findings: list[Finding] = []
    if not isinstance(given, dict) or not isinstance(when, dict):
        return findings
    for field in (
        "principal",
        "purpose",
        "tool",
        "actor_location",
        "subject_type",
        "agent_name",
    ):
        if field in given and not isinstance(given[field], str):
            findings.append(Finding("type", f"given.{field} must be a string", rule_id))
    if "subject_id" in given and not isinstance(given["subject_id"], str):
        findings.append(Finding("type", "given.subject_id must be a string", rule_id))
    if "is_admin" in given and not isinstance(given["is_admin"], bool):
        findings.append(Finding("type", "given.is_admin must be a boolean", rule_id))
    for field in ("attributes", "arguments", "subjects", "scopes", "context"):
        if field in given and not isinstance(given[field], dict):
            findings.append(
                Finding("type", f"given.{field} must be a mapping", rule_id)
            )
    if "roles" in given and (
        not isinstance(given["roles"], list)
        or any(not isinstance(item, str) for item in given["roles"])
    ):
        findings.append(
            Finding("type", "given.roles must be a list of strings", rule_id)
        )
    if "tool" in when and not isinstance(when["tool"], str):
        findings.append(Finding("type", "when.tool must be a string", rule_id))
    if "arguments" in when and not isinstance(when["arguments"], dict):
        findings.append(Finding("type", "when.arguments must be a mapping", rule_id))
    return findings


def _mapping_fields(
    value: Any, allowed: set[str], label: str, rule_id: str
) -> list[Finding]:
    if not isinstance(value, dict):
        return []
    unknown = sorted(set(value) - allowed)
    if not unknown:
        return []
    return [Finding("type", f"unknown {label} field {unknown[0]!r}", rule_id)]


def parse_scenario_document(
    data: Any, *, source: str
) -> tuple[ScenarioDocument | None, list[Finding]]:
    """Type-check one ``rule`` / ``scenarios`` document."""
    if not isinstance(data, dict):
        return None, [Finding("type", f"{source}: a scenario file must be a mapping")]
    unknown = sorted(set(data) - {"rule", "scenarios"})
    findings: list[Finding] = []
    if unknown:
        findings.append(Finding("type", f"{source}: unknown field {unknown[0]!r}"))
    rule_id = _check_id(data.get("rule"))
    if rule_id is None:
        findings.append(Finding("type", f"{source}: rule must name a rule id"))
        label = None
    else:
        label = rule_id
    raw = data.get("scenarios")
    if not isinstance(raw, list):
        findings.append(Finding("type", "scenarios must be a list", label))
        return None, findings
    scenarios: list[Scenario] = []
    for item in raw:
        scenario, item_findings = _scenario_from(
            item, source=source, rule_id=label or ""
        )
        findings.extend(item_findings)
        if scenario is not None:
            scenarios.append(scenario)
    if findings or label is None:
        return None, findings
    return ScenarioDocument(rule=label, scenarios=tuple(scenarios), source=source), []


def _is_scenario_data(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and "scenarios" in data
        and "rule" in data
        and "rule_type" not in data
    )


def _is_change_data(data: Any) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    keys = set(data)
    return keys <= {"ADDED", "MODIFIED", "REMOVED"} and bool(
        keys & {"ADDED", "MODIFIED", "REMOVED"}
    )


def _is_change_text(text: str) -> bool:
    return any(_HEADING_RE.match(line) for line in text.splitlines())


def _freshness_data(data: Any) -> tuple[Any, tuple[FreshnessRule, ...]]:
    if not isinstance(data, dict) or "freshness" not in data:
        return data, ()
    if set(data) - {"freshness", "rules"} or not isinstance(data["freshness"], dict):
        raise PolicyLanguageError([Finding("type", "freshness requires a mapping and optional rules")])
    rules = []
    for name, threshold in data["freshness"].items():
        if isinstance(threshold, dict) and set(threshold) == {"max_age"}:
            threshold = threshold["max_age"]
        # Numeric thresholds are seconds; strings accept s, m, h, d suffixes.
        try:
            if isinstance(threshold, bool):
                raise ValueError()
            if isinstance(threshold, str):
                match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([smhd])", threshold)
                if match is None:
                    raise ValueError()
                seconds = float(match[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]
            elif isinstance(threshold, (int, float)):
                seconds = threshold
            else:
                raise ValueError()
            rules.append(FreshnessRule(name, max_age=timedelta(seconds=seconds)))
        except (ValueError, TypeError, OverflowError, AttributeError) as exc:
            raise PolicyLanguageError([Finding("type", f"invalid freshness threshold for {name!r}")]) from exc
    return data.get("rules", []), tuple(rules)


def _rules_from_data(
    data: Any, source: str
) -> tuple[list[RuleDocument], list[Finding]]:
    items = data if isinstance(data, list) else [data]
    rules: list[RuleDocument] = []
    findings: list[Finding] = []
    for item in items:
        rule, item_findings = parse_rule(item, source=source)
        findings.extend(item_findings)
        if rule is not None:
            rules.append(rule)
    return rules, findings


def _scenarios_from_data(
    data: Any, source: str
) -> tuple[list[ScenarioDocument], list[Finding]]:
    items = data if isinstance(data, list) else [data]
    documents: list[ScenarioDocument] = []
    findings: list[Finding] = []
    for item in items:
        document, item_findings = parse_scenario_document(item, source=source)
        findings.extend(item_findings)
        if document is not None:
            documents.append(document)
    return documents, findings


def _raise_if(findings: list[Finding]) -> None:
    if findings:
        raise PolicyLanguageError(findings)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyLanguageError([Finding("parse", f"{path}: {exc}")]) from exc


def _scenario_paths(directory: Path) -> list[Path]:
    paths: list[Path] = []
    if not directory.is_dir():
        return paths
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith((".scenarios.yaml", ".scenarios.yml")):
            paths.append(path)
    return paths


def _policy_paths(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix not in {".yaml", ".yml"}:
            continue
        if path.name.endswith((".scenarios.yaml", ".scenarios.yml")):
            continue
        paths.append(path)
    return paths


def _read_dir(directory: Path) -> PolicySet:
    if not directory.is_dir():
        raise PolicyLanguageError([Finding("parse", f"{directory} is not a directory")])
    rules: list[RuleDocument] = []
    scenarios: list[ScenarioDocument] = []
    findings: list[Finding] = []
    freshness: dict[str, FreshnessRule] = {}
    for path in _policy_paths(directory):
        text = _read_text(path)
        if _is_change_text(text):
            findings.append(
                Finding("parse", f"{path} is a change file, not a policy file")
            )
            continue
        try:
            data = parse_yaml(text)
        except _YamlError as exc:
            findings.append(Finding("parse", f"{path}: {exc}"))
            continue
        if _is_change_data(data):
            findings.append(
                Finding("parse", f"{path} is a change file, not a policy file")
            )
            continue
        if _is_scenario_data(data) or (
            isinstance(data, list) and data and _is_scenario_data(data[0])
        ):
            found, item_findings = _scenarios_from_data(data, str(path))
            findings.extend(item_findings)
            scenarios.extend(found)
            continue
        data, fresh = _freshness_data(data)
        for item in fresh:
            if item.data_type in freshness:
                raise PolicyLanguageError([Finding("type", f"duplicate freshness rule {item.data_type!r}")])
            freshness[item.data_type] = item
        found_rules, item_findings = _rules_from_data(data, str(path))
        findings.extend(item_findings)
        rules.extend(found_rules)
    for path in _scenario_paths(directory):
        if any(item.source == str(path) for item in scenarios):
            continue
        try:
            data = parse_yaml(_read_text(path))
        except _YamlError as exc:
            findings.append(Finding("parse", f"{path}: {exc}"))
            continue
        found, item_findings = _scenarios_from_data(data, str(path))
        findings.extend(item_findings)
        scenarios.extend(found)
    _raise_if(findings)
    _raise_if(_duplicate_ids(rules))
    return PolicySet(
        rules=tuple(rules), scenarios=tuple(scenarios), source=str(directory),
        freshness=tuple(freshness.values())
    )


def _duplicate_ids(rules: list[RuleDocument]) -> list[Finding]:
    seen: set[str] = set()
    findings: list[Finding] = []
    for rule in rules:
        if rule.id in seen:
            findings.append(Finding("type", "duplicate rule id", rule.id))
        seen.add(rule.id)
    return findings


def _scenarios_for(
    rules: list[RuleDocument], directory: Path
) -> list[ScenarioDocument]:
    ids = {rule.id for rule in rules}
    kept: list[ScenarioDocument] = []
    findings: list[Finding] = []
    for path in _scenario_paths(directory):
        try:
            data = parse_yaml(_read_text(path))
        except _YamlError as exc:
            findings.append(Finding("parse", f"{path}: {exc}"))
            continue
        documents, item_findings = _scenarios_from_data(data, str(path))
        findings.extend(item_findings)
        for document in documents:
            if document.rule in ids:
                kept.append(document)
    _raise_if(findings)
    return kept


def read_policy_set(path: str | Path) -> PolicySet:
    """Read a policy file or a directory of policy and scenario files."""
    target = Path(path)
    if target.is_dir():
        return _read_dir(target)
    if not target.is_file():
        raise PolicyLanguageError([Finding("parse", f"{target} does not exist")])
    text = _read_text(target)
    if target.name.endswith((".scenarios.yaml", ".scenarios.yml")):
        return _read_dir(target.parent)
    if _is_change_text(text):
        raise PolicyLanguageError(
            [Finding("parse", f"{target} is a change file; pass it with --policy")]
        )
    try:
        data = parse_yaml(text)
    except _YamlError as exc:
        raise _parse_error(exc, str(target)) from exc
    if _is_scenario_data(data):
        return _read_dir(target.parent)
    if _is_change_data(data):
        raise PolicyLanguageError(
            [Finding("parse", f"{target} is a change file; pass it with --policy")]
        )
    data, freshness = _freshness_data(data)
    rules, findings = _rules_from_data(data, str(target))
    _raise_if(findings)
    _raise_if(_duplicate_ids(rules))
    scenarios = _scenarios_for(rules, target.parent)
    return PolicySet(rules=tuple(rules), scenarios=tuple(scenarios), source=str(target), freshness=freshness)


def source_hash(path: str | Path) -> str:
    """SHA-256 of the file a load read, or of a directory's policy files.

    A directory hash is the canonical hash of each policy file's name and
    SHA-256, sorted by name. Scenario files are not included: they do not
    change stored rows.
    """
    target = Path(path)
    if target.is_file():
        return hashlib.sha256(target.read_bytes()).hexdigest()
    if not target.is_dir():
        raise PolicyLanguageError([Finding("parse", f"{target} does not exist")])
    manifest = [
        {"path": child.name, "sha256": hashlib.sha256(child.read_bytes()).hexdigest()}
        for child in _policy_paths(target)
    ]
    return canonical_hash(manifest)


def _strip_fence(body: str) -> str:
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]
    return "\n".join(lines)


def _section_rules(body: str, source: str) -> tuple[list[Any], list[Finding]]:
    text = _strip_fence(body)
    if not text.strip():
        return [], []
    try:
        data = parse_yaml(text)
    except _YamlError as exc:
        return [], [Finding("parse", f"{source}: {exc}")]
    if data is None:
        return [], []
    items = data if isinstance(data, list) else [data]
    return items, []


def _change_from_sections(
    sections: dict[str, str], *, source: str, text: str
) -> PolicyChange:
    findings: list[Finding] = []
    added: list[RuleDocument] = []
    modified: list[RuleDocument] = []
    removed: list[str] = []
    for label, bucket in (("ADDED", added), ("MODIFIED", modified)):
        items, parse_findings = _section_rules(sections.get(label, ""), source)
        findings.extend(parse_findings)
        for item in items:
            rule, item_findings = parse_rule(item, source=source)
            findings.extend(item_findings)
            if rule is not None:
                bucket.append(rule)
    removed_items, parse_findings = _section_rules(sections.get("REMOVED", ""), source)
    findings.extend(parse_findings)
    for item in removed_items:
        if isinstance(item, str):
            ident = _check_id(item)
            if ident is None:
                findings.append(
                    Finding("type", f"{source}: REMOVED id is not a rule id")
                )
            else:
                removed.append(ident)
            continue
        if isinstance(item, dict) and set(item) == {"id"}:
            ident = _check_id(item.get("id"))
            if ident is None:
                findings.append(
                    Finding("type", f"{source}: REMOVED id is not a rule id")
                )
            else:
                removed.append(ident)
            continue
        findings.append(
            Finding("type", f"{source}: REMOVED entries name an id and nothing else")
        )
    seen: dict[str, str] = {}
    for rule in added:
        if rule.id in seen:
            findings.append(
                Finding("type", f"id appears in {seen[rule.id]} and ADDED", rule.id)
            )
        seen[rule.id] = "ADDED"
    for rule in modified:
        if rule.id in seen:
            findings.append(
                Finding("type", f"id appears in {seen[rule.id]} and MODIFIED", rule.id)
            )
        seen[rule.id] = "MODIFIED"
    for rule_id in removed:
        if rule_id in seen:
            findings.append(
                Finding("type", f"id appears in {seen[rule_id]} and REMOVED", rule_id)
            )
        seen[rule_id] = "REMOVED"
    _raise_if(findings)
    return PolicyChange(
        added=tuple(added),
        modified=tuple(modified),
        removed=tuple(removed),
        source=source,
        text=text,
    )


def parse_change(text: str, *, source: str = "") -> PolicyChange:
    """Parse a markdown change or a YAML change mapping."""
    if _is_change_text(text):
        sections: dict[str, list[str]] = {}
        current: str | None = None
        for line in text.splitlines():
            match = _HEADING_RE.match(line)
            if match:
                name = match.group(1)
                if name in sections:
                    raise PolicyLanguageError(
                        [Finding("parse", f"{source}: duplicate ## {name} section")]
                    )
                sections[name] = []
                current = name
                continue
            if current is not None:
                sections[current].append(line)
        if not sections:
            raise PolicyLanguageError(
                [Finding("parse", f"{source}: change has no sections")]
            )
        return _change_from_sections(
            {key: "\n".join(value) for key, value in sections.items()},
            source=source,
            text=text,
        )
    try:
        data = parse_yaml(text)
    except _YamlError as exc:
        raise _parse_error(exc, source or "change") from exc
    if not _is_change_data(data):
        raise PolicyLanguageError(
            [
                Finding(
                    "parse",
                    f"{source or 'change'}: not an ADDED/MODIFIED/REMOVED change",
                )
            ]
        )
    synthetic = {
        "ADDED": emit_yaml(data["ADDED"]) if data.get("ADDED") else "",
        "MODIFIED": emit_yaml(data["MODIFIED"]) if data.get("MODIFIED") else "",
        "REMOVED": emit_yaml(data["REMOVED"]) if data.get("REMOVED") else "",
    }
    return _change_from_sections(synthetic, source=source, text=text)


def read_change(path: str | Path) -> PolicyChange:
    """Read a change file from disk."""
    target = Path(path)
    return parse_change(_read_text(target), source=str(target))


def is_change_path(path: str | Path) -> bool:
    """Whether ``path`` is a change document rather than a policy set."""
    target = Path(path)
    if not target.is_file():
        return False
    text = _read_text(target)
    if _is_change_text(text):
        return True
    try:
        data = parse_yaml(text)
    except _YamlError:
        return False
    return _is_change_data(data)


# ── Validate, diff, apply ────────────────────────────────────────────────────


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _marker_findings(value: Any, rule_id: str | None) -> list[Finding]:
    findings: list[Finding] = []
    for text in _walk_strings(value):
        matches = list(CLARIFICATION_RE.finditer(text))
        if _CLARIFICATION_OPEN in text and not matches:
            findings.append(
                Finding(
                    "clarification",
                    "unresolved clarification marker",
                    rule_id,
                    strict=True,
                )
            )
        for match in matches:
            question = match.group(1).strip() or "unspecified question"
            findings.append(
                Finding(
                    "clarification",
                    f"unresolved clarification: {question}",
                    rule_id,
                    strict=True,
                )
            )
    return findings


def _rule_markers(rule: RuleDocument) -> list[Finding]:
    payload = {
        "id": rule.id,
        "regime": rule.regime,
        "citation": rule.citation,
        "applies_to": rule.applies_to,
        "rule": rule.rule,
    }
    return _marker_findings(payload, rule.id)


def strict_findings(policies: PolicySet) -> list[Finding]:
    """Uncited rules, untested rules, dangling references, and clarification markers."""
    findings: list[Finding] = []
    known = {rule.id for rule in policies.rules}
    tested = {document.rule for document in policies.scenarios if document.scenarios}
    for rule in policies.rules:
        if not rule.citation.strip():
            findings.append(
                Finding("uncited", "citation is empty", rule.id, strict=True)
            )
        if rule.id not in tested:
            findings.append(
                Finding("untested", "no colocated scenario", rule.id, strict=True)
            )
        findings.extend(_rule_markers(rule))
    for document in policies.scenarios:
        if document.rule not in known:
            findings.append(
                Finding(
                    "dangling",
                    f"scenario names unknown rule {document.rule}",
                    document.rule,
                    strict=True,
                )
            )
        for scenario in document.scenarios:
            cites = scenario.then.get("cites")
            if isinstance(cites, str) and cites not in known:
                findings.append(
                    Finding(
                        "dangling",
                        f"scenario {scenario.name!r} cites unknown rule {cites}",
                        document.rule,
                        strict=True,
                    )
                )
            findings.extend(_marker_findings(scenario, document.rule))
    return findings


def referential_findings(current: PolicySet, change: PolicyChange) -> list[Finding]:
    """MODIFIED and REMOVED ids must already exist. ADDED ids must not."""
    known = set(current.by_id())
    findings: list[Finding] = []
    for rule in change.added:
        if rule.id in known:
            findings.append(
                Finding(
                    "referential",
                    "ADDED rule already exists in the current policy set",
                    rule.id,
                )
            )
    for rule in change.modified:
        if rule.id not in known:
            findings.append(
                Finding(
                    "referential",
                    "MODIFIED rule does not match a rule in the current policy set",
                    rule.id,
                )
            )
    for rule_id in change.removed:
        if rule_id not in known:
            findings.append(
                Finding(
                    "referential",
                    "REMOVED rule does not match a rule in the current policy set",
                    rule_id,
                )
            )
    return findings


def validate(policies: PolicySet, change: PolicyChange | None = None) -> list[Finding]:
    """Type, referential, and strict findings for a policy set or a change.

    The caller decides which findings block. :func:`blocking_findings` drops the
    strict ones unless ``--strict`` is set. Referential failures always block.
    """
    findings: list[Finding] = []
    if change is None:
        findings.extend(strict_findings(policies))
        return _dedupe(findings)
    findings.extend(referential_findings(policies, change))
    if any(item.code == "referential" for item in findings):
        findings.extend(strict_findings(policies))
        findings.extend(_change_markers(change))
        return _dedupe(findings)
    merged = apply_change(policies, change)
    findings.extend(strict_findings(merged))
    findings.extend(_change_markers(change))
    return _dedupe(findings)


def _change_markers(change: PolicyChange) -> list[Finding]:
    findings: list[Finding] = []
    for rule in (*change.added, *change.modified):
        findings.extend(_rule_markers(rule))
    for rule_id in change.removed:
        findings.extend(_marker_findings(rule_id, rule_id))
    return findings


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[Finding] = []
    for finding in findings:
        key = (finding.code, finding.message, finding.rule_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def _body(rule: RuleDocument) -> tuple[Any, ...]:
    return (
        rule.id,
        rule.regime,
        rule.rule_type,
        rule.applies_to,
        rule.citation,
        canonical_json(rule.rule),
    )


def _identity(rule: RuleDocument) -> tuple[Any, ...]:
    return _body(rule) + (
        _format_dt(rule.effective_from),
        _format_dt(rule.effective_to),
    )


def _format_dt(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def diff_sets(before: PolicySet, after: PolicySet) -> PolicyChange:
    """The delta that turns ``before`` into ``after``."""
    previous = before.by_id()
    current = after.by_id()
    added = tuple(current[key] for key in sorted(set(current) - set(previous)))
    removed = tuple(sorted(set(previous) - set(current)))
    modified = tuple(
        current[key]
        for key in sorted(set(current) & set(previous))
        if _identity(previous[key]) != _identity(current[key])
    )
    return PolicyChange(added=added, modified=modified, removed=removed, source="diff")


def rule_to_data(rule: RuleDocument) -> dict[str, Any]:
    """The author-facing mapping for one rule."""
    data: dict[str, Any] = {
        "id": rule.id,
        "regime": rule.regime,
        "rule_type": rule.rule_type,
        "citation": rule.citation,
        "effective_from": _format_dt(rule.effective_from),
        "rule": rule.rule,
    }
    if rule.applies_to is not None:
        data["applies_to"] = rule.applies_to
    if rule.effective_to is not None:
        data["effective_to"] = _format_dt(rule.effective_to)
    return data


def render_change(change: PolicyChange) -> str:
    """Render a change as ``## ADDED``, ``## MODIFIED`` and ``## REMOVED``."""
    parts = [
        "## ADDED",
        _render_rule_list(change.added),
        "",
        "## MODIFIED",
        _render_rule_list(change.modified),
        "",
        "## REMOVED",
        _render_removed(change.removed),
        "",
    ]
    return "\n".join(parts).rstrip() + "\n"


def _render_rule_list(rules: tuple[RuleDocument, ...]) -> str:
    if not rules:
        return ""
    return emit_yaml([rule_to_data(rule) for rule in rules])


def _render_removed(ids: tuple[str, ...]) -> str:
    if not ids:
        return ""
    return emit_yaml([{"id": rule_id} for rule_id in ids])


def _ops(change: PolicyChange) -> dict[str, tuple[str, RuleDocument | None]]:
    ops: dict[str, tuple[str, RuleDocument | None]] = {}
    for rule in change.added:
        ops[rule.id] = ("ADDED", rule)
    for rule in change.modified:
        ops[rule.id] = ("MODIFIED", rule)
    for rule_id in change.removed:
        ops[rule_id] = ("REMOVED", None)
    return ops


def _same_change(
    left: tuple[str, RuleDocument | None], right: tuple[str, RuleDocument | None]
) -> bool:
    left_op, left_rule = left
    right_op, right_rule = right
    if left_op != right_op:
        return False
    if left_op == "REMOVED":
        return True
    if left_rule is None or right_rule is None:
        return False
    return _identity(left_rule) == _identity(right_rule)


def merge_changes(left: PolicyChange, right: PolicyChange) -> PolicyChange:
    """Combine two deltas against the same base.

    Disjoint edits combine. The same id with a different operation, or the same
    operation and a different rule, is an error: those two changes conflict.
    """
    merged = _ops(left)
    findings: list[Finding] = []
    for rule_id, (op, rule) in _ops(right).items():
        if rule_id not in merged:
            merged[rule_id] = (op, rule)
            continue
        if _same_change(merged[rule_id], (op, rule)):
            continue
        current_op = merged[rule_id][0]
        findings.append(
            Finding(
                "conflict",
                f"{current_op} and {op} both touch this rule",
                rule_id,
            )
        )
    if findings:
        raise PolicyLanguageError(findings)
    added = tuple(
        rule
        for rule_id, (op, rule) in sorted(merged.items())
        if op == "ADDED" and rule is not None
    )
    modified = tuple(
        rule
        for rule_id, (op, rule) in sorted(merged.items())
        if op == "MODIFIED" and rule is not None
    )
    removed = tuple(
        rule_id for rule_id, (op, _) in sorted(merged.items()) if op == "REMOVED"
    )
    return PolicyChange(added=added, modified=modified, removed=removed, source="merge")


def apply_change(
    current: PolicySet,
    change: PolicyChange,
    *,
    directory: Path | None = None,
    when: datetime | None = None,
) -> PolicySet:
    """Merge ``change`` into ``current``.

    With ``directory``, the merged rules are written back into that policy
    directory and the change is copied under ``archive/`` with ``when``'s date.
    """
    findings = referential_findings(current, change)
    if findings:
        raise PolicyLanguageError(findings)
    rules = current.by_id()
    for rule in change.added:
        rules[rule.id] = replace(rule, source="")
    for rule in change.modified:
        previous = rules[rule.id]
        rules[rule.id] = replace(rule, source=previous.source)
    for rule_id in change.removed:
        del rules[rule_id]
    ordered: list[RuleDocument] = []
    for rule in current.rules:
        if rule.id in rules and rule.id not in {item.id for item in change.added}:
            ordered.append(rules[rule.id])
    for rule in change.added:
        ordered.append(rules[rule.id])
    merged = PolicySet(
        rules=tuple(ordered),
        scenarios=current.scenarios,
        source=current.source,
        freshness=current.freshness,
    )
    if directory is not None:
        moment = when or utcnow()
        _write_policy_files(directory, current, merged)
        archive_change(change, directory / "archive", when=moment)
    return merged


def archive_change(change: PolicyChange, directory: Path, *, when: datetime) -> Path:
    """Copy a change into ``directory`` under a name that starts with the date."""
    directory.mkdir(parents=True, exist_ok=True)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    stamp = when.astimezone(timezone.utc).strftime("%Y-%m-%d")
    base = Path(change.source).name if change.source else "change.md"
    if base in {"", ".", ".."}:
        base = "change.md"
    stem = Path(base).stem
    suffix = Path(base).suffix
    dest = directory / f"{stamp}-{base}"
    counter = 2
    while dest.exists():
        dest = directory / f"{stamp}-{stem}-{counter}{suffix}"
        counter += 1
    dest.write_text(change.text or render_change(change), encoding="utf-8")
    return dest


def _write_policy_files(directory: Path, before: PolicySet, after: PolicySet) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    before_ids = before.by_id()
    after_ids = after.by_id()
    grouped: dict[str, list[str]] = {}
    for rule in before.rules:
        grouped.setdefault(rule.source, []).append(rule.id)
    written: set[str] = set()
    for source, ids in grouped.items():
        kept = [after_ids[rule_id] for rule_id in ids if rule_id in after_ids]
        path = Path(source) if source else None
        if not kept:
            if path is not None and path.is_file():
                data = parse_yaml(_read_text(path))
                if isinstance(data, dict) and "freshness" in data:
                    _write_rules(path, [])
                else:
                    path.unlink()
            continue
        if path is None:
            continue
        _write_rules(path, kept)
        written.add(source)
    for rule in after.rules:
        if rule.id in before_ids:
            continue
        path = directory / f"{rule.id}.yaml"
        if path.exists():
            raise PolicyLanguageError(
                [Finding("type", f"refusing to overwrite {path}", rule.id)]
            )
        _write_rules(path, [rule])


def _write_rules(path: Path, rules: list[RuleDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = parse_yaml(_read_text(path)) if path.is_file() else None
    if isinstance(existing, dict) and "freshness" in existing:
        text = emit_yaml({"freshness": existing["freshness"],
                          "rules": [rule_to_data(rule) for rule in rules]}) + "\n"
    elif len(rules) == 1:
        text = emit_yaml(rule_to_data(rules[0])) + "\n"
    else:
        text = emit_yaml([rule_to_data(rule) for rule in rules]) + "\n"
    path.write_text(text, encoding="utf-8")


# ── Explain and scenarios ────────────────────────────────────────────────────


def _payload(rule: RuleDocument) -> dict[str, Any]:
    body = json.loads(canonical_json(rule.rule))
    body["id"] = rule.id
    return body


def _policy_row(rule: RuleDocument, number: int) -> Policy:
    return Policy(
        id=number,
        regime=rule.regime,
        rule_type=rule.rule_type,
        applies_to=rule.applies_to,
        rule=_payload(rule),
        citation=rule.citation,
        effective_from=rule.effective_from,
        effective_to=rule.effective_to,
    )


def _snapshot(policies: PolicySet) -> tuple[PolicySnapshot, dict[int, str]]:
    rows = []
    mapping: dict[int, str] = {}
    for number, rule in enumerate(policies.rules, start=1):
        rows.append(_policy_row(rule, number))
        mapping[number] = rule.id
    return PolicySnapshot(policies=tuple(rows)), mapping


def _envelope_from(data: dict[str, Any]) -> Envelope:
    principal = data.get("principal", "")
    purpose = data.get("purpose", "")
    tool = data.get("tool", "")
    if (
        not isinstance(principal, str)
        or not isinstance(purpose, str)
        or not isinstance(tool, str)
    ):
        raise PolicyLanguageError(
            [Finding("type", "principal, purpose, and tool must be strings")]
        )
    return Envelope(
        principal=principal,
        purpose=purpose,
        tool=tool,
        actor_location=_as_str(data.get("actor_location", "")),
        agent_name=data.get("agent_name"),
        subject_type=data.get("subject_type"),
        subject_id=None
        if data.get("subject_id") is None
        else str(data.get("subject_id")),
        subjects=_string_dict(data.get("subjects") or {}),
        arguments=dict(data.get("arguments") or {}),
        roles=[str(item) for item in data.get("roles") or []],
        scopes=_string_dict(data.get("scopes") or {}),
        is_admin=bool(data.get("is_admin", False)),
        context=dict(data.get("context") or {}),
    )


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise PolicyLanguageError([Finding("type", "expected a mapping of strings")])
    return {str(key): str(item) for key, item in value.items()}


def scenario_case(
    scenario: Scenario,
) -> tuple[Envelope, dict[str, Any], tuple[str, ...]]:
    """The envelope, attributes, and purpose allowlist for one scenario."""
    given = dict(scenario.given)
    when = dict(scenario.when)
    tool = when.get("tool", given.get("tool", ""))
    purpose = given.get("purpose", _SCENARIO_PURPOSE)
    if not isinstance(purpose, str):
        purpose = str(purpose)
    data = dict(given)
    data["tool"] = tool
    data["purpose"] = purpose
    if "arguments" in when:
        data["arguments"] = when["arguments"]
    attributes = data.pop("attributes", {}) or {}
    if not isinstance(attributes, dict):
        raise PolicyLanguageError(
            [Finding("type", "scenario attributes must be a mapping")]
        )
    return _envelope_from(data), dict(attributes), (purpose,)


def explain(
    policies: PolicySet,
    envelope: Envelope,
    *,
    attributes: dict[str, Any] | None = None,
    purposes: tuple[str, ...] = (),
    as_of: datetime | None = None,
) -> Explanation:
    """Which rules engage for ``envelope``, and the checks each one produced.

    Engagement is the core rule: a policy that is effective at ``as_of`` is
    evaluated. ``applies_to`` is reported and does not, by itself, skip a rule.
    """
    moment = as_of or utcnow()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    snapshot, mapping = _snapshot(policies)
    evaluation = decide(
        envelope,
        snapshot,
        attributes=dict(attributes or {}),
        purposes=purposes,
        as_of=moment,
        request_id="policy-explain",
    )
    by_policy: dict[int, list[Check]] = {}
    other: list[Check] = []
    for check in evaluation.checks:
        if check.policy_id is None:
            other.append(check)
        else:
            by_policy.setdefault(check.policy_id, []).append(check)
    explained: list[RuleExplanation] = []
    for number, rule in enumerate(policies.rules, start=1):
        row = snapshot.policies[number - 1]
        if row.is_effective(moment):
            why = f"effective at {_format_dt(moment)}, so decide() evaluated it"
            engaged = True
        else:
            why = (
                f"outside the effective window at {_format_dt(moment)} "
                f"(effective_from {_format_dt(rule.effective_from)}, "
                f"effective_to {_format_dt(rule.effective_to) or 'open'})"
            )
            engaged = False
        explained.append(
            RuleExplanation(
                rule_id=mapping[number],
                engaged=engaged,
                why=why,
                applies_to=rule.applies_to,
                checks=tuple(by_policy.get(number, ())),
            )
        )
    return Explanation(
        outcome=evaluation.outcome,
        rules=tuple(explained),
        other_checks=tuple(other),
        as_of=moment,
    )


def render_explanation(explanation: Explanation) -> str:
    """The text form of an explanation."""
    lines = [f"outcome: {explanation.outcome.value}", ""]
    for rule in explanation.rules:
        state = "engaged" if rule.engaged else "not engaged"
        lines.append(f"{state} {rule.rule_id}")
        lines.append(f"  why: {rule.why}")
        lines.append(f"  applies_to: {rule.applies_to or '(none)'}")
        if not rule.checks:
            lines.append("  checks: none")
        for check in rule.checks:
            lines.append(f"  check: {check.check}")
            lines.append(f"  result: {check.result.value}")
            citation = check.citation if check.citation.strip() else "(none)"
            lines.append(f"  citation: {citation}")
        lines.append("")
    for check in explanation.other_checks:
        lines.append(f"other {check.check}")
        lines.append(f"  result: {check.result.value}")
        lines.append(f"  citation: {check.citation}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_scenarios(
    policies: PolicySet, *, as_of: datetime | None = None
) -> list[ScenarioResult]:
    """Run every colocated scenario against the policy set."""
    moment = as_of or utcnow()
    snapshot, mapping = _snapshot(policies)
    results: list[ScenarioResult] = []
    for document in policies.scenarios:
        for scenario in document.scenarios:
            results.append(_run_one(document.rule, scenario, snapshot, mapping, moment))
    return results


def _run_one(
    rule_id: str,
    scenario: Scenario,
    snapshot: PolicySnapshot,
    mapping: dict[int, str],
    as_of: datetime,
) -> ScenarioResult:
    envelope, attributes, purposes = scenario_case(scenario)
    evaluation = decide(
        envelope,
        snapshot,
        attributes=attributes,
        purposes=purposes,
        as_of=as_of,
        request_id=f"policy-test:{rule_id}:{scenario.name}",
    )
    # Colocated scenarios test policy rules in a hypothetical purpose context;
    # they never admit a workload or persist a governance decision. Execution
    # callers must pass the actual use-case register to decide().
    from kognita.governance import resolve_outcome
    checks = tuple(c for c in evaluation.checks if c.check != "USE_CASE_REGISTERED")
    evaluation = replace(evaluation, checks=checks, outcome=resolve_outcome(checks))
    expected = str(scenario.then["decision"])
    actual = evaluation.outcome.value
    problems: list[str] = []
    if actual != expected:
        problems.append(f"expected {expected}, got {actual}")
    cites = scenario.then.get("cites")
    basis_ids = {
        mapping[check.policy_id]
        for check in evaluation.basis()
        if check.policy_id is not None and check.policy_id in mapping
    }
    if isinstance(cites, str) and cites not in basis_ids:
        problems.append(f"expected the decision to cite {cites}")
    missing = [
        check.check for check in evaluation.basis() if not check.citation.strip()
    ]
    if missing:
        problems.append(f"decision check {missing[0]} has no citation")
    if problems:
        return ScenarioResult(rule_id, scenario.name, False, "; ".join(problems))
    return ScenarioResult(rule_id, scenario.name, True, actual)


def parse_envelope(
    text: str,
) -> tuple[Envelope, dict[str, Any], tuple[str, ...], datetime | None]:
    """Read an explain envelope from a JSON object."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PolicyLanguageError(
            [Finding("parse", f"envelope is not JSON: {exc}")]
        ) from exc
    if not isinstance(data, dict):
        raise PolicyLanguageError([Finding("type", "envelope must be a JSON object")])
    unknown = sorted(set(data) - _EXPLAIN_FIELDS)
    if unknown:
        raise PolicyLanguageError(
            [Finding("type", f"unknown envelope field {unknown[0]!r}")]
        )
    if "is_admin" in data and not isinstance(data["is_admin"], bool):
        raise PolicyLanguageError(
            [Finding("type", "envelope is_admin must be a boolean")]
        )
    attributes = data.get("attributes") or {}
    if not isinstance(attributes, dict):
        raise PolicyLanguageError(
            [Finding("type", "envelope attributes must be an object")]
        )
    purposes = data.get("purposes")
    if purposes is None:
        purpose_list: tuple[str, ...] = ()
    elif not isinstance(purposes, list) or any(
        not isinstance(item, str) for item in purposes
    ):
        raise PolicyLanguageError(
            [Finding("type", "envelope purposes must be a list of strings")]
        )
    else:
        purpose_list = tuple(purposes)
    as_of = None
    if data.get("as_of") is not None:
        as_of, error = _parse_datetime(data.get("as_of"), "as_of")
        if error or as_of is None:
            raise PolicyLanguageError([Finding("type", error or "as_of is invalid")])
    return _envelope_from(data), dict(attributes), purpose_list, as_of


def read_envelope(
    path_or_json: str,
) -> tuple[Envelope, dict[str, Any], tuple[str, ...], datetime | None]:
    """Read an envelope from a JSON file or a JSON object string."""
    candidate = Path(path_or_json)
    if candidate.is_file():
        return parse_envelope(_read_text(candidate))
    return parse_envelope(path_or_json)


# ── Compile to rows ──────────────────────────────────────────────────────────


def _row_rule_id(row: Policy) -> str | None:
    if not isinstance(row.rule, dict):
        return None
    ident = row.rule.get("id")
    if isinstance(ident, str) and ident:
        return ident
    return None


def _prefer_row(candidate: Policy, current: Policy, at: datetime) -> bool:
    candidate_live = candidate.is_effective(at)
    current_live = current.is_effective(at)
    if candidate_live != current_live:
        return candidate_live
    candidate_start = as_utc(candidate.effective_from)
    current_start = as_utc(current.effective_from)
    if candidate_start is None or current_start is None:
        return False
    return candidate_start > current_start


def _rows_by_rule_id(session: Session, at: datetime) -> dict[str, Policy]:
    chosen: dict[str, Policy] = {}
    for row in session.exec(select(Policy)).all():
        ident = _row_rule_id(row)
        if ident is None:
            continue
        current = chosen.get(ident)
        if current is None or _prefer_row(row, current, at):
            chosen[ident] = row
    return chosen


def _stored_body(row: Policy) -> tuple[Any, ...]:
    payload = dict(row.rule)
    ident = payload.pop("id", None)
    return (
        ident,
        row.regime,
        row.rule_type,
        row.applies_to,
        row.citation,
        canonical_json(payload),
    )


def compile_rules(
    session: Session,
    policies: PolicySet,
    *,
    at: datetime | None = None,
    close_absent: bool = False,
) -> int:
    """Insert the policy set. A changed effective row is closed and replaced.

    The first row for an id uses the file's ``effective_from``. A later compile
    of a different rule body closes that row at ``at`` and inserts the successor
    starting then. The comparison is the rule body, not the window: the file
    keeps the authored ``effective_from``, and the store keeps the history.

    With ``close_absent``, an effective row whose id is not in ``policies`` is
    closed at ``at``. A directory load and a change load pass that, because the
    set they compile is the whole policy. A single-file load does not.
    """
    moment = at or utcnow()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    existing = _rows_by_rule_id(session, moment)
    written = 0
    incoming = {rule.id for rule in policies.rules}
    for rule in policies.rules:
        payload = _payload(rule)
        row = existing.get(rule.id)
        if row is None:
            session.add(
                Policy(
                    regime=rule.regime,
                    rule_type=rule.rule_type,
                    applies_to=rule.applies_to,
                    rule=payload,
                    citation=rule.citation,
                    effective_from=rule.effective_from,
                    effective_to=rule.effective_to,
                )
            )
            written += 1
            continue
        if _stored_body(row) == _body(rule):
            continue
        if row.is_effective(moment):
            row.effective_to = moment
            session.add(row)
            session.add(
                Policy(
                    regime=rule.regime,
                    rule_type=rule.rule_type,
                    applies_to=rule.applies_to,
                    rule=payload,
                    citation=rule.citation,
                    effective_from=moment,
                    effective_to=rule.effective_to
                    if rule.effective_to is not None and rule.effective_to > moment
                    else None,
                )
            )
            written += 1
            continue
        start = as_utc(row.effective_from)
        if start is not None and start > moment:
            row.regime = rule.regime
            row.rule_type = rule.rule_type
            row.applies_to = rule.applies_to
            row.rule = payload
            row.citation = rule.citation
            row.effective_from = rule.effective_from
            row.effective_to = rule.effective_to
            session.add(row)
            written += 1
            continue
        session.add(
            Policy(
                regime=rule.regime,
                rule_type=rule.rule_type,
                applies_to=rule.applies_to,
                rule=payload,
                citation=rule.citation,
                effective_from=rule.effective_from,
                effective_to=rule.effective_to,
            )
        )
        written += 1
    if close_absent:
        for rule_id, row in existing.items():
            if rule_id in incoming or not row.is_effective(moment):
                continue
            row.effective_to = moment
            session.add(row)
            written += 1
    session.flush()
    return written


def policies_from_session(session: Session, *, at: datetime | None = None) -> PolicySet:
    """The effective policy rows, read back as rule documents."""
    moment = at or utcnow()
    rules: list[RuleDocument] = []
    for row in _rows_by_rule_id(session, moment).values():
        if not row.is_effective(moment):
            continue
        payload = dict(row.rule)
        ident = payload.pop("id", None)
        if not isinstance(ident, str):
            ident = f"policy-{row.id}"
        start = as_utc(row.effective_from)
        assert start is not None
        rules.append(
            RuleDocument(
                id=ident,
                regime=row.regime,
                rule_type=row.rule_type,
                citation=row.citation,
                effective_from=start,
                rule=payload,
                applies_to=row.applies_to,
                effective_to=as_utc(row.effective_to),
            )
        )
    rules.sort(key=lambda rule: rule.id)
    return PolicySet(rules=tuple(rules), source="database")


__all__ = [
    "CLARIFICATION_RE",
    "Explanation",
    "Finding",
    "PolicyChange",
    "PolicyLanguageError",
    "PolicySet",
    "RuleDocument",
    "RuleExplanation",
    "Scenario",
    "ScenarioDocument",
    "ScenarioResult",
    "apply_change",
    "archive_change",
    "blocking_findings",
    "compile_rules",
    "diff_sets",
    "emit_yaml",
    "explain",
    "format_finding",
    "is_change_path",
    "merge_changes",
    "parse_change",
    "parse_envelope",
    "parse_rule",
    "parse_scenario_document",
    "parse_yaml",
    "policies_from_session",
    "read_change",
    "read_envelope",
    "read_policy_set",
    "render_change",
    "render_explanation",
    "rule_to_data",
    "run_scenarios",
    "scenario_case",
    "source_hash",
    "strict_findings",
    "validate",
]
