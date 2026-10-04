"""The AI gateway — an explicit proxy in front of an OpenAI-compatible provider.

Agents point ``base_url`` at this process. The gateway does not intercept TLS
and it does not install a certificate. It governs the calls a deployment
configures to send here, and nothing else.

One request, in order:

1. Read the body just enough to build an :class:`~kognita.envelope.Envelope`.
   Principal and purpose come from authenticated headers or the bound client
   configuration. ``tool`` is ``model_call``. The subject is the model name.
2. Fill attributes the caller did not supply from the prompt, through
   :func:`kognita.governance.classifier_derived_envelope`. A typed attribute
   already on the request wins.
3. :func:`kognita.governance.decide` before any byte is forwarded. A denial
   is returned here.
4. On allow, redact with the egress guard and forward to the upstream.
5. Restore redacted spans in the response.
6. Classify the response before it is returned.
7. Record ``MODEL_CALL`` and ``EGRESS`` by hash and reference. The prompt and
   the response are not copied into the evidence.

Token totals and cost, when the provider response reports them, are added to
the :class:`~kognita.tools.Run` budget.

Identity, on this gateway only. Calls that do not come through here keep the
behaviour they already have.

- A call with no agent name is not a human, and it does not skip the registry.
  The call must name a registered agent or an approved system trigger. Either
  of those is admitted; the absence of both is a denial.
- Until agents carry their own credentials, an agent name is accepted only
  when the bound client configuration lists it.

The failure mode is set per use case. A use case is the purpose string on the
envelope; the retention store already records that string as ``use_case``.
The default is fail closed: if the evidence store cannot record the call, the
gateway refuses it. Degraded mode may proceed only for a local model and
content that is not client-identifying, and writes the decision and the model
evidence once the store accepts them. A caller classification can raise that
label and cannot lower it: while the store is down the classifier still runs,
on the values that would be forwarded after JSON parsing, not the raw wire
text and not only the extracted prompt. A call
is never forwarded without a decision. If the policy snapshot cannot be read,
every mode refuses.
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from sqlmodel import Session

from kognita.approvals import ApprovalError
from kognita.canonical import canonical_hash
from kognita.classify import PatternClassifier, classifier_record, most_sensitive
from kognita.db import session_scope
from kognita.egress import EgressGuard
from kognita.envelope import Check, Envelope, Evaluation, envelope_hash
from kognita.evidence import EvidenceWriter
from kognita.exceptions import KognitaError
from kognita.governance import (
    classifier_derived_envelope,
    decide,
    load_snapshot,
    record,
    resolve_outcome,
)
from kognita.retention import RetentionStore
from kognita.rules import build_registry
from kognita.tools import (
    Run,
    _budget_breaches,
    _budget_payload,
    _consume,
    _deny_for_budgets,
    _load_run,
    _refresh_run,
    _save_run,
)
from kognita.vocabulary import (
    ActorType,
    CheckResult,
    Classification,
    EgressDecision,
    EventType,
    FailureMode,
    Outcome,
    classification_rank,
)

_FORWARDED_HEADERS = frozenset({"authorization", "content-type", "accept"})


class _EvidenceStoreUnavailable(KognitaError):
    """Raised when a call cannot be written to the evidence store."""


class _UpstreamUnavailable(KognitaError):
    """The configured upstream could not be reached."""


class _RunNotFound(KognitaError):
    """The request named a run that is not in the store."""


@dataclass
class _DeferredModel:
    """What :meth:`Gateway._emit` needs once the evidence store accepts writes."""

    decision: EgressDecision
    token_map: dict[str, str]
    sent: bool
    sent_body: bytes
    actor_type: ActorType
    actor_id: str
    response_record: dict[str, Any] | None
    tokens: int | None
    cost_usd: float | None
    received_text: str = ""
    response_payload: dict[str, Any] | None = None
    prompt_template_version: str | None = None


@dataclass
class _DeferredEvidence:
    """A decision, and the model evidence of a call that already proceeded.

    Held in the gateway process until the evidence store accepts writes. The
    bytes are what the model was sent and what it returned, so the retained
    hashes match a call that was recorded at the time.
    """

    evaluation: Evaluation
    classification: Classification
    decided_at: datetime
    decision_budget: dict[str, Any] | None
    run: Run | None = None
    save_run: bool = False
    model: _DeferredModel | None = None


@dataclass(frozen=True)
class ClientConfiguration:
    """The authenticated client this gateway is bound to.

    ``agent_names`` is the set of agent names the client may present.
    ``system_triggers`` is the set of approved system triggers. ``agent_name``
    and ``system_trigger`` are the identities the session itself carries when
    a request header does not. Principal and purpose headers, when the
    request carries them, are the authenticated values; otherwise these
    fields are.
    """

    principal: str
    purpose: str
    agent_names: frozenset[str] = field(default_factory=frozenset)
    system_triggers: frozenset[str] = field(default_factory=frozenset)
    actor_location: str = ""
    agent_name: str | None = None
    system_trigger: str | None = None

    def __post_init__(self) -> None:
        names = set(self.agent_names)
        if self.agent_name:
            names.add(self.agent_name)
        triggers = set(self.system_triggers)
        if self.system_trigger:
            triggers.add(self.system_trigger)
        object.__setattr__(self, "agent_names", frozenset(names))
        object.__setattr__(self, "system_triggers", frozenset(triggers))


@dataclass
class GatewayResponse:
    """What the gateway returns to the caller. ``evaluation`` is set once one was recorded."""

    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)
    evaluation: Evaluation | None = None
    response_classifier: dict[str, Any] | None = None


class _GatewayPack:
    """Attributes and rules when a deployment does not pass its own pack.

    Policies in the store still apply. This pack does not invent subjects.
    """

    name = "gateway"

    def load_subjects(self, envelope: Envelope, session: Any) -> dict[str, Any]:
        return {}

    def resolve_attributes(self, envelope: Envelope, subjects: dict[str, Any]) -> dict[str, Any]:
        return {}

    def rules(self) -> dict[str, Any]:
        return build_registry()


Transport = Callable[[str, str, Mapping[str, str], bytes], tuple[int, Mapping[str, str], bytes]]


def _headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): "" if value is None else str(value) for key, value in headers.items()}


def _is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().strip("[]")
    return host in {"localhost", "127.0.0.1", "::1"}


def _upstream_url(upstream: str, path: str) -> str:
    suffix = path if path.startswith("/") else f"/{path}"
    return upstream.rstrip("/") + suffix


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return ""


def _forwarded_text(payload: Any) -> str:
    """Strings the provider reads after parsing the JSON body.

    The wire text is not that text. A unicode escape is decoded before the
    value is used, so the classifier has to see the parsed value.
    """
    parts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if value:
                parts.append(value)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                walk(item)
            return
        if isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return "\n".join(parts)


def _prompt_text(payload: Mapping[str, Any]) -> str:
    """The prose the provider will see. Typed fields other than text are left out."""
    parts: list[str] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                text = _content_text(message.get("content"))
                if text:
                    parts.append(text)
    for key in ("prompt", "input"):
        text = _content_text(payload.get(key))
        if text:
            parts.append(text)
    return "\n".join(parts)


def _response_text(payload: Mapping[str, Any]) -> str:
    parts: list[str] = []
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                text = _content_text(message.get("content"))
                if text:
                    parts.append(text)
            if isinstance(choice.get("text"), str) and choice["text"]:
                parts.append(choice["text"])
    return "\n".join(parts)


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        return value
    return None


def _template_version(headers: Mapping[str, str]) -> str | None:
    value = headers.get("prompt_template_version") or headers.get("prompt-template-version")
    return value or None


def _reported_model(
    requested: str, response: Mapping[str, Any] | None
) -> tuple[str, str | None]:
    """Model name and version as the provider reported them.

    The request names the model that was asked for. A response ``model`` is the
    name the provider says it served. ``model_version`` is that field when the
    provider sends one, otherwise the response ``model`` string. A provider that
    reports neither leaves the version empty.
    """
    if not isinstance(response, dict):
        return requested, None
    reported = response.get("model")
    name = reported if isinstance(reported, str) and reported else requested
    version = response.get("model_version")
    if isinstance(version, str) and version:
        return name, version
    if isinstance(reported, str) and reported:
        return name, reported
    return name, None


def _usage(payload: Mapping[str, Any]) -> tuple[int | None, float | None]:
    """Token total and cost, only when the provider response reports them."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None
    total = usage.get("total_tokens")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        numbers = [
            value
            for value in (prompt, completion)
            if not isinstance(value, bool) and isinstance(value, (int, float))
        ]
        total = sum(numbers) if numbers else None
    tokens = int(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else None
    cost_value = usage.get("cost_usd", usage.get("cost"))
    if isinstance(cost_value, bool) or not isinstance(cost_value, (int, float)):
        cost = None
    else:
        cost = float(cost_value)
    return tokens, cost


def _with_checks(evaluation: Evaluation, extra: Sequence[Check]) -> Evaluation:
    if not extra:
        return evaluation
    checks = evaluation.checks + tuple(extra)
    return Evaluation(
        request_id=evaluation.request_id,
        outcome=resolve_outcome(checks),
        checks=checks,
        attributes=evaluation.attributes,
        envelope=evaluation.envelope,
        as_of=evaluation.as_of,
        envelope_hash=envelope_hash(evaluation.envelope, evaluation.attributes, checks),
    )


def _identity(
    client: ClientConfiguration, headers: Mapping[str, str]
) -> tuple[str | None, str | None, list[Check]]:
    """Resolve the caller to an agent or a system trigger.

    A matching registered agent is not denied here; the registry and the kill
    switch still run inside :func:`kognita.governance.decide`. An approved
    system trigger is not denied here either. Anything else is.
    """
    claimed = headers.get("agent_name") or None
    agent_name = claimed or client.agent_name
    if agent_name:
        if agent_name not in client.agent_names:
            return agent_name, None, [
                Check(
                    check="AGENT_IDENTITY",
                    regime="INTERNAL",
                    result=CheckResult.FAIL,
                    citation=(
                        "agent name does not match the authenticated client configuration"
                    ),
                )
            ]
        return agent_name, None, []

    claimed_trigger = headers.get("system_trigger") or None
    trigger = claimed_trigger or client.system_trigger
    if trigger and trigger in client.system_triggers:
        return None, trigger, [
            Check(
                check="SYSTEM_TRIGGER",
                regime="INTERNAL",
                result=CheckResult.PASS,
                citation=f"approved system trigger {trigger}",
            )
        ]
    if trigger:
        return None, trigger, [
            Check(
                check="SYSTEM_TRIGGER",
                regime="INTERNAL",
                result=CheckResult.FAIL,
                citation="system trigger is not approved for this client configuration",
            )
        ]
    return None, None, [
        Check(
            check="AGENT_IDENTITY",
            regime="INTERNAL",
            result=CheckResult.FAIL,
            citation=(
                "no agent name; a registered agent or an approved system trigger is required"
            ),
        )
    ]


def _budget_block(
    run: Run, *, classification: Classification, now: datetime
) -> list[str]:
    """Caps already exhausted, plus the caps the next call would cross.

    Token and cost figures from a previous provider response count. This call
    does not yet know its own usage, so a budget that is already spent is a
    denial before anything is forwarded.
    """
    blocked: list[str] = []
    if run.max_tokens is not None and run.tokens_used >= run.max_tokens:
        blocked.append("max_tokens")
    if run.max_cost_usd is not None and run.cost_usd_used >= run.max_cost_usd:
        blocked.append("max_cost_usd")
    for name in _budget_breaches(
        run,
        classification=classification,
        cost_usd=None,
        tokens=None,
        now=now,
    ):
        if name not in blocked:
            blocked.append(name)
    return blocked


def _apply_usage(run: Run, *, tokens: int | None, cost_usd: float | None) -> None:
    if tokens is not None:
        run.tokens_used += tokens
    if cost_usd is not None:
        run.cost_usd_used += cost_usd


def _urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes
) -> tuple[int, Mapping[str, str], bytes]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, dict(exc.headers), raw
    except urllib.error.URLError as exc:
        raise _UpstreamUnavailable(str(exc.reason)) from exc


def _json_response(status: int, payload: Mapping[str, Any], **extra: Any) -> GatewayResponse:
    return GatewayResponse(
        status=status,
        body=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        **extra,
    )


def _denied(evaluation: Evaluation) -> GatewayResponse:
    return _json_response(
        403,
        {
            "outcome": evaluation.outcome.value,
            "request_id": evaluation.request_id,
            "checks": [check.to_dict() for check in evaluation.basis()],
        },
        evaluation=evaluation,
    )


def _copy_run(run: Run) -> Run:
    return replace(run, approvals_pending=list(run.approvals_pending))


def _client_data(classification: Classification) -> bool:
    """True at C2 and above. C2 is client-identifying; C3 is tighter."""
    return classification_rank(classification) >= classification_rank(Classification.C2)


def _degraded_may_proceed(upstream: str, classification: Classification) -> bool:
    """Degraded mode proceeds only for a local model with no client data."""
    return _is_local(upstream) and not _client_data(classification)


def _degraded_label(
    classifier: Any,
    payload: Any,
    classification: Classification,
    hint: Classification | None,
) -> Classification:
    """The label the degraded gate uses once the store has refused the write.

    ``hint`` is the caller classification. It is a floor: the classifier can
    raise it, and the caller cannot lower what the parsed values support.
    Those values are what the provider reads after JSON parsing. The raw wire
    text is not.
    """
    return most_sensitive(
        [classification, classifier.classify(_forwarded_text(payload), hint=hint)]
    )


def _with_classification(evaluation: Evaluation, classification: Classification) -> Evaluation:
    """The same decision, with the label the degraded gate actually used."""
    if evaluation.attributes.get("classification") == classification.value:
        return evaluation
    attributes = dict(evaluation.attributes)
    attributes["classification"] = classification.value
    return replace(
        evaluation,
        attributes=attributes,
        envelope_hash=envelope_hash(evaluation.envelope, attributes, evaluation.checks),
    )


def _refuse_degraded(
    evaluation: Evaluation, classification: Classification, upstream: str
) -> Evaluation:
    """The decision that refuses a degraded call the store cannot record."""
    extra: list[Check] = []
    if not _is_local(upstream):
        extra.append(
            Check(
                check="FAILURE_MODE",
                regime="INTERNAL",
                result=CheckResult.FAIL,
                citation=(
                    "a remote model may not be called while the evidence store is unavailable"
                ),
            )
        )
    if _client_data(classification):
        extra.append(
            Check(
                check="FAILURE_MODE",
                regime="INTERNAL",
                result=CheckResult.FAIL,
                citation=(
                    f"{classification.value} content is client-identifying or tighter and "
                    "may not be sent while the evidence store is unavailable"
                ),
            )
        )
    return _with_checks(evaluation, extra)


def _model_evidence(
    *,
    decision: EgressDecision,
    token_map: Mapping[str, str],
    sent: bool,
    sent_body: bytes,
    actor_type: ActorType,
    actor_id: str,
    response_record: dict[str, Any] | None,
    tokens: int | None,
    cost_usd: float | None,
    received_text: str = "",
    response_payload: Mapping[str, Any] | None = None,
    prompt_template_version: str | None = None,
) -> _DeferredModel:
    payload = dict(response_payload) if response_payload is not None else None
    return _DeferredModel(
        decision=decision,
        token_map=dict(token_map),
        sent=sent,
        sent_body=sent_body,
        actor_type=actor_type,
        actor_id=actor_id,
        response_record=response_record,
        tokens=tokens,
        cost_usd=cost_usd,
        received_text=received_text,
        response_payload=payload,
        prompt_template_version=prompt_template_version,
    )


class Gateway:
    """OpenAI-compatible proxy. ``transport`` replaces the network in tests.

    ``failure_mode`` maps a use case to :class:`~kognita.vocabulary.FailureMode`.
    A use case is the purpose string on the envelope. A use case with no entry
    fails closed.
    """

    def __init__(
        self,
        *,
        engine: Any,
        evidence: EvidenceWriter,
        upstream: str,
        client: ClientConfiguration,
        pack: Any | None = None,
        purposes: Sequence[str] = (),
        egress: EgressGuard | None = None,
        classifier: Any | None = None,
        transport: Transport | None = None,
        run: Run | None = None,
        provider: str | None = None,
        failure_mode: Mapping[str, FailureMode | str] | None = None,
    ) -> None:
        self.engine = engine
        self.evidence = evidence
        self.upstream = upstream
        self.provider = provider
        self.client = client
        self.pack = pack if pack is not None else _GatewayPack()
        self.purposes = purposes
        self.egress = egress if egress is not None else EgressGuard()
        self.classifier = classifier if classifier is not None else PatternClassifier()
        self.transport = transport if transport is not None else _urllib_transport
        self.run = run
        # Keys are use cases. A use case is the purpose string on the envelope.
        # A use case with no entry fails closed. There is no pass-through mode.
        self.failure_mode = {
            str(use_case): FailureMode(mode) for use_case, mode in (failure_mode or {}).items()
        }
        self._deferred: list[_DeferredEvidence] = []
        self._deferred_lock = threading.Lock()

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Serve ``http://{host}:{port}/v1`` until the process stops.

        The upstream URL is the provider origin, for example
        ``https://api.openai.com``. Clients set ``base_url`` to this server's
        ``/v1`` prefix.
        """
        self._make_server(host, port).serve_forever()

    def _make_server(self, host: str, port: int) -> ThreadingHTTPServer:
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                try:
                    length = int(self.headers.get("Content-Length", "0") or "0")
                except ValueError:
                    length = 0
                body = self.rfile.read(length) if length > 0 else b""
                result = gateway.handle(self.command, self.path, self.headers, body)
                payload = result.body
                self.send_response(result.status)
                for key, value in result.headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                self._method_not_allowed()

            def do_PUT(self) -> None:  # noqa: N802
                self._method_not_allowed()

            def _method_not_allowed(self) -> None:
                body = b'{"error":"the gateway only proxies model calls"}'
                self.send_response(405)
                self.send_header("content-type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: Any) -> None:
                sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

        return ThreadingHTTPServer((host, port), Handler)

    def handle(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        session: Session | None = None,
        run: Run | None = None,
    ) -> GatewayResponse:
        """Govern one request. A supplied ``session`` is left for the caller to commit."""
        self._flush_deferred()
        if session is None:
            try:
                with session_scope(self.engine) as owned:
                    return self._handle(owned, method, path, headers, body, run=run)
            except _EvidenceStoreUnavailable:
                return _json_response(503, {"error": "evidence store unavailable"})
        try:
            return self._handle(session, method, path, headers, body, run=run)
        except _EvidenceStoreUnavailable:
            session.rollback()
            return _json_response(503, {"error": "evidence store unavailable"})

    def _store(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except _EvidenceStoreUnavailable:
            raise
        except Exception as exc:
            raise _EvidenceStoreUnavailable(str(exc)) from exc

    def _handle(
        self,
        session: Session,
        method: str,
        path: str,
        raw_headers: Mapping[str, str],
        body: bytes,
        *,
        run: Run | None,
    ) -> GatewayResponse:
        headers = _headers(raw_headers)
        if method.upper() != "POST":
            return _json_response(405, {"error": "the gateway only proxies model calls"})
        try:
            payload = json.loads(body.decode("utf-8")) if body else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _json_response(400, {"error": "request body is not JSON"})
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), str) or not payload["model"]:
            return _json_response(400, {"error": "request body has no model"})
        if payload.get("stream") is True:
            return _json_response(
                400,
                {"error": "a streamed response cannot be restored or classified"},
            )

        typed: Classification | None = None
        if headers.get("classification"):
            try:
                typed = Classification(headers["classification"])
            except ValueError:
                return _json_response(400, {"error": "classification is not a known value"})

        agent_name, trigger, identity_checks = _identity(self.client, headers)
        principal = headers.get("principal") or self.client.principal
        purpose = headers.get("purpose") or self.client.purpose
        envelope = Envelope(
            principal=principal,
            purpose=purpose,
            tool="model_call",
            actor_location=self.client.actor_location,
            agent_name=agent_name,
            subject_type="model",
            subject_id=payload["model"],
        )
        now = self.evidence.clock()
        prompt = _prompt_text(payload)
        try:
            bound_run = self._bind_run(session, headers, run)
        except _RunNotFound:
            return _json_response(403, {"error": "run not found"})

        domain = envelope
        if envelope.subject_type == "model":
            domain = replace(envelope, subject_type=None, subject_id=None)
        subjects = self.pack.load_subjects(domain, session)
        attributes = dict(self.pack.resolve_attributes(envelope, subjects))
        if typed is not None:
            attributes["classification"] = typed.value
        if prompt:
            envelope, attributes = classifier_derived_envelope(
                prompt,
                envelope,
                attributes=attributes,
                classifier=self.classifier,
            )
        classification = Classification(attributes.get("classification") or Classification.C1)

        snapshot = self._store(lambda: load_snapshot(session, as_of=now))
        evaluation = decide(
            envelope,
            snapshot,
            attributes=attributes,
            subjects=subjects,
            rules=self.pack.rules(),
            purposes=self.purposes,
            engages=getattr(self.pack, "engages", None),
            as_of=now,
        )
        evaluation = _with_checks(evaluation, identity_checks)
        if bound_run is not None and evaluation.outcome is Outcome.ALLOW:
            self._store(lambda: _refresh_run(session, bound_run))
            blocked = _budget_block(bound_run, classification=classification, now=now)
            if blocked:
                evaluation = _deny_for_budgets(evaluation, blocked)

        local = _is_local(self.upstream)
        egress_decision = (
            self.egress.evaluate(classification, destination_is_local=local)
            if evaluation.outcome is Outcome.ALLOW
            else None
        )
        if egress_decision is EgressDecision.DENY:
            evaluation = _with_checks(
                evaluation,
                [
                    Check(
                        check="EGRESS",
                        regime="INTERNAL",
                        result=CheckResult.FAIL,
                        citation=(
                            f"{classification.value} content may not be sent to "
                            f"{self.upstream}"
                        ),
                    )
                ],
            )

        actor_type = _actor_type(agent_name, trigger)
        actor_id = _actor_id(agent_name, trigger)
        template_version = _template_version(headers)
        decision_budget = _budget_payload(bound_run) if bound_run is not None else None
        deferred: _DeferredEvidence | None = None

        if evaluation.outcome is not Outcome.ALLOW:
            egress_model = None
            if egress_decision is EgressDecision.DENY:
                egress_model = _model_evidence(
                    decision=EgressDecision.DENY,
                    token_map={},
                    sent=False,
                    sent_body=b"",
                    actor_type=actor_type,
                    actor_id=actor_id,
                    response_record=None,
                    tokens=None,
                    cost_usd=None,
                    prompt_template_version=template_version,
                )
            try:
                evaluation = self._record(session, evaluation, classification, bound_run, now)
                if egress_model is not None:
                    self._store(
                        lambda: self._emit_model(
                            session, evaluation, classification, egress_model, bound_run
                        )
                    )
            except _EvidenceStoreUnavailable:
                if not self._is_degraded(purpose):
                    raise
                self._abandon(session)
                self._enqueue(
                    _DeferredEvidence(
                        evaluation=evaluation,
                        classification=classification,
                        decided_at=now,
                        decision_budget=decision_budget,
                        run=_copy_run(bound_run) if bound_run is not None else None,
                        save_run=False,
                        model=egress_model,
                    )
                )
            return _denied(evaluation)

        consumed = False
        try:
            evaluation = self._record(session, evaluation, classification, bound_run, now)
            if bound_run is not None:
                _consume(bound_run, cost_usd=None, tokens=None, now=now)
                consumed = True
                self._store(lambda: _save_run(session, bound_run))
        except _EvidenceStoreUnavailable:
            if not self._is_degraded(purpose):
                raise
            self._abandon(session)
            classification = _degraded_label(
                self.classifier,
                payload,
                classification,
                typed,
            )
            evaluation = _with_classification(evaluation, classification)
            if not _degraded_may_proceed(self.upstream, classification):
                refused = _refuse_degraded(evaluation, classification, self.upstream)
                self._enqueue(
                    _DeferredEvidence(
                        evaluation=refused,
                        classification=classification,
                        decided_at=now,
                        decision_budget=decision_budget,
                        model=None,
                    )
                )
                return _denied(refused)
            if bound_run is not None and not consumed:
                _consume(bound_run, cost_usd=None, tokens=None, now=now)
            deferred = self._enqueue(
                _DeferredEvidence(
                    evaluation=evaluation,
                    classification=classification,
                    decided_at=now,
                    decision_budget=decision_budget,
                    run=_copy_run(bound_run) if bound_run is not None else None,
                    save_run=bound_run is not None,
                )
            )

        raw = body.decode("utf-8")
        if egress_decision is EgressDecision.REDACT:
            sent_text, token_map = self.egress.redactor.redact(raw)
        else:
            sent_text, token_map = raw, {}
        sent_body = sent_text.encode("utf-8")
        forward = {
            key: value for key, value in headers.items() if key in _FORWARDED_HEADERS
        }
        if "content-type" not in forward:
            forward["content-type"] = "application/json"
        forward_decision = egress_decision or EgressDecision.ALLOW

        try:
            status, upstream_headers, upstream_body = self.transport(
                "POST",
                _upstream_url(self.upstream, path),
                forward,
                sent_body,
            )
        except _UpstreamUnavailable:
            self._record_model_evidence(
                session,
                evaluation,
                classification=classification,
                deferred=deferred,
                model=_model_evidence(
                    decision=forward_decision,
                    token_map=token_map,
                    sent=True,
                    sent_body=sent_body,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    response_record=None,
                    tokens=None,
                    cost_usd=None,
                    prompt_template_version=template_version,
                ),
                run=bound_run,
                decision_budget=decision_budget,
                decided_at=now,
            )
            return _json_response(502, {"error": "upstream unavailable"}, evaluation=evaluation)

        try:
            upstream_text = upstream_body.decode("utf-8")
        except UnicodeDecodeError:
            self._record_model_evidence(
                session,
                evaluation,
                classification=classification,
                deferred=deferred,
                model=_model_evidence(
                    decision=forward_decision,
                    token_map=token_map,
                    sent=True,
                    sent_body=sent_body,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    response_record=None,
                    tokens=None,
                    cost_usd=None,
                    prompt_template_version=template_version,
                ),
                run=bound_run,
                decision_budget=decision_budget,
                decided_at=now,
            )
            return _json_response(
                502,
                {"error": "upstream response could not be classified"},
                evaluation=evaluation,
            )
        restored = (
            self.egress.redactor.restore(upstream_text, token_map)
            if token_map
            else upstream_text
        )
        restored_payload = _json_object(restored)
        response_source = _response_text(restored_payload) if restored_payload else restored
        response_record = classifier_record(self.classifier, response_source)
        tokens, cost_usd = _usage(restored_payload) if restored_payload else (None, None)
        if bound_run is not None and (tokens is not None or cost_usd is not None):
            _apply_usage(bound_run, tokens=tokens, cost_usd=cost_usd)
        self._record_model_evidence(
            session,
            evaluation,
            classification=classification,
            deferred=deferred,
            model=_model_evidence(
                decision=forward_decision,
                token_map=token_map,
                sent=True,
                sent_body=sent_body,
                actor_type=actor_type,
                actor_id=actor_id,
                response_record=response_record,
                tokens=tokens,
                cost_usd=cost_usd,
                received_text=upstream_text,
                response_payload=_json_object(upstream_text),
                prompt_template_version=template_version,
            ),
            run=bound_run,
            decision_budget=decision_budget,
            decided_at=now,
        )
        content_type = "application/json"
        for key, value in upstream_headers.items():
            if str(key).lower() == "content-type" and value:
                content_type = str(value)
                break
        return GatewayResponse(
            status=status,
            body=restored.encode("utf-8"),
            headers={"content-type": content_type},
            evaluation=evaluation,
            response_classifier=response_record,
        )

    def _is_degraded(self, use_case: str) -> bool:
        return self.failure_mode.get(use_case, FailureMode.FAIL_CLOSED) is FailureMode.DEGRADED

    def _abandon(self, session: Session) -> None:
        """Drop a write the store did not accept, so a later commit cannot keep it."""
        session.rollback()

    def _enqueue(self, item: _DeferredEvidence) -> _DeferredEvidence:
        with self._deferred_lock:
            self._deferred.append(item)
        return item

    def _flush_deferred(self) -> None:
        """Write calls that proceeded while the store was down, if it accepts them now."""
        with self._deferred_lock:
            if not self._deferred:
                return
            pending = list(self._deferred)
            try:
                with session_scope(self.engine) as session:
                    for item in pending:
                        self._write_deferred(session, item)
            except _EvidenceStoreUnavailable:
                return
            written = {id(item) for item in pending}
            self._deferred = [item for item in self._deferred if id(item) not in written]

    def _write_deferred(self, session: Session, item: _DeferredEvidence) -> None:
        if item.save_run and item.run is not None:
            run = item.run
            self._store(lambda: _save_run(session, run))
        evaluation = item.evaluation
        classification = item.classification
        self._store(
            lambda: record(
                session,
                evaluation,
                evidence=self.evidence,
                classification=classification,
                budget=item.decision_budget,
                now=item.decided_at,
            )
        )
        model = item.model
        if model is not None:
            self._store(
                lambda: self._emit_model(session, evaluation, classification, model, item.run)
            )

    def _emit_model(
        self,
        session: Session,
        evaluation: Evaluation,
        classification: Classification,
        model: _DeferredModel,
        run: Run | None,
    ) -> None:
        self._emit(
            session,
            evaluation,
            classification=classification,
            decision=model.decision,
            token_map=model.token_map,
            sent=model.sent,
            sent_body=model.sent_body,
            actor_type=model.actor_type,
            actor_id=model.actor_id,
            response_record=model.response_record,
            tokens=model.tokens,
            cost_usd=model.cost_usd,
            run=run,
            received_text=model.received_text,
            response_payload=model.response_payload,
            prompt_template_version=model.prompt_template_version,
        )

    def _record_model_evidence(
        self,
        session: Session,
        evaluation: Evaluation,
        *,
        classification: Classification,
        deferred: _DeferredEvidence | None,
        model: _DeferredModel,
        run: Run | None,
        decision_budget: dict[str, Any] | None,
        decided_at: datetime,
    ) -> None:
        """Record model evidence now, or keep it with the deferred decision.

        Fail closed raises when the store cannot take the write. Degraded keeps
        the decision and this evidence together until the store recovers.
        """
        if deferred is not None:
            if run is not None:
                deferred.run = _copy_run(run)
                deferred.save_run = True
            deferred.model = model
            return
        try:
            if run is not None and (model.tokens is not None or model.cost_usd is not None):
                self._store(lambda: _save_run(session, run))
            self._store(
                lambda: self._emit_model(session, evaluation, classification, model, run)
            )
        except _EvidenceStoreUnavailable:
            if not self._is_degraded(evaluation.envelope.purpose):
                raise
            self._abandon(session)
            self._enqueue(
                _DeferredEvidence(
                    evaluation=evaluation,
                    classification=classification,
                    decided_at=decided_at,
                    decision_budget=decision_budget,
                    run=_copy_run(run) if run is not None else None,
                    save_run=run is not None,
                    model=model,
                )
            )

    def _bind_run(
        self, session: Session, headers: Mapping[str, str], run: Run | None
    ) -> Run | None:
        if run is not None:
            return run
        run_id = headers.get("run_id")
        if run_id:
            try:
                return _load_run(session, run_id)
            except ApprovalError as exc:
                raise _RunNotFound(str(exc)) from exc
            except Exception as exc:
                raise _EvidenceStoreUnavailable(str(exc)) from exc
        return self.run

    def _record(
        self,
        session: Session,
        evaluation: Evaluation,
        classification: Classification,
        run: Run | None,
        now: datetime,
    ) -> Evaluation:
        budget = _budget_payload(run) if run is not None else None
        return self._store(
            lambda: record(
                session,
                evaluation,
                evidence=self.evidence,
                classification=classification,
                budget=budget,
                now=now,
            )
        )

    def _emit(
        self,
        session: Session,
        evaluation: Evaluation,
        *,
        classification: Classification,
        decision: EgressDecision,
        token_map: dict[str, str],
        sent: bool,
        sent_body: bytes,
        actor_type: ActorType,
        actor_id: str,
        response_record: dict[str, Any] | None,
        tokens: int | None,
        cost_usd: float | None,
        run: Run | None,
        received_text: str = "",
        response_payload: Mapping[str, Any] | None = None,
        prompt_template_version: str | None = None,
    ) -> None:
        local = _is_local(self.upstream)
        requested = evaluation.envelope.subject_id or ""
        use_case = evaluation.envelope.purpose
        model_name, model_version = _reported_model(requested, response_payload if sent else None)
        provider = self.provider or (urlparse(self.upstream).hostname or self.upstream)
        store = RetentionStore()
        prompt_hash = ""
        response_hash = ""
        if sent and sent_body:
            prompt_hash = store.retain_text(
                session,
                sent_body.decode("utf-8"),
                kind="prompt",
                use_case=use_case,
                correlation_id=evaluation.request_id,
            )
        if sent and received_text:
            response_hash = store.retain_text(
                session,
                received_text,
                kind="response",
                use_case=use_case,
                correlation_id=evaluation.request_id,
            )
        common: dict[str, Any] = {
            "destination": self.upstream,
            "destination_is_local": local,
            "classification": classification.value,
            "decision": decision.value,
            "redacted_spans": len(token_map),
            "manifest_hash": canonical_hash(sorted(token_map)) if sent else "",
            "sent": sent,
            "response_hash": response_hash,
        }
        if response_record is not None:
            common["response_classifier"] = response_record
        if tokens is not None:
            common["tokens"] = tokens
        if cost_usd is not None:
            common["cost_usd"] = cost_usd
        if run is not None:
            common["budget"] = _budget_payload(run)
        self._store(
            lambda: self.evidence.emit(
                session,
                correlation_id=evaluation.request_id,
                event_type=EventType.MODEL_CALL,
                actor_type=actor_type,
                actor_id=actor_id,
                classification=classification,
                payload={
                    **common,
                    "sent_bytes": len(sent_body),
                    "provider": provider,
                    "model": model_name,
                    "model_version": model_version,
                    "prompt_template_version": prompt_template_version,
                    "prompt_hash": prompt_hash,
                },
            )
        )
        self._store(
            lambda: self.evidence.emit(
                session,
                correlation_id=evaluation.request_id,
                event_type=EventType.EGRESS,
                actor_type=actor_type,
                actor_id=actor_id,
                classification=classification,
                payload={
                    **common,
                    "note": "Payload content is not copied to the evidence plane.",
                },
            )
        )


def _actor_type(agent_name: str | None, trigger: str | None) -> ActorType:
    if agent_name:
        return ActorType.AGENT
    return ActorType.SYSTEM


def _actor_id(agent_name: str | None, trigger: str | None) -> str:
    return agent_name or trigger or "gateway"


__all__ = [
    "ClientConfiguration",
    "Gateway",
    "GatewayResponse",
]
