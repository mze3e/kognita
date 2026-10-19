"""The MCP proxy — an explicit front for one or more MCP servers.

``kognita serve --mcp --root-config <config>`` binds this process. Agents
speak MCP to Kognita. Kognita speaks MCP to the backend servers named in the
config. The config also names the policy pack, the evidence database, and the
default actor context.

One call, in order:

1. Read the JSON-RPC body just enough to build an
   :class:`~kognita.envelope.Envelope`. Principal and purpose come from
   request headers or the default actor context. For ``tools/call``, ``tool``
   is the tool name and ``arguments`` are the tool arguments.
2. Classify those arguments through
   :func:`kognita.governance.classifier_derived_envelope`, which
   :func:`kognita.tools.run_governed` already applies. A typed classification
   is a floor, not an override: the classifier still runs and can only raise
   it. The classifier does not decide.
3. Resolve identity with the same rules as the AI gateway. An agent name
   is admitted only when the bound client configuration lists it and its
   workload credential verifies. With no agent name, a system trigger listed
   in that client configuration passes the credential check; all other
   governance checks still apply. Unlisted triggers are denied.
4. :func:`kognita.tools.run_governed` authorises and records the call. The
   backend is contacted only when that call is released: an allow, or a
   human approval that already has a live grant. A denial, an escalation,
   and an ungranted human approval return the outcome and the citations
   and do not call the backend. A released call returns the proxied result.
5. If the evidence store cannot record the call, the proxy refuses it. The
   backend is not called. This module has no degraded mode and no silent
   forward.

Notifications are not calls and are not forwarded. A method this proxy does
not implement is not forwarded either.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from sqlmodel import Session

from kognita.authority import _argument_values
from kognita.classify import PatternClassifier, most_sensitive
from kognita.db import create_all, make_engine, session_scope
from kognita.envelope import Check, Envelope, Evaluation
from kognita.evidence import EvidenceWriter
from kognita.exceptions import ConfigError, KognitaError
from kognita.gateway import ClientConfiguration, _bound_principal, _identity, _credential_checks
from kognita.tools import ToolRegistry, run_governed
from kognita.vocabulary import CheckResult, Classification, Outcome

_IMPLEMENTED = frozenset({"initialize", "ping", "tools/list", "tools/call"})
_PROTOCOL_VERSION = "2025-03-26"

Transport = Callable[[str, str, Mapping[str, str], bytes], tuple[int, Mapping[str, str], bytes]]


class _EvidenceStoreUnavailable(KognitaError):
    """Raised when a call cannot be written to the evidence store."""

    def __init__(self, message: str = "evidence store unavailable", *, rpc_id: Any = None) -> None:
        super().__init__(message)
        self.rpc_id = rpc_id


class _BackendUnavailable(KognitaError):
    """A named backend server could not be reached."""


class _InvalidParams(KognitaError):
    """The call was authorised, but it does not name a backend tool."""


class _FailClosedEvidence:
    """Turns an evidence-write failure into a refusal the proxy can catch.

    The decision write and the tool evidence both go through ``emit``. A
    failure there must not become a forward.
    """

    def __init__(self, inner: EvidenceWriter) -> None:
        self._inner = inner
        self.clock = inner.clock

    def emit(self, session: Session, **kwargs: Any) -> Any:
        try:
            return self._inner.emit(session, **kwargs)
        except _EvidenceStoreUnavailable:
            raise
        except Exception as exc:
            raise _EvidenceStoreUnavailable("evidence store unavailable") from exc


class _BackendPayload:
    """A backend JSON-RPC result or error, after the call was allowed."""

    def __init__(self, result: Any = None, error: Any = None) -> None:
        self.result = result
        self.error = error


@dataclass(frozen=True)
class BackendServer:
    """One MCP server the proxy fronts. ``url`` is that server's endpoint."""

    name: str
    url: str


@dataclass(frozen=True)
class RootConfig:
    """What ``--root-config`` names: servers, pack, evidence, actor.

    ``purposes`` is the list the proxy may allow. It is not ``actor.purpose``.
    Missing or empty denies every call.
    """

    servers: tuple[BackendServer, ...]
    policy_pack: str
    pack: Any
    evidence_database: str
    actor: ClientConfiguration
    purposes: tuple[str, ...] = ()


@dataclass
class McpResponse:
    """What the proxy returns to the caller. ``evaluation`` is set once one was recorded."""

    status: int
    body: bytes
    headers: dict[str, str]
    evaluation: Evaluation | None = None


def _headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): "" if value is None else str(value) for key, value in headers.items()}


def _json_response(
    status: int,
    payload: Mapping[str, Any] | None,
    *,
    evaluation: Evaluation | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> McpResponse:
    if payload is None:
        body = b""
        headers = dict(extra_headers or {})
    else:
        body = json.dumps(payload).encode("utf-8")
        headers = {"content-type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
    return McpResponse(status=status, body=body, headers=headers, evaluation=evaluation)


def _rpc_error(
    rpc_id: Any,
    *,
    code: int,
    message: str,
    data: Mapping[str, Any] | None = None,
    status: int = 400,
    evaluation: Evaluation | None = None,
) -> McpResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = dict(data)
    return _json_response(
        status,
        {"jsonrpc": "2.0", "id": rpc_id, "error": error},
        evaluation=evaluation,
    )


def _admitted_trigger(trigger: str | None, checks: Sequence[Check]) -> str | None:
    """The trigger identity admitted, or nothing.

    A failing trigger check must not be recorded as a system actor. Only the
    passing check from :func:`kognita.gateway._identity` counts.
    """
    if not trigger:
        return None
    for check in checks:
        if check.check == "SYSTEM_TRIGGER" and check.result is CheckResult.PASS:
            return trigger
    return None


class _AttributePack:
    """The caller's pack, with a typed classification written over the gap.

    The typed value is a floor for both the pack's own attribute and the
    classifier, never an override: :func:`kognita.tools.run_governed`
    still classifies free-text arguments and can only raise this value.
    """

    def __init__(self, inner: Any, classification: str | None) -> None:
        self._inner = inner
        self._classification = classification
        self.name = getattr(inner, "name", "mcp")

    def load_subjects(self, envelope: Envelope, session: Any) -> dict[str, Any]:
        loaded = self._inner.load_subjects(envelope, session)
        return dict(loaded or {})

    def resolve_attributes(self, envelope: Envelope, subjects: dict[str, Any]) -> dict[str, Any]:
        attrs = dict(self._inner.resolve_attributes(envelope, subjects) or {})
        if self._classification:
            # The header is a floor: it may raise the pack's label, never lower it.
            packed = attrs.get("classification")
            attrs["classification"] = (
                most_sensitive([packed, self._classification]).value
                if packed else self._classification
            )
        return attrs

    def rules(self) -> dict[str, Any]:
        return self._inner.rules()

    def engages(self, policy: Any, context: Any) -> bool:
        engages = getattr(self._inner, "engages", None)
        if engages is None:
            return True
        return bool(engages(policy, context))


def _load_pack(spec: str) -> Any:
    module_name, separator, attr = spec.partition(":")
    if not separator or not module_name or not attr:
        raise ConfigError("policy_pack must be 'module:attribute'")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"policy_pack {spec!r} could not be imported") from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise ConfigError(f"policy_pack {spec!r} has no attribute {attr!r}") from exc
    if isinstance(obj, type):
        try:
            obj = obj()
        except Exception as exc:
            raise ConfigError(f"policy_pack {spec!r} could not be constructed") from exc
    for method in ("load_subjects", "resolve_attributes", "rules"):
        if not callable(getattr(obj, method, None)):
            raise ConfigError(f"policy_pack {spec!r} has no {method}")
    return obj


def _actor_from_config(value: Any) -> ClientConfiguration:
    if not isinstance(value, dict):
        raise ConfigError("actor must be an object")
    names = value.get("agent_names") or []
    triggers = value.get("system_triggers") or []
    if isinstance(names, str) or not isinstance(names, (list, tuple)):
        raise ConfigError("actor.agent_names must be a list")
    if isinstance(triggers, str) or not isinstance(triggers, (list, tuple)):
        raise ConfigError("actor.system_triggers must be a list")
    agent_name = value.get("agent_name")
    system_trigger = value.get("system_trigger")
    if agent_name is not None and not isinstance(agent_name, str):
        raise ConfigError("actor.agent_name must be a string")
    if system_trigger is not None and not isinstance(system_trigger, str):
        raise ConfigError("actor.system_trigger must be a string")
    return ClientConfiguration(
        principal=str(value.get("principal") or ""),
        purpose=str(value.get("purpose") or ""),
        agent_names=frozenset(str(item) for item in names),
        system_triggers=frozenset(str(item) for item in triggers),
        actor_location=str(value.get("actor_location") or ""),
        agent_name=agent_name or None,
        system_trigger=system_trigger or None,
    )


def _servers_from_config(value: Any) -> tuple[BackendServer, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("servers must name one or more backend servers")
    servers: list[BackendServer] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ConfigError("each backend server must be an object")
        name = item.get("name")
        url = item.get("url")
        if not isinstance(name, str) or not name:
            raise ConfigError("each backend server needs a name")
        if name in seen:
            raise ConfigError(f"backend server {name!r} is named more than once")
        if not isinstance(url, str) or "://" not in url:
            raise ConfigError(f"backend server {name!r} needs a url")
        seen.add(name)
        servers.append(BackendServer(name=name, url=url))
    return tuple(servers)


def _purposes_from_config(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """The purpose list, when the file names one.

    A missing field is an empty list. A single string is not a list: that
    would turn ``actor.purpose``'s shape into an allowlist.
    """
    if "purposes" not in raw or raw["purposes"] is None:
        return ()
    value = raw["purposes"]
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ConfigError("purposes must be a list")
    return tuple(str(item) for item in value)


def load_root_config(path: str) -> RootConfig:
    """Read the root config: servers, policy pack, evidence database, actor.

    ``purposes``, when present, constrains the purpose header. ``actor.purpose``
    stays the purpose claimed when a request has none. Authorization is the
    use-case register, and an empty register denies the call.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise ConfigError(f"root config {path!r} could not be read") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"root config {path!r} is not JSON") from exc
    if not isinstance(raw, dict):
        raise ConfigError("root config must be a JSON object")
    missing = [
        key
        for key in ("servers", "policy_pack", "evidence_database", "actor")
        if key not in raw
    ]
    if missing:
        raise ConfigError("root config must name " + ", ".join(missing))
    policy_pack = raw["policy_pack"]
    evidence_database = raw["evidence_database"]
    if not isinstance(policy_pack, str) or not policy_pack:
        raise ConfigError("policy_pack must be 'module:attribute'")
    if not isinstance(evidence_database, str) or not evidence_database:
        raise ConfigError("evidence_database must be a path or URL")
    return RootConfig(
        servers=_servers_from_config(raw["servers"]),
        policy_pack=policy_pack,
        pack=_load_pack(policy_pack),
        evidence_database=evidence_database,
        actor=_actor_from_config(raw["actor"]),
        purposes=_purposes_from_config(raw),
    )


class _HttpCall:
    """A connection owned by one governed invocation, independent of RPC ids."""

    def __init__(self):
        from threading import Lock
        self._lock = Lock()
        self._connection = None
        self._socket = None
        self._stopped = False

    def _cancel(self, _rpc_id):
        import socket
        with self._lock:
            self._stopped = True
            connection = self._connection
            if connection is not None and self._socket is not None:
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        return 'connection_aborted'

    def __call__(self, method, url, headers, body):
        from http.client import HTTPConnection, HTTPSConnection
        from urllib.parse import urlsplit
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https'):
            raise _BackendUnavailable('unsupported backend URL')
        connection = (HTTPSConnection if parsed.scheme == 'https' else HTTPConnection)(
            parsed.hostname, parsed.port, timeout=60)
        try:
            connection.connect()
            connection.auto_open = 0
            with self._lock:
                if self._stopped:
                    raise _BackendUnavailable('cancelled')
                self._connection = connection
                self._socket = connection.sock
            path = parsed.path or '/'
            if parsed.query:
                path += '?' + parsed.query
            connection.request(method, path, body=body, headers=dict(headers))
            response = connection.getresponse()
            return response.status, dict(response.headers), response.read()
        except Exception as exc:
            raise _BackendUnavailable(str(exc)) from exc
        finally:
            connection.close()


def _urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes
) -> tuple[int, Mapping[str, str], bytes]:
    return _HttpCall()(method, url, headers, body)


def _package_version() -> str:
    from kognita import __version__

    return __version__


class McpProxy:
    """MCP server that fronts the backends named in its config.

    ``transport`` replaces the network in tests. The default speaks JSON-RPC
    over HTTP to each backend ``url``.
    """

    def __init__(
        self,
        *,
        engine: Any,
        evidence: EvidenceWriter,
        servers: Sequence[BackendServer],
        client: ClientConfiguration,
        pack: Any,
        purposes: Sequence[str] = (),
        classifier: Any | None = None,
        transport: Transport | None = None,
    ) -> None:
        if not servers:
            raise ConfigError("servers must name one or more backend servers")
        self.engine = engine
        self.evidence = _FailClosedEvidence(evidence)
        self.servers = tuple(servers)
        self._by_name = {server.name: server for server in self.servers}
        self.client = client
        self.pack = pack
        self.purposes = purposes
        self.classifier = classifier if classifier is not None else PatternClassifier()
        self.transport = transport if transport is not None else _urllib_transport

    @classmethod
    def from_config(
        cls,
        config: RootConfig,
        *,
        transport: Transport | None = None,
        classifier: Any | None = None,
    ) -> McpProxy:
        engine = make_engine(config.evidence_database)
        create_all(engine)
        return cls(
            engine=engine,
            evidence=EvidenceWriter(engine),
            servers=config.servers,
            client=config.actor,
            pack=config.pack,
            purposes=config.purposes,
            transport=transport,
            classifier=classifier,
        )

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Serve MCP JSON-RPC at ``http://{host}:{port}/`` until the process stops."""
        self._make_server(host, port).serve_forever()

    def _make_server(self, host: str, port: int) -> Any:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                try:
                    length = int(self.headers.get("Content-Length", "0") or "0")
                except ValueError:
                    length = 0
                body = self.rfile.read(length) if length > 0 else b""
                result = proxy.handle(self.command, self.path, self.headers, body)
                payload = result.body
                self.send_response(result.status)
                for key, value in result.headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if payload:
                    self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                self._method_not_allowed()

            def do_PUT(self) -> None:  # noqa: N802
                self._method_not_allowed()

            def _method_not_allowed(self) -> None:
                body = b'{"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"the MCP proxy only accepts POST"}}'
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
    ) -> McpResponse:
        """Govern one MCP request. A supplied ``session`` is left for the caller to commit."""
        if session is None:
            try:
                with session_scope(self.engine) as owned:
                    return self._handle(owned, method, path, headers, body)
            except _EvidenceStoreUnavailable as exc:
                return self._unavailable(exc.rpc_id)
        try:
            return self._handle(session, method, path, headers, body)
        except _EvidenceStoreUnavailable as exc:
            session.rollback()
            return self._unavailable(exc.rpc_id)

    def _unavailable(self, rpc_id: Any) -> McpResponse:
        return _rpc_error(
            rpc_id,
            code=-32001,
            message="evidence store unavailable",
            status=503,
        )

    def _denied(self, rpc_id: Any, evaluation: Evaluation) -> McpResponse:
        citations = [check.citation for check in evaluation.basis()]
        return _rpc_error(
            rpc_id,
            code=-32003,
            message=evaluation.outcome.value,
            data={
                "outcome": evaluation.outcome.value,
                "citations": citations,
                "request_id": evaluation.request_id,
                "checks": [check.to_dict() for check in evaluation.basis()],
            },
            status=403,
            evaluation=evaluation,
        )

    def _handle(
        self,
        session: Session,
        method: str,
        path: str,
        raw_headers: Mapping[str, str],
        body: bytes,
    ) -> McpResponse:
        del path
        if method.upper() != "POST":
            return _rpc_error(None, code=-32600, message="the MCP proxy only accepts POST", status=405)
        try:
            message = json.loads(body.decode("utf-8")) if body else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _rpc_error(None, code=-32700, message="request body is not JSON")
        if isinstance(message, list):
            return _rpc_error(None, code=-32600, message="batch requests are not accepted")
        if not isinstance(message, dict):
            return _rpc_error(None, code=-32600, message="request is not a JSON-RPC object")
        rpc_method = message.get("method")
        if not isinstance(rpc_method, str) or not rpc_method:
            return _rpc_error(message.get("id"), code=-32600, message="request has no method")
        if "id" not in message:
            return McpResponse(status=202, body=b"", headers={})
        rpc_id = message.get("id")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return _rpc_error(rpc_id, code=-32602, message="params must be an object")

        headers = _headers(raw_headers)
        typed: str | None = None
        if headers.get("classification"):
            try:
                typed = Classification(headers["classification"]).value
            except ValueError:
                return _rpc_error(rpc_id, code=-32602, message="classification is not a known value")

        if rpc_method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name:
                return _rpc_error(rpc_id, code=-32602, message="tools/call requires a name")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return _rpc_error(rpc_id, code=-32602, message="tools/call arguments must be an object")
            envelope_tool = tool_name
            envelope_arguments = arguments
        else:
            envelope_tool = rpc_method
            envelope_arguments = {}

        agent_name, trigger, identity_checks = _identity(self.client, headers)
        principal, purpose, principal_checks = _bound_principal(
            self.client, headers, self.purposes or ()
        )
        context: dict[str, Any] = {}
        admitted = _admitted_trigger(trigger, identity_checks)
        identity_checks = identity_checks + principal_checks + (
            [] if admitted else _credential_checks(session, agent_name, headers)
        )
        if admitted:
            context["system_trigger"] = admitted
        from kognita._isolation import _assurance
        # Preserve the argument-derived subject reference for pack loaders.
        # decide resolves the complete effective scope, including manifest
        # mappings, for all governance consumers.
        argument_clients = _argument_values(envelope_arguments, "client")
        subjects = {"client": next(iter(argument_clients))} if argument_clients and len(argument_clients) == 1 else {}
        envelope = Envelope(
            principal=principal,
            purpose=purpose,
            tool=envelope_tool,
            actor_location=self.client.actor_location,
            agent_name=agent_name,
            delegation_id=headers.get("delegation_id"),
            assurance_level=_assurance(headers),
            subjects=subjects,
            arguments=dict(envelope_arguments),
            context={**context, "_agent_credential": headers.get("agent_credential"),
                     "_require_client_run": True},
        )

        state = {"proxied": False}
        lifecycle = {}
        registry = ToolRegistry()

        def invoke(env: Envelope, evaluation: Evaluation, sess: Session) -> Any:
            from kognita._actions import _bounded
            from kognita.vocabulary import ActorType, EventType
            transport = _HttpCall() if self.transport is _urllib_transport else self.transport
            state['_transport'] = transport
            cancel = getattr(transport, '_cancel', None)
            seconds = permission.get('max_duration', 30.0)
            if bound_run is not None and bound_run.wall_clock_seconds is not None:
                from kognita.models import as_utc
                seconds = min(seconds, max(0, bound_run.wall_clock_seconds -
                    (as_utc(self.evidence.clock()) - as_utc(bound_run.started_at)).total_seconds()))
            from kognita.models import _Authority
            authority_ids = set(evaluation.attributes.get('_authority_grants', []))
            identity = env.delegation_id
            for _ in range(32):
                if not identity or identity in authority_ids:
                    break
                authority_ids.add(identity)
                row = sess.get(_Authority, identity)
                identity = row.body.get('_parent') if row else None
            # Persist admission before releasing external work. Holding SQLite's
            # write transaction during a backend call would block interventions.
            sess.commit()
            def live():
                from kognita.models import Agent, RunRecord
                from sqlmodel import select
                from kognita.credentials import verify
                current = sess.exec(select(Agent).where(Agent.name == agent_name)
                    .execution_options(populate_existing=True)).first()
                if not current or current.kill_switch:
                    return False
                if not verify(sess, agent_name, headers.get('agent_credential', '')):
                    return False
                if bound_run is not None:
                    row = sess.get(RunRecord, bound_run.id, populate_existing=True)
                    if row is None or row.control_state != 'live':
                        return False
                for identity in authority_ids:
                    row = sess.get(_Authority, identity, populate_existing=True)
                    if row is not None and row.ended:
                        return False
                return True
            outcome, value = _bounded(
                lambda: self._invoke(rpc_method, params, rpc_id, headers, state),
                seconds=seconds, agent=agent_name,
                run=headers.get('run_id'), delegation=envelope.delegation_id,
                cancel=(lambda: cancel(rpc_id)) if callable(cancel) else None,
                _authority=authority_ids, _check=live, _lifecycle=lifecycle)
            if bound_run is not None:
                _refresh_run(sess, bound_run)
            state['_outcome'] = outcome
            self.evidence.emit(sess, correlation_id=evaluation.request_id,
                event_type=EventType.TOOL_CALL, actor_type=ActorType.AGENT,
                actor_id=agent_name, payload={'tool': env.tool, 'outcome': outcome,
                    'delegation_id': env.delegation_id,
                    'run_id': headers.get('run_id'), 'citation': 'S6C-IN-FLIGHT'})
            if outcome == 'failed':
                raise value
            if outcome != 'completed':
                return _BackendPayload(error={'code': -32002, 'message': outcome})
            if bound_run is None and env.delegation_id:
                from kognita.canonical import canonical_json
                state["_sensitive_response"] = self.classifier.classify(canonical_json(value)) in (
                    Classification.C2, Classification.C3)
            return value

        from kognita.registry import get
        agent = get(session, agent_name) if agent_name else None
        permission = (agent.tool_allowlist or {}).get(envelope.tool, {}) if agent else {}
        # Synchronous transports without confirmed cancellation require rollback
        # for mutating operations. Read calls have no action to undo.
        if (permission.get('verb', 'read') in ('draft', 'queue', 'send', 'execute')
                and (self.transport is _urllib_transport
                     or not callable(getattr(self.transport, '_cancel', None)))
                and not (permission.get('reversible') is True and isinstance(permission.get('compensation'), str)
                         and permission['compensation'].strip())):
            identity_checks.append(Check(check='S6C-UNCANCELLABLE', regime='INTERNAL',
                result=CheckResult.FAIL, citation='S6C-UNCANCELLABLE'))

        def compensate(env, evaluation, sess):
            from kognita._actions import _bounded
            status, outcome = _bounded(lambda: self._invoke('tools/call',
                {'name': permission['compensation'], 'arguments': dict(env.arguments)},
                rpc_id, headers, {'proxied': False}), seconds=permission.get('max_duration', 30.0),
                agent=agent_name, run=None, delegation=None)
            if status != 'completed':
                raise _BackendUnavailable('compensation did not complete')
            if isinstance(outcome, _BackendPayload) and outcome.error is not None:
                raise _BackendUnavailable('compensation failed')
            return None

        if permission.get('reversible') and not (
                isinstance(permission.get('compensation'), str) and permission['compensation'].strip()):
            identity_checks.append(Check(check='S6C-REVERSIBILITY', regime='INTERNAL',
                result=CheckResult.FAIL, citation='S6C-REVERSIBILITY'))
        try:
            registry.register(envelope.tool, invoke,
                required_level=permission.get('required_level', 0), verb=permission.get('verb', 'read'),
                external=permission.get('external', False), client_data=permission.get('client_data', False),
                public_content=permission.get('public_content', False),
                reversible=permission.get('reversible', False),
                compensation=compensate if permission.get('reversible') else None,
                max_duration=permission.get('max_duration', 30.0))
        except ValueError:
            identity_checks.append(Check(check='S6C-TOOL-METADATA', regime='INTERNAL',
                result=CheckResult.FAIL, citation='S6C-TOOL-METADATA'))
            registry.register(envelope.tool, invoke)
        registry.get(envelope.tool)._backend_lifecycle = lifecycle
        registry.get(envelope.tool)._compensation_name = permission.get('compensation')
        from kognita.tools import Run, _refresh_run
        bound_run = Run(id=headers['run_id']) if headers.get('run_id') else None
        if bound_run is not None:
            from kognita.models import RunRecord, _RunScope
            scope = session.get(_RunScope, bound_run.id, populate_existing=True)
            identity = dict(agent=envelope.agent_name, human=envelope.principal,
                            purpose=envelope.purpose, delegation_id=envelope.delegation_id)
            if (session.get(RunRecord, bound_run.id) is None or scope is None
                    or any(scope.body.get(k) != v for k, v in identity.items())):
                identity_checks.append(Check(check='S6B-RUN-ISOLATION', regime='INTERNAL',
                    result=CheckResult.FAIL, citation='S6B-RUN-ISOLATION'))
                bound_run = None
            else:
                _refresh_run(session, bound_run)
        now = self.evidence.clock()
        try:
            result = run_governed(
                session,
                envelope,
                registry=registry,
                evidence=self.evidence,  # type: ignore[arg-type]
                pack=_AttributePack(self.pack, typed),
                purposes=self.purposes,
                as_of=now,
                now=now,
                classifier=self.classifier,
                run=bound_run,
                extra_checks=identity_checks,
            )
        except _BackendUnavailable:
            return _rpc_error(rpc_id, code=-32002, message="backend unavailable", status=502)
        except _InvalidParams as exc:
            return _rpc_error(rpc_id, code=-32602, message=str(exc))
        except _EvidenceStoreUnavailable as exc:
            exc.rpc_id = rpc_id
            raise
        except Exception as exc:
            if state["proxied"]:
                raise
            raise _EvidenceStoreUnavailable("evidence store unavailable", rpc_id=rpc_id) from exc

        evaluation = result.evaluation
        if state.get("_sensitive_response"):
            from kognita.tools import _response_outcome
            evaluation = _response_outcome(session, evaluation, [Check(
                check="S5-RUN-REQUIRED", citation="S5-RUN-REQUIRED",
                regime="INTERNAL", result=CheckResult.FAIL)], self.evidence)
        # Release is run_governed's decision, not "outcome is ALLOW".
        # Evaluation.allowed includes an ungranted HUMAN_APPROVAL, and a live
        # grant leaves the outcome as HUMAN_APPROVAL after the tool has run.
        released = evaluation.allowed and not result.approval_required
        if not released:
            return self._denied(rpc_id, evaluation)
        if state.get('_outcome') in ('timeout', 'cancelled', 'cancellation_requested', 'connection_aborted'):
            return _rpc_error(rpc_id, code=-32002, message=state['_outcome'], status=502, evaluation=evaluation)
        if rpc_method not in _IMPLEMENTED:
            return _rpc_error(
                rpc_id,
                code=-32601,
                message="method not found",
                status=200,
                evaluation=evaluation,
            )
        return self._succeeded(rpc_id, result.data, evaluation)

    def _succeeded(self, rpc_id: Any, data: Any, evaluation: Evaluation) -> McpResponse:
        if isinstance(data, _BackendPayload):
            if data.error is not None:
                error = data.error if isinstance(data.error, dict) else {"code": -32000, "message": str(data.error)}
                return _json_response(
                    200,
                    {"jsonrpc": "2.0", "id": rpc_id, "error": error},
                    evaluation=evaluation,
                )
            data = data.result
        return _json_response(
            200,
            {"jsonrpc": "2.0", "id": rpc_id, "result": data},
            evaluation=evaluation,
        )

    def _invoke(
        self,
        rpc_method: str,
        params: Mapping[str, Any],
        rpc_id: Any,
        headers: Mapping[str, str],
        state: dict[str, Any],
    ) -> Any:
        if rpc_method == "initialize":
            return self._initialize_result(params)
        if rpc_method == "ping":
            return {}
        if rpc_method == "tools/list":
            return self._tools_list(rpc_id, headers, state)
        if rpc_method == "tools/call":
            return self._tools_call(params, rpc_id, headers, state)
        return None

    def _initialize_result(self, params: Mapping[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        version = requested if isinstance(requested, str) and requested else _PROTOCOL_VERSION
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "kognita", "version": _package_version()},
        }

    def _route(self, tool_name: str) -> tuple[BackendServer, str]:
        if "/" in tool_name:
            server_name, _, bare = tool_name.partition("/")
            server = self._by_name.get(server_name)
            if server is None or not bare:
                raise _InvalidParams(f"no backend server named in {tool_name!r}")
            return server, bare
        if len(self.servers) == 1:
            return self.servers[0], tool_name
        raise _InvalidParams(
            f"{tool_name!r} does not name which backend server to call"
        )

    def _tools_call(
        self,
        params: Mapping[str, Any],
        rpc_id: Any,
        headers: Mapping[str, str],
        state: dict[str, Any],
    ) -> _BackendPayload:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str):
            raise _InvalidParams("tools/call requires a name")
        server, bare = self._route(name)
        payload = self._rpc(
            server,
            "tools/call",
            {"name": bare, "arguments": arguments},
            rpc_id,
            headers,
            state,
        )
        return payload

    def _tools_list(
        self,
        rpc_id: Any,
        headers: Mapping[str, str],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        tools: list[Any] = []
        several = len(self.servers) > 1
        for server in self.servers:
            payload = self._rpc(server, "tools/list", {}, rpc_id, headers, state)
            if payload.error is not None:
                raise _BackendUnavailable("backend tools/list failed")
            result = payload.result if isinstance(payload.result, dict) else {}
            listed = result.get("tools") if isinstance(result, dict) else None
            if not isinstance(listed, list):
                continue
            for tool in listed:
                if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                    continue
                item = dict(tool)
                if several:
                    item["name"] = f"{server.name}/{tool['name']}"
                tools.append(item)
        return {"tools": tools}

    def _rpc(
        self,
        server: BackendServer,
        method: str,
        params: Mapping[str, Any],
        rpc_id: Any,
        headers: Mapping[str, str],
        state: dict[str, Any],
    ) -> _BackendPayload:
        body = json.dumps(
            {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": dict(params)}
        ).encode("utf-8")
        forward = {"content-type": "application/json", "accept": "application/json"}
        state["proxied"] = True
        try:
            _status, _response_headers, raw = state.get("_transport", self.transport)("POST", server.url, forward, body)
        except _BackendUnavailable:
            raise
        except Exception as exc:
            raise _BackendUnavailable(str(exc)) from exc
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _BackendUnavailable("backend response is not JSON") from exc
        if not isinstance(parsed, dict):
            raise _BackendUnavailable("backend response is not JSON")
        if parsed.get("error"):
            return _BackendPayload(error=parsed["error"])
        return _BackendPayload(result=parsed.get("result"))


__all__ = [
    "BackendServer",
    "McpProxy",
    "McpResponse",
    "RootConfig",
    "load_root_config",
]
