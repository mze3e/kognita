"""Gateway overhead on ``kognita serve``.

An allowed ``POST /v1/chat/completions`` is timed through the gateway
``cmd_serve`` builds. Overhead is the call's wall time minus
``PatternClassifier`` inference and minus the time inside the local stand-in
that answers for the upstream. The test fails when that overhead is not under
50 ms.
"""
from __future__ import annotations

import json
import os
import platform
import threading
import time
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from sqlmodel import Session

from kognita.classify import PatternClassifier
from kognita.cli import build_parser, cmd_serve
from kognita.db import create_all, make_engine
from kognita.gateway import Gateway, _urllib_transport
from kognita.registry import register
from kognita.vocabulary import Classification

SECRET = "ana@example.org"
MODEL = "gateway-model"
PATH = "/v1/chat/completions"
UPSTREAM = "https://api.openai.com"
PROMPT = f"Email {SECRET} about the notes"
CALLS = 30
LIMIT_S = 0.050
_RESPONSE = json.dumps(
    {
        "model": MODEL,
        "choices": [
            {"message": {"role": "assistant", "content": "Reply to [EMAIL_1] tomorrow"}}
        ],
    }
).encode()
_BODY = json.dumps(
    {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}]}
).encode()


class _TimedClassifier(PatternClassifier):
    """The pattern classifier, plus the time spent inside its inference."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.inference_s = 0.0

    def classify(
        self, text: str, *, hint: Classification | None = None
    ) -> Classification:
        started = time.perf_counter()
        try:
            return super().classify(text, hint=hint)
        finally:
            self.inference_s += time.perf_counter() - started

    def calibrated_confidence(
        self, text: str, *, label: Classification | None = None
    ) -> float:
        started = time.perf_counter()
        try:
            return super().calibrated_confidence(text, label=label)
        finally:
            self.inference_s += time.perf_counter() - started


class _StandIn(BaseHTTPRequestHandler):
    """A fixed completion. ``model_s`` is time spent producing it."""

    bodies: list[bytes] = []
    model_s = 0.0

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        type(self).bodies.append(raw)
        started = time.perf_counter()
        payload = _RESPONSE
        type(self).model_s += time.perf_counter() - started
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _machine() -> str:
    cpu = platform.machine()
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    memory = "unknown"
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        meminfo = ""
    for line in meminfo.splitlines():
        if line.startswith("MemTotal"):
            memory = line.split(":", 1)[1].strip()
            break
    return (
        f"{platform.system()} {platform.release()} {platform.machine()}; "
        f"{os.cpu_count()} CPUs; {cpu}; MemTotal {memory}"
    )


def _seed(path: Path) -> None:
    engine = make_engine(path)
    create_all(engine)
    with Session(engine) as session:
        register(session, name="dossier-agent", owner_exec="Head of Research Ops")
        session.commit()
    engine.dispose()


def _listen(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _stop(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def test_gateway_overhead_is_under_50ms_excluding_classifier_inference(
    tmp_path: Path, monkeypatch
) -> None:
    """Each measured call stays under 50 ms after classifier and stand-in time."""
    _StandIn.bodies = []
    _StandIn.model_s = 0.0
    stand_in = ThreadingHTTPServer(("127.0.0.1", 0), _StandIn)
    stand_in_thread = _listen(stand_in)
    origin = f"http://127.0.0.1:{stand_in.server_address[1]}"
    db = tmp_path / "gateway.db"
    _seed(db)
    held: dict[str, Gateway] = {}

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        held["gateway"] = self

    monkeypatch.setattr(Gateway, "serve", serve)
    argv = [
        "serve",
        "--provider",
        "openai-compatible",
        "--upstream",
        UPSTREAM,
        "--db",
        str(db),
        "--principal",
        "alice",
        "--purpose",
        "COLLABORATION",
        "--purposes",
        "COLLABORATION",
        "--actor-location",
        "SG",
        "--agent",
        "dossier-agent",
    ]
    assert cmd_serve(build_parser().parse_args(argv)) == 0
    gateway = held["gateway"]
    assert gateway.upstream == UPSTREAM
    clock = _TimedClassifier()
    gateway.classifier = clock

    def transport(method, url, headers, body):
        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.hostname == "api.openai.com"
        forwarded = origin + parsed.path
        return _urllib_transport(method, forwarded, headers, body)

    gateway.transport = transport
    server = gateway._make_server("127.0.0.1", 0)
    thread = _listen(server)
    port = server.server_address[1]
    overheads: list[float] = []
    inference: list[float] = []
    upstream: list[float] = []
    try:
        for _ in range(CALLS):
            inference_before = clock.inference_s
            model_before = _StandIn.model_s
            started = time.perf_counter()
            connection = HTTPConnection("127.0.0.1", port, timeout=5)
            connection.request(
                "POST",
                PATH,
                body=_BODY,
                headers={
                    "agent_name": "dossier-agent",
                    "Content-Length": str(len(_BODY)),
                },
            )
            response = connection.getresponse()
            raw = response.read()
            elapsed = time.perf_counter() - started
            connection.close()
            assert response.status == 200, raw
            assert SECRET.encode() in raw
            spent_inference = clock.inference_s - inference_before
            spent_model = _StandIn.model_s - model_before
            assert spent_inference > 0
            overhead = elapsed - spent_inference - spent_model
            assert overhead >= 0
            overheads.append(overhead)
            inference.append(spent_inference)
            upstream.append(spent_model)
    finally:
        _stop(server, thread)
        _stop(stand_in, stand_in_thread)

    assert len(_StandIn.bodies) == CALLS
    assert all(SECRET.encode() not in body for body in _StandIn.bodies)
    assert all(b"[EMAIL_1]" in body for body in _StandIn.bodies)
    assert len(overheads) == CALLS
    maximum = max(overheads)
    mean = sum(overheads) / len(overheads)
    summary = (
        f"gateway overhead sample={CALLS} "
        f"max_ms={maximum * 1000:.3f} mean_ms={mean * 1000:.3f} "
        f"classifier_mean_ms={sum(inference) / len(inference) * 1000:.3f} "
        f"upstream_mean_ms={sum(upstream) / len(upstream) * 1000:.3f} "
        f"machine={_machine()}"
    )
    print(summary)
    assert maximum < LIMIT_S, summary
