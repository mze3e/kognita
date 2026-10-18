"""Gateway overhead on ``kognita serve``.

An allowed ``POST /v1/chat/completions`` is timed through the gateway
``cmd_serve`` builds. Overhead is the call's wall time minus
``PatternClassifier`` inference and minus the time inside the local stand-in
that answers for the upstream. Warm-up calls are dropped. The test fails
unless the 95th percentile of the sample is under 50 ms and the mean is
under 25 ms.
"""

from __future__ import annotations

import json
import math
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
# Steady state, not the first calls. A larger sample keeps one stalled call
# out of the 95th percentile.
WARMUP = 5
CALLS = 100
LIMIT_S = 0.050
MEAN_LIMIT_S = 0.025
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


def _nearest_rank(samples: list[float], fraction: float) -> float:
    """The value at the nearest rank. One slower sample does not set it."""
    ordered = sorted(samples)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


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
    """The steady sample stays under the overhead caps, after classifier time.

    The first calls are warm-up and are not measured. Thread and database
    startup on a cold server are outside the sample. The 95th percentile
    must stay under 50 ms and the mean under 25 ms.
    """
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

    def _call() -> tuple[float, float, float]:
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
        return overhead, spent_inference, spent_model

    try:
        for _ in range(WARMUP):
            _call()
        _StandIn.bodies.clear()
        _StandIn.model_s = 0.0
        clock.inference_s = 0.0
        for _ in range(CALLS):
            overhead, spent_inference, spent_model = _call()
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
    # A shared runner can stall one call without the gateway getting slower.
    # That maximum is printed and is not the measure. The measure is the
    # 95th percentile and the mean of the sample.
    p95 = _nearest_rank(overheads, 0.95)
    summary = (
        f"gateway overhead sample={CALLS} warmup_dropped={WARMUP} "
        f"p95_ms={p95 * 1000:.3f} mean_ms={mean * 1000:.3f} "
        f"max_ms={maximum * 1000:.3f} "
        f"classifier_mean_ms={sum(inference) / len(inference) * 1000:.3f} "
        f"upstream_mean_ms={sum(upstream) / len(upstream) * 1000:.3f} "
        f"machine={_machine()}"
    )
    print(summary)
    assert p95 < LIMIT_S, summary
    assert mean < MEAN_LIMIT_S, summary


def test_one_slow_call_does_not_set_the_overhead_percentile() -> None:
    """A single stall stays out of the 95th percentile and under the mean cap."""
    stalled = 0.267
    sample = [0.012] * (CALLS - 1) + [stalled]
    assert _nearest_rank(sample, 0.95) < LIMIT_S
    assert max(sample) == stalled
    assert sum(sample) / len(sample) < MEAN_LIMIT_S
