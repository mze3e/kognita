"""Gateway overhead on ``kognita serve``.

An allowed ``POST /v1/chat/completions`` is timed through the gateway
``cmd_serve`` builds. Overhead is the call's wall time minus
``PatternClassifier`` inference and minus the time inside the local stand-in
that answers for the upstream. Warm-up calls are dropped. The test fails
unless the 95th percentile of the sample is under 50 ms and the mean is
under 25 ms. A sample that misses is measured once more. The failure
reports both samples. The caps are not raised.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
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
# Closed history the decision reads on every call, plus an evidence chain.
# The numbers are large enough that loading every closed row misses the gate.
FILLED_EVIDENCE = 10_000
FILLED_CLOSED = 2_000
_CLOSED_FROM = "2020-01-01 00:00:00.000000"
_CLOSED_TO = "2020-02-01 00:00:00.000000"
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


def _live_collaboration(path: Path) -> None:
    """The closed history adopts the register, so the purpose needs a live entry."""
    from datetime import datetime, timezone

    from kognita.evidence import EvidenceWriter
    from kognita.usecases import model_card_id, register_model_card
    from test_usecases import _live_purpose

    at = datetime.now(timezone.utc)
    provider = urlparse(UPSTREAM).hostname
    engine = make_engine(path)
    evidence = EvidenceWriter(engine)
    with Session(engine) as session:
        register_model_card(
            session,
            provider=provider,
            model=MODEL,
            version=MODEL,
            intended_use="Gateway overhead fixture",
            limitations="None recorded",
            evaluation_results="Not evaluated",
            approver="Luis Ortega",
            approved_at=at,
            evidence=evidence,
            actor_id="Luis Ortega",
            at=at,
        )
        _live_purpose(
            session,
            evidence,
            "COLLABORATION",
            at=at,
            approved_models=(model_card_id(provider, MODEL, MODEL),),
        )
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


def _fill_store(path: Path) -> None:
    """Evidence plus closed decision history. None of it is in force."""
    connection = sqlite3.connect(path)
    evidence = []
    previous = "0" * 64
    payload = json.dumps(
        {
            "outcome": "ALLOW",
            "checks": [
                {
                    "check": "PURPOSE",
                    "regime": "INTERNAL",
                    "result": "pass",
                    "citation": "Lawful-basis registry",
                }
            ],
            "envelope_hash": "ab" * 32,
        }
    )
    for sequence in range(1, FILLED_EVIDENCE + 1):
        event_hash = f"{sequence:064x}"
        evidence.append(
            (
                sequence,
                sequence,
                f"seed-{sequence}",
                "POLICY_DECISION",
                "SYSTEM",
                "seed",
                "C1",
                payload,
                "cd" * 32,
                previous,
                event_hash,
                _CLOSED_FROM,
            )
        )
        previous = event_hash
    connection.executemany(
        """
        INSERT INTO evidence_events (
            id, sequence, correlation_id, event_type, actor_type, actor_id,
            classification, payload, payload_hash, prev_hash, event_hash, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        evidence,
    )
    policies = []
    revocations = []
    drills = []
    usecases = []
    for index in range(FILLED_CLOSED):
        if index == 0:
            rule = json.dumps(
                {
                    "dimension": "model",
                    "value": MODEL,
                    "on_violation": "fail",
                }
            )
            rule_type = "CIRCUIT_BREAKER"
            purpose = "COLLABORATION"
            revoked = MODEL
        else:
            rule = json.dumps({"on_violation": "fail"})
            rule_type = "PROHIBITED"
            purpose = f"closed-use-case-{index}"
            revoked = f"closed-model-{index}"
        policies.append(
            (
                "INTERNAL",
                rule_type,
                rule,
                f"closed policy {index}",
                _CLOSED_FROM,
                _CLOSED_TO,
                f"{index:064x}",
            )
        )
        revocations.append(
            (
                "model",
                revoked,
                f"closed revocation {index}",
                "owner",
                json.dumps(["owner"]),
                json.dumps([]),
                "",
                _CLOSED_FROM,
                _CLOSED_TO,
            )
        )
        drills.append((purpose, _CLOSED_FROM, _CLOSED_FROM, _CLOSED_TO))
        # A closed use-case row for COLLABORATION would hold version 1, which
        # the live entry registered below needs. The history uses other ids.
        closed_usecase = f"closed-use-case-{index}"
        usecases.append(
            (
                closed_usecase,
                1,
                json.dumps({"usecase_id": closed_usecase, "status": "retired"}),
                f"{index:064x}",
                _CLOSED_FROM,
                _CLOSED_TO,
            )
        )
    connection.executemany(
        """
        INSERT INTO policies (
            regime, rule_type, rule, citation, effective_from, effective_to, content_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        policies,
    )
    connection.executemany(
        """
        INSERT INTO revocations (
            dimension, value, citation, actor_id, accountable_owners, limit_usecases,
            below_classification, effective_from, effective_to
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        revocations,
    )
    connection.executemany(
        """
        INSERT INTO drill_dues (usecase_id, next_due, effective_from, effective_to)
        VALUES (?, ?, ?, ?)
        """,
        drills,
    )
    connection.executemany(
        """
        INSERT INTO use_cases (
            usecase_id, version, record, content_hash, effective_from, effective_to
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        usecases,
    )
    connection.commit()
    connection.close()


def _summary(
    overheads: list[float], inference: list[float], upstream: list[float]
) -> tuple[bool, str]:
    maximum = max(overheads)
    mean = sum(overheads) / len(overheads)
    # A shared runner can stall one call without the gateway getting slower.
    # That maximum is printed and is not the measure. The measure is the
    # 95th percentile and the mean of the sample.
    p95 = _nearest_rank(overheads, 0.95)
    text = (
        f"gateway overhead sample={CALLS} warmup_dropped={WARMUP} "
        f"p95_ms={p95 * 1000:.3f} mean_ms={mean * 1000:.3f} "
        f"max_ms={maximum * 1000:.3f} "
        f"classifier_mean_ms={sum(inference) / len(inference) * 1000:.3f} "
        f"upstream_mean_ms={sum(upstream) / len(upstream) * 1000:.3f} "
        f"machine={_machine()}"
    )
    return p95 < LIMIT_S and mean < MEAN_LIMIT_S, text


def _take_sample(
    call: Callable[[], tuple[float, float, float]],
) -> tuple[list[float], list[float], list[float]]:
    for _ in range(WARMUP):
        call()
    _StandIn.bodies.clear()
    overheads: list[float] = []
    inference: list[float] = []
    upstream: list[float] = []
    for _ in range(CALLS):
        overhead, spent_inference, spent_model = call()
        overheads.append(overhead)
        inference.append(spent_inference)
        upstream.append(spent_model)
    assert len(_StandIn.bodies) == CALLS
    assert all(SECRET.encode() not in body for body in _StandIn.bodies)
    assert all(b"[EMAIL_1]" in body for body in _StandIn.bodies)
    assert len(overheads) == CALLS
    return overheads, inference, upstream


def _require(call: Callable[[], tuple[float, float, float]]) -> None:
    """Hold the gate. One miss is measured again, and both samples are reported."""
    ok, summary = _summary(*_take_sample(call))
    print(summary)
    if ok:
        return
    again, second = _summary(*_take_sample(call))
    print(second)
    assert again, f"{summary} || {second}"


def _bench(tmp_path: Path, monkeypatch, *, fill: bool) -> None:
    _StandIn.bodies = []
    _StandIn.model_s = 0.0
    stand_in = ThreadingHTTPServer(("127.0.0.1", 0), _StandIn)
    stand_in_thread = _listen(stand_in)
    origin = f"http://127.0.0.1:{stand_in.server_address[1]}"
    db = tmp_path / "gateway.db"
    _seed(db)
    if fill:
        _fill_store(db)
        _live_collaboration(db)
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
        _require(_call)
    finally:
        _stop(server, thread)
        _stop(stand_in, stand_in_thread)


def test_gateway_overhead_is_under_50ms_excluding_classifier_inference(
    tmp_path: Path, monkeypatch
) -> None:
    """The steady sample stays under the overhead caps, after classifier time.

    The first calls are warm-up and are not measured. Thread and database
    startup on a cold server are outside the sample. The 95th percentile
    must stay under 50 ms and the mean under 25 ms. A sample that misses
    those caps is measured once more. The failure names both samples.
    """
    _bench(tmp_path, monkeypatch, fill=False)


def test_gateway_overhead_stays_under_50ms_with_a_filled_store(
    tmp_path: Path, monkeypatch
) -> None:
    """Closed history and an evidence chain do not move the same caps.

    The store holds 10,000 evidence rows and 2,000 closed policies,
    revocations, drill dues, and use-case versions. One closed breaker,
    revocation, and drill would deny this call if they were still in force.
    They are not. The caps are the same 50 ms and 25 ms.
    """
    _bench(tmp_path, monkeypatch, fill=True)


def test_a_passing_overhead_sample_is_not_measured_again(monkeypatch) -> None:
    """The second sample exists for a miss. A pass stops."""
    taken = {"n": 0}
    module = sys.modules[__name__]

    def take(call: Callable[[], tuple[float, float, float]]):
        taken["n"] += 1
        call()
        return [0.01] * CALLS, [0.001] * CALLS, [0.0] * CALLS

    monkeypatch.setattr(module, "_take_sample", take)
    monkeypatch.setattr(
        module,
        "_summary",
        lambda *_args: (True, "gateway overhead sample=100 p95_ms=9.000 mean_ms=8.000"),
    )
    _require(lambda: (0.01, 0.001, 0.0))
    assert taken["n"] == 1


def test_an_overhead_miss_is_measured_once_more(monkeypatch, capsys) -> None:
    """One miss is not the result. The next sample is, and both are printed."""
    taken = {"n": 0}
    module = sys.modules[__name__]

    def take(call: Callable[[], tuple[float, float, float]]):
        taken["n"] += 1
        return [0.01] * CALLS, [0.001] * CALLS, [0.0] * CALLS

    summaries = iter(
        [
            (False, "gateway overhead sample=100 p95_ms=220.490 mean_ms=51.415"),
            (True, "gateway overhead sample=100 p95_ms=9.100 mean_ms=8.200"),
        ]
    )
    monkeypatch.setattr(module, "_take_sample", take)
    monkeypatch.setattr(module, "_summary", lambda *_args: next(summaries))
    _require(lambda: (0.01, 0.001, 0.0))
    assert taken["n"] == 2
    printed = capsys.readouterr().out
    assert "p95_ms=220.490" in printed
    assert "p95_ms=9.100" in printed


def test_two_overhead_misses_report_both_samples(monkeypatch) -> None:
    """The failure names the first sample and the sample taken after it."""
    module = sys.modules[__name__]
    monkeypatch.setattr(
        module,
        "_take_sample",
        lambda _call: ([0.2] * CALLS, [0.001] * CALLS, [0.0] * CALLS),
    )
    monkeypatch.setattr(
        module,
        "_summary",
        lambda *_args: (False, "gateway overhead sample=100 p95_ms=220.490 mean_ms=51.415"),
    )
    try:
        _require(lambda: (0.2, 0.001, 0.0))
    except AssertionError as exc:
        text = str(exc)
    else:
        raise AssertionError("two misses must fail")
    assert text.count("p95_ms=220.490") == 2
    assert " || " in text


def test_one_slow_call_does_not_set_the_overhead_percentile() -> None:
    """A single stall stays out of the 95th percentile and under the mean cap."""
    stalled = 0.267
    sample = [0.012] * (CALLS - 1) + [stalled]
    assert _nearest_rank(sample, 0.95) < LIMIT_S
    assert max(sample) == stalled
    assert sum(sample) / len(sample) < MEAN_LIMIT_S
