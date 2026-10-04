"""Run the four governed-agent scenarios against the scaffolded store.

1. Allowed query: dossier-agent reads subject 2, and the evidence is logged.
2. Denied query: the agent asks for subject 1, is denied with citations, and
   nothing is retrieved.
3. Tampering: one evidence row is edited, then ``kognita evidence verify``
   reports the break.
4. Through the AI gateway: a prompt containing ana@example.org is redacted
   before the local stand-in sees it. Evidence keeps the manifest hash.

The gateway call is made before the row is edited, so the manifest is on the
chain that ``verify`` then reports as broken.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sqlalchemy import text
from sqlmodel import select

from kognita import (
    ClientConfiguration,
    EgressGuard,
    EgressPolicy,
    EventType,
    EvidenceWriter,
    Gateway,
    HashingEmbedder,
    Outcome,
    ask,
    canonical_json,
    make_engine,
    session_scope,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app  # noqa: E402


ROOT = Path(__file__).resolve().parent
DB = ROOT / "kognita.db"
UPSTREAM_BODY = ROOT / "upstream-request.json"
CITATION = "Data Sharing Charter s4, Schedule 1"


def _verify(db: Path) -> subprocess.CompletedProcess[str]:
    kognita = shutil.which("kognita")
    if kognita:
        command = [kognita, "evidence", "verify", "--db", str(db)]
    else:
        command = [
            sys.executable,
            "-m",
            "kognita.cli",
            "evidence",
            "verify",
            "--db",
            str(db),
        ]
    return subprocess.run(command, capture_output=True, text=True)


def _events(session, request_id: str):
    from kognita import EvidenceEvent

    return list(
        session.exec(
            select(EvidenceEvent)
            .where(EvidenceEvent.correlation_id == request_id)
            .order_by(EvidenceEvent.sequence)
        ).all()
    )


def _allowed(session, pack, evidence) -> None:
    answer = ask(
        session,
        app.QUESTION,
        app.envelope(app.ASSIGNED_SUBJECT),
        pack=pack,
        embedder=HashingEmbedder(),
        evidence=evidence,
        purposes=app.PURPOSES,
    )
    if answer.outcome is not Outcome.ALLOW or not answer.results:
        raise RuntimeError(
            f"allowed query did not read permitted data: {answer.to_dict()}"
        )
    kinds = [event.event_type for event in _events(session, answer.request_id)]
    if EventType.POLICY_DECISION not in kinds or EventType.RETRIEVAL not in kinds:
        raise RuntimeError(f"allowed query did not log evidence: {kinds}")
    print(
        f"allowed: {answer.outcome.value} subject {app.ASSIGNED_SUBJECT} "
        f"retrieved {len(answer.results)}"
    )


def _denied(session, pack, evidence) -> None:
    answer = ask(
        session,
        app.QUESTION,
        app.envelope(app.OTHER_SUBJECT),
        pack=pack,
        embedder=HashingEmbedder(),
        evidence=evidence,
        purposes=app.PURPOSES,
    )
    citations = [
        check.citation
        for check in (answer.evaluation.basis() if answer.evaluation else ())
    ]
    kinds = [event.event_type for event in _events(session, answer.request_id)]
    if (
        answer.outcome is not Outcome.DENY
        or answer.results
        or CITATION not in citations
    ):
        raise RuntimeError(f"denied query was not a cited denial: {answer.to_dict()}")
    if EventType.RETRIEVAL in kinds:
        raise RuntimeError("denied query retrieved data")
    print(f"denied: {answer.outcome.value} citations {citations} retrieved 0")


def _gateway(engine, pack, evidence) -> None:
    """Redact through the AI gateway into a local stand-in. No provider key."""
    bodies: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            bodies.append(body)
            response = json.dumps(
                {
                    "model": app.MODEL,
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Reply to [EMAIL_1] tomorrow",
                            }
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, fmt: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        gateway = Gateway(
            engine=engine,
            evidence=evidence,
            upstream=f"http://127.0.0.1:{port}",
            client=ClientConfiguration(
                principal=app.PRINCIPAL,
                purpose=app.PURPOSE,
                agent_names=frozenset({app.AGENT_NAME}),
                actor_location=app.SITE,
            ),
            pack=pack,
            purposes=app.PURPOSES,
            # The stand-in is the provider. Local trust would skip redaction.
            egress=EgressGuard(policy=EgressPolicy(trust_local=False)),
        )
        response = gateway.handle(
            "POST",
            "/v1/chat/completions",
            {"agent_name": app.AGENT_NAME},
            json.dumps(
                {
                    "model": app.MODEL,
                    "messages": [{"role": "user", "content": app.PROMPT}],
                }
            ).encode(),
        )
    finally:
        server.shutdown()
        server.server_close()

    if response.status != 200 or not bodies:
        raise RuntimeError(
            f"gateway did not forward a redacted call: {response.status} {response.body!r}"
        )
    forwarded = bodies[0]
    UPSTREAM_BODY.write_bytes(forwarded)
    if b"ana@example.org" in forwarded or b"[EMAIL_1]" not in forwarded:
        raise RuntimeError("stand-in saw the address, or the span was not redacted")
    print("gateway: stand-in received the redacted prompt")


def _manifest(session) -> None:
    from kognita import EvidenceEvent

    events = list(session.exec(select(EvidenceEvent)).all())
    blob = canonical_json([event.payload for event in events])
    if "ana@example.org" in blob or app.PROMPT in blob:
        raise RuntimeError("evidence stored the prompt")
    model_calls = [
        event for event in events if event.event_type is EventType.MODEL_CALL
    ]
    if not model_calls:
        raise RuntimeError("gateway wrote no MODEL_CALL evidence")
    manifest = model_calls[-1].payload.get("manifest_hash") or ""
    if len(manifest) != 64 or model_calls[-1].payload.get("redacted_spans", 0) < 1:
        raise RuntimeError(f"evidence has no manifest hash: {model_calls[-1].payload}")
    print(f"gateway: manifest {manifest}")


def _tamper(engine) -> str:
    from kognita import EvidenceEvent

    intact = _verify(DB)
    if intact.returncode != 0:
        raise RuntimeError(f"chain was already broken:\n{intact.stderr}")
    with session_scope(engine) as session:
        event = session.exec(
            select(EvidenceEvent).order_by(EvidenceEvent.sequence)
        ).first()
        if event is None:
            raise RuntimeError("no evidence row to edit")
        event.payload = {**event.payload, "tampered": True}
        session.add(event)
    with engine.connect() as connection:
        connection.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        connection.commit()
    engine.dispose()
    broken = _verify(DB)
    message = (broken.stderr or broken.stdout).strip()
    if broken.returncode == 0 or "BROKEN" not in message:
        raise RuntimeError(f"verify did not report the break: {message}")
    print(message)
    return message


def main() -> int:
    if not DB.is_file():
        print(
            f"no store at {DB}; run: kognita scaffold --template governed-agent",
            file=sys.stderr,
        )
        return 2
    engine = make_engine(DB)
    pack = app.DemoPack()
    evidence = EvidenceWriter(engine)
    try:
        with session_scope(engine) as session:
            _allowed(session, pack, evidence)
            _denied(session, pack, evidence)
        _gateway(engine, pack, evidence)
        with session_scope(engine) as session:
            _manifest(session)
        _tamper(engine)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
