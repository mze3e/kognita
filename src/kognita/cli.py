"""``kognita`` — operational commands for a governed store.

Deliberately small. This is not an application; it is the handful of things an
operator needs when the application is not running: what is installed, is the
evidence intact, and give me the artifact the auditor asked for.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from kognita.db import make_engine, session_scope
from kognita.evidence import ChainBreak, export_chain, verify_chain


def _probe(module: str) -> str:
    """Report an optional dependency's version, or why it is unavailable."""
    try:
        import_module(module)
    except ImportError:
        return "not installed"
    try:
        return version(module.replace("_", "-"))
    except PackageNotFoundError:
        return "installed"


def _vector_backend() -> str:
    """Which vector index this machine can actually run.

    ``enable_load_extension`` is compiled out of many stock Python builds, so
    reporting "sqlite-vec is installed" would be misleading — what matters is
    whether it loads here.
    """
    try:
        import sqlite3

        import sqlite_vec  # noqa: F401
    except ImportError:
        return "numpy (sqlite-vec not installed)"
    try:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        return "sqlite-vec available (numpy is still the default)"
    except Exception:
        return "numpy (sqlite-vec installed but this Python cannot load extensions)"


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what is installed and usable on this machine."""
    from kognita import __version__

    print(f"kognita {__version__}   python {sys.version.split()[0]}")
    print()
    print("core (always required)")
    for module in ("pydantic", "sqlmodel", "numpy", "dotenv"):
        print(f"  {module:<22} {_probe(module)}")
    print()
    print("optional extras")
    for label, module in (
        ("graph (graphiti)", "graphiti_core"),
        ("graph (kuzu)", "kuzu"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("groq", "groq"),
        ("gemini", "google.genai"),
        ("vec", "sqlite_vec"),
    ):
        print(f"  {label:<22} {_probe(module)}")
    print()
    print(f"vector backend           {_vector_backend()}")

    graph_ok = _probe("kuzu") != "not installed" and _probe("graphiti_core") != "not installed"
    print(f"graph engine             {'available' if graph_ok else 'unavailable (pip install kognita[graph])'}")

    if args.db:
        print()
        engine = make_engine(args.db)
        with session_scope(engine) as session:
            try:
                count = verify_chain(session)
                print(f"evidence chain           intact, {count} events")
            except ChainBreak as exc:
                print(f"evidence chain           BROKEN — {exc}")
                return 1
    return 0


def cmd_evidence_verify(args: argparse.Namespace) -> int:
    """Verify a store's evidence chain, or a previously exported artifact."""
    if args.file:
        from kognita.evidence import verify_export

        payload = json.loads(Path(args.file).read_text())
        try:
            count = verify_export(payload)
        except ChainBreak as exc:
            print(f"BROKEN: {exc}", file=sys.stderr)
            return 1
        print(f"export verified: {count} events, head {payload.get('head_hash', '')[:16]}")
        return 0

    engine = make_engine(args.db)
    with session_scope(engine) as session:
        try:
            count = verify_chain(session)
        except ChainBreak as exc:
            print(f"BROKEN: {exc}", file=sys.stderr)
            return 1
    print(f"evidence chain verified: {count} events")
    return 0


def cmd_evidence_export(args: argparse.Namespace) -> int:
    """Write a portable, self-verifying evidence artifact."""
    since: datetime | None = None
    if args.since:
        since = datetime.fromisoformat(args.since)
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

    engine = make_engine(args.db)
    with session_scope(engine) as session:
        payload: dict[str, Any] = export_chain(
            session, since=since, correlation_id=args.correlation_id
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text)
        print(
            f"wrote {payload['event_count']} events "
            f"({payload['interest_count']} of interest) to {args.output}"
        )
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kognita",
        description="Prove an AI answer was permitted — and evidence it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="report installed extras and store health")
    doctor.add_argument("--db", help="optional store to check the evidence chain of")
    doctor.set_defaults(func=cmd_doctor)

    evidence = sub.add_parser("evidence", help="evidence plane operations")
    evidence_sub = evidence.add_subparsers(dest="evidence_command", required=True)

    verify = evidence_sub.add_parser("verify", help="verify a chain is unbroken")
    verify.add_argument("--db", default="kognita.db", help="store path or URL")
    verify.add_argument("--file", help="verify an exported artifact instead of a store")
    verify.set_defaults(func=cmd_evidence_verify)

    export = evidence_sub.add_parser("export", help="write a portable audit artifact")
    export.add_argument("--db", default="kognita.db", help="store path or URL")
    export.add_argument("--since", help="ISO timestamp; marks events of interest")
    export.add_argument("--correlation-id", help="mark only this request's events")
    export.add_argument("-o", "--output", help="write to a file instead of stdout")
    export.set_defaults(func=cmd_evidence_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
