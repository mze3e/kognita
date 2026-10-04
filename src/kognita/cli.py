"""``kognita`` — operational commands for a governed store.

Deliberately small. This is not an application; it is the handful of things an
operator needs when the application is not running: what is installed, is the
evidence intact, and give me the artifact the auditor asked for.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from kognita.db import create_all, make_engine, session_scope
from kognita.evidence import ChainBreak, EvidenceWriter, export_chain, verify_chain
from kognita.exceptions import ConfigError
from kognita.gateway import ClientConfiguration, Gateway
from kognita.mcp import McpProxy, load_root_config


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

    graph_ok = (
        _probe("kuzu") != "not installed" and _probe("graphiti_core") != "not installed"
    )
    print(
        f"graph engine             {'available' if graph_ok else 'unavailable (pip install kognita[graph])'}"
    )

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
        print(
            f"export verified: {count} events, head {payload.get('head_hash', '')[:16]}"
        )
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


def _cmd_serve_mcp(args: argparse.Namespace) -> int:
    """Front the MCP servers named in ``--root-config``.

    The file names the backend servers, the policy pack, the evidence
    database, and the default actor context. ``--provider`` and ``--upstream``
    belong to the OpenAI-compatible gateway, not to this mode.
    """
    if not args.root_config:
        print("serve --mcp requires --root-config", file=sys.stderr)
        return 2
    try:
        config = load_root_config(args.root_config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    proxy = McpProxy.from_config(config)
    proxy.serve(args.host, args.port)
    return 0


def _template_dir(name: str) -> Path | None:
    """Where ``scaffold`` reads a template from.

    An installed wheel carries the template next to this module. A checkout
    keeps it under ``examples/`` so the core package does not import it.
    """
    candidates = (
        Path(__file__).resolve().parent / "_templates" / name,
        Path(__file__).resolve().parents[2] / "examples" / name,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def cmd_scaffold(args: argparse.Namespace) -> int:
    """Copy a template and seed its SQLite policy and evidence store."""
    source = _template_dir(args.template)
    if source is None:
        print(f"unknown template {args.template}", file=sys.stderr)
        return 2
    dest = Path(args.dest)
    if dest.exists() and any(dest.iterdir()):
        print(f"{dest} is not empty", file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix in {".pyc", ".db"}:
            continue
        shutil.copy2(path, dest / path.name)
    completed = subprocess.run(
        [
            sys.executable,
            str(dest / "app.py"),
            "seed",
            "--db",
            str(dest / "kognita.db"),
        ],
        cwd=dest,
    )
    if completed.returncode != 0:
        return completed.returncode
    print(f"created {dest}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the OpenAI-compatible AI gateway, or the MCP proxy.

    The AI gateway binds one client configuration. Agent names outside that
    configuration are denied. ``--provider`` accepts ``openai-compatible``
    only. ``--mcp --root-config`` fronts MCP servers instead, and does not
    start the model gateway.
    """
    if args.mcp:
        return _cmd_serve_mcp(args)
    if not args.upstream:
        print("serve --provider openai-compatible requires --upstream", file=sys.stderr)
        return 2
    engine = make_engine(args.db)
    create_all(engine)
    gateway = Gateway(
        engine=engine,
        evidence=EvidenceWriter(engine),
        upstream=args.upstream,
        client=ClientConfiguration(
            principal=args.principal,
            purpose=args.purpose,
            agent_names=frozenset(args.agent or []),
            system_triggers=frozenset(args.system_trigger or []),
            actor_location=args.actor_location,
        ),
    )
    gateway.serve(args.host, args.port)
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

    serve = sub.add_parser("serve", help="run the AI gateway or the MCP proxy")
    mode = serve.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--provider",
        choices=["openai-compatible"],
        help="wire format. openai-compatible only",
    )
    mode.add_argument(
        "--mcp",
        action="store_true",
        help="front one or more MCP servers",
    )
    serve.add_argument("--upstream", help="provider origin, not a base path")
    serve.add_argument(
        "--root-config",
        help="MCP config naming servers, policy pack, evidence database, and actor",
    )
    serve.add_argument("--db", default="kognita.db", help="policy and evidence store")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument(
        "--principal", default="", help="bound principal when the request has none"
    )
    serve.add_argument(
        "--purpose", default="", help="bound purpose when the request has none"
    )
    serve.add_argument("--actor-location", default="")
    serve.add_argument(
        "--agent",
        action="append",
        default=None,
        help="agent name this client configuration may present; repeatable",
    )
    serve.add_argument(
        "--system-trigger",
        action="append",
        default=None,
        help="approved system trigger this client may present; repeatable",
    )
    serve.set_defaults(func=cmd_serve)

    scaffold = sub.add_parser("scaffold", help="create an application from a template")
    scaffold.add_argument(
        "--template",
        required=True,
        choices=["governed-agent"],
        help="application template",
    )
    scaffold.add_argument(
        "--dest",
        default="governed-agent",
        help="directory to create (default: governed-agent)",
    )
    scaffold.set_defaults(func=cmd_scaffold)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
