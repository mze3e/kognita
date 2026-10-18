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

from kognita.canonical import canonical_json
from kognita.db import create_all, make_engine, session_scope
from kognita.evidence import ChainBreak, EvidenceWriter, export_chain, verify_chain
from kognita.exceptions import ConfigError
from kognita.gateway import ClientConfiguration, Gateway
from kognita.vocabulary import ActorType, EventType, FailureMode
from kognita.mcp import McpProxy, load_root_config
from kognita.policy import (
    PolicyLanguageError,
    apply_change,
    blocking_findings,
    compile_rules,
    diff_sets,
    explain,
    format_finding,
    is_change_path,
    policies_from_session,
    read_change,
    read_envelope,
    read_policy_set,
    render_change,
    render_explanation,
    run_scenarios,
    source_hash,
    validate,
)
from kognita.reconstruct import reconstruct, render_reconstruction


def _probe(module: str, distribution: str | None = None) -> str:
    """Report an optional dependency's version, or why it is unavailable."""
    try:
        import_module(module)
    except ImportError:
        return "not installed"
    try:
        return version(distribution or module.replace("_", "-"))
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
    for label, module, distribution in (
        ("openai", "openai", "openai"),
        ("vec", "sqlite_vec", "sqlite-vec"),
        ("docling", "docling", "docling"),
        ("hermes-agent", "hermes_cli", "hermes-agent"),
        ("openai-agents", "agents", "openai-agents"),
        ("langgraph", "langgraph", "langgraph"),
        ("pydantic-ai", "pydantic_ai", "pydantic-ai"),
    ):
        print(f"  {label:<22} {_probe(module, distribution)}")
    print()
    print(f"vector backend           {_vector_backend()}")

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


def _report_paths(output: str) -> tuple[Path, Path]:
    """JSON and readable paths for one reconstruct ``--output`` prefix."""
    path = Path(output)
    if path.suffix in {".json", ".txt", ".md"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_suffix(".json"), path.with_suffix(".txt")


def cmd_evidence_reconstruct(args: argparse.Namespace) -> int:
    """Write the reconstruction report as JSON and as a readable document.

    ``interaction_id`` is the correlation id on the evidence events. A hash
    mismatch is printed in the report; the command still writes both forms.
    """
    engine = make_engine(args.db)
    with session_scope(engine) as session:
        report = reconstruct(session, args.interaction_id)
    document = render_reconstruction(report)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        json_path, text_path = _report_paths(args.output)
        json_path.write_text(payload + "\n")
        text_path.write_text(document)
        print(f"wrote {json_path}")
        print(f"wrote {text_path}")
        return 0
    print(document, end="")
    print("--- json ---")
    print(payload)
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
    configuration are denied. ``--purposes`` is the purpose allowlist.
    ``--purpose`` is the purpose claimed when a request has none. With no
    ``--purposes``, every call is denied. ``--provider`` accepts
    ``openai-compatible`` only. ``--mcp --root-config`` fronts MCP servers
    instead, and does not start the model gateway. The proxy's allowlist is
    the ``purposes`` list in that file, not ``--purpose`` and not ``--purposes``.
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
        purposes=tuple(args.purposes or ()),
        failure_mode={args.purpose: args.failure_mode},
    )
    gateway.serve(args.host, args.port)
    return 0


def cmd_usecase_list(args: argparse.Namespace) -> int:
    """Print the use cases effective now."""
    from kognita.usecases import list_usecases

    engine = make_engine(args.db)
    create_all(engine)
    with session_scope(engine) as session:
        entries = list_usecases(session)
    if not entries:
        print("no registered use cases")
        return 0
    for entry in entries:
        status = entry.status or "not activated"
        owner = entry.business_owner or "(none)"
        print(
            f"{entry.usecase_id}  v{entry.version}  {status}  "
            f"tier {entry.proposed_tier}  accountable owner: {owner}"
        )
    return 0


def cmd_usecase_show(args: argparse.Namespace) -> int:
    """Print one use case as JSON."""
    from kognita.usecases import show_usecase

    engine = make_engine(args.db)
    create_all(engine)
    with session_scope(engine) as session:
        entry = show_usecase(session, args.usecase_id)
    if entry is None:
        print(
            f"Use-case register — '{args.usecase_id}' is not registered",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(entry.record, indent=2, sort_keys=True))
    return 0


def cmd_usecase_metrics(args: argparse.Namespace) -> int:
    """Print cycle time, decision quality and reliability from the interaction record."""
    from kognita.outcomes import outcome_metrics

    engine = make_engine(args.db)
    create_all(engine)
    with session_scope(engine) as session:
        reports = outcome_metrics(session, usecase_id=args.usecase_id)
    if not reports:
        print("no outcome metrics")
        return 0
    print(json.dumps([item.to_dict() for item in reports], indent=2, sort_keys=True))
    return 0


def cmd_usecase_coverage(args: argparse.Namespace) -> int:
    """Print review-queue coverage per use case and risk tier."""
    from kognita.review import coverage_report

    engine = make_engine(args.db)
    create_all(engine)
    with session_scope(engine) as session:
        report = coverage_report(session)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_usecase_validate(args: argparse.Namespace) -> int:
    """Fail when a use case is outside the register's rules."""
    from kognita.usecases import validate_usecase

    engine = make_engine(args.db)
    create_all(engine)
    with session_scope(engine) as session:
        findings = validate_usecase(session, args.usecase_id)
    if not findings:
        print("valid")
        return 0
    for citation in findings:
        print(citation)
    return 1


def _print_findings(findings: list[Any]) -> None:
    for finding in findings:
        print(format_finding(finding), file=sys.stderr)


def _policy_error(exc: PolicyLanguageError) -> int:
    _print_findings(exc.findings)
    return 1


def _findings_that_block_a_load(findings: list[Any], *, strict: bool) -> list[Any]:
    """What stops a load.

    A plain load blocks type and referential errors, and an uncited rule.
    Untested rules, dangling references, and clarification markers block only
    when ``strict`` is set, which is how a change load runs.
    """
    blocked = blocking_findings(findings, strict=strict)
    if strict:
        return blocked
    seen = {(item.code, item.message, item.rule_id) for item in blocked}
    for item in findings:
        if item.code != "uncited":
            continue
        key = (item.code, item.message, item.rule_id)
        if key in seen:
            continue
        seen.add(key)
        blocked.append(item)
    return blocked


def cmd_policy_load(args: argparse.Namespace) -> int:
    """Parse a policy set, or merge a change, and compile the rules to rows.

    A change file needs ``--policy``, the current policy set. Loading a change
    runs the strict checks and, when they pass, merges the delta into that
    directory, archives the change with the date, and compiles the result.

    A plain load refuses an uncited rule. An untested rule does not stop a
    plain load. A load that changes stored rows appends one POLICY_CHANGE
    evidence event in the same transaction.
    """
    path = Path(args.path)
    try:
        if is_change_path(path):
            if not args.policy:
                print(
                    "a change file needs --policy, the current policy set",
                    file=sys.stderr,
                )
                return 2
            policy_path = Path(args.policy)
            current = read_policy_set(policy_path)
            change = read_change(path)
            findings = _findings_that_block_a_load(
                validate(current, change), strict=True
            )
            if findings:
                _print_findings(findings)
                return 1
            directory = policy_path if policy_path.is_dir() else policy_path.parent
            merged = apply_change(
                current, change, directory=directory, when=datetime.now(timezone.utc)
            )
            written = _compile_policy(args.db, merged, source=path, close_absent=True)
            print(f"applied {path} into {directory}")
            print(f"compiled {written} rules into {args.db}")
            return 0
        policies = read_policy_set(path)
        findings = _findings_that_block_a_load(validate(policies), strict=False)
        if findings:
            _print_findings(findings)
            return 1
        written = _compile_policy(
            args.db, policies, source=path, close_absent=path.is_dir()
        )
        print(f"compiled {written} rules into {args.db}")
        return 0
    except PolicyLanguageError as exc:
        return _policy_error(exc)


def cmd_policy_diff(args: argparse.Namespace) -> int:
    """Show the ADDED, MODIFIED and REMOVED delta between two policy sets."""
    try:
        change = diff_sets(read_policy_set(args.before), read_policy_set(args.after))
    except PolicyLanguageError as exc:
        return _policy_error(exc)
    text = render_change(change)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return 0


def cmd_policy_validate(args: argparse.Namespace) -> int:
    """Type-check a policy set or a change.

    ``--strict`` also fails on an uncited rule, an untested rule, a dangling
    reference, and an unresolved ``[NEEDS CLARIFICATION: ...]`` marker.
    """
    path = Path(args.path)
    try:
        if is_change_path(path):
            if not args.policy:
                print(
                    "a change file needs --policy, the current policy set",
                    file=sys.stderr,
                )
                return 2
            findings = validate(read_policy_set(args.policy), read_change(path))
        else:
            findings = validate(read_policy_set(path))
    except PolicyLanguageError as exc:
        return _policy_error(exc)
    blocking = blocking_findings(findings, strict=args.strict)
    if blocking:
        _print_findings(blocking)
        return 1
    print("valid")
    return 0


def cmd_policy_explain(args: argparse.Namespace) -> int:
    """Show which rules engage for an envelope, and the citation of each check."""
    if not args.policy and not args.db:
        print("explain needs --policy or --db", file=sys.stderr)
        return 2
    try:
        envelope, attributes, purposes, as_of = read_envelope(args.envelope)
        if args.policy:
            policies = read_policy_set(args.policy)
        else:
            policies = _policies_in(args.db, as_of)
        report = explain(
            policies,
            envelope,
            attributes=attributes,
            purposes=purposes,
            as_of=as_of,
        )
    except PolicyLanguageError as exc:
        return _policy_error(exc)
    print(render_explanation(report), end="")
    return 0


def cmd_policy_test(args: argparse.Namespace) -> int:
    """Run the scenarios colocated with a policy set."""
    try:
        results = run_scenarios(read_policy_set(args.path))
    except PolicyLanguageError as exc:
        return _policy_error(exc)
    if not results:
        print("no scenarios")
        return 0
    failed = 0
    for result in results:
        if result.passed:
            print(f"PASS {result.rule_id}: {result.name}")
            continue
        failed += 1
        print(f"FAIL {result.rule_id}: {result.name}: {result.detail}", file=sys.stderr)
    return 1 if failed else 0


def _rule_body(rule: Any) -> str:
    """The rule body a load compares. Effective dates are not part of it."""
    return canonical_json(
        {
            "id": rule.id,
            "regime": rule.regime,
            "rule_type": rule.rule_type,
            "applies_to": rule.applies_to,
            "citation": rule.citation,
            "rule": rule.rule,
        }
    )


def _changed_ids(before: Any, after: Any) -> tuple[list[str], list[str], list[str]]:
    previous = {rule.id: rule for rule in before.rules}
    current = {rule.id: rule for rule in after.rules}
    added = sorted(current.keys() - previous.keys())
    removed = sorted(previous.keys() - current.keys())
    modified = sorted(
        rule_id
        for rule_id in previous.keys() & current.keys()
        if _rule_body(previous[rule_id]) != _rule_body(current[rule_id])
    )
    return added, modified, removed


def _compile_policy(db: str, policies: Any, *, source: Path, close_absent: bool) -> int:
    """Compile rows and, when any stored row changes, append POLICY_CHANGE.

    The event is written in the same transaction as the rows. A load that
    writes nothing appends nothing.
    """
    engine = make_engine(db)
    create_all(engine)
    moment = datetime.now(timezone.utc)
    digest = source_hash(source)
    with session_scope(engine) as session:
        before = policies_from_session(session, at=moment)
        written = compile_rules(session, policies, at=moment, close_absent=close_absent)
        after = policies_from_session(session, at=moment)
        added, modified, removed = _changed_ids(before, after)
        if added or modified or removed:
            EvidenceWriter(engine).emit(
                session,
                correlation_id="policy-load",
                event_type=EventType.POLICY_CHANGE,
                actor_type=ActorType.SYSTEM,
                actor_id="kognita policy load",
                payload={
                    "action": "LOAD",
                    "added": added,
                    "modified": modified,
                    "removed": removed,
                    "source_hash": digest,
                },
            )
        return written


def _policies_in(db: str, as_of: datetime | None) -> Any:
    engine = make_engine(db)
    create_all(engine)
    with session_scope(engine) as session:
        return policies_from_session(session, at=as_of)


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

    reconstruct_cmd = evidence_sub.add_parser(
        "reconstruct",
        help="answer the reconstruction test for one interaction",
    )
    reconstruct_cmd.add_argument(
        "interaction_id",
        help="correlation id of the evidence to reconstruct",
    )
    reconstruct_cmd.add_argument("--db", default="kognita.db", help="store path or URL")
    reconstruct_cmd.add_argument(
        "-o",
        "--output",
        help="path prefix; writes <prefix>.json and <prefix>.txt",
    )
    reconstruct_cmd.set_defaults(func=cmd_evidence_reconstruct)

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
    serve.add_argument(
        "--purposes",
        action="append",
        default=None,
        help=(
            "purpose the AI gateway may allow; repeatable. Distinct from "
            "--purpose, which is only the purpose claimed when a request has "
            "none. With no --purposes, the gateway denies every call"
        ),
    )
    serve.add_argument(
        "--failure-mode",
        default=FailureMode.FAIL_CLOSED.value,
        choices=[mode.value for mode in FailureMode],
        help=(
            "when the evidence store is unavailable, for --purpose. "
            "FAIL_CLOSED refuses the call. DEGRADED allows a local model "
            "with no client-identifying content and records it once the store recovers"
        ),
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

    usecase = sub.add_parser("usecase", help="use-case register")
    usecase_sub = usecase.add_subparsers(dest="usecase_command", required=True)

    usecase_list = usecase_sub.add_parser("list", help="list registered use cases")
    usecase_list.add_argument("--db", default="kognita.db", help="store path or URL")
    usecase_list.set_defaults(func=cmd_usecase_list)

    usecase_show = usecase_sub.add_parser("show", help="show one registered use case")
    usecase_show.add_argument("usecase_id", help="use case id")
    usecase_show.add_argument("--db", default="kognita.db", help="store path or URL")
    usecase_show.set_defaults(func=cmd_usecase_show)

    usecase_validate = usecase_sub.add_parser(
        "validate",
        help="fail when a use case is outside the register rules",
    )
    usecase_validate.add_argument(
        "usecase_id",
        nargs="?",
        help="one use case; omit to validate every use case",
    )
    usecase_validate.add_argument("--db", default="kognita.db", help="store path or URL")
    usecase_validate.set_defaults(func=cmd_usecase_validate)

    usecase_metrics = usecase_sub.add_parser(
        "metrics",
        help="report cycle time, decision quality and reliability",
    )
    usecase_metrics.add_argument(
        "usecase_id",
        nargs="?",
        help="one use case; omit to report every use case",
    )
    usecase_metrics.add_argument("--db", default="kognita.db", help="store path or URL")
    usecase_metrics.set_defaults(func=cmd_usecase_metrics)

    usecase_coverage = usecase_sub.add_parser(
        "coverage",
        help="report review coverage per use case and risk tier",
    )
    usecase_coverage.add_argument("--db", default="kognita.db", help="store path or URL")
    usecase_coverage.set_defaults(func=cmd_usecase_coverage)

    policy = sub.add_parser("policy", help="author, diff, and test policy files")
    policy_sub = policy.add_subparsers(dest="policy_command", required=True)

    load = policy_sub.add_parser(
        "load",
        help=(
            "parse, type-check, and compile rules to rows. Refuses an uncited "
            "rule. A load that changes stored rows appends a POLICY_CHANGE "
            "evidence event"
        ),
    )
    load.add_argument("path", help="policy file, policy directory, or change file")
    load.add_argument("--db", default="kognita.db", help="policy store")
    load.add_argument(
        "--policy",
        help="current policy set; required when path is a change file",
    )
    load.set_defaults(func=cmd_policy_load)

    diff = policy_sub.add_parser("diff", help="show the delta between two policy sets")
    diff.add_argument("before", help="policy file or directory")
    diff.add_argument("after", help="policy file or directory")
    diff.add_argument("-o", "--output", help="write the change document to a file")
    diff.set_defaults(func=cmd_policy_diff)

    validate_cmd = policy_sub.add_parser(
        "validate",
        help="fail on a broken policy set or change",
    )
    validate_cmd.add_argument(
        "path", help="policy file, policy directory, or change file"
    )
    validate_cmd.add_argument(
        "--policy",
        help="current policy set; required when path is a change file",
    )
    validate_cmd.add_argument(
        "--strict",
        action="store_true",
        help=(
            "also fail on an uncited rule, an untested rule, a dangling "
            "reference, and an unresolved [NEEDS CLARIFICATION: ...] marker"
        ),
    )
    validate_cmd.set_defaults(func=cmd_policy_validate)

    explain_cmd = policy_sub.add_parser(
        "explain",
        help="show which rules engage for an envelope, and why",
    )
    explain_cmd.add_argument(
        "--envelope",
        required=True,
        help="JSON file or JSON object. Optional keys: attributes, purposes, as_of",
    )
    explain_cmd.add_argument("--policy", help="policy file or directory")
    explain_cmd.add_argument("--db", help="policy store, when --policy is omitted")
    explain_cmd.set_defaults(func=cmd_policy_explain)

    test_cmd = policy_sub.add_parser("test", help="run colocated policy scenarios")
    test_cmd.add_argument("path", help="policy file, scenario file, or directory")
    test_cmd.set_defaults(func=cmd_policy_test)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
