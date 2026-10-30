"""``kognita`` — operational commands for a governed store.

Deliberately small. This is not an application; it is the handful of things an
operator needs when the application is not running: what is installed, is the
evidence intact, and give me the artifact the auditor asked for.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from kognita.breaker import (
    BREAKER_DIMENSIONS,
    BreakerError,
    affected_clients,
    evaluate_breakers,
    reset_breaker,
    run_kill_switch_drill,
    schedule_kill_switch_drill,
    set_breaker_threshold,
)
from kognita.canonical import canonical_json
from kognita.db import create_all, make_engine, session_scope
from kognita.evidence import ChainBreak, EvidenceWriter, export_chain, verify_chain
from kognita.revocation import DIMENSIONS, RevocationError, reinstate, revoke
from kognita.exceptions import ConfigError
from kognita.gateway import ClientConfiguration, Gateway
from kognita.vocabulary import ActorType, EventType, FailureMode
from kognita.mcp import McpProxy, load_root_config
from kognita.packs import (
    PackError,
    list_packs,
    load_source_hash,
    resolve_pack,
    update_pack,
)
from kognita.packs.catalog import (
    DEFAULT_PACK_SOURCE,
    builtin_data,
    default_managed_root,
    default_user_root,
)
from kognita.packs.update import pinned_names
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
        ("otel", "opentelemetry", "opentelemetry-api"),
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
    if args.otel:
        failed = _attach_otel(proxy.evidence._inner)
        if failed:
            return failed
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


def _attach_otel(evidence: EvidenceWriter) -> int | None:
    """Install OpenTelemetry, or return the exit code when it cannot be."""
    from kognita.otel import OtelError, attach

    try:
        print(attach(evidence), file=sys.stderr)
    except OtelError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return None


def _pack_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    user = Path(args.user_root) if getattr(args, "user_root", None) else default_user_root()
    managed = (
        Path(args.managed_root) if getattr(args, "managed_root", None) else default_managed_root()
    )
    return user, managed


def _load_named_pack(args: argparse.Namespace) -> Any:
    """Resolve a pack, compile it, and return the pack or an exit code.

    Compilation appends ``POLICY_CHANGE`` with ``source_hash`` when stored
    rows change. The user overlay is read and not written.
    """
    if getattr(args, "overlay", None) and not getattr(args, "pack", None):
        print("an overlay needs a pack", file=sys.stderr)
        return 2
    user, managed = _pack_roots(args)
    try:
        pack = resolve_pack(
            args.pack,
            overlay=args.overlay,
            user_root=user,
            managed_root=managed,
        )
    except PackError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    findings = _findings_that_block_a_load(validate(pack.policies), strict=False)
    if findings:
        _print_findings(findings)
        return 1
    _compile_policy(
        args.db,
        pack.policies,
        source=pack.policy_dir,
        close_absent=False,
        pack=pack.name,
        source_hash_value=load_source_hash(pack, args.overlay),
    )
    return pack


def _hmac_key(path: str | None) -> bytes | None:
    if not path:
        env = os.environ.get("KOGNITA_PACKS_HMAC_KEY")
        if not env:
            return None
        return env.encode("utf-8")
    data = Path(path).read_bytes()
    if data.endswith(b"\n"):
        data = data[:-1]
    if not data:
        raise PackError("hmac key is empty")
    return data


def cmd_packs_list(args: argparse.Namespace) -> int:
    """Print pack names found on disk."""
    user, managed = _pack_roots(args)
    for name in list_packs(user_root=user, managed_root=managed):
        print(name)
    return 0


def cmd_packs_show(args: argparse.Namespace) -> int:
    """Print one pack's version, origin, and rule ids."""
    user, managed = _pack_roots(args)
    try:
        pack = resolve_pack(
            args.name,
            overlay=getattr(args, "overlay", None),
            user_root=user,
            managed_root=managed,
        )
    except PackError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{pack.name}  {pack.version}  {pack.origin}")
    for rule in pack.policies.rules:
        print(f"{rule.id}  {rule.rule_type}")
    return 0


def cmd_packs_load(args: argparse.Namespace) -> int:
    """Compile a pack into the policy store."""
    args.pack = args.name
    loaded = _load_named_pack(args)
    if isinstance(loaded, int):
        return loaded
    print(f"loaded {loaded.name} {loaded.version}")
    return 0


def cmd_packs_update(args: argparse.Namespace) -> int:
    """Fetch pinned packs and install them only when integrity checks pass."""
    user, managed = _pack_roots(args)
    pins = Path(args.pins) if args.pins else builtin_data() / "pins.toml"
    try:
        key = _hmac_key(args.hmac_key)
        names = [args.name] if args.name else pinned_names(pins)
    except PackError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not names:
        print("no pinned packs to update", file=sys.stderr)
        return 1
    failed = 0
    for name in names:
        result = update_pack(
            name,
            db=args.db,
            source=args.source,
            ref=args.ref,
            root=args.root,
            expect=args.expect,
            signature=args.signature,
            hmac_key=key,
            pins=pins,
            managed_root=managed,
            user_root=user,
        )
        if result.refused:
            failed += 1
            print(f"refused {name}: {result.reason}", file=sys.stderr)
            continue
        print(f"updated {name} {result.version} {result.source_hash}")
    return 1 if failed else 0


def cmd_packs_diff(args: argparse.Namespace) -> int:
    """Diff a resolved pack against another policy directory."""
    user, managed = _pack_roots(args)
    try:
        pack = resolve_pack(
            args.name,
            overlay=getattr(args, "overlay", None),
            user_root=user,
            managed_root=managed,
        )
        other = read_policy_set(args.other)
    except (PackError, PolicyLanguageError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{pack.name} {pack.version}")
    print(render_change(diff_sets(pack.policies, other)), end="")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the OpenAI-compatible AI gateway, or the MCP proxy.

    The AI gateway binds one client configuration. Agent names outside that
    configuration are denied. ``--purposes`` constrains a purpose header.
    ``--purpose`` is the purpose claimed when a request has none. Authorization
    is the use-case register: an empty register denies the call.
    ``--pack`` loads that pack first and, when ``--purposes`` is omitted, uses
    the pack's purposes as the header list. Those names do not authorize the
    call until they are registered use cases.
    ``--provider`` accepts ``openai-compatible`` only. ``--mcp --root-config``
    fronts MCP servers instead, and does not start the model gateway. The
    proxy's header list is the ``purposes`` list in that file, not ``--purpose``
    and not ``--purposes``. The register authorizes that call too.
    ``--provider-key``, or ``KOGNITA_PROVIDER_KEY`` when it is omitted, is the
    gateway's own upstream credential, sent as ``Authorization: Bearer`` on
    every forwarded call. A caller that presents a provider credential of its
    own (``authorization``, ``x-api-key``, ``api-key``, ``x-goog-api-key``) is
    denied and flagged; the gateway never forwards it. With no key configured,
    the gateway forwards no provider credential at all.
    """
    if args.mcp:
        if args.pack or args.overlay:
            print(
                "serve --pack is the AI gateway; the MCP proxy reads its pack from --root-config",
                file=sys.stderr,
            )
            return 2
        return _cmd_serve_mcp(args)
    if not args.upstream:
        print("serve --provider openai-compatible requires --upstream", file=sys.stderr)
        return 2
    engine = make_engine(args.db)
    create_all(engine)
    evidence = EvidenceWriter(engine)
    if args.otel:
        failed = _attach_otel(evidence)
        if failed:
            return failed
    pack = None
    if args.pack:
        loaded = _load_named_pack(args)
        if isinstance(loaded, int):
            return loaded
        pack = loaded
        if not args.purposes:
            args.purposes = list(loaded.purposes)
    gateway = Gateway(
        engine=engine,
        evidence=evidence,
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
        pack=pack,
        provider_key=args.provider_key or os.environ.get("KOGNITA_PROVIDER_KEY") or None,
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


def _recorder_session(args: argparse.Namespace):
    engine = make_engine(args.db)
    create_all(engine)
    return session_scope(engine)


def cmd_recorder_runs(args: argparse.Namespace) -> int:
    """List recent runs from evidence."""
    from kognita.recorder import RecorderError, recent_runs

    try:
        with _recorder_session(args) as session:
            rows = recent_runs(
                session,
                actor=args.actor,
                tool=args.tool,
                outcome=args.outcome,
                limit=args.limit,
            )
    except RecorderError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    if not rows:
        print("no runs in evidence")
        return 0
    for row in rows:
        actors = ",".join(row["actors"]) or "-"
        tools = ",".join(row["tools"]) or "-"
        print(
            f"{row['run_id']}  {row['latest_outcome'] or '-'}  "
            f"{actors}  {tools}  decisions {row['decisions']}"
        )
    return 0


def cmd_recorder_show(args: argparse.Namespace) -> int:
    """Print one run's decisions, citations, and evidence."""
    from kognita.recorder import RecorderError, show_run

    try:
        with _recorder_session(args) as session:
            report = show_run(session, args.run_id)
    except RecorderError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_recorder_export(args: argparse.Namespace) -> int:
    """Write one run as a self-verifying evidence artifact."""
    from kognita.recorder import RecorderError, export_run

    try:
        with _recorder_session(args) as session:
            payload = export_run(session, args.run_id)
    except RecorderError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
        print(
            f"wrote {payload['event_count']} events "
            f"({payload['interest_count']} of interest) to {args.output}"
        )
    else:
        print(text)
    return 0


def cmd_recorder_alerts(args: argparse.Namespace) -> int:
    """Print flight-recorder alerts from evidence."""
    from kognita.recorder import alerts

    with _recorder_session(args) as session:
        report = alerts(session)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_recorder_effectiveness(args: argparse.Namespace) -> int:
    """Print effectiveness rates from evidence."""
    from kognita.recorder import effectiveness

    with _recorder_session(args) as session:
        report = effectiveness(session, usecase_id=args.usecase)
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


def _compile_policy(
    db: str,
    policies: Any,
    *,
    source: Path,
    close_absent: bool,
    pack: str | None = None,
    source_hash_value: str | None = None,
) -> int:
    """Compile rows and, when any stored row changes, append POLICY_CHANGE.

    The event is written in the same transaction as the rows. A load that
    writes nothing appends nothing. A pack load names the pack and uses the
    hash of the files that were compiled.
    """
    engine = make_engine(db)
    create_all(engine)
    moment = datetime.now(timezone.utc)
    digest = source_hash(source) if source_hash_value is None else source_hash_value
    with session_scope(engine) as session:
        before = policies_from_session(session, at=moment)
        written = compile_rules(session, policies, at=moment, close_absent=close_absent)
        after = policies_from_session(session, at=moment)
        added, modified, removed = _changed_ids(before, after)
        if added or modified or removed:
            payload: dict[str, Any] = {
                "action": "LOAD",
                "added": added,
                "modified": modified,
                "removed": removed,
                "source_hash": digest,
            }
            if pack:
                payload["pack"] = pack
            EvidenceWriter(engine).emit(
                session,
                correlation_id="policy-load" if not pack else f"pack-load:{pack}",
                event_type=EventType.POLICY_CHANGE,
                actor_type=ActorType.SYSTEM,
                actor_id="kognita policy load" if not pack else "kognita packs load",
                payload=payload,
            )
        return written


def _policies_in(db: str, as_of: datetime | None) -> Any:
    engine = make_engine(db)
    create_all(engine)
    with session_scope(engine) as session:
        return policies_from_session(session, at=as_of)


def _parse_when(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{value!r} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _revocation_target(args: argparse.Namespace) -> tuple[str, str]:
    for name in DIMENSIONS:
        value = getattr(args, name)
        if value:
            return name, value
    raise RevocationError("Revocation: name one dimension")


def _source_view(source):
    from dataclasses import asdict

    body = asdict(source)
    for key, value in list(body.items()):
        if isinstance(value, datetime):
            body[key] = value.isoformat()
    return body


def _cmd_source(args: argparse.Namespace) -> int:
    """Register, supersede, or list approved sources."""
    from sqlmodel import select

    from kognita.models import EvidenceEvent
    from kognita.sources import Source, _load, register_source, supersede_source

    engine = make_engine(args.db)
    create_all(engine)
    evidence = EvidenceWriter(engine)
    command = args.source_command
    try:
        with session_scope(engine) as session:
            if command == "list":
                rows = sorted(
                    (_source_view(item) for item in _load(session).entries),
                    key=lambda row: (row["source_id"], row["version"]),
                )
                _print_json(rows)
                return 0
            if command == "register":
                source = Source(
                    args.id,
                    args.kind,
                    args.version,
                    args.owner,
                    _parse_when(args.effective),
                    expiry=_parse_when(args.expiry) if args.expiry else None,
                )
                registered = register_source(
                    session, source, evidence=evidence, interaction_id=args.interaction
                )
                _print_json(_source_view(registered))
                return 0
            updated = supersede_source(
                session,
                args.id,
                version=args.version,
                by=args.by,
                evidence=evidence,
                interaction_id=args.interaction,
            )
            event = session.exec(
                select(EvidenceEvent)
                .where(EvidenceEvent.event_type == EventType.SOURCE_SUPERSEDED)
                .order_by(EvidenceEvent.sequence.desc())
            ).first()
            payload = event.payload if event is not None else {}
            _print_json({
                "source_id": updated.source_id,
                "version": updated.version,
                "superseded_by": updated.superseded_by,
                "claims": list(payload.get("claims") or []),
                "recommendations": list(payload.get("recommendations") or []),
            })
    except (ValueError, ChainBreak) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _cmd_affected_source(args: argparse.Namespace, source: str) -> int:
    """List dependents of one source. A window is optional."""
    from kognita.sources import affected_by_source

    definition = (getattr(args, "definition", "") or "").strip()
    if definition or any((getattr(args, name, "") or "").strip() for name in ("model", "policy", "usecase")):
        print("Affected clients: name one selector", file=sys.stderr)
        return 2
    start = end = None
    if (getattr(args, "start", "") or "") or (getattr(args, "end", "") or ""):
        if not args.start or not args.end:
            print(
                "Affected clients: --from and --to are both required when either is set",
                file=sys.stderr,
            )
            return 2
        try:
            start = _parse_when(args.start)
            end = _parse_when(args.end)
        except ValueError as exc:
            print(f"Affected clients: {exc}", file=sys.stderr)
            return 2
    engine = make_engine(args.db)
    create_all(engine)
    try:
        with session_scope(engine) as session:
            report = affected_by_source(session, source, start=start, end=end)
    except (ValueError, ChainBreak) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    _print_json(report)
    return 0


def _cmd_affected_definition(args: argparse.Namespace) -> int:
    """List work produced under a previous definition. A window is optional."""
    definition = (getattr(args, "definition", "") or "").strip()
    if args.model or args.policy or args.usecase:
        print(
            "Affected: --definition does not combine with --model, --policy, or --usecase",
            file=sys.stderr,
        )
        return 2
    try:
        start = _parse_when(args.start) if args.start else None
        end = _parse_when(args.end) if args.end else None
    except ValueError as exc:
        print(f"Affected: {exc}", file=sys.stderr)
        return 2
    from kognita.definitions import affected_by_definition

    engine = make_engine(args.db)
    create_all(engine)
    try:
        with session_scope(engine) as session:
            report = affected_by_definition(session, definition, start=start, end=end)
    except (ValueError, ChainBreak) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    _print_json(report)
    return 0


def cmd_evidence_affected(args: argparse.Namespace) -> int:
    """List affected clients for a model, policy, use case, or source."""
    source = (getattr(args, "source", "") or "").strip()
    if source:
        return _cmd_affected_source(args, source)
    definition = (getattr(args, "definition", "") or "").strip()
    if definition:
        return _cmd_affected_definition(args)
    if not (getattr(args, "start", "") or "") or not (getattr(args, "end", "") or ""):
        print("Affected clients: --from and --to are required", file=sys.stderr)
        return 2
    try:
        start = _parse_when(args.start)
        end = _parse_when(args.end)
    except ValueError as exc:
        print(f"Affected clients: {exc}", file=sys.stderr)
        return 2
    engine = make_engine(args.db)
    create_all(engine)
    try:
        with session_scope(engine) as session:
            report = affected_clients(
                session,
                model=args.model or "",
                policy_id=args.policy or "",
                usecase=args.usecase or "",
                start=start,
                end=end,
            )
    except BreakerError as exc:
        print(str(exc), file=sys.stderr)
        code = 2 if "name one of" in str(exc) else 1
        return code
    _print_json(report)
    return 0


def _cmd_definition_register(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from kognita.definitions import GovernedTerm, register_term

    engine = make_engine(args.db)
    create_all(engine)
    evidence = EvidenceWriter(engine)
    try:
        term = GovernedTerm(
            args.term,
            args.definition_ref,
            args.owner,
            args.version,
            _parse_when(args.effective),
            args.superseded_by or "",
        )
        with session_scope(engine) as session:
            saved = register_term(
                session, term, evidence=evidence, interaction_id=args.interaction,
            )
            payload = asdict(saved)
    except (ValueError, ChainBreak) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(canonical_json(payload))
    return 0


def _cmd_definition_list(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from kognita.definitions import _registered_terms

    engine = make_engine(args.db)
    create_all(engine)
    try:
        with session_scope(engine) as session:
            terms = sorted(
                _registered_terms(session),
                key=lambda item: (item.term.casefold(), item.version),
            )
            payload = [asdict(item) for item in terms]
    except (ValueError, ChainBreak) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(canonical_json(payload))
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    """Revoke one dimension. The next matching decision is a denial."""
    engine = make_engine(args.db)
    create_all(engine)
    evidence = EvidenceWriter(engine)
    try:
        dimension, value = _revocation_target(args)
        with session_scope(engine) as session:
            row = revoke(
                session,
                dimension,
                value,
                evidence=evidence,
                actor_id=args.actor,
                citation=args.citation,
                owner=args.owner or "",
                limit_usecases=tuple(args.limit_usecase or ()),
                below_classification=args.below or "",
            )
            payload = {
                "revocation_id": row.id,
                "dimension": row.dimension,
                "value": row.value,
                "citation": row.citation,
                "accountable_owners": list(row.accountable_owners),
                "limit_usecases": list(row.limit_usecases),
                "below_classification": row.below_classification,
            }
    except (RevocationError, BreakerError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_json(payload)
    return 0


def cmd_reinstate(args: argparse.Namespace) -> int:
    """Record one owner's cited reinstatement.

    A provider stays open until every accountable owner of every use case
    that lists it has reinstated it. ``effective_to`` is null until then.
    """
    engine = make_engine(args.db)
    create_all(engine)
    evidence = EvidenceWriter(engine)
    try:
        with session_scope(engine) as session:
            row = reinstate(
                session,
                args.revocation_id,
                evidence=evidence,
                actor_id=args.actor,
                citation=args.citation,
            )
            payload = {
                "revocation_id": row.id,
                "dimension": row.dimension,
                "value": row.value,
                "effective_to": row.effective_to.isoformat() if row.effective_to else None,
                "closed": row.effective_to is not None,
            }
    except RevocationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_json(payload)
    return 0


def cmd_breaker_threshold(args: argparse.Namespace) -> int:
    engine = make_engine(args.db)
    create_all(engine)
    evidence = EvidenceWriter(engine)
    try:
        with session_scope(engine) as session:
            row = set_breaker_threshold(
                session,
                args.dimension,
                args.value,
                threshold=args.threshold,
                window_seconds=args.window_seconds,
                evidence=evidence,
                actor_id=args.actor,
                citation=args.citation,
                owner=args.owner or "",
            )
            payload = {
                "id": row.id,
                "dimension": row.dimension,
                "value": row.value,
                "threshold": row.threshold,
                "window_seconds": row.window_seconds,
                "accountable_owner": row.accountable_owner,
                "citation": row.citation,
            }
    except BreakerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_json(payload)
    return 0


def cmd_breaker_evaluate(args: argparse.Namespace) -> int:
    engine = make_engine(args.db)
    create_all(engine)
    evidence = EvidenceWriter(engine)
    try:
        with session_scope(engine) as session:
            tripped = evaluate_breakers(session, evidence=evidence)
    except BreakerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_json(list(tripped))
    return 0


def cmd_breaker_reset(args: argparse.Namespace) -> int:
    engine = make_engine(args.db)
    create_all(engine)
    evidence = EvidenceWriter(engine)
    try:
        with session_scope(engine) as session:
            policy = reset_breaker(
                session,
                args.dimension,
                args.value,
                evidence=evidence,
                actor_id=args.actor,
                citation=args.citation,
            )
            payload = {
                "policy_id": policy.id,
                "dimension": args.dimension,
                "value": args.value,
                "effective_to": policy.effective_to.isoformat() if policy.effective_to else None,
            }
    except BreakerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_json(payload)
    return 0


def cmd_drill_due(args: argparse.Namespace) -> int:
    engine = make_engine(args.db)
    create_all(engine)
    evidence = EvidenceWriter(engine)
    try:
        due = _parse_when(args.at)
    except ValueError as exc:
        print(f"Kill-switch drill: {exc}", file=sys.stderr)
        return 2
    try:
        with session_scope(engine) as session:
            row = schedule_kill_switch_drill(
                session,
                args.usecase,
                due_at=due,
                evidence=evidence,
                actor_id=args.actor,
                citation=args.citation,
            )
            payload = {
                "usecase_id": row.usecase_id,
                "next_due": row.next_due.isoformat(),
            }
    except BreakerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_json(payload)
    return 0


def cmd_drill_run(args: argparse.Namespace) -> int:
    engine = make_engine(args.db)
    create_all(engine)
    evidence = EvidenceWriter(engine)
    try:
        with session_scope(engine) as session:
            payload = run_kill_switch_drill(
                session,
                args.usecase,
                args.agent,
                evidence=evidence,
                actor_id=args.actor,
                citation=args.citation,
                fallback=args.fallback,
            )
    except BreakerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_json(payload)
    return 0


def _cmd_evidence_authority(args):
    from kognita.authority import _lineage
    from kognita.evidence import verify_chain, ChainBreak
    engine = make_engine(args.db if "://" in args.db else "sqlite:///" + args.db)
    with session_scope(engine) as session:
        try:
            verify_chain(session)
            _print_json(_lineage(session, args.action_id))
        except (ValueError, ChainBreak) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return 0


def _cmd_claims(args):
    from dataclasses import asdict
    from kognita.claims import list_claims
    engine = make_engine(args.db)
    create_all(engine)
    try:
        with session_scope(engine) as session:
            claims = list_claims(
                session,
                client_id=getattr(args, "client", None),
                interaction_id=getattr(args, "interaction", None),
            )
    except (ValueError, ChainBreak) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if hasattr(args, "claim_id"):
        claims = [c for c in claims if c.claim_id == args.claim_id]
        if not claims:
            print("C1-UNKNOWN: claim is not recorded", file=sys.stderr)
            return 2
        print(canonical_json(asdict(claims[0])))
    else:
        print(canonical_json([asdict(c) for c in claims]))
    return 0


def _cmd_export_request(args: argparse.Namespace) -> int:
    from kognita.learning import ExportRequest, request_export

    engine = make_engine(args.db)
    create_all(engine)
    with session_scope(engine) as session:
        result = request_export(
            session,
            ExportRequest(
                purpose=args.purpose, anonymisation=args.anonymisation,
                validation=args.validation, approval=args.approval,
                requester=args.requester, records=tuple(args.record),
            ),
            evidence=EvidenceWriter(engine),
        )
    _print_json(result)
    return 0 if result["action"] == "APPROVED" else 2


def _cmd_export_exclude(args: argparse.Namespace) -> int:
    from kognita.learning import Exclusion, add_exclusion

    engine = make_engine(args.db)
    create_all(engine)
    try:
        with session_scope(engine) as session:
            exclusion = add_exclusion(
                session,
                Exclusion(reason=args.reason, client_id=args.client, record_id=args.record),
                evidence=EvidenceWriter(engine), actor_id=args.actor,
            )
    except ValueError as exc:
        _print_json({"action": "DENIED", "reason_codes": [str(exc)]})
        return 2
    _print_json({
        "action": "EXCLUDED", "client_id": exclusion.client_id,
        "record_id": exclusion.record_id, "reason": exclusion.reason,
        "recorded_at": exclusion.recorded_at.isoformat(),
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kognita",
        description="Prove an AI answer was permitted — and evidence it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    agent = sub.add_parser("agent")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    deploy = agent_sub.add_parser("deploy")
    deploy.add_argument("--manifest", required=True)
    deploy.add_argument("--db", default="kognita.db")
    deploy.add_argument("--actor", default="", help="local operator identity; defaults to the manifest owner")
    deploy.set_defaults(func=_agent_deploy)

    evaluation = sub.add_parser("eval", help="record evaluation results and report fairness")
    evaluation_sub = evaluation.add_subparsers(dest="eval_command", required=True)
    eval_record = evaluation_sub.add_parser("record", help="record an evaluation result")
    eval_record.add_argument("--db", default="kognita.db")
    eval_record.add_argument("--agent", required=True)
    eval_record.add_argument("--version", required=True)
    eval_record.add_argument("--model-version", required=True)
    eval_record.add_argument("--level", required=True, choices=["model", "agent", "system"])
    eval_record.add_argument("--metric", action="append", default=[], help="name=value")
    eval_record.set_defaults(func=_cmd_eval_record)
    eval_report = evaluation_sub.add_parser("report", help="fairness rates from origination evidence")
    eval_report.add_argument("--db", default="kognita.db")
    eval_report.add_argument("--segments", help="JSON file of attribute, client, value")
    eval_report.add_argument("--segment", action="append", default=[], help="attribute=client:value")
    eval_report.add_argument("--proxy", action="append", default=[])
    eval_report.add_argument("--threshold", required=True)
    eval_report.add_argument("--usecase", default="")
    eval_report.add_argument("--register-metric", action="store_true")
    eval_report.set_defaults(func=_cmd_eval_report)

    learning_export = sub.add_parser("export", help="govern record selection for learning")
    export_sub = learning_export.add_subparsers(dest="export_command", required=True)
    export_request = export_sub.add_parser("request", help="request an evidenced selection manifest")
    export_request.add_argument("--db", default="kognita.db", help="store path or URL")
    for control in ("purpose", "anonymisation", "validation", "approval", "requester"):
        export_request.add_argument(f"--{control}", default="")
    export_request.add_argument("--record", action="append", default=[], help="record id (repeatable)")
    export_request.set_defaults(func=_cmd_export_request)

    export_exclude = export_sub.add_parser("exclude", help="permanently exclude a client or record")
    export_exclude.add_argument("--db", default="kognita.db", help="store path or URL")
    target = export_exclude.add_mutually_exclusive_group(required=True)
    target.add_argument("--client", default="")
    target.add_argument("--record", default="")
    export_exclude.add_argument("--reason", required=True)
    export_exclude.add_argument("--actor", required=True)
    export_exclude.set_defaults(func=_cmd_export_exclude)

    doctor = sub.add_parser("doctor", help="report installed extras and store health")
    doctor.add_argument("--db", help="optional store to check the evidence chain of")
    doctor.set_defaults(func=cmd_doctor)

    claims = sub.add_parser("claims", help="inspect recorded claims")
    claims_sub = claims.add_subparsers(dest="claims_command", required=True)
    show_claim = claims_sub.add_parser("show")
    show_claim.add_argument("claim_id")
    show_claim.add_argument("--db", default="kognita.db")
    show_claim.set_defaults(func=_cmd_claims)
    list_claim = claims_sub.add_parser("list")
    list_claim.add_argument("--client")
    list_claim.add_argument("--interaction")
    list_claim.add_argument("--db", default="kognita.db")
    list_claim.set_defaults(func=_cmd_claims)

    source_cmd = sub.add_parser("source", help="approved sources and house-view changes")
    source_sub = source_cmd.add_subparsers(dest="source_command", required=True)
    source_register = source_sub.add_parser("register", help="register one approved source version")
    source_register.add_argument("--id", "--source", dest="id", required=True)
    source_register.add_argument("--kind", required=True)
    source_register.add_argument("--version", required=True)
    source_register.add_argument("--owner", required=True)
    source_register.add_argument("--effective", required=True, help="ISO timestamp")
    source_register.add_argument("--expiry", default="", help="ISO timestamp")
    source_register.add_argument("--interaction", default="source")
    source_register.add_argument("--db", default="kognita.db")
    source_register.set_defaults(func=_cmd_source)
    source_supersede = source_sub.add_parser("supersede", help="supersede one source version")
    source_supersede.add_argument("--id", "--source", dest="id", required=True)
    source_supersede.add_argument("--version", required=True)
    source_supersede.add_argument("--by", "--superseded-by", dest="by", required=True, help="successor version")
    source_supersede.add_argument("--interaction", default="source")
    source_supersede.add_argument("--db", default="kognita.db")
    source_supersede.set_defaults(func=_cmd_source)
    source_list = source_sub.add_parser("list", help="list registered source versions")
    source_list.add_argument("--db", default="kognita.db")
    source_list.set_defaults(func=_cmd_source)
    definition = sub.add_parser("definition", help="governed business definitions")
    definition_sub = definition.add_subparsers(dest="definition_command", required=True)
    register_definition = definition_sub.add_parser("register", help="register one term version")
    register_definition.add_argument("--term", required=True)
    register_definition.add_argument("--definition-ref", required=True)
    register_definition.add_argument("--owner", required=True)
    register_definition.add_argument("--version", required=True)
    register_definition.add_argument("--effective", required=True)
    register_definition.add_argument("--superseded-by", default="")
    register_definition.add_argument("--interaction", default="definition")
    register_definition.add_argument("--db", default="kognita.db")
    register_definition.set_defaults(func=_cmd_definition_register)
    list_definition = definition_sub.add_parser("list", help="list registered term versions")
    list_definition.add_argument("--db", default="kognita.db")
    list_definition.set_defaults(func=_cmd_definition_list)

    evidence = sub.add_parser("evidence", help="evidence plane operations")
    evidence_sub = evidence.add_subparsers(dest="evidence_command", required=True)

    authority = evidence_sub.add_parser("authority", help="reconstruct authority for an action request id")
    authority.add_argument("action_id")
    authority.add_argument("--db", default="kognita.db")
    authority.set_defaults(func=_cmd_evidence_authority)

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

    affected = evidence_sub.add_parser(
        "affected",
        help="list affected clients for a model, policy, use case, or source, or work produced under a previous definition",
    )
    affected.add_argument("--db", default="kognita.db", help="store path or URL")
    affected.add_argument("--model", default="", help="model version, name, or card")
    affected.add_argument("--policy", default="", help="policy id")
    affected.add_argument("--usecase", default="", help="use case id")
    affected.add_argument("--source", default="", help="source id")
    affected.add_argument(
        "--definition", default="", help="governed term or its definition reference",
    )
    affected.add_argument(
        "--from",
        dest="start",
        default="",
        help="window start, inclusive; required unless --source or --definition is set",
    )
    affected.add_argument(
        "--to",
        dest="end",
        default="",
        help="window end, exclusive; required unless --source or --definition is set",
    )
    affected.set_defaults(func=cmd_evidence_affected)

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
        "--provider-key",
        default=None,
        help=(
            "the gateway's own upstream credential, sent as Authorization: "
            "Bearer on every forwarded call; or set KOGNITA_PROVIDER_KEY. "
            "A caller-supplied provider credential is denied and flagged, "
            "never forwarded. With neither set, no provider credential is "
            "forwarded"
        ),
    )
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
            "purpose header the AI gateway accepts; repeatable. Distinct from "
            "--purpose, which is only the purpose claimed when a request has "
            "none. With no --purposes, the gateway uses the loaded pack's "
            "purposes as that header list. Authorization is the use-case "
            "register, and an empty register denies the call"
        ),
    )
    serve.add_argument("--pack", help="starter or installed pack to load before serving")
    serve.add_argument(
        "--overlay",
        help="policy directory merged on top of --pack; same rule id, user wins",
    )
    serve.add_argument(
        "--user-root",
        help="directory of user packs that shadow installed packs by name",
    )
    serve.add_argument(
        "--managed-root",
        help="directory written by kognita packs update",
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
    serve.add_argument(
        "--otel",
        action="store_true",
        help=(
            "export evidence to OpenTelemetry. Requires kognita[otel] and "
            "OTEL_EXPORTER_OTLP_ENDPOINT. Refuses to start without them"
        ),
    )
    serve.set_defaults(func=cmd_serve)

    recorder = sub.add_parser("recorder", help="flight recorder")
    recorder_sub = recorder.add_subparsers(dest="recorder_command", required=True)

    recorder_runs = recorder_sub.add_parser(
        "runs", help="recent runs filtered by actor, tool, and outcome"
    )
    recorder_runs.add_argument("--db", default="kognita.db", help="store path or URL")
    recorder_runs.add_argument("--actor", help="actor on the run")
    recorder_runs.add_argument("--tool", help="tool on the run")
    recorder_runs.add_argument("--outcome", help="decision outcome on the run")
    recorder_runs.add_argument("--limit", type=int, default=20)
    recorder_runs.add_argument("--json", action="store_true", help="print JSON")
    recorder_runs.set_defaults(func=cmd_recorder_runs)

    recorder_show = recorder_sub.add_parser(
        "show", help="decisions, citations, and evidence for one run"
    )
    recorder_show.add_argument("run_id")
    recorder_show.add_argument("--db", default="kognita.db", help="store path or URL")
    recorder_show.set_defaults(func=cmd_recorder_show)

    recorder_export = recorder_sub.add_parser(
        "export", help="export one run as self-verifying JSON"
    )
    recorder_export.add_argument("run_id")
    recorder_export.add_argument("--db", default="kognita.db", help="store path or URL")
    recorder_export.add_argument("-o", "--output", help="write to a file instead of stdout")
    recorder_export.set_defaults(func=cmd_recorder_export)

    recorder_alerts = recorder_sub.add_parser(
        "alerts", help="budget, denials, chain break, and confidence drift"
    )
    recorder_alerts.add_argument("--db", default="kognita.db", help="store path or URL")
    recorder_alerts.set_defaults(func=cmd_recorder_alerts)

    recorder_rates = recorder_sub.add_parser(
        "effectiveness",
        help="RM acceptance, edit, rejection, and wrong-client rates",
    )
    recorder_rates.add_argument("--db", default="kognita.db", help="store path or URL")
    recorder_rates.add_argument("--usecase", help="one use case; omit for every use case")
    recorder_rates.set_defaults(func=cmd_recorder_effectiveness)

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

    revoke_cmd = sub.add_parser(
        "revoke",
        help="revoke one dimension; the next matching decision is a denial",
    )
    target = revoke_cmd.add_mutually_exclusive_group(required=True)
    for name in DIMENSIONS:
        target.add_argument(f"--{name}", help=f"{name} to revoke")
    revoke_cmd.add_argument("--db", default="kognita.db", help="store path or URL")
    revoke_cmd.add_argument("--actor", required=True, help="person recording the revocation")
    revoke_cmd.add_argument("--citation", required=True, help="rule the revocation cites")
    revoke_cmd.add_argument(
        "--owner",
        default="",
        help="accountable owner, when the dimension does not name one",
    )
    revoke_cmd.add_argument(
        "--limit-usecase",
        action="append",
        default=None,
        help="provider only: use case that may still call it; repeatable",
    )
    revoke_cmd.add_argument(
        "--below",
        default="",
        help="provider only: exclusive classification ceiling, for example C2",
    )
    revoke_cmd.set_defaults(func=cmd_revoke)

    reinstate_cmd = sub.add_parser(
        "reinstate",
        help="close a revocation; the actor must be an accountable owner it names",
    )
    reinstate_cmd.add_argument("--db", default="kognita.db", help="store path or URL")
    reinstate_cmd.add_argument("--id", dest="revocation_id", type=int, required=True)
    reinstate_cmd.add_argument("--actor", required=True)
    reinstate_cmd.add_argument("--citation", required=True)
    reinstate_cmd.set_defaults(func=cmd_reinstate)

    breaker = sub.add_parser("breaker", help="circuit-breaker thresholds")
    breaker_sub = breaker.add_subparsers(dest="breaker_command", required=True)

    threshold = breaker_sub.add_parser("threshold", help="record how many flags trip a breaker")
    threshold.add_argument("--db", default="kognita.db", help="store path or URL")
    threshold.add_argument("--dimension", required=True, choices=BREAKER_DIMENSIONS)
    threshold.add_argument("--value", required=True)
    threshold.add_argument("--threshold", type=int, required=True)
    threshold.add_argument("--window-seconds", type=int, required=True)
    threshold.add_argument("--actor", required=True)
    threshold.add_argument("--citation", required=True)
    threshold.add_argument("--owner", default="")
    threshold.set_defaults(func=cmd_breaker_threshold)

    evaluate_cmd = breaker_sub.add_parser(
        "evaluate",
        help="trip each threshold whose flags in the window have reached it",
    )
    evaluate_cmd.add_argument("--db", default="kognita.db", help="store path or URL")
    evaluate_cmd.set_defaults(func=cmd_breaker_evaluate)

    reset_cmd = breaker_sub.add_parser(
        "reset",
        help="close a tripped breaker; the actor must be its accountable owner",
    )
    reset_cmd.add_argument("--db", default="kognita.db", help="store path or URL")
    reset_cmd.add_argument("--dimension", required=True, choices=BREAKER_DIMENSIONS)
    reset_cmd.add_argument("--value", required=True)
    reset_cmd.add_argument("--actor", required=True)
    reset_cmd.add_argument("--citation", required=True)
    reset_cmd.set_defaults(func=cmd_breaker_reset)

    drill = sub.add_parser("drill", help="kill-switch drills for a high-materiality use case")
    drill_sub = drill.add_subparsers(dest="drill_command", required=True)

    due = drill_sub.add_parser("due", help="set the next drill date")
    due.add_argument("--db", default="kognita.db", help="store path or URL")
    due.add_argument("--usecase", required=True)
    due.add_argument("--at", required=True, help="ISO timestamp the drill is due")
    due.add_argument("--actor", required=True)
    due.add_argument("--citation", required=True)
    due.set_defaults(func=cmd_drill_due)

    run = drill_sub.add_parser("run", help="engage the kill switch and record the denial")
    run.add_argument("--db", default="kognita.db", help="store path or URL")
    run.add_argument("--usecase", required=True)
    run.add_argument("--agent", required=True)
    run.add_argument("--actor", required=True)
    run.add_argument("--citation", required=True)
    run.add_argument("--fallback", required=True, help="how the fallback was reached")
    run.set_defaults(func=cmd_drill_run)

    packs = sub.add_parser("packs", help="list, load, update, and diff packs")
    packs_sub = packs.add_subparsers(dest="packs_command", required=True)
    packs_list = packs_sub.add_parser("list", help="list packs found on disk")
    packs_list.add_argument("--user-root")
    packs_list.add_argument("--managed-root")
    packs_list.set_defaults(func=cmd_packs_list)
    packs_show = packs_sub.add_parser("show", help="print one pack and its rules")
    packs_show.add_argument("name")
    packs_show.add_argument("--overlay")
    packs_show.add_argument("--user-root")
    packs_show.add_argument("--managed-root")
    packs_show.set_defaults(func=cmd_packs_show)
    packs_load = packs_sub.add_parser(
        "load", help="compile a pack into the policy store"
    )
    packs_load.add_argument("name")
    packs_load.add_argument("--overlay")
    packs_load.add_argument("--user-root")
    packs_load.add_argument("--managed-root")
    packs_load.add_argument("--db", default="kognita.db")
    packs_load.set_defaults(func=cmd_packs_load)
    packs_update = packs_sub.add_parser(
        "update",
        help="fetch a pinned pack from the Kognita repository and verify it",
    )
    packs_update.add_argument(
        "name",
        nargs="?",
        help="pack to update; omit to update every locally pinned pack",
    )
    packs_update.add_argument("--source", default=DEFAULT_PACK_SOURCE)
    packs_update.add_argument("--ref", default="main")
    packs_update.add_argument("--root", default="src/kognita/packs/data")
    packs_update.add_argument(
        "--expect",
        help="sha256:<hex> pin for this update; replaces the pin file for this call",
    )
    packs_update.add_argument(
        "--signature",
        help="hmac:<hex> over the canonical file manifest",
    )
    packs_update.add_argument(
        "--hmac-key",
        help="HMAC key file, or set KOGNITA_PACKS_HMAC_KEY",
    )
    packs_update.add_argument("--pins")
    packs_update.add_argument("--user-root")
    packs_update.add_argument("--managed-root")
    packs_update.add_argument("--db", default="kognita.db")
    packs_update.set_defaults(func=cmd_packs_update)
    packs_diff = packs_sub.add_parser(
        "diff", help="diff one pack's policy against another directory"
    )
    packs_diff.add_argument("name")
    packs_diff.add_argument("other", help="directory of the other policy version")
    packs_diff.add_argument("--overlay")
    packs_diff.add_argument("--user-root")
    packs_diff.add_argument("--managed-root")
    packs_diff.set_defaults(func=cmd_packs_diff)

    return parser


def _agent_deploy(args):
    from kognita.registry import register, approve_deployment, get
    from kognita.policy import parse_yaml

    text = Path(args.manifest).read_text()
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError:
        manifest = parse_yaml(text)
    if not isinstance(manifest, dict):
        raise ValueError("S1-DEPLOYMENT: manifest must be an object")
    args.actor = args.actor or manifest.get("owner_exec", "")
    level = manifest.get("autonomy_level", 0)
    if type(level) is not int or level != 0:
        raise ValueError("S1-PROMOTION: deploy at L0; raise autonomy through promote()")
    engine = make_engine(args.db)
    create_all(engine)
    try:
        with session_scope(engine) as session:
            current = get(session, manifest["name"])
            if current is not None and (current.owner_exec != args.actor or
                                        manifest["owner_exec"] != current.owner_exec):
                raise ValueError("S1-DEPLOYMENT: existing accountable owner required")
            agent = register(session, name=manifest["name"], owner_exec=manifest["owner_exec"],
                             version=manifest.get("version", "0.1.0"),
                             risk_class=manifest.get("risk_class", "MEDIUM"),
                             materiality_tier=manifest.get("materiality_tier", "T2"))
            from kognita.registry import _provider_key_finding
            manifest = _provider_key_finding(session, agent.name, manifest, EvidenceWriter(engine))
            for key in ("purpose", "use_cases", "permitted_systems", "tool_allowlist", "blast_radius", "field_allowlist",
                        "review_date", "environment", "principal", "deployment_date", "model"):
                value = manifest.get(key)
                if key in ("review_date", "deployment_date") and value is not None:
                    value = datetime.fromisoformat(str(value))
                setattr(agent, key, value)
            type(agent).model_validate(agent.model_dump())
            approve_deployment(session, agent.name, deployment=manifest["deployment"],
                               evidence=EvidenceWriter(engine), actor_id=args.actor)
            session.flush()
            payload = {"name": agent.name, "deployment_hash": agent.deployment_hash}
    except ValueError as exc:
        if str(exc).startswith("EVALUATION_"):
            _print_json({"action": "DENIED", "reason": str(exc)})
            return 2
        raise
    _print_json(payload)
    return 0


def _metric_pairs(items: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        if not sep or not name or name in metrics:
            raise ValueError("EVALUATION_METRICS_INVALID")
        try:
            if raw.lower() in {"true", "false"}:
                raise ValueError("bool")
            metrics[name] = float(raw)
        except ValueError as exc:
            raise ValueError("EVALUATION_METRICS_INVALID") from exc
    return metrics


def _cmd_eval_record(args: argparse.Namespace) -> int:
    from kognita.evaluation import EvaluationResult, record_evaluation

    engine = make_engine(args.db)
    create_all(engine)
    try:
        result = EvaluationResult(
            agent_id=args.agent,
            version=args.version,
            model_version=args.model_version,
            level=args.level,
            metrics=_metric_pairs(args.metric),
        )
        with session_scope(engine) as session:
            record_evaluation(session, result, evidence=EvidenceWriter(engine))
    except ValueError as exc:
        _print_json({"action": "DENIED", "reason": str(exc)})
        return 2
    _print_json({
        "action": "RECORDED",
        "agent_id": result.agent_id,
        "version": result.version,
        "model_version": result.model_version,
        "level": result.level,
        "metrics": dict(result.metrics),
    })
    return 0


def _segment_pairs(items: list[str]) -> dict[str, dict[str, str]]:
    segments: dict[str, dict[str, str]] = {}
    for item in items:
        name, sep, rest = item.partition("=")
        client, colon, value = rest.partition(":")
        if not sep or not colon or not name.strip() or not client.strip() or not value.strip():
            raise ValueError("FAIRNESS_SEGMENTS_REQUIRED")
        segments.setdefault(name.strip(), {})[client.strip()] = value.strip()
    return segments


def _cmd_eval_report(args: argparse.Namespace) -> int:
    from kognita.evaluation import fairness_report

    engine = make_engine(args.db)
    create_all(engine)
    try:
        threshold = float(args.threshold)
        if str(args.threshold).lower() in {"true", "false"}:
            raise ValueError("bool")
        segments = _segment_pairs(args.segment)
        if args.segments:
            loaded = json.loads(Path(args.segments).read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("FAIRNESS_SEGMENTS_REQUIRED")
            for name, mapping in loaded.items():
                if not isinstance(mapping, dict):
                    raise ValueError("FAIRNESS_SEGMENTS_REQUIRED")
                bucket = segments.setdefault(str(name), {})
                for client, value in mapping.items():
                    bucket[str(client)] = value
        with session_scope(engine) as session:
            report = fairness_report(
                session,
                segments=segments,
                disparity_threshold=threshold,
                proxy_attributes=tuple(args.proxy),
                evidence=EvidenceWriter(engine),
                usecase_id=args.usecase,
                register_monitored=args.register_metric,
            )
            payload = {
                "action": "REPORTED",
                "flagged": report.flagged,
                "threshold": report.threshold,
                "proxy_attributes": list(report.proxy_attributes),
                "risk_owners": list(report.risk_owners),
                "rates": [dict(row) for row in report.rates],
                "disparities": [dict(row) for row in report.disparities],
                "monitored_metric": report.monitored_metric,
            }
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        _print_json({"action": "DENIED", "reason": str(exc)})
        return 2
    _print_json(payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
