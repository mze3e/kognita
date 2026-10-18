"""Starter packs, the private-banking pack, and pack distribution.

Packs are policy files. ``packs update`` installs a pack only after a local
pin or an HMAC matches. A failed check or a failed fetch keeps the installed
pack and appends a refusal. Deciding does not fetch.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from sqlmodel import select

from kognita.cli import main
from kognita.db import make_engine, session_scope
from kognita.envelope import Envelope, RuleContext
from kognita.evidence import EvidenceWriter
from kognita.gateway import ClientConfiguration
from kognita.governance import decide, load_snapshot
from kognita.models import EvidenceEvent, Policy
from kognita.registry import register
from kognita.revocation import revoke
from kognita.packs import (
    PACK_NAMES,
    PRIVATE_BANKING_PACK,
    STARTER_PACKS,
    PackError,
    PackNetworkError,
    UpdateResult,
    builtin_data,
    content_hash,
    load_source_hash,
    resolve_pack,
    update_pack,
)
from kognita.packs.catalog import manifest_bytes, merge_policy_sets, read_pack_files
from kognita.policy import compile_rules
from kognita.rules import principal_equality
from kognita.vocabulary import CheckResult, EventType, Outcome
from test_gateway import BENIGN, PATH, _Upstream, _body, _gateway, _live, _register
from test_packaging import PACKAGE_ROOT, _engine_modules_on_disk

AS_OF = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
ROOT = builtin_data()


def _roots(tmp_path: Path) -> list[str]:
    return [
        "--user-root",
        str(tmp_path / "user"),
        "--managed-root",
        str(tmp_path / "managed"),
    ]


def _events(database: Path) -> list[tuple[EventType, dict]]:
    engine = make_engine(database)
    with session_scope(engine) as session:
        rows = session.exec(select(EvidenceEvent).order_by(EvidenceEvent.sequence)).all()
        return [(row.event_type, dict(row.payload)) for row in rows]


def _cited(evaluation, session) -> set[str]:
    rows = session.exec(select(Policy)).all()
    by_pk = {row.id: row.rule.get("id") for row in rows}
    return {
        by_pk.get(check.policy_id)
        for check in evaluation.basis()
        if check.policy_id is not None
    }


def _files_transport(name: str, files: dict[str, bytes]):
    def transport(url: str) -> bytes:
        path = unquote(urlparse(url).path)
        marker = f"/{name}/"
        index = path.rfind(marker)
        if index < 0:
            raise PackError(f"unexpected pack url {url}")
        rel = path[index + len(marker) :]
        if rel not in files:
            raise PackError(f"unexpected pack file {rel}")
        return files[rel]

    return transport


def _policy(equals: str = "subject_owner", **rule):
    body = {"equals": equals, "on_violation": "fail"}
    body.update(rule)
    return type(
        "Policy",
        (),
        {
            "id": "own-subject",
            "rule_type": "PRINCIPAL_EQUALITY",
            "regime": "ACCESS",
            "citation": "Principal equality. Source: Kognita test.",
            "rule": body,
        },
    )()


def _context(principal: str, **attributes) -> RuleContext:
    return RuleContext(
        envelope=Envelope(principal=principal, purpose="ACCESS", tool="read_record"),
        attributes=attributes,
    )


def test_pack_names_match_the_roadmap():
    assert STARTER_PACKS == (
        "role-based-access",
        "geo-fencing",
        "data-classification",
        "time-gated-access",
        "two-signature-approval-chains",
    )
    assert PRIVATE_BANKING_PACK == "private-banking"
    assert PACK_NAMES == STARTER_PACKS + (PRIVATE_BANKING_PACK,)


def test_every_pack_validates_strict_and_its_scenarios_pass(capsys):
    for name in PACK_NAMES:
        directory = ROOT / name / "policy"
        assert main(["policy", "validate", str(directory), "--strict"]) == 0
        assert main(["policy", "test", str(directory)]) == 0
    captured = capsys.readouterr()
    assert "FAIL" not in captured.err


def test_every_rule_cites_a_source_and_the_pin_matches_the_bytes():
    import tomllib

    pins = tomllib.loads((ROOT / "pins.toml").read_text(encoding="utf-8"))
    assert pins["repository"] == "https://github.com/mze3e/kognita"
    for name in PACK_NAMES:
        files = read_pack_files(ROOT / name)
        assert pins["packs"][name]["sha256"] == content_hash(files)
        assert pins["packs"][name]["version"] == "1.0.0"
        for rel, blob in files.items():
            if not rel.endswith(".yaml") or rel.endswith(".scenarios.yaml"):
                continue
            text = blob.decode("utf-8")
            assert "Source: Kognita" in text
            lowered = text.lower()
            assert "bmos" not in lowered
            assert "patent" not in lowered


def test_principal_equality_matches_only_the_same_string():
    match = principal_equality(_policy(), _context("alice", subject_owner="alice"))
    assert match[0].result is CheckResult.PASS
    assert "principal matches subject_owner" in match[0].citation

    mismatch = principal_equality(_policy(), _context("alice", subject_owner="bob"))
    assert mismatch[0].result is CheckResult.FAIL
    assert "does not match" in mismatch[0].citation

    email = principal_equality(_policy(), _context("alice@corp", subject_owner="alice"))
    assert email[0].result is CheckResult.FAIL

    missing = principal_equality(_policy(), _context("alice"))
    assert missing[0].result is CheckResult.FAIL
    assert "could not be evaluated" in missing[0].citation

    blank = principal_equality(_policy(), _context("", subject_owner="alice"))
    assert blank[0].result is CheckResult.FAIL

    number = principal_equality(_policy(), _context("alice", subject_owner=1))
    assert number[0].result is CheckResult.FAIL

    narrowed = principal_equality(
        _policy(tools=["release_payment"]),
        _context("alice", subject_owner="bob"),
    )
    assert narrowed == []


def test_user_directory_shadows_a_builtin_pack_and_update_does_not_write_it(tmp_path):
    user = tmp_path / "user"
    managed = tmp_path / "managed"
    dest = user / "geo-fencing"
    shutil.copytree(ROOT / "geo-fencing", dest)
    pack_file = dest / "pack.toml"
    pack_file.write_text(
        pack_file.read_text(encoding="utf-8").replace('version = "1.0.0"', 'version = "9.9.9"'),
        encoding="utf-8",
    )
    builtin_before = (ROOT / "geo-fencing" / "pack.toml").read_bytes()
    pack = resolve_pack("geo-fencing", user_root=user, managed_root=managed)
    assert pack.origin == "overlay"
    assert pack.version == "9.9.9"
    assert (ROOT / "geo-fencing" / "pack.toml").read_bytes() == builtin_before

    files = read_pack_files(ROOT / "geo-fencing")
    database = tmp_path / "store.db"
    result = update_pack(
        "geo-fencing",
        db=str(database),
        managed_root=managed,
        user_root=user,
        transport=_files_transport("geo-fencing", files),
    )
    assert result.refused is False
    assert pack_file.read_text(encoding="utf-8").count('version = "9.9.9"') == 1
    assert (managed / "geo-fencing" / "pack.toml").is_file()
    assert "9.9.9" not in (managed / "geo-fencing" / "pack.toml").read_text(encoding="utf-8")


def test_policy_overlay_replaces_the_same_rule_id(tmp_path):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    source = (ROOT / "geo-fencing" / "policy" / "permitted-locations.yaml").read_text(
        encoding="utf-8"
    )
    (overlay / "permitted-locations.yaml").write_text(
        source.replace("An actor may act", "A user overlay may act"),
        encoding="utf-8",
    )
    builtin_before = (ROOT / "geo-fencing" / "policy" / "permitted-locations.yaml").read_bytes()
    pack = resolve_pack(
        "geo-fencing",
        overlay=overlay,
        user_root=tmp_path / "user",
        managed_root=tmp_path / "managed",
    )
    rule = next(item for item in pack.policies.rules if item.id == "permitted-locations")
    assert rule.citation.startswith("A user overlay may act")
    assert (ROOT / "geo-fencing" / "policy" / "permitted-locations.yaml").read_bytes() == builtin_before
    assert len(pack.policies.rules) == 2


def test_update_refuses_a_bad_hash_a_bad_signature_and_a_missing_network(tmp_path):
    managed = tmp_path / "managed"
    user = tmp_path / "user"
    database = tmp_path / "store.db"
    kept = content_hash(read_pack_files(ROOT / "geo-fencing"))
    marker = managed / "geo-fencing"
    marker.mkdir(parents=True)
    (marker / "marker").write_text("stay", encoding="utf-8")

    files = dict(read_pack_files(ROOT / "geo-fencing"))
    files["pack.toml"] = files["pack.toml"].replace(b'version = "1.0.0"', b'version = "9.9.9"')
    mismatch = update_pack(
        "geo-fencing",
        db=str(database),
        managed_root=managed,
        user_root=user,
        transport=_files_transport("geo-fencing", files),
    )
    assert mismatch.refused is True
    assert mismatch.reason == "integrity"
    assert mismatch.source_hash == kept
    assert (marker / "marker").read_text(encoding="utf-8") == "stay"
    assert not (marker / "pack.toml").exists()

    def offline(url: str) -> bytes:
        raise PackNetworkError("pack source could not be reached")

    offline_result = update_pack(
        "geo-fencing",
        db=str(database),
        managed_root=managed,
        user_root=user,
        transport=offline,
    )
    assert offline_result.refused is True
    assert offline_result.reason == "network"
    assert (marker / "marker").read_text(encoding="utf-8") == "stay"

    called: list[str] = []

    def unexpected(url: str) -> bytes:
        called.append(url)
        raise AssertionError(url)

    pins = tmp_path / "pins.toml"
    pins.write_text('repository = "https://github.com/mze3e/kognita"\n\n[packs]\n', encoding="utf-8")
    unpinned = update_pack(
        "geo-fencing",
        db=str(database),
        pins=pins,
        managed_root=managed,
        user_root=user,
        transport=unexpected,
    )
    assert unpinned.reason == "unpinned"
    assert called == []

    foreign = update_pack(
        "geo-fencing",
        db=str(database),
        source="https://github.com/other/repo",
        pins=pins,
        managed_root=managed,
        user_root=user,
        transport=unexpected,
    )
    assert foreign.reason == "source"
    assert called == []

    events = _events(database)
    assert [payload["action"] for _, payload in events] == ["PACK_UPDATE_REFUSED"] * 4
    assert events[0][1]["source_hash"] == kept
    assert all(payload["pack"] == "geo-fencing" for _, payload in events)


def test_update_installs_when_the_pin_or_the_signature_matches(tmp_path, monkeypatch):
    files = read_pack_files(ROOT / "geo-fencing")
    digest = content_hash(files)
    managed = tmp_path / "managed"
    user = tmp_path / "user"
    database = tmp_path / "store.db"
    installed = update_pack(
        "geo-fencing",
        db=str(database),
        managed_root=managed,
        user_root=user,
        transport=_files_transport("geo-fencing", files),
    )
    assert installed == UpdateResult(
        pack="geo-fencing",
        refused=False,
        reason="",
        source_hash=digest,
        version="1.0.0",
    )
    assert content_hash(read_pack_files(managed / "geo-fencing")) == digest
    kind, payload = _events(database)[0]
    assert kind is EventType.POLICY_CHANGE
    assert payload["action"] == "PACK_UPDATE"
    assert payload["source_hash"] == digest

    body = manifest_bytes(files)
    mac = hmac.new(b"pack-key", body, hashlib.sha256).hexdigest()
    signed_db = tmp_path / "signed.db"
    pins = tmp_path / "empty.toml"
    pins.write_text("[packs]\n", encoding="utf-8")
    signed = update_pack(
        "geo-fencing",
        db=str(signed_db),
        pins=pins,
        signature=f"hmac:{mac}",
        hmac_key=b"pack-key",
        managed_root=tmp_path / "signed-managed",
        user_root=user,
        transport=_files_transport("geo-fencing", files),
    )
    assert signed.refused is False
    assert signed.source_hash == digest

    bad = update_pack(
        "geo-fencing",
        db=str(tmp_path / "bad.db"),
        pins=pins,
        signature="hmac:" + "ab" * 32,
        hmac_key=b"pack-key",
        managed_root=tmp_path / "bad-managed",
        user_root=user,
        transport=_files_transport("geo-fencing", files),
    )
    assert bad.reason == "signature"
    assert not (tmp_path / "bad-managed" / "geo-fencing").exists()

    wrong_pin = tmp_path / "wrong.toml"
    wrong_pin.write_text(
        '[packs."geo-fencing"]\nversion = "1.0.0"\nsha256 = "' + "cd" * 32 + '"\n',
        encoding="utf-8",
    )
    replaced = update_pack(
        "geo-fencing",
        db=str(tmp_path / "expect.db"),
        pins=wrong_pin,
        expect=f"sha256:{digest}",
        managed_root=tmp_path / "expect-managed",
        user_root=user,
        transport=_files_transport("geo-fencing", files),
    )
    assert replaced.refused is False

    def explode(*args, **kwargs):
        raise RuntimeError("evidence store unavailable")

    monkeypatch.setattr("kognita.packs.update.EvidenceWriter.emit", explode)
    previous = tmp_path / "restore" / "geo-fencing"
    previous.mkdir(parents=True)
    (previous / "marker").write_text("previous", encoding="utf-8")
    (previous / "pack.toml").write_text(
        'name = "geo-fencing"\nversion = "0.0.1"\npurposes = ["ACCESS"]\nfiles = ["marker"]\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="evidence store unavailable"):
        update_pack(
            "geo-fencing",
            db=str(tmp_path / "restore.db"),
            managed_root=tmp_path / "restore",
            user_root=user,
            transport=_files_transport("geo-fencing", files),
        )
    assert (previous / "marker").read_text(encoding="utf-8") == "previous"
    assert b'version = "0.0.1"' in (previous / "pack.toml").read_bytes()


def test_time_gate_uses_the_injected_clock_and_not_context():
    moment = datetime(2026, 6, 2, 10, 30, tzinfo=timezone.utc)
    pack = resolve_pack(
        "time-gated-access",
        user_root=Path("/no/user"),
        managed_root=Path("/no/managed"),
        clock=lambda: moment,
    )
    envelope = Envelope(
        principal="alice",
        purpose="ACCESS",
        tool="read_record",
        context={"local_hour": "21", "local_weekday": "Sun"},
    )
    attributes = pack.resolve_attributes(envelope, {})
    assert attributes["local_hour"] == "10"
    assert attributes["local_weekday"] == "Tue"
    assert envelope.context["local_hour"] == "21"


def test_rule_overlay_replaces_a_core_evaluator():
    def replacement(policy, context):
        return []

    pack = resolve_pack(
        "geo-fencing",
        user_root=Path("/no/user"),
        managed_root=Path("/no/managed"),
    ).overlay_rules({"ATTRIBUTE_ALLOWLIST": replacement})
    assert pack.rules()["ATTRIBUTE_ALLOWLIST"] is replacement


def test_private_banking_fixture_denies_cross_border_and_booking_centre(session):
    pack = resolve_pack(
        "private-banking",
        user_root=Path("/no/user"),
        managed_root=Path("/no/managed"),
    )
    compile_rules(session, pack.policies, at=AS_OF)
    session.flush()

    def outcome(client: str, product: str = "note-sg"):
        envelope = Envelope(
            principal="rm.lee",
            purpose="CLIENT_DISCUSSION",
            tool="discuss_product",
            actor_location="SG",
            subjects={"client": client, "product": product},
        )
        subjects = pack.load_subjects(envelope)
        attributes = pack.resolve_attributes(envelope, subjects)
        evaluation = decide(
            envelope,
            load_snapshot(session, as_of=AS_OF),
            attributes=attributes,
            subjects=subjects,
            rules=pack.rules(),
            purposes=pack.purposes,
            engages=pack.engages,
            as_of=AS_OF,
        )
        return evaluation

    allowed = outcome("C-100")
    assert allowed.outcome is Outcome.ALLOW
    denied_border = outcome("C-CROSS")
    assert denied_border.outcome is Outcome.DENY
    assert "cross-border-solicitation" in _cited(denied_border, session)
    denied_centre = outcome("C-BOOK")
    assert denied_centre.outcome is Outcome.DENY
    assert "booking-centre" in _cited(denied_centre, session)

    missing = Envelope(
        principal="rm.lee",
        purpose="CLIENT_DISCUSSION",
        tool="discuss_product",
        actor_location="SG",
        subjects={"client": "C-100"},
    )
    bare = pack.resolve_attributes(missing, {"client": {"booking_centre": "SG"}})
    assert "contact_count" not in bare
    with pytest.raises(LookupError):
        pack.load_subjects(
            Envelope(
                principal="rm.lee",
                purpose="CLIENT_DISCUSSION",
                tool="discuss_product",
                subjects={"client": "C-MISSING"},
            )
        )


def test_a_pack_cannot_skip_a_breaker_or_a_revocation(session, evidence: EvidenceWriter):
    """A tripped breaker and an open revocation decide even when the pack declines."""
    pack = resolve_pack(
        "private-banking",
        user_root=Path("/no/user"),
        managed_root=Path("/no/managed"),
    )
    compile_rules(session, pack.policies, at=AS_OF)
    session.add(
        Policy(
            regime="CIRCUIT_BREAKER",
            rule_type="CIRCUIT_BREAKER",
            rule={
                "dimension": "usecase",
                "value": "CLIENT_DISCUSSION",
                "on_violation": "fail",
                "accountable_owner": "Head of Research Ops",
            },
            citation=(
                "Circuit breaker — usecase CLIENT_DISCUSSION. "
                "Source: Kognita private-banking pack."
            ),
            effective_from=AS_OF,
        )
    )
    register(session, name="rm-assistant", owner_exec="Head of Research Ops")
    revoke(
        session,
        "tool",
        "discuss_product",
        evidence=evidence,
        actor_id="Head of Research Ops",
        citation=(
            "Revocation: discuss_product stays stopped. "
            "Source: Kognita private-banking pack."
        ),
        owner="Head of Research Ops",
        now=AS_OF,
    )
    session.flush()
    envelope = Envelope(
        principal="rm.lee",
        purpose="CLIENT_DISCUSSION",
        tool="discuss_product",
        actor_location="SG",
        subjects={"client": "C-100", "product": "note-sg"},
    )
    subjects = pack.load_subjects(envelope)
    attributes = pack.resolve_attributes(envelope, subjects)
    snapshot = load_snapshot(session, as_of=AS_OF)

    def evaluate(engages):
        return decide(
            envelope,
            snapshot,
            attributes=attributes,
            subjects=subjects,
            rules=pack.rules(),
            purposes=pack.purposes,
            engages=engages,
            as_of=AS_OF,
        )

    denied = evaluate(pack.engages)
    assert denied.outcome is Outcome.DENY
    basis = {check.check for check in denied.basis()}
    assert "CIRCUIT_BREAKER" in basis
    assert "REVOCATION" in basis
    assert any("discuss_product is revoked" in check.citation for check in denied.basis())

    skipped = evaluate(lambda _policy, _context: False)
    assert skipped.outcome is Outcome.DENY
    names = {check.check for check in skipped.checks}
    assert "CIRCUIT_BREAKER" in names
    assert "REVOCATION" in names
    assert not any(check.check.startswith("ATTRIBUTE_ALLOWLIST") for check in skipped.checks)


def test_gateway_cites_the_geo_fence_and_does_not_call_upstream_on_denial(session, evidence):
    pack = resolve_pack(
        "geo-fencing",
        user_root=Path("/no/user"),
        managed_root=Path("/no/managed"),
    )
    compile_rules(session, pack.policies, at=AS_OF)
    _register(session)
    _live(session, purpose="ACCESS")
    session.flush()
    upstream = _Upstream(
        json.dumps(
            {"model": "gateway-model", "choices": [{"message": {"content": "ok"}}]}
        ).encode()
    )

    def call(location: str):
        gateway = _gateway(
            session.get_bind(),
            evidence,
            upstream,
            client=ClientConfiguration(
                principal="alice",
                purpose="ACCESS",
                agent_names=frozenset({"dossier-agent"}),
                actor_location=location,
            ),
            purposes=("ACCESS",),
            pack=pack,
        )
        return gateway.handle(
            "POST",
            PATH,
            {"agent_name": "dossier-agent"},
            _body(BENIGN),
            session=session,
        )

    allowed = call("SG")
    assert allowed.status == 200
    assert allowed.evaluation is not None
    assert allowed.evaluation.outcome is Outcome.ALLOW
    assert "permitted-locations" in _cited(allowed.evaluation, session)
    assert len(upstream.calls) == 1

    upstream.calls.clear()
    denied = call("KP")
    assert denied.status == 403
    assert denied.evaluation is not None
    assert denied.evaluation.outcome is Outcome.DENY
    assert "blocked-locations" in _cited(denied.evaluation, session)
    assert upstream.calls == []


def test_cli_lists_shows_loads_diffs_and_serves_a_pack(tmp_path, capsys, monkeypatch):
    roots = _roots(tmp_path)
    assert main(["packs", "list", *roots]) == 0
    listed = set(capsys.readouterr().out.splitlines())
    assert set(PACK_NAMES) <= listed

    assert main(["packs", "show", "geo-fencing", *roots]) == 0
    shown = capsys.readouterr().out
    assert "geo-fencing  1.0.0  builtin" in shown
    assert "permitted-locations  ATTRIBUTE_ALLOWLIST" in shown
    assert main(["packs", "show", "not-a-pack", *roots]) == 1

    database = tmp_path / "store.db"
    assert main(["packs", "load", "geo-fencing", "--db", str(database), *roots]) == 0
    events = _events(database)
    assert len(events) == 1
    assert events[0][1]["action"] == "LOAD"
    assert events[0][1]["pack"] == "geo-fencing"
    pack = resolve_pack("geo-fencing", user_root=tmp_path / "user", managed_root=tmp_path / "managed")
    assert events[0][1]["source_hash"] == load_source_hash(pack)
    assert main(["packs", "load", "geo-fencing", "--db", str(database), *roots]) == 0
    assert len(_events(database)) == 1
    capsys.readouterr()

    other = tmp_path / "other"
    shutil.copytree(ROOT / "geo-fencing" / "policy", other)
    target = other / "permitted-locations.yaml"
    target.write_text(
        target.read_text(encoding="utf-8").replace("An actor may act", "A later pack may act"),
        encoding="utf-8",
    )
    assert main(["packs", "diff", "geo-fencing", str(other), *roots]) == 0
    diff = capsys.readouterr().out
    assert diff.startswith("geo-fencing 1.0.0\n")
    assert "## MODIFIED" in diff
    assert "permitted-locations" in diff

    captured: dict = {}

    def serve(self, host, port):
        captured["gateway"] = self
        captured["host"] = host

    monkeypatch.setattr("kognita.cli.Gateway.serve", serve)
    assert (
        main(
            [
                "serve",
                "--provider",
                "openai-compatible",
                "--upstream",
                "https://api.openai.com",
                "--pack",
                "geo-fencing",
                "--db",
                str(tmp_path / "serve.db"),
                *roots,
            ]
        )
        == 0
    )
    assert captured["gateway"].pack.name == "geo-fencing"
    assert captured["gateway"].purposes == ("ACCESS",)
    assert (
        main(
            [
                "serve",
                "--mcp",
                "--pack",
                "geo-fencing",
                "--db",
                str(tmp_path / "mcp.db"),
            ]
        )
        == 2
    )

    seen: dict = {}

    def fake_update(name, **kwargs):
        seen["name"] = name
        seen.update(kwargs)
        return UpdateResult(
            pack=name, refused=False, reason="", source_hash="abc", version="1.0.0"
        )

    monkeypatch.setattr("kognita.cli.update_pack", fake_update)
    assert (
        main(
            [
                "packs",
                "update",
                "geo-fencing",
                "--source",
                "https://github.com/mze3e/kognita",
                "--expect",
                "sha256:" + "ab" * 32,
                "--db",
                str(database),
                *roots,
            ]
        )
        == 0
    )
    assert seen["name"] == "geo-fencing"
    assert seen["expect"] == "sha256:" + "ab" * 32
    assert seen["source"] == "https://github.com/mze3e/kognita"


def test_merge_policy_sets_keeps_a_base_rule_the_overlay_does_not_replace():
    base = resolve_pack(
        "geo-fencing",
        user_root=Path("/no/user"),
        managed_root=Path("/no/managed"),
    ).policies
    overlay_dir = ROOT / "role-based-access" / "policy"
    from kognita.policy import read_policy_set

    merged = merge_policy_sets(base, read_policy_set(overlay_dir))
    assert {rule.id for rule in merged.rules} >= {
        "permitted-locations",
        "blocked-locations",
        "named-role",
        "own-subject",
    }


def test_the_engine_does_not_import_packs_and_import_kognita_does_not_load_them():
    for stem in _engine_modules_on_disk():
        tree = ast.parse((PACKAGE_ROOT / f"{stem}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            assert not any(
                name == "kognita.packs" or name.startswith("kognita.packs.") for name in modules
            ), stem

    probe = (
        "import sys; import kognita; "
        "print('yes' if 'kognita.packs' in sys.modules else 'no')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "no"


def test_a_pack_load_hash_changes_when_an_overlay_is_compiled(tmp_path):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    source = (ROOT / "geo-fencing" / "policy" / "blocked-locations.yaml").read_text(encoding="utf-8")
    (overlay / "blocked-locations.yaml").write_text(source, encoding="utf-8")
    pack = resolve_pack(
        "geo-fencing",
        overlay=overlay,
        user_root=tmp_path / "user",
        managed_root=tmp_path / "managed",
    )
    assert load_source_hash(pack, overlay) != load_source_hash(pack)
