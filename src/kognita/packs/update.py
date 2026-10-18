"""Fetch a pack from the kognita repository and verify it before installing.

The trust anchor is a pinned SHA-256, or an HMAC the operator holds the key
for. The pin is local. A hash taken from the response is not a pin. A failed
check or a failed fetch leaves the installed pack where it is and appends a
``POLICY_CHANGE`` refusal. Deciding does not call this module.
"""

from __future__ import annotations

import hashlib
import hmac
import shutil
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse

from kognita.db import create_all, make_engine, session_scope
from kognita.evidence import EvidenceWriter
from kognita.packs.catalog import (
    DEFAULT_PACK_REF,
    DEFAULT_PACK_ROOT,
    DEFAULT_PACK_SOURCE,
    PackError,
    builtin_data,
    content_hash,
    default_managed_root,
    default_user_root,
    manifest_bytes,
    read_pack_files,
)
from kognita.vocabulary import ActorType, EventType

_HEX = set("0123456789abcdef")


class PackUpdateRefused(PackError):
    """An update was not installed."""

    def __init__(self, reason: str, *, pack: str = "", source_hash: str = "") -> None:
        self.reason = reason
        self.pack = pack
        self.source_hash = source_hash
        super().__init__(reason)


class PackNetworkError(PackError):
    """The pack source could not be reached."""


@dataclass(frozen=True)
class Pin:
    """A local integrity pin. Not fetched with the pack."""

    sha256: str = ""
    version: str = ""


@dataclass(frozen=True)
class UpdateResult:
    """What an update did. ``refused`` means the installed pack was kept."""

    pack: str
    refused: bool
    reason: str
    source_hash: str
    version: str = ""


def parse_source(source: str) -> tuple[str, str]:
    """The approved repository. Any other host or repository is refused."""
    parsed = urlparse(source.strip())
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in {
        "github.com",
        "www.github.com",
    }:
        raise PackError("pack source must be https://github.com/mze3e/kognita")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise PackError("pack source must be https://github.com/mze3e/kognita")
    owner = parts[0]
    repo = parts[1].removesuffix(".git")
    if (owner, repo) != ("mze3e", "kognita"):
        raise PackError("pack source must be https://github.com/mze3e/kognita")
    return owner, repo


def _safe_ref(ref: str) -> bool:
    if not ref or ".." in ref.split("/"):
        return False
    return all(
        part and all(char.isalnum() or char in "._-" for char in part)
        for part in ref.split("/")
    )


def _safe_rel(rel: str) -> bool:
    if not rel or rel.startswith(("/", "\\")) or "\\" in rel:
        return False
    parts = Path(rel).parts
    return bool(parts) and ".." not in parts


def raw_url(source: str, ref: str, root: str, name: str, rel: str) -> str:
    """An HTTPS URL for one file in the approved repository."""
    owner, repo = parse_source(source)
    if not _safe_ref(ref) or not _safe_rel(rel) and rel != "pack.toml":
        raise PackError("pack ref or path is not a relative file")
    if rel != "pack.toml" and not _safe_rel(rel):
        raise PackError("pack path is not a relative file")
    segments = [owner, repo, *ref.split("/"), *[part for part in root.split("/") if part], name]
    segments.extend(rel.split("/"))
    path = "/".join(quote(segment, safe="") for segment in segments)
    return f"https://raw.githubusercontent.com/{path}"


def urllib_transport(url: str, *, timeout: float = 10.0) -> bytes:
    """Fetch one HTTPS URL from the approved repository. Anything else is refused."""
    if not url.startswith("https://raw.githubusercontent.com/mze3e/kognita/"):
        raise PackError("pack fetch URL is not the approved repository")
    request = urllib.request.Request(url, headers={"User-Agent": "kognita-packs"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final = response.geturl()
            if not str(final).startswith("https://raw.githubusercontent.com/mze3e/kognita/"):
                raise PackError("pack fetch was redirected off the approved repository")
            return response.read()
    except PackError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PackNetworkError("pack source could not be reached") from exc


def fetch_pack_files(
    name: str,
    *,
    source: str,
    ref: str,
    root: str,
    transport: Callable[[str], bytes],
) -> dict[str, bytes]:
    """Download ``pack.toml`` and every file it lists. The list is part of the hash."""
    try:
        blob = transport(raw_url(source, ref, root, name, "pack.toml"))
    except PackNetworkError:
        raise
    try:
        meta = tomllib.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PackError("remote pack.toml could not be read") from exc
    if meta.get("name") != name:
        raise PackError("remote pack name does not match")
    listed = meta.get("files")
    if not isinstance(listed, list) or not listed or any(not isinstance(item, str) for item in listed):
        raise PackError("remote pack has no file list")
    files = {"pack.toml": blob}
    for rel in listed:
        if not _safe_rel(rel):
            raise PackError("remote pack path is not a relative file")
        files[rel] = transport(raw_url(source, ref, root, name, rel))
    return files


def load_pin(path: Path, name: str) -> Pin | None:
    """The pin for ``name`` in a local pin file. A missing file is no pin."""
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PackError(f"{path} could not be read") from exc
    entry = (data.get("packs") or {}).get(name)
    if not isinstance(entry, dict):
        return None
    digest = entry.get("sha256") or ""
    version = entry.get("version") or ""
    if not isinstance(digest, str) or not isinstance(version, str):
        raise PackError(f"{path} pin for {name} is not a string")
    return Pin(sha256=digest, version=version)


def _hex(value: str) -> bool:
    return bool(value) and len(value) == 64 and all(char in _HEX for char in value)


def verify_integrity(
    files: dict[str, bytes],
    *,
    expect: str | None,
    pin: Pin | None,
    signature: str | None,
    hmac_key: bytes | None,
) -> str:
    """Check a pinned hash, a signature, or both. Neither is a refusal.

    ``expect`` is an operator pin (``sha256:<hex>``) and replaces the pin file
    for this call. A signature is ``hmac:<hex>`` over the manifest bytes.
    """
    digest = content_hash(files)
    body = manifest_bytes(files)
    hashed = False
    signed = False
    if expect:
        token = expect.removeprefix("sha256:")
        if not expect.startswith("sha256:") or not _hex(token) or not hmac.compare_digest(digest, token):
            raise PackUpdateRefused("integrity")
        hashed = True
    elif pin is not None and pin.sha256:
        if not _hex(pin.sha256) or not hmac.compare_digest(digest, pin.sha256):
            raise PackUpdateRefused("integrity")
        hashed = True
    if signature is not None:
        if hmac_key is None:
            raise PackUpdateRefused("signature")
        token = signature.removeprefix("hmac:")
        mac = hmac.new(hmac_key, body, hashlib.sha256).hexdigest()
        if not signature.startswith("hmac:") or not _hex(token) or not hmac.compare_digest(mac, token):
            raise PackUpdateRefused("signature")
        signed = True
    if not hashed and not signed:
        raise PackUpdateRefused("unpinned")
    return digest


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def install_files(managed: Path, name: str, files: dict[str, bytes], user_root: Path) -> Callable[[], None]:
    """Install ``files`` and return a function that restores the previous tree."""
    dest = managed / name
    if user_root.exists() and (_within(dest, user_root) or _within(managed, user_root)):
        raise PackError("refusing to write the user overlay")
    for rel in files:
        if rel != "pack.toml" and not _safe_rel(rel):
            raise PackError("pack path is not a relative file")
    managed.mkdir(parents=True, exist_ok=True)
    tmp = managed / f".{name}.incoming"
    backup = managed / f".{name}.previous"
    if tmp.exists():
        shutil.rmtree(tmp)
    if backup.exists():
        shutil.rmtree(backup)
    for rel, data in files.items():
        path = tmp / rel
        if not path.resolve().is_relative_to(tmp.resolve()):
            raise PackError("pack path is not a relative file")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    had_previous = dest.exists()
    if had_previous:
        dest.rename(backup)
    try:
        tmp.rename(dest)
    except OSError:
        if had_previous and not dest.exists() and backup.exists():
            backup.rename(dest)
        raise

    def restore() -> None:
        if dest.exists():
            shutil.rmtree(dest)
        if had_previous and backup.exists():
            backup.rename(dest)
        elif backup.exists():
            shutil.rmtree(backup)

    return restore


def _discard_backup(managed: Path, name: str) -> None:
    backup = managed / f".{name}.previous"
    if backup.exists():
        shutil.rmtree(backup)


def _in_force_root(
    name: str, user_root: Path, managed: Path, builtin: Path
) -> Path | None:
    for root in (user_root / name, managed / name, builtin / name):
        if (root / "pack.toml").is_file():
            return root
    return None


def _in_force_hash(name: str, user_root: Path, managed: Path, builtin: Path) -> str:
    root = _in_force_root(name, user_root, managed, builtin)
    if root is None:
        return ""
    try:
        return content_hash(read_pack_files(root))
    except PackError:
        return ""


def _record(db: str, payload: dict) -> None:
    engine = make_engine(db)
    create_all(engine)
    with session_scope(engine) as session:
        EvidenceWriter(engine).emit(
            session,
            correlation_id=f"pack-update:{payload.get('pack', '')}",
            event_type=EventType.POLICY_CHANGE,
            actor_type=ActorType.SYSTEM,
            actor_id="kognita packs update",
            payload=payload,
        )


def _refuse(
    db: str,
    *,
    pack: str,
    reason: str,
    source: str,
    ref: str,
    kept: str,
    refused_hash: str = "",
) -> UpdateResult:
    payload: dict[str, str] = {
        "action": "PACK_UPDATE_REFUSED",
        "pack": pack,
        "reason": reason,
        "source": source,
        "ref": ref,
        "source_hash": kept,
    }
    if refused_hash:
        payload["refused_hash"] = refused_hash
    _record(db, payload)
    return UpdateResult(pack=pack, refused=True, reason=reason, source_hash=kept)


def update_pack(
    name: str,
    *,
    db: str,
    source: str = DEFAULT_PACK_SOURCE,
    ref: str = DEFAULT_PACK_REF,
    root: str = DEFAULT_PACK_ROOT,
    expect: str | None = None,
    signature: str | None = None,
    hmac_key: bytes | None = None,
    pins: Path | None = None,
    managed_root: Path | None = None,
    user_root: Path | None = None,
    builtin_root: Path | None = None,
    transport: Callable[[str], bytes] | None = None,
) -> UpdateResult:
    """Fetch ``name`` and install it only when the pin or the signature matches.

    A network failure, a bad pin, a bad signature, or a missing pin keeps the
    installed pack and records the refusal. The user overlay is not written.
    """
    managed = managed_root if managed_root is not None else default_managed_root()
    user = user_root if user_root is not None else default_user_root()
    builtin = builtin_root if builtin_root is not None else builtin_data()
    kept = _in_force_hash(name, user, managed, builtin)
    fetcher = transport or urllib_transport
    try:
        parse_source(source)
    except PackError:
        return _refuse(db, pack=name, reason="source", source=source, ref=ref, kept=kept)
    if signature is not None and hmac_key is None:
        return _refuse(db, pack=name, reason="signature", source=source, ref=ref, kept=kept)
    pin_path = pins if pins is not None else builtin / "pins.toml"
    try:
        pin = load_pin(pin_path, name)
    except PackError:
        return _refuse(db, pack=name, reason="unpinned", source=source, ref=ref, kept=kept)
    if not expect and not (pin and pin.sha256) and signature is None:
        return _refuse(db, pack=name, reason="unpinned", source=source, ref=ref, kept=kept)
    if user.exists() and (_within(managed / name, user) or _within(managed, user)):
        return _refuse(db, pack=name, reason="overlay", source=source, ref=ref, kept=kept)
    try:
        files = fetch_pack_files(name, source=source, ref=ref, root=root, transport=fetcher)
    except PackNetworkError:
        return _refuse(db, pack=name, reason="network", source=source, ref=ref, kept=kept)
    except PackError:
        return _refuse(db, pack=name, reason="integrity", source=source, ref=ref, kept=kept)
    try:
        digest = verify_integrity(
            files, expect=expect, pin=pin, signature=signature, hmac_key=hmac_key
        )
    except PackUpdateRefused as exc:
        refused_hash = ""
        try:
            refused_hash = content_hash(files)
        except PackError:
            refused_hash = ""
        return _refuse(
            db,
            pack=name,
            reason=exc.reason,
            source=source,
            ref=ref,
            kept=kept,
            refused_hash=refused_hash,
        )
    try:
        restore = install_files(managed, name, files, user)
    except PackError:
        return _refuse(
            db,
            pack=name,
            reason="overlay",
            source=source,
            ref=ref,
            kept=kept,
            refused_hash=digest,
        )
    try:
        version = str(tomllib.loads(files["pack.toml"].decode("utf-8")).get("version") or "")
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        restore()
        return _refuse(db, pack=name, reason="integrity", source=source, ref=ref, kept=kept)
    try:
        _record(
            db,
            {
                "action": "PACK_UPDATE",
                "pack": name,
                "version": version,
                "source": source,
                "ref": ref,
                "source_hash": digest,
            },
        )
    except Exception:
        restore()
        raise
    _discard_backup(managed, name)
    return UpdateResult(
        pack=name, refused=False, reason="", source_hash=digest, version=version
    )


def pinned_names(pins: Path) -> list[str]:
    """Pack names in a pin file, sorted. An unreadable file is an error."""
    try:
        data = tomllib.loads(pins.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, OSError) as exc:
        raise PackError(f"{pins} could not be read") from exc
    packs = data.get("packs") or {}
    if not isinstance(packs, dict):
        raise PackError(f"{pins} has no packs table")
    return sorted(str(name) for name in packs)
