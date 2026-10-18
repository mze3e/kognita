"""Pack lookup. One directory per pack. No central registry.

A user directory shadows a built-in pack of the same name. A policy overlay
merges extra rules on top of the resolved pack; the same rule id is the user's.
``packs update`` never writes either of those directories.

A pack's ``purposes`` list is the header list ``kognita serve --pack`` uses.
It does not authorize an execution path. Register each purpose as a use case
before serving, or the register denies the call.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kognita.canonical import canonical_hash
from kognita.envelope import Envelope
from kognita.exceptions import KognitaError
from kognita.policy import PolicySet, read_policy_set
from kognita.rules import Evaluator, build_registry

#: The five templates, named as the 0.4 roadmap names them.
STARTER_PACKS: tuple[str, ...] = (
    "role-based-access",
    "geo-fencing",
    "data-classification",
    "time-gated-access",
    "two-signature-approval-chains",
)
PRIVATE_BANKING_PACK = "private-banking"
PACK_NAMES: tuple[str, ...] = STARTER_PACKS + (PRIVATE_BANKING_PACK,)

DEFAULT_PACK_SOURCE = "https://github.com/mze3e/kognita"
DEFAULT_PACK_REF = "main"
DEFAULT_PACK_ROOT = "src/kognita/packs/data"

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class PackError(KognitaError):
    """A pack could not be resolved or read."""


def builtin_data() -> Path:
    """The packs shipped inside this package."""
    return Path(__file__).resolve().parent / "data"


def default_user_root() -> Path:
    """Whole packs that shadow a built-in name. Updates never write here."""
    override = os.environ.get("KOGNITA_PACKS_OVERLAY")
    if override:
        return Path(override)
    return Path.home() / ".kognita" / "packs"


def default_managed_root() -> Path:
    """Packs written by ``packs update``. Looked up after the user overlay."""
    override = os.environ.get("KOGNITA_PACKS_MANAGED")
    if override:
        return Path(override)
    return Path.home() / ".kognita" / "packs-managed"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def content_hash(files: Mapping[str, bytes]) -> str:
    """SHA-256 of the sorted name and SHA-256 of each pack file."""
    manifest = [
        {"path": path, "sha256": hashlib.sha256(files[path]).hexdigest()}
        for path in sorted(files)
    ]
    return canonical_hash(manifest)


def manifest_bytes(files: Mapping[str, bytes]) -> bytes:
    """The bytes a signature covers. The content hash is their SHA-256."""
    from kognita.canonical import canonical_json

    manifest = [
        {"path": path, "sha256": hashlib.sha256(files[path]).hexdigest()}
        for path in sorted(files)
    ]
    return canonical_json(manifest).encode("utf-8")


def _safe_rel(rel: str) -> bool:
    if not rel or rel.startswith(("/", "\\")) or "\\" in rel:
        return False
    parts = Path(rel).parts
    return bool(parts) and ".." not in parts and not any(part.startswith("/") for part in parts)


def read_pack_files(root: Path) -> dict[str, bytes]:
    """The bytes ``content_hash`` covers: ``pack.toml`` and every listed file."""
    meta_path = root / "pack.toml"
    if not meta_path.is_file():
        raise PackError(f"{root} has no pack.toml")
    meta = _meta(meta_path.read_bytes(), root)
    files = {"pack.toml": meta_path.read_bytes()}
    listed = meta["files"]
    disk = _policy_files(root)
    if disk != set(listed):
        raise PackError(f"{root.name} file list does not match the pack directory")
    for rel in listed:
        if not _safe_rel(rel):
            raise PackError(f"{root.name} path {rel!r} escapes the pack directory")
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise PackError(f"{root.name} path {rel!r} escapes the pack directory")
        files[rel] = path.read_bytes()
    return files


def _policy_files(root: Path) -> set[str]:
    policy = root / "policy"
    if not policy.is_dir():
        return set()
    found: set[str] = set()
    for path in policy.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            found.add(path.relative_to(root).as_posix())
    return found


def _meta(blob: bytes, root: Path) -> dict[str, Any]:
    try:
        data = tomllib.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PackError(f"{root} pack.toml could not be read") from exc
    if not isinstance(data, dict):
        raise PackError(f"{root} pack.toml could not be read")
    name = data.get("name")
    version = data.get("version")
    purposes = data.get("purposes")
    listed = data.get("files")
    if not isinstance(name, str) or not name.strip():
        raise PackError(f"{root} pack.toml needs a name")
    if not isinstance(version, str) or not version.strip():
        raise PackError(f"{root} pack.toml needs a version")
    if (
        not isinstance(purposes, list)
        or not purposes
        or any(not isinstance(item, str) or not item.strip() for item in purposes)
    ):
        raise PackError(f"{root} pack.toml needs a purpose list")
    if (
        not isinstance(listed, list)
        or not listed
        or any(not isinstance(item, str) for item in listed)
    ):
        raise PackError(f"{root} pack.toml needs a file list")
    return data


def merge_policy_sets(base: PolicySet, overlay: PolicySet) -> PolicySet:
    """Overlay rules replace the same id. Rules the base does not have are added."""
    by_id = dict(base.by_id())
    order = [item.id for item in base.rules]
    for item in overlay.rules:
        if item.id not in by_id:
            order.append(item.id)
        by_id[item.id] = item
    scenarios = {document.rule: document for document in base.scenarios}
    for document in overlay.scenarios:
        scenarios[document.rule] = document
    seen: list[str] = []
    documents = []
    for document in (*base.scenarios, *overlay.scenarios):
        if document.rule in seen:
            continue
        seen.append(document.rule)
        current = scenarios.get(document.rule)
        if current is not None:
            documents.append(current)
    return PolicySet(
        rules=tuple(by_id[rule_id] for rule_id in order),
        scenarios=tuple(documents),
        source=overlay.source or base.source,
    )


@dataclass(frozen=True)
class Pack:
    """One resolved pack: its policy set, its version, and its fixture."""

    name: str
    version: str
    origin: str
    root: Path
    policies: PolicySet
    purposes: tuple[str, ...]
    clients: dict[str, dict[str, Any]] = field(default_factory=dict)
    products: dict[str, dict[str, Any]] = field(default_factory=dict)
    stamp_time: bool = False
    rule_overlays: tuple[Mapping[str, Evaluator], ...] = ()
    clock: Callable[[], datetime] = _utc_now

    @property
    def policy_dir(self) -> Path:
        return self.root / "policy"

    def rules(self) -> dict[str, Evaluator]:
        """Core primitives, then each pack overlay. A later overlay wins."""
        return build_registry(*self.rule_overlays)

    def overlay_rules(self, registry: Mapping[str, Evaluator]) -> Pack:
        """Return this pack with ``registry`` applied after the ones it has."""
        return replace(self, rule_overlays=(*self.rule_overlays, dict(registry)))

    def load_subjects(self, envelope: Envelope, session: Any = None) -> dict[str, Any]:
        """Fixture rows for a private-banking client or product.

        A pack with no fixture does not invent subjects. A named subject the
        fixture does not hold is a miss, not an empty row.
        """
        if not self.clients and not self.products:
            return {}
        tables = {"client": self.clients, "product": self.products}
        loaded: dict[str, Any] = {}
        for kind, ref in envelope.all_subjects().items():
            table = tables.get(kind)
            if table is None or str(ref) not in table:
                raise LookupError(f"{kind} {ref!r} is not in the {self.name} pack")
            loaded[kind] = dict(table[str(ref)])
        return loaded

    def resolve_attributes(
        self, envelope: Envelope, subjects: dict[str, Any]
    ) -> dict[str, Any]:
        """Attributes the pack's rules read.

        The decision clock is the injected clock, not the caller's context.
        A missing client or product leaves the attribute unset so the rule
        fails closed.
        """
        attrs: dict[str, Any] = {}
        if self.stamp_time:
            moment = self.clock()
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            attrs["local_hour"] = f"{moment.hour:02d}"
            attrs["local_weekday"] = _WEEKDAYS[moment.weekday()]
        if envelope.roles:
            attrs["role"] = envelope.roles[0]
        subject = subjects.get("subject")
        if isinstance(subject, dict) and subject.get("subject_owner") is not None:
            attrs["subject_owner"] = subject.get("subject_owner")
        if self.clients or self.products:
            client = subjects.get("client") if isinstance(subjects.get("client"), dict) else {}
            product = subjects.get("product") if isinstance(subjects.get("product"), dict) else {}
            assert isinstance(client, dict)
            assert isinstance(product, dict)
            if client:
                attrs["booking_centre"] = client.get("booking_centre")
                attrs["client_domicile"] = client.get("client_domicile")
                attrs["solicitation"] = client.get("solicitation")
                attrs["research_distribution"] = client.get("research_distribution")
                attrs["relationship_manager"] = client.get("relationship_manager")
                if "contact_count" in client:
                    attrs["contact_count"] = client.get("contact_count")
            attrs["rm_location"] = envelope.actor_location
            if product:
                attrs["product_eligible"] = bool(product.get("eligible"))
                attrs["suitable"] = bool(product.get("suitable"))
                attrs["within_mandate"] = bool(product.get("within_mandate"))
                attrs["within_concentration"] = bool(product.get("within_concentration"))
        return attrs

    def engages(self, policy: Any, context: Any) -> bool:
        """Every rule in the pack is in force. Scoping lives in the rule.

        ``decide`` still applies a tripped ``CIRCUIT_BREAKER`` and an open
        revocation when this returns false. A pack cannot skip either check.
        """
        return True


def _tables(meta: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    raw = meta.get(key) or {}
    if not isinstance(raw, dict):
        raise PackError(f"pack.toml {key} must be a table")
    tables: dict[str, dict[str, Any]] = {}
    for name, row in raw.items():
        if not isinstance(row, dict):
            raise PackError(f"pack.toml {key}.{name} must be a table")
        tables[str(name)] = dict(row)
    return tables


def _load_directory(
    root: Path,
    *,
    origin: str,
    clock: Callable[[], datetime] | None,
    rule_overlays: tuple[Mapping[str, Evaluator], ...],
) -> Pack:
    meta = _meta((root / "pack.toml").read_bytes(), root)
    read_pack_files(root)
    policies = read_policy_set(root / "policy")
    return Pack(
        name=str(meta["name"]),
        version=str(meta["version"]),
        origin=origin,
        root=root,
        policies=policies,
        purposes=tuple(meta["purposes"]),
        clients=_tables(meta, "clients"),
        products=_tables(meta, "products"),
        stamp_time=meta.get("stamp_time") is True,
        rule_overlays=rule_overlays,
        clock=clock or _utc_now,
    )


def _candidate(root: Path | None, name: str) -> Path | None:
    if root is None or not root.is_dir():
        return None
    path = root / name
    if (path / "pack.toml").is_file():
        return path
    return None


def resolve_pack(
    name: str,
    *,
    overlay: str | Path | None = None,
    user_root: str | Path | None = None,
    managed_root: str | Path | None = None,
    builtin_root: str | Path | None = None,
    clock: Callable[[], datetime] | None = None,
    rule_overlays: tuple[Mapping[str, Evaluator], ...] = (),
) -> Pack:
    """Resolve ``name``. The user directory wins, then an update, then the builtin.

    ``overlay`` is a policy directory merged on top of that pack. It is not a
    second pack name, and it is not written by an update.
    """
    builtin = Path(builtin_root) if builtin_root is not None else builtin_data()
    user = Path(user_root) if user_root is not None else default_user_root()
    managed = Path(managed_root) if managed_root is not None else default_managed_root()
    root = _candidate(user, name)
    origin = "overlay"
    if root is None:
        root = _candidate(managed, name)
        origin = "managed"
    if root is None:
        root = _candidate(builtin, name)
        origin = "builtin"
    if root is None:
        raise PackError(f"pack {name!r} is not installed")
    pack = _load_directory(root, origin=origin, clock=clock, rule_overlays=rule_overlays)
    if pack.name != name:
        raise PackError(f"pack directory {name!r} declares name {pack.name!r}")
    if overlay is not None:
        extra = read_policy_set(overlay)
        pack = replace(pack, policies=merge_policy_sets(pack.policies, extra))
    return pack


def list_packs(
    *,
    user_root: str | Path | None = None,
    managed_root: str | Path | None = None,
    builtin_root: str | Path | None = None,
) -> list[str]:
    """Sorted pack names found on disk. The pin file is not a catalog."""
    builtin = Path(builtin_root) if builtin_root is not None else builtin_data()
    user = Path(user_root) if user_root is not None else default_user_root()
    managed = Path(managed_root) if managed_root is not None else default_managed_root()
    names: set[str] = set()
    for root in (builtin, managed, user):
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and (child / "pack.toml").is_file():
                names.add(child.name)
    return sorted(names)


def load_source_hash(pack: Pack, overlay: str | Path | None = None) -> str:
    """Hash of the policy files a load compiled, including a user overlay."""
    from kognita.policy import source_hash

    base = source_hash(pack.policy_dir)
    if overlay is None:
        return base
    return canonical_hash({"overlay": source_hash(overlay), "pack": base})
