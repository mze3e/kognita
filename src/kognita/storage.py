"""Persistence: serialize / deserialize GraphSnapshot plus optional Kuzu DB copy."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from kognita.types import Edge, Node


@dataclass
class GraphSnapshot:
    """JSON-serializable snapshot of a knowledge graph.

    Stores lightweight ``Node`` / ``Edge`` dataclasses plus arbitrary episode
    bookkeeping and metadata. The Kuzu database itself is not part of the
    snapshot; pass ``include_kuzu_db=`` to :func:`save_snapshot` to copy it
    alongside the JSON.
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
            "episodes": list(self.episodes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphSnapshot":
        nodes = {n["uuid"]: Node(**n) for n in data.get("nodes", [])}
        edges = [Edge(**e) for e in data.get("edges", [])]
        return cls(
            nodes=nodes,
            edges=edges,
            episodes=list(data.get("episodes", [])),
            metadata=dict(data.get("metadata", {})),
        )


def content_hash(data: bytes) -> str:
    """MD5 hex digest — useful for content-addressable dedup of source documents."""
    return hashlib.md5(data).hexdigest()


def save_snapshot(
    snapshot: GraphSnapshot,
    path: str | Path,
    *,
    include_kuzu_db: str | Path | None = None,
) -> str:
    """Write ``snapshot`` as JSON files under ``path`` (creates the directory).

    Layout written::

        path/
        ├── snapshot.json       # nodes + edges + episodes
        └── metadata.json       # snapshot.metadata + derived counts + timestamp
        └── kuzu_db/            # copied if include_kuzu_db is provided

    Returns the absolute path of the directory that was written.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    data = snapshot.to_dict()
    with (out / "snapshot.json").open("w") as f:
        json.dump(
            {"nodes": data["nodes"], "edges": data["edges"], "episodes": data["episodes"]},
            f,
            indent=2,
            default=str,
        )

    metadata = {
        **data["metadata"],
        "saved_at": datetime.now().isoformat(),
        "node_count": len(snapshot.nodes),
        "edge_count": len(snapshot.edges),
        "episode_count": len(snapshot.episodes),
        "kuzu_db_saved": False,
    }

    if include_kuzu_db:
        src = Path(include_kuzu_db)
        if src.exists():
            dst = out / "kuzu_db"
            if src.resolve() != dst.resolve():
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            metadata["kuzu_db_saved"] = True

    with (out / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return str(out.resolve())


def load_snapshot(path: str | Path) -> GraphSnapshot:
    """Load a :class:`GraphSnapshot` previously written by :func:`save_snapshot`."""
    base = Path(path)
    with (base / "snapshot.json").open("r") as f:
        data = json.load(f)

    metadata: dict[str, Any] = {}
    meta_file = base / "metadata.json"
    if meta_file.exists():
        with meta_file.open("r") as f:
            metadata = json.load(f)

    data["metadata"] = metadata
    return GraphSnapshot.from_dict(data)


def saved_kuzu_db_path(snapshot_dir: str | Path) -> str | None:
    """Return the path to a snapshot's copied Kuzu DB directory, if any."""
    candidate = Path(snapshot_dir) / "kuzu_db"
    return str(candidate) if candidate.exists() else None
