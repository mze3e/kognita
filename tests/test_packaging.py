"""The packaging claims, executable.

The README says the decision engine installs and runs on four dependencies and
that the graph engine is one optional backend rather than the identity of the
library. Both are claims about *imports*, so both are testable here rather than
being left to reviewer discipline.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import kognita

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "kognita"

#: Top-level modules that are deliberately outside the engine contract.
#: ``config`` and ``exceptions`` have their own, stricter contract; ``cli`` sits
#: above the engine and may import anything.
_NOT_ENGINE = {"__init__", "config", "exceptions", "cli"}


def _engine_modules_on_disk() -> set[str]:
    return {
        path.stem
        for path in PACKAGE_ROOT.glob("*.py")
        if path.stem not in _NOT_ENGINE
    }


def _engine_modules_in_contract() -> set[str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    contracts = config["tool"]["importlinter"]["contracts"]
    engine = next(
        c for c in contracts if c["name"] == "the decision engine reaches nothing optional"
    )
    return {name.removeprefix("kognita.") for name in engine["source_modules"]}


def test_every_engine_module_is_covered_by_the_contract():
    """A new engine module must be added to the import-linter contract.

    Flattening the engine into ``kognita`` bought a namespace that matches the
    pitch, at the cost of a contract that names its modules one by one. A module
    missing from that list is not merely undocumented — it is unconstrained, and
    free to import a graph database without anything failing.
    """
    on_disk = _engine_modules_on_disk()
    in_contract = _engine_modules_in_contract()

    unconstrained = on_disk - in_contract
    assert not unconstrained, (
        "these modules are not covered by the import-linter engine contract, so "
        "nothing stops them importing an optional dependency: "
        f"{sorted(unconstrained)}. Add them to source_modules in pyproject.toml."
    )

    stale = in_contract - on_disk
    assert not stale, f"contract names modules that no longer exist: {sorted(stale)}"


def test_importing_kognita_loads_no_optional_dependency():
    """``import kognita`` must not drag in a graph database or a provider SDK.

    Run in a subprocess: this test session has almost certainly imported the
    graph engine elsewhere, so checking ``sys.modules`` in-process would prove
    nothing.
    """
    probe = (
        "import sys; import kognita; "
        "leaked = sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'graphiti_core', 'kuzu', 'openai', 'anthropic'}); "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"import kognita pulled in optional dependencies: {leaked}"


def test_the_engine_is_what_the_top_level_namespace_exports():
    """The names the pitch is about must be reachable as ``from kognita import ...``."""
    for name in ("Envelope", "decide", "record", "EvidenceWriter", "EgressGuard",
                 "retrieve", "ToolRegistry", "verify_chain", "PolicySnapshot"):
        assert hasattr(kognita, name), f"{name} should be exported from kognita"
        assert name in kognita.__all__


def test_no_graph_name_is_exported_from_the_top_level():
    """The graph engine is reached at ``kognita.graph``, never advertised here."""
    for name in ("GraphEngine", "GraphConfig", "KuzuSession", "Kognita", "Node", "Edge"):
        assert name not in kognita.__all__, (
            f"{name!r} is a graph name and must not appear in kognita.__all__"
        )


@pytest.mark.parametrize(
    ("retired", "destination"),
    [
        ("Kognita", "kognita.graph"),
        ("KognitaConfig", "kognita.graph"),
        ("KuzuSession", "kognita.graph"),
    ],
)
def test_retired_graph_names_say_where_they_went(retired, destination):
    """A clean break still has to be navigable.

    ``from kognita import Kognita`` worked in 0.1.x. It is gone, and the failure
    should name the module that owns it rather than leaving a reader to guess.
    """
    with pytest.raises(AttributeError) as excinfo:
        getattr(kognita, retired)
    message = str(excinfo.value)
    assert destination in message
    assert "kognita[graph]" in message
