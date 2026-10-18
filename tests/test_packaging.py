"""The packaging claims, executable.

The README says the decision engine installs and runs on four dependencies and
that the package no longer ships a graph engine. Both are claims about
*imports* and files, so both are testable here rather than being left to
reviewer discipline.
"""

from __future__ import annotations

import ast
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
_NOT_ENGINE = {"__init__", "__main__", "config", "exceptions", "cli", "policy"}


def _engine_modules_on_disk() -> set[str]:
    return {
        path.stem for path in PACKAGE_ROOT.glob("*.py") if path.stem not in _NOT_ENGINE
    }


def _engine_modules_in_contract() -> set[str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    contracts = config["tool"]["importlinter"]["contracts"]
    engine = next(
        c
        for c in contracts
        if c["name"] == "the decision engine reaches nothing optional"
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

    Run in a subprocess: this test session may have imported a provider SDK
    elsewhere, so checking ``sys.modules`` in-process would prove nothing.
    """
    probe = (
        "import sys; import kognita; "
        "leaked = sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'graphiti_core', 'kuzu', 'openai', 'anthropic', "
        "'docling', 'docling_core', 'docling_parse', 'docling_ibm_models', "
        "'hermes_cli', 'hermes_bootstrap', 'run_agent', 'agents', 'langgraph', "
        "'langchain_core', 'pydantic_ai', 'opentelemetry'} "
        "or m == 'kognita.ingest' or m.startswith('kognita.ingest.') "
        "or m == 'kognita.otel' or m.startswith('kognita.otel.')); "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"import kognita pulled in optional dependencies: {leaked}"


def test_importing_an_adapter_does_not_import_its_framework():
    """The adapter module is importable with no framework installed."""
    probe = (
        "import sys; "
        "import kognita.adapters.hermes; "
        "import kognita.adapters.openai_agents; "
        "import kognita.adapters.langgraph; "
        "import kognita.adapters.pydantic_ai; "
        "banned = {'hermes_cli', 'hermes_bootstrap', 'run_agent', 'agents', "
        "'langgraph', 'langchain_core', 'pydantic_ai', 'openai'}; "
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in banned); "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"importing an adapter pulled in a framework: {leaked}"


def test_the_engine_is_what_the_top_level_namespace_exports():
    """The names the pitch is about must be reachable as ``from kognita import ...``."""
    for name in (
        "Envelope",
        "decide",
        "record",
        "EvidenceWriter",
        "EgressGuard",
        "retrieve",
        "ToolRegistry",
        "verify_chain",
        "PolicySnapshot",
    ):
        assert hasattr(kognita, name), f"{name} should be exported from kognita"
        assert name in kognita.__all__


def test_no_graph_name_is_exported_from_the_top_level():
    for name in (
        "GraphEngine",
        "GraphConfig",
        "KuzuSession",
        "Kognita",
        "Node",
        "Edge",
    ):
        assert (
            name not in kognita.__all__
        ), f"{name!r} is a graph name and must not appear in kognita.__all__"


def test_the_policy_language_is_outside_the_engine_and_still_constrained():
    """``kognita.policy`` sits above ``decide`` and stays on the core dependencies.

    The engine must not import it. The policy module must not import the CLI,
    the conformance kit, an adapter, or a YAML package. It reaches the use-case
    register only because ``decide`` does; the policy file does not name it.
    """
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    contracts = config["tool"]["importlinter"]["contracts"]
    engine = next(
        c
        for c in contracts
        if c["name"] == "the decision engine reaches nothing optional"
    )
    policy = next(
        c
        for c in contracts
        if c["name"] == "the policy language reaches nothing optional"
    )
    assert "kognita.policy" in engine["forbidden_modules"]
    assert "kognita.policy" not in engine["source_modules"]
    assert "kognita.__main__" not in engine["source_modules"]
    assert policy["source_modules"] == ["kognita.policy"]
    entry = next(
        c for c in contracts if c["name"] == "the module entry reaches nothing optional"
    )
    assert entry["source_modules"] == ["kognita.__main__"]
    for name in ("kognita.cli", "kognita.testing", "kognita.adapters", "yaml"):
        assert name in policy["forbidden_modules"]
    assert "kognita.usecases" not in (PACKAGE_ROOT / "policy.py").read_text()


def test_importing_kognita_does_not_load_the_policy_language():
    """The top-level import stays the decision engine."""
    probe = (
        "import sys; import kognita; "
        "print('yes' if 'kognita.policy' in sys.modules else 'no')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "no"


def test_the_package_ships_no_graph_engine():
    """The Graphiti + Kuzu engine left the package (ADR 0008)."""
    assert not (PACKAGE_ROOT / "graph").exists()
    with pytest.raises(ModuleNotFoundError):
        __import__("kognita.graph")
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = config["project"]["optional-dependencies"]
    assert "graph" not in extras
    pinned = [req for reqs in extras.values() for req in reqs]
    assert not [req for req in pinned if req.startswith(("graphiti", "kuzu"))]


_FRAMEWORK_EXTRAS = ("hermes-agent", "openai-agents", "langgraph", "pydantic-ai")
_ADAPTER_MODULES = {
    "hermes-agent": "kognita.adapters.hermes",
    "openai-agents": "kognita.adapters.openai_agents",
    "langgraph": "kognita.adapters.langgraph",
    "pydantic-ai": "kognita.adapters.pydantic_ai",
}


def test_framework_extras_are_one_per_adapter_and_stay_out_of_core():
    """Each approved framework is its own extra. The core dependencies do not move."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    dependencies = config["project"]["dependencies"]
    extras = config["project"]["optional-dependencies"]
    assert dependencies == [
        "pydantic>=2.7.0",
        "sqlmodel>=0.0.22",
        "numpy>=1.26.0",
        "python-dotenv>=1.0.0",
    ]
    for name in _FRAMEWORK_EXTRAS:
        assert name in extras, name
        assert extras[name], name
    all_extra = " ".join(extras["all"])
    for name in _FRAMEWORK_EXTRAS:
        assert name not in all_extra


def test_only_the_owning_adapter_may_import_a_framework():
    """The import-linter ignore list is the allow-list, and it names adapters only."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    contracts = config["tool"]["importlinter"]["contracts"]
    contract = next(
        c for c in contracts if c["name"] == "the core never imports a framework"
    )
    assert contract["source_modules"] == ["kognita"]
    ignored = contract["ignore_imports"]
    allowed = set(_ADAPTER_MODULES.values())
    for chain in ignored:
        source = chain.split("->", 1)[0].strip()
        assert source in allowed, chain


@pytest.mark.parametrize(
    "module", ["hermes", "openai_agents", "langgraph", "pydantic_ai"]
)
def test_each_framework_adapter_is_under_200_lines(module):
    path = PACKAGE_ROOT / "adapters" / f"{module}.py"
    lines = path.read_text().splitlines()
    assert len(lines) < 200, f"{path.name} is {len(lines)} lines"


@pytest.mark.parametrize(
    "removed", ["Kognita", "KognitaConfig", "KuzuSession", "GraphEngine"]
)
def test_removed_graph_names_explain_the_removal(removed):
    """A clean break still has to be navigable.

    ``from kognita import Kognita`` worked in 0.1.x, and ``kognita.graph`` in
    0.2 and 0.3. The failure should say the engine was removed and how to keep
    using it, rather than leaving a reader to guess.
    """
    with pytest.raises(AttributeError) as excinfo:
        getattr(kognita, removed)
    message = str(excinfo.value)
    assert "ADR 0008" in message
    assert "kognita[graph]==0.3.0" in message


def _contracts() -> list[dict]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return config["tool"]["importlinter"]["contracts"]


def test_otel_stays_out_of_the_engine_and_out_of_all():
    """OpenTelemetry is an extra. The flight recorder is engine code."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = config["project"]["optional-dependencies"]
    assert extras["otel"] == [
        "opentelemetry-api>=1.27.0",
        "opentelemetry-sdk>=1.27.0",
        "opentelemetry-exporter-otlp-proto-http>=1.27.0",
    ]
    assert "otel" not in " ".join(extras["all"])
    engine = next(
        c
        for c in _contracts()
        if c["name"] == "the decision engine reaches nothing optional"
    )
    assert "kognita.recorder" in engine["source_modules"]
    assert "kognita.otel" not in engine["source_modules"]
    assert "opentelemetry" in engine["forbidden_modules"]
    assert "kognita.otel" in engine["forbidden_modules"]
    contract = next(
        c
        for c in _contracts()
        if c["name"] == "OpenTelemetry is reached only from the otel extra"
    )
    assert contract["source_modules"] == ["kognita"]
    assert set(contract["forbidden_modules"]) == {"opentelemetry", "kognita.otel"}
    assert contract["ignore_imports"] == [
        "kognita.cli -> kognita.otel",
        "kognita.otel -> opentelemetry",
    ]


def test_only_the_otel_package_imports_opentelemetry():
    """The engine and the CLI do not import OpenTelemetry at module level."""
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        for offset, module in _imported_modules(path):
            otel_pkg = module == "kognita.otel" or module.startswith("kognita.otel.")
            sdk = module == "opentelemetry" or module.startswith("opentelemetry.")
            if otel_pkg:
                if rel != "cli.py" or offset == 0:
                    offenders.append(f"{rel} imports {module} at column {offset}")
                continue
            if sdk and (not rel.startswith("otel/") or offset == 0):
                offenders.append(f"{rel} imports {module} at column {offset}")
    assert not offenders, offenders


def test_the_engine_contract_forbids_docling_and_ingestion():
    """Core modules stay installable without the docling extra."""
    engine = next(
        c
        for c in _contracts()
        if c["name"] == "the decision engine reaches nothing optional"
    )
    forbidden = set(engine["forbidden_modules"])
    for name in (
        "kognita.ingest",
        "docling",
        "docling_core",
        "docling_parse",
        "docling_ibm_models",
    ):
        assert name in forbidden


def test_ingest_modules_are_in_the_docling_contract():
    """A new ingest module is forbidden from importing Docling unless it is the backend."""
    guard = next(
        c for c in _contracts() if c["name"] == "ingestion does not import docling"
    )
    listed = {name.removeprefix("kognita.ingest.") for name in guard["source_modules"]}
    on_disk = {
        path.stem
        for path in (PACKAGE_ROOT / "ingest").glob("*.py")
        if path.stem not in {"__init__", "docling_backend"}
    }
    assert listed == on_disk
    forbidden = set(guard["forbidden_modules"])
    assert {
        "docling",
        "docling_core",
        "docling_parse",
        "docling_ibm_models",
    } <= forbidden


def test_docling_extra_pins_format_extras_only():
    """The extra is an exact docling-slim pin, not the torch metapackage."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extra = config["project"]["optional-dependencies"]["docling"]
    assert extra == ["docling-slim[format-docx,format-html,format-markdown]==2.140.0"]
    requirement = extra[0]
    assert "feat-chunking" not in requirement
    assert "format-pdf" not in requirement
    assert not requirement.startswith("docling==")
    assert "torch" not in requirement


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text())
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.col_offset, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.col_offset, node.module or ""))
    return found


def _is_docling_module(module: str) -> bool:
    return module == "docling" or module.startswith(
        ("docling.", "docling_core", "docling_parse", "docling_ibm_models")
    )


def test_only_the_guarded_backend_imports_docling():
    """Docling imports are indented inside the backend. Nothing else imports it."""
    for path in (PACKAGE_ROOT / "ingest").glob("*.py"):
        for offset, module in _imported_modules(path):
            if path.name == "docling_backend.py" and _is_docling_module(module):
                assert offset > 0, f"{path.name} imports {module} at module level"
                continue
            assert not _is_docling_module(module), f"{path.name} imports {module}"
            if module.endswith("docling_backend"):
                assert offset > 0, f"{path.name} imports the backend at module level"


def test_importing_ingest_does_not_load_docling():
    """``import kognita.ingest`` must succeed with Docling absent from sys.modules."""
    probe = (
        "import sys; import kognita.ingest; "
        "leaked = sorted(m for m in sys.modules "
        "if m == 'docling' or m.startswith('docling.') or m.startswith('docling_')); "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"import kognita.ingest pulled in Docling: {leaked}"


def test_wheel_force_include_does_not_duplicate_package_files():
    """Hatch already ships every file under ``src/kognita``.

    Force-including a path inside the package adds those files a second time,
    and the wheel build refuses the duplicate archive entry.
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    for source in wheel.get("force-include", {}):
        assert not source.startswith("src/kognita"), source
    assert (PACKAGE_ROOT / "packs" / "data" / "pins.toml").is_file()
