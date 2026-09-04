# Contributing to Kognita

Thank you for your interest in contributing to Kognita! Here's how to get started.

## Development setup

**Requirements**: Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/mze3e/kognita.git
cd kognita
uv sync --extra dev
```

## Running tests

```bash
uv run pytest -v
```

The test suite includes:
- **Core functionality tests** (`test_core.py`) — decision engine, governance, evidence
- **Conformance tests** (`test_conformance.py`) — real-world compliance scenarios
- **BMOS conformance** (`test_bmos_conformance.py`) — two-signature and proposal-apply patterns
- **Packaging tests** (`test_packaging.py`) — import layering and architecture contracts

Every test is offline (no network, no API keys). The core is tested with zero optional dependencies.

## Code architecture

Kognita enforces strict layering via `import-linter` contracts in `pyproject.toml`:

- **The decision engine** (core modules like `governance`, `evidence`, `retrieval`) must stay installable on the four core dependencies alone. These modules are named in the `forbidden_modules` contract.
- **Optional features** (graph, embeddings, LLM adapters) live in `kognita.graph` and `kognita.adapters`, kept independent.
- **CLI and testing** (`kognita.cli`, `kognita.testing`) are loose, not imported by the engine.

To check contracts before committing:

```bash
uv run import-linter lint
```

## Code style

Install pre-commit hooks:

```bash
uv run pre-commit install
```

Linting and formatting are handled by [Ruff](https://docs.astral.sh/ruff/). To run manually:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Submitting a pull request

1. Fork the repository and create a branch: `git checkout -b feat/my-feature`
2. Make your changes and add or update tests
3. Ensure `uv run pytest` passes and `uv run import-linter lint` is clean
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Open a PR against `main`

Please keep PRs focused — one feature or bug fix per PR.

## Adding Architecture Decision Records (ADRs)

Major design decisions are documented as ADRs in `docs/decisions/`. For a new ADR:

1. Create `docs/decisions/NNNN-decision-title.md` (increment NNNN)
2. Follow the template in existing ADRs (Status, Context, Decision, Consequences)
3. Reference the ADR in your PR description
4. Update `CHANGELOG.md` if the decision affects public API

## Releasing (maintainers only)

1. Update the version in `pyproject.toml`
2. Move `[Unreleased]` entries to a new version section in `CHANGELOG.md`
3. Commit: `git commit -m "chore: release vX.Y.Z"`
4. Create a GitHub Release tagged `vX.Y.Z` — the publish workflow triggers automatically

### PyPI Trusted Publisher setup (one-time)

The publish workflow uses [OIDC Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API tokens needed.

To configure on PyPI:
1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher with:
   - **Repository owner**: `mze3e`
   - **Repository name**: `kognita`
   - **Workflow filename**: `publish.yml`
   - **Environment name**: `pypi`

Repeat for TestPyPI at https://test.pypi.org/manage/account/publishing/ using environment name `testpypi`.

## Reporting issues

Please use the [GitHub issue tracker](https://github.com/mze3e/kognita/issues).
