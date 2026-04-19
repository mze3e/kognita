# Contributing to Kognita

Thank you for your interest in contributing! Here's how to get started.

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

With coverage:

```bash
uv run pytest --cov=kognita --cov-report=term-missing
```

The test suite uses only pure unit tests (no API keys required).

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
2. Make your changes and add tests for new behaviour
3. Ensure `uv run pytest` passes
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Open a PR against `main`

Please keep PRs focused — one feature or bug fix per PR.

## Releasing (maintainers only)

1. Update the version in `pyproject.toml` and `src/kognita/__init__.py`
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
