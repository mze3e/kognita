## Description

<!-- What does this PR do? Link any related issues with "Fixes #N" or "Related to #N". -->

## Changes

<!-- Bullet-point list of what changed. -->

## Testing

<!-- How was this tested? Which tests did you add or update? -->

## Compliance & Architecture

- [ ] No new dependencies in the core (except pydantic, sqlmodel, numpy, python-dotenv)
- [ ] `uv run import-linter lint` passes (no layering violations)
- [ ] Decision logic is tested (no API keys required)
- [ ] Evidence chain is checked if applicable

## Checklist

- [ ] Tests added or updated for new behaviour
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] All tests pass (`uv run pytest`)
- [ ] Import layering is clean (`uv run import-linter check`)
