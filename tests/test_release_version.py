"""Release metadata must agree with the version exposed to callers."""
from pathlib import Path
import re


def test_release_version_matches_package_metadata():
    from kognita import __version__

    root = Path(__file__).resolve().parents[1]
    assert __version__ == "0.8.0"
    project_version = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
    assert project_version is not None
    assert project_version.group(1) == __version__
    lock_entry = re.search(r'name = "kognita"\nversion = "([^"]+)"', (root / "uv.lock").read_text())
    assert lock_entry is not None
    assert lock_entry.group(1) == __version__


def test_typescript_distribution_metadata_matches_release():
    """The npm distribution must retain its published import and CLI entry points."""
    import json
    from kognita import __version__
    root = Path(__file__).resolve().parents[1] / 'verify-ts'
    package = json.loads((root / 'package.json').read_text())
    lock = json.loads((root / 'package-lock.json').read_text())
    assert package['version'] == lock['version'] == lock['packages']['']['version'] == __version__
    assert package['private'] is False
    assert package['main'] == 'dist/_verify.js'
    assert package['exports']['.'] == {'types': './dist/_verify.d.ts', 'default': './dist/_verify.js'}
    assert package['bin']['kognita-verify'] == 'dist/_cli.js'
    assert not package.get('dependencies')
