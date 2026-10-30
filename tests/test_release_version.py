"""Release metadata must agree with the version exposed to callers."""
from pathlib import Path
import re


def test_release_version_matches_package_metadata():
    from kognita import __version__

    root = Path(__file__).resolve().parents[1]
    assert __version__ == "0.6.0"
    project_version = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
    assert project_version is not None
    assert project_version.group(1) == __version__
    lock_entry = re.search(r'name = "kognita"\nversion = "([^"]+)"', (root / "uv.lock").read_text())
    assert lock_entry is not None
    assert lock_entry.group(1) == __version__
