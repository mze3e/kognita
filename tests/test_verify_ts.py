"""T4: the independent Node verifier and Python share golden exports."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kognita.evidence import ChainBreak, verify_export

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / 'tests/fixtures/verify'
_CASES = json.loads((_FIXTURES / 'cases.json').read_text())


@pytest.fixture(scope='module')
def _cli():
    if not shutil.which('node') or not shutil.which('npm'):
        pytest.skip('T4: Node 22 and npm required; both are installed in CI')
    subprocess.run(['npm', 'ci', '--prefix', str(_ROOT / 'verify-ts')], check=True, capture_output=True)
    subprocess.run(['npm', 'run', 'build', '--prefix', str(_ROOT / 'verify-ts')], check=True, capture_output=True)
    return _ROOT / 'verify-ts/dist/_cli.js'


@pytest.mark.parametrize('case', _CASES, ids=lambda case: case['file'])
def test_shared_fixtures_agree(case, _cli):
    path = _FIXTURES / case['file']
    key = _FIXTURES / case['public_key'] if case['public_key'] else None
    artifact = json.loads(path.read_text())
    try:
        count = verify_export(artifact, public_key=key.read_bytes() if key else None)
        accepted = True
    except (ChainBreak, ValueError, TypeError, KeyError):
        # Malformed envelopes and chain breaks must both be refusals.
        accepted = False
    result = subprocess.run(['node', str(_cli), str(path), *(['--public-key', str(key)] if key else [])],
                            capture_output=True, text=True)
    assert accepted == case['valid']
    assert result.returncode == (0 if accepted else 1), result.stderr
    if accepted:
        assert json.loads(result.stdout)['event_count'] == count
    else:
        assert 'T4:' in result.stderr
