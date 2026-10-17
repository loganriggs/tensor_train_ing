"""Packaged snapshot API and legacy script share one authoritative module."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def test_legacy_import_is_canonical_module_and_shares_helper_monkeypatch(monkeypatch):
    canonical = importlib.import_module('tnlm.source_snapshot')
    legacy = importlib.import_module('codex_tools.source_snapshot')
    assert legacy is canonical
    for name in ('create_snapshot', 'verify_snapshot', 'run_snapshot', 'main'):
        assert getattr(legacy, name) is getattr(canonical, name)
    sentinel = object()
    monkeypatch.setattr(legacy, '_source_files', lambda root: sentinel)
    assert canonical._source_files(None) is sentinel


def test_legacy_direct_script_outside_repo_isolated_create_verify_run(tmp_path):
    source = tmp_path / 'source'
    package = source / 'tnlm'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'cli.py').write_text("print('frozen legacy command')\nraise SystemExit(7)\n")
    script = Path(__file__).resolve().parents[1] / 'codex_tools' / 'source_snapshot.py'
    snapshot = tmp_path / 'snapshot'
    prefix = [sys.executable, '-I', '-B', str(script)]
    environment = dict(os.environ, PYTHONPATH='', CUDA_VISIBLE_DEVICES='')
    created = subprocess.run(prefix + ['create', '--source', str(source), '--out', str(snapshot)],
                             cwd=tmp_path, capture_output=True, text=True, env=environment, timeout=30)
    assert created.returncode == 0, created.stderr
    manifest = json.loads(created.stdout)
    assert manifest['version'] == 1 and manifest['source_root'] == str(source)
    verified = subprocess.run(prefix + ['verify', str(snapshot)], cwd=tmp_path,
                              capture_output=True, text=True, env=environment, timeout=30)
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)['verified'] is True
    launched = subprocess.run(prefix + ['run', '--snapshot', str(snapshot), '--', 'probe'],
                              cwd=tmp_path, capture_output=True, text=True, env=environment, timeout=30)
    assert launched.returncode == 7 and 'frozen legacy command' in launched.stdout
