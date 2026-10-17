"""Immutable source snapshots use a fake CLI, never model or GPU execution."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest


def api():
    return importlib.import_module('codex_tools.source_snapshot')


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'original'
    package = root / 'tnlm'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'sibling.py').write_text("VALUE = 'captured-source'\n")
    (package / 'resources.json').write_text('{"captured": true}\n')
    (package / 'cli.py').write_text('''import argparse
import json
import os
from pathlib import Path
import sys
from .sibling import VALUE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--candidates')
    parser.add_argument('--report', required=True)
    parser.add_argument('--exit-code', type=int, default=0)
    args = parser.parse_args()
    result = {'value': VALUE, 'cwd': os.getcwd(), 'package_root': str(Path(__file__).parent.resolve()),
              'isolated': sys.flags.isolated, 'dont_write_bytecode': sys.dont_write_bytecode}
    for name in ('config', 'candidates'):
        value = getattr(args, name)
        result[name + '_arg'] = value
        result[name] = Path(value).read_text() if value else None
    Path(args.report).write_text(json.dumps(result))
    return args.exit_code

if __name__ == '__main__':
    raise SystemExit(main())
''')
    cache = package / '__pycache__'
    cache.mkdir()
    (cache / 'ignored.pyc').write_bytes(b'bytecode')
    (cache / 'ignored.txt').write_text('whole cache directory is excluded')
    (package / 'orphan.pyc').write_bytes(b'bytecode')
    config = root / 'config with space.yaml'
    config.write_text('captured: config\n')
    candidates = root / 'candidates.json'
    candidates.write_text('[{"name":"captured"}]\n')
    return root, config, candidates


def test_capture_tree_inputs_hashes_and_no_overwrite(source, tmp_path):
    root, config, candidates = source
    destination = tmp_path / 'snapshot'
    manifest = api().create_snapshot(root, destination, inputs=[config, candidates])
    assert manifest['version'] == 1
    assert manifest['source_root'] == str(root.resolve())
    assert manifest['working_directory'] == str(root.resolve())
    assert manifest['input_paths'] == {str(config.resolve()): 'inputs/0-config with space.yaml',
                                       str(candidates.resolve()): 'inputs/1-candidates.json'}
    expected = {'tnlm/__init__.py', 'tnlm/cli.py', 'tnlm/sibling.py', 'tnlm/resources.json',
                'inputs/0-config with space.yaml', 'inputs/1-candidates.json'}
    assert set(manifest['files']) == expected
    for name, metadata in manifest['files'].items():
        value = (destination / name).read_bytes()
        assert metadata == {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
    assert api().verify_snapshot(destination) == manifest
    before = (destination / 'manifest.json').read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        api().create_snapshot(root, destination, inputs=[])
    assert (destination / 'manifest.json').read_bytes() == before


@pytest.mark.parametrize('attached', [False, True])
def test_isolated_run_rewrites_inputs_preserves_cwd_and_ignores_original_mutation(
        source, tmp_path, monkeypatch, attached):
    root, config, candidates = source
    destination = tmp_path / 'snapshot'
    manifest = api().create_snapshot(root, destination, inputs=[config, candidates])
    (root / 'tnlm' / 'sibling.py').write_text("VALUE = 'wrong-current-source'\n")
    config.write_text('wrong current config')
    candidates.unlink()  # A frozen original identity must not need current bytes.
    monkeypatch.setenv('PYTHONPATH', str(root))
    workdir = tmp_path / 'working directory'
    workdir.mkdir()
    args = ([f'--config={config}', f'--candidates={candidates}'] if attached else
            ['--config', str(config), '--candidates', str(candidates)])
    assert api().run_snapshot(destination, args + ['--report', 'observed.json'], workdir=workdir) == 0
    report = json.loads((workdir / 'observed.json').read_text())
    assert report['value'] == 'captured-source'
    assert report['cwd'] == str(workdir.resolve())
    assert report['package_root'] == str(destination / 'tnlm')
    assert report['isolated'] == 1 and report['dont_write_bytecode'] is True
    assert report['config'] == 'captured: config\n'
    assert report['candidates'] == '[{"name":"captured"}]\n'
    for name, original in [('config', config), ('candidates', candidates)]:
        assert Path(report[name + '_arg']) == destination / manifest['input_paths'][str(original.resolve())]
    assert not list(destination.rglob('*.pyc'))
    assert api().verify_snapshot(destination) == manifest


def test_default_cwd_copied_input_acceptance_and_exit_status(source, tmp_path):
    root, config, candidates = source
    destination = tmp_path / 'snapshot'
    manifest = api().create_snapshot(root, destination, inputs=[config, candidates])
    copied = destination / manifest['input_paths'][str(config.resolve())]
    assert api().run_snapshot(destination,
        ['--config', str(copied), '--report', 'default-cwd.json', '--exit-code', '13']) == 13
    report = json.loads((root / 'default-cwd.json').read_text())
    assert report['cwd'] == str(root.resolve()) and report['config'] == config.read_text()


@pytest.mark.parametrize('flag', ['--config', '--candidates'])
@pytest.mark.parametrize('attached', [False, True])
def test_unpinned_input_rejected_without_running(source, tmp_path, flag, attached):
    root, _, _ = source
    destination = tmp_path / 'snapshot'
    api().create_snapshot(root, destination)
    unpinned = tmp_path / 'unpinned.yaml'
    unpinned.write_text('not captured')
    args = [f'{flag}={unpinned}'] if attached else [flag, str(unpinned)]
    with pytest.raises((ValueError, RuntimeError)):
        api().run_snapshot(destination, args + ['--report', 'must-not-exist.json'])
    assert not (root / 'must-not-exist.json').exists()


@pytest.mark.parametrize('relative', ['', 'inputs'])
def test_cwd_inside_snapshot_rejected(source, tmp_path, relative):
    root, config, _ = source
    destination = tmp_path / 'snapshot'
    api().create_snapshot(root, destination, inputs=[config])
    with pytest.raises((ValueError, RuntimeError)):
        api().run_snapshot(destination, ['--report', 'must-not-exist.json'],
                           workdir=destination / relative)
    assert not list(destination.rglob('must-not-exist.json'))


@pytest.mark.parametrize('damage', ['tamper', 'missing', 'extra', 'symlink'])
def test_integrity_rejects_damage_before_run(source, tmp_path, damage):
    root, config, _ = source
    destination = tmp_path / 'snapshot'
    api().create_snapshot(root, destination, inputs=[config])
    if damage == 'tamper':
        (destination / 'tnlm' / 'sibling.py').write_text('tampered')
    elif damage == 'missing':
        (destination / 'tnlm' / 'sibling.py').unlink()
    elif damage == 'extra':
        (destination / 'unexpected.py').write_text('extra')
    else:
        (destination / 'extra-link').symlink_to(config)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        api().verify_snapshot(destination)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        api().run_snapshot(destination, ['--report', 'must-not-exist.json'])
    assert not (root / 'must-not-exist.json').exists()


def test_failed_creation_cleans_destination(source, tmp_path):
    root, config, _ = source
    destination = tmp_path / 'failed'
    with pytest.raises((ValueError, OSError)):
        api().create_snapshot(root, destination, inputs=[config, tmp_path / 'missing-input'])
    assert not destination.exists()
    # Symlinked source trees cannot silently follow mutable outside code.
    (root / 'tnlm' / 'linked.py').symlink_to(config)
    with pytest.raises((ValueError, OSError)):
        api().create_snapshot(root, destination, inputs=[config])
    assert not destination.exists()


@pytest.mark.parametrize('flag', ['--conf', '--cand'])
@pytest.mark.parametrize('attached', [False, True])
def test_abbreviated_input_flags_rejected(source, tmp_path, flag, attached):
    root, config, _ = source
    destination = tmp_path / 'snapshot'
    api().create_snapshot(root, destination, inputs=[config])
    args = [f'{flag}={config}'] if attached else [flag, str(config)]
    with pytest.raises(ValueError):
        api().run_snapshot(destination, args + ['--report', 'must-not-exist.json'])
    assert not (root / 'must-not-exist.json').exists()


def test_relative_inputs_and_launch_references_bind_to_declared_roots(source, tmp_path):
    root, config, _ = source
    destination = tmp_path / 'snapshot'
    manifest = api().create_snapshot(root, destination, inputs=[config.name])
    assert str(config.resolve()) in manifest['input_paths']
    assert api().run_snapshot(destination, ['--config', config.name, '--report', 'relative.json']) == 0
    report = json.loads((root / 'relative.json').read_text())
    assert report['config'] == 'captured: config\n'
    assert Path(report['config_arg']).is_relative_to(destination)


def test_source_change_after_destination_claim_rejects_and_cleans(source, tmp_path, monkeypatch):
    root, config, _ = source
    destination = tmp_path / 'concurrent-edit'
    module = api()
    original = module._source_files
    calls = []
    def changing_source(path):
        calls.append(True)
        if len(calls) == 2:
            assert destination.exists()
            assert (destination / 'tnlm' / 'cli.py').is_file()
            (root / 'tnlm' / 'sibling.py').write_text("VALUE = 'concurrent edit'\n")
        return original(path)
    monkeypatch.setattr(module, '_source_files', changing_source)
    with pytest.raises(RuntimeError, match='source changed'):
        module.create_snapshot(root, destination, inputs=[config])
    assert len(calls) == 2
    assert not destination.exists()


def test_recorded_workdir_survives_later_invocation_from_other_directory(source, tmp_path, monkeypatch):
    root, config, _ = source
    launch = tmp_path / 'experiment'
    launch.mkdir()
    later = tmp_path / 'later-shell'
    later.mkdir()
    destination = tmp_path / 'snapshot'
    manifest = api().create_snapshot(root, destination, inputs=[config], workdir=launch)
    assert manifest['working_directory'] == str(launch)
    monkeypatch.chdir(later)
    # Relative report output and original input path resolve against recorded cwd,
    # not package source root or the caller's current shell directory.
    relative_config = '../original/' + config.name
    assert api().run_snapshot(destination, ['--config', relative_config, '--report', 'reentry.json']) == 0
    report = json.loads((launch / 'reentry.json').read_text())
    assert report['cwd'] == str(launch)
    assert report['config'] == 'captured: config\n'
    assert not (root / 'reentry.json').exists()
    assert not (later / 'reentry.json').exists()
    assert api().run_snapshot(destination, ['--report', 'override.json'], workdir=later) == 0
    assert json.loads((later / 'override.json').read_text())['cwd'] == str(later)


def test_legacy_manifest_without_workdir_keeps_source_root_default(source, tmp_path):
    root, _, _ = source
    destination = tmp_path / 'snapshot'
    api().create_snapshot(root, destination)
    manifest_path = destination / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    del manifest['working_directory']
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest))
    assert api().run_snapshot(destination, ['--report', 'legacy.json']) == 0
    assert json.loads((root / 'legacy.json').read_text())['cwd'] == str(root)


@pytest.mark.parametrize('value', [None, 7, 'relative/path'])
def test_invalid_recorded_workdir_rejected(source, tmp_path, value):
    root, _, _ = source
    destination = tmp_path / 'snapshot'
    api().create_snapshot(root, destination)
    path = destination / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['working_directory'] = value
    path.chmod(0o644)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='working directory'):
        api().verify_snapshot(destination)
