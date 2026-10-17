"""Automatic source snapshots launch only fresh, explicitly funded sweeps."""
import json
from pathlib import Path

import pytest
import yaml

from tnlm.cli import main


def arguments(config, tmp_path, *, dry):
    config.raw['project']['dry_run'] = dry
    config_file = tmp_path / 'base.yaml'
    config_file.write_text(yaml.safe_dump(config.to_dict()))
    grid = tmp_path / 'grid.yaml'
    grid.write_text('- name: first\n  overrides: {}\n')
    return ['sweep', '--config', str(config_file), '--candidates', str(grid),
            '--out', str(tmp_path / 'out'), '--max-seconds', '40',
            '--evaluation-max-seconds', '20', '--device', 'cpu']


def test_snapshot_dry_run_has_no_writes(config, tmp_path, capsys):
    argv = arguments(config, tmp_path, dry=True)
    before = set(tmp_path.rglob('*'))
    assert main([*argv, '--snapshot-sources']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['dry_run'] is True
    assert result['source_snapshot']['created'] is False
    assert result['source_snapshot']['path'] == str(tmp_path / 'out.sources')
    assert set(tmp_path.rglob('*')) == before


def test_snapshot_launch_forwards_controls_and_child_status(config, tmp_path, monkeypatch, capsys):
    import tnlm.source_snapshot as snapshot
    import tnlm.sweep as sweep

    argv = arguments(config, tmp_path, dry=False)
    events = []
    def create(root, destination, inputs, *, workdir):
        assert (root / 'tnlm/cli.py').is_file()
        assert destination == tmp_path / 'frozen'
        assert inputs == [tmp_path / 'base.yaml', tmp_path / 'grid.yaml']
        assert workdir == Path.cwd()
        events.append('create')
    def run(destination, child, *, workdir):
        assert events == ['create']
        assert destination == tmp_path / 'frozen'
        assert workdir == Path.cwd()
        assert '--snapshot-sources' not in child
        assert child == ['sweep', '--config', str(tmp_path / 'base.yaml'), '--candidates', str(tmp_path / 'grid.yaml'),
                      '--out', str(tmp_path / 'out'), '--max-seconds', '40.0',
                      '--evaluation-max-seconds', '20.0', '--max-prediction-tokens', '31',
                      '--bootstrap-samples', '7', '--device', 'cpu', '--no-cooldown',
                      '--reference-candidate', 'first', '--prune-completed-intermediates']
        return 17
    monkeypatch.setattr(snapshot, 'create_snapshot', create)
    monkeypatch.setattr(snapshot, 'run_snapshot', run)
    monkeypatch.setattr(sweep, 'run_sweep', lambda *a, **k: pytest.fail('parent launched mutable sources'))
    assert main([*argv, '--snapshot-sources', str(tmp_path / 'frozen'), '--no-cooldown',
                 '--reference-candidate', 'first', '--prune-completed-intermediates',
                 '--max-prediction-tokens', '31', '--bootstrap-samples', '7']) == 17
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('case', ['existing-output', 'snapshot-inside-output', 'output-inside-snapshot'])
def test_snapshot_rejects_reentry_and_overlap_before_copy(config, tmp_path, monkeypatch, case):
    import tnlm.source_snapshot as snapshot

    argv = arguments(config, tmp_path, dry=False)
    destination = tmp_path / 'out.sources'
    if case == 'existing-output':
        (tmp_path / 'out').mkdir()
    elif case == 'snapshot-inside-output':
        destination = tmp_path / 'out/frozen'
    else:
        destination = tmp_path / 'container'
        argv[argv.index('--out') + 1] = str(destination / 'out')
    monkeypatch.setattr(snapshot, 'create_snapshot', lambda *a, **k: pytest.fail('invalid snapshot copied'))
    with pytest.raises((ValueError, FileExistsError)):
        main([*argv, '--snapshot-sources', str(destination)])


def test_candidate_dry_policy_controls_snapshot_not_base_alone(config, tmp_path, capsys):
    argv = arguments(config, tmp_path, dry=False)
    (tmp_path / 'grid.yaml').write_text('- name: first\n  overrides:\n    project:\n      dry_run: true\n')
    assert main([*argv, '--snapshot-sources']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['dry_run'] is True
    assert not (tmp_path / 'out.sources').exists()


@pytest.mark.slow
def test_actual_frozen_child_cpu_sweep(config, tmp_path, monkeypatch):
    import hashlib
    from tnlm.data import prepare_data
    from tnlm.source_snapshot import verify_snapshot
    from helpers import FixtureTokenizer

    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')
    source = tmp_path / 'validation.jsonl'
    source.write_text(json.dumps({'id': 'separate-validation', 'text': 'nopqrstuvwxyz'}) + '\n')
    manifest = prepare_data(tmp_path / 'validation', source_jsonl=source,
        revision='snapshot-fixture-v1', tokenizer=FixtureTokenizer(),
        split_fractions={'train': 0., 'val': 1., 'test': 0., 'calibration': 0.})
    config.raw['evaluation'] = {'validation_manifest': str(manifest)}
    config.raw['project']['milestone_tokens'] = []
    config.train.rung_total_tokens = 7
    config.train.deterministic = True  # Exercise the actual YAML schema in the frozen child.
    argv = arguments(config, tmp_path, dry=False)
    original_config = (tmp_path / 'base.yaml').read_bytes()
    original_grid = (tmp_path / 'grid.yaml').read_bytes()
    import tnlm.source_snapshot as snapshot_api
    original_create = snapshot_api.create_snapshot
    def capture_then_edit_live_inputs(*args, **kwargs):
        manifest = original_create(*args, **kwargs)
        # Reproduce the near-miss: live inputs change before the child opens them.
        (tmp_path / 'base.yaml').write_text('model: {unknown_future_field: true}\n')
        (tmp_path / 'grid.yaml').write_text('not a candidate list\n')
        return manifest
    monkeypatch.setattr(snapshot_api, 'create_snapshot', capture_then_edit_live_inputs)
    monkeypatch.chdir(tmp_path)  # Launch outside the package's source root.
    assert main([*argv, '--snapshot-sources', '--no-cooldown',
                 '--max-prediction-tokens', '7', '--bootstrap-samples', '0']) == 0
    frozen = tmp_path / 'out.sources'
    inventory = verify_snapshot(frozen)
    assert inventory['working_directory'] == str(tmp_path)
    journal = json.loads((tmp_path / 'out/sweep.json').read_text())
    assert journal['status'] == 'complete'
    for name, digest in journal['identity']['runtime_source_sha256'].items():
        assert digest == inventory['files']['tnlm/' + name]['sha256']
    expected_config = frozen / inventory['input_paths'][str(tmp_path / 'base.yaml')]
    assert expected_config.read_bytes() == original_config
    assert (frozen / inventory['input_paths'][str(tmp_path / 'grid.yaml')]).read_bytes() == original_grid
    results = list((tmp_path / 'out').rglob('result.json'))
    training = [json.loads(path.read_text()) for path in results if 'parent' in path.parts]
    assert training and all(row['runtime']['determinism']['algorithms_enabled'] for row in training)
    assert hashlib.sha256((frozen / 'tnlm/cli.py').read_bytes()).hexdigest() == journal['identity']['runtime_source_sha256']['cli.py']
    ledger = json.loads((tmp_path / 'spend.json').read_text())
    assert len(ledger['runs']) == 2
    assert all(row['status'] == 'complete' for row in ledger['runs'].values())
