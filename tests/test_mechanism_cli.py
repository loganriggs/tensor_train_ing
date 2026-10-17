"""End-to-end command interfaces for the known-mechanism probes."""
import json

import yaml

from tnlm.cli import main
from tnlm.data import load_mechanism_tasks


def test_prepare_mechanisms_cli(tmp_path, capsys):
    assert main(['prepare_mechanisms', '--out', str(tmp_path / 'tasks'), '--seed', '17',
                 '--train-examples', '8', '--validation-examples', '8']) == 0
    result = json.loads(capsys.readouterr().out)
    metadata, rows = load_mechanism_tasks(result['manifest'])
    assert len(rows) == 16
    assert {row['task'] for row in rows} == {'boolean_hierarchy', 'associative_retrieval'}
    assert metadata['seed'] == 17


def test_mechanism_evaluation_cli_dryrun_needs_no_files(config, tmp_path, capsys):
    config.raw['project']['dry_run'] = True
    path = tmp_path / 'dry-mechanisms.yaml'
    path.write_text(yaml.safe_dump(config.to_dict()))
    out = tmp_path / 'dry-out'
    assert main(['evaluate_mechanisms', '--config', str(path), '--checkpoint', 'missing',
                 '--task-manifest', 'missing', '--out', str(out), '--max-seconds', '30',
                 '--component', '0:mlp']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['dry_run'] and result['components'] == ['0:mlp']
    assert result['max_prompt_tokens'] == config.raw['data']['context']
    assert not out.exists()




def test_real_mechanism_evaluation_cli(checkpoint_fixture, tmp_path, capsys):
    cfg, checkpoint, tasks = checkpoint_fixture
    path = tmp_path / 'mechanisms.yaml'
    path.write_text(yaml.safe_dump(cfg.to_dict()))
    out = tmp_path / 'cli-mechanisms'
    assert main(['evaluate_mechanisms', '--config', str(path), '--checkpoint', str(checkpoint),
                 '--task-manifest', str(tasks), '--out', str(out), '--max-seconds', '60',
                 '--max-examples', '16', '--component', '0:mlp']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['summary']['examples'] == 16
    assert set(result['per_task']) == {'boolean_hierarchy', 'associative_retrieval'}
    assert set(result['summary']['branch_removal']) == {'0:mlp'}
    assert result['billing']['status'] == 'complete'
    assert result['frozen_state']['unchanged']
    assert not result['benchmark_training_exposure_verified']
    assert json.loads((out / 'mechanism_evaluation.json').read_text()) == result
