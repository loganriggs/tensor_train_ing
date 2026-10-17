"""Training admission rejects forecast exhaustion before an update or checkpoint."""
import errno
import importlib
import json

import pytest
import torch


def test_forecast_rejection_precedes_optimizer_and_initial_checkpoint(config, tmp_path, monkeypatch):
    training = importlib.import_module('tnlm.train')
    def no_space(directory, required):
        assert required > 0
        raise OSError(errno.ENOSPC, 'forecast exceeds capacity')
    def forbidden(*args, **kwargs):
        raise AssertionError('optimizer executed before storage admission')
    monkeypatch.setattr(training, 'require_checkpoint_space', no_space)
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    run = tmp_path / 'rejected'
    with pytest.raises(OSError, match='forecast exceeds capacity'):
        training.train_run(config, run, max_steps=1, max_seconds=60, device='cpu')
    assert not (run / 'initial').exists()
    assert not (run / 'metrics.jsonl').exists()
    assert not run.exists()
    assert not (tmp_path / 'spend.json').exists()


@pytest.mark.parametrize('optimizer', ['adamw', 'muon'])
def test_forecast_covers_actual_tied_checkpoint_bytes(config, tmp_path, optimizer):
    from tnlm.train import train_run
    config.train.optimizer = optimizer
    config.model.tie_embeddings = True
    config.raw['project']['checkpoint_every_steps'] = 1
    run = tmp_path / 'measured'
    result = train_run(config, run, max_steps=3, max_seconds=60, device='cpu')
    plan = result['checkpoint_storage_plan']
    saved = [p for p in run.iterdir() if p.is_dir() and (p / 'manifest.json').exists()]
    assert len(saved) <= plan['checkpoint_count']
    for directory in saved:
        actual_bytes = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
        assert actual_bytes <= plan['per_checkpoint_estimate_bytes']
    report = json.loads((run / 'checkpoint_storage_plan.json').read_text())
    assert report['plan'] == plan
    assert report['admission']['required_bytes'] == plan['required_additional_bytes']


@pytest.mark.parametrize('milestones, expected', [([], {90}), ([7, 14, 14, 90], {7, 14, 90})])
def test_custom_milestones_share_forecast_and_trainer_schedule(config, tmp_path, milestones, expected):
    from tnlm.train import train_run
    config.raw['project']['milestone_tokens'] = milestones
    config.train.global_prediction_tokens_per_update = 50
    run = tmp_path / 'custom'
    result = train_run(config, run, max_steps=2, max_seconds=60, device='cpu')
    actual = {int(p.name.removeprefix('milestone-')) for p in run.glob('milestone-*')}
    assert actual == expected
    assert result['checkpoint_storage_plan']['count_breakdown']['milestones'] == len(expected)
    assert (run / 'initial').is_dir() and (run / 'final').is_dir()


def test_search_plan_exposes_configured_retention_without_launch(config):
    from tnlm.train import plan_search
    config.raw["project"]["milestone_tokens"] = []
    result = plan_search(config)
    plan = result["checkpoint_storage_plan"]
    assert plan["milestone_policy"] == {"mode": "explicit", "tokens": []}
    assert plan["count_breakdown"]["milestones"] == 1  # mandatory90%branch
    assert plan["required_additional_bytes"] > 0
