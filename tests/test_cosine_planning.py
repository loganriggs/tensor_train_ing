import json
import yaml
import pytest

from tnlm.checkpoint_plan import checkpoint_storage_plan, pending_milestone_tokens
from tnlm.cli import main
from tnlm.sweep import run_sweep


def test_cosine_retains_ordinary_milestones_without_cooldown_branch(config):
    config.train.schedule = 'cosine'
    config.train.rung_total_tokens = 713
    config.raw['project']['milestone_tokens'] = [7]
    assert list(pending_milestone_tokens(config, start_tokens=0, token_limit=713,
                                        include_branch=True)) == [7]
    for distributed in (False, True):
        plan = checkpoint_storage_plan(config, distributed=distributed,
                                       distributed_world_size=2 if distributed else 1)
        assert plan['count_breakdown']['separate_90_percent_branch'] == 0
        assert plan['count_breakdown']['milestones'] == 1


def test_cosine_sweep_requires_explicit_no_cooldown_before_writes(config, tmp_path):
    config.train.schedule = 'cosine'
    config.train.rung_total_tokens = 713
    with pytest.raises(ValueError, match='no-cooldown'):
        run_sweep(config, [{'name':'cosine', 'overrides':{}}], tmp_path / 'bad',
                  max_seconds=60, evaluation_max_seconds=60, device='cpu')
    assert not (tmp_path / 'bad').exists()
    config.raw['project']['dry_run'] = True
    plan = run_sweep(config, [{'name':'cosine', 'overrides':{}}], tmp_path / 'dry',
                     max_seconds=60, evaluation_max_seconds=60, device='cpu', cooldown=False)
    assert plan['identity']['trajectories'][0]['cooldown_steps'] is None
    assert plan['identity']['trajectories'][0]['config']['train']['schedule'] == 'cosine'
    assert not (tmp_path / 'dry').exists()


def test_cosine_cli_plan_declares_schedule(config, tmp_path, capsys):
    config.train.schedule = 'cosine'
    config.train.rung_total_tokens = 713
    path = tmp_path / 'cosine.yaml'
    path.write_text(yaml.safe_dump(config.to_dict()))
    assert main(['plan_search', '--config', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['schedule'] == 'cosine'
    assert result['warmup_steps'] == 100
    assert result['planned_optimizer_steps'] == 102


def test_cosine_completion_needs_consistent_token_and_scheduler_endpoints():
    from tnlm.evaluate import _training_descriptor
    payload = {'config': {'train': {'schedule': 'cosine', 'rung_total_tokens':713,
                                    'global_prediction_tokens_per_update':7}},
               'progress': {'tokens':713, 'scheduler_step':102}, 'extra':{'cooldown_steps':None}}
    assert _training_descriptor(payload)['completion_status'] == 'complete'
    for step in (None, 101, 103):
        payload['progress']['scheduler_step'] = step
        assert _training_descriptor(payload)['completion_status'] == 'unknown'
    payload['progress'].update(tokens=700, scheduler_step=100)
    assert _training_descriptor(payload)['completion_status'] == 'partial'
