"""Config validation reports storage mistakes without launching or writing."""
import json
from types import SimpleNamespace

import pytest
import yaml

from tnlm import cli
from tnlm.checkpoint_plan import checkpoint_storage_plan
from tnlm.checkpoint_storage import RESERVE_BYTES
from tnlm.config import Config, ModelConfig, TrainConfig


def config():
    return Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4,
                  d_value=4, d_mlp=12, vocab_size=16, padded_vocab=16,
                  attention_backend='reference'),
                  train=TrainConfig(global_prediction_tokens_per_update=100,
                                    rung_total_tokens=1000),
                  raw={'project': {'milestone_tokens': []}})


@pytest.mark.parametrize('headroom, status', [(-1, 'insufficient'), (0, 'ok'), (1, 'ok')])
def test_free_space_boundary_is_advisory_and_creates_nothing(tmp_path, monkeypatch, headroom, status):
    cfg = config()
    required = checkpoint_storage_plan(cfg)['required_additional_bytes']
    seen = []
    def disk_usage(path):
        seen.append(path)
        return SimpleNamespace(free=required + RESERVE_BYTES + headroom)
    monkeypatch.setattr(cli.shutil, 'disk_usage', disk_usage)
    destination = tmp_path / 'not-created' / 'run'
    report = cli.validate_config(cfg, device='cpu', run_dir=destination)
    check = report['checkpoint_disk_check']
    assert check['status'] == status
    assert check['required_additional_bytes'] == required
    assert check['reserve_bytes'] == RESERVE_BYTES
    assert check['free_bytes'] == required + RESERVE_BYTES + headroom
    assert check['destination'] == str(destination)
    assert check['filesystem_probe_path'] == str(tmp_path)
    assert seen == [tmp_path]
    assert bool(report['warnings']) == (status == 'insufficient')
    assert not destination.parent.exists()
    json.dumps(report, allow_nan=False)


def test_default_checks_current_directory_not_spend_ledger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config()
    cfg.raw['project']['spend_ledger'] = '/not/the/checkpoint/filesystem/spend.json'
    seen = []
    def disk_usage(path):
        seen.append(path)
        return SimpleNamespace(free=10**12)
    monkeypatch.setattr(cli.shutil, 'disk_usage', disk_usage)
    check = cli.validate_config(cfg, device='cpu')['checkpoint_disk_check']
    assert seen == [tmp_path]
    assert check['destination_source'] == 'current_working_directory'
    assert check['status'] == 'ok'
    assert not list(tmp_path.iterdir())


def test_space_probe_error_warns_without_failing_validation(tmp_path, monkeypatch):
    def disk_usage(path):
        raise PermissionError('filesystem probe denied')
    monkeypatch.setattr(cli.shutil, 'disk_usage', disk_usage)
    report = cli.validate_config(config(), device='cpu', run_dir=tmp_path / 'new')
    check = report['checkpoint_disk_check']
    assert check['status'] == 'unavailable'
    assert check['free_bytes'] is None
    assert 'filesystem probe denied' in report['warnings'][0]
    assert not list(tmp_path.iterdir())


def test_retention_policy_changes_disk_warning(tmp_path, monkeypatch):
    cfg = config()
    default_bytes = checkpoint_storage_plan(cfg)['required_additional_bytes']
    cfg.raw['project'].update(keep_recovery_checkpoints=0, save_initial_checkpoint=False)
    screening_bytes = checkpoint_storage_plan(cfg)['required_additional_bytes']
    assert screening_bytes < default_bytes
    monkeypatch.setattr(cli.shutil, 'disk_usage',
                        lambda path: SimpleNamespace(free=screening_bytes + RESERVE_BYTES))
    screening = cli.validate_config(cfg, device='cpu', run_dir=tmp_path)
    assert screening['checkpoint_disk_check']['status'] == 'ok'
    cfg.raw['project'].update(keep_recovery_checkpoints=2, save_initial_checkpoint=True)
    default = cli.validate_config(cfg, device='cpu', run_dir=tmp_path)
    assert default['checkpoint_disk_check']['status'] == 'insufficient'


def test_cli_run_dir_reports_warning_in_json_and_exits_successfully(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cli.full_config(config())))
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    destination = tmp_path / 'future' / 'run'
    assert cli.main(['validate_config', '--config', str(path), '--device', 'cpu',
                     '--run-dir', str(destination)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['checkpoint_disk_check']['destination'] == str(destination)
    assert report['checkpoint_disk_check']['status'] == 'insufficient'
    assert 'Adjust retention' in report['warnings'][0]
    assert not destination.parent.exists()
