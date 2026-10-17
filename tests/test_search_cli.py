import json
import yaml

from tnlm.cli import main


def test_named_grid_cli_is_read_only(config, tmp_path, capsys):
    base = tmp_path / 'base.yaml'
    base.write_text(yaml.safe_dump(config.to_dict()))
    candidates = tmp_path / 'candidates.yaml'
    candidates.write_text(yaml.safe_dump([
        {'name': 'low', 'overrides': {'train': {'lr': 0.0001}}},
        {'name': 'high', 'overrides': {'train': {'lr': 0.0003}}},
    ]))
    before = set(tmp_path.rglob('*'))
    assert main(['plan_search', '--config', str(base), '--candidates', str(candidates)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [run['name'] for run in result['runs']] == ['low', 'high']
    assert set(tmp_path.rglob('*')) == before


def test_explicit_cooldown_endpoint_descriptor():
    from tnlm.evaluate import _training_descriptor
    payload = {'config': {'train': {'rung_total_tokens': 50,
                                   'global_prediction_tokens_per_update': 8}},
               'progress': {'tokens': 50, 'cooldown_start_tokens': 40,
                            'scheduler_step': 2, 'cooldown_end_tokens': 50},
               'extra': {'cooldown_steps': 2}}
    result = _training_descriptor(payload)
    assert result['segment_target_prediction_tokens'] == 50
    assert result['completion_status'] == 'complete'
    payload['progress']['tokens'] = 48
    payload['progress']['scheduler_step'] = 1
    assert _training_descriptor(payload)['completion_status'] == 'partial'
    for bad in (True, 48, 57, None):
        payload['progress']['cooldown_end_tokens'] = bad
        assert _training_descriptor(payload)['completion_status'] == 'unknown'


def test_cli_passes_explicit_endpoint(config, tmp_path, capsys, monkeypatch):
    import tnlm.cli as cli
    path = tmp_path / 'base.yaml'
    path.write_text(yaml.safe_dump(config.to_dict()))
    observed = {}
    def fake_run(cfg, run_dir, **kwargs):
        observed.update(kwargs)
        return {'endpoint': kwargs['cooldown_end_tokens']}
    monkeypatch.setattr(cli, 'train_run', fake_run)
    assert main(['cooldown', '--config', str(path), '--run-dir', str(tmp_path / 'branch'),
                 '--resume', str(tmp_path / 'parent'), '--max-steps', '2', '--max-seconds', '60',
                 '--cooldown-steps', '2', '--cooldown-end-tokens', '50', '--device', 'cpu']) == 0
    assert json.loads(capsys.readouterr().out)['endpoint'] == 50
    assert observed['cooldown_steps'] == 2
