import json
import yaml

from tnlm.cli import main


def test_cli_sweep_forwards_frozen_controls(config, tmp_path, monkeypatch, capsys):
    import tnlm.sweep as sweep
    observed = {}
    def fake(cfg, candidates, out_dir, **kwargs):
        observed.update(kwargs)
        observed['candidates'] = candidates
        return {'dry_run': True}
    monkeypatch.setattr(sweep, 'run_sweep', fake)
    config_path = tmp_path / 'base.yaml'
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    candidates = tmp_path / 'grid.yaml'
    candidates.write_text('- name: small\n  overrides: {}\n')
    assert main(['sweep', '--config', str(config_path), '--candidates', str(candidates),
                 '--out', str(tmp_path / 'out'), '--max-seconds', '40',
                 '--evaluation-max-seconds', '20', '--max-prediction-tokens', '32',
                 '--bootstrap-samples', '0', '--device', 'cpu', '--no-cooldown',
                 '--prune-completed-intermediates']) == 0
    assert json.loads(capsys.readouterr().out)['dry_run']
    assert observed == {'max_seconds':40., 'evaluation_max_seconds':20.,
                        'max_prediction_tokens':32, 'bootstrap_samples':0,
                        'device':'cpu', 'cooldown':False, 'reference_candidate':None,
                        'prune_completed_intermediates':True,
                        'candidates':[{'name':'small','overrides':{}}]}
    assert not (tmp_path / 'out').exists()


def test_cli_sweep_dryrun_resolves_cooldown_without_writes(config, tmp_path, capsys):
    config.raw['project']['dry_run'] = True
    path = tmp_path / 'dry.yaml'
    path.write_text(yaml.safe_dump(config.to_dict()))
    candidates = tmp_path / 'grid.yaml'
    candidates.write_text('- name: first\n  overrides: {}\n')
    before = set(tmp_path.rglob('*'))
    assert main(['sweep', '--config', str(path), '--candidates', str(candidates),
                 '--out', str(tmp_path / 'out'), '--max-seconds', '40',
                 '--evaluation-max-seconds', '20', '--device', 'cpu']) == 0
    plan = json.loads(capsys.readouterr().out)
    trajectory = plan['identity']['trajectories'][0]
    assert trajectory['branch_tokens'] == 91
    assert trajectory['cooldown_steps'] == 2
    assert trajectory['cooldown_end_tokens'] == 100
    assert set(tmp_path.rglob('*')) == before
