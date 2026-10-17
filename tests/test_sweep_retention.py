"""Opt-in retention tests touch only fresh isolated tmp_path sweep artifacts."""
import copy
import json
from pathlib import Path

import pytest
import torch

from helpers import FixtureTokenizer
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import prepare_data
from tnlm.sweep import run_sweep
from tnlm.sweep_retention import intermediate_payload, retention_forecast


@pytest.fixture
def cfg(tmp_path):
    torch.set_num_threads(1)
    paths = {}
    for split, texts in [('train', ['abcdefghijklm', 'training words']), ('val', ['nopqrst', 'uvwxyz'])]:
        source = tmp_path / f'{split}.jsonl'
        source.write_text(''.join(json.dumps({'id': split + str(i), 'text': text}) + '\n' for i, text in enumerate(texts)))
        paths[split] = prepare_data(tmp_path / split, source_jsonl=source, revision='retention-fixture',
            tokenizer=FixtureTokenizer(), split_fractions={name: float(name == split)
            for name in ('train', 'val', 'test', 'calibration')})
    return Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                    d_mlp=12, padded_vocab=50257, attention_backend='reference'),
        train=TrainConfig(compile=False, bf16_compute=False, rung_total_tokens=31,
                          global_prediction_tokens_per_update=7),
        raw={'data': {'manifest': str(paths['train']), 'context': 4},
             'evaluation': {'validation_manifest': str(paths['val'])},
             'project': {'dry_run': False, 'gpu_hourly_rate_usd': 0.,
                         'spend_ledger': str(tmp_path / 'ledger.json'), 'milestone_tokens': [7, 14, 21],
                         'checkpoint_every_steps': 1, 'checkpoint_every_seconds': 1000000}})


def execute(cfg, out, *, prune=True, two=True):
    candidates = [{'name': 'a', 'overrides': {}}]
    if two:
        candidates.append({'name': 'b', 'overrides': {'train': {'lr': .001}}})
    return run_sweep(cfg, candidates, out, max_seconds=60, evaluation_max_seconds=60,
                     device='cpu', max_prediction_tokens=14, bootstrap_samples=5,
                     prune_completed_intermediates=prune)


def forbid_stages(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('completed charged/comparison stages must not run again')
    for name in ('train_run', 'evaluate_checkpoint', 'compare_evaluations', 'summarize_runs'):
        monkeypatch.setattr('tnlm.sweep.' + name, forbidden)


@pytest.mark.slow
def test_actual_two_candidate_default_vs_prune_retained_proofs_and_reentry(cfg, tmp_path, monkeypatch):
    default = execute(cfg, tmp_path / 'default', prune=False)
    out = tmp_path / 'pruned'
    import fcntl
    import tnlm.sweep as module
    original_unlink = module.unlink_payload
    lock_checks = []
    def locked_unlink(*args):
        with Path(cfg.raw['project']['spend_ledger'] + '.lock').open('a') as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_checks.append(True)
        return original_unlink(*args)
    monkeypatch.setattr(module, 'unlink_payload', locked_unlink)
    result = execute(cfg, out)
    assert lock_checks
    assert default['status'] == result['status'] == 'complete'
    assert list((tmp_path / 'default').glob('*/parent/milestone-*/state.pt'))
    for name in ('a', 'b'):
        for phase in ('parent', 'cooldown'):
            stage = out / name / phase
            assert (stage / 'initial/state.pt').is_file()
            assert (stage / 'final/state.pt').is_file()
            assert (stage / 'metrics.jsonl').is_file() and (stage / 'result.json').is_file()
            assert not list(stage.glob('milestone-*/state.pt'))
            assert not list(stage.glob('recovery-*/state.pt'))
        assert (out / name / 'evaluation/evaluation.json').is_file()
        operation = result['prune_operations'][name]
        assert operation['status'] == 'complete' and operation['files']
        assert operation['original_payload_bytes'] == sum(info['bytes'] for info in operation['files'].values())
        for relative, info in operation['files'].items():
            assert intermediate_payload(info['artifact']) and not (out / relative).exists()
            manifest = json.loads((out / relative).with_name('manifest.json').read_text())
            assert manifest['files']['state.pt'] == {'sha256': info['sha256'], 'bytes': info['bytes']}
        left = json.loads((tmp_path / 'default' / name / 'evaluation/evaluation.json').read_text())
        right = json.loads((out / name / 'evaluation/evaluation.json').read_text())
        assert left['ce'] == right['ce'] and left['model_sha256'] == right['model_sha256']
    ledger_before = Path(cfg.raw['project']['spend_ledger']).read_bytes()
    forbid_stages(monkeypatch)
    assert execute(cfg, out)['status'] == 'complete'
    assert Path(cfg.raw['project']['spend_ledger']).read_bytes() == ledger_before


def interrupt_after_one(cfg, out, monkeypatch):
    import tnlm.sweep as module
    original = module.unlink_payload
    count = []
    def interrupted(*args):
        count.append(True)
        if len(count) == 2:
            raise RuntimeError('injected interrupted prune')
        return original(*args)
    monkeypatch.setattr(module, 'unlink_payload', interrupted)
    with pytest.raises(RuntimeError, match='interrupted prune'):
        execute(cfg, out, two=False)
    monkeypatch.setattr(module, 'unlink_payload', original)
    journal = json.loads((out / 'sweep.json').read_text())
    operation = journal['prune_operations']['a']
    assert operation['status'] == 'planned'
    assert sum(not (out / path).exists() for path in operation['files']) == 1
    return journal


@pytest.mark.slow
def test_interrupted_prune_recovers_without_retraining(cfg, tmp_path, monkeypatch):
    out = tmp_path / 'interrupted'
    interrupt_after_one(cfg, out, monkeypatch)
    # Summary had not run yet; only charged stage callbacks are forbidden.
    def forbidden(*args, **kwargs):
        raise AssertionError('charged stage retried during prune recovery')
    monkeypatch.setattr('tnlm.sweep.train_run', forbidden)
    monkeypatch.setattr('tnlm.sweep.evaluate_checkpoint', forbidden)
    before = Path(cfg.raw['project']['spend_ledger']).read_bytes()
    result = execute(cfg, out, two=False)
    assert result['status'] == 'complete' and result['prune_operations']['a']['status'] == 'complete'
    assert Path(cfg.raw['project']['spend_ledger']).read_bytes() == before


@pytest.mark.parametrize('tamper', ['retained', 'surviving-slated'])
@pytest.mark.slow
def test_pending_prune_rejects_tamper_before_any_more_deletion(cfg, tmp_path, monkeypatch, tamper):
    out = tmp_path / 'tampered'
    journal = interrupt_after_one(cfg, out, monkeypatch)
    surviving = [out / path for path in journal['prune_operations']['a']['files'] if (out / path).exists()]
    target = out / 'a/parent/result.json' if tamper == 'retained' else surviving[0]
    with target.open('ab') as stream:
        stream.write(b'tampered')
    before = {str(path): path.read_bytes() for path in surviving}
    forbid_stages(monkeypatch)
    with pytest.raises(ValueError, match='hash mismatch'):
        execute(cfg, out, two=False)
    assert all(path.exists() and path.read_bytes() == before[str(path)] for path in surviving)


@pytest.mark.slow
def test_external_active_pin_blocks_deletion(cfg, tmp_path, monkeypatch):
    import tnlm.sweep as module
    original_unlink = module.unlink_payload
    calls = []
    def should_not_unlink(*args):
        calls.append(True)
        return original_unlink(*args)
    original_eval = module.evaluate_checkpoint
    def pin_after_evaluation(*args, **kwargs):
        result = original_eval(*args, **kwargs)
        ledger_path = Path(cfg.raw['project']['spend_ledger'])
        ledger = json.loads(ledger_path.read_text())
        out = Path(args[2]).parents[1]
        ledger['runs']['external-active'] = {'status': 'running', 'reserved_usd': 1.,
            'parent_checkpoint': str((out / 'a/parent/milestone-000000000027').resolve())}
        ledger_path.write_text(json.dumps(ledger))
        return result
    monkeypatch.setattr(module, 'evaluate_checkpoint', pin_after_evaluation)
    monkeypatch.setattr(module, 'unlink_payload', should_not_unlink)
    out = tmp_path / 'pinned'
    with pytest.raises(ValueError, match='pins'):
        execute(cfg, out, two=False)
    assert not calls
    assert list((out / 'a/parent').glob('milestone-*/state.pt'))


def test_prune_forecast_dry_plan_and_completed_stage_exclusion(cfg, tmp_path):
    cfg.raw['project']['dry_run'] = True
    default = execute(cfg, tmp_path / 'dry-default', prune=False)
    pruned = execute(cfg, tmp_path / 'dry-pruned')
    runs = pruned['identity']['trajectories']
    first = runs[0]['parent_storage_plan']['required_additional_bytes'] + runs[0]['cooldown_storage_plan']['required_additional_bytes']
    retained = 2 * (runs[0]['parent_storage_plan']['per_checkpoint_estimate_bytes'] + runs[0]['cooldown_storage_plan']['per_checkpoint_estimate_bytes'])
    second = runs[1]['parent_storage_plan']['required_additional_bytes'] + runs[1]['cooldown_storage_plan']['required_additional_bytes']
    assert pruned['required_additional_bytes'] == max(first, retained + second)
    assert pruned['required_additional_bytes'] < default['required_additional_bytes']
    forecast = retention_forecast(runs, ['a/parent', 'a/cooldown'])
    assert forecast['required_additional_bytes'] == second
    assert not (tmp_path / 'dry-pruned').exists()


def test_payload_selector_preserves_endpoints_and_nested_or_wrong_files():
    assert intermediate_payload('milestone-000000000027/state.pt')
    assert intermediate_payload('recovery-000000000001/state.pt')
    for name in ('initial/state.pt', 'final/state.pt', 'milestone-1/manifest.json',
                 'nested/milestone-1/state.pt', '../milestone-1/state.pt', '/milestone-1/state.pt'):
        assert not intermediate_payload(name)


@pytest.mark.parametrize('redirect', ['parent', 'candidate'])
@pytest.mark.slow
def test_pending_prune_rejects_ancestor_symlink_without_outside_deletion(cfg, tmp_path, monkeypatch, redirect):
    out = tmp_path / 'redirected'
    journal = interrupt_after_one(cfg, out, monkeypatch)
    target = out / 'a' / 'parent' if redirect == 'parent' else out / 'a'
    outside = tmp_path / 'outside-copy'
    target.rename(outside)
    target.symlink_to(outside, target_is_directory=True)
    before = {str(path): path.read_bytes() for path in outside.rglob('*') if path.is_file()}
    forbid_stages(monkeypatch)
    with pytest.raises(ValueError, match='symlink'):
        execute(cfg, out, two=False)
    assert {str(path): path.read_bytes() for path in outside.rglob('*') if path.is_file()} == before


@pytest.mark.slow
def test_screening_sweep_skips_initial_and_recovery_preserves_cooldown(cfg, tmp_path):
    cfg.raw['project'].update(keep_recovery_checkpoints=0, save_initial_checkpoint=False,
                              milestone_tokens=[])
    out = tmp_path / 'screening'
    result = execute(cfg, out, two=False)
    assert result['status'] == 'complete'
    run = result['identity']['trajectories'][0]
    parent, cooldown = run['parent_storage_plan'], run['cooldown_storage_plan']
    assert parent['checkpoint_count'] == 2  # Required branch + final.
    assert cooldown['checkpoint_count'] == 1
    forecast = result['retention_forecast']
    assert forecast['new_final_retained_bytes'] == (parent['per_checkpoint_estimate_bytes'] +
                                                     cooldown['per_checkpoint_estimate_bytes'])
    for phase in ('parent', 'cooldown'):
        assert not (out/'a'/phase/'initial').exists()
        assert not list((out/'a'/phase).glob('recovery-*'))
        assert (out/'a'/phase/'final/state.pt').is_file()
    assert (out/'a/evaluation/evaluation.json').is_file()
