import copy
import json
from pathlib import Path

import pytest
import torch

from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import prepare_data
from tnlm.sweep import run_sweep


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(1)
    train = tmp_path / 'train.jsonl'
    train.write_text(''.join(json.dumps({'id': f'train{i}', 'text': f'Training document {i} has useful words.'}) + '\n' for i in range(10)))
    training = prepare_data(tmp_path / 'training-data', source_jsonl=train, revision='offline-train',
        split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    val = tmp_path / 'val.jsonl'
    val.write_text(''.join(json.dumps({'id': f'val{i}', 'text': f'Unique validation paragraph number {i}.'}) + '\n' for i in range(2)))
    validation = prepare_data(tmp_path / 'validation-data', source_jsonl=val, revision='offline-val',
        split_fractions={'train': 0., 'val': 1., 'test': 0., 'calibration': 0.})
    return Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4, d_mlp=12,
                    padded_vocab=50257, attention_backend='reference'),
        train=TrainConfig(compile=False, bf16_compute=False, global_prediction_tokens_per_update=7, rung_total_tokens=31),
        raw={'data': {'manifest': str(training), 'context': 7, 'micro_batch_size': 1},
             'evaluation': {'validation_manifest': str(validation)},
             'project': {'dry_run': False, 'gpu_hourly_rate_usd': 0., 'spend_ledger': str(tmp_path / 'ledger.json'),
                         'milestone_tokens': []}})


def invoke(config, out, **kwargs):
    return run_sweep(config, [{'name': 'tiny', 'overrides': {}}], out, max_seconds=60,
                     evaluation_max_seconds=60, device='cpu', max_prediction_tokens=14, bootstrap_samples=5, **kwargs)


def test_real_cpu_exact_endpoint_billing_reuse_and_corruption(config, tmp_path, monkeypatch):
    out = tmp_path / 'sweep'
    def execute():
        return run_sweep(config, [{'name': 'tiny', 'overrides': {}},
            {'name': 'wider', 'overrides': {'model': {'d_model': 16, 'heads': 2, 'd_mlp': 24}}}],
            out, max_seconds=60, evaluation_max_seconds=60, device='cpu',
            max_prediction_tokens=14, bootstrap_samples=5)
    result = execute()
    assert result['status'] == 'complete'
    assert len(result['stages']) == 8
    parent = result['stages']['tiny/parent']['result']
    assert parent['checkpoint']['tokens'] == 31 and parent['branch']['tokens'] == 28
    cooled = result['stages']['tiny/cooldown']['result']
    assert cooled['tokens'] == 31
    payload = torch.load(Path(cooled['path']) / 'state.pt', weights_only=False, mmap=True)
    assert payload['progress']['scheduler_step'] == 1
    assert payload['extra']['cooldown_end_tokens'] == 31
    ledger = json.loads((tmp_path / 'ledger.json').read_text())
    assert len(ledger['runs']) == 6
    assert all(entry['status'] == 'complete' and entry['reserved_usd'] == 0 for entry in ledger['runs'].values())
    assert (out / 'summary' / 'validation.json').exists()
    from tnlm.evaluate import compare_evaluations
    authoritative = compare_evaluations(out / 'tiny' / 'evaluation' / 'evaluation.json',
        out / 'wider' / 'evaluation' / 'evaluation.json', tmp_path / 'independent-comparison',
        bootstrap_samples=5, seed=0)
    summary = json.loads((out / 'summary' / 'validation.json').read_text())
    comparison = summary['comparisons'][0]
    assert summary['reference_candidate'] == 'tiny' and not summary['automatic_ranking_eligible']
    assert comparison['delta_ce_right_minus_left'] == authoritative['delta_ce_right_minus_left']
    assert comparison['delta_ce_95_ci'] == authoritative['uncertainty']['delta_ce_95_ci']
    assert comparison['training_comparison'] == authoritative['training_comparison']
    assert Path(comparison['path']).is_file()
    def forbidden(*args, **kwargs):
        raise AssertionError('completed stage must not execute')
    monkeypatch.setattr('tnlm.sweep.train_run', forbidden)
    monkeypatch.setattr('tnlm.sweep.evaluate_checkpoint', forbidden)
    monkeypatch.setattr('tnlm.sweep.summarize_runs', forbidden)
    monkeypatch.setattr('tnlm.sweep.compare_evaluations', forbidden)
    assert execute()['status'] == 'complete'
    # A terminal journal cannot hide a nonterminal billing entry.
    key = str((out / 'tiny' / 'parent').resolve())
    ledger['runs'][key]['status'] = 'failed'
    (tmp_path / 'ledger.json').write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='billing'):
        execute()
    ledger['runs'][key]['status'] = 'complete'
    (tmp_path / 'ledger.json').write_text(json.dumps(ledger))
    with (out / 'tiny' / 'evaluation' / 'evaluation.json').open('a') as stream:
        stream.write('corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        execute()


def test_failure_stops_and_never_retries_failed_stage(config, tmp_path, monkeypatch):
    calls = []
    def crashed(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('injected crash')
    monkeypatch.setattr('tnlm.sweep.train_run', crashed)
    out = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='injected crash'):
        invoke(config, out)
    assert json.loads((out / 'sweep.json').read_text())['stages']['tiny/parent']['status'] == 'failed'
    with pytest.raises(RuntimeError, match='explicit recovery'):
        invoke(config, out)
    assert calls == [1]


def test_dry_run_complete_plan_no_writes_and_frozen_controls(config, tmp_path):
    config.raw['project']['dry_run'] = True
    before = copy.deepcopy(config.to_dict())
    out = tmp_path / 'dry'
    result = invoke(config, out)
    assert result['dry_run'] and not out.exists()
    trajectory = result['identity']['trajectories'][0]
    assert trajectory['cooldown_steps'] == 1 and trajectory['cooldown_end_tokens'] == 31
    assert trajectory['data']['validation']['sha256']
    assert result['identity']['controls']['max_prediction_tokens'] == 14
    assert result['runtime_cap_based_usd'] == 0
    assert config.to_dict() == before
    assert not (tmp_path / 'ledger.json').exists()


@pytest.mark.parametrize('name', ['summary', 'sweep.json', 'sweep.json.tmp', 'comparisons'])
def test_reserved_name_before_any_output(config, tmp_path, name):
    with pytest.raises(ValueError, match='reserved'):
        run_sweep(config, [{'name': name, 'overrides': {}}], tmp_path / 'bad',
                  max_seconds=60, evaluation_max_seconds=60, device='cpu')
    assert not (tmp_path / 'bad').exists() and not (tmp_path / 'ledger.json').exists()


def test_invalid_controls_mixed_dryrun_and_tiny_branch_before_output(config, tmp_path):
    for kwargs in ({'max_seconds': float('nan')}, {'max_prediction_tokens': True}, {'bootstrap_samples': -1}):
        args = dict(max_seconds=60, evaluation_max_seconds=60, max_prediction_tokens=14)
        args.update(kwargs)
        with pytest.raises(ValueError):
            run_sweep(config, [{'name': 'tiny', 'overrides': {}}], tmp_path / 'bad', device='cpu', **args)
    with pytest.raises(ValueError, match='dry_run'):
        run_sweep(config, [{'name': 'a', 'overrides': {}}, {'name': 'b', 'overrides': {'project': {'dry_run': True}}}],
                  tmp_path / 'bad', max_seconds=60, evaluation_max_seconds=60, device='cpu')
    config.train.rung_total_tokens = 7
    with pytest.raises(ValueError, match='proper'):
        invoke(config, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_unjournaled_outputs_never_overwritten(config, tmp_path):
    out = tmp_path / 'existing'
    out.mkdir()
    (out / 'precious').write_text('keep')
    with pytest.raises(FileExistsError):
        invoke(config, out)
    assert (out / 'precious').read_text() == 'keep'
    assert not (tmp_path / 'ledger.json').exists()


def test_charged_partial_endpoint_stops_before_eval_and_never_retries(config, tmp_path, monkeypatch):
    from tnlm.train import train_run
    calls = []
    def partial(*args, **kwargs):
        calls.append(1)
        kwargs['max_steps'] = 1
        return train_run(*args, **kwargs)
    monkeypatch.setattr('tnlm.sweep.train_run', partial)
    out = tmp_path / 'partial'
    with pytest.raises(ValueError, match='endpoint'):
        invoke(config, out)
    ledger = json.loads((tmp_path / 'ledger.json').read_text())
    assert len(ledger['runs']) == 1
    assert next(iter(ledger['runs'].values()))['status'] == 'complete'
    assert not (out / 'tiny' / 'evaluation').exists()
    with pytest.raises(RuntimeError, match='explicit recovery'):
        invoke(config, out)
    assert calls == [1]


def test_running_journal_and_changed_identity_fail_closed(config, tmp_path, monkeypatch):
    out = tmp_path / 'interrupted'
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt('interrupted')
    monkeypatch.setattr('tnlm.sweep.train_run', interrupted)
    with pytest.raises(KeyboardInterrupt):
        invoke(config, out)
    path = out / 'sweep.json'
    journal = json.loads(path.read_text())
    journal['stages']['tiny/parent'] = {'status': 'running'}
    path.write_text(json.dumps(journal))
    with pytest.raises(RuntimeError, match='explicit recovery'):
        invoke(config, out)
    config.train.lr *= 2
    with pytest.raises(ValueError, match='identity mismatch'):
        invoke(config, out)


def test_lock_contention_before_any_charge(config, tmp_path):
    import fcntl
    out = tmp_path / 'locked'
    out.mkdir()
    with (out / '.sweep.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='holds the output lock'):
            invoke(config, out)
    assert not (tmp_path / 'ledger.json').exists()


def test_disk_rejection_before_any_charge(config, tmp_path, monkeypatch):
    import errno
    def reject(*args):
        raise OSError(errno.ENOSPC, 'injected insufficient disk')
    monkeypatch.setattr('tnlm.sweep.require_checkpoint_space', reject)
    out = tmp_path / 'full'
    with pytest.raises(OSError) as error:
        invoke(config, out)
    assert error.value.errno == errno.ENOSPC
    assert not (tmp_path / 'ledger.json').exists()
    journal = json.loads((out / 'sweep.json').read_text())
    assert journal['stages'] == {}


def test_runtime_source_drift_before_stage_stops_before_charge(config, tmp_path, monkeypatch):
    import tnlm.sweep as sweep
    calls = []
    def changed_sources():
        calls.append(1)
        return {'model.py': 'original' if len(calls) == 1 else 'changed'}
    monkeypatch.setattr(sweep, '_sources', changed_sources)
    with pytest.raises(ValueError, match='runtime sources changed'):
        invoke(config, tmp_path / 'changed-source')
    assert not (tmp_path / 'ledger.json').exists()


def test_missing_reference_rejected_before_writes(config, tmp_path):
    with pytest.raises(ValueError, match='reference_candidate'):
        invoke(config, tmp_path / 'invalid-reference', reference_candidate='missing')
    assert not (tmp_path / 'invalid-reference').exists()
    assert not (tmp_path / 'ledger.json').exists()


def test_explicit_reference_frozen_and_single_candidate_no_comparisons(config, tmp_path, monkeypatch):
    config.raw['project']['dry_run'] = True
    plan = invoke(config, tmp_path / 'dry-reference', reference_candidate='tiny')
    assert plan['identity']['controls']['reference_candidate'] == 'tiny'
    assert plan['identity']['controls']['comparison_seed'] == 0
    assert plan['comparison_pairs'] == []
