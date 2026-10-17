"""Small CPU integration checks: data replay, optimizer replay and norm lifecycle."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import PackedSampler, prepare_data, manifest_hash
from tnlm.checkpoint import load_checkpoint
from tnlm.model import TensorLM
from tnlm.normalization import lagged_modules, spectral_step
from tnlm.train import SpendLedger, train_run, make_optimizer, optimizer_groups, seed_all
from tnlm.cli import validate_config, main
from helpers import FixtureTokenizer, state


@pytest.mark.parametrize('normalization', ['pre_rms', 'branch_rms', 'lagged', 'spectral_rms'])
def test_interrupted_training_exact_replay(config, tmp_path, normalization):
    config.model.normalization = normalization
    if normalization == 'lagged':
        config.model.final_normalization = 'lagged'
    whole = train_run(config, tmp_path / 'whole', max_steps=3, max_seconds=60, device='cpu')
    first = train_run(config, tmp_path / 'first', max_steps=1, max_seconds=60, device='cpu')
    resumed = train_run(config, tmp_path / 'resumed', max_steps=2, max_seconds=60,
                        device='cpu', resume=first['checkpoint'])
    expected, actual = state(tmp_path / 'whole' / 'final'), state(tmp_path / 'resumed' / 'final')
    assert expected['progress'] == actual['progress']
    exposure = expected['progress']['optimizer_exposure']
    assert exposure['complete_history'] and exposure['observed_steps'] == 3
    assert [row['optimizer_exposure']['observed_steps'] for row in whole['history']] == [1, 2, 3]
    for index, group in enumerate(exposure['groups']):
        assert group['sum_lr'] == pytest.approx(
            sum(row['optimizer_group_lrs'][index] for row in whole['history']), rel=1e-15, abs=0)
    assert whole['history'][1]['batch_hashes'] == resumed['history'][0]['batch_hashes']
    assert whole['history'][2]['batch_hashes'] == resumed['history'][1]['batch_hashes']
    for key in expected['model']:
        assert torch.equal(expected['model'][key], actual['model'][key]), key
    restored = TensorLM(config.model)
    restored.load_state_dict(actual['model'])
    restored.eval()
    probe = torch.tensor([[1, 2, 3, 4]])
    docs = torch.zeros_like(probe)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe, docs)[:, :3], restored(probe[:, :3], docs[:, :3]), rtol=1e-5, atol=1e-6)
    assert actual['extra']['operator_audit']['class'] == ('P' if normalization == 'lagged' else 'N')
    assert resumed['compile_enabled'] is False
    if normalization == 'lagged':
        assert actual['model']['final_norm.calibrated']
        assert actual['progress']['calibration']['tokens'] == 8
        assert expected['progress']['calibration'] == actual['progress']['calibration']
    ledger = json.loads((tmp_path / 'spend.json').read_text())
    assert all(row['status'] == 'complete' for row in ledger['runs'].values())


def test_accumulation_matches_single_normalized_objective(config, tmp_path):
    result = train_run(config, tmp_path / 'train', max_steps=1, max_seconds=60, device='cpu')
    model = TensorLM(config.model)
    optimizer = make_optimizer(model, config.train)
    sampler = PackedSampler(config.raw['data']['manifest'], context=4, seed=config.train.seed)
    load_checkpoint(tmp_path / 'train' / 'initial', model, optimizer, sampler)
    optimizer.zero_grad()
    spectral_step(model)
    objective = 0
    count = 0
    while count < 7:
        idx, docs, targets = sampler.next_batch()
        valid = (targets.flatten() != -100).nonzero().flatten()
        targets.flatten()[valid[7 - count:]] = -100
        _, loss, n = model(idx, docs, targets, reduction='sum')
        objective = objective + loss
        count += int(n)
    (objective / count).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
    for group in optimizer.param_groups:
        group['lr'] = config.train.lr / 100
    optimizer.step()
    actual = state(tmp_path / 'train' / 'final')['model']
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, actual[key], rtol=1e-5, atol=1e-7)
    assert result['progress']['tokens'] == 7


def test_cooldown_forks_and_resumes_schedule(config, tmp_path):
    initial = train_run(config, tmp_path / 'source', max_steps=2, max_seconds=60, device='cpu')
    whole = train_run(config, tmp_path / 'cool-whole', resume=initial['checkpoint'],
                      cooldown_steps=3, max_steps=3, max_seconds=60, device='cpu')
    first = train_run(config, tmp_path / 'cool-first', resume=initial['checkpoint'],
                      cooldown_steps=3, max_steps=1, max_seconds=60, device='cpu')
    rest = train_run(config, tmp_path / 'cool-rest', resume=first['checkpoint'],
                     cooldown_steps=3, max_steps=2, max_seconds=60, device='cpu')
    assert [h['lr'] for h in whole['history']] == [h['lr'] for h in first['history'] + rest['history']]
    assert whole['history'][0]['lr'] > whole['history'][-1]['lr']
    assert whole['progress'] == rest['progress']
    a, b = state(tmp_path / 'cool-whole' / 'final'), state(tmp_path / 'cool-rest' / 'final')
    assert all(torch.equal(a['model'][key], b['model'][key]) for key in a['model'])


def test_optimizer_rule_and_dry_run(config, tmp_path):
    model = TensorLM(config.model)
    groups = optimizer_groups(model, .1)
    decay = {id(p) for p in groups[0]['params']}
    assert id(model.blocks[0].mlp.l.weight) in decay
    assert id(model.blocks[0].mlp.r.weight) in decay
    assert id(model.embed.weight) not in decay and id(model.unembed.weight) not in decay
    assert id(model.final_norm.weight) not in decay
    config.raw['project']['dry_run'] = True
    assert train_run(config, tmp_path / 'dry', max_steps=1, max_seconds=1)['dry_run']
    assert not (tmp_path / 'dry').exists()
    assert not (tmp_path / 'spend.json').exists()
    result = validate_config(config)
    assert result['parameters'] == model.param_counts()


def test_spend_ledger_caps_and_failure(config, tmp_path, monkeypatch):
    ledger = SpendLedger(tmp_path / 'budget.json', total_budget=10, reserve=2)
    ledger.begin('one', 4, 3600)
    ledger.begin('two', 4, 3600)
    with pytest.raises(ValueError, match='budget exhausted'):
        ledger.begin('three', 1, 3600)
    with pytest.raises(ValueError, match='duplicate'):
        ledger.begin('one', 0, 1)
    ledger.finish('one', 1800, 'failed')
    ledger.begin('three', 1, 3600)
    with pytest.raises(ValueError, match='run budget'):
        ledger.begin('four', 1, 3600, run_budget=.5)
    def fail(*args):
        raise RuntimeError('synthetic initialization failure')
    monkeypatch.setattr('tnlm.train.TensorLM', fail)
    with pytest.raises(RuntimeError, match='initialization failure'):
        train_run(config, tmp_path / 'failed', max_steps=1, max_seconds=60, device='cpu')
    entry = next(iter(json.loads((tmp_path / 'spend.json').read_text())['runs'].values()))
    assert entry['status'] == 'failed' and entry['reserved_usd'] == 0


def test_promotion_preserves_schedule_and_rejects_model_changes(config, tmp_path):
    first = train_run(config, tmp_path / 'promote-source', max_steps=1, max_seconds=60, device='cpu')
    before = state(tmp_path / 'promote-source' / 'final')
    config.train.rung_total_tokens = 1_000_000
    promoted = train_run(config, tmp_path / 'promoted', max_steps=1, max_seconds=60,
                         device='cpu', resume=first['checkpoint'])
    assert promoted['warmup_steps'] == before['progress']['warmup_steps'] == 100
    assert promoted['history'][0]['lr'] == config.train.lr * .02
    config.model.residual_alpha = .5
    with pytest.raises(ValueError, match='resume config mismatch'):
        train_run(config, tmp_path / 'bad-promotion', max_steps=1, max_seconds=60,
                  device='cpu', resume=first['checkpoint'])


def test_reject_silent_cooldown_schedule_change(config, tmp_path):
    first = train_run(config, tmp_path / 'constant', max_steps=1, max_seconds=60, device='cpu')
    cooldown = train_run(config, tmp_path / 'cooldown', max_steps=1, max_seconds=60,
                         device='cpu', resume=first['checkpoint'], cooldown_steps=3)
    with pytest.raises(ValueError, match='requires explicit matching cooldown_steps'):
        train_run(config, tmp_path / 'wrong-mode', max_steps=1, max_seconds=60,
                  device='cpu', resume=cooldown['checkpoint'])


def test_cli_plan_and_train_dryrun(config, tmp_path, capsys):
    import yaml
    cfg = config.to_dict()
    cfg['project']['dry_run'] = True
    path = tmp_path / 'tiny.yaml'
    path.write_text(yaml.safe_dump(cfg))
    assert main(['validate_config', '--config', str(path)]) == 0
    assert json.loads(capsys.readouterr().out)['operator_audit']['class'] == 'N'
    assert main(['train', '--config', str(path), '--run-dir', str(tmp_path / 'cli-dry'),
                 '--max-steps', '1', '--max-seconds', '1']) == 0
    assert json.loads(capsys.readouterr().out)['dry_run'] is True
    assert not (tmp_path / 'cli-dry').exists()


def test_reject_padded_targets_before_forward(config, tmp_path, monkeypatch):
    original = PackedSampler.next_batch
    def invalid_batch(self):
        idx, docs, targets = original(self)
        docs[0, 0] = -1
        targets[0, 0] = 7
        return idx, docs, targets
    monkeypatch.setattr(PackedSampler, 'next_batch', invalid_batch)
    with pytest.raises(ValueError, match='padded-token targets'):
        train_run(config, tmp_path / 'bad-padding', max_steps=1, max_seconds=60, device='cpu')
    assert (tmp_path / 'bad-padding' / 'initial' / 'manifest.json').is_file()
    assert not (tmp_path / 'bad-padding' / 'final').exists()


def test_exact_token_cap_masks_partial_final_update(config, tmp_path):
    config.train.rung_total_tokens = 10
    result = train_run(config, tmp_path / 'partial', max_steps=10, max_seconds=60, device='cpu')
    assert result['progress']['tokens'] == 10
    assert result['progress']['steps'] == 2
    assert result['stop_reason'] == 'token_limit'


def test_spectral_meta_counts_without_allocating_weights(config):
    config.model.normalization = 'spectral_rms'
    report = validate_config(config)
    actual = TensorLM(config.model)
    assert report['parameters'] == actual.param_counts()
    assert report['operator_audit'] == actual.operator_audit()


def test_long_trajectories_bound_history_but_keep_all_metrics(config, tmp_path):
    config.raw['project']['history_limit'] = 2
    result = train_run(config, tmp_path / 'bounded-history', max_steps=3, max_seconds=60, device='cpu')
    assert len(result['history']) == 2
    assert [row['step'] for row in result['history']] == [2, 3]
    rows = [json.loads(line) for line in (tmp_path / 'bounded-history' / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1, 2, 3]
    assert sum(row['prediction_tokens'] for row in rows) == result['progress']['tokens']
    assert result['cuda_peak_allocated_bytes'] is None
    assert result['warmup_fraction_of_completed_updates'] == 1
    assert result['warmup_dominates_completed_segment'] is True


def test_profile_measures_complete_steps_and_preserves_billing(config, tmp_path):
    from tnlm.train import profile_run
    config.raw['project']['history_limit'] = 1
    result = profile_run(config, tmp_path / 'profile', steps=2, warmup_steps=1,
                         max_seconds=60, device='cpu')
    assert result['completed_timed_steps'] == 2
    assert result['completed_total_steps'] == 3
    assert result['timed_prediction_tokens'] == 14
    assert result['prediction_tokens_per_second'] > 0
    assert result['billing']['elapsed_seconds'] >= result['timed_training_seconds']
    assert result['billing']['status'] == 'complete'
    assert (tmp_path / 'profile' / 'profile.json').is_file()
    saved = state(tmp_path / 'profile' / 'final')
    assert saved['progress']['steps'] == 3


def test_summary_keeps_training_loss_and_segment_cost_explicit(config, tmp_path):
    from tnlm.train import summarize_runs
    train_run(config, tmp_path / 'summary-source', max_steps=3, max_seconds=60, device='cpu')
    report = summarize_runs([tmp_path / 'summary-source'], tmp_path / 'summary', max_plot_points=2)
    saved = json.loads((tmp_path / 'summary' / 'comparison.json').read_text())
    assert report['runs'] == 1
    row = saved['runs'][0]
    assert row['run_prediction_tokens'] == row['global_prediction_tokens'] == 21
    assert row['operator_class'] == 'N'
    assert row['run_billed_usd'] == 0
    assert row['device'] == 'cpu'
    assert saved['curves'][0]['original_metric_points'] == 3
    assert saved['curves'][0]['points'][-1]['tokens'] == 21
    for name in ('loss-vs-tokens.svg', 'loss-vs-time.svg', 'loss-vs-cost.svg'):
        assert '<svg' in (tmp_path / 'summary' / name).read_text()


def test_polynomial_degree_reporting_keeps_normalizers_explicit(config):
    assert validate_config(config)['global_polynomial_degree_upper_bound'] is None
    config.model.normalization = 'lagged'
    config.model.final_normalization = 'lagged'
    polynomial = validate_config(config)
    assert polynomial['operator_audit']['class'] == 'P'
    assert polynomial['global_polynomial_degree_upper_bound'] == 10
    config.model.attention = 'softmax'
    assert validate_config(config)['global_polynomial_degree_upper_bound'] is None


def test_recovery_retention_keeps_two_and_full_branchpoints(config, tmp_path):
    config.raw['project']['checkpoint_every_steps'] = 1
    train_run(config, tmp_path / 'recovery', max_steps=4, max_seconds=60, device='cpu')
    directory = tmp_path / 'recovery'
    assert sorted(p.name for p in directory.glob('recovery-*')) == ['recovery-00000003', 'recovery-00000004']
    assert (directory / 'initial' / 'manifest.json').is_file()
    assert (directory / 'final' / 'manifest.json').is_file()
    for checkpoint in directory.glob('recovery-*'):
        assert state(checkpoint)['optimizer'] is not None


def test_ledger_records_checkpoint_parent_for_retention(config, tmp_path):
    first = train_run(config, tmp_path / 'pin-source', max_steps=1, max_seconds=60, device='cpu')
    train_run(config, tmp_path / 'pin-child', max_steps=1, max_seconds=60,
              device='cpu', resume=first['checkpoint'])
    ledger = json.loads((tmp_path / 'spend.json').read_text())
    assert ledger['runs'][str((tmp_path / 'pin-child').resolve())]['parent_checkpoint'] == first['checkpoint']


def test_resume_cannot_reset_project_budget_with_new_ledger(config, tmp_path):
    first = train_run(config, tmp_path / 'ledger-source', max_steps=1, max_seconds=60, device='cpu')
    config.raw['project']['spend_ledger'] = str(tmp_path / 'new-ledger.json')
    with pytest.raises(ValueError, match='retain the project spend ledger'):
        train_run(config, tmp_path / 'ledger-child', max_steps=1, max_seconds=60,
                  device='cpu', resume=first['checkpoint'])


def test_deep_lagged_calibration_does_not_execute_unscaled_downstream(config, tmp_path):
    import time
    from tnlm.train import calibrate_lagged
    config.model.layers = 8
    config.model.d_model = 32
    config.model.d_key = 8
    config.model.d_value = 8
    config.model.d_mlp = 48
    config.model.normalization = 'lagged'
    config.model.final_normalization = 'lagged'
    source = tmp_path / 'deep-cal-source.jsonl'
    source.write_text(json.dumps({'id': 'deep-prefix', 'text': ''.join(chr(i) for i in range(16))}) + '\n')
    manifest = prepare_data(tmp_path / 'deep-cal-data', source_jsonl=source, revision='deep-cal-v1',
                            tokenizer=FixtureTokenizer(),
                            split_fractions={'train': 0., 'val': 0., 'test': 0., 'calibration': 1.})
    config.raw['data'].update(manifest=str(manifest), context=8)
    seed_all(101)
    model = TensorLM(config.model).eval()
    calibration_sampler = PackedSampler(config.raw['data']['manifest'], split='calibration',
                                        context=8, seed=config.train.seed)
    probe, docs, _ = calibration_sampler.next_batch()
    with torch.no_grad():
        assert not torch.isfinite(model(probe, docs)).all()  # actual initialization regression
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    calibration = calibrate_lagged(model, config, config.raw['data']['manifest'],
                                   torch.device('cpu'), deadline=time.monotonic() + 60)
    assert calibration['tokens'] == 8
    assert calibration['passes'] == 17
    assert calibration['forward_token_exposures'] == 8 * 18
    assert calibration['verification_passes'] == 1
    assert calibration['verification_logits_finite'] is True
    assert len(calibration['sites']) == 17
    assert all(not m._forward_hooks and not m.collecting for m in lagged_modules(model))
    assert all(torch.equal(before[k], v) for k, v in model.named_parameters())
    with torch.no_grad():
        assert torch.isfinite(model(probe, docs)).all()
        torch.testing.assert_close(model(probe, docs)[:, :2], model(probe[:, :2], docs[:, :2]), rtol=1e-5, atol=1e-5)
    assert model.operator_audit()['class'] == 'P'


def test_launch_reservation_includes_declared_overrun_margin(tmp_path):
    ledger = SpendLedger(tmp_path / 'margin.json', total_budget=10, reserve=2)
    assert ledger.begin('one', 4, 3600, cost_margin_fraction=.25) == 5
    with pytest.raises(ValueError, match='budget exhausted'):
        ledger.begin('two', 3, 3600, cost_margin_fraction=.25)
    with pytest.raises(ValueError, match='run budget'):
        ledger.begin('three', 4, 3600, cost_margin_fraction=.25, run_budget=4.9)
    ledger.finish('one', 3600, 'complete')
    row = json.loads((tmp_path / 'margin.json').read_text())['runs']['one']
    assert row['spent_usd'] == 4  # margin is reservation, not invented spend
    assert row['reserved_usd'] == 0


def test_finite_stored_statistic_with_overflowing_compute_divisor_is_rejected():
    from tnlm.normalization import LaggedRMS
    from tnlm.train import _check_lagged_divisor
    norm = LaggedRMS(4)
    norm.mean_sq.fill_(1e80)  # FP64 buffer is finite; current FP32 divisor is not
    assert torch.isfinite(norm.mean_sq)
    with pytest.raises(FloatingPointError, match='compute divisor'):
        _check_lagged_divisor(norm)


def test_analysis_cli_dryrun_needs_no_analysis_module_or_probes(config, tmp_path, capsys):
    import yaml
    cfg = config.to_dict()
    cfg['project']['dry_run'] = True
    path = tmp_path / 'analysis-dry.yaml'
    path.write_text(yaml.safe_dump(cfg))
    assert main(['analyze', '--config', str(path), '--checkpoint', 'not-present',
                 '--out', str(tmp_path / 'analysis-dry'), '--context', '2048', '--device', 'cuda']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['dry_run'] and result['context'] == 2048
    assert not (tmp_path / 'analysis-dry').exists()
    assert not (tmp_path / 'spend.json').exists()


def test_stage_caps_and_explicit_reserve_pool(tmp_path):
    ledger = SpendLedger(tmp_path / 'stages.json', total_budget=10, reserve=2,
                          stage_budgets={'A': 3, 'B': 3, 'C': 1, 'D': 1, 'Reserve': 2})
    ledger.begin('a-active', 3, 3600, stage='A')
    with pytest.raises(ValueError, match='stage A budget exhausted'):
        ledger.begin('a-extra', 1, 3600, stage='A')
    ledger.begin('b-active', 3, 3600, stage='B')
    ledger.begin('recovery', 2, 3600, stage='Reserve')
    ledger.begin('c-active', 1, 3600, stage='C')
    ledger.begin('d-active', 1, 3600, stage='D')
    with pytest.raises(ValueError, match='project budget exhausted'):
        ledger.begin('overflow', .01, 3600, stage='Reserve')
    ledger.finish('a-active', 1800, 'complete')
    ledger.begin('a-released', 1.5, 3600, stage='A')
    ledger.finish('a-released', 1800, 'complete')
    with pytest.raises(ValueError, match='active runtime reservation'):
        ledger.finish('a-released', 1800, 'complete')


def test_external_charges_record_truth_and_block_overspending(tmp_path):
    ledger = SpendLedger(tmp_path / 'charges.json', total_budget=10, reserve=2)
    row = ledger.record_charge('api-usage', 3, stage='Reserve', description='observed API invoice')
    assert row['exceeds_stage_budget'] and not row['exceeds_project_budget']
    with pytest.raises(ValueError, match='stage Reserve budget exhausted'):
        ledger.begin('reserve-job', .1, 3600, stage='Reserve')
    ledger.record_charge('storage', 8, stage='Reserve', description='observed storage invoice')
    with pytest.raises(ValueError, match='project budget exhausted'):
        ledger.begin('no-budget', .1, 3600)
    saved = json.loads((tmp_path / 'charges.json').read_text())
    assert sum(row['spent_usd'] for row in saved['runs'].values()) == 11
    assert saved['runs']['storage']['exceeds_project_budget']
    with pytest.raises(ValueError, match='duplicate'):
        ledger.record_charge('storage', 1, stage='Reserve', description='duplicate receipt')


def test_cost_plan_uses_actual_stage_reservations_without_writing(config, tmp_path):
    from tnlm.train import plan_search
    ledger = SpendLedger(tmp_path / 'spend.json', reserve=10)
    ledger.begin('existing-job', 10, 3600, stage='B')
    config.raw['project'].update(stage='B', reserve_usd=10,
        measured_prediction_tokens_per_second=50, gpu_hourly_rate_usd=2, estimated_setup_seconds=3)
    before = (tmp_path / 'spend.json').read_bytes()
    plan = plan_search(config)
    assert plan['unallocated_stage_usd'] == 165
    assert plan['unallocated_regular_usd'] == 980
    assert plan['unallocated_project_usd'] == 990
    assert plan['stage_budgets_usd']['Reserve'] == 10
    assert plan['proposed_runs'][0]['projected_seconds'] == 5
    assert plan['proposed_runs'][0]['projected_usd'] == pytest.approx(5 * 2 / 3600)
    assert (tmp_path / 'spend.json').read_bytes() == before


@pytest.mark.skipif(importlib.util.find_spec('tnlm.analysis') is None,
                    reason='Milestone incomplete: Claude-owned tnlm.analysis has not landed')
@pytest.mark.parametrize('normalization', ['pre_rms', 'branch_rms'])
def test_first_milestone_train_resume_and_actual_analysis(config, tmp_path, capsys, normalization):
    """This gate must run when the real diagnostic module arrives; no stub qualifies."""
    import yaml
    source = tmp_path / 'milestone-validation.jsonl'
    source.write_text(''.join(json.dumps({'id': f'heldout-{i}', 'text': text}) + '\n'
                    for i, text in enumerate(['nopqrst', 'uvwxyz'])))
    validation = prepare_data(tmp_path / 'milestone-validation', source_jsonl=source,
                    revision='milestone-validation-v1', tokenizer=FixtureTokenizer(),
                    split_fractions={'train': 0., 'val': 1., 'test': 0., 'calibration': 0.})
    config.model.normalization = normalization
    config.raw['evaluation'] = {'validation_manifest': str(validation), 'seed': 0}
    whole = train_run(config, tmp_path / 'milestone-whole', max_steps=4, max_seconds=60, device='cpu')
    first = train_run(config, tmp_path / 'milestone-first', max_steps=1, max_seconds=60, device='cpu')
    resumed = train_run(config, tmp_path / 'milestone-resumed', max_steps=3, max_seconds=60,
                        device='cpu', resume=first['checkpoint'])
    assert [h['batch_hashes'] for h in whole['history'][1:]] == [h['batch_hashes'] for h in resumed['history']]
    expected, actual = state(Path(whole['checkpoint'])), state(Path(resumed['checkpoint']))
    assert all(torch.equal(expected['model'][key], actual['model'][key]) for key in expected['model'])
    model = TensorLM(config.model).eval()
    model.load_state_dict(actual['model'])
    probe, docs, _ = PackedSampler(config.raw['data']['manifest'], context=4).next_batch()
    with torch.no_grad():
        torch.testing.assert_close(model(probe, docs)[:, :2], model(probe[:, :2], docs[:, :2]), rtol=1e-5, atol=1e-6)
    cfg_file = tmp_path / 'milestone.yaml'
    cfg_file.write_text(yaml.safe_dump(config.to_dict()))
    out = tmp_path / 'milestone-analysis'
    assert main(['analyze', '--config', str(cfg_file), '--checkpoint', resumed['checkpoint'],
                 '--out', str(out), '--batches', '1', '--max-seconds', '60', '--device', 'cpu']) == 0
    result = json.loads(capsys.readouterr().out)
    for diagnostic in ('local_tensor', 'read_write_graph', 'normalizer_sensitivity',
                       'ablation', 'fixed_denominator', 'low_rank'):
        assert result[diagnostic], f'missing real diagnostic: {diagnostic}'
    for filename in result['files']:
        assert json.loads((out / filename).read_text()), f'empty artifact: {filename}'
    assert result['evaluation_provenance']['frozen_state']['unchanged']
    assert result['billing']['status'] == 'complete'
    assert (out / 'analysis-invocation-result.json').is_file()

@pytest.mark.parametrize('behavior', ['clean', 'mutate', 'nonfinite', 'hook', 'collecting', 'gradient', 'ablate'])
def test_analysis_wrapper_certifies_frozen_state_and_honest_billing(config, tmp_path, monkeypatch, behavior):
    """Wrapper contract checks complement, rather than replace, real diagnostics."""
    import argparse
    import sys
    import types
    from tnlm.cli import _run_analysis
    source = tmp_path / 'analysis-val.jsonl'
    source.write_text(json.dumps({'id': 'probe-only', 'text': 'nopqrstuvwxyz'}) + '\n')
    manifest = prepare_data(tmp_path / 'analysis-val', source_jsonl=source,
                            revision='probe-v1', tokenizer=FixtureTokenizer(),
                            split_fractions={'train': 0., 'val': 1., 'test': 0., 'calibration': 0.})
    config.raw['evaluation'] = {'validation_manifest': str(manifest)}
    saved = train_run(config, tmp_path / 'analysis-source', max_steps=1, max_seconds=60, device='cpu')
    module = types.ModuleType('tnlm.analysis')
    def diagnostic(model, batches, out):
        if behavior == 'mutate':
            with torch.no_grad():
                next(model.parameters()).add_(1)
        elif behavior == 'hook':
            model.register_forward_hook(lambda *args: None)
        elif behavior == 'collecting':
            model.collecting = True
        elif behavior == 'gradient':
            next(model.parameters()).requires_grad_(False)
        elif behavior == 'ablate':
            model.blocks[0].ablate = frozenset({'mlp'})
        return {'diagnostic': float('nan') if behavior == 'nonfinite' else 'contract fixture',
                'probe_random': float(torch.rand(()))}
    module.analyze = diagnostic
    monkeypatch.setitem(sys.modules, 'tnlm.analysis', module)
    out = tmp_path / 'analysis-result'
    args = argparse.Namespace(context=4, batches=1, max_seconds=60, checkpoint=saved['checkpoint'],
                              out=str(out), device='cpu', run_budget_usd=None)
    if behavior == 'clean':
        result = _run_analysis(config, args)
        assert result['evaluation_provenance']['frozen_state']['unchanged']
        assert result['evaluation_provenance']['probe_seed'] == 0
        assert result['probe_random'] == float(torch.rand((), generator=torch.Generator().manual_seed(0)))
        assert 'cli.py' in result['evaluation_provenance']['runtime_source_sha256']
        assert len(result['evaluation_provenance']['batch_hashes']) == 1
        assert result['billing']['status'] == 'complete'
    else:
        with pytest.raises((RuntimeError, ValueError)):
            _run_analysis(config, args)
        ledger = json.loads(Path(config.raw['project']['spend_ledger']).read_text())
        assert ledger['runs'][str(out.resolve())]['status'] == 'failed'
        assert not (out / 'analysis-invocation-result.json').exists()

@pytest.mark.parametrize('invalid_target', [-2, 50257])
def test_invalid_discarded_targets_fail_before_token_cap(config, tmp_path, monkeypatch, invalid_target):
    # Even targets discarded by a short rung belong to the declared raw batch.
    config.train.rung_total_tokens = 1
    original = PackedSampler.next_batch
    def invalid_batch(self):
        idx, docs, targets = original(self)
        targets[0, -1] = invalid_target
        return idx, docs, targets
    monkeypatch.setattr(PackedSampler, 'next_batch', invalid_batch)
    with pytest.raises(ValueError, match='target outside real vocabulary'):
        train_run(config, tmp_path / 'invalid-discard', max_steps=1, max_seconds=60, device='cpu')
    assert not (tmp_path / 'invalid-discard' / 'final').exists()
