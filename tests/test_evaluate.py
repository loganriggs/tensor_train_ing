"""Offline CPU held-out evaluation and source-target fairness checks."""
import json

import pytest
import torch
import torch.nn.functional as F

from tnlm.checkpoint import load_checkpoint
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import PackedSampler, prepare_data
from tnlm.model import TensorLM
from tnlm.train import evaluate_checkpoint, train_run
from helpers import FixtureTokenizer


@pytest.fixture(params=['pre_rms', 'lagged'])
def trained(tmp_path, request):
    torch.set_num_threads(1)
    source = tmp_path / 'train.jsonl'
    source.write_text(''.join(json.dumps({'id': str(i), 'text': 'abcdefg'}) + '\n' for i in range(20)))
    train_manifest = prepare_data(tmp_path / 'training-data', source_jsonl=source, revision='offline-v1',
        tokenizer=FixtureTokenizer(), split_fractions={'train': .8, 'val': 0., 'test': 0., 'calibration': .2})
    validation = tmp_path / 'validation.jsonl'
    validation.write_text(''.join(json.dumps({'id': f'validation-{i}', 'text': text}) + '\n'
                                  for i, text in enumerate(['hijklmn', 'opqrstu'])))
    val_manifest = prepare_data(tmp_path / 'validation-data', source_jsonl=validation, revision='offline-val-v1',
        tokenizer=FixtureTokenizer(), split_fractions={'train': 0., 'val': 1., 'test': 0., 'calibration': 0.})
    cfg = Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                                   d_mlp=12, padded_vocab=50257, normalization=request.param,
                                   final_normalization='lagged' if request.param == 'lagged' else 'rms'),
        train=TrainConfig(compile=False, bf16_compute=False,
                          global_prediction_tokens_per_update=7, rung_total_tokens=100),
        raw={'data': {'manifest': str(train_manifest), 'context': 7, 'micro_batch_size': 1, 'calibration_tokens': 8},
             'evaluation': {'validation_manifest': str(val_manifest)},
             'project': {'dry_run': False, 'gpu_hourly_rate_usd': .01,
                         'spend_ledger': str(tmp_path / 'spend.json')}})
    result = train_run(cfg, tmp_path / 'train', max_steps=1, max_seconds=60, device='cpu')
    return cfg, result['checkpoint']


def test_context_independent_targets_manual_ce_and_frozen_state(trained, tmp_path):
    cfg, checkpoint = trained
    a = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'eval2', context=2,
                            max_prediction_tokens=14, max_seconds=60, bootstrap_samples=25)
    b = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'eval7', context=7,
                            max_prediction_tokens=14, max_seconds=60, bootstrap_samples=25)
    assert a['target_plan_sha256'] == b['target_plan_sha256']
    assert a['target_plan'] == b['target_plan']
    assert a['prediction_tokens'] == b['prediction_tokens'] == 14
    assert all(row['tokens'] == 7 for row in a['per_document'])
    assert a['available_prefix_length_histogram'] == {'1': 8, '2': 6}
    assert b['available_prefix_length_histogram'] == {str(i): 2 for i in range(1, 8)}
    assert a['sampler']['final']['epoch'] == 0
    assert a['sampler']['final']['cursor'] == a['sampler']['rows_scanned']
    model = TensorLM(cfg.model).eval()
    load_checkpoint(checkpoint, model, restore_rng=False)
    # Independent source documents, no packed sampler/target-plan code involved.
    expected_sum = 0.
    with torch.no_grad():
        for text in ['hijklmn', 'opqrstu']:
            tokens = torch.tensor([[ord(c) for c in text] + [50256]])
            logits = model(tokens[:, :-1])
            expected_sum += float(F.cross_entropy(logits.float().flatten(0, 1),
                                                 tokens[:, 1:].flatten(), reduction='sum'))
    assert b['ce'] == pytest.approx(expected_sum / 14, abs=1e-6)
    assert a['frozen_state']['unchanged'] and b['frozen_state']['unchanged']
    assert a['model_sha256'] == b['model_sha256']
    assert b['uncertainty']['unit'] == 'documents, not training seeds'
    assert len(b['uncertainty']['ce_95_ci']) == 2
    assert b['validation_below_one_million_targets']
    assert not b['exposure']['final_test_exposed']
    saved = json.loads((tmp_path / 'eval7' / 'evaluation.json').read_text())
    assert saved['billing']['status'] == 'complete'
    assert saved['billing']['reserved_usd'] == 0
    assert saved['billing']['spent_usd'] > 0
    assert saved['billing']['parent_checkpoint'] == checkpoint


def test_partial_plan_single_doc_ci_and_dryrun(trained, tmp_path):
    cfg, checkpoint = trained
    report = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'partial', context=2,
                                 max_prediction_tokens=3, max_seconds=60)
    assert report['prediction_tokens'] == 3
    assert len(report['target_plan']['ranges']) == 1
    assert report['uncertainty']['ce_95_ci'] is None
    cfg.raw['project']['dry_run'] = True
    ledger_before = (tmp_path / 'spend.json').read_bytes()
    dry = evaluate_checkpoint(cfg, 'missing-checkpoint', tmp_path / 'dry', max_seconds=1)
    assert dry['dry_run']
    assert not (tmp_path / 'dry').exists()
    assert (tmp_path / 'spend.json').read_bytes() == ledger_before


def test_failure_releases_evaluation_reservation(trained, tmp_path, monkeypatch):
    cfg, checkpoint = trained
    def fail(*args, **kwargs):
        raise RuntimeError('injected evaluator failure')
    monkeypatch.setattr('tnlm.evaluate.TensorLM', fail)
    with pytest.raises(RuntimeError, match='injected evaluator failure'):
        evaluate_checkpoint(cfg, checkpoint, tmp_path / 'failed', max_seconds=60)
    row = json.loads((tmp_path / 'spend.json').read_text())['runs'][str((tmp_path / 'failed').resolve())]
    assert row['status'] == 'failed' and row['reserved_usd'] == 0
    assert row['elapsed_seconds'] > 0
    assert not (tmp_path / 'failed' / 'evaluation.json').exists()


def test_checkpoint_config_identity_is_strict(trained, tmp_path):
    cfg, checkpoint = trained
    cfg.raw['evaluation']['unrecorded_override'] = True
    with pytest.raises(ValueError, match='config mismatch'):
        evaluate_checkpoint(cfg, checkpoint, tmp_path / 'changed-validation', max_seconds=60)


def test_validation_packing_seed_is_independent_of_training_seed(trained, tmp_path):
    cfg, checkpoint = trained
    report = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'fixed-probes', context=2,
                                 max_prediction_tokens=14, max_seconds=60, bootstrap_samples=25)
    assert cfg.train.seed == 101
    assert report['evaluation_seed'] == report['sampler']['initial']['seed'] == 0
    assert report['uncertainty']['seed'] == 0
    assert len(report['available_prefix_plan_sha256']) == 64
    import copy
    other_cfg = copy.deepcopy(cfg)
    other_cfg.train.seed = 202
    other = train_run(other_cfg, tmp_path / 'other-training-seed', max_steps=1, max_seconds=60, device='cpu')
    other_report = evaluate_checkpoint(other_cfg, other['checkpoint'], tmp_path / 'other-fixed-probes',
                        context=2, max_prediction_tokens=14, max_seconds=60, bootstrap_samples=25)
    assert report['target_plan_sha256'] == other_report['target_plan_sha256']
    assert report['available_prefix_plan_sha256'] == other_report['available_prefix_plan_sha256']
    assert report['sampler']['initial']['seed'] == other_report['sampler']['initial']['seed']
    assert report['model_sha256'] != other_report['model_sha256']


def test_paired_comparison_real_reports_and_known_constant(trained, tmp_path):
    import copy
    from tnlm.train import compare_evaluations

    cfg, checkpoint = trained
    left = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'compare-eval2', context=2,
                               max_prediction_tokens=14, max_seconds=60, bootstrap_samples=0)
    right = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'compare-eval7', context=7,
                                max_prediction_tokens=14, max_seconds=60, bootstrap_samples=0)
    assert left['probe']['critical_model_sources_match_checkpoint']
    assert set(left['probe']['critical_model_source_matches']) == {'model.py', 'attention.py', 'normalization.py', 'config.py'}
    assert 'tnlm/attention.py' in left['probe']['runtime_source_sha256']
    same = compare_evaluations(left, tmp_path / 'compare-eval2' / 'evaluation.json',
                               tmp_path / 'same', bootstrap_samples=31)
    assert same['delta_ce_right_minus_left'] == 0
    assert same['uncertainty']['delta_ce_95_ci'] == [0, 0]
    across = compare_evaluations(left, right, tmp_path / 'across', bootstrap_samples=31)
    assert not across['contexts_match']
    assert not across['available_prefix_plans_match']
    assert across['delta_ce_right_minus_left'] == pytest.approx(right['ce'] - left['ce'], abs=1e-12)
    assert (tmp_path / 'across' / 'comparison.json').is_file()
    shifted = copy.deepcopy(left)
    shift = .375
    for row in shifted['per_document']:
        row['loss_sum'] += shift * row['tokens']
    shifted['loss_sum'] = sum(row['loss_sum'] for row in shifted['per_document'])
    shifted['ce'] = shifted['loss_sum'] / shifted['prediction_tokens']
    known = compare_evaluations(left, shifted, tmp_path / 'known', bootstrap_samples=31)
    assert known['delta_ce_right_minus_left'] == pytest.approx(shift, abs=1e-14)
    assert known['uncertainty']['delta_ce_95_ci'] == pytest.approx([shift, shift], abs=1e-14)


def test_paired_comparison_rejects_inconsistent_reports_before_output(trained, tmp_path):
    import copy
    import hashlib
    from tnlm.train import compare_evaluations

    cfg, checkpoint = trained
    report = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'reject-source', context=2,
                                 max_prediction_tokens=14, max_seconds=60, bootstrap_samples=0)

    def rehash_plan(value):
        value['target_plan_sha256'] = hashlib.sha256(json.dumps(value['target_plan'], sort_keys=True,
                                                    separators=(',', ':')).encode()).hexdigest()

    changes = [
        ('changed-plan', lambda r: r['target_plan'].update(ordering='changed')),
        ('changed-plan-valid-hash', lambda r: (r['target_plan'].update(ordering='changed'), rehash_plan(r))),
        ('same-context-prefix', lambda r: r.update(available_prefix_plan_sha256='0' * 64)),
        ('doc-count', lambda r: r['per_document'][0].update(tokens=6)),
        ('bad-range', lambda r: (r['target_plan']['ranges'][0].update(target_end=99), rehash_plan(r))),
        ('stale-loss', lambda r: r.update(loss_sum=r['loss_sum'] + 1)),
        ('stale-ce', lambda r: r.update(ce=r['ce'] + 1)),
        ('stale-total', lambda r: r.update(prediction_tokens=13)),
        ('not-frozen', lambda r: r['frozen_state'].update(unchanged=False)),
        ('nan-doc-loss', lambda r: r['per_document'][0].update(loss_sum=float('nan'))),
    ]
    for name, change in changes:
        bad = copy.deepcopy(report)
        change(bad)
        out = tmp_path / name
        with pytest.raises(ValueError):
            compare_evaluations(report, bad, out, bootstrap_samples=2)
        assert not out.exists()
    with pytest.raises(ValueError, match='bootstrap_samples'):
        compare_evaluations(report, report, tmp_path / 'too-many', bootstrap_samples=100001)
    assert not (tmp_path / 'too-many').exists()


def test_evaluation_allows_billing_stage_changes_without_changing_checkpoint(trained, tmp_path):
    cfg, checkpoint = trained
    cfg.raw['project'].update(stage='B', gpu_hourly_rate_usd=.02)
    report = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'stage-b-eval', context=2,
                                 max_prediction_tokens=3, max_seconds=60, bootstrap_samples=0)
    assert report['billing']['stage'] == 'B'
    assert report['billing']['rate_usd_per_hour'] == .02
    assert report['frozen_state']['unchanged']


def test_mathematically_valid_zero_cross_entropy_can_be_compared(trained, tmp_path):
    import copy
    from tnlm.train import compare_evaluations
    cfg, checkpoint = trained
    report = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'zero-base', context=2,
                                 max_prediction_tokens=14, max_seconds=60, bootstrap_samples=0)
    perfect = copy.deepcopy(report)
    for row in perfect['per_document']:
        row['loss_sum'] = 0.0
    perfect['loss_sum'] = perfect['ce'] = 0.0
    compared = compare_evaluations(perfect, perfect, tmp_path / 'zero-comparison', bootstrap_samples=20)
    assert compared['delta_ce_right_minus_left'] == 0
    assert compared['uncertainty']['delta_ce_95_ci'] == [0.0, 0.0]


def test_overlapping_content_is_rejected_before_model_loading(trained, tmp_path, monkeypatch):
    cfg, _ = trained
    source = tmp_path / 'leaking-validation.jsonl'
    source.write_text(json.dumps({'id': 'different-id-same-document', 'text': 'abcdefg'}) + '\n')
    manifest = prepare_data(tmp_path / 'leaking-validation', source_jsonl=source, revision='leak-fixture-v1',
        tokenizer=FixtureTokenizer(), split_fractions={'train': 0., 'val': 1., 'test': 0., 'calibration': 0.})
    cfg.raw['evaluation']['validation_manifest'] = str(manifest)
    run = train_run(cfg, tmp_path / 'leak-training', max_steps=1, max_seconds=60, device='cpu')
    def forbidden_model(*args, **kwargs):
        raise AssertionError('overlap gate must run before model loading')
    monkeypatch.setattr('tnlm.evaluate.TensorLM', forbidden_model)
    with pytest.raises(ValueError, match='exact whole-document content overlap'):
        evaluate_checkpoint(cfg, run['checkpoint'], tmp_path / 'leak-eval', max_seconds=60)
    audit = json.loads((tmp_path / 'leak-eval' / 'data-overlap-audit.json').read_text())
    assert audit['overlap_counts']['train'] == 1
    assert audit['overlap_counts']['calibration'] == 1
    assert audit['has_overlap']
    assert not (tmp_path / 'leak-eval' / 'evaluation.json').exists()
    row = json.loads((tmp_path / 'spend.json').read_text())['runs'][str((tmp_path / 'leak-eval').resolve())]
    assert row['status'] == 'failed' and row['reserved_usd'] == 0


def test_comparison_rejects_unproven_content_separation(trained, tmp_path):
    import copy
    from tnlm.train import compare_evaluations
    cfg, checkpoint = trained
    report = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'audit-report', context=2,
                                 max_prediction_tokens=14, max_seconds=60, bootstrap_samples=0)
    contaminated = copy.deepcopy(report)
    contaminated['data_overlap_audit']['has_overlap'] = True
    with pytest.raises(ValueError, match='separation audit'):
        compare_evaluations(report, contaminated, tmp_path / 'bad-audit-comparison', bootstrap_samples=20)
    assert not (tmp_path / 'bad-audit-comparison').exists()
    wrong_selection = copy.deepcopy(report)
    wrong_selection['data_overlap_audit']['selected_validation_ids_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='audit differs from target plan'):
        compare_evaluations(report, wrong_selection, tmp_path / 'wrong-audit-selection', bootstrap_samples=20)
    assert not (tmp_path / 'wrong-audit-selection').exists()


def test_training_descriptor_uses_loaded_checkpoint_not_caller_probe_or_mutation(trained, tmp_path, monkeypatch):
    import tnlm.checkpoint as checkpoint_module
    cfg, checkpoint = trained
    original_load = checkpoint_module.load_checkpoint
    def load_then_change_caller(*args, **kwargs):
        payload = original_load(*args, **kwargs)
        # Simulate caller-side changes after compatibility validation. The report
        # must use the loaded immutable payload rather than this live Config.
        cfg.train.optimizer = 'muon'
        cfg.train.seed = 999
        cfg.train.rung_total_tokens = 9999
        cfg.raw['data']['context'] = 333
        return payload
    monkeypatch.setattr(checkpoint_module, 'load_checkpoint', load_then_change_caller)
    report = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'checkpoint-derived', context=2,
                                 max_prediction_tokens=3, max_seconds=60, bootstrap_samples=0)
    descriptor = report['training_descriptor']
    assert descriptor['source'] == 'checkpoint_payload'
    assert descriptor['optimizer'] == 'adamw' and descriptor['seed'] == 101
    assert descriptor['width'] == 8 and descriptor['depth'] == 1
    assert descriptor['train_context'] == 7 and report['context'] == 2
    assert descriptor['rung_total_tokens'] == 100 and descriptor['actual_prediction_tokens'] == 7
    assert descriptor['cooldown_status'] == 'uncooled' and descriptor['completion_status'] == 'partial'
    assert descriptor['width_scaling_base'] == 'disabled'
    assert descriptor['aux_adamw_lr'] == 'default_to_lr'


def test_comparison_reports_real_cooled_and_different_token_budgets(trained, tmp_path):
    from tnlm.evaluate import compare_evaluations
    cfg, checkpoint = trained
    uncooled = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'uncooled-eval', context=2,
                                   max_prediction_tokens=3, max_seconds=60, bootstrap_samples=0)
    cooled_run = train_run(cfg, tmp_path / 'cooldown-branch', resume=checkpoint,
                           cooldown_steps=2, max_steps=2, max_seconds=60, device='cpu')
    cooled = evaluate_checkpoint(cfg, cooled_run['checkpoint'], tmp_path / 'cooled-eval', context=2,
                                 max_prediction_tokens=3, max_seconds=60, bootstrap_samples=0)
    descriptor = cooled['training_descriptor']
    assert descriptor['cooldown_status'] == 'cooled' and descriptor['cooldown_steps'] == 2
    assert descriptor['cooldown_start_tokens'] == 7
    assert descriptor['actual_prediction_tokens'] == descriptor['segment_target_prediction_tokens'] == 21
    assert descriptor['completion_status'] == 'complete'
    compared = compare_evaluations(uncooled, cooled, tmp_path / 'cooldown-contrast', bootstrap_samples=0)
    dimensions = compared['training_comparison']['dimensions']
    assert dimensions['cooldown_status']['status'] == 'different'
    assert dimensions['actual_prediction_tokens']['status'] == 'different'
    assert dimensions['completion_status']['status'] == 'different'
    assert dimensions['optimizer']['status'] == 'matched'
    assert not compared['automatic_ranking_eligible']
    assert not compared['training_comparison']['automatic_ranking_eligible']
    assert 'cooled and uncooled' in compared['training_comparison']['scope']
    continued_run = train_run(cfg, tmp_path / 'longer-parent', resume=checkpoint,
                              max_steps=1, max_seconds=60, device='cpu')
    continued = evaluate_checkpoint(cfg, continued_run['checkpoint'], tmp_path / 'continued-eval', context=2,
                                    max_prediction_tokens=3, max_seconds=60, bootstrap_samples=0)
    token_comparison = compare_evaluations(uncooled, continued, tmp_path / 'tokens-contrast', bootstrap_samples=0)
    assert token_comparison['training_comparison']['dimensions']['cooldown_status']['status'] == 'matched'
    assert token_comparison['training_comparison']['dimensions']['actual_prediction_tokens']['status'] == 'different'


def test_legacy_missing_training_metadata_never_implies_equal_training(trained, tmp_path):
    import copy
    from tnlm.evaluate import compare_evaluations
    cfg, checkpoint = trained
    report = evaluate_checkpoint(cfg, checkpoint, tmp_path / 'legacy-source', context=2,
                                 max_prediction_tokens=3, max_seconds=60, bootstrap_samples=0)
    legacy = copy.deepcopy(report)
    legacy.pop('training_descriptor')
    result = compare_evaluations(legacy, legacy, tmp_path / 'legacy-comparison', bootstrap_samples=0)
    assert result['delta_ce_right_minus_left'] == 0
    assert all(row['status'] == 'unknown' for row in result['training_comparison']['dimensions'].values())
    assert not result['automatic_ranking_eligible']
    one_missing = compare_evaluations(report, legacy, tmp_path / 'one-legacy', bootstrap_samples=0)
    assert one_missing['training_comparison']['dimensions']['seed']['status'] == 'unknown'


def test_descriptor_distributed_schedule_completion_and_optional_missing_evidence():
    from tnlm.evaluate import _training_descriptor, _training_comparison
    payload = {'config': {'train': {'optimizer': 'muon', 'seed': 3, 'lr': .001, 'aux_adamw_lr': None,
                                   'rung_total_tokens': 1000, 'global_prediction_tokens_per_update': 10,
                                   'schedule': 'constant_with_separate_cooldown'},
                          'model': {'d_model': 16, 'layers': 2, 'normalization': 'pre_rms',
                                    'attention': 'signed_product', 'width_scaling_base': 8,
                                    'width_lr_exponent': .5, 'width_logit_exponent': .5}, 'data': {'context': 512}},
               'progress': {'tokens': 120, 'steps': 12, 'scheduler_step': 2, 'cooldown_start_tokens': 100,
                            'distributed_schedule': {'cooldown_steps': 2}}}
    descriptor = _training_descriptor(payload)
    assert descriptor['completion_status'] == 'complete'
    assert descriptor['segment_target_prediction_tokens'] == 120
    other = dict(descriptor, width_scaling_base='disabled', width_lr_exponent=1., width_logit_exponent=0.,
                 normalization='branch_rms', attention='offset_square', aux_adamw_lr=.003, rung_total_tokens=2000)
    compared = _training_comparison({'training_descriptor': descriptor}, {'training_descriptor': other})
    for dimension in ('width_scaling_base', 'width_lr_exponent', 'width_logit_exponent',
                      'normalization', 'attention', 'aux_adamw_lr', 'rung_total_tokens'):
        assert compared['dimensions'][dimension]['status'] == 'different'
    assert not compared['automatic_ranking_eligible']
    legacy = _training_descriptor({'config': {}, 'progress': {}})
    assert legacy['cooldown_status'] == legacy['completion_status'] == 'unknown'
    comparison = _training_comparison({'training_descriptor': legacy}, {'training_descriptor': legacy})
    assert all(row['status'] == 'unknown' for row in comparison['dimensions'].values())
    payload['progress']['scheduler_step'] = 1
    assert _training_descriptor(payload)['completion_status'] == 'unknown'


def test_depth_and_optimizer_comparison_evidence_is_checkpoint_derived():
    from tnlm.evaluate import _training_descriptor, _training_comparison
    payload = {'config': {'model': {'depth_scaling_base': None, 'depth_lr_exponent': .5,
                                    'depth_residual_base': 12, 'depth_residual_exponent': 1.,
                                    'residual_alpha': 1.},
                          'train': {'weight_decay_hidden': .1, 'beta1': .9, 'beta2': .98}},
               'progress': {'warmup_steps': 100}}
    descriptor = _training_descriptor(payload)
    assert descriptor['depth_scaling_base'] == 'disabled'
    assert descriptor['depth_residual_base'] == 12
    changes = dict(depth_scaling_base=12, depth_lr_exponent=.25, depth_residual_base=6,
                   depth_residual_exponent=.5, residual_alpha=.7, weight_decay_hidden=.2,
                   beta1=.8, beta2=.99, warmup_steps=200)
    other = dict(descriptor, **changes)
    result = _training_comparison({'training_descriptor':descriptor}, {'training_descriptor':other})
    assert all(result['dimensions'][key]['status'] == 'different' for key in changes)
    legacy = _training_descriptor({'config': {}, 'progress': {}})
    result = _training_comparison({'training_descriptor':descriptor}, {'training_descriptor':legacy})
    assert all(result['dimensions'][key]['status'] == 'unknown' for key in changes)
