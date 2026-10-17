"""Training-seed replication stays separate from document uncertainty."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from tnlm.config import Config
from tnlm.evaluate import _training_descriptor


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def report(seed, condition):
    cfg = Config()
    cfg.train.seed = seed
    cfg.train.lr = .01 if condition == 'left' else .02
    cfg.train.rung_total_tokens = 14
    cfg.train.global_prediction_tokens_per_update = 7
    cfg.raw = {'data': {'context': 4}}
    progress = {'tokens': 14, 'steps': 2, 'scheduler_step': 2, 'warmup_steps': 100,
                'context_history': [{'context': 4, 'start_tokens': 0,
                                     'start_step': 0, 'source': 'fresh'}]}
    progress['batch_history'] = [{'batch_tokens': 7, 'lr': cfg.train.lr, 'aux_adamw_lr': None,
        'start_tokens': 0, 'start_step': 0, 'source': 'fresh'}]
    descriptor = _training_descriptor({'config': cfg.to_dict(), 'progress': progress,
                                       'extra': {'cooldown_steps': None}})
    plan = {'split': 'val', 'validation_manifest_sha256': 'a' * 64,
            'ranges': [{'document_id': name, 'target_start': 1, 'target_end': 3, 'count': 2}
                       for name in ('a', 'b')]}
    ce = 1.0 + (0.2 if seed == 101 else 0.4) * (condition == 'right')
    model_hash = hashlib.sha256(f'{seed}:{condition}'.encode()).hexdigest()
    return {
        'kind': 'held_out_validation', 'split': 'val', 'evaluation_protocol_version': 2,
        'frozen_state': {'unchanged': True, 'model_before': model_hash,
                         'model_after': model_hash, 'buffers_before': 'b' * 64,
                         'buffers_after': 'b' * 64},
        'model_sha256': model_hash, 'checkpoint_state_sha256': model_hash,
        'checkpoint_manifest_sha256': model_hash,
        'checkpoint': f'/fixture/{seed}/{condition}', 'context': 4, 'evaluation_seed': 0,
        'training_manifest_sha256': 'c' * 64, 'validation_manifest_sha256': 'a' * 64,
        'available_prefix_plan_sha256': 'd' * 64,
        'target_plan': plan, 'target_plan_sha256': digest(plan),
        'data_overlap_audit': {'has_overlap': False, 'training_manifest_hash': 'c' * 64,
            'validation_manifest_hash': 'a' * 64, 'selected_validation_documents': 2,
            'selected_validation_ids_sha256': hashlib.sha256(
                json.dumps(['a', 'b'], separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()},
        'per_document': [{'document_id': name, 'tokens': 2, 'loss_sum': 2 * ce} for name in ('a', 'b')],
        'prediction_tokens': 4, 'loss_sum': 4 * ce, 'ce': ce,
        'training_descriptor': descriptor,
    }


@pytest.fixture
def pairs(tmp_path):
    rows = []
    for seed in (101, 202):
        row = {'seed': seed}
        for condition in ('left', 'right'):
            path = tmp_path / f'{condition}-{seed}.json'
            path.write_text(json.dumps(report(seed, condition)))
            row[condition] = path
        rows.append(row)
    return rows


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_two_seed_statistics_and_immutable_inputs(pairs, tmp_path):
    from tnlm.replication import compare_seed_pairs

    before = {str(path): path.read_bytes() for pair in pairs for path in (pair['left'], pair['right'])}
    result = compare_seed_pairs(pairs, tmp_path / 'result', allowed_between_dimensions=['lr', 'batch_history'])
    aggregate = result['aggregate']
    assert aggregate['seeds'] == 2
    assert aggregate['mean_paired_delta_ce'] == pytest.approx(.3)
    assert aggregate['paired_delta_sample_sd'] == pytest.approx(math.sqrt(.02))
    assert aggregate['paired_delta_standard_error'] == pytest.approx(.1)
    assert result['automatic_ranking_eligible'] is False
    assert not {'ci95', 'p_value', 'winner'} & aggregate.keys()
    serialized = json.dumps(result)
    assert 'paired' in serialized
    for path, content in before.items():
        assert Path(path).read_bytes() == content
        assert hashlib.sha256(content).hexdigest() in serialized
    assert (tmp_path / 'result').is_dir()


@pytest.mark.parametrize('case', [
    'seed', 'duplicate-seed', 'target', 'prefix', 'context', 'horizon',
    'normalization', 'within-lr', 'within-mlp', 'history', 'partial', 'metadata', 'duplicate-checkpoint',
])
def test_invalid_replication_rejects_without_output(pairs, tmp_path, case):
    from tnlm.replication import compare_seed_pairs

    if case == 'duplicate-seed':
        pairs[1]['seed'] = pairs[0]['seed']
    else:
        def change(value):
            d = value['training_descriptor']
            if case == 'seed':
                d['seed'] = 99
            elif case == 'target':
                value['target_plan']['ordering'] = 'different'
                value['target_plan_sha256'] = digest(value['target_plan'])
            elif case == 'prefix':
                value['available_prefix_plan_sha256'] = 'e' * 64
            elif case == 'context':
                value['context'] = 5
            elif case == 'horizon':
                d['rung_total_tokens'] = 15
            elif case == 'normalization':
                d['normalization'] = 'branch_rms'
            elif case == 'within-lr':
                d['lr'] = .03
            elif case == 'within-mlp':
                d['configuration']['model']['d_mlp'] += 1
            elif case == 'history':
                d['context_history_status'] = 'unknown'
            elif case == 'partial':
                d['completion_status'] = 'partial'
            elif case == 'metadata':
                value.pop('training_descriptor')
            elif case == 'duplicate-checkpoint':
                other = json.loads(pairs[0]['right'].read_text())
                value['checkpoint_state_sha256'] = other['checkpoint_state_sha256']
        rewrite(pairs[1]['right'], change)
    with pytest.raises(ValueError):
        compare_seed_pairs(pairs, tmp_path / 'rejected', allowed_between_dimensions=['lr', 'batch_history'])
    assert not (tmp_path / 'rejected').exists()


def test_real_two_seeds_two_conditions_cpu(tmp_path):
    import torch
    from helpers import FixtureTokenizer
    from tnlm.config import ModelConfig, TrainConfig
    from tnlm.data import prepare_data
    from tnlm.evaluate import evaluate_checkpoint
    from tnlm.replication import compare_seed_pairs
    from tnlm.train import train_run

    torch.set_num_threads(1)
    manifests = {}
    for split, text in [('train', 'abcdefgh'), ('val', 'ijklmnop')]:
        source = tmp_path / f'{split}.jsonl'
        source.write_text(json.dumps({'id': split, 'text': text * 200}) + '\n')
        fractions = {name: float(name == split) for name in ('train', 'val', 'test', 'calibration')}
        manifests[split] = prepare_data(
            tmp_path / f'{split}-data', source_jsonl=source, revision='offline-replication',
            max_tokens=1000, tokenizer=FixtureTokenizer(), split_fractions=fractions,
        )
    rows = []
    for seed in (101, 202):
        pair = {'seed': seed}
        for condition, lr in [('left', .01), ('right', .02)]:
            cfg = Config(
                model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                                  d_mlp=12, vocab_size=128, padded_vocab=128,
                                  attention_backend='reference'),
                train=TrainConfig(compile=False, bf16_compute=False, seed=seed, lr=lr,
                                  global_prediction_tokens_per_update=7, rung_total_tokens=14),
                raw={'data': {'manifest': str(manifests['train']), 'context': 4, 'micro_batch_size': 1},
                     'evaluation': {'validation_manifest': str(manifests['val']), 'seed': 0},
                     'project': {'dry_run': False, 'gpu_hourly_rate_usd': 0,
                                 'milestone_tokens': [], 'spend_ledger': str(tmp_path / 'spend.json')}},
            )
            name = f'{condition}-{seed}'
            trained = train_run(cfg, tmp_path / name, max_steps=2, max_seconds=60, device='cpu')
            evaluate_checkpoint(cfg, trained['checkpoint'], tmp_path / f'{name}-eval',
                                max_prediction_tokens=10, max_seconds=60, bootstrap_samples=0)
            pair[condition] = tmp_path / f'{name}-eval/evaluation.json'
        rows.append(pair)
    result = compare_seed_pairs(rows, tmp_path / 'replicated', allowed_between_dimensions=['lr', 'batch_history'])
    assert result['aggregate']['seeds'] == 2
    assert result['validation']['unique_prediction_tokens'] == 10
    assert result['validation']['repeated_prediction_tokens_per_condition'] == 20
    assert (tmp_path / 'replicated/replication.json').is_file()


def test_identical_model_outcomes_with_distinct_checkpoints_are_valid(pairs, tmp_path):
    from tnlm.replication import compare_seed_pairs

    first = json.loads(pairs[0]['left'].read_text())

    def same_model(value):
        model_hash = first['model_sha256']
        value['model_sha256'] = model_hash
        value['frozen_state']['model_before'] = model_hash
        value['frozen_state']['model_after'] = model_hash

    rewrite(pairs[1]['left'], same_model)
    result = compare_seed_pairs(pairs, tmp_path / 'identical-outcomes', allowed_between_dimensions=['lr', 'batch_history'])
    assert result['aggregate']['seeds'] == 2
    assert result['per_seed'][0]['left']['model_sha256'] == result['per_seed'][1]['left']['model_sha256']
    assert result['per_seed'][0]['left']['checkpoint_state_sha256'] != result['per_seed'][1]['left']['checkpoint_state_sha256']


def test_hidden_architecture_difference_requires_explicit_config_path(pairs, tmp_path):
    from tnlm.replication import compare_seed_pairs

    for pair in pairs:
        rewrite(pair['right'], lambda value: value['training_descriptor']['configuration']['model'].update(d_mlp=2048))
    with pytest.raises(ValueError):
        compare_seed_pairs(pairs, tmp_path / 'undeclared', allowed_between_dimensions=['lr', 'batch_history'])
    assert not (tmp_path / 'undeclared').exists()
    result = compare_seed_pairs(
        pairs, tmp_path / 'declared',
        allowed_between_dimensions=['lr', 'batch_history', 'config.model.d_mlp'],
    )
    assert result['aggregate']['seeds'] == 2
