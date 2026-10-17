import copy

import pytest

from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.search import plan_sweep


def base(tmp_path):
    return Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                  d_mlp=12, vocab_size=16, padded_vocab=16, attention_backend='reference'),
                  train=TrainConfig(rung_total_tokens=1000), raw={'project': {
                  'spend_ledger': str(tmp_path / 'absent' / 'ledger.json'),
                  'gpu_hourly_rate_usd': 2., 'measured_prediction_tokens_per_second': 100.,
                  'estimated_setup_seconds': 5.}, 'data': {'context': 8}})


def test_shapes_aggregate_stable_hashes_and_no_mutation_or_writes(tmp_path):
    cfg = base(tmp_path)
    candidates = [{'name': 'small', 'overrides': {}}, {'name': 'wide', 'overrides': {
        'model': {'d_model': 16, 'heads': 2, 'd_key': 8, 'd_value': 8, 'd_mlp': 24},
        'train': {'lr': .001}, 'data': {'context': 16}}}]
    before_cfg, before_candidates = copy.deepcopy(cfg.to_dict()), copy.deepcopy(candidates)
    result = plan_sweep(cfg, candidates)
    assert result == plan_sweep(cfg, candidates)
    assert cfg.to_dict() == before_cfg and candidates == before_candidates
    assert list(tmp_path.iterdir()) == []
    runs = result['runs']
    assert runs[0]['config_sha256'] != runs[1]['config_sha256']
    assert runs[1]['config']['model']['heads'] == 2
    assert runs[1]['config']['data']['context'] == 16
    assert result['required_additional_bytes'] == sum(run['plan']['checkpoint_storage_plan']['required_additional_bytes'] for run in runs)
    assert runs[1]['plan']['checkpoint_storage_plan']['per_checkpoint_estimate_bytes'] > runs[0]['plan']['checkpoint_storage_plan']['per_checkpoint_estimate_bytes']
    assert result['projected_usd'] == pytest.approx(2 * 15 * 2 / 3600)
    assert result['cost_projection_complete']
    # Candidate names identify runs but do not alter resolved config identity.
    renamed = plan_sweep(cfg, [{'name': 'renamed', 'overrides': {}}])
    assert renamed['runs'][0]['config_sha256'] == runs[0]['config_sha256']


def test_incomplete_cost_is_not_invented(tmp_path):
    cfg = base(tmp_path)
    result = plan_sweep(cfg, [{'name': 'priced', 'overrides': {}},
        {'name': 'unknown', 'overrides': {'project': {'gpu_hourly_rate_usd': None}}}])
    assert result['projected_usd'] is None and not result['cost_projection_complete']
    assert result['unpriced_candidates'] == ['unknown']
    assert result['known_projected_usd'] == pytest.approx(15 * 2 / 3600)
    assert result['projected_seconds'] == 30


@pytest.mark.parametrize('name', ['../bad', '.', '..', '/abs', 'a/b', 'a\\b', '', 'a b'])
def test_unsafe_names_reject(tmp_path, name):
    with pytest.raises(ValueError, match='safe path'):
        plan_sweep(base(tmp_path), [{'name': name, 'overrides': {}}])


@pytest.mark.parametrize('candidates', [[], [{'name': 'same', 'overrides': {}}] * 2,
    [{'name': 'a', 'overrides': {}, 'execute': True}],
    [{'name': 'a', 'overrides': {'typo': {}}}],
    [{'name': 'a', 'overrides': {'model': {'typo': 4}}}],
    [{'name': 'a', 'overrides': {'train': {'max_steps': 4}}}],
    [{'name': 'a', 'overrides': {'model': {'d_model': 7}}}]])
def test_invalid_candidates_reject(tmp_path, candidates):
    with pytest.raises(ValueError):
        plan_sweep(base(tmp_path), candidates)
    assert list(tmp_path.iterdir()) == []
