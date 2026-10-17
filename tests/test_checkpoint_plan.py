import copy
import json
import random

import numpy as np
import pytest
import torch

from tnlm.checkpoint_plan import checkpoint_retention_policy, checkpoint_storage_plan
from tnlm.config import Config, ModelConfig, TrainConfig


def cfg(**kwargs):
    return Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                  d_mlp=12, vocab_size=16, padded_vocab=16, attention_backend='reference', **kwargs),
                  train=TrainConfig(global_prediction_tokens_per_update=1_000_000), raw={})


def test_retention_reachable_and_branch_dedup():
    config = cfg()
    plan = checkpoint_storage_plan(config, token_limit=10_000_000, max_steps=2)
    assert plan['count_breakdown'] == {'initial': 1, 'final': 1, 'milestones': 2,
            'separate_90_percent_branch': 0, 'rolling_recovery_including_staging': 2}
    # 90% of this horizon equals an existing 100M milestone.
    limit = 1_000_000_000
    serial = checkpoint_storage_plan(config, token_limit=limit)
    distributed = checkpoint_storage_plan(config, token_limit=limit, distributed=True, distributed_world_size=2)
    assert serial['count_breakdown']['milestones'] == 16
    assert distributed['count_breakdown']['milestones'] == 16
    assert distributed['count_breakdown']['separate_90_percent_branch'] == 1
    cooled = checkpoint_storage_plan(config, token_limit=10_000_000, cooldown=True)
    regular = checkpoint_storage_plan(config, token_limit=10_000_000)
    assert regular['count_breakdown']['milestones'] == cooled['count_breakdown']['milestones'] + 1


def test_resume_and_zero_update_counts_and_huge_horizon():
    config = cfg()
    plan = checkpoint_storage_plan(config, start_tokens=2_000_000, token_limit=4_000_001, max_steps=1)
    assert plan['reachable_tokens'] == 3_000_000
    assert plan['count_breakdown']['milestones'] == 0
    assert plan['count_breakdown']['rolling_recovery_including_staging'] == 1
    plan = checkpoint_storage_plan(config, max_steps=0)
    assert plan['checkpoint_count'] == 2
    plan = checkpoint_storage_plan(config, token_limit=10**40)
    assert plan['count_breakdown']['milestones'] == 10**32 + 6
    json.dumps(plan, allow_nan=False)


def test_meta_no_real_allocation_or_rng_mutation(monkeypatch):
    from tnlm.model import TensorLM
    original = TensorLM.__init__
    def checked(self, *args, **kwargs):
        assert torch.empty(0).device.type == 'meta'
        original(self, *args, **kwargs)
        assert all(p.device.type == 'meta' for p in self.parameters())
    monkeypatch.setattr(TensorLM, '__init__', checked)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    checkpoint_storage_plan(cfg(normalization='spectral_rms'))
    assert random.getstate() == python_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_ties_dtype_muon_and_export_estimate():
    untied = checkpoint_storage_plan(cfg(), max_steps=0)
    tied = checkpoint_storage_plan(cfg(tie_embeddings=True), max_steps=0)
    assert untied['tensor_bytes_breakdown']['parameters'] - tied['tensor_bytes_breakdown']['parameters'] == 16 * 8 * 4
    config = cfg()
    config.train.optimizer = 'muon'
    muon = checkpoint_storage_plan(config, max_steps=0)
    assert muon['tensor_bytes_breakdown']['optimizer_moments'] < untied['tensor_bytes_breakdown']['optimizer_moments']
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        double = checkpoint_storage_plan(cfg(), max_steps=0)
    finally:
        torch.set_default_dtype(old)
    assert double['tensor_bytes_breakdown']['parameters'] == 2 * untied['tensor_bytes_breakdown']['parameters']
    exported = checkpoint_storage_plan(cfg(), max_steps=0, export_weights=True)
    assert exported['required_additional_bytes'] == untied['required_additional_bytes'] + exported['export_weights_estimate_bytes']


@pytest.mark.parametrize('arguments', [{'start_tokens': -1}, {'token_limit': float('inf')},
    {'max_steps': True}, {'distributed_world_size': 0}, {'distributed_world_size': 2}, {'cooldown': 1}])
def test_invalid_numeric_arguments(arguments):
    with pytest.raises(ValueError):
        checkpoint_storage_plan(cfg(), **arguments)


def test_custom_milestones_empty_duplicate_and_branch_policy():
    from tnlm.checkpoint_plan import pending_milestone_tokens
    config = cfg()
    config.raw = {'project': {'milestone_tokens': []}}
    serial = checkpoint_storage_plan(config, token_limit=20_000_000)
    assert serial['count_breakdown']['milestones'] == 1  # mandatory 90%
    distributed = checkpoint_storage_plan(config, token_limit=20_000_000, distributed=True, distributed_world_size=2)
    assert distributed['count_breakdown']['milestones'] == 0
    assert distributed['count_breakdown']['separate_90_percent_branch'] == 1
    assert list(pending_milestone_tokens(config, start_tokens=0, token_limit=20)) == []
    assert list(pending_milestone_tokens(config, start_tokens=0, token_limit=20, include_branch=True)) == [18]
    config.raw['project']['milestone_tokens'] = [7, 14, 14]
    plan = checkpoint_storage_plan(config, token_limit=16)
    assert plan['milestone_policy'] == {'mode': 'explicit', 'tokens': [7, 14]}
    assert plan['count_breakdown']['milestones'] == 2  # 90%=14 deduplicates
    assert list(pending_milestone_tokens(config, start_tokens=7, token_limit=16, include_branch=True)) == [14]
    assert list(pending_milestone_tokens(config, start_tokens=0, token_limit=16, reachable_tokens=10)) == [7]
    distributed = checkpoint_storage_plan(config, token_limit=16, distributed=True, distributed_world_size=2)
    assert distributed['count_breakdown']['milestones'] == 2
    assert distributed['count_breakdown']['separate_90_percent_branch'] == 1


@pytest.mark.parametrize('values', [None, (), [True], [0], [-1], [1.5], ['7']])
def test_invalid_custom_milestone_policy(values):
    from tnlm.checkpoint_plan import pending_milestone_tokens
    config = cfg()
    config.raw = {'project': {'milestone_tokens': values}}
    with pytest.raises(ValueError):
        checkpoint_storage_plan(config, max_steps=0)
    with pytest.raises(ValueError):
        pending_milestone_tokens(config, start_tokens=0, token_limit=10)


def test_default_pending_schedule_is_lazy_and_matches_planner():
    from itertools import islice
    from tnlm.checkpoint_plan import pending_milestone_tokens
    config = cfg()
    values = pending_milestone_tokens(config, start_tokens=0, token_limit=10**40)
    assert list(islice(values, 8)) == [1_000_000, 2_000_000, 4_000_000, 8_000_000,
                                     16_000_000, 32_000_000, 100_000_000, 200_000_000]
    tokens = list(pending_milestone_tokens(config, start_tokens=0, token_limit=200_000_000,
                                          reachable_tokens=123_000_000, include_branch=True))
    plan = checkpoint_storage_plan(config, token_limit=200_000_000, max_steps=123)
    assert len(tokens) == plan['count_breakdown']['milestones']


def test_distributed_ties_count_model_fqns_but_unique_optimizer_state():
    serial = checkpoint_storage_plan(cfg(tie_embeddings=True), max_steps=0, export_weights=True)
    distributed = checkpoint_storage_plan(cfg(tie_embeddings=True), max_steps=0, export_weights=True,
                                          distributed=True, distributed_world_size=2)
    assert distributed['tensor_bytes_breakdown']['parameters'] == serial['tensor_bytes_breakdown']['parameters'] + 16 * 8 * 4
    assert distributed['tensor_bytes_breakdown']['optimizer_moments'] == serial['tensor_bytes_breakdown']['optimizer_moments']
    assert distributed['export_weights_estimate_bytes'] == serial['export_weights_estimate_bytes']


def test_retention_defaults_preserve_existing_behavior():
    config = cfg()
    policy = checkpoint_retention_policy(config)
    assert policy == {'keep_recovery_checkpoints': 2, 'save_initial_checkpoint': True}
    plan = checkpoint_storage_plan(config)
    assert plan['checkpoint_retention_policy'] == policy
    assert plan['count_breakdown']['initial'] == 1
    assert plan['count_breakdown']['rolling_recovery_including_staging'] == 3


@pytest.mark.parametrize('distributed', [False, True])
@pytest.mark.parametrize('keep', [0, 1, 2, 5])
@pytest.mark.parametrize('save_initial', [False, True])
def test_screening_retention_plans_preserve_required_checkpoints(distributed, keep, save_initial):
    config = cfg()
    config.raw = {'project': {'milestone_tokens': [], 'keep_recovery_checkpoints': keep,
                              'save_initial_checkpoint': save_initial}}
    options = {'distributed': distributed, 'distributed_world_size': 2 if distributed else 1}
    plan = checkpoint_storage_plan(config, token_limit=10_000_000, **options)
    counts = plan['count_breakdown']
    assert counts['initial'] == int(save_initial)
    assert counts['final'] == 1
    assert counts['rolling_recovery_including_staging'] == (keep + 1 if keep else 0)
    assert counts['milestones'] + counts['separate_90_percent_branch'] == 1
    assert plan['required_additional_bytes'] == plan['checkpoint_count'] * plan['per_checkpoint_estimate_bytes']
    # A cooldown has no separate branch, but still retains its final endpoint.
    cooled = checkpoint_storage_plan(config, token_limit=10_000_000, cooldown=True, **options)
    assert cooled['checkpoint_count'] == plan['checkpoint_count'] - 1


@pytest.mark.parametrize('keep', [0, 1, 2])
@pytest.mark.parametrize('updates', [0, 1, 2])
def test_screening_recovery_staging_limited_by_reachable_updates(keep, updates):
    config = cfg()
    config.raw = {'project': {'keep_recovery_checkpoints': keep, 'save_initial_checkpoint': False,
                              'milestone_tokens': []}}
    plan = checkpoint_storage_plan(config, max_steps=updates)
    assert plan['count_breakdown']['rolling_recovery_including_staging'] == (
        min(updates, keep + 1) if keep else 0)
    assert plan['count_breakdown']['initial'] == 0
    assert plan['count_breakdown']['final'] == 1


@pytest.mark.parametrize('field,value', [
    ('keep_recovery_checkpoints', value) for value in [-1, True, False, 1.0, '1', None]
] + [('save_initial_checkpoint', value) for value in [0, 1, 'false', None]])
def test_screening_retention_strict_validation_before_model_construction(field, value, monkeypatch):
    config = cfg()
    config.raw = {'project': {field: value}}
    def unexpected_model(*args, **kwargs):
        raise AssertionError('invalid retention must fail before model construction')
    monkeypatch.setattr('tnlm.checkpoint_plan.TensorLM.__init__', unexpected_model)
    with pytest.raises(ValueError, match=f'project.{field}'):
        checkpoint_retention_policy(config)
    with pytest.raises(ValueError, match=f'project.{field}'):
        checkpoint_storage_plan(config, max_steps=0)
