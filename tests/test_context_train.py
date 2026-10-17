"""Explicit serial context branches preserve training state and consumed-stream masks."""
import copy
import json

import numpy as np
import pytest
import torch

from helpers import state
from tnlm.data import PackedSampler
from tnlm.evaluate import _training_descriptor
from tnlm.train import train_run


def run(cfg, path, **kwargs):
    return train_run(cfg, path, max_seconds=60, device='cpu', **kwargs)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('optimizer,normalization', [
    ('adamw', 'pre_rms'), ('muon', 'pre_rms'), ('adamw', 'lagged'),
])
def test_context_branch_exact_resume(config, tmp_path, optimizer, normalization):
    config.train.optimizer = optimizer
    config.model.normalization = normalization
    if normalization == 'lagged':
        config.model.final_normalization = 'lagged'
    parent = run(config, tmp_path / 'parent', max_steps=1)
    old = state(tmp_path / 'parent/final')
    longer = copy.deepcopy(config)
    longer.raw['data']['context'] = 9
    whole = run(longer, tmp_path / 'whole', max_steps=2,
                resume=parent['checkpoint'], context_fork=True)
    first = run(longer, tmp_path / 'first', max_steps=1,
                resume=parent['checkpoint'], context_fork=True)
    resumed = run(longer, tmp_path / 'resumed', max_steps=1,
                  resume=first['checkpoint'])
    a, b = state(tmp_path / 'whole/final'), state(tmp_path / 'resumed/final')
    for key in ('model', 'optimizer', 'rng', 'sampler', 'progress'):
        equal(a[key], b[key])
    pending_resume = run(longer, tmp_path / 'pending-resumed', max_steps=2,
                         resume=str(tmp_path / 'whole/initial'))
    pending = state(tmp_path / 'pending-resumed/final')
    for key in ('model', 'optimizer', 'rng', 'sampler', 'progress'):
        equal(a[key], pending[key])
    assert whole['training_descriptor']['context_history_status'] == 'complete'
    equal(whole['training_descriptor']['context_history'], _training_descriptor(a)['context_history'])
    equal(pending_resume['training_descriptor']['context_history'], _training_descriptor(a)['context_history'])
    initial = state(tmp_path / 'whole/initial')
    for key in ('model', 'optimizer', 'rng'):
        equal(old[key], initial[key])
    assert initial['progress']['tokens'] == old['progress']['tokens']
    assert initial['progress']['warmup_steps'] == old['progress']['warmup_steps']
    assert initial['progress']['scheduler_step'] == old['progress']['scheduler_step']
    assert 'context_transition' in initial['sampler']
    sampler = PackedSampler(longer.raw['data']['manifest'], context=9,
                            batch_size=1, seed=longer.train.seed)
    sampler.load_state_dict(initial['sampler'])
    targets = sampler.next_batch()[2]
    assert (targets[0, :8] == -100).all()
    assert targets[0, 8] != -100
    assert a['progress']['context_fork_exposure']['masked_prediction_targets'] == 8
    assert a['progress']['context_fork_exposure']['prefix_tokens_at_or_before_watermark'] > 0
    assert len(_training_descriptor(a)['context_history']) == 2
    assert a['progress']['context_history'][0]['context'] == 4
    assert a['progress']['context_history'][-1]['context'] == 9
    if normalization == 'lagged':
        equal(old['progress']['calibration'], initial['progress']['calibration'])


@pytest.mark.parametrize('change', ['same', 'shrink', 'microbatch', 'ordinary'])
def test_invalid_context_changes_reject_before_billing(config, tmp_path, change):
    parent = run(config, tmp_path / 'parent', max_steps=1)
    before = (tmp_path / 'spend.json').read_bytes()
    cfg = copy.deepcopy(config)
    cfg.raw['data']['context'] = 9
    if change == 'same':
        cfg.raw['data']['context'] = 4
    elif change == 'shrink':
        cfg.raw['data']['context'] = 3
    elif change == 'microbatch':
        cfg.raw['data']['micro_batch_size'] = 2
    with pytest.raises(ValueError):
        run(cfg, tmp_path / 'bad', max_steps=1, resume=parent['checkpoint'],
            context_fork=change != 'ordinary')
    assert not (tmp_path / 'bad').exists()
    assert (tmp_path / 'spend.json').read_bytes() == before


@pytest.mark.parametrize('flag,resume', [(1, 'missing'), (True, None), ('yes', None)])
def test_invalid_flag_before_dry_run(config, tmp_path, flag, resume):
    config.raw['project']['dry_run'] = True
    with pytest.raises(ValueError, match='context_fork'):
        run(config, tmp_path / 'bad', max_steps=1, resume=resume, context_fork=flag)
    assert not (tmp_path / 'bad').exists()
    assert not (tmp_path / 'spend.json').exists()


def test_context_fork_dry_run_does_not_read_placeholder(config, tmp_path):
    config.raw['project']['dry_run'] = True
    config.raw['data']['context'] = 9
    report = run(config, tmp_path / 'dry', max_steps=1,
                 resume='missing-parent', context_fork=True)
    assert report['context_fork'] is True
    assert report['new_context'] == 9
    assert report['parent'] == 'missing-parent'
    assert report['parent_validation'] == 'pending'
    assert not (tmp_path / 'dry').exists()


def test_legacy_resume_does_not_invent_fresh_context_history(config, tmp_path, monkeypatch):
    import tnlm.checkpoint as checkpoint

    original = checkpoint.save_checkpoint

    def legacy_save(*args, **kwargs):
        kwargs['progress'] = copy.deepcopy(kwargs['progress'])
        kwargs['progress'].pop('context_history', None)
        return original(*args, **kwargs)

    monkeypatch.setattr(checkpoint, 'save_checkpoint', legacy_save)
    parent = run(config, tmp_path / 'legacy', max_steps=1)
    monkeypatch.setattr(checkpoint, 'save_checkpoint', original)
    resumed = run(config, tmp_path / 'resumed', max_steps=1, resume=parent['checkpoint'])
    assert 'context_history' not in resumed['progress']
    assert 'context_fork_exposure' not in resumed['progress']


def test_cosine_context_fork_keeps_global_horizon_and_multiplier(config, tmp_path):
    from helpers import FixtureTokenizer
    from tnlm.data import prepare_data

    source = tmp_path / 'long.jsonl'
    source.write_text(json.dumps({'id': 'long', 'text': 'abcdefgh' * 200}) + '\n')
    manifest = prepare_data(
        tmp_path / 'long-data', source_jsonl=source, revision='offline-context-cosine',
        max_tokens=1000, tokenizer=FixtureTokenizer(),
        split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.},
    )
    config.model.vocab_size = config.model.padded_vocab = 128
    config.train.schedule = 'cosine'
    config.train.rung_total_tokens = 713
    config.raw['data']['manifest'] = str(manifest)
    config.raw['project']['milestone_tokens'] = []
    parent = run(config, tmp_path / 'parent', max_steps=99)
    longer = copy.deepcopy(config)
    longer.raw['data']['context'] = 9
    branch = run(longer, tmp_path / 'branch', max_steps=3,
                 resume=parent['checkpoint'], context_fork=True)
    assert branch['progress']['tokens'] == 713
    assert branch['progress']['scheduler_step'] == 102
    assert branch['progress']['warmup_steps'] == 100
    assert [row['optimizer_group_lrs'][0] for row in branch['history']] == pytest.approx([
        config.train.lr, .5 * config.train.lr, 0.,
    ])
    assert branch['training_descriptor']['schedule'] == 'cosine'
    assert branch['training_descriptor']['completion_status'] == 'complete'
    assert branch['training_descriptor']['context_history_status'] == 'complete'
    equal(branch['training_descriptor']['context_history'],
          _training_descriptor(state(tmp_path / 'branch/final'))['context_history'])
    promoted = copy.deepcopy(longer)
    promoted.train.rung_total_tokens += 1
    before = (tmp_path / 'spend.json').read_bytes()
    with pytest.raises(ValueError, match='horizon'):
        run(promoted, tmp_path / 'promoted', max_steps=1,
            resume=parent['checkpoint'], context_fork=True)
    assert (tmp_path / 'spend.json').read_bytes() == before


def test_context_fork_with_explicit_cooldown_endpoint(config, tmp_path):
    parent = run(config, tmp_path / 'parent', max_steps=1)
    longer = copy.deepcopy(config)
    longer.raw['data']['context'] = 9
    options = {'cooldown_steps': 3, 'cooldown_end_tokens': 25}
    whole = run(longer, tmp_path / 'whole', max_steps=3,
                resume=parent['checkpoint'], context_fork=True, **options)
    first = run(longer, tmp_path / 'first', max_steps=1,
                resume=parent['checkpoint'], context_fork=True, **options)
    resumed = run(longer, tmp_path / 'resumed', max_steps=2,
                  resume=first['checkpoint'], **options)
    a, b = state(tmp_path / 'whole/final'), state(tmp_path / 'resumed/final')
    for key in ('model', 'optimizer', 'rng', 'sampler', 'progress'):
        equal(a[key], b[key])
    assert whole['progress']['tokens'] == resumed['progress']['tokens'] == 25
    assert whole['progress']['scheduler_step'] == 3
    assert whole['progress']['cooldown_start_tokens'] == 7
    assert whole['training_descriptor']['completion_status'] == 'complete'
    assert whole['training_descriptor']['context_history_status'] == 'complete'
    equal(whole['training_descriptor']['context_history'], _training_descriptor(a)['context_history'])
    with pytest.raises(ValueError, match='endpoint'):
        run(longer, tmp_path / 'wrong-endpoint', max_steps=1,
            resume=first['checkpoint'], cooldown_steps=3, cooldown_end_tokens=26)
