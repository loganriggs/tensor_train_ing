import copy
import math
from pathlib import Path

import pytest
import torch

from helpers import state
from tnlm.runtime import full_config, resume_validator
from tnlm.schedules import schedule_multiplier, validate_schedule
from tnlm.train import plan_search, train_run


def test_cosine_independent_endpoints_warmup_decay_and_no_restart():
    n = 1000
    w = validate_schedule('cosine', n)
    assert w == 100
    values = [schedule_multiplier(j, n, schedule='cosine') for j in range(n)]
    assert values[:w] == [(j + 1) / w for j in range(w)]
    assert values[w - 1] == 1. and values[-1] == 0.
    assert all(a >= b for a, b in zip(values[w - 1:], values[w:]))
    for j in [w, 500, n - 2]:
        assert values[j] == pytest.approx(math.cos(math.pi * (j + 1 - w) / (2 * (n - w))) ** 2)
    with pytest.raises(ValueError, match='exhausted'):
        schedule_multiplier(n, n, schedule='cosine')
    assert validate_schedule('cosine', 10001) == 101


def test_constant_preserves_saved_warmup_after_promotion():
    assert schedule_multiplier(49, 20000, warmup_steps=100) == .5
    assert schedule_multiplier(199, 20000, warmup_steps=100) == 1.
    for n in [1, 50, 1000, 20000]:
        for step in [0, 49, 99, 500]:
            assert schedule_multiplier(step, n) == min(1., (step + 1) / max(100, math.ceil(.01 * n)))


@pytest.mark.parametrize('schedule,n', [('unknown', 1000), ('cosine', 100), ('cosine', 1)])
def test_invalid_schedule_planning_before_writes(config, tmp_path, schedule, n):
    config.train.schedule = schedule
    config.train.rung_total_tokens = n * config.train.global_prediction_tokens_per_update
    with pytest.raises(ValueError):
        plan_search(config)
    with pytest.raises(ValueError):
        train_run(config, tmp_path / 'bad', max_steps=1, max_seconds=60, device='cpu')
    assert not (tmp_path / 'bad').exists() and not (tmp_path / 'spend.json').exists()


@pytest.mark.parametrize('optimizer_kind', ['adamw', 'muon'])
@pytest.mark.slow
def test_real_cpu_cosine_global_resume_final_partial_and_precharge_rejections(config, tmp_path, optimizer_kind, monkeypatch):
    config.train.schedule = 'cosine'
    config.train.rung_total_tokens = 713  # 102 updates of7; final update has6targets.
    config.train.optimizer = optimizer_kind
    config.train.aux_adamw_lr = .003
    config.model.width_scaling_base = 4 if optimizer_kind == 'adamw' else None
    config.raw['project']['milestone_tokens'] = []
    whole = train_run(config, tmp_path / 'whole', max_steps=102, max_seconds=120, device='cpu')
    first = train_run(config, tmp_path / 'first', max_steps=100, max_seconds=120, device='cpu')
    resumed = train_run(config, tmp_path / 'resumed', max_steps=2, max_seconds=120,
                        device='cpu', resume=first['checkpoint'])
    a, b = state(Path(whole['checkpoint'])), state(Path(resumed['checkpoint']))
    assert a['progress'] == b['progress']
    assert a['sampler'] == b['sampler']
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])
    for name in a['model']:
        assert torch.equal(a['model'][name], b['model'][name])
    def exact(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                exact(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for lvalue, rvalue in zip(left, right):
                exact(lvalue, rvalue)
        else:
            assert left == right
    exact(a['optimizer'], b['optimizer'])
    assert resumed['history'][0]['lr'] == first['history'][-1]['lr'] * .5
    assert resumed['history'][-1]['lr'] == 0.
    assert whole['history'][-1]['prediction_tokens'] == 6
    groups = b['optimizer']['param_groups'] if optimizer_kind == 'adamw' else b['optimizer']['muon']['param_groups'] + b['optimizer']['adamw']['param_groups']
    assert all(group['lr'] == 0 for group in groups)
    assert groups[0]['base_lr'] != groups[1]['base_lr']
    assert whole['training_descriptor']['schedule'] == 'cosine'
    assert whole['training_descriptor']['completion_status'] == 'complete'
    assert not list((tmp_path / 'whole').glob('milestone-*'))
    ledger_before = (tmp_path / 'spend.json').read_bytes()
    with pytest.raises(ValueError, match='separate cooldown'):
        train_run(config, tmp_path / 'cooldown', max_steps=2, max_seconds=60, device='cpu',
                  resume=first['checkpoint'], cooldown_steps=2, cooldown_end_tokens=713)
    config.train.rung_total_tokens = 714  # Same ceil(D/B), still a different frozen horizon.
    with pytest.raises(ValueError, match='original token horizon'):
        train_run(config, tmp_path / 'promotion', max_steps=2, max_seconds=60, device='cpu', resume=first['checkpoint'])
    assert (tmp_path / 'spend.json').read_bytes() == ledger_before
    assert not (tmp_path / 'promotion').exists() and not (tmp_path / 'cooldown').exists()
    # The original constant schedule still permits upward rung promotion.
    config.train.schedule = 'constant_with_separate_cooldown'
    saved = full_config(config)
    config.train.rung_total_tokens += 7
    resume_validator(config)(saved)
    # The early compatibility path must verify corruption before unpickling.
    config.train.schedule = 'cosine'
    config.train.rung_total_tokens = 713
    with (Path(first['checkpoint']) / 'state.pt').open('ab') as stream:
        stream.write(b'corrupt')
    def forbidden_load(*args, **kwargs):
        raise AssertionError('corrupt state reached deserialization')
    monkeypatch.setattr(torch, 'load', forbidden_load)
    with pytest.raises(ValueError, match='checksum'):
        train_run(config, tmp_path / 'corrupt-resume', max_steps=1, max_seconds=60,
                  device='cpu', resume=first['checkpoint'])
    assert (tmp_path / 'spend.json').read_bytes() == ledger_before


def test_cosine_dry_run_resume_placeholder_does_not_read_checkpoint(config, tmp_path, monkeypatch):
    import tnlm.distributed_train as distributed
    config.train.schedule = 'cosine'
    config.train.rung_total_tokens = 713
    config.raw['project']['dry_run'] = True
    missing = tmp_path / 'missing'
    assert train_run(config, tmp_path / 'serial-dry', max_steps=1, max_seconds=60, device='cpu', resume=missing)['dry_run']
    monkeypatch.setattr(distributed, '_group', lambda: (0, 2))
    monkeypatch.setattr(distributed, '_phase', lambda label, callback: callback())
    monkeypatch.setattr(distributed, '_agree', lambda *args: None)
    result = distributed.distributed_train_run(config, tmp_path / 'distributed-dry',
        max_steps=1, max_seconds=60, device='cpu', resume=missing)
    assert result['dry_run']
    assert not (tmp_path / 'serial-dry').exists() and not (tmp_path / 'distributed-dry').exists()
