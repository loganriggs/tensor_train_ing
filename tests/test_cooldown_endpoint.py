"""Exact token endpoints preserve partial-update and branch replay semantics."""
import pytest
import torch
from tnlm.train import train_run


def run(config, path, **kwargs):
    return train_run(config, path, device='cpu', max_seconds=60, **kwargs)


def test_exact_endpoint_whole_and_resumed(config, tmp_path):
    parent = run(config, tmp_path / 'parent', max_steps=2)
    options = dict(cooldown_steps=3, cooldown_end_tokens=32)
    whole = run(config, tmp_path / 'whole', resume=parent['checkpoint'], max_steps=3, **options)
    first = run(config, tmp_path / 'first', resume=parent['checkpoint'], max_steps=1, **options)
    resumed = run(config, tmp_path / 'resumed', resume=first['checkpoint'], max_steps=2, **options)
    assert whole['progress']['tokens'] == resumed['progress']['tokens'] == 32
    assert whole['progress']['cooldown_end_tokens'] == 32
    assert whole['history'][-1]['tokens'] == 32
    a = torch.load(tmp_path / 'whole/final/state.pt', weights_only=False)
    b = torch.load(tmp_path / 'resumed/final/state.pt', weights_only=False)
    assert a['progress'] == b['progress']
    for key in a['model']:
        assert torch.equal(a['model'][key], b['model'][key]), key
    for endpoint in (None, 33):
        with pytest.raises(ValueError, match='endpoint'):
            run(config, tmp_path / f'changed-{endpoint}', resume=first['checkpoint'], max_steps=1,
                cooldown_steps=3, cooldown_end_tokens=endpoint)


@pytest.mark.parametrize('endpoint', [True, 0, -1, 3.5, 28, 36])
def test_invalid_endpoint(config, tmp_path, endpoint):
    parent = run(config, tmp_path / 'parent', max_steps=2)
    with pytest.raises(ValueError, match='cooldown'):
        run(config, tmp_path / 'invalid', resume=parent['checkpoint'], max_steps=3,
            cooldown_steps=3, cooldown_end_tokens=endpoint)


def test_endpoint_requires_cooldown(config, tmp_path):
    with pytest.raises(ValueError, match='cooldown'):
        run(config, tmp_path / 'invalid', max_steps=1, cooldown_end_tokens=20)
