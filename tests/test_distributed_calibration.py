"""Local versus explicitly reduced calibration with an initialized-group sentinel.

These tests exercise collective selection and sufficient-statistic mathematics;
actual transport/FSDP coordination belongs to the separate multiprocess tests.
"""
import copy
import time

import torch
import pytest

from tnlm.calibration import calibrate_lagged
from tnlm.model import TensorLM
from tnlm.normalization import lagged_modules


def calibrated_model(config):
    config.model.normalization = 'lagged'
    config.model.final_normalization = 'lagged'
    torch.manual_seed(19)
    return TensorLM(config.model)


def test_calibration_default_stays_local_with_initialized_group(config, monkeypatch):
    original = calibrated_model(config)
    baseline = copy.deepcopy(original)
    expected = calibrate_lagged(baseline, config, config.raw['data']['manifest'], torch.device('cpu'),
                                deadline=time.monotonic() + 60)
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)

    def forbidden(*args, **kwargs):
        raise AssertionError('default calibration unexpectedly entered a collective')

    monkeypatch.setattr(torch.distributed, 'all_reduce', forbidden)
    actual = calibrate_lagged(original, config, config.raw['data']['manifest'], torch.device('cpu'),
                              deadline=time.monotonic() + 60)
    assert actual == expected
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(original.state_dict()[name], tensor, rtol=0, atol=0)
    assert all(not norm.collecting and norm._valid is None for norm in lagged_modules(original))


def test_explicit_distributed_calibration_reduces_paired_statistics_once_per_site(config, monkeypatch):
    model = calibrated_model(config)
    baseline = copy.deepcopy(model)
    expected = calibrate_lagged(baseline, config, config.raw['data']['manifest'], torch.device('cpu'),
                                deadline=time.monotonic() + 60)
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    calls = []

    def identical_peer_sum(stats, *args, **kwargs):
        assert stats.dtype == torch.float64 and stats.shape == (2,)
        assert stats[1] > 0
        calls.append(stats.clone())
        stats.mul_(2)  # independent algebra: identical peer doubles sum AND count

    monkeypatch.setattr(torch.distributed, 'all_reduce', identical_peer_sum)
    actual = calibrate_lagged(model, config, config.raw['data']['manifest'], torch.device('cpu'),
                              deadline=time.monotonic() + 60, distributed=True)
    assert len(calls) == len(lagged_modules(model))
    assert actual['tokens'] == expected['tokens'] == 8
    assert actual['batch_hashes'] == expected['batch_hashes']
    for norm, reference, stats in zip(lagged_modules(model), lagged_modules(baseline), calls):
        torch.testing.assert_close(norm.mean_sq, stats[0] / stats[1], rtol=0, atol=0)
        torch.testing.assert_close(norm.mean_sq, reference.mean_sq, rtol=0, atol=0)
        assert norm.acc_count == norm.acc_sum == 0


def mock_collective_control(monkeypatch, model, *, failure=None, disagreement=False):
    import torch.distributed.fsdp as fsdp
    import tnlm.distributed_checkpoint as checkpoint
    events, commits = [], []
    monkeypatch.setattr(fsdp, 'FSDPModule', TensorLM)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    model.reset_iter_state = lambda: events.append(('reset', None))
    def phase(label, callback):
        events.append(('phase', label))
        value = callback()
        if label == 'calibration site commit/divisor':
            commits.append(value)
        if failure == 'commit' and len(commits) == 2 and label == 'calibration site commit/divisor':
            raise ValueError('simulated peer failure after site commit')
        if failure == 'output' and label == 'calibration verification output validation':
            raise ValueError('simulated peer invalid verification output')
        return value
    def agree(label, value):
        events.append(('agree', label))
        if disagreement and label == 'calibration batch identity':
            raise ValueError('simulated peer batch identity mismatch')
    monkeypatch.setattr(checkpoint, '_phase', phase)
    monkeypatch.setattr(checkpoint, '_agree', agree)
    return events


def assert_restored(model, before, modes):
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert [module.training for module in model.modules()] == modes
    assert all(not norm.collecting and norm._valid is None and norm.acc_sum == norm.acc_count == 0
               for norm in lagged_modules(model))
    assert all(not module._forward_hooks for module in model.modules())


def test_collective_control_agrees_before_forward_and_preserves_serial_statistics(config, monkeypatch):
    model = calibrated_model(config)
    model.blocks[0].mlp.eval()  # preserve mixed module modes, not only root.training
    modes = [module.training for module in model.modules()]
    baseline = copy.deepcopy(model)
    expected = calibrate_lagged(baseline, config, config.raw['data']['manifest'], torch.device('cpu'),
                                deadline=time.monotonic() + 60)
    events = mock_collective_control(monkeypatch, model)
    def forbidden(*args, **kwargs):
        raise AssertionError('replicated prefix must not reduce statistics')
    monkeypatch.setattr(torch.distributed, 'all_reduce', forbidden)
    report = calibrate_lagged(model, config, config.raw['data']['manifest'], torch.device('cpu'),
                              deadline=time.monotonic() + 60, collective_control=True)
    assert report['tokens'] == expected['tokens']
    assert report['sites'] == expected['sites']
    assert report['batch_hashes'] == expected['batch_hashes']
    assert report['prefix_replication_world_size'] == 2
    assert report['aggregate_forward_token_exposures'] == report['forward_token_exposures'] * 2
    for position, event in enumerate(events):
        if event == ('phase', 'calibration stopped site forward'):
            assert events[position - 1] == ('agree', 'calibration batch identity')
    assert sum(event[0] == 'reset' for event in events) == len(report['batch_hashes']) * (len(lagged_modules(model)) + 1)
    assert [module.training for module in model.modules()] == modes
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize('failure', ['commit', 'output', 'identity', 'deadline'])
def test_collective_calibration_failure_rolls_back_buffers_hooks_and_modes(config, monkeypatch, failure):
    model = calibrated_model(config)
    model.blocks[0].mlp.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    events = mock_collective_control(monkeypatch, model, failure=failure, disagreement=failure == 'identity')
    deadline = time.monotonic() - 1 if failure == 'deadline' else time.monotonic() + 60
    with pytest.raises((ValueError, TimeoutError), match='simulated|runtime cap'):
        calibrate_lagged(model, config, config.raw['data']['manifest'], torch.device('cpu'),
                         deadline=deadline, collective_control=True)
    assert_restored(model, before, modes)
    if failure in ('identity', 'deadline'):
        assert not any(event[0] == 'reset' for event in events), 'entered forward before agreed preflight'


@pytest.mark.parametrize('invalid', ['double_statistics', 'pending_gradient', 'missing_reset', 'pending_statistics'])
def test_collective_calibration_rejects_unsupported_or_pending_state(config, monkeypatch, invalid):
    model = calibrated_model(config)
    mock_collective_control(monkeypatch, model)
    if invalid == 'pending_gradient':
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
    elif invalid == 'missing_reset':
        model.reset_iter_state = None
    elif invalid == 'pending_statistics':
        lagged_modules(model)[0].acc_sum.fill_(3.)
    expected = {'double_statistics': 'distributed=False', 'pending_gradient': 'cleared parameter gradients',
                'missing_reset': 'reset_iter_state', 'pending_statistics': 'pending statistic'}[invalid]
    with pytest.raises(ValueError, match=expected):
        calibrate_lagged(model, config, config.raw['data']['manifest'], torch.device('cpu'),
                         deadline=time.monotonic() + 60, collective_control=True,
                         distributed=invalid == 'double_statistics')
    if invalid == 'pending_gradient':
        assert torch.equal(parameter.grad, torch.ones_like(parameter))
    if invalid == 'pending_statistics':
        assert lagged_modules(model)[0].acc_sum == 3.


def test_collective_mode_requires_group_before_phase_helpers(config, monkeypatch):
    import tnlm.distributed_checkpoint as checkpoint
    model = calibrated_model(config)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: False)
    def forbidden(*args, **kwargs):
        raise AssertionError('phase helper used before process group validation')
    monkeypatch.setattr(checkpoint, '_phase', forbidden)
    with pytest.raises(ValueError, match='initialized process group'):
        calibrate_lagged(model, config, config.raw['data']['manifest'], torch.device('cpu'),
                         deadline=time.monotonic() + 60, collective_control=True)


def test_unexpected_forward_error_resets_iteration_and_rolls_back(config, monkeypatch):
    model = calibrated_model(config)
    events = mock_collective_control(monkeypatch, model)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    def fail_forward(*args):
        raise ValueError('simulated forward failure')
    handle = model.register_forward_pre_hook(fail_forward)
    try:
        with pytest.raises(ValueError, match='simulated forward failure'):
            calibrate_lagged(model, config, config.raw['data']['manifest'], torch.device('cpu'),
                             deadline=time.monotonic() + 60, collective_control=True)
    finally:
        handle.remove()
    assert sum(event[0] == 'reset' for event in events) == 1
    assert_restored(model, before, modes)
