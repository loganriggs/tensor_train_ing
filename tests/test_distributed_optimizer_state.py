"""AdamW checkpoint state allocation must never execute a synthetic update."""
import copy

import pytest
import torch

from tnlm.distributed_checkpoint import _state


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize('amsgrad', [False, True])
@pytest.mark.parametrize('default_dtype', [torch.float32, torch.float64])
def test_state_allocation_preserves_weights_and_first_update(dtype, amsgrad, default_dtype, monkeypatch):
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        torch.manual_seed(81)
        model = torch.nn.Linear(3, 2, dtype=dtype)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, amsgrad=amsgrad)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.01, amsgrad=amsgrad)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        original_step = optimizer.step
        def forbidden(*args, **kwargs):
            raise AssertionError('state extraction must not call optimizer.step')
        monkeypatch.setattr(optimizer, 'step', forbidden)
        _state(model, optimizer)
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key]) and torch.isfinite(value).all()
        for parameter, values in optimizer.state.items():
            assert values['step'].dtype == default_dtype
            assert values['step'].device.type == 'cpu'
            assert float(values['step']) == 0
            for name in ('exp_avg', 'exp_avg_sq', *(['max_exp_avg_sq'] if amsgrad else [])):
                assert values[name].dtype == parameter.dtype
                assert values[name].stride() == parameter.stride()
                assert torch.count_nonzero(values[name]) == 0
        monkeypatch.setattr(optimizer, 'step', original_step)
        # Nonzero exactly representable gradients keep FP16's true first update
        # well-defined even with default epsilon below its representable range.
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            parameter.grad = torch.full_like(parameter, .5)
            expected.grad = torch.full_like(expected, .5)
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            assert torch.equal(actual, expected) and torch.isfinite(actual).all()
            for key in optimizer.state[actual]:
                assert torch.equal(optimizer.state[actual][key], reference_optimizer.state[expected][key])
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize('fused,capturable', [(False, False), (True, False), (False, True)])
def test_scalar_counter_allocation_rules(fused, capturable, monkeypatch):
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), fused=fused, capturable=capturable)
        monkeypatch.setattr(optimizer, 'step', lambda *a, **kw: pytest.fail('unexpected step'))
        _state(model, optimizer)
        for parameter, values in optimizer.state.items():
            assert values['step'].dtype == (torch.float32 if fused else torch.float64)
            assert values['step'].device == (parameter.device if fused or capturable else torch.device('cpu'))
    finally:
        torch.set_default_dtype(previous_dtype)


def test_partial_existing_state_and_frozen_parameters_preserved(monkeypatch):
    model = torch.nn.Linear(2, 2)
    extra = torch.nn.Parameter(torch.ones(2))
    model.register_parameter('extra', extra)
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    original = {key: value.clone() for key, value in optimizer.state[model.weight].items()}
    original_objects = dict(optimizer.state[model.weight])
    model.bias.requires_grad_(False)
    assert extra not in optimizer.state and model.bias not in optimizer.state
    monkeypatch.setattr(optimizer, 'step', lambda *a, **kw: pytest.fail('unexpected step'))
    _state(model, optimizer)
    for key, value in optimizer.state[model.weight].items():
        assert value is original_objects[key]
        assert torch.equal(value, original[key])
    assert float(optimizer.state[extra]['step']) == 0
    assert model.bias not in optimizer.state


@pytest.mark.parametrize('field,value', [('lr', float('nan')), ('weight_decay', float('inf')),
                                        ('eps', float('inf')), ('betas', (.9, float('nan')))])
def test_nonfinite_hyperparameters_rejected_before_allocation(field, value, monkeypatch):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.param_groups[0][field] = value
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    monkeypatch.setattr(optimizer, 'step', lambda *a, **kw: pytest.fail('unexpected step'))
    with pytest.raises(ValueError, match='nonfinite Python'):
        _state(model, optimizer)
    assert not optimizer.state
    assert all(torch.equal(tensor, before[key]) for key, tensor in model.state_dict().items())


def test_entirely_frozen_empty_optimizer_rejected_without_allocation(monkeypatch):
    model = torch.nn.Linear(2, 2).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters())
    monkeypatch.setattr(optimizer, 'step', lambda *a, **kw: pytest.fail('unexpected step'))
    with pytest.raises(ValueError, match='entirely frozen'):
        _state(model, optimizer)
    assert not optimizer.state


def _muon_fixture():
    from tnlm.config import ModelConfig
    from tnlm.model import TensorLM
    from tnlm.optim import MuonAdamW
    model = TensorLM(ModelConfig(layers=1, d_model=8, heads=1, d_key=4,
                                d_value=4, d_mlp=12, vocab_size=16, padded_vocab=16,
                                attention_backend='reference'))
    optimizer = MuonAdamW(model, lr=.01, aux_lr=.002, weight_decay=.1, betas=(.9, .95))
    return model, optimizer


def test_muon_children_no_synthetic_step_and_first_update(monkeypatch):
    model, optimizer = _muon_fixture()
    reference = copy.deepcopy(model)
    from tnlm.optim import MuonAdamW
    expected = MuonAdamW(reference, lr=.01, aux_lr=.002, weight_decay=.1, betas=(.9, .95))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    def forbidden(*args, **kwargs):
        raise AssertionError('no synthetic child or wrapper step')
    with monkeypatch.context() as patch:
        for opt in (optimizer, optimizer.muon, optimizer.adamw):
            patch.setattr(opt, 'step', forbidden)
        state = _state(model, optimizer)
    assert state['optimizer']['state']
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)
    for value in optimizer.muon.state.values():
        assert set(value) == {'momentum_buffer'}
        assert torch.count_nonzero(value['momentum_buffer']) == 0
    for actual, target in zip(model.parameters(), reference.parameters()):
        actual.grad = torch.full_like(actual, .25)
        target.grad = torch.full_like(target, .25)
    optimizer.step()
    expected.step()
    for actual, target in zip(model.parameters(), reference.parameters()):
        assert torch.equal(actual, target)
        for key, value in optimizer.state[actual].items():
            assert torch.equal(value, expected.state[target][key])


def test_muon_children_get_set_replays_and_preserves_partial_state():
    from torch.distributed.checkpoint.state_dict import set_state_dict
    from tnlm.distributed_checkpoint import _optimizer_children, _layout
    model, optimizer = _muon_fixture()
    # Initialize just one logical matrix, preserving its exact momentum object.
    parameter = optimizer.muon.param_groups[0]['params'][0]
    momentum = torch.full_like(parameter, .125)
    optimizer.muon.state[parameter]['momentum_buffer'] = momentum
    state = copy.deepcopy(_state(model, optimizer))
    assert optimizer.muon.state[parameter]['momentum_buffer'] is momentum
    restored, recovered = _muon_fixture()
    _state(restored, recovered)
    set_state_dict(restored, _optimizer_children(recovered), model_state_dict=state['model'],
                   optim_state_dict=state['optimizer'])
    for actual, target in zip(model.parameters(), restored.parameters()):
        actual.grad = torch.full_like(actual, .5)
        target.grad = torch.full_like(target, .5)
    optimizer.step()
    recovered.step()
    for actual, target in zip(model.parameters(), restored.parameters()):
        assert torch.equal(actual, target)
        for key, value in optimizer.state[actual].items():
            assert torch.equal(value, recovered.state[target][key])
    layout = _layout(model, optimizer)
    assert layout['optimizer'] == 'tnlm.optim.MuonAdamW'
    assert len(layout['optimizer_children']) == 2
    altered = _layout(restored, recovered)
    recovered.muon.param_groups[0]['params'].reverse()
    assert _layout(restored, recovered) != altered


def test_muon_nonfinite_child_rejected_before_any_allocation():
    model, optimizer = _muon_fixture()
    optimizer.adamw.param_groups[0]['lr'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        _state(model, optimizer)
    assert not optimizer.muon.state and not optimizer.adamw.state


@pytest.mark.parametrize('overlap', [False, True])
def test_muon_rejects_duplicate_and_overlapping_child_parameters(overlap):
    model, optimizer = _muon_fixture()
    parameter = optimizer.muon.param_groups[0]['params'][0]
    destination = optimizer.adamw if overlap else optimizer.muon
    destination.param_groups[0]['params'].append(parameter)
    with pytest.raises(ValueError, match='disjoint.*duplicates'):
        _state(model, optimizer)
    assert not optimizer.muon.state and not optimizer.adamw.state
