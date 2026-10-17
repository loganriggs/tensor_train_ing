"""Exact serial-seed equivalence for bounded canonical CPU component generation."""
import gc
import weakref

import pytest
import torch

from tnlm.config import ModelConfig
from tnlm.initialization import canonical_initialization_components
from tnlm.model import TensorLM
from tnlm.normalization import lagged_modules


CASES = [('pre_rms', 'signed_product', 'bilinear', False),
         ('pre_rms', 'signed_product', 'bilinear', True),
         ('spectral_rms', 'lower_degree_mix', 'affine_bilinear', False),
         ('spectral_rms', 'offset_square', 'bilinear', True),
         ('lagged', 'signed_product', 'bilinear', False),
         ('branch_rms', 'offset_square', 'affine_bilinear', False),
         ('sandwich_rms', 'lower_degree_mix', 'bilinear', False),
         ('group_rms', 'signed_product', 'bilinear', False),
         ('pre_rms', 'softmax', 'swiglu', False)]


def configuration(package='pre_rms', attention='signed_product', mlp='bilinear', tied=False):
    return ModelConfig(layers=2, d_model=8, heads=1, d_key=4, d_value=4, d_mlp=12,
                       vocab_size=17, padded_vocab=32, normalization=package,
                       final_normalization='lagged' if package == 'lagged' else 'rms',
                       attention=attention, mlp=mlp, tie_embeddings=tied, attention_backend='reference')


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_components_state_and_final_rng_bitwise_equal_ordinary_model(case, dtype):
    torch.set_num_threads(1)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        cfg = configuration(*case)
        torch.manual_seed(137)
        reference = TensorLM(cfg)
        expected_rng = torch.get_rng_state().clone()
        expected_buffers = {name: value.clone() for name, value in reference.named_buffers()}
        torch.manual_seed(137)
        actual, buffers, prefixes = {}, {}, []
        for prefix, component in canonical_initialization_components(cfg):
            prefixes.append(prefix)
            for name, value in component.state_dict().items():
                actual[prefix + '.' + name] = value.clone()
            for name, value in component.named_buffers():
                buffers[prefix + '.' + name] = value.clone()
            assert all(value.device.type == 'cpu' for value in (*component.parameters(), *component.buffers()))
            for norm in lagged_modules(component):
                assert norm.mean_sq == 1 and not norm.calibrated
                assert norm.acc_sum == norm.acc_count == 0
            del component
        if cfg.tie_embeddings:
            actual['unembed.weight'] = actual['embed.weight']
            assert 'unembed' not in prefixes
        else:
            assert prefixes[-1] == 'unembed'
        assert prefixes[:4] == ['embed', 'blocks.0', 'blocks.1', 'final_norm']
        assert actual.keys() == reference.state_dict().keys()
        for name, value in reference.state_dict().items():
            assert torch.equal(value, actual[name]), name
        assert buffers.keys() == expected_buffers.keys()
        for name, value in expected_buffers.items():
            assert torch.equal(value, buffers[name]), name
        assert torch.equal(torch.get_rng_state(), expected_rng)
    finally:
        torch.set_default_dtype(previous_dtype)


def test_component_lifetime_is_bounded_and_reset_adapter_has_one_real_unit(monkeypatch):
    import tnlm.model as model_module
    references = []
    original_block = model_module.Block
    original_reset = model_module.TensorLM.reset_parameters

    def observed_block(cfg):
        gc.collect()
        assert not any(reference() is not None for reference in references), 'previous real source block retained'
        block = original_block(cfg)
        if not next(block.parameters()).is_meta:
            references.append(weakref.ref(block))
        return block

    reset_real_units = []
    def observed_reset(model):
        real_units = []
        if not model.embed.weight.is_meta:
            real_units.append('embed')
        real_units.extend('block' for block in model.blocks if not next(block.parameters()).is_meta)
        if not model.unembed.weight.is_meta:
            real_units.append('unembed')
        assert len(real_units) <= 1
        reset_real_units.append(real_units)
        return original_reset(model)

    monkeypatch.setattr(model_module, 'Block', observed_block)
    monkeypatch.setattr(model_module.TensorLM, 'reset_parameters', observed_reset)
    torch.manual_seed(51)
    for prefix, component in canonical_initialization_components(configuration()):
        del component
    gc.collect()
    assert not any(reference() is not None for reference in references)
    assert len(references) == 4  # two blocks in constructor pass, two in reset pass
    assert reset_real_units == [[], ['embed'], ['block'], ['block'], ['unembed']]


def test_consumer_rng_interference_fails_explicitly():
    torch.manual_seed(13)
    iterator = canonical_initialization_components(configuration())
    _, component = next(iterator)
    del component
    torch.rand(1)
    with pytest.raises(RuntimeError, match='changed CPU RNG'):
        next(iterator)
