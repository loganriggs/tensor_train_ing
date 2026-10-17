import torch

from tnlm.model import TensorLM
from tnlm.shapes import _BASE, count_params, memory_plan, preset


def test_counts_match_brief_and_instantiated_models():
    assert count_params(preset("screen_100m"))["matrices"] == 101_056_512
    assert count_params(preset("screen_100m", polynomial=False))["matrices"] == 101_842_944
    tiny = preset("screen_100m", layers=1, d_model=64, heads=1, d_mlp=128, padded_vocab=128, vocab_size=100)
    real = TensorLM(tiny)
    assert count_params(tiny)["total"] == sum(p.numel() for p in real.parameters())


def test_polynomial_presets_are_parameter_matched():
    for name in _BASE:
        p, s = (count_params(preset(name, poly))["matrices"] for poly in (True, False))
        assert abs(p - s) / s < 0.01, name


def test_memory_plan_monotone_in_gpus():
    cfg = preset("funded_32b", activation_checkpointing=True)
    a, b = (memory_plan(cfg, gpus=g, micro_tokens_per_gpu=8192)["total_GB"] for g in (8, 16))
    assert b < a
