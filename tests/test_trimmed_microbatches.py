"""Unscored independent rows do not affect the training objective or gradients."""
import copy

import pytest
import torch

from tnlm.config import ModelConfig
from tnlm.model import TensorLM
from tnlm.normalization import spectral_step


def sample():
    idx = torch.tensor([
        [1, 2, 3, 4, 5],
        [20, 21, 22, 23, 24],
        [6, 7, 8, 9, 10],
        [0, 0, 0, 0, 0],
    ])
    docs = torch.tensor([
        [0, 0, 1, 1, -1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [-1, -2, -3, -4, -5],
    ])
    targets = torch.tensor([
        [2, -100, 4, -100, -100],
        [-100, -100, -100, -100, -100],
        [-100, 8, 9, 10, -100],
        [-100, -100, -100, -100, -100],
    ])
    return idx, docs, targets


@pytest.mark.parametrize('normalization', ['pre_rms', 'group_rms', 'spectral_rms'])
@pytest.mark.parametrize('attention', ['signed_product', 'softmax'])
@pytest.mark.parametrize('loss_chunk_tokens', [0, 3])
def test_full_and_trimmed_objective_all_parameter_gradients(
    normalization, attention, loss_chunk_tokens,
):
    from tnlm.train import _trim_unscored_rows

    torch.set_num_threads(1)
    torch.manual_seed(917)
    cfg = ModelConfig(
        layers=2, d_model=8, heads=2, d_key=4, d_value=4, d_mlp=12,
        vocab_size=32, padded_vocab=32, normalization=normalization,
        attention=attention, attention_backend='reference', dropout=0.,
        loss_chunk_tokens=loss_chunk_tokens,
    )
    full = TensorLM(cfg).double().train()
    trimmed = copy.deepcopy(full)
    spectral_step(full)
    spectral_step(trimmed)
    idx, docs, targets = sample()
    reduced = _trim_unscored_rows(idx, docs, targets, enabled=True)
    assert reduced[0].shape == (2, 5)  # Preserve the full causal time axis.
    assert torch.equal(reduced[0], idx[[0, 2]])
    before_rng = torch.get_rng_state().clone()
    _, full_loss, full_count = full(idx, docs, targets=targets, reduction='sum')
    assert torch.equal(before_rng, torch.get_rng_state())
    _, reduced_loss, reduced_count = trimmed(reduced[0], reduced[1], targets=reduced[2], reduction='sum')
    assert full_count == reduced_count == 5
    assert torch.equal(before_rng, torch.get_rng_state())
    # Independent dense CE establishes the objective, rather than just comparing
    # two executions of the optimized loss path.
    reference_logits = full(idx, docs)
    reference = torch.nn.functional.cross_entropy(
        reference_logits.reshape(-1, 32), targets.reshape(-1),
        ignore_index=-100, reduction='sum',
    )
    torch.testing.assert_close(full_loss, reference, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(reduced_loss, reference, atol=1e-11, rtol=1e-11)
    # Global normalization and an arbitrary upstream multiplier remain unchanged.
    (full_loss * .37 / 7).backward()
    (reduced_loss * .37 / 7).backward()
    for (name, parameter), (other_name, other) in zip(
        full.named_parameters(), trimmed.named_parameters(), strict=True,
    ):
        assert name == other_name
        assert parameter.grad is not None and other.grad is not None, name
        torch.testing.assert_close(parameter.grad, other.grad, atol=1e-11, rtol=1e-10, msg=name)


def test_disabled_preserves_original_tensor_objects():
    from tnlm.train import _trim_unscored_rows

    batch = sample()
    result = _trim_unscored_rows(*batch, enabled=False)
    assert all(actual is original for actual, original in zip(result, batch))


def test_empty_and_fully_scored_shapes():
    from tnlm.train import _trim_unscored_rows

    idx, docs, targets = sample()
    empty_targets = torch.full_like(targets, -100)
    empty = _trim_unscored_rows(idx, docs, empty_targets, enabled=True)
    assert all(value.shape == (0, 5) for value in empty)
    scored_targets = idx.clone()
    scored = _trim_unscored_rows(idx, docs, scored_targets, enabled=True)
    assert all(torch.equal(actual, expected) for actual, expected in zip(scored, (idx, docs, scored_targets)))


@pytest.mark.parametrize('normalization,final_normalization,expected', [
    ('pre_rms', 'rms', True),
    ('lagged', 'lagged', False),
    ('pre_rms', 'lagged', False),
])
@pytest.mark.parametrize('compile_requested', [False, True])
def test_training_gate_includes_final_lagged_norm(
    config, tmp_path, monkeypatch, normalization, final_normalization, expected, compile_requested,
):
    import importlib

    training = importlib.import_module('tnlm.train')
    original = training._trim_unscored_rows
    observed = []
    observed_buckets = []

    def spy(idx, docs, targets, *, enabled, bucket_rows=False):
        observed.append(enabled)
        observed_buckets.append(bucket_rows)
        return original(idx, docs, targets, enabled=enabled, bucket_rows=bucket_rows)

    monkeypatch.setattr(training, '_trim_unscored_rows', spy)
    # Actual graph capture is covered separately; this test checks trainer routing.
    monkeypatch.setattr(torch, 'compile', lambda model: model)
    config.train.compile = compile_requested
    config.model.normalization = normalization
    config.model.final_normalization = final_normalization
    result = training.train_run(
        config, tmp_path / 'run', max_steps=1, max_seconds=60, device='cpu',
    )
    assert observed and all(enabled is expected for enabled in observed)
    assert all(bool(bucketed) is (compile_requested and expected) for bucketed in observed_buckets)
    assert result['progress']['steps'] == 1
    assert result['progress']['tokens'] == 7
