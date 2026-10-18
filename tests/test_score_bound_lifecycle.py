"""Training score-bound capture survives compilation and backward recomputation."""
from types import SimpleNamespace

import pytest
import torch

from tnlm.config import ModelConfig
from tnlm.model import TensorLM
from tnlm.qk_clip import score_bounds
from tnlm.training_stability import BatchScoreBounds


def setup_model(*, checkpoint=False):
    torch.manual_seed(17)
    cfg = SimpleNamespace(
        model=ModelConfig(layers=2, d_model=16, heads=2, d_key=8, d_value=8,
            d_mlp=24, vocab_size=97, padded_vocab=128, attention_backend='reference',
            activation_checkpointing=checkpoint),
        train=SimpleNamespace(qk_clip_source='batch_bound', qk_clip_tau=0.,
                              qk_clip_every_steps=0))
    model = TensorLM(cfg.model).train()
    return model, BatchScoreBounds(model, cfg)


def inputs(offset=0):
    idx = (torch.arange(12).reshape(2, 6) * 3 + offset) % 97
    docs = torch.zeros_like(idx)
    docs[1, 3:] = 1
    return idx, docs


def copy_bounds(model):
    return [{key: value.clone() for key, value in layer.items()}
            for layer in score_bounds(model)]


def assert_bounds_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key])


def test_aot_eager_compiled_recording_tracks_changed_inputs():
    eager, eager_helper = setup_model()
    compiled_model, compiled_helper = setup_model()
    compiled = torch.compile(compiled_model, backend='aot_eager')
    observations = []
    for offset in (0, 41):
        idx, docs = inputs(offset)
        eager_helper.begin_update()
        compiled_helper.begin_update()
        eager(idx, doc_ids=docs)
        compiled(idx, doc_ids=docs)
        eager_helper.observe()
        compiled_helper.observe()
        observed = copy_bounds(compiled_model)
        assert_bounds_equal(observed, copy_bounds(eager))
        assert compiled_helper.finish_update() == pytest.approx(eager_helper.finish_update(), rel=1e-5)
        observations.append(observed)
    assert any(not torch.equal(a[key], b[key])
               for a, b in zip(*observations) for key in a)


def test_checkpoint_backward_and_attribute_overwrite_preserve_last_training_bounds(monkeypatch):
    import tnlm.qk_clip as clipping
    model, helper = setup_model(checkpoint=True)
    helper.begin_update()
    snapshots = []
    for offset in (0, 41):
        idx, docs = inputs(offset)
        _, loss, _ = model(idx, doc_ids=docs, targets=(idx + 1) % 97, reduction='sum')
        helper.observe()
        snapshots.append(copy_bounds(model))
        loss.backward()
    expected_max = max(float(value.max()) for snapshot in snapshots
                       for layer in snapshot for value in layer.values())
    assert helper.finish_update() == pytest.approx(expected_max)
    # A probe or backward recomputation can assign these mutable module attrs.
    # Clipping must reinstall the captured final training-forward snapshot.
    for block in model.blocks:
        block.attn.last_score_bound = {key: torch.full_like(value, 1.e9)
                                      for key, value in block.attn.last_score_bound.items()}
    calls = []
    def inspect_clip(actual_model, tau):
        calls.append(tau)
        assert_bounds_equal(score_bounds(actual_model), snapshots[-1])
        return {'clipped_heads': 0, 'max_bound_before': expected_max,
                'factors': {}, 'tau': tau, 'source': 'per-batch norm bound'}
    monkeypatch.setattr(clipping, 'qk_clip_from_bounds_', inspect_clip)
    helper.clip(.1)
    assert calls == [.1]


@pytest.mark.parametrize('bad', ['missing', 'shape', 'negative', 'nan', 'inf'])
def test_invalid_recorded_bounds_reject_before_optimizer(bad):
    model, helper = setup_model()
    helper.begin_update()
    idx, docs = inputs()
    model(idx, doc_ids=docs)
    if bad == 'missing':
        model.blocks[0].attn.last_score_bound = None
    else:
        value = model.blocks[0].attn.last_score_bound['S1']
        if bad == 'shape':
            value = torch.ones(model.cfg.heads + 1)
        else:
            value = value.clone()
            value[0] = {'negative': -1., 'nan': float('nan'), 'inf': float('inf')}[bad]
        model.blocks[0].attn.last_score_bound['S1'] = value
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        helper.observe()
        helper.finish_update()


def test_snapshot_owns_storage_when_forward_bound_buffer_is_reused(monkeypatch):
    import tnlm.qk_clip as clipping
    model, helper = setup_model()
    idx, docs = inputs()
    helper.begin_update()
    model(idx, doc_ids=docs)
    expected = copy_bounds(model)
    helper.observe()
    for layer in score_bounds(model):
        for bound in layer.values():
            bound.fill_(1.e9)
    def inspect_clip(actual, tau):
        assert_bounds_equal(score_bounds(actual), expected)
        return {'clipped_heads': 0, 'factors': {}, 'tau': tau}
    monkeypatch.setattr(clipping, 'qk_clip_from_bounds_', inspect_clip)
    helper.clip(.1)
