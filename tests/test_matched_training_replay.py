"""Independent bare-loop replay gate for future matched-work GPU benchmarks.

CPU eager only: this verifies the benchmark objective/trajectory, not throughput.
"""
import json

import pytest
import torch

from tnlm.checkpoint import load_checkpoint
from tnlm.data import PackedSampler, batch_hash
from tnlm.model import TensorLM
from tnlm.normalization import spectral_step
from tnlm.train import make_optimizer, train_run


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('optimizer_name', ['adamw', 'muon'])
@pytest.mark.parametrize('normalization', ['pre_rms', 'spectral_rms'])
def test_bare_loop_matches_trainer_from_initial_checkpoint(config, tmp_path,
                                                          optimizer_name, normalization):
    config.train.optimizer = optimizer_name
    config.model.normalization = normalization
    config.raw['data'].update(micro_batch_size=4, context=4)
    # Forces a quota-filling second batch, and a partial final update.
    config.train.global_prediction_tokens_per_update = 17
    config.train.rung_total_tokens = 39
    output = tmp_path / 'trainer'
    train_run(config, output, max_steps=3, max_seconds=60, device='cpu')
    metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    model = TensorLM(config.model)
    optimizer = make_optimizer(model, config.train)
    sampler = PackedSampler(config.raw['data']['manifest'], split='train',
                            batch_size=4, context=4, seed=config.train.seed)
    load_checkpoint(output / 'initial', model, optimizer, sampler)
    model.train()
    assert [row['prediction_tokens'] for row in metrics] == [17, 17, 5]
    assert any(row['microbatches'] > 1 for row in metrics)
    omitted = 0
    for record in metrics:
        # Reproduce observed rates, without introducing another schedule implementation.
        for group, lr in zip(optimizer.param_groups, record['optimizer_group_lrs'], strict=True):
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        spectral_step(model)
        remaining = record['prediction_tokens']
        total_loss = 0.0
        hashes, positions = [], 0
        while remaining:
            batch = sampler.next_batch()
            idx, docs, original_targets = batch
            targets = original_targets.clone()
            valid_indices = torch.where(targets.reshape(-1) != -100)[0]
            used = min(remaining, len(valid_indices))
            targets.reshape(-1)[valid_indices[used:]] = -100
            if not used:
                continue
            hashes.append(batch_hash(batch))
            # Independent row selection; retain every prefix position of scored rows.
            selected = (targets != -100).any(dim=1)
            omitted += int((~selected).sum())
            idx, docs, targets = idx[selected], docs[selected], targets[selected]
            positions += idx.numel()
            _, loss, count = model(idx, doc_ids=docs, targets=targets, reduction='sum')
            assert int(count) == used
            loss.backward()
            total_loss += float(loss.detach())
            remaining -= used
        assert hashes == record['batch_hashes']
        assert positions == record['forward_input_positions']
        assert total_loss / record['prediction_tokens'] == pytest.approx(record['loss'], rel=1e-7)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.div_(record['prediction_tokens'])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip,
                                              error_if_nonfinite=True)
        assert float(norm) == pytest.approx(record['grad_norm'], rel=1e-7)
        optimizer.step()
    final = torch.load(output / 'final/state.pt', map_location='cpu', weights_only=False)
    assert omitted > 0
    assert sampler.state_dict() == final['sampler']
    torch.testing.assert_close(model.state_dict(), final['model'], rtol=0, atol=0)
    assert_state_equal(optimizer.state_dict(), final['optimizer'])
