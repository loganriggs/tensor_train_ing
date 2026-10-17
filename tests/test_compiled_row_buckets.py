"""Bound row-shape specialization without changing the scored objective."""
import copy

import pytest
import torch

from tnlm.config import ModelConfig
from tnlm.model import TensorLM
from tnlm.train import _trim_unscored_rows


@pytest.mark.parametrize('batch_size', [1, 3, 16, 31, 128, 1024])
def test_bucket_bound_and_original_row_identity(batch_size):
    idx = torch.arange(batch_size).view(-1, 1).expand(-1, 3).clone()
    docs = idx + 2000
    shapes = set()
    for scored in range(1, batch_size + 1):
        targets = torch.full_like(idx, -100)
        targets[-scored:, 1] = 1
        x, d, y = _trim_unscored_rows(idx, docs, targets, enabled=True, bucket_rows=True)
        shapes.add(x.shape[0])
        selected = x[:, 0]
        assert torch.equal(selected, selected.sort().values)
        assert torch.equal(x, idx[selected])
        assert torch.equal(d, docs[selected])
        assert torch.equal(y, targets[selected])
        assert int((y != -100).sum()) == scored
        if scored == 1:
            assert x.shape == (1, 3)
    assert len(shapes) <= 5


@pytest.mark.parametrize('chunk', [0, 64])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.slow
def test_compiled_all_row_counts_preserve_objective_and_stabilize(chunk, checkpoint):
    # Counting backend audits Dynamo capture only, not Inductor/CUDA kernels.
    torch.set_num_threads(1)
    torch._dynamo.reset()
    try:
        torch.manual_seed(173)
        cfg = ModelConfig(layers=1, d_model=8, heads=2, d_key=4,
                          d_value=4, d_mlp=12, vocab_size=32, padded_vocab=32,
                          attention_backend='reference', loss_chunk_tokens=chunk,
                          activation_checkpointing=checkpoint)
        eager = TensorLM(cfg).double().train()
        model = copy.deepcopy(eager)
        captures = []

        def backend(graph, inputs):
            captures.append(graph)
            return graph.forward

        compiled = torch.compile(model, backend=backend, fullgraph=True)
        for cycle in range(2):
            for scored in [16, *range(1, 16)]:
                idx = torch.randint(0, 32, (16, 8))
                docs = torch.zeros_like(idx)
                docs[:, 3 + scored % 3:] = 1
                targets = torch.full_like(idx, -100)
                targets[-scored:, cycle] = 2
                batch = _trim_unscored_rows(idx, docs, targets, enabled=True, bucket_rows=True)
                eager.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                _, expected, expected_n = eager(idx, docs, targets=targets, reduction='sum')
                _, actual, actual_n = compiled(batch[0], batch[1], targets=batch[2], reduction='sum')
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
                assert actual_n == expected_n == scored
                (expected / 16).backward()
                (actual / 16).backward()
                for p, q in zip(eager.parameters(), model.parameters(), strict=True):
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-10)
            if cycle == 0:
                warm_captures = len(captures)
                assert 1 <= warm_captures <= 5
            else:
                assert len(captures) == warm_captures
    finally:
        torch._dynamo.reset()
