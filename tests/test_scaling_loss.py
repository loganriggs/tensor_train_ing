"""Independent CPU acceptance checks for recomputing linear cross-entropy."""
import pytest
import torch
import torch.nn.functional as F

from tnlm.kernels.fused_ce import linear_cross_entropy_sum


def _fixture(dtype=torch.float64, noncontiguous=False, all_ignored=False):
    generator = torch.Generator().manual_seed(1407)
    if noncontiguous:
        x = torch.randn(7, 22, generator=generator, dtype=dtype).T[::2].detach().requires_grad_()
        w = torch.randn(7, 26, generator=generator, dtype=dtype).T[::2].detach().requires_grad_()
        targets = torch.randint(0, 13, (22,), generator=generator)[::2]
        assert not x.is_contiguous() and not w.is_contiguous() and not targets.is_contiguous()
    else:
        x = torch.randn(11, 7, generator=generator, dtype=dtype).requires_grad_()
        w = torch.randn(13, 7, generator=generator, dtype=dtype).requires_grad_()
        targets = torch.randint(0, 13, (11,), generator=generator)
    targets[::3] = -100
    if all_ignored:
        targets.fill_(-100)
    return x, w, targets


@pytest.mark.parametrize('chunk', [1, 2, 4, 11, 30])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_sum_and_gradients_match_dense(chunk, dtype, noncontiguous):
    x, w, targets = _fixture(dtype, noncontiguous)
    reference_x = x.detach().clone().requires_grad_()
    reference_w = w.detach().clone().requires_grad_()
    actual = linear_cross_entropy_sum(x, w, targets, chunk=chunk)
    expected = F.cross_entropy(reference_x @ reference_w.T, targets, ignore_index=-100, reduction='sum')
    actual_grads = torch.autograd.grad(actual, (x, w))
    expected_grads = torch.autograd.grad(expected, (reference_x, reference_w))
    tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=5e-12, atol=5e-12)
    torch.testing.assert_close(actual, expected, **tolerance)
    for observed, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(observed, reference, **tolerance)


@pytest.mark.parametrize('chunk', [1, 3, 30])
@pytest.mark.parametrize('scale', [-2.75, 0., .125, 3.5])
@pytest.mark.parametrize('all_ignored', [False, True])
def test_arbitrary_upstream_scale_and_ignored_targets(chunk, scale, all_ignored):
    x, w, targets = _fixture(all_ignored=all_ignored)
    reference_x = x.detach().clone().requires_grad_()
    reference_w = w.detach().clone().requires_grad_()
    actual = linear_cross_entropy_sum(x, w, targets, chunk=chunk)
    expected = F.cross_entropy(reference_x @ reference_w.T, targets, reduction='sum')
    upstream = torch.tensor(scale, dtype=x.dtype)
    actual_grads = torch.autograd.grad(actual, (x, w), grad_outputs=upstream)
    expected_grads = torch.autograd.grad(expected, (reference_x, reference_w), grad_outputs=upstream)
    torch.testing.assert_close(actual, expected, rtol=5e-12, atol=5e-12)
    for observed, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(observed, reference, rtol=5e-12, atol=5e-12)
    if all_ignored:
        assert actual.item() == 0.
        assert all(torch.count_nonzero(gradient).item() == 0 for gradient in actual_grads)


@pytest.mark.parametrize('chunk', [1, 2, 8])
def test_fp64_gradcheck(chunk):
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(4, 3, dtype=torch.float64, generator=generator).requires_grad_()
    w = torch.randn(5, 3, dtype=torch.float64, generator=generator).requires_grad_()
    targets = torch.tensor([0, -100, 4, 2])
    assert torch.autograd.gradcheck(lambda features, weights: linear_cross_entropy_sum(features, weights, targets, chunk=chunk),
                                   (x, w), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('chunk', [1, 4, 30])
def test_tied_embedding_and_output_weight_accumulate_both_paths(chunk):
    generator = torch.Generator().manual_seed(67)
    weight = torch.randn(9, 5, dtype=torch.float64, generator=generator).requires_grad_()
    transform = torch.randn(5, 5, dtype=torch.float64, generator=generator).requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference_transform = transform.detach().clone().requires_grad_()
    input_ids = torch.tensor([1, 2, 1, 7, 4, 2])
    targets = torch.tensor([2, 1, -100, 4, 2, 8])
    x = F.embedding(input_ids, weight) @ transform
    reference_x = F.embedding(input_ids, reference_weight) @ reference_transform
    actual = linear_cross_entropy_sum(x, weight, targets, chunk=chunk)
    expected = F.cross_entropy(reference_x @ reference_weight.T, targets, reduction='sum')
    actual_grads = torch.autograd.grad(actual * 1.75, (weight, transform))
    expected_grads = torch.autograd.grad(expected * 1.75, (reference_weight, reference_transform))
    torch.testing.assert_close(actual, expected, rtol=5e-12, atol=5e-12)
    for observed, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(observed, reference, rtol=5e-12, atol=5e-12)
