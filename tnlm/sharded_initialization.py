"""Fresh FSDP initialization from canonical CPU components, one at a time."""
from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor, Shard

from .initialization import sharded_model_for_restore


@torch.no_grad()
def _copy_component(model, prefix, component):
    """Copy CPU slices directly into local storage; never move a full matrix to CUDA."""
    destination = dict(model.named_parameters()) | dict(model.named_buffers())
    copied = set()
    for name, source in list(component.named_parameters()) + list(component.named_buffers()):
        key = f'{prefix}.{name}'
        target = destination[key]
        if source.device.type != 'cpu' or source.dtype != target.dtype or source.shape != target.shape:
            raise ValueError(f'incompatible canonical initialization tensor: {key}')
        if isinstance(target, DTensor):
            mesh = target.device_mesh
            if mesh.ndim != 1 or len(target.placements) != 1 or not isinstance(target.placements[0], Shard):
                raise ValueError('canonical initialization requires a one-dimensional sharded mesh')
            axis = target.placements[0].dim
            chunk = (source.shape[axis] + mesh.size() - 1) // mesh.size()
            start = min(mesh.get_local_rank() * chunk, source.shape[axis])
            source = source.narrow(axis, start, min(chunk, source.shape[axis] - start))
            target = target.to_local()
        if source.shape != target.shape:
            raise ValueError(f'canonical shard shape mismatch: {key}')
        target.copy_(source)
        copied.add(key)
    return copied


def sharded_model_from_cpu_components(cfg, mesh, device):
    """Match CPU TensorLM construction and RNG using bounded component storage.

    Every rank calls after the same CPU seed. Host scratch holds one component
    (a block or vocabulary matrix); destination storage holds local shards and
    replicated buffers. This is not a peak-memory guarantee for subsequent FSDP
    gathers, calibration, or optimizer/checkpoint work.
    """
    from .initialization import canonical_initialization_components

    model = sharded_model_for_restore(cfg, mesh, device)
    copied = set()
    components = canonical_initialization_components(getattr(cfg, 'model', cfg))
    while True:
        try:
            prefix, component = next(components)
        except StopIteration:
            break
        copied.update(_copy_component(model, prefix, component))
        # Release the consumer's reference before the generator allocates again.
        del component
    expected = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
    if copied != expected:
        raise ValueError(f'incomplete canonical initialization: missing={sorted(expected - copied)}, '
                         f'extra={sorted(copied - expected)}')
    return model
