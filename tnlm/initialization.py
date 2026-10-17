"""Memory-bounded initialization building blocks for FSDP models.

sharded_model_for_restore allocates zero placeholders for checkpoint loading.
canonical_initialization_components streams exact ordinary CPU initialization;
sharded_initialization combines these without a full real model allocation.
Never execute a zero-placeholder model before loading or filling all its state.
"""
from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor

from .model import TensorLM
from .parallel import shard_model


def sharded_model_for_restore(cfg, mesh, device):
    """Construct meta -> shard -> materialize local storage -> zero all state.

    ``cfg`` is a ModelConfig (or a Config with .model). Persistent buffers are
    overwritten by DCP; nonpersistent statistic accumulators correctly stay zero.
    Create the optimizer only after this helper, then perform the validated load.
    Memory includes local parameter shards and replicated buffers, never a real
    unsharded model during construction. This helper does not restore RNG state.
    """
    cfg = getattr(cfg, 'model', cfg)
    device = torch.device(device)
    if device.type == 'meta' or device.type != mesh.device_type:
        raise ValueError('restore device must be real and match the FSDP device mesh')
    with torch.device('meta'):
        model = TensorLM(cfg)
    if any(not value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise RuntimeError('restore construction allocated real model state before sharding')
    model = shard_model(model, mesh=mesh)
    model.to_empty(device=device)
    if cfg.tie_embeddings and model.unembed.weight is not model.embed.weight:
        model.unembed.weight = model.embed.weight
    with torch.no_grad():
        for parameter in model.parameters():
            if not isinstance(parameter, DTensor):
                raise RuntimeError('restore model has an unsharded parameter')
            parameter.to_local().zero_()
        for buffer in model.buffers():
            (buffer.to_local() if isinstance(buffer, DTensor) else buffer).zero_()
    if device.type == 'cpu':
        for unit in [model, *model.blocks]:
            unit.set_force_sum_reduction_for_comms(True)
    return model


CANONICAL_INITIALIZATION_VERSION = 'serial-cpu-components-v1'


def canonical_initialization_components(cfg):
    """Yield exact ordinary TensorLM CPU initialization, one component at a time.

    Prefixes are embed, blocks.N, final_norm, and unembed (omitted for tied
    embeddings). The caller seeds CPU RNG before iteration; complete consumption
    leaves that RNG exactly where ordinary TensorLM(cfg) construction leaves it.
    No CUDA RNG or model execution is involved. CPU default dtype is preserved.

    Pass one constructs/discards each CPU component in original constructor order,
    recording tiny RNG states. Pass two reconstructs each under fork_rng and uses
    the authoritative TensorLM.reset_parameters through a meta-only adapter whose
    current component is the sole real allocation. This preserves constructor RNG
    draws, spectral vectors/gamma, gains, affine biases, and lagged buffers exactly.

    The consumer MUST release each yielded module (and its tensors) before
    advancing the iterator. Keeping yielded modules defeats the memory bound.
    Unrelated global CPU RNG draws/default-dtype changes during iteration are
    rejected. Partial consumption leaves a partial RNG trajectory, not the final
    canonical state. Peak source storage is one CPU block/vocabulary component;
    this does not include consumer-owned destination shards or retained outputs.
    """
    import copy
    from torch import nn
    from .model import Block
    from .normalization import make_norm

    cfg = copy.deepcopy(getattr(cfg, 'model', cfg))
    dtype = torch.get_default_dtype()
    with torch.device('meta'):
        adapter = TensorLM(cfg)
    meta_embed, meta_head = adapter.embed, adapter.unembed
    adapter.blocks = nn.ModuleList([])
    constructors = [('embed', lambda: nn.Embedding(cfg.padded_vocab, cfg.d_model))]
    constructors.extend((f'blocks.{index}', lambda: Block(cfg)) for index in range(cfg.layers))
    constructors.extend((('final_norm', lambda: make_norm(cfg.final_normalization, cfg.d_model, cfg)),
                         ('unembed', lambda: nn.Linear(cfg.d_model, cfg.padded_vocab, bias=False))))
    constructor_rng = {}
    for prefix, constructor in constructors:
        constructor_rng[prefix] = torch.get_rng_state().clone()
        with torch.device('cpu'):
            component = constructor()
        del component
    for prefix, constructor in constructors:
        # The tied head constructor consumed RNG in pass one but its weight was
        # discarded by ordinary TensorLM before reset. Its output aliases embed.
        if prefix == 'unembed' and cfg.tie_embeddings:
            continue
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(constructor_rng[prefix])
            with torch.device('cpu'):
                component = constructor()
        adapter.embed, adapter.unembed = meta_embed, meta_head
        adapter.blocks = nn.ModuleList([])
        if prefix == 'embed':
            adapter.embed = component
        elif prefix == 'unembed':
            adapter.unembed = component
        elif prefix.startswith('blocks.'):
            adapter.blocks = nn.ModuleList([component])
        if prefix != 'final_norm':
            TensorLM.reset_parameters(adapter)
        expected_rng = torch.get_rng_state().clone()
        try:
            yield prefix, component
        finally:
            adapter.embed, adapter.unembed = meta_embed, meta_head
            adapter.blocks = nn.ModuleList([])
            del component
        if torch.get_default_dtype() != dtype or not torch.equal(torch.get_rng_state(), expected_rng):
            raise RuntimeError('canonical initialization consumer changed CPU RNG/default dtype between components')
