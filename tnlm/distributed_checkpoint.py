"""Synchronous shared-filesystem FSDP2 recovery checkpoints, AdamW or MuonAdamW children.

Every rank must call each API in identical order with a live process group. Exact
replay requires the same world size, PyTorch version, model/optimizer layout and
rank-to-data mapping. Loading clones local model/optimizer state for numeric preflight,
so peak memory includes another local state copy, without a full-model gather.
Schedulers/scalers are not supported; caller-owned scheduling belongs in progress
(the intended BF16 path needs no AMP scaler). A failed process/collective requires relaunch, not continued
use of the broken group. This is a checkpoint primitive, not a distributed trainer.
Only exact PackedSampler/RankPackedSampler types are supported for safe shallow
cursor preflight. Load only trusted local files (DCP and torch pickle are not authenticity checks).

State extraction explicitly allocates missing trainable AdamW zero moments and
zero step counters, or Muon zero momentum buffers, using public tensor constructors,
without taking a synthetic optimizer step or touching parameter values. Existing state/counters are retained.
This changes an empty optimizer's in-memory state. Entirely frozen models with an
empty optimizer are unsupported recovery targets; use weight-only export instead.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

from .optim import MuonAdamW
from .checkpoint_storage import checkpoint_space_estimate, require_checkpoint_space
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.tensor import DTensor

from .checkpoint import (_digest, _git, _plain, _publish, _restore_rng, _rng_state,
                         _sync_directory, _validate_rng)

FORMAT = 'tnlm-dcp-v1'


def export_distributed_weights(path, model, *, config, progress, manifest_hash,
                               parent=None, max_state_bytes=4_000_000_000):
    """Collectively export a bounded full CPU checkpoint for existing evaluators.

    All ranks participate; only rank zero writes. This is an evaluation artifact,
    with no optimizer/sampler state, and cannot resume training. Admission bounds
    the full persistent tensor payload, not peak host memory or serialization IO.
    Export retains spectral and normalization buffers and all explicit norm nodes.
    """
    from .checkpoint import save_checkpoint
    from .model import TensorLM
    from .parallel import full_state_dict
    from .train import _assert_no_analysis_state

    rank, _ = _group()
    def preflight():
        if type(max_state_bytes) is not int or max_state_bytes <= 0:
            raise ValueError('positive integer export state-byte limit required')
        _assert_no_analysis_state(model)
        state = model.state_dict()
        _finite(state)
        for module in model.modules():
            for name in ('acc_sum', 'acc_count'):
                value = getattr(module, name, None)
                if value is not None and bool(value.ne(0).any()):
                    raise ValueError('export requires committed normalization statistics')
        size = sum(value.numel() * value.element_size() for value in state.values())
        if size > max_state_bytes:
            raise ValueError(f'full export needs {size} tensor bytes, exceeds {max_state_bytes}')
        target = Path(path).resolve()
        if target.exists():
            raise ValueError('export destination already exists')
        return {'path': str(target), 'state_bytes': size, 'max_state_bytes': max_state_bytes,
                'config': _plain(config), 'progress': _plain(progress),
                'manifest_hash': manifest_hash, 'parent': _plain(parent)}
    request = _phase('export preflight', preflight)
    _agree('export request', request)
    _agree('export replicated buffers', _phase('export buffer check', lambda: _replicated_buffers(model)))
    state = _phase('full CPU state gathering', lambda: full_state_dict(model))
    def publish():
        if rank:
            return None
        # Meta construction avoids allocating a second full CPU parameter copy.
        with torch.random.fork_rng(devices=[]), torch.device('meta'):
            portable = TensorLM(model.cfg)
        portable.load_state_dict(state, strict=True, assign=True)
        if model.cfg.tie_embeddings:
            if not torch.equal(state['embed.weight'], state['unembed.weight']):
                raise ValueError('tied vocabulary weights differ in gathered state')
            portable.unembed.weight = portable.embed.weight
        # Nonpersistent collection accumulators are absent from the state dict.
        for module in portable.modules():
            for name, value in module._buffers.items():
                if value is not None and value.is_meta:
                    if name not in module._non_persistent_buffers_set:
                        raise ValueError('export left a persistent buffer on meta')
                    module._buffers[name] = torch.zeros(value.shape, dtype=value.dtype, device='cpu')
        return save_checkpoint(request['path'], portable, None, None,
            config=config, progress=progress, manifest_hash=manifest_hash, parent=parent,
            extra={'artifact_kind': 'distributed_evaluation_weights', 'training_recovery': False,
                   'source_world_size': dist.get_world_size(), 'persistent_tensor_bytes': request['state_bytes']})
    manifest = _phase('evaluation artifact publication', publish)
    result = [manifest]
    dist.broadcast_object_list(result, src=0)
    return result[0]


def _group():
    if not dist.is_initialized():
        raise ValueError('distributed checkpoint requires an initialized process group')
    return dist.get_rank(), dist.get_world_size()


def _phase(label, callback):
    """Turn recoverable rank-local validation/IO errors into all-rank failures."""
    value, error = None, None
    try:
        value = callback()
    except (Exception, CheckpointException) as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise ValueError(f'{label} failed: ' + '; '.join(f'rank {i}: {e}' for i, e in enumerate(errors) if e))
    return value


def _agree(label, value):
    encoded = _phase(f'{label} serialization', lambda: json.dumps(value, sort_keys=True, allow_nan=False))
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, encoded)
    if len(set(values)) != 1:
        raise ValueError(f'{label} differs across ranks')


def _finite(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError('nonfinite Python optimizer/checkpoint value')
    if isinstance(value, torch.Tensor):
        local = value.to_local() if isinstance(value, DTensor) else value
        if (local.is_floating_point() or local.is_complex()) and not bool(torch.isfinite(local).all()):
            raise ValueError('nonfinite local checkpoint tensor')
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite(item)


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value


def _optimizer_children(optimizer):
    """Expose only exact supported optimizers; never pass the nested wrapper to DCP."""
    if type(optimizer) is torch.optim.AdamW:
        return [optimizer]
    if (type(optimizer) is MuonAdamW and type(optimizer.muon) is torch.optim.Muon
            and type(optimizer.adamw) is torch.optim.AdamW):
        children = [optimizer.muon, optimizer.adamw]
        parameters = [p for child in children for group in child.param_groups for p in group['params']]
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError('MuonAdamW children must have disjoint parameter groups without duplicates')
        return children
    raise ValueError('distributed checkpoints support exactly torch.optim.AdamW or MuonAdamW')


def _layout(model, optimizer):
    children = _optimizer_children(optimizer)
    names = {id(p): name for name, p in model.named_parameters()}
    states = model.state_dict()
    layout = {'model_class': f'{type(model).__module__}.{type(model).__qualname__}',
            'model': {name: {'shape': list(t.shape), 'dtype': str(t.dtype),
                            'placements': [str(p) for p in t.placements] if isinstance(t, DTensor) else None,
                            'mesh': {'shape': list(t.device_mesh.mesh.shape),
                                     'device_type': t.device_mesh.device_type,
                                     'rank_order': t.device_mesh.mesh.tolist()} if isinstance(t, DTensor) else None}
                      for name, t in states.items()},
            'optimizer': 'torch.optim.AdamW',
            'parameter_groups': [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]}
    if type(optimizer) is MuonAdamW:
        layout['optimizer'] = 'tnlm.optim.MuonAdamW'
        layout['optimizer_children'] = [
            {'class': f'{type(child).__module__}.{type(child).__qualname__}',
             'parameter_groups': [[names[id(p)] for p in group['params']] for group in child.param_groups]}
            for child in children]
    return layout


def _check(model, optimizer, sampler):
    from .data import PackedSampler, RankPackedSampler
    if type(sampler) not in (PackedSampler, RankPackedSampler):
        raise ValueError('distributed checkpoints support exactly PackedSampler and RankPackedSampler')
    if type(sampler) is RankPackedSampler and (sampler.rank != dist.get_rank() or sampler.world_size != dist.get_world_size()):
        raise ValueError('rank sampler identity differs from the process group')
    layout = _layout(model, optimizer)
    _finite(model.state_dict())
    _finite(optimizer.state_dict())
    for module in model.modules():
        for name in ('acc_count', 'acc_sum'):
            value = getattr(module, name, None)
            if value is not None and bool(value.ne(0).any()):
                raise ValueError('checkpoint requires committed lagged statistics at a step boundary')
    if any(p.grad is not None for g in optimizer.param_groups for p in g['params']):
        raise ValueError('checkpoint requires cleared gradients at an optimizer-step boundary')
    return layout


def _allocate_adamw_state(optimizer):
    """Match AdamW lazy state allocation without executing an optimizer step.

    Step counters use default scalar dtype (FP64 only for FP64 global default),
    except fused mode always uses FP32. Capturable/fused counters live beside the
    parameter; ordinary counters live on CPU. Moments preserve parameter layout.
    """
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError('distributed checkpoints currently support exactly torch.optim.AdamW')
    _finite(optimizer.state_dict())  # Reject invalid hyperparameters before allocation.
    eligible = [(group, parameter) for group in optimizer.param_groups
                for parameter in group['params'] if parameter.requires_grad]
    if not eligible and not optimizer.state:
        raise ValueError('entirely frozen model with empty AdamW state is unsupported for recovery')
    for group, parameter in eligible:
        if optimizer.state.get(parameter):
            continue
        fused = bool(group.get('fused'))
        step_dtype = (torch.float32 if fused or torch.get_default_dtype() != torch.float64
                      else torch.float64)
        step_device = parameter.device if group.get('capturable') or fused else torch.device('cpu')
        state = {
            'step': torch.zeros((), dtype=step_dtype, device=step_device),
            'exp_avg': torch.zeros_like(parameter, memory_format=torch.preserve_format),
            'exp_avg_sq': torch.zeros_like(parameter, memory_format=torch.preserve_format),
        }
        if group['amsgrad']:
            state['max_exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        optimizer.state[parameter] = state


def _state(model, optimizer):
    children = _optimizer_children(optimizer)
    # Validate every child first, so unsupported/invalid state cannot partially
    # allocate a preceding child. Empty state would trigger DCP's synthetic step.
    _finite(optimizer.state_dict())
    for child in children:
        eligible = [p for group in child.param_groups for p in group['params'] if p.requires_grad]
        if not eligible and not child.state:
            raise ValueError('entirely frozen optimizer child with empty state is unsupported for recovery')
        if type(child) is torch.optim.Muon and any(p.ndim != 2 or p.is_complex() for p in eligible):
            raise ValueError('Muon recovery requires real logical 2-D matrix parameters')
    for child in children:
        if type(child) is torch.optim.AdamW:
            _allocate_adamw_state(child)
        else:
            for group in child.param_groups:
                for parameter in group['params']:
                    if parameter.requires_grad and 'momentum_buffer' not in child.state.get(parameter, {}):
                        child.state[parameter]['momentum_buffer'] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format)
    dcp_optimizer = optimizer if type(optimizer) is torch.optim.AdamW else children
    model_state, optim_state = get_state_dict(model, dcp_optimizer)
    return {'model': model_state, 'optimizer': optim_state}


def _replicated_buffers(model, loaded_model_state=None):
    """Digest live buffers, or staged persistent values plus live nonpersistent ones."""
    digest = hashlib.sha256()
    for name, value in sorted(model.named_buffers()):
        if loaded_model_state is not None and name in loaded_model_state:
            value = loaded_model_state[name]
        if isinstance(value, DTensor):
            raise ValueError('distributed buffer placements are not supported by this checkpoint format')
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _runtime():
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision()}


def save_distributed_checkpoint(path, model, optimizer, sampler, *, config, progress,
                                manifest_hash, parent=None):
    """All-rank synchronous save; immutable destination on a shared filesystem."""
    rank, world = _group()
    path = _phase('checkpoint path validation', lambda: Path(path).resolve())
    layout = _phase('save preflight', lambda: _check(model, optimizer, sampler))
    shared = _phase('save metadata', lambda: {
        'config': _plain(config), 'progress': _plain(progress),
        'manifest_hash': manifest_hash, 'parent': _plain(parent),
        'world_size': world, 'torch_version': str(torch.__version__),
        'backend': str(dist.get_backend()), 'layout': layout,
        'runtime': _runtime(),
        'optimizer_hyperparameters_scope': 'informational save-time snapshot; checksummed DCP optimizer state is recovery authority',
        'optimizer_hyperparameters': [[{key: _plain(value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value)
                                       for key, value in group.items() if key != 'params'}
                                      for group in child.param_groups] for child in _optimizer_children(optimizer)],
    })
    _phase('metadata serialization', lambda: json.dumps(shared, allow_nan=False))
    _agree('checkpoint path and global metadata', {'path': str(path), **shared})
    buffers = _phase('replicated buffer digest', lambda: _replicated_buffers(model))
    _agree('replicated buffers', buffers)
    local = _phase('rank recovery state', lambda: {'rank': rank, 'world_size': world,
                    'rng': _rng_state(), 'sampler': sampler.state_dict()})
    _agree('global sampler stream', {'sampler_type': type(sampler).__name__,
                                    'stream': local['sampler'].get('stream')})
    state = _phase('canonical state collection', lambda: _state(model, optimizer))
    _phase('canonical state finite check', lambda: _finite(state))
    code = _phase('checkpoint code provenance', lambda: _git() if rank == 0 else None)
    estimate = _phase('checkpoint space estimate', lambda: checkpoint_space_estimate(
        {'state': state, 'recovery': local, 'metadata': shared, 'code': code}))
    estimates = [None] * world
    dist.all_gather_object(estimates, estimate)
    disk_admission = _phase('checkpoint disk admission', lambda: require_checkpoint_space(
        path.parent, sum(estimates)) if rank == 0 else None)
    admission = [disk_admission]
    dist.broadcast_object_list(admission, src=0)
    disk_admission = admission[0]
    def staging_directory():
        if rank:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'checkpoint already exists: {path}')
        return tempfile.mkdtemp(prefix=f'.{path.name}.tmp-', dir=path.parent)
    staging = _phase('staging creation', staging_directory)
    broadcast = [staging]
    dist.broadcast_object_list(broadcast, src=0)
    staging = Path(broadcast[0])
    try:
        _phase('DCP synchronous save', lambda: dcp.save(state, checkpoint_id=staging / 'dcp'))
        def save_rank():
            with (staging / f'rank-{rank:05d}.pt').open('wb') as stream:
                torch.save(local, stream)
                stream.flush()
                os.fsync(stream.fileno())
        _phase('rank recovery save', save_rank)
        def publish():
            if rank:
                return None
            files = {}
            for artifact in sorted(staging.rglob('*')):
                if artifact.is_file():
                    with artifact.open('rb') as stream:
                        os.fsync(stream.fileno())
                    files[str(artifact.relative_to(staging))] = {'sha256': _digest(artifact), 'bytes': artifact.stat().st_size}
            manifest = {'format': FORMAT, **shared, 'files': files, 'code': code,
                        'disk_admission': disk_admission,
                        'replicated_buffers_sha256': buffers,
                        'created_utc': datetime.now(timezone.utc).isoformat(),
                        'replay_scope': 'same-world same-layout rank-local RNG/data cursor; group failure requires relaunch'}
            with (staging / 'manifest.json').open('w') as stream:
                json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            for directory in sorted((p for p in staging.rglob('*') if p.is_dir()), reverse=True):
                _sync_directory(directory)
            _sync_directory(staging)
            _publish(staging, path)
            _sync_directory(path.parent)
            return manifest
        manifest = _phase('atomic publication', publish)
        broadcast = [manifest]
        dist.broadcast_object_list(broadcast, src=0)
        return broadcast[0]
    finally:
        # All recoverable errors have already synchronized ranks via _phase.
        # Hard process/transport failures may leave staging; never a valid save.
        if rank == 0 and staging.exists():
            shutil.rmtree(staging)


def load_distributed_checkpoint(path, model, optimizer, sampler, *, expected_config=None,
                                expected_manifest_hash=None, restore_rng=True):
    """All-rank trusted-local load, refusing changed world/version/layout."""
    rank, world = _group()
    path = _phase('checkpoint path validation', lambda: Path(path).resolve())
    layout = _phase('load preflight', lambda: _check(model, optimizer, sampler))
    _agree('load request', {'path': str(path), 'expected_config': _plain(expected_config),
                          'expected_manifest_hash': expected_manifest_hash, 'restore_rng': restore_rng})
    def verify():
        if rank:
            return None
        try:
            manifest = json.loads((path / 'manifest.json').read_text())
            if manifest['format'] != FORMAT or manifest['world_size'] != world:
                raise ValueError('checkpoint format/world size mismatch')
            if manifest['torch_version'] != str(torch.__version__) or manifest['layout'] != layout or manifest['backend'] != str(dist.get_backend()) or manifest['runtime'] != _runtime():
                raise ValueError('checkpoint version/backend/layout mismatch')
            if expected_config is not None and manifest['config'] != _plain(expected_config):
                raise ValueError('checkpoint config mismatch')
            if expected_manifest_hash is not None and manifest['manifest_hash'] != expected_manifest_hash:
                raise ValueError('data manifest hash mismatch')
            files = manifest['files']
            required = {f'rank-{i:05d}.pt' for i in range(world)} | {'dcp/.metadata'}
            if not required <= set(files) or not any(name.endswith('.distcp') for name in files):
                raise ValueError('incomplete checkpoint shard set')
            actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()} - {'manifest.json'}
            if actual != set(files):
                raise ValueError('checkpoint file set differs from manifest')
            for name, info in files.items():
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('invalid checkpoint artifact path')
                artifact = path / relative
                if artifact.stat().st_size != info['bytes'] or _digest(artifact) != info['sha256']:
                    raise ValueError(f'checkpoint checksum mismatch: {name}')
            return manifest
        except (OSError, KeyError, TypeError) as exc:
            raise ValueError('incomplete or invalid distributed checkpoint') from exc
    manifest = _phase('checkpoint verification', verify)
    broadcast = [manifest]
    dist.broadcast_object_list(broadcast, src=0)
    manifest = broadcast[0]
    _agree('destination layout and runtime', {'layout': layout, 'runtime': _runtime()})
    def rank_state():
        local = torch.load(path / f'rank-{rank:05d}.pt', map_location='cpu', weights_only=False)
        if local['rank'] != rank or local['world_size'] != world:
            raise ValueError('rank recovery identity mismatch')
        if restore_rng:
            _validate_rng(local['rng'])
            # Check CPU RNG encodings without changing caller generators.
            random.Random().setstate(local['rng']['python'])
            np.random.RandomState().set_state(local['rng']['numpy'])
            torch.Generator(device='cpu').set_state(local['rng']['torch'])
        # Shallow-copy avoids duplicating token memmaps. PackedSampler loaders
        # replace cursors/order rather than mutating shared document/token arrays.
        # Reject custom sampler classes whose load mutates shared backing storage.
        probe = copy.copy(sampler)
        probe.load_state_dict(local['sampler'])
        return local
    local = _phase('rank recovery verification', rank_state)
    _agree('restored global sampler stream', {'sampler_type': type(sampler).__name__,
                                             'stream': local['sampler'].get('stream')})
    # Clone sharded tensors so DCP IO cannot alter live weights before all ranks
    # have loaded and validated their local numeric state.
    state = _phase('load state allocation', lambda: _clone(_state(model, optimizer)))
    _phase('DCP synchronous load', lambda: dcp.load(state, checkpoint_id=path / 'dcp'))
    _phase('loaded numeric validation', lambda: _finite(state))
    # Validate replicated-state identity while DCP tensors are still detached
    # staging copies. A bad manifest/digest must not install weights, optimizer,
    # sampler cursor or RNG before reporting this mismatch. Nonpersistent buffers
    # are intentionally absent from DCP and remain the preflighted live values.
    staged_buffers = _phase('staged buffer digest', lambda: _replicated_buffers(model, state['model']))
    _agree('staged replicated buffers', staged_buffers)
    if staged_buffers != manifest['replicated_buffers_sha256']:
        raise ValueError('staged replicated buffers differ from checkpoint')
    _phase('state installation', lambda: set_state_dict(model, optimizer if type(optimizer) is torch.optim.AdamW
           else _optimizer_children(optimizer),
           model_state_dict=state['model'], optim_state_dict=state['optimizer']))
    _phase('sampler restoration', lambda: sampler.load_state_dict(local['sampler']))
    if restore_rng:
        _phase('RNG restoration', lambda: _restore_rng(local['rng']))
    buffers = _phase('restored buffer digest', lambda: _replicated_buffers(model))
    _agree('restored replicated buffers', buffers)
    if buffers != manifest['replicated_buffers_sha256']:
        raise ValueError('restored replicated buffers differ from checkpoint')
    return {**manifest, **local}
