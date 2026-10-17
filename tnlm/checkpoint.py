"""Atomic recovery checkpoints. Load only trusted local files: torch pickle can execute code."""
from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

from .checkpoint_storage import checkpoint_space_estimate, require_checkpoint_space


def _plain(value):
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _git():
    def run(*args):
        result = subprocess.run(['git', *args], cwd=Path(__file__).parent,
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    status = run('status', '--porcelain')
    package = Path(__file__).parent
    return {'revision': run('rev-parse', 'HEAD'),
            'dirty': None if status is None else bool(status),
            'source_sha256': {f'tnlm/{p.relative_to(package).as_posix()}': _digest(p)
                              for p in sorted(package.rglob('*.py'))}}


def _rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _validate_rng(state):
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('Checkpoint has CUDA RNG state but CUDA is unavailable')
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from checkpoint')


def _restore_rng(state):
    _validate_rng(state)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(staging, path):
    # Linux renameat2(RENAME_NOREPLACE) closes the concurrent-writer race,
    # including an existing empty destination directory.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(path), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(path))


def _require_finite(value, location):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f'Nonfinite checkpoint tensor: {location}')
    elif isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, f'{location}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, f'{location}[{index}]')


def save_checkpoint(path, model, optimizer, sampler, *, config, progress,
                    manifest_hash, scheduler=None, scaler=None, parent=None, extra=None):
    """Save full recovery state to a new immutable directory; return its manifest.

    A sibling staging directory is flushed before atomic publication. Readers
    require the final manifest and verify the state bytes before unpickling.
    """
    _require_finite(model.state_dict(), 'model')
    if optimizer is not None:
        _require_finite(optimizer.state_dict(), 'optimizer')
    for name, module in model.named_modules():
        for buffer_name in ('acc_count', 'acc_sum'):
            value = getattr(module, buffer_name, None)
            if value is not None and bool(torch.as_tensor(value).ne(0).any()):
                raise ValueError(f'Checkpoint requires an optimizer-step boundary: '
                                 f'{name}.{buffer_name} has unfinished statistic collection')
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Checkpoint already exists: {path}')
    payload = {
        'format_version': 1, 'model': model.state_dict(),
        'model_class': f'{type(model).__module__}.{type(model).__qualname__}',
        'optimizer_class': (f'{type(optimizer).__module__}.{type(optimizer).__qualname__}'
                            if optimizer is not None else None),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'sampler': sampler.state_dict() if sampler is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'rng': _rng_state(), 'config': _plain(config), 'progress': _plain(progress),
        'manifest_hash': manifest_hash, 'parent': _plain(parent), 'extra': extra,
        'code': _git(), 'environment': {
            'python': sys.version, 'platform': platform.platform(),
            'distributions': dict(sorted(
                (dist.metadata['Name'], dist.version)
                for dist in importlib.metadata.distributions()
                if dist.metadata['Name'])),
            'torch': str(torch.__version__), 'numpy': np.__version__,
            'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'torch_build': torch.__config__.show(),
            'devices': [{'name': torch.cuda.get_device_name(i),
                         'capability': list(torch.cuda.get_device_capability(i))}
                        for i in range(torch.cuda.device_count())],
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'kernel_revision': os.environ.get('TNLM_KERNEL_REVISION'),
        },
    }
    disk_admission = require_checkpoint_space(path.parent, checkpoint_space_estimate(
        {'payload': payload, 'manifest': {'data_manifest_hash': manifest_hash, 'parent': _plain(parent)}}))
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{path.name}.tmp-', dir=path.parent))
    try:
        state_path = staging / 'state.pt'
        with state_path.open('wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        manifest = {'format_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'files': {'state.pt': {'sha256': _digest(state_path),
                                            'bytes': state_path.stat().st_size}},
                    'data_manifest_hash': manifest_hash, 'parent': _plain(parent),
                    'disk_admission': disk_admission}
        with (staging / 'manifest.json').open('w') as f:
            json.dump(manifest, f, sort_keys=True, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        _sync_directory(staging)
        _publish(staging, path)
        _sync_directory(path.parent)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_checkpoint(path, model, optimizer=None, sampler=None, *,
                    expected_manifest_hash=None, expected_config=None,
                    scheduler=None, scaler=None, restore_rng=True, map_location='cpu',
                    config_validator=None):
    """Verify and restore a *trusted local* checkpoint; return the full payload.

    Checksums detect corruption, not malicious pickle content or authenticity.
    Config and data identity are checked before mutating model or optimizer.
    Optional config_validator(saved_config) may raise ValueError for caller-defined
    compatibility rules; strict expected_config checking remains independent.
    """
    path = Path(path)
    try:
        manifest = json.loads((path / 'manifest.json').read_text())
        info = manifest['files']['state.pt']
        digest, size = info['sha256'], info['bytes']
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest) or
                type(size) is not int or size <= 0 or 'data_manifest_hash' not in manifest):
            raise ValueError('invalid state checksum metadata')
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f'Incomplete or invalid checkpoint: {path}') from e
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    state_path = path / 'state.pt'
    if not state_path.is_file() or state_path.stat().st_size != info['bytes'] or _digest(state_path) != info['sha256']:
        raise ValueError('Checkpoint state checksum/size mismatch')
    payload = torch.load(state_path, map_location=map_location, weights_only=False)
    if payload.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint state format')
    if payload.get('parent') != manifest.get('parent'):
        raise ValueError('Checkpoint parent identity is inconsistent')
    if payload['manifest_hash'] != manifest['data_manifest_hash']:
        raise ValueError('Checkpoint data identity is inconsistent')
    if expected_manifest_hash is not None and payload['manifest_hash'] != expected_manifest_hash:
        raise ValueError('Data manifest hash mismatch')
    if expected_config is not None and payload['config'] != _plain(expected_config):
        raise ValueError('Checkpoint config mismatch')
    if config_validator is not None:
        compatibility = config_validator(payload['config'])
        if compatibility is not None:
            payload['config_compatibility'] = compatibility
    for name, obj in [('optimizer', optimizer), ('sampler', sampler),
                      ('scheduler', scheduler), ('scaler', scaler)]:
        if obj is not None and payload[name] is None:
            raise ValueError(f'Checkpoint is missing {name} state')
    _require_finite(payload['model'], 'model')
    _require_finite(payload['optimizer'], 'optimizer')
    # PyTorch's strict load reports missing keys only AFTER copying valid tensors.
    # Check shapes and complete persistent-buffer coverage before any mutation.
    current = model.state_dict()
    incoming = payload['model']
    if current.keys() != incoming.keys():
        raise ValueError('Checkpoint model keys/buffers mismatch')
    for name, value in current.items():
        saved = incoming[name]
        if isinstance(value, torch.Tensor) and (not isinstance(saved, torch.Tensor) or value.shape != saved.shape):
            raise ValueError(f'Checkpoint model shape mismatch: {name}')
    if optimizer is not None:
        identity = f'{type(optimizer).__module__}.{type(optimizer).__qualname__}'
        if payload.get('optimizer_class') not in (None, identity):
            raise ValueError('Checkpoint optimizer class mismatch')
        def validate_groups(actual_optimizer, saved_state):
            if not isinstance(saved_state, dict):
                raise ValueError('Checkpoint optimizer state layout mismatch')
            if 'param_groups' in saved_state:
                groups = saved_state['param_groups']
                if len(groups) != len(actual_optimizer.param_groups) or any(
                        len(saved.get('params', [])) != len(actual['params'])
                        for saved, actual in zip(groups, actual_optimizer.param_groups)):
                    raise ValueError('Checkpoint optimizer parameter groups mismatch')
            elif set(saved_state) == {'muon', 'adamw'} and all(
                    isinstance(getattr(actual_optimizer, name, None), torch.optim.Optimizer)
                    for name in saved_state):
                for name, child_state in saved_state.items():
                    validate_groups(getattr(actual_optimizer, name), child_state)
            else:
                raise ValueError('Checkpoint optimizer state layout mismatch')
        validate_groups(optimizer, payload['optimizer'])
    if restore_rng:
        _validate_rng(payload['rng'])
    model.load_state_dict(incoming, strict=True)
    for name, obj in [('optimizer', optimizer), ('sampler', sampler),
                      ('scheduler', scheduler), ('scaler', scaler)]:
        if obj is not None:
            obj.load_state_dict(payload[name])
    # These deliberately nonpersistent statistics belong to the abandoned
    # accumulation, not the restored step-boundary checkpoint.
    for module in model.modules():
        for name in ('acc_count', 'acc_sum'):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor):
                with torch.no_grad():
                    value.zero_()
    if restore_rng:
        _restore_rng(payload['rng'])
    return payload
