"""Best-effort checkpoint disk admission; no reservation, writes, or deletion.

Estimates include complete tensor backing storages (including views), deduplicated
aliases, pickle metadata, 10% serialization margin and 1 MiB fixed overhead.
Admission keeps an additional 64 MiB free. Concurrent writers/compression/filesystem
allocation can defeat this advisory check; publication must still handle IO errors.
"""
from __future__ import annotations

import errno
import os
import pickle
import shutil

import torch
from torch.distributed.tensor import DTensor

RESERVE_BYTES = 64 * 1024 * 1024
SERIALIZATION_OVERHEAD_BYTES = 1024 * 1024


class _CountingSink:
    def __init__(self):
        self.count = 0

    def write(self, value):
        self.count += len(value)
        return len(value)


def checkpoint_space_estimate(payload) -> int:
    """Count metadata without retaining its serialization; count each storage once.

    DTensors contribute only their local tensor storage, suitable for per-rank DCP
    estimates. The estimate does not gather tensors or copy their contents.
    Payloads must be pickleable, as required by torch.save itself.
    """
    storages = {}

    class Counter(pickle.Pickler):
        def persistent_id(self, obj):
            if not isinstance(obj, torch.Tensor):
                return None
            local = obj.to_local() if isinstance(obj, DTensor) else obj
            # Sparse tensors have separate index/value backing storages.
            if local.layout == torch.sparse_coo:
                parts = (local._indices(), local._values())
            elif local.layout != torch.strided:
                raise ValueError(f"unsupported tensor layout for disk estimate: {local.layout}")
            else:
                parts = (local,)
            keys = []
            for part in parts:
                if part.device.type == 'meta':
                    raise ValueError('cannot estimate checkpoint storage for meta tensors')
                storage = part.untyped_storage()
                key = (str(part.device), storage._cdata)
                storages[key] = storage.nbytes()
                keys.append(key)
            return ('tensor', str(local.dtype), tuple(local.shape), tuple(keys))

    sink = _CountingSink()
    Counter(sink, protocol=pickle.HIGHEST_PROTOCOL).dump(payload)
    total = sum(storages.values()) + sink.count
    return total + (total + 9) // 10 + SERIALIZATION_OVERHEAD_BYTES


def require_checkpoint_space(directory, required_bytes: int) -> dict:
    """Check existing filesystem free space before staging, with 64 MiB reserve.

    Nothing is created and no space is reserved. Callers should aggregate per-rank
    estimates when DCP ranks share a filesystem, rather than checking each alone.
    """
    if isinstance(required_bytes, bool) or not isinstance(required_bytes, int) or required_bytes < 0:
        raise ValueError('required_bytes must be a nonnegative integer')
    path = os.path.abspath(os.fspath(directory))
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            raise FileNotFoundError(path)
        path = parent
    free = shutil.disk_usage(path).free
    info = {'free_bytes': free, 'required_bytes': required_bytes, 'reserve_bytes': RESERVE_BYTES}
    if free < required_bytes + RESERVE_BYTES:
        raise OSError(errno.ENOSPC, f'checkpoint staging needs {required_bytes} bytes plus '
                      f'{RESERVE_BYTES} reserve; only {free} bytes free', os.fspath(directory))
    return info
