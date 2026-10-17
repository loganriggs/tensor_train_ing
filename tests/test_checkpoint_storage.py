import errno
from collections import namedtuple

import pytest
import torch

from tnlm.checkpoint_storage import (
    RESERVE_BYTES, SERIALIZATION_OVERHEAD_BYTES,
    checkpoint_space_estimate, require_checkpoint_space,
)


def test_aliases_views_and_storage_count():
    tensor = torch.empty(4096, dtype=torch.float64)
    single = checkpoint_space_estimate({'tensor': tensor})
    alias = checkpoint_space_estimate({'tensor': tensor, 'alias': tensor, 'view': tensor[:1]})
    clone = checkpoint_space_estimate({'tensor': tensor, 'clone': tensor.clone()})
    view = checkpoint_space_estimate(tensor[:1])
    assert view >= tensor.untyped_storage().nbytes() + SERIALIZATION_OVERHEAD_BYTES
    assert alias - single < 1024
    assert clone - single >= tensor.untyped_storage().nbytes()


def test_mixed_dtype_metadata_and_finite_bytes():
    payload = {'float': torch.empty(20, dtype=torch.float64),
               'byte': torch.empty(30, dtype=torch.uint8), 'metadata': 'x' * 100000}
    estimate = checkpoint_space_estimate(payload)
    assert isinstance(estimate, int)
    assert estimate >= SERIALIZATION_OVERHEAD_BYTES + 110000 + 190
    assert estimate < SERIALIZATION_OVERHEAD_BYTES + 120000
    assert checkpoint_space_estimate({'a': 1}) >= SERIALIZATION_OVERHEAD_BYTES


def test_sparse_storage_and_meta_rejection():
    tensor = torch.sparse_coo_tensor([[0, 1]], [2., 3.], (4,))
    assert checkpoint_space_estimate(tensor) >= SERIALIZATION_OVERHEAD_BYTES + 24
    with pytest.raises(ValueError, match='meta'):
        checkpoint_space_estimate(torch.empty(2, device='meta'))


def test_admission_no_writes_and_exact_boundary(tmp_path, monkeypatch):
    usage = namedtuple('Usage', 'total used free')
    requested = tmp_path / 'absent' / 'checkpoint'
    seen = []
    def disk(path):
        seen.append(path)
        return usage(100, 0, RESERVE_BYTES + 100)
    monkeypatch.setattr('tnlm.checkpoint_storage.shutil.disk_usage', disk)
    info = require_checkpoint_space(requested, 100)
    assert info == {'free_bytes': RESERVE_BYTES + 100, 'required_bytes': 100,
                    'reserve_bytes': RESERVE_BYTES}
    assert seen == [str(tmp_path)]
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(OSError) as error:
        require_checkpoint_space(requested, 101)
    assert error.value.errno == errno.ENOSPC
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('value', [-1, 1.5, float('inf'), True])
def test_invalid_required_size(tmp_path, value):
    with pytest.raises(ValueError):
        require_checkpoint_space(tmp_path, value)


def test_dtensor_counts_local_storage(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard
    dist.init_process_group('gloo', init_method=f'file://{tmp_path / "group"}', rank=0, world_size=1)
    try:
        mesh = init_device_mesh('cpu', (1,))
        distributed = distribute_tensor(torch.arange(128), mesh, [Shard(0)])
        assert checkpoint_space_estimate(distributed) == checkpoint_space_estimate(distributed.to_local())
    finally:
        dist.destroy_process_group()
