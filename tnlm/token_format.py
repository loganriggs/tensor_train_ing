"""Lightweight token-file format shared by readers and tokenization workers."""
from pathlib import Path

import numpy as np

TOKEN_DTYPE = '<i4'  # On-disk tokens: little-endian signed 32-bit integers.
SPLIT_ORDER = ('train', 'val', 'test', 'calibration')


def read_split_tokens(manifest_dir, split):
    """Read a split's token payload using the dataset's on-disk format.

    Returns a read-only np.memmap, or a read-only empty ndarray for an empty
    split (NumPy cannot map an empty file). This low-level reader validates the
    split name and byte alignment; PackedSampler verifies manifest checksums
    and declared token counts before calling it.
    """
    if split not in SPLIT_ORDER:
        raise ValueError(f'invalid split: {split}')
    path = Path(manifest_dir) / f'{split}.tokens.bin'
    size = path.stat().st_size
    if size % np.dtype(TOKEN_DTYPE).itemsize:
        raise ValueError(f'token payload size is not aligned to {TOKEN_DTYPE}: {split}')
    if not size:
        tokens = np.empty(0, dtype=TOKEN_DTYPE)
        tokens.flags.writeable = False
        return tokens
    return np.memmap(path, dtype=TOKEN_DTYPE, mode='r')
