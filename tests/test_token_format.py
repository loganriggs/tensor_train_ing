"""The public reader must not reinterpret int32 token bytes as uint16 pairs."""
import struct

import numpy as np
import pytest

from tnlm.data import TOKEN_DTYPE, read_split_tokens


def test_token_reader_preserves_values_count_and_read_only_mapping(tmp_path):
    expected = [0, 1, 256, 50256, 17]
    (tmp_path / 'train.tokens.bin').write_bytes(struct.pack('<5i', *expected))
    tokens = read_split_tokens(tmp_path, 'train')
    assert TOKEN_DTYPE == '<i4'
    assert isinstance(tokens, np.memmap)
    assert tokens.tolist() == expected
    assert not tokens.flags.writeable
    with pytest.raises(ValueError):
        tokens[0] = 99


def test_empty_split_is_read_only_and_typed(tmp_path):
    (tmp_path / 'val.tokens.bin').touch()
    tokens = read_split_tokens(tmp_path, 'val')
    assert tokens.shape == (0,) and tokens.dtype == np.dtype(TOKEN_DTYPE)
    assert not tokens.flags.writeable


@pytest.mark.parametrize('split', ['unknown', '../train', '/train'])
def test_reader_rejects_invalid_split(tmp_path, split):
    with pytest.raises(ValueError, match='invalid split'):
        read_split_tokens(tmp_path, split)


def test_reader_rejects_partial_token(tmp_path):
    (tmp_path / 'train.tokens.bin').write_bytes(b'\x01\x00\x00')
    with pytest.raises(ValueError, match='aligned'):
        read_split_tokens(tmp_path, 'train')


def test_tokenization_workers_keep_lightweight_imports():
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c',
        'import sys; import tnlm.data_preparation; assert "torch" not in sys.modules'],
        check=True, capture_output=True, timeout=30)
