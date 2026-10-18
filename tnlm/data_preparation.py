"""Bounded, ordered tokenization workers for immutable dataset preparation."""
from __future__ import annotations

import hashlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

from .token_format import TOKEN_DTYPE

_TOKENIZER = None


def _init_tokenizer():
    global _TOKENIZER
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding('gpt2')


def _encode_batch(batch, tokenizer=None):
    tokenizer = _TOKENIZER if tokenizer is None else tokenizer
    result = []
    for doc_id, text, source_offset, content_hash, rank in batch:
        processed = text.replace('\r\n', '\n').replace('\r', '\n')
        encoded = tokenizer.encode(processed, allowed_special=set(), disallowed_special=())
        # Packed results avoid millions of Python integer objects in IPC queues.
        wide_tokens = np.asarray(encoded, dtype='<i8')
        if len(wide_tokens) and (wide_tokens.min() < 0 or wide_tokens.max() >= 50257):
            raise ValueError('tokens must belong to the real GPT2 vocabulary')
        tokens = wide_tokens.astype(TOKEN_DTYPE)
        result.append((doc_id, source_offset, content_hash, rank,
                       hashlib.sha256(processed.encode('utf-8')).hexdigest(), tokens))
    return result


def _batches(rows, byte_limit):
    batch, size = [], 0
    for row in rows:
        row_bytes = len(row[1].encode('utf-8'))
        if batch and (size + row_bytes > byte_limit or len(batch) >= 128):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += row_bytes
    if batch:
        yield batch


def ordered_tokenization(rows, *, tokenizer, workers, batch_bytes):
    """Yield canonical-order results with bounded queued input and output.

    A source document is indivisible, so a single document larger than the byte
    budget is processed alone. Peak memory therefore also depends on the largest
    source document. No pool is started in the serial path.
    """
    batches = iter(_batches(rows, batch_bytes))
    if workers == 1:
        for batch in batches:
            yield from _encode_batch(batch, tokenizer)
        return
    pool = ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context('spawn'),
                               initializer=_init_tokenizer)
    pending = deque()
    try:
        for _ in range(2 * workers):
            batch = next(batches, None)
            if batch is None:
                break
            pending.append(pool.submit(_encode_batch, batch))
        while pending:
            # Completion order must never affect membership, offsets or caps.
            yield from pending.popleft().result()
            batch = next(batches, None)
            if batch is not None:
                pending.append(pool.submit(_encode_batch, batch))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
