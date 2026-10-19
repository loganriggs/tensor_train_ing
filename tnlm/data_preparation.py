"""Bounded, ordered tokenization workers for immutable dataset preparation."""
from __future__ import annotations

import hashlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import zlib

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


def reclaim_candidate_pages(ledger):
    """Return disposable ledger free pages to the filesystem without a full copy."""
    ledger.commit()
    # Drain the PRAGMA cursor: incremental_vacuum may expose one result per page.
    ledger.execute('PRAGMA incremental_vacuum').fetchall()


def consume_candidates(ledger, *, batch_bytes, reclaim_bytes=32 * 1024 * 1024):
    """Consume compressed ranked rows, reclaiming storage as tokens are emitted.

    Input batches hold at most 128 compressed documents and the byte budget,
    except for an indivisible oversized document. Text is decompressed lazily.
    No SELECT remains active during DELETE/commit/vacuum. Deleting a queued row
    is safe because preparation is atomic and has no partial-resume contract.
    """
    released = 0
    while True:
        columns = 'SELECT id, text, source_offset, content_hash, rank FROM candidates'
        # Consumed rows are deleted, so the index front is the next batch.
        # Avoid a disjunctive keyset predicate that can rescan a large index.
        cursor = ledger.execute(columns + ' ORDER BY rank ASC, id ASC LIMIT 128')
        rows, size = [], 0
        try:
            for row in cursor:
                if rows and size + len(row[1]) > batch_bytes:
                    break
                rows.append(row)
                size += len(row[1])
                if size >= batch_bytes:
                    break
        finally:
            cursor.close()
        if not rows:
            reclaim_candidate_pages(ledger)
            return
        ledger.executemany('DELETE FROM candidates WHERE id=?', ((row[0],) for row in rows))
        released += size
        if released >= reclaim_bytes:
            reclaim_candidate_pages(ledger)
            released = 0
        for doc_id, compressed, offset, content_hash, rank in rows:
            yield doc_id, zlib.decompress(compressed).decode('utf-8'), offset, content_hash, rank
