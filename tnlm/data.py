"""Immutable document datasets and deterministic packed batches.

Targets never cross document boundaries; padding targets are -100. Packing is
row-local, and document fragments retain original token offsets in documents.jsonl.
"""
from __future__ import annotations

import hashlib
import sqlite3
import zlib
import importlib.metadata
import json
import math
import os
import shutil
import random
from pathlib import Path

import numpy as np
import torch

from .token_format import TOKEN_DTYPE, SPLIT_ORDER, read_split_tokens

PACKING_VERSION = 'document-shuffle-row-overlap-incremental-v2'


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest_hash(path):
    """SHA256 of a completed manifest, validating its independently stored digest."""
    path = Path(path)
    if path.is_dir():
        path /= 'manifest.json'
    digest = _sha(path)
    if path.with_name('manifest.sha256').read_text().strip() != digest:
        raise ValueError('manifest checksum mismatch')
    return digest


def validate_tokenizer_compatibility(training_manifest, validation_manifest, *, allow_test_only=False):
    """Require the same declared tokenization semantics for checkpoint probes.

    Production fingerprints bind vocabulary ranks, regex and special tokens.
    Injected offline fixtures carry no such fingerprint; their weaker metadata
    agreement is explicitly labeled and cannot match production data.
    """
    records, hashes = [], []
    for source in (training_manifest, validation_manifest):
        path = Path(source)
        if path.is_dir():
            path /= 'manifest.json'
        hashes.append(manifest_hash(path))
        record = json.loads(path.read_text()).get('tokenizer')
        if not isinstance(record, dict):
            raise ValueError('missing tokenizer identity in dataset manifest')
        records.append(record)
    left, right = records
    fixture = any(record.get('injected') or record.get('test_only') for record in records)
    if fixture:
        if not allow_test_only:
            raise ValueError('test-only tokenizer data is restricted to CPU fixture probes')
        if (any(record.get('injected') is not True or record.get('test_only') is not True
                for record in records) or left != right or not left.get('name')
                or type(left.get('eos_id')) is not int):
            raise ValueError('incompatible test-only tokenizer metadata')
        identity = dict(left)
        status = 'test_only_metadata_match'
    else:
        fields = ('name', 'eos_id', 'fingerprint_sha256', 'vocabulary_sha256',
                  'regex_pattern', 'special_tokens')
        for record in records:
            if (any(key not in record for key in fields)
                    or not isinstance(record['name'], str) or not record['name']
                    or type(record['eos_id']) is not int
                    or not isinstance(record['regex_pattern'], str)
                    or not isinstance(record['special_tokens'], dict)):
                raise ValueError('incomplete tokenizer identity in dataset manifest')
            if record['name'] != 'gpt2' or record['eos_id'] != 50256:
                raise ValueError('production tokenizer must use the pinned GPT2 encoding and EOS')
            for key in ('fingerprint_sha256', 'vocabulary_sha256'):
                value = record[key]
                if (not isinstance(value, str) or len(value) != 64
                        or any(c not in '0123456789abcdef' for c in value)):
                    raise ValueError('invalid tokenizer fingerprint in dataset manifest')
        identity = {key: left[key] for key in fields}
        if identity != {key: right[key] for key in fields}:
            raise ValueError('validation tokenizer identity differs from training tokenizer')
        status = 'fingerprint_match'
    return {'status': status, 'training_manifest_sha256': hashes[0],
            'validation_manifest_sha256': hashes[1], 'identity': identity,
            'scope': 'checksum-bound declared tokenizer identity; injected fixtures are not production evidence'}


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def _load_heldout_exclusion(ledger, manifest):
    """Authenticate and stream the fixed validation identity into an indexed ledger."""
    path = Path(manifest)
    if path.is_dir():
        path /= 'manifest.json'
    identity = manifest_hash(path)
    info = json.loads(path.read_text())
    documents = path.with_name('documents.jsonl')
    expected = info.get('artifacts', {}).get('documents.jsonl')
    if not isinstance(expected, str) or _sha(documents) != expected:
        raise ValueError('heldout documents artifact checksum mismatch')
    ledger.execute('CREATE TABLE heldout (kind TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(kind, value)) WITHOUT ROWID')
    count = 0
    with documents.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('split') != 'val':
                continue
            if not isinstance(row.get('id'), str):
                raise ValueError('heldout validation document requires string ID')
            for name in ('content_sha256', 'processed_sha256'):
                value = row.get(name)
                if (not isinstance(value, str) or len(value) != 64
                        or any(c not in '0123456789abcdef' for c in value)):
                    raise ValueError('heldout validation document requires content fingerprints')
            ledger.executemany('INSERT OR IGNORE INTO heldout VALUES (?, ?)',
                               [('id', row['id']), ('raw', row['content_sha256']),
                                ('processed', row['processed_sha256'])])
            count += 1
            if count % 8192 == 0:
                ledger.commit()
    if not count:
        raise ValueError('heldout manifest has no validation documents')
    if _sha(documents) != expected or manifest_hash(path) != identity:
        raise ValueError('heldout validation metadata changed during reading')
    ledger.commit()
    return {'manifest_sha256': identity, 'documents_sha256': expected,
            'validation_documents': count, 'excluded_unique_documents': 0,
            'rule': 'exclude every source document matching a fixed validation ID, raw content SHA256, or CRLF/CR-to-LF processed content SHA256, before selection and from all output splits'}


def _prepare_data(output_dir, *, source_jsonl=None, revision, max_tokens=50_000_000,
                 max_documents=100_000, tokenizer=None, split_fractions=None,
                 hf_dataset=None, hf_config=None, hf_data_files=None, hf_shard_checksums=None, hf_local_shards=None,
                 workers=1, tokenization_batch_bytes=1048576, exclude_validation_manifest=None,
                 on_shard_scanned=None):
    """Freeze bounded documents. HF mode requires a full commit and explicit shards.

    Local revisions are user-supplied labels supplemented by source SHA256. Tokens
    map to source document token offsets, not character offsets (BPE may split bytes).
    Existing output directories are rejected to prevent accidental replacement.
    """
    if on_shard_scanned is not None:
        if not callable(on_shard_scanned) or hf_dataset is None:
            raise ValueError('on_shard_scanned requires a callable and an HF dataset')
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    if type(tokenization_batch_bytes) is not int or tokenization_batch_bytes < 1:
        raise ValueError('tokenization_batch_bytes must be a positive integer')
    if tokenizer is not None and workers != 1:
        raise ValueError('injected test tokenizers require workers=1')
    if not revision or revision in ('main', 'master', 'latest'):
        raise ValueError('an immutable source revision is required')
    if any(type(value) is not int or value <= 0 for value in (max_tokens, max_documents)):
        raise ValueError('positive integer data bounds required')
    if (source_jsonl is None) == (hf_dataset is None):
        raise ValueError('specify exactly one local JSONL or HF dataset')
    fractions = {'train': .97, 'val': .01, 'test': .01, 'calibration': .01} if split_fractions is None else split_fractions
    if set(fractions) != {'train', 'val', 'test', 'calibration'} or any(not math.isfinite(v) or v < 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError('split fractions must specify train/val/test/calibration and sum to one')
    fractions = {split: fractions[split] for split in SPLIT_ORDER}
    if tokenizer is None:
        import tiktoken
        tokenizer = tiktoken.get_encoding('gpt2')
        ranks = sorted((k.hex(), v) for k, v in tokenizer._mergeable_ranks.items())
        tok_meta = {'name': 'gpt2', 'package': 'tiktoken', 'version': importlib.metadata.version('tiktoken'),
                    'vocabulary_sha256': _digest(json.dumps(ranks)), 'eos_id': tokenizer.eot_token,
                    'special_tokens': tokenizer._special_tokens,
                    'regex_pattern': tokenizer._pat_str}
        tok_meta['fingerprint_sha256'] = _digest(json.dumps({
            'ranks': ranks, 'regex_pattern': tokenizer._pat_str,
            'special_tokens': tokenizer._special_tokens}, sort_keys=True))
    else:
        tok_meta = {'name': getattr(tokenizer, 'name', type(tokenizer).__name__),
                    'injected': True, 'test_only': True, 'eos_id': tokenizer.eot_token}
    if tokenizer.eot_token != 50256:
        raise ValueError('GPT2 EOS must be 50256')
    if source_jsonl is not None:
        source_jsonl = Path(source_jsonl).resolve()
        source = {'kind': 'jsonl', 'revision': revision, 'path': str(source_jsonl), 'sha256': _sha(source_jsonl)}
        def records():
            with source_jsonl.open() as f:
                for line_number, line in enumerate(f, 1):
                    if line.strip():
                        yield json.loads(line), line_number
    else:
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()) or not hf_data_files:
            raise ValueError('HF requires full 40-character commit and explicit data_files')
        shards = [hf_data_files] if isinstance(hf_data_files, str) else hf_data_files
        if not isinstance(shards, (list, tuple)) or not all(isinstance(x, str) and x and not any(c in x for c in '*?[') for x in shards):
            raise ValueError('HF data_files must be explicit shard paths, without patterns')
        if not isinstance(hf_shard_checksums, dict) or set(hf_shard_checksums) != set(shards):
            raise ValueError('HF shard_checksums must cover every explicit shard exactly')
        if any(not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest.lower()) for digest in hf_shard_checksums.values()):
            raise ValueError('HF shard checksums must be SHA256 hex digests')
        if len(set(shards)) != len(shards):
            raise ValueError('HF shard paths must be unique')
        if hf_local_shards is not None and set(hf_local_shards) != set(shards):
            raise ValueError('HF local shard paths must cover every declared shard exactly')
        verified = []
        for shard in sorted(shards):
            if hf_local_shards is None:
                from huggingface_hub import hf_hub_download
                path = Path(hf_hub_download(repo_id=hf_dataset, repo_type='dataset',
                                            revision=revision, filename=shard))
            else:
                path = Path(hf_local_shards[shard])
            if _sha(path) != hf_shard_checksums[shard].lower():
                raise ValueError(f'HF raw shard checksum mismatch: {shard}')
            verified.append((shard, path))
        source = {'kind': 'huggingface', 'dataset': hf_dataset, 'config': hf_config,
                  'revision': revision, 'shards': sorted(shards),
                  'shard_sha256': {k: v.lower() for k, v in hf_shard_checksums.items()},
                  'shard_hash_verification': 'SHA256 of every local raw shard verified before parquet reading'}
        def records():
            import pyarrow.parquet as pq
            for shard, path in verified:
                row = 0
                parquet = pq.ParquetFile(path)
                try:
                    schema = getattr(parquet, 'schema_arrow', None)
                    projection = ({'columns': [name for name in ('text', 'id', 'doc_id')
                                               if name in schema.names]}
                                  if schema is not None else {})
                    for batch in parquet.iter_batches(batch_size=256, **projection):
                        for record in batch.to_pylist():
                            yield record, {'shard': shard, 'row': row}
                            row += 1
                    if _sha(path) != hf_shard_checksums[shard].lower():
                        raise ValueError(f'HF raw shard changed during reading: {shard}')
                finally:
                    close = getattr(parquet, 'close', None)
                    try:
                        if close is not None:
                            close()
                    finally:
                        # Do not retain a bound close method or the last Arrow
                        # batch while the next shard's reader/buffers allocate.
                        close = parquet = batch = record = None
                if on_shard_scanned is not None:
                    on_shard_scanned(shard, path)
    out = Path(output_dir)
    # Both duplicate detection and ranked membership live on disk: increasing
    # max_documents must not grow a Python heap or document metadata list.
    selected_count = 0
    worst_rank = None
    scan_records = unique_documents = 0
    ledger_path = out / '.seen.sqlite'
    ledger = sqlite3.connect(ledger_path)
    ledger.execute('PRAGMA auto_vacuum=INCREMENTAL')
    ledger.execute('PRAGMA cache_size=-32768')
    ledger.execute('PRAGMA temp_store=FILE')
    ledger.execute('CREATE TABLE seen (id TEXT PRIMARY KEY, content_hash TEXT NOT NULL)')
    ledger.execute('CREATE TABLE candidates (id TEXT PRIMARY KEY, text BLOB NOT NULL, source_offset TEXT NOT NULL, content_hash TEXT NOT NULL, rank TEXT NOT NULL)')
    ledger.execute('CREATE INDEX candidate_rank ON candidates(rank ASC, id ASC)')
    streams = {}
    tokenized = None
    total = document_count = 0
    offsets = dict.fromkeys(fractions, 0)
    heldout_exclusion = None
    try:
        if exclude_validation_manifest is not None:
            heldout_exclusion = _load_heldout_exclusion(ledger, exclude_validation_manifest)
        for record, source_offset in records():
            scan_records += 1
            text = record.get('text')
            if not isinstance(text, str):
                raise ValueError('every record requires string text')
            content_hash = _digest(text)
            stable_id = record.get('id', record.get('doc_id'))
            doc_id = content_hash if stable_id is None else str(stable_id)
            previous = ledger.execute('SELECT content_hash FROM seen WHERE id=?', (doc_id,)).fetchone()
            if previous:
                if previous[0] != content_hash:
                    raise ValueError(f'conflicting content for duplicate document ID: {doc_id}')
                continue
            ledger.execute('INSERT INTO seen VALUES (?, ?)', (doc_id, content_hash))
            unique_documents += 1
            if unique_documents % 8192 == 0:
                ledger.commit()
            if heldout_exclusion is not None:
                processed_hash = (_digest(text.replace('\r\n', '\n').replace('\r', '\n'))
                                  if '\r' in text else content_hash)
                match = ledger.execute(
                    "SELECT 1 FROM heldout WHERE (kind='id' AND value=?) OR "
                    "(kind='raw' AND value=?) OR (kind='processed' AND value=?) LIMIT 1",
                    (doc_id, content_hash, processed_hash)).fetchone()
                if match:
                    heldout_exclusion['excluded_unique_documents'] += 1
                    continue
            rank = _digest('selection\0' + doc_id)
            retain = selected_count < max_documents or rank < worst_rank
            if retain:
                if selected_count == max_documents:
                    # Match the legacy heap tie-break: evict the smallest ID
                    # among the largest ranks; equal ranks do not replace.
                    ledger.execute('DELETE FROM candidates WHERE id=(SELECT id FROM candidates ORDER BY rank DESC, id ASC LIMIT 1)')
                else:
                    selected_count += 1
                ledger.execute('INSERT INTO candidates VALUES (?, ?, ?, ?, ?)',
                               (doc_id, zlib.compress(text.encode('utf-8'), level=1), json.dumps(source_offset), content_hash, rank))
                if selected_count == max_documents:
                    worst_rank = ledger.execute('SELECT rank FROM candidates ORDER BY rank DESC, id ASC LIMIT 1').fetchone()[0]
        ledger.commit()
        if source_jsonl is not None and _sha(source_jsonl) != source['sha256']:
            raise ValueError('local source changed during preparation')
        streams = {split: (out / f'{split}.tokens.bin').open('wb') for split in fractions}
        from .data_preparation import ordered_tokenization, consume_candidates, reclaim_candidate_pages
        # Duplicate conflicts are checked over the full source before this point.
        # Reclaim their ledger and progressively release selected source payloads
        # so large preparations do not retain all text beside the final tokens.
        ledger.execute('DROP TABLE seen')
        reclaim_candidate_pages(ledger)
        rows = consume_candidates(ledger, batch_bytes=tokenization_batch_bytes)
        tokenized = ordered_tokenization(rows, tokenizer=tokenizer, workers=workers,
                                         batch_bytes=tokenization_batch_bytes)
        with (out / 'documents.jsonl').open('w') as metadata:
            for doc_id, serialized_offset, content_hash, rank, processed_hash, encoded in tokenized:
                remaining = max_tokens - total
                encoded_count = len(encoded)
                eos_included = encoded_count + 1 <= remaining
                tokens = encoded[:remaining]
                if len(tokens) and (tokens.min() < 0 or tokens.max() >= 50257):
                    raise ValueError('tokens must belong to the real GPT2 vocabulary')
                token_count = len(tokens) + int(eos_included)
                split_rank = int(_digest(doc_id), 16)
                cumulative = 0.
                for split, fraction in fractions.items():
                    cumulative += fraction
                    if split_rank < int(cumulative * 2**256):
                        break
                start = offsets[split]
                tokens.tofile(streams[split])
                if eos_included:
                    streams[split].write(np.asarray([tokenizer.eot_token], dtype=TOKEN_DTYPE).tobytes())
                offsets[split] += token_count
                doc = {'id': doc_id, 'numeric_id': document_count, 'split': split,
                       'selection_sha256': rank,
                       'content_sha256': content_hash, 'processed_sha256': processed_hash,
                       'source_offset': json.loads(serialized_offset),
                       'token_start': start, 'token_end': offsets[split],
                       'source_token_start': 0, 'source_token_end': token_count,
                       'processed_token_start': 0, 'processed_token_end': min(token_count, encoded_count),
                       'synthetic_eos_token_offset': encoded_count if eos_included else None,
                       'truncated': not eos_included}
                metadata.write(json.dumps(doc, sort_keys=True) + '\n')
                document_count += 1
                total += token_count
                if total >= max_tokens:
                    break
    finally:
        if tokenized is not None:
            tokenized.close()
        for stream in streams.values():
            stream.close()
        ledger.close()
        ledger_path.unlink(missing_ok=True)
        Path(str(ledger_path) + '-journal').unlink(missing_ok=True)
    artifacts = {p.name: _sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    manifest = {'format_version': 1, 'source': source, 'tokenizer': tok_meta,
                'split_fractions': fractions, 'split_order': list(SPLIT_ORDER),
                'tokens': offsets, 'documents': document_count,
                'scan_records': scan_records, 'scan_unique_documents': unique_documents,
                'max_tokens': max_tokens, 'max_documents': max_documents,
                'preprocessing': 'CRLF/CR to LF; otherwise unchanged',
                'deduplication': 'identical stable-ID/content duplicates collapse; conflicting contents rejected',
                'membership': 'SHA256(UTF8(stable document ID)), fixed fraction intervals',
                'hash_domains': {'selection': 'UTF8(selection + NUL + stable document ID)',
                                 'split': 'UTF8(stable document ID)'},
                'bounded_selection': 'smallest domain-separated selection SHA256 ranks across entire declared source; canonical rank order then token cap',
                'preparation_cost': 'full declared source scan and HF raw-shard download/hash; output token/document bounds do not bound scan cost',
                'token_offset_semantics': 'source_token_* are legacy processed document stream offsets including synthetic EOS; processed_token_* exclude EOS; no original character/byte alignment',
                'boundary_loss_policy': 'ignore cross-document transitions and padding',
                'artifacts': artifacts}
    if heldout_exclusion is not None:
        manifest['heldout_exclusion'] = heldout_exclusion
    # Sidecar precedes the manifest: manifest presence is the completion marker.
    encoded_manifest = json.dumps(manifest, sort_keys=True, indent=2) + '\n'
    (out / 'manifest.sha256').write_text(_digest(encoded_manifest) + '\n')
    for artifact in out.iterdir():
        with artifact.open('rb') as f:
            os.fsync(f.fileno())
    # Derived mmap metadata is ready before the completion marker is published.
    # It is deliberately outside the immutable manifest's artifact identity.
    from .sampler_index import ensure_sampler_index
    ensure_sampler_index(out, manifest)
    temporary_manifest = out / '.manifest.json.tmp'
    with temporary_manifest.open('w') as f:
        f.write(encoded_manifest)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_manifest, out / 'manifest.json')
    directory_fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    parent_fd = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return out / 'manifest.json'


def prepare_data(output_dir, *, source_jsonl=None, revision, max_tokens=50_000_000,
                 max_documents=100_000, tokenizer=None, split_fractions=None,
                 hf_dataset=None, hf_config=None, hf_data_files=None, hf_shard_checksums=None, hf_local_shards=None,
                 workers=1, tokenization_batch_bytes=1048576, exclude_validation_manifest=None,
                 on_shard_scanned=None):
    """Prepare an immutable dataset; remove this invocation's output on failure.

    HF raw-shard SHA256 values are checked against downloaded pinned files or
    hf_local_shards={declared_path: local_path}. Every declared shard is fully read.
    Output bounds do not bound source scanning/download costs; choose shards first.
    Injected tokenizers mark test-only data and require workers=1. Parallel GPT2
    tokenization preserves canonical document order and the exact manifest format.
    Queue memory is bounded by workers and tokenization_batch_bytes, except that
    an indivisible document larger than the budget is processed alone. Selection
    and metadata use disk rather than Python collections proportional to the cap.
    exclude_validation_manifest optionally excludes matching validation IDs and
    exact raw/normalized document hashes before selection from every output split.
    An optional HF-only on_shard_scanned(shard, path) callback runs once after
    the entire shard is consumed, its second checksum passes and its reader is
    closed. The preparer never deletes raw files; a callback may manage its own
    staging. Callback failures abort preparation and remove incomplete output.
    Successful publication fsyncs all files and atomically installs the manifest.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        return _prepare_data(out, source_jsonl=source_jsonl, revision=revision,
                             max_tokens=max_tokens, max_documents=max_documents,
                             tokenizer=tokenizer, split_fractions=split_fractions,
                             hf_dataset=hf_dataset, hf_config=hf_config,
                             hf_data_files=hf_data_files, hf_shard_checksums=hf_shard_checksums,
                             hf_local_shards=hf_local_shards, workers=workers,
                             tokenization_batch_bytes=tokenization_batch_bytes,
                             exclude_validation_manifest=exclude_validation_manifest,
                             on_shard_scanned=on_shard_scanned)
    except BaseException:
        shutil.rmtree(out)
        raise


class PackedSampler:
    """Deterministic epoch permutations, row packing, and resumable next-batch order."""
    def __init__(self, manifest, split='train', batch_size=1, context=512, seed=0):
        self.path = Path(manifest)
        if self.path.is_dir():
            self.path /= 'manifest.json'
        self.manifest_hash = manifest_hash(self.path)
        self.manifest = json.loads(self.path.read_text())
        root = self.path.parent
        for name, expected in self.manifest['artifacts'].items():
            if Path(name).name != name or _sha(root / name) != expected:
                raise ValueError(f'artifact checksum mismatch: {name}')
        if split not in self.manifest['tokens'] or batch_size <= 0 or context <= 0:
            raise ValueError('invalid split or batch shape')
        self.split, self.batch_size, self.context, self.seed = split, batch_size, context, seed
        required = {'documents.jsonl', *(f'{name}.tokens.bin' for name in SPLIT_ORDER)}
        if not required.issubset(self.manifest['artifacts']):
            raise ValueError('dataset manifest lacks required metadata/token artifacts')
        if self.manifest.get('format_version') != 1:
            raise ValueError('unsupported dataset format')
        counts = self.manifest['tokens']
        if set(counts) != set(SPLIT_ORDER) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError('invalid dataset token counts')
        for name, count in counts.items():
            if (root / f'{name}.tokens.bin').stat().st_size != count * np.dtype(TOKEN_DTYPE).itemsize:
                raise ValueError(f'token payload size mismatch: {name}')
        from .sampler_index import load_sampler_index
        self.docs = load_sampler_index(root, self.manifest, split)
        if not self.docs:
            raise ValueError(f'empty split: {split}')
        self.tokens = read_split_tokens(root, split)
        self.packing_version = PACKING_VERSION
        self.epoch = self.cursor = 0
        self._order()
        self._batch_mappings = []
        self._context_transition = None

    def _permutation(self, epoch):
        if getattr(self, '_order_epoch', None) == epoch:
            return self.order
        if not hasattr(self.docs, 'cache_dir'):
            order = np.arange(len(self.docs), dtype=np.int64)
            random.Random(f'{self.seed}:{epoch}').shuffle(order)
            order.flags.writeable = False
            return order
        import fcntl
        import os
        import tempfile
        key = {'version': 1, 'manifest': self.manifest_hash, 'split': self.split,
               'seed': str(self.seed), 'epoch': epoch, 'documents': len(self.docs)}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        cache = self.docs.cache_dir / 'packing'
        cache.mkdir(exist_ok=True)
        path, meta = cache / f'{digest}.npy', cache / f'{digest}.json'
        with (cache / f'{digest}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                saved = json.loads(meta.read_text())
                if saved['identity'] != key or _sha(path) != saved['sha256']:
                    raise ValueError('invalid permutation cache')
                order = np.load(path, mmap_mode='r', allow_pickle=False)
                if order.shape != (len(self.docs),) or order.dtype != np.dtype('<i8'):
                    raise ValueError('invalid permutation cache shape')
                return order
            except (OSError, ValueError, KeyError, TypeError):
                pass
            # The first builder owns one compact array; subsequent ranks map
            # the same read-only file. Python's shuffle retains the old order.
            order = np.arange(len(self.docs), dtype='<i8')
            random.Random(f'{self.seed}:{epoch}').shuffle(order)
            fd, temporary = tempfile.mkstemp(prefix='order-', dir=cache)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    np.save(stream, order, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                checksum = _sha(temporary)
                os.replace(temporary, path)
                temporary = None
                fd, temporary = tempfile.mkstemp(prefix='order-meta-', dir=cache)
                with os.fdopen(fd, 'w') as stream:
                    json.dump({'identity': key, 'sha256': checksum}, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, meta)
                temporary = None
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
            return np.load(path, mmap_mode='r', allow_pickle=False)

    def _epoch_rows(self, order, epoch):
        """Cache the deterministic context-dependent row count, not packed rows."""
        if not hasattr(self.docs, "cache_dir"):
            rows, fill = self._count_lengths(self._lengths(order))
            return rows + bool(fill)
        import fcntl
        import os
        import tempfile
        key = {'version': 2, 'manifest': self.manifest_hash, 'split': self.split,
               'seed': str(self.seed), 'epoch': epoch, 'context': self.context}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        cache = self.docs.cache_dir / 'packing'
        cache.mkdir(exist_ok=True)
        path = cache / f'rows-{digest}.json'
        with (cache / f'rows-{digest}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                saved = json.loads(path.read_text())
                payload = saved['payload']
                checksum = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                if saved['sha256'] != checksum or payload['identity'] != key:
                    raise ValueError('invalid row count cache')
                rows = payload['rows']
                if type(rows) is not int or not 0 < rows <= self.manifest['tokens'][self.split]:
                    raise ValueError('invalid row count')
                checkpoints = payload['checkpoints']
                expected_d = [*range(0, len(order), 4096), len(order)]
                if (not isinstance(checkpoints, list) or len(checkpoints) != len(expected_d)
                        or any(not isinstance(row, list) or len(row) != 3
                               or any(type(v) is not int for v in row)
                               or row[0] != d or row[1] < 0 or not 0 <= row[2] <= self.context
                               for d, row in zip(expected_d, checkpoints))
                        or checkpoints[0] != [0, 0, 0]
                        or any(left[1] > right[1] for left, right in zip(checkpoints, checkpoints[1:]))
                        or checkpoints[-1][1] + bool(checkpoints[-1][2]) != rows):
                    raise ValueError('invalid sparse row checkpoints')
                self._packing_prefix = (epoch, self.context, checkpoints)
                return rows
            except (OSError, ValueError, KeyError, TypeError):
                pass
            # Random traversal of the structured mmap would fault all 40 bytes
            # per document into each rank's RSS. Read bounded chunks through the
            # file API and keep only the 8-byte lengths for this first scan.
            lengths = np.empty(len(self.docs), dtype=np.int64)
            with (self.docs.cache_dir / f'{self.split}.index.bin').open('rb') as stream:
                for start in range(0, len(self.docs), 65536):
                    count = min(65536, len(self.docs) - start)
                    chunk = np.fromfile(stream, dtype=self.docs.rows.dtype, count=count)
                    if len(chunk) != count:
                        raise ValueError('truncated document sampler index during row count')
                    np.subtract(chunk['token_end'], chunk['token_start'],
                                out=lengths[start:start + count])
            rows = fill = 0
            checkpoints = [[0, 0, 0]]
            for start in range(0, len(order), 4096):
                end = min(start + 4096, len(order))
                rows, fill = self._count_lengths(
                    (int(lengths[index]) for index in order[start:end]), rows=rows, fill=fill)
                checkpoints.append([end, rows, fill])
            del lengths
            rows += bool(fill)
            payload = {'identity': key, 'rows': rows, 'checkpoints': checkpoints}
            checksum = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            fd, temporary = tempfile.mkstemp(prefix='rows-', dir=cache)
            try:
                with os.fdopen(fd, 'w') as stream:
                    json.dump({'payload': payload, 'sha256': checksum}, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            self._packing_prefix = (epoch, self.context, checkpoints)
            return rows

    def _length(self, index):
        if not hasattr(self.docs, 'starts'):
            doc = self.docs[index]
            return doc['token_end'] - doc['token_start']
        return self.docs.length(index)

    def _start(self, index):
        return self.docs.start(index) if hasattr(self.docs, 'starts') else self.docs[index]['token_start']

    def _lengths(self, order, stop=None):
        for index in order if stop is None else order[:stop]:
            yield self._length(index)

    def _count_lengths(self, lengths, *, rows=0, fill=0):
        # O(documents), including context=1 and very long individual documents.
        width = self.context + 1
        for remaining in lengths:
            capacity = width - fill
            if remaining < capacity:
                fill += remaining
                continue
            rows += 1
            remaining -= capacity
            fill = 0
            if remaining:
                remaining += 1
                complete = (remaining - 1) // self.context
                rows += complete
                fill = remaining - complete * self.context
                if complete and fill == 1:
                    fill = 0
        return rows, fill

    def _order(self):
        # A one-row replay mask belongs only to its source epoch. A terminal
        # singleton can place the fork at epoch end before the mask is consumed.
        self._context_transition = None
        self.order = self._permutation(self.epoch)
        self._order_epoch = self.epoch
        self.epoch_row_count = self._epoch_rows(self.order, self.epoch)
        self.document_cursor = self.token_cursor = 0

    def _prefixes(self, order, epoch):
        if not hasattr(self.docs, 'cache_dir'):
            return [[0, 0, 0]]
        cached = getattr(self, '_packing_prefix', None)
        if cached is None or cached[:2] != (epoch, self.context):
            self._epoch_rows(order, epoch)
        return self._packing_prefix[2]

    def _prefix_count(self, order, epoch, document_cursor):
        checkpoints = self._prefixes(order, epoch)
        point = min(document_cursor // 4096, len(checkpoints) - 1)
        d, rows, fill = checkpoints[point]
        return self._count_lengths((self._length(i) for i in order[d:document_cursor]),
                                   rows=rows, fill=fill)

    def _seek_row(self, order, cursor, epoch=None):
        """Seek from a sparse document prefix; no token/row materialization."""
        from bisect import bisect_left
        epoch = self.epoch if epoch is None else epoch
        checkpoints = self._prefixes(order, epoch)
        point = max(0, bisect_left(checkpoints, cursor, key=lambda item: item[1]) - 1)
        d, rows, fill = checkpoints[point]
        t = 0
        remaining_rows = cursor - rows
        width = self.context + 1
        while remaining_rows:
            if d == len(order):
                if fill and remaining_rows == 1:
                    return d, 0
                raise ValueError('invalid canonical row seek')
            length = self._length(order[d])
            remaining = length - t
            if fill == 0 and remaining >= width:
                count = min(remaining_rows, (remaining - 1) // self.context)
                t += count * self.context
                remaining_rows -= count
                if t == length - 1:
                    d, t = d + 1, 0
                continue
            capacity = width - fill
            if remaining < capacity:
                fill += remaining
                d, t = d + 1, 0
            else:
                if remaining == capacity:
                    d, t = d + 1, 0
                else:
                    t += capacity - 1
                remaining_rows -= 1
                fill = 0
        return d, t

    def _fork_position(self, source):
        epoch, d, t = source['epoch'], source['document_cursor'], source['token_cursor']
        order = self._permutation(epoch)
        if d == len(order):
            epoch, d, t = epoch + 1, 0, 0
            order = self._permutation(epoch)
        # Number of complete canonical new-context rows up to the scored token.
        rows, fill = self._prefix_count(order, epoch, d)
        cursor, _ = self._count_lengths((t + 1,), rows=rows, fill=fill)
        document_cursor, token_cursor = self._seek_row(order, cursor, epoch)
        return epoch, cursor, document_cursor, token_cursor, d, t

    def fork_context(self, context):
        """Return a new packing stream, preserving consumed-stream target watermark.

        Earlier canonical inputs may replay as prefix but their targets are masked
        once. This certifies consumed sampler rows, not optimizer quota selection:
        callers may previously have discarded overflow targets. Read-only corpus
        storage is shared; original cursor and last-batch mappings stay unchanged.
        """
        import copy
        if type(context) is not int or context <= 0:
            raise ValueError('context must be a positive integer')
        original = PackedSampler.state_dict(self)
        check = copy.copy(self)
        PackedSampler.load_state_dict(check, original)
        if context == self.context:
            return check
        transition = original.get('context_transition')
        source = copy.deepcopy(transition['source_state'] if transition else original)
        fork = copy.copy(self)
        fork.context = context
        epoch, cursor, d, t, _, _ = fork._fork_position(source)
        target = {**source, 'context': context, 'epoch': epoch, 'cursor': cursor,
                  'document_cursor': d, 'token_cursor': t,
                  'context_transition': {'version': 1, 'source_context': source['context'], 'source_state': source}}
        PackedSampler.load_state_dict(fork, target)
        return fork

    def _next_row(self):
        """Build one row only, retaining no prior/future packed rows."""
        row, fill = [], 0
        width = self.context + 1
        while self.document_cursor < len(self.order) and fill < width:
            i = int(self.order[self.document_cursor])
            length = self._length(i)
            count = min(width - fill, length - self.token_cursor)
            row.append((i, self._start(i) + self.token_cursor, count))
            fill += count
            self.token_cursor += count
            if self.token_cursor == length:
                self.document_cursor += 1
                self.token_cursor = 0
            elif fill == width:
                self.token_cursor -= 1
        return row

    def next_batch(self):
        idx = torch.full((self.batch_size, self.context), 50256, dtype=torch.long)
        ids = torch.full_like(idx, -1)
        targets = torch.full_like(idx, -100)
        self._batch_mappings = []
        for b in range(self.batch_size):
            if self.cursor >= self.epoch_row_count:
                self.epoch += 1
                self.cursor = 0
                self._order()
            transition = self._context_transition
            row_start_document = self.document_cursor
            row = self._next_row()
            self._context_transition = None
            self.cursor += 1
            tok, docids, mappings = [], [], []
            for i, start, count in row:
                doc = self.docs[i]
                packed_start = len(tok)
                mappings.append({'document_id': doc['id'], 'numeric_id': doc['numeric_id'],
                                 'content_sha256': doc['content_sha256'],
                                 'processed_sha256': doc['processed_sha256'],
                                 'source_offset': doc['source_offset'],
                                 'synthetic_eos_token_offset': doc['synthetic_eos_token_offset'],
                                 'processed_document_token_count': doc['processed_token_end'],
                                 'packed_token_start': packed_start,
                                 'packed_token_end': packed_start + count,
                                 'source_token_start': start - doc['token_start'],
                                 'source_token_end': start - doc['token_start'] + count})
                tok.extend(self.tokens[start:start + count].tolist())
                docids.extend([self.docs.numeric_id(i)] * count)
            self._batch_mappings.append({'epoch': self.epoch, 'row_cursor': self.cursor - 1,
                                         'input_length': min(self.context, len(tok)),
                                         'ranges': mappings})
            n = min(self.context, len(tok))
            token_row = torch.tensor(tok, dtype=torch.long)
            document_row = torch.tensor(docids, dtype=torch.long)
            idx[b, :n] = token_row[:n]
            ids[b, :n] = document_row[:n]
            pairs = min(self.context, len(tok) - 1)
            if pairs > 0:
                same_document = document_row[:pairs] == document_row[1:pairs + 1]
                targets[b, :pairs] = torch.where(same_document, token_row[1:pairs + 1], -100)
            if transition is not None:
                source = transition['source_state']
                _, _, _, _, frontier_doc, frontier_token = self._fork_position(source)
                exposure = {'prefix_tokens_at_or_before_watermark': 0, 'masked_prediction_targets': 0}
                for row_document, mapping in enumerate(mappings):
                    position = row_start_document + row_document
                    for packed in range(mapping['packed_token_start'], min(n, mapping['packed_token_end'])):
                        offset = mapping['source_token_start'] + packed - mapping['packed_token_start']
                        if position < frontier_doc or (position == frontier_doc and offset <= frontier_token):
                            exposure['prefix_tokens_at_or_before_watermark'] += 1
                    for packed in range(mapping['packed_token_start'], min(self.context, mapping['packed_token_end'] - 1)):
                        target_offset = mapping['source_token_start'] + packed - mapping['packed_token_start'] + 1
                        if position < frontier_doc or (position == frontier_doc and target_offset <= frontier_token):
                            exposure['masked_prediction_targets'] += int(targets[b, packed] != -100)
                            targets[b, packed] = -100
                self._batch_mappings[-1]['context_fork_exposure'] = exposure
            # Unique negative padding IDs prevent padding from sharing document scope.
            if n < self.context:
                ids[b, n:] = -torch.arange(1, self.context - n + 1)
        return idx, ids, targets

    def batch_mappings(self):
        """Copy last batch lineage; ranges include the target-lookahead slot T.

        Each range maps packed positions [start,end) to source document token
        offsets. Positions >= input_length are lookahead or padding, not inputs.
        Calling this method never advances the sampler.
        """
        import copy
        return copy.deepcopy(self._batch_mappings)

    def batch_context_exposure(self):
        """Local last-batch context-transition counts, excluding padding/lookahead."""
        keys = ('prefix_tokens_at_or_before_watermark', 'masked_prediction_targets')
        return {key: sum(row.get('context_fork_exposure', {}).get(key, 0)
                         for row in self._batch_mappings) for key in keys}

    __next__ = next_batch

    def __iter__(self):
        return self

    def state_dict(self):
        import copy
        state = {k: getattr(self, k) for k in ('packing_version', 'manifest_hash', 'split', 'batch_size', 'context', 'seed', 'epoch', 'cursor', 'document_cursor', 'token_cursor')}
        if self._context_transition is not None:
            state['context_transition'] = copy.deepcopy(self._context_transition)
        return state

    def load_state_dict(self, state):
        for k in ('packing_version', 'manifest_hash', 'split', 'batch_size', 'context', 'seed'):
            if state.get(k) != getattr(self, k):
                raise ValueError(f'sampler state mismatch: {k}')
        epoch, cursor = state['epoch'], state['cursor']
        document_cursor, token_cursor = state['document_cursor'], state['token_cursor']
        if any(type(value) is not int or value < 0 for value in (epoch, cursor, document_cursor, token_cursor)):
            raise ValueError('invalid sampler cursor')
        # Build/validate in locals: a rejected restore must leave the live stream
        # unchanged, including cached last-batch mappings.
        order = self._permutation(epoch)
        epoch_row_count = self._epoch_rows(order, epoch)
        if cursor > epoch_row_count or document_cursor > len(order):
            raise ValueError('invalid sampler cursor')
        if document_cursor == len(order):
            if token_cursor != 0 or cursor != epoch_row_count:
                raise ValueError('inconsistent completed epoch cursor')
        else:
            length = self._length(order[document_cursor])
            if token_cursor >= length or cursor == epoch_row_count:
                raise ValueError('invalid document token cursor')
            previous_rows, previous_fill = self._prefix_count(order, epoch, document_cursor)
            at_document_start = token_cursor == 0 and previous_fill == 0 and previous_rows == cursor
            if not at_document_start:
                previous_rows, previous_fill = self._count_lengths(
                    (token_cursor + 1,), rows=previous_rows, fill=previous_fill)
                if token_cursor >= length - 1 or previous_fill != 0 or previous_rows != cursor:
                    raise ValueError('inconsistent row/document/token cursors')
        import copy
        transition = copy.deepcopy(state.get('context_transition'))
        if transition is not None:
            if not isinstance(transition, dict) or set(transition) != {'version', 'source_context', 'source_state'} or type(transition['version']) is not int or transition['version'] != 1:
                raise ValueError('invalid context transition schema')
            source = transition['source_state']
            source_context = transition['source_context']
            if (type(source_context) is not int or source_context <= 0 or not isinstance(source, dict)
                    or type(source.get('context')) is not int or source.get('context') != source_context
                    or set(source) != {'packing_version', 'manifest_hash', 'split', 'batch_size', 'context', 'seed', 'epoch', 'cursor', 'document_cursor', 'token_cursor'}):
                raise ValueError('invalid context transition source')
            checker = copy.copy(self)
            checker.context = source_context
            checker._context_transition = None
            PackedSampler.load_state_dict(checker, source)
            expected = self._fork_position(source)[:4]
            if (epoch, cursor, document_cursor, token_cursor) != expected:
                raise ValueError('context transition cursor mismatch')
        self.epoch, self.order, self.epoch_row_count = epoch, order, epoch_row_count
        self._order_epoch = epoch
        self.cursor = cursor
        self.document_cursor, self.token_cursor = document_cursor, token_cursor
        self._context_transition = transition
        self._batch_mappings = []



class RankPackedSampler(PackedSampler):
    """Rank slices of one deterministic global packed-row stream.

    A call consumes ``world_size * batch_size`` rows globally, returning this
    rank's contiguous slice. Every rank must call equally often. Only local rows
    materialize token tensors; other rows advance packing metadata. Epochs may
    repeat within a global batch for tiny corpora, exactly as in PackedSampler.
    This partitions stream positions, not unique document identities.
    """
    def __init__(self, manifest, *, rank, world_size, split='train', batch_size=1,
                 context=512, seed=0):
        if (type(world_size) is not int or world_size < 1 or type(rank) is not int
                or not 0 <= rank < world_size):
            raise ValueError('invalid sampler rank/world_size')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.rank, self.world_size = rank, world_size
        super().__init__(manifest, split=split, batch_size=batch_size, context=context, seed=seed)

    def _skip_rows(self, count):
        for _ in range(count):
            if self.cursor >= self.epoch_row_count:
                self.epoch += 1
                self.cursor = 0
                self._order()
            self._next_row()
            self._context_transition = None
            self.cursor += 1

    def next_batch(self):
        self._skip_rows(self.rank * self.batch_size)
        batch = super().next_batch()
        self._skip_rows((self.world_size - self.rank - 1) * self.batch_size)
        return batch

    __next__ = next_batch

    def state_dict(self):
        return {'rank_packing_version': 1, 'rank': self.rank, 'world_size': self.world_size,
                'stream': super().state_dict()}

    def load_state_dict(self, state):
        for key, expected in (('rank_packing_version', 1), ('rank', self.rank),
                              ('world_size', self.world_size)):
            if type(state.get(key)) is not int or state[key] != expected:
                raise ValueError(f'rank sampler state mismatch: {key}')
        super().load_state_dict(state['stream'])


def batch_hash(batch):
    h = hashlib.sha256()
    values = batch.values() if isinstance(batch, dict) else batch
    for value in values:
        value = value.detach().cpu().contiguous()
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def audit_validation_overlap(training_manifest, validation_manifest, document_ids=None):
    """Audit exact full-document hash equality using metadata only.

    Compare selected validation documents against train/calibration/final-test
    metadata, independent of ID namespaces. Original and processed hashes are
    compared separately. This does not detect partial/near duplicates or establish
    general independence. Token payloads, targets, and raw text are never read.
    """
    hash_fields = ('content_sha256', 'processed_sha256')
    compared_splits = ('train', 'calibration', 'test')
    sample_limit = 10

    def checked_metadata(path):
        path = Path(path)
        if path.is_dir():
            path /= 'manifest.json'
        digest = manifest_hash(path)
        manifest = json.loads(path.read_text())
        expected = manifest.get('artifacts', {}).get('documents.jsonl')
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('missing or invalid documents.jsonl checksum')
        metadata_path = path.with_name('documents.jsonl')
        if _sha(metadata_path) != expected:
            raise ValueError('documents.jsonl artifact checksum mismatch')
        return digest, metadata_path

    def documents(path):
        seen = set()
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    doc = json.loads(line)
                    doc_id = doc['id']
                    if not isinstance(doc_id, str) or doc_id in seen:
                        raise ValueError('non-string or duplicate document ID')
                    if doc['split'] not in SPLIT_ORDER:
                        raise ValueError('unknown document split')
                    for field in hash_fields:
                        value = doc[field]
                        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value.lower()):
                            raise ValueError(f'invalid {field}')
                        doc[field] = value.lower()
                    seen.add(doc_id)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f'invalid document metadata at line {line_number}: {error}') from error
                yield doc

    training_hash, training_path = checked_metadata(training_manifest)
    validation_hash, validation_path = checked_metadata(validation_manifest)
    requested = None if document_ids is None else ({document_ids} if isinstance(document_ids, str) else {str(value) for value in document_ids})
    index = {split: {field: {} for field in hash_fields} for split in compared_splits}
    for doc in documents(training_path):
        if doc['split'] in index:
            for field in hash_fields:
                index[doc['split']][field].setdefault(doc[field], doc['id'])
    counts = dict.fromkeys(compared_splits, 0)
    samples = {split: [] for split in compared_splits}
    selected = set()
    first_validation_id = {field: {} for field in hash_fields}
    duplicate_hashes = {field: set() for field in hash_fields}
    duplicate_ids = set()
    duplicate_samples = []
    for doc in documents(validation_path):
        doc_id = doc['id']
        if doc['split'] != 'val' or (requested is not None and doc_id not in requested):
            continue
        selected.add(doc_id)
        for split in compared_splits:
            matched = False
            for field in hash_fields:
                other_id = index[split][field].get(doc[field])
                if other_id is not None:
                    matched = True
                    if len(samples[split]) < sample_limit:
                        samples[split].append({'validation_document_id': doc_id,
                                               'other_document_id': other_id,
                                               'hash_kind': field, 'sha256': doc[field]})
            counts[split] += int(matched)
        for field in hash_fields:
            previous = first_validation_id[field].setdefault(doc[field], doc_id)
            if previous != doc_id:
                duplicate_ids.update((previous, doc_id))
                duplicate_hashes[field].add(doc[field])
                if len(duplicate_samples) < sample_limit:
                    duplicate_samples.append({'validation_document_id': doc_id,
                                              'other_document_id': previous,
                                              'hash_kind': field, 'sha256': doc[field]})
    if requested is not None and requested - selected:
        missing = sorted(requested - selected)
        raise ValueError(f'unknown requested validation document IDs: {missing[:sample_limit]} (total {len(missing)})')
    return {'training_manifest_hash': training_hash, 'validation_manifest_hash': validation_hash,
            'selected_validation_documents': len(selected),
            'selected_validation_ids_sha256': hashlib.sha256(json.dumps(sorted(selected), separators=(',', ':'), ensure_ascii=True).encode()).hexdigest(),
            'overlap_counts': counts,
            'overlap_samples': samples,
            'duplicate_validation': {'documents_with_duplicate_content': len(duplicate_ids),
                                     'hash_group_counts': {field: len(values) for field, values in duplicate_hashes.items()},
                                     'samples': duplicate_samples},
            'has_overlap': bool(any(counts.values()) or duplicate_ids),
            'scope': 'exact whole-document original/processed SHA256 equality only; no near-duplicate or general-independence claim',
            'sample_limit': sample_limit}


# Backwards-compatible synthetic-task API; implementation lives in tasks.py.
from .tasks import (MECHANISM_VERSION, _MECHANISM_KEYS, _MECHANISM_TASKS,
                    _mechanism_bounds, _mechanism_partitions, _mechanism_row,
                    load_mechanism_tasks, prepare_mechanism_tasks)
