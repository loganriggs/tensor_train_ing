"""Verified, optional mmap document index; immutable dataset manifests stay unchanged."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import tempfile

import numpy as np

from .token_format import TOKEN_DTYPE

SPLITS = ('train', 'val', 'test', 'calibration')
INDEX_NAME = '.sampler-index-v1'
INDEX_DTYPE = np.dtype([(field, '<i8') for field in
    ('token_start', 'token_end', 'numeric_id', 'metadata_offset', 'metadata_length')])
_ROW = struct.Struct('<qqqqq')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _identity(root, manifest):
    expected = hashlib.sha256((json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()).hexdigest()
    path = root / 'manifest.json'
    if path.exists():
        actual = _sha(path)
        if actual != (root/'manifest.sha256').read_text().strip():
            raise ValueError('manifest checksum mismatch')
        if json.loads(path.read_text()) != manifest:
            raise ValueError('sampler index manifest identity mismatch')
        expected = actual
    documents_sha = manifest.get('artifacts', {}).get('documents.jsonl')
    if not isinstance(documents_sha, str) or _sha(root/'documents.jsonl') != documents_sha:
        raise ValueError('document metadata checksum mismatch')
    return {'manifest_sha256': expected, 'documents_sha256': documents_sha}


def _validate_manifest(root, manifest):
    if manifest.get('format_version') != 1:
        raise ValueError('unsupported dataset format')
    counts = manifest.get('tokens', {})
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError('invalid dataset token counts')
    if type(manifest.get('documents')) is not int or manifest['documents'] < 0:
        raise ValueError('invalid dataset document count')
    required = {'documents.jsonl', *(f'{split}.tokens.bin' for split in SPLITS)}
    if not required.issubset(manifest.get('artifacts', {})):
        raise ValueError('dataset manifest lacks required metadata/token artifacts')
    for split, count in counts.items():
        if (root/f'{split}.tokens.bin').stat().st_size != count * np.dtype(TOKEN_DTYPE).itemsize:
            raise ValueError(f'token payload size mismatch: {split}')
    return counts


def _build(root, manifest, identity, target):
    counts = _validate_manifest(root, manifest)
    covered = dict.fromkeys(SPLITS, 0)
    sizes = dict.fromkeys(SPLITS, 0)
    db = sqlite3.connect(target/'seen.sqlite')
    db.execute('PRAGMA cache_size=-16384')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE seen (id TEXT PRIMARY KEY, numeric_id INTEGER UNIQUE NOT NULL)')
    streams = {split: (target/f'{split}.index.bin').open('wb') for split in SPLITS}
    document_count = offset = 0
    try:
        with (root/'documents.jsonl').open('rb') as source:
            for line_number, line in enumerate(source, 1):
                try:
                    doc = json.loads(line)
                    name = doc['split']
                    doc_id, numeric_id = doc['id'], doc['numeric_id']
                    start, end = doc['token_start'], doc['token_end']
                    if name not in covered or not isinstance(doc_id, str):
                        raise ValueError('invalid split or document ID')
                    if type(numeric_id) is not int or not 0 <= numeric_id < 2**63:
                        raise ValueError('invalid numeric document ID')
                    if (type(start) is not int or type(end) is not int or start != covered[name]
                            or not start < end <= counts[name] or end >= 2**63):
                        raise ValueError('document token ranges must cover each split contiguously')
                    for field in ('content_sha256', 'processed_sha256'):
                        digest = doc[field]
                        if (not isinstance(digest, str) or len(digest) != 64
                                or any(c not in '0123456789abcdef' for c in digest.lower())):
                            raise ValueError(f'invalid {field}')
                    try:
                        db.execute('INSERT INTO seen VALUES (?, ?)', (doc_id, numeric_id))
                    except sqlite3.IntegrityError as error:
                        raise ValueError('duplicate document ID or numeric document ID') from error
                    streams[name].write(_ROW.pack(start, end, numeric_id, offset, len(line)))
                    sizes[name] += 1
                    covered[name] = end
                    document_count += 1
                    if document_count % 8192 == 0:
                        db.commit()
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f'invalid document metadata at line {line_number}: {error}') from error
                offset += len(line)
        if covered != counts or document_count != manifest['documents']:
            raise ValueError('document metadata does not cover declared dataset')
        if _identity(root, manifest) != identity:
            raise ValueError('sampler index source identity changed during build')
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        for stream in streams.values():
            stream.close()
        db.close()
    (target/'seen.sqlite').unlink()
    (target/'seen.sqlite-journal').unlink(missing_ok=True)
    header = {'index_version': 1, 'identity': identity, 'row_bytes': INDEX_DTYPE.itemsize,
              'fields': list(INDEX_DTYPE.names), 'splits': {split: {
                  'count': sizes[split], 'bytes': (target/f'{split}.index.bin').stat().st_size,
                  'sha256': _sha(target/f'{split}.index.bin')} for split in SPLITS}}
    with (target/'index.json').open('w') as stream:
        json.dump(header, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _verify(root, manifest, identity, cache):
    try:
        header = json.loads((cache/'index.json').read_text())
        if (header['index_version'] != 1 or header['identity'] != identity
                or header['row_bytes'] != INDEX_DTYPE.itemsize
                or header['fields'] != list(INDEX_DTYPE.names) or set(header['splits']) != set(SPLITS)):
            raise ValueError('stale sampler index identity or format')
        metadata_bytes = (root/'documents.jsonl').stat().st_size
        total = 0
        for split in SPLITS:
            info = header['splits'][split]
            n = info['count']
            path = cache/f'{split}.index.bin'
            if (type(n) is not int or n < 0 or info['bytes'] != n * INDEX_DTYPE.itemsize
                    or path.stat().st_size != info['bytes'] or _sha(path) != info['sha256']):
                raise ValueError('sampler index payload checksum or size mismatch')
            total += n
            last = 0
            if n:
                with path.open('rb') as payload:
                    for start in range(0, n, 65536):
                        part = np.frombuffer(payload.read(min(65536, n-start) * INDEX_DTYPE.itemsize), dtype=INDEX_DTYPE)
                        if (int(part['token_start'][0]) != last
                                or np.any(part['token_end'] <= part['token_start'])
                                or np.any(part['token_start'][1:] != part['token_end'][:-1])
                                or np.any(part['numeric_id'] < 0)
                                or np.any(part['metadata_offset'] < 0)
                                or np.any(part['metadata_length'] <= 0)
                                or np.any(part['metadata_offset'] > metadata_bytes - part['metadata_length'])):
                            raise ValueError('invalid sampler index row structure')
                        last = int(part['token_end'][-1])
            if last != manifest['tokens'][split]:
                raise ValueError('sampler index token coverage mismatch')
        if total != manifest['documents']:
            raise ValueError('sampler index document count mismatch')
        return header
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f'invalid document sampler index; explicitly rebuild to replace: {error}') from error


def ensure_sampler_index(root, manifest, *, rebuild=False):
    """Build a missing cache once; reject corruption unless explicitly rebuilding.

    Also supports preparation before the manifest's atomic publication: its
    canonical bytes determine the identity when manifest.json is not present.
    """
    root = Path(root)
    _validate_manifest(root, manifest)
    identity = _identity(root, manifest)
    cache = root/INDEX_NAME
    with (root/(INDEX_NAME+'.lock')).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists() and not rebuild:
            _verify(root, manifest, identity, cache)
            return cache
        temporary = Path(tempfile.mkdtemp(prefix=INDEX_NAME+'.building-', dir=root))
        backup = None
        try:
            _build(root, manifest, identity, temporary)
            _verify(root, manifest, identity, temporary)
            if cache.exists():
                backup = Path(tempfile.mkdtemp(prefix=INDEX_NAME+'.old-', dir=root))
                backup.rmdir()
                os.rename(cache, backup)
            try:
                os.rename(temporary, cache)
            except BaseException:
                if backup is not None:
                    os.rename(backup, cache)
                    backup = None
                raise
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
            if backup is not None:
                shutil.rmtree(backup)
        return cache


class _IndexedDocument(Mapping):
    def __init__(self, owner, index):
        self.owner, self.index, self._full = owner, index, None

    def _document(self):
        if self._full is None:
            self._full = self.owner.full_document(self.index)
        return self._full

    def __getitem__(self, key):
        if key in ('token_start', 'token_end', 'numeric_id'):
            self.owner._touch()
            return int(self.owner.rows[key][self.index])
        if key == 'split':
            return self.owner.split
        return self._document()[key]

    def __iter__(self):
        return iter(self._document())

    def __len__(self):
        return len(self._document())


class IndexedDocuments(Sequence):
    """Lazy document rows with bounded process-local mmap residency on Linux.

    Every 1,024 indexed accesses releases this mapping's resident pages with
    MADV_DONTNEED. Subsequent reads fault pages back from the shared filesystem
    cache; this does not modify the file or evict another process's mapping.
    This trades additional page faults for a bounded access window between
    releases; RSS also depends on OS page size and fault-around behavior, so
    this is not an exact byte limit. Direct access to public array
    views bypasses accounting and should be followed by release_pages(). On
    platforms without madvise correctness is unchanged, but residency is not
    bounded by this mechanism.
    """
    PAGE_RELEASE_INTERVAL = 1024

    def __init__(self, root, cache, split, identity):
        self.root, self.cache_dir, self.split, self.identity = Path(root), cache, split, identity
        path = cache/f'{split}.index.bin'
        self.rows = (np.memmap(path, dtype=INDEX_DTYPE, mode='r') if path.stat().st_size
                     else np.empty(0, dtype=INDEX_DTYPE))
        self._index_touches = 0
        self._metadata_file = (self.root/'documents.jsonl').open('rb')
        self.starts, self.ends, self.numeric_ids = (self.rows[name] for name in ('token_start', 'token_end', 'numeric_id'))

    def release_pages(self):
        self._index_touches = 0
        mapping = getattr(self.rows, '_mmap', None)
        advice = getattr(mmap, 'MADV_DONTNEED', None)
        if mapping is None or advice is None or not hasattr(mapping, 'madvise'):
            return False
        try:
            mapping.madvise(advice)
        except (OSError, ValueError):
            return False
        return True

    def _touch(self):
        self._index_touches += 1
        if self._index_touches >= self.PAGE_RELEASE_INTERVAL:
            self.release_pages()

    def length(self, index):
        self._touch()
        return int(self.ends[index]) - int(self.starts[index])

    def start(self, index):
        self._touch()
        return int(self.starts[index])

    def numeric_id(self, index):
        self._touch()
        return int(self.numeric_ids[index])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._touch()
        return _IndexedDocument(self, index)

    def full_document(self, index):
        self._touch()
        row = self.rows[index]
        raw = os.pread(self._metadata_file.fileno(), int(row['metadata_length']),
                       int(row['metadata_offset']))
        return json.loads(raw)


def load_sampler_index(root, manifest, split, *, rebuild=False):
    if split not in SPLITS:
        raise ValueError(f'invalid split: {split}')
    cache = ensure_sampler_index(root, manifest, rebuild=rebuild)
    header = json.loads((cache/'index.json').read_text())
    return IndexedDocuments(root, cache, split, header['identity'])
