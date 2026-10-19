"""Disposable compressed source staging retains canonical dataset bytes."""
import hashlib
import json
import random
import sqlite3
import zlib

from tnlm.data import prepare_data
from tnlm.data_preparation import consume_candidates


def test_ranked_consumption_reclaims_database_and_preserves_unicode(tmp_path):
    path = tmp_path/'staging.sqlite'
    db = sqlite3.connect(path)
    db.execute('PRAGMA auto_vacuum=INCREMENTAL')
    db.execute('CREATE TABLE candidates (id TEXT PRIMARY KEY, text BLOB, source_offset TEXT, content_hash TEXT, rank TEXT)')
    db.execute('CREATE INDEX candidate_rank ON candidates(rank ASC,id ASC)')
    randomizer = random.Random(71)
    expected = []
    for i in range(128):
        text = 'é😀\r\n' + randomizer.randbytes(32768).hex()
        rank = hashlib.sha256(str(i).encode()).hexdigest()
        row = (str(i), text, str(i), hashlib.sha256(text.encode()).hexdigest(), rank)
        expected.append(row)
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?)',
                   (row[0], zlib.compress(text.encode(), 1), *row[2:]))
    db.commit()
    before = path.stat().st_size
    consumed = consume_candidates(db, batch_bytes=65536, reclaim_bytes=262144)
    actual = list(consumed)
    after = path.stat().st_size
    assert actual == sorted(expected, key=lambda row: (row[4], row[0]))
    assert db.execute('SELECT count(*) FROM candidates').fetchone()[0] == 0
    assert after < before / 10
    db.close()


def test_prepare_staging_is_compressed_and_duplicates_still_scanned(tmp_path, monkeypatch):
    import tnlm.data_preparation as staging
    source = tmp_path/'source.jsonl'
    text = 'Large UTF8 document. é😀\r\n' * 1000
    source.write_text(json.dumps({'id': 'same', 'text': text})+'\n'+json.dumps({'id': 'same', 'text': text})+'\n')
    original = staging.ordered_tokenization
    observations = []
    def inspect(rows, **kwargs):
        db = sqlite3.connect(tmp_path/'out/.seen.sqlite')
        stored = db.execute('SELECT text FROM candidates').fetchone()[0]
        assert isinstance(stored, bytes) and zlib.decompress(stored).decode() == text
        assert len(stored) < len(text.encode()) / 10
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='seen'").fetchall()
        db.close()
        observations.append(True)
        return original(rows, **kwargs)
    monkeypatch.setattr(staging, 'ordered_tokenization', inspect)
    class Tokenizer:
        name, eot_token = 'staging-test', 50256
        def encode(self, text, **kwargs):
            return [123, 456]
    manifest = prepare_data(tmp_path/'out', source_jsonl=source, revision='fixture', tokenizer=Tokenizer())
    info = json.loads(manifest.read_text())
    assert observations == [True]
    assert info['scan_records'] == 2 and info['scan_unique_documents'] == 1
    assert info['documents'] == 1
    assert not (tmp_path/'out/.seen.sqlite').exists()


def _hf_fixture(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    shard = tmp_path/'source.parquet'
    pq.write_table(pa.table({'id': ['one', 'two', 'three'], 'text': ['A', 'B', 'C']}), shard)
    class Tokenizer:
        name, eot_token = 'shard-callback-test', 50256
        def encode(self, text, **kwargs):
            return [ord(text)]
    return shard, dict(hf_dataset='fixture/fineweb', revision='a'*40,
        hf_data_files=['data/source.parquet'], hf_local_shards={'data/source.parquet': str(shard)},
        hf_shard_checksums={'data/source.parquet': hashlib.sha256(shard.read_bytes()).hexdigest()},
        tokenizer=Tokenizer())


def test_shard_callback_after_complete_scan_and_close_preserves_bytes(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    shard, kwargs = _hf_fixture(tmp_path)
    original = pq.ParquetFile
    readers = []
    class Reader:
        def __init__(self, path):
            self.reader = original(path)
            self.schema_arrow = self.reader.schema_arrow
            self.closed, self.rows = False, 0
            readers.append(self)
        def iter_batches(self, **kw):
            for batch in self.reader.iter_batches(**kw):
                self.rows += batch.num_rows
                yield batch
        def close(self):
            self.reader.close()
            self.closed = True
    monkeypatch.setattr(pq, 'ParquetFile', Reader)
    calls = []
    def scanned(name, path):
        assert readers[-1].closed and readers[-1].rows == 3
        assert path == shard and path.exists()
        calls.append(name)
    left = prepare_data(tmp_path/'with-hook', **kwargs, on_shard_scanned=scanned)
    right = prepare_data(tmp_path/'without-hook', **kwargs)
    assert calls == ['data/source.parquet']
    assert left.read_bytes() == right.read_bytes()
    info = json.loads(left.read_text())
    for artifact in info['artifacts']:
        assert (left.parent/artifact).read_bytes() == (right.parent/artifact).read_bytes()


def test_shard_callback_not_called_when_second_hash_fails(tmp_path, monkeypatch):
    import pytest
    import tnlm.data as data
    shard, kwargs = _hf_fixture(tmp_path)
    original, count = data._sha, 0
    def changed(path):
        nonlocal count
        if path == shard:
            count += 1
            if count == 2:
                return '0'*64
        return original(path)
    monkeypatch.setattr(data, '_sha', changed)
    calls = []
    with pytest.raises(ValueError, match='changed during reading'):
        prepare_data(tmp_path/'bad', **kwargs, on_shard_scanned=lambda *a: calls.append(a))
    assert calls == [] and not (tmp_path/'bad').exists()


def test_shard_callback_failure_cleans_output(tmp_path):
    import pytest
    shard, kwargs = _hf_fixture(tmp_path)
    def failed(*args):
        raise RuntimeError('callback failure')
    with pytest.raises(RuntimeError, match='callback failure'):
        prepare_data(tmp_path/'failed-hook', **kwargs, on_shard_scanned=failed)
    assert shard.exists() and not (tmp_path/'failed-hook').exists()
    with pytest.raises(ValueError, match='callable'):
        prepare_data(tmp_path/'invalid-hook', **kwargs, on_shard_scanned=True)
    source = tmp_path/'source.jsonl'
    source.write_text('{"text":"hello"}\n')
    with pytest.raises(ValueError, match='HF dataset'):
        prepare_data(tmp_path/'local-hook', source_jsonl=source, revision='fixture',
                     on_shard_scanned=lambda *args: None)


def test_shard_reader_and_last_batch_released_before_next_reader(tmp_path, monkeypatch):
    import gc
    import weakref
    import pyarrow.parquet as pq
    readers, batches, closed = [], [], []
    class Batch:
        def to_pylist(self):
            return [{'id': str(len(readers)), 'text': 'A'}]
    class Reader:
        def __init__(self, path):
            gc.collect()
            assert not readers or readers[-1]() is None
            assert not batches or batches[-1]() is None
            readers.append(weakref.ref(self))
        def iter_batches(self, **kwargs):
            batch = Batch()
            batches.append(weakref.ref(batch))
            yield batch
        def close(self):
            closed.append(True)
    monkeypatch.setattr(pq, 'ParquetFile', Reader)
    paths = {}
    for name in ('a.parquet', 'b.parquet'):
        path = tmp_path/name
        path.write_bytes(b'mocked parquet payload')
        paths[name] = str(path)
    class Tokenizer:
        name, eot_token = 'lifecycle-test', 50256
        def encode(self, text, **kwargs):
            return [123]
    manifest = prepare_data(tmp_path/'released', hf_dataset='fixture/fineweb', revision='a'*40,
        hf_data_files=list(paths), hf_local_shards=paths,
        hf_shard_checksums={name: hashlib.sha256(b'mocked parquet payload').hexdigest()
                            for name in paths}, tokenizer=Tokenizer())
    assert json.loads(manifest.read_text())['documents'] == 2
    assert closed == [True, True]
    gc.collect()
    assert all(ref() is None for ref in [*readers, *batches])
