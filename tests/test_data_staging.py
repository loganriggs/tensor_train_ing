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
