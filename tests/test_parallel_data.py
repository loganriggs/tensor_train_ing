"""Parallel preparation preserves the frozen dataset's byte-level contract."""
import hashlib
import json

import numpy as np
import pytest
import tiktoken

from tnlm.data import SPLIT_ORDER, prepare_data


def _source(path, records):
    with path.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')
    return path


def _documents(manifest):
    return [json.loads(line) for line in manifest.with_name('documents.jsonl').read_text().splitlines()]


def _artifacts(manifest):
    return {path.relative_to(manifest.parent): path.read_bytes() for path in manifest.parent.rglob('*') if path.is_file()}


def _rank(record):
    return hashlib.sha256(('selection\0' + record['id']).encode()).hexdigest()


@pytest.mark.parametrize('batch_bytes', [1, 131, 1048576])
def test_parallel_real_gpt2_is_byte_identical_to_serial(tmp_path, batch_bytes):
    texts = ['', 'café\r\nTensor networks αβ.', '<|endoftext|> is literal text.',
             '😺\rNew line! ', 'long ' * 300]
    records = [{'id': str(i), 'text': texts[i % len(texts)]} for i in range(37)]
    records.extend([records[0], records[-1]])
    source = _source(tmp_path / 'source.jsonl', records)
    kwargs = dict(source_jsonl=source, revision='parallel-fixture-v1', max_documents=31,
                  max_tokens=900, split_fractions=dict.fromkeys(SPLIT_ORDER, .25))
    serial = prepare_data(tmp_path / 'serial', workers=1, **kwargs)
    parallel = prepare_data(tmp_path / 'parallel', workers=2,
                            tokenization_batch_bytes=batch_bytes, **kwargs)
    assert _artifacts(parallel) == _artifacts(serial)
    metadata = json.loads(parallel.read_text())
    assert metadata['format_version'] == 1
    assert metadata['scan_records'] == 39
    assert metadata['scan_unique_documents'] == 37
    assert metadata['tokenizer']['name'] == 'gpt2'
    assert not metadata['tokenizer'].get('test_only')
    assert set(metadata['artifacts']) == {'documents.jsonl', *(s + '.tokens.bin' for s in SPLIT_ORDER)}


@pytest.mark.parametrize('cut', [0, 1])
def test_parallel_token_cap_preserves_eos_and_canonical_order(tmp_path, cut):
    records = [{'id': str(i), 'text': 'café\r\nhello ' + str(i)} for i in range(19)]
    selected = sorted(records, key=_rank)[:7]
    encoder = tiktoken.get_encoding('gpt2')
    encoded = [encoder.encode(r['text'].replace('\r\n', '\n'),
                              allowed_special=set(), disallowed_special=()) + [50256]
               for r in selected]
    cap = sum(map(len, encoded[:2])) - cut
    source = _source(tmp_path / 'source.jsonl', records)
    manifest = prepare_data(tmp_path / 'data', source_jsonl=source, revision='v1',
                            max_documents=7, max_tokens=cap, workers=2,
                            tokenization_batch_bytes=10,
                            split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    docs = _documents(manifest)
    assert [d['id'] for d in docs] == [r['id'] for r in selected[:2]]
    stored = np.fromfile(manifest.with_name('train.tokens.bin'), dtype='<i4').tolist()
    assert stored == (encoded[0] + encoded[1])[:cap]
    assert docs[0]['synthetic_eos_token_offset'] == len(encoded[0]) - 1
    assert docs[1]['truncated'] is bool(cut)
    assert docs[1]['synthetic_eos_token_offset'] == (None if cut else len(encoded[1]) - 1)
    assert sum(json.loads(manifest.read_text())['tokens'].values()) == cap


def test_parallel_selection_is_source_order_independent(tmp_path):
    records = [{'id': str(i), 'text': f'document {i} 🐈'} for i in range(73)]
    manifests = [prepare_data(tmp_path / name, source_jsonl=_source(tmp_path / f'{name}.jsonl', rows),
                              revision='v1', max_documents=23, max_tokens=137, workers=2,
                              tokenization_batch_bytes=79)
                 for name, rows in [('forward', records), ('reverse', records[::-1])]]
    docs = [[{k: v for k, v in doc.items() if k != 'source_offset'}
             for doc in _documents(manifest)] for manifest in manifests]
    assert docs[0] == docs[1]
    for split in SPLIT_ORDER:
        assert manifests[0].with_name(f'{split}.tokens.bin').read_bytes() == manifests[1].with_name(f'{split}.tokens.bin').read_bytes()


def test_parallel_duplicate_conflict_after_token_budget_still_rejected(tmp_path):
    rows = [{'id': str(i), 'text': 'original'} for i in range(100)]
    rows.append({'id': '99', 'text': 'changed'})
    source = _source(tmp_path / 'source.jsonl', rows)
    with pytest.raises(ValueError, match='conflicting content'):
        prepare_data(tmp_path / 'failed', source_jsonl=source, revision='v1',
                     workers=2, max_documents=1, max_tokens=1)
    assert not (tmp_path / 'failed').exists()


@pytest.mark.parametrize('field,value', [('workers', 0), ('workers', -1), ('workers', True),
                                         ('workers', 1.5), ('tokenization_batch_bytes', 0),
                                         ('tokenization_batch_bytes', True),
                                         ('tokenization_batch_bytes', 1.5)])
def test_parallel_controls_reject_invalid_values_and_clean_output(tmp_path, field, value):
    source = _source(tmp_path / 'source.jsonl', [{'id': 'one', 'text': 'hello'}])
    with pytest.raises(ValueError, match=field):
        prepare_data(tmp_path / 'failed', source_jsonl=source, revision='v1', **{field: value})
    assert not (tmp_path / 'failed').exists()


def test_parallel_rejects_unshareable_injected_tokenizer(tmp_path):
    class FixtureTokenizer:
        eot_token = 50256

        def encode(self, text, **kwargs):
            return [1]

    source = _source(tmp_path / 'source.jsonl', [{'id': 'one', 'text': 'hello'}])
    with pytest.raises(ValueError, match='tokenizer'):
        prepare_data(tmp_path / 'failed', source_jsonl=source, revision='v1',
                     workers=2, tokenizer=FixtureTokenizer())
    assert not (tmp_path / 'failed').exists()


def test_large_source_selection_uses_global_rank_with_small_output(tmp_path):
    # Selection must scan beyond the output cap, without retaining the corpus.
    count = 12000
    source = _source(tmp_path / 'source.jsonl',
                     ({'id': str(i), 'text': 'bounded selection ' * 20} for i in range(count)))
    manifest = prepare_data(tmp_path / 'data', source_jsonl=source, revision='v1',
                            workers=2, max_documents=3, max_tokens=10000,
                            tokenization_batch_bytes=256)
    expected = sorted(({'id': str(i)} for i in range(count)), key=_rank)[:3]
    assert [doc['id'] for doc in _documents(manifest)] == [r['id'] for r in expected]
    metadata = json.loads(manifest.read_text())
    assert metadata['scan_records'] == metadata['scan_unique_documents'] == count
    assert metadata['documents'] == 3
    assert not list(manifest.parent.glob('*.sqlite*'))


def test_parallel_worker_failure_cancels_pending_and_removes_output(tmp_path, monkeypatch):
    import tnlm.data_preparation as preparation

    futures, shutdown = [], []

    class FailedFuture:
        cancelled = False

        def result(self):
            raise RuntimeError('injected tokenization worker failure')

        def cancel(self):
            self.cancelled = True

    class FailingPool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, batch):
            future = FailedFuture()
            futures.append(future)
            return future

        def shutdown(self, **kwargs):
            shutdown.append(kwargs)

    monkeypatch.setattr(preparation, 'ProcessPoolExecutor', FailingPool)
    source = _source(tmp_path / 'source.jsonl',
                     [{'id': str(i), 'text': 'worker text'} for i in range(12)])
    with pytest.raises(RuntimeError, match='tokenization worker failure'):
        prepare_data(tmp_path / 'failed', source_jsonl=source, revision='v1',
                     workers=2, tokenization_batch_bytes=1)
    assert len(futures) == 4
    assert all(future.cancelled for future in futures[1:])
    assert shutdown == [{'wait': True, 'cancel_futures': True}]
    assert not (tmp_path / 'failed').exists()


def test_parallel_prefetch_is_bounded_and_generator_close_reaps_pool(monkeypatch):
    import tnlm.data_preparation as preparation

    seen, futures, shutdown = [], [], []

    def rows():
        for i in range(10000):
            seen.append(i)
            yield str(i), 'x', i, 'hash', 'rank'

    class Future:
        cancelled = False

        def __init__(self, batch):
            self.batch = batch

        def result(self):
            return self.batch

        def cancel(self):
            self.cancelled = True

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, batch):
            future = Future(batch)
            futures.append(future)
            return future

        def shutdown(self, **kwargs):
            shutdown.append(kwargs)

    monkeypatch.setattr(preparation, 'ProcessPoolExecutor', Pool)
    encoded = preparation.ordered_tokenization(rows(), tokenizer=None, workers=2, batch_bytes=1)
    assert next(encoded)[0] == '0'
    assert len(futures) == 4
    assert len(seen) <= 5  # Four queued documents plus the batcher's one-row lookahead.
    encoded.close()
    assert all(future.cancelled for future in futures[1:])
    assert shutdown == [{'wait': True, 'cancel_futures': True}]


@pytest.mark.parametrize('name', ['max_documents', 'max_tokens'])
@pytest.mark.parametrize('value', [True, 1.5, 0, -1])
def test_data_bounds_are_strict_integers(tmp_path, name, value):
    from tnlm.data import prepare_data
    source = tmp_path / 'source.jsonl'
    source.write_text('{"id":"a","text":"hello"}\n')
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='positive integer data bounds'):
        prepare_data(out, source_jsonl=source, revision='test', **{name: value})
    assert not out.exists()
