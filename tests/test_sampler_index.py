"""Compact sampler compatibility against the frozen pre-index implementation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil

import numpy as np
import pytest
import torch

from helpers import FixtureTokenizer
from legacy_sampler_oracle import PackedSampler as LegacySampler, RankPackedSampler as LegacyRankSampler
from tnlm.data import PackedSampler, RankPackedSampler, prepare_data


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / 'source.jsonl'
    lengths = (1, 2, 3, 4, 5, 12, 33, 80)
    source.write_text(''.join(json.dumps({'id': f'doc-{i}',
        'text': ''.join(chr(97 + (i + j) % 26) for j in range(lengths[i % len(lengths)]))}) + '\n'
        for i in range(48)))
    return prepare_data(tmp_path / 'data', source_jsonl=source, revision='index-fixture-v1',
        tokenizer=FixtureTokenizer(), split_fractions={'train': .5, 'val': .5, 'test': 0., 'calibration': 0.})


def _same_next(current, legacy):
    actual, expected = current.next_batch(), legacy.next_batch()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert current.batch_mappings() == legacy.batch_mappings()
    assert current.batch_context_exposure() == legacy.batch_context_exposure()
    assert current.state_dict() == legacy.state_dict()
    json.dumps(current.state_dict(), allow_nan=False)  # Cursors must remain Python ints.


@pytest.mark.parametrize('context', [1, 4, 17, 128])
@pytest.mark.parametrize('batch_size,seed', [(1, 0), (3, 17)])
@pytest.mark.parametrize('split', ['train', 'val'])
def test_compact_sampler_exact_legacy_order_batches_mappings_and_resume(corpus, context, batch_size, seed, split):
    kwargs = dict(context=context, batch_size=batch_size, seed=seed, split=split)
    current, legacy = PackedSampler(corpus, **kwargs), LegacySampler(corpus, **kwargs)
    assert isinstance(current.order, np.ndarray)
    assert current.order.dtype == np.int64
    assert not isinstance(current.docs, (list, tuple))
    for epoch in range(4):
        assert list(current._permutation(epoch)) == legacy._permutation(epoch)
    for _ in range(12):
        _same_next(current, legacy)
    saved = legacy.state_dict()
    resumed, legacy_resumed = PackedSampler(corpus, **kwargs), LegacySampler(corpus, **kwargs)
    resumed.load_state_dict(saved)
    legacy_resumed.load_state_dict(current.state_dict())
    for _ in range(12):
        _same_next(resumed, legacy_resumed)
    assert [dict(current.docs[i]) for i in range(len(current.docs))] == legacy.docs


@pytest.mark.parametrize('old,new,calls', [(1, 4, 0), (4, 17, 3), (17, 4, 5), (128, 256, 12)])
def test_compact_context_fork_exact_lineage_and_replay(corpus, old, new, calls):
    current, legacy = PackedSampler(corpus, context=old, seed=31), LegacySampler(corpus, context=old, seed=31)
    for _ in range(calls):
        _same_next(current, legacy)
    before = current.state_dict()
    fork, old_fork = current.fork_context(new), legacy.fork_context(new)
    assert current.state_dict() == before == legacy.state_dict()
    # Restore the transition before consuming its replay-masked first row.
    restored = PackedSampler(corpus, context=new, seed=31)
    restored.load_state_dict(old_fork.state_dict())
    assert fork.state_dict() == restored.state_dict() == old_fork.state_dict()
    for _ in range(16):
        _same_next(restored, old_fork)


@pytest.mark.parametrize('rank', [0, 1, 2])
def test_compact_rank_sampler_matches_legacy_global_stream(corpus, rank):
    kwargs = dict(rank=rank, world_size=3, batch_size=2, context=17, seed=11)
    current, legacy = RankPackedSampler(corpus, **kwargs), LegacyRankSampler(corpus, **kwargs)
    for _ in range(16):
        _same_next(current, legacy)
    saved = legacy.state_dict()
    resumed = RankPackedSampler(corpus, **kwargs)
    resumed.load_state_dict(saved)
    for _ in range(8):
        _same_next(resumed, legacy)


def test_preparation_publishes_auxiliary_index_without_changing_manifest(corpus):
    from tnlm.sampler_index import ensure_sampler_index, load_sampler_index
    root, manifest = corpus.parent, json.loads(corpus.read_text())
    before = corpus.read_bytes(), corpus.with_suffix('.sha256').read_bytes()
    assert (root / '.sampler-index-v1/index.json').is_file()
    path = ensure_sampler_index(root, manifest)
    assert path == root / '.sampler-index-v1'
    docs = load_sampler_index(root, manifest, 'train')
    assert isinstance(docs.starts, np.ndarray) and isinstance(docs.ends, np.ndarray)
    assert not docs.starts.flags.writeable
    assert not docs.ends.flags.writeable
    assert '.sampler-index-v1/index.json' not in manifest['artifacts']
    assert (corpus.read_bytes(), corpus.with_suffix('.sha256').read_bytes()) == before


def test_concurrent_index_build_publishes_one_valid_cache(corpus):
    from tnlm.sampler_index import ensure_sampler_index, load_sampler_index
    root, manifest = corpus.parent, json.loads(corpus.read_text())
    shutil.rmtree(root / '.sampler-index-v1')
    before = corpus.read_bytes()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: ensure_sampler_index(root, manifest), range(8)))
    assert len(set(results)) == 1
    docs = load_sampler_index(root, manifest, 'train')
    assert len(docs) > 0
    assert corpus.read_bytes() == before
    assert docs.full_document(0)['split'] == 'train'
    current, legacy = PackedSampler(corpus, context=17), LegacySampler(corpus, context=17)
    _same_next(current, legacy)


def test_corrupt_index_rejected_then_explicitly_rebuilt(corpus):
    from tnlm.sampler_index import ensure_sampler_index
    root, manifest = corpus.parent, json.loads(corpus.read_text())
    index = ensure_sampler_index(root, manifest) / 'train.index.bin'
    with index.open('r+b') as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError):
        PackedSampler(corpus, context=17)
    ensure_sampler_index(root, manifest, rebuild=True)
    _same_next(PackedSampler(corpus, context=17), LegacySampler(corpus, context=17))


def test_stale_index_manifest_identity_rejected_then_explicitly_rebuilt(corpus):
    from tnlm.sampler_index import ensure_sampler_index
    root, manifest = corpus.parent, json.loads(corpus.read_text())
    ensure_sampler_index(root, manifest)
    manifest['source']['revision'] = 'changed-source-revision'
    encoded = json.dumps(manifest, sort_keys=True).encode()
    corpus.write_bytes(encoded)
    corpus.with_suffix('.sha256').write_text(hashlib.sha256(encoded).hexdigest() + '\n')
    with pytest.raises(ValueError):
        PackedSampler(corpus, context=17)
    ensure_sampler_index(root, manifest, rebuild=True)
    _same_next(PackedSampler(corpus, context=17), LegacySampler(corpus, context=17))


def test_corrupt_metadata_rejected_even_with_existing_index(corpus):
    PackedSampler(corpus, context=17)
    metadata = corpus.with_name('documents.jsonl')
    with metadata.open('ab') as stream:
        stream.write(b'{}\n')
    with pytest.raises(ValueError):
        PackedSampler(corpus, context=17)


@pytest.mark.parametrize('which', ['permutation', 'row_count'])
def test_disposable_packing_cache_corruption_rebuilds_exact_stream(corpus, which):
    sampler = PackedSampler(corpus, context=17, seed=51)
    cache = sampler.docs.cache_dir / 'packing'
    if which == 'permutation':
        path = next(cache.glob('*.npy'))
        path.write_bytes(b'corrupt cached permutation')
    else:
        path = next(cache.glob('rows-*.json'))
        payload = json.loads(path.read_text())
        payload['payload']['rows'] = 1  # Leave checksum unchanged.
        path.write_text(json.dumps(payload))
    current, legacy = PackedSampler(corpus, context=17, seed=51), LegacySampler(corpus, context=17, seed=51)
    assert current.epoch_row_count == legacy.epoch_row_count
    for _ in range(12):
        _same_next(current, legacy)


def test_release_pages_keeps_metadata_and_exact_sampler_stream(corpus):
    current, legacy = PackedSampler(corpus, context=17, seed=42), LegacySampler(corpus, context=17, seed=42)
    docs = current.docs
    expected_documents = [dict(doc) for doc in legacy.docs]
    index_path = docs.cache_dir / 'train.index.bin'
    before = index_path.read_bytes()
    for _ in range(12):
        released = docs.release_pages()
        assert type(released) is bool
        for i, expected in enumerate(expected_documents):
            assert docs.length(i) == expected['token_end'] - expected['token_start']
            assert docs.start(i) == expected['token_start']
            assert docs.numeric_id(i) == expected['numeric_id']
            assert docs.full_document(i) == expected
        _same_next(current, legacy)
    assert index_path.read_bytes() == before


@pytest.mark.parametrize('fallback', ['empty', 'no_advice', 'no_method', 'os_error', 'value_error'])
def test_release_pages_unsupported_or_failed_advice_is_safe(corpus, monkeypatch, fallback):
    from types import SimpleNamespace
    import tnlm.sampler_index as index_module
    manifest = json.loads(corpus.read_text())
    docs = index_module.load_sampler_index(corpus.parent, manifest, 'test' if fallback == 'empty' else 'train')
    if fallback == 'no_advice':
        monkeypatch.delattr(index_module.mmap, 'MADV_DONTNEED', raising=False)
    elif fallback == 'no_method':
        monkeypatch.setattr(docs, 'rows', SimpleNamespace(_mmap=object()))
    elif fallback in ('os_error', 'value_error'):
        def reject(_advice):
            raise (OSError if fallback == 'os_error' else ValueError)('unsupported advice')
        monkeypatch.setattr(docs, 'rows', SimpleNamespace(_mmap=SimpleNamespace(madvise=reject)))
    docs._index_touches = 100
    assert docs.release_pages() is False
    assert docs._index_touches == 0


@pytest.mark.parametrize('accessor', ['length', 'start', 'numeric_id', 'full_document', 'getitem', 'field'])
def test_index_accesses_trigger_periodic_page_release(corpus, monkeypatch, accessor):
    from tnlm.sampler_index import load_sampler_index
    docs = load_sampler_index(corpus.parent, json.loads(corpus.read_text()), 'train')
    row = docs[0]
    docs._index_touches = 0
    assert docs.PAGE_RELEASE_INTERVAL == 1024
    monkeypatch.setattr(docs, 'PAGE_RELEASE_INTERVAL', 3)
    releases = []
    original_release = docs.release_pages

    def record_release():
        releases.append(docs._index_touches)
        return original_release()

    monkeypatch.setattr(docs, 'release_pages', record_release)
    operation = (lambda: docs[0]) if accessor == 'getitem' else (
        (lambda: row['token_start']) if accessor == 'field' else lambda: getattr(docs, accessor)(0))
    for _ in range(7):
        operation()
    assert releases == [3, 3]
    assert docs._index_touches == 1


@pytest.fixture(scope='module')
def large_sparse_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp('sparse-packing-corpus')
    source = root / 'source.jsonl'
    lengths = (0, 1, 2, 3, 4, 11, 37)
    with source.open('w') as stream:
        for i in range(9001):
            length = 1025 if i % 997 == 0 else lengths[i % len(lengths)]
            stream.write(json.dumps({'id': f'large-doc-{i}', 'text': chr(97 + i % 26) * length}) + '\n')
    return prepare_data(root / 'data', source_jsonl=source, revision='sparse-packing-v1',
        tokenizer=FixtureTokenizer(), split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})


@pytest.mark.parametrize('context', [1, 4, 17])
def test_sparse_prefix_seek_restore_and_fork_match_legacy(large_sparse_corpus, context, monkeypatch):
    current = PackedSampler(large_sparse_corpus, context=context, seed=53)
    legacy = LegacySampler(large_sparse_corpus, context=context, seed=53)
    assert len(current.docs) == 9001
    lengths = [legacy.docs[i]['token_end'] - legacy.docs[i]['token_start'] for i in legacy.order]
    assert 1 in lengths and max(lengths) > 1000
    candidates = {0, legacy.epoch_row_count - 1, legacy.epoch_row_count}
    for position in (4095, 4096, 4097, 8191, 8192, 8193, 8999, 9001):
        rows, fill = legacy._count_lengths(lengths[:position])
        assert current._prefix_count(current.order, 0, position) == (rows, fill)
        candidates.update(row for row in (rows - 1, rows, rows + 1) if 0 <= row <= legacy.epoch_row_count)
    baseline = legacy.state_dict()
    for row in sorted(candidates):
        current.load_state_dict(baseline)
        legacy.load_state_dict(baseline)
        document, token = legacy._seek_row(legacy.order, row)
        assert current._seek_row(current.order, row) == (document, token)
        saved = {**baseline, 'cursor': row, 'document_cursor': document, 'token_cursor': token}
        current.load_state_dict(saved)
        legacy.load_state_dict(saved)
        # Fork before reading; includes replay masks in the first new-context row.
        fork = current.fork_context(context + 5)
        legacy_fork = legacy.fork_context(context + 5)
        assert fork.state_dict() == legacy_fork.state_dict()
        for _ in range(2):
            _same_next(fork, legacy_fork)
            _same_next(current, legacy)
    # Warm-cache near-end prefix lookups must avoid scanning the whole corpus.
    current.load_state_dict(baseline)
    touched = []
    original = current._length

    def count_access(index):
        touched.append(index)
        return original(index)

    monkeypatch.setattr(current, '_length', count_access)
    assert current._prefix_count(current.order, 0, 9000) == legacy._count_lengths(lengths[:9000])
    assert len(touched) <= 4096


def test_sparse_prefix_cache_corruption_rebuilds_near_end(large_sparse_corpus):
    current = PackedSampler(large_sparse_corpus, context=7, seed=72)
    legacy = LegacySampler(large_sparse_corpus, context=7, seed=72)
    cache = current.docs.cache_dir / 'packing'
    for path in cache.glob('rows-*.json'):
        payload = json.loads(path.read_text())
        identity = payload['payload']['identity']
        if identity.get('context') == 7 and identity.get('seed') == '72':
            assert payload['payload']['checkpoints']
            payload['payload']['checkpoints'][-1][1] += 100  # Checksum intentionally stale.
            path.write_text(json.dumps(payload))
            break
    else:
        pytest.fail('expected sparse packing prefix cache')
    rebuilt = PackedSampler(large_sparse_corpus, context=7, seed=72)
    row = legacy.epoch_row_count - 1
    document, token = legacy._seek_row(legacy.order, row)
    saved = {**legacy.state_dict(), 'cursor': row, 'document_cursor': document, 'token_cursor': token}
    rebuilt.load_state_dict(saved)
    legacy.load_state_dict(saved)
    for _ in range(3):
        _same_next(rebuilt, legacy)
