import json

import pytest
import torch

from tnlm.data import PackedSampler, batch_hash, prepare_data


class TinyTokenizer:
    name = 'offline-fixture'
    eot_token = 50256

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def prepare(tmp_path, text=('abcd', 'ef', 'ghijklmnop')):
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps({'id': str(i), 'text': t}) + '\n' for i, t in enumerate(text)))
    return prepare_data(tmp_path / 'data', source_jsonl=source, revision='fixture-v1',
                        tokenizer=TinyTokenizer(), split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})


def test_replay_and_boundaries(tmp_path):
    manifest = prepare(tmp_path)
    sampler = PackedSampler(manifest, context=4, batch_size=2, seed=17)
    sampler.next_batch()
    state = sampler.state_dict()
    expected = [batch_hash(sampler.next_batch()) for _ in range(5)]
    resumed = PackedSampler(manifest, context=4, batch_size=2, seed=17)
    resumed.load_state_dict(state)
    assert [batch_hash(resumed.next_batch()) for _ in range(5)] == expected
    for _ in range(5):
        idx, ids, targets = resumed.next_batch()
        assert idx.shape == ids.shape == targets.shape == (2, 4)
        assert idx.dtype == ids.dtype == targets.dtype == torch.int64
        for b in range(2):
            for j in range(3):
                if ids[b, j] != ids[b, j + 1] or ids[b, j] < 0:
                    assert targets[b, j] == -100
                elif targets[b, j] != -100:
                    assert targets[b, j] == idx[b, j + 1]


def test_padding_and_provenance(tmp_path):
    manifest = prepare(tmp_path, ('a\r\nb',))
    metadata = json.loads(manifest.read_text())
    assert metadata['tokens']['train'] == 4
    doc = json.loads(manifest.with_name('documents.jsonl').read_text())
    assert doc['content_sha256'] != doc['processed_sha256']
    assert doc['source_token_start'] == 0 and doc['source_token_end'] == 4
    idx, ids, targets = PackedSampler(manifest, context=8).next_batch()
    assert targets[0].tolist() == [10, 98, 50256, -100, -100, -100, -100, -100]
    assert (ids[0, 4:] < 0).all()


def test_checksum_and_state_rejection(tmp_path):
    manifest = prepare(tmp_path)
    sampler = PackedSampler(manifest, context=4)
    state = sampler.state_dict()
    state['context'] = 99
    with pytest.raises(ValueError, match='context'):
        sampler.load_state_dict(state)
    with manifest.with_name('train.tokens.bin').open('ab') as f:
        f.write(b'x')
    with pytest.raises(ValueError, match='checksum'):
        PackedSampler(manifest)


def test_immutable_bound_and_split_membership(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps({'id': str(i), 'text': 'abcdefgh'}) + '\n' for i in range(30)))
    manifests = [prepare_data(tmp_path / name, source_jsonl=source, revision='v1', tokenizer=TinyTokenizer(),
                              max_tokens=100, max_documents=9) for name in ('one', 'two')]
    assert manifests[0].read_bytes() == manifests[1].read_bytes()
    metadata = json.loads(manifests[0].read_text())
    assert sum(metadata['tokens'].values()) <= 100 and metadata['documents'] == 9
    with pytest.raises(FileExistsError):
        prepare_data(tmp_path / 'one', source_jsonl=source, revision='v1', tokenizer=TinyTokenizer())
    with pytest.raises(ValueError, match='revision'):
        prepare_data(tmp_path / 'bad', source_jsonl=source, revision='main', tokenizer=TinyTokenizer())


@pytest.mark.parametrize('context', [1, 2, 3, 4, 7, 32])
def test_every_within_document_pair_once_and_lineage(tmp_path, context):
    from collections import Counter
    manifest = prepare(tmp_path, ('a', '', 'bcdefghijkl', 'mnop', 'q', 'rstuvw'))
    sampler = PackedSampler(manifest, context=context, seed=13)
    expected = Counter((doc['id'], offset) for doc in sampler.docs
                       for offset in range(doc['token_end'] - doc['token_start'] - 1))
    observed = Counter()
    for _ in range(sampler.epoch_row_count):
        idx, ids, targets = sampler.next_batch()
        mapping = sampler.batch_mappings()[0]
        for j in range(context):
            if targets[0, j] == -100:
                continue
            matching = [r for r in mapping['ranges'] if r['packed_token_start'] <= j < r['packed_token_end']]
            assert len(matching) == 1
            r = matching[0]
            source_offset = r['source_token_start'] + j - r['packed_token_start']
            observed[r['document_id'], source_offset] += 1
            assert ids[0, j] == r['numeric_id']
        # Metadata access is read-only and defensive.
        mapping['ranges'].clear()
        assert sampler.batch_mappings()[0]['ranges']
    assert observed == expected


def test_failed_preparation_cleanup_and_eos(tmp_path):
    source = tmp_path / 'bad.jsonl'
    source.write_text('{"id":"one","text":"abc"}\n{"id":"bad","text":12}\n')
    with pytest.raises(ValueError, match='string text'):
        prepare_data(tmp_path / 'failed', source_jsonl=source, revision='v1', tokenizer=TinyTokenizer())
    assert not (tmp_path / 'failed').exists()
    tokenizer = TinyTokenizer()
    tokenizer.eot_token = 999
    with pytest.raises(ValueError, match='EOS'):
        prepare_data(tmp_path / 'eos', source_jsonl=source, revision='v1', tokenizer=tokenizer)
    assert not (tmp_path / 'eos').exists()


def test_manifest_hash_and_test_tokenizer_metadata(tmp_path):
    from tnlm.data import manifest_hash
    manifest = prepare(tmp_path)
    assert manifest_hash(manifest) == manifest_hash(manifest.parent)
    assert json.loads(manifest.read_text())['tokenizer']['test_only'] is True
    manifest.write_text(manifest.read_text() + ' ')
    with pytest.raises(ValueError, match='manifest checksum'):
        manifest_hash(manifest)


def test_hf_requires_explicit_upstream_checksums(tmp_path):
    common = dict(hf_dataset='HuggingFaceFW/fineweb', revision='a' * 40,
                  tokenizer=TinyTokenizer(), hf_data_files=['data/shard.parquet'])
    with pytest.raises(ValueError, match='cover every explicit shard'):
        prepare_data(tmp_path / 'missing', **common)
    with pytest.raises(ValueError, match='SHA256'):
        prepare_data(tmp_path / 'invalid', **common,
                     hf_shard_checksums={'data/shard.parquet': 'invalid'})
    assert not (tmp_path / 'missing').exists()
    assert not (tmp_path / 'invalid').exists()


def test_hf_checksum_metadata_with_mocked_download_and_parquet(tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace
    calls = []
    shard = tmp_path / 'verified.parquet'
    shard.write_bytes(b'tiny parquet fixture bytes')
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()

    def download(**kwargs):
        calls.append(kwargs)
        return str(shard)

    class ParquetFile:
        def __init__(self, path):
            assert path == shard

        def iter_batches(self, **kwargs):
            yield SimpleNamespace(to_pylist=lambda: [{'id': 'remote-document', 'text': 'hello'}])

    parquet = SimpleNamespace(ParquetFile=ParquetFile)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules, 'pyarrow', SimpleNamespace(parquet=parquet))
    monkeypatch.setitem(sys.modules, 'pyarrow.parquet', parquet)
    manifest = prepare_data(tmp_path / 'remote', hf_dataset='HuggingFaceFW/fineweb',
                            revision='a' * 40, hf_data_files=['data/shard.parquet'],
                            hf_shard_checksums={'data/shard.parquet': digest.upper()},
                            tokenizer=TinyTokenizer())
    metadata = json.loads(manifest.read_text())
    assert metadata['source']['shard_sha256'] == {'data/shard.parquet': digest}
    assert 'verified before parquet reading' in metadata['source']['shard_hash_verification']
    assert calls == [{'repo_id': 'HuggingFaceFW/fineweb', 'repo_type': 'dataset',
                      'revision': 'a' * 40, 'filename': 'data/shard.parquet'}]
    doc = json.loads(manifest.with_name('documents.jsonl').read_text())
    assert doc['source_offset'] == {'shard': 'data/shard.parquet', 'row': 0}
    calls.clear()
    local_manifest = prepare_data(tmp_path / 'local', hf_dataset='HuggingFaceFW/fineweb',
                                  revision='a' * 40, hf_data_files=['data/shard.parquet'],
                                  hf_shard_checksums={'data/shard.parquet': digest},
                                  hf_local_shards={'data/shard.parquet': shard},
                                  tokenizer=TinyTokenizer())
    assert not calls
    assert local_manifest.read_bytes() == manifest.read_bytes()


def test_hf_local_shard_checksum_rejects_wrong_bytes(tmp_path):
    shard = tmp_path / 'bad.parquet'
    shard.write_bytes(b'wrong raw shard content')
    with pytest.raises(ValueError, match='raw shard checksum mismatch'):
        prepare_data(tmp_path / 'rejected', hf_dataset='HuggingFaceFW/fineweb',
                     revision='a' * 40, hf_data_files=['shard.parquet'],
                     hf_shard_checksums={'shard.parquet': '0' * 64},
                     hf_local_shards={'shard.parquet': shard}, tokenizer=TinyTokenizer())
    assert not (tmp_path / 'rejected').exists()


def test_failed_atomic_manifest_publication_cleans_output(tmp_path, monkeypatch):
    import tnlm.data as data
    source = tmp_path / 'source.jsonl'
    source.write_text('{"id":"one","text":"abc"}\n')

    def fail_replace(*args):
        raise OSError('simulated publication failure')

    monkeypatch.setattr(data.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='publication failure'):
        prepare_data(tmp_path / 'failed', source_jsonl=source, revision='v1', tokenizer=TinyTokenizer())
    assert not (tmp_path / 'failed').exists()


def test_split_fraction_order_and_manifest_replay(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps({'id': str(i), 'text': 'abc'}) + '\n' for i in range(100)))
    fractions = {'train': .5, 'val': .2, 'test': .2, 'calibration': .1}
    manifests = []
    for i, configuration in enumerate((fractions, dict(reversed(list(fractions.items()))))):
        manifests.append(prepare_data(tmp_path / str(i), source_jsonl=source, revision='v1',
                                      split_fractions=configuration, tokenizer=TinyTokenizer()))
    replay = json.loads(manifests[0].read_text())['split_fractions']
    manifests.append(prepare_data(tmp_path / 'replay', source_jsonl=source, revision='v1',
                                  split_fractions=replay, tokenizer=TinyTokenizer()))
    assert len({p.read_bytes() for p in manifests}) == 1
    assert json.loads(manifests[0].read_text())['split_order'] == ['train', 'val', 'test', 'calibration']


def test_hash_selection_source_order_independent_and_bounded(tmp_path):
    import hashlib
    records = [{'id': str(i), 'text': f'document {i}'} for i in range(50)]
    prepared = []
    for name, sequence in [('forward', records), ('reverse', records[::-1])]:
        source = tmp_path / f'{name}.jsonl'
        source.write_text(''.join(json.dumps(r) + '\n' for r in sequence))
        prepared.append(prepare_data(tmp_path / name, source_jsonl=source, revision='v1',
                                     max_documents=7, max_tokens=35, tokenizer=TinyTokenizer()))
    load = lambda p: [json.loads(line) for line in p.with_name('documents.jsonl').read_text().splitlines()]
    first, second = map(load, prepared)
    expected = sorted(records, key=lambda r: hashlib.sha256(('selection\0' + r['id']).encode()).hexdigest())[:7]
    assert [d['id'] for d in first] == [r['id'] for r in expected[:len(first)]]
    assert [{k: v for k, v in d.items() if k != 'source_offset'} for d in first] == [
        {k: v for k, v in d.items() if k != 'source_offset'} for d in second]
    for split in ('train', 'val', 'test', 'calibration'):
        assert prepared[0].with_name(f'{split}.tokens.bin').read_bytes() == prepared[1].with_name(f'{split}.tokens.bin').read_bytes()
    for manifest in prepared:
        metadata = json.loads(manifest.read_text())
        assert metadata['scan_records'] == metadata['scan_unique_documents'] == 50
        assert sum(metadata['tokens'].values()) <= 35
        assert not list(manifest.parent.glob('*.sqlite*'))
    split = next(s for s, count in json.loads(prepared[0].read_text())['tokens'].items() if count)
    samplers = [PackedSampler(p, split=split, context=3, seed=11) for p in prepared]
    assert [batch_hash(samplers[0].next_batch()) for _ in range(5)] == [
        batch_hash(samplers[1].next_batch()) for _ in range(5)]


def test_conflicting_duplicate_outside_selected_subset_rejected(tmp_path):
    source = tmp_path / 'duplicates.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in [
        {'id': 'one', 'text': 'same'}, {'id': 'one', 'text': 'same'},
        {'id': 'two', 'text': 'abc'}, {'id': 'one', 'text': 'changed'}]))
    with pytest.raises(ValueError, match='conflicting content'):
        prepare_data(tmp_path / 'rejected', source_jsonl=source, revision='v1',
                     max_documents=1, max_tokens=1, tokenizer=TinyTokenizer())
    assert not (tmp_path / 'rejected').exists()


def test_packing_version_and_synthetic_eos_metadata(tmp_path):
    manifest = prepare(tmp_path, ('abc',))
    doc = json.loads(manifest.with_name('documents.jsonl').read_text())
    assert doc['processed_token_end'] == 3
    assert doc['source_token_end'] == 4
    assert doc['synthetic_eos_token_offset'] == 3
    sampler = PackedSampler(manifest)
    state = sampler.state_dict()
    assert state['packing_version']
    state['packing_version'] = 'unknown'
    with pytest.raises(ValueError, match='packing_version'):
        sampler.load_state_dict(state)


def test_selection_and_split_hashes_are_independent(tmp_path):
    import hashlib
    from collections import Counter
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps({'id': str(i), 'text': 'a'}) + '\n' for i in range(1000)))
    fractions = {'train': .25, 'val': .25, 'test': .25, 'calibration': .25}
    manifest = prepare_data(tmp_path / 'selected', source_jsonl=source, revision='v1',
                            max_documents=50, tokenizer=TinyTokenizer(), split_fractions=fractions)
    docs = [json.loads(line) for line in manifest.with_name('documents.jsonl').read_text().splitlines()]
    counts = Counter(doc['split'] for doc in docs)
    assert set(counts) == set(fractions)
    assert all(5 <= count <= 25 for count in counts.values())
    expected_ids = sorted(map(str, range(1000)), key=lambda doc_id: hashlib.sha256(('selection\0' + doc_id).encode()).digest())[:50]
    assert [doc['id'] for doc in docs] == expected_ids
    for doc in docs:
        rank = int.from_bytes(hashlib.sha256(doc['id'].encode()).digest(), 'big')
        expected_split = list(fractions)[rank // (2**254)]
        assert doc['split'] == expected_split


def reference_epoch(sampler, epoch):
    """Independent token-by-token legacy packing, only for tiny fixtures."""
    import random
    order = list(range(len(sampler.docs)))
    random.Random(f'{sampler.seed}:{epoch}').shuffle(order)
    rows, current = [], []
    for i in order:
        doc = sampler.docs[i]
        tokens = sampler.tokens[doc['token_start']:doc['token_end']].tolist()
        for offset, token in enumerate(tokens):
            current.append((token, doc['numeric_id'], offset, doc['id']))
            if len(current) == sampler.context + 1:
                rows.append(current)
                current = current[-1:] if offset + 1 < len(tokens) else []
    if current:
        rows.append(current)
    return rows


@pytest.mark.parametrize('context', [1, 2, 3, 5, 13])
def test_incremental_matches_independent_legacy_reference(tmp_path, context):
    manifest = prepare(tmp_path, ('a', '', 'bcdefghijkl', 'mnop', 'q', 'rstuvw', 'xyz0123456789'))
    sampler = PackedSampler(manifest, context=context, seed=29)
    assert not hasattr(sampler, 'rows')
    for epoch in range(3):
        reference = reference_epoch(sampler, epoch)
        for row_cursor, row in enumerate(reference):
            idx, ids, targets = sampler.next_batch()
            assert sampler.epoch_row_count == len(reference)
            expected_idx = [item[0] for item in row[:context]]
            expected_ids = [item[1] for item in row[:context]]
            padding = context - len(expected_idx)
            expected_idx += [50256] * padding
            expected_ids += list(range(-1, -padding - 1, -1))
            expected_targets = [row[j + 1][0] if j + 1 < len(row) and row[j][1] == row[j + 1][1]
                                else -100 for j in range(context)]
            assert idx[0].tolist() == expected_idx
            assert ids[0].tolist() == expected_ids
            assert targets[0].tolist() == expected_targets
            mapping = sampler.batch_mappings()[0]
            assert (mapping['epoch'], mapping['row_cursor']) == (epoch, row_cursor)
            reconstructed = []
            for r in mapping['ranges']:
                reconstructed.extend((r['numeric_id'], offset, r['document_id'])
                                     for offset in range(r['source_token_start'], r['source_token_end']))
            assert reconstructed == [item[1:] for item in row]
            state = sampler.state_dict()
            restored = PackedSampler(manifest, context=context, seed=29)
            # Loading must install stored document/token cursors, not pack prior rows.
            def forbid_replay():
                raise AssertionError('restore replayed rows')
            restored._next_row = forbid_replay
            restored.load_state_dict(state)
            del restored._next_row
            assert restored.state_dict() == state
            assert batch_hash(restored.next_batch()) == batch_hash(sampler.next_batch())
            sampler.load_state_dict(state)


def test_row_count_large_document_without_materialized_rows():
    sampler = PackedSampler.__new__(PackedSampler)
    sampler.docs = [{'token_start': 0, 'token_end': 1_000_000_000}]
    sampler.seed, sampler.epoch, sampler.context = 1, 0, 1
    sampler._order()
    assert sampler.epoch_row_count == 999_999_999
    assert sampler.order == [0]
    assert not hasattr(sampler, 'rows')
    assert sampler._next_row() == [(0, 0, 2)]
    assert sampler._next_row() == [(0, 1, 2)]


def test_real_local_parquet_gpt2_checksum_lineage_and_replay(tmp_path):
    """Exercise actual Parquet decoding and GPT2, without any Hub downloads."""
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tiktoken

    texts = ['café\r\nTensor networks αβ.', 'A second real document: 12345.', 'Final document.']
    shard = tmp_path / 'tiny.parquet'
    pq.write_table(pa.table({'id': ['real-0', 'real-1', 'real-2'], 'text': texts}), shard)
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    shard_name = 'fixtures/tiny.parquet'
    kwargs = dict(hf_dataset='fixture/local-only-parquet', revision='a' * 40,
                  hf_data_files=[shard_name], hf_shard_checksums={shard_name: digest},
                  hf_local_shards={shard_name: shard},
                  split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    manifest = prepare_data(tmp_path / 'prepared', **kwargs)
    metadata = json.loads(manifest.read_text())
    assert metadata['source']['shard_sha256'] == {shard_name: digest}
    assert metadata['tokenizer']['name'] == 'gpt2'
    assert len(metadata['tokenizer']['fingerprint_sha256']) == 64
    assert not metadata['tokenizer'].get('test_only', False)
    docs = [json.loads(line) for line in manifest.with_name('documents.jsonl').read_text().splitlines()]
    encoding = tiktoken.get_encoding('gpt2')
    for doc in docs:
        source_row = int(doc['id'].split('-')[1])
        assert doc['source_offset'] == {'shard': shard_name, 'row': source_row}
        processed = texts[source_row].replace('\r\n', '\n')
        expected = encoding.encode(processed) + [50256]
        assert doc['processed_token_end'] == len(expected) - 1
        assert doc['synthetic_eos_token_offset'] == len(expected) - 1
        assert doc['content_sha256'] == hashlib.sha256(texts[source_row].encode()).hexdigest()
        import numpy as np
        stored = np.fromfile(manifest.with_name('train.tokens.bin'), dtype='<i4')
        assert stored[doc['token_start']:doc['token_end']].tolist() == expected
    sampler = PackedSampler(manifest, context=64, seed=19)
    idx, ids, targets = sampler.next_batch()
    transitions = ids[:, :-1] != ids[:, 1:]
    assert int(transitions.sum()) >= 2
    assert (targets[:, :-1][transitions] == -100).all()
    assert (targets[ids < 0] == -100).all()
    assert int((idx == 50256).logical_and(ids >= 0).sum()) == 3
    assert all(r['source_offset']['shard'] == shard_name for r in sampler.batch_mappings()[0]['ranges'])
    state = sampler.state_dict()
    hashes = [batch_hash(sampler.next_batch()) for _ in range(4)]
    restored = PackedSampler(manifest, context=64, seed=19)
    restored.load_state_dict(state)
    assert [batch_hash(restored.next_batch()) for _ in range(4)] == hashes
    shard.write_bytes(shard.read_bytes() + b'corrupted-after-verification')
    with pytest.raises(ValueError, match='raw shard checksum mismatch'):
        prepare_data(tmp_path / 'rejected', **kwargs)
    assert not (tmp_path / 'rejected').exists()


def audit_manifest(tmp_path, name, records):
    """Minimal independently checksummed metadata; token payload deliberately absent."""
    import hashlib
    directory = tmp_path / name
    directory.mkdir()
    docs = []
    for doc_id, split, text in records:
        docs.append({'id': doc_id, 'split': split,
                     'content_sha256': hashlib.sha256(text.encode()).hexdigest(),
                     'processed_sha256': hashlib.sha256(text.replace('\r\n', '\n').replace('\r', '\n').encode()).hexdigest()})
    metadata = directory / 'documents.jsonl'
    metadata.write_text(''.join(json.dumps(doc) + '\n' for doc in docs))
    manifest = directory / 'manifest.json'
    manifest.write_text(json.dumps({'artifacts': {'documents.jsonl': hashlib.sha256(metadata.read_bytes()).hexdigest(),
                                                 'train.tokens.bin': 'unopened-token-payload'}}))
    directory.joinpath('manifest.sha256').write_text(hashlib.sha256(manifest.read_bytes()).hexdigest() + '\n')
    return manifest


def test_validation_overlap_all_splits_namespaces_and_internal_duplicates(tmp_path):
    from tnlm.data import audit_validation_overlap
    training = audit_manifest(tmp_path, 'training', [('train/source-id', 'train', 'identical'),
                                                    ('calibration/source-id', 'calibration', 'newline\r\ncollision'),
                                                    ('test/source-id', 'test', 'final-test'),
                                                    ('irrelevant', 'val', 'ignore-me')])
    validation = audit_manifest(tmp_path, 'validation', [('validation-a', 'val', 'identical'),
                                                        ('validation-b', 'val', 'newline\ncollision'),
                                                        ('validation-c', 'val', 'final-test'),
                                                        ('validation-d', 'val', 'identical'),
                                                        ('validation-e', 'val', 'unique'),
                                                        ('not-validation', 'train', 'identical')])
    result = audit_validation_overlap(training, validation)
    assert result['selected_validation_documents'] == 5
    assert result['overlap_counts'] == {'train': 2, 'calibration': 1, 'test': 1}
    assert result['duplicate_validation']['documents_with_duplicate_content'] == 2
    assert result['duplicate_validation']['hash_group_counts'] == {'content_sha256': 1, 'processed_sha256': 1}
    assert result['has_overlap']
    calibration = result['overlap_samples']['calibration']
    assert len(calibration) == 1 and calibration[0]['hash_kind'] == 'processed_sha256'
    assert calibration[0]['validation_document_id'] == 'validation-b'
    assert calibration[0]['other_document_id'] == 'calibration/source-id'
    assert 'whole-document' in result['scope'] and 'near-duplicate' in result['scope']
    json.dumps(result, allow_nan=False)


def test_validation_overlap_selected_disjoint_unknown_and_bounded_samples(tmp_path):
    from tnlm.data import audit_validation_overlap
    training = audit_manifest(tmp_path, 'training', [('source', 'train', 'shared')])
    validation = audit_manifest(tmp_path, 'validation', [('unique', 'val', 'disjoint')] +
                                [(f'alias-{i}', 'val', 'shared') for i in range(20)])
    result = audit_validation_overlap(training, validation, document_ids='unique')
    import hashlib
    assert result['selected_validation_ids_sha256'] == hashlib.sha256(json.dumps(['unique'], separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    assert result['selected_validation_documents'] == 1
    assert not result['has_overlap']
    assert result['overlap_counts'] == {'train': 0, 'calibration': 0, 'test': 0}
    assert result['duplicate_validation']['documents_with_duplicate_content'] == 0
    with pytest.raises(ValueError, match='unknown requested'):
        audit_validation_overlap(training, validation, document_ids=['unknown'])
    result = audit_validation_overlap(training, validation)
    assert result['overlap_counts']['train'] == 20
    assert result['duplicate_validation']['documents_with_duplicate_content'] == 20
    assert len(result['overlap_samples']['train']) == result['sample_limit']
    assert len(result['duplicate_validation']['samples']) == result['sample_limit']


def test_validation_overlap_rejects_corrupt_metadata_sidecar_and_invalid_hashes(tmp_path):
    import hashlib
    from tnlm.data import audit_validation_overlap
    training = audit_manifest(tmp_path, 'training', [('source', 'train', 'shared')])
    validation = audit_manifest(tmp_path, 'validation', [('candidate', 'val', 'other')])
    metadata = validation.with_name('documents.jsonl')
    original = metadata.read_text()
    metadata.write_text(original + ' ')
    with pytest.raises(ValueError, match='artifact checksum'):
        audit_validation_overlap(training, validation)
    metadata.write_text(original)
    sidecar = validation.with_name('manifest.sha256')
    original_sidecar = sidecar.read_text()
    sidecar.write_text('0' * 64)
    with pytest.raises(ValueError, match='manifest checksum'):
        audit_validation_overlap(training, validation)
    sidecar.write_text(original_sidecar)
    doc = json.loads(original)
    doc['processed_sha256'] = 'invalid'
    metadata.write_text(json.dumps(doc) + '\n')
    manifest = json.loads(validation.read_text())
    manifest['artifacts']['documents.jsonl'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    validation.write_text(json.dumps(manifest))
    sidecar.write_text(hashlib.sha256(validation.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='invalid processed_sha256'):
        audit_validation_overlap(training, validation)


def test_rejected_restore_preserves_state_mappings_and_next_batch(tmp_path):
    manifest = prepare(tmp_path, ('abcdefghijklm', 'nopqrst', 'uvwxy'))
    sampler = PackedSampler(manifest, context=3, seed=47)
    sampler.next_batch()
    state, mappings = sampler.state_dict(), sampler.batch_mappings()
    reference = PackedSampler(manifest, context=3, seed=47)
    reference.load_state_dict(state)
    expected_hash = batch_hash(reference.next_batch())
    malformed = [dict(state, epoch=99, cursor=10**9),
                 dict(state, epoch=99, document_cursor=10**9),
                 dict(state, cursor=1, document_cursor=0, token_cursor=0),
                 dict(state, token_cursor=10**9)]
    for candidate in malformed:
        with pytest.raises(ValueError, match='cursor'):
            sampler.load_state_dict(candidate)
        assert sampler.state_dict() == state
        assert sampler.batch_mappings() == mappings
    assert batch_hash(sampler.next_batch()) == expected_hash


@pytest.mark.parametrize('corruption', ['duplicate_numeric_id', 'duplicate_id', 'range_overlap',
                                        'missing_coverage', 'wrong_token_count'])
def test_sampler_rejects_checksum_valid_broken_lineage(tmp_path, corruption):
    import hashlib
    manifest = prepare(tmp_path)
    metadata_path = manifest.with_name('documents.jsonl')
    docs = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    metadata = json.loads(manifest.read_text())
    if corruption == 'duplicate_numeric_id':
        docs[1]['numeric_id'] = docs[0]['numeric_id']
    elif corruption == 'duplicate_id':
        docs[1]['id'] = docs[0]['id']
    elif corruption == 'range_overlap':
        docs[1]['token_start'] -= 1
    elif corruption == 'missing_coverage':
        docs.pop()
        metadata['documents'] -= 1
    else:
        metadata['tokens']['train'] += 1
    metadata_path.write_text(''.join(json.dumps(doc) + '\n' for doc in docs))
    metadata['artifacts']['documents.jsonl'] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(metadata))
    manifest.with_name('manifest.sha256').write_text(hashlib.sha256(manifest.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='document|payload'):
        PackedSampler(manifest)
