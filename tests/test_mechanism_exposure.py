"""Declared-corpus mechanism exposure proofs; never infer lifetime exposure."""
import hashlib
import json

import pytest

from tnlm.data import prepare_data
from tnlm.tasks import audit_mechanism_training, load_mechanism_tasks, prepare_mechanism_tasks


class OfflineTokenizer:
    name = 'exposure-fixture'
    eot_token = 50256

    def encode(self, text, **kwargs):
        return list(text.encode())


def freeze(out, source, max_tokens=100_000):
    return prepare_data(out, source_jsonl=source, revision='fixture-v1', tokenizer=OfflineTokenizer(),
                        max_tokens=max_tokens,
                        split_fractions={'train': .7, 'val': 0., 'test': 0., 'calibration': .3})


def test_precise_benchmark_source_train_and_calibration_verified(tmp_path):
    task_manifest = prepare_mechanism_tasks(tmp_path / 'tasks')
    corpus = freeze(tmp_path / 'corpus', task_manifest.with_name('train.jsonl'))
    result = audit_mechanism_training(task_manifest, corpus)
    assert result['verified'] and result['status'] == 'verified'
    assert result['source_matches_benchmark_train_sha256']
    assert result['coverage']['complete'] and result['coverage']['fraction'] == 1
    assert result['coverage']['matched_declared_train_rows'] == 64
    assert all(result['coverage']['documents_by_split'].values())
    assert not result['has_validation_overlap']
    assert not any(result['validation_composition_overlap'].values())
    assert 'model-lifetime' in result['scope'] and 'not optimizer visitation' in result['scope']
    assert result['coverage']['truncated_documents'] == 0
    # Corpus token bytes are not needed by this metadata-only audit.
    for payload in corpus.parent.glob('*.tokens.bin'):
        payload.unlink()
    assert audit_mechanism_training(task_manifest, corpus) == result
    json.dumps(result, allow_nan=False)


def test_truncated_selection_has_partial_correctly_scoped_coverage(tmp_path):
    task_manifest = prepare_mechanism_tasks(tmp_path / 'tasks')
    corpus = freeze(tmp_path / 'partial', task_manifest.with_name('train.jsonl'), max_tokens=1)
    result = audit_mechanism_training(task_manifest, corpus)
    assert result['verified']
    assert result['coverage']['matched_declared_train_rows'] == 1
    assert result['coverage']['truncated_documents'] == 1
    assert result['coverage']['fraction'] == 1 / 64 and not result['coverage']['complete']
    assert sum(len(compositions) for split in result['exposed_compositions'].values() for compositions in split.values()) == 1
    assert 'parent-composition upper bounds for truncated documents' in result['scope']


@pytest.mark.parametrize('source_kind', ['mixed', 'validation', 'generic', 'renamed_train'])
def test_generic_or_validation_corpora_unknown_or_overlap(tmp_path, source_kind):
    task_manifest = prepare_mechanism_tasks(tmp_path / 'tasks')
    _, train = load_mechanism_tasks(task_manifest, 'train')
    _, validation = load_mechanism_tasks(task_manifest, 'val')
    if source_kind == 'validation':
        source = task_manifest.with_name('val.jsonl')
    else:
        source = tmp_path / 'external.jsonl'
        if source_kind == 'mixed':
            records = train[:3] + validation[:3]
        elif source_kind == 'renamed_train':
            records = [dict(row, id='another-namespace/' + row['id']) for row in train[:3]]
        else:
            records = [{'id': 'generic', 'text': 'Unrelated arbitrary prose with unknown mechanisms.'}]
        source.write_text(''.join(json.dumps(row) + '\n' for row in records))
    corpus = freeze(tmp_path / 'corpus', source)
    result = audit_mechanism_training(task_manifest, corpus)
    assert not result['verified'] and result['reasons']
    assert not result['source_matches_benchmark_train_sha256']
    expected_overlap = source_kind in ('mixed', 'validation')
    assert result['has_validation_overlap'] is expected_overlap
    assert result['status'] == ('overlap' if expected_overlap else 'unknown')
    if expected_overlap:
        assert any(result['validation_composition_overlap'].values())
        assert any(result['exact_validation_row_matches'].values())


def resign(corpus):
    metadata = json.loads(corpus.read_text())
    metadata['artifacts']['documents.jsonl'] = hashlib.sha256(corpus.with_name('documents.jsonl').read_bytes()).hexdigest()
    corpus.write_text(json.dumps(metadata))
    corpus.with_name('manifest.sha256').write_text(hashlib.sha256(corpus.read_bytes()).hexdigest())


def test_corrupt_metadata_and_sidecar_rejected(tmp_path):
    task_manifest = prepare_mechanism_tasks(tmp_path / 'tasks')
    corpus = freeze(tmp_path / 'corpus', task_manifest.with_name('train.jsonl'))
    path = corpus.with_name('documents.jsonl')
    original = path.read_bytes()
    path.write_bytes(original + b'bad')
    with pytest.raises(ValueError, match='metadata checksum'):
        audit_mechanism_training(task_manifest, corpus)
    path.write_bytes(original)
    corpus.with_name('manifest.sha256').write_text('0' * 64)
    with pytest.raises(ValueError, match='manifest checksum'):
        audit_mechanism_training(task_manifest, corpus)


@pytest.mark.parametrize('field', ['content_sha256', 'processed_sha256', 'source_offset', 'id'])
def test_changed_training_identity_cannot_certify_even_if_resigned(tmp_path, field):
    task_manifest = prepare_mechanism_tasks(tmp_path / 'tasks')
    corpus = freeze(tmp_path / 'corpus', task_manifest.with_name('train.jsonl'))
    path = corpus.with_name('documents.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = next(row for row in rows if row['split'] in ('train', 'calibration'))
    selected[field] = 123456 if field == 'source_offset' else ('different-id' if field == 'id' else '0' * 64)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    resign(corpus)
    result = audit_mechanism_training(task_manifest, corpus)
    assert not result['verified']
    assert result['coverage']['unmatched_documents'] == 1


def test_invalid_training_count_metadata_rejected_even_if_resigned(tmp_path):
    task_manifest = prepare_mechanism_tasks(tmp_path / 'tasks')
    corpus = freeze(tmp_path / 'corpus', task_manifest.with_name('train.jsonl'))
    metadata = json.loads(corpus.read_text())
    metadata['documents'] += 1
    corpus.write_text(json.dumps(metadata))
    corpus.with_name('manifest.sha256').write_text(hashlib.sha256(corpus.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='count/coverage'):
        audit_mechanism_training(task_manifest, corpus)


def test_inconsistent_partial_document_lineage_rejected(tmp_path):
    task_manifest = prepare_mechanism_tasks(tmp_path / 'tasks')
    corpus = freeze(tmp_path / 'partial', task_manifest.with_name('train.jsonl'), max_tokens=1)
    path = corpus.with_name('documents.jsonl')
    row = json.loads(path.read_text())
    row['truncated'] = False
    path.write_text(json.dumps(row) + '\n')
    resign(corpus)
    with pytest.raises(ValueError, match='processed-token/EOS lineage'):
        audit_mechanism_training(task_manifest, corpus)
