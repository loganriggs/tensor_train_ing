"""Held-out exclusion authenticates references and blocks cross-ID leakage."""
import json

import pytest

from tnlm.data import SPLIT_ORDER, prepare_data


def _source(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def _prepare(tmp_path, name, rows, **kwargs):
    return prepare_data(tmp_path / name,
                        source_jsonl=_source(tmp_path / f'{name}.jsonl', rows),
                        revision='heldout-fixture-v1', **kwargs)


def _reference(tmp_path):
    return _prepare(tmp_path, 'reference', [
        {'id': 'validation-id', 'text': 'private validation wording'},
        {'id': 'raw-reference', 'text': 'same raw bytes'},
        {'id': 'normalized-reference', 'text': 'normalized\r\nvalidation\rtext'},
    ], split_fractions={split: float(split == 'val') for split in SPLIT_ORDER})


def _rows():
    return [
        {'id': 'validation-id', 'text': 'same ID with changed text must be excluded'},
        {'id': 'different-raw-id', 'text': 'same raw bytes'},
        {'id': 'different-normalized-id', 'text': 'normalized\nvalidation\ntext'},
        {'id': 'unrelated-one', 'text': 'independent training document'},
        {'id': 'unrelated-two', 'text': 'another independent document'},
    ]


def _docs(manifest):
    return [json.loads(line) for line in manifest.with_name('documents.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('target_split', SPLIT_ORDER)
def test_heldout_id_raw_and_processed_matches_excluded_from_every_split(tmp_path, target_split):
    reference = _reference(tmp_path)
    manifest = _prepare(tmp_path, 'filtered', _rows(),
                        exclude_validation_manifest=reference,
                        split_fractions={split: float(split == target_split) for split in SPLIT_ORDER})
    docs = _docs(manifest)
    assert {doc['id'] for doc in docs} == {'unrelated-one', 'unrelated-two'}
    assert {doc['split'] for doc in docs} == {target_split}
    metadata = json.loads(manifest.read_text())
    exclusion = metadata['heldout_exclusion']
    assert exclusion['validation_documents'] == 3
    assert exclusion['excluded_unique_documents'] == 3
    assert exclusion['manifest_sha256'] == reference.with_name('manifest.sha256').read_text().strip()
    assert exclusion['documents_sha256'] == json.loads(reference.read_text())['artifacts']['documents.jsonl']
    assert metadata['scan_records'] == metadata['scan_unique_documents'] == 5
    assert not list(manifest.parent.glob('*.sqlite*'))


def test_heldout_filter_serial_parallel_are_identical(tmp_path):
    reference = _reference(tmp_path)
    source = _source(tmp_path / 'source.jsonl', _rows() + [_rows()[1]])
    common = dict(source_jsonl=source, revision='v1', exclude_validation_manifest=reference,
                  tokenization_batch_bytes=1)
    serial = prepare_data(tmp_path / 'serial', workers=1, **common)
    parallel = prepare_data(tmp_path / 'parallel', workers=2, **common)
    assert {p.relative_to(serial.parent): p.read_bytes() for p in serial.parent.rglob('*') if p.is_file()} == {
        p.relative_to(parallel.parent): p.read_bytes() for p in parallel.parent.rglob('*') if p.is_file()}
    assert json.loads(parallel.read_text())['heldout_exclusion']['excluded_unique_documents'] == 3


def test_excluded_duplicate_id_conflicts_still_reject(tmp_path):
    reference = _reference(tmp_path)
    rows = _rows() + [{'id': 'validation-id', 'text': 'conflicting duplicate source text'}]
    with pytest.raises(ValueError, match='conflicting content'):
        _prepare(tmp_path, 'failed', rows, exclude_validation_manifest=reference,
                 max_tokens=1, max_documents=1)
    assert not (tmp_path / 'failed').exists()


@pytest.mark.parametrize('artifact', ['manifest.json', 'documents.jsonl'])
def test_heldout_reference_tampering_rejected_and_output_cleaned(tmp_path, artifact):
    reference = _reference(tmp_path)
    path = reference.with_name(artifact)
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='checksum|hash'):
        _prepare(tmp_path, 'failed', _rows(), exclude_validation_manifest=reference)
    assert not (tmp_path / 'failed').exists()


def test_reference_without_validation_documents_rejected(tmp_path):
    reference = _prepare(tmp_path, 'reference', [{'id': 'train-only', 'text': 'hello'}],
                         split_fractions={split: float(split == 'train') for split in SPLIT_ORDER})
    with pytest.raises(ValueError, match='val|held.out'):
        _prepare(tmp_path, 'failed', _rows(), exclude_validation_manifest=reference)
    assert not (tmp_path / 'failed').exists()


def test_reference_training_documents_are_not_excluded(tmp_path):
    reference = _prepare(tmp_path, 'reference',
                         [{'id': str(i), 'text': f'unique reference text {i}'} for i in range(30)],
                         split_fractions={'train': .5, 'val': .5, 'test': 0., 'calibration': 0.})
    docs = _docs(reference)
    train = next(doc for doc in docs if doc['split'] == 'train')
    val = next(doc for doc in docs if doc['split'] == 'val')
    rows = [{'id': doc['id'], 'text': f"unique reference text {doc['id']}"} for doc in [train, val]]
    filtered = _prepare(tmp_path, 'filtered', rows, exclude_validation_manifest=reference)
    assert [doc['id'] for doc in _docs(filtered)] == [train['id']]


def test_default_preparation_omits_exclusion_provenance(tmp_path):
    manifest = _prepare(tmp_path, 'ordinary', _rows())
    assert 'heldout_exclusion' not in json.loads(manifest.read_text())
    assert len(_docs(manifest)) == len(_rows())
