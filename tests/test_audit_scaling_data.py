"""Independent release gate checks metadata and int32 payloads without Torch."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('audit_scaling_data', Path(__file__).parents[1] / 'scripts/audit_scaling_data.py')
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def make_dataset(root, prefix):
    root.mkdir()
    docs = [{'id': prefix + split, 'split': split, 'token_start': 0, 'token_end': 2,
             'content_sha256': digest(prefix + split + 'raw'),
             'processed_sha256': digest(prefix + split + 'processed')} for split in audit_module.SPLITS]
    metadata = root / 'documents.jsonl'
    metadata.write_text(''.join(json.dumps(doc) + '\n' for doc in docs))
    artifacts = {'documents.jsonl': audit_module.sha256(metadata)}
    for split in audit_module.SPLITS:
        path = root / (split + '.tokens.bin')
        path.write_bytes(bytes(8))
        artifacts[path.name] = audit_module.sha256(path)
    manifest = {'artifacts': artifacts, 'tokens': dict.fromkeys(audit_module.SPLITS, 2), 'documents': 4,
                'format_version': 1, 'tokenizer': {'name': 'test'}, 'source': {
                    'kind': 'huggingface', 'dataset': 'FineWeb', 'config': None, 'revision': 'fixed',
                    'shard_hash_verification': 'verified'},
                'split_order': list(audit_module.SPLITS), 'split_fractions': dict.fromkeys(audit_module.SPLITS, .25)}
    for key in ('preprocessing', 'deduplication', 'membership', 'hash_domains', 'bounded_selection',
                'boundary_loss_policy', 'token_offset_semantics'):
        manifest[key] = 'same-policy'
    write_json(root / 'manifest.json', manifest)
    return manifest, docs


@pytest.fixture
def fixture(tmp_path):
    validation, val_docs = make_dataset(tmp_path / 'validation', 'heldout-')
    current, docs = make_dataset(tmp_path / 'release', 'new-')
    vp = tmp_path / 'validation/manifest.json'
    current['heldout_exclusion'] = {'rule': 'same-exclusion',
        'manifest_sha256': audit_module.sha256(vp),
        'documents_sha256': validation['artifacts']['documents.jsonl'], 'validation_documents': 1}
    rp = tmp_path / 'reference.json'
    write_json(rp, current)
    cp = tmp_path / 'release/manifest.json'
    write_json(cp, current)
    kwargs = {'minimum_train_tokens': 2, 'expected_validation_documents': 1,
              'expected_validation_manifest_sha256': audit_module.sha256(vp),
              'expected_reference_manifest_sha256': audit_module.sha256(rp)}
    return cp, rp, vp, kwargs, current, docs, val_docs


def run(fixture):
    cp, rp, vp, kwargs, *_ = fixture
    return audit_module.audit(cp, rp, vp, **kwargs)


def test_release_passes_all_artifacts_and_fixed_validation(fixture):
    report = run(fixture)
    assert report['status'] == 'passed', report
    assert report['validation_documents'] == 1
    assert report['fixed_validation_unchanged']
    assert len(report['artifact_checks']) == 5
    assert not any(n for split in report['overlap'].values() for n in split.values())


@pytest.mark.parametrize('split', audit_module.SPLITS)
@pytest.mark.parametrize('field', ['id', 'content_sha256', 'processed_sha256'])
def test_independent_overlap_in_every_split_even_with_valid_checksums(fixture, split, field):
    cp, _, _, _, manifest, docs, val_docs = fixture
    heldout = next(doc for doc in val_docs if doc['split'] == 'val')
    next(doc for doc in docs if doc['split'] == split)[field] = heldout[field]
    metadata = cp.with_name('documents.jsonl')
    metadata.write_text(''.join(json.dumps(doc) + '\n' for doc in docs))
    manifest['artifacts']['documents.jsonl'] = audit_module.sha256(metadata)
    write_json(cp, manifest)
    report = run(fixture)
    assert report['status'] == 'failed'
    assert report['overlap'][split][field] == 1


@pytest.mark.parametrize('field', ['tokenizer', 'preprocessing', 'deduplication', 'revision', 'heldout'])
def test_policy_or_provenance_drift_rejects(fixture, field):
    cp, _, _, _, manifest, *_ = fixture
    if field == 'revision':
        manifest['source']['revision'] = 'changed'
    elif field == 'heldout':
        manifest['heldout_exclusion']['manifest_sha256'] = '0' * 64
    else:
        manifest[field] = 'changed'
    write_json(cp, manifest)
    report = run(fixture)
    assert report['status'] == 'failed' and report['errors']


def test_token_payload_requires_int32_size_even_when_checksum_matches(fixture):
    cp, _, _, _, manifest, *_ = fixture
    payload = cp.with_name('train.tokens.bin')
    payload.write_bytes(bytes(4))
    manifest['artifacts']['train.tokens.bin'] = audit_module.sha256(payload)
    write_json(cp, manifest)
    assert 'int32 token payload size mismatch: train' in run(fixture)['errors']


def test_artifact_corruption_and_train_minimum_fail(fixture):
    cp, _, _, kwargs, *_ = fixture
    cp.with_name('train.tokens.bin').write_bytes(b'x' * 8)
    kwargs['minimum_train_tokens'] = 3
    errors = run(fixture)['errors']
    assert 'artifact checksum mismatch: train.tokens.bin' in errors
    assert 'train token count below required minimum' in errors


@pytest.mark.parametrize('which', ['manifest', 'documents'])
def test_validation_mutation_rejects(fixture, which):
    vp = fixture[2]
    path = vp if which == 'manifest' else vp.with_name('documents.jsonl')
    path.write_bytes(path.read_bytes() + b' ')
    assert run(fixture)['status'] == 'failed'


def test_cli_failure_report_and_exclusive_output(fixture, tmp_path):
    cp, rp, vp, kwargs, *_ = fixture
    out = tmp_path / 'audit.json'
    argv = ['--manifest', str(cp), '--reference-manifest', str(rp), '--validation-manifest', str(vp),
            '--minimum-train-tokens', '999', '--expected-validation-documents', '1',
            '--expected-validation-manifest-sha256', kwargs['expected_validation_manifest_sha256'],
            '--expected-reference-manifest-sha256', kwargs['expected_reference_manifest_sha256'], '--output', str(out)]
    assert audit_module.main(argv) == 1
    assert json.loads(out.read_text())['status'] == 'failed'
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        audit_module.main(argv)
    assert out.read_bytes() == before
