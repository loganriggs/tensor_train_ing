"""Independent streaming release audit for held-out FineWeb scale-up datasets.

Only metadata/provenance and exact full-document hashes are checked. This does
not detect near duplicates or independently retokenize the source corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

VALIDATION_SHA256 = 'd93d1555f9808f7a6031fd4e6245fccecbe6e46d8e1307766a6f458502f9c4c5'
REFERENCE_SHA256 = '390b37a6c2fd0e3c51e028ef0d3719f91b690c3b72c4cbd220ed5f100e886543'
SPLITS = ('train', 'val', 'test', 'calibration')
HASH_FIELDS = ('content_sha256', 'processed_sha256')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path, expected=None):
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'pinned manifest checksum mismatch: {path}')
    return json.loads(payload), digest


def document_stream(path, digest):
    with Path(path).open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            try:
                document = json.loads(line)
                if not isinstance(document['id'], str) or document['split'] not in SPLITS:
                    raise ValueError('invalid ID/split')
                for field in HASH_FIELDS:
                    value = document[field]
                    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                        raise ValueError('invalid ' + field)
                yield document
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'{path}: invalid metadata line {line_number}: {error}') from error


def audit(manifest, reference_manifest, validation_manifest, *, minimum_train_tokens=12_500_000_000,
          expected_validation_manifest_sha256=VALIDATION_SHA256,
          expected_reference_manifest_sha256=REFERENCE_SHA256, expected_validation_documents=4496):
    started = time.monotonic()
    report = {'status': 'failed', 'manifest': str(Path(manifest).resolve()),
              'reference_manifest': str(Path(reference_manifest).resolve()),
              'validation_manifest': str(Path(validation_manifest).resolve()),
              'minimum_train_tokens': minimum_train_tokens, 'errors': [],
              'scope': 'all output splits; exact stable ID/raw/processed whole-document matches',
              'limitations': ['No near-duplicate or partial-content audit.',
                  'Tokenizer/preprocessing/deduplication policies are compared by provenance; source retokenization and all-pairs deduplication are not repeated.']}
    errors = report['errors']
    try:
        if type(minimum_train_tokens) is not int or minimum_train_tokens < 0:
            raise ValueError('minimum_train_tokens must be a nonnegative integer')
        ref, ref_sha = read_manifest(reference_manifest, expected_reference_manifest_sha256)
        validation, val_sha = read_manifest(validation_manifest, expected_validation_manifest_sha256)
        current, current_sha = read_manifest(manifest)
        report.update(manifest_sha256=current_sha, reference_manifest_sha256=ref_sha,
                      validation_manifest_sha256=val_sha)
        for field in ('format_version', 'tokenizer', 'preprocessing', 'deduplication', 'membership',
                      'hash_domains', 'split_order', 'split_fractions', 'bounded_selection',
                      'boundary_loss_policy', 'token_offset_semantics'):
            if field not in ref or current.get(field) != ref[field]:
                errors.append('reference policy mismatch: ' + field)
        for field in ('kind', 'dataset', 'config', 'revision', 'shard_hash_verification'):
            if current.get('source', {}).get(field) != ref.get('source', {}).get(field):
                errors.append('reference source mismatch: ' + field)
        exclusion = current.get('heldout_exclusion', {})
        for field in ('rule', 'manifest_sha256', 'documents_sha256', 'validation_documents'):
            if field not in ref.get('heldout_exclusion', {}) or exclusion.get(field) != ref['heldout_exclusion'][field]:
                errors.append('reference heldout exclusion mismatch: ' + field)
        expected_docs_sha = validation['artifacts']['documents.jsonl']
        if exclusion.get('manifest_sha256') != val_sha or exclusion.get('documents_sha256') != expected_docs_sha:
            errors.append('heldout exclusion does not bind fixed validation artifacts')
        heldout = {field: set() for field in ('id', *HASH_FIELDS)}
        digest = hashlib.sha256()
        validation_docs = Path(validation_manifest).with_name('documents.jsonl')
        validation_count = 0
        for doc in document_stream(validation_docs, digest):
            if doc['split'] == 'val':
                validation_count += 1
                for field in heldout:
                    heldout[field].add(doc[field])
        if digest.hexdigest() != expected_docs_sha:
            raise ValueError('fixed validation documents checksum mismatch')
        if validation_count != expected_validation_documents:
            errors.append(f'fixed validation document count {validation_count} != {expected_validation_documents}')
        report['validation_documents'] = validation_count
        counts = current.get('tokens', {})
        if set(counts) != set(SPLITS) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError('invalid token counts')
        report['tokens'] = counts
        if counts['train'] < minimum_train_tokens:
            errors.append('train token count below required minimum')
        artifacts = current.get('artifacts', {})
        required = {'documents.jsonl', *(split + '.tokens.bin' for split in SPLITS)}
        if not required.issubset(artifacts):
            raise ValueError('missing required artifacts')
        root = Path(manifest).parent
        report['artifact_checks'] = {}
        for name, expected in artifacts.items():
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('unsafe artifact name')
            if name == 'documents.jsonl':
                continue  # Hash the exact bytes being independently audited below.
            path = root / name
            actual = sha256(path)
            passed = actual == expected
            report['artifact_checks'][name] = {'sha256': actual, 'matches': passed, 'bytes': path.stat().st_size}
            if not passed:
                errors.append('artifact checksum mismatch: ' + name)
        for split in SPLITS:
            if (root / (split + '.tokens.bin')).stat().st_size != counts[split] * 4:
                errors.append('int32 token payload size mismatch: ' + split)
        overlap = {split: {field: 0 for field in heldout} for split in SPLITS}
        examples = []
        covered = dict.fromkeys(SPLITS, 0)
        document_count = 0
        digest = hashlib.sha256()
        for doc in document_stream(root / 'documents.jsonl', digest):
            split = doc['split']
            start, end = doc['token_start'], doc['token_end']
            if type(start) is not int or type(end) is not int or start != covered[split] or not start < end <= counts[split]:
                raise ValueError('document token ranges do not cover splits contiguously')
            covered[split] = end
            document_count += 1
            for field, values in heldout.items():
                if doc[field] in values:
                    overlap[split][field] += 1
                    if len(examples) < 20:
                        examples.append({'split': split, 'document_id': doc['id'], 'match': field})
        report.update(documents=document_count, overlap=overlap, overlap_examples=examples)
        actual = digest.hexdigest()
        report['artifact_checks']['documents.jsonl'] = {'sha256': actual, 'matches': actual == artifacts['documents.jsonl']}
        if actual != artifacts['documents.jsonl']:
            errors.append('artifact checksum mismatch: documents.jsonl')
        if covered != counts or document_count != current.get('documents'):
            errors.append('metadata coverage/count differs from manifest')
        if any(count for row in overlap.values() for count in row.values()):
            errors.append('fixed validation overlaps output metadata')
        # Detect concurrent mutation of pinned inputs and the release manifest.
        validation_unchanged = (sha256(validation_manifest) == val_sha
                                and sha256(validation_docs) == expected_docs_sha)
        if not validation_unchanged:
            errors.append('fixed validation changed during audit')
        if sha256(reference_manifest) != ref_sha or sha256(manifest) != current_sha:
            errors.append('reference or release manifest changed during audit')
        report['fixed_validation_unchanged'] = validation_unchanged
        report['status'] = 'passed' if not errors else 'failed'
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f'{type(error).__name__}: {error}')
    report['elapsed_seconds'] = time.monotonic() - started
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path, default=Path('data/fineweb_3b_heldout/manifest.json'))
    parser.add_argument('--validation-manifest', type=Path, default=Path('data/fineweb_confirmation_320m/manifest.json'))
    parser.add_argument('--minimum-train-tokens', type=int, default=12_500_000_000)
    parser.add_argument('--expected-validation-manifest-sha256', default=VALIDATION_SHA256)
    parser.add_argument('--expected-reference-manifest-sha256', default=REFERENCE_SHA256)
    parser.add_argument('--expected-validation-documents', type=int, default=4496)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve output exclusively before the potentially long audit.
    with args.output.open('x') as stream:
        json.dump({'status': 'running'}, stream)
    report = audit(args.manifest, args.reference_manifest, args.validation_manifest,
        minimum_train_tokens=args.minimum_train_tokens,
        expected_validation_manifest_sha256=args.expected_validation_manifest_sha256,
        expected_reference_manifest_sha256=args.expected_reference_manifest_sha256,
        expected_validation_documents=args.expected_validation_documents)
    fd, temporary = tempfile.mkstemp(dir=args.output.parent, prefix='.audit-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({'status': report['status'], 'audit': str(args.output), 'errors': report['errors']}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
