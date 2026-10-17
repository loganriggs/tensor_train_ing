"""Immutable exact-label synthetic tasks with whole compositions held out.

Provides bounded Boolean hierarchy and associative retrieval benchmarks. These
mechanism probes are distinct from language-model quality and final-test claims.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest_hash(path):
    """SHA256 of a completed manifest, validating its independently stored digest."""
    path = Path(path)
    if path.is_dir():
        path /= 'manifest.json'
    digest = _sha(path)
    if path.with_name('manifest.sha256').read_text().strip() != digest:
        raise ValueError('manifest checksum mismatch')
    return digest


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


MECHANISM_VERSION = 'exact-compositions-v1'
_MECHANISM_TASKS = ('boolean_hierarchy', 'associative_retrieval')
_MECHANISM_KEYS = ('A', 'B', 'C', 'D')


def _mechanism_partitions(seed):
    import itertools
    boolean = list(itertools.product(('xor', 'and'), repeat=3))
    mappings = list(itertools.product(range(4), repeat=4))
    random.Random(f'{MECHANISM_VERSION}:mappings:{seed}').shuffle(mappings)
    return {'boolean_hierarchy': {'train': [ops for ops in boolean if ops.count('and') % 2 == 0],
                                  'val': [ops for ops in boolean if ops.count('and') % 2 == 1]},
            'associative_retrieval': {'train': mappings[:192], 'val': mappings[192:]}}


def _mechanism_row(task, split, composition, instance):
    """Canonical exact-label row constructor; neither tokenizes nor downloads."""
    if task == 'boolean_hierarchy':
        bits, operators = list(instance), list(composition)
        def apply(op, a, b):
            return a ^ b if op == 'xor' else a & b
        answer = apply(operators[2], apply(operators[0], *bits[:2]), apply(operators[1], *bits[2:]))
        composition_id = 'boolean:' + '-'.join(operators)
        prompt = f'Compute ({bits[0]} {operators[0].upper()} {bits[1]}) {operators[2].upper()} ({bits[2]} {operators[1].upper()} {bits[3]}). Answer:'
        structure = {'bits': bits, 'operators': operators}
    else:
        mapping = list(zip(_MECHANISM_KEYS, composition))
        answer = composition[_MECHANISM_KEYS.index(instance)]
        composition_id = 'retrieval:' + ''.join(map(str, composition))
        prompt = 'Mappings: ' + ', '.join(f'{key}={value}' for key, value in mapping) + f'. Query: {instance}. Answer:'
        structure = {'key_values': [list(pair) for pair in mapping], 'query': instance}
    identity = {'task': task, 'split': split, 'composition_id': composition_id, **structure}
    return {'id': task + '/' + _digest(json.dumps(identity, sort_keys=True, separators=(',', ':'))),
            **identity, 'prompt': prompt, 'answer': f' {answer}', 'text': prompt + f' {answer}'}


def _mechanism_bounds(seed, train_examples, validation_examples):
    if type(seed) is not int:
        raise ValueError('mechanism seed must be an integer')
    for name, count, maximum in (('train', train_examples, 36), ('val', validation_examples, 42)):
        if type(count) is not int or count < 8 or count > maximum or count % 2:
            raise ValueError(f'{name} examples per task must be even, >=8 and <= {maximum}; balanced unique Boolean domain bound')


def prepare_mechanism_tasks(output_dir, seed=0, train_examples=32, validation_examples=16):
    """Freeze two tiny exact-label tasks with whole compositions held out.

    Counts are PER TASK. Boolean outputs are exactly balanced; all four eligible
    triples occur in each split. Each triple uses both primitive operators across
    the training partition at every site. Retrieval holds out entire mappings,
    independent of query; keys are unique and values may repeat. Domain enumeration
    is fixed at 128 Boolean assignments and 1024 retrieval mapping/query pairs.
    These synthetic tasks are not a final test or a language-model quality proxy.
    """
    import itertools
    _mechanism_bounds(seed, train_examples, validation_examples)
    partitions = _mechanism_partitions(seed)
    rows = {'train': [], 'val': []}
    for split, count in (('train', train_examples), ('val', validation_examples)):
        generator = random.Random(f'{MECHANISM_VERSION}:boolean:{seed}:{split}')
        selected = []
        remaining = {' 0': [], ' 1': []}
        for composition in partitions['boolean_hierarchy'][split]:
            candidates = [_mechanism_row('boolean_hierarchy', split, composition, bits)
                          for bits in itertools.product(range(2), repeat=4)]
            for label in remaining:
                pool = [row for row in candidates if row['answer'] == label]
                generator.shuffle(pool)
                selected.append(pool.pop())  # coverage of every held-out triple
                remaining[label].extend(pool)
        for label in remaining:
            generator.shuffle(remaining[label])
            selected.extend(remaining[label][:count // 2 - 4])
        generator.shuffle(selected)
        rows[split].extend(selected)
        generator = random.Random(f'{MECHANISM_VERSION}:retrieval:{seed}:{split}')
        pools = {str(label): [] for label in range(4)}
        for composition in partitions['associative_retrieval'][split]:
            for key in _MECHANISM_KEYS:
                row = _mechanism_row('associative_retrieval', split, composition, key)
                pools[row['answer'].strip()].append(row)
        for pool in pools.values():
            generator.shuffle(pool)
        rows[split].extend(pools[str(i % 4)].pop() for i in range(count))
    metadata = {'format_version': 1, 'generator_version': MECHANISM_VERSION, 'seed': seed,
                'counts_are_per_task': True, 'examples_per_task': {'train': train_examples, 'val': validation_examples},
                'domain_bounds': {'boolean_assignments_total': 128, 'retrieval_mapping_query_pairs_total': 1024,
                                  'balanced_boolean_max_per_split': {'train': 36, 'val': 42}},
                'composition_splits': {task: {split: sorted({row['composition_id'] for row in rows[split] if row['task'] == task})
                                             for split in rows} for task in _MECHANISM_TASKS},
                'composition_definition': {'boolean_hierarchy': 'operator triple [left,right,root]; even AND count train, odd val; all four bits enumerated',
                                           'associative_retrieval': 'full A/B/C/D to 0/1/2/3 mapping, values may repeat; seeded permutation of256 maps, first192 train, last64 val'},
                'label_definition': {'boolean_hierarchy': 'root(left(bit0,bit1),right(bit2,bit3)); xor is exclusive-or, and is conjunction',
                                     'associative_retrieval': 'value for exact query key'},
                'tokenizer': {'name': 'gpt2', 'answer_contract': 'leading-space digit 0..3; evaluator verifies exactly one token'},
                'scope': 'exact-label synthetic mechanism probes with held-out compositions; no final-test or LM-quality claim'}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        files = {}
        for split in rows:
            path = out / f'{split}.jsonl'
            with path.open('w') as stream:
                for row in rows[split]:
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            files[path.name] = {'sha256': _sha(path), 'bytes': path.stat().st_size}
        metadata['files'] = files
        encoded = json.dumps(metadata, sort_keys=True, indent=2) + '\n'
        with (out / 'manifest.sha256').open('w') as stream:
            stream.write(_digest(encoded) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        with (out / '.manifest.json.tmp').open('w') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(out / '.manifest.json.tmp', out / 'manifest.json')
        for directory in (out, out.parent):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return out / 'manifest.json'
    except BaseException:
        shutil.rmtree(out)
        raise


def load_mechanism_tasks(manifest, split='val'):
    """Verify checksums AND semantic labels/composition isolation; return metadata, rows.

    Reads both small bounded splits to verify isolation, but returns only requested
    rows. No tokenizer, raw external data, model, or network is involved.
    """
    if split not in ('train', 'val'):
        raise ValueError('mechanism split must be train or val')
    path = Path(manifest)
    if path.is_dir():
        path /= 'manifest.json'
    manifest_hash(path)
    metadata = json.loads(path.read_text())
    if metadata.get('generator_version') != MECHANISM_VERSION or metadata.get('format_version') != 1:
        raise ValueError('unsupported mechanism generator version')
    try:
        counts = metadata['examples_per_task']
        seed = metadata['seed']
        _mechanism_bounds(seed, counts['train'], counts['val'])
        partitions = _mechanism_partitions(seed)
        declared = metadata['composition_splits']
        if set(metadata['files']) != {'train.jsonl', 'val.jsonl'} or metadata.get('counts_are_per_task') is not True:
            raise ValueError('invalid mechanism files/count semantics')
        loaded = {}
        all_ids = set()
        for name in ('train', 'val'):
            artifact = path.with_name(f'{name}.jsonl')
            info = metadata['files'][artifact.name]
            if artifact.stat().st_size != info['bytes'] or _sha(artifact) != info['sha256']:
                raise ValueError(f'mechanism artifact checksum/size mismatch: {artifact.name}')
            rows = []
            with artifact.open() as stream:
                for line in stream:
                    if len(rows) >= counts[name] * len(_MECHANISM_TASKS):
                        raise ValueError('mechanism rows exceed declared bounded counts')
                    row = json.loads(line)
                    task = row['task']
                    if task not in _MECHANISM_TASKS or row['split'] != name:
                        raise ValueError('invalid mechanism task/split')
                    if task == 'boolean_hierarchy':
                        composition, instance = tuple(row['operators']), tuple(row['bits'])
                        if len(instance) != 4 or any(type(bit) is not int or bit not in (0, 1) for bit in instance):
                            raise ValueError('invalid Boolean bits')
                    else:
                        pairs = row['key_values']
                        if len(pairs) != 4 or [pair[0] for pair in pairs] != list(_MECHANISM_KEYS):
                            raise ValueError('invalid retrieval keys')
                        composition, instance = tuple(pair[1] for pair in pairs), row['query']
                        if any(type(value) is not int or value not in range(4) for value in composition) or instance not in _MECHANISM_KEYS:
                            raise ValueError('invalid retrieval mapping/query')
                    if composition not in partitions[task][name]:
                        raise ValueError('mechanism composition violates held-out partition')
                    expected = _mechanism_row(task, name, composition, instance)
                    if row != expected or row['id'] in all_ids:
                        raise ValueError('invalid mechanism exact label/prompt/identity or duplicate row')
                    all_ids.add(row['id'])
                    rows.append(row)
            for task in _MECHANISM_TASKS:
                task_rows = [row for row in rows if row['task'] == task]
                if len(task_rows) != counts[name]:
                    raise ValueError('mechanism task count mismatch')
                compositions = sorted({row['composition_id'] for row in task_rows})
                if declared[task][name] != compositions:
                    raise ValueError('mechanism declared composition lists mismatch')
                if task == 'boolean_hierarchy':
                    if sum(row['answer'] == ' 1' for row in task_rows) * 2 != counts[name] or len(compositions) != 4:
                        raise ValueError('Boolean labels unbalanced or composition coverage missing')
            loaded[name] = rows
        for task in _MECHANISM_TASKS:
            if set(declared[task]['train']) & set(declared[task]['val']):
                raise ValueError('mechanism split composition overlap')
        return metadata, loaded[split]
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f'invalid mechanism metadata/rows: {error}') from error


def audit_mechanism_training(task_manifest, training_manifest):
    """Audit exact benchmark exposure in the DECLARED frozen train/calibration corpus.

    Verification binds the benchmark train.jsonl source SHA256 and every exposed
    document's ID, whole-text hashes, and source line. It does not prove optimizer
    visitation, token payload correctness, or full model-lifetime/pretraining
    exposure. Partial documents conservatively expose their parent composition.
    Generic corpora remain unknown; exact validation-text matches are still flagged.
    Only task JSONL text and training metadata are read, never corpus/token payloads.
    """
    task_path, training_path = Path(task_manifest), Path(training_manifest)
    if task_path.is_dir():
        task_path /= 'manifest.json'
    if training_path.is_dir():
        training_path /= 'manifest.json'
    task_metadata, train_rows = load_mechanism_tasks(task_path, 'train')
    _, validation_rows = load_mechanism_tasks(task_path, 'val')
    task_hash = manifest_hash(task_path)
    training_hash = manifest_hash(training_path)
    training = json.loads(training_path.read_text())
    expected = training.get('artifacts', {}).get('documents.jsonl')
    metadata_path = training_path.with_name('documents.jsonl')
    if not isinstance(expected, str) or _sha(metadata_path) != expected:
        raise ValueError('training document metadata checksum mismatch')
    counts = training.get('tokens')
    if training.get('format_version') != 1 or not isinstance(counts, dict) or set(counts) != {'train', 'val', 'test', 'calibration'}:
        raise ValueError('invalid training manifest format/token counts')
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError('invalid training token counts')
    if type(training.get('documents')) is not int or training['documents'] < 0:
        raise ValueError('invalid training document count')
    source = training.get('source', {})
    if not isinstance(source, dict):
        raise ValueError('invalid training source metadata')
    source_matches = source.get('kind') == 'jsonl' and source.get('sha256') == task_metadata['files']['train.jsonl']['sha256']
    train_by_id = {row['id']: (row, line_number) for line_number, row in enumerate(train_rows, 1)}
    all_rows = [('train', row) for row in train_rows] + [('val', row) for row in validation_rows]
    hash_fields = ('content_sha256', 'processed_sha256')
    hashes = {field: {} for field in hash_fields}
    row_hashes = {}
    for split, row in all_rows:
        text = row['text']
        pair = {'content_sha256': _digest(text),
                'processed_sha256': _digest(text.replace('\r\n', '\n').replace('\r', '\n'))}
        row_hashes[row['id']] = pair
        for field in hash_fields:
            hashes[field].setdefault(pair[field], []).append((split, row))
    exposed = {split: {task: set() for task in _MECHANISM_TASKS} for split in ('train', 'calibration')}
    validation_overlap = {task: set() for task in _MECHANISM_TASKS}
    validation_matches = {split: [] for split in exposed}
    document_counts = dict.fromkeys(exposed, 0)
    matched_ids = set()
    unmatched = []
    unmatched_count = truncated_count = 0
    seen_ids, seen_numeric = set(), set()
    covered = dict.fromkeys(counts, 0)
    actual_documents = 0
    with metadata_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if actual_documents >= training['documents']:
                raise ValueError('training metadata exceeds declared document count')
            try:
                doc = json.loads(line)
                doc_id, split = doc['id'], doc['split']
                numeric_id = doc['numeric_id']
                start, end = doc['token_start'], doc['token_end']
                if not isinstance(doc_id, str) or doc_id in seen_ids or split not in covered:
                    raise ValueError('invalid or duplicate document ID/split')
                if type(numeric_id) is not int or numeric_id < 0 or numeric_id in seen_numeric:
                    raise ValueError('invalid or duplicate numeric document ID')
                if type(start) is not int or type(end) is not int or start != covered[split] or not start < end <= counts[split]:
                    raise ValueError('noncontiguous or invalid document token range')
                for field in hash_fields:
                    value = doc[field]
                    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value.lower()):
                        raise ValueError(f'invalid {field}')
                    doc[field] = value.lower()
                if type(doc['truncated']) is not bool:
                    raise ValueError('invalid document truncation flag')
                length = end - start
                expected_processed_end = length if doc['truncated'] else length - 1
                expected_eos = None if doc['truncated'] else length - 1
                if (doc.get('source_token_start') != 0 or doc.get('source_token_end') != length
                        or doc.get('processed_token_start') != 0 or doc.get('processed_token_end') != expected_processed_end
                        or doc.get('synthetic_eos_token_offset') != expected_eos):
                    raise ValueError('inconsistent document processed-token/EOS lineage')
                seen_ids.add(doc_id)
                seen_numeric.add(numeric_id)
                covered[split] = end
                actual_documents += 1
                if split not in exposed:
                    continue
                document_counts[split] += 1
                truncated_count += int(doc['truncated'])
                known = train_by_id.get(doc_id)
                identity_matches = (known is not None and all(doc[field] == row_hashes[doc_id][field] for field in hash_fields)
                                    and type(doc.get('source_offset')) is int and doc['source_offset'] == known[1])
                if identity_matches:
                    matched_ids.add(doc_id)
                else:
                    unmatched_count += 1
                    if len(unmatched) < 10:
                        unmatched.append(doc_id)
                matched_validation_ids = set()
                for field in hash_fields:
                    for row_split, row in hashes[field].get(doc[field], []):
                        exposed[split][row['task']].add(row['composition_id'])
                        if row_split == 'val':
                            validation_overlap[row['task']].add(row['composition_id'])
                            if row['id'] not in matched_validation_ids and len(validation_matches[split]) < 10:
                                validation_matches[split].append({'corpus_document_id': doc_id, 'validation_row_id': row['id'],
                                                                  'hash_kind': field, 'sha256': doc[field]})
                            matched_validation_ids.add(row['id'])
            except (KeyError, TypeError, IndexError, ValueError) as error:
                raise ValueError(f'invalid training document metadata at line {line_number}: {error}') from error
    if covered != counts or actual_documents != training['documents']:
        raise ValueError('training metadata count/coverage mismatch')
    has_overlap = any(validation_overlap.values())
    total_exposed = sum(document_counts.values())
    verified = bool(source_matches and not unmatched_count and total_exposed and not has_overlap)
    reasons = []
    if not source_matches:
        reasons.append('source SHA256/kind is not precisely benchmark train.jsonl; unrecognized corpus compositions remain unknown')
    if unmatched_count:
        reasons.append('some declared train/calibration documents do not match benchmark train stable ID, both text hashes, and source line')
    if not total_exposed:
        reasons.append('declared corpus has no train/calibration documents')
    if has_overlap:
        reasons.append('exact validation-row text/composition appears in declared train/calibration metadata')
    return {'verified': verified, 'status': 'overlap' if has_overlap else ('verified' if verified else 'unknown'),
            'reasons': reasons, 'benchmark_manifest_hash': task_hash, 'training_manifest_hash': training_hash,
            'source_matches_benchmark_train_sha256': source_matches,
            'coverage': {'declared_train_rows': len(train_rows), 'train_calibration_documents': total_exposed,
                         'documents_by_split': document_counts, 'matched_declared_train_rows': len(matched_ids),
                         'fraction': len(matched_ids) / len(train_rows), 'complete': len(matched_ids) == len(train_rows),
                         'truncated_documents': truncated_count, 'unmatched_documents': unmatched_count,
                         'unmatched_document_id_samples': unmatched},
            'exposed_compositions': {split: {task: sorted(values) for task, values in task_sets.items()} for split, task_sets in exposed.items()},
            'validation_composition_overlap': {task: sorted(values) for task, values in validation_overlap.items()},
            'exact_validation_row_matches': validation_matches, 'has_validation_overlap': bool(has_overlap),
            'scope': 'declared checksum-bound train/calibration corpus metadata only; exact whole-document text equality and parent-composition upper bounds for truncated documents; not optimizer visitation, token payload proof, near-duplicate audit, or full model-lifetime/pretraining exposure'}
