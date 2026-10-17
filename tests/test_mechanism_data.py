"""Exact-label tests independent of the mechanism generator's label implementation."""
import hashlib
import json
from collections import Counter

import pytest

from tnlm.data import load_mechanism_tasks, prepare_mechanism_tasks


def test_exact_labels_held_out_compositions_and_shared_primitives(tmp_path):
    manifest = prepare_mechanism_tasks(tmp_path / 'tasks', seed=71)
    metadata, train = load_mechanism_tasks(manifest, 'train')
    _, validation = load_mechanism_tasks(manifest, 'val')
    assert len(train) == 64 and len(validation) == 32
    assert metadata['counts_are_per_task'] is True
    assert metadata['examples_per_task'] == {'train': 32, 'val': 16}
    assert metadata['tokenizer']['name'] == 'gpt2'
    for task in ('boolean_hierarchy', 'associative_retrieval'):
        left = {row['composition_id'] for row in train if row['task'] == task}
        right = {row['composition_id'] for row in validation if row['task'] == task}
        assert left and right and left.isdisjoint(right)
        assert sorted(left) == metadata['composition_splits'][task]['train']
        assert sorted(right) == metadata['composition_splits'][task]['val']
    for rows in (train, validation):
        labels = Counter(row['answer'] for row in rows if row['task'] == 'boolean_hierarchy')
        assert labels[' 0'] == labels[' 1']
        for row in rows:
            assert row['prompt'].endswith('Answer:')
            assert row['text'] == row['prompt'] + row['answer']
            assert len(row['answer']) == 2 and row['answer'][0] == ' ' and row['answer'][1] in '0123'
            if row['task'] == 'boolean_hierarchy':
                bits, ops = row['bits'], row['operators']
                # Independent truth tables, not calls into production helpers.
                truth = {'and': {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1},
                         'xor': {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}}
                left = truth[ops[0]][tuple(bits[:2])]
                right = truth[ops[1]][tuple(bits[2:])]
                answer = truth[ops[2]][left, right]
                assert row['answer'] == f' {answer}'
            else:
                mapping = dict(row['key_values'])
                assert len(mapping) == 4 and set(mapping) == set('ABCD')
                assert set(mapping.values()) <= set(range(4))
                assert row['answer'] == f" {mapping[row['query']]}"
    boolean_train = [row for row in train if row['task'] == 'boolean_hierarchy']
    for site in range(3):
        assert {row['operators'][site] for row in boolean_train} == {'xor', 'and'}
    assert len({row['id'] for row in train + validation}) == len(train + validation)
    assert 'no final-test' in metadata['scope'] and 'LM-quality' in metadata['scope']


def test_deterministic_immutable_hashes_and_maximum_balanced_bounds(tmp_path):
    first = prepare_mechanism_tasks(tmp_path / 'first', seed=19, train_examples=36, validation_examples=42)
    second = prepare_mechanism_tasks(tmp_path / 'second', seed=19, train_examples=36, validation_examples=42)
    assert first.read_bytes() == second.read_bytes()
    metadata, train = load_mechanism_tasks(first, 'train')
    _, validation = load_mechanism_tasks(first, 'val')
    assert len(train) == 72 and len(validation) == 84
    for name, expected in metadata['files'].items():
        artifact = first.with_name(name)
        assert artifact.read_bytes() == second.with_name(name).read_bytes()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected['sha256']
        assert artifact.stat().st_size == expected['bytes']
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_mechanism_tasks(first.parent)
    assert first.read_bytes() == before


@pytest.mark.parametrize('kwargs', [{'train_examples': 0}, {'train_examples': 6}, {'train_examples': 9},
                                    {'train_examples': 38}, {'validation_examples': 44},
                                    {'validation_examples': True}, {'seed': 'not-an-integer'}])
def test_invalid_or_impossible_unique_bounds_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        prepare_mechanism_tasks(tmp_path / 'invalid', **kwargs)
    assert not (tmp_path / 'invalid').exists()


def resign(manifest):
    metadata = json.loads(manifest.read_text())
    for name in metadata['files']:
        artifact = manifest.with_name(name)
        metadata['files'][name] = {'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                   'bytes': artifact.stat().st_size}
    manifest.write_text(json.dumps(metadata))
    manifest.with_name('manifest.sha256').write_text(hashlib.sha256(manifest.read_bytes()).hexdigest())


def test_corrupt_artifacts_and_manifest_rejected(tmp_path):
    manifest = prepare_mechanism_tasks(tmp_path / 'tasks')
    artifact = manifest.with_name('val.jsonl')
    original = artifact.read_bytes()
    artifact.write_bytes(original + b'corrupt')
    with pytest.raises(ValueError, match='checksum/size'):
        load_mechanism_tasks(manifest)
    artifact.write_bytes(original)
    manifest.write_text(manifest.read_text() + ' ')
    with pytest.raises(ValueError, match='manifest checksum'):
        load_mechanism_tasks(manifest)


@pytest.mark.parametrize('tamper', ['answer', 'composition', 'prompt', 'duplicate', 'declared_compositions'])
def test_semantic_tampering_rejected_even_with_updated_checksums(tmp_path, tamper):
    manifest = prepare_mechanism_tasks(tmp_path / 'tasks')
    artifact = manifest.with_name('val.jsonl')
    rows = [json.loads(line) for line in artifact.read_text().splitlines()]
    if tamper == 'answer':
        rows[0]['answer'] = ' 9'
    elif tamper == 'composition':
        operators = rows[0]['operators']
        operators[0] = 'and' if operators[0] == 'xor' else 'xor'
    elif tamper == 'prompt':
        rows[0]['prompt'] = 'Wrong problem. Answer:'
    elif tamper == 'duplicate':
        rows[1] = rows[0]
    else:
        metadata = json.loads(manifest.read_text())
        metadata['composition_splits']['boolean_hierarchy']['val'] = []
        manifest.write_text(json.dumps(metadata))
    artifact.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    resign(manifest)
    with pytest.raises(ValueError, match='mechanism'):
        load_mechanism_tasks(manifest)


def test_atomic_publication_failure_cleans_output(tmp_path, monkeypatch):
    import tnlm.data as data
    def fail(*args):
        raise OSError('publication failure')
    monkeypatch.setattr(data.os, 'replace', fail)
    with pytest.raises(OSError, match='publication failure'):
        prepare_mechanism_tasks(tmp_path / 'failed')
    assert not (tmp_path / 'failed').exists()
