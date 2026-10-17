"""Context forks conserve independently enumerated consumed-stream target IDs."""
import copy
import json

import pytest
import torch

from helpers import FixtureTokenizer
from tnlm.data import PackedSampler, RankPackedSampler, batch_hash, prepare_data


def manifest(tmp_path, lengths):
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps({'id': f'doc{i}', 'text': 'a' * (n - 1)}) + '\n' for i, n in enumerate(lengths)))
    return prepare_data(tmp_path / 'data', source_jsonl=source, revision='context-fixture', tokenizer=FixtureTokenizer(),
        split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})


def identities(sampler, batch):
    result = []
    for b, row in enumerate(sampler.batch_mappings()):
        for item in row['ranges']:
            for j in range(item['packed_token_start'], min(sampler.context, item['packed_token_end'] - 1)):
                if batch[2][b, j] != -100:
                    result.append((row['epoch'], item['document_id'], item['source_token_start'] + j - item['packed_token_start'] + 1))
    return result


def expected_remaining(sampler):
    state = sampler.state_dict()
    source = state.get('context_transition', {}).get('source_state', state)
    epoch, d, t = source['epoch'], source['document_cursor'], source['token_cursor']
    order = sampler._permutation(epoch)
    if d == len(order):
        epoch, d, t = epoch + 1, 0, 0
        order = sampler._permutation(epoch)
    expected = []
    for position in range(d, len(order)):
        doc = sampler.docs[order[position]]
        length = doc['token_end'] - doc['token_start']
        first = t + 1 if position == d else 1
        expected.extend((epoch, doc['id'], offset) for offset in range(first, length))
    return epoch, expected


@pytest.mark.parametrize('old,new', [(1, 7), (7, 1), (2, 3), (3, 2), (4, 9), (9, 4)])
def test_every_old_boundary_exact_independent_remaining_targets(tmp_path, old, new):
    path = manifest(tmp_path, [1, 2, 3, 5, 9, 17, 2, 1])
    source = PackedSampler(path, context=old, seed=19)
    for boundary in range(source.epoch_row_count + 1):
        before = copy.deepcopy(source.state_dict())
        before_mappings = source.batch_mappings()
        epoch, expected = expected_remaining(source)
        fork = source.fork_context(new)
        assert source.state_dict() == before and source.batch_mappings() == before_mappings
        for i in source._permutation(epoch + 1):
            doc = source.docs[i]
            expected.extend((epoch + 1, doc['id'], offset) for offset in range(1, doc['token_end'] - doc['token_start']))
        actual = []
        while len(actual) < len(expected):
            actual.extend(identities(fork, fork.next_batch()))
        assert actual[:len(expected)] == expected, (old, new, boundary, before, actual, expected)
        assert len(actual) == len(set(actual))
        source.next_batch()


def test_pending_transition_repeated_forks_and_exact_state_replay(tmp_path):
    path = manifest(tmp_path, [31])
    source = PackedSampler(path, context=3)
    source.next_batch()
    fork = source.fork_context(7)
    assert 'context_transition' in fork.state_dict()
    repeated = fork.fork_context(2).fork_context(9)
    epoch, expected = expected_remaining(source)
    restored = PackedSampler(path, context=9)
    restored.load_state_dict(repeated.state_dict())
    found = []
    while repeated.cursor < repeated.epoch_row_count:
        a, b = repeated.next_batch(), restored.next_batch()
        assert batch_hash(a) == batch_hash(b)
        assert repeated.state_dict() == restored.state_dict()
        found.extend(identities(repeated, a))
    assert found == expected
    assert 'context_transition' not in repeated.state_dict()


def test_replayed_prefix_inputs_are_retained_but_targets_masked(tmp_path):
    path = manifest(tmp_path, [31])
    source = PackedSampler(path, context=3)
    source.next_batch()
    fork = source.fork_context(7)
    batch = fork.next_batch()
    assert fork.batch_mappings()[0]['ranges'][0]['source_token_start'] == 0
    assert batch[0].shape == (1, 7)
    assert batch[2][0, :3].tolist() == [-100] * 3
    assert identities(fork, batch)[0][2] == 4


def test_reject_transition_before_mutation_and_no_row_walk(tmp_path, monkeypatch):
    path = manifest(tmp_path, [10001, 2])
    source = PackedSampler(path, context=1)
    for _ in range(2003):
        source.next_batch()
    def forbidden(*args):
        raise AssertionError('fork must not iterate packed rows')
    monkeypatch.setattr(PackedSampler, '_next_row', forbidden)
    fork = source.fork_context(7)
    before = fork.state_dict()
    for mutate in [lambda s: s['context_transition'].update(version=True),
                   lambda s: s['context_transition'].update(source_context=9),
                   lambda s: s['context_transition']['source_state'].update(seed=999),
                   lambda s: s['context_transition']['source_state'].update(cursor=100),
                   lambda s: s.update(cursor=1)]:
        state = copy.deepcopy(before)
        mutate(state)
        with pytest.raises(ValueError):
            fork.load_state_dict(state)
        assert fork.state_dict() == before
    for context in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            source.fork_context(context)


def test_rank_fork_global_before_slice_and_skipped_transition(tmp_path):
    path = manifest(tmp_path, [3, 29, 2, 17, 1])
    ranks = [RankPackedSampler(path, rank=r, world_size=3, context=2, batch_size=2, seed=5) for r in range(3)]
    for sampler in ranks:
        sampler.next_batch()
    serial = PackedSampler(path, context=2, batch_size=6, seed=5)
    serial.next_batch()
    forked = [sampler.fork_context(7) for sampler in ranks]
    reference = serial.fork_context(7)
    clones = [RankPackedSampler(path, rank=r, world_size=3, context=7, batch_size=2, seed=5) for r in range(3)]
    for clone, fork in zip(clones, forked):
        clone.load_state_dict(fork.state_dict())
    epoch, independently_expected = expected_remaining(serial)
    for future_epoch in range(epoch + 1, epoch + 6):
        for i in serial._permutation(future_epoch):
            doc = serial.docs[i]
            independently_expected.extend((future_epoch, doc['id'], offset) for offset in range(1, doc['token_end'] - doc['token_start']))
    observed = []
    for _ in range(4):
        expected = reference.next_batch()
        batches = [sampler.next_batch() for sampler in forked]
        for clone, sampler, batch in zip(clones, forked, batches):
            assert batch_hash(clone.next_batch()) == batch_hash(batch)
            assert clone.state_dict() == sampler.state_dict()
            observed.extend(identities(sampler, batch))
        for index in range(3):
            assert torch.equal(torch.cat([batch[index] for batch in batches]), expected[index])
        streams = [sampler.state_dict()['stream'] for sampler in forked]
        assert streams[0] == streams[1] == streams[2]
        assert streams[0]['cursor'] == reference.cursor

    assert observed == independently_expected[:len(observed)]


def test_final_singleton_pending_mask_never_crosses_epoch(tmp_path):
    path = manifest(tmp_path, [2, 1])
    for seed in range(100):
        source = PackedSampler(path, context=1, seed=seed)
        lengths = [source.docs[i]['token_end'] - source.docs[i]['token_start'] for i in source.order]
        if lengths == [2, 1]:
            break
    else:
        raise AssertionError('fixture permutation unavailable')
    source.next_batch()
    assert source.document_cursor == 1 and source.token_cursor == 0
    fork = source.fork_context(2)
    assert fork.cursor == fork.epoch_row_count
    assert 'context_transition' in fork.state_dict()
    repeated = fork.fork_context(3)
    for sampler in (fork, repeated):
        actual = []
        while len(actual) < 2:
            actual.extend(identities(sampler, sampler.next_batch()))
        expected = []
        for epoch in (1, 2):
            for i in source._permutation(epoch):
                doc = source.docs[i]
                expected.extend((epoch, doc['id'], offset) for offset in range(1, doc['token_end'] - doc['token_start']))
        assert actual == expected
