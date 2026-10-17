"""Rank streams reproduce the serial global batch, including epoch rollover."""
import copy

import pytest
import torch

from tnlm.data import PackedSampler, RankPackedSampler, batch_hash


@pytest.mark.parametrize('world_size,local_batch,context', [(1, 2, 4), (2, 1, 4), (3, 2, 7), (4, 3, 128)])
def test_rank_concatenation_matches_serial_stream(config, world_size, local_batch, context):
    manifest = config.raw['data']['manifest']
    serial = PackedSampler(manifest, batch_size=world_size * local_batch, context=context, seed=17)
    ranks = [RankPackedSampler(manifest, rank=rank, world_size=world_size,
                              batch_size=local_batch, context=context, seed=17)
             for rank in range(world_size)]
    for step in range(8):
        expected = serial.next_batch()
        actual = [next(sampler) for sampler in ranks]
        for field, wanted in enumerate(expected):
            assert torch.equal(torch.cat([batch[field] for batch in actual]), wanted)
        assert [mapping for sampler in ranks for mapping in sampler.batch_mappings()] == serial.batch_mappings()
        for sampler in ranks:
            stream = sampler.state_dict()['stream']
            reference = serial.state_dict()
            assert {k: v for k, v in stream.items() if k != 'batch_size'} == {
                k: v for k, v in reference.items() if k != 'batch_size'}
    for sampler in ranks:
        state = copy.deepcopy(sampler.state_dict())
        expected = [batch_hash(sampler.next_batch()) for _ in range(3)]
        restored = RankPackedSampler(manifest, rank=sampler.rank, world_size=world_size,
                                     batch_size=local_batch, context=context, seed=17)
        restored.load_state_dict(state)
        assert [batch_hash(next(restored)) for _ in range(3)] == expected


@pytest.mark.parametrize('field,value', [('rank', 1), ('world_size', 3), ('rank_packing_version', 2)])
def test_rank_restore_identity_rejected_before_mutation(config, field, value):
    sampler = RankPackedSampler(config.raw['data']['manifest'], rank=0, world_size=2, context=4)
    next(sampler)
    before, mappings = sampler.state_dict(), sampler.batch_mappings()
    invalid = copy.deepcopy(before)
    invalid[field] = value
    with pytest.raises(ValueError, match='rank sampler state mismatch'):
        sampler.load_state_dict(invalid)
    assert sampler.state_dict() == before and sampler.batch_mappings() == mappings


@pytest.mark.parametrize('rank,world', [(0, 0), (-1, 2), (2, 2), (True, 2), (0, 1.5)])
def test_rank_identity_validated_before_loading(rank, world):
    with pytest.raises(ValueError, match='rank/world_size'):
        RankPackedSampler('missing', rank=rank, world_size=world)
