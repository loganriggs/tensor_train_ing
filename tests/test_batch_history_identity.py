import copy

import pytest

from tnlm.evaluate import _training_comparison, _training_descriptor


def payload():
    return {'config': {'train': {'global_prediction_tokens_per_update': 32, 'lr': .02,
                                'aux_adamw_lr': None}},
            'progress': {'tokens': 128, 'steps': 10, 'batch_history': [
                {'batch_tokens': 8, 'lr': .01, 'aux_adamw_lr': None,
                 'start_tokens': 0, 'start_step': 0, 'source': 'fresh'},
                {'batch_tokens': 32, 'lr': .02, 'aux_adamw_lr': None,
                 'start_tokens': 64, 'start_step': 8, 'source': 'batch_branch',
                 'lr_scale': 2., 'parent': {'path': '/some/parent'},
                 'source_sampler': {'cursor': 123}, 'optimizer_groups_before': [{}],
                 'optimizer_groups_after': [{}]}]}}


def comparison(left, right):
    return _training_comparison({'training_descriptor': _training_descriptor(left)},
                                {'training_descriptor': _training_descriptor(right)})['dimensions']


def test_semantic_history_keeps_intervention_and_strips_artifact_details():
    original = payload()
    before = copy.deepcopy(original)
    descriptor = _training_descriptor(original)
    assert descriptor['batch_history_status'] == 'complete'
    assert descriptor['batch_history'][0]['lr_scale'] == 1
    assert descriptor['batch_history'][1] == {
        'batch_tokens': 32, 'lr': .02, 'aux_adamw_lr': None,
        'start_tokens': 64, 'start_step': 8, 'source': 'batch_branch', 'lr_scale': 2.}
    other = copy.deepcopy(original)
    other['progress']['batch_history'][1]['parent']['path'] = '/another/parent'
    other['progress']['batch_history'][1]['source_sampler']['cursor'] = 999
    assert comparison(original, other)['batch_history']['status'] == 'matched'
    assert original == before


def test_equal_current_policy_does_not_hide_different_training_history():
    left, right = payload(), payload()
    right['progress']['batch_history'][1]['start_tokens'] = 72
    dimensions = comparison(left, right)
    assert dimensions['global_prediction_tokens_per_update']['status'] == 'matched'
    assert dimensions['lr']['status'] == 'matched'
    assert dimensions['batch_history']['status'] == 'different'


def test_missing_and_partial_histories_are_not_assumed_equal():
    missing = payload()
    del missing['progress']['batch_history']
    assert _training_descriptor(missing)['batch_history_status'] == 'unknown'
    assert comparison(missing, missing)['batch_history']['status'] == 'unknown'
    partial = payload()
    partial['progress']['batch_history'][0].update(
        source='legacy_unknown', start_tokens=None, start_step=None)
    descriptor = _training_descriptor(partial)
    assert descriptor['batch_history_status'] == 'partial'
    assert descriptor['batch_history'][0]['start_tokens'] is None
    assert comparison(partial, partial)['batch_history']['status'] == 'unknown'


@pytest.mark.parametrize('entry,field,value', [
    (0, 'source', 'batch_branch'), (0, 'start_tokens', 1), (0, 'lr_scale', 2),
    (0, 'batch_tokens', True), (0, 'lr', float('nan')), (0, 'aux_adamw_lr', False),
    (1, 'source', 'fresh'), (1, 'start_tokens', -1), (1, 'start_tokens', 129),
    (1, 'start_step', 11), (1, 'start_step', True), (1, 'lr_scale', 0),
    (1, 'lr_scale', float('inf')), (1, 'batch_tokens', 16), (1, 'lr', .03),
    (1, 'aux_adamw_lr', .02),
])
def test_malformed_or_endpoint_mismatched_history_is_unknown(entry, field, value):
    data = payload()
    data['progress']['batch_history'][entry][field] = value
    descriptor = _training_descriptor(data)
    assert descriptor['batch_history'] is None
    assert descriptor['batch_history_status'] == 'unknown'


def test_auxiliary_lr_is_recorded_and_validated_separately():
    data = payload()
    data['config']['train']['aux_adamw_lr'] = .004
    data['progress']['batch_history'][0]['aux_adamw_lr'] = .002
    data['progress']['batch_history'][1]['aux_adamw_lr'] = .004
    assert _training_descriptor(data)['batch_history_status'] == 'complete'


def test_later_branch_cannot_go_backwards():
    data = payload()
    third = copy.deepcopy(data['progress']['batch_history'][1])
    third.update(start_tokens=63, start_step=7, lr_scale=1)
    data['progress']['batch_history'].append(third)
    assert _training_descriptor(data)['batch_history_status'] == 'unknown'


@pytest.mark.parametrize('field', ['lr_scale', 'aux_adamw_lr'])
def test_branch_does_not_infer_missing_policy_evidence(field):
    data = payload()
    del data['progress']['batch_history'][1][field]
    assert _training_descriptor(data)['batch_history_status'] == 'unknown'
