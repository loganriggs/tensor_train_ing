"""Batch comparison uses independently specified unequal-size document losses."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tnlm.comparison_batch import compare_evaluation_pairs
from tnlm.evaluate import compare_evaluations


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def report(losses):
    names, counts = ['a', 'b', 'c'], [1, 3, 2]
    plan = {'split': 'val', 'validation_manifest_sha256': 'a' * 64,
            'ranges': [{'document_id': name, 'target_start': 1, 'target_end': count + 1, 'count': count}
                       for name, count in zip(names, counts)]}
    return {'kind': 'held_out_validation', 'split': 'val', 'evaluation_protocol_version': 2,
            'frozen_state': {'unchanged': True, 'model_before': 'b' * 64, 'model_after': 'b' * 64,
                             'buffers_before': 'c' * 64, 'buffers_after': 'c' * 64},
            'model_sha256': 'b' * 64, 'context': 4, 'evaluation_seed': 0,
            'training_manifest_sha256': 'd' * 64, 'validation_manifest_sha256': 'a' * 64,
            'available_prefix_plan_sha256': 'e' * 64, 'target_plan': plan, 'target_plan_sha256': digest(plan),
            'data_overlap_audit': {'has_overlap': False, 'training_manifest_hash': 'd' * 64,
                'validation_manifest_hash': 'a' * 64, 'selected_validation_documents': 3,
                'selected_validation_ids_sha256': hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest()},
            'per_document': [{'document_id': name, 'tokens': count, 'loss_sum': loss}
                             for name, count, loss in zip(names, counts, losses)],
            'prediction_tokens': sum(counts), 'loss_sum': sum(losses), 'ce': sum(losses) / sum(counts)}


@pytest.fixture
def inputs(tmp_path):
    result = []
    for name, losses in [('left', [1., 6., 6.]), ('right', [2., 3., 10.]), ('third', [3., 9., 6.])]:
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(report(losses)))
        result.append(path)
    return result


def test_unequal_document_point_delta_authoritative_ci_order_and_hashes(inputs, tmp_path):
    left, right, third = inputs
    pairs = [{'name': '../harmless label/with spaces', 'left': left, 'right': right},
             {'name': '/absolute-looking-label', 'left': third, 'right': left}]
    before = copy.deepcopy(pairs)
    raw = {str(path): path.read_bytes() for path in inputs}
    result = compare_evaluation_pairs(pairs, tmp_path / 'batch', bootstrap_samples=123, seed=11)
    assert pairs == before
    assert [row['name'] for row in result['pairs']] == [row['name'] for row in pairs]
    # Weighted by six target tokens, not an unweighted average of document CEs.
    assert result['pairs'][0]['result']['delta_ce_right_minus_left'] == pytest.approx((1 - 3 + 4) / 6)
    for row, pair in zip(result['pairs'], pairs):
        expected = compare_evaluations(json.loads(raw[str(pair['left'])]), json.loads(raw[str(pair['right'])]),
                                       None, bootstrap_samples=123, seed=11)
        assert row['result'] == expected
    for path, payload in raw.items():
        assert result['inputs'][path] == {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
        assert Path(path).read_bytes() == payload
    assert json.loads((tmp_path / 'batch/comparisons.json').read_text()) == result
    assert {path.name for path in (tmp_path / 'batch').iterdir()} == {'comparisons.json'}
    assert not result['automatic_ranking_eligible']


def test_each_resolved_input_read_once_even_with_alias_and_reuse(inputs, tmp_path, monkeypatch):
    left, right, _ = inputs
    alias = tmp_path / 'alias.json'
    alias.symlink_to(left)
    original = Path.read_bytes
    reads = {}
    def counted(path):
        key = str(path.resolve())
        reads[key] = reads.get(key, 0) + 1
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', counted)
    result = compare_evaluation_pairs([
        {'name': 'forward', 'left': left, 'right': right},
        {'name': 'reverse', 'left': right, 'right': alias},
        {'name': 'self', 'left': alias, 'right': left}], tmp_path / 'batch', bootstrap_samples=0)
    assert reads == {str(left): 1, str(right): 1}
    assert result['pairs'][2]['result']['delta_ce_right_minus_left'] == 0.


@pytest.mark.parametrize('defect', ['late-schema', 'late-nonmatching', 'duplicate-label', 'empty-label', 'nonpath'])
def test_all_pairs_validate_before_output(inputs, tmp_path, defect):
    left, right, third = inputs
    pairs = [{'name': 'valid-first', 'left': left, 'right': right},
             {'name': 'late', 'left': left, 'right': third}]
    if defect == 'late-schema':
        pairs[1]['extra'] = True
    elif defect == 'late-nonmatching':
        value = json.loads(third.read_text())
        value['target_plan']['ordering'] = 'different'
        value['target_plan_sha256'] = digest(value['target_plan'])
        third.write_text(json.dumps(value))
    elif defect == 'duplicate-label':
        pairs[1]['name'] = pairs[0]['name']
    elif defect == 'empty-label':
        pairs[1]['name'] = '  '
    else:
        pairs[1]['right'] = {}
    with pytest.raises(ValueError):
        compare_evaluation_pairs(pairs, tmp_path / 'no-output', bootstrap_samples=0)
    assert not (tmp_path / 'no-output').exists()


def test_existing_output_preserved(inputs, tmp_path):
    left, right, _ = inputs
    out = tmp_path / 'existing'
    out.mkdir()
    marker = out / 'precious'
    marker.write_text('unchanged')
    with pytest.raises(FileExistsError):
        compare_evaluation_pairs([{'name': 'a', 'left': left, 'right': right}], out, bootstrap_samples=0)
    assert marker.read_text() == 'unchanged' and list(out.iterdir()) == [marker]


@pytest.mark.parametrize('kwargs', [{'bootstrap_samples': -1}, {'bootstrap_samples': True},
    {'bootstrap_samples': 100001}, {'seed': -1}, {'seed': True}])
def test_invalid_controls_no_output(inputs, tmp_path, kwargs):
    with pytest.raises(ValueError):
        compare_evaluation_pairs([{'name': 'a', 'left': inputs[0], 'right': inputs[1]}], tmp_path / 'bad', **kwargs)
    assert not (tmp_path / 'bad').exists()


def test_independent_paired_ratio_bootstrap_with_explicit_draws(tmp_path, monkeypatch):
    import numpy as np
    counts = [1, 9]
    paths = []
    for label, sums in [('left', [2., 20.]), ('right', [3., 11.])]:
        value = report([1., 6., 6.])
        value['target_plan']['ranges'] = [
            {'document_id': name, 'target_start': 1, 'target_end': count + 1, 'count': count}
            for name, count in zip(['a', 'b'], counts)]
        value['target_plan_sha256'] = digest(value['target_plan'])
        value['per_document'] = [{'document_id': name, 'tokens': count, 'loss_sum': loss}
                                 for name, count, loss in zip(['a', 'b'], counts, sums)]
        value.update(prediction_tokens=10, loss_sum=sum(sums), ce=sum(sums) / 10)
        value['data_overlap_audit'].update(selected_validation_documents=2,
            selected_validation_ids_sha256=hashlib.sha256(json.dumps(['a', 'b'], separators=(',', ':')).encode()).hexdigest())
        path = tmp_path / (label + '.json')
        path.write_text(json.dumps(value))
        paths.append(path)
    draws = [[0, 0], [0, 1], [1, 0], [1, 1]]
    seen = []
    class Draws:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 2, 2)
            draw = draws[len(seen)]
            seen.append(draw)
            return np.array(draw)
    monkeypatch.setattr('tnlm.evaluate.np.random.default_rng', lambda seed: Draws())
    result = compare_evaluation_pairs([{'name': 'explicit-draws', 'left': paths[0], 'right': paths[1]}],
                                      tmp_path / 'result', bootstrap_samples=4, seed=7)
    comparison = result['pairs'][0]['result']
    assert comparison['delta_ce_right_minus_left'] == -.8
    # Independent ratios are [1,-.8,-.8,-1]; linear percentile interpolation.
    assert comparison['uncertainty']['delta_ce_95_ci'] == pytest.approx([-.985, .865])
    assert seen == draws
