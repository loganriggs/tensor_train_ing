"""Batch comparison CLI input modes and pair-file path semantics."""
import json

import pytest
import yaml

from tnlm.cli import main


def test_batch_paths_resolve_from_pair_file(tmp_path, monkeypatch, capsys):
    import tnlm.comparison_batch as batch

    plans = tmp_path / 'plans'
    plans.mkdir()
    pairs = plans / 'pairs.yaml'
    pairs.write_text(yaml.safe_dump([{'name': 'width comparison', 'left': 'a.json', 'right': 'b.json'}]))
    destination = tmp_path / 'out'

    def compare(actual, out, *, bootstrap_samples, seed):
        assert actual == [{'name': 'width comparison', 'left': str(plans / 'a.json'), 'right': str(plans / 'b.json')}]
        assert out == str(destination)
        assert (bootstrap_samples, seed) == (73, 19)
        return {'forwarded': True}

    monkeypatch.setattr(batch, 'compare_evaluation_pairs', compare)
    assert main(['compare_evaluations', '--pairs', str(pairs), '--out', str(destination),
                 '--bootstrap-samples', '73', '--seed', '19']) == 0
    assert json.loads(capsys.readouterr().out) == {'forwarded': True}
    assert not destination.exists()


@pytest.mark.parametrize('inputs', [[], ['--left', 'a'], ['--right', 'b'],
    ['--pairs', 'pairs.yaml', '--left', 'a'], ['--pairs', 'pairs.yaml', '--right', 'b']])
def test_incomplete_or_ambiguous_modes_rejected_before_io(inputs, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(['compare_evaluations', *inputs, '--out', str(tmp_path / 'out')])
    assert error.value.code == 2
    assert not (tmp_path / 'out').exists()


def test_legacy_single_pair_mode_preserved(tmp_path, monkeypatch, capsys):
    import tnlm.train as train

    def compare(left, right, out, *, bootstrap_samples, seed):
        assert (left, right, out) == ('a.json', 'b.json', str(tmp_path / 'out'))
        assert (bootstrap_samples, seed) == (1000, 0)
        return {'legacy': True}

    monkeypatch.setattr(train, 'compare_evaluations', compare)
    assert main(['compare_evaluations', '--left', 'a.json', '--right', 'b.json',
                 '--out', str(tmp_path / 'out')]) == 0
    assert json.loads(capsys.readouterr().out) == {'legacy': True}
