"""The paired-seed CLI preserves explicit pair selection and report path roots."""
import json

import yaml

from tnlm.cli import main


def test_seed_pair_cli_resolves_paths_from_pair_file(tmp_path, monkeypatch, capsys):
    import tnlm.replication as replication

    directory = tmp_path / 'plans'
    directory.mkdir()
    path = directory / 'pairs.yaml'
    pairs = [{'seed': seed, 'left': f'left-{seed}.json', 'right': f'right-{seed}.json'}
             for seed in (101, 202)]
    path.write_text(yaml.safe_dump(pairs))
    out = tmp_path / 'report'

    def compare(actual, destination, *, allowed_between_dimensions):
        assert [pair['seed'] for pair in actual] == [101, 202]
        assert actual[0]['left'] == str(directory / 'left-101.json')
        assert actual[1]['right'] == str(directory / 'right-202.json')
        assert destination == str(out)
        assert allowed_between_dimensions == ['width', 'config.model.d_mlp']
        return {'forwarded': True}

    monkeypatch.setattr(replication, 'compare_seed_pairs', compare)
    assert main(['compare_seed_pairs', '--pairs', str(path), '--out', str(out),
                 '--allow-differences', 'width', 'config.model.d_mlp']) == 0
    assert json.loads(capsys.readouterr().out) == {'forwarded': True}
    assert not out.exists()
