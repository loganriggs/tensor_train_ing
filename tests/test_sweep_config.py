import hashlib
import json
import pytest
import yaml
from tnlm.cli import main


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@pytest.fixture
def journal(tmp_path):
    config = {'model': {'layers': 12}, 'train': {'lr': .00045},
              'project': {'dry_run': False, 'spend_ledger': 'original.json'}}
    row = {'name': 'chosen', 'config': config, 'config_sha256': digest(config)}
    identity = {'version': 1, 'trajectories': [row]}
    value = {'identity': identity, 'identity_sha256': digest(identity)}
    path = tmp_path/'sweep.json';path.write_text(json.dumps(value))
    return tmp_path, value, config


def test_cli_stdout_and_file_preserve_exact_legacy_config(journal, tmp_path, capsys):
    root, value, config = journal
    before = (root/'sweep.json').read_bytes()
    argv = ['sweep-config', '--sweep', str(root), '--candidate', 'chosen']
    assert main(argv) == 0
    output = capsys.readouterr().out
    assert yaml.safe_load(output) == config
    assert 'deterministic' not in yaml.safe_load(output)['train']
    assert digest(config) in output and hashlib.sha256(before).hexdigest() in output
    target = tmp_path/'config.yaml'
    assert main(argv+['--out', str(target)]) == 0
    assert capsys.readouterr().out == ''
    assert target.read_text() == output
    with pytest.raises(FileExistsError):main(argv+['--out', str(target)])
    assert (root/'sweep.json').read_bytes() == before


@pytest.mark.parametrize('case', ['unknown', 'duplicate', 'identity-tamper', 'config-tamper'])
def test_rejects_ambiguous_or_corrupt_input_before_output(journal, tmp_path, case):
    root, value, config = journal
    candidate = 'chosen'
    if case == 'unknown':candidate = 'absent'
    elif case == 'duplicate':
        value['identity']['trajectories'] *= 2
        value['identity_sha256'] = digest(value['identity'])
    else:
        value['identity']['trajectories'][0]['config']['train']['lr'] = 1.
        if case == 'config-tamper':value['identity_sha256'] = digest(value['identity'])
    (root/'sweep.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):
        main(['sweep-config', '--sweep', str(root), '--candidate', candidate, '--out', str(tmp_path/'bad.yaml')])
    assert not (tmp_path/'bad.yaml').exists()
