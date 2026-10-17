"""Context history distinguishes extension branches from early long-context runs."""
import copy
import json

import pytest
import yaml

from tnlm.cli import main
from tnlm.data import PackedSampler
from tnlm.evaluate import _training_comparison, _training_descriptor
from tnlm.runtime import full_config, record_context_fork, resume_validator


def test_fork_validator_preserves_all_other_math_and_cosine_horizon(config):
    saved = copy.deepcopy(full_config(config))
    config.raw['data']['context'] = 8
    resume_validator(config, context_fork=True)(saved)
    with pytest.raises(ValueError, match='mismatch'):
        resume_validator(config)(saved)
    for section, key, value in [('train', 'lr', .003), ('data', 'micro_batch_size', 2),
                                ('model', 'residual_alpha', .25)]:
        invalid = copy.deepcopy(saved)
        invalid[section][key] = value
        with pytest.raises(ValueError, match='mismatch'):
            resume_validator(config, context_fork=True)(invalid)
    saved['train']['schedule'] = config.train.schedule = 'cosine'
    config.train.rung_total_tokens += 1
    with pytest.raises(ValueError, match='exact original token horizon'):
        resume_validator(config, context_fork=True)(saved)


def descriptor(history):
    return _training_descriptor({'config': {'data': {'context': 8}},
        'progress': {'tokens': 21, 'steps': 3, 'context_history': history}})


def test_context_history_distinguishes_fork_and_from_scratch():
    short = {'context': 4, 'start_tokens': 0, 'start_step': 0, 'source': 'fresh'}
    progress = {'tokens': 7, 'steps': 1, 'context_history': [short]}
    parent = {'path': '/parent', 'manifest_sha256': 'a' * 64}
    source = {'document_cursor': 1, 'token_cursor': 3}
    record_context_fork(progress, old_context=4, new_context=8, parent=parent, source_state=source)
    parent['path'] = '/mutated'
    source['token_cursor'] = 99
    assert progress['context_history'][1]['parent']['path'] == '/parent'
    assert progress['context_history'][1]['source_sampler']['token_cursor'] == 3
    fork = descriptor(progress['context_history'])
    fresh = descriptor([{'context': 8, 'start_tokens': 0, 'start_step': 0, 'source': 'fresh'}])
    assert fork['train_context'] == fresh['train_context'] == 8
    assert fork['context_history_status'] == fresh['context_history_status'] == 'complete'
    comparison = _training_comparison({'training_descriptor': fork}, {'training_descriptor': fresh})
    assert comparison['dimensions']['train_context']['status'] == 'matched'
    assert comparison['dimensions']['context_history']['status'] == 'different'


def test_legacy_history_never_infers_complete_or_matched_lifetime_context():
    progress = {'tokens': 7, 'steps': 1}
    record_context_fork(progress, old_context=4, new_context=8, parent={}, source_state={})
    report = descriptor(progress['context_history'])
    assert report['context_history_status'] == 'partial'
    assert report['context_history'][0]['start_tokens'] is None
    comparison = _training_comparison({'training_descriptor': report}, {'training_descriptor': report})
    assert comparison['dimensions']['context_history']['status'] == 'unknown'
    assert descriptor(None)['context_history_status'] == 'unknown'


@pytest.mark.parametrize('bad', [
    [], [{'context': 8, 'start_tokens': 7, 'start_step': 1, 'source': 'fresh'}],
    [{'context': 4, 'start_tokens': 0, 'start_step': 0, 'source': 'fresh'}],
    [{'context': True, 'start_tokens': 0, 'start_step': 0, 'source': 'fresh'}],
    [{'context': 8, 'start_tokens': 99, 'start_step': 1, 'source': 'context_fork'}],
])
def test_inconsistent_history_is_unknown(bad):
    assert descriptor(bad)['context_history_status'] == 'unknown'


def test_context_exposure_distinguishes_prefix_inputs_and_masked_targets(config):
    sampler = PackedSampler(config.raw['data']['manifest'], context=3)
    sampler.next_batch()
    fork = sampler.fork_context(7)
    fork.next_batch()
    assert fork.batch_context_exposure() == {
        'prefix_tokens_at_or_before_watermark': 4, 'masked_prediction_targets': 3}
    fork.next_batch()
    assert fork.batch_context_exposure() == {
        'prefix_tokens_at_or_before_watermark': 0, 'masked_prediction_targets': 0}


def test_cli_context_fork_forwarding(config, tmp_path, monkeypatch, capsys):
    import tnlm.cli as cli
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(full_config(config)))
    def run(cfg, run_dir, **kwargs):
        assert kwargs['context_fork'] is True
        assert kwargs['resume'] == 'parent'
        return {'forwarded': True}
    monkeypatch.setattr(cli, 'train_run', run)
    assert main(['train', '--config', str(path), '--run-dir', str(tmp_path / 'run'),
                 '--max-steps', '1', '--max-seconds', '60', '--device', 'cpu',
                 '--resume', 'parent', '--context-fork']) == 0
    assert json.loads(capsys.readouterr().out) == {'forwarded': True}


def test_real_context_branch_summary_preserves_history(config, tmp_path):
    from tnlm.train import train_run
    from tnlm.reporting import summarize_runs

    parent = train_run(config, tmp_path / 'parent', max_steps=1, max_seconds=60, device='cpu')
    config.raw['data']['context'] = 9
    branch = train_run(config, tmp_path / 'branch', max_steps=1, max_seconds=60,
                       device='cpu', resume=parent['checkpoint'], context_fork=True)
    summarize_runs([tmp_path / 'branch'], tmp_path / 'summary')
    row = json.loads((tmp_path / 'summary' / 'comparison.json').read_text())['runs'][0]
    assert row['context'] == 9
    assert row['context_history_status'] == 'complete'
    assert row['context_history'] == branch['training_descriptor']['context_history']
    assert [entry['context'] for entry in row['context_history']] == [4, 9]


def test_distributed_cli_dry_context_fork_needs_parent_without_initializing_group(config, tmp_path, capsys, monkeypatch):
    import torch.distributed as dist

    def forbidden(*args, **kwargs):
        raise AssertionError('dry run must not initialize a process group')
    monkeypatch.setattr(dist, 'init_process_group', forbidden)
    config.raw['project']['dry_run'] = True
    config.raw['data']['context'] = 2048
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(full_config(config)))
    args = ['train', '--config', str(path), '--run-dir', str(tmp_path / 'run'),
            '--max-steps', '1', '--max-seconds', '60', '--distributed', '--context-fork']
    with pytest.raises(ValueError, match='explicit parent'):
        main(args)
    assert main([*args, '--resume', 'placeholder']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['context_fork'] is True and report['new_context'] == 2048
    assert report['parent'] == 'placeholder' and report['parent_validation'] == 'pending'
    assert not (tmp_path / 'run').exists()
