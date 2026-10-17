"""Real CPU trace capture preserves the complete training trajectory."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tnlm.cli import main
from tnlm.train import profile_run


def test_trace_matches_microbatches_and_preserves_training(config, tmp_path):
    traced = profile_run(config, tmp_path / 'traced', steps=2, warmup_steps=1,
                         max_seconds=60, device='cpu', trace_steps=1)
    plain = profile_run(config, tmp_path / 'plain', steps=2, warmup_steps=1,
                        max_seconds=60, device='cpu')
    trace = traced['kernel_profile']['traces'][0]
    assert traced['kernel_profile']['trace_count'] == 1
    path = Path(trace['path'])
    raw = path.read_bytes()
    assert trace['sha256'] == hashlib.sha256(raw).hexdigest()
    assert trace['bytes'] == len(raw)
    timeline = json.loads(raw)
    counts = Counter(event.get('name') for event in timeline['traceEvents'] if event.get('ph') == 'X')
    metrics = [json.loads(row) for row in (tmp_path / 'traced/metrics.jsonl').read_text().splitlines()]
    assert counts['tnlm.forward'] == counts['tnlm.backward'] == metrics[1]['microbatches']
    assert counts['tnlm.optimizer'] == 1
    assert counts['tnlm.sampler'] >= metrics[1]['microbatches']
    assert counts['tnlm.transfer'] == metrics[1]['microbatches']
    assert traced['completed_timed_steps'] == 2
    assert traced['timed_prediction_tokens'] == plain['timed_prediction_tokens'] == 14
    left = torch.load(Path(traced['checkpoint']) / 'state.pt', weights_only=False)
    right = torch.load(Path(plain['checkpoint']) / 'state.pt', weights_only=False)
    for name, value in left['model'].items():
        torch.testing.assert_close(value, right['model'][name], rtol=0, atol=0)
    torch.testing.assert_close(left['optimizer'], right['optimizer'], rtol=0, atol=0)
    assert left['sampler'] == right['sampler']
    assert not (tmp_path / 'plain/traces').exists()


def test_trace_dry_run_never_starts_capture(config, tmp_path, monkeypatch):
    config.raw['project']['dry_run'] = True
    monkeypatch.setattr(torch.profiler, 'profile', lambda **kwargs: pytest.fail('dry run captured'))
    result = profile_run(config, tmp_path / 'dry', steps=2, warmup_steps=1,
                         max_seconds=60, device='cpu', trace_steps=2)
    assert result['dry_run'] and result['trace_steps'] == 2
    assert not (tmp_path / 'dry').exists()


@pytest.mark.parametrize('count', [-1, 101, True, 1.5])
def test_invalid_trace_bound_rejected_before_training(config, tmp_path, count):
    with pytest.raises(ValueError, match='trace_steps'):
        profile_run(config, tmp_path / 'bad', steps=2, max_seconds=60, device='cpu', trace_steps=count)
    assert not (tmp_path / 'bad').exists()


def test_cli_trace_bound_forwarded(config, tmp_path, monkeypatch, capsys):
    import yaml
    import tnlm.cli as cli
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config.to_dict()))
    def profile(*args, **kwargs):
        assert kwargs['trace_steps'] == 3
        assert kwargs['kernels'] is False
        return {'trace_steps': 3}
    monkeypatch.setattr(cli, 'profile_run', profile)
    assert main(['profile', '--config', str(path), '--run-dir', str(tmp_path / 'out'),
                 '--max-seconds', '60', '--device', 'cpu', '--trace-steps', '3']) == 0
    assert json.loads(capsys.readouterr().out)['trace_steps'] == 3
