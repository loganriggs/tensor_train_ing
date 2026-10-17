import os
import pytest
import torch
from tnlm.runtime import (deterministic_execution, determinism_metadata,
                          full_config, resume_validator)
from tnlm.train import train_run


@pytest.mark.parametrize('failure', [False, True])
def test_policy_scope_and_failure_restore(monkeypatch, failure):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'prior-value')
    before = determinism_metadata()
    def execute():
        with deterministic_execution(True, device='cpu'):
            now = determinism_metadata()
            assert now['algorithms_enabled'] and not now['warn_only']
            assert now['cublas_workspace_config'] == ':4096:8'
            assert not now['cudnn_benchmark'] and now['cudnn_deterministic']
            if failure:
                raise ValueError('injected')
    if failure:
        with pytest.raises(ValueError, match='injected'): execute()
    else: execute()
    assert determinism_metadata() == before


def test_late_cuda_setup_rejects_without_mutation(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    before = determinism_metadata()
    with pytest.raises(RuntimeError, match='fresh process'):
        with deterministic_execution(True, device='cuda'): pytest.fail('entered')
    assert determinism_metadata() == before


def test_existing_cuda_workspace_supported(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with deterministic_execution(True, device='cuda'):
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':16:8'


@pytest.mark.parametrize('value', [1, 'true', None])
def test_invalid_policy_before_writes(config, tmp_path, value):
    config.train.deterministic = value
    with pytest.raises(ValueError, match='Boolean'):
        train_run(config, tmp_path/'invalid', max_steps=1, max_seconds=30, device='cpu')
    assert not (tmp_path/'invalid').exists()


def test_dry_run_leaves_policy_unchanged(config, tmp_path):
    config.train.deterministic = True
    config.raw['project']['dry_run'] = True
    before = determinism_metadata()
    assert train_run(config, tmp_path/'dry', max_steps=1, max_seconds=30, device='cpu')['dry_run']
    assert before == determinism_metadata()


def test_real_training_records_policy_and_restores(config, tmp_path):
    config.train.deterministic = True
    before = determinism_metadata()
    result = train_run(config, tmp_path/'trained', max_steps=2, max_seconds=60, device='cpu')
    assert determinism_metadata() == before
    assert result['train_config']['deterministic'] is True
    assert result['runtime']['determinism']['algorithms_enabled'] is True
    state = torch.load(tmp_path/'trained/final/state.pt', weights_only=False)
    assert state['config']['train']['deterministic'] is True
    assert state['extra']['runtime']['determinism']['cublas_workspace_config'] == ':4096:8'


def test_resume_policy_is_pinned_and_legacy_false_works(config):
    config.train.deterministic = False
    legacy = full_config(config)
    del legacy['train']['deterministic']
    resume_validator(config)(legacy)
    config.train.deterministic = True
    with pytest.raises(ValueError, match='resume config mismatch'):
        resume_validator(config)(legacy)


def test_cuda_workspace_retained_after_first_initialization(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    initialized = [False]
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: initialized[0])
    before = torch.are_deterministic_algorithms_enabled()
    with deterministic_execution(True, device='cuda'):
        initialized[0] = True
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert torch.are_deterministic_algorithms_enabled() == before
    # A subsequent sweep stage can reuse the initialized CUDA context.
    with deterministic_execution(True, device='cuda'):
        assert torch.are_deterministic_algorithms_enabled()


def test_distributed_cli_policy_precedes_group_setup(config, monkeypatch):
    from types import SimpleNamespace
    import tnlm.cli as cli
    config.train.deterministic = True
    before = determinism_metadata()
    def setup(cfg, args):
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8')
        raise RuntimeError('group setup failed')
    monkeypatch.setattr(cli, '_distributed_run_with_policy', setup)
    with pytest.raises(RuntimeError, match='group setup failed'):
        cli._distributed_run_from_cli(config, SimpleNamespace(device='cpu'))
    assert determinism_metadata() == before
