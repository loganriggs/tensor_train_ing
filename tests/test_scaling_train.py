"""Trainer-level scaling gates: optimization preserves objective and replay."""
import copy

import pytest
import torch

from helpers import state
from tnlm.train import train_run


@pytest.mark.parametrize('normalization,checkpointing', [
    ('pre_rms', False), ('pre_rms', True), ('spectral_rms', True), ('lagged', False)])
def test_chunked_loss_and_recomputation_preserve_training_and_resume(config, tmp_path, normalization, checkpointing):
    config.model.normalization = normalization
    if normalization == 'lagged':
        config.model.final_normalization = 'lagged'
    baseline = train_run(config, tmp_path / 'dense', max_steps=3, max_seconds=60, device='cpu')
    optimized = copy.deepcopy(config)
    optimized.model.loss_chunk_tokens = 3  # includes partial chunks and ignored targets
    optimized.model.activation_checkpointing = checkpointing
    whole = train_run(optimized, tmp_path / 'optimized', max_steps=3, max_seconds=60, device='cpu')
    first = train_run(optimized, tmp_path / 'first', max_steps=1, max_seconds=60, device='cpu')
    resumed = train_run(optimized, tmp_path / 'resumed', resume=first['checkpoint'],
                        max_steps=2, max_seconds=60, device='cpu')
    dense_state = state(tmp_path / 'dense' / 'final')
    whole_state = state(tmp_path / 'optimized' / 'final')
    resumed_state = state(tmp_path / 'resumed' / 'final')
    assert baseline['progress'] == whole['progress'] == resumed['progress']
    assert [row['batch_hashes'] for row in baseline['history']] == [row['batch_hashes'] for row in whole['history']]
    assert [row['batch_hashes'] for row in whole['history'][1:]] == [row['batch_hashes'] for row in resumed['history']]
    for name, tensor in whole_state['model'].items():
        assert torch.equal(tensor, resumed_state['model'][name]), name
        torch.testing.assert_close(tensor, dense_state['model'][name], rtol=2e-5, atol=2e-6, msg=name)
    for actual, expected in zip(whole['history'], baseline['history']):
        assert actual['loss'] == pytest.approx(expected['loss'], rel=2e-6, abs=2e-6)
    for key, moments in whole_state['optimizer']['state'].items():
        for name, tensor in moments.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor, resumed_state['optimizer']['state'][key][name])


def test_scaling_config_reports_loss_workspace_and_reference_attention(config):
    from tnlm.cli import validate_config
    config.raw['data'].update(context=128, micro_batch_size=2)
    dense = validate_config(config)
    config.model.loss_chunk_tokens = 17
    config.model.activation_checkpointing = True
    chunked = validate_config(config)
    assert dense['estimated_one_fp32_training_logit_chunk_bytes'] == 256 * config.model.vocab_size * 4
    assert chunked['estimated_one_fp32_training_logit_chunk_bytes'] == 17 * config.model.vocab_size * 4
    assert not chunked['scaling']['training_materializes_full_logits']
    assert chunked['scaling']['inference_materializes_full_logits']
    assert chunked['scaling']['activation_checkpointing']
    assert chunked['scaling']['default_training_world_size'] == 1
    assert chunked['scaling']['distributed_training']['validated_transport'] == 'CPU/gloo'
    assert not chunked['scaling']['distributed_training']['compile_supported']
    assert 'dense reference' in chunked['memory_note']
    assert dense['parameters'] == chunked['parameters']


@pytest.mark.parametrize('invalid', ['lagged_recompute', 'negative_chunk'])
def test_invalid_scaling_combinations_fail_before_spend(config, tmp_path, invalid):
    # Dataclasses remain mutable, so launch must recheck safety after construction.
    if invalid == 'lagged_recompute':
        config.model.normalization = 'lagged'
        config.model.final_normalization = 'lagged'
        config.model.activation_checkpointing = True
        message = 'double-count lagged'
    else:
        config.model.loss_chunk_tokens = -1
        message = 'nonnegative integer'
    with pytest.raises(ValueError, match=message):
        train_run(config, tmp_path / 'invalid-scaling', max_steps=1, max_seconds=60, device='cpu')
    assert not (tmp_path / 'invalid-scaling').exists()
    assert not (tmp_path / 'spend.json').exists()


def test_compiled_chunked_recomputed_training_executes(config, tmp_path):
    """Exercise the configured compiler path as a real optimizer/checkpoint run."""
    config.model.activation_checkpointing = True
    config.model.loss_chunk_tokens = 3
    config.train.compile = True
    result = train_run(config, tmp_path / 'compiled', max_steps=1, max_seconds=180, device='cpu')
    assert result['progress']['steps'] == 1
    assert result['progress']['tokens'] == 7
    assert torch.isfinite(torch.tensor(result['history'][0]['loss']))
    payload = state(tmp_path / 'compiled' / 'final')
    assert payload['optimizer']['state']
    assert all(torch.isfinite(value).all() for value in payload['model'].values())


@pytest.mark.parametrize('normalization', ['pre_rms', 'spectral_rms'])
def test_muon_auxiliary_lr_and_exact_composite_optimizer_replay(config, tmp_path, normalization):
    config.model.normalization = normalization
    config.model.loss_chunk_tokens = 3
    config.model.activation_checkpointing = True
    config.train.optimizer = 'muon'
    config.train.aux_adamw_lr = .003
    whole = train_run(config, tmp_path / 'muon-whole', max_steps=3, max_seconds=60, device='cpu')
    first = train_run(config, tmp_path / 'muon-first', max_steps=1, max_seconds=60, device='cpu')
    resumed = train_run(config, tmp_path / 'muon-resumed', resume=first['checkpoint'],
                        max_steps=2, max_seconds=60, device='cpu')
    expected = state(tmp_path / 'muon-whole' / 'final')
    actual = state(tmp_path / 'muon-resumed' / 'final')
    def exact(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                exact(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for l, r in zip(left, right):
                exact(l, r)
        else:
            assert left == right
    exact(expected['model'], actual['model'])
    exact(expected['optimizer'], actual['optimizer'])
    assert set(actual['optimizer']) == {'muon', 'adamw'}
    assert actual['optimizer']['muon']['state'] and actual['optimizer']['adamw']['state']
    for index, row in enumerate(whole['history'], start=1):
        assert row['optimizer_group_lrs'][0] == pytest.approx(config.train.lr * index / 100)
        assert row['optimizer_group_lrs'][1:] == pytest.approx([.003 * index / 100] * 2)
    assert [row['batch_hashes'] for row in whole['history'][1:]] == [row['batch_hashes'] for row in resumed['history']]


def test_head_coefficients_are_scalars_for_decay_and_muon(config):
    from tnlm.model import TensorLM
    from tnlm.optim import param_split
    from tnlm.train import optimizer_groups
    config.model.attention = 'lower_degree_mix'
    model = TensorLM(config.model)
    hidden, _, small = param_split(model)
    coef = model.blocks[0].attn.coef
    assert coef.ndim == 2
    assert id(coef) in {id(parameter) for parameter in small}
    assert id(coef) not in {id(parameter) for parameter in hidden}
    decay, no_decay = optimizer_groups(model, .1)
    assert id(coef) not in {id(parameter) for parameter in decay['params']}
    assert id(coef) in {id(parameter) for parameter in no_decay['params']}


def test_validation_resolves_actual_backend_without_allocating_weights(config):
    from tnlm.cli import validate_config
    config.model.attention_backend = 'auto'
    assert validate_config(config, device='cpu')['scaling']['resolved_attention_backend'] == 'reference'
    assert validate_config(config, device='cuda:0')['scaling']['resolved_attention_backend'] == 'triton'
    config.model.attention = 'softmax'
    config.model.attention_backend = 'triton'
    assert validate_config(config, device='cpu')['scaling']['resolved_attention_backend'] == 'flex'
    config.model.attention_backend = 'reference'
    assert validate_config(config, device='cpu')['scaling']['resolved_attention_backend'] == 'softmax'


@pytest.mark.parametrize('leak', ['ablation', 'denominator'])
def test_analysis_interventions_cannot_leak_into_training(config, tmp_path, monkeypatch, leak):
    from tnlm.model import TensorLM
    model = TensorLM(config.model)
    if leak == 'ablation':
        model.blocks[0].ablate = frozenset({'mlp'})
    else:
        model.blocks[0].mlp_in.denom_mode = 'fixed'
    monkeypatch.setattr('tnlm.train.TensorLM', lambda cfg: model)
    with pytest.raises(RuntimeError, match='analysis .* leaked into training'):
        train_run(config, tmp_path / 'leaked-analysis', max_steps=1, max_seconds=60, device='cpu')
    assert not (tmp_path / 'leaked-analysis' / 'initial').exists()
    assert not (tmp_path / 'leaked-analysis' / 'final').exists()


@pytest.mark.parametrize('launch', ['torchrun', 'initialized'])
def test_distributed_launch_rejected_before_spend(config, tmp_path, monkeypatch, launch):
    if launch == 'torchrun':
        monkeypatch.setenv('WORLD_SIZE', '2')
    else:
        monkeypatch.setenv('WORLD_SIZE', '1')
        monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
        monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    with pytest.raises(RuntimeError, match='distributed training requires coordinated'):
        train_run(config, tmp_path / 'distributed', max_steps=1, max_seconds=60, device='cpu')
    assert not (tmp_path / 'distributed').exists()
    assert not (tmp_path / 'spend.json').exists()


def test_width_scaled_adamw_schedule_and_resume(config, tmp_path):
    config.model.width_scaling_base = 2
    config.model.width_lr_exponent = .5
    config.model.width_logit_exponent = .5
    config.train.lr = .01
    whole = train_run(config, tmp_path / 'width-whole', max_steps=3, max_seconds=60, device='cpu')
    first = train_run(config, tmp_path / 'width-first', max_steps=1, max_seconds=60, device='cpu')
    resumed = train_run(config, tmp_path / 'width-resume', max_steps=2, max_seconds=60,
                        device='cpu', resume=first['checkpoint'])
    for row in whole['history']:
        expected = config.train.lr * row['step'] / whole['progress']['warmup_steps']
        assert row['optimizer_group_lrs'] == pytest.approx([.5 * expected, expected])
    expected_state = state(tmp_path / 'width-whole' / 'final')
    actual = state(tmp_path / 'width-resume' / 'final')
    for name, tensor in expected_state['model'].items():
        assert torch.equal(tensor, actual['model'][name]), name
    assert expected_state['optimizer']['param_groups'] == actual['optimizer']['param_groups']
    for name, moments in expected_state['optimizer']['state'].items():
        for key, value in moments.items():
            assert torch.equal(value, actual['optimizer']['state'][name][key])
    assert whole['progress'] == resumed['progress']


def test_validation_reports_flex_and_capacity_assumptions(config):
    from tnlm.cli import validate_config
    config.model.attention = 'softmax'
    config.model.attention_backend = 'auto'
    report = validate_config(config, device='cuda:0')
    assert report['scaling']['resolved_attention_backend'] == 'flex'
    assert report['memory_plan']['gpus'] == 1
    assert report['memory_plan']['micro_tokens_per_gpu'] == 4
    assert 'not measured peak memory' in report['memory_plan_assumptions']['scope']


def test_distributed_optimizer_capability_report(config):
    from tnlm.cli import validate_config
    config.train.optimizer = "muon"
    report = validate_config(config, device="cpu")["scaling"]["distributed_training"]
    assert report["optimizer"] == "muon"
    assert report["supported_optimizers"] == ["adamw", "muon"]
    assert report["validated_transport"] == "CPU/gloo"
    assert not report["compile_supported"]


def test_validation_reports_retained_checkpoint_storage(config):
    from tnlm.cli import validate_config
    plan = validate_config(config, device="cpu")["checkpoint_storage_plan"]
    assert plan["required_additional_bytes"] >= plan["per_checkpoint_estimate_bytes"] * 2
    assert plan["count_breakdown"]["initial"] == plan["count_breakdown"]["final"] == 1
    assert plan["world_size"] == 1 and not plan["distributed"]
    assert "no reservation" in plan["scope"]
