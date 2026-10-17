import random

import numpy as np
import pytest
import torch

from tnlm.checkpoint import load_checkpoint, save_checkpoint


class Sampler:
    def __init__(self):
        self.cursor = 0

    def state_dict(self):
        return {'cursor': self.cursor}

    def load_state_dict(self, state):
        self.cursor = state['cursor']


class Model(torch.nn.Linear):
    def __init__(self):
        super().__init__(3, 2)
        self.register_buffer('running_rms', torch.ones(2))
        self.register_buffer('spectral_u', torch.tensor([0.6, 0.8]))


def setup_state():
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    sampler = Sampler()
    optimizer.zero_grad()
    model(torch.ones(2, 3)).square().sum().backward()
    optimizer.step()
    scheduler.step()
    sampler.cursor = 17
    model.running_rms.fill_(3)
    return model, optimizer, sampler, scheduler


def test_full_state_replay(tmp_path):
    model, optimizer, sampler, scheduler = setup_state()
    path = tmp_path / 'step1'
    save_checkpoint(path, model, optimizer, sampler, config={'seed': 42},
                    progress={'steps': 1, 'tokens': 6}, manifest_hash='abc', scheduler=scheduler)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    expected_weights = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer.zero_grad()
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    next_weights = {k: v.clone() for k, v in model.state_dict().items()}
    sampler.cursor = 99
    model.running_rms.zero_()
    payload = load_checkpoint(path, model, optimizer, sampler, scheduler=scheduler,
                              expected_config={'seed': 42}, expected_manifest_hash='abc')
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert sampler.cursor == 17
    assert payload['progress'] == {'steps': 1, 'tokens': 6}
    assert payload['environment']['torch']
    for k, v in model.state_dict().items():
        assert torch.equal(v, expected_weights[k])
    optimizer.zero_grad()
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    # The second identical optimizer update proves Adam's moments were restored.
    for k, v in model.state_dict().items():
        assert torch.equal(v, next_weights[k])


def test_reject_corrupt_incomplete_and_overwrite(tmp_path):
    model, optimizer, sampler, scheduler = setup_state()
    path = tmp_path / 'saved'
    kwargs = dict(config={}, progress={}, manifest_hash='abc')
    save_checkpoint(path, model, optimizer, sampler, **kwargs)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, optimizer, sampler, **kwargs)
    with (path / 'state.pt').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(path, model)
    (tmp_path / 'partial').mkdir()
    torch.save({}, tmp_path / 'partial' / 'state.pt')
    with pytest.raises(ValueError, match='Incomplete'):
        load_checkpoint(tmp_path / 'partial', model)


@pytest.mark.parametrize('kwargs,match', [
    ({'expected_manifest_hash': 'wrong'}, 'manifest hash'),
    ({'expected_config': {'seed': 0}}, 'config mismatch'),
])
def test_identity_failure_does_not_mutate_model(tmp_path, kwargs, match):
    model, optimizer, sampler, scheduler = setup_state()
    save_checkpoint(tmp_path / 'saved', model, optimizer, sampler,
                    config={'seed': 42}, progress={}, manifest_hash='abc')
    with torch.no_grad():
        model.weight.fill_(77)
    with pytest.raises(ValueError, match=match):
        load_checkpoint(tmp_path / 'saved', model, **kwargs)
    assert torch.all(model.weight == 77)


def test_failed_write_is_not_published(tmp_path, monkeypatch):
    model, optimizer, sampler, scheduler = setup_state()

    def fail_save(payload, stream):
        stream.write(b'partial bytes')
        raise OSError('simulated disk failure')

    monkeypatch.setattr(torch, 'save', fail_save)
    with pytest.raises(OSError, match='disk failure'):
        save_checkpoint(tmp_path / 'saved', model, optimizer, sampler,
                        config={}, progress={}, manifest_hash='abc')
    assert not (tmp_path / 'saved').exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('buffer_name', ['acc_count', 'acc_sum'])
def test_unfinished_stat_collection_rejected(tmp_path, buffer_name):
    model = Model()
    model.register_buffer('acc_count', torch.zeros(()), persistent=False)
    model.register_buffer('acc_sum', torch.zeros(()), persistent=False)
    getattr(model, buffer_name).fill_(1)
    with pytest.raises(ValueError, match='unfinished statistic collection'):
        save_checkpoint(tmp_path / 'saved', model, None, None,
                        config={}, progress={}, manifest_hash='abc')
    assert not (tmp_path / 'saved').exists()
    getattr(model, buffer_name).zero_()
    save_checkpoint(tmp_path / 'saved', model, None, None,
                    config={}, progress={}, manifest_hash='abc')
    payload = load_checkpoint(tmp_path / 'saved', model)
    assert len(payload['code']['source_sha256']['tnlm/checkpoint.py']) == 64
    assert 'torch_build' in payload['environment']
    assert 'deterministic_algorithms' in payload['environment']


def test_config_validator_runs_before_mutation(tmp_path):
    model, optimizer, sampler, scheduler = setup_state()
    save_checkpoint(tmp_path / 'saved', model, optimizer, sampler,
                    config={'rung_total_tokens': 100}, progress={}, manifest_hash='abc')
    saved = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.fill_(77)

    def reject(config):
        assert config == {'rung_total_tokens': 100}
        raise ValueError('incompatible promotion')

    with pytest.raises(ValueError, match='incompatible promotion'):
        load_checkpoint(tmp_path / 'saved', model, config_validator=reject)
    assert torch.all(model.weight == 77)

    def accept(config):
        assert config['rung_total_tokens'] <= 200

    load_checkpoint(tmp_path / 'saved', model, config_validator=accept)
    assert torch.equal(model.weight, saved)
    # A compatibility callback does not bypass separately requested strict checks.
    with pytest.raises(ValueError, match='config mismatch'):
        load_checkpoint(tmp_path / 'saved', model, config_validator=accept,
                        expected_config={'rung_total_tokens': 200})


@pytest.mark.parametrize('location', ['model', 'optimizer'])
def test_nonfinite_state_not_published(tmp_path, location):
    model, optimizer, sampler, scheduler = setup_state()
    if location == 'model':
        with torch.no_grad():
            model.weight[0, 0] = float('nan')
    else:
        state = optimizer.state[model.weight]
        state['exp_avg'][0, 0] = float('inf')
    with pytest.raises(ValueError, match=f'Nonfinite checkpoint tensor: {location}'):
        save_checkpoint(tmp_path / 'saved', model, optimizer, sampler,
                        config={}, progress={}, manifest_hash='abc')
    assert not (tmp_path / 'saved').exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('location', ['model', 'optimizer'])
def test_checksummed_nonfinite_load_does_not_mutate(tmp_path, location):
    import hashlib
    import json

    model, optimizer, sampler, scheduler = setup_state()
    path = tmp_path / 'saved'
    save_checkpoint(path, model, optimizer, sampler,
                    config={}, progress={}, manifest_hash='abc')
    state_path = path / 'state.pt'
    payload = torch.load(state_path, weights_only=False)
    assert payload['model_class'].endswith('.Model')
    assert payload['optimizer_class'].endswith('.AdamW')
    assert any(name.lower() == 'torch' for name in payload['environment']['distributions'])
    if location == 'model':
        payload['model']['weight'][0, 0] = float('nan')
    else:
        first_state = next(iter(payload['optimizer']['state'].values()))
        first_state['exp_avg'].fill_(float('inf'))
    torch.save(payload, state_path)
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['files']['state.pt'] = {
        'sha256': hashlib.sha256(state_path.read_bytes()).hexdigest(),
        'bytes': state_path.stat().st_size,
    }
    (path / 'manifest.json').write_text(json.dumps(manifest))
    with torch.no_grad():
        model.weight.fill_(77)
    with pytest.raises(ValueError, match=f'Nonfinite checkpoint tensor: {location}'):
        load_checkpoint(path, model, optimizer)
    assert torch.all(model.weight == 77)
    assert torch.isfinite(optimizer.state[model.weight]['exp_avg']).all()


def _rewrite_payload(path, transform):
    import hashlib
    import json
    state_path = path / 'state.pt'
    payload = torch.load(state_path, weights_only=False)
    transform(payload)
    torch.save(payload, state_path)
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['files']['state.pt'] = {
        'sha256': hashlib.sha256(state_path.read_bytes()).hexdigest(),
        'bytes': state_path.stat().st_size,
    }
    (path / 'manifest.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('damage', ['missing_buffer', 'shape', 'optimizer_class', 'optimizer_groups', 'cuda_rng'])
def test_incompatible_load_preflight_does_not_mutate(tmp_path, monkeypatch, damage):
    model, optimizer, sampler, scheduler = setup_state()
    path = tmp_path / 'saved'
    save_checkpoint(path, model, optimizer, sampler, config={}, progress={}, manifest_hash='abc')

    def transform(payload):
        if damage == 'missing_buffer':
            del payload['model']['spectral_u']
        elif damage == 'shape':
            payload['model']['spectral_u'] = torch.ones(7)
        elif damage == 'optimizer_class':
            payload['optimizer_class'] = 'torch.optim.sgd.SGD'
        elif damage == 'optimizer_groups':
            payload['optimizer']['param_groups'][0]['params'] = []
        else:
            payload['rng']['cuda'] = [torch.zeros(10, dtype=torch.uint8)]
    _rewrite_payload(path, transform)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with torch.no_grad():
        model.weight.fill_(77)
    with pytest.raises(ValueError):
        load_checkpoint(path, model, optimizer)
    assert torch.all(model.weight == 77)


def test_load_clears_abandoned_nonpersistent_accumulation(tmp_path):
    model = Model()
    model.register_buffer('acc_count', torch.zeros(()), persistent=False)
    model.register_buffer('acc_sum', torch.zeros(()), persistent=False)
    save_checkpoint(tmp_path / 'saved', model, None, None, config={}, progress={}, manifest_hash='abc')
    model.acc_count.fill_(17)
    model.acc_sum.fill_(99)
    load_checkpoint(tmp_path / 'saved', model)
    assert model.acc_count == 0 and model.acc_sum == 0


def test_publish_race_never_overwrites_existing_directory(tmp_path, monkeypatch):
    import tnlm.checkpoint as checkpoint_module
    publish = checkpoint_module._publish
    path = tmp_path / 'saved'

    def competing_writer(staging, destination):
        destination.mkdir()
        publish(staging, destination)

    monkeypatch.setattr(checkpoint_module, '_publish', competing_writer)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, Model(), None, None, config={}, progress={}, manifest_hash='abc')
    assert path.is_dir() and list(path.iterdir()) == []
    assert list(tmp_path.iterdir()) == [path]


def test_malformed_manifest_rejected_cleanly(tmp_path):
    import json
    path = tmp_path / 'saved'
    save_checkpoint(path, Model(), None, None, config={}, progress={}, manifest_hash='abc')
    manifest = json.loads((path / 'manifest.json').read_text())
    del manifest['files']['state.pt']['bytes']
    (path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Incomplete or invalid'):
        load_checkpoint(path, Model())

def test_code_identity_includes_nested_backend_sources():
    import hashlib
    from pathlib import Path
    import tnlm.checkpoint as checkpoint
    package = Path(checkpoint.__file__).parent
    identity = checkpoint._git()['source_sha256']
    sources = list(package.rglob('*.py'))
    assert sources
    for source in sources:
        key = f'tnlm/{source.relative_to(package).as_posix()}'
        assert identity[key] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert 'tnlm/kernels/poly_attn.py' in identity

@pytest.mark.parametrize('child', ['muon', 'adamw'])
def test_composite_optimizer_mismatch_rejected_before_model_mutation(tmp_path, child):
    from tnlm.config import ModelConfig, TrainConfig
    from tnlm.model import TensorLM
    from tnlm.train import make_optimizer
    model = TensorLM(ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4,
                                 d_mlp=12, vocab_size=16, padded_vocab=16))
    optimizer = make_optimizer(model, TrainConfig(optimizer='muon', lr=.01, aux_adamw_lr=.003))
    path = tmp_path / 'composite'
    save_checkpoint(path, model, optimizer, None, config={}, progress={}, manifest_hash='abc')
    def damage(payload):
        payload['optimizer'][child]['param_groups'][0]['params'].append(999999)
    _rewrite_payload(path, damage)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(77)
    with pytest.raises(ValueError, match='optimizer parameter groups mismatch'):
        load_checkpoint(path, model, optimizer)
    assert all(torch.all(parameter == 77) for parameter in model.parameters())


def test_disk_admission_rejects_before_any_writes(tmp_path, monkeypatch):
    import errno
    from collections import namedtuple
    usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr('tnlm.checkpoint_storage.shutil.disk_usage', lambda _: usage(1, 1, 0))
    def forbidden_save(*args, **kwargs):
        raise AssertionError('torch.save reached before disk admission')
    monkeypatch.setattr(torch, 'save', forbidden_save)
    model, optimizer, sampler, _ = setup_state()
    with pytest.raises(OSError) as error:
        save_checkpoint(tmp_path / 'newparent' / 'saved', model, optimizer, sampler,
                        config={}, progress={}, manifest_hash='test')
    assert error.value.errno == errno.ENOSPC
    assert list(tmp_path.iterdir()) == []


def test_disk_admission_recorded(tmp_path):
    from tnlm.checkpoint_storage import RESERVE_BYTES
    model, optimizer, sampler, _ = setup_state()
    manifest = save_checkpoint(tmp_path / 'saved', model, optimizer, sampler,
                               config={}, progress={}, manifest_hash='test')
    info = manifest['disk_admission']
    assert info['required_bytes'] >= (tmp_path / 'saved' / 'state.pt').stat().st_size
    assert info['reserve_bytes'] == RESERVE_BYTES
    assert info['free_bytes'] >= info['required_bytes'] + info['reserve_bytes']
