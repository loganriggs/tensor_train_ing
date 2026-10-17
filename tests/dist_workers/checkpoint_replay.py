"""Bounded two-rank CPU/gloo FSDP2 checkpoint and exact next-update replay gate."""
import json
import os
from pathlib import Path
import random
import sys
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tnlm.config import ModelConfig
from tnlm.data import RankPackedSampler, batch_hash, manifest_hash, prepare_data
from tnlm.distributed_checkpoint import load_distributed_checkpoint, save_distributed_checkpoint
from tnlm.model import TensorLM
from tnlm.normalization import SIGMA_FLOOR, SpectralLinear, lagged_modules, spectral_step
from tnlm.parallel import full_state_dict, global_count, shard_model


def copied(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: copied(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copied(item) for item in value]
    return value


def identical(left, right):
    if isinstance(left, torch.Tensor):
        a = left.to_local() if hasattr(left, 'to_local') else left
        b = right.to_local() if hasattr(right, 'to_local') else right
        assert torch.equal(a, b), float((a - b).abs().max())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            identical(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            identical(a, b)
    else:
        assert left == right


def certify_portable(cfg, checkpoint, source_state, expected_logits, probe, *, analysis_path=None):
    from tnlm.checkpoint import load_checkpoint
    portable = TensorLM(cfg).eval()
    exported = load_checkpoint(checkpoint, portable, restore_rng=False)
    assert exported['optimizer'] is None and exported['sampler'] is None
    assert exported['extra']['training_recovery'] is False
    identical(source_state, portable.state_dict())
    expected_class = 'P' if cfg.normalization == 'lagged' else 'N'
    assert portable.operator_audit()['class'] == expected_class
    # Reconstruct spectral matrices directly from gathered source parameters and
    # power-iteration buffers; do not query a sharded module outside its forward.
    for name, module in portable.named_modules():
        if isinstance(module, SpectralLinear):
            weight = source_state[f'{name}.weight']
            sigma = (source_state[f'{name}.u'] @ weight @ source_state[f'{name}.v']).abs().clamp_min(SIGMA_FLOOR)
            expected = source_state[f'{name}.gamma'] * weight / sigma
            torch.testing.assert_close(module.effective_weight(), expected, rtol=0, atol=0)
    with torch.no_grad():
        logits = portable(probe, torch.zeros_like(probe))
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        # Prefix length and later token content are both changed independently.
        changed = probe.clone()
        changed[:, 2:] = torch.tensor([[101, 102]])
        torch.testing.assert_close(logits[:, :2], portable(changed, torch.zeros_like(changed))[:, :2],
                                   rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(logits[:, :2], portable(probe[:, :2], torch.zeros_like(probe[:, :2])),
                                   rtol=1e-6, atol=1e-7)
    if analysis_path is not None:
        from tnlm.analysis import analyze
        portable.train()  # Verify restoration of a meaningful nondefault entry mode.
        parameter_state = copied(dict(portable.named_parameters()))
        buffer_state = copied(dict(portable.named_buffers()))
        modes = {name: module.training for name, module in portable.named_modules()}
        hooks = {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks), dict(module._backward_hooks))
                 for name, module in portable.named_modules()}
        attrs = {name: {attr: getattr(module, attr) for attr in
                 ('ablate', 'collecting', '_valid', 'denom_mode', 'denom_saved', 'denom_const') if hasattr(module, attr)}
                 for name, module in portable.named_modules()}
        targets = torch.tensor([[98, 99, 100, -100]])
        result = analyze(portable, [(probe, torch.zeros_like(probe), targets)], str(analysis_path))
        assert result['operator_class'] == expected_class
        assert set(('local_tensor', 'read_write_graph', 'normalizer_sensitivity',
                    'ablation', 'fixed_denominator', 'low_rank')) <= result.keys()
        assert (analysis_path / 'analysis.json').is_file()
        identical(parameter_state, dict(portable.named_parameters()))
        identical(buffer_state, dict(portable.named_buffers()))
        for name, module in portable.named_modules():
            assert module.training == modes[name]
            assert (dict(module._forward_hooks), dict(module._forward_pre_hooks), dict(module._backward_hooks)) == hooks[name]
            assert {attr: getattr(module, attr) for attr in attrs[name]} == attrs[name]
        print(f'{cfg.normalization}: FP32 trained/exported actual analysis and frozen state/mode/hooks passed', flush=True)



class Tokenizer:
    name, eot_token = 'offline', 50256
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    dist.init_process_group('gloo', timeout=timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(sys.argv[1])
    if rank == 0:
        source = root / 'source.jsonl'
        source.write_text(''.join(json.dumps({'id': str(i), 'text': 'abcdefghi'}) + '\n' for i in range(8)))
        prepare_data(root / 'data', source_jsonl=source, revision='offline-dcp-v1', tokenizer=Tokenizer(),
                     split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    dist.barrier()
    manifest = root / 'data' / 'manifest.json'
    mesh = init_device_mesh('cpu', (world,))
    for package in ('pre_rms', 'spectral_rms', 'lagged'):
        torch.manual_seed(81)
        cfg = ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4, d_mlp=12,
                          padded_vocab=50257, normalization=package,
                          final_normalization='lagged' if package == 'lagged' else 'rms', attention_backend='reference')
        reference = TensorLM(cfg)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.001)
        torch.manual_seed(81)
        model = shard_model(TensorLM(cfg), mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=4, seed=3)
        settings = {'model_package': package}
        path = root / package
        # Saving an empty optimizer materializes zero moments but must not skip
        # the true first-step bias-correction counter or alter any weight.
        original = copied(model.state_dict())
        save_distributed_checkpoint(root / f'{package}-initial', model, optimizer, sampler,
             config=settings, progress={'steps': 0}, manifest_hash=manifest_hash(manifest))
        assert optimizer.state
        assert all(float(state['step']) == 0 for state in optimizer.state.values())
        identical(original, model.state_dict())

        def update(batch):
            optimizer.zero_grad(set_to_none=True)
            spectral_step(model)
            idx, docs, targets = batch
            for norm in lagged_modules(model):
                norm.collecting = True
                norm.set_valid_mask(docs >= 0)
            _, loss, count = model(idx, docs, targets, reduction='sum')
            (loss / global_count(count)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            for norm in lagged_modules(model):
                norm.collecting = False
                norm.set_valid_mask(None)
                norm.commit()
            optimizer.zero_grad(set_to_none=True)

        load_distributed_checkpoint(root / f'{package}-initial', model, optimizer, sampler,
                                    expected_config=settings)
        first_batch = sampler.next_batch()
        update(first_batch)
        gathered = [None] * world
        dist.all_gather_object(gathered, first_batch)
        global_batch = tuple(torch.cat([batch[i] for batch in gathered], dim=0) for i in range(3))
        reference_optimizer.zero_grad(set_to_none=True)
        spectral_step(reference)
        for norm in lagged_modules(reference):
            norm.collecting = True
            norm.set_valid_mask(global_batch[1] >= 0)
        _, ref_loss, ref_count = reference(*global_batch, reduction='sum')
        (ref_loss / ref_count).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
        reference_optimizer.step()
        for norm in lagged_modules(reference):
            norm.collecting = False
            norm.set_valid_mask(None)
            norm.commit(distributed=False)
        for name, value in model.state_dict().items():
            full = value.full_tensor() if hasattr(value, 'full_tensor') else value
            torch.testing.assert_close(full, reference.state_dict()[name], rtol=1e-12, atol=1e-12)
        del reference, reference_optimizer
        random.seed(100 + rank)
        np.random.seed(200 + rank)
        torch.manual_seed(300 + rank)
        save_distributed_checkpoint(path, model, optimizer, sampler, config=settings,
                                   progress={'steps': 1}, manifest_hash=manifest_hash(manifest), parent='initial')
        expected_rng = (random.random(), np.random.rand(), torch.rand(3))
        batch = sampler.next_batch()
        expected_batch = batch_hash(batch)
        update(batch)
        expected_model = copied(model.state_dict())
        expected_optimizer = copied(optimizer.state_dict())
        sampler.next_batch()
        torch.rand(19)
        random.random()
        np.random.rand()
        payload = load_distributed_checkpoint(path, model, optimizer, sampler,
                     expected_config=settings, expected_manifest_hash=manifest_hash(manifest))
        assert payload['progress'] == {'steps': 1}
        assert random.random() == expected_rng[0]
        assert np.random.rand() == expected_rng[1]
        assert torch.equal(torch.rand(3), expected_rng[2])
        batch = sampler.next_batch()
        assert batch_hash(batch) == expected_batch
        update(batch)
        identical(expected_model, model.state_dict())
        identical(expected_optimizer, optimizer.state_dict())

        # Explicit bounded export keeps the ordinary analysis/evaluation path usable.
        from tnlm.distributed_checkpoint import export_distributed_weights
        from tnlm.checkpoint import load_checkpoint
        try:
            export_distributed_weights(root / f'{package}-too-large', model, config=settings,
                progress={'steps': 2}, manifest_hash=manifest_hash(manifest), max_state_bytes=1)
        except ValueError as error:
            assert 'exceeds' in str(error)
        else:
            raise AssertionError('export byte cap ignored')
        export_distributed_weights(root / f'{package}-evaluation', model, config=settings,
            progress={'steps': 2}, manifest_hash=manifest_hash(manifest), parent=str(path))
        probe = torch.tensor([[97, 98, 99, 100]])
        model.eval()
        with torch.no_grad():
            expected_logits = model(probe, torch.zeros_like(probe))
        source_state = full_state_dict(model)
        if rank == 0:
            certify_portable(cfg, root / f'{package}-evaluation', source_state, expected_logits, probe)
        dist.barrier()

        # Recoverable invalid requests must fail collectively before live state changes.
        for label, load_kwargs in [('config', {'expected_config': {'wrong': True}}),
                                   ('manifest', {'expected_manifest_hash': 'wrong'})]:
            before = copied(model.state_dict())
            try:
                load_distributed_checkpoint(path, model, optimizer, sampler, **load_kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{label} mismatch accepted')
            identical(before, model.state_dict())
        try:
            save_distributed_checkpoint(path, model, optimizer, sampler, config=settings,
                progress={'steps': 2}, manifest_hash=manifest_hash(manifest))
        except ValueError:
            pass
        else:
            raise AssertionError('overwrite accepted')
        incompatible = RankPackedSampler(manifest, rank=rank, world_size=world, context=4,
                                         seed=7 if rank == 1 else 3)
        before = copied(model.state_dict())
        try:
            load_distributed_checkpoint(path, model, optimizer, incompatible)
        except ValueError as error:
            assert 'sampler' in str(error)
        else:
            raise AssertionError('rank-local incompatible sampler accepted')
        identical(before, model.state_dict())
        if rank == 0:
            print(f'{package}: world={world} exact next-batch/RNG/optimizer/update replay', flush=True)

    # One-rank input errors are collective validation failures, not deadlocks.
    before_cursor = sampler.state_dict()
    if rank == 1:
        sampler.next_batch()
    try:
        save_distributed_checkpoint(root / 'divergent-stream', model, optimizer, sampler,
            config=settings, progress={'steps': 2}, manifest_hash=manifest_hash(manifest))
    except ValueError as error:
        assert 'global sampler stream' in str(error)
    else:
        raise AssertionError('divergent global row cursors accepted')
    sampler.load_state_dict(before_cursor)
    invalid_optimizer = torch.optim.SGD(model.parameters(), lr=.01) if rank == 1 else optimizer
    try:
        save_distributed_checkpoint(root / 'unsupported', model, invalid_optimizer, sampler,
                                    config={}, progress={}, manifest_hash='abc')
    except ValueError as error:
        assert 'AdamW' in str(error)
    else:
        raise AssertionError('unsupported optimizer accepted')
    parameter = next(model.parameters())
    if rank == 1:
        parameter.grad = torch.zeros_like(parameter)
    try:
        save_distributed_checkpoint(root / 'pending-grad', model, optimizer, sampler,
                                    config={}, progress={}, manifest_hash='abc')
    except ValueError as error:
        assert 'cleared gradients' in str(error)
    else:
        raise AssertionError('pending gradients accepted')
    optimizer.zero_grad(set_to_none=True)

    # Validate local sharded numeric storage and replicated normalization state.
    local_parameter = parameter.to_local() if hasattr(parameter, 'to_local') else parameter
    original_parameter = local_parameter.detach().clone()
    if rank == 1:
        with torch.no_grad():
            local_parameter.reshape(-1)[0] = float('nan')
    try:
        save_distributed_checkpoint(root / 'nonfinite', model, optimizer, sampler,
                                    config={}, progress={}, manifest_hash='abc')
    except ValueError as error:
        assert 'nonfinite' in str(error)
    else:
        raise AssertionError('nonfinite local shard accepted')
    finally:
        with torch.no_grad():
            local_parameter.copy_(original_parameter)
    norm = lagged_modules(model)[0]
    old_mean = norm.mean_sq.clone()
    if rank == 1:
        norm.mean_sq.add_(1)
    try:
        save_distributed_checkpoint(root / 'divergent-buffers', model, optimizer, sampler,
                                    config={}, progress={}, manifest_hash='abc')
    except ValueError as error:
        assert 'replicated buffers' in str(error)
    else:
        raise AssertionError('divergent replicated buffers accepted')
    finally:
        norm.mean_sq.copy_(old_mean)
    if rank == 1:
        norm.acc_count.fill_(1)
    try:
        save_distributed_checkpoint(root / 'pending-statistics', model, optimizer, sampler,
                                    config={}, progress={}, manifest_hash='abc')
    except ValueError as error:
        assert 'committed lagged statistics' in str(error)
    else:
        raise AssertionError('unfinished lagged statistics accepted')
    finally:
        norm.acc_count.zero_()

    # Shared-filesystem admission sums all local estimates; rank-zero ENOSPC
    # propagates collectively before staging creation or any serializer writes.
    import errno
    import tnlm.distributed_checkpoint as checkpoint_module
    original_admission = checkpoint_module.require_checkpoint_space
    original_dcp_save = checkpoint_module.dcp.save
    original_torch_save = torch.save
    def no_space(directory, required):
        assert rank == 0 and required > 2 * 1024 * 1024
        raise OSError(errno.ENOSPC, 'injected disk admission ENOSPC')
    def forbidden_write(*args, **kwargs):
        raise AssertionError('serializer reached before admission')
    checkpoint_module.require_checkpoint_space = no_space
    checkpoint_module.dcp.save = forbidden_write
    torch.save = forbidden_write
    try:
        try:
            save_distributed_checkpoint(root / 'no-space-parent' / 'saved', model, optimizer, sampler,
                                        config={}, progress={}, manifest_hash='abc')
        except ValueError as error:
            assert 'disk admission' in str(error) and 'ENOSPC' in str(error)
        else:
            raise AssertionError('ENOSPC checkpoint accepted')
    finally:
        checkpoint_module.require_checkpoint_space = original_admission
        checkpoint_module.dcp.save = original_dcp_save
        torch.save = original_torch_save
    dist.barrier()
    assert not (root / 'no-space-parent').exists()
    admission = json.loads((root / 'lagged' / 'manifest.json').read_text())['disk_admission']
    assert admission['required_bytes'] >= sum(p.stat().st_size for p in (root / 'lagged').rglob('*') if p.is_file())

    # Inject a failure after DCP shards finish but before all rank files finish.
    # No completion manifest/destination may become visible.
    original_save = torch.save
    def fail_rank_file(value, stream, *args, **kwargs):
        if rank == 1 and str(getattr(stream, 'name', '')).endswith('rank-00001.pt'):
            raise OSError('injected rank-file write failure')
        return original_save(value, stream, *args, **kwargs)
    torch.save = fail_rank_file
    try:
        try:
            save_distributed_checkpoint(root / 'incomplete', model, optimizer, sampler,
                                        config={}, progress={}, manifest_hash='abc')
        except ValueError as error:
            assert 'rank-file write failure' in str(error)
        else:
            raise AssertionError('partial checkpoint published')
    finally:
        torch.save = original_save
    dist.barrier()
    assert not (root / 'incomplete').exists()
    assert not list(root.glob('.incomplete.tmp-*'))

    # Version/world/layout mismatches are rejected before loading any live tensor.
    manifest_path = root / 'lagged' / 'manifest.json'
    original_manifest = manifest_path.read_text()
    for field, changed in [('world_size', 3), ('torch_version', 'wrong'),
                           ('layout', {})]:
        if rank == 0:
            metadata = json.loads(original_manifest)
            metadata[field] = changed
            manifest_path.write_text(json.dumps(metadata))
        dist.barrier()
        before = copied(model.state_dict())
        try:
            load_distributed_checkpoint(root / 'lagged', model, optimizer, sampler)
        except ValueError as error:
            assert 'mismatch' in str(error)
        else:
            raise AssertionError(f'{field} mismatch accepted')
        identical(before, model.state_dict())
    if rank == 0:
        manifest_path.write_text(original_manifest)
    dist.barrier()

    # A manifest buffer-identity mismatch must be rejected before installing
    # any staged model/optimizer/sampler/RNG state. Live model is at step2 while
    # this checkpoint is step1, making accidental installation observable.
    before_model = copied(model.state_dict())
    before_optimizer = copied(optimizer.state_dict())
    before_sampler = sampler.state_dict()
    from tnlm.checkpoint import _rng_state
    before_rng = _rng_state()
    if rank == 0:
        metadata = json.loads(original_manifest)
        metadata['replicated_buffers_sha256'] = '0' * 64
        manifest_path.write_text(json.dumps(metadata))
    dist.barrier()
    try:
        load_distributed_checkpoint(root / 'lagged', model, optimizer, sampler)
    except ValueError as error:
        assert 'staged replicated buffers' in str(error)
    else:
        raise AssertionError('invalid replicated-buffer identity accepted')
    identical(before_model, model.state_dict())
    identical(before_optimizer, optimizer.state_dict())
    assert before_sampler == sampler.state_dict()
    after_rng = _rng_state()
    assert before_rng['python'] == after_rng['python']
    assert before_rng['numpy'][0] == after_rng['numpy'][0]
    assert np.array_equal(before_rng['numpy'][1], after_rng['numpy'][1])
    assert before_rng['numpy'][2:] == after_rng['numpy'][2:]
    assert torch.equal(before_rng['torch'], after_rng['torch'])
    assert before_rng['cuda'] == after_rng['cuda'] is None
    if rank == 0:
        manifest_path.write_text(original_manifest)
        print('buffer identity mismatch: model/optimizer/sampler/RNG unchanged before installation', flush=True)
    dist.barrier()

    # Corrupt one committed rank artifact; every rank must reject before DCP load.
    if rank == 0:
        with (root / 'lagged' / 'rank-00001.pt').open('ab') as stream:
            stream.write(b'corrupt')
    dist.barrier()
    try:
        load_distributed_checkpoint(root / 'lagged', model, optimizer, sampler)
    except ValueError as error:
        assert 'checksum' in str(error)
    else:
        raise AssertionError('corruption accepted')
    # Production-dtype analysis gates use genuinely FP32 constructed, trained
    # and exported sources. FP64 export checks above remain unchanged.
    torch.set_default_dtype(torch.float32)
    for package in ('pre_rms', 'spectral_rms', 'lagged'):
        torch.manual_seed(91)
        cfg = ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4, d_mlp=12,
                          padded_vocab=50257, normalization=package,
                          final_normalization='lagged' if package == 'lagged' else 'rms', attention_backend='reference')
        model = shard_model(TensorLM(cfg), mesh)
        for unit in [model, *model.blocks]:
            unit.set_force_sum_reduction_for_comms(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=4, seed=3)
        update(sampler.next_batch())
        model.eval()
        destination = root / f'{package}-fp32-evaluation'
        export_distributed_weights(destination, model, config={'model_package': package},
            progress={'steps': 1}, manifest_hash=manifest_hash(manifest), max_state_bytes=10_000_000)
        source_state = full_state_dict(model)
        with torch.no_grad():
            expected_logits = model(probe, torch.zeros_like(probe))
        # Convert rank-zero diagnostic errors into an agreed failure; no peer
        # waits indefinitely for a bare rank-zero exception before a barrier.
        from tnlm.distributed_checkpoint import _phase
        _phase('FP32 exported analysis certification', lambda: certify_portable(cfg, destination,
               source_state, expected_logits, probe, analysis_path=root / f'{package}-fp32-analysis') if rank == 0 else None)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
