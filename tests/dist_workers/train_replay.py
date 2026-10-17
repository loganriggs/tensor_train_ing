"""CPU/gloo FSDP2 trainer: serial global objective and exact interrupted replay."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import sys
import warnings

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tnlm.calibration import calibrate_lagged
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import PackedSampler, RankPackedSampler, prepare_data
from tnlm.distributed_checkpoint import load_distributed_checkpoint
from tnlm.distributed_train import distributed_train_run
from tnlm.model import TensorLM
from tnlm.normalization import lagged_modules, spectral_step
from tnlm.parallel import full_state_dict, shard_model
from tnlm.runtime import seed_all
from tnlm.train import _mask_to_count, make_optimizer


class Tokenizer:
    name, eot_token = 'offline-distributed-train', 50256
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def flatten_equal(left, right):
    if isinstance(left, torch.Tensor):
        a = left.to_local() if hasattr(left, 'to_local') else left
        b = right.to_local() if hasattr(right, 'to_local') else right
        assert torch.equal(a, b)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            flatten_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            flatten_equal(a, b)
    else:
        assert left == right


def main():
    warnings.filterwarnings('ignore', message='FSDP2-wrapped module .* returned a view tensor')
    import time
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(sys.argv[1])
    if rank == 0:
        source = root / 'source.jsonl'
        source.write_text(''.join(json.dumps({'id': str(i), 'text': ['ab', 'abcd', 'abcdefg'][i % 3]}) + '\n' for i in range(30)))
        prepare_data(root / 'data', source_jsonl=source, revision='offline-trainer-v1', tokenizer=Tokenizer(),
                     split_fractions={'train': .8, 'val': 0., 'test': 0., 'calibration': .2})
    dist.barrier()
    manifest = root / 'data' / 'manifest.json'
    mesh = init_device_mesh('cpu', (world,))
    clipping_observations = []
    original_clip = torch.nn.utils.clip_grad_norm_
    def observed_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        square_sum = torch.zeros((), dtype=torch.float64)
        sharded = False
        for parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if hasattr(gradient, 'to_local'):
                sharded = True
                assert len(gradient.placements) == 1 and gradient.placements[0].is_shard()
                gradient = gradient.to_local()
            square_sum += gradient.double().square().sum()
        if sharded:
            dist.all_reduce(square_sum)
        exact_norm = float(square_sum.sqrt())
        result = original_clip(parameters, *args, **kwargs)
        clipping_observations.append(exact_norm)
        return result
    torch.nn.utils.clip_grad_norm_ = observed_clip
    for package in ('pre_rms', 'spectral_rms', 'lagged', 'pre_rms_fp32'):
        torch.set_default_dtype(torch.float32 if package.endswith('_fp32') else torch.float64)
        normalization = package.removesuffix('_fp32')
        cfg = Config(model=ModelConfig(layers=1, d_model=8, heads=1, d_key=4, d_value=4, d_mlp=12,
                          padded_vocab=50257, normalization=normalization,
                          final_normalization='lagged' if package == 'lagged' else 'rms', attention_backend='reference'),
                     train=TrainConfig(compile=False, bf16_compute=False, lr=.01, grad_clip=.01,
                          global_prediction_tokens_per_update=7, rung_total_tokens=8),
                     raw={'data': {'manifest': str(manifest), 'context': 3, 'micro_batch_size': 1, 'calibration_tokens': 8},
                          'project': {'dry_run': False, 'gpu_hourly_rate_usd': .01,
                                      'spend_ledger': str(root / 'spend.json'), 'checkpoint_every_steps': 1,
                                      'milestone_tokens': [7, 7]}})
        if package in ('pre_rms', 'spectral_rms'):
            cfg.raw['project'].update(keep_recovery_checkpoints=1 if package == 'pre_rms' else 0,
                                      save_initial_checkpoint=False)
        whole = distributed_train_run(cfg, root / f'{package}-whole', max_steps=2, max_seconds=120,
                    export_max_state_bytes=10_000_000 if package.endswith('_fp32') else None)
        assert (whole['evaluation_checkpoint'] is not None) == package.endswith('_fp32')
        if package in ('pre_rms', 'spectral_rms'):
            assert not (root / f'{package}-whole' / 'initial').exists()
            recoveries = sorted((root / f'{package}-whole').glob('recovery-*'))
            assert len(recoveries) == (1 if package == 'pre_rms' else 0)
            if recoveries:
                assert recoveries[0].name == 'recovery-00000002'
        assert (root / f'{package}-whole' / 'milestone-000000000007').is_dir()
        assert (root / f'{package}-whole' / 'branchpoint-000000000007').is_dir()
        assert whole['checkpoint_storage_plan']['count_breakdown']['milestones'] == 1
        assert whole['checkpoint_storage_plan']['count_breakdown']['separate_90_percent_branch'] == 1
        distributed_exact_norms = clipping_observations[-2:]
        first = distributed_train_run(cfg, root / f'{package}-first', max_steps=1, max_seconds=120)
        resumed = distributed_train_run(cfg, root / f'{package}-resumed', max_steps=1, max_seconds=120,
                                        resume=first['checkpoint'])
        assert whole['progress'] == resumed['progress']
        assert whole['progress']['tokens'] == 8 and whole['progress']['steps'] == 2
        assert [row['prediction_tokens'] for row in whole['history']] == [7, 1]
        assert len(whole['history'][0]['rounds']) >= 2  # true gradient accumulation
        assert any(any(value == 0 for value in row['selected_counts']) for step in whole['history'] for row in step['rounds'])
        assert any(len(set(row['raw_counts'])) > 1 for step in whole['history'] for row in step['rounds'])
        assert all(row['grad_norm'] > cfg.train.grad_clip for row in whole['history'])
        assert whole['history'][1]['rounds'] == resumed['history'][0]['rounds']
        loaded = []
        for result in (whole, resumed):
            seed_all(cfg.train.seed)
            model = shard_model(TensorLM(cfg.model), mesh)
            optimizer = make_optimizer(model, cfg.train)
            sampler = RankPackedSampler(manifest, rank=rank, world_size=world, context=3, seed=cfg.train.seed)
            load_distributed_checkpoint(result['checkpoint'], model, optimizer, sampler)
            loaded.append((full_state_dict(model), get_optimizer_state_dict(model, optimizer,
                            options=StateDictOptions(full_state_dict=True, cpu_offload=True))))
        if rank == 0:
            flatten_equal(loaded[0], loaded[1])

        # Independent serial reference consumes identical global rows and selects
        # the same quota in rank order; it never uses the distributed trainer.
        seed_all(cfg.train.seed)
        reference = TensorLM(cfg.model)
        if lagged_modules(reference):
            calibrate_lagged(reference, cfg, manifest, torch.device('cpu'),
                             deadline=time.monotonic() + 60, distributed=False)
        reference.train()
        optimizer = make_optimizer(reference, cfg.train)
        sampler = PackedSampler(manifest, batch_size=world, context=3, seed=cfg.train.seed)
        reference_norms = []
        reference_exact_norms = []
        for step, wanted in enumerate((7, 1)):
            for group in optimizer.param_groups:
                group['lr'] = cfg.train.lr * (step + 1) / 100
            optimizer.zero_grad(set_to_none=True)
            spectral_step(reference)
            accumulated = 0
            while accumulated < wanted:
                idx, docs, targets = sampler.next_batch()
                targets = _mask_to_count(targets, wanted - accumulated)
                n = int((targets != -100).sum())
                for norm in lagged_modules(reference):
                    norm.collecting = True
                    norm.set_valid_mask(docs >= 0)
                _, loss, _ = reference(idx, docs, targets, reduction='sum')
                (loss / wanted).backward()
                for norm in lagged_modules(reference):
                    norm.collecting = False
                    norm.set_valid_mask(None)
                accumulated += n
            reference_norms.append(float(torch.nn.utils.clip_grad_norm_(reference.parameters(), cfg.train.grad_clip)))
            reference_exact_norms.append(clipping_observations[-1])
            optimizer.step()
            for norm in lagged_modules(reference):
                norm.commit(distributed=False)
        ref_optimizer = get_optimizer_state_dict(reference, optimizer, options=StateDictOptions(full_state_dict=True))
        if rank == 0:
            parameter_error = max(float((value - loaded[0][0][name]).abs().max())
                                  for name, value in reference.state_dict().items() if value.is_floating_point())
            moment_error = max(float((value - loaded[0][1]['state'][name][key]).abs().max())
                               for name, values in ref_optimizer['state'].items()
                               for key, value in values.items())
            print(f'{package}: max parameter error={parameter_error:.3e}, max AdamW state error={moment_error:.3e}', flush=True)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(value, loaded[0][0][name], rtol=2e-6 if package.endswith('_fp32') else 1e-10,
                                           atol=2e-7 if package.endswith('_fp32') else 1e-12)
            for name, values in ref_optimizer['state'].items():
                for key, value in values.items():
                    # FP32 GEMM/reduction ordering differs between batched serial and
                    # rank-local math; keep FP64 gates tight and bound FP32 moments.
                    torch.testing.assert_close(value, loaded[0][1]['state'][name][key],
                        rtol=1e-5 if package.endswith('_fp32') else 1e-10,
                        atol=1e-8 if package.endswith('_fp32') else 1e-12)
            assert len(reference_norms) == len(whole['history']) == len(reference_exact_norms) == len(distributed_exact_norms) == 2
            for step, (expected, record, exact_serial, exact_distributed) in enumerate(zip(
                    reference_norms, whole['history'], reference_exact_norms, distributed_exact_norms)):
                absolute = abs(expected - record['grad_norm'])
                relative = absolute / max(abs(expected), 1e-30)
                print(f'{package} step{step}: serial clip={expected:.12g}, distributed clip={record["grad_norm"]:.12g}, '
                      f'abs={absolute:.3e}, rel={relative:.3e}, serial FP64 sum={exact_serial:.12g}, '
                      f'distributed FP64 sum={exact_distributed:.12g}', flush=True)
                if package.endswith('_fp32'):
                    # FP32 norm reduction over vocabulary matrices loses more
                    # precision than rank sums. Compare actual gradients via an
                    # independent FP64 squared-sum reduction and separately cap
                    # the observed standard norm error (measured <=1.7e-5 relative).
                    assert abs(exact_serial - exact_distributed) <= 1e-6 * exact_serial + 1e-7
                    assert abs(expected - exact_serial) <= 3e-5 * exact_serial
                    assert abs(record['grad_norm'] - exact_distributed) <= 3e-5 * exact_distributed
                else:
                    assert abs(exact_serial - exact_distributed) <= 1e-10 * exact_serial + 1e-12
                    assert absolute <= 1e-10 * expected + 1e-12
            print(f'{package}: serial global normalization/clipping and exact distributed resume passed', flush=True)

    # Cooldown branches preserve their saved base LR and schedule after interruption.
    cfg.train.rung_total_tokens = 100
    parent = distributed_train_run(cfg, root / 'cool-parent', max_steps=1, max_seconds=120)
    cooldown_endpoint = parent['progress']['tokens'] + 2 * cfg.train.global_prediction_tokens_per_update - 1
    cool_whole = distributed_train_run(cfg, root / 'cool-whole', max_steps=2, max_seconds=120,
                                      resume=parent['checkpoint'], cooldown_steps=2, cooldown_end_tokens=cooldown_endpoint)
    cool_first = distributed_train_run(cfg, root / 'cool-first', max_steps=1, max_seconds=120,
                                      resume=parent['checkpoint'], cooldown_steps=2, cooldown_end_tokens=cooldown_endpoint)
    cool_resumed = distributed_train_run(cfg, root / 'cool-resumed', max_steps=1, max_seconds=120,
                                        resume=cool_first['checkpoint'], cooldown_steps=2, cooldown_end_tokens=cooldown_endpoint)
    assert cool_whole['progress']['tokens'] == cooldown_endpoint
    assert cool_whole['training_descriptor']['cooldown_status'] == 'cooled'
    assert cool_whole['training_descriptor']['completion_status'] == 'complete'
    assert cool_first['training_descriptor']['completion_status'] == 'partial'
    assert cool_whole['training_descriptor']['segment_target_prediction_tokens'] == cooldown_endpoint
    assert cool_whole['training_descriptor'] == cool_resumed['training_descriptor']
    assert cool_whole['progress'] == cool_resumed['progress']
    assert [row['lr'] for row in cool_whole['history']] == [row['lr'] for row in cool_first['history'] + cool_resumed['history']]
    cfg.train.compile = True
    try:
        distributed_train_run(cfg, root / 'unsupported', max_steps=1, max_seconds=120)
    except ValueError as error:
        assert 'uncompiled AdamW' in str(error)
    else:
        raise AssertionError('compile accepted')
    if rank == 0:
        ledger = json.loads((root / 'spend.json').read_text())
        assert len(ledger['runs']) == 16  # 4 cases * 3 segments + 4 cooldown segments
        assert all(row['status'] == 'complete' and row['reserved_usd'] == 0 for row in ledger['runs'].values())
        assert not (root / 'unsupported').exists()
        print('single instance ledger and cooldown replay passed', flush=True)
    # Dry runs launch no work and make no ledger/output changes.
    cfg.raw['project']['dry_run'] = True
    previous = (root / 'spend.json').read_text()
    dry = distributed_train_run(cfg, root / 'dry', max_steps=1, max_seconds=1)
    assert dry['dry_run'] and not (root / 'dry').exists()
    assert (root / 'spend.json').read_text() == previous
    cfg.raw['project']['dry_run'] = False
    cfg.train.compile = False
    # A one-rank invalid batch fails before forward, finalizes one failed bill,
    # and leaves only the valid initialization checkpoint as recovery state.
    original_batch = RankPackedSampler.next_batch
    def invalid_batch(self):
        idx, docs, targets = original_batch(self)
        if self.rank == 1:
            targets[0, 0] = cfg.model.vocab_size
        return idx, docs, targets
    RankPackedSampler.next_batch = invalid_batch
    try:
        try:
            distributed_train_run(cfg, root / 'invalid-batch', max_steps=1, max_seconds=120)
        except ValueError as error:
            assert 'target outside real vocabulary' in str(error)
        else:
            raise AssertionError('invalid rank-local target accepted')
    finally:
        RankPackedSampler.next_batch = original_batch
    if rank == 0:
        ledger = json.loads((root / 'spend.json').read_text())
        failed = ledger['runs'][str((root / 'invalid-batch').resolve())]
        assert failed['status'] == 'failed' and failed['reserved_usd'] == 0
        assert (root / 'invalid-batch' / 'initial' / 'manifest.json').is_file()
        assert not (root / 'invalid-batch' / 'final').exists()
        print('FP32/gloo export, dryrun and collective failure accounting passed', flush=True)
    # Forecast rejection is collective and precedes even the initial save.
    import errno
    import importlib
    training_module = importlib.import_module('tnlm.distributed_train')
    old_admission = training_module.require_checkpoint_space
    old_save = training_module.save_distributed_checkpoint
    def no_forecast_space(directory, required):
        assert rank == 0 and required > 0
        raise OSError(errno.ENOSPC, 'injected retention capacity failure')
    def no_initial_save(*args, **kwargs):
        raise AssertionError('checkpoint written before retention admission')
    training_module.require_checkpoint_space = no_forecast_space
    training_module.save_distributed_checkpoint = no_initial_save
    try:
        try:
            distributed_train_run(cfg, root / 'capacity-failure', max_steps=1, max_seconds=120)
        except ValueError as error:
            assert 'checkpoint retention admission' in str(error)
            assert 'injected retention capacity failure' in str(error)
        else:
            raise AssertionError('insufficient retention capacity accepted')
    finally:
        training_module.require_checkpoint_space = old_admission
        training_module.save_distributed_checkpoint = old_save
    assert not (root / 'capacity-failure' / 'initial').exists()
    if rank == 0:
        ledger = json.loads((root / 'spend.json').read_text())
        assert str((root / 'capacity-failure').resolve()) not in ledger['runs']
        assert not (root / 'capacity-failure').exists()
        print('collective retention admission rejects before checkpoint writes', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
