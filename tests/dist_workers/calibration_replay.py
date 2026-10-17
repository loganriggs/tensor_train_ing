"""Two-rank early-exit calibration, collective failure cleanup and retry gates."""
from datetime import timedelta
import json
from pathlib import Path
import sys
import time
import warnings

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tnlm.calibration import calibrate_lagged
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import PackedSampler, prepare_data
from tnlm.model import TensorLM
from tnlm.normalization import lagged_modules
from tnlm.parallel import full_state_dict, global_count
from tnlm.runtime import seed_all
from tnlm.sharded_initialization import sharded_model_from_cpu_components


class Tokenizer:
    name, eot_token = 'offline-calibration', 50256
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def clean(model, mode=True):
    assert model.training is mode
    for norm in lagged_modules(model):
        assert not norm.collecting and norm._valid is None
        assert float(norm.acc_count) == 0 and float(norm.acc_sum) == 0
        assert not norm._forward_hooks


def state_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def main():
    warnings.filterwarnings('ignore', message='FSDP2-wrapped module .* returned a view tensor')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(sys.argv[1])
    if rank == 0:
        source = root / 'prefix.jsonl'
        source.write_text(json.dumps({'id': 'deep-prefix', 'text': ''.join(chr(i) for i in range(24))}) + '\n')
        prepare_data(root / 'data', source_jsonl=source, revision='offline-collective-cal-v1', tokenizer=Tokenizer(),
                     split_fractions={'train': 0., 'val': 0., 'test': 0., 'calibration': 1.})
    dist.barrier()
    manifest = root / 'data' / 'manifest.json'
    cfg = Config(model=ModelConfig(layers=8, d_model=32, heads=2, d_key=8, d_value=16, d_mlp=48,
                          padded_vocab=50257, normalization='lagged', final_normalization='lagged', attention_backend='reference'),
                 train=TrainConfig(compile=False, bf16_compute=False, seed=101),
                 raw={'data': {'context': 8, 'calibration_tokens': 8}})
    mesh = init_device_mesh('cpu', (world,))
    seed_all(101)
    serial = TensorLM(cfg.model).train()
    probe, docs, targets = PackedSampler(manifest, split='calibration', context=8).next_batch()
    with torch.no_grad():
        assert not torch.isfinite(serial(probe, docs)).all()  # actual depth regression
    serial_report = calibrate_lagged(serial, cfg, manifest, torch.device('cpu'),
                                    deadline=time.monotonic() + 60, distributed=False)
    serial.train()
    seed_all(101)
    model = sharded_model_from_cpu_components(cfg.model, mesh, torch.device('cpu')).train()
    before = full_state_dict(model)
    report = calibrate_lagged(model, cfg, manifest, torch.device('cpu'), deadline=time.monotonic() + 60,
                             distributed=False, collective_control=True)
    clean(model)
    assert report['tokens'] == 8 and report['passes'] == 17
    assert report['batch_hashes'] == serial_report['batch_hashes']
    assert report['sites'] == serial_report['sites']
    state = full_state_dict(model)
    if rank == 0:
        state_equal(state, serial.state_dict())
        for name, parameter in model.named_parameters():
            assert torch.equal(state[name], before[name]), name
    with torch.no_grad():
        actual = model(probe, docs)
        expected = serial(probe, docs)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    reference_optimizer = torch.optim.AdamW(serial.parameters(), lr=.001)
    _, loss, count = model(probe, docs, targets, reduction='sum')
    (loss / global_count(count)).backward()
    _, ref_loss, ref_count = serial(probe, docs, targets, reduction='sum')
    (ref_loss / ref_count).backward()
    optimizer.step()
    reference_optimizer.step()
    updated = full_state_dict(model)
    if rank == 0:
        for name, value in serial.state_dict().items():
            torch.testing.assert_close(updated[name], value, rtol=1e-6, atol=1e-7)
        print('17-site deep overflow calibration: exact state, unchanged weights, full forward/backward/update passed', flush=True)

    original_batch = PackedSampler.next_batch
    for failure in ('rank_deadline', 'invalid_token', 'batch_mismatch'):
        seed_all(101)
        model = sharded_model_from_cpu_components(cfg.model, mesh, torch.device('cpu')).train()
        before = full_state_dict(model)
        calls, forwards = [0], [0]
        def count_forward(module, args):
            forwards[0] += 1
        handle = model.register_forward_pre_hook(count_forward)
        def injected_batch(self):
            batch = original_batch(self)
            calls[0] += 1
            # Fail on the second site after one statistic has been committed,
            # proving rollback rather than merely empty-prefix cleanup.
            if rank == 1 and calls[0] == 2:
                batch[0][0, 0] = cfg.model.vocab_size if failure == 'invalid_token' else (int(batch[0][0, 0]) + 1)
            return batch
        if failure != 'rank_deadline':
            PackedSampler.next_batch = injected_batch
        try:
            try:
                calibrate_lagged(model, cfg, manifest, torch.device('cpu'),
                    deadline=time.monotonic() - 1 if failure == 'rank_deadline' and rank == 1 else time.monotonic() + 60,
                    distributed=False, collective_control=True)
            except (ValueError, TimeoutError) as error:
                error_text = str(error)
            else:
                raise AssertionError(f'{failure} accepted')
        finally:
            PackedSampler.next_batch = original_batch
            handle.remove()
        clean(model)
        assert forwards[0] == (0 if failure == 'rank_deadline' else 1), (failure, forwards[0])
        after = full_state_dict(model)
        if rank == 0:
            state_equal(before, after)
        # A recoverable pre-forward error leaves the live group/model usable.
        retried = calibrate_lagged(model, cfg, manifest, torch.device('cpu'),
                      deadline=time.monotonic() + 60, distributed=False, collective_control=True)
        clean(model)
        assert retried['sites'] == serial_report['sites']
        with torch.no_grad():
            assert torch.isfinite(model(probe, docs)).all()
        if rank == 0:
            print(f'{failure}: rejected before offending forward, state rollback, mode/hooks/stat cleanup and retry passed; {error_text}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
