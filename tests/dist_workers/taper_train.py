"""Small real FSDP lifecycle test with an explicitly compressed two-step warmup."""
from datetime import timedelta
import importlib
import json
from pathlib import Path
import sys
import warnings

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tnlm.config import Config, ModelConfig, TrainConfig
from tnlm.data import PackedSampler, RankPackedSampler, prepare_data
from tnlm.distributed_checkpoint import load_distributed_checkpoint
from tnlm.model import TensorLM
from tnlm.parallel import full_state_dict, shard_model
from tnlm.train import _mask_to_count, make_optimizer
from tnlm.runtime import seed_all
from tnlm.training_taper import TaperTraining
from train_replay import flatten_equal


def main():
    warnings.filterwarnings('ignore', message='FSDP2-wrapped module .* returned a view tensor')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    root = Path(sys.argv[1])
    class Tokenizer:
        name, eot_token = 'taper-fixture', 50256
        def encode(self, text, **kwargs):
            return list(text.encode())
    if rank == 0:
        source = root / 'source.jsonl'
        source.write_text(''.join(json.dumps({'id': str(i), 'text': ['ab', 'abcde', 'abcdefg'][i % 3]})+'\n' for i in range(30)))
        prepare_data(root / 'data', source_jsonl=source, revision='taper-test', tokenizer=Tokenizer(),
                     split_fractions={'train': 1., 'val': 0., 'test': 0., 'calibration': 0.})
    dist.barrier()
    trainer = importlib.import_module('tnlm.distributed_train')
    real_validate = trainer.validate_schedule
    def short_warmup(*args, **kwargs):
        real_validate(*args, **kwargs)
        return 2
    trainer.validate_schedule = short_warmup
    manifest = root / 'data/manifest.json'
    cfg = Config(model=ModelConfig(layers=1, d_model=4, heads=1, d_key=2, d_value=2, d_mlp=8,
                     padded_vocab=50257, normalization='taper', final_normalization='taper', attention_backend='reference'),
                 train=TrainConfig(compile=False, bf16_compute=False, lr=.001, taper_anchor_lambda=.1,
                     global_prediction_tokens_per_update=5, rung_total_tokens=22),
                 raw={'data': {'manifest': str(manifest), 'context': 3, 'micro_batch_size': 1},
                      'project': {'dry_run': False, 'gpu_hourly_rate_usd': 0., 'spend_ledger': str(root/'spend.json'),
                                  'keep_recovery_checkpoints': 0, 'save_initial_checkpoint': False,
                                  'milestone_tokens': [5, 10, 20]}})
    run = trainer.distributed_train_run
    whole = run(cfg, root/'whole', max_steps=5, max_seconds=120)
    assert whole['progress']['steps'] == 5 and whole['progress']['tokens'] == 22
    assert [r['taper_gate'] for r in whole['history']][:2] == [1., 1.]
    assert whole['history'][-1]['taper_gate'] == 0.
    assert whole['history'][2]['taper_gate_used'] == 1.
    assert whole['history'][2]['taper_anchor'] > 0.
    assert whole['operator_audit']['class'] == 'P', whole['operator_audit']
    assert any(0 in rd['selected_counts'] for row in whole['history'] for rd in row['rounds'])
    assert any(len(set(rd['selected_counts'])) > 1 for row in whole['history'] for rd in row['rounds'])
    assert 'taper_constants' in whole['history'][1]
    mesh = init_device_mesh('cpu', (dist.get_world_size(),))
    def state(result):
        model = shard_model(TensorLM(cfg.model), mesh)
        for unit in [model, *model.blocks]:
            unit.set_force_sum_reduction_for_comms(True)
        opt = make_optimizer(model, cfg.train)
        sampler = RankPackedSampler(manifest, rank=rank, world_size=dist.get_world_size(), context=3, seed=cfg.train.seed)
        payload = load_distributed_checkpoint(result['checkpoint'], model, opt, sampler)
        values = (full_state_dict(model), get_optimizer_state_dict(model, opt,
                  options=StateDictOptions(full_state_dict=True, cpu_offload=True)), sampler.state_dict(), payload['progress'])
        return values
    expected = state(whole)
    # Independent global-batch objective: token means across both ranks before
    # weighting by this round's share of the global update quota.
    seed_all(cfg.train.seed)
    reference = TensorLM(cfg.model)
    opt = make_optimizer(reference, cfg.train)
    serial = PackedSampler(manifest, batch_size=dist.get_world_size(), context=3, seed=cfg.train.seed)
    taper = TaperTraining(reference, cfg, total_steps=5, warmup_steps=2, distributed=False)
    for step, wanted in enumerate((5, 5, 5, 5, 2)):
        taper.before_step(step)
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group['lr'] = cfg.train.lr * min(1., (step + 1) / 2)
        accumulated = 0
        anchor_sum = ce_sum = 0.
        while accumulated < wanted:
            idx, docs, targets = serial.next_batch()
            targets = _mask_to_count(targets, wanted - accumulated)
            selected = int((targets != -100).sum())
            if not selected:
                continue
            taper.set_valid_mask(docs >= 0 if taper.collecting else targets != -100)
            _, loss, _ = reference(idx, docs, targets, reduction='sum')
            anchor = taper.anchor()
            objective = loss / wanted
            if anchor is not None:
                objective = objective + cfg.train.taper_anchor_lambda * anchor * selected / wanted
                anchor_sum += float(anchor.detach()) * selected
            objective.backward()
            taper.set_valid_mask(None)
            ce_sum += float(loss.detach())
            accumulated += selected
        torch.nn.utils.clip_grad_norm_(reference.parameters(), cfg.train.grad_clip)
        opt.step()
        taper.after_step(step + 1)
        torch.testing.assert_close(torch.tensor(whole['history'][step]['taper_anchor']),
                                   torch.tensor(anchor_sum / wanted), rtol=2e-4, atol=1e-7)
        torch.testing.assert_close(torch.tensor(whole['history'][step]['loss']),
                                   torch.tensor(ce_sum / wanted), rtol=2e-6, atol=1e-6)
    if rank == 0:
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, expected[0][name], rtol=2e-4, atol=2e-6, msg=lambda m: name + ': ' + m)
        print('Taper distributed selected-target anchor agrees with serial global objective', flush=True)
    for boundary in (5, 10):
        resumed = run(cfg, root/f'resume{boundary}', max_steps=5, max_seconds=120,
                      resume=root/'whole'/f'milestone-{boundary:012d}')
        actual = state(resumed)
        if rank == 0:
            flatten_equal(expected, actual)
        assert resumed['history'][-1]['taper_gate'] == 0.
    parent = root/'whole'/'milestone-000000000010'
    cool = run(cfg, root/'cool', max_steps=3, max_seconds=120, resume=parent,
               cooldown_steps=3, cooldown_end_tokens=22)
    first = run(cfg, root/'cool-first', max_steps=1, max_seconds=120, resume=parent,
                cooldown_steps=3, cooldown_end_tokens=22)
    rest = run(cfg, root/'cool-rest', max_steps=2, max_seconds=120, resume=first['checkpoint'],
               cooldown_steps=3, cooldown_end_tokens=22)
    assert cool['history'][0]['taper_gate_used'] == whole['history'][2]['taper_gate_used']
    assert [r['taper_gate'] for r in cool['history']] == [r['taper_gate'] for r in whole['history'][2:]]
    left, right = state(cool), state(rest)
    if rank == 0:
        flatten_equal(left, right)
        print('Taper distributed freeze, final polynomial gate, exact resume, and cooldown passed', flush=True)
    cfg.train.taper_anchor_lambda = 0.
    zero = run(cfg, root/'zero-anchor', max_steps=5, max_seconds=120)
    assert zero['history'][2]['taper_anchor'] > 0.
    assert zero['history'][-1]['taper_gate'] == 0.
    cfg.train.taper_end_fraction = .2
    try:
        run(cfg, root/'bad-endpoint', max_steps=1, max_seconds=120)
    except ValueError as error:
        assert 'endpoint' in str(error)
    else:
        raise AssertionError('Taper endpoint before warmup accepted')
    assert not (root/'bad-endpoint').exists()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
