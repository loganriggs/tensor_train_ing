"""Collective FSDP2 training with exact global-target normalization.

All ranks call with an initialized default process group and shared filesystem.
One instance reservation is owned by rank zero. Recoverable validation failures
are collective; transport/process failures require relaunch with a new group.
Compile and Muon are explicitly unsupported here. CPU/gloo is the tested path.
Fresh runs stream canonical CPU components into local parameter shards. Lagged
calibration replays the same training-only prefix on all ranks, with collective
validation and explicit FSDP cleanup after every early-stopped forward.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from .checkpoint_plan import checkpoint_retention_policy, checkpoint_storage_plan, pending_milestone_tokens
from .checkpoint_storage import require_checkpoint_space
from .optimizer_exposure import update_optimizer_exposure
from .budget import SpendLedger
from .calibration import _check_lagged_divisor, calibrate_lagged
from .config import Config
from .data import RankPackedSampler, batch_hash, manifest_hash
from .distributed_checkpoint import (_agree, _group, _phase, load_distributed_checkpoint,
                                    save_distributed_checkpoint, export_distributed_weights)
from .initialization import CANONICAL_INITIALIZATION_VERSION, sharded_model_for_restore
from .sharded_initialization import sharded_model_from_cpu_components
from .normalization import lagged_modules, spectral_step
from .runtime import full_config, resume_validator, seed_all, record_context_fork, training_execution, determinism_metadata
from .schedules import validate_schedule, schedule_multiplier, schedule_steps
from .train import _assert_no_analysis_state, _mask_to_count, lr_multiplier, make_optimizer


@training_execution(collective=True, default_device='cpu')
def distributed_train_run(cfg, run_dir, *, max_steps, max_seconds, device='cpu',
                          resume=None, cooldown_steps=None, cooldown_end_tokens=None, run_budget_usd=None, export_max_state_bytes=None, context_fork=False):
    """Train bounded new updates; resume/cooldown uses a fresh run identity.

    RankPackedSampler slices one global stream. Each round assigns remaining
    prediction targets in rank order; every rank executes backward, including
    zero-target ranks. FSDP gradient SUM divides local summed losses by the exact
    global update quota before backward, then clips the global sharded gradient.
    """
    rank, world = _group()
    cfg = _phase('configuration load', lambda: Config.from_yaml(str(cfg)) if isinstance(cfg, (str, Path)) else cfg)
    shared = _phase('launch identity', lambda: {'config': full_config(cfg), 'run_dir': str(Path(run_dir).resolve()),
                 'max_steps': max_steps, 'max_seconds': max_seconds, 'device_type': torch.device(device).type,
                 'resume': str(Path(resume).resolve()) if resume else None, 'context_fork': context_fork, 'cooldown_steps': cooldown_steps,
                 'cooldown_end_tokens': cooldown_end_tokens,
                 'run_budget_usd': run_budget_usd, 'export_max_state_bytes': export_max_state_bytes})
    _agree('launch identity', shared)
    def context_preflight():
        if type(context_fork) is not bool or (context_fork and resume is None):
            raise ValueError('context_fork must be Boolean and requires an explicit parent checkpoint')
    _phase('context fork preflight', context_preflight)
    project, data = cfg.raw.get('project', {}), cfg.raw.get('data', {})
    def schedule_preflight():
        if cfg.train.schedule == 'cosine' and cooldown_end_tokens is not None:
            raise ValueError('cosine does not support a separate cooldown')
        return validate_schedule(cfg.train.schedule,
            schedule_steps(cfg.train.rung_total_tokens, cfg.train.global_prediction_tokens_per_update),
            cooldown_steps=cooldown_steps)
    _phase('schedule preflight', schedule_preflight)
    if project.get('dry_run', True):
        return {'dry_run': True, 'command': 'distributed_train', 'world_size': world,
                'run_dir': shared['run_dir'], 'tokens': cfg.train.rung_total_tokens,
                'optimizer': cfg.train.optimizer, 'compile_supported': False,
                'context_fork': context_fork, 'context': data.get('context', 512),
                'new_context': data.get('context', 512), 'parent': shared['resume'],
                'parent_validation': 'pending' if resume else 'not applicable'}

    parent_config = None
    if resume is not None:
        def parent_preflight():
            saved = json.loads((Path(resume) / 'manifest.json').read_text())['config']
            resume_validator(cfg, context_fork=context_fork)(saved)
            return saved
        parent_config = _phase('parent horizon/context preflight', parent_preflight)
        _agree('parent configuration', parent_config)
    old_context = parent_config.get('data', {}).get('context', 512) if context_fork else data.get('context', 512)

    def preflight():
        if export_max_state_bytes is not None and (type(export_max_state_bytes) is not int or export_max_state_bytes <= 0):
            raise ValueError('export_max_state_bytes must be a positive integer or None')
        if cfg.train.compile or cfg.train.optimizer not in ('adamw', 'muon'):
            raise ValueError('distributed trainer supports uncompiled AdamW or Muon+AdamW; compile is unsupported')
        if cfg.train.optimizer == 'muon' and cfg.train.aux_adamw_lr == 0:
            raise ValueError('zero auxiliary AdamW LR is not supported by the current Muon optimizer factory')
        if (type(max_steps) is not int or max_steps <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0 or
                cfg.train.rung_total_tokens <= 0 or cfg.train.global_prediction_tokens_per_update <= 0):
            raise ValueError('positive distributed step/time/token limits required')
        if not cfg.train.fp32_optimizer_state or not cfg.train.fp32_norm_reductions:
            raise ValueError('distributed reference requires FP32 state and norm reductions')
        if type(cfg.model.loss_chunk_tokens) is not int or cfg.model.loss_chunk_tokens < 0:
            raise ValueError('loss_chunk_tokens must be a nonnegative integer')
        if cfg.model.activation_checkpointing and 'lagged' in (cfg.model.normalization, cfg.model.final_normalization):
            raise ValueError('lagged collection cannot use activation recomputation')
        if cooldown_end_tokens is not None and (type(cooldown_end_tokens) is not int or cooldown_end_tokens <= 0 or cooldown_steps is None):
            raise ValueError('cooldown_end_tokens requires a positive integer and cooldown_steps')
        if cooldown_steps is not None and (type(cooldown_steps) is not int or cooldown_steps <= 0 or not resume):
            raise ValueError('cooldown requires positive cooldown_steps and explicit parent')
        if data.get('document_boundary_mask', True) is not True or data.get('reset_position_at_document', True) is not True:
            raise ValueError('distributed reference requires document masking and position reset')
        if int(project.get('history_limit', 1000)) <= 0:
            raise ValueError('history_limit must be positive')
        checkpoint_retention_policy(cfg)
        from .training_stability import stability_interval, qk_clip_policy
        if qk_clip_policy(cfg)[0] > 0:
            raise ValueError('QK clipping currently supports serial training only')
        if stability_interval(cfg):
            raise ValueError('periodic stability logging currently supports serial training only')
        if float(project.get('checkpoint_every_seconds', 300)) <= 0:
            raise ValueError('checkpoint interval must be positive')
        dev = torch.device(device)
        if dev.type not in ('cpu', 'cuda'):
            raise ValueError('distributed trainer supports CPU or CUDA only')
        if dev.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA requested but unavailable')
            dev = torch.device('cuda', dev.index if dev.index is not None else int(os.environ.get('LOCAL_RANK', rank)))
            torch.cuda.set_device(dev)
        if (dev.type == 'cpu' and dist.get_backend() != 'gloo') or (dev.type == 'cuda' and dist.get_backend() != 'nccl'):
            raise ValueError('use gloo for CPU and NCCL for CUDA')
        rate = project.get('gpu_hourly_rate_usd')
        if rate is None or not math.isfinite(rate) or rate < 0 or (dev.type == 'cuda' and rate <= 0):
            raise ValueError('declare actual instance gpu_hourly_rate_usd (explicit zero allowed for CPU)')
        if Path(shared['run_dir']).exists():
            raise ValueError('distributed run directory already exists')
        return dev
    dev = _phase('launch preflight', preflight)
    retention_policy = checkpoint_retention_policy(cfg)
    run_dir = Path(shared['run_dir'])
    run_id = str(run_dir)
    if resume is None:
        launch_storage_plan = _phase('checkpoint launch forecast', lambda: checkpoint_storage_plan(
            cfg, max_steps=max_steps, distributed_world_size=world, distributed=True,
            export_weights=export_max_state_bytes is not None))
        _agree('checkpoint launch forecast', launch_storage_plan)
        _phase('checkpoint retention admission', lambda: require_checkpoint_space(
            run_dir.parent, launch_storage_plan['required_additional_bytes']) if rank == 0 else None)
    ledger = SpendLedger(project.get('spend_ledger', 'runs/spend.json'), project.get('total_budget_usd', 1000),
                         project.get('reserve_usd', 150), stage_budgets=project.get('stage_budgets'))
    _phase('instance admission', lambda: ledger.begin(run_id, project['gpu_hourly_rate_usd'], max_seconds,
           run_budget_usd, parent_checkpoint=resume, cost_margin_fraction=project.get('cost_margin_fraction', .1),
           stage=project.get('stage', 'A')) if rank == 0 else None)
    started, status = time.monotonic(), 'failed'
    model, optimizer = None, None

    def max_value(value):
        tensor = torch.tensor(float(value), dtype=torch.float64, device=dev)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor)

    def broadcast(value):
        values = [value]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def expired():
        return max_value(time.monotonic() - started) >= max_seconds

    try:
        _phase('run directory creation', lambda: run_dir.mkdir(parents=True, exist_ok=False) if rank == 0 else None)
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(dev)
        sampler = _phase('rank sampler construction', lambda: RankPackedSampler(data['manifest'], rank=rank,
                         world_size=world, batch_size=data.get('micro_batch_size', 1),
                         context=old_context, seed=cfg.train.seed))
        data_hash = _phase('data identity', lambda: manifest_hash(data['manifest']))
        _agree('immutable dataset', data_hash)
        def tokenizer_check():
            tokenizer = sampler.manifest['tokenizer']
            if tokenizer.get('test_only') and dev.type != 'cpu':
                raise ValueError('test-only tokenizers are restricted to CPU fixtures')
            if data.get('tokenizer') is not None and data['tokenizer'] not in (tokenizer['name'], tokenizer.get('fingerprint_sha256')):
                raise ValueError('configured tokenizer differs from immutable dataset')
        _phase('tokenizer identity', tokenizer_check)
        if expired():
            raise TimeoutError('runtime cap during dataset setup')
        seed_all(cfg.train.seed)
        mesh = init_device_mesh(dev.type, (world,))
        streamed_fresh = not resume
        if resume:
            model = _phase('sharded restore allocation', lambda: sharded_model_for_restore(cfg.model, mesh, dev))
        else:
            model = _phase('canonical component initialization', lambda: sharded_model_from_cpu_components(cfg.model, mesh, dev))
        _phase('analysis-state guard', lambda: _assert_no_analysis_state(model))
        progress = {'steps': 0, 'tokens': 0, 'scheduler_step': 0, 'skipped_updates': [],
                    'initialization': CANONICAL_INITIALIZATION_VERSION if streamed_fresh else 'ordinary_cpu_tensorlm',
                    'context_history': [{'context': data.get('context', 512), 'start_tokens': 0, 'start_step': 0, 'source': 'fresh'}],
                    'input_tokens_seen': 0, 'discarded_prediction_targets': 0,
                    'warmup_steps': validate_schedule(cfg.train.schedule, schedule_steps(cfg.train.rung_total_tokens, cfg.train.global_prediction_tokens_per_update))}
        parent = None
        if not resume and lagged_modules(model):
            calibration = calibrate_lagged(model, cfg, data['manifest'], dev,
                deadline=started + max_seconds, distributed=False, collective_control=True)
            _agree('calibration report', calibration)
            progress['calibration'] = calibration
        if dev.type == 'cpu':
            # FP32 FSDP may choose NCCL-only PREMUL_SUM even at divide factor1.
            # Force supported SUM collectives for gloo without changing math.
            for unit in [model, *model.blocks]:
                unit.set_force_sum_reduction_for_comms(True)
        optimizer = _phase('optimizer construction', lambda: make_optimizer(model, cfg.train))
        if resume:
            def parent_check():
                metadata = json.loads((Path(resume) / 'manifest.json').read_text())
                compatibility = resume_validator(cfg, context_fork=context_fork)(metadata['config'])
                old_schedule = metadata['progress'].get('distributed_schedule', {})
                if old_schedule.get('cooldown_steps') is not None and cooldown_steps != old_schedule['cooldown_steps']:
                    raise ValueError('cooled checkpoint requires explicit matching cooldown_steps')
                if old_schedule.get('cooldown_steps') is not None and cooldown_end_tokens != old_schedule.get('cooldown_end_tokens'):
                    raise ValueError('cannot change an existing cooldown endpoint')
                return compatibility
            compatibility = _phase('parent trajectory compatibility', parent_check)
            payload = load_distributed_checkpoint(resume, model, optimizer, sampler,
                                                  expected_manifest_hash=data_hash)
            progress = payload['progress']
            if compatibility.get('defaulted_fields'):
                progress.setdefault('config_compatibility_history', []).append(
                    {'checkpoint': str(Path(resume).resolve()), **compatibility})
            parent = _phase('parent checkpoint identity', lambda: {'path': str(Path(resume).resolve()),
                      'manifest_sha256': hashlib.sha256((Path(resume) / 'manifest.json').read_bytes()).hexdigest()})
            del payload
            if context_fork:
                source_state = sampler.state_dict()['stream']
                sampler = _phase('global context stream fork', lambda: sampler.fork_context(data.get('context', 512)))
                _agree('forked global sampler stream', sampler.state_dict()['stream'])
                _phase('context history accounting', lambda: record_context_fork(progress,
                    old_context=old_context, new_context=data.get('context', 512), parent=parent,
                    source_state=source_state))
                _agree('context fork history', progress)
            _phase('restored lagged divisor validation', lambda: [_check_lagged_divisor(norm) for norm in lagged_modules(model)])
        previous = progress.get('distributed_schedule', {})
        branch_base = (previous['branch_base_lr'] if previous.get('branch_base_lr') is not None and
                       (previous.get('cooldown_steps') is not None or cooldown_steps is None)
                       else [g['lr'] for g in optimizer.param_groups])
        if cooldown_steps is not None and previous.get('cooldown_steps') is None:
            progress['scheduler_step'] = 0
            progress['cooldown_start_tokens'] = progress['tokens']
        progress['distributed_schedule'] = {'cooldown_steps': cooldown_steps, 'branch_base_lr': branch_base}
        if cooldown_end_tokens is not None:
            progress['distributed_schedule']['cooldown_end_tokens'] = cooldown_end_tokens
        norms = lagged_modules(model)
        model.train()
        token_limit = (progress['cooldown_start_tokens'] + cooldown_steps * cfg.train.global_prediction_tokens_per_update
                       if cooldown_steps is not None else cfg.train.rung_total_tokens)
        def endpoint_limit():
            if cooldown_end_tokens is None:
                return token_limit
            if not token_limit - cfg.train.global_prediction_tokens_per_update < cooldown_end_tokens <= token_limit:
                raise ValueError('cooldown endpoint must match the cooldown step horizon')
            return cooldown_end_tokens
        token_limit = _phase('cooldown endpoint validation', endpoint_limit)
        if cooldown_end_tokens is not None:
            progress['cooldown_end_tokens'] = cooldown_end_tokens
        storage_plan = _phase('checkpoint retention forecast', lambda: checkpoint_storage_plan(
            cfg, start_tokens=progress['tokens'], token_limit=token_limit, max_steps=max_steps,
            distributed_world_size=world, distributed=True, cooldown=cooldown_steps is not None,
            export_weights=export_max_state_bytes is not None))
        _agree('checkpoint retention forecast', storage_plan)
        storage_admission = _phase('checkpoint retention admission', lambda: require_checkpoint_space(
            run_dir, storage_plan['required_additional_bytes']) if rank == 0 else None)
        storage_admission = broadcast(storage_admission)
        _phase('checkpoint retention report', lambda: (run_dir / 'checkpoint_storage_plan.json').write_text(
            json.dumps({'plan': storage_plan, 'admission': storage_admission}, indent=2, allow_nan=False) + '\n')
            if rank == 0 else None)
        def save(name):
            progress['runtime_determinism'] = determinism_metadata()
            return save_distributed_checkpoint(run_dir / name, model, optimizer, sampler,
                    config=full_config(cfg), progress=progress, manifest_hash=data_hash, parent=parent)
        if expired():
            raise TimeoutError('runtime cap during model initialization or checkpoint restoration')
        write_start = time.monotonic()
        if retention_policy['save_initial_checkpoint']:
            save('initial')
        write_allowance = (3 * max_value(time.monotonic() - write_start)
                           if retention_policy['save_initial_checkpoint'] else 0.0)
        rolling = deque()
        history = deque(maxlen=int(project.get('history_limit', 1000)))
        last_recovery = time.monotonic()
        slowest = 0.
        completed_updates = completed_warmup_updates = 0
        stop_reason = 'max_steps'
        checkpoint_every = int(project.get('checkpoint_every_steps', 0))
        interval = float(project.get('checkpoint_every_seconds', 300))
        branchpoint = (9 * token_limit) // 10 if cooldown_steps is None and cfg.train.schedule != 'cosine' else None
        branchpoint_saved = branchpoint is None or progress['tokens'] >= branchpoint
        pending_milestones = list(pending_milestone_tokens(cfg, start_tokens=progress['tokens'],
            token_limit=token_limit, reachable_tokens=storage_plan['reachable_tokens']))
        total_steps = schedule_steps(cfg.train.rung_total_tokens, cfg.train.global_prediction_tokens_per_update)
        for _ in range(max_steps):
            _phase('analysis-state guard', lambda: _assert_no_analysis_state(model))
            elapsed = max_value(time.monotonic() - started)
            if elapsed + slowest + write_allowance >= max_seconds:
                stop_reason = 'runtime_cap'
                break
            if progress['tokens'] >= token_limit:
                stop_reason = 'token_limit'
                break
            step_started = time.monotonic()
            schedule_step = progress['scheduler_step']
            mult = (lr_multiplier(schedule_step, total_steps, cooldown_steps=cooldown_steps) if cooldown_steps is not None
                    else schedule_multiplier(schedule_step, total_steps, schedule=cfg.train.schedule, warmup_steps=progress['warmup_steps']))
            for group, base in zip(optimizer.param_groups, branch_base):
                group['lr'] = (base if cooldown_steps is not None else group.get('base_lr', cfg.train.lr)) * mult
            optimizer.zero_grad(set_to_none=True)
            _phase('spectral iteration', lambda: spectral_step(model))
            wanted = min(cfg.train.global_prediction_tokens_per_update, token_limit - progress['tokens'])
            prefix_at_watermark = masked_transition_targets = 0
            accumulated, local_loss, input_count, discarded, rounds, empty_rounds = 0, 0., 0, 0, [], 0
            while accumulated < wanted:
                if expired():
                    raise TimeoutError('runtime cap during collective accumulation; recover from last complete checkpoint')
                def prepare_batch():
                    batch = sampler.next_batch()
                    idx, docs, targets = (tensor.to(dev) for tensor in batch)
                    if (idx < 0).any() or (idx >= cfg.model.vocab_size).any():
                        raise ValueError('input outside real vocabulary')
                    if ((docs < 0) & (targets != -100)).any():
                        raise ValueError('padded-token targets must be -100')
                    if ((targets != -100) & ((targets < 0) | (targets >= cfg.model.vocab_size))).any():
                        raise ValueError('target outside real vocabulary')
                    return batch, idx, docs, targets
                batch, idx, docs, targets = _phase('batch validation', prepare_batch)
                if 'context_fork_exposure' in progress:
                    exposure = sampler.batch_context_exposure()
                    prefix_at_watermark += exposure['prefix_tokens_at_or_before_watermark']
                    masked_transition_targets += exposure['masked_prediction_targets']
                raw_count = int((targets != -100).sum())
                counts, hashes = [None] * world, [None] * world
                dist.all_gather_object(counts, raw_count)
                dist.all_gather_object(hashes, batch_hash(batch))
                input_count += int((docs >= 0).sum())
                remaining = wanted - accumulated
                quotas = [max(0, min(count, remaining - sum(counts[:i]))) for i, count in enumerate(counts)]
                selected = quotas[rank]
                targets = _mask_to_count(targets, selected)
                discarded += raw_count - selected
                rounds.append({'raw_counts': counts, 'selected_counts': quotas, 'batch_hashes': hashes})
                if not sum(quotas):
                    empty_rounds += 1
                    if empty_rounds > 100:
                        raise ValueError('no valid prediction targets in global stream')
                    continue
                empty_rounds = 0
                for norm in norms:
                    norm.collecting = True
                    norm.set_valid_mask(docs >= 0)
                try:
                    def forward():
                        with torch.autocast(dev.type, dtype=torch.bfloat16,
                                            enabled=cfg.train.bf16_compute and dev.type == 'cuda'):
                            return model(idx, doc_ids=docs, targets=targets, reduction='sum')
                    _, loss, n = _phase('collective forward', forward)
                    def check_loss():
                        if int(n) != selected or not bool(torch.isfinite(loss)):
                            raise FloatingPointError('invalid collective loss/count')
                    _phase('loss validation', check_loss)
                    _phase('collective backward', lambda: (loss / wanted).backward())
                finally:
                    for norm in norms:
                        norm.collecting = False
                        norm.set_valid_mask(None)
                local_loss += float(loss.detach())
                accumulated += sum(quotas)
            gradient_norm = _phase('global sharded clipping', lambda: torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), cfg.train.grad_clip, error_if_nonfinite=True))
            _phase('optimizer update', lambda: optimizer.step())
            _phase('optimizer exposure accounting', lambda: update_optimizer_exposure(progress, optimizer))
            for norm in norms:
                _phase('lagged global commit', lambda norm=norm: norm.commit())
                _phase('lagged divisor validation', lambda norm=norm: _check_lagged_divisor(norm))
            optimizer.zero_grad(set_to_none=True)
            if dev.type == 'cuda':
                torch.cuda.synchronize(dev)
            values = [local_loss, input_count, discarded]
            if 'context_fork_exposure' in progress:
                values.extend([prefix_at_watermark, masked_transition_targets])
            totals = torch.tensor(values, dtype=torch.float64, device=dev)
            dist.all_reduce(totals)
            completed_updates += 1
            completed_warmup_updates += int(cooldown_steps is None and schedule_step < progress['warmup_steps'])
            progress['steps'] += 1
            progress['tokens'] += accumulated
            progress['scheduler_step'] += 1
            progress['input_tokens_seen'] += int(totals[1])
            progress['discarded_prediction_targets'] += int(totals[2])
            if 'context_fork_exposure' in progress:
                progress['context_fork_exposure']['prefix_tokens_at_or_before_watermark'] += int(totals[3])
                progress['context_fork_exposure']['masked_prediction_targets'] += int(totals[4])
            seconds = max_value(time.monotonic() - step_started)
            slowest = max(slowest, seconds)
            run_elapsed = broadcast(time.monotonic() - started if rank == 0 else None)
            record = {'run_elapsed_seconds': run_elapsed,
                      'run_billed_usd': run_elapsed * project['gpu_hourly_rate_usd'] / 3600, 'step': progress['steps'], 'tokens': progress['tokens'], 'prediction_tokens': accumulated,
                      'loss': float(totals[0]) / accumulated, 'grad_norm': float(gradient_norm),
                      'lr': optimizer.param_groups[0]['lr'],
                      'optimizer_group_lrs': [group['lr'] for group in optimizer.param_groups],
                      'optimizer_exposure': progress['optimizer_exposure'], 'rounds': rounds, 'seconds': seconds}
            history.append(record)
            _phase('metrics append', lambda: _append_metric(run_dir, record) if rank == 0 else None)
            while pending_milestones and progress['tokens'] >= pending_milestones[0]:
                save(f'milestone-{pending_milestones.pop(0):012d}')
            if not branchpoint_saved and progress['tokens'] >= branchpoint:
                save(f'branchpoint-{progress["tokens"]:012d}')
                branchpoint_saved = True
            due = ((checkpoint_every > 0 and progress['steps'] % checkpoint_every == 0) or
                   max_value(time.monotonic() - last_recovery) >= interval)
            if retention_policy['keep_recovery_checkpoints'] and due:
                name = f'recovery-{progress["steps"]:08d}'
                write_start = time.monotonic()
                save(name)
                write_allowance = max(write_allowance, 3 * max_value(time.monotonic() - write_start))
                if rank == 0:
                    rolling.append(run_dir / name)
                def prune():
                    with ledger.locked() as ledger_data:
                        pinned = {row.get('parent_checkpoint') for row in ledger_data['runs'].values()}
                        while len(rolling) > retention_policy['keep_recovery_checkpoints']:
                            oldest = rolling.popleft()
                            if str(oldest.resolve()) not in pinned:
                                shutil.rmtree(oldest)
                _phase('recovery retention', prune if rank == 0 else lambda: None)
                last_recovery = time.monotonic()
        save('final')
        evaluation_checkpoint = None
        if export_max_state_bytes is not None:
            export_parent = _phase('evaluation source checkpoint identity', lambda: {
                'path': str(run_dir / 'final'),
                'manifest_sha256': hashlib.sha256((run_dir / 'final' / 'manifest.json').read_bytes()).hexdigest()})
            export_distributed_weights(run_dir / 'evaluation', model, config=full_config(cfg), progress=progress,
                manifest_hash=data_hash, parent=export_parent, max_state_bytes=export_max_state_bytes)
            evaluation_checkpoint = str(run_dir / 'evaluation')
        peak_allocated = int(max_value(torch.cuda.max_memory_allocated(dev))) if dev.type == 'cuda' else None
        peak_reserved = int(max_value(torch.cuda.max_memory_reserved(dev))) if dev.type == 'cuda' else None
        devices = [None] * world
        dist.all_gather_object(devices, str(dev))
        from .evaluate import _training_descriptor
        training_descriptor = _training_descriptor({'config': full_config(cfg), 'progress': progress})
        result = {'training_descriptor': training_descriptor, 'checkpoint_storage_plan': storage_plan, 'checkpoint_storage_admission': storage_admission,
                  'metrics_path': str(run_dir / 'metrics.jsonl'), 'spend_ledger': str(ledger.path.resolve()),
                  'history_limit': int(project.get('history_limit', 1000)),
                  'parameters': model.param_counts(), 'model_config': full_config(cfg)['model'], 'train_config': full_config(cfg)['train'],
                  'data': {'manifest_sha256': data_hash, 'context': data.get('context', 512),
                           'tokenizer': sampler.manifest['tokenizer']},
                  'runtime': {'determinism': determinism_metadata(), 'device': dev.type, 'devices': devices, 'torch': str(torch.__version__),
                              'backend': str(dist.get_backend()), 'world_size': world,
                              'parameter_dtype': str(next(model.parameters()).dtype),
                              'bf16_autocast': cfg.train.bf16_compute and dev.type == 'cuda',
                              'initialization': ('meta_sharded_restore' if resume else
                                  'canonical_cpu_components'),
                              'cuda_peak_scope': 'maximum rank-local peak, not sum across ranks'},
                  'cuda_peak_allocated_bytes': peak_allocated, 'cuda_peak_reserved_bytes': peak_reserved,
                  'compile_enabled': False, 'warmup_steps': progress['warmup_steps'],
                  'warmup_fraction_of_completed_updates': completed_warmup_updates / completed_updates if completed_updates else None,
                  'warmup_dominates_completed_segment': completed_warmup_updates > .1 * completed_updates,
                  'checkpoint': str(run_dir / 'final'), 'evaluation_checkpoint': evaluation_checkpoint, 'progress': progress, 'history': list(history),
                  'world_size': world, 'run_id': run_id, 'parent': parent, 'stop_reason': stop_reason,
                  'operator_audit': model.operator_audit(), 'gradient_reduction': 'SUM',
                  'normalization': 'local summed loss / exact global prediction-token quota before backward',
                  'billing_rate_scope': 'gpu_hourly_rate_usd is aggregate hourly cost of the whole distributed instance/job',
                  'runtime_scope': 'collective-boundary caps; individual operations cannot be preempted; transport failures require relaunch'}
        _phase('result serialization', lambda: json.dumps(result, allow_nan=False))
        status = 'complete'
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if model is not None:
            for norm in lagged_modules(model):
                norm.collecting = False
                norm.set_valid_mask(None)
        _phase('instance billing finalization', lambda: ledger.finish(run_id, time.monotonic() - started, status) if rank == 0 else None)
    billing = _phase('billing read', lambda: json.loads(ledger.path.read_text())['runs'][run_id] if rank == 0 else None)
    result['billing'] = broadcast(billing)
    _phase('result write', lambda: (run_dir / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n') if rank == 0 else None)
    return result


def _append_metric(run_dir, record):
    with (run_dir / 'metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
