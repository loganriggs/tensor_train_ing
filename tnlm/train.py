"""Token-normalized reference training, explicit cooldowns, and local cost accounting.

Only hidden linear-map weights receive decay. AdamW and Muon preserve distinct
L/R parameters; Muon uses auxiliary AdamW for vocabulary weights and scalars.
Activation recomputation is optional and forbidden for lagged statistics.
Dollar accounting requires the caller's actual hourly rate;
it includes initialization, compilation and checkpoint time, not external API costs.
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from .schedules import validate_schedule, schedule_multiplier, schedule_steps
from .checkpoint_plan import checkpoint_storage_plan, pending_milestone_tokens, checkpoint_retention_policy
from .checkpoint_storage import require_checkpoint_space
from .optimizer_exposure import update_optimizer_exposure
from .budget import STAGE_BUDGETS, SpendLedger
from .calibration import _check_lagged_divisor, calibrate_lagged
from .config import Config
from .evaluate import compare_evaluations, evaluate_checkpoint, evaluate_mechanisms
from .reporting import summarize_runs
from .runtime import evaluation_validator, full_config, resume_validator, seed_all, training_execution, determinism_metadata
from .model import TensorLM
from .training_stability import stability_interval, qk_clip_policy, TrainingStabilityProbe
from .training_taper import TaperTraining, validate_taper_config
from .normalization import lagged_modules, spectral_step


def _assert_no_analysis_state(model):
    for name, module in model.named_modules():
        if getattr(module, 'ablate', frozenset()):
            raise RuntimeError(f'analysis ablation leaked into training: {name}')
        if any(getattr(module, field, None) is not None
               for field in ('denom_mode', 'denom_saved', 'denom_const')):
            raise RuntimeError(f'analysis denominator state leaked into training: {name}')


def optimizer_groups(model, weight_decay):
    from .optim import hidden_matrix
    vocabulary = {id(model.embed.weight), id(model.unembed.weight)}
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (decay if hidden_matrix(name, param) and id(param) not in vocabulary else no_decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(model, cfg):
    if cfg.optimizer == 'muon' or getattr(getattr(model, 'cfg', None), 'width_scaling_base', None) is not None:
        from .optim import make_optimizer as make_scaling_optimizer
        return make_scaling_optimizer(model, cfg)
    if cfg.optimizer != "adamw":
        raise ValueError("optimizer must be adamw or muon")
    return torch.optim.AdamW(optimizer_groups(model, cfg.weight_decay_hidden),
                             lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def lr_multiplier(step, total_steps, *, cooldown_steps=None):
    if cooldown_steps is not None:
        if cooldown_steps <= 0:
            raise ValueError("cooldown_steps must be positive")
        return max(0.0, 1 - step / cooldown_steps)
    warmup = max(100, math.ceil(total_steps * 0.01))
    return min(1.0, (step + 1) / warmup)


def plan_search(cfg):
    project, data = cfg.raw.get('project', {}), cfg.raw.get('data', {})
    if cfg.train.rung_total_tokens <= 0 or cfg.train.global_prediction_tokens_per_update <= 0:
        raise ValueError('positive rung/token batch required for planning')
    planned_steps = schedule_steps(cfg.train.rung_total_tokens, cfg.train.global_prediction_tokens_per_update)
    warmup_steps = validate_schedule(cfg.train.schedule, planned_steps)
    controller = SpendLedger(project.get('spend_ledger', 'runs/spend.json'),
                             project.get('total_budget_usd', 1000), project.get('reserve_usd', 150),
                             stage_budgets=project.get('stage_budgets'))
    stage = project.get('stage', 'A')
    if stage not in controller.stage_budgets:
        raise ValueError(f'unknown budget stage: {stage}')
    throughput = project.get('measured_prediction_tokens_per_second')
    if throughput is not None:
        throughput = float(throughput)
        if not math.isfinite(throughput) or throughput <= 0:
            raise ValueError('measured throughput must be finite and positive')
    setup_seconds = float(project.get('estimated_setup_seconds', 0))
    if not math.isfinite(setup_seconds) or setup_seconds < 0:
        raise ValueError('estimated setup time must be finite and nonnegative')
    seconds = cfg.train.rung_total_tokens / throughput + setup_seconds if throughput is not None else None
    rate = project.get('gpu_hourly_rate_usd')
    if rate is not None:
        rate = float(rate)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError('hourly rate must be finite and nonnegative')
    snapshot = json.loads(controller.path.read_text()) if controller.path.exists() else {'runs': {}}
    allocated = sum(r.get('spent_usd', 0) + r.get('reserved_usd', 0) for r in snapshot['runs'].values())
    stage_allocated = sum(r.get('spent_usd', 0) + r.get('reserved_usd', 0)
        for r in snapshot['runs'].values() if r.get('stage', 'A') == stage)
    regular_allocated = sum(r.get('spent_usd', 0) + r.get('reserved_usd', 0)
        for r in snapshot['runs'].values() if r.get('stage', 'A') != 'Reserve')
    return {"dry_run": project.get('dry_run', True), "manifest": data.get('manifest'),
            "context": data.get('context', 512), "tokens": cfg.train.rung_total_tokens,
            "prediction_tokens_per_update": cfg.train.global_prediction_tokens_per_update,
            "normalization": cfg.model.normalization, "optimizer": cfg.train.optimizer, "schedule": cfg.train.schedule,
            "checkpoint_storage_plan": checkpoint_storage_plan(cfg),
            "rate_usd_per_hour": project.get('gpu_hourly_rate_usd'),
            "total_budget_usd": project.get('total_budget_usd', 1000),
            "reserve_usd": project.get('reserve_usd', 150),
            "stage": project.get('stage', 'A'),
            "stage_budgets_usd": controller.stage_budgets,
            "unallocated_project_usd": controller.total - allocated,
            "unallocated_regular_usd": controller.total - controller.reserve - regular_allocated,
            "unallocated_stage_usd": controller.stage_budgets[stage] - stage_allocated,
            "cost_margin_fraction": project.get('cost_margin_fraction', .1),
            "warmup_steps": warmup_steps,
            "planned_optimizer_steps": planned_steps,
            "warmup_dominates_rung": warmup_steps > .1 * planned_steps,
            "proposed_runs": [{"normalization": cfg.model.normalization,
                "tokens": cfg.train.rung_total_tokens, "projected_seconds": seconds,
                "projected_usd": seconds * rate / 3600 if seconds is not None and rate is not None else None}],
            "note": "one configured trajectory; projected cost needs measured valid-target throughput and actual hourly rate"}


def _mask_to_count(targets, count):
    targets = targets.clone()
    indices = (targets.flatten() != -100).nonzero().flatten()
    targets.flatten()[indices[count:]] = -100
    return targets


def _trim_unscored_rows(idx, docs, targets, *, enabled, bucket_rows=False):
    """Omit independent rows with zero loss, preserving each retained prefix.

    Call only without stochastic layers or lagged-statistic collection: those
    can observe otherwise unscored rows. This does not change sampler exposure.
    Compiled forwards use at most five row counts for a fixed input batch size:
    chunked CE can specialize its Python loop separately for each row count.
    Fill buckets with original zero-loss rows, never synthetic tokens/prefixes.
    """
    if not enabled:
        return idx, docs, targets
    keep = (targets != -100).any(dim=1)
    if bool(keep.all()):
        return idx, docs, targets
    if bucket_rows:
        scored_rows = int(keep.sum())
        if scored_rows:
            original_rows = idx.shape[0]
            buckets = sorted({1, *((original_rows + d - 1) // d for d in (8, 4, 2, 1))})
            retained_rows = next(size for size in buckets if size >= scored_rows)
            if retained_rows == original_rows:
                return idx, docs, targets
            extra = retained_rows - scored_rows
            if extra:
                keep[(~keep).nonzero().flatten()[:extra]] = True
    return idx[keep], docs[keep], targets[keep]


@training_execution()
def train_run(cfg, run_dir, *, max_steps, max_seconds, device='cuda', resume=None,
              cooldown_steps=None, cooldown_end_tokens=None, run_budget_usd=None,
              context_fork=False, batch_tokens=None, lr_scale=None, _step_profiler=None, _throughput_canary=None):
    """Run a bounded trajectory; resume into a NEW run directory/ledger identity.

    max_steps counts new updates. Resume restores RNG and next-batch order before
    any forward. Cooldown forks keep optimizer/data state but use a fresh linear
    schedule from the checkpoint's current LR. Checkpoints occur at step boundaries.
    """
    if type(context_fork) is not bool or (context_fork and resume is None):
        raise ValueError('context_fork requires a Boolean flag and explicit parent checkpoint')
    if isinstance(cfg, (str, Path)):
        cfg = Config.from_yaml(str(cfg))
    from .batch_branch import branch_config, validate_parent as validate_batch_parent, apply_branch, history_entry
    parent_cfg = cfg
    cfg, batch_change = branch_config(cfg, batch_tokens=batch_tokens, lr_scale=lr_scale,
        resume=resume, context_fork=context_fork, cooldown_steps=cooldown_steps)
    parent_validator = resume_validator(parent_cfg, context_fork=context_fork)
    retention_policy = checkpoint_retention_policy(cfg)
    stability_every = stability_interval(cfg)
    clip_tau, clip_every = qk_clip_policy(cfg)
    if validate_taper_config(cfg):
        taper_total = schedule_steps(cfg.train.rung_total_tokens, cfg.train.global_prediction_tokens_per_update)
        taper_warmup = validate_schedule(cfg.train.schedule, taper_total)
        if round(cfg.train.taper_end_fraction * taper_total) <= taper_warmup:
            raise ValueError("TaperNorm endpoint must be strictly after positive warmup_steps")
    if cfg.train.schedule == 'cosine' and cooldown_end_tokens is not None:
        raise ValueError('cosine does not support a separate cooldown')
    validate_schedule(cfg.train.schedule, schedule_steps(cfg.train.rung_total_tokens, cfg.train.global_prediction_tokens_per_update), cooldown_steps=cooldown_steps)
    if cfg.raw.get('project', {}).get('dry_run', True):
        planned = plan_search(cfg)
        return {**planned, 'batch_branch': batch_change, 'context_fork': context_fork,
                'new_context': cfg.raw.get('data', {}).get('context', 512),
                'parent': str(resume) if resume is not None else None,
                'parent_validation': 'pending' if resume is not None else None}
    # This entry point owns one sampler, ledger reservation, and full-state save.
    # FSDP primitives alone do not make those operations collective-safe.
    try:
        launched_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    except ValueError as error:
        raise ValueError('WORLD_SIZE must be a positive integer') from error
    if launched_world_size < 1:
        raise ValueError('WORLD_SIZE must be a positive integer')
    distributed = torch.distributed
    active_world_size = (distributed.get_world_size() if distributed.is_available()
                         and distributed.is_initialized() else 1)
    if max(launched_world_size, active_world_size) > 1:
        raise RuntimeError('train_run supports one process; distributed training requires coordinated '
                           'sampling, token budgets, clipping, billing, and checkpoint replay')
    from .checkpoint import save_checkpoint, load_checkpoint, _digest
    from .data import PackedSampler, batch_hash

    project, data = cfg.raw.get('project', {}), cfg.raw.get('data', {})
    if type(cfg.model.loss_chunk_tokens) is not int or cfg.model.loss_chunk_tokens < 0:
        raise ValueError('loss_chunk_tokens must be a nonnegative integer')
    if cfg.model.activation_checkpointing and 'lagged' in (cfg.model.normalization, cfg.model.final_normalization):
        raise ValueError('activation checkpointing would double-count lagged statistics')
    if cooldown_steps is not None and (cooldown_steps <= 0 or resume is None):
        raise ValueError('cooldown requires positive cooldown_steps and an explicit parent checkpoint')
    if cooldown_end_tokens is not None and (type(cooldown_end_tokens) is not int or cooldown_end_tokens <= 0 or cooldown_steps is None):
        raise ValueError('cooldown_end_tokens requires a positive integer and cooldown_steps')
    history_limit = int(project.get('history_limit', 1000))
    if history_limit <= 0:
        raise ValueError('history_limit must be positive; full metrics persist in metrics.jsonl')
    if max_steps <= 0 or max_seconds <= 0:
        raise ValueError("positive max_steps/max_seconds required")
    if cfg.train.rung_total_tokens <= 0 or cfg.train.global_prediction_tokens_per_update <= 0:
        raise ValueError("positive token limits required")
    if data.get('document_boundary_mask', True) is not True or data.get('reset_position_at_document', True) is not True:
        raise ValueError("this reference requires document masking and reset positions")
    if not cfg.train.fp32_optimizer_state or not cfg.train.fp32_norm_reductions:
        raise ValueError("reference requires FP32 optimizer state and norm reductions")
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rate = project.get('gpu_hourly_rate_usd')
    if rate is None or (dev.type == 'cuda' and rate <= 0):
        raise ValueError("declare actual gpu_hourly_rate_usd (explicit 0 allowed for CPU)")
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    if _throughput_canary is not None:
        _throughput_canary.bind_output(run_dir / 'canary.json')
    # Fresh launches can reject the entire retention schedule before model/data
    # setup or spend admission. Resume receives its exact cursor after loading.
    if resume is None:
        launch_storage_plan = checkpoint_storage_plan(cfg, max_steps=max_steps)
        require_checkpoint_space(run_dir.parent, launch_storage_plan['required_additional_bytes'])
    old_context = data.get('context', 512)
    if resume is not None:
        # Trusted-local metadata preflight before billing; canonical load below
        # still verifies checksums and restores full state before any forward.
        metadata = json.loads((Path(resume) / 'manifest.json').read_text())
        state_path = Path(resume) / 'state.pt'
        info = metadata['files']['state.pt']
        if metadata.get('format_version') != 1 or state_path.stat().st_size != info['bytes'] or _digest(state_path) != info['sha256']:
            raise ValueError('checkpoint checksum/version mismatch before resume metadata deserialization')
        saved = torch.load(state_path, map_location='cpu', weights_only=False, mmap=True)
        parent_validator(saved['config'])
        if batch_change is not None:
            validate_batch_parent(saved, cfg)
        old_context = saved['config'].get('data', {}).get('context', 512)
        del saved
    ledger = SpendLedger(project.get('spend_ledger', 'runs/spend.json'),
                         project.get('total_budget_usd', 1000), project.get('reserve_usd', 150),
                         stage_budgets=project.get('stage_budgets'))
    run_id = str(run_dir.resolve())
    ledger.begin(run_id, rate, max_seconds, run_budget_usd, parent_checkpoint=resume,
                 cost_margin_fraction=project.get('cost_margin_fraction', .1), stage=project.get('stage', 'A'))
    started = time.monotonic()
    status = 'failed'
    try:
        sampler = PackedSampler(data['manifest'], split='train',
                                batch_size=data.get('micro_batch_size', 1),
                                context=old_context, seed=cfg.train.seed)
        from .data import manifest_hash as hash_manifest
        manifest_hash = hash_manifest(data['manifest'])
        tokenizer = sampler.manifest['tokenizer']
        if tokenizer.get('test_only') and dev.type != 'cpu':
            raise ValueError('test-only tokenizer manifests are restricted to CPU fixtures')
        if data.get('tokenizer') is not None and data['tokenizer'] not in (tokenizer['name'], tokenizer.get('fingerprint_sha256')):
            raise ValueError('configured tokenizer does not match immutable manifest')
        run_dir.mkdir(parents=True)
        seed_all(cfg.train.seed)
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(dev)
        model = TensorLM(cfg.model).to(dev)
        _assert_no_analysis_state(model)
        optimizer = make_optimizer(model, cfg.train)
        progress = {"steps": 0, "tokens": 0, "skipped_updates": [], "scheduler_step": 0,
                    "input_tokens_seen": 0, "discarded_prediction_targets": 0,
                    "warmup_steps": validate_schedule(cfg.train.schedule, schedule_steps(cfg.train.rung_total_tokens, cfg.train.global_prediction_tokens_per_update))}
        parent = None
        if resume:
            payload = load_checkpoint(resume, model, optimizer, sampler,
                                      expected_manifest_hash=manifest_hash,
                                      config_validator=parent_validator, map_location='cpu')
            progress.update(payload['progress'])
            if payload.get('config_compatibility', {}).get('defaulted_fields'):
                progress.setdefault('config_compatibility_history', []).append(
                    {'checkpoint': str(Path(resume).resolve()), **payload['config_compatibility']})
            parent = {'path': str(Path(resume).resolve()),
                      'manifest_sha256': hashlib.sha256((Path(resume) / 'manifest.json').read_bytes()).hexdigest()}
            if batch_change is not None:
                validate_batch_parent(payload, cfg)
                apply_branch(optimizer, sampler, progress, parent_cfg, cfg, parent, batch_change)
                import yaml
                (run_dir / 'branch-config.yaml').write_text(yaml.safe_dump(full_config(cfg), sort_keys=False))
            if context_fork:
                from .runtime import record_context_fork
                source_state = sampler.state_dict()
                sampler = sampler.fork_context(data.get('context', 512))
                record_context_fork(progress, old_context=old_context,
                    new_context=data.get('context', 512), parent=parent, source_state=source_state)
            if payload.get('extra', {}).get('cooldown_steps') is not None and cooldown_steps is None:
                raise ValueError('cooldown checkpoint requires explicit matching cooldown_steps')
        if not resume:
            progress['batch_history'] = [history_entry(cfg, tokens=0, steps=0, source='fresh')]
            progress['context_history'] = [{'context': data.get('context', 512),
                'start_tokens': 0, 'start_step': 0, 'source': 'fresh'}]
        resume_extra = payload.get('extra', {}) if resume else {}
        if resume:
            del payload  # CPU source tensors need not stay resident after restoration
        branch_base_lr = [g['lr'] for g in optimizer.param_groups]
        if cooldown_steps is not None:
            previous_cooldown = resume_extra.get('cooldown_steps')
            if previous_cooldown is not None:
                if previous_cooldown != cooldown_steps:
                    raise ValueError('cannot change an existing cooldown horizon')
                if cooldown_end_tokens != resume_extra.get('cooldown_end_tokens'):
                    raise ValueError('cannot change an existing cooldown endpoint')
                branch_base_lr = resume_extra['branch_base_lr']
            else:
                progress['scheduler_step'] = 0
                progress['cooldown_start_tokens'] = progress['tokens']
        norms = lagged_modules(model)
        if norms and not resume:
            progress['calibration'] = calibrate_lagged(model, cfg, data['manifest'], dev,
                                                       deadline=started + max_seconds)
        for norm in norms:
            _check_lagged_divisor(norm)
        # Collection mutates buffers and must never be traced/recomputed.
        forward = torch.compile(model) if cfg.train.compile and not norms else model
        total_steps = schedule_steps(cfg.train.rung_total_tokens, cfg.train.global_prediction_tokens_per_update)
        if cooldown_steps is not None:
            token_limit = progress['cooldown_start_tokens'] + cooldown_steps * cfg.train.global_prediction_tokens_per_update
            if cooldown_end_tokens is not None:
                lower = token_limit - cfg.train.global_prediction_tokens_per_update
                if not lower < cooldown_end_tokens <= token_limit:
                    raise ValueError('cooldown endpoint must match the cooldown step horizon')
                token_limit = cooldown_end_tokens
                progress['cooldown_end_tokens'] = cooldown_end_tokens
        else:
            token_limit = cfg.train.rung_total_tokens
        storage_plan = checkpoint_storage_plan(cfg, start_tokens=progress['tokens'],
            token_limit=token_limit, max_steps=max_steps, cooldown=cooldown_steps is not None)
        storage_admission = require_checkpoint_space(run_dir, storage_plan['required_additional_bytes'])
        (run_dir / 'checkpoint_storage_plan.json').write_text(json.dumps(
            {'plan': storage_plan, 'admission': storage_admission}, indent=2, allow_nan=False) + '\n')
        taper = TaperTraining(model, cfg, total_steps=total_steps, warmup_steps=progress["warmup_steps"])
        taper.before_step(progress["steps"])
        taper.set_valid_mask(None)
        def save(name):
            save_checkpoint(run_dir / name, model, optimizer, sampler,
                                   config=full_config(cfg), progress=progress,
                                   manifest_hash=manifest_hash, parent=parent,
                                   extra={"operator_audit": model.operator_audit(),
                                          "cooldown_steps": cooldown_steps,
                                          "cooldown_end_tokens": cooldown_end_tokens,
                                          "branch_base_lr": branch_base_lr,
                                          "runtime": {"determinism": determinism_metadata(), "device": str(dev),
                                              "autocast_enabled": cfg.train.bf16_compute and dev.type == "cuda",
                                              "autocast_dtype": "bfloat16" if cfg.train.bf16_compute and dev.type == "cuda" else None}})
            return run_dir / name
        write_start = time.monotonic()
        if retention_policy['save_initial_checkpoint']:
            save('initial')
        checkpoint_write_seconds = (3 * (time.monotonic() - write_start)
                                    if retention_policy['save_initial_checkpoint'] else 0.0)
        stability_probe = TrainingStabilityProbe(cfg, dev) if stability_every or clip_tau > 0 else None
        last_recovery_time = time.monotonic()
        rolling = deque()
        pending_milestones = list(pending_milestone_tokens(cfg, start_tokens=progress['tokens'],
            token_limit=token_limit, reachable_tokens=storage_plan['reachable_tokens'],
            include_branch=cooldown_steps is None and cfg.train.schedule != 'cosine'))
        history = deque(maxlen=history_limit)
        slowest_step_seconds = 0.0
        completed_new_updates = completed_warmup_updates = 0
        model.train()
        empty_batches = 0
        checkpoint_every = int(project.get('checkpoint_every_steps', 0))
        checkpoint_seconds = float(project.get('checkpoint_every_seconds', 300))
        keep_recovery = retention_policy['keep_recovery_checkpoints']
        if checkpoint_seconds <= 0:
            raise ValueError('checkpoint interval must be positive')
        stop_reason = 'max_steps'
        region = getattr(_step_profiler, 'region', lambda _name: nullcontext())
        for local_step in range(max_steps):
            _assert_no_analysis_state(model)
            elapsed = time.monotonic() - started
            # Avoid starting another step if the observed slowest step won't fit.
            if elapsed >= max_seconds or elapsed + slowest_step_seconds + checkpoint_write_seconds >= max_seconds:
                stop_reason = 'runtime_cap'
                break
            if progress['tokens'] >= token_limit:
                stop_reason = 'token_limit'
                break
            step_start = time.monotonic()
            if _step_profiler is not None:
                _step_profiler.start_step(local_step)
            taper.before_step(progress['steps'])
            schedule_step = progress.get('scheduler_step', progress['steps'])
            mult = (lr_multiplier(schedule_step, total_steps, cooldown_steps=cooldown_steps)
                    if cooldown_steps is not None else schedule_multiplier(schedule_step, total_steps, schedule=cfg.train.schedule, warmup_steps=progress['warmup_steps']))
            for group, base in zip(optimizer.param_groups, branch_base_lr):
                group['lr'] = (base if cooldown_steps is not None else group.get('base_lr', cfg.train.lr)) * mult
            optimizer.zero_grad(set_to_none=True)
            with region('tnlm.spectral_step'):
                spectral_step(model)
            wanted = min(cfg.train.global_prediction_tokens_per_update, token_limit - progress['tokens'])
            valid_total, loss_total, hashes = 0, 0.0, []
            anchor_total = 0.0
            forward_input_positions, unscored_rows_omitted = 0, 0
            while valid_total < wanted:
                if time.monotonic() - started >= max_seconds:
                    raise TimeoutError("runtime cap during accumulation; last complete checkpoint is recoverable")
                with region('tnlm.sampler'):
                    batch = sampler.next_batch()
                with region('tnlm.prepare_batch'):
                    if 'context_fork_exposure' in progress:
                        for key, value in sampler.batch_context_exposure().items():
                            progress['context_fork_exposure'][key] += value
                    # The packed stream is CPU-resident. Validate and select the
                    # exact target quota before transfer, avoiding device-to-host
                    # synchronization for each of these input/count checks.
                    idx, docs, targets = batch
                    if (idx < 0).any() or (idx >= cfg.model.vocab_size).any():
                        raise ValueError("input token outside real vocabulary")
                    if ((docs < 0) & (targets != -100)).any():
                        raise ValueError('padded-token targets must be -100')
                    if ((targets != -100) & ((targets < 0) | (targets >= cfg.model.vocab_size))).any():
                        raise ValueError("target outside real vocabulary")
                    raw_valid = int((targets != -100).sum())
                    targets = _mask_to_count(targets, wanted - valid_total)
                    progress['input_tokens_seen'] += int((docs >= 0).sum())
                    n = int((targets != -100).sum())
                    progress['discarded_prediction_targets'] += raw_valid - n
                    if not n:
                        empty_batches += 1
                        if empty_batches > 100:
                            raise ValueError("no valid prediction targets in sampler")
                        continue
                    empty_batches = 0
                    hashes.append(batch_hash(batch))
                    original_rows = idx.shape[0]
                    idx, docs, targets = _trim_unscored_rows(idx, docs, targets,
                        enabled=not norms and not taper.enabled and cfg.model.dropout == 0,
                        bucket_rows=cfg.train.compile and not norms)
                    unscored_rows_omitted += original_rows - idx.shape[0]
                    forward_input_positions += idx.numel()
                with region('tnlm.transfer'):
                    idx, docs, targets = (x.to(dev) for x in (idx, docs, targets))
                if taper.enabled:
                    taper.set_valid_mask((docs >= 0) if taper.collecting else (targets != -100))
                for norm in norms:
                    norm.collecting = True
                    norm.set_valid_mask(docs >= 0)
                try:
                    with region('tnlm.forward'), torch.autocast(dev.type, dtype=torch.bfloat16,
                                        enabled=cfg.train.bf16_compute and dev.type == 'cuda'):
                        _, loss, actual_n = forward(idx, doc_ids=docs, targets=targets, reduction='sum')
                    anchor = taper.anchor()
                    objective = loss if anchor is None else loss + taper.anchor_lambda * anchor * n
                    if anchor is not None:
                        anchor_total += float(anchor.detach()) * n
                    with region('tnlm.validate_loss'):
                        if int(actual_n) != n or not torch.isfinite(objective):
                            raise FloatingPointError("invalid loss/count; update aborted")
                    with region('tnlm.backward'):
                        objective.backward()
                finally:
                    taper.set_valid_mask(None)
                    for norm in norms:
                        norm.collecting = False
                        norm.set_valid_mask(None)
                valid_total += n
                with region('tnlm.loss_accounting'):
                    loss_total += float(loss.detach())
            with region('tnlm.optimizer'):
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.div_(valid_total)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip, error_if_nonfinite=True)
                optimizer.step()
            update_optimizer_exposure(progress, optimizer)
            for norm in norms:
                norm.commit()
                _check_lagged_divisor(norm)
            if dev.type == 'cuda':
                with region('tnlm.synchronize'):
                    torch.cuda.synchronize(dev)
            progress['steps'] += 1
            taper.after_step(progress['steps'])
            progress['tokens'] += valid_total
            progress['scheduler_step'] = schedule_step + 1
            completed_new_updates += 1
            completed_warmup_updates += int(cooldown_steps is None and schedule_step < progress['warmup_steps'])
            clipping = None
            if clip_tau > 0 and progress['steps'] % clip_every == 0:
                with region('tnlm.qk_clip'):
                    clipping = stability_probe.clip(model, clip_tau)
            stability = None
            if stability_every and progress['steps'] % stability_every == 0:
                with region('tnlm.stability_probe'):
                    stability = stability_probe.sample(model)
            record = {"step": progress['steps'], "tokens": progress['tokens'],
                      "prediction_tokens": valid_total,
                      "loss": loss_total / valid_total, "grad_norm": float(grad_norm),
                      "optimizer_group_lrs": [group['lr'] for group in optimizer.param_groups],
                      "optimizer_exposure": progress["optimizer_exposure"],
                      "lr": optimizer.param_groups[0]['lr'], "batch_hashes": hashes,
                      "microbatches": len(hashes), "forward_input_positions": forward_input_positions,
                      "unscored_rows_omitted": unscored_rows_omitted,
                      "seconds": time.monotonic() - step_start,
                      "run_elapsed_seconds": time.monotonic() - started,
                      "run_billed_usd": (time.monotonic() - started) * rate / 3600}
            record.update(taper.metrics(anchor_total / valid_total))
            if clipping is not None:
                record['qk_clip'] = clipping
            if stability is not None:
                record['stability'] = stability
            if _step_profiler is not None:
                _step_profiler.finish_step()
            slowest_step_seconds = max(slowest_step_seconds, time.monotonic() - step_start)
            history.append(record)
            with open(run_dir / 'metrics.jsonl', 'a') as out:
                out.write(json.dumps(record, allow_nan=False) + '\n')
            if _throughput_canary is not None:
                _throughput_canary.observe(record)
            while pending_milestones and progress['tokens'] >= pending_milestones[0]:
                milestone = pending_milestones.pop(0)
                save(f'milestone-{milestone:012d}')
            recovery_due = ((checkpoint_every > 0 and progress['steps'] % checkpoint_every == 0) or
                time.monotonic() - last_recovery_time >= max(checkpoint_seconds, 20 * checkpoint_write_seconds))
            if keep_recovery and recovery_due:
                write_start = time.monotonic()
                rolling.append(save(f"recovery-{progress['steps']:08d}"))
                checkpoint_write_seconds = time.monotonic() - write_start
                last_recovery_time = time.monotonic()
                # Serialize pruning against branch admission. Branch parents remain
                # immutable and retained even if the rolling window has moved on.
                while len(rolling) > keep_recovery:
                    candidate = rolling.popleft()
                    with ledger.locked() as ledger_data:
                        references = {r.get('parent_checkpoint') for r in ledger_data['runs'].values()}
                        if str(candidate.resolve()) not in references:
                            shutil.rmtree(candidate)
        if _throughput_canary is not None:
            _throughput_canary.require_complete()
        final = save('final')
        from .evaluate import _training_descriptor
        training_descriptor = _training_descriptor({'config': full_config(cfg), 'progress': progress,
            'extra': {'cooldown_steps': cooldown_steps, 'cooldown_end_tokens': cooldown_end_tokens}})
        result = {"training_descriptor": training_descriptor, "batch_branch": batch_change, "checkpoint": str(final), "checkpoint_storage_plan": storage_plan,
                  "checkpoint_storage_admission": storage_admission, "progress": progress, "history": list(history),
                  "metrics_path": str(run_dir / "metrics.jsonl"),
                  "spend_ledger": str(ledger.path.resolve()),
                  "run_id": run_id, "parent": parent,
                  "history_limit": history_limit,
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None,
                  "operator_audit": model.operator_audit(), "parameters": model.param_counts(),
                  "runtime": {"determinism": determinism_metadata(), "device": str(dev), "torch": str(torch.__version__),
                              "bf16_autocast": cfg.train.bf16_compute and dev.type == "cuda"},
                  "model_config": full_config(cfg)["model"], "train_config": full_config(cfg)["train"],
                  "data": {"manifest_sha256": manifest_hash, "context": data.get("context", 512),
                           "tokenizer": tokenizer},
                  "stop_reason": stop_reason,
                  "compile_enabled": bool(cfg.train.compile and not norms),
                  "warmup_steps": progress["warmup_steps"],
                  "warmup_fraction_of_completed_updates": completed_warmup_updates / completed_new_updates if completed_new_updates else None,
                  "warmup_dominates_completed_segment": completed_warmup_updates > .1 * completed_new_updates,
                  "lagged_collection_policy": "all nonpadding tokens; once per optimizer step",
                  "elapsed_seconds_before_final_accounting": time.monotonic() - started}
        if _throughput_canary is not None:
            result['throughput_canary'] = _throughput_canary.report()
        if _step_profiler is not None:
            result['kernel_profile'] = _step_profiler.report()
        (run_dir / 'result.json').write_text(json.dumps(result, indent=2))
        status = 'complete'
        return result
    finally:
        try:
            if _step_profiler is not None:
                _step_profiler.close()
        finally:
            ledger.finish(run_id, time.monotonic() - started, status)


def profile_run(cfg, run_dir, *, steps, warmup_steps=2, max_seconds, device='cuda', run_budget_usd=None, kernels=False, trace_steps=0):
    """Measure complete training steps, excluding only declared warmup from timing.

    Initialization, compiler warmup and checkpointing remain in billed wall time.
    This is a fresh bounded training trajectory, not a forward-only microbenchmark.
    """
    if isinstance(cfg, (str, Path)):
        cfg = Config.from_yaml(str(cfg))
    if steps <= 0 or warmup_steps < 0:
        raise ValueError('positive timed steps and nonnegative warmup_steps required')
    if type(trace_steps) is not int or not 0 <= trace_steps <= 100:
        raise ValueError('trace_steps must be an integer in [0, 100]')
    if kernels and torch.device(device).type != 'cuda':
        raise ValueError('kernel profiling requires a CUDA device')
    from .profiling import KernelProfile
    profiler = KernelProfile(warmup_steps, trace_dir=Path(run_dir) / 'traces',
                             trace_steps=trace_steps, cuda=torch.device(device).type == 'cuda',
                             kernels=kernels) if kernels or trace_steps else None
    result = train_run(cfg, run_dir, max_steps=steps + warmup_steps,
                       max_seconds=max_seconds, device=device, run_budget_usd=run_budget_usd,
                       **({'_step_profiler': profiler} if profiler is not None else {}))
    if result.get('dry_run'):
        return {**result, 'command': 'profile', 'warmup_steps': warmup_steps, 'timed_steps': steps, 'kernels': kernels, 'trace_steps': trace_steps}
    timed_seconds, tokens, timed_steps, observed = 0.0, 0, 0, 0
    microbatches, forward_positions, omitted_rows = 0, 0, 0
    metrics = Path(result['metrics_path'])
    if metrics.exists():
        with metrics.open() as source:
            for index, line in enumerate(source):
                record = json.loads(line)
                observed += 1
                if index >= warmup_steps:
                    timed_steps += 1
                    timed_seconds += record['seconds']
                    tokens += record['prediction_tokens']
                    microbatches += record['microbatches']
                    forward_positions += record['forward_input_positions']
                    omitted_rows += record['unscored_rows_omitted']
    report = {'command': 'profile', 'device': str(device), 'determinism': result.get('runtime', {}).get('determinism'), 'kernels': kernels, 'trace_steps': trace_steps,
              'requested_timed_steps': steps, 'completed_timed_steps': timed_steps,
              'warmup_steps': warmup_steps, 'completed_total_steps': observed,
              'timed_training_seconds': timed_seconds, 'timed_prediction_tokens': tokens,
              'timed_microbatches': microbatches,
              'timed_forward_input_positions': forward_positions,
              'timed_unscored_rows_omitted': omitted_rows,
              'forward_positions_per_prediction_target': forward_positions / tokens if tokens else None,
              'mean_microbatches_per_step': microbatches / timed_steps if timed_steps else None,
              'prediction_tokens_per_second': tokens / timed_seconds if timed_seconds > 0 else None,
              'mean_training_step_seconds': timed_seconds / timed_steps if timed_steps else None,
              'cuda_peak_allocated_bytes': result['cuda_peak_allocated_bytes'],
              'cuda_peak_reserved_bytes': result['cuda_peak_reserved_bytes'],
              'checkpoint': result['checkpoint'], 'operator_audit': result['operator_audit'],
              'parameters': result['parameters'], 'compile_enabled': result['compile_enabled'],
              'stop_reason': result['stop_reason'],
              'note': 'step timers include batch preparation, forward/backward and optimizer work; any compilation or recompilation after declared warmup remains in measured time; checkpoint saves, metric writes and profiler finalization are outside step timers; training initialization, warmup and saves remain billed'}
    ledger = Path(cfg.raw.get('project', {}).get('spend_ledger', 'runs/spend.json'))
    if profiler is not None:
        report['kernel_profile'] = result['kernel_profile']
    report['billing'] = json.loads(ledger.read_text())['runs'][str(Path(run_dir).resolve())]
    (Path(run_dir) / 'profile.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


