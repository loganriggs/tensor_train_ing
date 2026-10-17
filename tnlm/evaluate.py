"""Frozen held-out and exact-label mechanism evaluation and matched comparisons."""
from __future__ import annotations

import json
import copy
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .budget import SpendLedger
from .calibration import _check_lagged_divisor
from .config import Config
from .model import TensorLM
from .normalization import lagged_modules
from .runtime import evaluation_validator


def _context_history_descriptor(progress, context):
    """Summarize known context segments without inferring legacy history."""
    history = progress.get('context_history')
    if not isinstance(history, list) or not history:
        return None, 'unknown'
    entries = []
    previous_tokens = previous_step = previous_context = None
    complete = True
    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            return None, 'unknown'
        ctx, tokens, step, source = (entry.get(key) for key in ('context', 'start_tokens', 'start_step', 'source'))
        if type(ctx) is not int or ctx <= 0:
            return None, 'unknown'
        if index == 0 and source == 'legacy_unknown' and tokens is None and step is None:
            complete = False
        else:
            if any(type(value) is not int or value < 0 for value in (tokens, step)):
                return None, 'unknown'
            if index == 0:
                if source != 'fresh' or tokens != 0 or step != 0:
                    return None, 'unknown'
            elif (source != 'context_fork' or ctx <= previous_context
                  or (previous_tokens is not None and tokens < previous_tokens)
                  or (previous_step is not None and step < previous_step)):
                return None, 'unknown'
            if (type(progress.get('tokens')) is not int or tokens > progress['tokens']
                    or type(progress.get('steps')) is not int or step > progress['steps']):
                return None, 'unknown'
        entries.append({'context': ctx, 'start_tokens': tokens, 'start_step': step, 'source': source})
        previous_context, previous_tokens, previous_step = ctx, tokens, step
    if previous_context != context:
        return None, 'unknown'
    return entries, 'complete' if complete else 'partial'


def _batch_history_descriptor(progress, train):
    """Report semantic batch/LR segments, without artifact-specific identities."""
    history = progress.get('batch_history')
    if not isinstance(history, list) or not history:
        return None, 'unknown'
    def rate(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    entries = []
    previous_tokens = previous_step = None
    complete = True
    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            return None, 'unknown'
        batch, lr, aux, tokens, step, source = (entry.get(key) for key in
            ('batch_tokens', 'lr', 'aux_adamw_lr', 'start_tokens', 'start_step', 'source'))
        scale = entry.get('lr_scale', 1.)
        if (type(batch) is not int or batch <= 0 or not rate(lr)
                or 'aux_adamw_lr' not in entry or (aux is not None and not rate(aux))
                or not rate(scale) or scale <= 0):
            return None, 'unknown'
        if index == 0 and source == 'legacy_unknown' and tokens is None and step is None:
            complete = False
        else:
            if any(type(value) is not int or value < 0 for value in (tokens, step)):
                return None, 'unknown'
            if index == 0:
                if source != 'fresh' or tokens != 0 or step != 0:
                    return None, 'unknown'
            elif (source != 'batch_branch' or 'lr_scale' not in entry
                  or (previous_tokens is not None and tokens < previous_tokens)
                  or (previous_step is not None and step < previous_step)):
                return None, 'unknown'
            if (type(progress.get('tokens')) is not int or tokens > progress['tokens']
                    or type(progress.get('steps')) is not int or step > progress['steps']):
                return None, 'unknown'
        if index == 0 and scale != 1:
            return None, 'unknown'
        entries.append({'batch_tokens': batch, 'lr': lr, 'aux_adamw_lr': aux,
                        'start_tokens': tokens, 'start_step': step, 'source': source,
                        'lr_scale': scale})
        previous_tokens, previous_step = tokens, step
    last = entries[-1]
    if (last['batch_tokens'] != train.get('global_prediction_tokens_per_update')
            or last['lr'] != train.get('lr') or last['aux_adamw_lr'] != train.get('aux_adamw_lr')):
        return None, 'unknown'
    return entries, 'complete' if complete else 'partial'


def _training_descriptor(payload):
    """Describe checkpoint training evidence, never evaluation caller overrides."""
    mapping = lambda value: value if isinstance(value, dict) else {}
    config, progress, extra = (mapping(payload.get(key)) for key in ('config', 'progress', 'extra'))
    train, model, data = (mapping(config.get(key)) for key in ('train', 'model', 'data'))
    context_history, context_history_status = _context_history_descriptor(progress, data.get('context', 512))
    batch_history, batch_history_status = _batch_history_descriptor(progress, train)
    def integer(value, minimum=0):
        return value if type(value) is int and value >= minimum else None
    def text(value):
        return value if isinstance(value, str) and value else None
    def number(value, nonnegative=True):
        return value if type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0) else None
    schedule = mapping(progress.get('distributed_schedule'))
    candidates = []
    for source, values in (('progress.distributed_schedule', schedule), ('extra', extra)):
        if 'cooldown_steps' in values:
            raw = values['cooldown_steps']
            if raw is None or integer(raw, 1) is not None:
                candidates.append((source, raw))
    notes = []
    cooldown_steps = None
    start = integer(progress.get('cooldown_start_tokens'))
    if candidates and len({value for _, value in candidates}) > 1:
        cooldown_status = 'unknown'
        notes.append('conflicting checkpoint cooldown metadata')
    elif candidates:
        cooldown_steps = candidates[0][1]
        cooldown_status = 'uncooled' if cooldown_steps is None else 'cooled'
        if cooldown_status == 'uncooled' and start is not None:
            cooldown_status = 'unknown'
            notes.append('cooldown start conflicts with explicit uncooled metadata')
    elif start is not None:
        cooldown_status = 'cooled'
        notes.append('cooldown branch start exists but its step horizon is missing')
    else:
        cooldown_status = 'unknown'
        notes.append('checkpoint has no explicit cooldown branch evidence')
    tokens = integer(progress.get('tokens'))
    rung = integer(train.get('rung_total_tokens'), 1)
    batch = integer(train.get('global_prediction_tokens_per_update'), 1)
    scheduler_step = integer(progress.get('scheduler_step'))
    target = rung if cooldown_status == 'uncooled' else None
    if cooldown_status == 'cooled' and start is not None and cooldown_steps is not None and batch is not None:
        target = start + cooldown_steps * batch
    endpoint_invalid = False
    if 'cooldown_end_tokens' in progress:
        endpoint = integer(progress['cooldown_end_tokens'], 1)
        if (cooldown_status != 'cooled' or endpoint is None or start is None
                or cooldown_steps is None or batch is None
                or not start + (cooldown_steps - 1) * batch < endpoint <= start + cooldown_steps * batch):
            endpoint_invalid = True
            target = None
            notes.append('invalid explicit cooldown token endpoint')
        else:
            target = endpoint
    completion = 'unknown'
    if target is not None and tokens is not None:
        completion = 'complete' if tokens >= target else 'partial'
        if cooldown_status == 'cooled' and scheduler_step is not None and (tokens >= target) != (scheduler_step >= cooldown_steps):
            completion = 'unknown'
            notes.append('cooldown token and scheduler-step completion evidence disagree')
    elif not endpoint_invalid and cooldown_status == 'cooled' and cooldown_steps is not None and scheduler_step is not None:
        completion = 'complete' if scheduler_step >= cooldown_steps else 'partial'
    if train.get('schedule') == 'cosine':
        expected_steps = (rung + batch - 1) // batch if rung is not None and batch is not None else None
        if (cooldown_status != 'uncooled' or expected_steps is None or scheduler_step is None
                or tokens is None or scheduler_step > expected_steps or tokens > rung
                or (tokens == rung) != (scheduler_step == expected_steps)):
            completion = 'unknown'
            notes.append('cosine token and scheduler horizon evidence is missing or inconsistent')
    return {'version': 1, 'source': 'checkpoint_payload',
            'config_compatibility': copy.deepcopy(payload.get('config_compatibility', {})),
            'configuration': copy.deepcopy({key: value for key, value in config.items() if key != 'project'}),
            'optimizer': text(train.get('optimizer')), 'seed': integer(train.get('seed')),
            'width': integer(model.get('d_model'), 1), 'depth': integer(model.get('layers'), 1),
            'train_context': integer(data.get('context'), 1), 'rung_total_tokens': rung,
            'context_history': context_history, 'context_history_status': context_history_status,
            'context_history_scope': 'checkpoint-recorded context segments; train_context is the current segment, not a claim of uniform lifetime context',
            'batch_history': batch_history, 'batch_history_status': batch_history_status,
            'batch_history_scope': 'checkpoint-recorded batch/LR segments; current batch and LR do not imply a uniform lifetime policy; null aux_adamw_lr falls back to that segment LR',
            'normalization': text(model.get('normalization')), 'attention': text(model.get('attention')),
            'width_scaling_base': ('disabled' if 'width_scaling_base' in model and model['width_scaling_base'] is None
                                   else integer(model.get('width_scaling_base'), 1)),
            'width_lr_exponent': number(model.get('width_lr_exponent'), nonnegative=False),
            'width_logit_exponent': number(model.get('width_logit_exponent'), nonnegative=False),
            'depth_scaling_base': ('disabled' if 'depth_scaling_base' in model and model['depth_scaling_base'] is None
                                   else integer(model.get('depth_scaling_base'), 1)),
            'depth_lr_exponent': number(model.get('depth_lr_exponent'), nonnegative=False),
            'depth_residual_base': ('disabled' if 'depth_residual_base' in model and model['depth_residual_base'] is None
                                   else integer(model.get('depth_residual_base'), 1)),
            'depth_residual_exponent': number(model.get('depth_residual_exponent'), nonnegative=False),
            'residual_alpha': number(model.get('residual_alpha'), nonnegative=False),
            'weight_decay_hidden': number(train.get('weight_decay_hidden')),
            'beta1': number(train.get('beta1')), 'beta2': number(train.get('beta2')),
            'aux_adamw_lr': ('default_to_lr' if 'aux_adamw_lr' in train and train['aux_adamw_lr'] is None
                             else number(train.get('aux_adamw_lr'))),
            'actual_prediction_tokens': tokens, 'optimizer_steps': integer(progress.get('steps')),
            'global_prediction_tokens_per_update': batch, 'lr': number(train.get('lr')),
            'schedule': text(train.get('schedule')), 'scheduler_step': scheduler_step,
            'warmup_steps': integer(progress.get('warmup_steps'), 1),
            'cooldown_status': cooldown_status, 'cooldown_steps': cooldown_steps,
            'cooldown_start_tokens': start, 'segment_target_prediction_tokens': target,
            'completion_status': completion, 'evidence_notes': notes,
            'completion_scope': 'declared training segment token/step endpoint only; no automatic experiment completion, fairness, or ranking claim'}


def _training_comparison(left, right):
    """Report evidence dimensions; matching targets never certifies fair ranking."""
    fields = ('optimizer', 'seed', 'width', 'depth', 'train_context', 'context_history', 'batch_history', 'rung_total_tokens',
              'actual_prediction_tokens', 'global_prediction_tokens_per_update', 'lr',
              'normalization', 'attention', 'width_scaling_base', 'width_lr_exponent',
              'width_logit_exponent', 'aux_adamw_lr', 'schedule', 'cooldown_status', 'cooldown_steps',
              'segment_target_prediction_tokens', 'completion_status', 'warmup_steps',
              'depth_scaling_base', 'depth_lr_exponent', 'depth_residual_base',
              'depth_residual_exponent', 'residual_alpha', 'weight_decay_hidden', 'beta1', 'beta2')
    def descriptor(report):
        value = report.get('training_descriptor')
        return value if isinstance(value, dict) and value.get('version') == 1 and value.get('source') == 'checkpoint_payload' else {}
    descriptors = descriptor(left), descriptor(right)
    dimensions = {}
    for field in fields:
        values = [value.get(field) for value in descriptors]
        if field == 'cooldown_steps':
            values = ['not_applicable' if desc.get('cooldown_status') == 'uncooled' and field in desc and value is None else value
                      for desc, value in zip(descriptors, values)]
        known = all(value is not None and value != 'unknown' for value in values)
        if field == 'context_history':
            known = known and all(desc.get('context_history_status') == 'complete' for desc in descriptors)
        if field == 'batch_history':
            known = known and all(desc.get('batch_history_status') == 'complete' for desc in descriptors)
        status = ('matched' if values[0] == values[1] else 'different') if known else 'unknown'
        dimensions[field] = {'status': status, 'left': values[0], 'right': values[1]}
    return {'dimensions': dimensions,
            'matched_dimensions': [key for key, value in dimensions.items() if value['status'] == 'matched'],
            'different_dimensions': [key for key, value in dimensions.items() if value['status'] == 'different'],
            'unknown_dimensions': [key for key, value in dimensions.items() if value['status'] == 'unknown'],
            'automatic_ranking_eligible': False,
            'scope': 'training evidence summary only; general matched-target contrasts remain allowed; cooled and uncooled checkpoints or different training budgets are not automatically fair rankings; equality of missing evidence is never inferred'}


def evaluate_checkpoint(cfg, checkpoint, out_dir, *, context=None,
                        max_prediction_tokens=1_000_000, max_seconds,
                        device='cpu', bootstrap_samples=1000, run_budget_usd=None):
    """Evaluate frozen validation targets, with a context-independent source-offset plan.

    Validation manifest must be pinned in the training config before checkpointing.
    Context is the only probe override; no training configuration is rewritten.
    Uncertainty resamples documents, not independently trained models or seeds.
    """
    import hashlib
    import torch.nn.functional as F
    from .checkpoint import load_checkpoint
    from .data import PackedSampler, manifest_hash, validate_tokenizer_compatibility

    if isinstance(cfg, (str, Path)):
        cfg = Config.from_yaml(str(cfg))
    project, data = cfg.raw.get('project', {}), cfg.raw.get('data', {})
    probe_context = data.get('context', 512) if context is None else context
    if project.get('dry_run', True):
        return {'dry_run': True, 'command': 'evaluate', 'checkpoint': str(checkpoint),
                'context': probe_context, 'max_prediction_tokens': max_prediction_tokens,
                'split': 'val', 'target_policy': 'canonical document order; source target offsets 1..L-1'}
    if (type(probe_context) is not int or probe_context <= 0 or
            type(max_prediction_tokens) is not int or max_prediction_tokens <= 0 or
            type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 100_000 or
            not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('positive evaluation bounds and bootstrap_samples in [0, 100000] required')
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    rate = project.get('gpu_hourly_rate_usd')
    if rate is None or (dev.type == 'cuda' and rate <= 0):
        raise ValueError('declare actual gpu_hourly_rate_usd (explicit 0 allowed for CPU)')
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f'evaluation directory already exists: {out_dir}')
    ledger = SpendLedger(project.get('spend_ledger', 'runs/spend.json'),
                         project.get('total_budget_usd', 1000), project.get('reserve_usd', 150),
                         stage_budgets=project.get('stage_budgets'))
    run_id = str(out_dir.resolve())
    ledger.begin(run_id, rate, max_seconds, run_budget_usd, parent_checkpoint=checkpoint,
                 cost_margin_fraction=project.get('cost_margin_fraction', .1), stage=project.get('stage', 'A'))
    started, status = time.monotonic(), 'failed'

    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError('evaluation runtime cap exhausted')

    def tensor_hash(items):
        digest = hashlib.sha256()
        for name, value in sorted(items):
            value = value.detach().cpu().contiguous()
            digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def file_hash(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                deadline()
                digest.update(block)
        return digest.hexdigest()

    try:
        out_dir.mkdir(parents=True, exist_ok=False)
        train_hash = manifest_hash(data['manifest'])
        validation_path = cfg.raw.get('evaluation', {}).get('validation_manifest', data['manifest'])
        tokenizer_compatibility = validate_tokenizer_compatibility(
            data['manifest'], validation_path, allow_test_only=dev.type == 'cpu')
        evaluation_seed = int(cfg.raw.get('evaluation', {}).get('seed', 0))
        sampler = PackedSampler(validation_path, split='val', batch_size=1,
                                context=probe_context, seed=evaluation_seed)
        deadline()
        ranges, remaining = [], max_prediction_tokens
        for doc in sampler.docs:
            length = doc['token_end'] - doc['token_start']
            count = min(max(0, length - 1), remaining)
            if count:
                ranges.append({'document_id': doc['id'], 'target_start': 1,
                               'target_end': count + 1, 'count': count})
                remaining -= count
            if not remaining:
                break
        if not ranges:
            raise ValueError('validation split has no prediction targets')
        from .data import audit_validation_overlap
        overlap_audit = audit_validation_overlap(data['manifest'], validation_path,
                                                 document_ids=[row['document_id'] for row in ranges])
        (out_dir / 'data-overlap-audit.json').write_text(json.dumps(overlap_audit, indent=2) + '\n')
        if overlap_audit['has_overlap']:
            raise ValueError('held-out validation has exact whole-document content overlap; see data-overlap-audit.json')
        deadline()
        model = TensorLM(cfg.model)  # CPU loading avoids transient duplicate GPU state.
        payload = load_checkpoint(checkpoint, model, config_validator=evaluation_validator(cfg),
                                  expected_manifest_hash=train_hash, restore_rng=False,
                                  map_location='cpu')
        model.to(dev).eval()
        checkpoint_progress, checkpoint_code = payload['progress'], payload['code']
        training_descriptor = _training_descriptor(payload)
        del payload  # discard unused CPU optimizer tensors during long validation
        for norm in lagged_modules(model):
            norm.collecting = False
            norm.set_valid_mask(None)
            _check_lagged_divisor(norm)
        deadline()
        before_model = tensor_hash(model.state_dict().items())
        before_buffers = tensor_hash(model.named_buffers())
        plan = {'version': 1, 'validation_manifest_sha256': sampler.manifest_hash,
                'split': 'val', 'ordering': 'immutable documents.jsonl order',
                'offset_semantics': 'processed document stream; includes synthetic EOS if present',
                'ranges': ranges}
        plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        selected = {row['document_id']: row for row in ranges}
        totals = {row['document_id']: {'loss_sum': 0.0, 'tokens': 0} for row in ranges}
        prefixes, forward_input_tokens, forwarded_batches = {}, 0, 0
        prefix_digest = hashlib.sha256()
        initial_sampler = sampler.state_dict()
        with torch.inference_mode():
            for _ in range(sampler.epoch_row_count):
                deadline()
                idx, docs, original_targets = sampler.next_batch()
                targets = torch.full_like(original_targets, -100)
                positions = []
                for mapping in sampler.batch_mappings()[0]['ranges']:
                    row = selected.get(mapping['document_id'])
                    if row is None:
                        continue
                    start, end = mapping['packed_token_start'], mapping['packed_token_end']
                    for j in range(start, min(end - 1, probe_context)):
                        offset = mapping['source_token_start'] + j - start + 1
                        if row['target_start'] <= offset < row['target_end']:
                            if original_targets[0, j] == -100:
                                raise ValueError('selected validation target was masked by sampler')
                            targets[0, j] = original_targets[0, j]
                            positions.append((j, mapping['document_id'], j - start + 1, offset))
                if not positions:
                    continue
                if ((docs < 0) & (targets != -100)).any():
                    raise ValueError('padded-token targets in evaluation')
                logits = model(idx.to(dev), doc_ids=docs.to(dev))
                losses = F.cross_entropy(logits.float().flatten(0, 1), targets.to(dev).flatten(),
                                         reduction='none', ignore_index=-100).cpu()
                if not torch.isfinite(losses).all():
                    raise FloatingPointError('nonfinite validation loss')
                for j, doc_id, prefix, offset in positions:
                    if offset != selected[doc_id]['target_start'] + totals[doc_id]['tokens']:
                        raise ValueError('duplicate or missing source target offset in evaluation')
                    totals[doc_id]['loss_sum'] += float(losses[j])
                    totals[doc_id]['tokens'] += 1
                    prefixes[prefix] = prefixes.get(prefix, 0) + 1
                    prefix_digest.update(json.dumps([doc_id, offset, prefix], separators=(',', ':')).encode() + b'\n')
                forwarded_batches += 1
                forward_input_tokens += int((docs >= 0).sum())
                deadline()
        for row in ranges:
            if totals[row['document_id']]['tokens'] != row['count']:
                raise ValueError('validation target plan was not evaluated exactly once')
        after_model = tensor_hash(model.state_dict().items())
        after_buffers = tensor_hash(model.named_buffers())
        if before_model != after_model or before_buffers != after_buffers:
            raise RuntimeError('evaluation changed checkpoint model or normalization buffers')
        counts = np.asarray([row['tokens'] for row in totals.values()], dtype=np.int64)
        sums = np.asarray([row['loss_sum'] for row in totals.values()], dtype=np.float64)
        tokens, loss_sum = int(counts.sum()), float(sums.sum())
        ci = None
        if len(counts) >= 2 and bootstrap_samples:
            rng = np.random.default_rng(evaluation_seed)
            estimates = np.empty(bootstrap_samples, dtype=np.float64)
            for i in range(bootstrap_samples):
                deadline()
                draw = rng.integers(0, len(counts), size=len(counts))
                estimates[i] = sums[draw].sum() / counts[draw].sum()
            ci = np.quantile(estimates, [.025, .975]).tolist()
        runtime_sources = {f'tnlm/{p.name}': file_hash(p)
                           for p in sorted(Path(__file__).parent.glob('*.py'))}
        critical_sources = ('model.py', 'attention.py', 'normalization.py', 'config.py')
        source_matches = {name: runtime_sources.get(f'tnlm/{name}') ==
                          checkpoint_code.get('source_sha256', {}).get(f'tnlm/{name}')
                          for name in critical_sources}
        checkpoint_path = Path(checkpoint)
        result = {'kind': 'held_out_validation', 'evaluation_protocol_version': 2,
                  'data_overlap_audit': overlap_audit, 'tokenizer_compatibility': tokenizer_compatibility,
                  'split': 'val', 'context': probe_context,
                  'prediction_tokens': tokens, 'loss_sum': loss_sum, 'ce': loss_sum / tokens,
                  'validation_below_one_million_targets': tokens < 1_000_000,
                  'uncertainty': {'unit': 'documents, not training seeds', 'method': 'token-weighted ratio document bootstrap',
                                  'samples': bootstrap_samples, 'seed': evaluation_seed, 'ce_95_ci': ci,
                                  'documents': len(totals)},
                  'target_plan': plan, 'target_plan_sha256': plan_hash,
                  'per_document': [{'document_id': key, **row} for key, row in totals.items()],
                  'available_prefix_length_histogram': {str(k): v for k, v in sorted(prefixes.items())},
                  'available_prefix_plan_sha256': prefix_digest.hexdigest(),
                  'evaluation_seed': evaluation_seed,
                  'training_manifest_sha256': train_hash, 'validation_manifest_sha256': sampler.manifest_hash,
                  'checkpoint': str(checkpoint_path.resolve()),
                  'checkpoint_manifest_sha256': file_hash(checkpoint_path / 'manifest.json'),
                  'checkpoint_state_sha256': file_hash(checkpoint_path / 'state.pt'),
                  'checkpoint_progress': checkpoint_progress, 'training_descriptor': training_descriptor,
                  'operator_audit': model.operator_audit(),
                  'model_sha256': before_model, 'frozen_state': {'model_before': before_model, 'model_after': after_model,
                       'buffers_before': before_buffers, 'buffers_after': after_buffers, 'unchanged': True},
                  'sampler': {'initial': initial_sampler, 'final': sampler.state_dict(),
                              'rows_scanned': sampler.epoch_row_count, 'forwarded_batches': forwarded_batches},
                  'probe': {'context_override': context, 'batch_size': 1, 'device': str(dev),
                            'dtype': str(next(model.parameters()).dtype), 'autocast': False,
                            'checkpoint_code': checkpoint_code,
                            'runtime_source_sha256': runtime_sources,
                            'critical_model_sources_match_checkpoint': all(source_matches.values()),
                            'critical_model_source_matches': source_matches,
                            'evaluator_source_sha256': file_hash(Path(__file__)),
                            'torch_version': str(torch.__version__),
                            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                            'float32_matmul_precision': torch.get_float32_matmul_precision()},
                  'exposure': {'split': 'val', 'unique_prediction_tokens': tokens,
                               'forward_input_tokens': forward_input_tokens,
                               'final_test_exposed': False}}
        deadline()
        status = 'complete'
    finally:
        ledger.finish(run_id, time.monotonic() - started, status)
    result['billing'] = json.loads(ledger.path.read_text())['runs'][run_id]
    (out_dir / 'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result



def compare_evaluations(left, right, out_dir, *, bootstrap_samples=1000, seed=0):
    """Compare matched validation targets with a paired document bootstrap.

    Delta is CE_right - CE_left. Intervals describe nominal document uncertainty,
    not training-seed variability or correction for architecture/hyperparameter search.
    Passing out_dir=None validates and returns the report without writing files.
    """
    import hashlib

    if type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 100_000:
        raise ValueError('bootstrap_samples must be in [0, 100000]')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')

    def validate(value):
        report = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
        if not isinstance(report, dict) or report.get('kind') != 'held_out_validation' or report.get('split') != 'val':
            raise ValueError('comparison requires held_out_validation reports')
        frozen = report.get('frozen_state', {})
        if (frozen.get('unchanged') is not True or not frozen.get('model_before') or
                frozen.get('model_before') != frozen.get('model_after') or
                frozen.get('model_before') != report.get('model_sha256') or
                not frozen.get('buffers_before') or frozen.get('buffers_before') != frozen.get('buffers_after')):
            raise ValueError('evaluation lacks consistent frozen-state proof')
        plan = report.get('target_plan')
        if not isinstance(plan, dict) or plan.get('split') != 'val':
            raise ValueError('invalid validation target plan')
        digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != report.get('target_plan_sha256'):
            raise ValueError('target-plan SHA256 mismatch')
        if not plan.get('validation_manifest_sha256') or plan['validation_manifest_sha256'] != report.get('validation_manifest_sha256'):
            raise ValueError('validation manifest identity mismatch')
        context = report.get('context')
        prefix_hash = report.get('available_prefix_plan_sha256')
        if type(context) is not int or context <= 0 or not isinstance(prefix_hash, str) or len(prefix_hash) != 64 or any(c not in '0123456789abcdef' for c in prefix_hash):
            raise ValueError('invalid context or prefix provenance')
        audit = report.get('data_overlap_audit', {})
        if (report.get('evaluation_protocol_version') != 2 or audit.get('has_overlap') is not False or
                audit.get('training_manifest_hash') != report.get('training_manifest_sha256') or
                audit.get('validation_manifest_hash') != report.get('validation_manifest_sha256')):
            raise ValueError('evaluation lacks certified exact-document separation audit')
        ranges = plan.get('ranges')
        if not isinstance(ranges, list) or not ranges:
            raise ValueError('empty target plan')
        planned = {}
        for row in ranges:
            if not isinstance(row, dict):
                raise ValueError('invalid target range')
            name, start, end, count = (row.get(k) for k in ('document_id', 'target_start', 'target_end', 'count'))
            if (not isinstance(name, str) or name in planned or
                    any(type(x) is not int for x in (start, end, count)) or
                    start < 1 or end <= start or count != end - start):
                raise ValueError('invalid or duplicate target range/count')
            planned[name] = count
        ids_digest = hashlib.sha256(json.dumps(sorted(planned), separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
        if audit.get('selected_validation_documents') != len(planned) or audit.get('selected_validation_ids_sha256') != ids_digest:
            raise ValueError('document separation audit differs from target plan')
        rows = report.get('per_document')
        if not isinstance(rows, list):
            raise ValueError('missing per-document losses')
        by_document = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('invalid per-document loss')
            name, tokens, loss = row.get('document_id'), row.get('tokens'), row.get('loss_sum')
            if (not isinstance(name, str) or name in by_document or type(tokens) is not int or
                    tokens <= 0 or not isinstance(loss, (float, int)) or isinstance(loss, bool) or
                    not math.isfinite(loss) or loss < 0 or planned.get(name) != tokens):
                raise ValueError('invalid per-document ID/count/loss')
            by_document[name] = (tokens, float(loss))
        if set(by_document) != set(planned):
            raise ValueError('per-document IDs differ from target plan')
        total_tokens = sum(row[0] for row in by_document.values())
        total_loss = sum(row[1] for row in by_document.values())
        if (type(report.get('prediction_tokens')) is not int or report['prediction_tokens'] != total_tokens or
                not isinstance(report.get('loss_sum'), (int, float)) or
                not math.isclose(report['loss_sum'], total_loss, rel_tol=1e-12, abs_tol=1e-10) or
                not isinstance(report.get('ce'), (int, float)) or
                not math.isclose(report['ce'], total_loss / total_tokens, rel_tol=1e-12, abs_tol=1e-10)):
            raise ValueError('aggregate losses/counts disagree with per-document state')
        return report, by_document

    left, left_docs = validate(left)
    right, right_docs = validate(right)
    if left['target_plan'] != right['target_plan'] or left['validation_manifest_sha256'] != right['validation_manifest_sha256']:
        raise ValueError('evaluations have different target plans or validation manifests')
    if set(left_docs) != set(right_docs) or any(left_docs[k][0] != right_docs[k][0] for k in left_docs):
        raise ValueError('evaluations have different per-document targets/counts')
    prefix_match = left['available_prefix_plan_sha256'] == right['available_prefix_plan_sha256']
    if left['context'] == right['context'] and not prefix_match:
        raise ValueError('same-context evaluations have different available prefix plans')
    names = sorted(left_docs)
    counts = np.asarray([left_docs[k][0] for k in names], dtype=np.int64)
    left_sums = np.asarray([left_docs[k][1] for k in names], dtype=np.float64)
    right_sums = np.asarray([right_docs[k][1] for k in names], dtype=np.float64)
    differences = right_sums - left_sums
    tokens = int(counts.sum())
    delta = float(differences.sum() / tokens)
    ci = None
    if len(names) >= 2 and bootstrap_samples:
        rng = np.random.default_rng(seed)
        estimates = np.empty(bootstrap_samples, dtype=np.float64)
        for i in range(bootstrap_samples):
            draw = rng.integers(0, len(names), size=len(names))
            estimates[i] = differences[draw].sum() / counts[draw].sum()
        ci = np.quantile(estimates, [.025, .975]).tolist()

    def identity(report):
        return {key: report.get(key) for key in (
            'checkpoint', 'checkpoint_manifest_sha256', 'checkpoint_state_sha256',
            'checkpoint_progress', 'training_descriptor', 'model_sha256', 'operator_audit', 'probe',
            'context', 'evaluation_seed', 'available_prefix_plan_sha256',
            'available_prefix_length_histogram', 'ce')}

    result = {'kind': 'matched_validation_comparison', 'left': identity(left), 'right': identity(right),
              'validation_manifest_sha256': left['validation_manifest_sha256'],
              'target_plan_sha256': left['target_plan_sha256'],
              'prediction_tokens': tokens, 'documents': len(names),
              'contexts_match': left['context'] == right['context'],
              'available_prefix_plans_match': prefix_match,
              'training_comparison': _training_comparison(left, right),
              'automatic_ranking_eligible': False,
              'prefix_comparison': 'identical available prefixes' if prefix_match else 'different available prefixes across contexts',
              'delta_ce_right_minus_left': delta,
              'uncertainty': {'method': 'paired document bootstrap of token-weighted CE difference',
                              'unit': 'nominal document uncertainty; not training seeds or HPO correction',
                              'bootstrap_samples': bootstrap_samples, 'seed': seed, 'delta_ce_95_ci': ci},
              'interpretation': 'matched-target predictive loss comparison; training schedule/budget differences and unknowns are explicitly reported; no automatic fair ranking, promotion, or interpretability conclusion'}
    if out_dir is None:
        return result
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result



def evaluate_mechanisms(cfg, checkpoint, task_manifest, out_dir, *, max_examples=32,
                        max_seconds=300, device='cpu', components=('0:attn', '0:mlp'),
                        run_budget_usd=None, max_prompt_tokens=None):
    """Answer-only exact-label probes, branch removal, and activation transfer.

    Generator composition holdout does not verify checkpoint training exposure.
    These tasks are neither an LM-quality proxy nor a unique causal mediation test.
    An explicit max_prompt_tokens overrides the configured training context for this
    probe only; prompts are never truncated.
    """
    import hashlib
    import importlib.metadata
    import torch.nn.functional as F
    from .checkpoint import load_checkpoint
    from .data import load_mechanism_tasks, manifest_hash
    from .tasks import audit_mechanism_training

    if isinstance(cfg, (str, Path)):
        cfg = Config.from_yaml(str(cfg))
    project = cfg.raw.get('project', {})
    components = tuple(components)
    cap = cfg.raw.get('data', {}).get('context', 512) if max_prompt_tokens is None else max_prompt_tokens
    if project.get('dry_run', True):
        return {'dry_run': True, 'command': 'evaluate_mechanisms', 'checkpoint': str(checkpoint),
                'task_manifest': str(task_manifest), 'max_examples': max_examples,
                'max_prompt_tokens': cap, 'components': list(components)}
    if (type(max_examples) is not int or max_examples <= 0 or type(cap) is not int or cap <= 0 or
            not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('positive bounded mechanism evaluation limits required')
    parsed = []
    for component in components:
        try:
            layer, branch = component.split(':')
            layer = int(layer)
        except (AttributeError, ValueError) as error:
            raise ValueError('component must be layer:attn or layer:mlp') from error
        if branch not in ('attn', 'mlp') or not 0 <= layer < cfg.model.layers or component in [c[0] for c in parsed]:
            raise ValueError('invalid or duplicate branch-removal component')
        parsed.append((component, layer, branch))
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    rate = project.get('gpu_hourly_rate_usd')
    if rate is None or (dev.type == 'cuda' and rate <= 0):
        raise ValueError('declare actual gpu_hourly_rate_usd (explicit 0 allowed for CPU)')
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f'mechanism evaluation directory already exists: {out_dir}')
    ledger = SpendLedger(project.get('spend_ledger', 'runs/spend.json'),
                         project.get('total_budget_usd', 1000), project.get('reserve_usd', 150),
                         stage_budgets=project.get('stage_budgets'))
    run_id = str(out_dir.resolve())
    ledger.begin(run_id, rate, max_seconds, run_budget_usd, parent_checkpoint=checkpoint,
                 cost_margin_fraction=project.get('cost_margin_fraction', .1), stage=project.get('stage', 'A'))
    started, status = time.monotonic(), 'failed'

    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError('mechanism evaluation runtime cap exhausted')

    def file_hash(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                deadline()
                digest.update(block)
        return digest.hexdigest()

    def tensor_hash(items):
        digest = hashlib.sha256()
        for name, value in sorted(items):
            value = value.detach().cpu().contiguous()
            digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    try:
        out_dir.mkdir(parents=True, exist_ok=False)
        metadata, all_rows = load_mechanism_tasks(task_manifest, split='val')
        rows = all_rows[:max_examples]
        if not rows:
            raise ValueError('mechanism validation split is empty')
        task_path = Path(task_manifest)
        if task_path.is_dir():
            task_path /= 'manifest.json'
        task_hash = manifest_hash(task_path)
        training_hash = manifest_hash(cfg.raw['data']['manifest'])
        import tiktoken
        tokenizer = tiktoken.get_encoding('gpt2')
        ranks = sorted((key.hex(), value) for key, value in tokenizer._mergeable_ranks.items())
        tokenizer_fingerprint = hashlib.sha256(json.dumps({
            'ranks': ranks, 'regex_pattern': tokenizer._pat_str,
            'special_tokens': tokenizer._special_tokens}, sort_keys=True).encode()).hexdigest()
        training_path = Path(cfg.raw['data']['manifest'])
        if training_path.is_dir():
            training_path /= 'manifest.json'
        training_bytes = training_path.read_bytes()
        if hashlib.sha256(training_bytes).hexdigest() != training_hash:
            raise ValueError('training manifest changed during mechanism evaluation')
        training_tokenizer = json.loads(training_bytes)['tokenizer']
        if (training_tokenizer.get('name') != 'gpt2' or training_tokenizer.get('eos_id') != 50256 or
                training_tokenizer.get('fingerprint_sha256') != tokenizer_fingerprint):
            raise ValueError('mechanism prompts require the checkpoint training GPT2 tokenizer fingerprint')
        del ranks
        training_exposure = audit_mechanism_training(task_path, training_path)
        if (training_exposure['benchmark_manifest_hash'] != task_hash or
                training_exposure['training_manifest_hash'] != training_hash):
            raise ValueError('manifest changed during mechanism exposure audit')
        deadline()
        encoded = []
        candidate_ids = {}
        for task, digits in [('boolean_hierarchy', range(2)), ('associative_retrieval', range(4))]:
            ids = [tokenizer.encode_ordinary(f' {digit}') for digit in digits]
            if any(len(tokens) != 1 for tokens in ids):
                raise ValueError('mechanism candidate answers must be exactly one GPT2 token')
            candidate_ids[task] = [tokens[0] for tokens in ids]
        for row in rows:
            deadline()
            prompt = tokenizer.encode_ordinary(row['prompt'])
            answer = tokenizer.encode_ordinary(row['answer'])
            if len(answer) != 1 or answer[0] not in candidate_ids[row['task']]:
                raise ValueError('mechanism answer must be exactly one task-candidate token')
            if not prompt or len(prompt) > cap:
                raise ValueError('mechanism prompt exceeds explicit context cap; no truncation permitted')
            if tokenizer.encode_ordinary(row['text']) != prompt + answer:
                raise ValueError('mechanism prompt/answer token boundary is not additive')
            if max(prompt + answer) >= cfg.model.vocab_size:
                raise ValueError('mechanism GPT2 tokens exceed checkpoint vocabulary')
            encoded.append((row, prompt, answer[0]))
        model = TensorLM(cfg.model)
        payload = load_checkpoint(checkpoint, model, expected_manifest_hash=training_hash,
                                  config_validator=evaluation_validator(cfg), restore_rng=False, map_location='cpu')
        checkpoint_progress, checkpoint_code = payload['progress'], payload['code']
        training_descriptor = _training_descriptor(payload)
        del payload
        model.to(dev).eval()
        for norm in lagged_modules(model):
            norm.collecting = False
            norm.set_valid_mask(None)
            _check_lagged_divisor(norm)
        before_parameters = tensor_hash(model.named_parameters())
        before_buffers = tensor_hash(model.named_buffers())
        original_ablations = [block.ablate for block in model.blocks]
        examples = []
        forward_tokens = 0
        reuse_predictions = 0
        cached_vectors = {}

        def score(prompt, answer, task):
            nonlocal forward_tokens
            deadline()
            idx = torch.tensor([prompt], dtype=torch.long, device=dev)
            logits = model(idx, doc_ids=torch.zeros_like(idx))[0, -1].float()
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('nonfinite mechanism logits')
            target = torch.tensor([answer], dtype=torch.long, device=dev)
            ce = float(F.cross_entropy(logits[None], target))
            candidates = candidate_ids[task]
            prediction = int(logits.argmax())
            constrained = candidates[int(logits[candidates].argmax())]
            forward_tokens += len(prompt)
            deadline()
            return {'answer_ce': ce, 'full_vocab_prediction_token': prediction,
                    'full_vocab_correct': prediction == answer,
                    'candidate_prediction_token': constrained, 'candidate_correct': constrained == answer}

        try:
            with torch.inference_mode():
                for row, prompt, answer in encoded:
                    for block in model.blocks:
                        block.ablate = frozenset()
                    captures, handles = {}, []
                    def capture(name):
                        def hook(module, inputs, output):
                            vector = output[0, -1].detach().cpu().clone()
                            if vector.ndim != 1 or vector.numel() != cfg.model.d_model or not bool(torch.isfinite(vector).all()):
                                raise FloatingPointError('invalid final-position mechanism component vector')
                            captures[name] = vector
                        return hook
                    try:
                        for name, layer, branch in parsed:
                            module = getattr(model.blocks[layer], branch)
                            handles.append(module.register_forward_hook(capture(name)))
                        baseline = score(prompt, answer, row['task'])
                    finally:
                        for handle in handles:
                            handle.remove()
                    if set(captures) != set(components):
                        raise RuntimeError('mechanism baseline did not execute every requested component')
                    cached_vectors[row['id']] = captures
                    contrasts = {}
                    for name, layer, branch in parsed:
                        model.blocks[layer].ablate = frozenset({branch})
                        try:
                            measured = score(prompt, answer, row['task'])
                        finally:
                            model.blocks[layer].ablate = frozenset()
                        measured['delta_answer_ce'] = measured['answer_ce'] - baseline['answer_ce']
                        measured['delta_full_vocab_correct'] = int(measured['full_vocab_correct']) - int(baseline['full_vocab_correct'])
                        measured['delta_candidate_correct'] = int(measured['candidate_correct']) - int(baseline['candidate_correct'])
                        contrasts[name] = measured
                    examples.append({'id': row['id'], 'task': row['task'], 'composition_id': row['composition_id'],
                                     'prompt': row['prompt'], 'answer': row['answer'], 'answer_token': answer,
                                     'prompt_tokens': len(prompt), 'baseline': baseline, 'branch_removal': contrasts})
                # Capture every donor on the unmodified model before any transfer.
                # Only d-dimensional final-position vectors survive baseline forwards.
                for target, (row, prompt, answer) in zip(examples, encoded):
                    target['branch_reuse'] = {}
                    for name, layer, branch in parsed:
                        transfers = {}
                        for donor_class in ('same_answer', 'different_answer'):
                            eligible = [candidate for candidate in rows
                                if candidate['task'] == row['task']
                                and candidate['composition_id'] != row['composition_id']
                                and ((candidate['answer'] == row['answer']) == (donor_class == 'same_answer'))]
                            if not eligible:
                                transfers[donor_class] = {'status': 'skipped',
                                    'reason': 'no same-task different-composition donor with ' + donor_class.replace('_', ' ')}
                                continue
                            donor = min(eligible, key=lambda candidate: candidate['id'])
                            vector = cached_vectors[donor['id']][name]
                            def patch(module, inputs, output):
                                patched = output.clone()
                                patched[0, -1] = vector.to(device=output.device, dtype=output.dtype)
                                return patched
                            module = getattr(model.blocks[layer], branch)
                            handle = module.register_forward_hook(patch)
                            try:
                                measured = score(prompt, answer, row['task'])
                            finally:
                                handle.remove()
                            reuse_predictions += 1
                            baseline = target['baseline']
                            measured.update(status='patched', donor_id=donor['id'],
                                donor_composition_id=donor['composition_id'], donor_answer=donor['answer'],
                                donor_task=donor['task'], donor_prompt_tokens=len(next(
                                    item[1] for item in encoded if item[0]['id'] == donor['id'])),
                                delta_answer_ce=measured['answer_ce'] - baseline['answer_ce'],
                                delta_full_vocab_correct=int(measured['full_vocab_correct']) - int(baseline['full_vocab_correct']),
                                delta_candidate_correct=int(measured['candidate_correct']) - int(baseline['candidate_correct']))
                            transfers[donor_class] = measured
                        target['branch_reuse'][name] = transfers
        finally:
            for block, original in zip(model.blocks, original_ablations):
                block.ablate = original
        after_parameters = tensor_hash(model.named_parameters())
        after_buffers = tensor_hash(model.named_buffers())
        if before_parameters != after_parameters or before_buffers != after_buffers:
            raise RuntimeError('mechanism evaluation mutated checkpoint parameters/buffers')

        def summarize(selected):
            n = len(selected)
            def metrics(values):
                return {'answer_ce': sum(v['answer_ce'] for v in values) / n,
                        'full_vocab_accuracy': sum(v['full_vocab_correct'] for v in values) / n,
                        'candidate_accuracy': sum(v['candidate_correct'] for v in values) / n}
            base = metrics([row['baseline'] for row in selected])
            contrast = {}
            for name in components:
                values = [row['branch_removal'][name] for row in selected]
                contrast[name] = {**metrics(values),
                    'paired_mean_delta_answer_ce': sum(v['delta_answer_ce'] for v in values) / n,
                    'paired_mean_delta_full_vocab_accuracy': sum(v['delta_full_vocab_correct'] for v in values) / n,
                    'paired_mean_delta_candidate_accuracy': sum(v['delta_candidate_correct'] for v in values) / n}
            reuse = {}
            for name in components:
                reuse[name] = {}
                for donor_class in ('same_answer', 'different_answer'):
                    values = [row['branch_reuse'][name][donor_class] for row in selected]
                    patched = [value for value in values if value['status'] == 'patched']
                    count = len(patched)
                    summary = {'patched_examples': count, 'skipped_examples': n - count}
                    if count:
                        summary.update(answer_ce=sum(v['answer_ce'] for v in patched) / count,
                            full_vocab_accuracy=sum(v['full_vocab_correct'] for v in patched) / count,
                            candidate_accuracy=sum(v['candidate_correct'] for v in patched) / count,
                            paired_mean_delta_answer_ce=sum(v['delta_answer_ce'] for v in patched) / count,
                            paired_mean_delta_full_vocab_accuracy=sum(v['delta_full_vocab_correct'] for v in patched) / count,
                            paired_mean_delta_candidate_accuracy=sum(v['delta_candidate_correct'] for v in patched) / count)
                    else:
                        summary['status'] = 'unavailable: no eligible donors'
                    reuse[name][donor_class] = summary
            return {'examples': n, 'baseline': base, 'branch_removal': contrast, 'branch_reuse': reuse}

        runtime_sources = {str(path.relative_to(Path(__file__).parent.parent)): file_hash(path)
                           for path in sorted(Path(__file__).parent.rglob('*.py'))}
        checkpoint_path = Path(checkpoint)
        result = {'kind': 'known_mechanism_evaluation', 'split': 'val', 'examples': examples,
                  'summary': summarize(examples),
                  'per_task': {task: summarize([row for row in examples if row['task'] == task])
                               for task in sorted({row['task'] for row in examples})},
                  'per_composition': {comp: summarize([row for row in examples if row['composition_id'] == comp])
                                      for comp in sorted({row['composition_id'] for row in examples})},
                  'probe_ids': [row['id'] for row in examples],
                  'task_manifest_sha256': task_hash, 'task_metadata': metadata,
                  'training_manifest_sha256': training_hash,
                  'checkpoint': str(checkpoint_path.resolve()),
                  'checkpoint_manifest_sha256': file_hash(checkpoint_path / 'manifest.json'),
                  'checkpoint_state_sha256': file_hash(checkpoint_path / 'state.pt'),
                  'checkpoint_progress': checkpoint_progress, 'training_descriptor': training_descriptor,
                  'checkpoint_code': checkpoint_code,
                  'runtime_source_sha256': runtime_sources, 'operator_audit': model.operator_audit(),
                  'probe': {'device': str(dev), 'max_examples': max_examples, 'max_prompt_tokens': cap,
                            'explicit_prompt_cap_override': max_prompt_tokens, 'components': list(components),
                            'autocast': False, 'tokenizer': 'gpt2',
                            'tiktoken_version': importlib.metadata.version('tiktoken'),
                            'tokenizer_fingerprint_sha256': tokenizer_fingerprint,
                            'candidate_token_ids': candidate_ids,
                            'reuse_policy': {'donor_selection': 'lexicographically smallest immutable ID among selected same-task different-composition probes, separately by answer agreement',
                                'patch_site': 'component output after projection, before output normalization',
                                'patched_positions': 'target final prompt position only; donor final prompt position',
                                'normalizers': 'live', 'cached_vector_elements': len(examples) * len(components) * cfg.model.d_model}},
                  'frozen_state': {'parameters_before': before_parameters, 'parameters_after': after_parameters,
                                   'buffers_before': before_buffers, 'buffers_after': after_buffers, 'unchanged': True},
                  'exposure': {'answer_prediction_tokens': len(examples) * (1 + len(components)),
                               'reuse_answer_prediction_tokens': reuse_predictions,
                               'total_answer_prediction_tokens': len(examples) * (1 + len(components)) + reuse_predictions,
                               'forward_input_tokens': forward_tokens, 'final_test_exposed': False},
                  'generator_composition_holdout_verified': True,
                  'benchmark_training_exposure_verified': training_exposure['verified'],
                  'training_exposure_audit': training_exposure,
                  'interpretation': 'training-exposure verification covers only declared checksum-bound train/calibration corpus metadata, not optimizer visitation or lifetime exposure; exact-label task evidence, not an LM-quality proxy or unique causal mediation claim; activation transfer contrasts do not prove component reuse'}
        deadline()
        # Serialize and test the output path while the reservation remains active.
        pending = out_dir / '.mechanism_evaluation.json.tmp'
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        deadline()
        status = 'complete'
    finally:
        ledger.finish(run_id, time.monotonic() - started, status)
    result['billing'] = json.loads(ledger.path.read_text())['runs'][run_id]
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    os.replace(pending, out_dir / 'mechanism_evaluation.json')
    return result

