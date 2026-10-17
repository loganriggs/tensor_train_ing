"""Explicit post-warmup batch/LR interventions on full serial checkpoints."""
import copy
import math


def branch_config(cfg, *, batch_tokens, lr_scale, resume, context_fork, cooldown_steps):
    if batch_tokens is None and lr_scale is None:
        return cfg, None
    if resume is None or batch_tokens is None or lr_scale is None:
        raise ValueError('batch branch requires resume, batch_tokens and lr_scale together')
    if type(batch_tokens) is not int or batch_tokens <= 0:
        raise ValueError('batch_tokens must be a positive integer')
    if type(lr_scale) not in (int, float) or not math.isfinite(lr_scale) or lr_scale <= 0:
        raise ValueError('lr_scale must be finite and positive')
    if context_fork or cooldown_steps is not None or cfg.train.schedule != 'constant_with_separate_cooldown':
        raise ValueError('batch branch requires constant schedule without cooldown or context fork')
    if batch_tokens == cfg.train.global_prediction_tokens_per_update:
        raise ValueError('batch branch requires a different batch token quota')
    child = copy.deepcopy(cfg)
    child.train.global_prediction_tokens_per_update = batch_tokens
    for field in ('lr', 'aux_adamw_lr'):
        value = getattr(child.train, field)
        if value is not None:
            scaled = value * lr_scale
            if not math.isfinite(scaled) or scaled <= 0:
                raise ValueError('scaled learning rates must be finite and positive')
            setattr(child.train, field, scaled)
    return child, {'old_batch_tokens': cfg.train.global_prediction_tokens_per_update,
                   'new_batch_tokens': batch_tokens, 'lr_scale': float(lr_scale)}


def validate_parent(payload, child):
    progress, extra = payload['progress'], payload.get('extra', {})
    if (extra.get('cooldown_steps') is not None or 'cooldown_start_tokens' in progress
            or progress.get('distributed_schedule', {}).get('cooldown_steps') is not None):
        raise ValueError('batch branch cannot start from a cooldown checkpoint')
    warmup, step = progress.get('warmup_steps'), progress.get('scheduler_step')
    if (type(warmup) is not int or warmup <= 0 or type(step) is not int or step < warmup):
        raise ValueError('batch branch requires completed parent warmup with recorded scheduler state')
    if (type(progress.get('tokens')) is not int or type(progress.get('steps')) is not int
            or progress['tokens'] < 0 or progress['steps'] < 0):
        raise ValueError('batch branch requires valid cumulative progress')
    if progress['tokens'] >= child.train.rung_total_tokens:
        raise ValueError('batch branch requires a token horizon beyond the parent cursor')


def history_entry(cfg, *, tokens, steps, source):
    return {'batch_tokens': cfg.train.global_prediction_tokens_per_update,
            'lr': cfg.train.lr, 'aux_adamw_lr': cfg.train.aux_adamw_lr,
            'start_tokens': tokens, 'start_step': steps, 'source': source}


def apply_branch(optimizer, sampler, progress, parent_cfg, child_cfg, parent, spec):
    def summary():
        return [{k: copy.deepcopy(v) for k, v in group.items() if k != 'params'}
                for group in optimizer.param_groups]
    before = summary()
    changes = []
    for group in optimizer.param_groups:
        rates = {key: float(group.get(key, group['lr'])) * spec['lr_scale'] for key in ('lr', 'base_lr')}
        if any(not math.isfinite(value) or value <= 0 for value in rates.values()):
            raise ValueError('scaled optimizer group rates must be finite and positive')
        changes.append(rates)
    for group, rates in zip(optimizer.param_groups, changes):
        group.update(rates)
    if 'batch_history' not in progress:
        progress['batch_history'] = [history_entry(parent_cfg, tokens=None, steps=None, source='legacy_unknown')]
    entry = history_entry(child_cfg, tokens=progress['tokens'], steps=progress['steps'], source='batch_branch')
    entry.update(lr_scale=spec['lr_scale'], parent=copy.deepcopy(parent),
                 source_sampler=copy.deepcopy(sampler.state_dict()),
                 optimizer_groups_before=before, optimizer_groups_after=summary(),
                 optimizer_policy='moments, step counters, betas, momentum and weight decay preserved; all group rates scaled',
                 schedule_policy='parent warmup and scheduler counters preserved; constant LR continuation')
    progress['batch_history'].append(entry)
