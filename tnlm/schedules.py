"""Global-step LR controls: legacy warmup/constant and fixed-horizon cosine."""
import math

CONSTANT = 'constant_with_separate_cooldown'
COSINE = 'cosine'


def schedule_steps(tokens, batch):
    if type(tokens) is not int or tokens <= 0 or type(batch) is not int or batch <= 0:
        raise ValueError('positive integer token horizon and batch required')
    return (tokens + batch - 1) // batch


def validate_schedule(schedule, total_steps, *, cooldown_steps=None):
    if schedule not in (CONSTANT, COSINE):
        raise ValueError('unsupported schedule')
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError('schedule requires positive integer total_steps')
    warmup = max(100, (total_steps + 99) // 100)
    if schedule == COSINE:
        if cooldown_steps is not None:
            raise ValueError('cosine does not support a separate cooldown')
        if total_steps <= warmup:
            raise ValueError('cosine requires total_steps greater than warmup_steps')
    return warmup


def schedule_multiplier(step, total_steps, *, schedule=CONSTANT, warmup_steps=None):
    """Zero-indexed GLOBAL update; cosine reaches zero at N-1 without restarts."""
    default_warmup = validate_schedule(schedule, total_steps)
    warmup = default_warmup if warmup_steps is None else warmup_steps
    if type(step) is not int or step < 0 or type(warmup) is not int or warmup <= 0:
        raise ValueError('step/warmup must be nonnegative/positive integers')
    if schedule == CONSTANT:
        return min(1., (step + 1) / warmup)
    if warmup != default_warmup or step >= total_steps:
        raise ValueError('cosine horizon/warmup mismatch or exhausted schedule')
    if step < warmup:
        return (step + 1) / warmup
    if step == total_steps - 1:
        return 0.
    return .5 * (1. + math.cos(math.pi * (step + 1 - warmup) / (total_steps - warmup)))
