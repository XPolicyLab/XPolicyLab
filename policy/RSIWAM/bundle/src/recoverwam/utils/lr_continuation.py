"""Absolute-step cosine continuation, preserving the full Adam training state."""
import math


def continuation_multiplier(step, spec):
    start, end = int(spec['start_step']), int(spec['end_step'])
    high, low = float(spec['start_ratio']), float(spec['end_ratio'])
    if not (0 <= start < end and 0 < low <= high <= 1):
        raise ValueError('Invalid LR continuation interval or ratios')
    if spec.get('schedule', 'cosine') != 'cosine':
        raise ValueError('The released continuation uses cosine decay')
    progress = min(max((int(step) - start) / (end - start), 0.0), 1.0)
    return low + (high - low) * (1 + math.cos(math.pi * progress)) / 2


def joint_multiplier(step, warmup_steps, tail):
    """Continuous joint schedule: warmup, original 80k cosine, low-LR tail."""
    tail_ratio = continuation_multiplier(step, tail)
    boundary, warmup = int(tail['start_step']), int(warmup_steps)
    if not 0 <= warmup < boundary:
        raise ValueError('Joint warmup must end before the low-LR tail starts')
    step = max(int(step), 0)
    if step > boundary:
        return tail_ratio
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / (boundary - warmup)
    floor = float(tail['start_ratio'])
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def validate_continuation_state(scheduler, optimizer, step, spec):
    if not int(spec['start_step']) <= step < int(spec['end_step']):
        raise ValueError(f'Resume step {step} is outside the continuation interval')
    return _validate_restored_lrs(scheduler, optimizer, step, continuation_multiplier(step, spec))


def validate_joint_state(scheduler, optimizer, step, warmup_steps, tail):
    if not 0 <= step < int(tail['end_step']):
        raise ValueError(f'Resume step {step} is outside the joint schedule')
    return _validate_restored_lrs(scheduler, optimizer, step, joint_multiplier(step, warmup_steps, tail))


def _validate_restored_lrs(scheduler, optimizer, step, ratio):
    scheduler = getattr(scheduler, 'scheduler', scheduler)
    if scheduler.last_epoch != step:
        raise ValueError(f'Scheduler step {scheduler.last_epoch} != trainer step {step}')
    actual = [float(group['lr']) for group in optimizer.param_groups]
    expected = [float(base) * ratio for base in scheduler.base_lrs]
    if len(actual) != len(expected) or not all(
        math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-15)
        for x, y in zip(actual, expected)
    ):
        raise ValueError(f'Restored LR discontinuity: actual={actual}, expected={expected}')
    return actual
