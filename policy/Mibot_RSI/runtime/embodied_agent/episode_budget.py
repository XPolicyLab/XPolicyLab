"""Episode step policy; no robot motion or benchmark success logic is changed."""
import numbers


def read_budget(env):
    """The official step budget, read without writing it: the record configure_budget(env, 'benchmark') returns.

    The submission path (v2/submission.py) serves the robot inside RoboDojo's own evaluator, where step_lim stays
    exactly as the task set it; no other budget mode exists there."""
    limit = env.step_lim
    if isinstance(limit, bool) or not isinstance(limit, numbers.Integral) or limit <= 0:
        raise ValueError(f'the official step_lim must be a positive whole number, not {limit!r}')
    limit = int(limit)
    return {'budget_mode': 'benchmark', 'benchmark_compatible': True, 'benchmark_step_limit': limit,
            'step_limit': limit, 'step_limit_multiplier': 1.0}


def configure_budget(env, mode, exploration_step_limit=None):
    if mode not in ('exploration', 'feasibility', 'benchmark'):
        raise ValueError('budget mode must be exploration, feasibility or benchmark')
    if exploration_step_limit is not None:
        if mode != 'exploration' or type(exploration_step_limit) is not int or exploration_step_limit <= 0:
            raise ValueError('exploration step limit must be positive and used only in exploration mode')
    original = int(env.step_lim)
    # RoboDojo compares its monotonically increasing counter with step_lim.
    # Infinity disables that comparison without resetting time or observations.
    actual = exploration_step_limit if mode == 'exploration' else original * (3 if mode == 'feasibility' else 1)
    env.step_lim = float('inf') if actual is None else actual
    return {'budget_mode': mode, 'benchmark_compatible': mode == 'benchmark',
            'benchmark_step_limit': original,
            'step_limit': actual,
            'step_limit_multiplier': None if actual is None else actual / original}


def budget_observation(policy, steps):
    limit = policy['step_limit']
    return {**policy, 'physical_steps_remaining': None if limit is None else max(0, limit - int(steps))}


def budget_instructions(mode, exploration_step_limit=None):
    if mode == 'exploration':
        if exploration_step_limit is not None:
            return (f'This is a bounded exploration episode with a physical control-step limit of '
                    f'{exploration_step_limit}. It is not benchmark-compatible. Watch '
                    'physical_steps_remaining; use the budget for informative RGB-grounded tests '
                    'and record what they taught. Task completion is optional in an exploration '
                    'test; return both arms when pursuing a complete task trial and still possible. ')
        return ('This is an exploration episode with NO physical control-step limit. '
                'physical_steps_remaining=null means unlimited, not zero or unknown. '
                'Unlimited steps permit useful comparisons. Answering one question does not '
                'by itself end the investigation: use the current scene for the next unresolved '
                'task phase when it can provide evidence, or explain why a fresh scene is better. '
                'Choose the next physical action for the evidence it can provide. If pursuing a full task trial, '
                'return both arms while the episode remains active. Return-step estimates '
                'describe motion duration only, not a deadline. ')
    if mode == 'benchmark':
        return ('This is a benchmark episode with the original physical step budget. '
                'Task completion has priority over post-task arm return: do not abandon a feasible '
                'completion solely to reserve return steps. Use explicit short trajectory durations '
                'and batch waypoints when safe so long orientation interpolation does not consume the '
                'budget. After the task is complete, or no useful completion attempt remains, use the '
                'remaining budget to return both arms. Observation metadata estimates bookmark-return '
                'steps and should guide that final choice. ')
    if mode == 'feasibility':
        return ('This is a feasibility episode with a physical control-step limit equal to three times the '
                'original benchmark limit. It is not benchmark-compatible. Use the extra budget to '
                'continue useful attempts, recover from failures, and finish every required outcome. '
                'Track physical_steps_remaining and leave enough room for the best remaining completion '
                'attempts. After the task is complete, or no useful completion attempt remains, use the '
                'remaining budget to return both arms. ')
    raise ValueError('budget mode must be exploration, feasibility or benchmark')
