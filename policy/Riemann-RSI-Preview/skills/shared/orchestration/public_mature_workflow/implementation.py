"""Reuse existing public mature factories behind the standard policy protocol."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import traceback

import numpy as np

from skills.shared.motion.public_observation_command_clock.implementation import (
    PublicHistory, PublicCommandClock, PublicActionBudgetReached,
)


def save(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        raise TypeError(type(obj).__name__)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, default=convert, indent=2) + '\n')
    temporary.replace(path)


def configure_planners(full_warmup):
    """Configure only this policy process, never the official eval process."""
    import torch
    import env.planner_manager.curobo_planner as planner
    torch.cuda.set_device(0)
    planner.BATCH_NUM = 1
    if not full_warmup:
        from curobo.batch_motion_planner import BatchMotionPlanner
        from curobo.motion_planner import MotionPlanner
        def skip_warmup(self, *args, **kwargs): return True
        MotionPlanner.warmup = skip_warmup
        BatchMotionPlanner.warmup = skip_warmup
        planner.CuroboPlanner._prewarm_alternate_modes = lambda self: None


def run(controller, profile, initial, exchange, output_root, *, horizon, prepare=configure_planners):
    """All arguments are policy-owned declarations or current public inputs."""
    from harness.runtime.mature_teacher_boundary import PublicRGBRecorder, PolicyActionClock
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    record = {'policy_error': None, 'traceback': None, 'official_success_claimed': False,
              'official_score_assigned': False, 'action_count_is_acknowledged_only': True}
    context = clock = None
    try:
        parameters = dict(profile['parameters'])
        if parameters.get('official_eval', True) is not True:
            raise ValueError('Only official evaluation is supported')
        for name, value in profile.get('environment', {}).items():
            if name in ('HOME', 'PATH', 'PYTHONPATH', 'LD_PRELOAD', 'CUDA_VISIBLE_DEVICES'):
                raise ValueError('Task parameters cannot alter process isolation: ' + name)
            os.environ[name] = str(value)
        options = dict(parameters, official_eval=True, output_dir=str(root / 'policy'),
                       public_instruction=initial.get('instruction', ''))
        prepare(bool(options.get('full_curobo_warmup', False)))
        history = PublicHistory(initial, root / 'policy')
        public_recorder = PublicRGBRecorder(history)
        args = SimpleNamespace(**options)
        mode = getattr(controller, '__name__', '').split('.')
        if len(mode) > 1 and mode[1] in ('pick_from_conveyor_by_image', 'match_and_pick_from_conveyor'):
            # Reuse the original idempotent policy-only capacity hook, before
            # this task's numeric planner/cache construction.
            from skills.shared.motion.public_conveyor_workspace.implementation import install_conveyor_planner_capacity
            install_conveyor_planner_capacity()
        environment = controller.create_public_environment(history.current_public_observation, public_recorder, args)
        if environment.step_lim != horizon or environment.dt != .004:
            raise ValueError('Public task factory differs from the declared official horizon')
        history.public_environment = environment
        from skills.shared.motion.declared_x5_command_targets.implementation import clock_environment
        clock = PublicCommandClock(clock_environment(environment), history, exchange, time_scale=options['motion_speed'])
        context = controller.create(environment, public_recorder, args, PolicyActionClock(clock))
        context.run()
    except PublicActionBudgetReached:
        record['own_command_budget_reached'] = True
    except Exception as error:
        # This is an explicit policy fallback, not an evaluator abort/score path.
        record.update(policy_error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        print(json.dumps({'event': 'policy_bug_enter_finite_hold', 'error': record['policy_error']}), flush=True)
        traceback.print_exc()
    finally:
        if clock is not None:
            record['acknowledged_actions'] = clock.action_count
            record['phases'] = clock.action_status
        if context is not None:
            try: record['context'] = context.result(record['policy_error'])
            except Exception as error: record['diagnostics_error'] = repr(error)
        save(root / 'policy-diagnostics.json', record)
