"""RSI-6 isolated adapter. No environment methods are called.

Call at initialization and EACH command boundary, after the existing native
update has returned. Caller supplies its update stamp, never a guessed step.
Keep records outside SFT observations. Benchmark mode is off by default.
"""
import json
from pathlib import Path


class EquationFeedback:
    def __init__(self, *, enabled=False):
        self.enabled = enabled
        self.previous_score = None
        self.previous_step = None

    def sample(self, env, *, action_id, native_updated_step,
               reward_registered, decision_reason, env_idx=0):
        if not self.enabled:
            return None
        step = int(env.take_action_cnt[env_idx])
        if self.previous_step is not None and step < self.previous_step:
            raise ValueError('Create a new adapter for each episode')
        manager = env.reward_manager
        binary_mode = (manager.score_meta[env_idx]['mode'] is None
                       and manager.final_score_meta[env_idx]['mode'] is None
                       and not hasattr(env, 'get_score'))
        fresh = reward_registered and native_updated_step == step
        ended = bool(env.end_flag[env_idx])
        success = bool(ended and env.success[env_idx]) if fresh else None
        score = float(success) if fresh and binary_mode else None
        delta = (score - self.previous_score
                 if score is not None and self.previous_score is not None else None)
        baseline = score is not None and self.previous_score is None
        names = ('correct_token_xy_within_0_018m', 'token_x_axis_within_45deg',
                 'token_y_axis_within_45deg', 'all_robot_back_to_origin')
        record = {
            'task': 'solve_equation', 'action_id': action_id,
            'uses_privileged_reward_feedback': True,
            'native_step': step, 'score_updated_step': native_updated_step,
            'remaining_steps': max(0, int(env.step_lim) - step),
            'current_score': score, 'delta_score': delta,
            'is_initial_baseline': baseline, 'native_success': success,
            'episode_ended': ended if fresh else None,
            'score_scale': {'raw': [0, 1], 'display': [0, 100],
                            'display_multiplier': 100} if binary_mode else None,
            'source': 'EvalEnv.end_flag AND EvalEnv.success; binary run_eval mapping',
            'status': 'available' if score is not None else 'unknown',
            'stage_conditions': {name: {'current': None, 'historical': None}
                                 for name in names},
            'stage_conditions_reason': 'Native checks are not individually cached',
            'decision_reason': decision_reason,
        }
        if score is not None:
            self.previous_score = score
        self.previous_step = step
        return record

    @staticmethod
    def append_sidecar(path, record):
        if record is None:
            return
        with Path(path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    @staticmethod
    def agent_diagnostic(record):
        """Add to next agent message separately; never mutate SFT observations."""
        return None if record is None else {'privileged_reward_feedback': record}
