"""RSI-5: read-only diagnostics for audited general_pickup, opt-in only."""
import json
from pathlib import Path


class PickupFeedback:
    def __init__(self, env, sidecar):
        self.env = env
        self.path = Path(sidecar)
        self.previous = None
        self.baseline = None
        self.counter = 0
        self.latest = None
        rm = env.reward_manager
        expected = [[('is_lift', {'label': 'target', 'z_threshold': 0.1})]]
        # Registration must already have happened; no empty/reset state accepted.
        self.valid = (env.task_name == 'general_pickup' and env.step_lim == 200
                      and int(env.take_action_cnt[0]) == 0
                      and rm.check_list[0] == expected
                      and rm.score_meta[0]['mode'] is None
                      and rm.final_score_meta[0]['mode'] is None
                      and not rm.final_check_list[0]
                      and not rm.trigger_check_list[0]
                      and not rm.query_list[0] and not rm.trigger_query_list[0]
                      and not env.end_flag[0])

    def sample(self, action_id, reason):
        e = self.env
        step = int(e.take_action_cnt[0])
        # take_action_batch already calls reward_manager.step + is_episode_end.
        # success alone starts True, and is NOT a completion flag.
        success = bool(e.end_flag[0] and e.success[0]) if self.valid else None
        score = float(success) if success is not None else None
        history = (not bool(e.reward_manager.check_list[0])) if self.valid else None
        if self.baseline is None and self.valid and step == 0:
            self.baseline = score
            self.previous = score
        delta = score - self.previous if score is not None and self.previous is not None else None
        row = dict(task='general_pickup', action_id=action_id, decision_reason=reason,
                   native_step=step, remaining_steps=max(0, int(e.step_lim)-step),
                   current_score=score, delta_score=delta, initial_baseline=self.baseline,
                   raw_score_scale=[0, 1], display_score=None if score is None else 100*score,
                   display_score_scale=[0, 100], native_success=success,
                   stage_conditions={'lift_10cm': {'current': None, 'historically_achieved': history}},
                   source='EvalEnv.end_flag AND success; RewardManager.check_list (latched)',
                   status='available' if self.valid else 'unknown_contract_or_initialization',
                   uses_privileged_reward_feedback=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.previous = score
        self.latest = row
        return row

    def command_done(self, result):
        self.counter += 1
        self.sample(f'primitive-{self.counter}', result.get('note') or 'primitive completed')


class FeedbackResults(list):
    """Covers normal, rejected and early-return primitive result paths."""
    def __init__(self, feedback):
        super().__init__()
        self.feedback = feedback

    def append(self, result):
        super().append(result)
        self.feedback.command_done(result)
