"""Toolbox collection diagnostics; observe native reward calls, never initiate them."""
import copy
import json
import math
import os
from pathlib import Path
import importlib.util

_spec=importlib.util.spec_from_file_location('rsi3_toolbox_supervisor',Path(__file__).with_name('toolbox_supervisor.py'))
supervision=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(supervision)


class RewardFeedback:
    def __init__(self, env):
        self.env = env
        self.enabled = (os.environ.get('ROBODAWN_RUN_MODE') == 'collection'
                        and os.environ.get('ROBODAWN_PRIVILEGED_REWARD_FEEDBACK') == '1')
        self.cache = None
        self.pending = []
        self.events = []
        self.baseline = None
        self.previous = None
        self.previous_observation = None
        self.serial = 0
        self.error = None
        self.action_context = {}
        self.score_serial = 0

    def __enter__(self):
        if not self.enabled:
            return self
        if getattr(self.env, 'task_name', None) != 'store_tools_in_toolbox':
            raise ValueError('This reward adapter is audited only for store_tools_in_toolbox')
        root = os.environ.get('ROBODAWN_REWARD_SIDECAR_DIR')
        if not root:
            raise ValueError('Collection reward feedback requires dedicated sidecar directory')
        self.path = Path(root)
        self.supervisor = supervision.core.Supervisor('store_tools_in_toolbox', int(self.env.step_lim))
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = (self.path / 'reward_feedback.jsonl').open('x', encoding='utf-8')
        self.rm = self.env.reward_manager
        self.original = self.rm.get_reward
        self.had_override = 'get_reward' in vars(self.rm)
        self.old_override = vars(self.rm).get('get_reward')

        def observe_native_call(*args, **kwargs):
            result = self.original(*args, **kwargs)
            try:
                self.capture(result)
            except Exception as exc:
                self.cache = None
                self.error = type(exc).__name__ + ': ' + str(exc)
            return result  # preserve native return identity and call count

        self.wrapper = observe_native_call
        self.rm.get_reward = self.wrapper
        self.env._rsi3_reward_feedback = self
        return self

    def capture(self, rewards):
        rm = self.rm
        step = int(self.env.take_action_cnt[0])
        meta = copy.deepcopy(rm.score_meta[0])
        supported = (meta == {'mode': 'transition', 'gradient': [25, 50, 75, 100]}
                     and rm.final_score_meta[0]['mode'] is None)
        gate = rm._gated_score_lst
        score = float(gate[0]) if supported and gate is not None else None
        if score is not None and not math.isfinite(score):
            score = None
        native = bool(float(rewards[0]) > 1 - 1e-3)
        self.score_serial += 1  # called only after the existing native evaluator
        count = int(rm.score_completed_count[0]) if supported else None
        self.cache = {'native_step': step, 'current_score': score, 'native_success': native,
                      'score_mode': meta.get('mode'), 'completed_transition_count': count,
                      'status': 'fresh' if score is not None else 'unknown_score_schema_or_gate'}
        self.cache.update(score_event='native-score-%08d' % self.score_serial if score is not None else None,
                          score_evaluated=True if score is not None else None)
        if self.baseline is None and score is not None:
            self.baseline = {'native_step': step, 'score': score,
                             'source': 'first observed native get_reward gate; no fabricated zero'}
            self.previous = score
            self.previous_observation = score
        remaining = []
        for event in self.pending:
            if event['native_step'] == step:
                self.emit(event)
            else:
                remaining.append(event)
        self.pending = remaining

    def snapshot(self):
        step = int(self.env.take_action_cnt[0])
        fresh = self.cache is not None and self.cache['native_step'] == step
        data = copy.deepcopy(self.cache) if fresh else {
            'native_step': step, 'current_score': None, 'native_success': None,
            'completed_transition_count': None, 'status': 'unknown_no_same_step_native_gate'}
        count = data['completed_transition_count']
        data.update(uses_privileged_reward_feedback=True, source='native get_reward return + _gated_score_lst copy',
                    source_update_step=self.cache['native_step'] if self.cache else None,
                    remaining_steps=max(0, int(self.env.step_lim) - step),
                    score_scale={'raw_min': 0, 'raw_max': 100, 'display': 'raw points; no normalization',
                                 'tiers': [0, 25, 50, 75, 100]},
                    initial_baseline=copy.deepcopy(self.baseline),
                    stage_conditions={'currently_satisfied': None,
                                      'current_status_reason': 'no read-only per-tool predicate cache audited',
                                      'historically_reached': [count >= n for n in range(1, 5)] if count is not None else None},
                    read_error=self.error)
        return data

    def primitive_completed(self, result, action_id):
        event = {'action_id': action_id, 'command': result.get('command'),
                 'action_ok': result.get('ok'), 'execution_note': result.get('note'),
                 'decision_id': self.action_context.get('turn'),
                 'decision_reason': self.action_context.get('decision_reason'),
                 'steps_cost': result.get('steps_cost'),
                 'zero_step_action': result.get('steps_cost') == 0,
                 'stage_context': self.action_context.get('stage_context'),
                 'trigger_status': 'primitive_complete; phase checkpoint assessed at next observation',
                 'stage_acceptance': 'pending_checkpoint',
                 'early_stop_reason': None,
                 'native_step': int(self.env.take_action_cnt[0])}
        event.update(retention_warning=result.get('retention_warning'),native_terminal=result.get('native_terminal',False),
                     budget_assessment=result.get('budget_assessment'))
        cost=result.get('steps_cost')
        if isinstance(cost,int) and 0<=cost<=event['native_step']:
            self.supervisor.record_cost(result.get('kind') or 'unknown',event['native_step']-cost,event['native_step'])
        if self.cache is not None and self.cache['native_step'] == event['native_step']:
            self.emit(event)
        else:
            self.pending.append(event)  # next existing native gate, before next control action

    def emit(self, event):
        data = self.snapshot()
        if data['native_step'] != event['native_step']:
            data.update(current_score=None, native_success=None, status='unknown_gate_not_observed_at_action_end')
            data['native_step'] = event['native_step']
            data['remaining_steps'] = max(0, int(self.env.step_lim) - event['native_step'])
            data['stage_conditions']['historically_reached'] = None
            data['completed_transition_count'] = None
        score = data['current_score']
        data['delta_score'] = score - self.previous if score is not None and self.previous is not None else None
        if score is not None:
            self.previous = score
        else:
            self.previous = None
        data['delta_basis'] = 'previous completed primitive; first uses initial_baseline'
        data.update(event)
        data['supervisor'] = supervision.primitive_feedback(self.supervisor,data,event)
        self.file.write(json.dumps(data, ensure_ascii=False) + '\n')
        self.file.flush()
        self.events.append(data)

    def observation(self):
        if not self.enabled:
            return None
        data = self.snapshot()
        score = data['current_score']
        data['delta_score'] = score - self.previous_observation if score is not None and self.previous_observation is not None else None
        self.previous_observation = score
        data['delta_basis'] = 'previous agent observation; first uses initial_baseline'
        data['completed_action_feedback'] = copy.deepcopy(self.events)
        self.events.clear()
        return data

    def __exit__(self, *exc):
        if not self.enabled:
            return
        try:
            for event in self.pending:
                self.emit(event)
        finally:
            if self.rm.get_reward is self.wrapper:
                if self.had_override:
                    self.rm.get_reward = self.old_override
                else:
                    del self.rm.get_reward
            del self.env._rsi3_reward_feedback
            self.file.close()


def note_primitive(env, result):
    observer = getattr(env, '_rsi3_reward_feedback', None)
    if observer is not None:
        observer.serial += 1
        observer.primitive_completed(result, 'primitive-%04d' % observer.serial)


def begin_batch(env, envelope):
    observer = getattr(env, '_rsi3_reward_feedback', None)
    if observer is not None:
        observer.action_context = copy.deepcopy(envelope.get('plan') or {})


def budget_primitive(env, planned_steps, *, is_finishing=False, kind=None):
    observer=getattr(env,'_rsi3_reward_feedback',None)
    if observer is None:return {'decision':'allow','reason':'supervisor_disabled'}
    snap=supervision.snapshot(observer.snapshot())
    return observer.supervisor.budget(snap,planned_steps=planned_steps,includes_settle=True,
                                      finish_reserve=100,is_finishing=is_finishing,kind=kind)


def prompt_block(observation):
    feedback = (observation or {}).get('privileged_reward_feedback')
    if not feedback:
        return ''
    return ('\nPRIVILEGED COLLECTION DIAGNOSTICS (sidecar only; not RGB/proprioception SFT observation):\n'
            + json.dumps(feedback, ensure_ascii=False)
            + '\nCumulative tiers are not current tool placement. delta=0 alone is not failure; use RGB, '
              'physical settling needs and remaining native budget. Never wait merely to farm score.\n')
