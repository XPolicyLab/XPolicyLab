"""Synchronous dual-arm executor, independent of API and simulator implementation."""
from __future__ import annotations
import math
from copy import deepcopy
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ..runtime.protocol import ARMS, MAX_GROUP_SECONDS, contact_error, validate_plan, validate_decision
from .robot_geometry import gripper_for_gap, jaw_gap_mm

GRIPPER_COMMAND_STEP = .15
GRIPPER_SETTLE_STEPS = 2
GRIPPER_CLOSE_SETTLE_STEPS = 6
GRIPPER_EPS = 1e-5


def execution_limits():
    from .motion import JOINT_SPEED_RAD_S, START_STOP_RAMP_S
    return {'joint_speed_rad_s': JOINT_SPEED_RAD_S,
            'controller': 'ramped', 'endpoint_ramp_s': START_STOP_RAMP_S,
            'max_group_sim_seconds': MAX_GROUP_SECONDS,
            'gripper_command_delta_per_step': GRIPPER_COMMAND_STEP, 'gripper_settle_steps': GRIPPER_SETTLE_STEPS,
            'gripper_close_settle_steps': GRIPPER_CLOSE_SETTLE_STEPS,
            'timing_note': 'One joint-speed cap: 2 rad/s, with 0.12 s startup and stopping ramps. Ramp time is included in the control-step budget. Positive duration_s may lengthen playback. No-op motion consumes zero steps.'}


class Replan(RuntimeError):
    """Stop the current action; obtain fresh observations before planning again."""


class StopExecution(RuntimeError):
    """A policy lifecycle/budget stop, including during a visual review."""


class StepBudgetExhausted(StopExecution):
    """All configured task action steps have been acknowledged."""


class Ended(RuntimeError):
    pass




def angle(a, b):
    return float(Rotation.from_matrix(a.T @ b).magnitude())


class Executor:
    def __init__(self, backend, trace=lambda event: None, deadline_check=lambda: None):
        self.b = backend
        self.trace = trace
        self.deadline_check = deadline_check
        self.group_count = 0
        self.last_replan = None
        self.last_feedback = None
        self._active_goals = {}

    def _step(self, q, g, group=None):
        self.deadline_check()
        if self.b.ended():
            raise Ended()
        self.b.step_phase = self.last_feedback['phase'] if self.last_feedback else 'execution'
        if hasattr(self.b, 'set_collision_grippers'):
            self.b.set_collision_grippers(self.b.gripper_commands(), g)
            self.b.collision_context = {'phase': self.b.step_phase}
            self.b.validate_pair(q)
        self.b.step(q, g)
        if self.last_feedback is not None:
            self.last_feedback['executed_steps'] += 1
        if self.b.ended():
            raise Ended()
        if hasattr(self.b, 'set_collision_grippers'):
            self.b.set_collision_grippers(g, g)
        self.b.validate_pair(self.b.joints())
        if group is not None:
            for arm in ARMS:
                if group.get(arm) is not None:
                    self._contact(group[arm], self.b.pose(arm), 'execution')

    @staticmethod
    def _gripper_targets(group, when, starts):
        goals = starts.copy()
        for a in ARMS:
            c = group['grippers'][a]
            if c['when'] != when or c['command'] == 'keep':
                continue
            goals[a] = {'open': 1., 'close': 0.}.get(c['command'],
                gripper_for_gap(c['width_mm']) if c['command'] == 'width' else starts[a])
        return {a: starts[a] if abs(goals[a]-starts[a]) <= GRIPPER_EPS else goals[a] for a in ARMS}

    @staticmethod
    def _settle_steps(starts, goals):
        # A new narrowing command gets a short stationary hold before any next move.
        # This is time to settle, not evidence that an object was grasped.
        return (GRIPPER_CLOSE_SETTLE_STEPS if any(goals[a] < starts[a]-GRIPPER_EPS for a in ARMS)
                else GRIPPER_SETTLE_STEPS)

    @classmethod
    def _gripper_event_steps(cls, starts, goals):
        if goals == starts:
            return 0
        return (max(1, math.ceil(max(abs(goals[a]-starts[a]) for a in ARMS)/GRIPPER_COMMAND_STEP))
                + cls._settle_steps(starts, goals))

    def _gripper_step_budget(self, group):
        starts, steps = self.b.gripper_commands().copy(), 0
        for when in ('before', 'after'):
            goals = self._gripper_targets(group, when, starts)
            steps += self._gripper_event_steps(starts, goals)
            starts = goals
        return steps

    def _grippers(self, group, when):
        starts = self.b.gripper_commands().copy()
        goals = self._gripper_targets(group, when, starts)
        if hasattr(self.b, 'gripper_estimates'):
            old_widths = dict(self.b.held_width_mm)
            new_widths = self.b.gripper_estimates(group, when)
            if goals != starts:
                self.b.set_collision_grippers(starts, goals, old_widths, new_widths)
                self.b.collision_context = {'phase': when+'_gripper'}
                self.b.validate_pair(self.b.joints())
            self.b.held_width_mm = {a: old_widths[a] if goals[a] > starts[a] else new_widths[a] for a in ARMS}
        if goals == starts:
            if hasattr(self.b, 'gripper_estimates'):
                self.b.held_width_mm = new_widths
            return
        q = self.b.joints()
        n = max(1, math.ceil(max(abs(goals[a]-starts[a]) for a in ARMS) / GRIPPER_COMMAND_STEP))
        for t in np.linspace(0, 1, n + 1)[1:]:
            self._step(q, {a: starts[a] + t*(goals[a]-starts[a]) for a in ARMS}, group)
        # Explicit gripper event settling; no implicit open/close skills.
        settle_steps = self._settle_steps(starts, goals)
        changed=[a for a in ARMS if goals[a]!=starts[a]]
        for _ in range(settle_steps):
            self._step(q, goals, group)
        if hasattr(self.b, 'gripper_estimates'):
            self.b.held_width_mm = new_widths
        if self.last_feedback is not None:
            for a in changed:
                self.last_feedback['gripper_events'].append({
                    'arm':a, 'when':when, 'settle_steps':settle_steps,
                    'commanded_gap_mm':jaw_gap_mm(goals[a]),
                    'feedback_semantics':'command only; actual gap/contact/possession unavailable'})
        self.trace({'event':'gripper','when':when,'command':goals,'settle_steps':settle_steps})

    def _prepare(self, group):
        from .motion import compile_motion
        return compile_motion(self, group)

    @staticmethod
    def _contact(target, pose, phase):
        if target['contact_constraint'] is not None:
            limit = target['contact_constraint']['tolerance_mm']/1000
            error = contact_error(target, pose)
            if error > limit + 1e-8:
                raise Replan(f'contact plane constraint violated at {phase}: error {error*1000:.3f} mm, '
                             f'limit {limit*1000:.3f} mm; use a shorter target or change the explicit pose')

    def execute_group(self, group):
        self._active_goals={}
        self.last_timing={}
        self.b.path_diagnostics={}
        if hasattr(self.b, 'last_collision'):
            self.b.last_collision = None
        self.last_feedback={'group_index':self.group_count,'status':'planning',
                            'phase':'planning','executed_steps':0,
                            'arrival':{},'gripper_events':[], 'path_diagnostics':{}}
        try:
            self._execute_group(group)
            self.last_feedback['status']='complete'
        except Replan as exc:
            self.last_feedback.update(status='replan',reason=str(exc),
                                      failure_phase=self.last_feedback['phase'])
            if getattr(exc, 'collision', None) is not None:
                self.last_feedback['collision'] = deepcopy(exc.collision)
            raise
        finally:
            self._update_feedback()

    def _update_feedback(self):
        if self.last_feedback is None or self.b.ended():
            return
        self.last_feedback['sim_time_s']=self.b.sim_time()
        for a,(goal_p,goal_r) in self._active_goals.items():
            p,r=self.b.pose(a)
            delta=np.asarray(goal_p)-p
            self.last_feedback['arrival'][a]={
                'remaining_translation_world_m':delta.tolist(),
                'position_error_mm':float(np.linalg.norm(delta))*1000,
                'orientation_error_deg':math.degrees(angle(r,goal_r))}

    def _execute_group(self, group):
        active, goals, qs = self._prepare(group)
        self._active_goals=goals
        self.last_feedback['status']='executing'
        if 'sampled_contact_point_speed_m_s' in self.last_timing:
            self.last_feedback['sampled_contact_point_speed_m_s']=self.last_timing['sampled_contact_point_speed_m_s']
        self.last_feedback['path_diagnostics'] = self.last_timing.get('path_diagnostics', {})
        self.trace({'event': 'group_started', 'group': self.group_count,
                    'requested': group, 'planned_motion_steps': len(qs), 'timing': self.last_timing})
        self.last_feedback['phase']='before_gripper'
        self._grippers(group, 'before')
        commands = self.b.gripper_commands()
        lag_count = 0
        self.last_feedback['phase']='motion'
        for q in qs:
            self._step(q, commands, group)
            actual = self.b.joints()
            lag = max(np.max(abs(actual[a]-q[a])) for a in ARMS)
            lag_count = lag_count + 1 if lag > .2 else 0
            self.trace({'event': 'control_step', 'sim_time_s': self.b.sim_time(),
                        'target_joints': q, 'actual_joints': actual,
                        'commanded_grippers': self.b.gripper_commands(), 'max_joint_error_rad': lag})
            if lag > .5 or lag_count >= 3:
                raise Replan('joint tracking deviation; fresh observation required')
        # A short measured convergence phase,  do not release while still tracking.
        self.last_feedback['phase']='convergence'
        for _ in range(min(12, max(0,self.b.remaining_steps()-self._gripper_step_budget(group)))):
            if all(np.linalg.norm(self.b.pose(a)[0]-goals[a][0]) <= .003 and
                   angle(self.b.pose(a)[1],goals[a][1]) <= math.radians(3) for a in active):
                break
            if not qs: break
            self._step(qs[-1], commands, group)
        for a in active:
            p, r = self.b.pose(a)
            if np.linalg.norm(p-goals[a][0]) > .005 or angle(r, goals[a][1]) > math.radians(5):
                raise Replan('arrival mismatch; after-arrival gripper commands cancelled')
        self.last_feedback['phase']='after_gripper'
        self._grippers(group, 'after')
        self.last_feedback['phase']='finished'
        self.group_count += 1
        self.trace({'event': 'group_completed', 'group': self.group_count-1})

    def execute(self, plan, checkpoint=None):
        """Execute one group, or two with an obligatory fail-closed visual gate."""
        plan = deepcopy(plan)
        validate_plan(plan)  # Validate the entire queue before any planning or motion.
        self.last_replan = None
        self.last_feedback = None
        self._active_goals = {}
        try:
            if len(plan['groups']) == 2 and not callable(checkpoint):
                raise Replan('two groups require a checkpoint reviewer; no motion executed')
            minimum = self.minimum_queue_steps(plan['groups'])
            remaining = self.b.remaining_steps()
            self.trace({'event': 'group_budget_check', 'minimum_steps': minimum, 'remaining_steps': remaining, 'groups': len(plan['groups'])})
            if minimum > remaining:
                raise Replan(f'action plan needs at least {minimum} environment steps; only {remaining} remain. '
                             'No motion executed. Choose a smaller next action within the remaining budget. '
                             'Keep normal motion timing and necessary settling.')
            for i, group in enumerate(plan['groups']):
                self.deadline_check()
                if self.b.ended():
                    raise Ended()
                self.execute_group(group)
                if i + 1 == len(plan['groups']):
                    continue
                self.deadline_check()
                if self.b.ended():
                    raise Ended()
                self.last_feedback['phase'] = 'checkpoint'
                self.trace({'event': 'checkpoint_started', 'plan_group_index': i})
                try:
                    # Reviewers may only approve/reject; they cannot rewrite pending targets.
                    verdict = validate_decision(checkpoint(i, deepcopy(group), deepcopy(plan['groups'][i+1:])))
                except (Ended, StopExecution):
                    raise
                except Exception as exc:
                    # Recheck lifecycle limits before treating reviewer failure as a replan.
                    self.deadline_check()
                    raise Replan('checkpoint unavailable or invalid: '+str(exc)) from exc
                self.deadline_check()
                if self.b.ended():
                    raise Ended()
                self.trace({'event': 'checkpoint_decision', 'plan_group_index': i, **verdict})
                self.last_feedback['checkpoint'] = deepcopy(verdict)
                if verdict['decision'] != 'CONTINUE':
                    raise Replan('checkpoint: '+verdict['reason'])
                # Group two is compiled only now, from the acknowledged post-group-one state.
        except Replan as e:
            self.last_replan = str(e)
            if self.last_feedback is None:
                self.last_feedback={'group_index':self.group_count,'arrival':{},'gripper_events':[],
                                    'path_diagnostics':{},'executed_steps':0,'phase':'group_precheck'}
            self.last_feedback.update(status='replan',reason=str(e))
            self.last_feedback.setdefault('failure_phase', self.last_feedback['phase'])
            self.b.hold_current()
            self.trace({'event': 'plan_cancelled', 'reason': str(e)})
            # Feedback describes measured execution only; the interrupted target is not resumed.
            return 'REPLAN'
        return 'COMPLETE'

    def minimum_group_steps(self, group):
        """Pre-IK lower bound from explicit duration and gripper events.

        Joint travel is unknown until planning. Trajectory budgeting checks the
        resulting path before execution.
        """
        return max(0, math.ceil(group['duration_s']/self.b.control_dt)) + self._gripper_step_budget(group)

    def minimum_queue_steps(self, groups):
        """Lower bound for the whole plan, carrying commanded jaws between groups."""
        starts, steps = self.b.gripper_commands().copy(), 0
        for group in groups:
            steps += max(0, math.ceil(group['duration_s']/self.b.control_dt))
            for when in ('before', 'after'):
                goals = self._gripper_targets(group, when, starts)
                steps += self._gripper_event_steps(starts, goals)
                starts = goals
        return steps
