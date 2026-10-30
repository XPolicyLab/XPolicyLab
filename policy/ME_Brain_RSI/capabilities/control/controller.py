"""GP4: bounded contact probe, RGB test lift, latched closure, one retry.

Uses only RGB, instruction, own arm joints/EE pose and static public geometry.
The gripper observation is command echo, not physical aperture or contact.
No evaluator success signal; the native 200-step budget remains unchanged.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import bounded_pose, quat_matrix
from .contracts import restrict_observation, validate_observation, validate_action, ARMS
from .reachability import select_trajectory, recheck_lift
from .feedback import TargetMotion


class Policy:
    def __init__(self, calibration, perceptor):
        self.calibration, self.perceptor = calibration, perceptor
        self.step = self.age = self.index = self.dwell = 0
        self.home = None
        self.plan = []
        self.grasp = None
        self.reason = 'clear_initial_view'
        self.diagnostics = {}
        self.finished = False
        self.descend_history = []
        self.adapted_contact = None
        self.motion_plan = {}
        self.approach_replanned = False
        self.own_state = None
        self.gripper_target = {s: 1. for s in ARMS}
        self.retry_count = 0
        self.retry_pending = False
        self.retry_reason = None
        self.verdict_streak = {'following': 0, 'stationary': 0}
        self.target_motion = TargetMotion(calibration)
        self.attachment = {'status': 'unknown', 'reason': 'not_tested'}
        self.contact_pose = None
        self.events = []
        self.cautious_unverified = False

    def tcp(self, pose):
        return pose[:3]+quat_matrix(pose[3:]) @ self.calibration['tool_offset_in_ee']

    def finish(self, reason):
        self.finished = True
        self.reason = reason
        self.events.append({'step': self.step, 'event': reason})

    def next_phase(self):
        self.index += 1
        self.age = self.dwell = 0

    def grasp_plan(self, grasp):
        side, poses, self.motion_plan = select_trajectory(
            grasp, self.calibration, self.own_state, self.approach_replanned or self.retry_count > 0)
        approach, contact, warmup, middle, high = poses
        return [(side, approach, 1., 'approach', 2, 42),
                (side, contact, 1., 'descend', 2, 24),
                (side, contact, 0., 'close', 14, 20),
                (side, warmup, 0., 'test_lift', 2, 14),
                (side, warmup, 0., 'verify_grasp', 2, 10),
                (side, middle, 0., 'lift_middle', 2, 20),
                (side, high, 0., 'lift', 3, 24),
                (side, high, 0., 'hold', 15, 18)]

    def detect(self, obs):
        previous = self.grasp
        candidate = self.perceptor.detect(obs['vision']['cam_head']['color'], obs['instruction'])
        if previous is not None:
            # Same grounded instance only. No category/episode-specific routing.
            a, b = np.asarray(previous.bbox), np.asarray(candidate.bbox)
            overlap = np.prod(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])))
            if (candidate.label != previous.label or np.linalg.norm(np.array(candidate.xy)-previous.xy) > .06
                    or overlap/max(1., min(np.prod(a[2:]-a[:2]), np.prod(b[2:]-b[:2]))) < .35
                    or (candidate.xy[0] < 0) != (previous.xy[0] < 0)):
                raise ValueError('Retry target association uncertain')
        self.grasp = candidate
        self.plan = self.grasp_plan(candidate)
        self.target_motion.initialize(obs['vision']['cam_head']['color'],
            getattr(self.perceptor, 'last_debug', {}).get('support'), candidate.surface_z)
        self.descend_history = []
        self.verdict_streak = {'following': 0, 'stationary': 0}
        self.attachment = {'status': 'unknown', 'reason': 'new_grasp'}
        self.cautious_unverified = False
        self.index = self.age = self.dwell = 0
        self.retry_pending = False

    def near_surface_stall(self, current, target, angle):
        if self.grasp is None or self.age < 8 or len(self.descend_history) < 6:
            return False
        actual, desired = self.tcp(current), self.tcp(target)
        recent = np.asarray(self.descend_history[-6:])
        tip_extra = np.linalg.norm(np.array(self.calibration['contact_tip_offset_in_ee'])-
                                   self.calibration['tool_offset_in_ee'])
        # A finite closing probe at measured height; NOT a force/contact claim.
        # Never push down to compensate for a stalled Cartesian target.
        return bool(np.linalg.norm(actual[:2]-desired[:2]) < .015 and angle < .13
                    and .003 < actual[2]-desired[2] < .05
                    and self.calibration['table_z']+tip_extra+.005 <= actual[2]
                    and actual[2] <= self.grasp.surface_z+.025
                    and np.max(np.linalg.norm(recent-recent[0], axis=1)) < .002)

    def enter_close(self, side, current):
        lift, checked = recheck_lift(current, side, self.own_state, self.calibration)
        self.contact_pose = current.copy()
        self.target_motion.anchor(self.tcp(current))
        self.adapted_contact = {'tcp': self.tcp(current).tolist(), 'step': self.step,
                                'reason': 'measured_pose_probe_not_contact_sensor'}
        self.motion_plan['contact_recheck'] = checked
        for j in range(self.index+1, len(self.plan)):
            s, old, grip, name, dwell, limit = self.plan[j]
            pose = {'close': current, 'test_lift': lift[0], 'verify_grasp': lift[0],
                    'lift_middle': lift[1], 'lift': lift[2], 'hold': lift[2]}[name]
            self.plan[j] = (s, pose.copy(), grip, name, dwell, limit)
        self.next_phase()

    def retry(self, side, current, reason):
        # Reserve 110 actions for clearance, perception, a new grasp and lift.
        # Retry is within this episode, not another simulation or seed.
        if self.retry_count >= 1 or self.step > 85:
            self.finish(reason+'_no_retry_budget')
            return
        self.retry_count += 1
        self.retry_pending = True
        self.retry_reason = reason
        target = current.copy()
        target[2] += .055
        # Retain closure while clearing. Open only after clearance is reached.
        self.plan = [(side, target, self.gripper_target[side], 'retry_clear', 2, 12),
                     (side, target, 1., 'retry_open', 5, 7)]
        self.index = self.age = self.dwell = 0
        self.events.append({'step': self.step, 'event': 'bounded_retry', 'reason': reason})
        self.reason = 'retry_clear'

    def assess(self, side, current):
        self.attachment = self.target_motion.assess(self.tcp(current))
        for key in self.verdict_streak:
            self.verdict_streak[key] = (self.verdict_streak[key]+1
                if self.attachment['status'] == key else 0)

    def act(self, raw):
        obs = restrict_observation(raw)
        validate_observation(obs)
        self.own_state = obs['state']
        current = {s: np.asarray(obs['state'][s+'_ee_pose'], float) for s in ARMS}
        if self.home is None:
            self.home = {s: current[s].copy() for s in ARMS}
            for s in ARMS:
                target = self.home[s].copy()
                target[0] += -.10 if s == 'left' else .10
                target[2] += .04
                self.plan.append((s, target, 1., 'clear_view', 1, 12))
        self.step += 1
        if self.grasp is not None and not self.finished and not self.retry_pending:
            self.target_motion.update(obs['vision']['cam_head']['color'])
        # Never feed the returned, rate-limited previous gripper command back.
        action = {s+'_'+k: (current[s].copy() if k == 'ee_pose' else np.array([self.gripper_target[s]]))
                  for s in ARMS for k in ('ee_pose', 'ee_joint_state')}
        if not self.finished and self.index >= len(self.plan):
            if self.grasp is None or self.retry_pending:
                try:
                    self.detect(obs)
                    self.reason = 'rgb_grasp'
                except ValueError as error:
                    self.finish(str(error))
            else:
                self.finish('lift_command_complete_not_a_success_claim')
        if not self.finished and self.step < 195:
            side, target, grip, name, dwell, limit = self.plan[self.index]
            self.age += 1
            self.gripper_target[side] = grip
            speed = (.006 if name in ('descend', 'test_lift') or
                     (name.startswith('lift') and self.cautious_unverified)
                     else .010 if name.startswith('lift') else .018)
            action[side+'_ee_pose'] = bounded_pose(current[side], target, speed, .18)
            action[side+'_ee_joint_state'] = np.array([grip])
            distance = float(np.linalg.norm(current[side][:3]-target[:3]))
            angle = float(Rotation.from_matrix(quat_matrix(current[side][3:]).T @ quat_matrix(target[3:])).magnitude())
            reached = distance < .009 and angle < .13
            self.dwell = self.dwell+1 if reached else 0
            self.reason = name
            if name == 'descend':
                self.descend_history.append(self.tcp(current[side]))
                if self.dwell >= dwell or self.near_surface_stall(current[side], target, angle):
                    try:
                        self.enter_close(side, current[side])
                        action[side+'_ee_pose'] = current[side].copy()
                        self.reason = 'contact_pose_hold'
                    except ValueError:
                        self.retry(side, current[side], 'contact_lift_unreachable')
                elif self.age >= limit:
                    self.retry(side, current[side], 'descend_motion_timeout')
            elif name == 'verify_grasp':
                self.assess(side, current[side])
                if self.verdict_streak['following'] >= 2:
                    self.next_phase()
                elif self.verdict_streak['stationary'] >= 2:
                    self.retry(side, current[side], 'visible_empty_grasp')
                elif self.age >= limit:
                    # Occlusion is NOT a failed grasp. Preserve prior feasible
                    # behavior at reduced speed and keep checking visible RGB;
                    # never label this branch confirmed attachment or success.
                    self.cautious_unverified = True
                    self.events.append({'step': self.step, 'event': 'unverified_cautious_lift'})
                    self.next_phase()
            elif name in ('lift_middle', 'lift', 'hold'):
                self.assess(side, current[side])
                if self.verdict_streak['stationary'] >= 2:
                    action[side+'_ee_pose'] = current[side].copy()
                    self.finish('target_not_following_hold_closed')
                elif self.dwell >= dwell:
                    self.next_phase()
                elif self.age >= limit:
                    self.finish(name+'_motion_timeout_hold_closed')
            elif self.dwell >= dwell:
                self.next_phase()
            elif self.age >= limit:
                if name == 'clear_view':
                    self.next_phase()
                elif name == 'approach' and not self.approach_replanned and self.step < 70:
                    self.approach_replanned = True
                    try:
                        self.plan = self.grasp_plan(self.grasp)
                        self.index = self.age = self.dwell = 0
                        self.reason = 'compact_approach_replan'
                    except ValueError as error:
                        self.finish(str(error))
                else:
                    self.finish(name+'_motion_timeout_hold_gripper_target')
        elif self.step >= 195 and not self.finished:
            self.finish('local_budget_guard_hold_gripper_target')
        # Closed target remains latched on every terminal/default action.
        self.diagnostics = {'step': self.step, 'phase': self.reason, 'waypoint': self.index,
            'age': self.age, 'finished': self.finished,
            'rgb_grasp': None if self.grasp is None else self.grasp.json(),
            'measured_tcp': {s: self.tcp(current[s]).tolist() for s in ARMS},
            'commanded_ee': {s: action[s+'_ee_pose'].tolist() for s in ARMS},
            'commanded_gripper': {s: float(action[s+'_ee_joint_state'][0]) for s in ARMS},
            'gripper_command_echo': {s: np.asarray(obs['state'].get(s+'_ee_joint_state', [])).tolist() for s in ARMS},
            'gripper_feedback_is_sensor': False, 'adapted_contact': self.adapted_contact,
            'motion_plan': self.motion_plan, 'approach_replanned': self.approach_replanned,
            'retry_count': self.retry_count, 'retry_reason': self.retry_reason,
            'cautious_unverified': self.cautious_unverified,
            'attachment_evidence': self.attachment, 'events': self.events[-8:]}
        validate_action(action)
        return action
