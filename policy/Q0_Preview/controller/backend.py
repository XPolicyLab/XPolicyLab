"""Remote policy backend: standard observations and static robot geometry only."""
from __future__ import annotations
import copy
import threading
from pathlib import Path
from types import SimpleNamespace
from functools import lru_cache
from .collision import RobotGeometry, pair_clearance
import numpy as np
from scipy.spatial.transform import Rotation
from ..runtime.protocol import ARMS
from .robot_geometry import (grasp_frame, flange_pose, matrix_to_quat_wxyz,
    POSE_FINGER_LINKS, FINGERTIP_REACH_M, GRASP_POINT_OFFSET_M, jaw_gap_mm,
    TABLE_SURFACE_Z, TOOL_TABLE_GUARD_Z)
from .executor import Replan, Ended, StepBudgetExhausted, angle, execution_limits
from .joint_limits import JointLimits


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class Backend:
    def __init__(self, observation, max_control_steps):
        if type(max_control_steps) is not int or max_control_steps <= 0:
            raise ValueError('An explicit positive task control-step limit is required')
        self.raw = observation
        self.observation_index = 0
        self.geometry = RobotGeometry()
        self.control_dt = 1./float(observation['additional_info']['frequency'])
        self.max_control_steps, self.steps = max_control_steps, 0
        self.actions_sent = 0
        self.step_phase = 'initial'
        self.on_step_acknowledged = lambda: None
        self.cmd_g = self.grippers()
        self.held_width_mm = {a: None for a in ARMS}
        self._collision_ranges = None
        self.collision_context = {}
        self.last_collision = None
        self.origin_poses = {a:self.pose(a) for a in ARMS}
        self.origin_ee_positions = {a:array(self.raw['state'][f'{a}_ee_pose'])[:3].copy() for a in ARMS}
        self.joint_limits = {a:JointLimits.from_urdf([f'joint{i+1}' for i in range(len(self.joints()[a]))]) for a in ARMS}
        self.robots = {}
        # Infer base transforms from permitted joint/EEF observations and static FK.
        for a,q in self.joints().items():
            self.joint_limits[a].validate(q)
            local = self.geometry.frames(q)['link6']
            measured = array(self.raw['state'][f'{a}_ee_pose'])
            world_r = Rotation.from_quat(measured[3:], scalar_first=True).as_matrix()
            base_r = world_r @ local[:3,:3].T
            base_p = measured[:3] - base_r @ local[:3,3]
            self.robots[a] = SimpleNamespace(entity_origin_pose=np.r_[base_p, matrix_to_quat_wxyz(base_r)])
        self.planners = {a:LazyPlanner(self,a) for a in ARMS}
        self.path_diagnostics = {}
        self._condition = threading.Condition()
        self._pending = None
        self._delivered = self._acknowledged = self._cancelled = False
        self._error = None

    def joints(self):
        return {a:array(self.raw['state'][f'{a}_arm_joint_state']).astype(float).copy() for a in ARMS}

    def grippers(self):
        """Standard state contains a command echo, NEVER a measured opening."""
        return {a:float(array(self.raw['state'][f'{a}_ee_joint_state']).reshape(-1)[0]) for a in ARMS}

    def gripper_commands(self):
        return dict(self.cmd_g)

    def pose(self, arm):
        return grasp_frame(array(self.raw['state'][f'{arm}_ee_pose']).astype(float))

    def sim_time(self):
        return self.steps*self.control_dt

    def remaining_steps(self):
        return max(0,self.max_control_steps-self.steps)

    def budget_state(self):
        return {'control_steps_used': self.steps,
                'control_steps_sent': self.actions_sent,
                'unacknowledged_actions': self.actions_sent-self.steps,
                'control_step_limit': self.max_control_steps,
                'control_steps_remaining': self.remaining_steps(),
                'control_time_remaining_s': self.remaining_steps()*self.control_dt,
                'control_step_limit_source': 'audited static task definition',
                'sim_time_s': self.sim_time(), 'control_dt_s': self.control_dt}

    def ended(self):
        return self._cancelled

    def step(self, q, g):
        with self._condition:
            if self._cancelled: raise Ended()
            if self.remaining_steps() == 0:
                raise StepBudgetExhausted('task_control_step_limit')
            if any(not np.isfinite(g[a]) or not 0 <= g[a] <= 1 for a in ARMS):
                raise ValueError('Invalid normalized gripper command')
            action = {}
            for a in ARMS:
                self.joint_limits[a].validate(q[a])
                action[f'{a}_arm_joint_state'] = np.asarray(q[a],dtype=np.float32).copy()
                action[f'{a}_ee_joint_state'] = np.array([g[a]],dtype=np.float32)
            self.cmd_g = dict(g)
            self._pending, self._delivered, self._acknowledged = action,False,False
            self._condition.notify_all()
            self._condition.wait_for(lambda:self._acknowledged or self._cancelled)
            if self._cancelled: raise Ended()

    def hold_current(self):
        pass  # No hidden action or physics step.

    def planner_target_pose(self, a, seed, p, r):
        # Anchor both pose planning and IK to the actual joint seed, not the previous
        # requested target (which might have been rejected or only partially reached).
        reference = matrix_to_quat_wxyz(self.fk(a, seed)[1])
        return flange_pose(p, r, reference_quaternion=reference)

    def ik(self, a, seed, p, r):
        return self.ik_candidates(a, seed, p, r)[0]

    def ik_candidates(self, a, seed, p, r):
        result = self.planners[a].solve_ik_to_joint(seed, self.planner_target_pose(a, seed, p, r),
                    copy.deepcopy(self.robots[a].entity_origin_pose))
        if result.get('status') != 'Success':
            raise Replan(f'{a} Cartesian segment is not reachable; add/change explicit keypoints')
        candidates = []
        for raw in result.get('joint_candidates', [result.get('joint_value')]):
            try:
                q = self.nearest_joint_equivalent(a, raw, seed)
            except Replan:
                continue
            if not any(np.max(abs(q-other)) < 1e-3 for other in candidates):
                candidates.append(q)
        if not candidates:
            raise Replan(f'{a} has no IK candidate within configured joint limits')
        return sorted(candidates, key=lambda q:float(np.linalg.norm(q-seed)))

    def nearest_joint_equivalent(self, arm, q, seed):
        return self.joint_limits[arm].nearest(q, seed)

    def plan_free(self, a, start, p, r):
        from .motion import plan_path
        return plan_path(self, a, start, p, r)

    def alternative_paths(self, a, start, p, r, check_deadline):
        from .motion import path_alternatives
        return path_alternatives(self, a, start, p, r, check_deadline)

    def reference_fk(self, a, q):
        planner = self.planners[a]
        kin = planner.motion_planner.compute_kinematics(planner._build_joint_state(q))
        tool = kin.tool_poses.to_dict()[planner.ee_link]
        p = array(tool.position).reshape(-1)[:3] - np.asarray(planner.frame_bias)
        quat = array(tool.quaternion).reshape(-1)[:4]
        local_r = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        origin = array(self.robots[a].entity_origin_pose).reshape(7)
        base_r = Rotation.from_quat(origin[3:], scalar_first=True).as_matrix()
        world_r = base_r @ local_r
        flange_p = origin[:3] + base_r @ p
        return flange_p + GRASP_POINT_OFFSET_M*world_r[:, 0], world_r

    @lru_cache(maxsize=4096)
    def _robot_geometry(self, a, joints, finger):
        origin = array(self.robots[a].entity_origin_pose).reshape(7)
        bias = np.zeros(3)  # bundled X5 configuration has zero frame_bias
        T = self.geometry.frames(joints, finger)['link6']
        base = Rotation.from_quat(origin[3:], scalar_first=True).as_matrix()
        rotation = base @ T[:3,:3]
        position = origin[:3] + base @ (T[:3,3]-bias)
        spheres = self.geometry.world_spheres(joints, origin, finger, bias)
        return (position+GRASP_POINT_OFFSET_M*rotation[:,0], rotation), spheres

    def fk(self, a, q):
        return self._robot_geometry(a, tuple(np.asarray(q,dtype=float)), .044)[0]

    def gripper_estimates(self, group, when, previous=None):
        widths = dict(self.held_width_mm if previous is None else previous)
        for a in ARMS:
            command = group['grippers'][a]
            if command['when'] == when:
                if command['command'] == 'open':
                    widths[a] = None
                elif command.get('held_width_mm') is not None:
                    widths[a] = float(command['held_width_mm'])
        return widths

    @staticmethod
    def collision_gap_range(command, held_width):
        gap = jaw_gap_mm(command)
        if held_width is None and gap < .8:
            # A close command alone cannot establish the width of a held object.
            return (.8, jaw_gap_mm(1.))
        gap = max(.8, gap, held_width or 0.)
        return (gap, gap)

    def set_collision_grippers(self, before, after, held_before=None, held_after=None):
        held_before = self.held_width_mm if held_before is None else held_before
        held_after = held_before if held_after is None else held_after
        self._collision_ranges = {}
        for a in ARMS:
            lo = self.collision_gap_range(before[a], held_before[a])
            hi = self.collision_gap_range(after[a], held_after[a])
            self._collision_ranges[a] = (min(lo[0], hi[0]), max(lo[1], hi[1]))

    def validate_pair(self, q):
        spheres = {}
        for a in ARMS:
            self.joint_limits[a].validate(q[a])
            p,r = self.fk(a,q[a])
            if min(p[2],(p+FINGERTIP_REACH_M*r[:,0])[2]) < TOOL_TABLE_GUARD_Z:
                raise Replan(f'{a} tool would penetrate the table')
            gaps = (self._collision_ranges or {}).get(a,
                self.collision_gap_range(self.cmd_g[a], self.held_width_mm[a]))
            fingers = [max(0., gap/2000.-.0004) for gap in gaps]
            lo = self._robot_geometry(a,tuple(np.asarray(q[a],float)),fingers[0])[1]
            hi = self._robot_geometry(a,tuple(np.asarray(q[a],float)),fingers[1])[1]
            spheres[a] = ((lo[0]+hi[0])/2, np.maximum(lo[1],hi[1])+np.linalg.norm(hi[0]-lo[0],axis=1)/2,lo[2])
        distance,links,points = pair_clearance(spheres['left'],spheres['right'], details=True)
        if distance < .002:
            self.last_collision = {'clearance_mm': distance*1000, 'required_mm': 2.,
                'links': list(links), **points, **self.collision_context,
                'gap_ranges_mm': copy.deepcopy(self._collision_ranges),
                'geometry_source': 'command and held-width estimate; not measured opening'}
            error = Replan(f'inter-arm clearance {distance*1000:.1f} mm at {links}; required 2.0 mm; see collision feedback')
            error.collision = copy.deepcopy(self.last_collision)
            raise error

    def observation(self):
        # The limit comes from the audited static task card, never live evaluator state.
        calibration = {name: {'K':cam['intrinsic_matrix'].tolist(),
            'camera_to_world':cam['extrinsic_matrix'].tolist(),
            'convention':'USD/OpenGL: +X right, +Y up, -Z forward'}
            for name,cam in self.raw['vision'].items()}
        states = {}
        for a in ARMS:
            p, r = self.pose(a)
            home_delta = array(self.raw['state'][f'{a}_ee_pose']).reshape(-1)[:3] - self.origin_ee_positions[a]
            home_angle = float(np.degrees(angle(r, self.origin_poses[a][1])))
            states[a] = {'grasp_point': p.tolist(), 'tool_axis': r[:, 0].tolist(),
                         'jaw_axis': r[:, 1].tolist(),
                         'gripper_pose_points': [(p-.02*r[:,1]).tolist(), (p+.02*r[:,1]).tolist()],
                         'gripper_pose_finger_links': list(POSE_FINGER_LINKS),
                         'commanded_gripper': self.cmd_g[a],
                         'flange_point': (p-GRASP_POINT_OFFSET_M*r[:,0]).tolist(),
                         'fingertip_center': (p+.01257*r[:,0]-.000002*r[:,1]-.00014012*r[:,2]).tolist(),
                         'distal_pad_bbox_center': (p+.00507*r[:,0]-.000002*r[:,1]-.00144044*r[:,2]).tolist(),
                         'commanded_gap_mm': jaw_gap_mm(self.cmd_g[a]),
                         'collision_gap_range_mm': list(self.collision_gap_range(self.cmd_g[a], self.held_width_mm[a])),
                         'held_width_estimate_mm': self.held_width_mm[a],
                         'collision_geometry_source': 'command and held-width estimate; not measured opening',
                          'gripper_state_semantics': 'command echo only; actual opening/contact/possession unavailable',
                         'home_position_error_m': float(np.linalg.norm(home_delta)),
                         'home_orientation_error_deg': home_angle,
                         'near_home': bool(np.max(np.abs(home_delta)) <= .15 and home_angle <= 20),
                         'home_grasp_point': self.origin_poses[a][0].tolist(),
                         'home_tool_axis': self.origin_poses[a][1][:, 0].tolist(),
                         'home_jaw_axis': self.origin_poses[a][1][:, 1].tolist()}
        return {'source_observation_index':self.observation_index, 'instruction': str(self.raw.get('instruction', self.raw.get('instructions', ''))),
                'state': states, 'calibration': calibration,
                'home_check': {'position_per_axis_m': .15, 'orientation_deg': 20,
                               'note': 'Observed initial-pose check using benchmark default tolerances; task-specific completion and official result remain authoritative.'},
                **self.budget_state(), 'execution_limits': execution_limits(),
                'sim_time_source': 'relative time from acknowledged action steps and supplied control frequency'}, self.raw['vision']

    def update(self, observation):
        with self._condition:
            if abs(1./observation['additional_info']['frequency']-self.control_dt)>1e-9:
                raise ValueError('Control frequency changed during episode')
            if self._pending is not None and not self._delivered:
                raise RuntimeError('Observation arrived before pending action delivery')
            self.raw = observation
            self.observation_index += 1
            if self._pending is not None and self._delivered:
                self.steps += 1
                self._pending = None
                self._delivered = False
                self._acknowledged = True
                self.on_step_acknowledged()
            self._condition.notify_all()

    def next_action(self):
        with self._condition:
            self._condition.wait_for(lambda:self._pending is not None or self._cancelled or self._error is not None or self.remaining_steps() == 0)
            if self._error is not None: raise RuntimeError('Policy worker failed') from self._error
            if self._cancelled: raise RuntimeError('Episode ended; reset before inference')
            if self.remaining_steps() == 0:
                raise StepBudgetExhausted('task_control_step_limit')
            if self._delivered: raise RuntimeError('A fresh update_obs is required after each action')
            self._delivered = True
            self.actions_sent += 1
            return {k:v.copy() for k,v in self._pending.items()}

    def cancel(self):
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def close(self):
        self.cancel()
        self._robot_geometry.cache_clear()
        # Break LazyPlanner -> Backend cycles so GPU planners are released on reset.
        for planner in self.planners.values():
            planner.backend = None
            planner.instance = None
        self.planners.clear()

    def fail(self, error):
        with self._condition:
            self._error = error
            self._condition.notify_all()


class LazyPlanner:
    def __init__(self, backend, arm):
        self.backend, self.arm, self.instance = backend,arm,None
        self.frame_bias = np.zeros(3)

    def __getattr__(self, name):
        if self.instance is None:
            from .curobo_planner import CuroboPlanner
            b = self.backend
            self.instance = CuroboPlanner(robot_origin_pose=[0,0,0,1,0,0,0],
                active_joints_name=list(b.joint_limits[self.arm].names),
                all_joints=list(b.joint_limits[self.arm].names), dt=b.control_dt,
                yml_path=str(Path(__file__).with_name('calibration')/'curobo.yml'),
                # CuRobo expects the top surface in the arm's base frame.
                table_height=TABLE_SURFACE_Z-b.robots[self.arm].entity_origin_pose[2])
            p,r = b.fk(self.arm,b.joints()[self.arm])
            reference = b.reference_fk(self.arm,b.joints()[self.arm])
            if np.linalg.norm(p-reference[0])>.0005 or angle(r,reference[1])>.002:
                raise RuntimeError('Bundled URDF/CuRobo FK mismatch')
        return getattr(self.instance,name)
