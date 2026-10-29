"""RoboDojo backend. This module runs only in the simulator process."""
import base64
import copy
import gzip
import json
import math
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
from geometry import tcp_from_link, link_from_tcp, make_target, interpolate, errors, vector, return_steps_estimate
from episode_budget import configure_budget, budget_observation, read_budget
from evidence import Evidence
from motion_evidence import HeldPoseMonitor
from observation_capture import capture_observation
from motion_timing import (speed_limits, within, move_timing, waypoint_timing, mark_timing, gripper_timing,
                           gripper_width_floor, motion_profile, ramp_time, progress)


def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [plain(v) for v in x]
    if hasattr(x, 'detach'): return x.detach().cpu().numpy().tolist()
    if hasattr(x, 'tolist'): return x.tolist()
    return x


def largest_joint_lag(names, commanded, measured, minimum_rad=.01):
    """The joint furthest from its commanded position after a control step (named in a tracking_error report),
    or None when every joint is within minimum_rad of its command (the path error then lies elsewhere)."""
    commanded, measured = np.asarray(commanded, dtype=float), np.asarray(measured, dtype=float)
    i = int(np.argmax(np.abs(commanded - measured)))
    if abs(commanded[i] - measured[i]) < minimum_rad:
        return None
    return {'joint': str(names[i]), 'commanded_rad': round(float(commanded[i]), 4),
            'measured_rad': round(float(measured[i]), 4)}


PLAN_CHECKS = ('Dry run: the inverse kinematics and per-step joint-change checks this motion would make before each '
               'step, each step starting from the joints planned for the one before (an executed motion starts each '
               'step from the joints it measures); nothing moved and no step was used. It cannot predict contact, '
               'tracking_error, slipping, gripper contact or final accuracy, so feasible is not a guarantee.')


# Settings of development-only features that the submission adapter refuses (v2/submission.py).
SUBMISSION_REFUSED = ('ROBOT_EXPLORATION_STEP_LIMIT', 'ROBOT_EXTEND_MULTIPLIER', 'ROBOT_TEACHER_ACCESS',
                      'ROBOT_OUTER_STAGE_CONFIG', 'ROBOT_CURRICULUM_MANIFEST')


class RoboDojoAdapter:
    def __init__(self, env, output, environ=None, submission=False):
        """`environ` holds the ROBOT_* settings (default: this process's environment). With submission=True
        (the official evaluator's process, v2/submission.py) the step budget is read, never written, and the
        development-only settings are refused."""
        self.env, self.output = env, Path(output)
        self.environ = os.environ if environ is None else environ
        if submission:
            refused = [key for key in SUBMISSION_REFUSED if self.environ.get(key)]
            if self.environ.get('ROBOT_BUDGET_MODE', 'benchmark') != 'benchmark':
                refused.append('ROBOT_BUDGET_MODE=' + str(self.environ.get('ROBOT_BUDGET_MODE')))
            if refused:
                raise ValueError('the submission adapter runs the official budget only; refused: ' + ', '.join(refused))
            self.budget = read_budget(env)
        else:
            exploration_limit = self.environ.get('ROBOT_EXPLORATION_STEP_LIMIT')
            self.budget = configure_budget(env, self.environ.get('ROBOT_BUDGET_MODE', 'exploration'),
                                           int(exploration_limit) if exploration_limit else None)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rm = env.robot_manager
        self.arms = {r.arm_name.split('_')[0]: r for r in self.rm.robot_list if r.type == 'target'}
        if any(r.robot_name != 'x5' for r in self.arms.values()):
            raise ValueError('This adapter calibration supports X5 only; supply another robot calibration explicitly')
        self.hz = float(env.obs_manager.collect_freq)
        self.substeps_per_control = int(env.obs_manager.collect_interval)
        self.limits = speed_limits(self.environ)  # ROBOT_SPEED_LIMITS; the original values by default
        # ROBOT_MOTION_PROFILE: None (constant speed, the original) unless a smoothed profile is configured.
        self.profile = motion_profile(self.environ.get('ROBOT_MOTION_PROFILE'))
        self.squeeze = self.environ.get('ROBOT_GRIPPER_SQUEEZE') == '1'  # widths below 0 (off by default)
        # X5 robot_config.yml nominal grasp offset; finger pad center from its URDF/STL.
        self.offset = [float(next(iter(self.arms.values())).gripper_bias), -0.000002, -0.00144]
        self.width_zero = .000807
        self.grips = {a: 1. for a in self.arms}
        self.marks = {'initial': self.state()}
        self.current_id, self.low_step, self.last_obs = None, 0, None
        self.finish_requested = False
        if self.environ.get('ROBOT_TEACHER_ACCESS'):
            # No privileged input: v2 never sets this (v2/episode.py removes it), and an evaluation refuses it.
            raise ValueError('ROBOT_TEACHER_ACCESS: this adapter has no privileged teacher input')
        self.outer_collector = None
        config = self.environ.get('ROBOT_OUTER_STAGE_CONFIG')
        if config and self.environ.get('ROBOT_CONTROLLER_OWNER') == 'candidate_program':
            from outer_stage_state import Collector
            self.outer_collector = Collector(env, self.output.parent/'outer_assessment', json.loads(config), self.hz)
        self.trace = (self.output / 'transitions.jsonl').open('a', buffering=1)
        self.low_trace = gzip.open(self.output / 'actuator_commands.jsonl.gz', 'at')
        self.commands = (self.output / 'commands.jsonl').open('a', buffering=1)
        self.evidence = Evidence(self.output)
        original_control = self.rm.control_robot
        # close() puts the original back: RoboDojo's evaluator runs many episodes on one robot manager, and a
        # wrapper left in place would wrap the next episode's again and write to this episode's closed log.
        self._original_control = original_control

        def recorded_control(*args, **kwargs):
            before = self.state()
            original_control(*args, **kwargs)
            targets = {}
            for a, robot in self.arms.items():
                entity = self.rm.robot_key[self.rm.robot_list.index(robot)]
                targets[a] = {
                    'arm_position_target_rad': plain(entity.data.joint_pos_target[0, robot.arm_joint_indices]),
                    'finger_position_target_m': plain(entity.data.joint_pos_target[0, robot.gripper_joint_indices]),
                }
            self.low_trace.write(json.dumps({'substep': self.low_step, 'time_s': self.low_step / (self.hz * self.substeps_per_control),
                'command_id': self.current_id, 'measured_before_physics': before, 'applied_targets': targets}) + '\n')
            self.low_step += 1
        self.rm.control_robot = recorded_control

    def state(self):
        result = {}
        for a, robot in self.arms.items():
            link = self.rm.get_real_endpose(robot)[0]
            fingers = self.rm.get_end_effector_real_val(robot)[0]
            # X5 commands joint7 as the master gripper joint and joint8 as its
            # symmetric mimic. Expose only the master measurement so an actor
            # cannot use simulator-only differential finger motion as contact
            # evidence. The width is an estimate from the nominal coupling and
            # pad geometry, matching what a single-encoder hardware interface
            # can provide.
            gripper_position = float(np.asarray(fingers, dtype=float).reshape(-1)[0])
            result[a] = {'tcp': tcp_from_link(link, self.offset),
                'link_pose_wxyz': plain(link), 'joint_positions_rad': plain(self.rm.get_joint(robot)[0]),
                'gripper_position_measured_m': gripper_position,
                'gripper_width_m': float(max(0., 2. * gripper_position + self.width_zero)),
                'gripper_width_source': 'single measured master joint + nominal symmetric coupling and pad geometry',
                'gripper_command_normalized': self.grips.get(a, 1.) if hasattr(self, 'grips') else None}
        return result

    def describe(self):
        return {'api_version': '0.5', 'backend': 'robodojo_x5', 'clock_mode': 'paused_between_commands',
            'task': self.capture()['task'], 'control_hz': self.hz,
            'completion_requirements': ['Complete the physical task', 'Return both arms to their initial poses before finishing'],
            'terminal_semantics': 'The evaluator may terminate before arm return; no further motion is then possible. '
                                  'Report arm-return state separately; do not infer it from terminal status.',
            'initial_robot_state': self.marks['initial'],
            'physics_hz': self.hz * self.substeps_per_control,
            **budget_observation(self.budget, self.env.take_action_cnt[0]),
            'frames': {'world': '+x left base toward right base, +y forward across table, +z up',
                       'tcp': '+x approaches along fingers, +y across jaws, +z transverse; frozen at command start',
                       'base': 'selected robot base axes; frozen at command start'},
            'tcp': {'name': 'grasp_center', 'offset_in_link6_m': self.offset, 'orientation': 'same as link6',
                    'source': 'X5 nominal gripper_bias and URDF/STL finger pad geometry; not physical hand-eye calibration'},
            'gripper_width_range_m': [0, .088807],
            **({'gripper_squeeze_min_m': round(gripper_width_floor(next(iter(self.arms.values())).gripper_scale[0],
                                                                   self.width_zero, True), 4)} if self.squeeze else {}),
            'arms': {a: {'joint_names': list(r.arm_joints_name), 'base_pose_wxyz': list(r.entity_origin_pose)} for a, r in self.arms.items()},
            'capabilities': {'rgb': ['cam_head', 'cam_left_wrist', 'cam_right_wrist'],
                'measured_gripper_master_joint': True, 'independent_finger_positions': False,
                'tcp_cartesian_linear': True, 'tcp_rotation': True, 'joint_bookmarks': True,
                'dual_arm_waypoints': True, 'hold': True, 'object_poses': False, 'force_sensor': False,
                'global_collision_planning': False, 'camera_metric_projection': False,
                'camera_calibration': False, 'real_robot_adapter': False},
            'limits': {'speed_m_s': self.limits['speed_m_s'], 'speed_rad_s': self.limits['speed_rad_s'],
                'mark_joint_speed_rad_s': self.limits['mark_joint_speed_rad_s'], 'joint_speed_rad_s': 5.,
                'tracking_abort_m': .02, 'tracking_abort_rad': .25, 'position_tolerance_m': .003, 'orientation_tolerance_rad': .06},
            # Only a configured (non-default) profile is described, so the default description is unchanged.
            **({'motion_profile': dict(self.profile)} if getattr(self, 'profile', None) else {}),
            'usage': 'robot.arm("left").move_by(dx_m=.01, frame="world"); rotate_by(axis="z",angle_deg=10,frame="tcp"); '
                'move_to(position_m=[x,y,z],quaternion_xyzw=[x,y,z,w]); gripper.set_width(width_m=.04); '
                'move_to_mark("initial"); robot.mark_pose("name"); robot.execute_trajectory([{"left":{"position_m":[x,y,z]},"duration_s":1.0}]); '
                'robot.hold(duration_s=1.0); '
                'display(report) shows its post-action images; display(robot.observe()) shows current images. '
                'reached refers to robot motion, never object/task success. robot.finish() ends this physical attempt.'}

    def capture(self):
        step = int(self.env.take_action_cnt[0])
        oid = f'frame_{step:06d}'
        folder = self.output / 'observations' / oid
        if not folder.exists():
            started = time.time()
            obs, capture_details = capture_observation(self.env)
            meta = {'observation_id': oid, 'control_step': step, 'control_time_s': step / self.hz,
                'wall_time_s': time.time(), 'capture_wall_duration_s': time.time() - started,
                'synchronization': 'physics held; six renders, one final observation read; hardware timestamps unavailable',
                'capture_details': capture_details,
                'task': obs['instruction'], 'robot_state': self.state(),
                **budget_observation(self.budget, step),
                'episode_ended': bool(self.env.end_flag[0]), 'cameras': []}
            meta['return_to_initial_steps_estimate'] = return_steps_estimate(
                meta['robot_state'], self.marks['initial'], self.hz, self.limits['mark_joint_speed_rad_s'])
            folder.mkdir(parents=True)
            png_started = time.perf_counter()
            for name, camera in obs['vision'].items():
                arr = np.asarray(camera['color'], dtype=np.uint8)
                # PNG level changes encoding effort/size, not decoded pixels.
                # The default level 6 dominates frame recording time.
                Image.fromarray(arr).save(folder / f'{name}.png', compress_level=1)
                meta['cameras'].append({'name': name, 'shape': list(arr.shape)})
            meta['recording_details'] = {'image_format': 'PNG', 'compress_level': 1,
                'lossless': True, 'png_save_wall_s': time.perf_counter() - png_started}
            (folder / 'observation.json').write_text(json.dumps(meta, indent=2))
        self.last_obs = oid
        value = json.loads((folder / 'observation.json').read_text())
        if self.outer_collector is not None:
            self.outer_collector.capture(value)
        return value

    def observation(self, oid=None):
        meta = self.capture() if oid is None else self._read_observation(oid)
        folder = self.output / 'observations' / meta['observation_id']
        return {**meta, 'images': [{'camera': c['name'], 'png_base64': base64.b64encode((folder / (c['name'] + '.png')).read_bytes()).decode()}
                                for c in meta['cameras']]}

    def _read_observation(self, oid):
        if not isinstance(oid, str) or not oid.startswith('frame_') or not oid[6:].isdigit():
            raise ValueError('Invalid observation ID')
        return json.loads((self.output / 'observations' / oid / 'observation.json').read_text())

    def _step(self, joints, command_id):
        before = self.capture()
        action = {}
        for a, robot in self.arms.items():
            action[f'{a}_arm_joint_state'] = plain(joints[a])
            action[f'{a}_ee_joint_state'] = [self.grips[a]]
        low_start = self.low_step
        self.env.take_action(action)
        after = self.capture()
        monitor = getattr(self, '_pose_monitor', None)
        if monitor is not None: monitor.update(after)
        self.trace.write(json.dumps({'command_id': command_id, 'controller_owner': getattr(self, 'environ', os.environ).get('ROBOT_CONTROLLER_OWNER', 'external_client'),
            'observation_id': before['observation_id'], 'next_observation_id': after['observation_id'],
            'control_time_s': before['control_time_s'], 'requested_joint_action': action,
            'applied_actuator_substeps': [low_start, self.low_step], 'measured_after': after['robot_state']}) + '\n')

    def _width_command(self, arm, width, name='width_m'):
        """The normalized finger command of a gripper width target, checked against the gripper's range."""
        try:
            width = float(width)
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be a number') from None
        robot = self.arms[arm]
        # With ROBOT_GRIPPER_SQUEEZE=1 a target below 0 is allowed, down to the robot's own firmest
        # closing command: the position-controlled fingers then press harder on what they hold.
        floor = gripper_width_floor(robot.gripper_scale[0], self.width_zero, self.squeeze)
        if not math.isfinite(width) or not floor - 1e-9 <= width <= .088807:
            raise ValueError(f'{name} outside [{floor:.4f}, .088807]')
        q = (width - self.width_zero) / 2.
        return float((q - robot.gripper_scale[0]) / (robot.gripper_scale[1] - robot.gripper_scale[0]))

    def _width_target(self, arm, width):
        self.grips[arm] = self._width_command(arm, width)

    def _motion_widths(self, entries):
        """{arm: (width_m, finger command)} for the gripper_width_m given with an arm motion; `entries` maps each
        moving arm to its motion parameters (a move's params or a waypoint's arm entry). Checked before anything
        moves, so an invalid width moves nothing."""
        widths = {}
        for arm, entry in entries.items():
            if isinstance(entry, dict) and entry.get('gripper_width_m') is not None:
                value = entry['gripper_width_m']
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError('gripper_width_m must be a number')
                command = self._width_command(arm, value, 'gripper_width_m')
                widths[arm] = (float(entry['gripper_width_m']), command)
        return widths

    def _trajectory(self, targets, duration, record, joint_targets=None, ramp_s=0., widths=None):
        """Step along the straight TCP path to `targets` (or, with joint_targets, joint-linearly) over `duration`;
        ramp_s > 0 (a smoothed motion profile) spreads the path over time with ramps at both ends. `widths`
        ({arm: (width_m, finger command)}, from _motion_widths) sets those grippers' targets with the first step
        actually taken, so the fingers move toward them during the motion at their own speed, as after
        gripper.set_width; a motion that stops before its first step leaves the gripper targets unchanged."""
        start = self.state()
        widths_applied = False
        steps = max(1, int(math.ceil(duration * self.hz)))
        ramp_fraction = ramp_s / duration if ramp_s else 0.
        status, reason = 'partial', 'time_budget'
        reached_fraction = 0.
        planned = None
        tracking_lag = None
        for i in range(steps):
            if record['cancel'].is_set(): status, reason = 'cancelled', 'cancel_requested'; break
            if self.env.end_flag[0]: reason = 'benchmark_terminal'; break
            fraction = (i + 1) / steps
            now = self.state()
            qs, desired = {}, {}
            for a, robot in self.arms.items():
                if joint_targets is not None:
                    qs[a] = np.asarray(start[a]['joint_positions_rad']) + fraction * (np.asarray(joint_targets[a]) - start[a]['joint_positions_rad'])
                else:
                    desired[a] = interpolate(start[a]['tcp'], targets[a], progress(fraction, ramp_fraction))
                    pose = link_from_tcp(desired[a], self.offset)
                    ik = self.rm.solve_ik(target_pose=pose.tolist(), env_idx=0, robot=robot)
                    if ik['status'] != 'Success':
                        reason = f'ik_failed:{a}'; break
                    qs[a] = np.asarray(ik['joint_value'], dtype=float)
                jump = np.max(np.abs(qs[a] - now[a]['joint_positions_rad']))
                if not np.all(np.isfinite(qs[a])) or jump > 5. / self.hz:
                    reason = f'joint_jump_or_speed_limit:{a}'; break
            else:
                if widths and not widths_applied:
                    for a, (_, command) in widths.items():
                        self.grips[a] = command
                    widths_applied = True
                self._step(qs, record['id'])
                reached_fraction = fraction
                actual = self.state()
                if joint_targets is None:
                    deviation = {a: errors(actual[a]['tcp'], desired[a]) for a in self.arms}
                    off_path = [a for a, (p, r) in deviation.items() if p > .02 or r > .25]
                    if off_path:
                        # Name, for each arm off its path, the joint furthest behind its command.
                        lags = {a: largest_joint_lag(self.arms[a].arm_joints_name, qs[a],
                                                     actual[a]['joint_positions_rad']) for a in off_path}
                        tracking_lag = {a: lag for a, lag in lags.items() if lag}
                        reason = 'tracking_error'; break
                if i == steps - 1: status, reason = 'reached', 'trajectory_completed'
                continue
            break
        final = self.state()
        errs = {a: {'position_m': errors(final[a]['tcp'], targets[a])[0],
                    'orientation_rad': errors(final[a]['tcp'], targets[a])[1]} for a in self.arms}
        if status == 'reached' and any(e['position_m'] > .003 or e['orientation_rad'] > .06 for e in errs.values()):
            status, reason = 'partial', 'final_tolerance_not_met'
        result = {'status': status, 'termination_reason': reason, 'target_tcp': targets,
            'measured_final_state': final, 'tcp_error': errs, 'path_fraction': reached_fraction}
        if tracking_lag:
            result['tracking_lag'] = tracking_lag
        if widths:
            # The measured end width (single master-joint estimate); the fingers may still be moving.
            result['gripper'] = {a: {'target_width_m': w, 'applied': widths_applied,
                                     'width_m': final[a]['gripper_width_m']} for a, (w, _) in widths.items()}
        return result

    def _move_request(self, state, params):
        """Targets, duration, timing and ramp seconds (motion profile) of a move request starting from `state`
        (executed or dry run)."""
        targets = {a: s['tcp'] for a, s in state.items()}
        arm = params['arm']
        targets[arm] = make_target(targets[arm], params, np.roll(self.arms[arm].entity_origin_pose[3:], -1))
        distance, angle = errors(state[arm]['tcp'], targets[arm])
        speed = float(params.get('speed_m_s', .2))
        angular_speed = float(params.get('speed_rad_s', .5))
        if not within(speed, self.limits['speed_m_s']) or not within(angular_speed, self.limits['speed_rad_s']):
            raise ValueError('Speed exceeds supported limits')
        duration, timing = move_timing(distance, angle, speed, angular_speed, self.limits, getattr(self, 'profile', None))
        return targets, duration, timing, timing.get('ramp_s', 0.)

    def _waypoint_request(self, tcps, wp):
        """Targets, requested duration, per-arm (distance, angle) and the motion profile's ramp seconds (None
        without a profile; added to the requested duration) of one trajectory waypoint starting from the TCP
        poses `tcps`. Its duration and speeds are judged by _waypoint_rejection and _speed_excess."""
        target = dict(tcps)
        duration = float(wp['duration_s'])
        if not duration >= .04: raise ValueError('Waypoint duration_s must be at least .04 seconds')
        motions = {}
        for a in self.arms:
            if a in wp:
                target[a] = make_target(target[a], wp[a], np.roll(self.arms[a].entity_origin_pose[3:], -1))
                motions[a] = errors(tcps[a], target[a])
        profile = getattr(self, 'profile', None)
        ramp = round(ramp_time(profile, max((d for d, _ in motions.values()), default=0.) / duration,
                               max((r for _, r in motions.values()), default=0.) / duration), 3) if profile else None
        return target, duration, motions, ramp

    def _speed_excess(self, motions, duration, margins=None):
        """The arms of a waypoint (`motions`: arm -> (distance_m, angle_rad)) that would move faster than the speed
        limits over `duration`: each with its distance, angle, speeds (unrounded, so a slight excess still shows),
        which limits it is `over` and the shortest duration within the limits (rounded up to 0.01 s), which for
        `margins` (arm -> (metres, radians)) also covers that much more distance and angle."""
        linear, angular = self.limits['speed_m_s'], self.limits['speed_rad_s']
        excess = {}
        for a, (d, r) in motions.items():
            over = [k for k, v, limit in (('speed_m_s', d, linear), ('speed_rad_s', r, angular)) if v / duration > limit + 1e-8]
            if over:
                extra_d, extra_r = (margins or {}).get(a, (0., 0.))
                extra_r = min(extra_r, max(0., math.pi - r))  # an orientation is never more than π away
                excess[a] = {'distance_m': round(d, 4), 'angle_rad': round(r, 4), 'speed_m_s': d / duration,
                             'speed_rad_s': r / duration, 'over': over,
                             'required_duration_s': math.ceil(max((d + extra_d) / linear, (r + extra_r) / angular) * 100
                                                              - 1e-9) / 100}
        return excess

    def _waypoint_rejection(self, tcps, waypoints):
        """Every waypoint checked before the first one runs, each starting where the one before ends (as in a dry
        run): the rejection of the whole trajectory when a waypoint lasts longer than 20 s or would move an arm faster
        than the speed limits (every such waypoint and arm listed), otherwise None.

        While it runs, each waypoint is checked again from the measured pose. A reached waypoint ends within 3 mm and
        0.06 rad of the target it executed, and a target given relative to the arm's pose (delta_m, a rotation, or an
        arm the waypoint leaves alone) carries over how far that pose already was from the plan (a tcp-frame delta_m
        adds its length times the angle error); so the bound grows by 3 mm and 0.06 rad per waypoint until an absolute
        target (position_m, quaternion_xyzw) resets it. The duration suggested for an absolute target covers that
        bound, so it still passes when checked again."""
        excess, too_long = [], []
        bound = {a: (0., 0.) for a in self.arms}  # (m, rad): measured pose vs the plan when this waypoint starts
        for index, wp in enumerate(waypoints, 1):
            target, duration, motions, _ = self._waypoint_request(tcps, wp)
            if duration > 20.:
                too_long.append((index, duration))
            entries = {a: (wp.get(a) if isinstance(wp.get(a), dict) else {}) for a in self.arms}
            # How far each arm's executed target may lie from the planned one: none for an absolute target, but a
            # quaternion turned further about a tcp-frame axis follows the start orientation (up to twice its error).
            target_error = {}
            for a, entry in entries.items():
                position, angle = bound[a]
                tcp_turn = entry.get('frame', 'world') == 'tcp' and bool(entry.get('rotation_axis')) and bool(entry.get('angle_deg'))
                if entry.get('position_m') is not None:
                    position = 0.
                elif entry.get('delta_m') is not None and entry.get('frame', 'world') == 'tcp':
                    position += float(np.linalg.norm(np.asarray(entry['delta_m'], dtype=float))) * angle
                if entry.get('quaternion_xyzw') is not None:
                    angle = 2 * angle if tcp_turn else 0.
                target_error[a] = (position, angle)
            # An absolute target's distance and angle grow by at most the start error plus the target's own error.
            margins = {a: (bound[a][0] if entries[a].get('position_m') is not None else 0.,
                           bound[a][1] + target_error[a][1] if entries[a].get('quaternion_xyzw') is not None else 0.)
                       for a in motions}
            excess += [{'waypoint': index, 'arm': a, 'duration_s': duration, **e}
                       for a, e in self._speed_excess(motions, duration, margins).items()]
            bound = {a: (p + .003, r + .06) for a, (p, r) in target_error.items()}
            tcps = target
        if not (excess or too_long):
            return None
        rejected = {'status': 'rejected',
                    'termination_reason': 'duration_exceeds_timeout' if too_long else 'exceeds_speed_limit'}
        if too_long:
            rejected.update(waypoint=too_long[0][0], required_duration_s=too_long[0][1])
        if excess:
            rejected.update(speed_excess=excess, limits={k: self.limits[k] for k in ('speed_m_s', 'speed_rad_s')})
        return rejected

    @staticmethod
    def _duration_rejection(params, duration, timing):
        # timeout_s is optional: without it only the 20 s duration limit applies.
        timeout = float(params['timeout_s']) if params.get('timeout_s') is not None else 30.
        if not .04 <= duration <= 20. or not 0 < timeout <= 30. or duration > timeout:
            return {'status': 'rejected', 'termination_reason': 'duration_exceeds_timeout', 'required_duration_s': duration,
                    'timing': timing}
        return None

    def _ik_from(self, robot, pose, seed):
        """IK of a link pose seeded with the joint values `seed`, through the same solver as rm.solve_ik (which seeds
        from the measured joints): a dry run seeds each step with the joints it planned for the step before."""
        planner = self.rm.ik_solver[robot.robot_name]
        return planner.solve_ik_to_joint([float(v) for v in seed], pose.tolist(),
                                         real_robot_pose=copy.deepcopy(robot.entity_origin_pose))

    def _dry_path(self, start, targets, duration, cancel=None, ramp_s=0.):
        """The per-step checks of a Cartesian motion (IK along the straight path, the joint-change limit), made
        without stepping the simulator. `start` holds each arm's 'tcp' and 'joint_positions_rad'; a set `cancel`
        (e.g. at the python_exec time limit) ends the check early; ramp_s as in _trajectory."""
        steps = max(1, int(math.ceil(duration * self.hz)))
        ramp_fraction = ramp_s / duration if ramp_s else 0.
        q = {a: np.asarray(start[a]['joint_positions_rad'], dtype=float) for a in self.arms}
        low, high = {a: v.copy() for a, v in q.items()}, {a: v.copy() for a, v in q.items()}
        status, reason, done = 'feasible', None, steps
        for i in range(steps):
            if cancel is not None and cancel.is_set():
                status, reason, done = 'cancelled', 'cancel_requested', i
                break
            fraction = (i + 1) / steps
            for a, robot in self.arms.items():
                pose = link_from_tcp(interpolate(start[a]['tcp'], targets[a], progress(fraction, ramp_fraction)),
                                     self.offset)
                ik = self._ik_from(robot, pose, q[a])
                if ik['status'] != 'Success':
                    status, reason = 'would_stop', f'ik_failed:{a}'; break
                planned = np.asarray(ik['joint_value'], dtype=float)
                if not np.all(np.isfinite(planned)) or np.max(np.abs(planned - q[a])) > 5. / self.hz:
                    status, reason = 'would_stop', f'joint_jump_or_speed_limit:{a}'; break
                q[a] = planned
                low[a], high[a] = np.minimum(low[a], planned), np.maximum(high[a], planned)
            else:
                continue
            done = i
            break
        return {'status': status, 'termination_reason': reason, 'path_fraction': done / steps, 'planned_steps': done,
                'target_tcp': targets,
                'planned_joint_range_rad': {a: {'joints': [str(n) for n in self.arms[a].arm_joints_name],
                                                'min': [round(float(v), 4) for v in low[a]],
                                                'max': [round(float(v), 4) for v in high[a]]} for a in self.arms},
                'planned_joints_rad': {a: [float(v) for v in q[a]] for a in self.arms}}

    def _waypoint_widths(self, waypoints):
        """The gripper widths of each waypoint (see _motion_widths), all checked before the first one runs."""
        return [self._motion_widths({a: wp[a] for a in self.arms if a in wp}) if isinstance(wp, dict) else {}
                for wp in waypoints]

    def _plan(self, params, cancel=None):
        """Dry run of a move or trajectory request: what its per-step checks would decide. Nothing moves."""
        params = dict(params)
        motion = params.pop('motion', None)
        state = self.state()
        note = {'moved': False, 'checks': PLAN_CHECKS}
        if motion == 'move':
            targets, duration, timing, ramp = self._move_request(state, params)
            widths = self._motion_widths({params['arm']: params})
            rejected = self._duration_rejection(params, duration, timing)
            if rejected:
                return {**rejected, 'moved': False}
            result = self._dry_path(state, targets, duration, cancel, ramp)
            result.pop('planned_joints_rad')
            if widths:  # checked against the gripper's range and echoed; finger motion is not simulated
                result['gripper'] = {a: {'target_width_m': w} for a, (w, _) in widths.items()}
            return {**result, 'timing': timing, **note}
        if motion == 'trajectory':
            waypoints = params.get('waypoints')
            if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 100: raise ValueError('Expected 1..100 waypoints')
            start = {a: {'tcp': s['tcp'], 'joint_positions_rad': s['joint_positions_rad']} for a, s in state.items()}
            widths = self._waypoint_widths(waypoints)
            rejected = self._waypoint_rejection({a: s['tcp'] for a, s in state.items()}, waypoints)
            if rejected:
                return {**rejected, **note}
            reports = []
            for wp, wp_widths in zip(waypoints, widths):
                target, duration, motions, ramp = self._waypoint_request({a: s['tcp'] for a, s in start.items()}, wp)
                report = self._dry_path(start, target, duration + (ramp or 0.), cancel, ramp or 0.)
                report['timing'] = waypoint_timing(motions, duration, self.limits, ramp)
                if wp_widths:
                    report['gripper'] = {a: {'target_width_m': w} for a, (w, _) in wp_widths.items()}
                joints = report.pop('planned_joints_rad')
                # The next waypoint starts where this one was planned to end.
                start = {a: {'tcp': target[a], 'joint_positions_rad': joints[a]} for a in self.arms}
                reports.append(report)
                if report['status'] != 'feasible': break
            return {'status': reports[-1]['status'], 'waypoint_reports': reports,
                    'planned_steps': sum(r['planned_steps'] for r in reports),
                    'timing': [r['timing'] for r in reports], **note}
        raise ValueError("plan needs motion='move' or motion='trajectory'")

    def _execute(self, method, params, record):
        if method in ('history', 'inspect', 'annotate'):
            return getattr(self.evidence, method)(**params)
        if method == 'describe': return self.describe()
        if method == 'observe': return self.observation()
        if method == 'replay': return self.observation(params['observation_id'])
        if method == 'mark_pose':
            name = params['name']
            if name == 'initial': raise ValueError('initial bookmark is reserved')
            self.marks[name] = self.state()
            (self.output / 'bookmarks.json').write_text(json.dumps(self.marks, indent=2))
            return {'bookmark': name, 'state': self.marks[name]}
        if method == 'finish':
            self.finish_requested = True
            return {'status': 'attempt_finished', 'observation_id': self.capture()['observation_id']}
        if self.env.end_flag[0] or self.finish_requested:
            return {'status': 'rejected', 'termination_reason': 'episode_already_ended'}
        if method == 'plan':
            return self._plan(params, record['cancel'])
        state = self.state()
        targets = {a: s['tcp'] for a, s in state.items()}
        joint_targets, ramp, widths = None, 0., None
        if method == 'move':
            targets, duration, timing, ramp = self._move_request(state, params)
            widths = self._motion_widths({params['arm']: params})
        elif method == 'gripper':
            duration = float(params.get('duration_s', .4))
            if not .04 <= duration <= 8.: raise ValueError('Gripper duration outside .04..8 seconds')
            timing = gripper_timing(duration)
            self._width_target(params['arm'], params['width_m'])
            joint_targets = {a: s['joint_positions_rad'] for a, s in state.items()}
        elif method == 'hold':
            # Both arms keep their joints while simulated time passes: the simulation otherwise pauses
            # between commands, so e.g. whether a released object stays put shows only once time advances. Bounds as a
            # gripper command's; the held-pose monitor watches both arms.
            duration = float(params.get('duration_s', 1.))
            if not .04 <= duration <= 8.: raise ValueError('Hold duration outside .04..8 seconds')
            timing = gripper_timing(duration)
            joint_targets = {a: s['joint_positions_rad'] for a, s in state.items()}
        elif method == 'move_to_mark':
            arm, name = params['arm'], params['name']
            if arm not in ('left', 'right', 'both'):
                raise ValueError("move_to_mark arm must be 'left', 'right' or 'both'")
            # 'both' moves the two arms' joints to the bookmark at once, in one duration (one set of steps).
            arms = ('left', 'right') if arm == 'both' else (arm,)
            joint_targets = {a: s['joint_positions_rad'] for a, s in state.items()}
            for a in arms:
                joint_targets[a] = self.marks[name][a]['joint_positions_rad']
                targets[a] = self.marks[name][a]['tcp']
            joint_speed = params.get('speed_rad_s')
            if joint_speed is not None:
                joint_speed = float(joint_speed)
                if not within(joint_speed, self.limits['mark_joint_speed_rad_s']):
                    raise ValueError(f"move_to_mark speed_rad_s must be in (0, {self.limits['mark_joint_speed_rad_s']:g}] rad/s")
            change = max(float(np.max(np.abs(np.asarray(joint_targets[a]) - state[a]['joint_positions_rad']))) for a in arms)
            duration, timing = mark_timing(change, self.limits, joint_speed)
        elif method == 'trajectory':
            reports = []
            waypoints = params['waypoints']
            if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 100: raise ValueError('Expected 1..100 waypoints')
            all_widths = self._waypoint_widths(waypoints)
            rejected = self._waypoint_rejection({a: s['tcp'] for a, s in self.state().items()}, waypoints)
            if rejected:
                return rejected
            for wp, widths in zip(waypoints, all_widths):
                state = self.state()
                target, duration, motions, ramp = self._waypoint_request({a: s['tcp'] for a, s in state.items()}, wp)
                excess = self._speed_excess(motions, duration)
                if excess:
                    # Checked again from the measured pose: the waypoint before may have ended up to 3 mm and 0.06 rad
                    # from its target, which a waypoint planned right at the speed limit cannot absorb.
                    reports.append({'status': 'rejected', 'termination_reason': 'exceeds_speed_limit',
                                    'speed_excess': [{'waypoint': len(reports) + 1, 'arm': a, 'duration_s': duration,
                                                      **e} for a, e in excess.items()],
                                    'limits': {k: self.limits[k] for k in ('speed_m_s', 'speed_rad_s')},
                                    'measured_final_state': state})
                    if widths:
                        reports[-1]['gripper'] = {a: {'target_width_m': w, 'applied': False}
                                                  for a, (w, _) in widths.items()}
                    break
                report = self._trajectory(target, duration + (ramp or 0.), record, ramp_s=ramp or 0., widths=widths)
                report['timing'] = waypoint_timing(motions, duration, self.limits, ramp)
                reports.append(report)
                if report['status'] != 'reached': break
            # A waypoint rejected after the first has run leaves the trajectory partial: earlier waypoints moved.
            status = 'partial' if reports[-1]['status'] == 'rejected' and len(reports) > 1 else reports[-1]['status']
            return {'status': status, 'waypoint_reports': reports,
                    'timing': [r['timing'] for r in reports if 'timing' in r]}
        else: raise ValueError('Unknown method')
        rejected = self._duration_rejection(params, duration, timing)
        if rejected:
            if widths:
                rejected['gripper'] = {a: {'target_width_m': w, 'applied': False} for a, (w, _) in widths.items()}
            return rejected
        result = self._trajectory(targets, duration, record, joint_targets=joint_targets, ramp_s=ramp, widths=widths)
        result['timing'] = timing
        if method == 'gripper':
            # A squeeze below 0 is judged against a closed gripper, so closing on nothing still reads as reached.
            error = abs(self.state()[params['arm']]['gripper_width_m'] - max(float(params['width_m']), 0.))
            result['gripper_width_error_m'] = error
            if error > .003:
                result['status'], result['termination_reason'] = 'partial', 'gripper_width_not_reached_contact_or_tracking_unknown'
        return result

    def execute(self, record):
        self.current_id = record['id']
        method, params = record['request']['method'], record['request'].get('params', {})
        if method in ('history', 'inspect', 'annotate', 'plan'):
            # Read-only: no observation, no command log entry, no physical step.
            return self._execute(method, params, record)
        initial = self.capture()
        start = initial['observation_id']
        monitor = HeldPoseMonitor(method, params, initial, getattr(self, 'hz', None))
        self._pose_monitor = monitor
        started = time.time()
        try:
            if record['cancel'].is_set():
                result = {'status': 'cancelled', 'termination_reason': 'cancelled_before_start'}
            else:
                result = self._execute(method, params, record)
        except Exception as error:
            if method not in ('observe', 'replay', 'describe'):
                # Preserve failed/partially executed requests as evidence too.
                try: end = self.capture()['observation_id']
                except Exception: end = None  # Never claim a stale pre-action image is the post-action state.
                result = {'command_id': record['id'], 'status': 'fault',
                          'termination_reason': str(error), 'start_observation_id': start,
                          'end_observation_id': end}
                motion = monitor.report()
                if motion: result['uncommanded_tcp_motion'] = motion
                self.commands.write(json.dumps({'request': record['request'], 'result': result,
                    'started_wall_s': started, 'finished_wall_s': time.time()}) + '\n')
            raise
        finally:
            self._pose_monitor = None
        if method not in ('observe', 'replay', 'describe'):
            result.update(command_id=record['id'], start_observation_id=start,
                          end_observation_id=self.capture()['observation_id'])
            motion = monitor.report()
            if motion: result['uncommanded_tcp_motion'] = motion
            self.commands.write(json.dumps({'request': record['request'], 'result': result,
                'started_wall_s': started, 'finished_wall_s': time.time()}) + '\n')
            self.low_trace.flush()
        return result

    def close(self):
        if getattr(self, '_original_control', None) is not None:
            self.rm.control_robot = self._original_control
        self.trace.close(); self.low_trace.close(); self.commands.close()
        if self.outer_collector is not None:
            self.outer_collector.close()
