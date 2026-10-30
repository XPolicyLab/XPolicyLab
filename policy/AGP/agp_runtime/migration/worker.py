"""Single simulator-owner thread inside the official episode hook."""

import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from migration.kinematics import x5_tcp_transform


def transform(pose):
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat(np.asarray(pose[3:])[[1, 2, 3, 0]]).as_matrix()
    return T


def pose(T):
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]]


def atomic(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, indent=2))
    tmp.replace(path)


def resample_joints(points, start, max_step):
    current = np.asarray(start, dtype=float)
    result = []
    for end in points:
        end = np.asarray(end, dtype=float)
        if end.shape != current.shape or not np.isfinite(end).all():
            raise ValueError("Invalid joint trajectory")
        n = max(1, int(np.ceil(np.max(np.abs(end - current)) / max_step)))
        result.extend(current + (end - current) * i / n for i in range(1, n + 1))
        current = end
    return result


def resample_tcp(points, start, fk, max_translation, max_rotation):
    """Refine joint segments, preserving corners and bounding sampled TCP deltas.

    These are command endpoint bounds, not instantaneous physical speed or
    acceleration guarantees. Joint limits must already have been sampled.
    """
    if not all(np.isfinite(x) and x > 0 for x in (max_translation, max_rotation)):
        raise ValueError('TCP sample limits must be positive and finite')
    current = np.asarray(start, dtype=float)
    current_tcp = fk(current)
    result = []
    evaluations = 1
    for end in points:
        end = np.asarray(end, dtype=float)
        n = 1
        for attempt in range(16):
            evaluations += n
            if evaluations > 20000:
                raise ValueError('TCP resampling exceeds FK work allowance')
            segment = [current + (end-current)*i/n for i in range(1, n)] + [end.copy()]
            frames = [current_tcp] + [fk(q) for q in segment]
            frames = np.asarray(frames)
            if frames.shape != (n+1, 4, 4) or not np.isfinite(frames).all():
                raise ValueError('Invalid TCP forward kinematics')
            translation = np.linalg.norm(np.diff(frames[:, :3, 3], axis=0), axis=1)
            rotation = (Rotation.from_matrix(frames[:-1, :3, :3]).inv() *
                        Rotation.from_matrix(frames[1:, :3, :3])).magnitude()
            ratio = max(float(translation.max()/max_translation),
                        float(rotation.max()/max_rotation))
            if ratio <= 1.0 + 1e-9:
                break
            n = max(n+1, int(math.ceil(n*ratio)))
            if n > 10000:
                raise ValueError('TCP resampling exceeds supported segment size')
        else:
            raise ValueError('TCP resampling did not converge')
        result.extend(segment)
        current, current_tcp = end, frames[-1]
    return result


class Worker:
    def __init__(self, env, session):
        self.env = env
        self.session = session
        self.bridge = session / "bridge"
        self.bridge.mkdir(exist_ok=True)
        self.rm = env.robot_manager
        self.arms = {}
        self.grips = {}
        for arm in ('left', 'right'):
            robot = next(r for r in self.rm.robot_list
                         if r.type == 'target' and r.arm_name.startswith(arm))
            base = transform(robot.entity_origin_pose)
            self.arms[arm] = {'robot': robot, 'base': base, 'invbase': np.linalg.inv(base),
                              'initial': self.rm.get_joint(robot)[0].copy(),
                              'tcp': x5_tcp_transform(float(robot.gripper_bias))}
            articulation = self.rm.robot_key[self.rm.robot_list.index(robot)]
            self.arms[arm]['gripper_limits_m'] = articulation.data.joint_pos_limits[
                0, robot.gripper_joint_indices].cpu().numpy().copy()
            self.grips[robot.arm_name] = 1.0
        self.started = time.monotonic()
        self.tools = 0
        self.motion_sequence = 0
        self.dt = 1 / env.obs_manager.collect_freq
        # Native commands address the X5 grasp center along link6 +X.
        goal = session / "goal"
        goal.mkdir(exist_ok=True)
        (goal / "INSTRUCTION.md").write_text(str(env.obs_manager.instruction[0]))
        frames = {}
        for arm, context in self.arms.items():
            robot = context['robot']
            link6 = transform(self.rm.get_real_endpose(robot, is_relative=False)[0])
            fingers = {name: pose(np.linalg.inv(link6) @ transform(
                self.rm.get_link_pose(robot, name, is_relative=False)[0])).tolist()
                for name in ('link7', 'link8')}
            frames[arm] = {'finger_poses_in_link6': fingers,
                           'T_link6_tcp': context['tcp'].tolist()}
            if any(abs(fingers[name][0] - 0.08657) > 0.002 for name in fingers):
                raise RuntimeError(f'{arm} X5 USD fingers disagree with URDF TCP convention')
        atomic(session/'robot_frame_validation.json', frames)
        atomic(session / "control_task.json", {"release_requirement": "support"})
        atomic(session / "worker_ready.json", {"arms": list(self.arms), "pid": os.getpid(),
            "task": os.environ["AGENT_TASK"], "action_budget": int(env.step_lim),
            "control_dt_s": self.dt, "dual_arm_exercise": os.environ.get("AGENT_DUAL_EXERCISE")=="1"})

    def tcp_base(self, arm):
        context = self.arms[arm]
        return (context['invbase'] @ transform(self.rm.get_real_endpose(
            context['robot'], is_relative=False)[0]) @ context['tcp'])

    def _transit_joint_velocity(self, arm):
        context = self.arms[arm]
        robot = context['robot']
        measured = np.asarray(self.rm.get_end_effector_real_val(robot)[0])
        limits = np.asarray(context['gripper_limits_m'])
        opened = (self.grips[robot.arm_name] == 1. and measured.shape == (2,)
                  and limits.shape == (2, 2) and np.isfinite(measured).all()
                  and np.isfinite(limits).all()
                  and np.all(np.abs(measured - limits[:, 1]) <= .001))
        # Calibrated only for open-gripper free travel. Closed/partial fingers
        # retain the established transit rate; this is not a possession sensor.
        return 2.5 if opened else 1.5

    def _prepare(self, arm, command, args):
        context = self.arms[arm]
        robot = context['robot']
        return self._prepare_from_state(arm, command, args,
            self.rm.get_joint(robot)[0], self.tcp_base(arm), self._transit_joint_velocity(arm))

    def _prepare_from_state(self, arm, command, args, start, tcp, transit_velocity):
        context = self.arms[arm]
        robot = context['robot']
        start = np.asarray(start).copy()
        tcp = np.asarray(tcp).copy()
        target = None
        profile = "transit" if command == "home" else args.get("speed_profile", "precision")
        raw_points = None
        velocity = 0.5 if profile == "precision" else transit_velocity
        # Official within-action interpolation uses 80 percent of each interval.
        max_step = velocity * self.dt * 0.6
        if command == "hold":
            points = [start] * args["samples"]
        elif command == "gripper":
            if args.get('action') not in {'open', 'close'}:
                return {'ok': False, 'error': 'Gripper action must be open or close'}
            points = [start] * 10
        elif command == "home":
            plan = self.rm.planner[robot.robot_name].plan_joint(start, context['initial'])
            if plan["status"] != "Success":
                return {"ok": False, "error": "Home planning rejected",
                        "motion_started": False, "motion_executed": False}
            raw_points = np.asarray(plan["position"])
            points = resample_joints(raw_points, start, max_step)
        elif command == "move_joints":
            points = resample_joints([np.asarray(args["joints"])], start, max_step)
        elif command in {"move_ee", "move_delta"}:
            target = tcp.copy()
            if command == "move_ee":
                target = transform(
                    [args["position"][k] for k in "xyz"]
                    + [args["rotation"][k] for k in "wxyz"]
                )
            else:
                target[:3, 3] += np.asarray(args.get("dpos", [0, 0, 0]))
                target[:3, :3] = (
                    Rotation.from_euler(
                        "xyz", args.get("drot_deg", [0, 0, 0]), degrees=True
                    ).as_matrix()
                    @ target[:3, :3]
                )
            planner = self.rm.planner[robot.robot_name]
            if args.get("mode", "linear") == "plan":
                end = pose(context['base'] @ target @ np.linalg.inv(context['tcp']))
                ik = planner.solve_ik_to_joint(start, end, robot.entity_origin_pose)
                if ik["status"] != "Success":
                    return {"ok": False, "error": "Planning IK rejected",
                            "motion_started": False, "motion_executed": False}
                plan = planner.plan_joint(start, ik["joint_value"])
                if plan["status"] != "Success":
                    return {
                        "ok": False,
                        "error": "Planning rejected",
                        "motion_started": False,
                        "motion_executed": False,
                    }
                raw_points = np.asarray(plan["position"])
                points = resample_joints(raw_points, start, max_step)
            else:
                distance = np.linalg.norm(target[:3, 3] - tcp[:3, 3])
                angle = (
                    Rotation.from_matrix(target[:3, :3])
                    * Rotation.from_matrix(tcp[:3, :3]).inv()
                ).magnitude()
                n = max(
                    1,
                    math.ceil(
                        distance
                        / ((0.05 if profile == "precision" else 0.20) * self.dt * 0.6)
                    ),
                    math.ceil(
                        angle
                        / ((0.18 if profile == "precision" else 0.7) * self.dt * 0.6)
                    ),
                )
                interp = Slerp(
                    [0, 1],
                    Rotation.from_matrix(np.stack([tcp[:3, :3], target[:3, :3]])),
                )
                points = []
                previous = start
                for i in range(1, n + 1):
                    T = np.eye(4)
                    T[:3, 3] = tcp[:3, 3] + (target[:3, 3] - tcp[:3, 3]) * i / n
                    T[:3, :3] = interp(i / n).as_matrix()
                    result = planner.solve_ik_to_joint(
                        previous,
                        pose(context['base'] @ T @ np.linalg.inv(context['tcp'])),
                        robot.entity_origin_pose,
                    )
                    if result["status"] != "Success":
                        return {
                            "ok": False,
                            "error": f"Linear IK rejected at sample {i}/{n}",
                            "motion_started": False,
                            "motion_executed": False,
                        }
                    q = np.asarray(result["joint_value"])
                    if np.max(np.abs(q - previous)) > 0.2:
                        return {
                            "ok": False,
                            "error": "Linear IK branch discontinuity",
                            "max_joint_jump_rad": float(np.max(np.abs(q - previous))),
                            "motion_started": False,
                            "motion_executed": False,
                        }
                    points.extend(resample_joints([q], previous, max_step))
                    previous = q
        else:
            raise ValueError("Unsupported command")
        if command in {'move_ee', 'move_delta'}:
            try:
                points = resample_tcp(points, start,
                    lambda q: self._predicted_tcp(arm, q),
                    (.05 if profile == 'precision' else .20) * self.dt * .6,
                    (.18 if profile == 'precision' else .7) * self.dt * .6)
            except Exception as exc:
                return {'ok': False, 'error': f'Cartesian rate preparation rejected: {exc}',
                        'motion_started': False, 'motion_executed': False}
        return {"ok": True, "start": np.asarray(start).copy(), "target": target,
                "points": points, "raw_points": raw_points, "profile": profile,
                "joint_velocity_limit_rad_s": velocity,
                "command": command, "args": args,
                "gripper_command": (1. if args['action']=='open' else 0.) if command=='gripper' else None}

    def action_dict(self, joints):
        """Both arms always enter the same official physical timestep."""
        result = {}
        for arm, ctx in self.arms.items():
            r = ctx['robot']
            q = joints.get(arm, self.rm.get_joint(r)[0])
            result[self.rm.process_name(r.arm_name)] = np.asarray(q).tolist()
            result[self.rm.process_name(r.gripper_name)] = [self.grips[r.arm_name]]
        return result

    def _errors(self, arm, plan):
        achieved = self.tcp_base(arm)
        target = plan['target']
        q = self.rm.get_joint(self.arms[arm]['robot'])[0]
        joint_error = float(np.max(np.abs(q - plan['points'][-1])))
        position = float(np.linalg.norm(achieved[:3,3]-target[:3,3])) if target is not None else None
        rotation = float((Rotation.from_matrix(achieved[:3,:3]) *
                          Rotation.from_matrix(target[:3,:3]).inv()).magnitude()) if target is not None else None
        args = plan['args']
        reached = (position <= args.get('position_tolerance_m', .002) and
                   rotation <= args.get('rotation_tolerance_rad', .02)) if target is not None else joint_error <= .003
        gripper_error = drive_error = None
        endpoint = None
        if plan['command'] == 'gripper':
            r = self.arms[arm]['robot']
            measured = np.asarray(self.rm.get_end_effector_real_val(r)[0])
            mapping = r.gripper_move
            index = (1 if args['action']=='open' else 0) if mapping['sign']==1 else (0 if args['action']=='open' else 1)
            base_target = float(r.gripper_scale[index])
            desired = np.array([base_target, base_target*mapping['mimic'][1]+mapping['mimic'][2]])
            limits = self.arms[arm]['gripper_limits_m']
            endpoint = np.clip(desired, limits[:,0], limits[:,1])
            drive_error = float(np.max(np.abs(measured-desired)))
            gripper_error = float(np.max(np.abs(measured-endpoint)))
            endpoint = endpoint.tolist()
            reached = gripper_error < .003
        return {'target_reached': bool(reached), 'position_error_m': position,
                'rotation_error_rad': rotation, 'joint_error_rad': joint_error,
                'gripper_error_m': gripper_error, 'gripper_drive_error_m': drive_error,
                'gripper_endpoint_m': endpoint,
                'gripper_measured_m': measured.tolist() if plan['command']=='gripper' else None,
                'gripper_drive_target_m': desired.tolist() if plan['command']=='gripper' else None,
                'achieved_tcp_pose_base': pose(achieved).tolist(),
                'position_residual_m': (target[:3,3]-achieved[:3,3]).tolist() if target is not None else None}

    def execute(self, command, args):
        if command == 'estimate':
            return self.estimate(args['stages'], args.get('preview'), args.get('preview_source'))
        if command != 'actions':
            return {'ok': False, 'error': 'Use explicit actions map',
                    'motion_started': False, 'motion_executed': False, 'low_level_actions': 0}
        return self.execute_actions(args['actions'])

    def _validate_motion(self, arm, start, points):
        robot = self.arms[arm]['robot']
        return self.rm.planner[robot.robot_name].validate_joint_trajectory(start, points)

    def _prepare_stage(self, actions, starts=None):
        reject = lambda error, **extra: dict(ok=False, error=error, **extra)
        if not isinstance(actions, dict) or not actions or set(actions)-set(self.arms):
            return reject('Expected one or both explicit arm commands')
        plans = {}
        for arm, action in actions.items():
            if not isinstance(action, dict) or not isinstance(action.get('args'), dict):
                return reject('Each arm requires command and args')
            a = action['args']
            for key, lo, hi in [('max_samples',1,1000),('settle_samples',0,10),('samples',1,25)]:
                if key in a and (type(a[key]) is not int or not lo <= a[key] <= hi):
                    return reject(f'{key} must be an integer in [{lo},{hi}]')
            for key, lo, hi in [('position_tolerance_m',.0005,.01),('rotation_tolerance_rad',.005,.1)]:
                if key in a and (isinstance(a[key],bool) or not isinstance(a[key],(int,float)) or not lo<=a[key]<=hi):
                    return reject(f'{key} outside supported range')
            if starts is None:
                plan = self._prepare(arm, action['command'], a)
            else:
                state = starts[arm]
                plan = self._prepare_from_state(arm, action['command'], a,
                    state['joints'], state['tcp'], state['transit_velocity'])
            if not plan['ok']:
                return reject(plan['error'], rejected_arm=arm, arms={arm:plan})
            plan['full_samples'] = len(plan['points'])
            plan['run_samples'] = min(len(plan['points']),a.get('max_samples',len(plan['points'])))
            plan['checkpoint'] = plan['run_samples'] < plan['full_samples']
            plans[arm] = plan
        return {'ok': True, 'plans': plans}

    def _preflight_stage(self, plans):
        preflight = {}
        for arm, p in plans.items():
            if p['command'] in {'hold', 'gripper'}:
                preflight[arm] = {'ok': True, 'scope': 'current-joint contact hold'}
                continue
            try:
                preflight[arm] = self._validate_motion(
                    arm, p['start'], np.asarray(p['points'][:p['run_samples']]))
                if type(preflight[arm].get('ok')) is not bool:
                    raise ValueError('Invalid preflight result')
            except Exception as exc:
                preflight[arm] = {'ok': False, 'error': f'Preflight unavailable: {exc}'}
        return preflight

    def _predicted_tcp(self, arm, joints):
        context = self.arms[arm]
        planner = self.rm.planner[context['robot'].robot_name]
        return transform(planner.forward_ee_pose_base(joints)) @ context['tcp']

    def _preview_state(self):
        return {'count': int(self.env.take_action_cnt[0]),
            'time': float(self.env.sim.sim.current_time), 'ended': bool(self.env.end_flag[0]),
            'grips': dict(self.grips),
            'arms': {arm: {'joints': np.asarray(self.rm.get_joint(c['robot'])[0]).tolist(),
                'fingers': np.asarray(self.rm.get_end_effector_real_val(c['robot'])[0]).tolist(),
                'tcp': self.tcp_base(arm).tolist()} for arm,c in self.arms.items()}}

    def estimate(self, stages, preview=None, preview_source=None):
        if preview is None:
            return self._estimate_core(stages)
        from migration.preview import validate_request, make_preview
        preview = validate_request(preview)
        if isinstance(stages,list) and stages and any(a.get('command') == 'gripper' for a in stages[0].values()):
            raise ValueError('Stage0 gripper preview unsupported')
        before = self._preview_state()
        candidates = []
        result = self._estimate_core(stages, (preview, candidates))
        if not candidates:
            unchanged = before == self._preview_state()
            result['preview'] = {'status':'unavailable','reason':'Stage0 unavailable or preflight rejected',
                'measured_state_unchanged':unchanged}
            if not unchanged:
                result.update(ok=False, estimate_complete=False, reason='INVALID_ESTIMATE: measured state changed')
            return result
        try:
            bound = candidates[0]
            if isinstance(bound,Exception):
                raise bound
            if not isinstance(preview_source,dict) or preview_source.get('sim_time_s') != result['state']['sim_time_s']:
                raise ValueError('Capture simulator time differs from estimate start')
            rendered = make_preview(self.session, preview, preview_source, bound)
            if before != self._preview_state():
                raise RuntimeError('Measured simulator state changed during preview estimate')
            rendered['preview']['nominal_fits'] = result['nominal_fits']
            result.update(rendered)
        except Exception as exc:
            result['preview'] = {'status':'unavailable','reason':str(exc)}
            result.pop('images',None)
        unchanged = before == self._preview_state()
        result['preview']['measured_state_unchanged'] = unchanged
        if not unchanged:
            result.update(ok=False, estimate_complete=False, reason='INVALID_ESTIMATE: measured state changed')
            result['preview'].update(status='unavailable', reason='Measured simulator state changed during preview estimate')
            result.pop('images',None)
        return result

    def _estimate_core(self, stages, preview_capture=None):
        count = int(self.env.take_action_cnt[0])
        remaining = max(0, self.env.step_lim - count)
        starts = {}
        state_arms = {}
        for arm, context in self.arms.items():
            robot = context['robot']
            joints = np.asarray(self.rm.get_joint(robot)[0]).copy()
            tcp = self.tcp_base(arm).copy()
            fingers = np.asarray(self.rm.get_end_effector_real_val(robot)[0]).copy()
            starts[arm] = {'joints': joints, 'tcp': tcp,
                           'transit_velocity': self._transit_joint_velocity(arm)}
            state_arms[arm] = {'joints': joints.tolist(), 'tcp_pose_base': pose(tcp).tolist(),
                'gripper_command': self.grips[robot.arm_name], 'gripper_measured_m': fingers.tolist()}
        result = {'ok': False, 'estimate_complete': False, 'low_level_actions': 0,
            'motion_started': False, 'motion_executed': False,
            'state': {'low_level_actions': count, 'remaining_actions': int(remaining),
                      'sim_time_s': float(self.env.sim.sim.current_time),
                      'episode_ended': bool(self.env.end_flag[0]), 'arms': state_arms},
            'stages': [], 'total_samples': None, 'known_prefix_samples': 0,
            'total_additional_settle_cap': 0, 'nominal_plus_settle_cap': None,
            'nominal_fits': None, 'nominal_plus_settle_cap_fits': None,
            'collision_scope': 'Sampled robot bounds/self and installed static fixture only; '
                               'no task objects, attached objects or inter-arm collision guarantee.',
            'limitations': 'Future states assume tracking of each executed-prefix endpoint without '
                'contact, slip or obstruction. Future mechanical gripper state is unknown, so later '
                'stages use 1.5 rad/s transit. No possession or success inference. Planning can change '
                'planner RNG/cache; execution replans. Settling cap excludes retries and corrections.'}
        if not isinstance(stages, list) or not 1 <= len(stages) <= 12:
            return dict(result, failed_stage=None, reason='Expected 1 to 12 stages')
        total = 0
        settle_total = 0
        for index, actions in enumerate(stages):
            try:
                prepared = self._prepare_stage(actions, starts)
                if not prepared['ok']:
                    return dict(result, failed_stage=index, reason=prepared['error'])
                plans = prepared['plans']
                preflight = self._preflight_stage(plans)
                stage_samples = max(p['run_samples'] for p in plans.values())
                settle_cap = max((p['args'].get('settle_samples', 3) for p in plans.values()
                    if not p['checkpoint'] and p['command'] in
                    {'move_ee', 'move_delta', 'move_joints', 'home'}), default=0)
                arms = {}
                predicted = {}
                for arm, plan in plans.items():
                    endpoint = np.asarray(plan['points'][plan['run_samples'] - 1]).copy()
                    tcp = self._predicted_tcp(arm, endpoint)
                    predicted[arm] = {'joints': endpoint, 'tcp': tcp, 'transit_velocity': 1.5}
                    arms[arm] = {'full_samples': plan['full_samples'],
                        'scheduled_samples': plan['run_samples'], 'checkpoint': plan['checkpoint'],
                        'joint_velocity_limit_rad_s': plan['joint_velocity_limit_rad_s'],
                        'predicted_end_joints': endpoint.tolist(),
                        'predicted_end_tcp': pose(tcp).tolist(), 'preflight': preflight[arm]}
                entry = {'stage_index': index, 'arms': arms, 'stage_samples': stage_samples,
                    'additional_settle_cap': settle_cap, 'cumulative_samples': total + stage_samples,
                    'nominal_fits': total + stage_samples <= remaining,
                    'nominal_plus_settle_cap_fits': total + stage_samples + settle_total + settle_cap <= remaining}
                result['stages'].append(entry)
                if not all(check['ok'] for check in preflight.values()):
                    return dict(result, failed_stage=index, reason='Joint trajectory preflight rejected')
                if index == 0 and preview_capture is not None:
                    from migration.preview import bind_candidate
                    request, captured = preview_capture
                    try:
                        captured.append(bind_candidate(state_arms, plans, request))
                    except Exception as exc:
                        captured.append(exc)
                starts.update(predicted)
                # Shared physics would advance for both arms. No future finger
                # measurements exist, including after a hypothetical open command.
                for state in starts.values():
                    state['transit_velocity'] = 1.5
                total += stage_samples
                settle_total += settle_cap
                result['known_prefix_samples'] = total
                result['total_additional_settle_cap'] = settle_total
            except Exception as exc:
                return dict(result, failed_stage=index, reason=str(exc))
        return dict(result, ok=True, estimate_complete=True, total_samples=total,
                    nominal_plus_settle_cap=total + settle_total,
                    nominal_fits=total <= remaining,
                    nominal_plus_settle_cap_fits=total + settle_total <= remaining,
                    failed_stage=None, reason=None)

    def execute_actions(self, actions):
        reject = lambda error, **extra: dict(ok=False, error=error, motion_started=False,
                                             motion_executed=False, low_level_actions=0, **extra)
        if not actions or set(actions)-set(self.arms):
            return reject('Expected one or both explicit arm commands')
        if self.env.is_episode_end():
            return reject('Episode ended')
        self.motion_sequence += 1
        prepared = self._prepare_stage(actions)
        if not prepared['ok']:
            return reject(prepared['error'], **{k:v for k,v in prepared.items() if k not in {'ok','error'}})
        plans = prepared['plans']
        count = int(self.env.take_action_cnt[0])
        remaining = max(0,self.env.step_lim-count)
        scheduled = max(p['run_samples'] for p in plans.values())
        directory = self.session/'motion_diagnostics'
        directory.mkdir(exist_ok=True)
        diagnostics = {}
        prefixes = {}
        for arm,p in plans.items():
            delta = np.diff(np.vstack([p['start'],p['points']]),axis=0)
            d = {'arm':arm,'command':p['command'],'args':p['args'],'profile':p['profile'],
                 'action_count_before':count,'dt_s':self.dt,
                 'joint_velocity_limit_rad_s':p.get('joint_velocity_limit_rad_s'),
                 'planner_input_samples':len(p['raw_points']) if p['raw_points'] is not None else None,
                 'output_samples':p['full_samples'],'scheduled_samples':p['run_samples'],
                 'joint_path_length_rad':np.sum(np.abs(delta),axis=0).tolist(),
                 'linf_path_length_rad':float(np.max(np.abs(delta),axis=1).sum()),
                 'max_command_velocity_rad_s':float(np.max(np.abs(delta))/self.dt),
                 'max_command_acceleration_rad_s2':float(np.max(np.abs(np.diff(
                     np.vstack([np.zeros_like(p['start']),delta,np.zeros_like(p['start'])]),axis=0)))/self.dt**2),
                 'expected_duration_s':scheduled*self.dt}
            prefix=directory/f"{count:04d}_{self.motion_sequence:04d}_{arm}_{p['command']}"
            prefixes[arm]=prefix; diagnostics[arm]=d
            np.savez_compressed(prefix.with_suffix('.npz'), start=p['start'],
                planned=p['raw_points'] if p['raw_points'] is not None else np.empty((0,len(p['start']))),
                commanded=np.asarray(p['points']))
        if scheduled>remaining:
            for arm,d in diagnostics.items():
                d.update(executed_samples=0,stop_reason='insufficient action budget')
                atomic(prefixes[arm].with_suffix('.json'),d)
            return reject('Planned motion exceeds remaining action budget',
                          planned_samples=scheduled,remaining_actions=int(remaining))
        # Validate FINAL resampled execution prefixes, after both preparations and
        # before either gripper state or physics changes. Holds and gripper-only
        # commands keep measured joints and preserve intended contact semantics.
        preflight = self._preflight_stage(plans)
        for arm, result in preflight.items():
            diagnostics[arm]['preflight'] = result
        rejected = [arm for arm, result in preflight.items() if not result['ok']]
        if rejected:
            for arm, d in diagnostics.items():
                d.update(executed_samples=0, stop_reason='joint trajectory preflight rejected')
                atomic(prefixes[arm].with_suffix('.json'), d)
            return reject('Joint trajectory preflight rejected', rejected_arm=rejected[0],
                          arms={arm: {'ok': result['ok'], 'preflight': result,
                                      'motion_started': False, 'motion_executed': False}
                                for arm, result in preflight.items()})
        starts={arm:self.rm.get_joint(ctx['robot'])[0].copy() for arm,ctx in self.arms.items()}
        feedback={arm:[q.copy()] for arm,q in starts.items()}
        sent={arm:[] for arm in self.arms}
        prospective_grips=dict(self.grips)
        for arm,p in plans.items():
            if p['command']=='gripper':
                prospective_grips[self.arms[arm]['robot'].arm_name]=p['gripper_command']
        self.grips=prospective_grips
        sim_start=float(self.env.sim.sim.current_time)
        wall_start=time.monotonic()
        stop_reason=None
        faults={}
        trajectory_samples=0
        settle_samples=0
        last={arm:q.copy() for arm,q in starts.items()}
        def sample(targets):
            nonlocal stop_reason
            if self.env.is_episode_end():
                stop_reason = ('official action budget exhausted' if self.env.take_action_cnt[0]>=self.env.step_lim
                               else 'official episode ended')
                return False
            self.env.take_action(self.action_dict(targets))
            for arm,ctx in self.arms.items():
                q=self.rm.get_joint(ctx['robot'])[0].copy()
                feedback[arm].append(q); sent[arm].append(np.asarray(targets[arm]).copy())
                if np.max(np.abs(q-targets[arm]))>.08:
                    faults[arm]=f'{arm} joint tracking error exceeded 0.08 rad'
                    stop_reason=faults[arm]
            if not self.env.end_flag[0]: self.env.get_obs()
            return True
        for i in range(scheduled):
            for arm,p in plans.items():
                last[arm]=p['points'][min(i,p['run_samples']-1)]
            if not sample(last): break
            trajectory_samples+=1
            if stop_reason: break
        # Count every convergence sample. No bias or uncounted physics. A checkpoint
        # deliberately discards its unexecuted suffix and never settles toward it.
        eligible={arm:p for arm,p in plans.items() if not p['checkpoint'] and
                  p['command'] in {'move_ee','move_delta','move_joints','home'}}
        stalled={arm:0 for arm in eligible}
        previous={arm:self._errors(arm,p)['joint_error_rad'] for arm,p in eligible.items()}
        local_settles={arm:0 for arm in plans}
        settle_reasons={}
        if trajectory_samples==scheduled and not stop_reason:
            for n in range(max((p['args'].get('settle_samples',3) for p in eligible.values()),default=0)):
                pending=[arm for arm,p in eligible.items() if n<p['args'].get('settle_samples',3)
                         and not self._errors(arm,p)['target_reached'] and stalled[arm]<2]
                if not pending: break
                released_cartesian = [arm for arm,p in eligible.items()
                    if arm not in pending and p['command'] in {'move_ee','move_delta'}
                    and not self._errors(arm,p)['target_reached']]
                if released_cartesian:
                    # A measured-pose replacement could jump beyond the sampled
                    # Cartesian bound. End this pair instead of sending it or
                    # continuing to press an exhausted/stalled endpoint.
                    stop_reason = ('Cartesian endpoint no longer settling: ' +
                                   ', '.join(released_cartesian))
                    break
                # Stop pressing a failed endpoint while the partner continues.
                # Keep the original target in plans for honest residual reporting.
                for arm,p in eligible.items():
                    if arm not in pending and not self._errors(arm,p)['target_reached']:
                        last[arm]=self.rm.get_joint(self.arms[arm]['robot'])[0].copy()
                if not sample(last): break
                settle_samples+=1
                for arm in pending:
                    local_settles[arm]+=1
                    error=self._errors(arm,plans[arm])['joint_error_rad']
                    stalled[arm]=stalled[arm]+1 if previous[arm]-error<.00005 else 0
                    previous[arm]=error
                if stop_reason: break
            for arm,p in eligible.items():
                if not self._errors(arm,p)['target_reached']:
                    settle_reasons[arm] = ('endpoint progress stalled' if stalled[arm]>=2
                                           else 'endpoint settle allowance exhausted')
        executed=int(self.env.take_action_cnt[0])-count
        duration=float(self.env.sim.sim.current_time)-sim_start
        results={}
        tracking={arm:float(np.max(np.abs(np.asarray(feedback[arm][1:])-np.asarray(sent[arm]))))
                  if sent[arm] else None for arm in self.arms}
        for arm,p in plans.items():
            errors=self._errors(arm,p)
            complete=trajectory_samples>=p['full_samples'] and not p['checkpoint']
            reached=bool(complete and errors['target_reached'] and arm not in faults)
            outcome='reached' if reached else ('checkpoint' if p['checkpoint'] and trajectory_samples>=p['run_samples'] and arm not in faults else 'held_short')
            result=dict(errors, ok=bool(reached or p['command']!='gripper' or p['args'].get('action')=='close'),
                target_reached=reached, motion_started=executed>0,motion_executed=executed>0,
                motion_outcome=outcome,planned_samples=p['full_samples'],
                trajectory_samples=min(trajectory_samples,p['run_samples']),settle_samples=local_settles[arm],
                padding_hold_samples=executed-min(trajectory_samples,p['run_samples'])-local_settles[arm],
                requested_trajectory_samples=p['run_samples'],
                stop_reason=(None if reached or outcome=='checkpoint' else
                             faults.get(arm) or stop_reason or settle_reasons.get(arm)),
                max_joint_tracking_error_rad=tracking[arm],
                position_tolerance_m=p['args'].get('position_tolerance_m',.002),
                rotation_tolerance_rad=p['args'].get('rotation_tolerance_rad',.02))
            results[arm]=result
            diagnostics[arm].update(result,executed_samples=executed,actual_duration_s=duration)
            np.save(prefixes[arm].with_name(prefixes[arm].name+'_feedback.npy'),np.asarray(feedback[arm]))
            np.save(prefixes[arm].with_name(prefixes[arm].name+'_executed.npy'),np.asarray(sent[arm]))
            atomic(prefixes[arm].with_suffix('.json'),diagnostics[arm])
        np.savez_compressed(directory/f'{count:04d}_{self.motion_sequence:04d}_shared_feedback.npz',
            **{f'{a}_measured':np.asarray(v) for a,v in feedback.items()},
            **{f'{a}_commanded':np.asarray(v) for a,v in sent.items()})
        return {'ok':not faults and all(r['ok'] for r in results.values()),'arms':results,
                'motion_started':executed>0,'motion_executed':executed>0,
                'low_level_actions':executed,'planned_samples':scheduled,
                'trajectory_samples':trajectory_samples,'settle_samples':settle_samples,
                'sim_duration_s':duration,'wall_duration_s':time.monotonic()-wall_start,
                'stop_reason':stop_reason or ('endpoint progress stalled' if any('stalled' in v for v in settle_reasons.values())
                                               else ('endpoint settle allowance exhausted' if settle_reasons else None)),
                'tracking_by_arm':tracking,
                'episode_ended':bool(self.env.end_flag[0])}

    def serve(self):
        processed = set()
        deadline = time.monotonic() + float(os.environ.get("AGENT_TIMEOUT_S", "3600"))
        while (
            time.monotonic() < deadline
            and not (self.session / "policy_finished.json").exists()
        ):
            for path in sorted(self.bridge.glob("req_*.json")):
                if path.name in processed:
                    continue
                req = json.loads(path.read_text())
                rid = req["id"]
                response = self.bridge / f"resp_{rid:06d}.json"
                if response.exists():
                    processed.add(path.name)
                    continue
                claim = self.bridge / f"started_{rid:06d}.json"
                if claim.exists():
                    raise RuntimeError(
                        "Unresolved worker execution claim. Replay forbidden."
                    )
                with claim.open("x") as stream:
                    json.dump(req, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                directory_fd = os.open(self.bridge, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                self.tools += 1
                count_before = self.env.take_action_cnt[0]
                try:
                    result = (
                        self.observe()
                        if req["cmd"] == "observe"
                        else self.execute(req["cmd"], req["args"])
                    )
                except Exception as exc:
                    traceback.print_exc()
                    result = {
                        "ok": False,
                        "error": str(exc),
                        "motion_started": self.env.take_action_cnt[0] > count_before,
                        "low_level_actions": self.env.take_action_cnt[0] - count_before,
                    }
                atomic(response, dict(result, id=rid))
                processed.add(path.name)
            time.sleep(0.01)
        if not (self.session / "policy_finished.json").exists():
            atomic(self.session / "policy_timeout.json", {"timeout": True})


def run_episode(env):
    session = Path(os.environ["AGENT_SESSION"]).resolve()
    session.mkdir(parents=True, exist_ok=True)
    worker = None
    try:
        worker = Worker(env, session)
        worker.serve()
    except Exception:
        (session / "worker_error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        # Early policy exit is evaluated as an official termination, never left
        # with EvalEnv's optimistic initial success value.
        if not env.is_episode_end():
            env.success[0] = False
            env.is_episode_end()
        # Post-policy diagnostics are archived only for the evaluator/operator.
        # They are never included in observations or MCP policy responses.
        try:
            checks = []
            def record_checks(node, path):
                if isinstance(node, tuple) and len(node) == 2 and isinstance(node[0], str):
                    value = env.reward_manager.call_func_parser(node, 0)
                    checks.append({"path": path, "predicate": node[0],
                                   "args": str(node[1]), "value": float(value)})
                elif isinstance(node, list):
                    for i, child in enumerate(node):
                        record_checks(child, path + [i])
            record_checks(env.reward_manager.check_list[0], [])
            atomic(session / "evaluator_diagnostics.json", {"checks": checks})
        except Exception:
            (session / "diagnostics_error.txt").write_text(traceback.format_exc())
        atomic(
            session / "episode_metrics.json",
            {
                "task": os.environ["AGENT_TASK"],
                "action_budget": int(env.step_lim),
                "low_level_actions": env.take_action_cnt[0],
                "official_success": bool(env.success[0]),
                "sim_time_s": float(env.sim.sim.current_time),
                "tool_requests": worker.tools if worker else 0,
                "wall_duration_s": time.monotonic() - worker.started
                if worker
                else None,
            },
        )
