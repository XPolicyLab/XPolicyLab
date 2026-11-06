"""Public RGB/state evidence and resource checks for policy adapters."""
from rsi.resource_locations import load_manifest as load_resource_manifest
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from harness.runtime.native_rule_evaluation.clock import upgrade_official_action_format
from harness.runtime.native_rule_evaluation.recording import jsonable

SIDES = ('left', 'right')
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
STATE_WIDTHS = {side + '_' + name: width for side in SIDES for name, width in
                (('arm_joint_state', 6), ('ee_pose', 7), ('ee_joint_state', 1))}


def project(observation):
    """Keep public RGB and proprioception, including invalid feedback as evidence.

    The actor's original projection owns finite-feedback admission. This
    recorder must retain the actual invalid frame rather than replacing it.
    """
    return {'state': {key: np.asarray(observation['state'][key]).copy() for key in STATE_WIDTHS},
            'vision': {camera: {'colors': np.asarray(observation['vision'][camera][
                'color' if 'color' in observation['vision'][camera] else 'colors'])[..., :3].copy()}
                for camera in CAMERAS}}


class PublicEvidence:
    """Full evaluator trajectory; the actor receives a detached current frame."""

    def __init__(self, output_root):
        self.output_root = output_root
        self.images = {camera: [] for camera in CAMERAS}
        self.public_grippers = {side: [] for side in SIDES}
        self.last_observation = None
        for side in SIDES:
            setattr(self, 'state_' + side + '_arm', [])
            setattr(self, 'state_' + side + '_ee', [])

    def append(self, observation, robots):
        self.last_observation = project(observation)
        state = self.last_observation['state']
        for side in SIDES:
            getattr(self, 'state_' + side + '_arm').append(state[side + '_arm_joint_state'])
            getattr(self, 'state_' + side + '_ee').append(state[side + '_ee_pose'])
            self.public_grippers[side].append(state[side + '_ee_joint_state'])
        for camera in CAMERAS:
            self.images[camera].append(self.last_observation['vision'][camera]['colors'])


def verify_resources(task):
    resources = load_resource_manifest(Path(task) / 'resources.json')
    def check(path, expected):
        path = Path(path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned public toast resource changed: ' + str(path))
    for key in ('public_robot_urdf', 'public_robot_usd', 'public_robot_config'):
        check(resources[key]['path'], resources[key]['sha256'])
    root = Path(resources['public_robot_urdf']['path']).parent
    for name, expected in resources['public_robot_meshes'].items():
        check(root / name, expected)
    assets = resources['public_toast_assets']
    for name, expected in assets['files'].items():
        check(Path(assets['path']) / name, expected)
    return resources


def robot_inputs(env, resources, description):
    """Static descriptions and numeric planners; never pass the native manager."""
    parameters = resources['public_robot_parameters']
    robots = [description(arm, name, tuple(parameters['gripper_scale']),
        int(parameters['sign']), tuple(parameters['mimic'][1:]))
        for arm, name in resources['planner_robot_names'].items()]
    for declared in robots:
        actual = next(robot for robot in env.robot_manager.robot_list
                      if robot.type == 'target' and robot.arm_name == declared.arm_name)
        if (actual.robot_name != declared.robot_name or actual.ee_type != 'gripper'
                or tuple(actual.gripper_scale) != declared.gripper_scale
                or int(actual.gripper_move['sign']) != declared.gripper_sign
                or tuple(actual.gripper_move['mimic'][1:]) != declared.gripper_mimic):
            raise ValueError('Native robot differs from the declared public robot')
    planners = {robot.robot_name: env.robot_manager.planner[robot.robot_name] for robot in robots}
    limits = {arm: np.asarray(resources['public_joint_limits'], dtype=np.float32)
              for arm in resources['planner_robot_names']}
    return robots, planners, limits


class ActionExchange:
    """One normalized action, one official step, one paired public capture."""

    interval = 10

    def __init__(self, env, recorder, *, time_scale=1.4):
        if not np.isfinite(time_scale) or time_scale <= 0.:
            raise ValueError('Invalid declared motion time scale')
        self.time_scale = float(time_scale)
        self.env, self.recorder = env, recorder
        self.phase = lambda: 'public_rule_action'
        self.normalized_actions, self.action_sim_steps, self.action_status = [], [], []

    def __call__(self, action):
        if self.env.end_flag[0]:
            raise RuntimeError('A terminal episode cannot receive another action')
        expected = {side + '_' + name for side in SIDES for name in ('arm_joint_state', 'ee_joint_state')}
        if set(action) != expected:
            raise ValueError('Expected a complete normalized fourteen-dimensional action')
        for key, value in action.items():
            value = np.asarray(value, dtype=float)
            if value.shape != ((1,) if '_ee_' in key else (6,)) or not np.isfinite(value).all():
                raise ValueError('Invalid public action: ' + key)
            if '_ee_' in key and not 0. <= value[0] <= 1.:
                raise ValueError('Invalid normalized jaw command')
        previous = int(self.env.take_action_cnt[0])
        self.env.take_action(deepcopy(action))
        count = int(self.env.take_action_cnt[0])
        if count != previous + 1:
            raise RuntimeError('Official take_action must advance exactly one control')
        phase = str(self.phase())
        self.recorder._sim_step = count * self.interval
        self.normalized_actions.append(deepcopy(action))
        self.action_sim_steps.append(count * self.interval)
        self.action_status.append({'phase': phase, 'action_count': count})
        self.recorder.record(command=action, phase=phase)
        return project(self.recorder.public.last_observation), bool(self.env.end_flag[0])

    def save_actions(self, directory, save_npz=False):
        directory = Path(directory)
        arrays = {key: np.asarray([row[key] for row in self.normalized_actions], dtype=np.float32)
                  for key in (side + '_' + name for side in SIDES
                              for name in ('arm_joint_state', 'ee_joint_state'))}
        arrays['sim_step'] = np.asarray(self.action_sim_steps, dtype=np.int64)
        (directory / 'official_actions.json').write_text(json.dumps(jsonable({
            'action_count': len(self.normalized_actions), 'status': self.action_status,
            'frequency_hz': 25., 'motion_time_scale': self.time_scale,
            'gripper_encoding': 'normalized_0_1', 'physical_width_or_contact': None,
            'target_storage': 'trajectory.hdf5'}), indent=2) + '\n')
        upgrade_official_action_format(directory, arrays)


def run_actor(env, recorder, resources, exchange, args, actors, *, context, public):
    robots, planners, limits = robot_inputs(env, resources, public.PublicRobotDescription)
    def step(action):
        frame, ended = exchange(action)
        return public.PublicStepReply(frame, ended)
    actor = context(project(recorder.public.last_observation), robots, planners, limits, step,
        assets_dir=resources['public_toast_assets']['path'], output_root=recorder.output_root)
    actors.append(actor)
    exchange.phase = lambda: actor.phase
    try:
        try:
            actor.run()
            while not env.end_flag[0]:
                actor.phase = 'official_final_hold'
                actor._official_bridge.execute([{}] * 10, actor.phase)
        except getattr(public, 'PublicRobotFeedbackUnavailable', ()) as error:
            # Preserve the STD donor's finite-command continuation. No invalid
            # observation is replaced or sent back into the actor.
            if not exchange.normalized_actions:
                raise
            actor.phase = 'public_feedback_refusal_native_hold'
            held = deepcopy(exchange.normalized_actions[-1])
            actor.events.append({'kind': 'public_feedback_unavailable', 'field': error.key,
                'measured_contact': 'unknown', 'observations_replaced': False})
            while not env.end_flag[0]:
                exchange(held)
    except public.NativeEpisodeEnded:
        if not env.end_flag[0]:
            raise
