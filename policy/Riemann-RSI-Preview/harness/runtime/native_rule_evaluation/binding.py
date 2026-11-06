"""Evaluator-owned resource verification and restricted public tool binding."""
from rsi.resource_locations import load_manifest as load_resource_manifest
from functools import partial
import hashlib
import importlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import numpy as np


def load_public_interfaces(policy_package):
    robot = importlib.import_module(policy_package + '.skills.public_robot.implementation')
    motion_math = importlib.import_module(policy_package + '.skills.motion_math.implementation')
    for name in ('RobotConfiguration', 'PublicPlanner', 'PublicRobotManager', 'PublicEnvironment', 'ActionSubmission', 'PublicRecorder'):
        if not hasattr(robot, name):
            raise ValueError('Declared public robot capability is missing: ' + name)
    return SimpleNamespace(robot=robot, as_numpy=motion_math.as_numpy)


def verify_resources(root, *, require_public_cube_templates=False):
    resources = load_resource_manifest(Path(root) / 'resources.json')
    required = ('public_robot_urdf', 'public_robot_usd', 'public_robot_config', 'public_robot_meshes')
    if require_public_cube_templates:
        required += ('public_cube_templates',)
    for name in required:
        if name not in resources:
            raise ValueError('Pinned public resource declaration is missing: ' + name)
    for name, item in resources.items():
        if not isinstance(item, dict) or 'path' not in item:
            continue
        path = Path(item['path'])
        if 'sha256' in item:
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Pinned public resource changed: ' + name)
        if 'files' in item:
            for filename, expected in item['files'].items():
                if hashlib.sha256((path / filename).read_bytes()).hexdigest() != expected:
                    raise ValueError('Pinned public resource file changed: ' + name + '/' + filename)
        if 'sha256' not in item and 'files' not in item:
            raise ValueError('Pinned public resource has no byte declaration: ' + name)
    for name, expected in resources['public_robot_meshes'].items():
        path = Path(resources['public_robot_urdf']['path']).resolve().parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned public robot visual changed: ' + name)
    return resources


def robot_configurations(resources, public_robot):
    configuration = resources['public_robot_parameters']
    limits = tuple(tuple(item) for item in resources['public_joint_limits'])
    return {arm: public_robot.RobotConfiguration(arm_name=arm, robot_name=name, gripper_name=arm.replace('arm', 'ee'), gripper_scale=tuple(configuration['gripper_scale']), gripper_move=MappingProxyType({'sign': configuration['sign'], 'mimic': tuple(configuration['mimic'])}), arm_joint_indices=tuple(range(6)), joint_limits=limits) for arm, name in resources['planner_robot_names'].items()}



def fk(planner, joints, as_numpy):
    state = planner._build_joint_state(np.asarray(joints, dtype=np.float32))
    kinematics = planner.motion_planner.compute_kinematics(state)
    return (as_numpy(kinematics.tool_poses.position).reshape(-1, 3)[0], as_numpy(kinematics.tool_poses.quaternion).reshape(-1, 4)[0])


def action_submission(bridge, public_robot):
    return public_robot.ActionSubmission(bridge.execute, time_scale=bridge.time_scale,
                                         get_time_scale=lambda: bridge.time_scale,
                                         set_time_scale=lambda value: setattr(bridge, 'time_scale', float(value)))


def public_environment(env, recorder, resources, public_robot, as_numpy):
    configurations = robot_configurations(resources, public_robot)
    for robot in env.robot_manager.robot_list:
        if robot.type != 'target':
            continue
        declared = configurations[robot.arm_name]
        if robot.robot_name != declared.robot_name or tuple(robot.gripper_scale) != declared.gripper_scale:
            raise ValueError('Native robot differs from the pinned public configuration')
        articulation = env.robot_manager.robot_key[env.robot_manager.robot_list.index(robot)]
        actual_limits = as_numpy(articulation.data.joint_pos_limits[0])[np.asarray(robot.arm_joint_indices, dtype=int)]
        if not np.array_equal(actual_limits, np.asarray(declared.joint_limits, dtype=np.float32)):
            raise ValueError('Native limits differ from the pinned public USD declaration')
    planners = {name: public_robot.PublicPlanner(planner.plan_path, planner.plan_joint, partial(fk, planner, as_numpy=as_numpy)) for name, planner in env.robot_manager.planner.items()}
    manager = public_robot.PublicRobotManager(configurations, planners, recorder.public)
    return public_robot.PublicEnvironment(manager, float(env.dt), int(env.step_lim))
