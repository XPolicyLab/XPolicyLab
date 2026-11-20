"""Fresh numerical tools built from pinned, public robot files only."""
from rsi.rule_configuration import official_root as _rule_official_root
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
from rsi.resource_locations import load_manifest as load_resource_manifest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def declarations(resources=None):
    data = load_resource_manifest(ROOT / 'configs/public-x5-numeric-tools.json')
    if resources is not None:
        # A task may declare its own (float32 USD or float64 URDF) limits.
        # Neither source is read from an articulation.
        for name in ('public_robot_urdf', 'public_robot_config', 'public_planner_robot_config',
                     'public_robot_parameters', 'planner_robot_names', 'public_joint_limits'):
            if name in resources:
                data[name] = resources[name]
    return data


def create(resources, output_root, *, arm_joints_only=False):
    """Return static descriptors and cuRobo objects; no simulator is imported."""
    import yaml
    import torch
    import env.planner_manager.curobo_planner as module
    from curobo._src.geom.types import Cuboid, SceneCfg
    if Path(module.__file__).resolve() != (_rule_official_root() / 'env/planner_manager/curobo_planner.py'):
        raise ValueError('Unexpected numerical planner implementation')
    data = declarations(resources)
    for name in ('public_robot_urdf', 'public_robot_config', 'public_planner_robot_config'):
        item = data[name]
        path = Path(item['path'])
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Pinned public numerical resource changed: ' + name)
    robot_root = Path(data['public_robot_urdf']['path']).parent
    for member, expected in data['public_robot_meshes'].items():
        if hashlib.sha256((robot_root / member).read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned public robot mesh changed: ' + member)
    config = yaml.safe_load(Path(data['public_planner_robot_config']['path']).read_text())
    kinematics = config['robot_cfg']['kinematics']
    kinematics['asset_root_path'] = str(robot_root)
    kinematics['urdf_path'] = data['public_robot_urdf']['path']
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    filename = root / 'declared-x5.yml'
    with filename.open('x') as stream:
        yaml.safe_dump(config, stream)
    parameters = data['public_robot_parameters']
    # Older task declarations omit joint names. Read the pinned public YAML,
    # never a native manager or live articulation.
    robot_config = yaml.safe_load(Path(data['public_robot_config']['path']).read_text())
    arm = list(parameters.get('arm_joint_names', robot_config['arm_joints_name']))
    fingers = list(parameters.get('finger_joint_names', robot_config['gripper_joints_name']))
    robots = tuple(SimpleNamespace(type='target', ee_type='gripper', arm_name=side,
        robot_name=name, gripper_name=side.replace('arm', 'ee'),
        gripper_scale=tuple(parameters['gripper_scale']),
        gripper_move={'sign': parameters['sign'], 'mimic': tuple(parameters['mimic'])},
        arm_joints_name=arm, arm_joint_indices=tuple(range(6)),
        urdf_path=data['public_robot_urdf']['path'])
        for side, name in data['planner_robot_names'].items())
    planners = {}
    for name in sorted({robot.robot_name for robot in robots}):
        planner = module.CuroboPlanner(robot_origin_pose=np.array([0., 0., 0., 1., 0., 0., 0.]),
            active_joints_name=arm, all_joints=arm if arm_joints_only else arm + fingers,
            dt=.004, yml_path=str(filename), table_height=-.025 if arm_joints_only else 0.)
        if planner.device_cfg.device != torch.device('cuda:0'):
            raise ValueError('Numeric tools must stay on the leased policy device')
        if not arm_joints_only:
            for field in ('motion_planner', 'motion_planner_batch', 'ik_solver'):
                getattr(planner, field).update_world(SceneCfg(cuboid=[Cuboid(name='table',
                    pose=[0., 0., -.05, 1., 0., 0., 0.], dims=[3., 3., .05])]))
        planners[name] = planner
    return SimpleNamespace(robot_list=robots, planner=planners,
        ik_solver=dict(planners), process_name=lambda name: name if name.endswith('_state') else name + '_joint_state')
