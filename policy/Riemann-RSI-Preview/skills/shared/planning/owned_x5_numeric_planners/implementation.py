"""Owned public X5 planners; receives static resources, never a native environment."""
from rsi.rule_configuration import official_root as _rule_official_root, python_prefix as _rule_python_prefix, settings as _rule_settings
import hashlib,importlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from skills.shared.motion.normalized_pair_joint_execution.implementation import PublicRobotDescription
OFFICIAL_ROOT=_rule_official_root()
OFFICIAL_COMMIT=_rule_settings()['official_commit']
OFFICIAL_PREFIX=_rule_python_prefix()
def verify_static_resources(resources):

    def pinned(path, expected):
        path = Path(path)
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned public resource changed: ' + str(path))
    official = resources['official_runtime']
    if Path(official['path']) != OFFICIAL_ROOT or official['commit'] != OFFICIAL_COMMIT:
        raise ValueError('Only the exact official public runtime is declared')
    actual = subprocess.check_output(['git', '-c', 'safe.directory=' + str(OFFICIAL_ROOT), '-C', str(OFFICIAL_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != OFFICIAL_COMMIT:
        raise ValueError('Official source changed')
    for member, digest in official['files'].items():
        pinned(OFFICIAL_ROOT / member, digest)
    for name in ('public_robot_urdf', 'public_robot_usd', 'public_robot_config', 'public_planner_robot_config'):
        record = resources[name]
        pinned(record['path'], record['sha256'])
    robot_root = Path(resources['public_robot_urdf']['path']).parent
    for member, digest in resources['public_robot_meshes'].items():
        pinned(robot_root / member, digest)
    for member, digest in resources['public_robot_usd_layers']['files'].items():
        pinned(Path(resources['public_robot_usd_layers']['path']) / member, digest)
    for member, digest in resources['public_camera']['files'].items():
        pinned(Path(resources['public_camera']['path']) / member, digest)
    return resources

def create(resources, output_root):
    """Only public inputs enter; planners are fresh objects from pinned bytes."""
    if Path(sys.prefix) != OFFICIAL_PREFIX:
        raise ValueError('Owned native planners require the configured official evaluator Python')
    verify_static_resources(resources)
    import yaml
    import torch
    from curobo._src.geom.types import Cuboid, SceneCfg
    module = importlib.import_module('env.planner_manager.curobo_planner')
    if Path(module.__file__).resolve() != OFFICIAL_ROOT / 'env/planner_manager/curobo_planner.py':
        raise ValueError('Planner imported from an undeclared source')
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    configuration = yaml.safe_load(Path(resources['public_planner_robot_config']['path']).read_text())
    kinematics = configuration['robot_cfg']['kinematics']
    kinematics['asset_root_path'] = str(Path(resources['public_robot_urdf']['path']).parent)
    kinematics['urdf_path'] = resources['public_robot_urdf']['path']
    filename = output_root / 'owned-x5-public-planner.yml'
    with filename.open('x') as stream:
        yaml.safe_dump(configuration, stream)
    parameters = resources['public_robot_parameters']
    arm_joints = list(parameters['arm_joint_names'])
    all_joints = arm_joints + list(parameters['finger_joint_names'])
    descriptions = tuple((PublicRobotDescription(arm, name, tuple(parameters['gripper_scale']), int(parameters['sign']), tuple(parameters['mimic'][1:])) for arm, name in resources['planner_robot_names'].items()))
    planners = {}
    for name in sorted({robot.robot_name for robot in descriptions}):
        planner = module.CuroboPlanner(robot_origin_pose=np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), active_joints_name=arm_joints, all_joints=all_joints, dt=0.004, yml_path=str(filename), table_height=0.0)
        if planner.device_cfg.device != torch.device('cuda:0'):
            raise ValueError('Parent must bind an owned cuda:0 lease before constructing planners')
        for solver_name in ('motion_planner', 'motion_planner_batch', 'ik_solver'):
            scene = SceneCfg(cuboid=[Cuboid(name='table', pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0], dims=[3.0, 3.0, 0.05])])
            getattr(planner, solver_name).update_world(scene)
        planners[name] = planner
    limits = {robot.arm_name: resources['public_joint_limits'] for robot in descriptions}
    return descriptions, planners, limits
