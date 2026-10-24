"""Kinematics of RoboDojo's ARX X5 arms: FK, IK and planned paths through RoboDojo's own
cuRobo planner (env/planner_manager/curobo_planner.py), in the env frame.

The runtime worker calls these with the simulator's planner; the kinematics service
(services/servers/serve_kinematics.py) with a planner of its own, built the way
RoboDojo builds it, for a policy that runs without the simulator (the XPolicyLab
connector). Same code, same planner configuration, so both answer alike.

Importing this module does not import cuRobo, torch or RoboDojo.
"""

import math


def forward(planner_of, groups, command):
    """{arm: link6 pose [x, y, z, qw, qx, qy, qz]} for the joints in `command` (14 values).
    groups: [{"name", "joints", "base"}] with base the arm's mount pose; planner_of(group)
    is that arm's planner."""
    import numpy as np
    from utils.transformer import matrix_to_pose, pose_to_matrix

    poses = {}
    for group in groups:
        planner = planner_of(group)
        joints = [float(command[i]) for i in group["joints"]]
        kinematics = planner.motion_planner.compute_kinematics(planner._build_joint_state(joints))
        tool = kinematics.tool_poses.get_link_pose(planner.ee_link)
        position = tool.position.detach().reshape(-1)[:3].cpu().numpy() - np.asarray(planner.frame_bias, dtype=float)
        quaternion = tool.quaternion.detach().reshape(-1)[:4].cpu().numpy()
        base = pose_to_matrix(list(group["base"]))
        poses[group["name"]] = [float(v) for v in matrix_to_pose(base @ pose_to_matrix([*position, *quaternion]))]
    return poses


def inverse(planner, group, current, pose):
    """6 joints placing link6 at `pose` starting from the arm's `current` joints, or None."""
    result = planner.solve_ik_to_joint(list(current), list(pose), real_robot_pose=list(group["base"]))
    if result.get("status") != "Success":
        return {"success": False, "joints": None}
    return {"success": True, "joints": [float(v) for v in result["joint_value"]]}


def plan(planner, group, current, interval, pose=None, joints=None, raw=False):
    """Waypoints at the policy rate (every `interval` physics steps) from `current` to a
    link6 pose or to joints. raw returns the planner's physics-rate path."""
    import numpy as np

    if (pose is None) == (joints is None):
        raise ValueError("plan needs exactly one of pose or joints")
    if pose is not None:
        result = planner.plan_path(list(current), list(pose), real_robot_pose=list(group["base"]))
    else:
        result = planner.plan_joint(list(current), list(joints))
    if raw:
        position = result.get("position")
        return {"status": str(result.get("status")),
                "position": None if position is None else np.asarray(position, dtype=float).tolist()}
    if result.get("status") != "Success":
        return {"success": False, "waypoints": []}
    path = np.asarray(result["position"], dtype=float).reshape(-1, len(group["joints"]))
    # The planner returns physics-rate samples; ticks run at the policy rate.
    indices = list(range(int(interval) - 1, len(path), int(interval)))
    if not indices or indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)
    return {"success": True, "waypoints": path[indices].tolist()}


def standalone_planner(robodojo_root, robot="x5", base_z=0.765, dt=0.004):
    """RoboDojo's CuroboPlanner for one arm model, built as its robot manager builds it
    (robot_manager._setup_planner), without the simulator. Needs RoboDojo on sys.path."""
    import os
    import sys

    import yaml

    root = os.fspath(robodojo_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from env.planner_manager.curobo_planner import CuroboPlanner

    robot_dir = os.path.join(root, "Assets", "Robots", robot)
    with open(os.path.join(robot_dir, "robot_config.yml")) as handle:
        config = yaml.safe_load(handle)
    joints = _arm_joints(config)
    return CuroboPlanner(robot_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], active_joints_name=joints,
                         all_joints=joints, dt=dt, yml_path=os.path.join(robot_dir, "curobo.yml"),
                         table_height=0.74 - base_z)


def _arm_joints(config):
    """The arm joint names in a RoboDojo robot_config.yml (robot_config.arm_joints_name)."""
    stack = [config]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "arm_joints_name" in node:
                return list(node["arm_joints_name"])
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    raise ValueError("robot_config.yml has no arm_joints_name")


def fk_error(planner_of, groups, command, measured):
    """Largest distance between FK of `command` and the measured link6 positions."""
    predicted = forward(planner_of, groups, command)
    return max(math.dist(predicted[group["name"]][:3], measured[group["name"]][:3]) for group in groups)
