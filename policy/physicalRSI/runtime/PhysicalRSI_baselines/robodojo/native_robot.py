"""Reset-time robot frame diagnostics, separate from policy observations."""

import numpy as np

from .native_camera import array
from .planner_bridge import quaternion, to_base


def capture(env, indices):
    manager = env.robot_manager
    robots = [r for r in manager.robot_list if r.type == "target"]
    if len(robots) != 2 or {r.arm_name for r in robots} != {"left_arm", "right_arm"}:
        raise ValueError("Robot diagnostics require exactly two named target arms")
    origins = array(manager.scene.env_origins)
    if origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("Invalid native environment origins")
    rows = []
    for index in indices:
        for robot in robots:
            poses = {}
            for label, link in (("base", robot.base_link), ("ee", robot.ee_link_name)):
                pair = {}
                for frame, relative in (("world", False), ("environment", True)):
                    pose = array(
                        manager.get_link_pose(
                            robot,
                            link_name=link,
                            env_idx_list=[index],
                            is_relative=relative,
                        )[index]
                    )
                    if pose.shape != (7,):
                        raise ValueError("Invalid native link pose")
                    quaternion(pose[3:].tolist())
                    pair[frame] = pose
                if not np.allclose(
                    pair["world"][:3] - origins[index],
                    pair["environment"][:3],
                    atol=1e-5,
                    rtol=0,
                ):
                    raise ValueError("Native world/environment translation mismatch")
                # Quaternions q and -q represent the same orientation.
                if abs(np.dot(pair["world"][3:], pair["environment"][3:])) < 1 - 1e-3:
                    raise ValueError("Native world/environment rotation mismatch")
                poses[label] = {key: value.tolist() for key, value in pair.items()}
            arm = array(manager.get_joint(robot, env_idx_list=[index])[index])
            fingers = array(
                manager.get_end_effector_real_val(robot, env_idx_list=[index])[index]
            )
            arm_names, finger_names = (
                list(robot.arm_joints_name),
                list(robot.gripper_joints_name),
            )
            names = arm_names + finger_names
            if (
                not arm_names
                or not finger_names
                or len(set(names)) != len(names)
                or arm.shape != (len(arm_names),)
                or fingers.shape != (len(finger_names),)
            ):
                raise ValueError("Native joint names and values disagree")
            position, orientation = to_base(
                poses["base"]["world"],
                poses["ee"]["world"][:3],
                poses["ee"]["world"][3:],
            )
            rows.append(
                dict(
                    env_idx=index,
                    arm=robot.arm_name,
                    robot=robot.robot_name,
                    environment_origin=origins[index].tolist(),
                    base_link=robot.base_link,
                    ee_link=robot.ee_link_name,
                    poses=poses,
                    base_ee_pose=[*position, *orientation],
                    arm_joints=dict(zip(arm_names, arm.tolist())),
                    raw_gripper_joints=dict(zip(finger_names, fingers.tolist())),
                )
            )
    return dict(
        schema="physicalrsi.robodojo.robot-diagnostics/v1",
        robots=rows,
        scope="reset-time native link frames and raw joints; no planner FK or action qualification",
        physical_qualification=False,
    )
