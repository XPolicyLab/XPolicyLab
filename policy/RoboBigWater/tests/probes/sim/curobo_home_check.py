"""From the home configuration (all joints zero): are IK and the planner consistent with forward kinematics?"""
import os
import numpy as np, torch, time
import transforms3d as t3d
from env.planner_manager.curobo_planner import CuroboPlanner

ROOT = os.environ["ROBODOJO_REPO"] + "/Assets/Robots/x5"
names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
origin = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
planner = CuroboPlanner(robot_origin_pose=origin, active_joints_name=names, all_joints=names, dt=0.004,
                        yml_path=f"{ROOT}/curobo.yml", table_height=0.74 - 0.765)
print("cspace", planner.cspace_joint_names, "tool", planner.tool_frames, "ee", planner.ee_link)


def fk(q):
    kin = planner.motion_planner.compute_kinematics(planner._build_joint_state(np.asarray(q, dtype=np.float32)))
    pose = kin.tool_poses.get_link_pose(planner.ee_link)
    return np.asarray(pose.position.detach().cpu()).reshape(-1)[:3], np.asarray(pose.quaternion.detach().cpu()).reshape(-1)[:4]


def show(tag, q, target):
    p, quat = fk(q)
    ang = 2 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(quat, target[3:]))))))
    print(f"  {tag}: q={np.round(q,3)} fk_pos={np.round(p,4)} pos_err={np.linalg.norm(p-np.asarray(target[:3])):.4f} m rot_err={ang:.2f} deg")


for label, q0 in (("home", np.zeros(6)), ("bent", np.array([0.0, 0.9, 0.9, -0.3, 0.0, 0.0]))):
    p0, quat0 = fk(q0)
    print(f"== start {label}: fk pos {np.round(p0,4)} quat {np.round(quat0,4)}")
    for name, delta in (("down 5 cm", [0, 0, -0.05]), ("forward 10 cm", [0.10, 0, 0]), ("up 5 cm", [0, 0, 0.05])):
        target = list(p0 + np.asarray(delta)) + list(quat0)
        ik = planner.solve_ik_to_joint(np.asarray(q0, dtype=np.float32), target, real_robot_pose=origin)
        print(f" {name}: ik {ik['status']}")
        if ik["status"] == "Success":
            show("ik", ik["joint_value"], target)
        plan = planner.plan_path(np.asarray(q0, dtype=np.float32), target, real_robot_pose=origin)
        print(f" {name}: plan {plan['status']}", None if plan["status"] != "Success" else plan["position"].shape)
        if plan["status"] == "Success":
            show("plan first", plan["position"][0], target)
            show("plan last", plan["position"][-1], target)
print("joint limits", planner.motion_planner.kinematics.get_joint_limits().position if hasattr(planner.motion_planner, "kinematics") else "n/a")
