"""cuRobo without Isaac Sim: does the motion planner work on its own?

Builds the planner exactly as RoboDojo's RobotManager does for the X5 arm, then
runs inverse kinematics and motion planning to a pose 5 cm below the pose of
the default joint configuration.
"""
import os, sys, time, traceback
import numpy as np
import torch

print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), "| torch devices", torch.cuda.device_count(),
      "| current", torch.cuda.current_device(), flush=True)
from env.planner_manager.curobo_planner import CuroboPlanner

ROOT = os.environ["ROBODOJO_REPO"] + "/Assets/Robots/x5"
root_pose = [-0.3, -0.45, 0.765, 0.707, 0, 0, 0.707]
joints_name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
t0 = time.time()
planner = CuroboPlanner(robot_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], active_joints_name=joints_name,
                        all_joints=joints_name, dt=0.004, yml_path=f"{ROOT}/curobo.yml", table_height=0.74 - root_pose[2])
print(f"planner built in {time.time()-t0:.1f}s, device {planner.device_cfg.device}", flush=True)

if os.environ.get("SWITCH_DEVICE"):
    # what happens inside the Isaac Sim process: the simulator makes another GPU the current one
    torch.cuda.set_device(int(os.environ["SWITCH_DEVICE"]))
    _ = torch.zeros(8, device=f"cuda:{os.environ['SWITCH_DEVICE']}")
    print("current device switched to", torch.cuda.current_device(), flush=True)
q0 = np.array([0.0, 0.9, 0.9, -0.3, 0.0, 0.0], dtype=np.float32)
state = planner._build_joint_state(q0)
kin = planner.motion_planner.compute_kinematics(state)
pose = kin.tool_poses.get_link_pose(planner.ee_link) if hasattr(kin, "tool_poses") else None
pos = np.asarray(pose.position.detach().cpu()).reshape(-1)[:3]
quat = np.asarray(pose.quaternion.detach().cpu()).reshape(-1)[:4]
print("fk base frame", pos, quat, flush=True)
# planner works in the arm base frame when real_robot_pose equals robot_origin_pose
origin = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
target = list(pos) + list(quat)
target[2] -= 0.05
for name, call in (
    ("ik", lambda: planner.solve_ik_to_joint(q0, target, real_robot_pose=origin)),
    ("plan_path", lambda: planner.plan_path(q0, target, real_robot_pose=origin)),
    ("plan_path again", lambda: planner.plan_path(q0, target, real_robot_pose=origin)),
):
    t0 = time.time()
    try:
        r = call()
        extra = r["position"].shape if "position" in r else r.get("joint_value")
        print(f"{name}: {r['status']} in {time.time()-t0:.2f}s {extra}", flush=True)
    except Exception as e:
        print(f"{name}: EXCEPTION {type(e).__name__}: {str(e)[:150]}", flush=True)
        traceback.print_exc(limit=3)
        break
print("STANDALONE-DONE", flush=True)
