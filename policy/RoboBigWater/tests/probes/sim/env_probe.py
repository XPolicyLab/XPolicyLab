"""M1 probe: bring the environment up once and print the facts the server design depends on."""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--task", default="stack_bowls")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--layout", type=int, default=0)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
os.environ.setdefault("ROBODOJO_RUN_ID", time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(args.out, exist_ok=True)
os.chdir(args.out)

t0 = time.time()
app = AppLauncher(args).app
t_app = time.time() - t0

import numpy as np
import torch
import transforms3d as t3d

from roboshell.server.envsetup import build_env, start_episode

report = {"t_app_s": round(t_app, 1)}


def dump():
    with open(os.path.join(args.out, "probe.json"), "w") as handle:
        json.dump(report, handle, indent=1, default=lambda o: np.asarray(o).tolist())


import sys, traceback


def fail(kind, value, trace):
    traceback.print_exception(kind, value, trace)
    dump()
    print("PROBE-FAILED", flush=True)
    os._exit(1)


sys.excepthook = fail

t0 = time.time()
env = build_env(app, args.task, args.seed, device_id=args.device_id)
report["t_build_s"] = round(time.time() - t0, 1)
t0 = time.time()
start_episode(env, args.layout)
report["t_reset_s"] = round(time.time() - t0, 1)
report["step_lim"] = int(env.step_lim)
report["dt"] = float(env.dt)
report["collect_interval"] = int(env.obs_manager.collect_interval)
report["env_origin"] = env.robot_manager.scene.env_origins[0].cpu().numpy()
report["cwd"] = os.getcwd()
report["save_dir"] = env.save_dir
dump()

rm = env.robot_manager
robots = [r for r in rm.robot_list if r.type == "target"]
report["robots"] = []
for robot in robots:
    key = rm.robot_key[rm.robot_list.index(robot)]
    links = {}
    for name in key.body_names:
        links[name] = rm.get_link_pose(robot, name, env_idx_list=[0], is_relative=True)[0]
    joints = rm.get_joint(robot, env_idx_list=[0])[0]
    ee = rm.get_real_endpose(robot, env_idx_list=[0])[0]
    ik = rm.solve_ik(target_pose=list(ee), env_idx=0, robot=robot)
    entry = {
        "arm_name": robot.arm_name,
        "gripper_name": robot.gripper_name,
        "ee_link": robot.ee_link_name,
        "entity_origin_pose": robot.entity_origin_pose,
        "gripper_scale": robot.gripper_scale,
        "gripper_bias": robot.gripper_bias,
        "body_names": list(key.body_names),
        "joint_names": list(key.joint_names),
        "links": links,
        "arm_joints": joints,
        "gripper_joints": rm.get_end_effector_real_val(robot, env_idx_list=[0])[0],
        "ee_pose": ee,
        "ik_status": ik["status"],
        "ik_joint": ik.get("joint_value"),
        "ik_joint_err": None if ik["status"] != "Success" else float(np.abs(np.asarray(ik["joint_value"]) - joints).max()),
    }
    # express the finger links in the ee frame: where is the point between the fingers?
    R = t3d.quaternions.quat2mat(ee[3:])
    for name in ("link7", "link8"):
        if name in links:
            entry[f"{name}_in_ee"] = R.T @ (np.asarray(links[name][:3]) - np.asarray(ee[:3]))
    # IK for a 5 cm move straight down and a planned path to the same target
    target = list(ee)
    target[2] -= 0.05
    t0 = time.time()
    ik2 = rm.solve_ik(target_pose=target, env_idx=0, robot=robot)
    entry["ik_down5cm"] = {"status": ik2["status"], "t_s": round(time.time() - t0, 3),
                           "joint_delta": None if ik2["status"] != "Success" else np.asarray(ik2["joint_value"]) - joints}
    t0 = time.time()
    plan = rm.planner[robot.robot_name].plan_path(np.asarray(joints, dtype=np.float32), target, real_robot_pose=list(robot.entity_origin_pose))
    entry["plan_down5cm"] = {"status": plan["status"], "t_s": round(time.time() - t0, 3),
                             "points": None if plan["status"] != "Success" else int(plan["position"].shape[0])}
    report["robots"].append(entry)
dump()

t0 = time.time()
obs = env.get_obs()
report["t_obs_s"] = round(time.time() - t0, 3)
report["instruction"] = obs["instruction"]
report["state_keys"] = {k: np.asarray(v).shape for k, v in obs["state"].items()}
report["state"] = obs["state"]
report["vision"] = {}
import cv2

for name, cam in obs["vision"].items():
    color = np.asarray(cam["color"]) if "color" in cam else None
    report["vision"][name] = {
        "keys": list(cam.keys()),
        "shape": None if color is None else color.shape,
        "dtype": None if color is None else str(color.dtype),
        "intrinsic": cam.get("intrinsic_matrix"),
        "extrinsic": cam.get("extrinsic_matrix"),
    }
    if color is not None:
        cv2.imwrite(os.path.join(args.out, f"{name}.png"), cv2.cvtColor(color[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
dump()

# one official action step that holds the current pose: timing and step accounting
action = {}
for robot in robots:
    name = robot.arm_name
    action[f"{name}_joint_state"] = rm.get_joint(robot, env_idx_list=[0])[0].tolist()
    action[rm.process_name(robot.gripper_name)] = list(obs["state"][rm.process_name(robot.gripper_name)])
report["action_keys"] = {k: len(v) for k, v in action.items()}
t0 = time.time()
for _ in range(25):
    env.take_action(action)
report["t_25_actions_s"] = round(time.time() - t0, 3)
report["take_action_cnt"] = int(env.take_action_cnt[0])
report["end_flag"] = bool(env.end_flag[0])
report["success"] = bool(env.success[0])
report["reward"] = [float(x) for x in env.reward_manager.get_reward(final_check=False)]
if hasattr(env, "get_score"):
    report["score"] = [float(x) for x in env.reward_manager.get_score()]
report["gpu_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
report["objects"] = sorted(getattr(env.scene_manager, "object_dict", {}).keys()) if hasattr(env.scene_manager, "object_dict") else None
dump()
print("PROBE-DONE", flush=True)
os._exit(0)  # a normal shutdown can hang on a dialog that needs a display
