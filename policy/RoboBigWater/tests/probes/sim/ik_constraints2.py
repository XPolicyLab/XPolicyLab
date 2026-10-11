"""Second diagnostic: is the current pose itself solvable, does a looser position tolerance help, and how far from the base are the targets?"""
import json, os, sys, numpy as np
sys.path.insert(0, os.environ["ROBOSHELL_ROOT"])
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from env.planner_manager.curobo_planner import CuroboPlanner
from roboshell.server import geometry as geo, motion

ROOT = os.environ["ROBODOJO_REPO"] + "/Assets/Robots/x5"
names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
planner = CuroboPlanner(robot_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], active_joints_name=names, all_joints=names, dt=0.004, yml_path=f"{ROOT}/curobo.yml", table_height=0.74 - 0.765)
ORIGINS = {"left": [-0.3, -0.45, 0.765, 0.707, 0, 0, 0.707], "right": [0.3, -0.45, 0.765, 0.707, 0, 0, 0.707]}
TCP = np.eye(4); TCP[:3, 3] = [0.145, 0, 0]
loose = InverseKinematics(InverseKinematicsCfg.create(robot=planner.robot_cfg, scene_model=planner.scene_model, device_cfg=planner.device_cfg,
    num_seeds=64, seed_solver_num_seeds=64, position_tolerance=0.01, orientation_tolerance=0.2, self_collision_check=False, use_cuda_graph=False, max_batch_size=1, multi_env=False, max_goalset=1))
default = planner.ik_solver


class Robot:
    def __init__(self, tag): self.entity_origin_pose = ORIGINS[tag]


def fk_world(q, tag):
    kin = planner.motion_planner.compute_kinematics(planner._build_joint_state(np.asarray(q, dtype=np.float32)))
    pose = kin.tool_poses.get_link_pose(planner.ee_link)
    p = np.asarray(pose.position.detach().cpu()).reshape(-1)[:3]; quat = np.asarray(pose.quaternion.detach().cpu()).reshape(-1)[:4]
    return geo.pose_to_matrix(ORIGINS[tag]) @ geo.pose_to_matrix(list(p) + list(quat))


def ik(pose, seed, tag, solver):
    planner.ik_solver = solver
    r = planner.solve_ik_to_joint(np.asarray(seed, dtype=np.float32), geo.matrix_to_pose(pose), real_robot_pose=ORIGINS[tag])
    planner.ik_solver = default
    return r["status"] == "Success"


n = 0
for episode in sys.argv[1:]:
    data = np.load(os.path.join(episode, "targets.npz")); targets = data["targets"]; arms = [str(a) for a in data["arms"]]
    for line in open(os.path.join(episode, "commands.jsonl")):
        c = json.loads(line); req, fb = c["request"], c["feedback"]
        if req.get("cmd") != "move" or fb.get("plan_ok") is not False or c["steps"][0] == 0:
            continue
        tag = req["arm"]; col = arms.index(tag) * 7; q = targets[c["steps"][0] - 1, col:col + 6]
        ee = fk_world(q, tag); tcp = ee @ TCP
        d = np.array([req.get("dx", 0), req.get("dy", 0), req.get("dz", 0)])
        target = tcp.copy(); target[:3, 3] += d; target_ee = target @ np.linalg.inv(TCP)
        base = np.array(ORIGINS[tag][:3])
        reach_now = np.linalg.norm(ee[:3, 3] - base); reach_target = np.linalg.norm(target_ee[:3, 3] - base)
        approach = tcp[:3, 0]
        row = [f"{tag} d=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})", f"link6 reach now {reach_now:.2f} m -> {reach_target:.2f} m", f"approach {np.round(approach,2)}",
               f"cur pose IK default {'ok' if ik(ee, q, tag, default) else 'FAIL'}", f"target IK default {'ok' if ik(target_ee, q, tag, default) else 'FAIL'}",
               f"target IK loose {'ok' if ik(target_ee, q, tag, loose) else 'FAIL'}", f"joints {np.round(q, 2)}"]
        print(" | ".join(row)); n += 1
        if n >= 14: break
    if n >= 14: break
