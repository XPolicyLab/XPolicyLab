"""Which IK constraint blocks the recorded failures: orientation tolerance, self-collision, or reach?
Usage: ROBOSHELL_ROOT=... ROBODOJO_REPO=... python ik_constraints.py EPISODE_DIR..."""
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


def make_solver(orientation_tolerance=0.02, self_collision=True, position_tolerance=0.001):
    cfg = InverseKinematicsCfg.create(robot=planner.robot_cfg, scene_model=planner.scene_model, device_cfg=planner.device_cfg,
                                      num_seeds=32, seed_solver_num_seeds=32, position_tolerance=position_tolerance,
                                      orientation_tolerance=orientation_tolerance, self_collision_check=self_collision,
                                      use_cuda_graph=False, max_batch_size=1, multi_env=False, max_goalset=1)
    return InverseKinematics(cfg)


solvers = {
    "default": planner.ik_solver,
    "orient 6deg": make_solver(orientation_tolerance=0.1),
    "orient 20deg": make_solver(orientation_tolerance=0.35),
    "no self-collision": make_solver(self_collision=False),
    "both relaxed": make_solver(orientation_tolerance=0.35, self_collision=False),
}


class Robot:
    def __init__(self, tag): self.entity_origin_pose = ORIGINS[tag]


def fk_world(q, tag):
    kin = planner.motion_planner.compute_kinematics(planner._build_joint_state(np.asarray(q, dtype=np.float32)))
    pose = kin.tool_poses.get_link_pose(planner.ee_link)
    p = np.asarray(pose.position.detach().cpu()).reshape(-1)[:3]; quat = np.asarray(pose.quaternion.detach().cpu()).reshape(-1)[:4]
    return geo.pose_to_matrix(ORIGINS[tag]) @ geo.pose_to_matrix(list(p) + list(quat))


motion.IK_ATTEMPTS = 1
counts = {k: 0 for k in solvers}; cases = 0
for episode in sys.argv[1:]:
    data = np.load(os.path.join(episode, "targets.npz")); targets = data["targets"]; arms = [str(a) for a in data["arms"]]
    for line in open(os.path.join(episode, "commands.jsonl")):
        c = json.loads(line); req, fb = c["request"], c["feedback"]
        if req.get("cmd") != "move" or fb.get("plan_ok") is not False or c["steps"][0] == 0:
            continue
        tag = req["arm"]; col = arms.index(tag) * 7; q = targets[c["steps"][0] - 1, col:col + 6]
        ee = fk_world(q, tag); target = (ee @ TCP); target[:3, 3] += [req.get("dx", 0), req.get("dy", 0), req.get("dz", 0)]; target = target @ np.linalg.inv(TCP)
        cases += 1
        row = []
        for name, solver in solvers.items():
            planner.ik_solver = solver
            try:
                motion.plan_line(planner, Robot(tag), q, ee, target, None); counts[name] += 1; row.append(f"{name}: ok")
            except motion.PlanFailure as f:
                row.append(f"{name}: fail@{(f.detail or '').split(',')[0].replace('no solution at ', '')}")
        print(f"{tag} d=({req.get('dx',0):+.2f},{req.get('dy',0):+.2f},{req.get('dz',0):+.2f}) | " + " | ".join(row))
planner.ik_solver = solvers["default"]
print("cases", cases, "solved:", counts)
