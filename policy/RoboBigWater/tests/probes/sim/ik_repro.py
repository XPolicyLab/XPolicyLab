"""Reproduce the IK failures of recorded episodes offline. Usage: ROBOSHELL_ROOT=... ROBODOJO_REPO=... python ik_repro.py EPISODE_DIR..."""
import glob, json, os, sys, numpy as np
sys.path.insert(0, os.environ["ROBOSHELL_ROOT"])
from env.planner_manager.curobo_planner import CuroboPlanner
from roboshell.server import geometry as geo, motion

ROOT = os.environ["ROBODOJO_REPO"] + "/Assets/Robots/x5"
names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
planner = CuroboPlanner(robot_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], active_joints_name=names, all_joints=names, dt=0.004, yml_path=f"{ROOT}/curobo.yml", table_height=0.74 - 0.765)
ORIGINS = {"left": [-0.3, -0.45, 0.765, 0.707, 0, 0, 0.707], "right": [0.3, -0.45, 0.765, 0.707, 0, 0, 0.707]}
TCP = np.eye(4); TCP[:3, 3] = [0.145, 0, 0]


class Robot:
    def __init__(self, tag): self.entity_origin_pose = ORIGINS[tag]


def fk_world(q, tag):
    kin = planner.motion_planner.compute_kinematics(planner._build_joint_state(np.asarray(q, dtype=np.float32)))
    pose = kin.tool_poses.get_link_pose(planner.ee_link)
    p = np.asarray(pose.position.detach().cpu()).reshape(-1)[:3]; quat = np.asarray(pose.quaternion.detach().cpu()).reshape(-1)[:4]
    return geo.pose_to_matrix(ORIGINS[tag]) @ geo.pose_to_matrix(list(p) + list(quat))


stats = {"cases": 0, "fail_1": 0, "fail_retry": 0}
for episode in sys.argv[1:]:
    targets = np.load(os.path.join(episode, "targets.npz"))["targets"]
    arms = [str(a) for a in np.load(os.path.join(episode, "targets.npz"))["arms"]]
    for line in open(os.path.join(episode, "commands.jsonl")):
        c = json.loads(line); req, fb = c["request"], c["feedback"]
        if req.get("cmd") != "move" or fb.get("plan_ok") is not False:
            continue
        step = c["steps"][0]
        if step == 0 or step > len(targets):
            continue
        tag = req["arm"]; col = arms.index(tag) * 7
        q = targets[step - 1, col:col + 6]
        ee = fk_world(q, tag)
        tcp = ee @ TCP
        target_tcp = tcp.copy(); target_tcp[:3, 3] += [req.get("dx", 0), req.get("dy", 0), req.get("dz", 0)]
        target_ee = target_tcp @ np.linalg.inv(TCP)
        stats["cases"] += 1
        outcome = []
        for attempts in (1, 4):
            motion.IK_ATTEMPTS = attempts
            try:
                motion.plan_line(planner, Robot(tag), q, ee, target_ee, None); outcome.append("ok")
            except motion.PlanFailure as f:
                outcome.append("FAIL " + (f.detail or ""))
                stats["fail_1" if attempts == 1 else "fail_retry"] += 1
        j6 = q[5]
        print(f"{os.path.basename(episode)[:26]} {tag} d=({req.get('dx',0):+.2f},{req.get('dy',0):+.2f},{req.get('dz',0):+.2f}) tcp z={tcp[2,3]:.2f} joint6={j6:+.2f} recorded='{(fb.get('plan_detail') or '')[:45]}' | 1 try: {outcome[0][:60]} | 4 tries: {outcome[1][:60]}")
print(stats)
