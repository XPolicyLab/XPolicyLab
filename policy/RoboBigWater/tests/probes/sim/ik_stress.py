"""How often does cuRobo IK fail on clearly reachable straight lines, and do retries fix it? No simulator needed."""
import os, sys, numpy as np
sys.path.insert(0, os.environ["ROBOSHELL_ROOT"])
from env.planner_manager.curobo_planner import CuroboPlanner
from roboshell.server import geometry as geo, motion

ROOT = os.environ["ROBODOJO_REPO"] + "/Assets/Robots/x5"
names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
origin = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
planner = CuroboPlanner(robot_origin_pose=origin, active_joints_name=names, all_joints=names, dt=0.004, yml_path=f"{ROOT}/curobo.yml", table_height=0.74 - 0.765)


class Robot:
    entity_origin_pose = origin


def fk(q):
    kin = planner.motion_planner.compute_kinematics(planner._build_joint_state(np.asarray(q, dtype=np.float32)))
    pose = kin.tool_poses.get_link_pose(planner.ee_link)
    p = np.asarray(pose.position.detach().cpu()).reshape(-1)[:3]; quat = np.asarray(pose.quaternion.detach().cpu()).reshape(-1)[:4]
    return geo.pose_to_matrix(list(p) + list(quat))


rng = np.random.default_rng(1)
# start from a "pointing down over the table" configuration, like the agent after `point down`
q0 = np.array([0.0, 0.9, 0.9, -0.3, 0.0, 0.0])
down = fk(q0).copy(); down[:3, :3] = motion.np.array([[0, 0, 1.0], [0, -1.0, 0], [1.0, 0, 0]]) @ np.eye(3)  # approach -z in base frame
for attempts in (1, motion.IK_ATTEMPTS):
    motion.IK_ATTEMPTS = attempts
    fails, total, examples = 0, 0, []
    for trial in range(60):
        r = planner.solve_ik_to_joint(q0.astype(np.float32), geo.matrix_to_pose(down), real_robot_pose=origin)
        if r["status"] != "Success":
            continue
        start = np.asarray(r["joint_value"], dtype=float)
        delta = rng.uniform([-0.15, -0.15, -0.10], [0.15, 0.15, 0.10])
        target = down.copy(); target[:3, 3] += delta
        total += 1
        try:
            motion.plan_line(planner, Robot(), start, down, target, None)
        except motion.PlanFailure as f:
            fails += 1
            if len(examples) < 3: examples.append((np.round(delta, 3).tolist(), f.detail))
    print(f"IK attempts={attempts}: {fails}/{total} lines failed; examples {examples}")
