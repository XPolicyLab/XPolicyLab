"""Operator-side determinism check with contact: push a bowl, then replay the episode.

Uses the true object pose from a previous episode of the same layout, so it must never run inside the agent container.
Usage: push_and_replay.py REFERENCE_EPISODE_DIR
"""
import json, subprocess, sys
import numpy as np

ROBO = [sys.executable, "roboshell/client/robo.py"]
ADMIN = [sys.executable, "roboshell/client/robo_admin.py"]


def robo(*args):
    reply = json.loads(subprocess.run(ROBO + [str(a) for a in args], capture_output=True, text=True).stdout)
    print(args, {k: reply.get(k) for k in ("plan_ok", "error_m", "sim_time_s", "episode_over", "plan_fail_reason")})
    return reply


def go(arm, dx, dy, dz):
    while max(abs(dx), abs(dy), abs(dz)) > 1e-4:
        step = [float(np.clip(v, -0.2, 0.2)) for v in (dx, dy, dz)]
        robo("move", arm, "--dx", round(step[0], 4), "--dy", round(step[1], 4), "--dz", round(step[2], 4))
        dx, dy, dz = dx - step[0], dy - step[1], dz - step[2]


reference = dict(np.load(sys.argv[1] + "/final_state.npz"))
layout = json.load(open(sys.argv[1] + "/result.json"))["layout"]
objects = {k: v for k, v in reference.items() if k.startswith("object_")}
for key, value in objects.items():
    print(key, np.round(value[:3], 3))
subprocess.run(ADMIN + ["reset", "--task", "stack_bowls", "--layout", str(layout)])
name, pose = min(objects.items(), key=lambda kv: kv[1][0])
robo("point", "left", "down", "--open", "y")
import os
start = json.load(open(os.path.join(os.environ["ROBO_OBS_DIR"], "state.json")))["left"]["tcp_pos"]
go("left", pose[0] - 0.14 - start[0], pose[1] - start[1], 0.0)
go("left", 0, 0, 0.80 - start[2])
robo("gripper", "left", "close")
go("left", 0.18, 0, 0)
go("left", 0, 0, 0.1)
robo("done")
result = json.loads(subprocess.run(ADMIN + ["result"], capture_output=True, text=True).stdout)
print({k: result[k] for k in ("end_reason", "commands", "action_steps", "sim_time_s", "progress_score")})
final = dict(np.load(result["directory"] + "/final_state.npz"))
for key in objects:
    print("moved", key, np.round(np.linalg.norm(final[key][:3] - objects[key][:3]), 4), "m")
report = json.loads(subprocess.run(ADMIN + ["replay", result["directory"]], capture_output=True, text=True).stdout)
print("REPLAY", {k: report[k] for k in ("steps_recorded", "steps_replayed", "max_object_pos_diff_m", "max_joint_diff_rad")})
print(json.dumps(report["differences"]))
