"""Operator-side check of the pick tool near the reach limit: reset a layout, run pick at given points, report stages.

Usage: pick_reach.py LAYOUT ARM X Y Z [ARM X Y Z ...]   (server on ROBO_SERVER / ROBO_ADMIN_SERVER)
"""
import json, os, subprocess, sys
import numpy as np

ROBO = [sys.executable, "roboshell/client/robo.py"]
ADMIN = [sys.executable, "roboshell/client/robo_admin.py"]


def call(prog, *args):
    out = subprocess.run(prog + [str(a) for a in args], capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"raw": out[-300:]}


layout = int(sys.argv[1]); picks = sys.argv[2:]
print("reset", call(ADMIN, "reset", "--task", "stack_bowls", "--layout", layout).get("episode_id"))
for i in range(0, len(picks), 4):
    arm, x, y, z = picks[i:i + 4]
    reply = call(ROBO, "pick", arm, "--x", x, "--y", y, "--z", z)
    print("pick", arm, (x, y, z), "approach:", reply.get("approach"), "plan_ok:", reply.get("plan_ok"), reply.get("plan_fail_reason"), reply.get("plan_detail"))
    for s in reply.get("stages", []):
        print("   ", s)
    state = json.load(open(os.path.join(os.environ.get("ROBO_OBS_DIR", "obs"), "state.json")))[arm]
    print("    tcp now", np.round(state["tcp_pos"], 3), "approach_dir", np.round(state["approach_dir"], 2), "gripper", reply.get("gripper"))
    call(ROBO, "home", arm)
call(ROBO, "done")
result = call(ADMIN, "result")
print({k: result.get(k) for k in ("end_reason", "commands", "sim_time_s", "progress_score")})
final = dict(np.load(result["directory"] + "/final_state.npz"))
for key, value in final.items():
    if key.startswith("object_"):
        print("final", key, np.round(value[:3], 3))
