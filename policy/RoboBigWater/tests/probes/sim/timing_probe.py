"""Measure, against a running server: action steps and tracking error of moves, and the gripper stroke time."""
import json, os, subprocess, sys
ROBO = [sys.executable, "roboshell/client/robo.py"]; ADMIN = [sys.executable, "roboshell/client/robo_admin.py"]
def call(prog, *a):
    out = subprocess.run(prog + [str(x) for x in a], capture_output=True, text=True).stdout
    try: return json.loads(out)
    except Exception: return {"raw": out[-200:]}
call(ADMIN, "reset", "--task", sys.argv[1], "--layout", 0)
log = None
def last():
    import glob
    d = sorted(glob.glob(os.path.join(os.environ["PROBE_RUN_ROOT"], "*_s*_l*")))[-1]
    rows = [json.loads(l) for l in open(os.path.join(d, "commands.jsonl"))]
    return rows[-1]
def show(name, fb):
    row = last(); s = row["steps"]
    print(f"{name:34s} steps={s[1]-s[0]:3d} settle={fb.get('settle_steps')} plan_ok={fb.get('plan_ok')} error_mm={None if fb.get('error_m') is None else round(fb['error_m']*1000,2)} err_deg={fb.get('error_deg')}")
show("point left down --open x", call(ROBO, "point", "left", "down", "--open", "x"))
for args in (("--dz", -0.05), ("--dz", -0.10), ("--dx", 0.10), ("--dx", -0.20), ("--dy", -0.15), ("--dz", 0.15), ("--dy", 0.15)):
    show("move left %s %s" % args, call(ROBO, "move", "left", *args))
show("rotate left --yaw 90", call(ROBO, "rotate", "left", "--yaw", 90))
show("rotate left --yaw -90", call(ROBO, "rotate", "left", "--yaw", -90))
show("home both", call(ROBO, "home", "both"))
print("gripper stroke (measured opening after each control step; GRIPPER_STEPS of this server:", os.environ.get("ROBOSHELL_GRIPPER_STEPS"), ")")
for target in ("close", "open"):
    call(ROBO, "gripper", "left", target); row = last(); seq = [row["server_only"]["gripper_measured"]["left"]]; n0 = row["steps"][1] - row["steps"][0]
    for _ in range(18):
        call(ROBO, "wait", 0.04); seq.append(last()["server_only"]["gripper_measured"]["left"])
    print(f"  {target}: first reading after {n0} step(s):", seq)
call(ROBO, "done")
