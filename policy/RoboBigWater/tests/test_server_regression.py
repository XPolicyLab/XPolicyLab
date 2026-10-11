#!/usr/bin/env python3
"""Contract regression test against a running robo-server. Standard library only.

Usage: ROBO_SERVER=... ROBO_ADMIN_SERVER=... python3 tests/test_server_regression.py TASK
"""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = sys.argv[1] if len(sys.argv) > 1 else "stack_bowls"
OBS = tempfile.mkdtemp(prefix="robo_obs_")
ENV = dict(os.environ, ROBO_OBS_DIR=OBS)
RESULTS = []


def robo(*args):
    done = subprocess.run([sys.executable, f"{ROOT}/roboshell/client/robo.py", *map(str, args)], env=ENV, capture_output=True, text=True)
    try:
        reply = json.loads(done.stdout)
    except ValueError:
        reply = {"_raw": done.stdout, "_err": done.stderr[-200:]}
    return done.returncode, reply


def admin(*args):
    done = subprocess.run([sys.executable, f"{ROOT}/roboshell/client/robo_admin.py", *map(str, args)], env=ENV, capture_output=True, text=True)
    return done.returncode, done.stdout.strip()


def state():
    with open(os.path.join(OBS, "state.json")) as handle:
        return json.load(handle)


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition)))
    print(("PASS" if condition else "FAIL"), name, detail, flush=True)


def close(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ---- episode 1: commands ------------------------------------------------
code, instruction = admin("reset", "--task", TASK, "--layout", 0)
check("reset returns the instruction", code == 0 and len(instruction) > 5, repr(instruction))
code, reply = robo("obs")
check("obs is free", code == 0 and reply["budget_left"] == 60, reply)
start = state()
check("obs files", all(os.path.getsize(os.path.join(OBS, f)) > 1000 for f in ("head.png", "wrist_l.png", "wrist_r.png")))
check("state has cameras and both arms", set(start["cameras"]) == {"head", "wrist_l", "wrist_r"} and "left" in start and "right" in start)

code, reply = robo("point", "left", "down")
check("point without --open is a usage error", code == 1)
code, reply = robo("point", "left", "down", "--open", "z")
check("point down --open z is rejected and free", code == 1 and "error" in reply, reply.get("error"))
code, reply = robo("obs")
check("budget untouched by rejected commands", reply["budget_left"] == 60)

for arm, axis, index in (("left", "x", 0), ("right", "y", 1)):
    code, reply = robo("point", arm, "down", "--open", axis)
    s = state()[arm]
    check(f"point {arm} down --open {axis}", code == 0 and reply["plan_ok"] and reply["error_deg"] < 1.0 and reply["error_m"] < 0.005,
          f"err {reply.get('error_m')} m {reply.get('error_deg')} deg method sim_time {reply.get('sim_time_s')}")
    check(f"{arm} approach is -z and fingers open along {axis}", close(s["approach_dir"], [0, 0, -1], 0.03) and abs(abs(s["open_dir"][index]) - 1) < 0.03,
          f"{s['approach_dir']} {s['open_dir']}")

before = state()["left"]["tcp_pos"]
code, reply = robo("move", "left", "--dx", 0.05, "--dy", 0.10, "--dz", -0.08)
after = state()["left"]["tcp_pos"]
check("move left reaches the target", code == 0 and reply["error_m"] < 0.003 and close([a - b for a, b in zip(after, before)], [0.05, 0.10, -0.08], 0.004),
      f"delta {[round(a - b, 4) for a, b in zip(after, before)]} err {reply.get('error_m')}")
check("move keeps the orientation", close(state()["left"]["approach_dir"], [0, 0, -1], 0.03))

code, reply = robo("move", "right", "--dy", 0.5)
check("move is clipped to 0.20 m", code == 0 and reply["clipped"] and abs(reply["reached_tcp"]["pos"][1] - (start["right"]["tcp_pos"][1] + 0.20)) < 0.02 or reply.get("plan_ok") is False,
      f"{reply.get('clipped')} {reply.get('reached_tcp', {}).get('pos')}")

before = state()["left"]
code, reply = robo("rotate", "left", "--yaw", 30)
after = state()["left"]
check("rotate yaw 30 (world) keeps the position", code == 0 and reply["error_deg"] < 1.0 and close(after["tcp_pos"], before["tcp_pos"], 0.004),
      f"err {reply.get('error_deg')} deg pos shift {[round(a - b, 4) for a, b in zip(after['tcp_pos'], before['tcp_pos'])]}")
check("approach still -z after yaw", close(after["approach_dir"], [0, 0, -1], 0.03), after["approach_dir"])
code, reply = robo("rotate", "left", "--pitch", 20, "--frame", "tool")
check("rotate in the tool frame", code == 0 and reply["plan_ok"] and reply["error_deg"] < 1.5, f"err {reply.get('error_deg')}")
code, reply = robo("rotate", "left", "--roll", 200)
check("rotate is clipped to 90 deg", reply.get("clipped") is True, reply.get("plan_fail_reason"))

code, reply = robo("gripper", "left", 0.5)
check("gripper 0.5", code == 0 and abs(reply["gripper"] - 0.5) < 0.08, reply.get("gripper"))
code, reply = robo("gripper", "left", "close")
check("gripper close", code == 0 and reply["gripper"] < 0.3, reply.get("gripper"))
code, reply = robo("gripper", "left", "open")
check("gripper open", code == 0 and reply["gripper"] > 0.9, reply.get("gripper"))

code, reply = robo("status")
check("status repeats the last feedback", code == 0 and reply.get("cmd") == "gripper")
t0 = reply["sim_time_s"]
code, reply = robo("wait", 1)
check("wait 1 s advances the simulation by 1 s", code == 0 and abs(reply["sim_time_s"] - t0 - 1.0) < 0.05, reply.get("sim_time_s"))
code, reply = robo("move", "up", "--dx", 0.1)
check("unknown arm is a usage error", code == 1)

code, reply = robo("home", "both")
s = state()
check("home both returns to the start pose", code == 0 and close(s["left"]["tcp_pos"], start["left"]["tcp_pos"], 0.01) and close(s["right"]["tcp_pos"], start["right"]["tcp_pos"], 0.01),
      f"{s['left']['tcp_pos']} {s['right']['tcp_pos']}")
budget = reply["budget_left"]
code, reply = robo("done")
check("done returns a boolean and ends the episode", code == 0 and reply["success"] is False and reply.get("episode_over"))
check("done reveals nothing else", not ({"progress_score", "end_reason", "success_official"} & set(reply)), sorted(reply))
code, reply = robo("move", "left", "--dx", 0.01)
check("commands after the end return exit code 3", code == 3)
code, text = admin("result")
first = json.loads(text)
check("result file", first["end_reason"] == "done" and first["done_called"] and first["commands"] == 60 - budget, {k: first[k] for k in ("end_reason", "commands", "action_steps")})

# ---- replay ---------------------------------------------------------------
code, text = admin("replay", first["directory"])
report = json.loads(text)
check("replay runs the same number of steps", report["steps_replayed"] == report["steps_recorded"], f"{report['steps_replayed']} / {report['steps_recorded']}")
check("replay reproduces the final state", report["max_object_pos_diff_m"] < 1e-3 and report["max_joint_diff_rad"] < 1e-3,
      f"objects {report['max_object_pos_diff_m']:.2e} m, joints {report['max_joint_diff_rad']:.2e} rad")
print("REPLAY", json.dumps({k: report[k] for k in ("max_object_pos_diff_m", "max_joint_diff_rad", "success_recorded", "success_replayed")}))

# ---- episode 2: command budget -------------------------------------------
admin("reset", "--task", TASK, "--layout", 1)
last = None
for index in range(60):
    code, last = robo("wait", 0.04)
check("the 60th command ends the episode", last.get("episode_over") and last["budget_left"] == 0, last)
code, text = admin("result")
check("end reason budget", json.loads(text)["end_reason"] == "budget", json.loads(text)["end_reason"])

# ---- episode 3: simulation time ------------------------------------------
admin("reset", "--task", TASK, "--layout", 2)
for index in range(10):
    code, last = robo("wait", 5)
    if last.get("episode_over") or code == 3:
        break
code, text = admin("result")
result = json.loads(text)
check("the step limit ends the episode", result["end_reason"] == "sim_time" and result["action_steps"] == result["step_lim"], {k: result[k] for k in ("end_reason", "action_steps", "step_lim")})

failed = [name for name, ok in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
for name in failed:
    print("  FAILED:", name)
sys.exit(1 if failed else 0)
