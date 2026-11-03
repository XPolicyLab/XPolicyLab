#!/usr/bin/env python3
"""Exercise a live RoboDojo runtime worker: ticks, shield, kinematics, planning, cameras.

    python3 scripts/robodojo_runtime_check.py --endpoint http://127.0.0.1:8770 --task stack_bowls

Writes camera images and a JSON report to --output. This checks the plumbing;
it does not attempt the task and its outcome is not a benchmark result.
"""

import argparse
import base64
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness_policy.core.robot import RemoteEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8770")
    parser.add_argument("--task", default="stack_bowls")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="runs/runtime-check")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = RemoteEnvironment(args.endpoint, timeout=600)
    report = {}

    env.tools()
    started = time.perf_counter()
    observation = env.reset(args.task, args.seed)
    report["reset_seconds"] = time.perf_counter() - started
    episode = observation.episode_id
    report["instruction"] = observation.instruction
    profile = env.call("profile")
    report["profile"] = {key: profile[key] for key in (
        "task", "step_limit", "control_hz", "physics_steps_per_tick", "groups", "arm_bases", "fk_check_error_m")}
    token = env.call("claim_outcome", episode_id=episode)["token"]
    try:
        env.call("claim_outcome", episode_id=episode)
        report["second_claim"] = "ALLOWED (bug)"
    except RuntimeError as error:
        report["second_claim"] = f"refused: {error}"

    state = env.call("state", episode_id=episode)
    step = state["step"]
    hold = list(state["joint_position"])

    # Hold position for 25 ticks and time them.
    durations = []
    for _ in range(25):
        started = time.perf_counter()
        result = env.call("tick", episode_id=episode, step=step, command=hold)
        durations.append(time.perf_counter() - started)
        assert result["executed"], result
        step = result["state"]["step"]
    report["tick_seconds"] = {"mean": sum(durations) / len(durations), "max": max(durations)}

    # Shield: a 1 rad jump on one joint is rejected, and so is a far target.
    jump = list(hold)
    jump[1] += 1.0
    result = env.call("tick", episode_id=episode, step=step, command=jump)
    report["shield_joint_jump"] = result["shield"]["verdict"], [r["check"] for r in result["shield"]["reasons"]]
    step = result["state"]["step"]
    far = list(result["state"]["joint_position"])
    far[1] += 1.4
    far[2] -= 1.4
    result = env.call("tick", episode_id=episode, step=step, command=far)
    report["shield_far_target"] = result["executed"], result["shield"]["verdict"], [r["check"] for r in result["shield"]["reasons"]]
    step = result["state"]["step"]

    # Kinematics: FK of the measured joints against the simulator pose.
    state = env.call("state", episode_id=episode)
    fk = env.call("fk", command=state["joint_position"])
    report["fk_error_m"] = {arm: math.dist(fk[arm][:3], state[f"{arm.split('_')[0]}_ee_pose"][:3]) for arm in fk}

    # IK on the current left pose should return joints whose FK matches it.
    left = state["left_ee_pose"]
    ik = env.call("ik", episode_id=episode, arm="left_arm", pose=left)
    report["ik_current_pose"] = ik["success"]

    # Plan 5 cm up and follow the waypoints tick by tick.
    target = list(left)
    target[2] += 0.05
    plan = env.call("plan", episode_id=episode, arm="left_arm", pose=target)
    report["plan_success"] = plan["success"]
    report["plan_waypoints"] = len(plan["waypoints"])
    if plan["success"]:
        started = time.perf_counter()
        joints = list(state["joint_position"])
        for waypoint in plan["waypoints"] + [plan["waypoints"][-1]] * 5:
            joints[0:6] = waypoint
            result = env.call("tick", episode_id=episode, step=step, command=joints)
            step = result["state"]["step"]
        report["plan_follow_seconds"] = time.perf_counter() - started
        reached = result["state"]["left_ee_pose"]
        report["plan_arrival_error_m"] = math.dist(reached[:3], target[:3])

    started = time.perf_counter()
    view = env.call("observe", episode_id=episode)
    report["observe_seconds"] = time.perf_counter() - started
    for camera, url in view["images"].items():
        header, data = url.split(",", 1)
        (out / f"{camera}.jpg").write_bytes(base64.b64decode(data))
    report["intrinsics"] = view["intrinsics"]
    report["camera_to_world"] = view["camera_to_world"]
    report["outcome"] = env.call("outcome", token=token)
    try:
        env.call("outcome", token="wrong")
        report["wrong_token"] = "ALLOWED (bug)"
    except RuntimeError as error:
        report["wrong_token"] = f"refused: {error}"
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
