#!/usr/bin/env python3
"""Compare two recorded RoboDojo episodes step by step.

Use it to cross-check the pinned ``stack_bowls`` worker (``adapter.py``) against
the general worker (``general.py``) on the same task, seed, agent and budget.
It reads ``events.jsonl`` and ``summary.json`` only. It does not start a
simulator and it is not a benchmark comparison.
"""

from pathlib import Path
import argparse
import base64
import json
import math
import struct
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from harness_policy.core.trace import read_events

# Keys the general worker adds or fills from the task instead of constants.
EXPECTED_EXTRA_KEYS = {"scene_config", "layout_count", "layout_source"}


def _png_size(url):
    prefix = "data:image/png;base64,"
    if not isinstance(url, str) or not url.startswith(prefix):
        return None
    header = base64.b64decode(url[len(prefix):len(prefix) + 44])
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return list(struct.unpack(">II", header[16:24]))


def _max_abs(left, right):
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return None
    worst = 0.0
    for a, b in zip(left, right):
        if isinstance(a, list) or isinstance(b, list):
            inner = _max_abs(a, b)
            if inner is None:
                return None
            worst = max(worst, inner)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            worst = max(worst, abs(float(a) - float(b)))
        elif a != b:
            return None
    return worst


class Report:
    def __init__(self, atol):
        self.atol = atol
        self.differences = []
        self.numeric = {}

    def exact(self, where, left, right):
        if left != right:
            self.differences.append({"where": where, "left": left, "right": right})

    def close(self, where, left, right):
        delta = _max_abs(left, right) if isinstance(left, list) else (
            abs(float(left) - float(right))
            if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
        )
        if delta is None or not math.isfinite(delta):
            self.differences.append({"where": where, "left": left, "right": right})
            return
        self.numeric[where] = delta
        if delta > self.atol:
            self.differences.append({"where": where, "max_abs_diff": delta, "atol": self.atol})


def _load(run):
    events = list(read_events(run))
    summary = json.loads((run / "summary.json").read_text())
    reset = next((event for event in events if event["event"] == "reset"), None)
    decisions = {event["step"]: event for event in events if event["event"] == "decision"}
    transitions = {event["step"]: event for event in events if event["event"] == "transition"}
    errors = [event["message"] for event in events if event["event"] == "error"]
    return {"summary": summary, "reset": reset, "decisions": decisions,
            "transitions": transitions, "errors": errors}


def _compare_observation(report, where, left, right):
    report.exact(f"{where}.step", left["step"], right["step"])
    report.exact(f"{where}.instruction", left["instruction"], right["instruction"])
    left_state, right_state = left["state"], right["state"]
    only_left = sorted(set(left_state) - set(right_state))
    only_right = sorted(set(right_state) - set(left_state) - EXPECTED_EXTRA_KEYS)
    if only_left or only_right:
        report.differences.append({"where": f"{where}.state_keys",
                                   "only_left": only_left, "only_right": only_right})
    for key in sorted(set(left_state) & set(right_state)):
        a, b = left_state[key], right_state[key]
        if key in ("joint_position", "left_ee_pose", "right_ee_pose"):
            report.close(f"{where}.state.{key}", a, b)
        elif key in ("seed", "task", "stack"):
            report.exact(f"{where}.state.{key}", a, b)
        elif key == "physics_device":
            continue
        else:
            report.exact(f"{where}.state.{key}", a, b)
    report.exact(f"{where}.cameras", sorted(left["images"]), sorted(right["images"]))
    for camera in sorted(set(left["images"]) & set(right["images"])):
        report.exact(f"{where}.image_size.{camera}",
                     _png_size(left["images"][camera]), _png_size(right["images"][camera]))


def compare(left_run, right_run, atol):
    left, right = _load(left_run), _load(right_run)
    report = Report(atol)
    for side, data in (("left", left), ("right", right)):
        if data["reset"] is None:
            report.differences.append({"where": f"{side}.reset", "error": "no reset event"})
    if left["reset"] and right["reset"]:
        report.exact("tools", left["reset"]["tools"], right["reset"]["tools"])
        _compare_observation(report, "reset", left["reset"]["observation"], right["reset"]["observation"])
    for key in ("task", "seed", "status", "success", "steps"):
        report.exact(f"summary.{key}", left["summary"].get(key), right["summary"].get(key))
    report.exact("errors", left["errors"], right["errors"])
    for step in sorted(set(left["decisions"]) | set(right["decisions"])):
        a, b = left["decisions"].get(step), right["decisions"].get(step)
        if a is None or b is None:
            report.differences.append({"where": f"decision[{step}]", "missing": "left" if a is None else "right"})
            continue
        report.exact(f"decision[{step}].name", a["action"]["name"], b["action"]["name"])
        report.close(f"decision[{step}].position",
                     a["action"]["arguments"].get("position"), b["action"]["arguments"].get("position"))
    for step in sorted(set(left["transitions"]) | set(right["transitions"])):
        a, b = left["transitions"].get(step), right["transitions"].get(step)
        if a is None or b is None:
            report.differences.append({"where": f"transition[{step}]", "missing": "left" if a is None else "right"})
            continue
        ta, tb = a["transition"], b["transition"]
        for flag in ("success", "terminated", "truncated"):
            report.exact(f"transition[{step}].{flag}", ta[flag], tb[flag])
        report.close(f"transition[{step}].reward", ta["reward"], tb["reward"])
        report.close(f"transition[{step}].executed_position",
                     ta["info"].get("executed_position"), tb["info"].get("executed_position"))
        report.exact(f"transition[{step}].videos", sorted(ta.get("videos", {})), sorted(tb.get("videos", {})))
        _compare_observation(report, f"transition[{step}].observation", ta["observation"], tb["observation"])
    return {
        "left": str(left_run),
        "right": str(right_run),
        "left_environment": left["reset"] and left["reset"].get("environment"),
        "right_environment": right["reset"] and right["reset"].get("environment"),
        "atol": atol,
        "match": not report.differences,
        "differences": report.differences,
        "max_abs_diff": report.numeric,
    }


def main():
    parser = argparse.ArgumentParser(description="Step-by-step parity of two recorded RoboDojo episodes")
    parser.add_argument("left", type=Path, help="Episode directory, e.g. runs/EXPERIMENT/seed-0")
    parser.add_argument("right", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance for joint, pose, action and reward values")
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        raise SystemExit("atol must be a nonnegative finite number")
    result = compare(args.left, args.right, args.atol)
    print(json.dumps(result, indent=2))
    return 0 if result["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
