"""How fast the reactive layer acts, against one model decision (paper: Tab. reactive (a)).

Times, with the runtime's own call_reactive (audit hook and budget included):
- one `when` guard and one `watch` guard evaluated on an observation;
- one call of intercept's controller (skills/reflexes) on a 640x480 head-camera frame, on
  a new frame (tracking and projection) and between frames;
and, from recorded runs (--runs), the time from a model request to its reply.

A guard is evaluated before every command of a bounded execution, so it stops the motion
on the first tick whose observation satisfies it (tests/test_reflexes.py checks that).

  python scripts/reactive_latency.py --runs runs/eval/eval-1.4.1 > latency.json
"""

import argparse
import glob
import json
import os
from pathlib import Path
import statistics
import sys
import time
import tomllib
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.geometry import DOWN, link6_for_tip  # noqa: E402
from harness_policy.runtime.reactive import ReactiveLayer, call_reactive  # noqa: E402
from skills.reflexes import PatchTracker, intercept_controller  # noqa: E402


def timed(function, repeats):
    """Milliseconds per call: median and 99th percentile."""
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return {"median_ms": round(statistics.median(samples), 4), "p99_ms": round(samples[int(0.99 * (len(samples) - 1))], 4)}


def head_frame(embodiment, u, v):
    """A head-camera frame (gray table, a red 60-pixel square with a dark band at (u, v), the
    size of a conveyor object in the head camera) with the embodiment's calibration."""
    calibration = embodiment["calibration"]["cam_head"]
    image = np.full((480, 640, 3), 120, dtype=np.uint8)
    image[v - 30:v + 30, u - 30:u + 30] = (200, 30, 30)
    image[v - 4:v + 4, u - 30:u + 30] = (60, 20, 20)
    return SimpleNamespace(images={"cam_head": image},
                           intrinsics={"cam_head": np.asarray(calibration["intrinsics"], dtype=float)},
                           camera_to_world={"cam_head": np.asarray(calibration["camera_to_world"], dtype=float)})


def model_latency(run_dirs):
    """Seconds from each model request to its reply in the runs' events.jsonl files."""
    seconds = []
    for run in run_dirs:
        for path in glob.glob(os.path.join(run, "**", "events.jsonl"), recursive=True):
            asked = None
            for line in open(path, errors="ignore"):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "model_request":
                    asked = event.get("time")
                elif event.get("event") == "model_reply" and asked is not None and event.get("time"):
                    seconds.append(event["time"] - asked)
                    asked = None
    if not seconds:
        return None
    seconds.sort()
    return {"calls": len(seconds), "median_s": round(statistics.median(seconds), 2),
            "p90_s": round(seconds[int(0.9 * (len(seconds) - 1))], 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embodiment", default=str(ROOT / "embodiments" / "arx_x5" / "embodiment.toml"))
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--budget-ms", type=float, default=20.0)
    parser.add_argument("--runs", nargs="*", default=[], help="run directories with events.jsonl files")
    args = parser.parse_args()
    embodiment = tomllib.loads(Path(args.embodiment).read_text())
    budget = args.budget_ms / 1000.0
    observation = {"step": 10, "t": 0.4, "left_ee": [0.0, -0.2, 1.0], "left_ee_pose": [0.0, -0.2, 1.0, *DOWN],
                   "left_gripper": 1.0}

    when = ReactiveLayer(budget)
    when.when(lambda o: o["left_ee"][2] > 1.2, "TOO_HIGH")
    watch = ReactiveLayer(budget)
    watch.watch("left_ee.2", high=1.2)

    first = head_frame(embodiment, 320, 300)
    mask = np.zeros((480, 640), dtype=bool)
    mask[270:330, 290:350] = True
    tracker = PatchTracker(first.images["cam_head"], mask)
    controller, _ = intercept_controller(tracker, "cam_head", 0.80, DOWN, "left_arm",
                                         lambda tip, orientation: link6_for_tip(tip, orientation, 0.150))
    frames = [head_frame(embodiment, 320 + k % 20, 300) for k in range(64)]
    counter = {"k": 0}

    def new_frame():
        counter["k"] += 1
        view = dict(observation, t=0.1 * counter["k"], observation=frames[counter["k"] % len(frames)])
        call_reactive("controller", controller, view, budget_seconds=budget)

    held = dict(observation, t=0.5, observation=frames[0])
    call_reactive("controller", controller, held, budget_seconds=budget)

    result = {
        "control_period_ms": round(1000.0 / float(embodiment.get("control_hz", 25.0)), 1),
        "budget_ms": args.budget_ms,
        "when_guard": timed(lambda: when.evaluate(observation), args.repeats),
        "watch_guard": timed(lambda: watch.evaluate(observation), args.repeats),
        "intercept_controller_new_frame": timed(new_frame, min(args.repeats, 500)),
        "intercept_controller_between_frames": timed(
            lambda: call_reactive("controller", controller, held, budget_seconds=budget), args.repeats),
        "model_decision": model_latency(args.runs),
    }
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
