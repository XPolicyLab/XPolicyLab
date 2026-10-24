#!/usr/bin/env python3
"""The evidence for a learned policy's description: its scores run alone, per task.

    python scripts/policy_card.py runs/simate-alone-dev [runs/...] [--json]

Reads the episodes of `harness run configs/experiments/policy-alone-dev.toml --script
configs/scripts/policy_alone.json` (the policy handed each whole task from the start
pose, alone) and sorts the tasks by their mean score (0-1, RoboDojo's: 1 on success,
else the process score) into the three kinds a description names:

    reliable     mean score >= 0.7: it usually finishes by itself
    partial      0.1 <= mean < 0.7: it often does part and leaves the rest
    unreliable   mean < 0.1: it rarely helps

(the thresholds the G0.5 card of Z2 was derived with). The description in
services/<name>.toml generalises these tasks into kinds of manipulation and never
names a task; this script gives the tasks behind each kind.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src")]

from harness_policy.runtime.report import TASK_DIMENSION, aggregate, load_episodes  # noqa: E402

KINDS = (("reliable", 0.7), ("partial", 0.1), ("unreliable", float("-inf")))


def card(directories) -> dict:
    """{"policy", "episodes", "kinds": {kind: [(task, dimension, mean score, episodes)]}}."""
    episodes = load_episodes(directories)
    if not episodes:
        raise SystemExit(f"No episodes under {', '.join(map(str, directories))}")
    result = aggregate(episodes)
    kinds = {kind: [] for kind, _ in KINDS}
    for task, entry in sorted(result["tasks"].items(), key=lambda item: -(item[1]["score"] or 0.0)):
        if entry["score"] is None:
            continue
        mean = entry["score"] / 100
        kind = next(name for name, floor in KINDS if mean >= floor)
        kinds[kind].append((task, TASK_DIMENSION.get(task, "?"), round(mean, 2), entry["episodes"]))
    return {"episodes": len(episodes), "tasks": len(result["tasks"]), "kinds": kinds,
            "dimensions": {name: round(d["score"], 1) for name, d in result["dimensions"].items()
                           if d["score"] is not None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = card(args.runs)
    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    print(f"{result['episodes']} episodes, {result['tasks']} tasks; Score by dimension: "
          + ", ".join(f"{name} {score}" for name, score in result["dimensions"].items()))
    for kind, tasks in result["kinds"].items():
        print(f"\n{kind} ({len(tasks)}):")
        for task, dimension, mean, count in tasks:
            print(f"  {task:34s} {dimension:15s} {mean:.2f}  ({count} episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
