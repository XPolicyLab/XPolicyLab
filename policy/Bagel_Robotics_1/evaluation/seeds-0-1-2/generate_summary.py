#!/usr/bin/env python3
"""Regenerate the unchanged RoboDojo summary from the published score ledger.

This groups already-scored outcomes; it does not run or rescore any episode.
Python standard library only. See README.md for the pinned RoboDojo checkout.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

SCRIPTS = {
    "scripts/robodojo.sh": "7008023f0fd27fc5f8f2adad30b1286a5e6cce432be981b82cf2ffa3498d6ea3",
    "scripts/internal/summarize_result.py": "9ec361af587e1788d83d3cf8670e05b817ff252d33c33829a6337dc16ae370d2",
}
POLICY = "Bagel-Robotics-1_with_labeled_replacements_and_repair"
SUITES = {"precision": 400, "memory": 300, "generalization": 600,
          "long_horizon": 400, "open": 400}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty output directory; never use a live eval_result directory")
    parser.add_argument("--view", choices=["reported", "original"], default="reported")
    args = parser.parse_args()
    root = args.robodojo_root.resolve()
    for name, expected in SCRIPTS.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            parser.error(f"{name} must match RoboDojo f57e1633ffed59761bc8e568cfe7530c4b33381c")
    ledger = Path(__file__).with_name("episode-scores.csv")
    with ledger.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 6300:
        parser.error("Expected 6,300 reported outcomes")
    seen = set()
    for row in rows:
        row["seed"] = int(row["seed"])
        row["layout_id"] = int(row["layout_id"])
        if row["success"] not in ("true", "false"):
            parser.error("Invalid success field")
        row["success"] = row["success"] == "true"
        row["partial_score"] = float(row["partial_score"])
        if not math.isfinite(row["partial_score"]) or not 0 <= row["partial_score"] <= 1:
            parser.error("Invalid native partial score")
        key = row["seed"], row["task"], row["layout_id"]
        if key in seen:
            parser.error(f"Duplicate scored identity: {key}")
        seen.add(key)
    counts = collections.Counter((r["seed"], r["suite"]) for r in rows)
    expected = {(s, suite): count for s in range(3) for suite, count in SUITES.items()}
    if counts != expected:
        parser.error("Seed/suite coverage differs from the published selection")
    selected = rows if args.view == "reported" else [r for r in rows if r["record_type"] == "original"]
    if len(selected) != (6300 if args.view == "reported" else 6271):
        parser.error("Unexpected selection count")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty; existing results will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    eval_root = output / "eval_result" / "RoboDojo"
    grouped = collections.defaultdict(list)
    for row in selected:
        grouped[row["seed"], row["task"]].append(row)
    policy = POLICY if args.view == "reported" else "Bagel-Robotics-1_original_selection"
    for (seed, task), episodes in sorted(grouped.items()):
        episodes.sort(key=lambda r: r["layout_id"])
        n = len(episodes)
        target = (eval_root / task / policy / "arx_x5" /
                  f"{seed}_ckpt_name=cp1000,action_type=ee" / "2026-11-09_00-00-00" / "_result.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "eval_time": n,
            "success_rate": sum(r["success"] for r in episodes) / n,
            "score": 100 * math.fsum(r["partial_score"] for r in episodes) / n,
            "details": {str(r["layout_id"]): {
                "layout_id": r["layout_id"], "success": r["success"], "score": r["partial_score"],
                "record_type": r["record_type"], "replaces_episode": r["replaces_episode"],
            } for r in episodes},
            "_aggregation": {
                "source": "episode-scores.csv", "view": args.view, "checkpoint": 1000,
                "note": "Score-field projection of retained results; outcomes and actual layout IDs unchanged."
            },
        }
        target.write_text(json.dumps(data, indent=2) + "\n")
    env = dict(os.environ, ROBODOJO_EVAL_ROOT=str(eval_root))
    result = subprocess.run(["bash", str(root / "scripts/robodojo.sh"), "summarize"],
                            cwd=root, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    (output / "summarize.log").write_text(result.stdout)
    if result.returncode:
        raise SystemExit(f"Summarizer failed ({result.returncode}); see {output / 'summarize.log'}")
    shutil.copyfile(eval_root / "_summary.md", output / "summary.md")
    print(f"Generated {output / 'summary.md'} from {len(selected)} retained outcomes.")
    print("Score-only export: rollout videos are not copied; native video-check messages remain in summarize.log.")


if __name__ == "__main__":
    main()
