#!/usr/bin/env python3
"""Audit one exact RoboDojo official-54 evaluation run.

RoboDojo's official summarizer scans the global result tree and selects the
latest result for each policy/task/seed.  This companion audit never replaces
that summarizer; it pins every input to one run id so a final report cannot
accidentally mix in an older or newer evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DIMENSIONS = {
    "Generalization": [
        "stack_bowls",
        "push_T",
        "pack_objects_into_box",
        "fold_clothes",
        "hang_mugs",
        "sweep_blocks",
        "pour_liquid_into_cup",
        "make_toast",
        "arrange_largest_number",
        "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones",
        "stack_blocks",
    ],
    "Precision": [
        "fasten_screws",
        "plug_in_charger",
        "insert_tubes",
        "pour_balls_into_vase",
        "play_Xylophone",
        "deposit_coin",
        "insert_key",
        "build_tower",
    ],
    "Long-Horizon": [
        "put_bottles_into_dustbin",
        "fill_pen_holder",
        "classify_objects",
        "play_tic_tac_toe",
        "fill_egg_holder",
        "organize_table",
        "make_kong",
        "play_stacking_toy",
    ],
    "Memory": [
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_blocks",
        "swap_T",
        "press_by_number",
        "imitate_sorting_sequence",
    ],
    "Open": [
        "align_blocks",
        "general_pickup",
        "stack_blocks_by_language",
        "solve_equation",
        "classify_objects_by_language",
        "pick_from_conveyor_by_image",
        "store_tools_in_toolbox",
        "pour_by_language",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robodojo-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--policy", default="RecoverWAM")
    parser.add_argument("--embodiment", default="arx_x5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--action-type", default="ee")
    parser.add_argument("--runner-summary", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_tasks(path: Path) -> list[str]:
    tasks = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"duplicate entries in task file: {path}")
    return tasks


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def detail_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    details = payload.get("details", {})
    if not isinstance(details, dict):
        raise ValueError("details is not a JSON object")
    entries = []
    for key, value in details.items():
        if not isinstance(value, dict):
            raise ValueError(f"details[{key!r}] is not a JSON object")
        entries.append(value)
    return entries


def rate(numerator: float, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def main() -> int:
    args = parse_args()
    tasks = load_tasks(args.tasks_file)
    task_set = set(tasks)
    base_tasks = [task for task in tasks if not task.endswith("_random")]
    expected_bases = {task for values in DIMENSIONS.values() for task in values}
    issues: list[str] = []

    if len(tasks) != 54:
        issues.append(f"task inventory has {len(tasks)} entries, expected 54")
    if set(base_tasks) != expected_bases:
        missing = sorted(expected_bases - set(base_tasks))
        extra = sorted(set(base_tasks) - expected_bases)
        issues.append(f"official 42-cell inventory mismatch: missing={missing}, extra={extra}")

    run_bucket = f"{args.seed}_ckpt_name={args.ckpt_name},action_type={args.action_type}"
    result_root = args.robodojo_root / "eval_result" / "RoboDojo"
    task_records: dict[str, dict[str, Any]] = {}

    for task in tasks:
        expected = 25 if task.endswith("_random") or f"{task}_random" in task_set else 50
        result_path = (
            result_root
            / task
            / args.policy
            / args.embodiment
            / run_bucket
            / f"{args.run_id}_{task}"
            / "_result.json"
        )
        record: dict[str, Any] = {
            "expected_episodes": expected,
            "result_path": str(result_path),
            "episodes": 0,
            "successes": 0,
            "score_sum": 0.0,
        }
        if not result_path.is_file():
            issues.append(f"missing result: {task}")
            task_records[task] = record
            continue
        try:
            payload = load_json(result_path)
            entries = detail_entries(payload)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"invalid result for {task}: {exc}")
            task_records[task] = record
            continue

        layout_ids = [entry.get("layout_id") for entry in entries]
        if len(layout_ids) != len(set(layout_ids)):
            issues.append(f"duplicate layout_id in result: {task}")
        if len(entries) != expected:
            issues.append(f"episode count for {task}: found {len(entries)}, expected {expected}")
        success_count = sum(bool(entry.get("success", False)) for entry in entries)
        score_sum = sum(float(entry.get("score", 0.0) or 0.0) for entry in entries)
        record.update(
            {
                "episodes": len(entries),
                "successes": success_count,
                "score_sum": score_sum,
                "success_rate": rate(success_count, len(entries)),
                "score": rate(score_sum, len(entries)),
            }
        )
        task_records[task] = record

    runner_pass = None
    if args.runner_summary is not None:
        try:
            runner = load_json(args.runner_summary)
            runner_results = runner.get("results", [])
            runner_pass = bool(
                isinstance(runner_results, list)
                and len(runner_results) == 54
                and all(row.get("status") == "PASS" for row in runner_results)
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"invalid runner summary: {exc}")
            runner_pass = False
        if not runner_pass:
            issues.append("runner summary does not contain exactly 54 PASS results")

    cells: dict[str, dict[str, Any]] = {}
    for base in base_tasks:
        parts = [task_records[base]]
        random_name = f"{base}_random"
        if random_name in task_records:
            parts.append(task_records[random_name])
        episodes = sum(int(part["episodes"]) for part in parts)
        successes = sum(int(part["successes"]) for part in parts)
        score_sum = sum(float(part["score_sum"]) for part in parts)
        cells[base] = {
            "episodes": episodes,
            "successes": successes,
            "score_sum": score_sum,
            "success_rate": rate(successes, episodes),
            "score": rate(score_sum, episodes),
        }
        if episodes != 50:
            issues.append(f"official cell {base}: found {episodes} episodes, expected 50")

    dimensions: dict[str, dict[str, Any]] = {}
    for dimension, names in DIMENSIONS.items():
        episodes = sum(int(cells.get(name, {}).get("episodes", 0)) for name in names)
        successes = sum(int(cells.get(name, {}).get("successes", 0)) for name in names)
        score_sum = sum(float(cells.get(name, {}).get("score_sum", 0.0)) for name in names)
        dimensions[dimension] = {
            "cells": len(names),
            "episodes": episodes,
            "successes": successes,
            "success_rate": rate(successes, episodes),
            "score": rate(score_sum, episodes),
        }

    episodes = sum(int(record["episodes"]) for record in task_records.values())
    successes = sum(int(record["successes"]) for record in task_records.values())
    score_sum = sum(float(record["score_sum"]) for record in task_records.values())
    if episodes != 2100:
        issues.append(f"total episode count is {episodes}, expected 2100")

    report = {
        "official_mean_success_rate": sum(d["success_rate"] for d in dimensions.values()) / 5,
        "official_mean_score": sum(d["score"] for d in dimensions.values()) / 5,
        "status": "complete" if not issues else "incomplete",
        "run_id": args.run_id,
        "checkpoint_name": args.ckpt_name,
        "policy": args.policy,
        "embodiment": args.embodiment,
        "seed": args.seed,
        "action_type": args.action_type,
        "runnable_tasks": len(tasks),
        "official_cells": len(cells),
        "episodes": episodes,
        "expected_episodes": 2100,
        "successes": successes,
        "success_rate": rate(successes, episodes),
        "score": rate(score_sum, episodes),
        "runner_summary_all_pass": runner_pass,
        "dimensions": dimensions,
        "cells": cells,
        "task_records": task_records,
        "issues": issues,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# RoboDojo exact-run audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Run: `{args.run_id}`",
        f"- Checkpoint label: `{args.ckpt_name}`",
        f"- Inventory: {len(tasks)} runnable tasks / {len(cells)} official cells",
        f"- Episodes: {episodes} / 2100",
        f"- Official five-dimension SR: {report['official_mean_success_rate']:.2f}%",
        f"- Pooled success rate: {report['success_rate']:.2f}%",
        f"- Official five-dimension Score: {report['official_mean_score']:.2f}",
        f"- Pooled score: {report['score']:.2f}",
        "",
        "| Dimension | Episodes | Success rate | Score |",
        "|---|---:|---:|---:|",
    ]
    for name, values in dimensions.items():
        lines.append(
            f"| {name} | {values['episodes']} | "
            f"{values['success_rate']:.2f}% | {values['score']:.2f} |"
        )
    if issues:
        lines.extend(["", "## Incomplete checks", ""])
        lines.extend(f"- {issue}" for issue in issues)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"status={report['status']} tasks={len(tasks)} cells={len(cells)} "
        f"episodes={episodes}/2100 success_rate={report['success_rate']:.2f}% "
        f"score={report['score']:.2f} issues={len(issues)}"
    )
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
