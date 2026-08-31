#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_tasks(path: Path) -> list[str]:
    tasks = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    tasks = [task for task in tasks if task]
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError(f"Expected exactly 50 unique tasks in {path}, got {len(tasks)}")
    return tasks


def load_conditions(value: str) -> list[str]:
    conditions = [item.strip() for item in value.split(",") if item.strip()]
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError(f"Conditions must be a non-empty unique list: {value!r}")
    invalid = set(conditions) - {"clean", "randomized"}
    if invalid:
        raise ValueError(f"Unsupported conditions: {sorted(invalid)}")
    return conditions


def build_records(
    run_dir: Path,
    tasks: list[str],
    conditions: list[str],
    expected_episodes: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for condition in conditions:
        task_config = f"demo_{condition}"
        for task in tasks:
            stem = f"{condition}_{task}"
            element_path = run_dir / "elements" / f"{stem}.json"
            exit_path = run_dir / "status" / f"{stem}.exit_code"
            record: dict[str, Any] = {
                "condition": condition,
                "task_config": task_config,
                "task": task,
                "status": "failed",
                "exit_code": None,
                "episodes": None,
                "success_count": None,
                "success_rate": None,
                "result_file": None,
                "stdout_log": str((run_dir / "logs" / f"{stem}.out").resolve()),
                "stderr_log": str((run_dir / "logs" / f"{stem}.err").resolve()),
                "error": None,
            }
            errors: list[str] = []
            if not exit_path.is_file():
                errors.append(f"missing exit status: {exit_path}")
            else:
                try:
                    exit_code = int(exit_path.read_text(encoding="utf-8").strip())
                    record["exit_code"] = exit_code
                    if exit_code != 0:
                        errors.append(f"client exit={exit_code}")
                except (OSError, ValueError) as exc:
                    errors.append(f"invalid exit status: {exc}")

            if not element_path.is_file():
                errors.append(f"missing element summary: {element_path}")
            else:
                try:
                    payload = json.loads(element_path.read_text(encoding="utf-8"))
                    record.update(
                        episodes=payload.get("episodes"),
                        success_count=payload.get("success_count"),
                        success_rate=payload.get("success_rate"),
                        result_file=payload.get("result_file"),
                    )
                    if payload.get("task") != task:
                        errors.append(f"task mismatch: {payload.get('task')!r}")
                    if payload.get("task_config") != task_config:
                        errors.append(f"config mismatch: {payload.get('task_config')!r}")
                    if payload.get("episodes") != expected_episodes:
                        errors.append(
                            f"episodes={payload.get('episodes')}, expected={expected_episodes}"
                        )
                    rate = payload.get("success_rate")
                    count = payload.get("success_count")
                    if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
                        errors.append(f"invalid success_rate={rate!r}")
                    if not isinstance(count, int) or not 0 <= count <= expected_episodes:
                        errors.append(f"invalid success_count={count!r}")
                    result_file = payload.get("result_file")
                    if not result_file or not Path(result_file).is_file():
                        errors.append(f"missing RoboTwin result file: {result_file}")
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(f"invalid element summary: {exc}")

            if errors:
                record["error"] = "; ".join(errors)
            else:
                record["status"] = "passed"
            records.append(record)
    return records


def metrics(records: list[dict[str, Any]], condition: str | None = None) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if condition is None or record["condition"] == condition
    ]
    passed = [record for record in selected if record["status"] == "passed"]
    complete = len(passed) == len(selected)
    successes = sum(int(record["success_count"]) for record in passed)
    episodes = sum(int(record["episodes"]) for record in passed)
    return {
        "tasks_expected": len(selected),
        "tasks_passed": len(passed),
        "complete": complete,
        "macro_average_success_rate": (
            sum(float(record["success_rate"]) for record in passed) / len(selected)
            if complete and selected
            else None
        ),
        "aggregate_success_count": successes,
        "aggregate_episodes": episodes,
        "aggregate_success_rate": successes / episodes if complete and episodes else None,
    }


def write_outputs(
    run_dir: Path,
    records: list[dict[str, Any]],
    conditions: list[str],
) -> bool:
    condition_metrics = {condition: metrics(records, condition) for condition in conditions}
    overall = metrics(records)
    complete = overall["complete"]
    payload = {
        "complete": complete,
        "conditions": condition_metrics,
        "overall": overall,
        "records": records,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (run_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# OLA-SEM local 50-task evaluation",
        "",
        "| condition | completed | macro average | aggregate |",
        "|---|---:|---:|---:|",
    ]
    for condition in conditions:
        item = condition_metrics[condition]
        macro = item["macro_average_success_rate"]
        aggregate = item["aggregate_success_rate"]
        lines.append(
            f"| {condition} | {item['tasks_passed']}/{item['tasks_expected']} | "
            f"{macro * 100:.2f}% | {aggregate * 100:.2f}% |"
            if macro is not None and aggregate is not None
            else f"| {condition} | {item['tasks_passed']}/{item['tasks_expected']} | N/A | N/A |"
        )
    lines.extend(
        [
            "",
            "| condition | task | status | success | result/error |",
            "|---|---|---|---:|---|",
        ]
    )
    for record in records:
        success = (
            f"{record['success_count']}/{record['episodes']}"
            if record["success_count"] is not None
            else "N/A"
        )
        detail = record["result_file"] or record["error"] or ""
        lines.append(
            f"| {record['condition']} | {record['task']} | {record['status']} | "
            f"{success} | {detail} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tasks-file", required=True, type=Path)
    parser.add_argument("--conditions", default="clean,randomized")
    parser.add_argument("--expected-episodes", type=int, default=100)
    args = parser.parse_args()
    tasks = load_tasks(args.tasks_file)
    conditions = load_conditions(args.conditions)
    records = build_records(args.run_dir, tasks, conditions, args.expected_episodes)
    complete = write_outputs(args.run_dir, records, conditions)
    print(f"Summary written to {args.run_dir}; complete={complete}")
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
