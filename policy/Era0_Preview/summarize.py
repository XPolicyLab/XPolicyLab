#!/usr/bin/env python3
"""Render RoboDojo results in the standard capability-dimension report format."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (0, 1, 2)
SEED_RE = re.compile(r"^(\d+)(?:_|$)")
DIMENSIONS = {
    "Generalization": [
        "stack_bowls", "push_T", "pack_objects_into_box", "fold_clothes",
        "hang_mugs", "sweep_blocks", "pour_liquid_into_cup", "make_toast",
        "arrange_largest_number", "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones", "stack_blocks",
    ],
    "Precision": [
        "fasten_screws", "plug_in_charger", "insert_tubes",
        "pour_balls_into_vase", "play_Xylophone", "deposit_coin",
        "insert_key", "build_tower",
    ],
    "Long-Horizon": [
        "put_bottles_into_dustbin", "fill_pen_holder", "classify_objects",
        "play_tic_tac_toe", "fill_egg_holder", "organize_table",
        "make_kong", "play_stacking_toy",
    ],
    "Memory": [
        "cover_blocks", "match_and_pick_from_conveyor", "swap_blocks",
        "swap_T", "press_by_number", "imitate_sorting_sequence",
    ],
    "Open": [
        "align_blocks", "general_pickup", "stack_blocks_by_language",
        "solve_equation", "classify_objects_by_language",
        "pick_from_conveyor_by_image", "store_tools_in_toolbox", "pour_by_language",
    ],
}
ALL_TASKS = tuple(task for tasks in DIMENSIONS.values() for task in tasks)
GENERALIZATION_TASKS = frozenset(DIMENSIONS["Generalization"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate RoboDojo results using the capability-dimension summary format."
    )
    parser.add_argument("run_dir", type=Path, help="Evaluation output directory or its RoboDojo root.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path.")
    return parser.parse_args()


def subdirs(path: Path) -> list[Path]:
    return sorted((entry for entry in path.iterdir() if entry.is_dir()), key=lambda entry: entry.name) if path.is_dir() else []


def result_root(run_dir: Path) -> Path:
    return run_dir / "RoboDojo" if (run_dir / "RoboDojo").is_dir() else run_dir


def load_entries(result_file: Path) -> list[tuple[bool, float]]:
    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {result_file}: {exc}") from exc
    details = payload.get("details") if isinstance(payload, dict) else None
    if isinstance(details, dict):
        ordered = []
        for layout, row in details.items():
            try:
                ordered.append((int(layout), row))
            except (TypeError, ValueError):
                continue
        rows = [row for _, row in sorted(ordered)]
    elif isinstance(details, list):
        rows = details
    else:
        raise ValueError(f"{result_file}: details must be a mapping or list")
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{result_file}: no readable trajectory details")
    try:
        return [(bool(row.get("success", False)), float(row.get("score", 0.0) or 0.0)) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{result_file}: score must be numeric") from exc


def latest_result(run_path: Path) -> tuple[str, list[tuple[bool, float]]] | None:
    candidates = [entry for entry in subdirs(run_path) if (entry / "_result.json").is_file()]
    if not candidates:
        return None
    latest = candidates[-1]
    return latest.name, load_entries(latest / "_result.json")


def scan_result_root(
    root: Path,
    raw: dict[str, Any],
    timestamps: dict[tuple[str, str, int], str],
) -> set[str]:
    task_names = {entry.name for entry in subdirs(root)}
    for task_dir in subdirs(root):
        for policy_dir in subdirs(task_dir):
            for embodiment_dir in subdirs(policy_dir):
                for run_dir in subdirs(embodiment_dir):
                    match = SEED_RE.match(run_dir.name)
                    if match is None:
                        continue
                    latest = latest_result(run_dir)
                    if latest is None:
                        continue
                    timestamp, entries = latest
                    key = (policy_dir.name, task_dir.name, int(match.group(1)))
                    if timestamp >= timestamps.get(key, ""):
                        timestamps[key] = timestamp
                        raw[key[0]][key[1]][key[2]] = entries
    return task_names


def scan(root: Path) -> tuple[dict[str, Any], set[str]]:
    raw: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    timestamps: dict[tuple[str, str, int], str] = {}
    task_names = scan_result_root(root, raw, timestamps)

    # During a parallel sweep each worker writes its task beneath
    # <output>/<task>/RoboDojo/<task>/... . The launcher creates a top-level
    # RoboDojo symlink farm only after every worker has exited, so accept the
    # worker layout too for interruption recovery and live partial summaries.
    if root.name != "RoboDojo":
        for worker_dir in subdirs(root):
            nested_root = worker_dir / "RoboDojo"
            if nested_root.is_dir():
                task_names.update(scan_result_root(nested_root, raw, timestamps))
    return raw, task_names


def stats(entries: list[tuple[bool, float]]) -> tuple[float, float, int, int]:
    total = len(entries)
    successes = sum(success for success, _ in entries)
    return (successes / total * 100.0, sum(score for _, score in entries) / total * 100.0, successes, total) if total else (0.0, 0.0, 0, 0)


def merge_data(raw: dict[str, Any], task_names: set[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    random_of = {task: f"{task}_random" for task in GENERALIZATION_TASKS if f"{task}_random" in task_names}
    data: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    halves: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for policy, policy_raw in raw.items():
        for task in ALL_TASKS:
            base = policy_raw.get(task, {})
            if task not in random_of:
                for seed, entries in base.items():
                    data[policy][task][seed] = stats(entries)
                continue
            random = policy_raw.get(random_of[task], {})
            for seed in sorted(set(base) | set(random)):
                if seed in base:
                    halves[policy][task][seed]["standard"] = stats(base[seed])
                if seed in random:
                    halves[policy][task][seed]["random"] = stats(random[seed])
                merged = base.get(seed, []) + random.get(seed, [])
                if merged:
                    data[policy][task][seed] = stats(merged)
    return data, halves, random_of


def mean_std(values: list[float]) -> tuple[float, float] | None:
    return (sum(values) / len(values), statistics.pstdev(values) if len(values) > 1 else 0.0) if values else None


def fmt_stat(value: tuple[float, float] | None, integer: bool = False) -> str:
    if value is None:
        return ""
    fmt = "{:.0f}" if integer else "{:.2f}"
    return f"{fmt.format(value[0])} ±{fmt.format(value[1])}"


def dimension_seed(policy_data: dict[str, Any], tasks: list[str], seed: int) -> tuple[float, float] | None:
    cells = [policy_data.get(task, {}).get(seed) for task in tasks]
    cells = [cell for cell in cells if cell is not None]
    return (sum(cell[0] for cell in cells) / len(cells), sum(cell[1] for cell in cells) / len(cells)) if cells else None


def overview(data: dict[str, Any]) -> list[str]:
    lines = [
        "## Overview (by capability dimension)", "",
        "Progress † marks incomplete evaluation (126 = 42 tasks × 3 seeds). Cells show `mean ±std` across seeds and use every available episode (no minimum threshold). `Episodes` shows total (successes/episodes) aggregated across all completed (task, seed) cells. Policies are sorted by overall Avg SR.",
        "",
    ]
    header = ["Policy", "Progress", "Episodes"]
    for dimension in DIMENSIONS:
        header.extend((f"{dimension} SR (%)", f"{dimension} Score"))
    header.extend(("Avg SR (%)", "Avg Score"))
    lines.extend(("| " + " | ".join(header) + " |", "| --- | " + " | ".join(["---:"] * (len(header) - 1)) + " |"))
    rows = []
    for policy, policy_data in data.items():
        filled = sum(seed in policy_data.get(task, {}) for task in ALL_TASKS for seed in EXPECTED_SEEDS)
        if not filled:
            continue
        successes = sum(cell[2] for task in ALL_TASKS for cell in policy_data.get(task, {}).values())
        episodes = sum(cell[3] for task in ALL_TASKS for cell in policy_data.get(task, {}).values())
        total = len(ALL_TASKS) * len(EXPECTED_SEEDS)
        cells = [policy, "100%" if filled == total else f"{filled / total * 100:.1f}% † ({filled}/{total})", f"{successes}/{episodes}"]
        per_seed_overall = []
        for dimension, tasks in DIMENSIONS.items():
            values = [dimension_seed(policy_data, tasks, seed) for seed in EXPECTED_SEEDS]
            values = [value for value in values if value is not None]
            cells.extend((fmt_stat(mean_std([value[0] for value in values])), fmt_stat(mean_std([value[1] for value in values]))))
        for seed in EXPECTED_SEEDS:
            values = [dimension_seed(policy_data, tasks, seed) for tasks in DIMENSIONS.values()]
            values = [value for value in values if value is not None]
            if values:
                per_seed_overall.append((sum(value[0] for value in values) / len(values), sum(value[1] for value in values) / len(values)))
        cells.extend((fmt_stat(mean_std([value[0] for value in per_seed_overall])), fmt_stat(mean_std([value[1] for value in per_seed_overall]))))
        rows.append((-(mean_std([value[0] for value in per_seed_overall]) or (-1.0, 0.0))[0], policy, cells))
    for _, _, cells in sorted(rows):
        lines.append("| " + " | ".join(cells) + " |")
    print("\n".join(lines[4:]) + "\n")
    return lines + [""]


def generalization_split(halves: dict[str, Any]) -> list[str]:
    lines = [
        "## Generalization: Standard vs Random", "",
        "Each cell shows `mean ±std` across seeds, averaged over the 12 Generalization tasks with completed data for that half. Δ = Random − Standard (mean SR). Sorted by Standard SR. Statistics use every available episode (no minimum threshold).", "",
        "| Policy | Standard SR (%) | Standard Score | Random SR (%) | Random Score | Δ SR | Δ Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in sorted(halves):
        values: dict[str, tuple[tuple[float, float] | None, tuple[float, float] | None]] = {}
        for half in ("standard", "random"):
            per_seed = []
            for seed in EXPECTED_SEEDS:
                cells = [halves[policy][task].get(seed, {}).get(half) for task in DIMENSIONS["Generalization"]]
                cells = [cell for cell in cells if cell is not None]
                if cells:
                    per_seed.append((sum(cell[0] for cell in cells) / len(cells), sum(cell[1] for cell in cells) / len(cells)))
            values[half] = (mean_std([item[0] for item in per_seed]), mean_std([item[1] for item in per_seed]))
        standard, random = values["standard"], values["random"]
        if standard[0] is None and random[0] is None:
            continue
        cells = [policy, fmt_stat(standard[0]), fmt_stat(standard[1]), fmt_stat(random[0]), fmt_stat(random[1])]
        if standard[0] and random[0]:
            cells.extend((f"{random[0][0] - standard[0][0]:+.2f}", f"{random[1][0] - standard[1][0]:+.2f}"))
        else:
            cells.extend(("", ""))
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def task_row(task: str, task_data: dict[int, tuple[float, float, int, int]]) -> list[str]:
    cells, srs, scores = [f"<nobr>{task}</nobr>"], [], []
    for seed in EXPECTED_SEEDS:
        cell = task_data.get(seed)
        if cell is None:
            cells.extend(("", "", ""))
            continue
        sr, score, successes, episodes = cell
        cells.extend((f"{sr:.0f}", f"{score:.2f}", f"{successes}/{episodes}"))
        srs.append(sr)
        scores.append(score)
    cells.extend((fmt_stat(mean_std(srs), integer=True) if len(srs) == len(EXPECTED_SEEDS) else "", fmt_stat(mean_std(scores)) if len(scores) == len(EXPECTED_SEEDS) else ""))
    return cells


def average_row(policy_data: dict[str, Any], tasks: list[str], label: str) -> list[str]:
    cells, srs, scores = [label], [], []
    for seed in EXPECTED_SEEDS:
        values = [policy_data.get(task, {}).get(seed) for task in tasks]
        values = [value for value in values if value is not None]
        if not values:
            cells.extend(("", "", ""))
            continue
        sr = sum(value[0] for value in values) / len(values)
        score = sum(value[1] for value in values) / len(values)
        cells.extend((f"{sr:.1f}", f"{score:.2f}", f"{sum(value[2] for value in values)}/{sum(value[3] for value in values)}"))
        srs.append(sr)
        scores.append(score)
    cells.extend((fmt_stat(mean_std(srs), integer=True), fmt_stat(mean_std(scores))))
    return cells


def policy_table(policy: str, policy_data: dict[str, Any], halves: dict[str, Any], random_of: dict[str, str]) -> list[str]:
    lines = [f"## {policy}", "", "Per-seed columns: `SR (%)`, `Score`, `n` (successes/episodes actually observed for that seed).", ""]
    header = ["Task"] + [item for seed in EXPECTED_SEEDS for item in (f"Seed {seed} SR (%)", f"Seed {seed} Score", f"Seed {seed} n")] + ["Avg SR (%)", "Avg Score"]
    lines.extend(("| " + " | ".join(header) + " |", "| --- | " + " | ".join(["---:"] * (len(header) - 1)) + " |"))
    for dimension, tasks in DIMENSIONS.items():
        lines.append("| " + " | ".join([f"**{dimension}**"] + [""] * (len(header) - 1)) + " |")
        for task in tasks:
            lines.append("| " + " | ".join(task_row(task, policy_data.get(task, {}))) + " |")
            if task in random_of:
                for half, label in (("standard", f"{task} (standard)"), ("random", random_of[task])):
                    split = {seed: halves.get(task, {}).get(seed, {}).get(half) for seed in EXPECTED_SEEDS}
                    split = {seed: cell for seed, cell in split.items() if cell is not None}
                    if split:
                        lines.append("| " + " | ".join(task_row(label, split)) + " |")
        lines.append("| " + " | ".join(average_row(policy_data, tasks, f"**{dimension} Avg (per seed)**")) + " |")
    lines.append("| " + " | ".join(average_row(policy_data, list(ALL_TASKS), "**Avg (per seed)**")) + " |")
    return lines + [""]


def render(data: dict[str, Any], halves: dict[str, Any], random_of: dict[str, str]) -> str:
    lines = ["# RoboDojo Evaluation Summary", ""] + overview(data) + generalization_split(halves)
    for policy in sorted(data):
        lines.extend(policy_table(policy, data[policy], halves.get(policy, {}), random_of))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    root = result_root(run_dir)
    if not root.is_dir():
        raise SystemExit(f"Result directory not found: {root}")
    try:
        raw, task_names = scan(root)
        data, halves, random_of = merge_data(raw, task_names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not data:
        raise SystemExit(f"No _result.json files found below {root}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(data, halves, random_of), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps({"run_dir": str(run_dir), "result_root": str(root), "policies": sorted(data)}, indent=2) + "\n", encoding="utf-8")
    print(f"Input: {root}")
    print(f"Policies: {len(data)}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
