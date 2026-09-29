#!/usr/bin/env python3
"""Build a paper-style RoboDojo Table-1 summary for one checkpoint/seed.

The script scans the per-run ``robodojo_eval_results`` directories,
selects the most complete (then newest) result for each task, and applies the
official RoboDojo aggregation protocol:

* Generalization: 12 tasks, each formed from 25 standard + 25 random episodes.
* Other dimensions: 50 episodes for each task.
* Dimension metrics: equal-weight mean over tasks in that dimension.
* Average: equal-weight mean over the five capability dimensions.
* Display order and cell format: score / success rate, matching paper Table 1.

Episodes that RoboDojo flagged as ``unstable`` (a layout that collapsed before
the policy acted) never reach ``_result.json``'s ``details`` map: the simulator
decrements its episode counter and re-runs them. Counting ``details`` entries
therefore reproduces the official rule that an unstable episode leaves the
denominator entirely rather than scoring as a failure.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

DIMENSIONS: dict[str, list[str]] = {
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

GENERALIZATION = set(DIMENSIONS["Generalization"])
ALL_TASKS = [task for tasks in DIMENSIONS.values() for task in tasks]
if len(ALL_TASKS) != 42 or len(set(ALL_TASKS)) != 42:
    raise RuntimeError("RoboDojo Table-1 taxonomy must contain exactly 42 unique tasks")
VALID_RESULT_TASKS = set(ALL_TASKS) | {f"{task}_random" for task in GENERALIZATION}
GENERALIZATION_HALF_EPISODES = 25
STANDALONE_EPISODES = 50


@dataclass(frozen=True)
class Episode:
    success: bool
    score: float


@dataclass(frozen=True)
class Candidate:
    task: str
    path: Path
    episodes: tuple[Episode, ...]
    mtime_ns: int


@dataclass(frozen=True)
class Metric:
    score: float
    success_rate_pct: float


def _numeric_key(value: Any) -> tuple[int, str]:
    try:
        return int(value), ""
    except (TypeError, ValueError):
        return 2**63 - 1, str(value)


def load_episodes(path: Path) -> tuple[Episode, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return ()
    details = payload.get("details")
    if not isinstance(details, dict):
        return ()
    episodes: list[Episode] = []
    for _, value in sorted(details.items(), key=lambda item: _numeric_key(item[0])):
        if not isinstance(value, dict):
            continue
        try:
            score = float(value.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        episodes.append(Episode(success=bool(value.get("success", False)), score=score))
    return tuple(episodes)


def _task_from_relative_parts(parts: tuple[str, ...]) -> str | None:
    for index, part in enumerate(parts[:-1]):
        if part == "RoboDojo" and index + 1 < len(parts):
            task = parts[index + 1]
            return task if task in VALID_RESULT_TASKS else None
    return None


def _matches_legacy_seed_and_checkpoint(
    parts: tuple[str, ...],
    seed: int,
    ckpt_name: str,
) -> bool:
    prefix = f"{seed}_"
    marker = f"ckpt_name={ckpt_name},"
    return any(part.startswith(prefix) and marker in part for part in parts)


def _nearest_run_metadata(
    path: Path,
    eval_root: Path,
    cache: dict[Path, dict[str, Any] | None],
) -> dict[str, Any] | None:
    for parent in path.parents:
        if parent == eval_root.parent:
            break
        metadata_path = parent / "run_metadata.json"
        if metadata_path in cache:
            metadata = cache[metadata_path]
        elif metadata_path.is_file():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = payload if isinstance(payload, dict) else None
            except (OSError, json.JSONDecodeError, TypeError):
                metadata = None
            cache[metadata_path] = metadata
        else:
            cache[metadata_path] = None
            metadata = None
        if metadata is not None:
            return metadata
        if parent == eval_root:
            break
    return None


def _metadata_matches(
    metadata: dict[str, Any],
    *,
    checkpoint: Path,
    ckpt_name: str,
    seed: int,
) -> bool:
    try:
        metadata_checkpoint = Path(str(metadata["checkpoint"])).expanduser().resolve()
        metadata_seed = int(metadata["seed"])
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return (
        metadata_checkpoint == checkpoint
        and metadata_seed == seed
        and str(metadata.get("ckpt_name", "")) == ckpt_name
    )


def scan_candidates(
    eval_root: Path,
    *,
    checkpoint: Path,
    ckpt_name: str,
    seed: int,
) -> dict[str, Candidate]:
    selected: dict[str, Candidate] = {}
    if not eval_root.is_dir():
        return selected
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_stem = checkpoint.stem
    run_marker = f"_{checkpoint_stem}_"
    metadata_cache: dict[Path, dict[str, Any] | None] = {}
    for path in eval_root.rglob("_result.json"):
        try:
            relative = path.relative_to(eval_root)
        except ValueError:
            continue
        parts = relative.parts
        metadata = _nearest_run_metadata(path, eval_root, metadata_cache)
        if metadata is not None:
            if not _metadata_matches(
                metadata,
                checkpoint=checkpoint,
                ckpt_name=ckpt_name,
                seed=seed,
            ):
                continue
        else:
            # Backward compatibility for existing long-form result trees.
            if (
                not parts
                or run_marker not in parts[0]
                or not _matches_legacy_seed_and_checkpoint(
                    parts,
                    seed,
                    ckpt_name,
                )
            ):
                continue
        task = _task_from_relative_parts(parts)
        if task is None:
            continue
        episodes = load_episodes(path)
        if not episodes:
            continue
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        candidate = Candidate(task=task, path=path, episodes=episodes, mtime_ns=mtime_ns)
        previous = selected.get(task)
        if previous is None or (len(candidate.episodes), candidate.mtime_ns) > (
            len(previous.episodes),
            previous.mtime_ns,
        ):
            selected[task] = candidate
    return selected


def metric(episodes: tuple[Episode, ...]) -> Metric:
    if not episodes:
        raise ValueError("cannot aggregate an empty episode list")
    return Metric(
        score=fmean(item.score for item in episodes) * 100.0,
        success_rate_pct=fmean(float(item.success) for item in episodes) * 100.0,
    )


def collect_task_metrics(
    selected: dict[str, Candidate],
) -> tuple[dict[str, Metric], dict[str, dict[str, Any]]]:
    task_metrics: dict[str, Metric] = {}
    task_status: dict[str, dict[str, Any]] = {}
    for task in ALL_TASKS:
        if task in GENERALIZATION:
            standard = selected.get(task)
            random = selected.get(f"{task}_random")
            standard_count = len(standard.episodes) if standard else 0
            random_count = len(random.episodes) if random else 0
            complete = (
                standard_count >= GENERALIZATION_HALF_EPISODES
                and random_count >= GENERALIZATION_HALF_EPISODES
            )
            combined: tuple[Episode, ...] = ()
            if complete:
                combined = (
                    standard.episodes[:GENERALIZATION_HALF_EPISODES]
                    + random.episodes[:GENERALIZATION_HALF_EPISODES]
                )
                task_metrics[task] = metric(combined)
            task_status[task] = {
                "dimension": "Generalization",
                "complete": complete,
                "standard_episodes": standard_count,
                "random_episodes": random_count,
                "required": "25 standard + 25 random",
                "metric": asdict(task_metrics[task]) if complete else None,
                "standard_source": str(standard.path) if standard else None,
                "random_source": str(random.path) if random else None,
            }
            continue

        candidate = selected.get(task)
        count = len(candidate.episodes) if candidate else 0
        complete = count >= STANDALONE_EPISODES
        if complete:
            task_metrics[task] = metric(candidate.episodes[:STANDALONE_EPISODES])
        dimension = next(name for name, tasks in DIMENSIONS.items() if task in tasks)
        task_status[task] = {
            "dimension": dimension,
            "complete": complete,
            "episodes": count,
            "required": STANDALONE_EPISODES,
            "metric": asdict(task_metrics[task]) if complete else None,
            "source": str(candidate.path) if candidate else None,
        }
    return task_metrics, task_status


def aggregate_dimensions(task_metrics: dict[str, Metric]) -> dict[str, dict[str, Any]]:
    dimensions: dict[str, dict[str, Any]] = {}
    for name, tasks in DIMENSIONS.items():
        values = [task_metrics[task] for task in tasks if task in task_metrics]
        dimensions[name] = {
            "score": fmean(value.score for value in values) if values else None,
            "success_rate_pct": (
                fmean(value.success_rate_pct for value in values) if values else None
            ),
            "completed_tasks": len(values),
            "required_tasks": len(tasks),
            "complete": len(values) == len(tasks),
        }
    available = [value for value in dimensions.values() if value["score"] is not None]
    all_dimensions_available = len(available) == len(DIMENSIONS)
    dimensions["Average"] = {
        "score": fmean(value["score"] for value in available) if all_dimensions_available else None,
        "success_rate_pct": (
            fmean(value["success_rate_pct"] for value in available)
            if all_dimensions_available
            else None
        ),
        "completed_dimensions": len(available),
        "required_dimensions": len(DIMENSIONS),
        "complete": all(value["complete"] for value in available) and all_dimensions_available,
    }
    return dimensions


def format_cell(value: dict[str, Any], *, average: bool = False) -> str:
    score = value["score"]
    success = value["success_rate_pct"]
    if average:
        completed = value["completed_dimensions"]
        required = value["required_dimensions"]
    else:
        completed = value["completed_tasks"]
        required = value["required_tasks"]
    if score is None or success is None:
        return f"— ({completed}/{required})"
    suffix = "" if value["complete"] else f" † ({completed}/{required})"
    return f"{score:.2f} / {success:.2f}%{suffix}"


def build_markdown(
    *,
    checkpoint: Path,
    ckpt_name: str,
    seed: int,
    dimensions: dict[str, dict[str, Any]],
    task_status: dict[str, dict[str, Any]],
) -> str:
    ordered_columns = [*DIMENSIONS.keys(), "Average"]
    complete_tasks = sum(bool(value["complete"]) for value in task_status.values())
    protocol_complete = bool(dimensions["Average"]["complete"])
    lines = [
        "# RoboDojo Simulation Summary (Table 1 format)",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Evaluation key: `{ckpt_name}`; policy seed: `{seed}`",
        f"- Official task coverage: **{complete_tasks}/42**",
        f"- Paper protocol complete: **{'yes' if protocol_complete else 'no'}**",
        "- Cell format: **score / success rate**. `†` means a partial dimension; "
        "it must not be quoted as the final benchmark number.",
        "",
        "| Algorithm / Checkpoint | " + " | ".join(ordered_columns) + " |",
        "| --- | " + " | ".join(["---:"] * len(ordered_columns)) + " |",
    ]
    cells = [
        format_cell(dimensions[name], average=(name == "Average"))
        for name in ordered_columns
    ]
    lines.append(f"| {ckpt_name} | " + " | ".join(cells) + " |")
    lines += ["", "## Coverage by capability", ""]
    lines.append("| Dimension | Completed | Score | Success | Status |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for name in DIMENSIONS:
        value = dimensions[name]
        score = "—" if value["score"] is None else f"{value['score']:.2f}"
        success = "—" if value["success_rate_pct"] is None else f"{value['success_rate_pct']:.2f}%"
        status = "complete" if value["complete"] else "partial"
        lines.append(
            f"| {name} | {value['completed_tasks']}/{value['required_tasks']} | "
            f"{score} | {success} | {status} |"
        )

    lines += ["", "## Per-task protocol status", ""]
    lines.append("| Dimension | Task | Episodes | Score / Success | Status |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for dimension, tasks in DIMENSIONS.items():
        for task in tasks:
            status = task_status[task]
            if dimension == "Generalization":
                episodes = f"{status['standard_episodes']} std + {status['random_episodes']} random"
            else:
                episodes = f"{status['episodes']}/{status['required']}"
            task_metric = status["metric"]
            result = (
                f"{task_metric['score']:.2f} / {task_metric['success_rate_pct']:.2f}%"
                if task_metric
                else "—"
            )
            state = "complete" if status["complete"] else "incomplete"
            lines.append(f"| {dimension} | `{task}` | {episodes} | {result} | {state} |")
    return "\n".join(lines) + "\n"


def write_csv(
    path: Path,
    *,
    checkpoint: Path,
    ckpt_name: str,
    seed: int,
    dimensions: dict[str, dict[str, Any]],
) -> None:
    names = [*DIMENSIONS.keys(), "Average"]
    fieldnames = ["algorithm", "checkpoint", "seed", "paper_protocol_complete"]
    for name in names:
        key = name.lower().replace("-", "_")
        fieldnames += [
            f"{key}_score",
            f"{key}_success_rate_pct",
            f"{key}_coverage",
        ]
    row: dict[str, Any] = {
        "algorithm": ckpt_name,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "paper_protocol_complete": dimensions["Average"]["complete"],
    }
    for name in names:
        value = dimensions[name]
        key = name.lower().replace("-", "_")
        row[f"{key}_score"] = "" if value["score"] is None else f"{value['score']:.6f}"
        row[f"{key}_success_rate_pct"] = (
            "" if value["success_rate_pct"] is None else f"{value['success_rate_pct']:.6f}"
        )
        if name == "Average":
            row[f"{key}_coverage"] = (
                f"{value['completed_dimensions']}/{value['required_dimensions']}"
            )
        else:
            row[f"{key}_coverage"] = f"{value['completed_tasks']}/{value['required_tasks']}"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for table1_seedN.{md,csv,json}; defaults to eval-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    eval_root = args.eval_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else eval_root
    )
    selected = scan_candidates(
        eval_root,
        checkpoint=checkpoint,
        ckpt_name=args.ckpt_name,
        seed=args.seed,
    )
    task_metrics, task_status = collect_task_metrics(selected)
    dimensions = aggregate_dimensions(task_metrics)

    prefix = output_dir / f"table1_seed{args.seed}"
    markdown_path = prefix.with_suffix(".md")
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown(
        checkpoint=checkpoint,
        ckpt_name=args.ckpt_name,
        seed=args.seed,
        dimensions=dimensions,
        task_status=task_status,
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    write_csv(
        csv_path,
        checkpoint=checkpoint,
        ckpt_name=args.ckpt_name,
        seed=args.seed,
        dimensions=dimensions,
    )
    payload = {
        "checkpoint": str(checkpoint),
        "ckpt_name": args.ckpt_name,
        "seed": args.seed,
        "paper_protocol_complete": dimensions["Average"]["complete"],
        "dimensions": dimensions,
        "tasks": task_status,
        "selected_result_files": {
            task: {"path": str(value.path), "episodes": len(value.episodes)}
            for task, value in sorted(selected.items())
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    columns = [*DIMENSIONS.keys(), "Average"]
    print("[RoboDojo][Table1] score / success rate")
    print(" | ".join(columns))
    print(
        " | ".join(
            format_cell(dimensions[name], average=(name == "Average"))
            for name in columns
        )
    )
    print(f"[RoboDojo][Table1] markdown={markdown_path}")
    print(f"[RoboDojo][Table1] csv={csv_path}")
    print(f"[RoboDojo][Table1] json={json_path}")


if __name__ == "__main__":
    main()
