"""Seed-0 WALL-OSS ablation report; whole-task completeness fallback, never max.

Read-only CLI: consumes the runner's summary.json (PASS/exit_code/result_path)
and official WallX results at ONE explicitly named checkpoint and embodiment.
Prints JSON; never launches evaluation or edits RoboDojo's summarizer/results.
The upstream summarizer supplies task membership, episode counts and statistics.
"""

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import re


SUMMARIZER = Path(__file__).resolve().parents[2] / "third_party/RoboDojo/scripts/internal/summarize_result.py"
_spec = importlib.util.spec_from_file_location("_robodojo_summary_rules", SUMMARIZER)
_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rules)


@dataclass(frozen=True)
class RunResult:
    """One task/seed run's layout-sorted entries and run-level errors.

    Failed episodes are ordinary (False, score) entries, not run errors.
    An absent/unreadable run is represented by None or errors, never a zero score.
    """

    entries: tuple[tuple[bool, float], ...] = ()
    errors: tuple[str, ...] = ()
    path: str = ""


def _whole_task(results, task):
    halves = (task, task + "_random") if task in _rules.GENERALIZATION_TASKS else (task,)
    required = _rules.PAIRED_HALF_EPISODES if len(halves) == 2 else _rules.STANDALONE_EPISODES
    entries, paths, reasons = [], [], []
    for half in halves:
        run = results.get(half, {}).get(0)  # no averaging over other seeds
        if run is None:
            reasons.append(f"{half}: missing seed 0")
            continue
        if run.errors:
            reasons.extend(f"{half}: {error}" for error in run.errors)
        if len(run.entries) < required:
            reasons.append(f"{half}: {len(run.entries)}/{required} episodes")
        entries.extend(run.entries[:required])
        paths.append(run.path)
    return entries, paths, reasons


def select_seed0(harness, base, *, tasks=None):
    """Complete, error-free harness task wins irrespective of either score.

    Otherwise use the ENTIRE official task, same seed (including both halves).
    Missing/invalid base is an error: never omit a task from the denominator.
    """
    rows = []
    for task in (_rules.ALL_TASKS if tasks is None else tasks):
        if task.endswith("_random"):
            raise ValueError("report base tasks only; random halves are merged")
        entries, paths, reasons = _whole_task(harness, task)
        source = "harness+wall"
        if reasons:
            entries, paths, base_errors = _whole_task(base, task)
            if base_errors:
                raise ValueError(f"official WallX seed-0 fallback unavailable for {task}: {'; '.join(base_errors)}")
            source = "WallX"
        count, success_rate, score = _rules.stats(entries)
        rows.append({"task": task, "seed": 0, "source": source, "episodes": count,
                     "success_rate": success_rate, "score": score,
                     "fallback_reasons": reasons, "result_paths": paths})
    # Official overall is the unweighted mean of the five dimension means, the
    # same function summarize_result.overall_seed_value uses. A flat mean of the
    # 42 tasks weights Generalization (12) above Memory (6).
    policy_data = {row["task"]: {0: (row["success_rate"], row["score"])} for row in rows}
    dimensions = {}
    for name, members in _rules.DIMENSIONS.items():
        value = _rules.dimension_seed_value(policy_data, members, 0)
        if value is not None:
            dimensions[name] = {"success_rate": value[0], "score": value[1],
                                "tasks": sum(1 for task in members if task in policy_data)}
    overall = _rules.overall_seed_value(policy_data, 0)
    return {"seed": 0, "tasks": rows, "task_count": len(rows),
            "dimensions": dimensions,
            "success_rate": None if overall is None else overall[0],
            "score": None if overall is None else overall[1]}


def read_result(path, *, errors=()):
    """Use upstream layout sorting; reject malformed records/run errors first."""
    path = Path(path)
    problems = list(errors)
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("details"), dict):
            raise ValueError("missing details mapping")
        for key in ("error", "errors", "run_error", "run_errors"):
            if data.get(key):
                problems.append(f"{key}: {data[key]}")
        if data.get("status") in ("error", "FAIL", "interrupted") or data.get("exit_code", 0) not in (0, "0"):
            problems.append("run status/exit_code indicates an error")
        layouts = set()
        for layout, entry in data["details"].items():
            index = int(layout)
            if index in layouts:
                raise ValueError("duplicate normalized layout id")
            layouts.add(index)
            if not isinstance(entry, dict) or type(entry.get("success")) is not bool:
                raise ValueError("invalid success entry")
            score = entry.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("invalid score entry")
        entries = _rules.load_entries(str(path.parent))
        if entries is None:
            raise ValueError("upstream could not read entries")
        return RunResult(tuple(entries), tuple(problems), str(path))
    except (OSError, ValueError, TypeError) as error:
        return RunResult(errors=tuple(problems + [f"unreadable result: {error}"]), path=str(path))


def _path_identity(path, *, policy, embodiment, checkpoint=None):
    """Check task/policy/embodiment/seed/action mode without opening any other run."""
    path = Path(path)
    if path.name != "_result.json" or len(path.parts) < 6:
        raise ValueError("not a RoboDojo _result.json path")
    task, found_policy, found_embodiment, run, _timestamp = path.parts[-6:-1]
    if found_policy != policy or found_embodiment != embodiment:
        raise ValueError("result policy/embodiment mismatch")
    match = re.fullmatch(r"(\d+)_ckpt_name=(.+),action_type=ee", run)
    if match is None or (checkpoint is not None and match[2] != checkpoint):
        raise ValueError("result must use EE actions at the declared checkpoint")
    return task, int(match[1])


def read_harness_summary(path, *, policy="HarnessPolicyWall", embodiment="arx_x5"):
    """Latest runner row per task, including FAIL rows (never score-select)."""
    summary = json.loads(Path(path).read_text())
    results = {}
    for row in summary["results"]:
        task = row["task"]
        # Ignore other seeds even if their action mode/result is malformed.
        run_name = Path(row.get("result_path") or "").parent.parent.name
        seed_match = _rules.SEED_RE.match(run_name)
        if seed_match and int(seed_match[1]) != 0:
            continue
        problems = []
        if row.get("status") != "PASS" or str(row.get("exit_code")) != "0":
            problems.append(f"runner {row.get('status')}, exit={row.get('exit_code')}")
        for key in ("error", "errors", "run_error", "run_errors"):
            if row.get(key):
                problems.append(f"{key}: {row[key]}")
        try:
            result_task, seed = _path_identity(row.get("result_path") or "", policy=policy, embodiment=embodiment)
            if result_task != task:
                raise ValueError("result task mismatch")
        except ValueError as error:
            results.setdefault(task, {})[0] = RunResult(errors=tuple(problems + [str(error)]))
            continue
        if seed != 0:
            continue
        results.setdefault(task, {})[0] = read_result(row["result_path"], errors=problems)
    return results


def read_official(root, checkpoint, *, embodiment="arx_x5"):
    """WallX only, fixed checkpoint/embodiment/seed/mode; latest timestamp, not best."""
    results = {}
    tasks = _rules.ALL_TASKS + [task + "_random" for task in sorted(_rules.GENERALIZATION_TASKS)]
    for task in tasks:
        run_dir = Path(root) / task / "WallX" / embodiment / f"0_ckpt_name={checkpoint},action_type=ee"
        latest = _rules.latest_timestamp_dir(str(run_dir))
        if latest is not None:
            results[task] = {0: read_result(Path(latest) / "_result.json")}
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-summary", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--base-ckpt", required=True, help="one official WallX checkpoint for every task")
    parser.add_argument("--embodiment", default="arx_x5")
    args = parser.parse_args(argv)
    report = select_seed0(read_harness_summary(args.harness_summary, embodiment=args.embodiment),
                          read_official(args.base_root, args.base_ckpt, embodiment=args.embodiment))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
