"""55-line MIXED-SOURCE diagnostic, never a frozen official/paired result.

Public W + local WALL/Codex historical task means are floors. Latest selective
reruns REPLACE the whole task, even if lower. Gen is standard+random, 25+25;
standalone is 50. Incomplete reruns retain the native denominator (missing=0)
and cannot certify the line. Conditional targets never enter measured composite.
"""

import argparse
import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

CENTRAL = {"push_T": 65, "arrange_largest_number": 75, "play_Xylophone": 60,
           "swap_blocks": 50, "press_by_number": 90, "align_blocks": 65,
           "general_pickup": 85, "solve_equation": 70, "classify_objects_by_language": 65,
           "pack_objects_into_box": 45, "store_laptop_and_headphones": 60}
# Sweep is high-risk optional Tier2; its 35 target accounts for +0.4278 composite
# in the old central ~55 plan. No unseen P2/Simate premiums.
OPTIONAL = {"sweep_blocks": 35}
HIST_NAMES = ("wall-oss-seed0", "wall-oss-codex-seed0", "wall-oss-codex-recovery-seed0",
              "wall-oss-codex-hack-seed0", "wall-oss-codex-skills-seed0")
SELECTIVE_AGENT_VERSIONS = {"1.13.0", "1.13.1", "1.14.0", "1.15.0", "1.16.0", "1.17.0", "1.18.0", "1.19.0", "1.20.0", "1.21.0"}


def dimensions(robodojo):
    tree = ast.parse((robodojo / "scripts/internal/summarize_result.py").read_text())
    dims = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "DIMENSIONS" for t in n.targets))
    if [len(v) for v in dims.values()] != [12, 8, 8, 6, 8] or len({t for v in dims.values() for t in v}) != 42:
        raise ValueError("native dimension membership changed")
    return dims


def public_floors(path, dims):
    rows = re.findall(r"^\| (Gen|Prec|LH|Mem|Open) / (\w+) \| ([\d.]+) \|", path.read_text(), re.M)
    floors = {task: float(value) for _, task, value in rows}
    expected = {task for members in dims.values() for task in members}
    if set(floors) != expected or len(rows) != 42:
        raise ValueError("W table must contain exactly all 42 tasks")
    return floors


def read_entries(path, required):
    try:
        data = json.loads(path.read_text())
        details = data["details"]
        if not isinstance(details, dict):
            raise ValueError("details must be a mapping")
        values = {}
        for layout, row in details.items():
            index = int(layout)
            score = float(row["score"])
            if index in values or index < 0 or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid/duplicate layout or score")
            values[index] = score
        # Native layout IDs 0..required-1; never replace failed cases by later IDs.
        selected = [values[i] for i in range(required) if i in values]
        errors = [str(data[k]) for k in ("error", "errors", "run_error", "run_errors") if data.get(k)]
        if data.get("status") in ("FAIL", "error", "interrupted") or data.get("exit_code", 0) not in (0, "0"):
            errors.append("runner error")
        return selected, errors
    except (OSError, ValueError, TypeError, KeyError) as error:
        return [], [f"unreadable result {path}: {error}"]


def scan(root, tasks, *, policy=None):
    """One latest timestamp per task/seed0, not best repeated score. No videos/traces."""
    result_root = root / "eval_result/RoboDojo"
    found = {}
    for task in tasks:
        folder = result_root / task
        if not folder.is_dir():
            continue
        candidates = []
        for path in folder.glob("*/arx_x5/0_ckpt_name=*,action_type=*/*/_result.json"):
            if policy and (path.parts[-5] != policy or path.parts[-3] != f"0_ckpt_name={policy},action_type=ee"):
                continue
            if path.parts[-5] not in {"HarnessPolicyWall", "HarnessPolicyWallCodex", "HarnessPolicyWallCodexRecovery",
                                       "HarnessPolicyWallCodex1911", "HarnessPolicyWallCodexHack", "HarnessPolicyWallCodexSkills"} and policy is None:
                continue
            candidates.append(path)
        if candidates:
            found[task] = max(candidates, key=lambda p: (p.parent.name, str(p)))
    return found


def whole_task(paths, task, gen, *, conservative=False):
    halves = (task, task + "_random") if task in gen else (task,)
    required = 25 if len(halves) == 2 else 50
    observed, errors, scores, sources = 0, [], [], []
    complete = True
    for half in halves:
        path = paths.get(half)
        entries, problems = read_entries(path, required) if path else ([], [half + ": missing"])
        observed += len(entries)
        errors.extend(problems)
        complete &= len(entries) == required and not problems
        if path:
            sources.append(str(path))
        if conservative:
            scores.append(sum(entries) / required * 100)
        elif entries and not problems:
            scores.append(sum(entries) / len(entries) * 100)
        else:
            return None
    return {"score": sum(scores) / len(halves), "n": observed, "complete": bool(complete),
            "errors": errors, "paths": sources}


def means(rows, dims, column):
    dim = {name: sum(rows[t][column] for t in members) / len(members) for name, members in dims.items()}
    return dim, sum(dim.values()) / 5


def build(args):
    dims = dimensions(args.robodojo)
    gen = set(dims["Generalization"])
    tasks = [t for members in dims.values() for t in members]
    halves = tasks + [t + "_random" for t in gen]
    public = public_floors(args.public, dims)
    rows = {t: {"floor": public[t], "floor_source": "public W", "floor_n": "aggregate",
                "target": CENTRAL.get(t), "measured": None, "n": 0, "status": "FLOOR"} for t in tasks}
    issues = []
    hist_roots = args.hist_root if args.hist_root is not None else [args.machine / n for n in HIST_NAMES]
    for root in hist_roots:
        if not root.is_dir():
            issues.append("optional historical root absent: " + str(root))
            continue
        paths = scan(root, halves)
        for task in tasks:
            hist = whole_task(paths, task, gen)
            if hist and hist["score"] > rows[task]["floor"]:
                rows[task].update(floor=hist["score"], floor_source=root.name + (" (partial history)" if not hist["complete"] else ""),
                                  floor_n=hist["n"])
    # Explicit optional Tier2 can be planned, but neither target is a measurement.
    if args.tier2:
        for task, target in OPTIONAL.items():
            rows[task]["target"] = target
    new_roots = args.new_root if args.new_root is not None else [args.machine / "wall-oss-codex-skills-55line-seed0"]
    rerun_tasks = set()
    for root in new_roots:
        if not root.is_dir():
            continue
        plan_path = root / "rerun-plan.json"
        if not plan_path.is_file():
            issues.append("new root ignored without frozen rerun-plan.json: " + str(root))
            continue
        plan = json.loads(plan_path.read_text())
        policy = plan["policy"]
        if plan.get("agent_version") not in SELECTIVE_AGENT_VERSIONS or plan.get("seed") != 0 or plan.get("eval_num") != "native":
            raise ValueError("new results need frozen 1.13.0/1.13.1/1.14.0/1.15.0/1.16.0/1.17.0/1.18.0 seed0/native identity")
        if not policy.startswith("HarnessPolicyWallCodexSkills55") or not plan.get("identity"):
            raise ValueError("wrong/missing independent policy identity")
        planned = set(plan["tasks"])
        if not planned <= set(halves):
            raise ValueError("unknown task in rerun plan")
        for task in planned:
            base = task.removesuffix("_random")
            if base in gen and not {base, base + "_random"} <= planned:
                raise ValueError("new Gen plan requires both halves")
        planned_base = {t.removesuffix("_random") for t in planned}
        paths = scan(root, halves, policy=policy)
        # Runner FAIL is not discarded, including valid result files on error exit.
        summary_path = root / "summary.json"
        runner_errors = set()
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            for row in summary.get("results", []):
                if row.get("status") != "PASS" or str(row.get("exit_code")) != "0":
                    runner_errors.add(row["task"].removesuffix("_random"))
        for task in planned_base:
            measured = whole_task(paths, task, gen, conservative=True)
            complete = measured["complete"] and task not in runner_errors
            rows[task].update(measured=measured["score"], n=measured["n"],
                              status="MEASURED" if complete else "PENDING/FAILED (native-denominator lower bound)",
                              measured_source=root.name, identity=plan["identity"], measured_paths=measured["paths"])
            if measured["errors"]:
                issues.extend(measured["errors"])
        rerun_tasks.update(planned_base)
    for row in rows.values():
        row["selected"] = row["floor"] if row["measured"] is None else row["measured"]
        row["conditional"] = max(row["floor"], row["target"] or 0)
        row["gap"] = None if row["target"] is None else row["target"] - row["selected"]
    return dims, rows, rerun_tasks, issues


def render(args):
    dims, rows, rerun, issues = build(args)
    floor_dims, floor = means(rows, dims, "floor")
    selected_dims, selected = means(rows, dims, "selected")
    conditional_dims, conditional = means(rows, dims, "conditional")
    required_tier1 = set(CENTRAL)
    certified = required_tier1 <= rerun and all(rows[t]["status"] == "MEASURED" for t in rerun) and selected >= args.line
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    lines = ["# 55-line scoreboard", "", datetime.now(timezone.utc).isoformat(), "",
             "MIXED-SOURCE DIAGNOSTIC, not a frozen official or paired result. Failed cases retained.",
             "Public W is aggregate/non-paired; local hist only (no Simate/arxiv premiums).",
             "Historical partial means are labeled with n; Gen needs BOTH halves and uses their equal mean.",
             "Latest selective measurements replace whole tasks even when below floors. Missing rerun cases = 0/50; no certification while incomplete.",
             "Targets are TARGET(conditional), never measured uplift. Non-rerun tasks use max(public W, local hist).", "",
             f"Floor-only composite: {floor:.4f}; selective+floors: {selected:.4f}; delta vs {args.line:g}: {selected-args.line:+.4f}.",
             f"TARGET(conditional) Tier1{' + optional Tier2 sweep' if args.tier2 else ' only'}: {conditional:.4f}; delta: {conditional-args.line:+.4f}.",
             "Prior ~55 central plan INCLUDED optional sweep=35; without it the conditional plan is lower. No P2 lift credited.",
             "Line status: " + ("55 reached by measured selective + floors (still mixed-source)" if certified else "NOT CERTIFIED; do not claim measured 55"), "",
             "| Dimension/task | Floor | Floor source (n) | Measured | Target (conditional) | Selected | Gap to target | Status |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for dim, tasks in dims.items():
        for task in tasks:
            row = rows[task]
            lines.append(f"| {dim}/{task} | {fmt(row['floor'])} | {row['floor_source']} ({row['floor_n']}) | {fmt(row['measured'])} ({row['n']}/50) | {fmt(row['target'])} | {fmt(row['selected'])} | {fmt(row['gap'])} | {row['status']} |")
    lines += ["", "| Dimension | Floor | Selective+floors | Target (conditional) |",
              "| --- | ---: | ---: | ---: |"]
    for dim in dims:
        lines.append(f"| {dim} | {floor_dims[dim]:.4f} | {selected_dims[dim]:.4f} | {conditional_dims[dim]:.4f} |")
    lines += [f"| Composite (five equal dim means) | {floor:.4f} | {selected:.4f} | {conditional:.4f} |", "",
              "Rerun roots (in chronological order; latest planned task wins): " + ", ".join(str(p) for p in (args.new_root or [args.machine / 'wall-oss-codex-skills-55line-seed0'])),
              "", "Calls/tokens/cost per episode, paired difference and repeated-run spread: UNKNOWN; no default promotion."]
    if issues:
        lines += ["", "Input notes:"] + ["- " + issue for issue in issues]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--machine", type=Path, default=Path("/pfs/echo_folder/weiyang/tmp/harness"))
    p.add_argument("--robodojo", type=Path, default=Path(__file__).resolve().parents[1] / "third_party/RoboDojo")
    p.add_argument("--public", type=Path)
    p.add_argument("--hist-root", type=Path, action="append", help="override historical roots; repeat")
    p.add_argument("--new-root", type=Path, action="append", help="override selective roots; oldest to newest; repeat")
    p.add_argument("--markdown", type=Path)
    p.add_argument("--tier2", action="store_true", help="include conditional sweep=35 from the old ~55 central plan")
    p.add_argument("--line", type=float, default=55)
    args = p.parse_args(argv)
    args.public = args.public or args.machine / "score-target-65-design.md"
    report = render(args)
    # Machine artifact only; source/observations/results are read-only.
    (args.markdown or args.machine / "scoreboard-55line.md").write_text(report)
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
