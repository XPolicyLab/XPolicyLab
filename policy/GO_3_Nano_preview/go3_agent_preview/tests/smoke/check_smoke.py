#!/usr/bin/env python3
"""check_smoke.py -- decide whether a go3_agent_preview smoke run actually exercised both branches.

`smoke_test.sh` runs four tasks over the same small layout set. For each task this
reads RoboDojo's `_result.json` and the adapter's own `events.jsonl`, and checks
three things that fail differently:

  run        the eval completed the requested number of DISTINCT layouts, with no
             duplicated or missing layout ids, and no fatal error in the log;
  code       the code branch ran a program: `events.jsonl` records episode starts
             and code ticks, so a run that only returned zeros is caught here;
  wam        every switch that occurred was served: the backend answered
             (`wam_chunk`) and produced actions (`first_wam_action`). A task that
             never switched is reported as "not exercised", never as a pass, and
             a `wam_error` or a switch with no chunk is a failure.

Exit codes: 0 all checks pass, 2 a run did not complete, 3 a check failed,
4 a switch happened without a working backend.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# Conditions that invalidate a run outright, as opposed to merely noisy.
FATAL = [
    ("backend unreachable", r"GEAct backend request failed|Connection refused|connect call failed"),
    ("adapter import failed", r"ModuleNotFoundError: No module named 'XPolicyLab\.policy\.go3_agent_preview"),
    ("missing env_cfg", r"FileNotFoundError: .*env_cfg"),
    ("policy server never started", r"policy server did not open|exited before opening port"),
    ("switch rejected", r"skill switching requires|does not match backend robot"),
]
# Worth surfacing; does not by itself fail the run.
NOISY = [
    ("physx refill", r"\[PhysX\] broken envs="),
    ("kit restart", r"restarting \("),
    ("worker startup timeout", r"code_worker_startup_timeout"),
    ("code held after wam failure", r"holding current state"),
]

READY_CODE = r"\[go3_agent_preview\] code policy ready"
READY_WAM = r"\[GEAct\] proxy ready backend="


def read_text(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def find_result(results_root: str, task: str, run_id: str) -> str | None:
    """The sim writes eval_result/RoboDojo/<task>/<policy>/<env_cfg>/<seed>_<info>/<run_id>/."""
    hits = glob.glob(os.path.join(results_root, task, "*", "*", "*", run_id, "_result.json"))
    return sorted(hits)[-1] if hits else None


def read_events(events_root: str, task: str) -> list[dict]:
    path = os.path.join(events_root, task, "events.jsonl")
    records = []
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def check_run(result_path: str | None, log: str, want: int, seed: int) -> dict:
    """The run-level facts: distinct layouts, eval_time, and self-consistency."""
    problems, notes = [], []
    layouts: set[int] = set()
    eval_time = None
    sr = None
    if result_path is None:
        problems.append("no _result.json was written for this run")
    else:
        try:
            data = json.load(open(result_path, encoding="utf-8"))
        except (OSError, ValueError) as error:
            problems.append(f"unreadable _result.json: {error}")
            data = {}
        details = data.get("details") or {}
        for detail in details.values():
            if isinstance(detail, dict) and "layout_id" in detail:
                layouts.add(int(detail["layout_id"]))
        eval_time = data.get("eval_time")
        sr = data.get("success_rate")
        if eval_time != want:
            problems.append(f"eval_time={eval_time}, expected {want} (batches lost or run short)")
        if len(details) != eval_time:
            problems.append(f"details has {len(details)} entries but eval_time={eval_time}")
        if len(layouts) != want:
            problems.append(f"{len(layouts)} distinct layouts, expected {want}")
        elif layouts != set(range(min(layouts), min(layouts) + want)):
            notes.append(f"layouts are not contiguous: {sorted(layouts)}")

    if not re.search(READY_CODE, log):
        problems.append("log has no code-policy ready line (adapter never constructed)")
    for label, pattern in FATAL:
        if re.search(pattern, log):
            problems.append(f"{label}: matched /{pattern}/")
    for label, pattern in NOISY:
        count = len(re.findall(pattern, log))
        if count:
            notes.append(f"{label} x{count}")
    return {"layouts": sorted(layouts), "eval_time": eval_time, "sr": sr,
            "problems": problems, "notes": notes}


def check_branches(events: list[dict]) -> dict:
    """What the adapter itself says happened: code ran, and switches were served."""
    problems, notes = [], []
    starts = [e for e in events if e.get("event_type") == "episode_start"]
    code_ticks = [e for e in events if e.get("event_type") == "code_tick"]
    boundaries = [e for e in events if e.get("event_type") == "switch_boundary"]
    chunks = [e for e in events if e.get("event_type") == "wam_chunk"]
    firsts = [e for e in events if e.get("event_type") == "first_wam_action"]
    errors = [e for e in events if e.get("event_type") == "wam_error"]

    if not events:
        problems.append("no events.jsonl was written (code.out_dir not honoured, or the "
                        "policy server never reached an observation)")
        return {"starts": 0, "code_ticks": 0, "switches": 0, "wam_chunks": 0,
                "wam_errors": 0, "problems": problems, "notes": notes}
    if not starts:
        problems.append("events.jsonl has no episode_start record")
    if not code_ticks:
        problems.append("events.jsonl has no code_tick record (the code branch never ran)")

    if errors:
        latest = errors[-1].get("reason", "")
        problems.append(f"{len(errors)} wam_error record(s); last: {latest[:160]}")
    # A switch is only served if a chunk and an action followed it.
    if boundaries and not chunks:
        problems.append(f"{len(boundaries)} switch_boundary record(s) but no wam_chunk "
                        "(the backend never answered)")
    elif boundaries and not firsts:
        problems.append(f"{len(boundaries)} switch(es) but no first_wam_action")
    elif boundaries:
        notes.append(f"WAM exercised: {len(boundaries)} switch(es), {len(chunks)} chunk(s)")
    else:
        notes.append("no switch occurred: WAM not exercised on this task "
                     "(code path only; this is not a WAM pass)")

    return {"starts": len(starts), "code_ticks": len(code_ticks),
            "switches": len(boundaries), "wam_chunks": len(chunks),
            "wam_errors": len(errors), "problems": problems, "notes": notes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--events-root", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--expect-layouts", type=int, default=10)
    parser.add_argument("--expect-seed", type=int, default=0)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    tasks = [t for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        sys.exit("no tasks given")
    log_dir = args.log_dir or os.path.join(os.path.dirname(os.path.abspath(args.events_root)), "logs")

    rows, completed = [], True
    for task in tasks:
        run_id = f"smoke-{task}-s{args.expect_seed}"
        log = read_text(os.path.join(log_dir, f"{task}.log"))
        run = check_run(find_result(args.results_root, task, run_id), log,
                        args.expect_layouts, args.expect_seed)
        branches = check_branches(read_events(args.events_root, task))
        if run["eval_time"] is None:
            completed = False
        row = {"task": task, "run_id": run_id, "result": find_result(args.results_root, task, run_id),
               **run, **branches}
        row["problems"] = run["problems"] + branches["problems"]
        row["notes"] = run["notes"] + branches["notes"]
        rows.append(row)

    width = max(len(r["task"]) for r in rows)
    print(f"\n{'task':<{width}}  {'n':>3} {'distinct':>8} {'SR%':>6} {'switch':>6} {'chunk':>5}  verdict")
    for row in rows:
        n = row["eval_time"] if row["eval_time"] is not None else 0
        sr = f"{100 * row['sr']:.0f}" if isinstance(row["sr"], (int, float)) else "-"
        verdict = "ok" if not row["problems"] else "FAIL"
        print(f"{row['task']:<{width}}  {n:>3} {len(row['layouts']):>8} {sr:>6} "
              f"{row['switches']:>6} {row['wam_chunks']:>5}  {verdict}")
        for note in row["notes"]:
            print(f"{'':<{width}}  note: {note}")
        for problem in row["problems"]:
            print(f"{'':<{width}}  FAIL: {problem}")

    failed = [r for r in rows if r["problems"]]
    wam_unserved = any(r["switches"] and not r["wam_chunks"] for r in rows)
    ok = not failed
    print(f"\n{len(rows) - len(failed)}/{len(rows)} tasks passed")
    if args.json_out:
        json.dump({"tasks": rows, "ok": ok}, open(args.json_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    if ok:
        return 0
    if wam_unserved:
        return 4
    if not completed:
        return 2
    return 3


if __name__ == "__main__":
    sys.exit(main())
