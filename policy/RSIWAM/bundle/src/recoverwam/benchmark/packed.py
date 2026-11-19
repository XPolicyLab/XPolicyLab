#!/usr/bin/env python3
"""Packed RoboDojo evaluation: many simulator clients per GPU, shared servers.

Every task is split into disjoint layout-id chunks from the start.  A fixed pool
of worker slots (simulator GPU + policy port, several slots per GPU and per
policy server) pulls chunk jobs longest-first.  Each job is an independent
RoboDojo client with its own run id and resume manifest (the same shard
mechanism as ``robodojo_dynamic_tail.py``).  When a task finishes with fewer
stable layouts than its native target (unstable layouts), top-up jobs run the
next unused layout ids.  The canonical per-task result selects the lowest
``target`` stable layout ids, matching the dynamic-tail merge contract, and is
written where ``audit_robodojo_official54_run.py`` expects it.

Policy servers must already be running and reachable from the simulator host
on ``127.0.0.1:<policy_port>``; the servers must isolate per-client state
(XPolicyLab RecoverWAM virtual env ids).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from . import scheduler_io as tail


def log(root: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (root / "scheduler.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def job_run_id(config: dict[str, Any], task: str, index: int) -> str:
    return f"{config['run_id']}_packed_{task}_{index:03d}"


def job_paths(config: dict[str, Any], task: str, run_id: str) -> dict[str, str]:
    parent = tail.result_parent(config, task)
    root = Path(config["output_root"])
    return {
        "manifest": str(parent / f"_resume_{run_id}.json"),
        "result": str(parent / run_id / "_result.json"),
        "log": str(root / "logs" / f"{run_id}.log"),
        "status": str(root / "status" / f"{run_id}.rc"),
    }


def write_job_manifest(config: dict[str, Any], job: dict[str, Any], layout_count: int) -> None:
    task = job["task"]
    run_id = job["run_id"]
    save_dir = (
        Path("eval_result")
        / "RoboDojo"
        / task
        / config.get("policy_name", "RecoverWAM")
        / config.get("env_cfg", "arx_x5")
        / f"{int(config.get('seed', 0))}_ckpt_name={config['ckpt_name']},action_type={config['action_type']}"
        / run_id
    )
    manifest = {
        "run_id": run_id,
        "save_dir": str(save_dir),
        "task_name": task,
        "policy_name": config.get("policy_name", "RecoverWAM"),
        "config_name": config.get("env_cfg", "arx_x5"),
        "eval_seed": int(config.get("seed", 0)),
        "additional_info": f"ckpt_name={config['ckpt_name']},action_type={config['action_type']}",
        "success_nums": 0,
        "fail_nums": 0,
        "unstable_nums": 0,
        "total_score": 0.0,
        "completed_layout_ids": [],
        "abandoned_layout_ids": sorted(set(range(layout_count)) - set(job["layouts"])),
        "details": {},
        "restart_count": 0,
    }
    tail.atomic_json(Path(job["manifest"]), manifest)


def new_job(config, plan, task, layouts, estimate) -> dict[str, Any]:
    index = len([j for j in plan["jobs"] if j["task"] == task])
    run_id = job_run_id(config, task, index)
    job = {
        "task": task,
        "index": index,
        "run_id": run_id,
        "layouts": list(layouts),
        "estimate": float(estimate),
        "state": "pending",
        "attempts": 0,
        "slot": None,
        **job_paths(config, task, run_id),
    }
    write_job_manifest(config, job, plan["tasks"][task]["layout_count"])
    plan["jobs"].append(job)
    return job


def build_plan(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["output_root"])
    plan_path = root / "plan.json"
    if plan_path.exists():
        return tail.read_json(plan_path)
    for sub in ("logs", "status"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    tasks = tail.load_tasks(Path(config["tasks_file"]))
    official = set(tail.load_tasks(Path(config["official_tasks_file"])))
    elapsed = tail.reference_seconds(config)
    chunk = int(config.get("chunk_layouts", 12))
    plan: dict[str, Any] = {"config_snapshot": config, "tasks": {}, "jobs": []}
    for task in tasks:
        target = tail.native_target(task, official)
        count = tail.layout_count(config, task)
        if count < target:
            raise RuntimeError(f"{task}: {count} layouts < target {target}")
        per_layout = elapsed.get(task, float(target)) / max(target, 1)
        plan["tasks"][task] = {
            "target": target,
            "layout_count": count,
            "next_layout": target,
            "seconds_per_layout": per_layout,
            "state": "running",
        }
        groups = math.ceil(target / chunk)
        for layouts in tail.split_even(list(range(target)), groups):
            new_job(config, plan, task, layouts, per_layout * len(layouts))
    tail.atomic_json(plan_path, plan)
    return plan


def save_plan(config: dict[str, Any], plan: dict[str, Any]) -> None:
    tail.atomic_json(Path(config["output_root"]) / "plan.json", plan)


def stable_layouts(job: dict[str, Any]) -> dict[int, dict[str, Any]]:
    path = Path(job["result"])
    if not path.is_file():
        return {}
    details = tail.details_by_layout(tail.read_json(path))
    return {lid: row for lid, row in details.items() if lid in set(job["layouts"])}


def launch(
    config: dict[str, Any], job: dict[str, Any], slot: dict[str, Any], slot_index: int
) -> None:
    status = Path(job["status"])
    if status.exists():
        status.rename(status.with_suffix(f".rc.attempt{job['attempts']}"))
    shard = {
        "task": job["task"],
        "run_id": job["run_id"],
        "log": job["log"],
        "status": job["status"],
        "session": f"{config['session_prefix']}_s{slot_index:02d}",
        # A slot may point at a second simulator host; the policy port must
        # be reachable there (reverse SSH tunnel from the policy host).
        "ssh_host": slot.get("ssh_host", config.get("sim_ssh_host", "local")),
        "ssh_port": int(slot.get("ssh_port", config.get("sim_ssh_port", 22))),
        "policy_port": int(slot["policy_port"]),
        "gpu": int(slot["gpu"]),
        "host_label": slot.get("host_label", "sim"),
    }
    tail.launch_shard(config, shard)
    job.update(state="running", slot=slot_index, started_at=time.time())
    job["attempts"] += 1


def finalize_task(config, plan, task, root) -> None:
    info = plan["tasks"][task]
    jobs = [j for j in plan["jobs"] if j["task"] == task]
    stable: dict[int, dict[str, Any]] = {}
    for job in jobs:
        for lid, row in stable_layouts(job).items():
            if lid in stable:
                raise RuntimeError(f"duplicate layout {task}:{lid}")
            stable[lid] = row
    target = int(info["target"])
    if len(stable) < target:
        need = target - len(stable)
        start = int(info["next_layout"])
        stop = min(int(info["layout_count"]), start + need + 2)
        if start >= stop:
            info["state"] = "failed_exhausted"
            log(root, f"FAILED {task}: {len(stable)}/{target} stable and no layouts left")
            return
        info["next_layout"] = stop
        job = new_job(
            config, plan, task, range(start, stop), info["seconds_per_layout"] * (stop - start)
        )
        log(
            root,
            f"top-up {task}: stable={len(stable)}/{target} -> {job['run_id']} layouts={job['layouts']}",
        )
        return
    ordered = [stable[lid] for lid in sorted(stable)[:target]]
    successes = sum(bool(row.get("success")) for row in ordered)
    score = sum(float(row.get("score", 0.0) or 0.0) for row in ordered)
    merged = {
        "success_rate": successes / target,
        "eval_time": target,
        "score": score / target * 100.0,
        "details": {str(i): row for i, row in enumerate(ordered)},
    }
    tail.atomic_json(tail.canonical_result_path(config, task), merged)
    info.update(
        state="done",
        successes=successes,
        stable=len(stable),
        elapsed_s=sum(j.get("finished_at", 0) - j.get("started_at", 0) for j in jobs),
    )
    log(root, f"done {task}: {successes}/{target} success, stable candidates={len(stable)}")


def run(config: dict[str, Any], poll: int) -> int:
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(config)
    slots = list(config["slots"])
    busy: dict[int, dict[str, Any]] = {}
    for job in plan["jobs"]:  # resume: re-attach running jobs to their slots
        if job["state"] == "running" and job["slot"] is not None:
            busy[int(job["slot"])] = job
    max_attempts = int(config.get("max_job_attempts", 3))
    log(root, f"start jobs={len(plan['jobs'])} slots={len(slots)} tasks={len(plan['tasks'])}")
    while True:
        for slot_index, job in list(busy.items()):
            status = Path(job["status"])
            if not status.is_file():
                continue
            rc = status.read_text(encoding="utf-8").strip()
            job["finished_at"] = time.time()
            del busy[slot_index]
            got = len(stable_layouts(job))
            # A client that never reached the policy server (or died before its
            # first episode) can still exit 0; an empty chunk is retried rather
            # than accepted as "all layouts unstable".
            if got == len(job["layouts"]) or (rc == "0" and got > 0):
                job["state"] = "done"
                log(
                    root,
                    f"job done {job['run_id']} rc={rc} stable={got}/{len(job['layouts'])} "
                    f"wall={job['finished_at'] - job['started_at']:.0f}s slot={slot_index}",
                )
            elif job["attempts"] < max_attempts:
                job["state"] = "pending"
                log(root, f"job retry {job['run_id']} rc={rc} stable={got}/{len(job['layouts'])}")
            else:
                job["state"] = "done"
                log(root, f"job gave up {job['run_id']} rc={rc} stable={got}/{len(job['layouts'])}")
        for task, info in plan["tasks"].items():
            if info["state"] != "running":
                continue
            jobs = [j for j in plan["jobs"] if j["task"] == task]
            if jobs and all(j["state"] == "done" for j in jobs):
                finalize_task(config, plan, task, root)
        pending = sorted(
            (j for j in plan["jobs"] if j["state"] == "pending"),
            key=lambda j: (-j["estimate"], j["run_id"]),
        )
        for slot_index in range(len(slots)):
            if slot_index in busy or not pending:
                continue
            job = pending.pop(0)
            launch(config, job, slots[slot_index], slot_index)
            busy[slot_index] = job
            log(
                root,
                f"launch {job['run_id']} layouts={job['layouts'][0]}..{job['layouts'][-1]} "
                f"gpu={slots[slot_index]['gpu']} port={slots[slot_index]['policy_port']} slot={slot_index}",
            )
        save_plan(config, plan)
        states = [info["state"] for info in plan["tasks"].values()]
        done_jobs = sum(j["state"] == "done" for j in plan["jobs"])
        (root / "status.txt").write_text(
            f"tasks_done={states.count('done')}/{len(states)} jobs_done={done_jobs}/{len(plan['jobs'])} "
            f"running={len(busy)}\n",
            encoding="utf-8",
        )
        if not busy and all(s != "running" for s in states):
            break
        time.sleep(poll)
    failed = [t for t, info in plan["tasks"].items() if info["state"] != "done"]
    summary = {
        "run_id": config["run_id"],
        "eval_num": "native",
        "dimensions": ["all"],
        "counts": {
            "PASS": len(plan["tasks"]) - len(failed),
            "FAIL": len(failed),
            "SKIP": 0,
            "DRY_RUN": 0,
        },
        "results": [
            {
                "status": "PASS" if info["state"] == "done" else "FAIL",
                "task": task,
                "exit_code": "0" if info["state"] == "done" else "1",
                "eval_time": str(info["target"]),
                "elapsed_sec": str(round(info.get("elapsed_s", 0))),
                "result_path": str(tail.canonical_result_path(config, task)),
                "log_path": str(root / "scheduler.log"),
                "message": "ok (packed)" if info["state"] == "done" else info["state"],
            }
            for task, info in plan["tasks"].items()
        ],
    }
    tail.atomic_json(root / "summary.json", summary)
    log(root, f"finished failed_tasks={failed}")
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    return run(tail.read_json(args.config), args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
