#!/usr/bin/env python3
"""Evaluate a trained XPolicyLab policy (e.g. G0.5) with RoboDojo's own eval client, and
show the results in scripts/view_runs.py next to the harness runs.

    # The policy server must already run (XPolicyLab setup_policy_server.py, or
    # `bash scripts/robodojo.sh server` in the RoboDojo checkout). Then, in the harness-policy env:
    python scripts/robodojo_policy_eval.py run --policy G05 --port 19005 --tasks all --eval-num 2 \
        --gpu 0 --label baseline --output runs/g05-2case

    # Only convert results RoboDojo already wrote:
    python scripts/robodojo_policy_eval.py convert --policy G05 --label baseline \
        --since 2026-10-06_05-42-00 --output runs/g05-2case

`run` evaluates the tasks one after another through `scripts/robodojo.sh client`: RoboDojo's
own episode loop, layouts (pack --seed), step limits and scoring. `--tasks all` is the 42
tasks the harness suites run; the leaderboard also scores Generalization on the `*_random`
variants (`--tasks all,random`, with `--eval-num native` for the official episode counts).
`--eval-num 2` takes the first two layouts that settle, as the harness suites do. Use ONE
`run` per policy server: an XPolicyLab server keeps a single current observation, so two
clients of one server receive each other's actions.

RoboDojo writes eval_result/RoboDojo/<task>/<policy>/<env cfg>/<seed>_ckpt_name=<label>,
action_type=<type>/<run>/_result.json (layout, success and score of each episode, rewritten
after every batch) and episode_<n>_<camera>_<success|fail>.mp4. `convert` turns each task's
newest run into OUTPUT/<task>/seed-<layout>/ with summary.json, manifest.json, events.jsonl
and videos/, replacing episodes this script converted from an older run of that task (other
episodes in OUTPUT are left alone). A trained policy has no language model: no model calls,
cost or transcript.
Standard library only.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from harness_policy.runtime.report import DIMENSIONS  # noqa: E402

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
RUN_NAME = "%Y-%m-%d_%H-%M-%S"  # RoboDojo's run folder (ROBODOJO_RUN_ID) format


def expand_tasks(spec):
    tasks = []
    for item in (part.strip() for part in spec.split(",")):
        if item == "all":
            tasks += [task for group in DIMENSIONS.values() for task in group]
        elif item == "random":  # the Generalization tasks' randomized variants
            tasks += [f"{task}_random" for task in DIMENSIONS["Generalization"]]
        elif item in DIMENSIONS:
            tasks += DIMENSIONS[item]
        elif item:
            tasks.append(item)
    return list(dict.fromkeys(tasks))


def run_folder(results, task, policy, env_cfg, seed, label, action_type):
    """RoboDojo's folder for one task's runs (src/eval_client/eval_env.py, save_dir)."""
    return Path(results) / task / policy / env_cfg / f"{seed}_ckpt_name={label},action_type={action_type}"


def converted_from(episode):
    """The RoboDojo run an episode folder was converted from, or None (a harness episode, or none)."""
    try:
        source = json.loads((Path(episode) / "summary.json").read_text()).get("source")
    except (OSError, ValueError):
        return None
    return Path(source).resolve() if source else None


def convert(results, policy, env_cfg, seed, label, action_type, since, output, name, tasks=None):
    """Write the newest run (folder name >= since) of every task as harness episodes; returns the count."""
    output, converted = Path(output), 0
    task_dirs = [Path(results) / task for task in tasks] if tasks else sorted(Path(results).iterdir())
    for task_dir in task_dirs:
        folder = run_folder(results, task_dir.name, policy, env_cfg, seed, label, action_type)
        runs = sorted(path for path in folder.glob("[0-9]*")
                      if path.name >= since and (path / "_result.json").exists())
        if not runs:
            continue
        run = runs[-1].resolve()
        result = json.loads((run / "_result.json").read_text())
        # Episodes converted from an older run of the same policy, label and config would mix two
        # evaluations: replace them. Episodes from anywhere else (harness runs, other policies) stay.
        for old in (output / task_dir.name).glob("seed-*"):
            source = converted_from(old)
            if source is not None and source.parent == run.parent and source != run:
                shutil.rmtree(old)
        for index, detail in sorted(result.get("details", {}).items(), key=lambda item: int(item[0])):
            layout, success = int(detail["layout_id"]), bool(detail.get("success"))
            score = float(detail.get("score") or 0.0)
            episode = output / task_dir.name / f"seed-{layout}"
            if (episode / "summary.json").exists() and converted_from(episode) != run:
                print(f"{task_dir.name}/seed-{layout}: {episode} holds another evaluation; not overwritten "
                      "(use a separate --output per policy)", file=sys.stderr)
                continue
            (episode / "videos").mkdir(parents=True, exist_ok=True)
            tag = "success" if success else "fail"
            for camera in CAMERAS:
                video = run / f"episode_{int(index):07d}_{camera}_{tag}.mp4"
                if video.exists():
                    shutil.copyfile(video, episode / "videos" / f"episode-{camera}.mp4")
            # RoboDojo ends every episode itself (success, failure or step limit).
            summary = {"run": str(episode), "task": task_dir.name, "seed": layout, "status": "episode_ended",
                       "success": success, "reward": score, "process_score": 100 * score,
                       "model_calls": 0, "cost_usd": 0.0, "source": str(run)}
            manifest = {"config": {"agent": {"model": name},
                                   "runtime": {"label": f"{name} (RoboDojo eval client)"},
                                   "policy": policy, "env_cfg": env_cfg, "layout_pack_seed": seed,
                                   "ckpt_name": label, "action_type": action_type},
                        "source": str(run)}
            reset = {"event": "reset", "observation": {"state": {"task": task_dir.name, "layout_source": "eval",
                                                                  "layout_pack_seed": seed}}}
            outcome = {"event": "outcome", "outcome": {"reward": score, "success": success,
                                                       "process_score": 100 * score}}
            (episode / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (episode / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (episode / "events.jsonl").write_text(json.dumps(reset) + "\n" + json.dumps(outcome) + "\n")
            converted += 1
    return converted


def client_command(robodojo, task, args):
    return ["bash", str(Path(robodojo) / "scripts" / "robodojo.sh"), "client", "--task", task,
            "--policy-name", args.policy, "--policy-host", args.host, "--policy-port", str(args.port),
            "--eval-num", str(args.eval_num), "--env-gpu", str(args.gpu), "--env-cfg", args.env_cfg,
            "--seed", str(args.seed), "--ckpt", args.label, "--action-type", args.action_type]


def run(args):
    robodojo = Path(args.robodojo).resolve()
    output = Path(args.output).resolve()
    (output / "logs").mkdir(parents=True, exist_ok=True)
    # Isaac Lab logs under $TMPDIR/isaaclab; a shared /tmp/isaaclab may belong to another user.
    tmpdir = output / "tmp" / f"gpu{args.gpu}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    # A leftover ROBODOJO_RUN_ID would make every task resume one old run; EVAL_NUM would
    # override --eval-num native.
    env = {key: value for key, value in os.environ.items() if key not in ("ROBODOJO_RUN_ID", "EVAL_NUM")}
    env["TMPDIR"] = str(tmpdir)
    try:
        socket.create_connection((args.host, args.port), timeout=10).close()
    except OSError as error:
        sys.exit(f"policy server {args.host}:{args.port} is not reachable ({error}); start it first")
    since = datetime.now().strftime(RUN_NAME)
    results = robodojo / "eval_result" / "RoboDojo"
    for task in expand_tasks(args.tasks):
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"{stamp} {task}: running", flush=True)
        with open(output / "logs" / f"{task}.log", "ab") as log:
            code = subprocess.call(client_command(robodojo, task, args), cwd=robodojo, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        count = convert(results, args.policy, args.env_cfg, args.seed, args.label, args.action_type,
                        since, output, args.name or args.policy, tasks=[task])
        note = "" if code == 0 else f" -- RoboDojo client FAILED (exit {code}); results may be partial, see logs/{task}.log"
        print(f"{datetime.now().strftime('%H:%M:%S')} {task}: {count} episodes{note}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "convert"):
        sub = commands.add_parser(command)
        sub.add_argument("--policy", required=True, help="XPolicyLab policy directory name, e.g. G05")
        sub.add_argument("--name", help="name shown in the viewer (default: --policy)")
        sub.add_argument("--label", required=True, help="ckpt label recorded in RoboDojo's result path")
        sub.add_argument("--action-type", default="joint")
        sub.add_argument("--env-cfg", default="arx_x5")
        sub.add_argument("--seed", type=int, default=0, help="layout pack (RoboDojo --seed)")
        sub.add_argument("--robodojo", default=os.environ.get("HARNESS_ROBODOJO_ROOT", str(ROOT / "third_party" / "RoboDojo")))
        sub.add_argument("--output", required=True, help="viewer folder, e.g. runs/g05-2case")
    run_parser = commands.choices["run"]
    run_parser.add_argument("--tasks", required=True, help="task names, dimension names or 'all'")
    run_parser.add_argument("--eval-num", default="native", help="episodes per task, or 'native' (official counts)")
    run_parser.add_argument("--host", default="127.0.0.1")
    run_parser.add_argument("--port", type=int, required=True)
    run_parser.add_argument("--gpu", type=int, default=0, help="simulator GPU")
    convert_parser = commands.choices["convert"]
    convert_parser.add_argument("--since", default="", help="first run folder to use, e.g. 2026-10-06_05-42-00")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        results = Path(args.robodojo) / "eval_result" / "RoboDojo"
        count = convert(results, args.policy, args.env_cfg, args.seed, args.label, args.action_type,
                        args.since, args.output, args.name or args.policy)
        print(f"{count} episodes in {args.output}")


if __name__ == "__main__":
    main()
