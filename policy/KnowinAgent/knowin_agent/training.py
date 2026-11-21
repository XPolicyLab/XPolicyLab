"""Public online-training entrypoint using the existing hosted episode and Reviewer.

The simulator runs in a child process because Isaac's close can terminate Python.
All harness outputs live in a fresh external directory. This module does not own
GPU admission or start perception services; those are deployment prerequisites.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

POLICY_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Real RoboDojo practice -> review agent -> frozen knowledge (uses models and GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli.add_argument("bench_name", choices=["RoboDojo"])
    cli.add_argument("ckpt_name", choices=["no-checkpoint"], help="no learned-weight checkpoint")
    cli.add_argument("env_cfg_type", choices=["arx_x5"])
    cli.add_argument("action_type", choices=["joint"])
    cli.add_argument("seed", type=int)
    cli.add_argument("gpu_id", type=int, help="physical GPU id; CUDA and Kit renderer select this card")
    cli.add_argument("--task", required=True)
    cli.add_argument("--out", required=True, type=Path, help="new directory outside the policy repository")
    cli.add_argument("--knowledge-from", type=Path, help="optional frozen seed; omitted means cold start")
    cli.add_argument("--robodojo-root", type=Path, help="RoboDojo installation; otherwise use backend env lookup")
    cli.add_argument("--practice-max-executions", type=positive_int, default=150,
                     help="accepted practice executions (not native action steps)")
    cli.add_argument("--practice-wall-seconds", type=positive_seconds, default=10800.0,
                     help="hosted Agent wall budget; simulator startup/cleanup are additional")
    cli.add_argument("--review-wall-seconds", type=positive_seconds, default=2400.0)
    cli.add_argument("--test-budget", type=positive_int, default=100,
                     help="future Test execution budget shown to the reviewer; this command starts no Test")
    cli.add_argument("--video-stride", type=int, default=0, help="0 disables optional video; positive N records every N frames")
    cli.add_argument("--review-agent-command", default=os.environ.get("KNOWIN_AGENT_REVIEW_AGENT_CMD"),
                     help="optional shell-split argv; review prompt is appended; otherwise use configured Codex")
    return cli


def external_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path == POLICY_ROOT or POLICY_ROOT in path.parents:
        raise ValueError("--out must be outside the policy repository and immutable knowledge assets")
    if path.exists():
        raise FileExistsError(f"refusing to reuse training output: {path}; choose a new --out")
    return path


def practice_config(args: argparse.Namespace) -> dict:
    # Reuse the official adapter's environment/config handling without importing
    # XPolicyLab or starting a backend/model. The package root also works standalone.
    from deploy import episode_config
    from knowin_agent.assembly import resolve_config
    from knowin_agent.backends.robodojo.native import KIT_ARGS_ENV, OFFICIAL_KIT_ARGS
    from knowin_agent.episode import build_limits
    from knowin_agent.workspace.freeze import verify

    if args.seed < 0 or args.gpu_id < 0 or args.video_stride < 0:
        raise ValueError("seed, gpu_id and video_stride must be nonnegative")
    config = episode_config(args.task, use_environment_knowledge=False)
    config["phase"] = "practice"
    config.pop("knowledge_from", None)
    if args.knowledge_from:
        seed = args.knowledge_from.expanduser().resolve()
        verify(seed)
        config["knowledge_from"] = str(seed)
    backend = config["backend"]
    backend["seed"] = args.seed
    if args.robodojo_root:
        backend["robodojo_root"] = str(args.robodojo_root.expanduser().resolve())
    kit_args = os.environ.get(KIT_ARGS_ENV, OFFICIAL_KIT_ARGS)
    backend["kit_args"] = f"{kit_args} --/renderer/activeGpu={args.gpu_id} --/renderer/multiGpu/enabled=false"
    config["limits"].update(max_executions=args.practice_max_executions,
                            wall_seconds=args.practice_wall_seconds)
    config["video_stride"] = args.video_stride
    config["identity"] = {"entrypoint": "train.sh", "bench_name": args.bench_name,
                          "ckpt_name": args.ckpt_name, "env_cfg_type": args.env_cfg_type,
                          "action_type": args.action_type, "physical_gpu_id": args.gpu_id}
    # Validate all existing switches and limits before any expensive startup.
    resolved, _ = resolve_config(config)
    build_limits(resolved.get("limits"))
    return config


def review_command(args: argparse.Namespace, config: dict) -> list[str]:
    agent, switches = config["agent"], config["switches"]
    if args.review_agent_command:
        command = shlex.split(args.review_agent_command)
        if not command:
            raise ValueError("review agent command is empty")
    else:
        command = [agent["binary"], "exec", "--skip-git-repo-check", "--sandbox", "workspace-write",
                   "--model", switches["agent.model"], "-c",
                   f'model_reasoning_effort="{switches["agent.effort"]}"']
    # command_launcher intentionally inherits the environment. Prefix its argv with
    # env to select the same configured authentication home as practice, without
    # mutating this process's environment or involving a shell.
    if agent.get("codex_home_source"):
        home = str(Path(agent["codex_home_source"]).expanduser().resolve())
        command = ["env", f"CODEX_HOME={home}", *command]
    return command


def train(args: argparse.Namespace) -> Path:
    from knowin_agent.evolution.review import Reviewer, command_launcher
    from knowin_agent.workspace.freeze import verify

    if os.environ.get("EVAL_ENV_TYPE") == "debug" or os.environ.get("KNOWIN_AGENT_DRY_RUN") == "1":
        raise ValueError("train.sh performs real training; debug/dry-run flags are not supported")
    out = external_output(args.out)
    config = practice_config(args)
    command = review_command(args, config)
    # Claim the destination atomically. No resume, deletion, or overwrite is implicit.
    out.mkdir(parents=True, exist_ok=False)
    state = {"status": "practice", "task": args.task, "seed": args.seed, "started_at": time.time(),
             "practice_max_executions": args.practice_max_executions,
             "practice_wall_seconds": args.practice_wall_seconds,
             "review_wall_seconds": args.review_wall_seconds, "test_budget": args.test_budget,
             "video_stride": args.video_stride, "review_command": command}

    def record() -> None:
        (out / "training.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    config_path = out / "practice_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    record()
    print(f"[KnowinAgent][TRAIN] task={args.task}; outputs={out}; "
          f"practice<={args.practice_max_executions} accepted executions / {args.practice_wall_seconds}s Agent wall; "
          f"review<={args.review_wall_seconds}s; video_stride={args.video_stride}", flush=True)
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        env["PYTHONPATH"] = str(POLICY_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        practice = out / "practice"
        argv = [config["agent"]["python"], "-m", "knowin_agent.cli", "host", "--config", str(config_path),
                "--run-dir", str(practice)]
        state["practice_command"] = argv
        record()
        with (out / "practice.log").open("wb") as log:
            completed = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        state["practice_returncode"] = completed.returncode
        record()
        if completed.returncode != 0:
            raise RuntimeError(f"practice exited {completed.returncode}; inspect {out / 'practice.log'}")
        result = json.loads((practice / "result.json").read_text(encoding="utf-8"))
        # Even a zero simulator exit is insufficient without the completed freeze receipt.
        frozen = Path(result.get("frozen_dir", str(practice / "frozen"))).resolve()
        if frozen != (practice / "frozen").resolve():
            raise ValueError("practice result references unexpected frozen knowledge")
        verify(frozen)
        state.update(status="review", practice_stop_reason=result.get("stop_reason"),
                     practice_executions=result.get("executions"))
        record()
        reviewer = Reviewer(command_launcher(command), wall_seconds=args.review_wall_seconds,
                            test_budget=args.test_budget)
        merged = reviewer.review(args.task, [practice],
                                 Path(config["knowledge_from"]) if config.get("knowledge_from") else None,
                                 out / "review", test_budget=args.test_budget)
        verify(merged)
        state.update(status="finished", frozen_dir=str(merged), finished_at=time.time())
        record()
        print(f"[KnowinAgent][TRAIN] frozen knowledge: {merged}", flush=True)
        return merged
    except BaseException as exc:
        state.update(status="failed", finished_at=time.time(), error=f"{type(exc).__name__}: {exc}")
        record()
        raise


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        train(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[KnowinAgent][TRAIN][ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
