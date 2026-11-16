"""Run isolated XPolicyLab Playground/Test Debug jobs in a Slurm allocation."""

import argparse
import concurrent.futures
from dataclasses import replace
import os
from pathlib import Path
import shutil
import socket

from orchestrator.config import REPO, load_task
from orchestrator.files import hashes, write_json
from orchestrator.resources import GpuSlots
from orchestrator.session_runner import definition, prepare_inputs, prepare_test
from .execution import episode, read


DEBUG_TASKS = ("stack_bowls", "cover_blocks", "insert_tubes", "fill_pen_holder")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sif", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--per-gpu", type=int, default=1)
    parser.add_argument("--episode-timeout", type=int, default=28800)
    parser.add_argument("--agent-image", default="embodiedevo-agent:xpolicy")
    parser.add_argument("--material-mirror", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Launch under Slurm with an allocated GPU")
    args.run_dir = args.run_dir.resolve()
    args.sif = args.sif.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "runtime-tmp").mkdir()
    allocated = os.environ.get("SLURM_JOB_GPUS", "")
    if not allocated or any(not value.isdecimal() for value in allocated.split(",")):
        parser.error("Expected physical GPU indices in SLURM_JOB_GPUS")
    slots = GpuSlots(allocated.split(","), args.per_gpu, directory=args.run_dir / "gpu-slots")
    status = {"diagnostic": args.probe, "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "concurrency_limit": len(slots.gpus) * args.per_gpu,
              "completed": [], "errors": [], "status": "running"}
    status["runtime_profile"] = {"runtime": "apptainer", "sif": str(args.sif),
                                 "kit_threads": 8, "video_threads": 2,
                                 "reset_repairs": "existing EmbodiedEvo native reset helpers"}
    status["source_hashes"] = hashes(REPO / "integrations/robodojo_policy")
    write_json(args.run_dir / "status.json", status)

    def launch(cfg, task_run, run, phase, identity):
        with slots.lease() as gpu:
            return episode(cfg, task_run, run, phase, identity, gpu, args)

    pending = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=status["concurrency_limit"]) as pool:
        for task in DEBUG_TASKS:
            cfg = load_task("RoboDojo", env_task=task)
            cfg = replace(cfg, test=replace(cfg.test, scenes=5),
                          agent=replace(cfg.agent, image=args.agent_image))
            task_run = args.run_dir / "tasks" / task
            task_run.mkdir(parents=True)
            state = {"protocol": cfg.protocol, "definition": definition(cfg), "phase": "playground",
                     "inputs": prepare_inputs(cfg, task_run)}
            write_json(task_run / "protocol.json", state)
            future = pool.submit(launch, cfg, task_run, task_run / "playground", "playground", 0)
            pending[future] = (cfg, task_run, "playground")
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                cfg, task_run, phase = pending.pop(future)
                try:
                    future.result()
                    status["completed"].append({"task": cfg.simulator.env_task, "phase": phase})
                    if phase == "playground":
                        state = read(task_run / "protocol.json")
                        for part in ("skills", "lessons"):
                            source = task_run / "playground/episodes/episode_0000/workspace" / part
                            hashes(source)
                            shutil.copytree(source, task_run / "frozen_harness" / part)
                        state.update(phase="frozen", playground_valid=True,
                                     harness=hashes(task_run / "frozen_harness"))
                        write_json(task_run / "protocol.json", state)
                        settings = ["standard"] * (3 if cfg.simulator.env_task == "stack_bowls" else 5)
                        if len(settings) == 3:
                            settings += ["random"] * 2
                        _, seeds = prepare_test(cfg, task_run, settings=settings)
                        for index, (identity, setting) in enumerate(zip(seeds, settings, strict=True)):
                            effective = cfg
                            if setting == "random":
                                effective = replace(cfg, simulator=replace(cfg.simulator,
                                                    env_task=cfg.simulator.env_task + "_random"))
                            future = pool.submit(launch, effective, task_run,
                                task_run / "test" / f"scene_{index:04d}", "test", identity)
                            pending[future] = (effective, task_run, "test")
                except BaseException as exc:
                    status["errors"].append({"task": cfg.simulator.env_task, "phase": phase,
                                             "error": f"{type(exc).__name__}: {exc}"})
            write_json(args.run_dir / "status.json", status)
    status["status"] = "failed" if status["errors"] else "finished"
    write_json(args.run_dir / "status.json", status)
    return bool(status["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
