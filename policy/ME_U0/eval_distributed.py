"""Shard official RoboDojo task/seed evaluations across machines and GPU pairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys


def node_rank(output, count, requested):
    if requested != "auto":
        return int(requested)
    claims = output / "node_rank_claims"
    claims.mkdir(parents=True, exist_ok=True)
    owner = os.environ.get("ROBODOJO_NODE_ID")
    if not owner and shutil.which("nvidia-smi"):
        owner = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
        ).strip()
    owner = owner or socket.gethostname()
    for rank in range(count):
        owner_file = claims / f"rank_{rank}" / "owner_id"
        if owner_file.exists() and owner_file.read_text().strip() == owner:
            return rank
    for rank in range(count):
        claim = claims / f"rank_{rank}"
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        (claim / "owner_id").write_text(owner + "\n")
        return rank
    raise RuntimeError("No free node rank; use the same explicit rank when restarting a node")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="HF bundle directory containing model.pt and config.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--task-suite", default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--eval-num", default="native")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--node-rank", default="auto")
    parser.add_argument("--server-gpus", default="0,1,2,3")
    parser.add_argument("--sim-gpus", default="0,1,2,3")
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--action-chunk-size", type=int, default=36)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    policy_dir = Path(__file__).resolve().parent
    benchmark_dir = policy_dir.parents[2]
    checkpoint = str(Path(args.checkpoint).resolve())
    output = Path(args.output_root).resolve()
    rank = node_rank(output, args.num_nodes, args.node_rank)
    policy_gpus = args.server_gpus.split(",")
    simulator_gpus = args.sim_gpus.split(",")
    if len(policy_gpus) != len(simulator_gpus):
        parser.error("--server-gpus and --sim-gpus must have the same length")
    if not 0 <= rank < args.num_nodes:
        parser.error("--node-rank must be smaller than --num-nodes")
    if args.tasks:
        tasks = args.tasks.split(",")
    else:
        command = [sys.executable, str(benchmark_dir / "scripts/internal/task_inventory.py"), "--only-runnable"]
        if args.task_suite != "all":
            command.extend(["--dimension", args.task_suite])
        tasks = subprocess.check_output(command, cwd=benchmark_dir, text=True).splitlines()
    jobs = [(seed, task) for seed in args.seeds.split(",") for task in tasks]
    lane_count = len(policy_gpus)
    total_lanes = args.num_nodes * lane_count
    node_dir = output / "nodes" / f"node_{rank}"
    node_dir.mkdir(parents=True, exist_ok=True)
    processes = {}

    def stop(signum, frame):
        for process in list(processes.values()):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def run_lane(lane):
        shard = rank * lane_count + lane
        assigned = jobs[shard::total_lanes]
        lane_dir = node_dir / "lanes" / f"lane_{lane}"
        lane_dir.mkdir(parents=True, exist_ok=True)
        with (lane_dir / "assignments.tsv").open("w") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["seed", "task"])
            writer.writerows(assigned)
        print(f"node={rank} lane={lane} shard={shard}/{total_lanes} jobs={len(assigned)} policy_gpu={policy_gpus[lane]} sim_gpu={simulator_gpus[lane]}", flush=True)
        if args.dry_run:
            return 0
        failed = 0
        for seed, task in assigned:
            task_dir = lane_dir / "tasks" / f"seed_{seed}" / task
            task_dir.mkdir(parents=True, exist_ok=True)
            exit_file = task_dir / "exit_code"
            if exit_file.exists() and exit_file.read_text().strip() == "0":
                continue
            environment = os.environ.copy()
            environment.update(
                ROBODOJO_EVAL_ROOT=str(output / "results/RoboDojo"),
                ROBODOJO_RUN_ID=f"{output.name}_seed{seed}_{task}",
                ROBODOJO_FULL_RUN_ID=output.name,
                ROBODOJO_NODE_RANK=str(rank),
                ROBODOJO_LANE_ID=str(lane),
                ME_U0_POLICY_PORT=str(19200 + lane),
                ME_U0_NUM_INFERENCE_STEPS=str(args.num_inference_steps),
                ME_U0_ACTION_CHUNK_SIZE=str(args.action_chunk_size),
                EVAL_NUM=args.eval_num,
            )
            command = [
                "bash", str(policy_dir / "eval.sh"), "RoboDojo", task, checkpoint,
                "arx_x5", "joint", seed, policy_gpus[lane], simulator_gpus[lane],
                environment.get("ME_U0_POLICY_ENV", "me_u0"),
                environment.get("ME_U0_SIM_ENV", "robodojo"),
            ]
            print(f"node={rank} lane={lane} START seed={seed} task={task}", flush=True)
            with (task_dir / "eval.log").open("a") as log:
                process = subprocess.Popen(command, cwd=policy_dir, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                processes[lane] = process
                result = process.wait()
                del processes[lane]
            exit_file.write_text(str(result) + "\n")
            failed += result != 0
            print(f"node={rank} lane={lane} END seed={seed} task={task} rc={result}", flush=True)
        return failed

    with ThreadPoolExecutor(max_workers=lane_count) as executor:
        failed = sum(executor.map(run_lane, range(lane_count)))
    if not args.dry_run:
        (node_dir / "exit_code").write_text(str(int(failed > 0)) + "\n")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
