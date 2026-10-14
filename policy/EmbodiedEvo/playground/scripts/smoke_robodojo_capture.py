"""Repeated-reset XPolicyLab diagnostic on the configured Playground scene only."""

import argparse
import os
from pathlib import Path
import socket
import time
import traceback

from integrations.robodojo_policy.execution import episode, read
from orchestrator.config import REPO, load_task
from orchestrator.files import hashes, write_json
from orchestrator.session_runner import prepare_inputs
from orchestrator.slurm import allocated_gpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sif", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--env-task", default="insert_tubes")
    parser.add_argument("--probe-reset-cycles", type=int, default=12)
    parser.add_argument("--material-mirror", type=Path)
    parser.add_argument("--episode-timeout", type=int, default=3600)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or args.gpu not in allocated_gpus(os.environ.get("SLURM_JOB_GPUS", "")):
        parser.error("Run inside a Slurm allocation with the selected physical GPU")
    cfg = load_task("RoboDojo", env_task=args.env_task)
    if not 1 <= args.probe_reset_cycles <= cfg.playground.budget - 2:
        parser.error("Reset cycles must fit the Playground execution budget")
    args.run_dir = args.run_dir.resolve()
    args.sif = args.sif.resolve()
    args.probe = True
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "runtime-tmp").mkdir()
    prepare_inputs(cfg, args.run_dir)
    state = {"status": "running", "diagnostic": True, "model_called": False,
             "env_task": args.env_task, "reset_cycles": args.probe_reset_cycles,
             "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
             "started_at": time.time(), "source_hashes": hashes(REPO / "simulators"),
             "adapter_hashes": hashes(REPO / "integrations/robodojo_policy")}
    write_json(args.run_dir / "status.json", state)
    try:
        run = args.run_dir / "playground"
        episode(cfg, args.run_dir, run, "playground", cfg.playground.seed, args.gpu, args)
        import json

        records = [json.loads(line) for line in (run / "capture-resources.jsonl").read_text().splitlines()]
        assert len(records) == args.probe_reset_cycles + 1, records
        expected = {k: v for k, v in records[0].items() if k != "reset_count"}
        assert expected["camera_views"] == expected["render_products"] == 3, expected
        assert expected["stage_render_products"] >= 3, expected
        assert all({k: v for k, v in row.items() if k != "reset_count"} == expected for row in records), records
        result = read(run / "result.json")
        assert result["valid"] and result["diagnostic"], result
        state.update(status="passed", capture_resources=expected)
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        state["finished_at"] = time.time()
        write_json(args.run_dir / "status.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
