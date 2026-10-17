"""Pin frozen knowledge and run the official client against a coding agent."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from orchestrator.config import REPO, config_dict, config_from_dict
from orchestrator.files import hashes, write_json
from orchestrator.host_policy import require_allowed
from orchestrator.resources import sim_env
from orchestrator.session_runner import stop_process


def pin(args):
    require_allowed("local")
    source = args.source_run.resolve()
    state = json.loads((source / "protocol.json").read_text())
    frozen = source / "frozen_harness"
    if (state.get("phase") != "frozen" or not state.get("playground_valid")
            or state.get("harness") != hashes(frozen)):
        raise ValueError("Source must have an intact frozen harness")
    task = state["definition"]["config"]["simulator"]["env_task"]
    if task != args.env_task:
        raise ValueError(f"Frozen harness task {task} does not match {args.env_task}")
    cfg = config_from_dict(state["definition"]["config"])
    if cfg.simulator.backend != "robodojo":
        raise ValueError("The source must be a RoboDojo run")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "inputs", run / "inputs")
    shutil.copytree(frozen, run / "frozen_harness")
    shutil.copytree(REPO / "integrations", run / "integration-source/integrations",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    write_json(run / "integration-source-hashes.json", hashes(run / "integration-source"))
    # Only the protocol's budget description changes; task goals, primitives
    # and frozen knowledge remain the same. Record the exact replacements.
    replacements = {
        "Native `step(action)` calls have no step-count limit. Actions are counted for telemetry only.":
        "Native `step(action)` calls obey the official task action limit. Check status for native_steps_remaining.",
        "Native actions have no step-count limit;":
        "Native actions obey the official per-task step-count limit;",
        "`native_steps_remaining: null` (unlimited)":
        "`native_steps_remaining` (official remaining actions)",
        "There is no action-count budget.": "The official action-count budget applies.",
    }
    changes = []
    for relative in ("test.md", "primitives/primitives.md"):
        path = run / "inputs" / relative
        before = path.read_text()
        after = before
        for old, new in replacements.items():
            after = after.replace(old, new)
        path.write_text(after)
        changes.append({"path": relative, "before": before, "after": after})
    write_json(run / "protocol-description-changes.json", changes)
    (run / "official").mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    provision = json.loads((REPO / "vendor/robodojo/.embodiedevo-provisioned.json").read_text())
    spec = {
        "run_dir": str(run), "source_run": str(source), "env_task": args.env_task,
        "seed": args.seed, "episodes": args.episodes, "gpu": args.gpu, "port": port,
        "harness_hashes": hashes(run / "frozen_harness"), "task_config": config_dict(cfg),
        "vendor_provision": provision, "protocol": "xpolicylab-ws-frozen-harness-v1",
        "scope": "local official evaluator subset; not a verified leaderboard submission",
        "terminal_feedback": "binary success and termination via standard trial_end hook",
        "action_chunk_size": 1,
    }
    write_json(run / "adapter.json", spec)
    return cfg, run, spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--env-task", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    cfg, run, spec = pin(args)
    server = client = None
    read_fd = write_fd = None
    runtime = {"status": "starting", "pid": os.getpid(), "started_at": time.time()}
    write_json(run / "runtime.json", runtime)

    def cancelled(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancelled)
    try:
        environment = sim_env(args.gpu, "robodojo")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        policy_python = REPO / ".venv-robodojo-policy/bin/python"
        command = [str(policy_python), "-m", "integrations.robodojo_policy.bootstrap",
                   "server", "--run-dir", str(run)]
        server = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        until = time.monotonic() + 60
        while not (run / "policy-server.json").exists():
            if server.poll() is not None or time.monotonic() > until:
                raise RuntimeError("Policy server failed to start; inspect server-error.json")
            time.sleep(0.2)
        read_fd, write_fd = os.pipe()
        command = [str(cfg.python), "-m", "integrations.robodojo_policy.bootstrap", "client",
                   "--run-dir", str(run), "--ready-fd", str(write_fd)]
        client = subprocess.Popen(command, cwd=REPO, env=environment,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, pass_fds=(write_fd,), start_new_session=True)
        os.close(write_fd)
        write_fd = None
        runtime.update(status="running", server_pid=server.pid, client_pid=client.pid)
        write_json(run / "runtime.json", runtime)
        while client.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("Policy server exited during evaluation")
            time.sleep(1)
        results = list((run / "official/eval_result").rglob("_result.json"))
        if client.returncode != 0 or len(results) != 1:
            raise RuntimeError(f"Official client exited {client.returncode}; result files={len(results)}")
        result = json.loads(results[0].read_text())
        if result["eval_time"] != args.episodes:
            raise RuntimeError("Official evaluator did not score every requested episode")
        if hashes(run / "frozen_harness") != spec["harness_hashes"]:
            raise RuntimeError("Frozen harness changed during evaluation")
        errors = [p for p in (run / "episodes").glob("*/runtime.json")
                  if json.loads(p.read_text()).get("status") == "failed"]
        if errors:
            raise RuntimeError("Agent worker failure invalidates this smoke run")
        write_json(run / "result.json", {
            "valid": True, "scope": spec["scope"], "env_task": args.env_task,
            "official_result": str(results[0].relative_to(run)), **result,
        })
        runtime.update(status="finished", finished_at=time.time())
    except BaseException as exc:
        runtime.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=time.time())
        raise
    finally:
        stop_process(client)
        stop_process(server)
        for fd in (read_fd, write_fd):
            if fd is not None:
                os.close(fd)
        write_json(run / "runtime.json", runtime)


if __name__ == "__main__":
    main()
