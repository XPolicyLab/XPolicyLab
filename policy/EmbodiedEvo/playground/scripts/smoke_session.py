"""Exercise a real simulator and Docker client without invoking a language model.

Regular tasks use only the configured PlayGround seed. Test-only baselines require
an explicit diagnostic --seed and run directly in Test. This is an interface check,
never a scored Test scene; it neither opens nor creates a formal Test plan.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orchestrator.config import load_task
from orchestrator.session_runner import prepare_inputs, run_episode

CLIENT = """
import json
import os
from pathlib import Path
import subprocess

def call(*args, ok=True):
    p = subprocess.run(["python", "scripts/env.py", *args], capture_output=True, text=True)
    print(p.stdout, flush=True)
    assert (p.returncode == 0) == ok, p.stderr + p.stdout
    return json.loads(p.stdout)

if BASELINE:
    assert {p.name for p in Path(".").iterdir()} == {
        "instruction.md", "primitives", "scripts", "submission", "observations", "scratch",
    }
    assert {p.name for p in Path("submission").iterdir()} == {"solution.py"}
    assert {p.name for p in Path("scratch").iterdir()} <= {".gitkeep"}
    assert {p.name for p in Path("observations").iterdir()} <= {".gitkeep", "000000"}
call("status")
call("instruction")
p = Path("submission/solution.py")
p.write_text("remember = 17\\nprint(get_instruction())\\nget_observation()\\n")
call("exec")
p.write_text("assert remember == 17\\nprint('persistent variables verified')\\n")
call("exec")
status = call("status")
p.write_text(
    "obs = get_observation()\\n"
    "step({k: v for k, v in obs['state'].items() if k.endswith('joint_state')})\n"
)
action = call("exec")
assert action["native_steps_used"] > status["native_steps_used"]
if BASELINE:
    rejected = subprocess.run(
        ["python", "scripts/env.py", "reset"], capture_output=True, text=True,
    )
    assert rejected.returncode == 2, rejected.stdout + rejected.stderr
    assert "invalid choice" in rejected.stderr
    assert call("status")["executions_used"] == action["executions_used"]
else:
    call("reset", ok=status["phase"] == "playground")
if status["phase"] == "test":
    p.write_text("restart = reset\\nrestart()\\n")
    result = call("exec", ok=False)
    assert "PermissionError" in result["stderr"]
for name in (() if os.environ.get("EMBODIEDEVO_SMOKE_HOST_CLIENT") == "1" else (
    "instruction.md", "primitives/primitives.md", "scripts/env.py",
    "observations/000000/images.json",
)):
    try:
        with open(name, "a"):
            pass
    except OSError:
        pass
    else:
        raise AssertionError("Expected readonly mount: " + name)
call("observe")
call("finish")
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-task")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument(
        "--phase", choices=["playground", "test"],
        help="defaults to Test for baselines, Playground for regular tasks",
    )
    parser.add_argument(
        "--seed", type=int,
        help="explicit diagnostic scene for a Test-only baseline (no formal Test plan)",
    )
    parser.add_argument(
        "--host-client",
        action="store_true",
        help="fixed no-model host client: checks IPC only, NOT Docker isolation",
    )
    args = parser.parse_args(argv)
    cfg = load_task(args.task, env_task=args.env_task)
    phase = args.phase or ("test" if cfg.baseline else "playground")
    if cfg.baseline:
        if phase != "test":
            parser.error("Baseline has no Playground; use --phase test")
        if args.seed is None or args.seed < 0:
            parser.error("Baseline smoke requires an explicit nonnegative diagnostic --seed")
        seed = args.seed
    else:
        if args.seed is not None:
            parser.error("Regular task smoke uses only its configured Playground seed")
        seed = cfg.playground.seed
    client = f"BASELINE = {cfg.baseline!r}\n" + CLIENT
    cfg = replace(cfg, agent=replace(cfg.agent, network="none"))
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    prepare_inputs(cfg, run)

    def empty_home(path):
        path.mkdir(parents=True)
        return path

    def host_command(cfg, episode, socket_dir, phase):
        bootstrap = (
            "import os\n"
            f"os.chdir({str(episode / 'workspace')!r})\n"
            f"os.environ['EMBODIEDEVO_WORKSPACE'] = {str(episode / 'workspace')!r}\n"
            f"os.environ['EMBODIEDEVO_SOCKET'] = {str(socket_dir / 'session.sock')!r}\n"
            "os.environ['EMBODIEDEVO_SMOKE_HOST_CLIENT'] = '1'\n"
        )
        return None, [sys.executable, "-c", bootstrap + client]

    with (
        patch("orchestrator.launch_agent.adapt_for_docker", return_value=["python", "-c", client]),
        patch("orchestrator.launch_agent.stage_home", side_effect=empty_home),
        patch("orchestrator.session_runner.agent_command", side_effect=host_command)
        if args.host_client
        else nullcontext(),
    ):
        result = run_episode(
            cfg,
            run / "smoke",
            run,
            phase=phase,
            seed=seed,
            seconds=600,
            gpu=args.gpu,
        )
    from orchestrator.files import write_json

    write_json(
        run / "smoke.json",
        {
            "passed": result["valid"],
            "kind": "host-socket-smoke" if args.host_client else "docker-socket-smoke",
            "docker_isolation_checked": not args.host_client,
            "baseline": cfg.baseline,
            "phase": phase,
            "diagnostic_seed": seed,
            "scored": False,
        },
    )
    print(f"valid={result['valid']}; result={run / 'smoke/result.json'}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
