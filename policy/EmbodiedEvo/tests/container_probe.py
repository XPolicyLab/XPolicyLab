"""Network-disabled Docker agent for startup-only load tests (never uses Codex)."""

import os
from pathlib import Path
import time
from unittest.mock import patch


def probe_command(cfg, run, socket_dir, phase):
    name = f"embodiedevo-session-{os.getpid()}-{time.time_ns()}"
    argv = ["docker", "run", "--rm", "--init", "--name", name,
            "--network", "none", "--read-only", "--tmpfs", "/tmp:mode=1777",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "64", "--cpus", "1", "--memory", "256m",
            "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", "/workspace",
            "--env", "EAHARNESS_SOCKET=/run/eaharness/session.sock",
            "--env", "OPENBLAS_NUM_THREADS=1", "--env", "OMP_NUM_THREADS=1",
            "--mount", f"type=bind,src={run / 'workspace'},dst=/workspace",
            "--mount", f"type=bind,src={socket_dir},dst=/run/eaharness,readonly",
            "--mount", f"type=bind,src={Path(__file__).resolve()},dst=/probe.py,readonly"]
    for component in ("instruction.md", "primitives", "scripts", "observations", "skills", "lessons"):
        argv += ["--mount", f"type=bind,src={run / 'workspace' / component},dst=/workspace/{component},readonly"]
    return name, argv + [cfg.agent.image, "python3", "/probe.py"]


def run_container_probe(*args):
    # Only replace the agent command: production worker, workspace creation,
    # Unix socket, action bridge, Docker readiness and cleanup all run normally.
    from XPolicyLab.policy.EmbodiedEvo.runtime.worker import run_worker
    with patch("XPolicyLab.policy.EmbodiedEvo.runtime.worker.agent_command", probe_command):
        run_worker(*args)


def main():
    import subprocess
    import sys
    for component in ("skills", "lessons", "scripts", "observations"):
        try:
            (Path(component) / "forbidden-write").write_text("should fail")
        except OSError:
            pass
        else:
            raise AssertionError(f"Writable frozen mount: {component}")
    Path("submission/solution.py").write_text(
        "a = {k: v for k, v in get_observation()['state'].items() if k.endswith('joint_state')}\n"
        "step(a)\n")
    # Deliberately remains in the first step until the load client closes.
    raise SystemExit(subprocess.call([sys.executable, "scripts/env.py", "exec"]))


if __name__ == "__main__":
    main()
