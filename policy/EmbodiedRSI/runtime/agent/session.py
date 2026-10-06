"""Launch and stop one isolated Codex process for a Test episode."""
import os
import signal
import subprocess
import time
from XPolicyLab.policy.EmbodiedRSI.runtime.agent import launch as launch_agent

def agent_command(cfg, run, socket_dir, phase):
    if phase != "test":
        raise ValueError("This adapter runs Test episodes only")
    launch_agent.require_image(cfg.agent.image)
    if cfg.agent.network != "none":
        launch_agent.ensure_network()
    name = f"embodiedrsi-session-{os.getpid()}-{time.time_ns()}"
    argv = launch_agent.docker_argv(
        run / "workspace",
        # Keep the CLI's cache/database HOME outside the run. Only the mirrored
        # rollout JSONL belongs in the reviewable output directory.
        launch_agent.stage_home(socket_dir / "agent-home"),
        launch_agent.adapt_for_docker(
            launch_agent.agent_argv(
                cfg.agent.preset,
                "Read instruction.md. "
                "Work autonomously toward the phase goal. "
                "Interact using scripts/env.py; finish the session when done. "
                "Camera frames in observations/ are native-resolution PNGs. Use "
                "the current_cam_*.png files; never open raw_cam_* files or "
                "create copies in scratch, and inspect no more than two current "
                "frames per iteration to keep model requests focused.",
                model=cfg.agent.model,
                effort=cfg.agent.reasoning_effort,
            ),
            preset=cfg.agent.preset,
        ),
        image=cfg.agent.image,
        network=cfg.agent.network,
        cpus=cfg.agent.cpus,
        memory_mb=cfg.agent.memory_mb,
        container_name=name,
    )
    # The frozen workspace client uses these workspace-session-v1 socket names.
    mounts = [
        "--env",
        "EAHARNESS_SOCKET=/run/eaharness/session.sock",
        "--mount",
        f"type=bind,src={socket_dir},dst=/run/eaharness,readonly",
    ]
    fixed = ["instruction.md", "primitives", "scripts", "observations"]
    fixed += ["skills", "lessons"]
    for component in fixed:
        mounts += [
            "--mount",
            f"type=bind,src={run / 'workspace' / component},dst=/workspace/{component},readonly",
        ]
    argv[argv.index(cfg.agent.image) : argv.index(cfg.agent.image)] = mounts
    return name, argv



def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
