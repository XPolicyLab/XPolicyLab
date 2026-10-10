"""One host code executor and one isolated, fresh agent per official trial."""

import contextlib
import os
import random
import signal
import socket
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path

from XPolicyLab.policy.EmbodiedRSI.runtime.config import REPO, config_from_dict
from XPolicyLab.policy.EmbodiedRSI.runtime.agent.recovery import recovery_reason, resume_argv, rollout_state
from XPolicyLab.policy.EmbodiedRSI.runtime.files import codex_token_usage, write_json
from XPolicyLab.policy.EmbodiedRSI.runtime.execution.session import Session
from XPolicyLab.policy.EmbodiedRSI.runtime.execution.rpc import receive, send
from XPolicyLab.policy.EmbodiedRSI.runtime.agent.session import agent_command, stop_process
from XPolicyLab.policy.EmbodiedRSI.runtime.workspace import create_workspace

from XPolicyLab.policy.EmbodiedRSI.runtime.xpolicylab.bridge import BridgeTask


def run_worker(connection, specification, observation, step_limit, episode):
    def cancelled(*_):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit("worker cancelled")

    signal.signal(signal.SIGTERM, cancelled)
    run = Path(specification["run_dir"]) / "episodes" / f"episode_{episode:04d}"
    runtime_path = run / "runtime.json"
    cfg = config_from_dict(specification["task_config"])
    process = None
    container = None
    stop_mirror = threading.Event()
    session = None
    phase = specification.get("phase", "test")
    recovery = dict(specification.get("agent_recovery", {}))
    recoveries = []
    recovery_due = None
    recovery_id = None
    runtime = {"status": "starting", "pid": os.getpid(), "native_step_limit": step_limit,
               "started_at": time.time(), "supervisor_pid": specification.get("supervisor_pid"),
               "phase": phase, "env_task": cfg.simulator.env_task,
               "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "model": cfg.agent.model, "reasoning_effort": cfg.agent.reasoning_effort,
               "stdout_log": "agent/stdout.log", "stderr_log": "agent/stderr.log"}
    try:
        run.mkdir(parents=True, exist_ok=False)
        create_workspace(run, Path(specification["run_dir"]) / "inputs",
                         Path(specification["run_dir"]) / "frozen_harness")
        env = BridgeTask(connection, observation, step_limit, phase=phase)
        limits = cfg.test
        session = Session(env, run, phase=phase, seed=0, budget=limits.budget,
                          seconds=specification.get("seconds"), execution_seconds=limits.submission_timeout_sec,
                          fps=cfg.feedback_fps)
        with (tempfile.TemporaryDirectory(prefix="embodiedrsi-", ignore_cleanup_errors=True) as temporary,
              contextlib.ExitStack() as logs):
            socket_dir = Path(temporary)
            socket_dir.chmod(0o755)
            # Keep CLI/Docker diagnostics on the host, outside the agent's mounts.
            # Direct file descriptors avoid pipe backpressure and survive worker exits.
            (run / "agent").mkdir(mode=0o700, exist_ok=True)
            streams = {}
            for name in ("stdout", "stderr"):
                path = run / runtime[f"{name}_log"]
                streams[name] = logs.enter_context(path.open("ab", buffering=0))
                path.chmod(0o600)

            def mirror():
                source = None
                offset = 0
                while not stop_mirror.wait(0.5):
                    files = sorted((socket_dir / "agent-home/.codex/sessions").rglob("*.jsonl"))
                    if not files:
                        continue
                    if source != files[0]:
                        source, offset = files[0], 0
                    with source.open("rb") as stream:
                        stream.seek(offset)
                        data = stream.read()
                    if data:
                        destination = run / "agent/session.jsonl"
                        destination.parent.mkdir(exist_ok=True)
                        with destination.open("ab") as stream:
                            stream.write(data)
                        offset += len(data)

            thread = threading.Thread(target=mirror, daemon=True)
            thread.start()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(socket_dir / "session.sock"))
                os.chmod(socket_dir / "session.sock", 0o600)
                listener.listen(8)
                listener.settimeout(0.5)
                container, argv = agent_command(cfg, run, socket_dir, phase)
                stderr_offset = (run / "agent/stderr.log").stat().st_size
                process = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL,
                                           stdout=streams["stdout"], stderr=streams["stderr"],
                                           start_new_session=True)
                runtime.update(status="running", agent_pid=process.pid, container=container,
                               agent_started_at=time.time())
                write_json(runtime_path, runtime)
                write_json(run / "container-isolation.json", {
                    "container": container, "image": cfg.agent.image,
                    "agent_gpu_access": False, "vendor_mounted": False,
                    "frozen_knowledge_read_only": True,
                })
                while not session.closed:
                    if session.seconds and time.monotonic() - session.started >= session.seconds:
                        session.closed, session.reason = True, "time_budget"
                        break
                    try:
                        client, _ = listener.accept()
                    except TimeoutError:
                        if recovery_due is not None:
                            if time.monotonic() >= recovery_due:
                                stderr_offset = (run / "agent/stderr.log").stat().st_size
                                process = subprocess.Popen(
                                    resume_argv(argv, recovery_id), cwd=REPO, stdin=subprocess.DEVNULL,
                                    stdout=streams["stdout"], stderr=streams["stderr"],
                                    start_new_session=True)
                                recovery_due = None
                                runtime.update(status="running", agent_pid=process.pid,
                                               agent_recoveries=len(recoveries))
                                write_json(runtime_path, runtime)
                            continue
                        if process.poll() is not None:
                            if process.returncode:
                                files = sorted((socket_dir / "agent-home/.codex/sessions").rglob("*.jsonl"))
                                recovery_id, error = rollout_state(files[0]) if files else (None, None)
                                with (run / "agent/stderr.log").open("rb") as stderr:
                                    stderr.seek(max(stderr_offset, stderr.seek(0, 2) - 65536))
                                    error = error or stderr.read().decode(errors="replace")
                                reason = recovery_reason(
                                    state={**session.status(), "valid": session.valid}, returncode=process.returncode,
                                    idle_seconds=0, idle_limit=float("inf"), error=error,
                                ) if recovery and cfg.agent.preset == "codex" else None
                                if reason and recovery_id and len(recoveries) < recovery.get("max_restarts", 3):
                                    delay = min(recovery.get("max_backoff_seconds", 300),
                                                recovery.get("backoff_seconds", 60) * 2 ** len(recoveries))
                                    delay += random.uniform(0, recovery.get("jitter_seconds", 20))
                                    recoveries.append({"at": time.time(), "reason": reason,
                                                       "session_id": recovery_id, "delay_seconds": delay,
                                                       "status": session.status()})
                                    write_json(run / "agent/recovery.json", recoveries)
                                    runtime.update(status="agent_reconnecting", agent_recoveries=len(recoveries))
                                    write_json(runtime_path, runtime)
                                    # Only the CLI restarts. The Session and native environment stay alive.
                                    subprocess.run(["docker", "rm", "-f", container], stdin=subprocess.DEVNULL,
                                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                                    recovery_due = time.monotonic() + delay
                                    continue
                                if reason:
                                    runtime["failure"] = "agent_transport_error"
                                raise RuntimeError(f"Agent exited with status {process.returncode}")
                            session.closed, session.reason = True, "agent_exited"
                        continue
                    with client, contextlib.ExitStack() as close:
                        stream = client.makefile("rwb")
                        # Buffered close may flush a second time after send failed.
                        def close_stream():
                            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                                stream.close()
                        close.callback(close_stream)
                        client.settimeout(10)
                        request = receive(stream)
                        try:
                            response = session.handle(request)
                        except (ValueError, PermissionError) as exc:
                            response = {"ok": False, "error": str(exc), "status": session.status()}
                        session.record_rpc(request, response)
                        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                            send(stream, response)
                    if env.done:
                        session.closed, session.reason = True, "official_episode_end"
                session.save()
                # Give the mirrored transcript time to capture the final tool result.
                time.sleep(0.6)
                runtime.update(status="finished", success=env.success, native_steps=env.control_steps,
                               token_usage=codex_token_usage(socket_dir / "agent-home"),
                               agent_exit_code=process.poll(), reason=session.reason,
                               finished_at=time.time())
                write_json(runtime_path, runtime)
                stop_mirror.set()
                try:
                    subprocess.run(["docker", "rm", "-f", container], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=30, check=True)
                except (subprocess.SubprocessError, OSError) as exc:
                    # Preserve container identity when cleanup itself fails.
                    write_json(run / "agent/cleanup-error.json", {
                        "at": time.time(), "container": container,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                container = None
                stop_process(process)
        connection.send({"kind": "finished", "success": env.success})
    except BaseException:
        error = traceback.format_exc()
        run.mkdir(parents=True, exist_ok=True)
        runtime.update(status="failed", error=error, finished_at=time.time(),
                       agent_exit_code=process.poll() if process is not None else None)
        write_json(runtime_path, runtime)
        if session is not None:
            session.closed, session.valid, session.reason = True, False, "worker_failed"
            session.save()
        with contextlib.suppress(Exception):
            connection.send({"kind": "error", "error": error})
    finally:
        stop_mirror.set()
        if container:
            with contextlib.suppress(subprocess.TimeoutExpired):
                subprocess.run(["docker", "rm", "-f", container], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        stop_process(process)
        connection.close()
