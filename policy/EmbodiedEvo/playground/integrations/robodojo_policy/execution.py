"""Shared single-scene XPolicyLab execution for Debug and Slurm campaigns."""

from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import socket
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace

from orchestrator.config import CODE_ROOT, REPO, config_dict
from orchestrator.files import hashes, write_json
from orchestrator.robodojo_apptainer import apptainer_command
from orchestrator.session_runner import stop_process
from .transport import forward


def read(path):
    return json.loads(path.read_text())


def start_native(command, environment, run, stderr_tail):
    """Serialize GPU provisioning; retry only before any native code has started."""
    ready = run / "container-ready"
    if command[-2] == "-c":
        command = [*command[:-1], f"touch {shlex.quote(str(ready))}; " + command[-1]]
    lock = Path(os.environ.get("EMBODIEDEVO_RUNTIME_DIR", "/tmp")) / f"embodiedevo-native-start-{os.getuid()}.lock"
    attempts = []
    for attempt in range(3):
        process = reader = None
        try:
            with lock.open("a") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                process = subprocess.Popen(command, cwd=REPO, env=environment,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    start_new_session=True)
                def drain(stream=process.stderr):
                    with stream:
                        while chunk := stream.read(4096):
                            stderr_tail.append(chunk)
                reader = threading.Thread(target=drain, daemon=True)
                reader.start()
                deadline = time.monotonic() + 180
                while not ready.exists() and process.poll() is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Native container provisioning timed out")
                    time.sleep(0.2)
            if ready.exists() or process.returncode == 0:
                return process, reader
            reader.join(timeout=5)
            error = b"".join(stderr_tail).decode(errors="replace")
            if "nvidia-container-cli" not in error or "driver rpc error: timed out" not in error:
                return process, reader
            attempts.append({"at": time.time(), "attempt": attempt + 1,
                             "returncode": process.returncode, "stderr_tail": error})
            write_json(run / "native-launch-attempts.json", attempts)
            if attempt == 2:
                return process, reader
            time.sleep(30 * (attempt + 1))
            stderr_tail.clear()
        except BaseException:
            stop_process(process)
            if reader is not None:
                reader.join(timeout=5)
            raise


def episode(cfg, task_run, run, phase, identity, gpu, args, *, flat=False, seconds=None):
    from orchestrator.slurm import allocated_gpus

    installed_profile = REPO / ".venv-robodojo/runtime.json"
    profile = read(installed_profile) if installed_profile.is_file() else {}
    if override := os.environ.get("EMBODIEDEVO_ROBODOJO_RUNTIME_PROFILE"):
        profile.update(read(Path(override)))
    allocated = allocated_gpus(os.environ["SLURM_JOB_GPUS"]) if os.environ.get("SLURM_JOB_GPUS") else set()
    from orchestrator.host_policy import require_allowed

    require_allowed("local")
    if not os.environ.get("SLURM_JOB_ID") or str(gpu) not in allocated:
        raise ValueError("XPolicyLab requires the selected GPU to belong to this Slurm allocation")
    if cfg.baseline and phase != "test":
        raise ValueError("Baseline only runs Test")
    # Campaign Test creates identity.json before dispatching the scene.
    run.mkdir(parents=True, exist_ok=flat)
    shutil.copytree(task_run / "inputs", run / "inputs")
    inherited = phase == "test" and not cfg.baseline
    if inherited:
        shutil.copytree(task_run / "frozen_harness", run / "frozen_harness")
        if (task_run / "program-records.json").exists():
            shutil.copyfile(task_run / "program-records.json", run / "program-records.json")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    bank, ordinal = divmod(identity, 1_000_000)
    spec = {
        "run_dir": str(run), "task_config": config_dict(cfg), "env_task": cfg.simulator.env_task,
        "phase": phase, "seed": bank, "layout_ordinal": ordinal, "episodes": 1, "port": port,
        "gpu": gpu, "diagnostic": args.probe, "protocol": "xpolicylab-ws-workspace-session-v1",
        "flat_episode": flat, "seconds": seconds,
        "supervisor_pid": os.getpid(),
        "probe_reset_cycles": getattr(args, "probe_reset_cycles", 1),
    }
    session_run = run if flat else run / "episodes/episode_0000"
    worker_runtime = session_run / ("agent/runtime.json" if flat else "runtime.json")
    write_json(run / "adapter.json", spec)
    server = client = None
    stderr_tail = deque(maxlen=16)
    reader = None
    state = {"status": "starting", "phase": phase, "gpu": gpu, "started_at": time.time(),
             "supervisor_pid": os.getpid(), "backend": "robodojo", "execution_backend": "xpolicylab"}
    write_json(run / "runtime.json", state)
    try:
        server = subprocess.Popen([
            str(REPO / ".venv-robodojo-policy/bin/python"), "-B", "-m",
            "integrations.robodojo_policy.bootstrap", "server", "--run-dir", str(run),
        ], cwd=CODE_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + 90
        while not (run / "policy-server.json").exists():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Policy server did not start; inspect server-error.json")
            time.sleep(0.5)
        (run / "official").mkdir()
        profile["cache_dir"] = str(args.run_dir / "nv-cache")
        if args.material_mirror:
            profile["material_mirror"] = str(args.material_mirror.resolve())
        with tempfile.TemporaryDirectory(prefix="embodiedevo-xp-") as transport_dir:
            unix_path = str(Path(transport_dir) / "policy.sock")
            with forward(unix_path, (socket.AF_INET, ("127.0.0.1", port)), unix=True):
                temporary_root = Path(profile.get("temporary_root", args.run_dir / "runtime-tmp"))
                temporary_root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="sim-", dir=temporary_root,
                                                 ignore_cleanup_errors=True) as tmp:
                    bridge = Path(tmp)
                    (bridge / "home").mkdir()
                    (bridge / "work").mkdir()
                    native_argv = [
                        "-m", "integrations.robodojo_policy.bootstrap", "client", "--run-dir", str(run),
                        "--gpu", str(gpu), "--transport-socket", "/run/embodiedevo-policy/policy.sock",
                        "--cpu-threads", "8", "--video-threads", "2",
                    ]
                    if profile.get("reserve_monitor_pipe"):
                        native_argv += ["--reserve-monitor-pipe"]
                    capacity = profile.get("collision_stack_bytes_by_task", {}).get(cfg.simulator.env_task, 0)
                    if capacity:
                        native_argv += ["--collision-stack-bytes", str(capacity)]
                    patch_count = profile.get("rigid_patch_count_by_task", {}).get(cfg.simulator.env_task, 0)
                    if patch_count:
                        native_argv += ["--rigid-patch-count", str(patch_count)]
                    if cfg.simulator.env_task in profile.get("reuse_reset_rigid_bodies_tasks", []):
                        if phase != "playground":
                            raise ValueError("Fixed-scene rigid reuse is only supported in Playground")
                        native_argv += ["--reuse-reset-rigid-bodies"]
                    if cfg.simulator.env_task in profile.get("disable_retired_collision_tasks", []):
                        if phase != "playground":
                            raise ValueError("Retired collision cleanup is only supported in Playground")
                        native_argv += ["--disable-retired-collisions"]
                    command = apptainer_command(native_argv, profile, bridge, args.sif)
                    at = command.index(str(args.sif))
                    command[at:at] = ["--bind", f"{transport_dir}:/run/embodiedevo-policy"]
                    environment = dict(os.environ, NVIDIA_VISIBLE_DEVICES=str(gpu),
                                       NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics,display,video")
                    if cpus := environment.get("EMBODIEDEVO_SIMULATOR_CPUSET"):
                        command = ["taskset", "-c", cpus, *command]
                    try:
                        client, reader = start_native(command, environment, run, stderr_tail)
                        state.update(status="running", server_pid=server.pid, simulator_pid=client.pid,
                                     slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                                     dedicated_host=profile.get("dedicated_host"),
                                     cpuset=([int(cpu) for cpu in cpus.split(",")]
                                             if cpus else sorted(os.sched_getaffinity(0))))
                        write_json(run / "runtime.json", state)
                        until = time.monotonic() + args.episode_timeout
                        while client.poll() is None:
                            if worker_runtime.exists() and read(worker_runtime).get("status") == "failed":
                                raise RuntimeError("Policy worker failed; inspect agent/runtime.json")
                            if "ready_at" not in state and (session_run / "session-result.json").exists():
                                state["ready_at"] = time.time()
                                write_json(run / "runtime.json", state)
                            if (run / "client-error.json").exists():
                                raise RuntimeError("Native client failed; inspect client-error.json")
                            if server.poll() is not None or time.monotonic() > until:
                                raise RuntimeError("Policy server exited or episode watchdog expired")
                            time.sleep(1)
                    finally:
                        try:
                            for log in bridge.glob("kit/logs/Kit/Isaac-Sim/*/*.log"):
                                with log.open("rb") as stream:
                                    stream.seek(max(0, log.stat().st_size - 1024 * 1024))
                                    (run / "native-kit-tail.log").write_bytes(stream.read())
                        except OSError as exc:
                            state["native_log_capture_error"] = str(exc)
                    if client.returncode:
                        raise RuntimeError(f"Simulator exited with code {client.returncode}")
        result = read(session_run / "session-result.json")
        worker = read(worker_runtime)
        if not result["valid"] or worker["status"] != "finished":
            raise RuntimeError("Policy worker did not finish a valid session")
        if inherited:
            expected = read(task_run / "protocol.json")["harness"]
            if hashes(run / "frozen_harness") != expected or hashes(task_run / "frozen_harness") != expected:
                raise ValueError("Frozen knowledge changed")
        if phase == "test" and not args.probe:
            reports = list((run / "official/eval_result").rglob("_result.json"))
            if len(reports) != 1:
                raise ValueError("Official evaluator did not score exactly one scene")
            report = read(reports[0])
            details = list(report.get("details", {}).values())
            score = report.get("score")
            if (report.get("eval_time") != 1 or len(details) != 1
                    or details[0].get("layout_id") != ordinal
                    or isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 100):
                raise ValueError("Official result has a missing score or wrong scene identity")
            result.update(official_result=str(reports[0].relative_to(run)), score=score,
                          task_completed=bool(details[0]["success"]))
        if flat:
            timing = {"phase": phase, "stage_started_at": state["started_at"],
                      "stage_finished_at": time.time(),
                      "simulator_ready_at": worker.get("started_at"),
                      "agent_started_at": worker.get("agent_started_at"),
                      "token_usage": worker.get("token_usage", {})}
            write_json(run / "timing.json", timing)
            result["timing"] = timing
        result["diagnostic"] = args.probe
        write_json(run / "result.json", result)
        state.update(status="finished", valid=True)
        return result
    except BaseException as exc:
        state.update(status="failed", valid=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_process(client)
        stop_process(server)
        if reader is not None:
            reader.join(timeout=5)
        state["stderr_tail"] = b"".join(stderr_tail).decode(errors="replace")
        write_json(run / "runtime.json", dict(state, finished_at=time.time()))


def run_episode(cfg, run, experiment, *, phase, seed, seconds, gpu, harness=None,
                submission=None):
    """Keep campaign workspace/result paths while using the shared evaluator."""
    if submission is not None:
        raise ValueError("XPolicyLab sessions do not inherit submitted programs")
    expected = experiment / "frozen_harness" if phase == "test" and not cfg.baseline else None
    if harness != expected:
        raise ValueError("Unexpected frozen harness for this phase or Baseline")
    profile = read(Path(os.environ["EMBODIEDEVO_ROBODOJO_RUNTIME_PROFILE"]))
    sif = Path(profile["sif"])
    if not sif.is_file():
        raise FileNotFoundError(sif)
    for component in ("runtime-tmp", "nv-cache"):
        (experiment / component).mkdir(exist_ok=True)
    args = SimpleNamespace(run_dir=experiment, sif=sif, probe=False,
                           material_mirror=(Path(profile["material_mirror"]) if profile.get("material_mirror") else None),
                           episode_timeout=profile.get("episode_timeout", 28800))
    write_json(run / "episode-config.json", config_dict(cfg))
    return episode(cfg, experiment, run, phase, seed, gpu, args, flat=True, seconds=seconds)
