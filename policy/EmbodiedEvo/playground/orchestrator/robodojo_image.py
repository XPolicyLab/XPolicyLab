"""Use a pinned RoboDojo image as the simulator interpreter, never as the agent.

The provisioned .venv-robodojo/bin/python delegates here. EmbodiedEvo still owns
the session, adapter, budgets, source snapshot and all run artifacts.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path

from orchestrator.config import CODE_ROOT, REPO
from orchestrator.files import write_json


def option(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def docker_command(argv, profile, bridge, name):
    run_arg = option(argv, "--run-dir")
    if not run_arg:
        raise ValueError("The image simulator requires an explicit --run-dir")
    run = Path(run_arg).resolve()
    gpu = option(argv, "--gpu") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu is None or not gpu.isdecimal():
        raise ValueError("Select exactly one physical GPU for the image simulator")
    image = profile["image"]
    if not image.startswith("sha256:") or len(image) != 71:
        raise ValueError("Pin the simulator image by its full sha256 image ID")
    source = REPO / "vendor/robodojo"
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Provision a real repository-local RoboDojo source snapshot")
    args = [
        "docker", "run", "--rm", "--init", "--name", name,
        "--label", "embodiedevo.role=simulator", "--log-driver", "none",
        "--gpus", f"device={gpu}", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--shm-size", "8g", "--user", f"{os.getuid()}:{os.getgid()}",
        "--tmpfs", "/tmp:rw,exec,mode=1777,size=8g",
        "-v", f"{REPO}:{REPO}:ro",
        # Image-installed editable packages use this path. Bind the exact
        # repository snapshot there too, rather than importing another checkout.
        "-v", f"{source}:{profile['source_root']}:ro",
        "-v", f"{run.parent}:{run.parent}",
        "-v", f"{bridge}:/run/embodiedevo",
    ]
    if cpuset := os.environ.get("EMBODIEDEVO_SIMULATOR_CPUSET"):
        # A Docker daemon does not inherit a Slurm task's CPU affinity.
        args += ["--cpuset-cpus", cpuset]
    if socket_arg := option(argv, "--socket"):
        folder = Path(socket_arg).resolve().parent
        args += ["-v", f"{folder}:{folder}"]
    cache = Path(profile["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    args += ["-v", f"{cache}:/tmp/.nv"]
    if mirror := profile.get("material_mirror"):
        args += ["-v", f"{mirror}:/opt/embodiedevo-mdl:ro"]
    kit_args = (
        f"--ext-folder {shlex.quote(profile.get('kit_ext_cache', '/opt/eah/kit-ext-cache/3'))} "
        "--/app/extensions/registryEnabled=false "
        "--/app/extensions/fsWatcherEnabled=false "
        "--/crashreporter/url='' --/crashreporter/skipOldDumpUpload=true "
        "--/privacy/performance=false --/privacy/usage=false "
        "--portable-root /run/embodiedevo/kit"
    )
    if threads := profile.get("kit_usd_threads"):
        kit_args += f" --/plugins/carb.tasking.plugin/threadCount={threads}"
    environment = {
        "HOME": "/run/embodiedevo/home", "XDG_RUNTIME_DIR": "/run/embodiedevo/xdg",
        "PYTHONPATH": str(CODE_ROOT), "EMBODIEDEVO_REPO": str(REPO),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "NVIDIA_DRIVER_CAPABILITIES": "all",
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "EMBODIEDEVO_ROBODOJO_KIT_ARGS": kit_args,
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    if mirror:
        environment["EMBODIEDEVO_MDL_MIRROR"] = "/opt/embodiedevo-mdl"
    if profile.get("kit_usd_threads"):
        environment["PXR_WORK_THREAD_LIMIT"] = str(profile["kit_usd_threads"])
    if profile.get("video_threads"):
        environment["EMBODIEDEVO_ROBODOJO_VIDEO_THREADS"] = str(profile["video_threads"])
    if profile.get("renderer_barrier"):
        environment["EMBODIEDEVO_ROBODOJO_RENDERER_BARRIER"] = "1"
    if threads := option(argv, "--cpu-threads"):
        if int(threads) > 0:
            # USD can be imported by image_entrypoint before the worker runs.
            environment["PXR_WORK_THREAD_LIMIT"] = threads
    for key, value in environment.items():
        args += ["-e", f"{key}={value}"]
    inner = list(argv)
    if "--gpu" in inner:
        # Docker exposes only the selected physical card, at container index 0.
        inner[inner.index("--gpu") + 1] = "0"
    redirect = ""
    if "--ready-fd" in inner:
        inner[inner.index("--ready-fd") + 1] = "3"
        redirect = "exec 3>/run/embodiedevo/events; "
    invocation = [profile["python"], "-m", "simulators.image_entrypoint", *inner]
    gate_wait = ""
    if gate := option(argv, "--container-start-gate"):
        # This shell is reached only after the NVIDIA prestart hook succeeds.
        # Keep Kit/CUDA imports out of the GPU-hook provisioning phase.
        gate_wait = (
            f"touch {shlex.quote(str(run / 'container-ready'))}; "
            f"while [ ! -e {shlex.quote(str(Path(gate).resolve()))} ]; do "
            f"if [ -e {shlex.quote(str(run.parent / 'STOP'))} ]; then exit 143; fi; "
            "sleep 0.2; done; "
        )
    shell = (
        'unset DISPLAY; mkdir -p "$HOME" "$XDG_RUNTIME_DIR" /run/embodiedevo/kit; '
        'chmod 700 "$XDG_RUNTIME_DIR"; ' + gate_wait + redirect + "exec " + shlex.join(invocation)
    )
    return [*args, "-w", str(CODE_ROOT), "--entrypoint", "/bin/bash", image, "-c", shell]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    profile = json.loads(Path(os.environ.get(
        "EMBODIEDEVO_ROBODOJO_RUNTIME_PROFILE", REPO / ".venv-robodojo/runtime.json"
    )).read_text())
    run_arg = option(argv, "--run-dir")
    if not run_arg:
        raise ValueError("The image simulator requires an explicit --run-dir")
    run = Path(run_arg).resolve()
    run.parent.mkdir(parents=True, exist_ok=True)
    outgoing = option(argv, "--ready-fd")
    outgoing = int(outgoing) if outgoing is not None else None
    name = f"embodiedevo-sim-{os.getppid()}-{os.getpid()}"
    state = {
        "name": name, "owner": os.getppid(), "pid": os.getpid(),
        "hostname": socket.gethostname(), "image": profile["image"],
    }
    # Smoke commands create their own run directory and must see it absent.
    marker = run / "simulator-container.json" if run.exists() else None

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with tempfile.TemporaryDirectory(
        prefix="embodiedevo-image-", dir=os.environ.get("EMBODIEDEVO_SIMULATOR_TMPDIR")
    ) as temporary:
        bridge = Path(temporary)
        bridge.chmod(0o755)
        state["temporary_directory"] = str(bridge)
        incoming = process = None
        stderr_tail = deque(maxlen=16)
        stderr_reader = None
        try:
            if outgoing is not None:
                os.mkfifo(bridge / "events", mode=0o600)
                incoming = os.open(bridge / "events", os.O_RDWR | os.O_NONBLOCK)
            command = docker_command(argv, profile, bridge, name)
            if marker:
                write_json(marker, {**state, "status": "starting"})
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)

            def drain_stderr():
                while chunk := process.stderr.read(8192):
                    stderr_tail.append(chunk)

            stderr_reader = threading.Thread(target=drain_stderr, daemon=True)
            stderr_reader.start()
            while process.poll() is None:
                if incoming is None:
                    try:
                        process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        pass
                elif select.select([incoming], [], [], 0.2)[0]:
                    os.write(outgoing, os.read(incoming, 65536))
            if incoming is not None:
                while select.select([incoming], [], [], 0)[0]:
                    os.write(outgoing, os.read(incoming, 65536))
            return process.returncode
        finally:
            # Preserve native diagnostics before removing the private Kit home,
            # including native exit(0)/abort paths that bypass Python exceptions.
            if run.exists():
                for path in bridge.glob("kit/logs/Kit/Isaac-Sim/*/*.log"):
                    with contextlib.suppress(OSError):
                        with path.open("rb") as stream:
                            stream.seek(max(0, path.stat().st_size - 2 * 1024**2))
                            (run / "native-kit-tail.log").write_bytes(stream.read())
            with contextlib.suppress(subprocess.SubprocessError):
                subprocess.run(
                    ["docker", "rm", "-f", name], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=30, check=False,
                )
            if process is not None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
            if stderr_reader is not None:
                stderr_reader.join(timeout=1)
            if incoming is not None:
                os.close(incoming)
                os.close(outgoing)
            if marker:
                write_json(marker, {**state, "status": "stopped",
                                    "stderr_tail": b"".join(stderr_tail).decode(errors="replace"),
                                    "docker_exit_code": process.returncode if process else None})


if __name__ == "__main__":
    raise SystemExit(main())
