"""Host-level GPU binding and optional exclusive-GPU leases for simulators."""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from collections.abc import Iterator
from pathlib import Path


def sim_env(gpu: str | None, backend: str | None = None) -> dict[str, str]:
    """Bind the simulator to one GPU for both CUDA and EGL.

    Hide all other GPUs and select the first visible EGL device.
    """
    from orchestrator.host_policy import require_allowed

    require_allowed("local")
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    # Prevent native BLAS/OpenMP pools from multiplying across the simulator,
    # API workers and the agent. Oversubscription makes EGL aborts much more
    # likely when several campaigns are active on one host.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    if gpu is not None and gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["MUJOCO_EGL_DEVICE_ID"] = "0"
    if backend == "robodojo":
        env.update(
            {
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "PYTHONNOUSERSITE": "1",
                "PIP_USER": "0",
                "PIP_NO_COMPILE": "1",
                "XDG_RUNTIME_DIR": f"/tmp/robodojo-runtime-{os.getuid()}",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }
        )
        for path in (
            "/etc/vulkan/icd.d/nvidia_icd.json",
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
        ):
            if os.path.isfile(path):
                env["VK_ICD_FILENAMES"] = path
                break
    return env


@contextlib.contextmanager
def gpu_lease(gpu: str | None) -> Iterator[None]:
    """Optionally serialize simulators targeting one physical GPU.

    H100-class cards can host multiple simulator processes.  The default is
    therefore concurrent execution; set ``EMBODIEDEVO_EXCLUSIVE_GPU=1`` only
    when diagnosing a driver-specific native rendering failure.
    """
    if os.environ.get("EMBODIEDEVO_EXCLUSIVE_GPU") != "1" or gpu is None or str(gpu).strip() == "":
        yield
        return
    gpu_id = str(gpu).strip().replace("/", "_")
    path = Path("/tmp") / f"embodiedevo-simulator-gpu-{gpu_id}.lock"
    with path.open("w", encoding="utf-8") as fh:
        started = time.monotonic()
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        waited = time.monotonic() - started
        if waited >= 1.0:
            print(f"simulator: acquired GPU {gpu} after waiting {waited:.1f}s", flush=True)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class GpuSlots:
    """Process-shared slots, acquired before starting a simulator.

    All schedulers on this host must use the same per-GPU limit. File locks
    also serialize threads and are released if the scheduler exits.
    """

    def __init__(self, gpus, per_gpu, *, directory=None):
        self.gpus = [str(gpu).strip() for gpu in gpus]
        if not self.gpus or any(not gpu.isdigit() for gpu in self.gpus):
            raise ValueError("GPU pool must contain physical GPU indices")
        self.gpus = [str(int(gpu)) for gpu in self.gpus]
        if len(set(self.gpus)) != len(self.gpus) or per_gpu < 1:
            raise ValueError("GPU indices must be unique and per-GPU capacity positive")
        self.per_gpu = per_gpu
        self.directory = Path(directory or f"/tmp/embodiedevo-gpu-slots-{os.getuid()}")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    def acquire(self, *, allowed=None):
        """Return (physical GPU, open lock), or None when all slots are busy.

        Fill one slot on every card before using the next slot per card.
        """
        allowed = set(self.gpus if allowed is None else map(str, allowed))
        for slot in range(self.per_gpu):
            for gpu in self.gpus:
                if gpu not in allowed:
                    continue
                handle = (self.directory / f"gpu-{gpu}-slot-{slot}.lock").open("a")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                return gpu, handle
        return None

    @staticmethod
    def release(lease):
        if lease is not None:
            lease[1].close()

    @contextlib.contextmanager
    def lease(self):
        selected = None
        try:
            while selected is None:
                selected = self.acquire()
                if selected is None:
                    time.sleep(0.1)
            yield selected[0]
        finally:
            self.release(selected)
