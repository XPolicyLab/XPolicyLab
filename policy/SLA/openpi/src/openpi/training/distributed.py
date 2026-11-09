from __future__ import annotations

import dataclasses
import logging
import os
import time

import jax
from jax.experimental import multihost_utils


@dataclasses.dataclass(frozen=True)
class DistributedInfo:
    process_count: int
    process_index: int
    local_process_index: int
    node_count: int
    coordinator_address: str | None
    local_device_ids: tuple[int, ...]

    @property
    def is_distributed(self) -> bool:
        return self.process_count > 1

    @property
    def is_primary(self) -> bool:
        return self.process_index == 0


_INITIALIZED = False


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value:
            return int(value)
    return default


def _local_device_id(local_process_index: int) -> int:
    visible_count = len([device for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if device])
    if visible_count == 1:
        return 0
    return local_process_index


def _distributed_info_from_env() -> DistributedInfo:
    # Web platforms may set WORLD_SIZE/RANK for the job rather than its workers.
    # Only trust that layout inside a worker launcher or with explicit JAX config.
    launched = (
        bool(os.environ.get("JAX_PROCESS_COUNT"))
        or all(os.environ.get(name) for name in ("TORCHELASTIC_RUN_ID", "LOCAL_RANK", "LOCAL_WORLD_SIZE"))
        or bool(os.environ.get("SLURM_PROCID"))
    )
    if not launched:
        if _env_int(("WORLD_SIZE", "SLURM_NTASKS"), 1) > 1:
            logging.info(
                "No distributed worker launcher detected; using all local visible devices in one process. "
                "For distributed training, use torchrun/srun or explicitly set JAX_PROCESS_COUNT."
            )
        return DistributedInfo(
            process_count=1,
            process_index=0,
            local_process_index=0,
            node_count=1,
            coordinator_address=None,
            local_device_ids=(),
        )

    process_count = _env_int(("JAX_PROCESS_COUNT", "WORLD_SIZE", "SLURM_NTASKS"), 1)
    process_index = _env_int(("JAX_PROCESS_INDEX", "RANK", "SLURM_PROCID"), 0)
    local_process_index = _env_int(("JAX_LOCAL_PROCESS_INDEX", "LOCAL_RANK", "SLURM_LOCALID"), 0)
    local_process_count = _env_int(("JAX_LOCAL_PROCESS_COUNT", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"), 1)
    default_node_count = max(1, process_count // local_process_count)
    node_count = _env_int(("JAX_NODE_COUNT", "SLURM_JOB_NUM_NODES", "PET_NNODES", "NNODES"), default_node_count)
    master_addr = os.environ.get("JAX_COORDINATOR_ADDR", os.environ.get("MASTER_ADDR"))
    master_port = os.environ.get("JAX_COORDINATOR_PORT", os.environ.get("MASTER_PORT"))
    coordinator_address = f"{master_addr}:{master_port}" if master_addr and master_port else None
    local_device_ids = os.environ.get("JAX_LOCAL_DEVICE_IDS")

    return DistributedInfo(
        process_count=process_count,
        process_index=process_index,
        local_process_index=local_process_index,
        node_count=node_count,
        coordinator_address=coordinator_address,
        local_device_ids=tuple(map(int, local_device_ids.split(",")))
        if local_device_ids
        else (_local_device_id(local_process_index),),
    )


def initialize() -> DistributedInfo:
    """Initialize JAX distributed from launcher-provided environment variables."""
    global _INITIALIZED  # noqa: PLW0603

    env_info = _distributed_info_from_env()

    if env_info.process_count > 1 and not _INITIALIZED:
        kwargs = {
            "num_processes": env_info.process_count,
            "process_id": env_info.process_index,
            "local_device_ids": list(env_info.local_device_ids),
            "initialization_timeout": int(os.environ.get("JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT", "600")),
        }
        if env_info.coordinator_address is not None:
            kwargs["coordinator_address"] = env_info.coordinator_address

        logging.info("Initializing JAX distributed with %s", kwargs)
        jax.distributed.initialize(**kwargs)
        _INITIALIZED = True

    info = DistributedInfo(
        process_count=jax.process_count(),
        process_index=jax.process_index(),
        local_process_index=env_info.local_process_index,
        node_count=env_info.node_count,
        coordinator_address=env_info.coordinator_address,
        local_device_ids=tuple(device.local_hardware_id for device in jax.local_devices()),
    )
    logging.info(
        "JAX distributed: process=%s/%s local_process=%s nodes=%s local_devices=%s global_devices=%s coordinator=%s",
        info.process_index,
        info.process_count,
        info.local_process_index,
        info.node_count,
        jax.local_device_count(),
        jax.device_count(),
        info.coordinator_address,
    )
    return info


def is_primary_process() -> bool:
    return jax.process_index() == 0


def local_torch_device_ids(info: DistributedInfo) -> list[int]:
    return list(info.local_device_ids)


def show_tqdm(*, is_primary: bool) -> bool:
    default = "0" if os.environ.get("SLURM_JOB_ID") else "1"
    return is_primary and os.environ.get("OPENPI_SHOW_TQDM", default).lower() in {"1", "true", "yes", "on"}


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def progress_time_info(*, loop_start: float, start_step: int, step: int, total_steps: int) -> str:
    elapsed_s = time.perf_counter() - loop_start
    completed_steps = step - start_step + 1
    remaining_steps = total_steps - step - 1
    eta_s = elapsed_s / completed_steps * remaining_steps
    return f"progress={step + 1}/{total_steps}, elapsed={_format_duration(elapsed_s)}, eta={_format_duration(eta_s)}"


def sync_global_devices(name: str) -> None:
    if jax.process_count() <= 1:
        return

    multihost_utils.sync_global_devices(name)
