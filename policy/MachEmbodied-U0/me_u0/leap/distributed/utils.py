"""Distributed training utilities."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.distributed as dist


def is_dist_initialized() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process."""
    if is_dist_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if is_dist_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_logging(rank: int | None = None) -> None:
    """Configure logging so only rank 0 emits INFO-level messages.

    Non-rank-0 processes have their root logger level raised to WARNING,
    which silences ``logger.info()`` / ``logger.debug()`` across the
    entire process while still allowing warnings and errors through.

    Safe to call when distributed training is not active (rank defaults
    to 0, so nothing changes).
    """
    if rank is None:
        rank = get_rank()
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.getLogger().setLevel(logging.WARNING)


def barrier() -> None:
    """Synchronize all processes."""
    if is_dist_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            device_idx = get_local_rank()
            torch.cuda.set_device(device_idx)
            dist.barrier(device_ids=[device_idx])
        else:
            dist.barrier()


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """All-reduce a tensor across processes."""
    if is_dist_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather_object(obj: Any) -> list:
    """Gather an arbitrary Python object from all processes."""
    if not is_dist_initialized():
        return [obj]
    output = [None] * get_world_size()
    dist.all_gather_object(output, obj)
    return output


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a Python object from src rank to all processes."""
    if not is_dist_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce a tensor by taking the mean across all processes."""
    if not is_dist_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor
