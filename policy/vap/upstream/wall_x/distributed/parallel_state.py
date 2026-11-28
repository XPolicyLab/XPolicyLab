"""Sequence-parallel and data-parallel process group state."""

import logging

import torch.distributed as dist
from typing import Optional

logger = logging.getLogger(__name__)

# Global SP group state
_SP_GROUP: Optional[dist.ProcessGroup] = None
_SP_WORLD_SIZE: int = 1
_SP_RANK: int = 0

# Global DP group state
_DP_GROUP: Optional[dist.ProcessGroup] = None
_DP_WORLD_SIZE: int = 1
_DP_RANK: int = 0

# Track if parallel groups are initialized
_PARALLEL_INITIALIZED: bool = False


def init_sequence_parallel(
    sp_size: int = 1,
    backend: str = "nccl",
) -> None:
    """
    Initialize Sequence Parallelism and Data Parallelism process groups.

    Args:
        sp_size: SP group size (number of GPUs per group)
        backend: Distributed backend, default "nccl"

    Example:
        # 8 GPUs, sp_size=4 will create:
        #
        # 2 SP groups (consecutive partitioning):
        #   SP Group 0: [GPU 0, 1, 2, 3]
        #   SP Group 1: [GPU 4, 5, 6, 7]
        #
        # 4 DP groups (strided partitioning):
        #   DP Group 0: [GPU 0, 4]
        #   DP Group 1: [GPU 1, 5]
        #   DP Group 2: [GPU 2, 6]
        #   DP Group 3: [GPU 3, 7]
    """
    global _SP_GROUP, _SP_WORLD_SIZE, _SP_RANK
    global _DP_GROUP, _DP_WORLD_SIZE, _DP_RANK
    global _PARALLEL_INITIALIZED

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before init_sequence_parallel"
        )

    if _PARALLEL_INITIALIZED:
        # Already initialized, skip
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if sp_size == 1:
        # No SP, DP covers the entire world
        _SP_GROUP = None
        _SP_WORLD_SIZE = 1
        _SP_RANK = 0
        _DP_GROUP = None  # None means use the default world group
        _DP_WORLD_SIZE = world_size
        _DP_RANK = rank
        _PARALLEL_INITIALIZED = True
        if rank == 0:
            logger.info(
                "[Parallel] SP disabled, DP uses world group (size=%s)", world_size
            )
        return

    assert (
        world_size % sp_size == 0
    ), f"world_size ({world_size}) must be divisible by sp_size ({sp_size})"

    num_sp_groups = world_size // sp_size

    # ============ Create SP groups (consecutive partitioning) ============
    # GPUs within an SP group process different sequence segments of the same sample
    for i in range(num_sp_groups):
        ranks = list(range(i * sp_size, (i + 1) * sp_size))
        group = dist.new_group(ranks, backend=backend)
        if rank in ranks:
            _SP_GROUP = group
            _SP_WORLD_SIZE = sp_size
            _SP_RANK = ranks.index(rank)

    # ============ Create DP groups (strided partitioning) ============
    # GPUs within a DP group process different data samples, need gradient synchronization
    # Each DP group contains GPUs from the same position across different SP groups
    num_dp_groups = sp_size
    for i in range(num_dp_groups):
        # Select ranks in strided manner: [i, i+sp_size, i+2*sp_size, ...]
        ranks = list(range(i, world_size, sp_size))
        group = dist.new_group(ranks, backend=backend)
        if rank in ranks:
            _DP_GROUP = group
            _DP_WORLD_SIZE = len(ranks)  # = num_sp_groups
            _DP_RANK = ranks.index(rank)

    _PARALLEL_INITIALIZED = True

    if rank == 0:
        logger.info("[Parallel] Initialized SP+DP groups:")
        logger.info(
            "  - SP: %s groups, each with %s GPUs (consecutive)",
            num_sp_groups,
            sp_size,
        )
        logger.info(
            "  - DP: %s groups, each with %s GPUs (strided)",
            num_dp_groups,
            num_sp_groups,
        )


def is_sp_initialized() -> bool:
    """Check if SP is initialized (sp_size > 1)"""
    return _SP_WORLD_SIZE > 1


def is_parallel_initialized() -> bool:
    """Check if parallel groups are initialized"""
    return _PARALLEL_INITIALIZED


def get_sp_group() -> Optional[dist.ProcessGroup]:
    """Get the SP group that the current process belongs to"""
    return _SP_GROUP


def get_sp_world_size() -> int:
    """Get SP group size"""
    return _SP_WORLD_SIZE


def get_sp_rank() -> int:
    """Get the rank of current process within its SP group"""
    return _SP_RANK


def get_dp_group() -> Optional[dist.ProcessGroup]:
    """
    Get the DP group that the current process belongs to.

    Returns:
        DP process group. Returns None if sp_size=1 (use default world group)
    """
    return _DP_GROUP


def get_dp_world_size() -> int:
    """Get DP group size (i.e., number of SP groups)"""
    return _DP_WORLD_SIZE


def get_dp_rank() -> int:
    """Get the rank of current process within its DP group"""
    return _DP_RANK


def destroy_parallel_groups() -> None:
    """Destroy all parallel groups"""
    global _SP_GROUP, _SP_WORLD_SIZE, _SP_RANK
    global _DP_GROUP, _DP_WORLD_SIZE, _DP_RANK
    global _PARALLEL_INITIALIZED

    if _SP_GROUP is not None:
        dist.destroy_process_group(_SP_GROUP)
    if _DP_GROUP is not None:
        dist.destroy_process_group(_DP_GROUP)

    _SP_GROUP = None
    _SP_WORLD_SIZE = 1
    _SP_RANK = 0
    _DP_GROUP = None
    _DP_WORLD_SIZE = 1
    _DP_RANK = 0
    _PARALLEL_INITIALIZED = False


# Backward compatibility
def destroy_sp_group() -> None:
    """Destroy SP group (backward compatible interface)"""
    destroy_parallel_groups()
