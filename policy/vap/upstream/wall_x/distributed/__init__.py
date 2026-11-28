# Sequence Parallelism and Data Parallelism Support for wall-x
from .parallel_state import (
    # SP functions
    init_sequence_parallel,
    get_sp_group,
    get_sp_world_size,
    get_sp_rank,
    is_sp_initialized,
    destroy_sp_group,
    # DP functions (new)
    get_dp_group,
    get_dp_world_size,
    get_dp_rank,
    # Utility functions
    is_parallel_initialized,
    destroy_parallel_groups,
)
from .communication_op import (
    sp_shard,
    sp_gather,
    sp_all_to_all_4d,
    sp_all_gather_4d,
    reduce_sp_partial_grads,
)
