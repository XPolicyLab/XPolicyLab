"""Some operations(comm and mapping) that are used in context parallel."""

import torch
import torch.distributed as dist

from falcon.distributed import all_to_all_2_or_3D, get_cp_group


def all_to_all_flat(
    inputs: torch.Tensor | list[torch.Tensor],
    scatter_idx: int = 1,
    gather_idx: int = 0,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor | list[torch.Tensor]:
    """All-to-all without blocking/interleaving (flat contiguous CP split).

    With flat split, the all_to_all gather produces sequences already in global
    order, so no blocking permutation is needed.

    Args:
        inputs: single tensor or list of tensors to all-to-all.
        scatter_idx: dimension to scatter (default 1 = head dim).
        gather_idx: dimension to gather (default 0 = seq dim).
        group: process group (defaults to CP group).

    Returns:
        All-to-all'd tensor(s), same type as input.
    """
    if group is None:
        group = get_cp_group()

    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    outputs: list[torch.Tensor] = []
    for data in inputs:
        data = all_to_all_2_or_3D(data, scatter_idx=scatter_idx, gather_idx=gather_idx, group=group)
        outputs.append(data)

    return outputs if len(outputs) > 1 else outputs[0]
