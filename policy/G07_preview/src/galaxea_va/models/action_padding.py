import torch


def zero_padded_action_dims(
    action: torch.Tensor,
    action_dim_is_pad: torch.Tensor | None,
) -> torch.Tensor:
    """Zero dimensions marked by a boolean ``[B,D]`` embodiment padding mask."""
    if action_dim_is_pad is None:
        return action
    dim_mask = action_dim_is_pad.to(device=action.device, dtype=torch.bool).unsqueeze(1)
    return action.masked_fill(dim_mask, 0.0)
