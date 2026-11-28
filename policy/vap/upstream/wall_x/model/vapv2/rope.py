"""Text-only adaptation to (chunk_start, local_time, height, width) RoPE."""
import torch

def text_position_ids(valid):
    """Valid tokens get first-axis -length,...,-1; other axes zero.

    Coordinates use real token count, not padded capacity. V/A and Memory ids
    are never shifted. Negative text positions are symbolic language order,
    not physical age; text projections identify the modality. Coordinates may
    overlap Memory and are not claimed to be a disjoint namespace.
    """
    if valid.dtype != torch.bool or valid.ndim != 2:
        raise ValueError("Text validity must be bool [B,L]")
    order = valid.long().cumsum(dim=1) - valid.long().sum(dim=1, keepdim=True) - 1
    positions = torch.zeros(*valid.shape, 4, dtype=torch.long, device=valid.device)
    positions[..., 0] = order.masked_fill(~valid, 0)
    return positions
