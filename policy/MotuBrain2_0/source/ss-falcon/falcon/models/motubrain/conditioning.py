"""Sequence layout for external action conditions."""

import torch


def condition_key_layout(
    base_lengths: torch.Tensor,
    condition_lengths: torch.Tensor,
    condition_padded_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Index concatenated base/condition keys into sample order, excluding condition padding."""
    base = base_lengths.tolist()
    lengths = condition_lengths.tolist()
    padded = condition_padded_lengths.tolist()
    if len(base) != len(lengths) or len(lengths) != len(padded):
        raise ValueError("condition and attention lengths must contain the same samples")
    if any(length < 0 or length > padding for length, padding in zip(lengths, padded, strict=True)):
        raise ValueError("condition lengths must satisfy 0 <= length <= padded length")
    combined = base_lengths.to(dtype=torch.int32) + condition_lengths.to(base_lengths.device, dtype=torch.int32)
    cumulative = torch.cat([combined.new_zeros(1), combined.cumsum(0, dtype=torch.int32)])
    total = sum(base) + sum(lengths)
    sample = torch.repeat_interleave(
        torch.arange(len(base), device=base_lengths.device),
        combined,
        output_size=total,
    )
    position = torch.arange(total, device=base_lengths.device) - cumulative[sample]
    base_offsets = base_lengths.cumsum(0) - base_lengths
    condition_offsets = condition_padded_lengths.cumsum(0) - condition_padded_lengths
    indices = torch.where(
        position < base_lengths[sample],
        base_offsets[sample] + position,
        sum(base) + condition_offsets[sample] + position - base_lengths[sample],
    )
    return indices, cumulative, max(count + length for count, length in zip(base, lengths, strict=True))
