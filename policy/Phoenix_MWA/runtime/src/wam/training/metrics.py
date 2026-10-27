from collections.abc import Mapping

import torch


def gather_mean_scalars(
    accelerator,
    metrics: Mapping[str, torch.Tensor | float | int],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Reduce scalar metrics across ranks with one collective."""
    names = sorted(metrics)
    if not names:
        return {}

    local_values = []
    for name in names:
        value = torch.as_tensor(metrics[name], device=device, dtype=torch.float32)
        if value.numel() != 1:
            raise ValueError(f"Metric `{name}` must be scalar, got shape {tuple(value.shape)}.")
        local_values.append(value.reshape(()))

    local = torch.stack(local_values).reshape(1, -1)
    reduced = accelerator.gather(local).mean(dim=0).cpu().tolist()
    return {name: float(value) for name, value in zip(names, reduced)}
