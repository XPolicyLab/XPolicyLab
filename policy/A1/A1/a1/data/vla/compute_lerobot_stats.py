import os
import json
import argparse
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import LeRobotDataset. Ensure lerobot is installed and on PYTHONPATH."
    ) from e


def _init_running_stats(dimension: int) -> Dict[str, torch.Tensor]:
    device = torch.device("cpu")
    return {
        "count": torch.zeros((), dtype=torch.long, device=device),
        "sum": torch.zeros((dimension,), dtype=torch.float64, device=device),
        "sumsq": torch.zeros((dimension,), dtype=torch.float64, device=device),
        "min": torch.full((dimension,), float("inf"), dtype=torch.float64, device=device),
        "max": torch.full((dimension,), float("-inf"), dtype=torch.float64, device=device),
    }


def _update_running_stats(stats: Dict[str, torch.Tensor], batch_values: torch.Tensor) -> None:
    # batch_values: (batch, dim)
    if batch_values.numel() == 0:
        return
    batch_values64 = batch_values.to(dtype=torch.float64, device=stats["sum"].device)
    stats["count"] += torch.tensor(batch_values64.shape[0], dtype=torch.long, device=stats["count"].device)
    stats["sum"] += batch_values64.sum(dim=0)
    stats["sumsq"] += (batch_values64.square()).sum(dim=0)
    stats["min"] = torch.minimum(stats["min"], batch_values64.min(dim=0).values)
    stats["max"] = torch.maximum(stats["max"], batch_values64.max(dim=0).values)


def _finalize_running_stats(stats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Returns mean, std, min, max as float32 tensors
    count = stats["count"].item()
    if count == 0:
        raise ValueError("No samples found to compute statistics.")
    sum_v = stats["sum"]
    sumsq_v = stats["sumsq"]
    mean = (sum_v / count).to(dtype=torch.float32)
    var = (sumsq_v / count) - (mean.to(dtype=torch.float64).square())
    var = torch.clamp(var, min=0.0).to(dtype=torch.float32)
    std = torch.sqrt(var)
    min_v = stats["min"].to(dtype=torch.float32)
    max_v = stats["max"].to(dtype=torch.float32)
    return mean, std, min_v, max_v


def compute_stats(
    data_root_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
    max_batches: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    dataset = LeRobotDataset(data_root_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    first_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    state_key = "state" if "state" in first_batch else "observation.state" if "observation.state" in first_batch else None
    actions_key = "actions" if "actions" in first_batch else "action" if "action" in first_batch else None

    if state_key is None:
        raise KeyError("Could not find 'state' or 'observation.state' in dataset samples.")
    if actions_key is None:
        raise KeyError("Could not find 'actions' or 'action' in dataset samples.")

    # Probe dimensions
    sample_state = first_batch[state_key].reshape(1, -1)
    sample_actions = first_batch[actions_key].reshape(1, -1)
    state_dim = sample_state.shape[-1]
    actions_dim = sample_actions.shape[-1]

    # Running stats
    state_stats = _init_running_stats(state_dim)
    actions_stats = _init_running_stats(actions_dim)

    # For exact percentiles we gather all values; adjust if OOM.
    state_values_all = []  # list of (N_i, state_dim)
    actions_values_all = []  # list of (N_i, actions_dim)

    for batch_idx, batch in enumerate(dataloader):
        print(f"batch_idx/all batches: {batch_idx}/{len(dataloader)}")
        state = batch[state_key]
        actions = batch[actions_key]

        # Ensure shape (B, D)
        state_flat = state.view(state.shape[0], -1).to(torch.float32)
        actions_flat = actions.view(actions.shape[0], -1).to(torch.float32)

        _update_running_stats(state_stats, state_flat)
        _update_running_stats(actions_stats, actions_flat)

        state_values_all.append(state_flat.cpu())
        actions_values_all.append(actions_flat.cpu())

        if max_batches > 0 and (batch_idx + 1) >= max_batches:
            break

    # Finalize mean/std/min/max
    state_mean, state_std, state_min, state_max = _finalize_running_stats(state_stats)
    actions_mean, actions_std, actions_min, actions_max = _finalize_running_stats(actions_stats)

    # Quantiles (0.01, 0.99)
    state_all = torch.cat(state_values_all, dim=0)
    actions_all = torch.cat(actions_values_all, dim=0)
    q01 = 0.01
    q99 = 0.99
    state_q01 = torch.quantile(state_all, q01, dim=0)
    state_q99 = torch.quantile(state_all, q99, dim=0)
    actions_q01 = torch.quantile(actions_all, q01, dim=0)
    actions_q99 = torch.quantile(actions_all, q99, dim=0)

    return {
        "state": {
            "mean": state_mean,
            "std": state_std,
            "min": state_min,
            "max": state_max,
            "q01": state_q01,
            "q99": state_q99,
        },
        "actions": {
            "mean": actions_mean,
            "std": actions_std,
            "min": actions_min,
            "max": actions_max,
            "q01": actions_q01,
            "q99": actions_q99,
        },
    }


def _tensor_to_list(d: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, list]]:
    out: Dict[str, Dict[str, list]] = {}
    for key, stats in d.items():
        out[key] = {}
        for stat_name, tensor_val in stats.items():
            out[key][stat_name] = [float(x) for x in tensor_val.detach().cpu().tolist()]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stats for LeRobotDataset: state and actions.")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to write stats JSON. Defaults to <data_root_dir>/meta/stats.json",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="For debugging: limit number of batches processed (0 means all)",
    )
    args = parser.parse_args()

    stats = compute_stats(
        data_root_dir=args.data_root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
    )

    stats_json = _tensor_to_list(stats)

    output_path = (
        args.output_path
        if args.output_path is not None
        else os.path.join(args.data_root_dir, "meta", "stats.json")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=4)

    print(f"Saved stats to: {output_path}")


if __name__ == "__main__":
    main()


