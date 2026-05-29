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


def _init_running_stats(dimension: int, device: torch.device) -> Dict[str, torch.Tensor]:
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
    device: str = "auto",
    quantile_device: str = "cpu",
    quantile_method: str = "hist",
    quantile_bins: int = 2048,
    quantile_dim_chunk: int = 512,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # Resolve main compute device
    if device == "auto":
        main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            main_device = torch.device(device)
        except Exception as exc:
            raise ValueError(f"Invalid device string: {device}") from exc
        if main_device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU for main device.")
            main_device = torch.device("cpu")

    # Resolve quantile device
    if quantile_device in ("same", "auto", None):
        q_device = main_device
    else:
        try:
            q_device = torch.device(quantile_device)
        except Exception as exc:
            raise ValueError(f"Invalid quantile_device string: {quantile_device}") from exc
        if q_device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU for quantile device.")
            q_device = torch.device("cpu")

    dataset = LeRobotDataset(data_root_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(main_device.type == "cuda"),
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
    state_stats = _init_running_stats(state_dim, main_device)
    actions_stats = _init_running_stats(actions_dim, main_device)

    for batch_idx, batch in enumerate(dataloader):
        print(f"batch_idx/all batches: {batch_idx}/{len(dataloader)}")
        state = batch[state_key]
        actions = batch[actions_key]

        # Ensure shape (B, D)
        state_flat = (
            state.view(state.shape[0], -1)
            .to(device=main_device, dtype=torch.float32, non_blocking=True)
        )
        actions_flat = (
            actions.view(actions.shape[0], -1)
            .to(device=main_device, dtype=torch.float32, non_blocking=True)
        )

        _update_running_stats(state_stats, state_flat)
        _update_running_stats(actions_stats, actions_flat)

        if max_batches > 0 and (batch_idx + 1) >= max_batches:
            break

    # Finalize mean/std/min/max
    state_mean, state_std, state_min, state_max = _finalize_running_stats(state_stats)
    actions_mean, actions_std, actions_min, actions_max = _finalize_running_stats(actions_stats)

    # Quantiles (0.01, 0.99) without storing all values: histogram approximation
    q01 = 0.01
    q99 = 0.99
    if quantile_method not in ("hist", "none"):
        raise ValueError("Unsupported quantile_method. Use 'hist' or 'none'.")

    if quantile_method == "none":
        state_q01 = state_min
        state_q99 = state_max
        actions_q01 = actions_min
        actions_q99 = actions_max
    else:
        # Build histograms in one pass with per-dimension chunking to bound memory
        eps = 1e-12

        # Pre-move min/max to quantile device
        state_min_q = state_min.to(q_device)
        state_max_q = state_max.to(q_device)
        state_range_q = torch.clamp(state_max_q - state_min_q, min=eps)

        actions_min_q = actions_min.to(q_device)
        actions_max_q = actions_max.to(q_device)
        actions_range_q = torch.clamp(actions_max_q - actions_min_q, min=eps)

        state_hist = torch.zeros((state_dim, quantile_bins), dtype=torch.long, device=q_device)
        actions_hist = torch.zeros((actions_dim, quantile_bins), dtype=torch.long, device=q_device)
        total_seen = 0

        for batch_idx, batch in enumerate(dataloader):
            state = batch[state_key]
            actions = batch[actions_key]

            state_flat = (
                state.view(state.shape[0], -1)
                .to(device=q_device, dtype=torch.float32, non_blocking=True)
            )
            actions_flat = (
                actions.view(actions.shape[0], -1)
                .to(device=q_device, dtype=torch.float32, non_blocking=True)
            )

            batch_size_curr = state_flat.shape[0]
            if batch_size_curr == 0:
                continue

            for start in range(0, state_dim, quantile_dim_chunk):
                end = min(start + quantile_dim_chunk, state_dim)
                s_chunk = state_flat[:, start:end]
                s_min = state_min_q[start:end]
                s_range = state_range_q[start:end]
                s_idx = ((s_chunk - s_min) / s_range * quantile_bins).floor().to(torch.long)
                s_idx.clamp_(0, quantile_bins - 1)
                offsets = torch.arange(end - start, device=q_device).view(1, -1) * quantile_bins
                flat_idx = (s_idx + offsets).reshape(-1)
                binc = torch.bincount(flat_idx, minlength=(end - start) * quantile_bins)
                state_hist[start:end] += binc.view(end - start, quantile_bins)

            for start in range(0, actions_dim, quantile_dim_chunk):
                end = min(start + quantile_dim_chunk, actions_dim)
                a_chunk = actions_flat[:, start:end]
                a_min = actions_min_q[start:end]
                a_range = actions_range_q[start:end]
                a_idx = ((a_chunk - a_min) / a_range * quantile_bins).floor().to(torch.long)
                a_idx.clamp_(0, quantile_bins - 1)
                offsets = torch.arange(end - start, device=q_device).view(1, -1) * quantile_bins
                flat_idx = (a_idx + offsets).reshape(-1)
                binc = torch.bincount(flat_idx, minlength=(end - start) * quantile_bins)
                actions_hist[start:end] += binc.view(end - start, quantile_bins)

            total_seen += batch_size_curr

            if max_batches > 0 and (batch_idx + 1) >= max_batches:
                break

        # Compute quantiles from histograms
        if total_seen == 0:
            raise ValueError("No samples found to compute histogram quantiles.")

        def quantiles_from_hist(hist: torch.Tensor, vmin: torch.Tensor, vrange: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            cdf = hist.cumsum(dim=1)
            total = cdf[:, -1]
            # targets are 0-indexed counts
            t_low = torch.clamp((total.to(torch.float64) * q01).ceil().to(torch.long) - 1, min=0)
            t_high = torch.clamp((total.to(torch.float64) * q99).ceil().to(torch.long) - 1, min=0)
            idx_low = torch.searchsorted(cdf, t_low.unsqueeze(1), right=False).squeeze(1)
            idx_high = torch.searchsorted(cdf, t_high.unsqueeze(1), right=False).squeeze(1)
            bin_width = vrange / quantile_bins
            q_low_v = vmin + idx_low.to(vmin.dtype) * bin_width
            q_high_v = vmin + idx_high.to(vmin.dtype) * bin_width
            zero_mask = (vrange <= eps)
            q_low_v = torch.where(zero_mask, vmin, q_low_v)
            q_high_v = torch.where(zero_mask, vmin + vrange, q_high_v)
            return q_low_v, q_high_v

        state_q01_q, state_q99_q = quantiles_from_hist(state_hist, state_min_q, state_range_q)
        actions_q01_q, actions_q99_q = quantiles_from_hist(actions_hist, actions_min_q, actions_range_q)

        state_q01 = state_q01_q.to(main_device)
        state_q99 = state_q99_q.to(main_device)
        actions_q01 = actions_q01_q.to(main_device)
        actions_q99 = actions_q99_q.to(main_device)

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
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device for stats accumulation: auto/cpu/cuda/cuda:0/...",
    )
    parser.add_argument(
        "--quantile_device",
        type=str,
        default="same",
        help="Device for quantile computation and concatenation: same/auto/cpu/cuda/...",
    )
    parser.add_argument(
        "--quantile_method",
        type=str,
        default="hist",
        help="Quantile computation method: hist (approx) or none (skip)",
    )
    parser.add_argument(
        "--quantile_bins",
        type=int,
        default=2048,
        help="Number of histogram bins per dimension for approx quantiles",
    )
    parser.add_argument(
        "--quantile_dim_chunk",
        type=int,
        default=512,
        help="Process this many dims per chunk when building histograms",
    )
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
        device=args.device,
        quantile_device=args.quantile_device,
        quantile_method=args.quantile_method,
        quantile_bins=args.quantile_bins,
        quantile_dim_chunk=args.quantile_dim_chunk,
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


