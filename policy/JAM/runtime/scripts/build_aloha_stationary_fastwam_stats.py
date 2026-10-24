#!/usr/bin/env python3
"""Build train-split FastWAM statistics for a vector-action ALOHA dataset.

The script reads only action and state columns from LeRobot Parquet episodes.
It reproduces FastWAM's 32-step replicated action-window statistics without
decoding embedded images.  The optional half swap converts a right-first public
release into the left-first convention used by the policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


LEFT_FIRST_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
RIGHT_FIRST_NAMES = LEFT_FIRST_NAMES[7:] + LEFT_FIRST_NAMES[:7]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-proportion", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-window", type=int, default=32)
    parser.add_argument("--global-batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--swap-bimanual-halves", action="store_true")
    parser.add_argument("--expected-task", default=None)
    parser.add_argument("--source-revision", default=None)
    return parser.parse_args()


def _flatten_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for nested in value.values():
            result.extend(_flatten_names(nested))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for nested in value:
            result.extend(_flatten_names(nested))
        return result
    return [str(value)]


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    relative = str(info["data_path"]).format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return root / relative


def _fixed_list_tensor(table, name: str, expected_dim: int) -> torch.Tensor:
    column = table[name].combine_chunks()
    values = column.values.to_numpy(zero_copy_only=False)
    array = np.asarray(values, dtype=np.float32).reshape(len(column), expected_dim)
    return torch.from_numpy(array.copy())


def _swap_halves(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 14:
        raise ValueError(f"Expected a 14-D bimanual vector, got {tuple(value.shape)}")
    return torch.cat([value[..., 7:], value[..., :7]], dim=-1)


def _sliding_window_with_replication(value: torch.Tensor, window: int) -> torch.Tensor:
    indices = torch.arange(value.shape[0]).unsqueeze(1) + torch.arange(window).unsqueeze(0)
    indices.clamp_(max=value.shape[0] - 1)
    return value[indices]


def _episode_stats(value: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "min": value.amin(0),
        "max": value.amax(0),
        "mean": value.mean(0),
        # Match torch.Tensor.var(dim=0), used by BaseLerobotDataset.
        "var": value.var(0),
        "q01": torch.quantile(value, 0.01, dim=0),
        "q99": torch.quantile(value, 0.99, dim=0),
    }


def _combine(per_episode: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    mins = torch.stack([item["min"] for item in per_episode])
    maxs = torch.stack([item["max"] for item in per_episode])
    q01s = torch.stack([item["q01"] for item in per_episode])
    q99s = torch.stack([item["q99"] for item in per_episode])
    means = torch.stack([item["mean"] for item in per_episode])
    variances = torch.stack([item["var"] for item in per_episode])

    stepwise_mean = means.mean(0)
    stepwise_std = (variances + (means - stepwise_mean) ** 2).mean(0).sqrt()
    global_mean = means.mean((0, 1))
    global_std = (variances + (means - global_mean) ** 2).mean((0, 1)).sqrt()
    stepwise_min = mins.amin(0)
    stepwise_max = maxs.amax(0)
    stepwise_q01 = q01s.amin(0)
    stepwise_q99 = q99s.amax(0)
    return {
        "stepwise_min": stepwise_min,
        "stepwise_max": stepwise_max,
        "global_min": stepwise_min.amin(0),
        "global_max": stepwise_max.amax(0),
        "stepwise_q01": stepwise_q01,
        "stepwise_q99": stepwise_q99,
        "global_q01": stepwise_q01.amin(0),
        "global_q99": stepwise_q99.amax(0),
        "stepwise_mean": stepwise_mean,
        "stepwise_std": stepwise_std,
        "global_mean": global_mean,
        "global_std": global_std,
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(nested) for nested in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    args = _parse_args()
    if not 0.0 <= args.val_proportion < 1.0:
        raise ValueError("--val-proportion must be in [0, 1)")
    if min(args.action_window, args.global_batch, args.epochs) <= 0:
        raise ValueError("window, batch, and epochs must be positive")

    root = args.dataset_root.resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = [str(item.get("task", "")) for item in _read_jsonlines(root / "meta" / "tasks.jsonl")]
    if args.expected_task and args.expected_task not in tasks:
        raise RuntimeError(f"Expected task {args.expected_task!r}, found {tasks}")

    features = info["features"]
    action_dim = math.prod(features["action"]["shape"])
    state_dim = math.prod(features["observation.state"]["shape"])
    if action_dim != 14 or state_dim != 14:
        raise RuntimeError(f"Expected action/state 14-D, got {action_dim}/{state_dim}")
    action_names = _flatten_names(features["action"].get("names"))
    state_names = _flatten_names(features["observation.state"].get("names"))
    expected_source_names = RIGHT_FIRST_NAMES if args.swap_bimanual_halves else LEFT_FIRST_NAMES
    if action_names != expected_source_names or state_names != expected_source_names:
        raise RuntimeError(
            "Source action/state names do not match the declared ordering: "
            f"action={action_names}, state={state_names}"
        )

    total_episodes = int(info["total_episodes"])
    indices = list(range(total_episodes))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    split = int(total_episodes * (1.0 - args.val_proportion))
    train_indices = sorted(indices[:split])
    val_indices = sorted(indices[split:])

    action_stats: list[dict[str, torch.Tensor]] = []
    state_stats: list[dict[str, torch.Tensor]] = []
    episode_lengths: dict[int, int] = {}
    for episode_index in train_indices:
        path = _episode_path(root, info, episode_index)
        if not path.exists():
            raise FileNotFoundError(path)
        table = pq.read_table(path, columns=["action", "observation.state"])
        action = _fixed_list_tensor(table, "action", action_dim)
        state = _fixed_list_tensor(table, "observation.state", state_dim)
        if action.shape[0] != state.shape[0]:
            raise RuntimeError(f"{path}: action/state length mismatch")
        if not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(state).all()):
            raise RuntimeError(f"{path}: action/state contains non-finite values")
        if args.swap_bimanual_halves:
            action = _swap_halves(action)
            state = _swap_halves(state)
        episode_lengths[episode_index] = int(action.shape[0])
        action_stats.append(
            _episode_stats(_sliding_window_with_replication(action, args.action_window))
        )
        state_stats.append(_episode_stats(state.unsqueeze(1)))

    train_frames = sum(episode_lengths.values())
    payload = {
        "state": {"default": _combine(state_stats)},
        "action": {"default": _combine(action_stats)},
        "num_episodes": len(train_indices),
        "num_transition": train_frames,
    }
    _atomic_json(args.output, payload)

    steps_per_epoch = math.ceil(train_frames / args.global_batch)
    audit = {
        "dataset_root": str(root),
        "info_sha256": hashlib.sha256(
            (root / "meta" / "info.json").read_bytes()
        ).hexdigest(),
        "source_revision": args.source_revision,
        "source_codebase_version": info.get("codebase_version"),
        "tasks": tasks,
        "fps": info.get("fps"),
        "source_order": "right_then_left" if args.swap_bimanual_halves else "left_then_right",
        "stats_order": "left_then_right",
        "seed": args.seed,
        "val_proportion": args.val_proportion,
        "train_episode_indices": train_indices,
        "val_episode_indices": val_indices,
        "episode_lengths": episode_lengths,
        "train_frames": train_frames,
        "global_batch": args.global_batch,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": steps_per_epoch * args.epochs,
        "action_window": args.action_window,
    }
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    _atomic_json(audit_path, audit)
    print(json.dumps(audit, indent=2))
    print(f"Stats written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
