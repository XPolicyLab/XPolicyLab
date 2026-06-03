#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full-precision delta stats (mean/std/min/max/q01/q99) for multiple datasets under a root_dir.
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pytorch3d.transforms as pt
except Exception as e:
    raise ImportError(
        "pytorch3d is required. Please install pytorch3d and retry.\n"
        f"Original error: {e}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute full-precision delta stats for multiple datasets."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing multiple dataset subdirectories.",
    )
    parser.add_argument(
        "--input_euler_convention",
        type=str,
        default="XYZ",
        help='Euler convention for input rotations, e.g. "XYZ".',
    )
    parser.add_argument(
        "--input_quaternion_order",
        type=str,
        default="xyzw",
        choices=["xyzw", "wxyz"],
        help='Quaternion order of input data, e.g. "xyzw" or "wxyz".',
    )
    parser.add_argument(
        "--stats_file_policy",
        type=str,
        default="overwrite",
        choices=["skip", "overwrite"],
        help='How to handle existing stats files: "skip" or "overwrite".',
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--available_devices",
        type=str,
        default="cuda:0",
        help='Comma-separated device list, e.g. "cuda:0,cuda:1" or "cpu".',
    )
    parser.add_argument(
        "--show_inner_pbar",
        action="store_true",
        help="Show per-dataset inner parquet progress bar.",
    )
    return parser.parse_args()


def _sanitize_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _sanitize_np(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def _wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


def _ensure_2d_torch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    assert x.ndim == 2
    return x


def _as_quaternion_wxyz(q: torch.Tensor, quaternion_order: str) -> torch.Tensor:
    assert q.ndim == 2 and q.shape[1] == 4
    if quaternion_order == "wxyz":
        return q
    if quaternion_order == "xyzw":
        return q[:, [3, 0, 1, 2]]
    raise ValueError(f"Unknown quaternion_order: {quaternion_order}")


def _maybe_canonicalize_quaternion_wxyz(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    assert quaternion_wxyz.ndim == 2 and quaternion_wxyz.shape[1] == 4
    sign = torch.where(quaternion_wxyz[:, :1] < 0, -1.0, 1.0)
    return quaternion_wxyz * sign


def _normalize_rotation_type(rotation_type: str) -> str:
    return (rotation_type or "").lower().strip()


def rotation_input_to_matrix(
    raw_rotation: torch.Tensor,
    rotation_type: str,
    *,
    input_euler_convention: str,
    input_quaternion_order: str,
) -> torch.Tensor:
    normalized_rotation_type = _normalize_rotation_type(rotation_type)
    raw_rotation = _ensure_2d_torch(raw_rotation).to(torch.float64)
    raw_rotation = _sanitize_torch(raw_rotation)

    if normalized_rotation_type in ("euler_angles_rpy", "rpy", "euler_rpy", "euler"):
        if raw_rotation.shape[1] != 3:
            raise ValueError(
                f"rotation_type={rotation_type} expects 3 dims, got {raw_rotation.shape[1]}"
            )
        rotation_matrix = pt.euler_angles_to_matrix(raw_rotation, input_euler_convention)
        return _sanitize_torch(rotation_matrix)

    if "quat" in normalized_rotation_type or "quaternion" in normalized_rotation_type:
        if raw_rotation.shape[1] != 4:
            raise ValueError(
                f"rotation_type={rotation_type} expects 4 dims, got {raw_rotation.shape[1]}"
            )

        if "wxyz" in normalized_rotation_type:
            quaternion_order = "wxyz"
        elif "xyzw" in normalized_rotation_type:
            quaternion_order = "xyzw"
        else:
            quaternion_order = input_quaternion_order

        quaternion_wxyz = _as_quaternion_wxyz(raw_rotation, quaternion_order)
        quaternion_wxyz = _sanitize_torch(quaternion_wxyz)

        eps = 1e-12
        quaternion_norm = torch.linalg.norm(quaternion_wxyz, dim=-1, keepdim=True)
        invalid_mask = quaternion_norm < eps
        quaternion_wxyz = quaternion_wxyz / quaternion_norm.clamp_min(eps)
        if invalid_mask.any():
            quaternion_wxyz[invalid_mask.squeeze(-1)] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                dtype=quaternion_wxyz.dtype,
                device=quaternion_wxyz.device,
            )

        rotation_matrix = pt.quaternion_to_matrix(quaternion_wxyz)
        return _sanitize_torch(rotation_matrix)

    raise ValueError(
        f"Unsupported input rotation_type={rotation_type}. "
        "Expected only euler_angles_rpy or quaternion."
    )


def rotation_input_to_euler(
    raw_rotation: torch.Tensor,
    rotation_type: str,
    *,
    input_euler_convention: str,
    input_quaternion_order: str,
) -> torch.Tensor:
    normalized_rotation_type = _normalize_rotation_type(rotation_type)
    raw_rotation = _ensure_2d_torch(raw_rotation).to(torch.float64)
    raw_rotation = _sanitize_torch(raw_rotation)

    if normalized_rotation_type in ("euler_angles_rpy", "rpy", "euler_rpy", "euler"):
        if raw_rotation.shape[1] != 3:
            raise ValueError(
                f"rotation_type={rotation_type} expects 3 dims, got {raw_rotation.shape[1]}"
            )
        return _sanitize_torch(_wrap_to_pi_torch(raw_rotation))

    rotation_matrix = rotation_input_to_matrix(
        raw_rotation,
        rotation_type,
        input_euler_convention=input_euler_convention,
        input_quaternion_order=input_quaternion_order,
    )
    euler_angles = pt.matrix_to_euler_angles(rotation_matrix, input_euler_convention)
    euler_angles = _wrap_to_pi_torch(euler_angles)
    return _sanitize_torch(euler_angles)


def rotation_matrix_to_output(
    rotation_matrix: torch.Tensor,
    *,
    rotation_output_type: str,
    input_euler_convention: str,
    output_quaternion_wxyz: bool,
    canonicalize_output_quaternion: bool,
) -> torch.Tensor:
    rotation_output_type = rotation_output_type.lower().strip()
    rotation_matrix = _sanitize_torch(rotation_matrix)

    if rotation_output_type == "matrix":
        output_rotation = rotation_matrix.reshape(-1, 9)

    elif rotation_output_type == "rotation_6d":
        output_rotation = pt.matrix_to_rotation_6d(rotation_matrix)

    elif rotation_output_type == "quaternion":
        quaternion_wxyz = pt.matrix_to_quaternion(rotation_matrix)
        if canonicalize_output_quaternion:
            quaternion_wxyz = _maybe_canonicalize_quaternion_wxyz(quaternion_wxyz)
        output_rotation = (
            quaternion_wxyz
            if output_quaternion_wxyz
            else quaternion_wxyz[:, [1, 2, 3, 0]]
        )

    elif rotation_output_type == "euler_angles":
        euler_angles = pt.matrix_to_euler_angles(rotation_matrix, input_euler_convention)
        output_rotation = _wrap_to_pi_torch(euler_angles)

    elif rotation_output_type == "axis_angle":
        output_rotation = pt.matrix_to_axis_angle(rotation_matrix)

    else:
        raise ValueError(f"Unsupported rotation_output_type: {rotation_output_type}")

    return _sanitize_torch(output_rotation)


def load_json(json_path: Path) -> Any:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_row_value(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype != object and value.ndim == 1:
        return value
    if isinstance(value, (list, tuple)):
        flattened_parts = [np.asarray(part).reshape(-1) for part in value]
        return np.concatenate(flattened_parts, axis=0)
    if isinstance(value, np.ndarray) and value.dtype == object:
        flattened_parts = [np.asarray(part).reshape(-1) for part in list(value)]
        return np.concatenate(flattened_parts, axis=0)
    array_value = np.asarray(value)
    return array_value.reshape(-1)


def extract_key_matrix(
    dataframe: pd.DataFrame,
    original_key: str,
    start: int,
    end: int,
) -> np.ndarray:
    column = dataframe[original_key]
    extracted_rows = []
    for value in column:
        flattened_value = flatten_row_value(value)
        extracted_rows.append(flattened_value[start:end])
    matrix = np.stack(extracted_rows, axis=0)
    return _sanitize_np(matrix).astype(np.float64, copy=False)


def parse_key_metadata(modality_metadata: Dict[str, Any], full_key: str) -> Dict[str, Any]:
    group_name, key_name = full_key.split(".", 1)
    key_info = modality_metadata[group_name][key_name]
    return {
        "original_key": key_info["original_key"],
        "start": int(key_info["start"]),
        "end": int(key_info["end"]),
        "rotation_type": key_info.get("rotation_type", None),
    }

def has_modality_key(modality_metadata: Dict[str, Any], full_key: str) -> bool:
    try:
        group_name, key_name = full_key.split(".", 1)
    except ValueError:
        return False
    return group_name in modality_metadata and key_name in modality_metadata[group_name]


def filter_existing_modality_keys(
    modality_metadata: Dict[str, Any],
    candidate_keys: List[str],
) -> List[str]:
    return [full_key for full_key in candidate_keys if has_modality_key(modality_metadata, full_key)]

class ExactStats:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.blocks: List[np.ndarray] = []
        self.count = 0
        self.min_values = np.full(self.dim, np.inf, dtype=np.float64)
        self.max_values = np.full(self.dim, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        assert values.ndim == 2 and values.shape[1] == self.dim

        values_float64 = _sanitize_np(values)
        self.blocks.append(values_float64)
        self.count += values_float64.shape[0]
        self.min_values = np.minimum(self.min_values, np.min(values_float64, axis=0))
        self.max_values = np.maximum(self.max_values, np.max(values_float64, axis=0))

    def finalize(self) -> Dict[str, List[float]]:
        if self.count == 0:
            zeros = np.zeros(self.dim, dtype=np.float64)
            return {
                "mean": zeros.tolist(),
                "std": zeros.tolist(),
                "min": zeros.tolist(),
                "max": zeros.tolist(),
                "q01": zeros.tolist(),
                "q99": zeros.tolist(),
            }

        all_values = _sanitize_np(np.concatenate(self.blocks, axis=0))
        mean = all_values.mean(axis=0)
        std = all_values.std(axis=0, ddof=1) if all_values.shape[0] > 1 else np.zeros_like(mean)
        q01 = np.quantile(all_values, 0.01, axis=0)
        q99 = np.quantile(all_values, 0.99, axis=0)

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self.min_values.tolist(),
            "max": self.max_values.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


def make_position_stat_key(modality_key: str) -> str:
    return f"{modality_key}.delta_sub"


def make_rotation_relative_stat_key(modality_key: str, rotation_output_type: str) -> str:
    rotation_output_type = rotation_output_type.lower().strip()
    output_name_map = {
        "euler_angles": "euler_delta_rel",
        "rotation_6d": "rotation_6d_delta_rel",
        "quaternion": "quaternion_delta_rel",
        "axis_angle": "axis_angle_delta_rel",
        "matrix": "matrix_delta_rel",
    }
    if rotation_output_type not in output_name_map:
        raise ValueError(
            f"Unsupported rotation_output_type for delta_rel stats: {rotation_output_type}"
        )
    return f"{modality_key}.{output_name_map[rotation_output_type]}"


def make_rotation_subtractive_stat_key(modality_key: str, rotation_output_type: str) -> str:
    rotation_output_type = rotation_output_type.lower().strip()
    if rotation_output_type != "euler_angles":
        raise ValueError(
            f"Only euler_angles supports delta_sub, got: {rotation_output_type}"
        )
    return f"{modality_key}.euler_delta_sub"


def list_dataset_dirs(root_dir: Path, modality_json_relpath: str) -> List[Path]:
    dataset_dirs = []
    for dataset_dir in sorted([path for path in root_dir.iterdir() if path.is_dir()]):
        if (dataset_dir / "data").exists() and (dataset_dir / modality_json_relpath).exists():
            dataset_dirs.append(dataset_dir)
    return dataset_dirs


def compute_dataset_stats(
    dataset_dir: Path,
    *,
    modality_json_relpath: str,
    modality_keys_config: Dict[str, List[str]],
    time_col: str,
    episode_col: str,
    input_euler_convention: str,
    input_quaternion_order: str,
    device: str,
    rotation_output_types: List[str],
    output_quaternion_wxyz: bool,
    canonicalize_output_quaternion: bool,
    show_inner_pbar: bool = False,
) -> Dict[str, Any]:
    modality_metadata = load_json(dataset_dir / modality_json_relpath)
    candidate_position_keys = modality_keys_config.get("position", [])
    candidate_rotation_keys = modality_keys_config.get("rotation", [])

    position_keys = filter_existing_modality_keys(modality_metadata, candidate_position_keys)
    rotation_keys = filter_existing_modality_keys(modality_metadata, candidate_rotation_keys)

    if not position_keys and not rotation_keys:
        raise ValueError(
            f"No valid position/rotation keys found in {dataset_dir}. "
            f"Candidates: position={candidate_position_keys}, rotation={candidate_rotation_keys}"
        )

    key_metadata: Dict[str, Dict[str, Any]] = {}
    for modality_key in position_keys + rotation_keys:
        key_metadata[modality_key] = parse_key_metadata(modality_metadata, modality_key)

    rotation_output_dims = {
        "axis_angle": 3,
        "euler_angles": 3,
        "quaternion": 4,
        "rotation_6d": 6,
        "matrix": 9,
    }

    normalized_rotation_output_types = [
        rotation_output_type.lower().strip()
        for rotation_output_type in rotation_output_types
    ]
    for rotation_output_type in normalized_rotation_output_types:
        if rotation_output_type not in rotation_output_dims:
            raise ValueError(
                f"Unsupported rotation_output_type in rotation_output_types: {rotation_output_type}"
            )

    need_euler_delta_sub = "euler_angles" in normalized_rotation_output_types
    need_any_relative_rotation = len(normalized_rotation_output_types) > 0

    stats_bank: Dict[str, ExactStats] = {}

    for position_key in position_keys:
        stats_bank[make_position_stat_key(position_key)] = ExactStats(dim=3)

    for rotation_key in rotation_keys:
        for rotation_output_type in normalized_rotation_output_types:
            stats_bank[
                make_rotation_relative_stat_key(rotation_key, rotation_output_type)
            ] = ExactStats(dim=rotation_output_dims[rotation_output_type])
        if need_euler_delta_sub:
            stats_bank[
                make_rotation_subtractive_stat_key(rotation_key, "euler_angles")
            ] = ExactStats(dim=3)

    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.rglob("episode_*.parquet"))
    if not parquet_files:
        parquet_files = sorted(data_dir.rglob("*.parquet"))

    parquet_progress_bar = tqdm(
        parquet_files,
        desc=f"{dataset_dir.name}: parquets",
        unit="file",
        leave=False,
        disable=not show_inner_pbar,
    )

    for parquet_path in parquet_progress_bar:
        dataframe = pd.read_parquet(parquet_path)

        if episode_col not in dataframe.columns:
            dataframe["_tmp_episode_col"] = 0
            effective_episode_col = "_tmp_episode_col"
        else:
            effective_episode_col = episode_col

        if time_col not in dataframe.columns:
            raise KeyError(f"Missing time_col='{time_col}' in {parquet_path}")

        for _, episode_dataframe in dataframe.groupby(effective_episode_col, sort=False):
            episode_dataframe = episode_dataframe.sort_values(time_col, kind="mergesort")
            num_steps = len(episode_dataframe)
            if num_steps <= 1:
                continue

            for position_key in position_keys:
                metadata = key_metadata[position_key]
                raw_position = extract_key_matrix(
                    episode_dataframe,
                    metadata["original_key"],
                    metadata["start"],
                    metadata["end"],
                )
                if raw_position.shape[1] != 3:
                    raise ValueError(
                        f"{position_key}: expected 3D position, got {raw_position.shape[1]}"
                    )
                position_delta = raw_position[1:] - raw_position[:-1]
                stats_bank[make_position_stat_key(position_key)].update(position_delta)

            for rotation_key in rotation_keys:
                metadata = key_metadata[rotation_key]
                raw_rotation_type = metadata.get("rotation_type", None)
                normalized_rotation_type = _normalize_rotation_type(raw_rotation_type)

                if normalized_rotation_type not in (
                    "euler_angles_rpy",
                    "rpy",
                    "euler_rpy",
                    "euler",
                    "quaternion",
                    "quat",
                    "quaternion_xyzw",
                    "quaternion_wxyz",
                ):
                    raise ValueError(
                        f"{rotation_key}: unexpected rotation_type={raw_rotation_type}. "
                        "Only euler_angles_rpy or quaternion is allowed."
                    )

                raw_rotation = extract_key_matrix(
                    episode_dataframe,
                    metadata["original_key"],
                    metadata["start"],
                    metadata["end"],
                )
                raw_rotation_tensor = torch.from_numpy(raw_rotation).to(device=device)

                euler_angles = None
                if need_euler_delta_sub:
                    euler_angles = rotation_input_to_euler(
                        raw_rotation_tensor,
                        normalized_rotation_type,
                        input_euler_convention=input_euler_convention,
                        input_quaternion_order=input_quaternion_order,
                    )

                relative_rotation_matrix = None
                if need_any_relative_rotation:
                    rotation_matrix = rotation_input_to_matrix(
                        raw_rotation_tensor,
                        normalized_rotation_type,
                        input_euler_convention=input_euler_convention,
                        input_quaternion_order=input_quaternion_order,
                    )
                    relative_rotation_matrix = torch.bmm(
                        rotation_matrix[1:],
                        rotation_matrix[:-1].transpose(1, 2),
                    )
                    relative_rotation_matrix = _sanitize_torch(relative_rotation_matrix)

                if need_euler_delta_sub:
                    assert euler_angles is not None
                    euler_delta = _wrap_to_pi_torch(euler_angles[1:] - euler_angles[:-1])
                    euler_delta = _sanitize_torch(euler_delta)
                    stats_bank[
                        make_rotation_subtractive_stat_key(rotation_key, "euler_angles")
                    ].update(euler_delta.detach().cpu().numpy())

                if relative_rotation_matrix is not None:
                    for rotation_output_type in normalized_rotation_output_types:
                        output_rotation = rotation_matrix_to_output(
                            relative_rotation_matrix,
                            rotation_output_type=rotation_output_type,
                            input_euler_convention=input_euler_convention,
                            output_quaternion_wxyz=output_quaternion_wxyz,
                            canonicalize_output_quaternion=canonicalize_output_quaternion,
                        )
                        stats_bank[
                            make_rotation_relative_stat_key(rotation_key, rotation_output_type)
                        ].update(output_rotation.detach().cpu().numpy())

    output_stats: Dict[str, Any] = {}
    for stat_key, stat_calculator in stats_bank.items():
        output_stats[stat_key] = stat_calculator.finalize()
    return output_stats


def _assign_device_for_idx(available_devices: List[str], idx: int) -> str:
    if not available_devices:
        return "cpu"
    return available_devices[idx % len(available_devices)]


def _process_one_dataset(
    dataset_dir: Path,
    *,
    stats_filename: str,
    stats_file_policy: str,
    modality_json_relpath: str,
    modality_keys_config: Dict[str, List[str]],
    time_col: str,
    episode_col: str,
    input_euler_convention: str,
    input_quaternion_order: str,
    device: str,
    rotation_output_types: List[str],
    output_quaternion_wxyz: bool,
    canonicalize_output_quaternion: bool,
    show_inner_pbar: bool,
) -> Tuple[str, bool, Optional[str]]:
    try:
        stats_output_path = dataset_dir / "meta" / stats_filename

        if stats_output_path.exists() and stats_file_policy == "skip":
            return (dataset_dir.name, True, None)

        if stats_file_policy not in ("skip", "overwrite"):
            raise ValueError(
                f"stats_file_policy must be 'skip' or 'overwrite', got {stats_file_policy}"
            )

        stats = compute_dataset_stats(
            dataset_dir,
            modality_json_relpath=modality_json_relpath,
            modality_keys_config=modality_keys_config,
            time_col=time_col,
            episode_col=episode_col,
            input_euler_convention=input_euler_convention,
            input_quaternion_order=input_quaternion_order,
            device=device,
            rotation_output_types=rotation_output_types,
            output_quaternion_wxyz=output_quaternion_wxyz,
            canonicalize_output_quaternion=canonicalize_output_quaternion,
            show_inner_pbar=show_inner_pbar,
        )

        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return (dataset_dir.name, True, None)
    except Exception as e:
        return (dataset_dir.name, False, f"{type(e).__name__}: {e}")


def main():
    args = parse_args()

    root_dir = Path(args.root_dir)
    input_euler_convention = args.input_euler_convention
    input_quaternion_order = args.input_quaternion_order
    stats_file_policy = args.stats_file_policy
    max_workers = args.max_workers
    available_devices = [device.strip() for device in args.available_devices.split(",") if device.strip()]
    show_inner_pbar = args.show_inner_pbar

    modality_keys_config = {
        "position": [
            "action.left_arm_eef_position",
            "action.right_arm_eef_position",
            "action.single_arm_eef_position",
        ],
        "rotation": [
            "action.left_arm_eef_orientation",
            "action.right_arm_eef_orientation",
            "action.single_arm_eef_orientation",
        ],
    }

    modality_json_relpath = "meta/modality.json"
    time_col = "frame_index"
    episode_col = "episode_index"

    stats_filename = "stats_delta_state.json"

    rotation_output_types = [
        "axis_angle",
        "euler_angles",
        "quaternion",
        "rotation_6d",
        "matrix",
    ]

    output_quaternion_wxyz = True
    canonicalize_output_quaternion = True

    dataset_dirs = list_dataset_dirs(root_dir, modality_json_relpath)
    if not dataset_dirs:
        print(f"No datasets found under {root_dir} with {modality_json_relpath} and data/.")
        return

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    futures = []
    failed_datasets: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for dataset_idx, dataset_dir in enumerate(dataset_dirs):
            assigned_device = _assign_device_for_idx(available_devices, dataset_idx)
            futures.append(
                executor.submit(
                    _process_one_dataset,
                    dataset_dir,
                    stats_filename=stats_filename,
                    stats_file_policy=stats_file_policy,
                    modality_json_relpath=modality_json_relpath,
                    modality_keys_config=modality_keys_config,
                    time_col=time_col,
                    episode_col=episode_col,
                    input_euler_convention=input_euler_convention,
                    input_quaternion_order=input_quaternion_order,
                    device=assigned_device,
                    rotation_output_types=rotation_output_types,
                    output_quaternion_wxyz=output_quaternion_wxyz,
                    canonicalize_output_quaternion=canonicalize_output_quaternion,
                    show_inner_pbar=show_inner_pbar,
                )
            )

        progress_bar = tqdm(total=len(futures), desc="Datasets (parallel)", unit="ds")
        for future in as_completed(futures):
            dataset_name, ok, error_message = future.result()
            if not ok and error_message is not None:
                failed_datasets.append((dataset_name, error_message))
                progress_bar.set_postfix_str(f"{dataset_name}: FAILED")
            else:
                progress_bar.set_postfix_str(f"{dataset_name}: done/skip")
            progress_bar.update(1)
        progress_bar.close()

    if failed_datasets:
        print("\nSome datasets failed:")
        for dataset_name, error_message in failed_datasets:
            print(f"  - {dataset_name}: {error_message}")
    else:
        print("All datasets done (including skipped).")

    print("Done.")


if __name__ == "__main__":
    main()