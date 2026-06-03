from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _load_subset_mapping_json(json_path: str) -> Dict[str, List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if not isinstance(mapping, dict):
        raise TypeError(f"Mapping json must be dict[str, list[str]], got {type(mapping).__name__}")

    normalized: Dict[str, List[str]] = {}
    for robot_type, folders in mapping.items():
        if folders is None:
            normalized[robot_type] = []
            continue
        if not isinstance(folders, list):
            raise TypeError(
                f"Mapping json must be dict[str, list[str]]; "
                f"got value type {type(folders).__name__} for key {robot_type!r}"
            )
        for x in folders:
            if not isinstance(x, str) or not x:
                raise TypeError(
                    f"Each dataset folder name must be a non-empty str, got {x!r} under {robot_type!r}"
                )
        normalized[robot_type] = folders
    return normalized


def _normalize_to_list(x: Union[str, List[str], Tuple[str, ...]], name: str) -> List[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return list(x)
    raise TypeError(f"{name} must be str or list/tuple[str], got {type(x).__name__}")


def _normalize_dataset_weights(dataset_weights: Optional[Union[List[float], Tuple[float, ...]]], n: int) -> List[float]:
    if dataset_weights is None:
        return [1.0] * n

    weights = list(dataset_weights)
    if len(weights) != n:
        raise ValueError(
            f"dataset_weights must have the same length as abs_paths/rel_paths/mapping_jsons. "
            f"Got len(dataset_weights)={len(weights)}, expected {n}"
        )

    for i, w in enumerate(weights):
        if not isinstance(w, (int, float)):
            raise TypeError(f"dataset_weights[{i}] must be int/float, got {type(w).__name__}")
        if w < 0:
            raise ValueError(f"dataset_weights[{i}] must be >= 0, got {w}")

    return [float(w) for w in weights]


def generate_dataset_mixture(
    mixture_name: str,
    abs_paths: Union[str, List[str]],
    rel_paths: Union[str, List[str]],
    dataset_subset_mapping_jsons: Union[str, List[str]],
    intra_dataset_weight_mode: Optional[str] = None,  # 'episode' / 'frame' / None
    dataset_weights: Optional[Union[List[float], Tuple[float, ...]]] = None,
    lerobot_version: Optional[str] = None,
):
    """
    Generate dataset mixture from multiple roots using per-root subset-mapping json files.

    Args:
        mixture_name:
            Output dict key.
        abs_paths:
            One or multiple absolute dataset roots.
        rel_paths:
            One or multiple relative dataset roots.
        dataset_subset_mapping_jsons:
            One json mapping path per root. Each json is:
                {
                    "robot_type_a": ["dataset_1", "dataset_2"],
                    "robot_type_b": ["dataset_3"]
                }
        intra_dataset_weight_mode:
            Per-root local weighting method for datasets inside each dataset:
              - "episode": proportional to total_episodes
              - "frame": proportional to total_frames
              - None: uniform inside each root
        dataset_weights:
            Global weights for each dataset.
            If None, defaults to equal weights [1, 1, ..., 1].
        lerobot_version:
            Optional extra config.
    """

    abs_paths = _normalize_to_list(abs_paths, "abs_paths")
    rel_paths = _normalize_to_list(rel_paths, "rel_paths")
    dataset_subset_mapping_jsons = _normalize_to_list(
        dataset_subset_mapping_jsons, "dataset_subset_mapping_jsons"
    )

    n_roots = len(abs_paths)
    if len(rel_paths) != n_roots or len(dataset_subset_mapping_jsons) != n_roots:
        raise ValueError(
            "abs_paths, rel_paths, and dataset_subset_mapping_jsons must have the same length. "
            f"Got len(abs_paths)={len(abs_paths)}, "
            f"len(rel_paths)={len(rel_paths)}, "
            f"len(dataset_subset_mapping_jsons)={len(dataset_subset_mapping_jsons)}"
        )

    if intra_dataset_weight_mode not in ("episode", "frame", None):
        raise ValueError(
            f"intra_dataset_weight_mode must be one of ('episode', 'frame', None), got {intra_dataset_weight_mode!r}"
        )

    dataset_weights = _normalize_dataset_weights(dataset_weights, n_roots)

    per_root_entries: List[List[Dict[str, Any]]] = []
    declared_missing_by_root: List[Tuple[str, List[str]]] = []

    for root_idx, (abs_root, rel_root, mapping_json) in enumerate(
        zip(abs_paths, rel_paths, dataset_subset_mapping_jsons)
    ):
        if not os.path.exists(abs_root):
            raise FileNotFoundError(f"Root path does not exist: {abs_root}")

        if not os.path.isfile(mapping_json):
            raise FileNotFoundError(f"Mapping json does not exist: {mapping_json}")

        mapping = _load_subset_mapping_json(mapping_json)

        try:
            all_candidates = sorted(
                d for d in os.listdir(abs_root)
                if os.path.isdir(os.path.join(abs_root, d))
            )
        except Exception as e:
            raise RuntimeError(f"Failed to list root directory {abs_root}: {type(e).__name__}: {e}") from e

        folder_to_robot_type: Dict[str, str] = {}
        declared_folders = set()

        for robot_type, folders in mapping.items():
            for folder_name in folders:
                if folder_name in folder_to_robot_type and folder_to_robot_type[folder_name] != robot_type:
                    raise ValueError(
                        f"[{abs_root}] Dataset folder {folder_name!r} is assigned to multiple robot types: "
                        f"{folder_to_robot_type[folder_name]!r} and {robot_type!r}"
                    )
                folder_to_robot_type[folder_name] = robot_type
                declared_folders.add(folder_name)

        on_disk_folders = set(all_candidates)

        missing_declared = sorted(x for x in declared_folders if x not in on_disk_folders)
        if missing_declared:
            declared_missing_by_root.append((abs_root, missing_declared))

        root_entries: List[Dict[str, Any]] = []

        for dataset_name in all_candidates:
            if dataset_name not in declared_folders:
                continue

            raw_count = 0
            if intra_dataset_weight_mode in ("episode", "frame"):
                info_path = os.path.join(abs_root, dataset_name, "meta", "info.json")
                count_key = "total_episodes" if intra_dataset_weight_mode == "episode" else "total_frames"
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        meta_info = json.load(f)
                    raw_count = meta_info.get(count_key, 0) or 0
                except (FileNotFoundError, json.JSONDecodeError):
                    raw_count = 0

            root_entries.append(
                {
                    "dataset_name": dataset_name,
                    "robot_type": folder_to_robot_type[dataset_name],
                    "abs_root": abs_root,
                    "rel_root": rel_root,
                    "raw_count": float(raw_count),
                    "root_index": root_idx,
                }
            )

        per_root_entries.append(root_entries)

    if declared_missing_by_root:
        lines = [
            "Found dataset folders declared in mapping json but missing on disk:"
        ]
        for abs_root, names in declared_missing_by_root:
            lines.append(f"\n[Root] {abs_root}")
            for x in names[:200]:
                lines.append(f"  - {x}")
            if len(names) > 200:
                lines.append(f"  ... (and {len(names) - 200} more)")

        raise ValueError("\n".join(lines))

    for root_entries in per_root_entries:
        if not root_entries:
            continue

        if intra_dataset_weight_mode in ("episode", "frame"):
            total_raw = sum(x["raw_count"] for x in root_entries)
            if total_raw > 0:
                for x in root_entries:
                    x["local_weight"] = x["raw_count"] / total_raw
            else:
                uniform = 1.0 / len(root_entries)
                for x in root_entries:
                    x["local_weight"] = uniform
        else:
            uniform = 1.0 / len(root_entries)
            for x in root_entries:
                x["local_weight"] = uniform

    merged_entries: List[Dict[str, Any]] = []
    for root_idx, root_entries in enumerate(per_root_entries):
        gw = dataset_weights[root_idx]
        for x in root_entries:
            x["pre_norm_final_weight"] = x["local_weight"] * gw
            merged_entries.append(x)

    total_final = sum(x["pre_norm_final_weight"] for x in merged_entries)
    if total_final <= 0:
        raise ValueError("All final weights are zero. Please check dataset_weights and dataset counts.")

    for x in merged_entries:
        x["final_weight"] = x["pre_norm_final_weight"] / total_final

    mixture_list: List[Tuple[str, float, str, Dict[str, Any]]] = []
    for item in merged_entries:
        extra_config: Dict[str, Any] = {}
        if lerobot_version:
            extra_config["lerobot_version"] = lerobot_version

        full_rel_path = os.path.join(item["rel_root"], item["dataset_name"]).replace("\\", "/")
        mixture_list.append(
            (
                full_rel_path,
                round(item["final_weight"], 8),
                item["robot_type"],
                extra_config,
            )
        )

    return {mixture_name: mixture_list}

def merge_mixtures_with_group_ratios(
    source_mixtures,
    mixture_ratios,
    new_mixture_name,
):
    merged_candidates = []

    for mixture_name, target_group_ratio in mixture_ratios.items():
        if mixture_name not in source_mixtures:
            print(f"Warning: {mixture_name} not in source_mixtures, skipped.")
            continue

        dataset_list = source_mixtures[mixture_name]
        if not dataset_list:
            continue

        group_total_amount = sum(item[1] for item in dataset_list)
        if group_total_amount <= 0:
            print(f"Warning: {mixture_name} group total amount <= 0, skipped.")
            continue

        for item in dataset_list:
            path = item[0]
            raw_amount = item[1]
            robot_type = item[2]
            extras = item[3:]

            temp_weight = float(target_group_ratio) * (float(raw_amount) / float(group_total_amount))
            merged_candidates.append((path, temp_weight, robot_type) + extras)

    if not merged_candidates:
        return {new_mixture_name: []}

    total_weight = sum(x[1] for x in merged_candidates)
    if total_weight <= 0:
        return {new_mixture_name: []}

    final_list = []
    for item in merged_candidates:
        path = item[0]
        temp_weight = item[1]
        rest = item[2:]

        final_weight = round(temp_weight / total_weight, 8)
        final_list.append((path, final_weight) + rest)

    return {new_mixture_name: final_list}