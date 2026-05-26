import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union


_FINAL_ROBOT_TYPE_MAP = {
    "google_robot": "oxe_auge_google_robot",
    "jaco": "oxe_auge_jaco",
    "kinova3": "oxe_auge_kinova3",
    "kuka_iiwa": "oxe_auge_kuka_iiwa",
    "panda": "oxe_auge_panda",
    "sawyer": "oxe_auge_sawyer",
    "widowx": "oxe_auge_widowX",
    "xarm7": "oxe_auge_xarm7",
    "ur5e": "oxe_auge_ur5e",
}

FINAL_ROBOT_TYPE_LIST = [
    "oxe_auge_original",
    "oxe_auge_google_robot",
    "oxe_auge_jaco",
    "oxe_auge_kinova3",
    "oxe_auge_kuka_iiwa",
    "oxe_auge_panda",
    "oxe_auge_sawyer",
    "oxe_auge_widowX",
    "oxe_auge_xarm7",
    "oxe_auge_ur5e",
]


def _normalize_robot_token(tok: str) -> str:
    t = (tok or "").strip().lower()
    if t in ("widowx", "widow_x", "widow-x", "widowx "):
        return "widowx"
    return t


def _parse_robot_key(robot_key: str) -> List[str]:
    # "google_robot,jaco" -> ["google_robot","jaco"]
    parts = [p.strip() for p in str(robot_key).split(",")]
    out = []
    for p in parts:
        nt = _normalize_robot_token(p)
        if nt:
            out.append(nt)
    return out


def _unique_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _load_dataset_robot_mapping_json(
    dataset_robot_mapping_json: Union[str, Dict[str, List[str]]]
) -> Dict[str, List[str]]:
    if isinstance(dataset_robot_mapping_json, str):
        with open(dataset_robot_mapping_json, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        mapping = dataset_robot_mapping_json

    if not isinstance(mapping, dict):
        raise TypeError(
            f"dataset_robot_mapping_json must be dict or json file path, got {type(mapping).__name__}"
        )

    normalized: Dict[str, List[str]] = {}
    for robot_key, dataset_list in mapping.items():
        if dataset_list is None:
            normalized[robot_key] = []
            continue
        if not isinstance(dataset_list, (list, tuple)):
            raise TypeError(
                f"Value for key '{robot_key}' must be list[str], got {type(dataset_list).__name__}"
            )
        clean_list = []
        for dataset_name in dataset_list:
            if not isinstance(dataset_name, str) or not dataset_name:
                raise TypeError(
                    f"Each dataset_name must be non-empty str, got {dataset_name!r} under key '{robot_key}'"
                )
            clean_list.append(dataset_name)
        normalized[robot_key] = clean_list
    return normalized


def _pick_group_id(dataset_name: str, group_prefixes: List[str]) -> str:
    if not group_prefixes:
        return dataset_name

    prefixes_sorted = sorted(group_prefixes, key=len, reverse=True)
    for p in prefixes_sorted:
        if dataset_name.startswith(p):
            return p
    return dataset_name


def _normalize_group_weights(
    group_prefixes: Optional[List[str]],
    group_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    if group_prefixes is None:
        if group_weights is not None:
            raise ValueError("group_weights should be None when group_prefixes is None.")
        return {}

    if not isinstance(group_prefixes, list) or not all(isinstance(x, str) and x for x in group_prefixes):
        raise TypeError("group_prefixes must be list[str].")

    if len(set(group_prefixes)) != len(group_prefixes):
        raise ValueError(f"group_prefixes contains duplicates: {group_prefixes}")

    if group_weights is None:
        return {p: 1.0 for p in group_prefixes}

    if not isinstance(group_weights, dict):
        raise TypeError(f"group_weights must be dict[str, float], got {type(group_weights).__name__}")

    if len(group_weights) != len(group_prefixes):
        raise ValueError(
            f"group_weights must have the same number of items as group_prefixes. "
            f"Got len(group_weights)={len(group_weights)}, len(group_prefixes)={len(group_prefixes)}"
        )

    prefix_set = set(group_prefixes)
    weight_key_set = set(group_weights.keys())
    if weight_key_set != prefix_set:
        missing = sorted(prefix_set - weight_key_set)
        extra = sorted(weight_key_set - prefix_set)
        raise ValueError(
            "group_weights keys must match group_prefixes exactly.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {extra}"
        )

    normalized = {}
    for k, v in group_weights.items():
        if not isinstance(v, (int, float)):
            raise TypeError(f"group_weights[{k!r}] must be int/float, got {type(v).__name__}")
        if v < 0:
            raise ValueError(f"group_weights[{k!r}] must be >= 0, got {v}")
        normalized[k] = float(v)

    return normalized


def _read_dataset_count(abs_root: str, dataset_name: str, intra_group_weight_mode: Optional[str]) -> float:
    if intra_group_weight_mode not in ("episode", "frame"):
        return 0.0

    info_path = os.path.join(abs_root, dataset_name, "meta", "info.json")
    count_key = "total_episodes" if intra_group_weight_mode == "episode" else "total_frames"

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
        return float(meta_info.get(count_key, 0) or 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.0


def generate_oxe_auge_dataset_mixture(
    mixture_name: str,
    abs_path: str,
    rel_path: str,
    dataset_robot_mapping_json: Union[str, Dict[str, List[str]]],
    intra_group_weight_mode: Optional[str] = "episode",  # "episode" / "frame" / None
    lerobot_version: Optional[str] = None,
    group_prefixes: Optional[List[str]] = None,
    group_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Tuple[str, float, str, Dict[str, Any]]]]:
    """
    Generate OXE-Auge dataset mixture from a single root.

    Args:
        mixture_name:
            Output dict key.
        abs_path:
            Single absolute dataset root.
        rel_path:
            Single relative dataset root.
        dataset_robot_mapping_json:
            Dict or json path. Format:
                {
                    "google_robot,jaco": ["dataset_a", "dataset_b"],
                    "panda": ["dataset_c"]
                }
        intra_group_weight_mode:
            Weighting method inside each group:
              - "episode": proportional to total_episodes within each group
              - "frame": proportional to total_frames within each group
              - None: uniform within each group
        lerobot_version:
            Optional extra config.
        group_prefixes:
            Large-dataset group prefixes. Datasets with the same matched prefix belong to the same outer group.
            Datasets not matched by any prefix become standalone groups.
        group_weights:
            Global weights for prefix-groups.
    """

    if not isinstance(abs_path, str) or not abs_path:
        raise TypeError("For OXE-Auge version, abs_path must be a non-empty str.")
    if not isinstance(rel_path, str) or not rel_path:
        raise TypeError("For OXE-Auge version, rel_path must be a non-empty str.")

    if intra_group_weight_mode not in ("episode", "frame", None):
        raise ValueError(
            f"intra_group_weight_mode must be one of ('episode', 'frame', None), "
            f"got {intra_group_weight_mode!r}"
        )

    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"abs_path does not exist or is not a directory: {abs_path}")

    mapping = _load_dataset_robot_mapping_json(dataset_robot_mapping_json)
    normalized_group_weights = _normalize_group_weights(group_prefixes, group_weights)

    try:
        on_disk_dataset_folders = {
            d for d in os.listdir(abs_path)
            if os.path.isdir(os.path.join(abs_path, d))
        }
    except Exception as e:
        raise RuntimeError(
            f"Failed to list dataset root {abs_path}: {type(e).__name__}: {e}"
        ) from e

    per_dataset: Dict[str, Dict[str, Any]] = {}

    for robot_key, dataset_list in mapping.items():
        robot_tokens = _parse_robot_key(robot_key)

        # always include original
        robot_types = ["oxe_auge_original"]

        for tok in robot_tokens:
            if tok not in _FINAL_ROBOT_TYPE_MAP:
                raise ValueError(
                    f"Unknown robot_type token '{tok}' parsed from key '{robot_key}'. "
                    f"Allowed: {sorted(_FINAL_ROBOT_TYPE_MAP.keys())}"
                )
            robot_types.append(_FINAL_ROBOT_TYPE_MAP[tok])

        robot_types = _unique_preserve_order(robot_types)

        for dataset_name in dataset_list:
            if dataset_name not in on_disk_dataset_folders:
                raise FileNotFoundError(
                    f"Dataset folder declared in mapping but missing on disk: "
                    f"{os.path.join(abs_path, dataset_name)}"
                )

            if dataset_name not in per_dataset:
                per_dataset[dataset_name] = {
                    "abs_root": abs_path,
                    "rel_root": rel_path,
                    "raw_count": 0.0,
                    "robot_types": list(robot_types),
                }
            else:
                per_dataset[dataset_name]["robot_types"] = _unique_preserve_order(
                    per_dataset[dataset_name]["robot_types"] + robot_types
                )

    dataset_names = list(per_dataset.keys())
    if group_prefixes is None:
        for dataset_name in dataset_names:
            per_dataset[dataset_name]["group_id"] = dataset_name
    else:
        for dataset_name in dataset_names:
            per_dataset[dataset_name]["group_id"] = _pick_group_id(dataset_name, group_prefixes)

    if intra_group_weight_mode in ("episode", "frame"):
        for dataset_name, rec in per_dataset.items():
            rec["raw_count"] = _read_dataset_count(
                abs_root=rec["abs_root"],
                dataset_name=dataset_name,
                intra_group_weight_mode=intra_group_weight_mode,
            )

    groups: Dict[str, List[str]] = {}
    for dataset_name, rec in per_dataset.items():
        gid = rec["group_id"]
        groups.setdefault(gid, []).append(dataset_name)

    for gid, members in groups.items():
        if intra_group_weight_mode in ("episode", "frame"):
            total_raw = sum(per_dataset[dn]["raw_count"] for dn in members)
            if total_raw > 0:
                for dn in members:
                    per_dataset[dn]["local_weight"] = per_dataset[dn]["raw_count"] / total_raw
            else:
                uniform = 1.0 / len(members)
                for dn in members:
                    per_dataset[dn]["local_weight"] = uniform
        else:
            uniform = 1.0 / len(members)
            for dn in members:
                per_dataset[dn]["local_weight"] = uniform

    merged_entries: List[Dict[str, Any]] = []
    for dataset_name, rec in per_dataset.items():
        gid = rec["group_id"]
        global_group_weight = normalized_group_weights.get(gid, 1.0)
        pre_norm_final_weight = rec["local_weight"] * global_group_weight

        merged_entries.append(
            {
                "dataset_name": dataset_name,
                "rel_root": rec["rel_root"],
                "robot_types": rec["robot_types"],
                "pre_norm_final_weight": pre_norm_final_weight,
            }
        )

    total_final = sum(x["pre_norm_final_weight"] for x in merged_entries)

    for x in merged_entries:
        x["final_weight"] = x["pre_norm_final_weight"] / total_final

    extra_config_base: Dict[str, Any] = {}
    if lerobot_version:
        extra_config_base["lerobot_version"] = lerobot_version

    mixture_list: List[Tuple[str, float, str, Dict[str, Any]]] = []

    for item in merged_entries:
        dataset_rel_path = os.path.join(item["rel_root"], item["dataset_name"]).replace("\\", "/")
        weight = round(float(item["final_weight"]), 8)

        for rtype in item["robot_types"]:
            if rtype not in FINAL_ROBOT_TYPE_LIST:
                raise ValueError(
                    f"Internal error: generated robot_type '{rtype}' not in FINAL_ROBOT_TYPE_LIST"
                )
            mixture_list.append((dataset_rel_path, weight, rtype, dict(extra_config_base)))

    return {mixture_name: mixture_list}