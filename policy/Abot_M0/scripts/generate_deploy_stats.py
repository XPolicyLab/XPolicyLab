#!/usr/bin/env python3
"""Bootstrap Abot dataset metadata and write deploy dataset_statistics.json (CPU only)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1]
ABOT_ROOT = POLICY_DIR / "abot_m0"
if str(ABOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ABOT_ROOT))

from ABot.dataloader.gr00t_lerobot.datasets import (  # noqa: E402
    calculate_dataset_statistics,
    combine_modality_stats,
    generate_action_mask_for_used_keys,
)
from ABot.dataloader.gr00t_lerobot.schema import (  # noqa: E402
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
)

LE_ROBOT_DATA_GLOB = "data/*/*.parquet"
LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t.json"
DEFAULT_DATA_ROOT = "/mnt/xspark-data/xspark_shared/lerobot"
DEFAULT_REPO = "RoboDojo_sim_v21_video_abot"
FALLBACK_REPO = "RoboDojo_lerobot_v21_video"
UNNORM_KEY = "robodojo_sim"
ACTION_SUBKEYS = (
    "action.left_joints",
    "action.right_joints",
    "action.left_gripper",
    "action.right_gripper",
)


def modality_layout_to_train_layout(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.concatenate([x[..., 0:6], x[..., 7:13], x[..., 6:7], x[..., 13:14]], axis=-1)


def bootstrap_dataset(dataset_dir: Path, fallback_dir: Path, prepare_script: Path) -> None:
    meta_dir = dataset_dir / "meta"
    if not meta_dir.is_dir():
        print(f"[bootstrap] copying meta/ from {fallback_dir}")
        shutil.copytree(fallback_dir / "meta", meta_dir)

    data_link = dataset_dir / "data"
    if not data_link.exists():
        fallback_data = fallback_dir / "data"
        if not fallback_data.is_dir():
            raise FileNotFoundError(f"Missing parquet data in {fallback_data}")
        print(f"[bootstrap] linking data/ -> {fallback_data}")
        data_link.symlink_to(fallback_data, target_is_directory=True)

    required_video_dirs = (
        "videos/chunk-000/observation.images.cam_high",
        "videos/chunk-000/observation.images.cam_left_wrist",
        "videos/chunk-000/observation.images.cam_right_wrist",
    )
    missing_videos = [rel for rel in required_video_dirs if not (dataset_dir / rel).exists()]
    if missing_videos:
        raise FileNotFoundError(f"Dataset missing required video dirs: {missing_videos}")

    print(f"[bootstrap] running {prepare_script.name}")
    subprocess.run(
        [sys.executable, str(prepare_script), "--dataset-dir", str(dataset_dir)],
        check=True,
    )


def compute_stats_gr00t(dataset_dir: Path) -> dict:
    stats_path = dataset_dir / LE_ROBOT_STATS_FILENAME
    if stats_path.is_file():
        print(f"[stats] reusing existing {stats_path}")
        with stats_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    parquet_files = sorted(dataset_dir.glob(LE_ROBOT_DATA_GLOB))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {dataset_dir}/{LE_ROBOT_DATA_GLOB}")

    print(f"[stats] computing stats_gr00t.json from {len(parquet_files)} parquet files")
    le_statistics = calculate_dataset_statistics(parquet_files)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(le_statistics, handle, indent=2)
    print(f"[stats] wrote {stats_path}")
    return le_statistics


def expand_action_mask(
    subkey_mask: list[bool], le_modality_meta: LeRobotModalityMetadata
) -> list[bool]:
    full_mask: list[bool] = []
    for subkey, apply_linear in zip(
        ("left_joints", "right_joints", "left_gripper", "right_gripper"),
        subkey_mask,
    ):
        meta = le_modality_meta.get_key_meta(f"action.{subkey}")
        dim = meta.end - meta.start
        full_mask.extend([apply_linear] * dim)
    return full_mask


def build_train_layout_action_stats(dataset_dir: Path, le_statistics: dict) -> dict:
    modality_path = dataset_dir / "meta/modality.json"
    with modality_path.open("r", encoding="utf-8") as handle:
        modality_json = json.load(handle)

    le_modality_meta = LeRobotModalityMetadata.model_validate(modality_json)
    subkey_stats = {}
    for subkey in ("left_joints", "right_joints", "left_gripper", "right_gripper"):
        meta = le_modality_meta.get_key_meta(f"action.{subkey}")
        le_key = meta.original_key
        indices = np.arange(meta.start, meta.end)
        entry = {}
        for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
            values = np.asarray(le_statistics[le_key][stat_name])[indices]
            entry[stat_name] = values.tolist()
        subkey_stats[f"action.{subkey}"] = DatasetStatisticalValues.model_validate(entry)

    combined = combine_modality_stats(subkey_stats)
    action_modalities = {
        key: le_modality_meta.get_key_meta(key) for key in ACTION_SUBKEYS
    }
    subkey_mask = generate_action_mask_for_used_keys(action_modalities, ACTION_SUBKEYS)
    combined["mask"] = expand_action_mask(subkey_mask, le_modality_meta)
    return combined


def write_dataset_statistics(run_dirs: list[Path], action_stats: dict) -> None:
    payload = {UNNORM_KEY: {"action": action_stats}}
    for run_dir in run_dirs:
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "dataset_statistics.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[deploy] wrote {out_path}")


def summarize(action_stats: dict) -> None:
    min_train = np.asarray(action_stats["min"])
    max_train = np.asarray(action_stats["max"])
    mask = np.asarray(action_stats["mask"], dtype=bool)
    print("[verify] train-layout action stats")
    print(f"  dim 12 (left_gripper): min={min_train[12]:.4f}, max={max_train[12]:.4f}, mask={mask[12]}")
    print(f"  dim 13 (right_gripper): min={min_train[13]:.4f}, max={max_train[13]:.4f}, mask={mask[13]}")
    print(f"  dim 6 (right_joint_0): min={min_train[6]:.4f}, max={max_train[6]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-repo", type=Path, default=Path(DEFAULT_REPO))
    parser.add_argument("--fallback-repo", type=Path, default=Path(FALLBACK_REPO))
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[
            ABOT_ROOT / "checkpoints/RoboDojo-cotrain-abot-3500-joint-1",
            ABOT_ROOT / "checkpoints/RoboDojo-cotrain-abot-3500-joint-2",
        ],
        help="Checkpoint run dirs to receive dataset_statistics.json (repeatable)",
    )
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()

    dataset_dir = (args.data_root / args.dataset_repo).resolve()
    fallback_dir = (args.data_root / args.fallback_repo).resolve()
    prepare_script = ABOT_ROOT / "examples/RoboDojo/prepare_RoboDojo_abot.py"

    if not args.skip_bootstrap:
        bootstrap_dataset(dataset_dir, fallback_dir, prepare_script)

    le_statistics = compute_stats_gr00t(dataset_dir)
    action_stats = build_train_layout_action_stats(dataset_dir, le_statistics)
    summarize(action_stats)
    write_dataset_statistics([path.resolve() for path in args.run_dir], action_stats)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
