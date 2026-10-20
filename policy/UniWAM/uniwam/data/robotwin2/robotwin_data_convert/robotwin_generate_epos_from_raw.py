#!/usr/bin/env python3
"""
Backfill eef/endpose trajectories into Motus RobotWin dataset as `epos/*.pt`.

Source (raw):
  {raw_root}/{task}/{raw_subset}/data/episode{episode_id}.hdf5
Target:
  {target_root}/{subset}/{task}/epos/{episode_id}.pt

By default, this script aligns episode IDs to existing `qpos/*.pt` files under
the target dataset to avoid split/task/episode mismatch.

Output format (default):
  [left_xyz(3), left_rpy(3), left_gripper(1), right_xyz(3), right_rpy(3), right_gripper(1)]
  => [T, 14]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch


LOGGER = logging.getLogger("robotwin_generate_epos_from_raw")


RAW_SUBSET_MAP: Dict[str, str] = {
    "clean": "aloha-agilex_clean_50",
    "randomized": "aloha-agilex_randomized_500",
}


@dataclass
class Stats:
    tasks: int = 0
    episodes_total: int = 0
    success: int = 0
    skipped_existing: int = 0
    failed: int = 0
    missing_raw_hdf5: int = 0
    mismatched_length: int = 0


def _natural_key(text: str) -> Tuple:
    parts: List[object] = []
    current = ""
    is_digit = text[:1].isdigit()
    for ch in text:
        if ch.isdigit() == is_digit:
            current += ch
        else:
            parts.append(int(current) if is_digit else current)
            current = ch
            is_digit = ch.isdigit()
    if current:
        parts.append(int(current) if is_digit else current)
    return tuple(parts)


def _load_qpos_len(qpos_path: Path) -> int:
    data = torch.load(str(qpos_path), map_location="cpu")
    if isinstance(data, torch.Tensor):
        if data.ndim != 2:
            raise ValueError(f"qpos tensor must be 2D [T, D], got {tuple(data.shape)} at {qpos_path}")
        return int(data.shape[0])
    if isinstance(data, dict):
        for key in ("qpos", "joint_action", "action", "actions"):
            if key in data and isinstance(data[key], torch.Tensor) and data[key].ndim == 2:
                return int(data[key].shape[0])
    raise TypeError(f"Unsupported qpos format: {qpos_path}")


def _quat_to_rpy(quat: np.ndarray, quat_order: str = "wxyz") -> np.ndarray:
    """
    Convert quaternion to roll-pitch-yaw (radians).

    Args:
        quat: [T, 4]
        quat_order:
            - "wxyz": quat = [qw, qx, qy, qz]
            - "xyzw": quat = [qx, qy, qz, qw]
    Returns:
        rpy: [T, 3] in radians
    """
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"quat must be [T,4], got {quat.shape}")

    q = quat.astype(np.float64, copy=True)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Found near-zero quaternion norm.")
    q /= norm

    if quat_order == "wxyz":
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif quat_order == "xyzw":
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError(f"Unsupported quat_order={quat_order}")

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=1).astype(np.float32, copy=False)


def _extract_epos_from_hdf5(
    hdf5_path: Path,
    quat_order: str = "wxyz",
    include_gripper: bool = True,
) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h:
        required_keys = [
            "endpose/left_endpose",
            "endpose/left_gripper",
            "endpose/right_endpose",
            "endpose/right_gripper",
        ]
        missing = [k for k in required_keys if k not in h]
        if missing:
            raise KeyError(f"Missing keys in {hdf5_path}: {missing}")

        left_pose = np.asarray(h["endpose/left_endpose"][()])       # [T, 7]
        left_gripper = np.asarray(h["endpose/left_gripper"][()])    # [T]
        right_pose = np.asarray(h["endpose/right_endpose"][()])     # [T, 7]
        right_gripper = np.asarray(h["endpose/right_gripper"][()])  # [T]

    if left_pose.ndim != 2 or left_pose.shape[1] != 7:
        raise ValueError(f"left_endpose shape must be [T,7], got {left_pose.shape} at {hdf5_path}")
    if right_pose.ndim != 2 or right_pose.shape[1] != 7:
        raise ValueError(f"right_endpose shape must be [T,7], got {right_pose.shape} at {hdf5_path}")
    if left_gripper.ndim != 1:
        raise ValueError(f"left_gripper shape must be [T], got {left_gripper.shape} at {hdf5_path}")
    if right_gripper.ndim != 1:
        raise ValueError(f"right_gripper shape must be [T], got {right_gripper.shape} at {hdf5_path}")

    t = left_pose.shape[0]
    if not (right_pose.shape[0] == left_gripper.shape[0] == right_gripper.shape[0] == t):
        raise ValueError(
            f"T mismatch in {hdf5_path}: "
            f"left_pose={left_pose.shape[0]}, right_pose={right_pose.shape[0]}, "
            f"left_gripper={left_gripper.shape[0]}, right_gripper={right_gripper.shape[0]}"
        )

    left_xyz = left_pose[:, :3]
    right_xyz = right_pose[:, :3]
    left_quat = left_pose[:, 3:7]
    right_quat = right_pose[:, 3:7]

    left_rpy = _quat_to_rpy(left_quat, quat_order=quat_order)
    right_rpy = _quat_to_rpy(right_quat, quat_order=quat_order)

    if include_gripper:
        left_gripper = left_gripper.reshape(t, 1)
        right_gripper = right_gripper.reshape(t, 1)
        # [left_xyz(3), left_rpy(3), left_gripper(1), right_xyz(3), right_rpy(3), right_gripper(1)] => [T, 14]
        epos = np.concatenate([left_xyz, left_rpy, left_gripper, right_xyz, right_rpy, right_gripper], axis=1)
    else:
        # [left_xyz(3), left_rpy(3), right_xyz(3), right_rpy(3)] => [T, 12]
        epos = np.concatenate([left_xyz, left_rpy, right_xyz, right_rpy], axis=1)
    return epos.astype(np.float32, copy=False)


def _iter_target_tasks(target_root: Path, subset: str, tasks: Optional[set[str]]) -> Iterable[Path]:
    subset_dir = target_root / subset
    if not subset_dir.exists():
        return []
    task_dirs = [d for d in subset_dir.iterdir() if d.is_dir()]
    if tasks is not None:
        task_dirs = [d for d in task_dirs if d.name in tasks]
    task_dirs.sort(key=lambda p: _natural_key(p.name))
    return task_dirs


def _process_one_episode(
    qpos_path: Path,
    raw_hdf5_path: Path,
    out_path: Path,
    overwrite: bool,
    length_mode: str,
    align_with_qpos: bool,
    quat_order: str,
    include_gripper: bool,
    stats: Stats,
) -> None:
    stats.episodes_total += 1

    if out_path.exists() and not overwrite:
        stats.skipped_existing += 1
        return

    if not raw_hdf5_path.exists():
        stats.failed += 1
        stats.missing_raw_hdf5 += 1
        raise FileNotFoundError(f"Missing raw hdf5: {raw_hdf5_path}")

    epos = _extract_epos_from_hdf5(
        raw_hdf5_path,
        quat_order=quat_order,
        include_gripper=include_gripper,
    )
    print(epos.shape)
    print(epos)

    if align_with_qpos:
        qpos_len = _load_qpos_len(qpos_path)
        epos_len = int(epos.shape[0])
        if qpos_len != epos_len:
            stats.mismatched_length += 1
            if length_mode == "strict":
                stats.failed += 1
                raise ValueError(
                    f"Length mismatch episode={qpos_path.stem}: qpos_len={qpos_len}, epos_len={epos_len}"
                )
            # trim mode
            keep = min(qpos_len, epos_len)
            if keep <= 0:
                stats.failed += 1
                raise ValueError(
                    f"Non-positive aligned length episode={qpos_path.stem}: qpos_len={qpos_len}, epos_len={epos_len}"
                )
            epos = epos[:keep]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(epos), str(out_path))
    stats.success += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RobotWin raw endpose(eef) to Motus target dataset epos/*.pt"
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("./data/robotwin_raw_dataset"),
        help="Raw RobotWin root directory.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("./data/robotwin_dataset"),
        help="Motus converted RobotWin root directory.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="clean,randomized",
        help="Comma-separated subsets to process. Choices: clean,randomized",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Optional comma-separated task names. Empty means all tasks in target subset.",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="epos",
        help="Output directory name under each task directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing epos/*.pt",
    )
    parser.add_argument(
        "--quat-order",
        type=str,
        choices=["wxyz", "xyzw"],
        default="wxyz",
        help="Quaternion order in raw endpose datasets.",
    )
    parser.add_argument(
        "--without-gripper",
        action="store_true",
        help="Do not append left/right gripper values to output.",
    )
    parser.add_argument(
        "--align-with-qpos",
        action="store_true",
        default=True,
        help="Align length to qpos episode length (default: enabled).",
    )
    parser.add_argument(
        "--no-align-with-qpos",
        action="store_false",
        dest="align_with_qpos",
        help="Disable qpos length alignment.",
    )
    parser.add_argument(
        "--length-mode",
        type=str,
        choices=["trim", "strict"],
        default="trim",
        help="How to handle epos/qpos length mismatch when align-with-qpos is enabled.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=0,
        help="Optional limit of episodes per task for debugging (0 means no limit).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    raw_root: Path = args.raw_root
    target_root: Path = args.target_root
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()} or None

    if not raw_root.exists():
        raise FileNotFoundError(f"raw root not found: {raw_root}")
    if not target_root.exists():
        raise FileNotFoundError(f"target root not found: {target_root}")

    invalid = [s for s in subsets if s not in RAW_SUBSET_MAP]
    if invalid:
        raise ValueError(f"Unsupported subsets: {invalid}. Allowed: {list(RAW_SUBSET_MAP.keys())}")

    stats = Stats()

    for subset in subsets:
        raw_subset_name = RAW_SUBSET_MAP[subset]
        task_dirs = list(_iter_target_tasks(target_root, subset, tasks))
        LOGGER.info("Processing subset=%s tasks=%d", subset, len(task_dirs))

        for task_dir in task_dirs:
            stats.tasks += 1
            task_name = task_dir.name
            qpos_dir = task_dir / "qpos"
            if not qpos_dir.exists():
                LOGGER.warning("Skip task without qpos: %s", task_dir)
                continue

            out_dir = task_dir / args.output_dir_name
            qpos_files = sorted(qpos_dir.glob("*.pt"), key=lambda p: _natural_key(p.name))
            if args.episode_limit and args.episode_limit > 0:
                qpos_files = qpos_files[: args.episode_limit]

            if not qpos_files:
                LOGGER.warning("Skip task with no qpos files: %s", task_dir)
                continue

            raw_data_dir = raw_root / task_name / raw_subset_name / "data"
            if not raw_data_dir.exists():
                LOGGER.warning("Raw data dir missing, skip task: %s", raw_data_dir)
                continue

            LOGGER.info(
                "Task %s/%s: episodes=%d -> out=%s",
                subset, task_name, len(qpos_files), out_dir
            )

            for qpos_path in qpos_files:
                episode_id = qpos_path.stem
                raw_hdf5_path = raw_data_dir / f"episode{episode_id}.hdf5"
                out_path = out_dir / f"{episode_id}.pt"
                try:
                    _process_one_episode(
                        qpos_path=qpos_path,
                        raw_hdf5_path=raw_hdf5_path,
                        out_path=out_path,
                        overwrite=args.overwrite,
                        length_mode=args.length_mode,
                        align_with_qpos=args.align_with_qpos,
                        quat_order=args.quat_order,
                        include_gripper=not args.without_gripper,
                        stats=stats,
                    )
                except Exception as err:
                    LOGGER.error("Failed episode subset=%s task=%s id=%s err=%s", subset, task_name, episode_id, err)

    LOGGER.info(
        "Done. tasks=%d episodes=%d success=%d skipped_existing=%d failed=%d missing_raw_hdf5=%d mismatched_length=%d",
        stats.tasks,
        stats.episodes_total,
        stats.success,
        stats.skipped_existing,
        stats.failed,
        stats.missing_raw_hdf5,
        stats.mismatched_length,
    )
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


