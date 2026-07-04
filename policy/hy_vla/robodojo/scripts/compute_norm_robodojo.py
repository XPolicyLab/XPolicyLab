#!/usr/bin/env python
# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compute normalization stats for RoboDojo HDF5 datasets.

Accumulates Welford statistics for state (20-dim PosRotMat6D) and action
(20-dim RT-relative + absolute PosRotMat) in a single pass over the directory
tree, matching the UMI action computation pattern.

Action computation (mirrors ``compute_norm_umi.py``):
  * Source: EE poses from ``state/{left,right}_ee_poses`` + grippers (NOT raw
    joint actions), forming a 16-d PosQuat skeleton identical to UMI's
    ``observation.state``.
  * Each chunk is a sliding window over the downsampled state timeline.
  * Relative branch: ``dual_arm_poses_to_relative`` → (chunk, 20)
  * Absolute branch: ``convert_PosQuat2PosRotationMatrix_batch`` → (chunk, 20)

Output pickle layout::

    {
        "qpos_mean":         (20,)             # per-frame state (PosRotMat)
        "qpos_std":          (20,)
        "action_mean":       (chunk_size, 20)  # rel action (RT-relative)
        "action_std":        (chunk_size, 20)
        "action_mean_abs":   (chunk_size, 20)  # abs action (PosRotMat)
        "action_std_abs":    (chunk_size, 20)
        "first_frame":       None
    }

``--umi-coord-frame`` is set, state goes through ``convert_frame_robo_to_umi``
BEFORE stats accumulation, producing UMI-frame norm stats.  The dataset
training config must set ``umi_coord_frame=True`` to match.

``--umi-gripper-space`` (requires ``--umi-coord-frame``) additionally maps gripper
values from RoboTwin convention (0-1 norm) to UMI convention (0-90 mm).

Usage
-----
python scripts/compute_norm_robodojo.py \
        --hdf5-dir   /path/to/RoboDojo_hdf5 \
        --downsample-rate 3 \
        --chunk-size 20 \
        --output     /path/to/norm_stats.pkl

python scripts/compute_norm_robodojo.py \
        --hdf5-dir   /path/to/RoboDojo_hdf5 \
        --umi-coord-frame \
        --downsample-rate 3 \
        --chunk-size 20 \
        --output     /path/to/norm_stats_umi.pkl

python scripts/compute_norm_robodojo.py \
        --hdf5-dir   /path/to/RoboDojo_hdf5 \
        --umi-coord-frame \
        --umi-gripper-space \
        --downsample-rate 3 \
        --chunk-size 20 \
        --output     /path/to/norm_stats_umi_with_gripper.pkl
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# Make the in-repo package importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hy_vla.utils.transform_utils import (
    convert_PosQuat2PosRotationMatrix_batch,
    dual_arm_poses_to_relative,
    convert_frame_robo_to_umi,
)


# ---------------------------------------------------------------------------
# Welford online stats
# ---------------------------------------------------------------------------
def _update_welford(x: np.ndarray, count: int, mean, M2):
    for row in x:
        if count == 0:
            mean = row.astype(np.float64).copy()
            M2 = np.zeros_like(row, dtype=np.float64)
        count += 1
        delta = row.astype(np.float64) - mean
        mean += delta / count
        delta2 = row.astype(np.float64) - mean
        M2 += delta * delta2
    return count, mean, M2


def _finalize(count, mean, M2, kind: str, std_eps: float):
    if mean is None or count <= 1:
        raise RuntimeError(f"{kind} accumulator is empty (count={count}).")
    std = np.sqrt(M2 / (count - 1))
    zero_idx = np.where(std < std_eps)
    if len(zero_idx[0]) > 0:
        print(
            f"[warn] {len(zero_idx[0])} {kind} dimensions have zero std, "
            f"set to {std_eps}."
        )
        std[zero_idx] = std_eps
    return mean.astype(np.float32), std.astype(np.float32)


def _sanitize_quat(arr_16d: np.ndarray):
    """Replace zero-norm quaternions with identity [0, 0, 0, 1] in-place."""
    for cols in (slice(3, 7), slice(11, 15)):
        norms = np.linalg.norm(arr_16d[:, cols], axis=1)
        zero_mask = norms < 1e-8
        if zero_mask.any():
            arr_16d[zero_mask, cols] = [0, 0, 0, 1]


# ---------------------------------------------------------------------------
# Episode scanning (mirrors _scan_episodes in robodojo_dataset.py)
# ---------------------------------------------------------------------------
def _scan_episodes(hdf5_dir: str) -> list[dict]:
    episodes: list[dict] = []
    for task_dir in sorted(os.listdir(hdf5_dir)):
        task_path = os.path.join(hdf5_dir, task_dir)
        if not os.path.isdir(task_path):
            continue
        for robot_dir in sorted(os.listdir(task_path)):
            robot_path = os.path.join(task_path, robot_dir)
            if not os.path.isdir(robot_path):
                continue
            data_dir = os.path.join(robot_path, "data")
            if not os.path.isdir(data_dir):
                continue
            pattern = os.path.join(data_dir, "episode_*.hdf5")
            for h5_path in sorted(glob.glob(pattern)):
                basename = os.path.basename(h5_path)
                ep_id_str = basename.replace("episode_", "").replace(".hdf5", "")
                try:
                    ep_id = int(ep_id_str)
                except ValueError:
                    continue
                episodes.append({
                    "task_name": task_dir,
                    "robot_type": robot_dir,
                    "hdf5_path": h5_path,
                    "episode_id": ep_id,
                })
    return episodes


# ---------------------------------------------------------------------------
# Main accumulation
# ---------------------------------------------------------------------------
def compute(
    hdf5_dir: str,
    output_path: str,
    downsample_rate: int,
    chunk_size: int,
    umi_coord_frame: bool,
    umi_gripper_space: bool = False,
) -> None:
    print(f"[config] hdf5_dir         = {hdf5_dir}")
    print(f"[config] downsample_rate  = {downsample_rate}")
    print(f"[config] chunk_size       = {chunk_size}")
    print(f"[config] umi_coord_frame  = {umi_coord_frame}")
    print(f"[config] umi_gripper_space= {umi_gripper_space}")
    print(f"[config] output           = {output_path}")

    eps = _scan_episodes(hdf5_dir)
    if not eps:
        sys.exit(f"No episode HDF5 files found under {hdf5_dir}")
    print(f"[load] {len(eps)} episodes")

    count_qpos = 0
    mean_qpos = None
    M2_qpos = None

    count_rel = 0
    mean_rel = None
    M2_rel = None

    count_abs = 0
    mean_abs = None
    M2_abs = None

    for ep in tqdm(eps, desc="episodes"):
        h5_path = ep["hdf5_path"]
        if not os.path.isfile(h5_path):
            print(f"[warn] missing hdf5: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as f:
            # --- State: 16-dim PosQuat (same as UMI observation.state) ---
            state_grp = f["state"]
            left_ee = state_grp["left_ee_poses"][:].astype(np.float32)
            right_ee = state_grp["right_ee_poses"][:].astype(np.float32)
            left_grip = state_grp["left_ee_joint_states"][:].astype(np.float32)
            right_grip = state_grp["right_ee_joint_states"][:].astype(np.float32)
            qpos = np.concatenate([left_ee, left_grip, right_ee, right_grip],
                                  axis=1)  # (T, 16)

        # Temporal downsample: align state timeline.
        qpos = qpos[::downsample_rate]
        if qpos.shape[0] < 2:
            continue

        # HDF5 stores quaternions in wxyz; convert to xyzw (scipy convention).
        qpos[:, [3, 4, 5, 6]] = qpos[:, [4, 5, 6, 3]]       # left  wxyz→xyzw
        qpos[:, [11, 12, 13, 14]] = qpos[:, [12, 13, 14, 11]]  # right wxyz→xyzw

        # --- Optional: UMI coordinate transform ---
        if umi_coord_frame:
            qpos = convert_frame_robo_to_umi(qpos, convert_gripper=umi_gripper_space)

        # --- State accumulation (20-d PosRotMat6D) ---
        _sanitize_quat(qpos)
        qpos_20d = convert_PosQuat2PosRotationMatrix_batch(qpos, quat_order="xyzw")
        count_qpos, mean_qpos, M2_qpos = _update_welford(
            qpos_20d, count_qpos, mean_qpos, M2_qpos
        )

        # --- Action chunk accumulation (UMI-style: EE pose chunks) ---
        # Build sliding-window chunks over the downsampled state timeline.
        # Pad with the last frame so that the final timestep has a full chunk.
        repeated = np.tile(qpos[-1:, :], (chunk_size, 1))
        qpos_padded = np.concatenate([qpos, repeated], axis=0)
        action_chunks = np.lib.stride_tricks.sliding_window_view(
            qpos_padded, window_shape=(chunk_size,), axis=0
        ).copy()  # (M, 16, chunk_size)
        action_chunks = np.transpose(action_chunks, (0, 2, 1))  # (M, chunk_size, 16)

        # --- Relative branch (RT-relative) ---
        rel = np.zeros((action_chunks.shape[0], chunk_size, 20), dtype=np.float32)
        for n in range(rel.shape[0]):
            chunk = action_chunks[n].copy()
            _sanitize_quat(chunk)
            rel[n] = dual_arm_poses_to_relative(chunk)
        rel_2d = rel.reshape(rel.shape[0], -1)
        count_rel, mean_rel, M2_rel = _update_welford(
            rel_2d, count_rel, mean_rel, M2_rel
        )

        # --- Absolute branch (PosRotMat) ---
        abs_ = np.zeros_like(rel)
        for n in range(abs_.shape[0]):
            chunk = action_chunks[n].copy()
            _sanitize_quat(chunk)
            abs_[n] = convert_PosQuat2PosRotationMatrix_batch(
                chunk, quat_order="xyzw"
            )
        abs_2d = abs_.reshape(abs_.shape[0], -1)
        count_abs, mean_abs, M2_abs = _update_welford(
            abs_2d, count_abs, mean_abs, M2_abs
        )

    mean_qpos, std_qpos = _finalize(count_qpos, mean_qpos, M2_qpos, "qpos", 1e-4)
    mean_rel_f, std_rel_f = _finalize(count_rel, mean_rel, M2_rel, "rel-action", 1e-5)
    mean_abs_f, std_abs_f = _finalize(count_abs, mean_abs, M2_abs, "abs-action", 1e-5)

    mean_rel_f = mean_rel_f.reshape(chunk_size, -1)   # (chunk_size, 20)
    std_rel_f = std_rel_f.reshape(chunk_size, -1)
    mean_abs_f = mean_abs_f.reshape(chunk_size, -1)
    std_abs_f = std_abs_f.reshape(chunk_size, -1)

    print(f"[stat] qpos:      {mean_qpos.shape}")
    print(f"[stat] rel action: {mean_rel_f.shape}")
    print(f"[stat] abs action: {mean_abs_f.shape}")

    payload = {
        "qpos_mean": mean_qpos,
        "qpos_std": std_qpos,
        "action_mean": mean_rel_f,
        "action_std": std_rel_f,
        "action_mean_abs": mean_abs_f,
        "action_std_abs": std_abs_f,
        "first_frame": None,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    print(f"[save] wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hdf5-dir", required=True,
        help="Root directory of RoboDojo HDF5 episode tree.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination pkl path.",
    )
    parser.add_argument(
        "--downsample-rate", type=int, default=3,
        help="Temporal downsample rate (default: 3).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=20,
        help="Action chunk length (default: 20).",
    )
    parser.add_argument("--umi-coord-frame", action="store_true",
                        help="Apply convert_frame_robo_to_umi to state "
                             "before computing stats (UMI coordinate frame).")
    parser.add_argument("--umi-gripper-space", action="store_true",
                        help="Also convert gripper values from RoboTwin convention "
                             "(0-1 norm) to UMI convention (0-90 mm). "
                             "Requires --umi-coord-frame to be set.")
    args = parser.parse_args()

    if args.umi_gripper_space and not args.umi_coord_frame:
        sys.exit("--umi-gripper-space requires --umi-coord-frame to be set")

    if not os.path.isdir(args.hdf5_dir):
        sys.exit(f"--hdf5-dir not found: {args.hdf5_dir}")
    if args.downsample_rate < 1:
        sys.exit("--downsample-rate must be >= 1")
    if args.chunk_size < 1:
        sys.exit("--chunk-size must be >= 1")

    compute(
        hdf5_dir=args.hdf5_dir,
        output_path=args.output,
        downsample_rate=args.downsample_rate,
        chunk_size=args.chunk_size,
        umi_coord_frame=args.umi_coord_frame,
        umi_gripper_space=args.umi_gripper_space,
    )


if __name__ == "__main__":
    main()
