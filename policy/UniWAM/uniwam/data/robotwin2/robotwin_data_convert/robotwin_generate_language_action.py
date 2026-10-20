#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate language_action text files for an existing RobotWin dataset.

This script scans:
  target_root/{clean,randomized}/{task}/{input_dir_name}/*.pt
and writes:
  target_root/{clean,randomized}/{task}/language_action/*.txt

Each output text file contains one line per timestep. Line t summarizes actions
from a sliding window:
  data[t : min(t + window_size, T)]

Language action formatting aligns with LAP-style numeric summarization:
  - verbose + rotation
  - sum_decimal = "0f"
  - rotation rounded to nearest 10 degrees
  - supports delta-action inputs and absolute-pose inputs
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _round_to_nearest_n(value: float, n: int = 5) -> int:
    return int(round(value / n) * n)


def _format_numeric(val: float, sum_decimal: str) -> str:
    decimals = 0
    if isinstance(sum_decimal, str):
        if sum_decimal == "no_number":
            return ""
        if sum_decimal == "nearest_10":
            return str(int(round(val / 10) * 10))
        match = re.fullmatch(r"(\d+)f", sum_decimal)
        if match:
            decimals = int(match.group(1))
    return f"{val:.{decimals}f}"


def summarize_numeric_actions(
    arr_like: np.ndarray,
    sum_decimal: str,
    include_rotation: bool = False,
    rotation_precision: int = 10,
    include_gripper_action: bool = True,
) -> str | None:
    """
    Summarize actions into a natural-language template.

    Input shape:
      - [T, 7]/[7] or [T, 6]/[6]
    """
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 6:
        return None

    if sum_decimal in {"no_number", "nearest_10"}:
        decimals = 0
    else:
        match = re.fullmatch(r"(\d+)f", sum_decimal)
        if not match:
            return None
        decimals = int(match.group(1))

    dx_m = float(arr[..., 0].sum())
    dy_m = float(arr[..., 1].sum())
    dz_m = float(arr[..., 2].sum())
    dx = round(abs(dx_m * 100.0), decimals)
    dy = round(abs(dy_m * 100.0), decimals)
    dz = round(abs(dz_m * 100.0), decimals)

    if include_rotation:
        droll_rad = float(arr[..., 3].sum())
        dpitch_rad = float(arr[..., 4].sum())
        dyaw_rad = float(arr[..., 5].sum())
        droll = _round_to_nearest_n(abs(droll_rad * 180.0 / np.pi), rotation_precision)
        dpitch = _round_to_nearest_n(abs(dpitch_rad * 180.0 / np.pi), rotation_precision)
        dyaw = _round_to_nearest_n(abs(dyaw_rad * 180.0 / np.pi), rotation_precision)

    parts: list[str] = []

    fmt_dx = _format_numeric(dx, sum_decimal)
    fmt_dy = _format_numeric(dy, sum_decimal)
    fmt_dz = _format_numeric(dz, sum_decimal)
    if dx_m > 0 and dx != 0:
        parts.append(f"move forward {fmt_dx} cm")
    elif dx_m < 0 and dx != 0:
        parts.append(f"move back {fmt_dx} cm")
    if dz_m > 0 and dz != 0:
        parts.append(f"move up {fmt_dz} cm")
    elif dz_m < 0 and dz != 0:
        parts.append(f"move down {fmt_dz} cm")
    if dy_m > 0 and dy != 0:
        parts.append(f"move left {fmt_dy} cm")
    elif dy_m < 0 and dy != 0:
        parts.append(f"move right {fmt_dy} cm")

    if include_rotation:
        if droll_rad > 0 and droll != 0:
            parts.append(f"tilt left {droll} degrees")
        elif droll_rad < 0 and droll != 0:
            parts.append(f"tilt right {droll} degrees")
        if dpitch_rad > 0 and dpitch != 0:
            parts.append(f"tilt back {dpitch} degrees")
        elif dpitch_rad < 0 and dpitch != 0:
            parts.append(f"tilt forward {dpitch} degrees")
        if dyaw_rad > 0 and dyaw != 0:
            parts.append(f"rotate counterclockwise {dyaw} degrees")
        elif dyaw_rad < 0 and dyaw != 0:
            parts.append(f"rotate clockwise {dyaw} degrees")

    if include_gripper_action and arr.shape[-1] >= 7:
        g_last = float(arr[-1, 6])
        if g_last >= 0.5:
            parts.append("open gripper")
        else:
            parts.append("close gripper")
    return ", ".join(parts)


def summarize_bimanual_numeric_actions(
    arr_like: np.ndarray,
    sum_decimal: str,
    include_rotation: bool = False,
    include_gripper_action: bool = True,
) -> str | None:
    """
    Summarize bimanual actions:
      - Input shape [T, 14] or [14]
      - First 7 dims: left arm
      - Last 7 dims: right arm
    """
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] == 14:
        left_actions = arr[..., :7]
        right_actions = arr[..., 7:14]
    elif arr.shape[-1] == 12:
        left_actions = arr[..., :6]
        right_actions = arr[..., 6:12]
    else:
        return None

    left_summary = summarize_numeric_actions(
        left_actions,
        sum_decimal,
        include_rotation,
        include_gripper_action=include_gripper_action,
    )
    right_summary = summarize_numeric_actions(
        right_actions,
        sum_decimal,
        include_rotation,
        include_gripper_action=include_gripper_action,
    )

    if left_summary is None or right_summary is None:
        return None
    return f"Left arm: {left_summary}. Right arm: {right_summary}"


def _natural_key(path_or_name: str) -> list[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path_or_name)]


def _iter_task_dirs(subset_dir: Path) -> Iterable[Path]:
    task_dirs = [p for p in subset_dir.iterdir() if p.is_dir()]
    for task_dir in sorted(task_dirs, key=lambda p: _natural_key(p.name)):
        yield task_dir


def _load_input_array(input_path: Path) -> np.ndarray:
    data = torch.load(str(input_path), map_location="cpu")

    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        tensor = torch.tensor(data)
    elif isinstance(data, dict):
        tensor = None
        for key in ("qpos", "joint_action", "action", "actions"):
            if key in data:
                value = data[key]
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break
                if isinstance(value, np.ndarray):
                    tensor = torch.from_numpy(value)
                    break
                if isinstance(value, list):
                    tensor = torch.tensor(value)
                    break
        if tensor is None:
            raise TypeError(f"Unsupported dict tensor format: {input_path}")
    else:
        raise TypeError(f"Unsupported input format: {input_path}")

    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Input tensor must be 2D [T, D], got shape={tuple(tensor.shape)} at {input_path}")
    if tensor.shape[1] not in (6, 7, 8, 12, 14, 16):
        raise ValueError(
            f"Unsupported input dim D={tensor.shape[1]} at {input_path}, expected one of 6/7/8/12/14/16"
        )
    return tensor.numpy()


def _angle_diff_rad(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Shortest signed angular difference in radians."""
    return (curr - prev + np.pi) % (2.0 * np.pi) - np.pi


def _quat_to_rpy(quat: np.ndarray, quat_order: str = "wxyz") -> np.ndarray:
    """Convert quaternion [T,4] to RPY [T,3] (radians)."""
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"quat must be [T,4], got {quat.shape}")

    q = quat.astype(np.float64, copy=True)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Found near-zero quaternion norm in pose data.")
    q /= norm

    if quat_order == "wxyz":
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif quat_order == "xyzw":
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError(f"Unsupported quat_order={quat_order}")

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=1).astype(np.float32, copy=False)


def _delta_single_arm_from_abs(window_abs: np.ndarray, include_gripper: bool) -> np.ndarray:
    """
    Convert single-arm absolute xyzrpy(+gripper) to per-step deltas.
    Output shape:
      - include_gripper=True  -> [max(T-1,1), 7]
      - include_gripper=False -> [max(T-1,1), 6]
    """
    t = window_abs.shape[0]
    if t <= 1:
        base = np.zeros((1, 6), dtype=np.float32)
        if include_gripper:
            g = np.array([[float(window_abs[-1, 6])]], dtype=np.float32)
            return np.concatenate([base, g], axis=1)
        return base

    dpos = (window_abs[1:, :3] - window_abs[:-1, :3]).astype(np.float32, copy=False)
    drot = _angle_diff_rad(window_abs[1:, 3:6], window_abs[:-1, 3:6]).astype(np.float32, copy=False)
    if include_gripper:
        g = window_abs[1:, 6:7].astype(np.float32, copy=False)
        return np.concatenate([dpos, drot, g], axis=1)
    return np.concatenate([dpos, drot], axis=1)


def _to_delta_window(
    window_actions: np.ndarray,
    input_mode: str,
    input_dir_name: str,
    quat_order: str,
) -> tuple[np.ndarray, bool]:
    """
    Convert window data to LAP-style delta action window.

    Returns:
      (delta_window, include_gripper_action)
    """
    dim = int(window_actions.shape[1])
    mode = input_mode
    if mode == "auto":
        dir_hint = input_dir_name.lower()
        if dim in (8, 16):
            mode = "absolute_xyzquat"
        elif dir_hint in {"epos", "eef", "endpose", "pose", "poses"}:
            mode = "absolute_xyzrpy"
        else:
            mode = "delta"

    if mode == "delta":
        if dim not in (7, 14):
            raise ValueError(
                f"input_mode=delta expects D in {{7,14}}, got D={dim}. "
                "Use --input-mode absolute_xyzrpy/absolute_xyzquat for absolute poses."
            )
        return window_actions, True

    if mode == "absolute_xyzquat":
        if dim == 8:
            xyz = window_actions[:, :3]
            quat = window_actions[:, 3:7]
            rpy = _quat_to_rpy(quat, quat_order=quat_order)
            g = window_actions[:, 7:8]
            abs_single = np.concatenate([xyz, rpy, g], axis=1)
            return _delta_single_arm_from_abs(abs_single, include_gripper=True), True
        if dim == 16:
            left_xyz = window_actions[:, :3]
            left_quat = window_actions[:, 3:7]
            left_g = window_actions[:, 7:8]
            right_xyz = window_actions[:, 8:11]
            right_quat = window_actions[:, 11:15]
            right_g = window_actions[:, 15:16]
            left_abs = np.concatenate([left_xyz, _quat_to_rpy(left_quat, quat_order), left_g], axis=1)
            right_abs = np.concatenate([right_xyz, _quat_to_rpy(right_quat, quat_order), right_g], axis=1)
            left_delta = _delta_single_arm_from_abs(left_abs, include_gripper=True)
            right_delta = _delta_single_arm_from_abs(right_abs, include_gripper=True)
            return np.concatenate([left_delta, right_delta], axis=1), True
        raise ValueError(f"input_mode=absolute_xyzquat expects D in {{8,16}}, got D={dim}")

    if mode == "absolute_xyzrpy":
        if dim == 7:
            return _delta_single_arm_from_abs(window_actions, include_gripper=True), True
        if dim == 6:
            return _delta_single_arm_from_abs(window_actions, include_gripper=False), False
        if dim == 14:
            left_delta = _delta_single_arm_from_abs(window_actions[:, :7], include_gripper=True)
            right_delta = _delta_single_arm_from_abs(window_actions[:, 7:14], include_gripper=True)
            return np.concatenate([left_delta, right_delta], axis=1), True
        if dim == 12:
            left_delta = _delta_single_arm_from_abs(window_actions[:, :6], include_gripper=False)
            right_delta = _delta_single_arm_from_abs(window_actions[:, 6:12], include_gripper=False)
            return np.concatenate([left_delta, right_delta], axis=1), False
        raise ValueError(f"input_mode=absolute_xyzrpy expects D in {{6,7,12,14}}, got D={dim}")

    raise ValueError(f"Unsupported input_mode={input_mode}")


def _summarize_window(
    window_actions: np.ndarray,
    input_mode: str,
    input_dir_name: str,
    quat_order: str,
) -> str:
    delta_window, include_gripper_action = _to_delta_window(
        window_actions=window_actions,
        input_mode=input_mode,
        input_dir_name=input_dir_name,
        quat_order=quat_order,
    )
    dim = delta_window.shape[1]
    if dim == 14:
        text = summarize_bimanual_numeric_actions(
            delta_window,
            sum_decimal="0f",
            include_rotation=True,
            include_gripper_action=include_gripper_action,
        )
    elif dim == 12:
        text = summarize_bimanual_numeric_actions(
            delta_window,
            sum_decimal="0f",
            include_rotation=True,
            include_gripper_action=False,
        )
    elif dim == 7:
        text = summarize_numeric_actions(
            delta_window,
            sum_decimal="0f",
            include_rotation=True,
            include_gripper_action=include_gripper_action,
        )
    elif dim == 6:
        text = summarize_numeric_actions(
            delta_window,
            sum_decimal="0f",
            include_rotation=True,
            include_gripper_action=False,
        )
    else:
        raise ValueError(f"Unsupported converted delta dim: {dim}")

    if text is None:
        raise ValueError(f"Failed to summarize window with shape={window_actions.shape} converted={delta_window.shape}")
    return text


@dataclass
class BackfillStats:
    tasks_scanned: int = 0
    episodes_total: int = 0
    episodes_generated: int = 0
    episodes_skipped_existing: int = 0
    episodes_failed: int = 0
    generated_lines: int = 0


class RobotWinLanguageActionBackfill:
    """Backfill language_action text files under RobotWin target_root."""

    def __init__(
        self,
        target_root: Path,
        subsets: list[str],
        window_size: int = 16,
        input_dir_name: str = "qpos",
        input_mode: str = "auto",
        quat_order: str = "wxyz",
        output_dir_name: str = "language_action",
        overwrite: bool = False,
    ) -> None:
        self.target_root = target_root
        self.subsets = subsets
        self.window_size = int(window_size)
        self.input_dir_name = input_dir_name
        self.input_mode = input_mode
        self.quat_order = quat_order
        self.output_dir_name = output_dir_name
        self.overwrite = overwrite

        self._validate()

    def _validate(self) -> None:
        if not self.target_root.exists():
            raise FileNotFoundError(f"target_root not found: {self.target_root}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if not self.subsets:
            raise ValueError("subsets is empty")
        if not self.input_dir_name.strip():
            raise ValueError("input_dir_name must be non-empty")
        if self.input_mode not in {"auto", "delta", "absolute_xyzrpy", "absolute_xyzquat"}:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")
        if self.quat_order not in {"wxyz", "xyzw"}:
            raise ValueError(f"Unsupported quat_order: {self.quat_order}")
        if not self.output_dir_name.strip():
            raise ValueError("output_dir_name must be non-empty")

    def _process_episode(self, input_path: Path, out_path: Path) -> tuple[bool, int]:
        arr = _load_input_array(input_path)
        timesteps = arr.shape[0]

        lines: list[str] = []
        for t in range(timesteps):
            end = min(t + self.window_size, timesteps)
            lines.append(
                _summarize_window(
                    window_actions=arr[t:end, :],
                    input_mode=self.input_mode,
                    input_dir_name=self.input_dir_name,
                    quat_order=self.quat_order,
                )
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True, timesteps

    def run(self) -> int:
        stats = BackfillStats()

        logger.info(
            "Language action backfill started: target_root=%s subsets=%s window_size=%d input_dir=%s input_mode=%s quat_order=%s output_dir=%s overwrite=%s",
            self.target_root,
            ",".join(self.subsets),
            self.window_size,
            self.input_dir_name,
            self.input_mode,
            self.quat_order,
            self.output_dir_name,
            self.overwrite,
        )

        for subset in self.subsets:
            subset_dir = self.target_root / subset
            if not subset_dir.exists():
                logger.warning("Subset directory not found, skip: %s", subset_dir)
                continue

            for task_dir in _iter_task_dirs(subset_dir):
                stats.tasks_scanned += 1
                input_dir = task_dir / self.input_dir_name
                output_dir = task_dir / self.output_dir_name

                if not input_dir.exists():
                    logger.warning("Missing input directory(%s), skip task: %s", self.input_dir_name, task_dir)
                    continue

                input_files = sorted(input_dir.glob("*.pt"), key=lambda p: _natural_key(p.name))
                if not input_files:
                    logger.warning("No %s .pt files found, skip task: %s", self.input_dir_name, task_dir)
                    continue

                for input_path in input_files:
                    stats.episodes_total += 1
                    out_path = output_dir / f"{input_path.stem}.txt"

                    if out_path.exists() and not self.overwrite:
                        stats.episodes_skipped_existing += 1
                        continue

                    try:
                        ok, num_lines = self._process_episode(input_path, out_path)
                        if ok:
                            stats.episodes_generated += 1
                            stats.generated_lines += num_lines
                    except Exception as err:  # noqa: BLE001
                        stats.episodes_failed += 1
                        logger.error("Failed episode: %s error=%s", input_path, err)

        logger.info("Language action backfill finished.")
        logger.info("  tasks_scanned=%d", stats.tasks_scanned)
        logger.info("  episodes_total=%d", stats.episodes_total)
        logger.info("  episodes_generated=%d", stats.episodes_generated)
        logger.info("  episodes_skipped_existing=%d", stats.episodes_skipped_existing)
        logger.info("  episodes_failed=%d", stats.episodes_failed)
        logger.info("  generated_lines=%d", stats.generated_lines)

        return 0 if stats.episodes_failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill language_action text files for RobotWin dataset.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("./data/robotwin_dataset"),
        help="Root directory of converted RobotWin dataset.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="clean,randomized",
        help="Comma-separated subset names to process.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Sliding window size for action accumulation.",
    )
    parser.add_argument(
        "--input-dir-name",
        type=str,
        default="qpos",
        help="Input directory name under each task (e.g., qpos, epos).",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["auto", "delta", "absolute_xyzrpy", "absolute_xyzquat"],
        default="auto",
        help=(
            "How to interpret input tensors. "
            "delta: [dx,dy,dz,droll,dpitch,dyaw,(gripper)] per step; "
            "absolute_xyzrpy: [x,y,z,roll,pitch,yaw,(gripper)] absolute pose; "
            "absolute_xyzquat: [x,y,z,q*,q*,q*,q*,(gripper)] absolute pose."
        ),
    )
    parser.add_argument(
        "--quat-order",
        type=str,
        choices=["wxyz", "xyzw"],
        default="wxyz",
        help="Quaternion order when input-mode=absolute_xyzquat.",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="language_action",
        help="Output directory name under each task.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing language_action/*.txt files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    try:
        runner = RobotWinLanguageActionBackfill(
            target_root=args.target_root,
            subsets=subsets,
            window_size=args.window_size,
            input_dir_name=args.input_dir_name,
            input_mode=args.input_mode,
            quat_order=args.quat_order,
            output_dir_name=args.output_dir_name,
            overwrite=args.overwrite,
        )
        code = runner.run()
        sys.exit(code)
    except Exception as err:  # noqa: BLE001
        logger.error("Language action backfill failed: %s", err)
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()


