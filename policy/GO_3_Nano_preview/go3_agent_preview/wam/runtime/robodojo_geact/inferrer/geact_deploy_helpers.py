from __future__ import annotations

import json
from typing import Any, Optional

import torch


def g2_sim_gripper_denormal(values: torch.Tensor) -> torch.Tensor:
    open_val, closed_val = -0.785, 0.0
    return values.clamp(0.0, 1.0) * (closed_val - open_val) + open_val


def g2_90d_gripper_unnormal(values: torch.Tensor) -> torch.Tensor:
    open_val, closed_val = 0.0, -0.91
    value = values * (closed_val - open_val) + open_val
    return value.clamp(-0.91, 0.0)


def load_geact_infer_stats(stat_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    with open(stat_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def to_tensor(key: str, sub_key: str) -> torch.Tensor:
        return torch.tensor(
            data[key][sub_key],
            device=device,
            dtype=torch.float32,
        )

    return {
        "a_mean": to_tensor("action", "mean"),
        "a_std": to_tensor("action", "std"),
        "s_mean": to_tensor("state", "mean"),
        "s_std": to_tensor("state", "std"),
    }


def resolve_geact_infer_settings(
    cfg: Any,
    *,
    robot_name: Optional[str] = None,
    stat_path: Optional[str] = None,
    joint_dim: Optional[int] = None,
    action_mode: Optional[str] = None,
) -> dict[str, Any]:
    infer_cfg = cfg.infer
    if infer_cfg is None:
        raise ValueError("cfg.infer must be set for GEAct deploy infer")

    robot = robot_name if robot_name is not None else infer_cfg.robot_name
    if not str(robot).strip():
        raise ValueError("cfg.infer.robot_name must be a non-empty string")

    stats = stat_path if stat_path is not None else infer_cfg.stat_path
    if not str(stats).strip():
        raise ValueError(
            "cfg.infer.stat_path must be a non-empty path "
            "(set it on the infer cfg; do not leave empty)"
        )

    return {
        "robot_name": str(robot).strip().lower().replace("-", "_"),
        "stat_path": str(stats),
        "joint_dim": joint_dim if joint_dim is not None else infer_cfg.joint_dim,
        "action_mode": (
            action_mode if action_mode is not None else infer_cfg.action_mode
        ),
    }


def apply_libero_deploy_overrides(
    cfg: Any,
    *,
    dense_action_steps: int | None = None,
    stat_path: str | None = None,
) -> None:
    for split in (cfg.data.train, cfg.data.val):
        split.cam_use = [["external_1", "waist"]]
        split.n_view = 2
        if hasattr(split, "sample_size") and not split.sample_size:
            split.sample_size = [256, 384]
    cfg.infer.robot_name = "libero"
    if stat_path is not None:
        resolved = str(stat_path).strip()
        if not resolved:
            raise ValueError("stat_path override must be a non-empty path")
        cfg.infer.stat_path = resolved
    cfg.infer.joint_dim = 6
    cfg.infer.action_mode = "abs"
    if dense_action_steps is not None:
        cfg.validation.dense_action_steps = int(dense_action_steps)
