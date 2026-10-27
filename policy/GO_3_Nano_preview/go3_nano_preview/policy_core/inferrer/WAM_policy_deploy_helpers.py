from __future__ import annotations

import json
from typing import Any, Optional

import torch






def load_WAM_policy_infer_stats(stat_path: str, device: torch.device) -> dict[str, torch.Tensor]:
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


def resolve_WAM_policy_infer_settings(
    cfg: Any,
    *,
    robot_name: Optional[str] = None,
    stat_path: Optional[str] = None,
    joint_dim: Optional[int] = None,
    action_mode: Optional[str] = None,
) -> dict[str, Any]:
    infer_cfg = cfg.infer
    if infer_cfg is None:
        raise ValueError("cfg.infer must be set for WAM_policy deploy infer")

    robot = robot_name if robot_name is not None else infer_cfg.robot_name
    if not str(robot).strip():
        raise ValueError("cfg.infer.robot_name must be a non-empty string")

    if str(robot).strip().lower() != 'robotwin':
        raise ValueError('Only robotwin robot type is supported')
    resolved_joint_dim = joint_dim if joint_dim is not None else infer_cfg.joint_dim
    if resolved_joint_dim != 6:
        raise ValueError('Robotwin requires six joints per arm')

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
