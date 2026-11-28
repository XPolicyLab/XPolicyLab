"""Typed config for the internal x2robot_dataset v1 backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from wall_x.config.data_config import DataConfig
from wall_x.config.registry import register_data_config


@register_data_config("x2robot_v1")
@dataclass
class X2RobotV1DataConfig(DataConfig):
    """x2robot_dataset v1 data config.

    Mirrors the knobs consumed by x2robot_dataset (X2RDataConfig /
    vg_processor); None means unset (backend default applies).
    """

    dataset_type: str = "x2robot_v1"
    dataset_config_path: Optional[str] = None
    dataset_path: Optional[List[Dict[str, Any]]] = None
    buffer_size: int = 2000
    preload_size: Optional[int] = None
    cache_dir: Optional[str] = None
    rgb_keys: Optional[List[str]] = None
    lowdim_keys: Optional[List[str]] = None
    tactile_keys: Optional[List[str]] = None
    instruction_path: Optional[Any] = None
    instruction_key: Optional[Any] = None
    multimodal_data_ratio: float = 0.0
    image_history_length: int = 1
    image_horizon: int = 1
    image_history_interval: int = 1
    obs_history_stride: int = 0
    action_history_length: int = 0
    event_level: bool = False
    # --- event-clip sampling (vg_processor) ---
    min_clip_length: Optional[int] = None
    max_clip_length: Optional[int] = None
    skip_interval: int = 2
    future_image_length: Optional[int] = None
    future_image_interval: Optional[int] = None
    # --- weighted / task-balanced sampling ---
    weighted_sampling: bool = False
    task_balance_alpha: Optional[float] = None
    task_balance_warmup_epochs: Optional[float] = None
    task_balance_tau_start: Optional[float] = None
    task_balance_tau_end: Optional[float] = None
    log_weighted_sampling: bool = False
    # --- frame filtering ---
    remove_static_strict: Optional[bool] = None
    remove_static_ratio: Optional[float] = None
    static_action_threshold: Optional[float] = None
    remove_scene_changes: Optional[bool] = None
    scene_change_dims: Optional[Any] = None
    # --- classifier-free-guidance caption dropout ---
    enable_cfg: Optional[bool] = None
    cfg_min_ratio: Optional[float] = None
    cfg_max_ratio: Optional[float] = None
    cfg_min_frame_threshold: Optional[int] = None
    cfg_max_frame_threshold: Optional[int] = None
    cfg_general_fixed_ratio: Optional[float] = None
    distribute_value_field_weights: Optional[Any] = None
    # --- normalizer stats (legacy top-level copies; prefer normalizer_config) ---
    customized_action_statistic_dof: Optional[str] = None
    min_key: Optional[str] = None
    delta_key: Optional[str] = None


__all__ = ["X2RobotV1DataConfig"]
