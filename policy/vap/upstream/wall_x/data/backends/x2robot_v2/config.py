"""Typed config for the internal x2robot_dataset_v2 backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from wall_x.config.data_config import DataConfig
from wall_x.config.registry import register_data_config


@register_data_config("x2robot_v2")
@dataclass
class X2RobotV2DataConfig(DataConfig):
    """x2robot_dataset_v2 data config."""

    dataset_type: str = "x2robot_v2"
    dataset_config_path: Optional[str] = None
    # Optional explicit validation-only dataset YAML.  When provided, the
    # backend consumes every episode in this manifest as validation data
    # instead of deriving a ratio split from ``dataset_config_path``.
    val_dataset_config_path: Optional[str] = None
    obs_horizon: int = 1
    history_length: int = 0
    history_interval: int = 1
    future_length: int = 0
    future_interval: int = 1
    text_processor: Optional[Dict[str, Any]] = None
    epilogue: Optional[Dict[str, Any]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    filters: Optional[List[Dict[str, Any]]] = None
    globals: Optional[Dict[str, Any]] = None
    image_history_length: int = 1
    image_horizon: int = 1
    image_history_interval: int = 1
    obs_history_stride: int = 0
    action_history_length: int = 0
    event_level: bool = False
    use_embodied_system_prompt_ratio: float = 0.0
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: Optional[bool] = None
    in_order: bool = False
    build_val_dataset: bool = True


__all__ = ["X2RobotV2DataConfig"]
