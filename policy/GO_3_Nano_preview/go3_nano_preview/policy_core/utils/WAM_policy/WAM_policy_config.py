from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WAM_policyConfig:
    model: Any
    train: Any
    data: Any
    cotrain: Any
    validation: Any
    goal_mode: Any = None
    infer: Any = None
    ema: dict = field(default_factory=dict)
    dsrl: dict = field(default_factory=dict)
    vlm_rm: dict = field(default_factory=dict)
    basic_fps: int = 30


def is_nested_config(value: object) -> bool:
    cls = type(value)
    name = getattr(cls, "__name__", "")
    annotations = getattr(cls, "__annotations__", None)
    if not annotations or not name.endswith("Config"):
        return False
    if name.startswith("WAM_policy"):
        return False
    return True


def nested_config_to_dict(config_obj: object) -> dict[str, Any]:
    annotations = getattr(type(config_obj), "__annotations__", {}) or {}
    names = list(annotations)
    for key in getattr(config_obj, "__dict__", {}):
        if not key.startswith("_") and key not in names:
            names.append(key)

    out: dict[str, Any] = {}
    for name in names:
        value = getattr(config_obj, name)
        if is_nested_config(value):
            out[name] = nested_config_to_dict(value)
        elif isinstance(value, list):
            out[name] = [
                nested_config_to_dict(item)
                if is_nested_config(item)
                else deepcopy(item)
                for item in value
            ]
        else:
            out[name] = deepcopy(value)
    return out


def as_dataset_kwargs(value: Any) -> dict[str, Any]:
    if value is None:
        raise KeyError("dataset split is not set")
    if isinstance(value, dict):
        return deepcopy(value)
    if is_nested_config(value) or getattr(type(value), "__annotations__", None):
        return nested_config_to_dict(value)
    raise TypeError(f"unsupported dataset split type: {type(value)!r}")


def require_dataset_split(cfg: Any, split: str) -> Any:
    value = getattr(cfg.data, split, None)
    if value is None:
        raise KeyError(f"cfg.data.{split} is not set")
    return value


def get_video_model_path(cfg: Any) -> str:
    return str(cfg.model.video.path)


def get_video_model_kwargs(cfg: Any) -> dict[str, Any]:
    video = nested_config_to_dict(cfg.model.video)
    return {k: v for k, v in video.items() if k != "path"}


def get_action_model_path(cfg: Any) -> str:
    return str(cfg.model.action.path)


def get_action_model_kwargs(cfg: Any) -> dict[str, Any]:
    action = nested_config_to_dict(cfg.model.action)
    return {k: v for k, v in action.items() if k != "path"}
