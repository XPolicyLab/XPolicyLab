"""Robot state/action conversion between the simulator vectors and the model vectors.

The checkpoint's ``inference_config.yaml`` (``processor``) fixes every parameter; the training-time pipeline it
reproduces is, per field::

    proprio:  raw slice -> x * scale + offset, clamp [-5, 5] -> zero-pad to the model width
    action:   model slice -> crop to the raw width -> (y - offset[t]) / scale[t] -> + proprio  (relative fields)

``scale``/``offset`` are the q01/q99 linear normalization (per action step for actions, global for proprio).
Model-only fields (``completed``) are zeros on input and dropped on output.
"""
from typing import Any

import numpy as np
import torch


class RobotProcessor:
    def __init__(self, cfg: dict[str, Any]) -> None:
        def tensors(section):
            return {key: (torch.tensor(spec["scale"], dtype=torch.float32),
                          torch.tensor(spec["offset"], dtype=torch.float32))
                    for key, spec in cfg["normalization"][section].items()}

        self.proprio_fields = cfg["proprio_fields"]  # raw order: {key, start, width, model_width}
        self.action_fields = cfg["action_fields"]
        self.completed = cfg["completed"]  # model-only key -> width, zeros on input
        self.model_proprio_keys = cfg["model_proprio_keys"]
        self.model_action_keys = cfg["model_action_keys"]
        self.relative_keys = cfg["relative_keys"]  # action fields stored relative to the current proprio
        self.proprio_norm = tensors("proprio")
        self.action_norm = tensors("action")  # scale/offset: [action_horizon, width]
        self.gripper_full_scale = float(cfg["gripper_full_scale"])
        widths = {field["key"]: field["model_width"] for field in self.action_fields}
        self.action_dim_is_pad = torch.cat([
            torch.arange(widths[key]) >= next(f["width"] for f in self.action_fields if f["key"] == key)
            if key in widths else torch.ones(self.completed[key], dtype=torch.bool)
            for key in self.model_action_keys
        ])

    @staticmethod
    def _normalize(x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return (x * scale + offset).clamp(-5.0, 5.0).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    def _split(self, vector: torch.Tensor, keys: list[str], fields: list[dict]) -> dict[str, torch.Tensor]:
        """Model vector -> raw-width fields (model padding and completed fields dropped)."""
        widths = {field["key"]: (field["model_width"], field["width"]) for field in fields}
        result, offset = {}, 0
        for key in keys:
            model_width, width = widths[key] if key in widths else (self.completed[key], None)
            if width is not None:
                result[key] = vector[..., offset: offset + width]
            offset += model_width
        if offset != vector.shape[-1]:
            raise ValueError(f"Model vector width {vector.shape[-1]} != configured width {offset}")
        return result

    def preprocess_proprio(self, vector: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw simulator state [D] -> (normalized model proprio [1, D_model], action padding mask [D_model])."""
        values = torch.as_tensor(vector, dtype=torch.float32).unsqueeze(0)
        fields = {}
        for field in self.proprio_fields:
            x = values[..., field["start"]: field["start"] + field["width"]]
            x = self._normalize(x, *self.proprio_norm[field["key"]])
            fields[field["key"]] = torch.nn.functional.pad(x, (0, field["model_width"] - field["width"]))
        proprio = torch.cat([
            fields[key] if key in fields else torch.zeros(1, self.completed[key], dtype=torch.float32)
            for key in self.model_proprio_keys
        ], dim=-1)
        return proprio, self.action_dim_is_pad

    def postprocess_action(self, action: torch.Tensor, proprio: torch.Tensor) -> np.ndarray:
        """Model action [T, D_model] (or [1, T, D_model]) and its input proprio [1, D_model] -> raw actions [T, D]."""
        if action.ndim == 3:
            (action,) = action  # one episode per policy
        action = action.to(dtype=torch.float32, device="cpu")
        proprio = proprio.to(dtype=torch.float32, device="cpu")
        action_fields = self._split(action, self.model_action_keys, self.action_fields)
        proprio_fields = self._split(proprio, self.model_proprio_keys, self.proprio_fields)
        horizon = action.shape[-2]
        raw = []
        for field in self.action_fields:
            key = field["key"]
            scale, offset = self.action_norm[key]
            value = (action_fields[key] - offset[:horizon]) / scale[:horizon]
            if key in self.relative_keys:
                proprio_scale, proprio_offset = self.proprio_norm[key]
                value = value + (proprio_fields[key][..., 0:1, :] - proprio_offset) / proprio_scale
            raw.append(value)
        return torch.cat(raw, dim=-1).numpy()
