"""Minimal evaluation-only state/action preprocessing for RoboDojo WAM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _tensor_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        try:
            array = np.asarray(value)
            if array.dtype.kind in "iufb":
                return torch.as_tensor(array, dtype=torch.float32)
        except (TypeError, ValueError):
            pass
        return [_tensor_tree(item) for item in value]
    return value


def load_dataset_stats_from_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return _tensor_tree(json.load(stream))


class ZScoreNormalizer:
    """Checkpoint-faithful global mean/std normalizer."""

    def __init__(self, shape_meta: dict[str, Any], stats: dict[str, Any]) -> None:
        self.fields: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {
            "action": {},
            "state": {},
        }
        for kind in self.fields:
            for meta in shape_meta[kind]:
                key = meta["key"]
                values = stats[kind][key]
                self.fields[kind][key] = (
                    torch.as_tensor(values["global_mean"], dtype=torch.float32),
                    torch.as_tensor(values["global_std"], dtype=torch.float32),
                )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        for kind in ("action", "state"):
            if kind not in batch:
                continue
            for key, (mean, std) in self.fields[kind].items():
                value = batch[kind][key]
                batch[kind][key] = torch.clamp(
                    (value - mean.to(value.device)) / (std.to(value.device) + 1e-8),
                    -5.0,
                    5.0,
                )
        return batch

    def backward(self, batch: dict[str, Any]) -> dict[str, Any]:
        for kind in ("action", "state"):
            if kind not in batch:
                continue
            for key, (mean, std) in self.fields[kind].items():
                value = batch[kind][key]
                batch[kind][key] = value * (std.to(value.device) + 1e-8) + mean.to(
                    value.device
                )
        return batch


class ConcatLeftAlign:
    def __init__(self, shape_meta: dict[str, Any], cfg: dict[str, Any]) -> None:
        self.shape_meta = shape_meta
        self.action_target_dim = int(cfg["action_target_dim"])
        self.state_target_dim = int(cfg["state_target_dim"])
        self.action_target_slices = list(cfg["action_target_slices"])
        self.state_target_slices = list(cfg["state_target_slices"])

    @staticmethod
    def _concat(values: dict[str, torch.Tensor], meta: list[dict[str, Any]]) -> torch.Tensor:
        result = torch.cat([values[item["key"]] for item in meta], dim=-1)
        if result.ndim != 2:
            raise ValueError(f"Expected [T,D] merged value, got {tuple(result.shape)}")
        return result

    @staticmethod
    def _scatter(
        value: torch.Tensor, target_dim: int, slices: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = value.new_zeros((value.shape[0], target_dim))
        is_pad = torch.ones(target_dim, dtype=torch.bool, device=value.device)
        for entry in slices:
            source_start, source_end = map(int, entry["source_slice"])
            target_start, target_end = map(int, entry["target_slice"])
            output[:, target_start:target_end] = value[:, source_start:source_end]
            is_pad[target_start:target_end] = False
        return output, is_pad

    @staticmethod
    def _gather(
        value: torch.Tensor, slices: list[dict[str, Any]]
    ) -> torch.Tensor:
        source_dim = max(int(entry["source_slice"][1]) for entry in slices)
        output = value.new_zeros((*value.shape[:-1], source_dim))
        for entry in slices:
            source_start, source_end = map(int, entry["source_slice"])
            target_start, target_end = map(int, entry["target_slice"])
            output[..., source_start:source_end] = value[..., target_start:target_end]
        return output

    @staticmethod
    def _split(
        value: torch.Tensor, meta: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        offset = 0
        for item in meta:
            width = int(item["shape"])
            output[item["key"]] = value[..., offset : offset + width]
            offset += width
        return output

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "action" in batch:
            action = self._concat(batch["action"], self.shape_meta["action"])
            batch["action"], batch["action_dim_is_pad"] = self._scatter(
                action, self.action_target_dim, self.action_target_slices
            )
        state = self._concat(batch["state"], self.shape_meta["state"])
        batch["state"], batch["state_dim_is_pad"] = self._scatter(
            state, self.state_target_dim, self.state_target_slices
        )
        return batch

    def backward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "state" in batch:
            state = self._gather(batch["state"], self.state_target_slices)
            batch["state"] = self._split(state, self.shape_meta["state"])
        action = self._gather(batch["action"], self.action_target_slices)
        batch["action"] = self._split(action, self.shape_meta["action"])
        return batch


@dataclass
class RuntimeProcessor:
    shape_meta: dict[str, Any]
    action_state_merger: ConcatLeftAlign
    _normalizer: ZScoreNormalizer | None = None

    def eval(self) -> "RuntimeProcessor":
        return self

    @property
    def normalizer(self) -> ZScoreNormalizer:
        if self._normalizer is None:
            raise RuntimeError("Dataset statistics have not been loaded")
        return self._normalizer

    def set_normalizer_from_stats(self, stats: dict[str, Any]) -> None:
        self._normalizer = ZScoreNormalizer(self.shape_meta, stats)

    def action_state_transform(self, batch: dict[str, Any]) -> dict[str, Any]:
        for kind in ("action", "state"):
            if kind not in batch:
                continue
            for meta in self.shape_meta[kind]:
                width = int(batch[kind][meta["key"]].shape[-1])
                if width != int(meta["raw_shape"]):
                    raise ValueError(
                        f"{kind}.{meta['key']} width {width} does not match "
                        f"expected {meta['raw_shape']}"
                    )
        return batch


def build_runtime_processor(cfg: Any) -> RuntimeProcessor:
    if hasattr(cfg, "items"):
        cfg = {key: value for key, value in cfg.items()}
    shape_meta = cfg["shape_meta"]
    if hasattr(shape_meta, "items"):
        shape_meta = {
            key: [dict(item) for item in value] for key, value in shape_meta.items()
        }
    merger_cfg = cfg["action_state_merger"]
    if hasattr(merger_cfg, "items"):
        merger_cfg = {key: value for key, value in merger_cfg.items()}
    merger_cfg.pop("_target_", None)
    return RuntimeProcessor(
        shape_meta=shape_meta,
        action_state_merger=ConcatLeftAlign(shape_meta, merger_cfg),
    ).eval()
