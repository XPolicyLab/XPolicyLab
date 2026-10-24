from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence as AbcSequence
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


class BaseModalityEncoder(nn.Module, ABC):
    """模态 encoder 的最小协议：输入 obs_dict，输出 {模态名: [B,L,D_native]}。"""

    def __init__(
        self,
        *,
        obs_shapes: Mapping[str, Sequence[int]] | None,
        obs_horizon: int = 1,
    ) -> None:
        super().__init__()
        self.obs_horizon = int(obs_horizon)
        if self.obs_horizon <= 0:
            raise ValueError("obs_horizon 必须为正整数")

        self._obs_shapes = _normalize_obs_shapes(obs_shapes or {})

    @property
    @abstractmethod
    def input_keys(self) -> Sequence[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_names(self) -> Sequence[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """返回该 encoder 输出 token 的原生 embedding 维度。"""
        raise NotImplementedError

    @property
    def output_dims(self) -> Dict[str, int]:
        """兼容辅助接口：把单值 embedding_dim 展开到每个输出名。"""
        return {str(name): int(self.embedding_dim) for name in self.output_names}

    @abstractmethod
    def forward(self, obs_dict: Mapping[str, torch.Tensor], training: bool = True) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def build_example_obs(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        example: Dict[str, torch.Tensor] = {}
        for key in self.input_keys:
            example[key] = self._build_missing_obs_tensor(
                key=key,
                batch_size=batch_size,
                time_steps=self.obs_horizon,
                device=device,
                dtype=dtype,
            )
        return example

    def _resolve_reference_shape(
        self,
        obs_dict: Mapping[str, torch.Tensor],
    ) -> tuple[int, int, torch.device, torch.dtype]:
        for key in self.input_keys:
            value = obs_dict.get(key)
            if value is None:
                continue
            if value.ndim < 2:
                raise ValueError(f"obs[{key}] 维度不足，期望至少 [B,T,...]")
            return int(value.shape[0]), int(value.shape[1]), value.device, value.dtype
        raise ValueError("obs_dict 缺少该 encoder 所有输入 key，无法编码。")

    def _build_missing_obs_tensor(
        self,
        *,
        key: str,
        batch_size: int,
        time_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if key not in self._obs_shapes:
            raise KeyError(f"obs_shapes 缺少 key: {key}")
        shape = tuple(int(v) for v in self._obs_shapes[key])
        tail = (1,) if len(shape) == 0 else shape
        fill_dtype = dtype if dtype.is_floating_point else torch.float32
        return torch.zeros((batch_size, time_steps) + tail, device=device, dtype=fill_dtype)



def _normalize_obs_shapes(obs_shapes: Mapping[str, Sequence[int]]) -> Dict[str, Tuple[int, ...]]:
    normalized: Dict[str, Tuple[int, ...]] = {}
    for raw_key, raw_shape in obs_shapes.items():
        key = str(raw_key).strip()
        if not key:
            continue

        if raw_shape is None:
            shape = ()
        elif isinstance(raw_shape, AbcSequence) and not isinstance(raw_shape, (str, bytes, bytearray)):
            shape = tuple(int(v) for v in raw_shape)
        else:
            shape = (int(raw_shape),)

        if any(v < 0 for v in shape):
            raise ValueError(f"obs_shapes[{key}] 不能包含负维度: {shape}")

        normalized[key] = shape
    return normalized


__all__ = ["BaseModalityEncoder"]
