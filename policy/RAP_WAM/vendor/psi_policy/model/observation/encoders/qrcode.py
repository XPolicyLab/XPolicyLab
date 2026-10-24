from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn

from psi_policy.model.observation.encoders.base import BaseModalityEncoder


class QRCodeEncoder(BaseModalityEncoder):
    """将二值低维信号编码为 token（embedding lookup）。"""

    def __init__(
        self,
        *,
        key: str | None = None,
        keys: Sequence[str] | None = None,
        obs_shapes: Mapping[str, Sequence[int]] | None,
        obs_horizon: int = 1,
        embedding_dim: int | None = None,
        **_: object,
    ) -> None:
        resolved_keys = _resolve_keys(key=key, keys=keys)
        self._keys = resolved_keys
        super().__init__(obs_shapes=obs_shapes, obs_horizon=obs_horizon)

        if embedding_dim is None:
            raise ValueError("QRCodeEncoder 需要显式配置 embedding_dim")
        self._embedding_dim = int(embedding_dim)
        if self._embedding_dim <= 0:
            raise ValueError("embedding_dim 必须为正整数")

        self.embedding = nn.Embedding(2, self._embedding_dim)

    @property
    def input_keys(self) -> Sequence[str]:
        return list(self._keys)

    @property
    def output_names(self) -> Sequence[str]:
        return list(self._keys)

    @property
    def embedding_dim(self) -> int:
        return int(self._embedding_dim)

    def _encode_one(self, value: torch.Tensor) -> torch.Tensor:
        flat = value.float().reshape(value.shape[0], value.shape[1], -1)
        indices = (flat.squeeze(-1) > 0).long()
        return self.embedding(indices)

    def validate_obs_contract(self, obs_dict: Mapping[str, torch.Tensor]) -> None:
        for key in self._keys:
            value = obs_dict.get(key)
            if value is None:
                raise KeyError(f"QRCodeEncoder 缺少输入 key: {key}")
            if value.ndim < 2:
                raise ValueError(f"low_dim obs[{key}] 维度不足，期望至少 [B,T,...]")
            if value.float().reshape(value.shape[0], value.shape[1], -1).shape[-1] != 1:
                raise ValueError(f"obs[{key}] 预期最后维度为 1，实际不满足标量 token 契约")

    def forward(self, obs_dict: Mapping[str, torch.Tensor], training: bool = True) -> Dict[str, torch.Tensor]:
        del training
        outputs: Dict[str, torch.Tensor] = {}
        for key in self._keys:
            outputs[key] = self._encode_one(obs_dict[key])

        return outputs



def _resolve_keys(*, key: str | None, keys: Sequence[str] | None) -> list[str]:
    candidates: list[str] = []
    if key is not None:
        candidates.append(str(key))
    if keys is not None:
        candidates.extend(str(v) for v in keys)

    cleaned: list[str] = []
    seen = set()
    for raw in candidates:
        cur = str(raw).strip()
        if not cur or cur in seen:
            continue
        seen.add(cur)
        cleaned.append(cur)

    if not cleaned:
        raise ValueError("QRCodeEncoder 至少需要一个 key")
    return cleaned


__all__ = ["QRCodeEncoder"]
