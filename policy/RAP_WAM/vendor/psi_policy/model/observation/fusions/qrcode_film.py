from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn

from psi_policy.model.observation.fusions.base import BaseFusion


class FilmAllFusion(BaseFusion):
    """使用 conditioner 模态对 targets 模态做 FiLM 调制。"""

    def __init__(
        self,
        *,
        conditioner: str,
        targets: Sequence[str],
        hidden_dim: int | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.conditioner = str(conditioner)
        self.targets = [str(v) for v in targets]
        if not self.conditioner:
            raise ValueError("conditioner 不能为空")
        if not self.targets:
            raise ValueError("targets 不能为空")

        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else 0
        if self.hidden_dim < 0:
            raise ValueError("hidden_dim 不能为负数")

        self._mlps = nn.ModuleDict()

    def _get_mlp(self, *, target_name: str, cond_dim: int, target_dim: int) -> nn.Module:
        key = target_name
        if key not in self._mlps:
            hidden = self.hidden_dim if self.hidden_dim > 0 else cond_dim
            self._mlps[key] = nn.Sequential(
                nn.Linear(cond_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, target_dim * 2),
            )
        return self._mlps[key]

    def _align_conditioner(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cond_len = int(cond.shape[1])
        target_len = int(target.shape[1])
        if cond_len == target_len:
            return cond
        if cond_len == 1:
            return cond.expand(-1, target_len, -1)
        if target_len % cond_len == 0:
            repeat_factor = target_len // cond_len
            return cond.repeat_interleave(repeat_factor, dim=1)

        raise ValueError(
            "FiLM conditioner 长度无法对齐目标长度: "
            f"conditioner_len={cond_len}, target_len={target_len}"
        )

    def validate_token_map(self, token_map: Mapping[str, torch.Tensor]) -> None:
        if self.conditioner not in token_map:
            raise KeyError(f"FiLM conditioner 不存在: {self.conditioner}")

        cond = token_map[self.conditioner]
        if cond.ndim != 3:
            raise ValueError(f"conditioner token 必须为 3 维 [B,L,D]，实际 {tuple(cond.shape)}")

        for target in self.targets:
            if target not in token_map:
                raise KeyError(f"FiLM target 不存在: {target}")
            target_tokens = token_map[target]
            if target_tokens.ndim != 3:
                raise ValueError(f"target token 必须为 3 维 [B,L,D]，实际 {tuple(target_tokens.shape)}")
            self._align_conditioner(cond=cond, target=target_tokens)

    def forward(
        self,
        *,
        token_map: Mapping[str, torch.Tensor],
        obs_dict: Mapping[str, torch.Tensor],
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        del obs_dict, training
        cond = token_map[self.conditioner]
        fused = {k: v for k, v in token_map.items()}
        for target in self.targets:
            target_tokens = fused[target]
            aligned_cond = self._align_conditioner(cond=cond, target=target_tokens)
            mlp = self._get_mlp(
                target_name=target,
                cond_dim=int(aligned_cond.shape[-1]),
                target_dim=int(target_tokens.shape[-1]),
            )
            film_params = mlp(aligned_cond)
            gamma, beta = film_params.chunk(2, dim=-1)
            fused[target] = gamma * target_tokens + beta

        return fused


__all__ = ["FilmAllFusion"]
