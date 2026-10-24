from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from psi_policy.model.observation.concat.base import BaseConcatStrategy


class TokenLevelConcat(BaseConcatStrategy):
    """按配置顺序做 token 维拼接（dim=1），并在出口统一 embedding 维度。"""

    def __init__(
        self,
        *,
        order: Sequence[str] | None = None,
        output_dim: int | None = None,
        input_dims: Mapping[str, int] | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.order = _normalize_order(order)
        self._configured_input_dims: Dict[str, int] = {}
        self._output_dim = int(output_dim) if output_dim is not None else 0
        if self._output_dim < 0:
            raise ValueError("output_dim 不能为负数")
        self.adapters = nn.ModuleDict()
        if input_dims is None:
            raise ValueError("TokenLevelConcat 构造时必须显式提供 input_dims")
        self.bind_input_dims(input_dims)

    @property
    def output_dim(self) -> int:
        if self._output_dim <= 0:
            raise RuntimeError("TokenLevelConcat 尚未确定 output_dim")
        return int(self._output_dim)

    def bind_input_dims(self, input_dims: Mapping[str, int]) -> None:
        normalized = _normalize_input_dims(input_dims)
        if not normalized:
            raise ValueError("TokenLevelConcat 需要非空 input_dims")

        if self._configured_input_dims:
            if normalized != self._configured_input_dims:
                raise ValueError(
                    "TokenLevelConcat 重复绑定了不一致的 input_dims: "
                    f"current={self._configured_input_dims}, new={normalized}"
                )
            return

        self.order = _resolve_order(configured=self.order, input_dims=normalized)
        target_dim = _resolve_output_dim(configured=self._output_dim, input_dims=normalized)
        modules: dict[str, nn.Module] = {}
        for name in self.order:
            dim = normalized[name]
            modules[name] = nn.Identity() if dim == target_dim else nn.Linear(dim, target_dim)

        self._configured_input_dims = normalized
        self._output_dim = target_dim
        self.adapters = nn.ModuleDict(modules)

    def forward(
        self,
        *,
        token_map: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, list[int]]:
        tensors: list[torch.Tensor] = []
        for name in self.order:
            if name not in token_map:
                raise KeyError(f"TokenLevelConcat 缺少输入 token: {name}")
            tensors.append(self.adapters[name](token_map[name]))

        part_lengths = [int(tensor.shape[1]) for tensor in tensors]
        return torch.cat(tensors, dim=1), part_lengths


def _normalize_input_dims(input_dims: Mapping[str, int]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for raw_name, raw_dim in input_dims.items():
        name = str(raw_name).strip()
        if not name:
            continue
        dim = int(raw_dim)
        if dim <= 0:
            raise ValueError(f"input_dims[{name}] 必须为正整数，实际 {dim}")
        normalized[name] = dim
    return normalized


def _normalize_order(order: Sequence[str] | None) -> list[str]:
    if order is None:
        raise ValueError("TokenLevelConcat.order 必须显式配置")

    normalized: list[str] = []
    seen = set()
    for raw_name in order:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("TokenLevelConcat.order 不能包含空字符串")
        if name in seen:
            raise ValueError(f"TokenLevelConcat.order 不能包含重复 token: {name}")
        seen.add(name)
        normalized.append(name)
    if not normalized:
        raise ValueError("TokenLevelConcat.order 不能为空")
    return normalized


def _resolve_order(*, configured: Sequence[str], input_dims: Mapping[str, int]) -> list[str]:
    configured_list = [str(name) for name in configured]
    available = list(input_dims.keys())
    missing = [name for name in available if name not in configured_list]
    unexpected = [name for name in configured_list if name not in input_dims]
    if missing or unexpected:
        raise ValueError(
            "TokenLevelConcat.order 必须与 input_dims 精确匹配: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return configured_list


def _resolve_output_dim(*, configured: int, input_dims: Mapping[str, int]) -> int:
    if configured > 0:
        return int(configured)

    unique_dims = sorted(set(int(v) for v in input_dims.values()))
    if len(unique_dims) == 1:
        return int(unique_dims[0])

    raise ValueError(
        "TokenLevelConcat 无法推导 output_dim：输入 token 维度不一致。"
        f"请显式配置 concat.output_dim，当前 input_dims={dict(input_dims)}"
    )


__all__ = ["TokenLevelConcat"]
