"""Typed contracts shared across model, training, and inference."""

from typing import Any, TypeAlias, TypedDict

import torch


class ConditionKVLayer(TypedDict, total=False):
    k: torch.Tensor
    v: torch.Tensor
    mask: torch.Tensor
    _video_condition_len: int


ConditionKVCache: TypeAlias = list[ConditionKVLayer]
TensorTree: TypeAlias = dict[str, Any]


class WAMOutput(TypedDict, total=False):
    action: torch.Tensor
    video: Any
