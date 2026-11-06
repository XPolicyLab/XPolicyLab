#!/usr/bin/env python

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


_ATTN_TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
_FFN_TARGET_NAMES = ("gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """Linear layer with an additive low-rank adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features, **factory_kwargs))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, **factory_kwargs))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        lora_hidden = F.linear(self.dropout(x), self.lora_A)
        lora_output = F.linear(lora_hidden, self.lora_B)
        return result + lora_output * self.scaling

    def freeze_base_parameters(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True

    def merge_into_base_(self):
        delta = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        self.weight.data.add_(delta.to(dtype=self.weight.dtype, device=self.weight.device))
        nn.init.zeros_(self.lora_B)

    def to_linear(self) -> nn.Linear:
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            delta = torch.matmul(self.lora_B, self.lora_A) * self.scaling
            merged_weight = self.weight + delta.to(dtype=self.weight.dtype, device=self.weight.device)
            linear.weight.copy_(merged_weight)
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        linear.train(self.training)
        return linear


def is_lora_parameter_name(name: str) -> bool:
    return name.endswith("lora_A") or name.endswith("lora_B")


def resolve_lora_target_linear_names(targets: Iterable[str]) -> tuple[str, ...]:
    resolved: list[str] = []
    for target in targets:
        if target == "attn":
            resolved.extend(_ATTN_TARGET_NAMES)
        elif target == "ffn":
            resolved.extend(_FFN_TARGET_NAMES)
        else:
            raise ValueError(f"Unsupported LoRA target group: {target}")
    return tuple(dict.fromkeys(resolved))


def apply_lora_to_linear_modules(
    module: nn.Module,
    *,
    target_names: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    num_replaced = 0
    target_name_set = set(target_names)
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, nn.Linear) and child_name in target_name_set:
            setattr(
                module,
                child_name,
                LoRALinear.from_linear(
                    child_module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
            num_replaced += 1
            continue

        num_replaced += apply_lora_to_linear_modules(
            child_module,
            target_names=target_names,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
    return num_replaced


def freeze_module_except_lora(module: nn.Module):
    for name, param in module.named_parameters():
        param.requires_grad = is_lora_parameter_name(name)


def merge_lora_weights_(module: nn.Module):
    for child_module in module.modules():
        if isinstance(child_module, LoRALinear):
            child_module.merge_into_base_()


def merge_and_unload_lora_modules_(module: nn.Module) -> int:
    num_replaced = 0
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, LoRALinear):
            setattr(module, child_name, child_module.to_linear())
            num_replaced += 1
            continue
        num_replaced += merge_and_unload_lora_modules_(child_module)
    return num_replaced
