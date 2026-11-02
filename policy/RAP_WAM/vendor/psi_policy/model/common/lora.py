from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def recursive_getattr(model: torch.nn.Module, module_name: str):
    output = model
    for name in str(module_name).split("."):
        output = getattr(output, name)
    return output


def recursive_setattr(model: torch.nn.Module, module_name: str, module: torch.nn.Module) -> None:
    output = model
    split_names = str(module_name).split(".")
    for name in split_names[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_names[-1], module)


class LoRALinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        lora_dim: int,
        lora_scaling: float,
    ) -> None:
        super().__init__()
        rows, columns = weight.shape
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
        self.lora_right_weight = torch.nn.Parameter(torch.zeros((columns, lora_dim)))
        self.lora_left_weight = torch.nn.Parameter(torch.zeros((lora_dim, rows)))
        self.lora_scaling = float(lora_scaling) / int(lora_dim)
        self._init_parameters()

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def _init_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_left_weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_input = input.to(dtype=self.weight.dtype)
        base_output = F.linear(base_input, self.weight, self.bias)

        lora_input = input.to(dtype=self.lora_right_weight.dtype)
        lora_output = (
            lora_input @ self.lora_right_weight @ self.lora_left_weight
        ) * self.lora_scaling
        return base_output + lora_output.to(dtype=base_output.dtype)


def convert_linear_layer_to_lora(
    model: torch.nn.Module,
    include_module_names: list[str],
    exclude_module_names: list[str],
    lora_dim: int = 0,
    lora_scaling: float = 1.0,
) -> torch.nn.Module:
    replace_names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(include_name in name for include_name in include_module_names):
            continue
        if any(exclude_name in name for exclude_name in exclude_module_names):
            continue
        replace_names.append(name)

    for name in replace_names:
        module = recursive_getattr(model, name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"Expected torch.nn.Linear at {name}, got {type(module)}")
        lora_module = LoRALinear(
            module.weight,
            module.bias,
            lora_dim=int(lora_dim),
            lora_scaling=float(lora_scaling),
        ).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, lora_module)
    return model


__all__ = [
    "LoRALinear",
    "convert_linear_layer_to_lora",
    "recursive_getattr",
    "recursive_setattr",
]
