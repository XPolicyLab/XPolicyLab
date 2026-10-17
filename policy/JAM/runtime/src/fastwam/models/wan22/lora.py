"""Low-rank adapters for the Wan-style ActionDiT (test-adapted RoboTwin finetunes).

Mirrors ImageWAM's ``imagewam.models.backbones.lora``: a ``LoRALinear`` wraps a
frozen released Linear, ``lora_disabled`` exposes the exact pre-LoRA function
(used as the frozen teacher by success distillation), and
``lora_merged_state_dict`` emits plain Linear keys so a saved checkpoint keeps
the released topology and loads through the unchanged deployment path.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init_seed: int | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"`rank` must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.enabled = True
        # B=0 keeps the initial function exactly equal to the released policy;
        # A comes from a local CPU generator so runtime construction order and
        # the trainer's global seeding cannot perturb it.
        lora_a = torch.empty(self.rank, base.in_features, dtype=torch.float32)
        if init_seed is None:
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        else:
            generator = torch.Generator(device="cpu").manual_seed(int(init_seed))
            bound = 1.0 / math.sqrt(float(base.in_features))
            lora_a.uniform_(-bound, bound, generator=generator)
        self.lora_A = nn.Parameter(lora_a.to(device=base.weight.device))
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, device=base.weight.device, dtype=torch.float32)
        )
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not self.enabled:
            return base_out
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return base_out + (lora_out * self.scaling).to(dtype=base_out.dtype)

    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_B.float() @ self.lora_A.float()) * float(self.scaling)
        return (self.base.weight.detach().float() + delta).to(dtype=self.base.weight.dtype)


@dataclass
class LoRAApplyResult:
    replaced: int
    target_suffixes: tuple[str, ...]


def _get_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora_to_linear_suffixes(
    module: nn.Module,
    *,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    init_seed: int | None = None,
) -> LoRAApplyResult:
    suffixes = tuple(str(item) for item in target_suffixes)
    replaced = 0
    for name, linear in list(module.named_modules()):
        if not isinstance(linear, nn.Linear) or isinstance(linear, LoRALinear):
            continue
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        parent, child_name = _get_parent(module, name)
        if isinstance(parent, LoRALinear):
            continue
        layer_seed = None if init_seed is None else int(init_seed) + replaced
        setattr(parent, child_name, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout, init_seed=layer_seed))
        replaced += 1
    logger.info("Applied LoRA to %d Linear layers; target_suffixes=%s", replaced, suffixes)
    return LoRAApplyResult(replaced=replaced, target_suffixes=suffixes)


def lora_modules(module: nn.Module) -> dict[str, LoRALinear]:
    return {name: child for name, child in module.named_modules() if isinstance(child, LoRALinear)}


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for child in module.modules():
        if isinstance(child, LoRALinear):
            params.extend([child.lora_A, child.lora_B])
    return params


@contextmanager
def lora_disabled(module: nn.Module):
    """Temporarily expose the frozen pre-LoRA function of ``module``."""
    layers = [child for child in module.modules() if isinstance(child, LoRALinear)]
    old = [bool(child.enabled) for child in layers]
    try:
        for child in layers:
            child.enabled = False
        yield
    finally:
        for child, enabled in zip(layers, old, strict=True):
            child.enabled = enabled


def lora_merged_state_dict(module: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """State dict of ``module`` with every LoRALinear rendered as a plain Linear."""
    wrapped = lora_modules(module)
    merged: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in module.state_dict().items():
        owner = None
        for name in wrapped:
            if key == f"{name}.base.weight" or key == f"{name}.base.bias":
                owner = name
                break
            if key == f"{name}.lora_A" or key == f"{name}.lora_B":
                owner = "__drop__"
                break
        if owner == "__drop__":
            continue
        if owner is not None:
            plain = key.replace(f"{owner}.base.", f"{owner}.")
            merged[plain] = wrapped[owner].merged_weight().detach().cpu() if key.endswith(".weight") else value.detach().cpu()
        else:
            merged[key] = value.detach().cpu()
    return merged


def remap_plain_linear_keys_to_lora_base(module: nn.Module, state_dict: dict) -> dict:
    """Map plain Linear checkpoint keys onto LoRALinear ``.base`` keys."""
    names = set(lora_modules(module))
    if not names:
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        mapped = key
        for name in names:
            if key == f"{name}.weight":
                mapped = f"{name}.base.weight"
                break
            if key == f"{name}.bias":
                mapped = f"{name}.base.bias"
                break
        remapped[mapped] = value
    return remapped
