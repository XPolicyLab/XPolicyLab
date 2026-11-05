"""Opt-in language LoRA; keep XVLA's native module and checkpoint namespace."""
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    """W x + (alpha/r) B A x; keep native weight/bias checkpoint names."""

    def __init__(self, base, spec):
        # Reuse the existing base weights without allocating/reinitializing W.
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.scaling = spec["alpha"] / spec["rank"]
        self.lora_dropout = nn.Dropout(spec["dropout"])
        self.lora_A = nn.Parameter(base.weight.new_empty(spec["rank"], base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, spec["rank"]))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value):
        result = F.linear(value, self.weight, self.bias)
        update = F.linear(F.linear(self.lora_dropout(value), self.lora_A), self.lora_B)
        return result + update * self.scaling


def resolve_spec(spec):
    if spec is None:
        return None
    spec = dict(spec)
    if set(spec) - {"rank", "alpha", "dropout"}:
        raise ValueError("Unknown VLM LoRA fields")
    rank = spec.get("rank", 16)
    alpha = float(spec.get("alpha", 32))
    dropout = float(spec.get("dropout", 0.0))
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
        raise ValueError("VLM LoRA rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("VLM LoRA alpha must be finite and positive")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("VLM LoRA dropout must be in [0, 1)")
    return dict(rank=int(rank), alpha=alpha, dropout=dropout)


def install_vlm_lora(model, spec):
    """Freeze VLM, adapt language-layer Linear modules; leave expert untouched.

    Install after vanilla weight loading for a new experiment, or inside XVLA
    construction when restoring a complete LoRA checkpoint. No wrapper
    replaces the VLM, preserving Qwen hooks, language/vision access and APIs.
    """
    spec = resolve_spec(spec)
    if spec is None:
        return
    if model.config.vlm_backend != "qwen3_5":
        raise ValueError("VLM LoRA currently supports Qwen3.5 only")
    if any(isinstance(module, LoRALinear) for module in model.vlm.modules()):
        if resolve_spec(getattr(model.config, "vlm_lora", None)) != spec:
            raise ValueError("Cannot change LoRA settings on an existing adapter")
        # Transformers may replace Parameter objects while loading shards.
        # requires_grad is not serialized, so enforce it after loading too.
        for name, parameter in model.vlm.named_parameters():
            parameter.requires_grad_("lora_" in name)
        return
    targets = [name for name, module in model.vlm.named_modules()
               if name.startswith("model.language_model.layers.")
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError("No Qwen language Linear modules found for VLM LoRA")
    model.vlm.requires_grad_(False)
    for name in targets:
        parent_name, leaf = name.rsplit(".", 1)
        parent = model.vlm.get_submodule(parent_name)
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), spec))
    # Save full base + adapters in XVLA's normal checkpoint; no external
    # adapter package or base checkpoint is needed for inference.
    model.config.vlm_lora = spec
