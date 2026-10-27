from __future__ import annotations


import torch
import torch.nn as nn

from awomo.awomo.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapper for an existing Linear layer."""

def remap_plain_linear_keys_to_lora_base(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    lora_names = {name for name, child in module.named_modules() if isinstance(child, LoRALinear)}
    if not lora_names:
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        mapped_key = key
        for name in lora_names:
            if key == f"{name}.weight":
                mapped_key = f"{name}.base.weight"
                break
            if key == f"{name}.bias":
                mapped_key = f"{name}.base.bias"
                break
        remapped[mapped_key] = value
    return remapped


def merge_lora_state_dict_to_plain(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = set()
    for key in state_dict:
        if key.endswith(".lora_A"):
            prefixes.add(key[: -len(".lora_A")])
    if not prefixes:
        return state_dict

    merged: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for prefix in prefixes:
        base_key = f"{prefix}.base.weight"
        a_key = f"{prefix}.lora_A"
        b_key = f"{prefix}.lora_B"
        if base_key not in state_dict or a_key not in state_dict or b_key not in state_dict:
            continue
        base = state_dict[base_key]
        lora_a = state_dict[a_key]
        lora_b = state_dict[b_key]
        delta = lora_b.float() @ lora_a.float()
        merged[f"{prefix}.weight"] = (base.float() + delta).to(dtype=base.dtype)
        consumed.update({base_key, a_key, b_key})
        bias_key = f"{prefix}.base.bias"
        if bias_key in state_dict:
            merged[f"{prefix}.bias"] = state_dict[bias_key]
            consumed.add(bias_key)

    for key, value in state_dict.items():
        if key in consumed:
            continue
        if ".lora_A" in key or ".lora_B" in key or ".base." in key:
            continue
        merged[key] = value
    return merged
