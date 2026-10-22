"""Expand only the official Wan48 patch/head to the agreed channel order."""

import math

import torch


def expand_wan48_state_dict(state: dict, model) -> dict:
    expected = model.state_dict()
    patch = state["patch_embedding.weight"]
    volume = math.prod(model.patch_size)
    if patch.shape[1] != 48 or model.patch_embedding.in_channels != 96:
        raise ValueError("Joint96 initialization requires an official 48-channel checkpoint and 96-channel model")
    output = dict(state)
    expanded = patch.new_zeros(expected["patch_embedding.weight"].shape)
    expanded[:, :48] = patch
    output["patch_embedding.weight"] = expanded
    for name in ("head.head.weight", "head.head.bias"):
        if name not in state:
            if name in expected:
                raise ValueError(f"Official checkpoint missing {name}")
            continue
        old = state[name]
        if old.shape[0] != volume * 48 or expected[name].shape[0] != volume * 96:
            raise ValueError(f"Unexpected Wan48 head layout: {name}")
        old = old.reshape(volume, 48, *old.shape[1:])
        new = old.new_zeros((volume, 96, *old.shape[2:]))
        new[:, :48] = old
        output[name] = new.reshape(expected[name].shape)
    missing = set(expected) - set(output)
    extra = set(output) - set(expected)
    incompatible = [name for name in set(output) & set(expected) if output[name].shape != expected[name].shape]
    if missing or extra or incompatible:
        raise ValueError(f"Joint96 initialization is not strict: missing={sorted(missing)}, extra={sorted(extra)}, incompatible={incompatible}")
    return output
