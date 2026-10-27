"""Factories for the three pinned pretrained model implementations."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from starVLA.model.modules.external_sources import SourceContract, import_external_package


def _checkpoint(path: str | Path) -> Any:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _plain_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return value
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")


def _select_checkpoint_state(
    state: dict[str, torch.Tensor],
    checkpoint_prefix: str | None,
) -> dict[str, torch.Tensor]:
    if checkpoint_prefix is None:
        return state
    selected = {
        key.removeprefix(checkpoint_prefix): value for key, value in state.items() if key.startswith(checkpoint_prefix)
    }
    if not selected:
        raise RuntimeError(f"Checkpoint does not contain state under prefix {checkpoint_prefix!r}")
    return selected


def build_dinov3_large(
    source: SourceContract,
    checkpoint_path: str | Path,
    checkpoint_prefix: str | None = None,
) -> nn.Module:
    import_external_package(source)
    backbone_module = importlib.import_module("dinov3.hub.backbones")
    model = backbone_module.dinov3_vitl16(pretrained=False)
    state = _select_checkpoint_state(_plain_state(_checkpoint(checkpoint_path)), checkpoint_prefix)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected DINOv3 checkpoint mismatch: {incompatible}")
    return model


def build_vggt_omega(source: SourceContract, checkpoint_path: str | Path) -> nn.Module:
    import_external_package(source)
    model_class = importlib.import_module("vggt_omega.models.vggt_omega").VGGTOmega
    model = model_class(enable_camera=True, enable_depth=True, enable_alignment=False)
    incompatible = model.load_state_dict(_plain_state(_checkpoint(checkpoint_path)), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected VGGT-Omega checkpoint mismatch: {incompatible}")
    model.requires_grad_(False).eval()
    return model


def _clean_vjepa_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("backbone.")
        cleaned[key] = value
    return cleaned


def build_vjepa2_1_large(source: SourceContract, checkpoint_path: str | Path) -> nn.Module:
    import_external_package(source)
    encoder_module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
    encoder = encoder_module.vit_large(
        img_size=(256, 256),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
        n_output_distillation=1,
    )
    payload = _checkpoint(checkpoint_path)
    if not isinstance(payload, dict) or "ema_encoder" not in payload:
        raise RuntimeError("V-JEPA 2.1 checkpoint must contain `ema_encoder`")
    state = _clean_vjepa_state(payload["ema_encoder"])
    incompatible = encoder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected V-JEPA 2.1 checkpoint mismatch: {incompatible}")
    encoder.requires_grad_(False).eval()
    return encoder
