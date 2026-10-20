"""Resolve the release's local assets without downloading or external fallbacks."""
from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ASSET_PATHS = {
    "text_encoder": "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
    "vae": "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
    "tokenizer": "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
    "dino": "dinov3_vitb16_timm/model.safetensors",
}


def bundle_root() -> Path:
    """Resolve all model assets relative to this standalone source release."""
    return Path(os.environ.get("CRWM_RELEASE_ROOT") or PACKAGE_ROOT).expanduser().resolve()


def pretrained_root() -> Path:
    return bundle_root() / "weights" / "third_party"


def asset_path(name: str) -> Path:
    path = pretrained_root() / ASSET_PATHS[name]
    present = path.is_dir() if name == "tokenizer" else path.is_file()
    if not present:
        raise FileNotFoundError(f"CR-WM release asset missing: {path}. Restore the complete release; inference is offline.")
    if name == "tokenizer":
        if not (path / "tokenizer_config.json").is_file() or not any(
            (path / filename).is_file() for filename in ("tokenizer.json", "spiece.model")
        ):
            raise FileNotFoundError(f"CR-WM tokenizer is incomplete: {path}. Restore tokenizer_config.json and tokenizer.json/spiece.model.")
    return path


def validate_assets() -> dict[str, Path]:
    """Check the full local dependency set before GPU model allocation."""
    return {name: asset_path(name) for name in ASSET_PATHS}
