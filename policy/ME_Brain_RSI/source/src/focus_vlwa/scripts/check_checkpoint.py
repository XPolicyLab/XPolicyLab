"""Validate released checkpoint keys and shapes without allocating model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from focus_vlwa.configs.model import load_model_config
from focus_vlwa.model.checkpoint import _target_key
from focus_vlwa.model.focus_vlwa import FocusVLWA


def validate_checkpoint(checkpoint: str | Path) -> tuple[list[str], list[str], list[str]]:
    """Return missing keys, unexpected keys, and shape mismatches."""
    path = Path(checkpoint)
    config = load_model_config(path if path.is_dir() else path.parent)
    if path.is_dir():
        path = path / "model.safetensors"
    with torch.device("meta"):
        expected = FocusVLWA(config).state_dict()

    loaded: set[str] = set()
    unexpected: list[str] = []
    shape_mismatches: list[str] = []
    with safe_open(path, framework="pt") as handle:
        aliases = handle.metadata() or {}
        for source_key in handle.keys():
            target_key = _target_key(source_key)
            if target_key not in expected:
                unexpected.append(source_key)
                continue
            if target_key in loaded:
                unexpected.append(f"{source_key} (duplicate target: {target_key})")
                continue
            source_shape = tuple(handle.get_slice(source_key).get_shape())
            target_shape = tuple(expected[target_key].shape)
            if source_shape != target_shape:
                shape_mismatches.append(f"{source_key}: {source_shape} != {target_shape}")
            loaded.add(target_key)
            for alias_key, stored_key in aliases.items():
                if stored_key != source_key:
                    continue
                alias_target_key = _target_key(alias_key)
                if alias_target_key not in expected or alias_target_key in loaded:
                    continue
                alias_shape = tuple(expected[alias_target_key].shape)
                if source_shape != alias_shape:
                    shape_mismatches.append(f"{alias_key}: {source_shape} != {alias_shape}")
                loaded.add(alias_target_key)

    missing = [key for key in expected if key not in loaded]
    tied_embedding = "joint_experts.vision_language_model.model.language_model.embed_tokens.weight"
    tied_head = "joint_experts.vision_language_model.lm_head.weight"
    if tied_embedding in missing and tied_head in loaded:
        missing.remove(tied_embedding)
    return missing, unexpected, shape_mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    missing, unexpected, shape_mismatches = validate_checkpoint(args.checkpoint)
    if missing or unexpected or shape_mismatches:
        raise SystemExit(
            "Checkpoint validation failed:\n"
            f"missing={missing[:20]}\n"
            f"unexpected={unexpected[:20]}\n"
            f"shape_mismatches={shape_mismatches[:20]}"
        )
    print(f"Checkpoint is compatible: {args.checkpoint}")


if __name__ == "__main__":
    main()
