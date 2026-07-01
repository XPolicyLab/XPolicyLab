#!/usr/bin/env python3
"""Build .merged_ckpt (base vae/tokenizer/text_encoder + finetuned transformer)."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def resolve_paths(checkpoint_path: str, base_model_path: str) -> tuple[Path, Path]:
    ckpt_root = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_root}")

    base_root = Path(base_model_path).expanduser().resolve()
    if not (base_root / "vae").is_dir():
        raise FileNotFoundError(f"Base model directory missing vae/: {base_root}")

    for transformer_path in (
        ckpt_root / "checkpoints" / "transformer",
        ckpt_root / "transformer",
    ):
        if (transformer_path / "config.json").exists():
            return base_root, transformer_path

    if (ckpt_root / "transformer" / "config.json").exists():
        return ckpt_root, ckpt_root / "transformer"

    raise FileNotFoundError(
        f"Transformer checkpoint not found under {ckpt_root}. "
        "Expected checkpoints/transformer/ or transformer/."
    )


def build_merged_ckpt(
    checkpoint_path: str,
    base_model_path: str,
    merged_dir: str | Path,
) -> Path:
    base_root, transformer_path = resolve_paths(checkpoint_path, base_model_path)
    merged = Path(merged_dir).expanduser().resolve()

    if merged.is_symlink() or merged.exists():
        if merged.is_dir() and not merged.is_symlink():
            shutil.rmtree(merged)
        else:
            merged.unlink()
    merged.mkdir(parents=True, exist_ok=True)

    for sub in ("vae", "text_encoder", "tokenizer"):
        src = base_root / sub
        if not src.exists():
            raise FileNotFoundError(f"Base model missing {sub}/: {src}")
        os.symlink(src, merged / sub)
    os.symlink(transformer_path.resolve(), merged / "transformer")

    print(f"[merged_ckpt] base={base_root}")
    print(f"[merged_ckpt] transformer={transformer_path}")
    print(f"[merged_ckpt] merged={merged}")
    return merged


def main() -> None:
    policy_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        default="/mnt/nas/final_ckpt/Lingbot_VA/checkpoint_step_5000",
    )
    parser.add_argument(
        "--base-model-path",
        default="/mnt/nas/shared_model_weights/lingbot-va-base",
    )
    parser.add_argument(
        "--merged-dir",
        default=str(policy_root / ".merged_ckpt"),
    )
    args = parser.parse_args()
    build_merged_ckpt(args.checkpoint_path, args.base_model_path, args.merged_dir)


if __name__ == "__main__":
    main()
