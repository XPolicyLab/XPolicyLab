#!/usr/bin/env python3
"""Build .merged_ckpt (base vae/tokenizer/text_encoder + finetuned transformer)."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _find_transformer(ckpt_root: Path) -> Path | None:
    """Locate a transformer/ dir under ``ckpt_root``.

    Supports several layouts (checked in order):
      * ``<root>/transformer``                          (already a step / merged dir)
      * ``<root>/checkpoints/transformer``
      * ``<root>/checkpoint_step_<N>/transformer``      (train.py layout, run root passed)
      * ``<root>/checkpoints/checkpoint_step_<N>/transformer``

    For the ``checkpoint_step_<N>`` layouts the largest ``N`` is chosen unless the
    ``LINGBOT_VA_STEP`` env var pins a specific step.
    """
    for transformer_path in (
        ckpt_root / "transformer",
        ckpt_root / "checkpoints" / "transformer",
    ):
        if (transformer_path / "config.json").exists():
            return transformer_path

    pinned = os.environ.get("LINGBOT_VA_STEP")
    candidates: list[tuple[int, Path]] = []
    for search_root in (ckpt_root, ckpt_root / "checkpoints"):
        if not search_root.is_dir():
            continue
        for step_dir in search_root.glob("checkpoint_step_*"):
            transformer_path = step_dir / "transformer"
            if not (transformer_path / "config.json").exists():
                continue
            try:
                step_num = int(step_dir.name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.append((step_num, transformer_path))

    if not candidates:
        return None

    if pinned is not None:
        pinned_num = int(pinned)
        for step_num, transformer_path in candidates:
            if step_num == pinned_num:
                return transformer_path
        raise FileNotFoundError(
            f"LINGBOT_VA_STEP={pinned} not found under {ckpt_root} "
            f"(available: {sorted(n for n, _ in candidates)})."
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_paths(checkpoint_path: str, base_model_path: str) -> tuple[Path, Path]:
    ckpt_root = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_root}")

    base_root = Path(base_model_path).expanduser().resolve()
    if not (base_root / "vae").is_dir():
        raise FileNotFoundError(f"Base model directory missing vae/: {base_root}")

    transformer_path = _find_transformer(ckpt_root)
    if transformer_path is None:
        raise FileNotFoundError(
            f"Transformer checkpoint not found under {ckpt_root}. Expected "
            "transformer/, checkpoints/transformer/, or "
            "[checkpoints/]checkpoint_step_<N>/transformer/."
        )
    return base_root, transformer_path


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
        default=os.environ.get("LINGBOT_VA_CHECKPOINT_PATH"),
        required=os.environ.get("LINGBOT_VA_CHECKPOINT_PATH") is None,
        help="LingBot-VA trained checkpoint dir (or set LINGBOT_VA_CHECKPOINT_PATH).",
    )
    parser.add_argument(
        "--base-model-path",
        default=os.environ.get("LINGBOT_VA_BASE_MODEL_PATH"),
        required=os.environ.get("LINGBOT_VA_BASE_MODEL_PATH") is None,
        help="lingbot-va-base weights dir (or set LINGBOT_VA_BASE_MODEL_PATH).",
    )
    parser.add_argument(
        "--merged-dir",
        default=str(policy_root / ".merged_ckpt"),
    )
    args = parser.parse_args()
    build_merged_ckpt(args.checkpoint_path, args.base_model_path, args.merged_dir)


if __name__ == "__main__":
    main()
