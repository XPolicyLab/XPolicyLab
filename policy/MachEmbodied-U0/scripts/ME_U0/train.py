"""Unified launcher for all retained ME_U0 training experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _configure_rank_local_compiler_cache() -> None:
    raw_root = os.environ.get("ME_U0_COMPILER_CACHE_NODE_ROOT", "").strip()
    if not raw_root:
        return
    raw_local_rank = os.environ.get("LOCAL_RANK", "0")
    try:
        local_rank = int(raw_local_rank)
    except ValueError as error:
        raise ValueError(
            f"LOCAL_RANK must be an integer, got {raw_local_rank!r}"
        ) from error
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {local_rank}")
    rank_root = Path(raw_root).expanduser().resolve(strict=False) / f"local_{local_rank}"
    cache_paths = {
        "TORCHINDUCTOR_CACHE_DIR": rank_root / "torchinductor",
        "TRITON_CACHE_DIR": rank_root / "triton",
        "CUDA_CACHE_PATH": rank_root / "cuda",
        "TORCH_EXTENSIONS_DIR": rank_root / "torch_extensions",
    }
    for name, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = os.fspath(path)


def main() -> None:
    _configure_rank_local_compiler_cache()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint path, latest, or auto")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    args = parser.parse_args()

    from leap.callbacks.atomic_checkpoint_callback import (
        resolve_resume_checkpoint,
        validate_deepspeed_checkpoint_world_size,
    )
    from leap.core.config import load_config
    from leap.trainers.ME_U0_trainer import MachEmbodiedUnifiedTrainer

    cfg = load_config(args.config, overrides=args.overrides)
    resume = (
        resolve_resume_checkpoint(args.resume, str(cfg.work_dir))
        if args.resume
        else None
    )
    if resume and os.environ.get("ME_U0_RESUME_VALIDATED") != "1":
        saved_world_size = validate_deepspeed_checkpoint_world_size(
            resume, int(os.environ.get("WORLD_SIZE", "1"))
        )
        if (
            cfg.get("distributed", {}).get("deepspeed_config")
            and saved_world_size is None
        ):
            raise ValueError(
                "DeepSpeed resume checkpoint has no optimizer rank shards: "
                f"{resume}"
            )
    trainer = MachEmbodiedUnifiedTrainer(cfg)
    if resume:
        trainer.load_checkpoint(str(resume))
    trainer.fit()


if __name__ == "__main__":
    main()
