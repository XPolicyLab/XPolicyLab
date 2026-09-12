#!/usr/bin/env python
"""Resolve production recipes and run one synthetic native CPU train step."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main() -> int:
    try:
        import h5py  # noqa: F401
        import torch
        from omegaconf import OmegaConf

        from tests.integration.test_train_step import _config, _write_episode
        from worldscape_policy.cli.config_composition import load_composed_config
        from worldscape_policy.cli.config_profiles import resolve_config_profiles
        from worldscape_policy.cli.train import run_config
    except ImportError as exc:
        print(f"SKIP: native smoke dependency unavailable: {exc}")
        return 0

    with tempfile.TemporaryDirectory(prefix="wsp-native-smoke-") as temporary:
        root = Path(temporary)
        data_root = root / "data"
        data_root.mkdir()
        checkpoint_dir = root / "output"
        _write_episode(data_root / "episode_000.hdf5", 0)
        os.environ.update(
            DATA_ROOT=str(data_root),
            PRETRAINED_MODEL_PATH=str(root / "unused-native-checkpoint"),
            WAN_CKPT_DIR=str(root / "wan"),
            CLIP_CKPT_DIR=str(root / "wan-image"),
            TOKENIZER_DIR=str(root / "tokenizer"),
            Qwen_CKPT_DIR=str(root / "qwen"),
            VLM_TOKEN_DIM="2560",
            VLM_CONTEXT_DIM="4096",
            OUTPUT_DIR=str(checkpoint_dir),
            WORLDSCAPE_CHECKPOINT=str(root / "unused-native-checkpoint"),
            WORLDSCAPE_HDF5_EPISODE=str(data_root / "episode_000.hdf5"),
            T2VA_DATA_ROOT=str(data_root),
            GOAL_IMAGE_DATA_ROOT=str(data_root),
            VIDEO_DATA_ROOT=str(data_root),
            WSP_MODE="interactive",
            VISUAL_PROMPT="demo",
            NATIVE_DATASET_NAME="worldscape_hdf5_demo",
        )
        for directory, name in (
            ("posttrain", "agilex.yaml"),
            ("posttrain", "libero.yaml"),
            ("posttrain", "robotwin2.yaml"),
            ("eval", "agilex.yaml"),
            ("pretrain", "mixed_three_mode_wan22.yaml"),
            ("pretrain", "mixed_three_mode_wan22_stage2.yaml"),
        ):
            os.environ["NATIVE_DATASET_NAME"] = (
                "worldscape_lerobot_demo"
                if directory == "posttrain" and name != "agilex.yaml"
                else "worldscape_hdf5_demo"
            )
            recipe_path = ROOT / "configs" / directory / name
            recipe = (
                load_composed_config(recipe_path)
                if directory in {"posttrain", "pretrain"}
                else OmegaConf.load(recipe_path)
            )
            resolved = resolve_config_profiles(recipe)
            OmegaConf.to_container(resolved, resolve=True)

        config = _config(data_root, checkpoint_dir)
        config.training.max_steps = 1
        config.training.save_every = 0
        config.training.save_at_end = False
        torch.manual_seed(0)
        trainer = run_config(config)
        if trainer.step != 1:
            raise RuntimeError(f"synthetic native smoke stopped at step {trainer.step}")
    print("native recipe config + synthetic CPU train-step smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
